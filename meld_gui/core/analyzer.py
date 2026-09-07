"""Document analysis built on top of the raw MELD scoring head.

The released scoring code answers one question: what is ``P(AI)`` for this
document? This module keeps that answer bit-for-bit identical while extracting
the intermediate signal the checkpoint already computes, so the UI can also
show *where* the evidence sits and *what* it looks like:

* per-token margins, stitched across overlapping windows and mapped back to
  character offsets in the original string;
* the same top-rho pooling applied per sentence, for the heatmap;
* family and operator attribution, read off the prototype logits.

Attribution is derived from prototypes that ship in the checkpoint but are not
documented as calibrated outputs on the model card, so it is reported as
indicative and labelled that way in the UI.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

from ..config import settings
from .chunking import Window, plan_windows
from .loader import ModelNotReady, manager
from .model import Meld
from .segmentation import Span, split_sentences, text_stats, word_count
from .thresholds import DEFAULT_FPR, OVERALL, ThresholdTable

#: Above this token count the per-token payload is dropped from the response
#: to keep the JSON manageable; sentence-level detail is always returned.
TOKEN_DETAIL_LIMIT = 24_000


@dataclass(slots=True)
class AnalysisOptions:
    stratum: str = OVERALL
    fpr: str = DEFAULT_FPR
    include_tokens: bool = True
    include_sentences: bool = True
    include_attribution: bool = True
    overlap: int | None = None
    batch_size: int | None = None


def _pool_top_rho_1d(values: torch.Tensor, rho: float) -> float:
    """Document-level pooling rule applied to an arbitrary set of tokens."""
    if values.numel() == 0:
        return 0.0
    k = max(1, math.ceil(values.numel() * rho))
    top = torch.topk(values, k=min(k, values.numel())).values
    return float(top.sum().item() / k)


class Analyzer:
    """Runs a full document analysis against the loaded checkpoint."""

    def __init__(self) -> None:
        self._manager = manager

    # ------------------------------------------------------------ inference
    def _forward_windows(
        self,
        model: Meld,
        tokenizer: Any,
        windows: Sequence[Window],
        device: str,
        batch_size: int,
    ) -> list[Any]:
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        results: list[Any] = []
        for i in range(0, len(windows), batch_size):
            batch = windows[i : i + batch_size]
            width = max(w.length for w in batch) + 2

            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            attention = torch.zeros((len(batch), width), dtype=torch.long)
            specials = torch.ones((len(batch), width), dtype=torch.long)

            for row, window in enumerate(batch):
                seq = [cls_id, *window.ids, sep_id]
                input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                attention[row, : len(seq)] = 1
                specials[row, 1 : 1 + window.length] = 0

            results.extend(
                model.forward_batch(
                    input_ids.to(device), attention.to(device), specials.to(device)
                )
            )
        return results

    # ------------------------------------------------------------- stitching
    @staticmethod
    def _stitch(
        windows: Sequence[Window],
        chunk_scores: Sequence[Any],
        n_tokens: int,
        n_families: int,
        n_ops: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge overlapping windows into one score per global token.

        Where windows overlap, the copy of a token that sat furthest from a
        window edge wins, because it was scored with the most context.
        """
        per_token = torch.zeros(n_tokens)
        family = torch.zeros(n_tokens, n_families)
        ops = torch.zeros(n_tokens, n_ops)
        priority = torch.full((n_tokens,), -1.0)

        for window, scores in zip(windows, chunk_scores):
            # Row layout is [CLS] + content + [SEP]; content starts at index 1.
            content = slice(1, 1 + window.length)
            values = scores.per_token[content]
            fam = scores.family_logits[content]
            op = scores.op_logits[content]

            local = torch.arange(window.length)
            rank = torch.minimum(local, window.length - 1 - local).float()
            positions = local + window.start

            better = rank > priority[positions]
            if not bool(better.any()):
                continue
            target = positions[better]
            source = local[better]

            priority[target] = rank[better]
            per_token[target] = values[source]
            family[target] = fam[source]
            ops[target] = op[source]
        return per_token, family, ops

    # ------------------------------------------------------------ attribution
    @staticmethod
    def _attribute(
        logits: torch.Tensor, labels: Sequence[str], keep_idx: torch.Tensor
    ) -> list[dict[str, Any]]:
        """Mean softmax over the tokens that drove the document score."""
        if keep_idx.numel() == 0 or logits.numel() == 0:
            return []
        selected = logits.index_select(0, keep_idx)
        probs = torch.softmax(selected, dim=-1).mean(0)
        order = torch.argsort(probs, descending=True)
        return [
            {
                "label": labels[int(i)],
                "probability": round(float(probs[int(i)].item()), 6),
            }
            for i in order
        ]

    # ---------------------------------------------------------------- public
    def analyze(self, text: str, options: AnalysisOptions | None = None) -> dict[str, Any]:
        options = options or AnalysisOptions()
        model, tokenizer, thresholds, device = self._manager.require()

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            raise ValueError("The document is empty.")
        if len(text) > settings.max_chars:
            raise ValueError(
                f"Document is {len(text):,} characters; the limit is "
                f"{settings.max_chars:,}. Split it into parts."
            )

        started = time.perf_counter()

        # verbose=False: the tokenizer warns whenever a sequence exceeds the
        # backbone's max length, but we never feed it whole -- it is cut into
        # windows below.
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        token_ids: list[int] = list(encoded["input_ids"])
        offsets: list[tuple[int, int]] = [tuple(o) for o in encoded["offset_mapping"]]
        if not token_ids:
            raise ValueError("The document produced no tokens.")

        overlap = options.overlap if options.overlap is not None else settings.chunk_overlap
        batch_size = options.batch_size or settings.batch_size
        windows = plan_windows(token_ids, model.max_length, overlap)
        chunk_scores = self._forward_windows(
            model, tokenizer, windows, device, batch_size
        )

        families = model.families
        ops = model.ops
        per_token, family_logits, op_logits = self._stitch(
            windows, chunk_scores, len(token_ids), len(families), len(ops)
        )

        # Document score: the released top-rho rule, applied over every token
        # in the document rather than only the first window.
        rho = model.rho
        k = max(1, math.ceil(len(token_ids) * rho))
        top = torch.topk(per_token, k=min(k, per_token.numel()))
        doc_score = float(top.values.sum().item() / k)
        doc_prob = float(torch.sigmoid(torch.tensor(doc_score)).item())
        keep_idx = top.indices

        decision = thresholds.decide(doc_score, doc_prob, options.stratum, options.fpr)

        stats = text_stats(text)
        elapsed = time.perf_counter() - started

        result: dict[str, Any] = {
            "decision": decision.as_dict(),
            "statistics": stats,
            "tokens": {
                "count": len(token_ids),
                "windows": len(windows),
                "truncated": False,
                "max_length": model.max_length,
                "rho": rho,
                "pooled_tokens": int(min(k, per_token.numel())),
            },
            "timing": {
                "seconds": round(elapsed, 3),
                "tokens_per_second": round(len(token_ids) / elapsed, 1) if elapsed else 0.0,
                "device": device,
            },
            "warnings": self._warnings(stats, len(windows)),
            "thresholds": {
                "all": {
                    row["key"]: row["thresholds"] for row in thresholds.describe()
                },
                "selected": decision.threshold,
            },
        }

        if options.include_attribution:
            result["attribution"] = {
                "families": self._attribute(family_logits, families, keep_idx),
                "operations": self._attribute(op_logits, ops, keep_idx),
                "note": (
                    "Derived from the family and operator prototypes stored in the "
                    "checkpoint. These are not calibrated outputs on the model card; "
                    "read them as a stylistic lean, not an identification."
                ),
            }

        if options.include_sentences:
            result["sentences"] = self._sentence_rows(
                text, offsets, per_token, rho, decision.threshold
            )
            result["evidence"] = sorted(
                result["sentences"], key=lambda r: r["score"], reverse=True
            )[:8]

        result["chunks"] = [
            {
                "index": window.index,
                "token_start": window.start,
                "token_end": window.end,
                "char_start": offsets[window.start][0] if window.start < len(offsets) else 0,
                "char_end": offsets[window.end - 1][1] if window.end - 1 < len(offsets) else len(text),
                "score": round(scores.doc_score, 4),
                "probability": round(scores.doc_prob, 6),
                "flagged": scores.doc_score > decision.threshold,
            }
            for window, scores in zip(windows, chunk_scores)
        ]

        if options.include_tokens and len(token_ids) <= TOKEN_DETAIL_LIMIT:
            result["token_detail"] = {
                "offsets": [[int(a), int(b)] for a, b in offsets],
                "scores": [round(float(v), 3) for v in per_token.tolist()],
            }

        return result

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _warnings(stats: dict[str, Any], n_windows: int) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        words = int(stats["words"])
        if words < settings.min_words:
            out.append(
                {
                    "level": "warning",
                    "message": (
                        f"Only {words} words. MELD is unreliable below "
                        f"{settings.min_words} words and tends to score short text "
                        "high regardless of origin. Treat this result as noise."
                    ),
                }
            )
        elif words < settings.min_words * 2:
            out.append(
                {
                    "level": "info",
                    "message": (
                        f"{words} words is close to the {settings.min_words}-word "
                        "floor; longer samples give steadier scores."
                    ),
                }
            )
        if n_windows > 1:
            out.append(
                {
                    "level": "info",
                    "message": (
                        f"Document exceeds the {2048}-token context and was scored in "
                        f"{n_windows} overlapping windows. Per-window scores are in the "
                        "Windows tab."
                    ),
                }
            )
        return out

    @staticmethod
    def _sentence_rows(
        text: str,
        offsets: Sequence[tuple[int, int]],
        per_token: torch.Tensor,
        rho: float,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Aggregate token margins onto sentence spans."""
        sentences = split_sentences(text)
        if not sentences:
            return []

        rows: list[dict[str, Any]] = []
        cursor = 0
        n_tokens = len(offsets)

        for span in sentences:
            # Offsets are non-decreasing, so walk the token list once.
            while cursor < n_tokens and offsets[cursor][1] <= span.start:
                cursor += 1
            idx = cursor
            picked: list[int] = []
            while idx < n_tokens and offsets[idx][0] < span.end:
                if offsets[idx][1] > offsets[idx][0]:
                    picked.append(idx)
                idx += 1

            if not picked:
                continue

            values = per_token[torch.tensor(picked, dtype=torch.long)]
            score = _pool_top_rho_1d(values, rho)
            rows.append(
                {
                    "index": span.index,
                    "start": span.start,
                    "end": span.end,
                    "text": span.text.strip(),
                    "words": word_count(span.text),
                    "tokens": len(picked),
                    "score": round(score, 4),
                    "probability": round(float(torch.sigmoid(torch.tensor(score)).item()), 6),
                    "mean": round(float(values.mean().item()), 4),
                    "max": round(float(values.max().item()), 4),
                    "flagged": score > threshold,
                }
            )
        return rows


analyzer = Analyzer()


def analyze_text(text: str, options: AnalysisOptions | None = None) -> dict[str, Any]:
    """Module-level convenience wrapper."""
    return analyzer.analyze(text, options)
