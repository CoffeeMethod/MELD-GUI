"""Faithful port of the MELD detector scoring head (v5 and v7+ checkpoints).

The upstream model card ships a minimal ``Meld`` class that returns only a
document level ``P(AI)``.  The checkpoint however carries a good deal more
signal: per-token style coordinates, per-family prototypes and per-operator
prototypes.  This module keeps the released maths bit-for-bit identical while
also handing back the intermediate tensors so the UI can build heatmaps and
attribution charts on top of them.

Reference implementation: https://huggingface.co/anon-review-meld-2026/meld
(``meld.py`` in that repository).

From v7 the head scores against one prototype per *generator* (``gen_protos``)
and a family's logit is its nearest member generator's logit, using the
``generator_family`` map in ``meld_config.json``. v5 checkpoints store one
prototype per family under ``family_protos``; they load into the same layout
with an identity map, exactly as the reference loader does, so both score
identically to their own reference code.

Provenance: the scoring head below is a port of the reference implementation
published on that model card, which is distributed under the MIT License
(Copyright (c) the MELD authors). That MIT notice is reproduced in full in the
NOTICE file at the repository root and continues to cover this derived work.
The remainder of this project is licensed under Apache License 2.0.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel

# Clamp bounds lifted verbatim from the released scoring code.
_TAU_CLAMP = 4.0
_MARGIN_CLAMP = 30.0


@dataclass(slots=True)
class ChunkScores:
    """Raw tensors for a single forward pass over one window of tokens."""

    per_token: torch.Tensor      # (L,) machine-vs-human margin per token
    valid: torch.Tensor          # (L,) bool, real content tokens only
    family_logits: torch.Tensor  # (L, n_families), nearest member generator
    op_logits: torch.Tensor      # (L, n_ops)
    doc_score: float             # top-rho pooled margin for this window
    doc_prob: float              # sigmoid(doc_score)


class Meld(nn.Module):
    """MELD detector (v5 and v7+ checkpoints).

    The scoring head is custom, so ``pipeline()`` /
    ``AutoModelForSequenceClassification`` cannot load this checkpoint; the
    module is constructed by hand and the state dict is loaded strictly.
    """

    def __init__(self, model_dir: str | Path) -> None:
        super().__init__()
        model_dir = Path(model_dir)
        self.model_dir = model_dir
        with open(model_dir / "meld_config.json", encoding="utf-8") as fh:
            self.cfg: dict[str, Any] = json.load(fh)

        r = self.cfg["style_rank"]
        backbone_cfg = AutoConfig.from_pretrained(model_dir)
        # The backbone's own config is the source of truth for the width. The
        # upstream repo swapped Ettin sizes between revisions, so a stale
        # backbone_hidden_size in meld_config.json must not decide the shape.
        hidden = int(getattr(backbone_cfg, "hidden_size", 0) or self.cfg["backbone_hidden_size"])
        self.null_reduction = str(self.cfg.get("null_reduction", "logsumexp"))

        self.backbone = AutoModel.from_config(backbone_cfg, attn_implementation="sdpa")
        self.style_proj = nn.Linear(hidden, r, bias=False)
        self.style_ln = nn.LayerNorm(r)
        self.human_anchors = nn.Parameter(torch.zeros(self.cfg["n_human_anchors"], r))
        n_fam = self.cfg["n_families"]
        gen_family = self.cfg.get("generator_family") or list(range(n_fam))
        n_gen = len(gen_family)
        self.gen_protos = nn.Parameter(torch.zeros(n_gen, r))
        self.gen_bias = nn.Parameter(torch.zeros(n_gen))
        member = torch.zeros(n_gen, n_fam, dtype=torch.bool)
        member[torch.arange(n_gen), torch.tensor(gen_family, dtype=torch.long)] = True
        self.register_buffer("gen_member", member, persistent=False)
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.op_protos = nn.Parameter(torch.zeros(self.cfg["n_ops"], r))
        self.op_bias = nn.Parameter(torch.zeros(self.cfg["n_ops"]))

        state = load_file(str(model_dir / "model.safetensors"))
        for old, new in (("family_protos", "gen_protos"), ("family_bias", "gen_bias")):
            if old in state:
                state[new] = state.pop(old)
        self.load_state_dict(state, strict=True)
        self.eval()

    # ------------------------------------------------------------------ meta
    @property
    def max_length(self) -> int:
        return int(self.cfg["max_length"])

    @property
    def rho(self) -> float:
        return float(self.cfg["rho"])

    @property
    def families(self) -> list[str]:
        return list(self.cfg["families"])

    @property
    def ops(self) -> list[str]:
        return list(self.cfg["ops"])

    # ------------------------------------------------------------- internals
    @staticmethod
    def _sqdist(u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Squared euclidean distance from every token to every prototype."""
        return (
            (u * u).sum(-1, keepdim=True)
            - 2.0 * u @ p.t()
            + (p * p).sum(-1).view(1, 1, -1)
        )

    def _pool_top_rho(self, per_token: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Document score = mean of the most machine-like ``rho`` fraction."""
        x = per_token.masked_fill(~valid, torch.finfo(per_token.dtype).min)
        x, _ = x.sort(dim=1, descending=True)
        k = (valid.sum(1).clamp(min=1) * self.cfg["rho"]).ceil().clamp(min=1).long()
        keep = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < k.unsqueeze(1)
        return torch.where(keep, x, torch.zeros_like(x)).sum(1) / k.float()

    # -------------------------------------------------------------- forward
    @torch.no_grad()
    def forward_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor,
    ) -> list[ChunkScores]:
        """Score a padded batch of pre-tokenised windows.

        Returns one :class:`ChunkScores` per row, trimmed to that row's length.
        """
        valid = attention_mask.bool() & ~special_tokens_mask.bool()
        h = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state.float()

        u = self.style_ln(self.style_proj(h))          # style coordinates
        tau = self.log_tau.clamp(-_TAU_CLAMP, _TAU_CLAMP).exp()

        anchor_logits = -tau * self._sqdist(u, self.human_anchors)
        if self.null_reduction == "max":
            human = anchor_logits.max(-1, keepdim=True).values
        else:
            human = torch.logsumexp(anchor_logits, -1, keepdim=True)
        gen = -tau * self._sqdist(u, self.gen_protos) + self.gen_bias.view(1, 1, -1)
        family = torch.where(
            self.gen_member, gen.unsqueeze(-1), gen.new_tensor(float("-inf"))
        ).amax(dim=2)
        op = -tau * self._sqdist(u, self.op_protos) + self.op_bias.view(1, 1, -1)

        tau_agg = self.cfg["tau_agg"]
        per_token = tau_agg * torch.logsumexp(
            (family - human).clamp(-_MARGIN_CLAMP, _MARGIN_CLAMP) / tau_agg, dim=-1
        )

        doc = self._pool_top_rho(per_token, valid)
        probs = torch.sigmoid(doc)

        out: list[ChunkScores] = []
        for i in range(input_ids.shape[0]):
            n = int(attention_mask[i].sum().item())
            out.append(
                ChunkScores(
                    per_token=per_token[i, :n].cpu(),
                    valid=valid[i, :n].cpu(),
                    family_logits=family[i, :n].cpu(),
                    op_logits=op[i, :n].cpu(),
                    doc_score=float(doc[i].item()),
                    doc_prob=float(probs[i].item()),
                )
            )
        return out

    @torch.no_grad()
    def score(self, texts: Sequence[str], tokenizer, device: str = "cpu"):
        """Reference-compatible entry point: returns ``(probs, raw_scores)``.

        Kept independent of the analyzer and identical to the reference scorer
        so results can be verified against it. v7+ checkpoints follow
        ``Scorer.score`` in the upstream ``meld.py`` (canonical text,
        consecutive windows, token cap, one joint top-rho pool); v5 checkpoints
        follow the original single-window model card code.
        """
        if self.cfg.get("scoring_contract"):
            scores = [self._score_contract(t, tokenizer, device) for t in texts]
            return [float(torch.sigmoid(torch.tensor(s))) for s in scores], scores
        enc = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg["max_length"],
            return_special_tokens_mask=True,
        ).to(device)
        rows = self.forward_batch(
            enc["input_ids"], enc["attention_mask"], enc["special_tokens_mask"]
        )
        return [r.doc_prob for r in rows], [r.doc_score for r in rows]

    def _score_contract(self, text: str, tokenizer, device: str, batch_size: int = 4) -> float:
        from .canonical import canonicalize

        ids = tokenizer(canonicalize(text), add_special_tokens=False, truncation=False)["input_ids"]
        ids = ids[: int(self.cfg.get("cap_tokens", 16384))]
        if not ids:
            raise ValueError("no scoreable tokens in input")
        width = self.max_length - 2
        windows = [ids[i : i + width] for i in range(0, len(ids), width)]
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        parts: list[torch.Tensor] = []
        for b0 in range(0, len(windows), batch_size):
            batch = windows[b0 : b0 + batch_size]
            span = max(len(w) for w in batch) + 2
            inp = torch.full((len(batch), span), pad, dtype=torch.long)
            att = torch.zeros((len(batch), span), dtype=torch.long)
            spec = torch.ones((len(batch), span), dtype=torch.long)
            for j, w in enumerate(batch):
                inp[j, 0] = tokenizer.cls_token_id
                inp[j, 1 : 1 + len(w)] = torch.tensor(w, dtype=torch.long)
                inp[j, 1 + len(w)] = tokenizer.sep_token_id
                att[j, : len(w) + 2] = 1
                spec[j, 1 : 1 + len(w)] = 0
            rows = self.forward_batch(inp.to(device), att.to(device), spec.to(device))
            parts.extend(r.per_token[1 : 1 + len(w)] for r, w in zip(rows, batch))
        tokens = torch.cat(parts)
        k = max(1, math.ceil(tokens.numel() * self.rho))
        return float(tokens.sort(descending=True).values[:k].mean())
