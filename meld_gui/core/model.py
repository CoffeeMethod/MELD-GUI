"""Faithful port of the MELD v5 detector scoring head.

The upstream model card ships a minimal ``Meld`` class that returns only a
document level ``P(AI)``.  The checkpoint however carries a good deal more
signal: per-token style coordinates, per-family prototypes and per-operator
prototypes.  This module keeps the released maths bit-for-bit identical while
also handing back the intermediate tensors so the UI can build heatmaps and
attribution charts on top of them.

Reference implementation: https://huggingface.co/anon-review-meld-2026/meld

Provenance: the scoring head below is a port of the reference implementation
published on that model card, which is distributed under the MIT License
(Copyright (c) the MELD authors). That MIT notice is reproduced in full in the
NOTICE file at the repository root and continues to cover this derived work.
The remainder of this project is licensed under Apache License 2.0.
"""

from __future__ import annotations

import json
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
    family_logits: torch.Tensor  # (L, n_families)
    op_logits: torch.Tensor      # (L, n_ops)
    doc_score: float             # top-rho pooled margin for this window
    doc_prob: float              # sigmoid(doc_score)


class Meld(nn.Module):
    """MELD v5 detector.

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
        hidden = self.cfg["backbone_hidden_size"]

        self.backbone = AutoModel.from_config(
            AutoConfig.from_pretrained(model_dir), attn_implementation="sdpa"
        )
        self.style_proj = nn.Linear(hidden, r, bias=False)
        self.style_ln = nn.LayerNorm(r)
        self.human_anchors = nn.Parameter(torch.zeros(self.cfg["n_human_anchors"], r))
        self.family_protos = nn.Parameter(torch.zeros(self.cfg["n_families"], r))
        self.family_bias = nn.Parameter(torch.zeros(self.cfg["n_families"]))
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.op_protos = nn.Parameter(torch.zeros(self.cfg["n_ops"], r))
        self.op_bias = nn.Parameter(torch.zeros(self.cfg["n_ops"]))

        self.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
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

        human = torch.logsumexp(
            -tau * self._sqdist(u, self.human_anchors), -1, keepdim=True
        )
        family = -tau * self._sqdist(u, self.family_protos) + self.family_bias.view(1, 1, -1)
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

        Kept identical to the model card so results can be verified against it.
        """
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
