"""Calibrated decision thresholds.

MELD's raw score is not calibrated against 0.5.  The checkpoint ships
``score_offsets``: for the overall validation set and for six document strata,
the raw score below which 99% / 95% / 90% of *human* texts fell.  Flagging is
therefore a choice of acceptable false-positive rate, not a fixed cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

FPR_KEYS = ("fpr_0.01", "fpr_0.05", "fpr_0.1")
DEFAULT_FPR = "fpr_0.01"
OVERALL = "overall"

#: Human readable labels for the strata shipped in the checkpoint.
STRATA_LABELS: dict[str, str] = {
    "overall": "General / mixed",
    "academic": "Academic writing",
    "creative": "Creative writing",
    "qa_social": "Q&A / social",
    "reviews": "Reviews",
    "web": "Web prose",
    "wiki": "Encyclopedic",
}

STRATA_HINTS: dict[str, str] = {
    "overall": "Use when the document type is unknown or mixed.",
    "academic": "Papers, essays, theses, technical reports.",
    "creative": "Fiction, poetry, screenwriting, narrative prose.",
    "qa_social": "Forum posts, comments, chat, short-form answers.",
    "reviews": "Product, film and service reviews.",
    "web": "Blog posts, news, marketing and general web copy.",
    "wiki": "Encyclopedic and reference-style articles.",
}

Verdict = Literal["human", "uncertain", "likely_ai", "ai"]

#: Bands are expressed as a margin (raw score minus the selected threshold).
#: They are a presentation aid layered on top of the single calibrated
#: decision; the flag itself is always ``score > threshold``.
_BANDS: tuple[tuple[float, Verdict], ...] = (
    (2.0, "ai"),
    (0.0, "likely_ai"),
    (-1.5, "uncertain"),
)

VERDICT_LABELS: dict[str, str] = {
    "human": "Likely human-written",
    "uncertain": "Inconclusive",
    "likely_ai": "Likely AI-generated",
    "ai": "Strong AI signal",
}


@dataclass(slots=True)
class Decision:
    """The outcome of comparing a raw score to a calibrated threshold."""

    score: float
    probability: float
    threshold: float
    margin: float
    flagged: bool
    verdict: Verdict
    verdict_label: str
    stratum: str
    stratum_label: str
    fpr: str
    fpr_pct: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "probability": self.probability,
            "threshold": self.threshold,
            "margin": self.margin,
            "flagged": self.flagged,
            "verdict": self.verdict,
            "verdict_label": self.verdict_label,
            "stratum": self.stratum,
            "stratum_label": self.stratum_label,
            "fpr": self.fpr,
            "fpr_pct": self.fpr_pct,
        }


class ThresholdTable:
    """Read-only view over ``meld_config.json['score_offsets']``."""

    def __init__(self, score_offsets: dict[str, Any]) -> None:
        self._raw = score_offsets
        self._overall = score_offsets[OVERALL]
        self._strata = score_offsets.get("strata", {})

    @property
    def strata(self) -> list[str]:
        return [OVERALL, *sorted(self._strata)]

    def n_human(self) -> int:
        return int(self._raw.get("n_human", 0))

    def entry(self, stratum: str) -> dict[str, Any]:
        if stratum == OVERALL:
            return self._overall
        if stratum not in self._strata:
            raise KeyError(f"unknown stratum {stratum!r}")
        return self._strata[stratum]

    def threshold(self, stratum: str = OVERALL, fpr: str = DEFAULT_FPR) -> float:
        if fpr not in FPR_KEYS:
            raise KeyError(f"unknown fpr level {fpr!r}")
        return float(self.entry(stratum)[fpr])

    def describe(self) -> list[dict[str, Any]]:
        """Everything the UI needs to render the calibration table."""
        rows = []
        for stratum in self.strata:
            entry = self.entry(stratum)
            rows.append(
                {
                    "key": stratum,
                    "label": STRATA_LABELS.get(stratum, stratum),
                    "hint": STRATA_HINTS.get(stratum, ""),
                    "n": int(entry.get("n", self._raw.get("n_human", 0))),
                    "thresholds": {k: float(entry[k]) for k in FPR_KEYS},
                }
            )
        return rows

    def decide(
        self,
        score: float,
        probability: float,
        stratum: str = OVERALL,
        fpr: str = DEFAULT_FPR,
    ) -> Decision:
        threshold = self.threshold(stratum, fpr)
        margin = score - threshold
        verdict: Verdict = "human"
        for cutoff, name in _BANDS:
            if margin >= cutoff:
                verdict = name
                break
        return Decision(
            score=score,
            probability=probability,
            threshold=threshold,
            margin=margin,
            flagged=score > threshold,
            verdict=verdict,
            verdict_label=VERDICT_LABELS[verdict],
            stratum=stratum,
            stratum_label=STRATA_LABELS.get(stratum, stratum),
            fpr=fpr,
            fpr_pct=float(fpr.split("_")[1]) * 100.0,
        )
