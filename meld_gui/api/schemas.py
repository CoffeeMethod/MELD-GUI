"""Request and response models for the REST API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.thresholds import DEFAULT_FPR, FPR_KEYS, OVERALL

FprLevel = Literal["fpr_0.01", "fpr_0.05", "fpr_0.1"]


class AnalyzeRequest(BaseModel):
    """A single document to score."""

    text: str = Field(..., description="The document to analyse.")
    stratum: str = Field(OVERALL, description="Calibration stratum for the threshold.")
    fpr: FprLevel = Field(DEFAULT_FPR, description="Accepted false-positive rate.")
    include_tokens: bool = Field(True, description="Return per-token scores for the heatmap.")
    include_sentences: bool = Field(True, description="Return per-sentence breakdown.")
    include_attribution: bool = Field(True, description="Return family/operator attribution.")
    save_history: bool = Field(True, description="Persist this run to history.")
    label: str = Field("", description="Optional name shown in history.")


class BatchDocument(BaseModel):
    name: str = ""
    text: str


class BatchRequest(BaseModel):
    documents: list[BatchDocument]
    stratum: str = OVERALL
    fpr: FprLevel = DEFAULT_FPR
    save_history: bool = True


class CompareRequest(BaseModel):
    """Two documents scored under identical settings."""

    documents: list[BatchDocument] = Field(..., min_length=2, max_length=6)
    stratum: str = OVERALL
    fpr: FprLevel = DEFAULT_FPR


class LoadRequest(BaseModel):
    device: str | None = Field(
        None, description="Torch device: cuda, cpu, or null for automatic."
    )


class ErrorResponse(BaseModel):
    detail: str
    code: str = "error"


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool


__all__ = [
    "AnalyzeRequest",
    "BatchDocument",
    "BatchRequest",
    "CompareRequest",
    "LoadRequest",
    "ErrorResponse",
    "HealthResponse",
    "FPR_KEYS",
]
