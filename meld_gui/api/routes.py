"""REST endpoints.

Everything the frontend does goes through this router, which means the whole
application is scriptable: point any HTTP client at the same routes and read
the generated docs at /docs.
"""

from __future__ import annotations

import csv
import io
import json
import time
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from .. import __version__
from ..config import settings
from ..core.analyzer import AnalysisOptions, analyzer
from ..core.extract import SUPPORTED, ExtractionError, extract
from ..core.loader import ModelNotReady, manager
from ..core.segmentation import text_stats
from ..core.thresholds import (
    FPR_KEYS,
    OVERALL,
    STRATA_HINTS,
    STRATA_LABELS,
    VERDICT_LABELS,
)
from ..services.history import history
from ..services.jobs import jobs

router = APIRouter(prefix="/api")


def _options(payload: Any) -> AnalysisOptions:
    return AnalysisOptions(
        stratum=getattr(payload, "stratum", OVERALL),
        fpr=getattr(payload, "fpr", "fpr_0.01"),
        include_tokens=getattr(payload, "include_tokens", True),
        include_sentences=getattr(payload, "include_sentences", True),
        include_attribution=getattr(payload, "include_attribution", True),
    )


def _guard(fn, *args, **kwargs):
    """Translate domain errors into tidy HTTP responses."""
    try:
        return fn(*args, **kwargs)
    except ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --------------------------------------------------------------------- meta
@router.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "model_loaded": manager.is_loaded,
    }


@router.get("/config", tags=["meta"])
def get_config() -> dict[str, Any]:
    """UI metadata that is available whether or not the model is loaded."""
    return {
        "version": __version__,
        "repo_id": settings.repo_id,
        "min_words": settings.min_words,
        "max_chars": settings.max_chars,
        "supported_files": sorted(SUPPORTED),
        "fpr_levels": [
            {"key": key, "label": f"{float(key.split('_')[1]) * 100:g}% FPR"}
            for key in FPR_KEYS
        ],
        "strata": [
            {"key": key, "label": label, "hint": STRATA_HINTS.get(key, "")}
            for key, label in STRATA_LABELS.items()
        ],
        "verdicts": VERDICT_LABELS,
    }


# -------------------------------------------------------------------- model
@router.get("/model", tags=["model"])
def model_status() -> dict[str, Any]:
    return manager.status()


@router.post("/model/load", tags=["model"])
def model_load(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    device = (payload or {}).get("device")
    manager.ensure_async(device)
    return manager.status()


@router.post("/model/unload", tags=["model"])
def model_unload() -> dict[str, Any]:
    manager.unload()
    return manager.status()


# ------------------------------------------------------------------ analyse
@router.post("/analyze", tags=["analysis"])
def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="No text supplied.")

    options = AnalysisOptions(
        stratum=payload.get("stratum", OVERALL),
        fpr=payload.get("fpr", "fpr_0.01"),
        include_tokens=bool(payload.get("include_tokens", True)),
        include_sentences=bool(payload.get("include_sentences", True)),
        include_attribution=bool(payload.get("include_attribution", True)),
    )
    result = _guard(analyzer.analyze, text, options)

    if payload.get("save_history", True):
        result["record_id"] = history.add(
            result, text, label=payload.get("label", ""), source=payload.get("source", "paste")
        )
    return result


@router.post("/analyze/file", tags=["analysis"])
async def analyze_file(
    file: UploadFile = File(...),
    stratum: str = Form(OVERALL),
    fpr: str = Form("fpr_0.01"),
    save_history: bool = Form(True),
) -> dict[str, Any]:
    data = await file.read()
    try:
        extracted = extract(data, file.filename or "upload.txt")
    except ExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    options = AnalysisOptions(stratum=stratum, fpr=fpr)
    result = _guard(analyzer.analyze, extracted.text, options)
    result["source_file"] = {
        "name": extracted.filename,
        "kind": extracted.kind,
        "pages": extracted.pages,
    }
    result["text"] = extracted.text
    if save_history:
        result["record_id"] = history.add(
            result, extracted.text, label=extracted.filename, source="file"
        )
    return result


@router.post("/extract", tags=["analysis"])
async def extract_only(file: UploadFile = File(...)) -> dict[str, Any]:
    """Extract text from a file without scoring it."""
    data = await file.read()
    try:
        extracted = extract(data, file.filename or "upload.txt")
    except ExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "name": extracted.filename,
        "kind": extracted.kind,
        "pages": extracted.pages,
        "text": extracted.text,
        "statistics": text_stats(extracted.text),
    }


@router.post("/compare", tags=["analysis"])
def compare(payload: dict[str, Any]) -> dict[str, Any]:
    """Score several documents under identical settings, side by side."""
    documents = payload.get("documents") or []
    if len(documents) < 2:
        raise HTTPException(status_code=400, detail="Supply at least two documents.")

    options = AnalysisOptions(
        stratum=payload.get("stratum", OVERALL),
        fpr=payload.get("fpr", "fpr_0.01"),
        include_tokens=False,
    )
    started = time.perf_counter()
    out = []
    for index, doc in enumerate(documents):
        text = (doc or {}).get("text", "")
        name = (doc or {}).get("name") or f"Document {index + 1}"
        if not text.strip():
            out.append({"name": name, "error": "Empty document."})
            continue
        result = _guard(analyzer.analyze, text, options)
        out.append({"name": name, "result": result})
    return {"documents": out, "seconds": round(time.perf_counter() - started, 3)}


# -------------------------------------------------------------------- batch
@router.post("/batch", tags=["batch"])
def batch_start(payload: dict[str, Any]) -> dict[str, Any]:
    documents = payload.get("documents") or []
    prepared = [
        (d.get("name") or f"Document {i + 1}", d.get("text", ""))
        for i, d in enumerate(documents)
        if (d or {}).get("text", "").strip()
    ]
    if not prepared:
        raise HTTPException(status_code=400, detail="No non-empty documents supplied.")
    if not manager.is_loaded:
        raise HTTPException(status_code=503, detail="Load the model first.")

    options = AnalysisOptions(
        stratum=payload.get("stratum", OVERALL),
        fpr=payload.get("fpr", "fpr_0.01"),
        include_tokens=False,
        include_sentences=False,
    )
    job = jobs.create(prepared, options, save_history=payload.get("save_history", True))
    return job.snapshot()


@router.post("/batch/files", tags=["batch"])
async def batch_files(
    files: list[UploadFile] = File(...),
    stratum: str = Form(OVERALL),
    fpr: str = Form("fpr_0.01"),
    save_history: bool = Form(True),
) -> dict[str, Any]:
    if not manager.is_loaded:
        raise HTTPException(status_code=503, detail="Load the model first.")

    prepared: list[tuple[str, str]] = []
    failures: list[dict[str, str]] = []
    for upload in files:
        data = await upload.read()
        name = upload.filename or "upload.txt"
        try:
            prepared.append((name, extract(data, name).text))
        except ExtractionError as exc:
            failures.append({"name": name, "error": str(exc)})

    if not prepared:
        detail = "; ".join(f"{f['name']}: {f['error']}" for f in failures)
        raise HTTPException(status_code=400, detail=detail or "No readable files.")

    options = AnalysisOptions(
        stratum=stratum, fpr=fpr, include_tokens=False, include_sentences=False
    )
    job = jobs.create(prepared, options, save_history=save_history)
    snapshot = job.snapshot()
    snapshot["skipped"] = failures
    return snapshot


@router.get("/batch/{job_id}", tags=["batch"])
def batch_status(job_id: str, results: bool = Query(False)) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    return job.snapshot(include_results=results)


@router.post("/batch/{job_id}/cancel", tags=["batch"])
def batch_cancel(job_id: str) -> dict[str, Any]:
    if not jobs.cancel(job_id):
        raise HTTPException(status_code=409, detail="Job is not cancellable.")
    return {"cancelled": True}


@router.get("/batch/{job_id}/export.csv", tags=["batch"])
def batch_export(job_id: str) -> StreamingResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    rows = [i.summary() for i in job.items]
    columns = [
        "name", "status", "verdict_label", "flagged", "probability",
        "score", "threshold", "words", "top_family", "error",
    ]
    return _csv_response(rows, columns, f"meld-batch-{job_id}.csv")


# ------------------------------------------------------------------ history
@router.get("/history", tags=["history"])
def history_list(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    q: str = Query(""),
    flagged_only: bool = Query(False),
) -> dict[str, Any]:
    return history.list(limit=limit, offset=offset, query=q, flagged_only=flagged_only)


@router.get("/history/stats", tags=["history"])
def history_stats() -> dict[str, Any]:
    return history.stats()


@router.get("/history/export.csv", tags=["history"])
def history_export() -> StreamingResponse:
    rows = list(history.export_rows())
    columns = [
        "id", "created_at", "label", "source", "words", "verdict",
        "flagged", "probability", "score", "threshold", "stratum", "fpr", "preview",
    ]
    return _csv_response(rows, columns, "meld-history.csv")


@router.get("/history/{record_id}", tags=["history"])
def history_get(record_id: str) -> dict[str, Any]:
    record = history.get(record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown record.")
    return record


@router.delete("/history/{record_id}", tags=["history"])
def history_delete(record_id: str) -> dict[str, Any]:
    if not history.delete(record_id):
        raise HTTPException(status_code=404, detail="Unknown record.")
    return {"deleted": True}


@router.delete("/history", tags=["history"])
def history_clear() -> dict[str, Any]:
    return {"deleted": history.clear()}


# ------------------------------------------------------------------ helpers
def _csv_response(
    rows: list[dict[str, Any]], columns: list[str], filename: str
) -> StreamingResponse:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
