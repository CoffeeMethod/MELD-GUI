"""Background batch jobs.

Batch scanning a folder of documents can take minutes, so it runs on a worker
thread and the UI polls for progress. Jobs are held in memory only: they are a
view over work in flight, while anything worth keeping is written to the
history store as each document finishes.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..core.analyzer import AnalysisOptions, analyzer
from .history import history


@dataclass
class BatchItem:
    name: str
    text: str
    status: str = "pending"      # pending | running | done | error | skipped
    error: str = ""
    result: dict[str, Any] | None = None
    record_id: str = ""

    def summary(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "record_id": self.record_id,
        }
        if self.result:
            decision = self.result["decision"]
            stats = self.result["statistics"]
            row.update(
                {
                    "score": decision["score"],
                    "probability": decision["probability"],
                    "threshold": decision["threshold"],
                    "flagged": decision["flagged"],
                    "verdict": decision["verdict"],
                    "verdict_label": decision["verdict_label"],
                    "words": stats["words"],
                    "top_family": (
                        self.result.get("attribution", {})
                        .get("families", [{}])[0]
                        .get("label", "")
                    ),
                }
            )
        return row


@dataclass
class BatchJob:
    id: str
    options: AnalysisOptions
    items: list[BatchItem]
    status: str = "queued"       # queued | running | done | cancelled | error
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    error: str = ""
    _cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def completed(self) -> int:
        return sum(1 for i in self.items if i.status in {"done", "error", "skipped"})

    def snapshot(self, include_results: bool = False) -> dict[str, Any]:
        done = [i for i in self.items if i.status == "done"]
        flagged = [i for i in done if i.result and i.result["decision"]["flagged"]]
        data: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "error": self.error,
            "total": len(self.items),
            "completed": self.completed,
            "flagged": len(flagged),
            "percent": round(self.completed / len(self.items) * 100, 1) if self.items else 0.0,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "elapsed": round((self.finished_at or time.time()) - self.created_at, 1),
            "stratum": self.options.stratum,
            "fpr": self.options.fpr,
            "items": [i.summary() for i in self.items],
        }
        if include_results:
            data["results"] = {i.name: i.result for i in done}
        return data


class JobRegistry:
    """In-memory registry of batch jobs, one worker at a time."""

    #: Jobs kept after completion before the oldest are dropped.
    MAX_JOBS = 20

    def __init__(self) -> None:
        self._jobs: dict[str, BatchJob] = {}
        self._lock = threading.Lock()

    def create(
        self,
        documents: list[tuple[str, str]],
        options: AnalysisOptions,
        save_history: bool = True,
    ) -> BatchJob:
        job = BatchJob(
            id=uuid.uuid4().hex[:12],
            options=options,
            items=[BatchItem(name=name, text=text) for name, text in documents],
        )
        with self._lock:
            self._jobs[job.id] = job
            self._evict()
        threading.Thread(
            target=self._run, args=(job, save_history), daemon=True,
            name=f"meld-batch-{job.id}",
        ).start()
        return job

    def _evict(self) -> None:
        if len(self._jobs) <= self.MAX_JOBS:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j.status in {"done", "cancelled", "error"}),
            key=lambda j: j.finished_at or j.created_at,
        )
        for job in finished[: len(self._jobs) - self.MAX_JOBS]:
            self._jobs.pop(job.id, None)

    def _run(self, job: BatchJob, save_history: bool) -> None:
        job.status = "running"
        try:
            for item in job.items:
                if job._cancel.is_set():
                    item.status = "skipped"
                    continue
                item.status = "running"
                try:
                    result = analyzer.analyze(item.text, job.options)
                    item.result = result
                    item.status = "done"
                    if save_history:
                        item.record_id = history.add(
                            result, item.text, label=item.name, source="batch"
                        )
                except Exception as exc:
                    item.status = "error"
                    item.error = f"{type(exc).__name__}: {exc}"
            job.status = "cancelled" if job._cancel.is_set() else "done"
        except Exception as exc:  # pragma: no cover - defensive
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished_at = time.time()

    def get(self, job_id: str) -> BatchJob | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status not in {"queued", "running"}:
            return False
        job._cancel.set()
        return True

    def list(self) -> list[dict[str, Any]]:
        return [
            {
                "id": j.id,
                "status": j.status,
                "total": len(j.items),
                "completed": j.completed,
                "created_at": j.created_at,
            }
            for j in sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        ]


jobs = JobRegistry()
