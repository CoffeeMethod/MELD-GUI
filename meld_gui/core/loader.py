"""Model lifecycle: locate, download, load, unload.

The GUI needs to stay responsive while a 1.6 GB checkpoint is fetched, so the
download runs on a worker thread and publishes progress that the frontend polls.
Loading is guarded by a lock: concurrent requests wait for one load rather than
racing to build several copies of the model.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import REQUIRED_FILES, settings
from .model import Meld
from .thresholds import ThresholdTable


@dataclass
class Progress:
    """Snapshot of an in-flight download or load, polled by the UI."""

    state: str = "idle"          # idle | downloading | loading | ready | error
    message: str = ""
    downloaded: int = 0
    total: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def percent(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(100.0, self.downloaded / self.total * 100.0)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["percent"] = round(self.percent, 2)
        data["elapsed"] = (
            round((self.finished_at or time.time()) - self.started_at, 1)
            if self.started_at
            else 0.0
        )
        return data


class ModelNotReady(RuntimeError):
    """Raised when inference is requested before the checkpoint is available."""


class ModelManager:
    """Owns the single :class:`Meld` instance shared by all requests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: Meld | None = None
        self._tokenizer: Any = None
        self._thresholds: ThresholdTable | None = None
        self._device: str = "cpu"
        self._model_path: Path | None = None
        self._load_seconds: float = 0.0
        self._error: str | None = None
        self.progress = Progress()
        self._worker: threading.Thread | None = None

    # ---------------------------------------------------------------- state
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def local_path(self) -> Path | None:
        """Return a directory holding a complete checkout, if one exists."""
        if settings.model_dir:
            candidate = Path(settings.model_dir)
            return candidate if self._complete(candidate) else None
        try:
            from huggingface_hub import snapshot_download

            # allow_patterns must match the download call: with local_files_only
            # the hub otherwise insists that *every* file in the repo is cached
            # and raises, even though we only ever fetch the six we need.
            path = Path(
                snapshot_download(
                    settings.repo_id,
                    allow_patterns=list(REQUIRED_FILES),
                    local_files_only=True,
                )
            )
            return path if self._complete(path) else None
        except Exception:
            return None

    @staticmethod
    def _complete(path: Path) -> bool:
        return path.is_dir() and all((path / f).exists() for f in REQUIRED_FILES)

    def status(self) -> dict[str, Any]:
        """Everything the model panel in the UI renders."""
        path = self._model_path or self.local_path()
        info: dict[str, Any] = {
            "loaded": self.is_loaded,
            "busy": self.is_busy,
            "downloaded": path is not None,
            "path": str(path) if path else None,
            "repo_id": settings.repo_id,
            "device": self._device if self.is_loaded else settings.resolve_device(),
            "requested_device": settings.device,
            "load_seconds": round(self._load_seconds, 2),
            "error": self._error,
            "progress": self.progress.as_dict(),
            "cuda_available": self._cuda_available(),
        }
        if self._model is not None and self._thresholds is not None:
            cfg = self._model.cfg
            info["model"] = {
                "version": cfg.get("version"),
                "architecture": cfg.get("architecture"),
                "max_length": cfg.get("max_length"),
                "rho": cfg.get("rho"),
                "style_rank": cfg.get("style_rank"),
                "families": cfg.get("families", []),
                "ops": cfg.get("ops", []),
                "n_human_anchors": cfg.get("n_human_anchors"),
                "released_step": cfg.get("released_step"),
                "parameters": sum(p.numel() for p in self._model.parameters()),
            }
            info["calibration"] = {
                "n_human": self._thresholds.n_human(),
                "strata": self._thresholds.describe(),
            }
        return info

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    # ------------------------------------------------------------ accessors
    def require(self) -> tuple[Meld, Any, ThresholdTable, str]:
        if self._model is None or self._tokenizer is None or self._thresholds is None:
            raise ModelNotReady(
                "The MELD checkpoint is not loaded. Download and load it from the "
                "Model panel before running an analysis."
            )
        return self._model, self._tokenizer, self._thresholds, self._device

    # -------------------------------------------------------------- actions
    def ensure_async(self, device: str | None = None) -> dict[str, Any]:
        """Kick off download+load on a worker thread if not already running."""
        with self._lock:
            if self.is_busy:
                return self.progress.as_dict()
            if self.is_loaded and (device is None or device == self._device):
                return self.progress.as_dict()
            self._error = None
            self.progress = Progress(
                state="starting", message="Preparing...", started_at=time.time()
            )
            self._worker = threading.Thread(
                target=self._run_load,
                args=(device,),
                daemon=True,
                name="meld-loader",
            )
            self._worker.start()
            return self.progress.as_dict()

    def load_blocking(self, device: str | None = None) -> None:
        """Synchronous load, for CLI use and tests."""
        path = self.local_path()
        if path is None:
            path = self._download()
        self._load(path, device)

    def _run_load(self, device: str | None) -> None:
        try:
            self.load_blocking(device)
        except Exception as exc:  # surfaced to the UI, never crashes the server
            self._error = f"{type(exc).__name__}: {exc}"
            self.progress.state = "error"
            self.progress.message = self._error
            self.progress.finished_at = time.time()

    def _download(self) -> Path:
        from huggingface_hub import snapshot_download

        self.progress.state = "downloading"
        self.progress.message = f"Downloading {settings.repo_id}"
        progress = self.progress

        # huggingface_hub reports progress through tqdm; swap in a subclass that
        # feeds our Progress object instead of writing to a terminal.
        try:
            from tqdm.auto import tqdm as _tqdm
        except Exception:  # pragma: no cover - tqdm ships with huggingface_hub
            _tqdm = None

        if _tqdm is None:
            return Path(
                snapshot_download(settings.repo_id, allow_patterns=list(REQUIRED_FILES))
            )

        class _UiTqdm(_tqdm):  # type: ignore[misc,valid-type]
            def __init__(self, *a: Any, **kw: Any) -> None:
                kw["disable"] = False
                super().__init__(*a, **kw)
                self._is_bytes = kw.get("unit") == "B"

            def update(self, n: float | None = 1) -> Any:
                out = super().update(n)
                if getattr(self, "_is_bytes", False):
                    progress.downloaded = int(self.n)
                    progress.total = int(self.total or 0)
                return out

        return Path(
            snapshot_download(
                settings.repo_id,
                allow_patterns=list(REQUIRED_FILES),
                tqdm_class=_UiTqdm,
            )
        )

    def _load(self, path: Path, device: str | None) -> None:
        self.progress.state = "loading"
        self.progress.message = "Loading weights"
        started = time.time()

        target = device or settings.resolve_device()
        if target.startswith("cuda") and not self._cuda_available():
            target = "cpu"

        from transformers import AutoTokenizer

        model = Meld(path).to(target)
        tokenizer = AutoTokenizer.from_pretrained(str(path))

        with self._lock:
            self._unload_locked()
            self._model = model
            self._tokenizer = tokenizer
            self._thresholds = ThresholdTable(model.cfg["score_offsets"])
            self._device = target
            self._model_path = path
            self._load_seconds = time.time() - started

        self.progress.state = "ready"
        self.progress.message = f"Loaded on {target}"
        self.progress.finished_at = time.time()

    def unload(self) -> None:
        with self._lock:
            self._unload_locked()
            self.progress = Progress(state="idle", message="Model unloaded")

    def _unload_locked(self) -> None:
        self._model = None
        self._tokenizer = None
        self._thresholds = None
        self._load_seconds = 0.0
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


manager = ModelManager()
