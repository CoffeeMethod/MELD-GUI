"""Model lifecycle: locate, download, load, unload.

The GUI needs to stay responsive while a ~4 GB checkpoint is fetched, so the
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

from ..config import OPTIONAL_FILES, REQUIRED_FILES, settings
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


class ModelLoadError(RuntimeError):
    """A load failure with a stable ``code`` the UI can branch on."""

    code = "error"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


def _hub_token() -> str | None:
    """Token huggingface_hub would pick up on its own (env var or ``hf auth login``)."""
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def translate_hub_error(exc: BaseException, token_sent: bool) -> ModelLoadError | None:
    """Turn a Hub download failure into an actionable message, or ``None``.

    The MELD repo is gated. Without a token, or with a token whose account has
    not accepted the conditions, the Hub answers 401/403, which raw tracebacks
    present as "Repository not found" or a bare HTTP error.
    """
    try:
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except Exception:  # pragma: no cover - old huggingface_hub
        return None

    status = getattr(getattr(exc, "response", None), "status_code", None)
    repo, url = settings.repo_id, settings.repo_url
    if isinstance(exc, GatedRepoError) or status in (401, 403):
        if token_sent:
            return ModelLoadError(
                f"Hugging Face refused access to {repo}. Open {url} while signed in "
                "as the account that owns the token and accept the access "
                "conditions. The token also needs read permission.",
                "access_denied",
            )
        return ModelLoadError(
            f"{repo} is gated. Open {url} and accept the conditions, create a read "
            "token at https://huggingface.co/settings/tokens, then paste it below "
            "or set MELD_HF_TOKEN.",
            "token_required",
        )
    if isinstance(exc, RepositoryNotFoundError):
        return ModelLoadError(f"Repository {repo} was not found on Hugging Face.", "not_found")
    return None


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
        self._error_code: str | None = None
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
                    allow_patterns=[*REQUIRED_FILES, *OPTIONAL_FILES],
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
            # Snapshot folders are named after the Hub commit they hold.
            "revision": path.name[:12] if path and not settings.model_dir else None,
            "can_update": not settings.model_dir,
            "repo_id": settings.repo_id,
            "device": self._device if self.is_loaded else settings.resolve_device(),
            "requested_device": settings.device,
            "load_seconds": round(self._load_seconds, 2),
            "error": self._error,
            "error_code": self._error_code,
            "needs_token": self._error_code in ("token_required", "access_denied"),
            "token_from_env": bool(settings.hf_token or _hub_token()),
            "repo_url": settings.repo_url,
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

    def threshold_table(self) -> ThresholdTable | None:
        """Calibration of the loaded model, else of the cached checkpoint."""
        if self._thresholds is not None:
            return self._thresholds
        path = self.local_path()
        if path is None:
            return None
        try:
            import json

            with open(path / "meld_config.json", encoding="utf-8") as fh:
                return ThresholdTable(json.load(fh)["score_offsets"])
        except Exception:
            return None

    # ------------------------------------------------------------ accessors
    def require(self) -> tuple[Meld, Any, ThresholdTable, str]:
        if self._model is None or self._tokenizer is None or self._thresholds is None:
            raise ModelNotReady(
                "The MELD checkpoint is not loaded. Download and load it from the "
                "Model panel before running an analysis."
            )
        return self._model, self._tokenizer, self._thresholds, self._device

    # -------------------------------------------------------------- actions
    def ensure_async(
        self,
        device: str | None = None,
        hf_token: str | None = None,
        update: bool = False,
    ) -> dict[str, Any]:
        """Kick off download+load on a worker thread if not already running.

        ``hf_token`` is used for this download only. It travels as a thread
        argument and is never stored on the manager, written to disk or logged.
        ``update`` asks the Hub for the latest revision instead of trusting
        whatever is already in the cache.
        """
        with self._lock:
            if self.is_busy:
                return self.progress.as_dict()
            if (
                not update
                and self.is_loaded
                and (device is None or device == self._device)
            ):
                return self.progress.as_dict()
            self._error = None
            self._error_code = None
            self.progress = Progress(
                state="starting", message="Preparing...", started_at=time.time()
            )
            self._worker = threading.Thread(
                target=self._run_load,
                args=(device, hf_token, update),
                daemon=True,
                name="meld-loader",
            )
            self._worker.start()
            return self.progress.as_dict()

    def load_blocking(
        self,
        device: str | None = None,
        hf_token: str | None = None,
        update: bool = False,
    ) -> None:
        """Synchronous load, for CLI use and tests.

        With ``update`` the cache is bypassed: the Hub is asked for its newest
        revision and only files that changed are downloaded. A fixed
        ``MELD_MODEL_DIR`` is never updated, since the app does not own it.
        """
        path = None if update and not settings.model_dir else self.local_path()
        if path is None:
            path = self._download(hf_token)
        if (
            update
            and self.is_loaded
            and path == self._model_path
            and (device is None or device == self._device)
        ):
            self.progress.state = "ready"
            self.progress.message = "Already on the latest version"
            self.progress.finished_at = time.time()
            return
        self._load(path, device)

    def _run_load(
        self, device: str | None, hf_token: str | None = None, update: bool = False
    ) -> None:
        try:
            self.load_blocking(device, hf_token, update)
        except Exception as exc:  # surfaced to the UI, never crashes the server
            if isinstance(exc, ModelLoadError):
                self._error = str(exc)
                self._error_code = exc.code
            else:
                self._error = f"{type(exc).__name__}: {exc}"
                self._error_code = "error"
            self.progress.state = "error"
            self.progress.message = self._error
            self.progress.finished_at = time.time()

    def _snapshot(self, token: str | None, **kwargs: Any) -> Path:
        """``snapshot_download`` with friendly errors and a completeness check."""
        from huggingface_hub import snapshot_download

        # An empty token must become None so huggingface_hub still falls back to
        # HF_TOKEN or a saved ``hf auth login``.
        token = token or None
        try:
            path = Path(
                snapshot_download(
                    settings.repo_id,
                    allow_patterns=[*REQUIRED_FILES, *OPTIONAL_FILES],
                    token=token,
                    **kwargs,
                )
            )
        except Exception as exc:
            friendly = translate_hub_error(exc, bool(token or _hub_token()))
            if friendly is None:
                raise
            raise friendly from None

        missing = [f for f in REQUIRED_FILES if not (path / f).exists()]
        if missing:
            raise ModelLoadError(
                f"{settings.repo_id} no longer contains: {', '.join(missing)}. "
                "The upstream repo layout changed; update MELD GUI.",
                "incompatible",
            )
        return path

    def _download(self, hf_token: str | None = None) -> Path:
        token = hf_token or settings.hf_token or None

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
            return self._snapshot(token)

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

        return self._snapshot(token, tqdm_class=_UiTqdm)

    def _load(self, path: Path, device: str | None) -> None:
        self.progress.state = "loading"
        self.progress.message = "Loading weights"
        started = time.time()

        target = device or settings.resolve_device()
        if target.startswith("cuda") and not self._cuda_available():
            target = "cpu"

        from transformers import AutoTokenizer

        try:
            model = Meld(path).to(target)
        except RuntimeError as exc:
            text = str(exc)
            if any(k in text for k in ("size mismatch", "Missing key", "Unexpected key")):
                raise ModelLoadError(
                    f"The checkpoint in {path} does not match this app's scoring head. "
                    "The Hugging Face repo was probably updated. Delete that folder so "
                    "it downloads again, or update MELD GUI. "
                    f"({text.splitlines()[0][:160]})",
                    "incompatible",
                ) from exc
            raise
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
