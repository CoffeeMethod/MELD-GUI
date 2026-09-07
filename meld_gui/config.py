"""Application settings.

Every value can be overridden with a ``MELD_``-prefixed environment variable,
which keeps deployment configuration out of the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

#: Upstream repository holding the released checkpoint.
DEFAULT_REPO_ID = "anon-review-meld-2026/meld"

#: Files required for a usable local checkout of the model.
REQUIRED_FILES = (
    "config.json",
    "meld_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _env(name: str, default: str) -> str:
    return os.environ.get(f"MELD_{name}", default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    """Runtime configuration for the server and the model."""

    repo_id: str = field(default_factory=lambda: _env("REPO_ID", DEFAULT_REPO_ID))
    #: Optional explicit path to a local model folder. Empty means "use the
    #: HuggingFace cache and download on demand".
    model_dir: str = field(default_factory=lambda: _env("MODEL_DIR", ""))
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))

    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    reload: bool = field(default_factory=lambda: _env_bool("RELOAD", False))

    #: Load the checkpoint as soon as the server boots rather than on first use.
    eager_load: bool = field(default_factory=lambda: _env_bool("EAGER_LOAD", False))

    data_dir: Path = field(
        default_factory=lambda: Path(_env("DATA_DIR", str(PROJECT_ROOT / "data")))
    )

    #: Token overlap between consecutive windows of a long document.
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 128))
    #: Windows scored per forward pass. Raise on a large GPU, lower on CPU.
    batch_size: int = field(default_factory=lambda: _env_int("BATCH_SIZE", 4))
    #: Largest accepted upload / paste, in characters.
    max_chars: int = field(default_factory=lambda: _env_int("MAX_CHARS", 400_000))
    #: Documents below this word count get an unreliability warning.
    min_words: int = field(default_factory=lambda: _env_int("MIN_WORDS", 100))

    history_limit: int = field(default_factory=lambda: _env_int("HISTORY_LIMIT", 500))

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def history_db(self) -> Path:
        return self.data_dir / "history.sqlite3"

    def resolve_device(self) -> str:
        """Turn ``auto`` into a concrete torch device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # pragma: no cover - torch always present in practice
            pass
        return "cpu"


settings = Settings()
