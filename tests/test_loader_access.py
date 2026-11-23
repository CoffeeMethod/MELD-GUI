"""Gated-repo handling: token plumbing and error messages. No network, no model."""

from __future__ import annotations

import sys
from pathlib import Path
import types

import pytest

from meld_gui.config import OPTIONAL_FILES, REQUIRED_FILES, Settings
from meld_gui.core import loader
from meld_gui.core.loader import ModelLoadError, ModelManager, translate_hub_error


def test_special_tokens_map_is_optional():
    assert "special_tokens_map.json" not in REQUIRED_FILES
    assert "special_tokens_map.json" in OPTIONAL_FILES


def test_token_read_from_env(monkeypatch):
    for name in ("MELD_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    assert Settings().hf_token == ""
    monkeypatch.setenv("HF_TOKEN", "hf_plain")
    assert Settings().hf_token == "hf_plain"
    monkeypatch.setenv("MELD_HF_TOKEN", "hf_meld")
    assert Settings().hf_token == "hf_meld"


def test_token_hidden_from_repr(monkeypatch):
    monkeypatch.setenv("MELD_HF_TOKEN", "hf_secret_value")
    assert "hf_secret_value" not in repr(Settings())


def _gated_error():
    from huggingface_hub.utils import GatedRepoError

    class _Resp:
        status_code = 401
        headers = {"X-Error-Code": "GatedRepo"}
        request = None  # huggingface_hub >= 1.0 reads response.request

    return GatedRepoError("gated", response=_Resp())


def test_gated_without_token_asks_for_one():
    pytest.importorskip("huggingface_hub")
    err = translate_hub_error(_gated_error(), token_sent=False)
    assert isinstance(err, ModelLoadError) and err.code == "token_required"
    assert "MELD_HF_TOKEN" in str(err)


def test_gated_with_token_says_accept_conditions():
    pytest.importorskip("huggingface_hub")
    err = translate_hub_error(_gated_error(), token_sent=True)
    assert err.code == "access_denied"
    assert "accept" in str(err).lower()


def test_unrelated_errors_pass_through():
    pytest.importorskip("huggingface_hub")
    assert translate_hub_error(ValueError("boom"), token_sent=False) is None


def test_token_reaches_snapshot_download_and_is_not_stored(monkeypatch, tmp_path):
    seen = {}

    def fake_snapshot(repo_id, **kwargs):
        seen.update(kwargs)
        for name in REQUIRED_FILES:
            (tmp_path / name).write_text("x")
        return str(tmp_path)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot
    fake_hub.get_token = lambda: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    mgr = ModelManager()
    path = mgr._download("hf_from_ui")
    assert seen["token"] == "hf_from_ui"
    assert path == tmp_path
    assert "hf_from_ui" not in repr(vars(mgr))


def test_incomplete_snapshot_is_reported(monkeypatch, tmp_path):
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda repo_id, **kw: str(tmp_path)  # empty folder
    fake_hub.get_token = lambda: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(ModelLoadError) as info:
        ModelManager()._download("hf_x")
    assert info.value.code == "incompatible"
    assert "model.safetensors" in str(info.value)


def test_status_reports_needs_token(monkeypatch):
    mgr = ModelManager()
    mgr._error, mgr._error_code = "gated", "token_required"
    monkeypatch.setattr(mgr, "local_path", lambda: None)
    monkeypatch.setattr(loader, "_hub_token", lambda: None)
    status = mgr.status()
    assert status["needs_token"] is True
    assert status["error_code"] == "token_required"
    assert status["repo_url"].startswith("https://huggingface.co/")


def test_update_bypasses_cache_and_asks_the_hub(monkeypatch):
    from meld_gui.core import loader as loader_mod

    mgr = loader_mod.ModelManager()
    calls = []
    monkeypatch.setattr(loader_mod.settings, "model_dir", "")
    monkeypatch.setattr(mgr, "local_path", lambda: calls.append("cache") or Path("old"))
    monkeypatch.setattr(mgr, "_download", lambda tok=None: calls.append("hub") or Path("new"))
    monkeypatch.setattr(mgr, "_load", lambda path, device: calls.append(path))

    mgr.load_blocking(update=False)
    assert calls == ["cache", Path("old")]

    calls.clear()
    mgr.load_blocking(update=True)
    assert calls == ["hub", Path("new")]
