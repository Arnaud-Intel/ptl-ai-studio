"""Resolving a GGUF whose filename is a pattern: the local cache first, so a
machine that already has the model runs with the network off."""
from __future__ import annotations

import huggingface_hub
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from pantherlake_ai_core import model_cache

REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
WANTED = "qwen2.5-1.5b-instruct-q4_k_m.gguf"


class _FakeApi:
    def list_repo_files(self, repo_id):
        return ["README.md", WANTED, "qwen2.5-1.5b-instruct-q8_0.gguf"]


def _refuse_network(*args, **kwargs):
    raise AssertionError("went to the Hub for a model that is already cached")


def test_a_pattern_is_matched_in_the_cache_without_the_network(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / WANTED).write_bytes(b"gguf")

    def snapshot_download(repo_id, local_files_only=False, allow_patterns=None, **kwargs):
        assert local_files_only, "the cache lookup must never reach the network"
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _refuse_network)
    monkeypatch.setattr(huggingface_hub, "HfApi", _refuse_network)
    downloading = []
    path = model_cache.resolve_gguf(REPO, "*q4_k_m.gguf", on_downloading=lambda: downloading.append(True))
    assert path == str(snapshot / WANTED)
    assert downloading == []  # nothing was fetched, so nothing announced one


def test_a_model_path_is_taken_as_given():
    assert model_cache.resolve_gguf(REPO, "*.gguf", local_path="D:/models/mine.gguf") == "D:/models/mine.gguf"


def test_nothing_cached_asks_the_hub_once_and_announces_the_download(monkeypatch):
    asked = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _missing)
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo_id, filename, **kw: asked.append(filename) or f"/cache/{filename}")
    downloading = []
    path = model_cache.resolve_gguf(REPO, "*q4_k_m.gguf", on_downloading=lambda: downloading.append(True))
    assert (path, asked, downloading) == (f"/cache/{WANTED}", [WANTED], [True])


def test_an_exact_filename_uses_the_hub_helper_which_has_its_own_cache_fallback(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo_id, filename, **kw: f"/cache/{filename}")
    monkeypatch.setattr(model_cache, "is_file_cached", lambda repo_id, filename: True)
    assert model_cache.resolve_gguf(REPO, WANTED) == f"/cache/{WANTED}"


def test_a_pattern_matching_nothing_says_so(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _missing)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: _NoMatches())
    with pytest.raises(FileNotFoundError, match="matches"):
        model_cache.resolve_gguf(REPO, "*.bogus")


def _missing(*args, **kwargs):
    raise LocalEntryNotFoundError("not cached")


class _NoMatches:
    def list_repo_files(self, repo_id):
        return ["README.md"]
