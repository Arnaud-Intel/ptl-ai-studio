"""Cheap presence check for Hugging Face Hub model snapshots, so a caller
can tell "about to download" from "loading from local disk" *before* the
slow, blocking load call -- without hooking into per-file download
progress, which would need separate instrumentation for every download
backend the bricks use (huggingface_hub, model_api, coqui, ...).
"""
from __future__ import annotations


def is_repo_cached(repo_id: str) -> bool:
    """True if `repo_id`'s full snapshot is already in the local Hugging
    Face Hub cache (loading it needs no network), False if any part of it
    would have to be fetched first."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot_download(repo_id, local_files_only=True)
        return True
    except LocalEntryNotFoundError:
        return False
