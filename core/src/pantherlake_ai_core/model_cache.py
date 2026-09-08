"""Hugging Face Hub model resolution shared by every brick: "the local path
for this model, fetched first if it isn't cached yet", plus a cheap
presence check so a caller can tell "about to download" from "loading from
local disk" *before* the slow, blocking fetch -- without hooking into
per-file download progress, which would need separate instrumentation for
every download backend the bricks use (huggingface_hub, model_api, ...).

    model_dir = resolve_snapshot("OpenVINO/whisper-base-fp16-ov", local_dir=args.model_path, on_downloading=cb)
    onnx_path = resolve_file("Xenova/detr-resnet-50", "onnx/model_quantized.onnx", on_downloading=cb)

`on_downloading` fires right before a fetch that actually has to happen --
never for a cache hit -- which is what lets the launcher show "Downloading
(first run only)" instead of a generic "Loading" for a multi-GB model.
"""
from __future__ import annotations

from typing import Callable


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


def is_file_cached(repo_id: str, filename: str) -> bool:
    """True if this one file of `repo_id` is already in the local cache."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        hf_hub_download(repo_id, filename, local_files_only=True)
        return True
    except LocalEntryNotFoundError:
        return False


def resolve_snapshot(
    repo_id: str,
    *,
    local_dir: str | None = None,
    on_downloading: Callable[[], None] | None = None,
) -> str:
    """Path to a model directory: `local_dir` as-is when given (a model the
    user converted or placed themselves -- what a brick's --model-path
    means, so it's never second-guessed), else `repo_id`'s snapshot in the
    Hub cache, downloaded first if it isn't there yet."""
    if local_dir:
        return local_dir
    from huggingface_hub import snapshot_download

    if on_downloading is not None and not is_repo_cached(repo_id):
        on_downloading()
    return snapshot_download(repo_id)


def resolve_file(
    repo_id: str,
    filename: str,
    *,
    local_path: str | None = None,
    on_downloading: Callable[[], None] | None = None,
) -> str:
    """Path to one model file: `local_path` as-is when given, else that
    file of `repo_id` from the Hub cache, downloaded first if needed."""
    if local_path:
        return local_path
    from huggingface_hub import hf_hub_download

    if on_downloading is not None and not is_file_cached(repo_id, filename):
        on_downloading()
    return hf_hub_download(repo_id, filename)
