"""Every model the demos can download, so they can be fetched before a show
instead of in front of an audience (BACKLOG R10).

One entry per model, naming the demos that use it and the exact files those
demos fetch -- not the whole repository, which for several of these is far
larger than what is actually loaded. `tests/test_models_prefetch.py` checks
each entry against the brick's own constants, so a model swapped in a brick
can't quietly leave this list behind.

Sizes come from the Hub and need a connection; everything else here works
offline, including "is it already cached", which is the question that
matters on the show machine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable

OPENVINO = "openvino"
PORTABLE = "portable"

# Never part of "do we have this model": a repository's README and its
# .gitattributes are documentation, no demo loads them, and treating them as
# required made a perfectly usable cached model look missing -- then fail to
# "download" on a flaky connection.
IGNORED_PATTERNS = ("*.md", ".gitattributes")


@dataclass(frozen=True)
class ModelSpec:
    key: str  # stable id, used by the API and the CLI
    label: str
    demos: tuple[str, ...]  # registry demo ids
    engine: str
    repo_id: str | None = None  # None for a model its own library fetches
    # Exactly the files the demos load; empty means the whole repository.
    patterns: tuple[str, ...] = ()
    note: str = ""
    # For models that aren't plain Hub files (a library downloads them itself).
    fetch: Callable[[], None] | None = field(default=None, compare=False, repr=False)


_SPEECH = ("live-translation", "meeting-notes", "voice-assistant")
_TEXT = ("doc-qa", "meeting-notes", "voice-assistant", "expense-extract", "smart-recall")
_CODING = ("code-review-assist", "html-creator")
_OCR = ("screen-ocr", "expense-extract", "smart-recall")
_DETECTION = ("object-detection", "smart-city-monitor")


def _fetch_wake_word() -> None:
    import openwakeword.utils

    openwakeword.utils.download_models(["hey jarvis"])


def _fetch_silero_vad() -> None:
    import torch

    torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("whisper-base-ov", "Whisper base (speech)", _SPEECH, OPENVINO, "OpenVINO/whisper-base-fp16-ov",
              note="The launcher's default size on the OpenVINO engine; tiny, medium and large-v3 download on demand."),
    ModelSpec("whisper-small-portable", "faster-whisper small (speech)", _SPEECH, PORTABLE, "Systran/faster-whisper-small"),
    ModelSpec("llm-1.5b-ov", "Qwen2.5 1.5B (text)", _TEXT, OPENVINO, "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"),
    ModelSpec("llm-1.5b-portable", "Qwen2.5 1.5B GGUF (text)", _TEXT, PORTABLE, "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
              ("*q4_k_m.gguf",)),
    ModelSpec("embed-ov", "Qwen3 embeddings", ("doc-qa", "smart-recall"), OPENVINO, "OpenVINO/Qwen3-Embedding-0.6B-int8-ov"),
    ModelSpec("embed-portable", "nomic embeddings GGUF", ("doc-qa", "smart-recall"), PORTABLE,
              "nomic-ai/nomic-embed-text-v1.5-GGUF", ("*Q4_K_M.gguf",)),
    ModelSpec("vlm-7b-ov", "Qwen2.5-VL 7B (reads images)", _OCR, OPENVINO, "OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov"),
    ModelSpec("yolo11s-ov", "YOLO11s (detection)", _DETECTION, OPENVINO, "OpenVINO/YOLO11s-int8-ov",
              ("yolo11s.xml", "yolo11s.bin")),
    ModelSpec("detr-portable", "DETR ResNet-50 (detection)", _DETECTION, PORTABLE, "Xenova/detr-resnet-50",
              ("onnx/model_quantized.onnx",)),
    ModelSpec("selfie-segmentation", "Selfie segmentation (background)", ("webcam-effects",), PORTABLE,
              "onnx-community/mediapipe_selfie_segmentation", ("onnx/model_quantized.onnx",)),
    ModelSpec("coder-30b-ov", "Qwen3-Coder 30B (code, HTML)", _CODING, OPENVINO,
              "OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov", note="The big one: about 17 GB."),
    ModelSpec("coder-1.5b-portable", "Qwen2.5-Coder 1.5B GGUF", _CODING, PORTABLE,
              "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", ("*q4_k_m.gguf",)),
    ModelSpec("openvoice", "OpenVoice (voice cloning)", ("voice-clone-studio",), PORTABLE, "myshell-ai/OpenVoice",
              ("checkpoints/base_speakers/EN/config.json", "checkpoints/base_speakers/EN/checkpoint.pth",
               "checkpoints/base_speakers/EN/en_default_se.pth", "checkpoints/converter/config.json",
               "checkpoints/converter/checkpoint.pth")),
    ModelSpec("chatterbox", "Chatterbox Turbo (voice cloning)", ("voice-clone-studio",), PORTABLE,
              "ResembleAI/chatterbox-turbo-ONNX",
              ("onnx/*_q4f16.onnx", "onnx/*_q4f16.onnx_data", "tokenizer.json"),
              note="The q4f16 graphs, which is the variant this brick loads."),
    ModelSpec("wake-word", "Wake word 'hey jarvis'", ("voice-assistant",), PORTABLE,
              note="A few MB, fetched by openwakeword itself.", fetch=_fetch_wake_word),
    ModelSpec("silero-vad", "Silero voice detection", ("voice-clone-studio",), PORTABLE,
              note="A few MB, fetched through torch.hub.", fetch=_fetch_silero_vad),
)

BY_KEY = {spec.key: spec for spec in MODELS}


def _hub_cache() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def cache_folder(spec: ModelSpec) -> Path | None:
    if not spec.repo_id:
        return None
    return _hub_cache() / f"models--{spec.repo_id.replace('/', '--')}"


def cached_bytes(spec: ModelSpec) -> int:
    """Bytes on disk for this model, partial downloads included -- what makes
    a progress bar move without asking the download backend anything.

    Two cache layouts exist: older hubs keep the data in `blobs/` with
    `snapshots/` linking to it, newer ones (1.x) write straight into
    `snapshots/` and have no `blobs/` at all. Counting only blobs, as this
    did at first, reported zero forever on a 1.x cache.
    """
    folder = cache_folder(spec)
    if folder is None or not folder.exists():
        return 0
    blobs = folder / "blobs"
    if blobs.exists():
        return sum(path.stat().st_size for path in blobs.glob("*") if path.is_file())
    return sum(
        path.stat().st_size
        for path in folder.rglob("*")
        if path.is_file() and not path.is_symlink()  # a link would count its target twice
    )


def is_cached(spec: ModelSpec) -> bool:
    """True when the demo could load this model with no network at all."""
    if spec.repo_id is None:
        return cached_bytes(spec) > 0 or _library_model_present(spec)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot_download(
            spec.repo_id,
            local_files_only=True,
            allow_patterns=list(spec.patterns) or None,
            ignore_patterns=list(IGNORED_PATTERNS),
        )
        return True
    except (LocalEntryNotFoundError, OSError):
        return False


def _library_model_present(spec: ModelSpec) -> bool:
    """Wake word and voice detection keep their own caches, outside the Hub's."""
    if spec.key == "wake-word":
        try:
            import openwakeword

            models = Path(openwakeword.__file__).parent / "resources" / "models"
            return any(models.glob("hey_jarvis*"))
        except Exception:
            return False
    if spec.key == "silero-vad":
        return (Path.home() / ".cache" / "torch" / "hub" / "snakers4_silero-vad_master").exists()
    return False


def remote_size(spec: ModelSpec) -> int | None:
    """What this model would download, from the Hub. None when that can't be
    asked (no connection, or a model its own library fetches)."""
    if spec.repo_id is None:
        return None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(spec.repo_id, files_metadata=True)
        return sum(
            sibling.size or 0
            for sibling in info.siblings
            if (not spec.patterns or any(fnmatch(sibling.rfilename, pattern) for pattern in spec.patterns))
            and not any(fnmatch(sibling.rfilename, pattern) for pattern in IGNORED_PATTERNS)
        )
    except Exception:
        return None


def _reporting_tqdm(on_progress: Callable[[int], None]):
    """A progress bar the Hub client updates as a download proceeds.

    Only bars that count bytes *and* know their total are followed, and each
    reports its own absolute position: the two transfer backends behave
    differently -- the Xet one runs an extra bar for network bytes, which are
    fewer than the file's (it de-duplicates), and its file bar only learns
    its total once the transfer starts. Summing every update would count the
    same bytes twice; taking each bar's position, capped at its total, does
    not. Watching the cache folder instead doesn't work at all there: the
    file is assembled elsewhere and appears whole at the end.
    """
    from tqdm.auto import tqdm as _tqdm

    bars: dict[int, int] = {}

    class _Reporter(_tqdm):
        def __init__(self, *args, **kwargs):
            # Drawn by us, not by the Hub: its bars would scribble over the
            # command's own line and mean nothing in the launcher.
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            self._counted = 0

        def update(self, n=1):
            result = super().update(n)
            self._counted += int(n or 0)  # self.n stays put while disabled
            if getattr(self, "unit", "") == "B" and (self.total or 0) > 0:
                bars[id(self)] = min(self._counted, int(self.total))
                on_progress(sum(bars.values()))
            return result

    return _Reporter


def download(spec: ModelSpec, on_progress: Callable[[int], None] | None = None) -> None:
    """Fetch exactly what the demos load. Resumes a partial download, and
    does nothing when everything is already cached. `on_progress`, if given,
    is called with the bytes fetched so far for this model."""
    if spec.fetch is not None:
        spec.fetch()
        return
    from huggingface_hub import snapshot_download

    extra = {"tqdm_class": _reporting_tqdm(on_progress)} if on_progress is not None else {}
    snapshot_download(
        spec.repo_id, allow_patterns=list(spec.patterns) or None, ignore_patterns=list(IGNORED_PATTERNS), **extra
    )


def for_demo(demo_id: str) -> list[ModelSpec]:
    return [spec for spec in MODELS if demo_id in spec.demos]
