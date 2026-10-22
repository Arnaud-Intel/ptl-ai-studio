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
SAMPLES = "samples"  # not a model: sample footage a demo plays, fetched the same way

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
    # For what is not on the Hub at all and can say so itself: whether it is
    # on disk, how many bytes of it are, and what the whole of it weighs.
    present: Callable[[], bool] | None = field(default=None, compare=False, repr=False)
    on_disk: Callable[[], int] | None = field(default=None, compare=False, repr=False)
    weighs: int | None = None
    # The one state of the repository the demos were tried with, for a
    # repository that is not Intel's or the model's own publisher's: what is
    # fetched is that, not whatever it holds on the day.
    revision: str | None = None
    # The licence its publisher puts on what is fetched, as an SPDX name, and
    # the model it was made from when the build is somebody else's work on
    # it. docs/MODEL_LICENCES.md has where each was read and, for a licence
    # that is not in PERMISSIVE, what was decided about it (BACKLOG R17).
    licence: str = ""
    made_from: str = ""
    # The laptop memory it needs to run, when that is more than most have:
    # under it, and with no discrete GPU to hold it, it is greyed out
    # wherever it is offered (`unfit`, the machine module).
    needs_memory_gb: int | None = None


# Licences that ask for a notice to be kept and nothing else. Anything else on
# a model here needs a decision written down before it is shown for a company.
PERMISSIVE = ("Apache-2.0", "MIT")

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


def _fetch_street_videos() -> None:
    from . import sample_videos

    sample_videos.download_all()


def _street_videos_spec() -> ModelSpec:
    from . import sample_videos

    return ModelSpec(
        "street-videos", "Sample videos for counting (6 clips)", ("smart-city-monitor",), SAMPLES,
        note="What the city monitor plays from disk, fetched from Wikimedia Commons and GitHub where their authors "
             "published them. See sample-data/videos/README.md for licences and credits.",
        fetch=_fetch_street_videos, present=sample_videos.all_present, on_disk=sample_videos.bytes_on_disk,
        weighs=sample_videos.total_bytes(),
        licence="CC0-1.0, CC-BY-3.0, CC-BY-4.0, CC-BY-SA-4.0 and public domain, a clip each",
    )


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("whisper-medium-ov", "Whisper medium (speech translation)", ("live-translation", "meeting-notes"), OPENVINO,
              "OpenVINO/whisper-medium-fp16-ov", licence="Apache-2.0", made_from="openai/whisper-medium",
              note="The default size for translating speech on the OpenVINO engine; the other sizes download on demand."),
    ModelSpec("whisper-base-ov", "Whisper base (speech)", ("voice-assistant",), OPENVINO, "OpenVINO/whisper-base-fp16-ov",
              licence="Apache-2.0", made_from="openai/whisper-base", note="What the voice assistant listens with."),
    ModelSpec("whisper-small-portable", "faster-whisper small (speech)", _SPEECH, PORTABLE, "Systran/faster-whisper-small",
              licence="MIT", made_from="openai/whisper-small"),
    # Meeting notes are written by a larger model on this engine (below).
    ModelSpec("llm-1.5b-ov", "Qwen2.5 1.5B (text)",
              (*(demo for demo in _TEXT if demo != "meeting-notes"), "video-commentary"), OPENVINO,
              "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov", licence="Apache-2.0", made_from="Qwen/Qwen2.5-1.5B-Instruct"),
    ModelSpec("notes-8b-ov", "Qwen3 8B (meeting notes, page plans)", ("meeting-notes", "page-agent"), OPENVINO,
              "OpenVINO/Qwen3-8B-int4-ov", licence="Apache-2.0", made_from="Qwen/Qwen3-8B",
              note="For the page agent, only when its plan is not written on the NPU. Document Q&A answers with it "
                   "too when it is here, and three other demos can be told to."),
    ModelSpec("notes-8b-npu-ov", "Qwen3 8B, NPU build (meeting notes, page plans)", ("meeting-notes", "page-agent"),
              OPENVINO, "OpenVINO/Qwen3-8B-int4-cw-ov", licence="Apache-2.0", made_from="Qwen/Qwen3-8B",
              note="What writes notes and page plans on the NPU. Document Q&A answers with it too when it is here, "
                   "and three other demos can be told to."),
    ModelSpec("flux2-klein-ov", "FLUX.2 klein 4B (draws pictures, redraws a portrait)", ("page-agent", "portrait-studio"), OPENVINO,
              "rupeshs/flux2-klein-4b-int4-ov", revision="2adc62976ad32ca941fc783e2599c45c9d0c3f5a",
              licence="Apache-2.0", made_from="black-forest-labs/FLUX.2-klein-4B",
              note="About 4.6 GB, and as much again the first time it is used: the build is not Intel's, and a copy "
                   "is made that the app can open."),
    ModelSpec("llm-1.5b-portable", "Qwen2.5 1.5B GGUF (text)", _TEXT, PORTABLE, "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
              ("*q4_k_m.gguf",), licence="Apache-2.0"),
    ModelSpec("embed-ov", "Qwen3 embeddings", ("doc-qa", "smart-recall"), OPENVINO, "OpenVINO/Qwen3-Embedding-0.6B-int8-ov",
              licence="Apache-2.0", made_from="Qwen/Qwen3-Embedding-0.6B"),
    ModelSpec("embed-portable", "nomic embeddings GGUF", ("doc-qa", "smart-recall"), PORTABLE,
              "nomic-ai/nomic-embed-text-v1.5-GGUF", ("*Q4_K_M.gguf",), licence="Apache-2.0"),
    ModelSpec("vlm-7b-ov", "Qwen2.5-VL 7B (reads images)", (*_OCR, "video-commentary"), OPENVINO,
              "OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov", licence="Apache-2.0", made_from="Qwen/Qwen2.5-VL-7B-Instruct"),
    ModelSpec("yolo11s-ov", "YOLO11s (detection)", _DETECTION, OPENVINO, "OpenVINO/YOLO11s-int8-ov",
              ("yolo11s.xml", "yolo11s.bin"), licence="AGPL-3.0", made_from="Ultralytics/YOLO11"),
    ModelSpec("detr-portable", "DETR ResNet-50 (detection)", _DETECTION, PORTABLE, "Xenova/detr-resnet-50",
              ("onnx/model_quantized.onnx",), licence="Apache-2.0", made_from="facebook/detr-resnet-50",
              note="The build's own page names no licence; Apache-2.0 is that of the model it was made from."),
    ModelSpec("selfie-segmentation", "Selfie segmentation (background)", ("webcam-effects",), PORTABLE,
              "onnx-community/mediapipe_selfie_segmentation", ("onnx/model_quantized.onnx",), licence="Apache-2.0",
              made_from="Google's MediaPipe selfie segmentation"),
    ModelSpec("coder-30b-ov", "Qwen3-Coder 30B (code, HTML)", (*_CODING, "page-agent"), OPENVINO,
              "OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov", licence="Apache-2.0",
              made_from="Qwen/Qwen3-Coder-30B-A3B-Instruct", needs_memory_gb=32,
              note="The big one: about 17 GB, so 32 GB of memory, or a discrete GPU such as the Arc Pro B60, to run."),
    ModelSpec("coder-1.5b-portable", "Qwen2.5-Coder 1.5B GGUF", _CODING, PORTABLE,
              "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF", ("*q4_k_m.gguf",), licence="Apache-2.0"),
    ModelSpec("openvoice", "OpenVoice (voice cloning)", ("voice-clone-studio",), PORTABLE, "myshell-ai/OpenVoice",
              ("checkpoints/base_speakers/EN/config.json", "checkpoints/base_speakers/EN/checkpoint.pth",
               "checkpoints/base_speakers/EN/en_default_se.pth", "checkpoints/converter/config.json",
               "checkpoints/converter/checkpoint.pth"), licence="MIT"),
    ModelSpec("wavmark", "WavMark (marks a cloned voice)", ("voice-clone-studio",), PORTABLE, "M4869/WavMark",
              ("step59000_*.model.pkl",), licence="MIT",
              note="10 MB. OpenVoice's converter marks what it says with it, and the wavmark package fetches it "
                   "the first time a voice is cloned with that model."),
    ModelSpec("chatterbox", "Chatterbox Turbo (voice cloning)", ("voice-clone-studio",), PORTABLE,
              "ResembleAI/chatterbox-turbo-ONNX",
              ("onnx/*_q4f16.onnx", "onnx/*_q4f16.onnx_data", "tokenizer.json"), licence="MIT",
              made_from="ResembleAI/chatterbox-turbo",
              note="The q4f16 graphs, which is the variant this brick loads."),
    ModelSpec("wake-word", "Wake word 'hey jarvis'", ("voice-assistant",), PORTABLE, licence="CC-BY-NC-SA-4.0",
              note="Only for who chooses a wake word over the button: openwakeword's models are for "
                   "non-commercial use. A few MB, fetched by the package itself.",
              fetch=_fetch_wake_word),
    ModelSpec("silero-vad", "Silero voice detection", ("voice-clone-studio",), PORTABLE, licence="MIT",
              note="A few MB, fetched through torch.hub.", fetch=_fetch_silero_vad),
    _street_videos_spec(),
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
    if spec.on_disk is not None:
        return spec.on_disk()
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
    if spec.present is not None:
        return spec.present()
    if spec.repo_id is None:
        return cached_bytes(spec) > 0 or _library_model_present(spec)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot_download(
            spec.repo_id,
            revision=spec.revision,
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
    if spec.weighs is not None:
        return spec.weighs
    if spec.repo_id is None:
        return None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(spec.repo_id, revision=spec.revision, files_metadata=True)
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
        spec.repo_id, revision=spec.revision, allow_patterns=list(spec.patterns) or None,
        ignore_patterns=list(IGNORED_PATTERNS), **extra
    )


def unfit(spec: ModelSpec) -> str | None:
    """Why this model cannot run on this laptop, or None (machine.unfit_reason)."""
    from .machine import unfit_reason

    return unfit_reason(spec.needs_memory_gb)


def for_demo(demo_id: str) -> list[ModelSpec]:
    return [spec for spec in MODELS if demo_id in spec.demos]
