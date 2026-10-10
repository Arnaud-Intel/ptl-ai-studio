"""The small language models a brick can choose between.

Most bricks that need a few sentences from a language model -- an answer
from documents, an expense line, a spoken reply, a comment in a mood -- use
this brick's default, Qwen2.5-1.5B: 58 tokens a second on the NPU, and the
weakest part of the demos it serves. Qwen3-8B is the other choice: a third
of the speed, and it reads what it is given more closely. Meeting Notes and
the Page Agent's planner have used it since October 2026; this is where the
others find it, so that which build goes on which chip is said once.

The NPU needs a build quantised for it (channel-wise): the standard one does
not compile there. See meeting_notes.session for what was tried.
"""
from __future__ import annotations

from dataclasses import dataclass

from pantherlake_ai_core import npu
from pantherlake_ai_core.engine import Engine


@dataclass(frozen=True)
class LanguageModel:
    key: str
    name: str
    # One line for a menu: what choosing it changes.
    says: str
    # The OpenVINO builds, on the NPU and elsewhere. None: the backend's own
    # default (llm_openvino, llm_portable).
    repo: str | None = None
    repo_npu: str | None = None


QUICK = LanguageModel("1.5b", "Qwen2.5 1.5B", "the fastest")
CAREFUL = LanguageModel(
    "8b", "Qwen3 8B", "more careful, slower",
    repo="OpenVINO/Qwen3-8B-int4-ov", repo_npu="OpenVINO/Qwen3-8B-int4-cw-ov",
)
MODELS: tuple[LanguageModel, ...] = (QUICK, CAREFUL)
BY_KEY = {model.key: model for model in MODELS}
DEFAULT = QUICK.key


def get(key: str | None) -> LanguageModel:
    """The model named `key`; the default for None or ""."""
    try:
        return BY_KEY[key or DEFAULT]
    except KeyError:
        raise ValueError(f"Unknown language model '{key}': one of {', '.join(BY_KEY)}.") from None


def choices(engine: Engine) -> tuple[LanguageModel, ...]:
    """What `engine` can load. The portable engine has the small model only:
    nothing here publishes the 8B in its format."""
    return MODELS if engine == Engine.OPENVINO else (QUICK,)


def on_disk(key: str | None, engine: Engine, device: str) -> bool:
    """Whether the model `key` can be loaded on `device` with nothing to
    download. The backend's own default is taken to be there: it is what
    every installation fetches first."""
    model = get(key)
    if model not in choices(engine):
        return False
    repo = repo_for(model.key, engine, device)
    if repo is None:
        return True
    from pantherlake_ai_core.model_cache import is_repo_cached

    return is_repo_cached(repo)


def preferred(wanted: str | None, engine: Engine, device: str) -> str:
    """The key of the model a brick loads when nobody chose one: `wanted`,
    if this laptop has it for `device`; the default if not. A better answer
    is not worth several gigabytes fetched because somebody pressed Start
    in front of an audience -- whoever wants the model fetches it first, or
    picks it by name."""
    return get(wanted).key if on_disk(wanted, engine, device) else DEFAULT


def repo_for(key: str | None, engine: Engine, device: str) -> str | None:
    """The repository to load for the model `key` on `device`, or None for
    the backend's own default. A model the engine does not have is refused
    here, before anything is downloaded."""
    model = get(key)
    if model not in choices(engine):
        raise ValueError(f"{model.name} needs the OpenVINO engine.")
    if engine != Engine.OPENVINO:
        return None
    return model.repo_npu if npu.is_npu(device) else model.repo
