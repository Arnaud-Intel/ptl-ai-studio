"""Where the image model comes from, and how the build on the Hub is made
one that OpenVINO GenAI opens.

The model is FLUX.2 [klein] 4B (Black Forest Labs, Apache-2.0): it draws a
picture from a sentence *and a picture to start from*, in four steps, which
is what restyling a portrait needs. The image model the Page Agent draws
with (FLUX.1-schnell) cannot do it: given a photo, it hands the photo back,
or draws somebody else (tried 2026-10-11, see the README).

Intel publishes no OpenVINO build of klein (2026-10-12). The one used here
was exported for another program, FastSD CPU, with the same exporter, and
differs from what `openvino_genai.Image2ImagePipeline` asks for in three
small ways -- none of them in the weights:

- its text encoder gives one output, the three layers the model reads set
  side by side; the pipeline asks the encoder for those three layers one by
  one, as `hidden_states.9`, `.18` and `.27`;
- its transformer takes the positions of its tokens (`img_ids`, `txt_ids`)
  as `[tokens, 4]`; the pipeline hands them over as `[1, tokens, 4]`;
- the figures its VAE is normalised with are in `vae_bn_stats.npz`; the
  pipeline reads them from the VAE's two configs.

`prepare` writes a copy with those three things changed, once, beside the
app's other caches (about 4.4 GB: the two large files are written again,
the rest is linked). A build that already is what the pipeline asks for --
one exported with a recent optimum-intel, or Intel's when there is one --
is used as it stands: `needs` finds nothing to change.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Callable

REPO = "rupeshs/flux2-klein-4b-int4-ov"
# The state of that repository this was tried with. It is somebody else's:
# what is fetched is this, not whatever it holds on the day.
REVISION = "2adc62976ad32ca941fc783e2599c45c9d0c3f5a"

PREPARED_ROOT = Path.home() / ".cache" / "pantherlake-ai-studio" / "models"
_LAYERS = (9, 18, 27)  # the text encoder's layers the model reads
_MARKER = "prepared.json"
_FORMAT = 1  # of the prepared copy: raised when `prepare` changes what it writes

TEXT_ENCODER, TRANSFORMER, VAE = "text encoder", "transformer", "vae"


def _snapshot(on_downloading: Callable[[], None] | None) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(snapshot_download(REPO, revision=REVISION, local_files_only=True))
    except (LocalEntryNotFoundError, OSError):
        if on_downloading is not None:
            on_downloading()
        return Path(snapshot_download(REPO, revision=REVISION))


def needs(model_dir: Path) -> list[str]:
    """Which of the three changes this build needs before the pipeline can
    open it; none for a build made for the pipeline."""
    import openvino as ov

    core = ov.Core()
    todo = []
    encoder = core.read_model(model_dir / "text_encoder" / "openvino_model.xml")
    names = {name for output in encoder.outputs for name in output.get_names()}
    if not {f"hidden_states.{layer}" for layer in _LAYERS} <= names:
        todo.append(TEXT_ENCODER)
    transformer = core.read_model(model_dir / "transformer" / "openvino_model.xml")
    for port in transformer.inputs:
        if port.get_names() & {"img_ids", "txt_ids"} and port.get_partial_shape().rank.get_length() == 2:
            todo.append(TRANSFORMER)
            break
    for part in ("vae_decoder", "vae_encoder"):
        config = json.loads((model_dir / part / "config.json").read_text(encoding="utf-8"))
        if "bn_running_mean_data" not in config or "bn_running_var_data" not in config:
            todo.append(VAE)
            break
    return todo


def split_text_encoder(source: Path, target: Path) -> None:
    """Save the encoder again with its one output cut back into the three
    layers it was made of, under the names the pipeline asks for."""
    import openvino as ov
    from openvino import opset13 as ops

    model = ov.Core().read_model(source)
    whole = model.outputs[0].get_node().input_value(0)
    width = whole.get_partial_shape()[2].get_length() // len(_LAYERS)
    results = []
    for index, layer in enumerate(_LAYERS):
        piece = ops.slice(whole, [index * width], [(index + 1) * width], [1], [2])
        piece.output(0).get_tensor().set_names({f"hidden_states.{layer}"})
        results.append(ops.result(piece))
    ov.save_model(ov.Model(results, model.get_parameters(), "text_encoder"), target, compress_to_fp16=False)


def batch_token_positions(source: Path, target: Path) -> None:
    """Save the transformer again taking `img_ids` and `txt_ids` with the
    batch in front, as the pipeline hands them over."""
    import openvino as ov
    from openvino import opset13 as ops

    model = ov.Core().read_model(source)
    for name in ("img_ids", "txt_ids"):
        old = next(parameter for parameter in model.get_parameters() if name in parameter.output(0).get_names())
        if old.get_partial_shape().rank.get_length() != 2:
            continue
        new = ops.parameter([-1, -1, old.get_partial_shape()[1].get_length()], old.get_element_type(), name=name)
        new.output(0).get_tensor().set_names({name})
        flat = ops.squeeze(new, ops.constant([0], ov.Type.i64))
        for consumer in list(old.output(0).get_target_inputs()):
            consumer.replace_source_output(flat.output(0))
        model.add_parameters([new])
        model.remove_parameter(old)
    model.validate_nodes_and_infer_types()
    ov.save_model(model, target, compress_to_fp16=False)


def vae_config_with_figures(config: dict, stats) -> dict:
    """The VAE's config with the figures it is normalised with, where the
    pipeline looks for them. `stats` is the build's `vae_bn_stats.npz`."""
    return {
        **config,
        "bn_eps": float(stats["batch_norm_eps"]),
        "bn_running_mean_data": [float(value) for value in stats["running_mean"]],
        "bn_running_var_data": [float(value) for value in stats["running_var"]],
    }


def _link(source: Path, target: Path) -> None:
    """The same file under a second name -- or a copy, where the disk will
    not link (another drive, a filesystem without links)."""
    try:
        os.link(source.resolve(), target)
    except OSError:
        shutil.copyfile(source, target)


def prepare(source: Path, target: Path, todo: list[str]) -> Path:
    """Write at `target` the copy of the build at `source` the pipeline
    opens. Built beside it and moved into place when whole: a copy that is
    there is a copy that is complete."""
    import numpy as np

    building = target.with_name(target.name + ".part")
    shutil.rmtree(building, ignore_errors=True)
    shutil.rmtree(target, ignore_errors=True)
    rewritten = set()
    if TEXT_ENCODER in todo:
        rewritten |= {"text_encoder/openvino_model.xml", "text_encoder/openvino_model.bin"}
    if TRANSFORMER in todo:
        rewritten |= {"transformer/openvino_model.xml", "transformer/openvino_model.bin"}
    if VAE in todo:
        rewritten |= {"vae_decoder/config.json", "vae_encoder/config.json"}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        if path.is_dir() or relative in rewritten:
            continue
        to = building / relative
        to.parent.mkdir(parents=True, exist_ok=True)
        _link(path, to)
    if TEXT_ENCODER in todo:
        split_text_encoder(source / "text_encoder" / "openvino_model.xml", building / "text_encoder" / "openvino_model.xml")
    if TRANSFORMER in todo:
        batch_token_positions(source / "transformer" / "openvino_model.xml", building / "transformer" / "openvino_model.xml")
    if VAE in todo:
        with np.load(source / "vae_bn_stats.npz") as stats:
            for part in ("vae_decoder", "vae_encoder"):
                config = json.loads((source / part / "config.json").read_text(encoding="utf-8"))
                (building / part).mkdir(parents=True, exist_ok=True)
                (building / part / "config.json").write_text(
                    json.dumps(vae_config_with_figures(config, stats), indent=2), encoding="utf-8")
    (building / _MARKER).write_text(json.dumps({"format": _FORMAT, "from": str(source), "changed": todo}), encoding="utf-8")
    building.replace(target)
    return target


def _prepared(target: Path) -> bool:
    try:
        return json.loads((target / _MARKER).read_text(encoding="utf-8")).get("format") == _FORMAT
    except (OSError, ValueError):
        return False


def resolve(
    model_dir: str | None = None,
    *,
    on_downloading: Callable[[], None] | None = None,
    on_preparing: Callable[[], None] | None = None,
) -> str:
    """The folder to hand the pipeline: the build named by `model_dir` (a
    brick's --model-path), else the one on the Hub, fetched first if it is
    not here -- in either case prepared first if it needs to be."""
    if model_dir:
        source = Path(model_dir)
        target = PREPARED_ROOT / ("prepared-" + hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:12])
    else:
        source = _snapshot(on_downloading)
        target = PREPARED_ROOT / f"flux2-klein-4b-int4-{REVISION[:8]}"
    if _prepared(target):
        return str(target)
    todo = needs(source)
    if not todo:
        return str(source)
    if on_preparing is not None:
        on_preparing()
    target.parent.mkdir(parents=True, exist_ok=True)
    return str(prepare(source, target, todo))
