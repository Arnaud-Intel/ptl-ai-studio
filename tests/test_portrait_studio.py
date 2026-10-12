"""Portrait Studio: a photo of a person drawn again in a chosen style
(bricks/portrait-studio, and its runner and route in the launcher).

No image model here. What is checked is what surrounds it: what the styles
tell the model, what is made of a photo before it is handed over, how the
build on the Hub is made one the pipeline opens (on models of a few numbers,
where OpenVINO is installed), and what the launcher refuses before it loads
anything."""
from __future__ import annotations

import base64
import io
import json
import re
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from launcher import app as launcher_app
from launcher import portrait_studio_runner
from launcher.errors import Conflict
from launcher.registry import REGISTRY
from pantherlake_ai_core import models as inventory
from pantherlake_ai_core import flux2_klein as klein
from portrait_studio import photo, styles
from portrait_studio.stylist import STEPS, Restyled


def _jpeg(width=64, height=48, colour=(200, 120, 40), exif=None) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(out, format="JPEG", **({"exif": exif} if exif else {}))
    return out.getvalue()


# --- the styles ---------------------------------------------------------------------------


def test_every_style_says_what_to_make_of_the_photo_and_what_to_leave_alone():
    assert len({style.id for style in styles.STYLES}) == len(styles.STYLES) >= 6
    for style in styles.STYLES:
        assert style.name and style.hint and style.prompt.startswith(style.does)
        assert "photo" in style.does  # an instruction about the photo it is given, not a picture described from nothing
        assert "keep the person" in style.prompt and "exactly as they are" in style.prompt
    assert {"cartoon", "corporate", "manga"} <= {style.id for style in styles.STYLES}  # the three that were asked for
    assert styles.get("manga").name == "Manga"
    with pytest.raises(ValueError, match="No style called 'cubist'.*cartoon"):
        styles.get("cubist")


def test_no_style_names_something_a_person_may_not_have():
    """The model draws every word it is given: "keep the beard" put a beard
    on a woman, "dark or light skin" made a light-skinned woman dark."""
    for style in styles.STYLES:
        told = style.prompt.lower()
        for word in ("beard", "glasses", "moustache", "facial hair", "freckles", "skin", "man", "woman", "young", "old",
                     "dark", "blonde", "bald"):
            if style.id == "corporate" and word == "dark":
                continue  # a dark suit: said of the clothes, not of the person
            assert not re.search(rf"\b{word}\b", told), f"{style.id} says '{word}'"
    # A style without colour does not ask for the person's colours to be kept; one that dresses them keeps the person only.
    assert "their colours" not in styles.get("manga").prompt and "their colours" not in styles.get("sketch").prompt
    assert "their colours" in styles.get("cartoon").prompt
    assert "clothes" in styles.get("corporate").keeps and "lettering" in styles.get("comic").does


# --- the photo ----------------------------------------------------------------------------


def test_a_photo_is_read_upright_and_cut_to_the_square_in_its_middle():
    from PIL import Image

    wide = Image.new("RGB", (160, 90), "white")
    wide.paste((255, 0, 0), (0, 0, 35, 90))  # a red band on the left: outside the middle square
    out = io.BytesIO()
    wide.save(out, format="PNG")
    cut = photo.square(photo.read(out.getvalue()), 512)
    assert cut.size == (512, 512) and cut.getpixel((2, 256)) == (255, 255, 255)  # the sides are gone
    # A phone says which way up its picture is beside the picture: it is turned before anything else.
    exif = Image.Exif()
    exif[0x0112] = 6  # "turn me a quarter"
    turned = photo.read(_jpeg(64, 48, exif=exif.tobytes()))
    assert turned.size == (48, 64) and turned.mode == "RGB"
    with pytest.raises(ValueError, match="512, 768, 1024"):
        photo.square(cut, 640)
    assert photo.DEFAULT_SIZE in photo.SIZES and all(size % 16 == 0 for size in photo.SIZES)


@pytest.mark.parametrize("data, says", [
    (b"", "no photo"),
    (b"not a picture at all", "could not be read as a picture"),
    (b"\xff\xd8\xff" + b"0" * 64, "could not be read as a picture"),
])
def test_what_is_not_a_photo_is_refused_in_plain_words(data, says):
    with pytest.raises(ValueError, match=says):
        photo.read(data)


def test_a_photo_too_large_is_refused_before_it_is_decoded(monkeypatch):
    monkeypatch.setattr(photo, "MAX_BYTES", 100)
    with pytest.raises(ValueError, match="use one under"):
        photo.read(_jpeg(200, 200))


# --- the model: making the build on the Hub one the pipeline opens ------------------------


def _write_build(root, *, for_the_pipeline=False):
    """A FLUX.2 klein build in miniature, laid out as on the Hub: a text
    encoder and a transformer of a few numbers, the VAE's two configs, and
    the files nothing changes."""
    import numpy as np
    import openvino as ov
    from openvino import opset13 as ops

    ids = ops.parameter([-1, -1], ov.Type.i64, name="input_ids")
    ids.output(0).get_tensor().set_names({"input_ids"})
    # Twelve numbers a token, the first four counting from the token, the next from ten times it, the last from a hundred.
    spread = ops.multiply(ops.unsqueeze(ops.convert(ids, "f32"), ops.constant([2], ov.Type.i64)),
                          ops.constant(np.array([1] * 4 + [10] * 4 + [100] * 4, dtype=np.float32)))
    if for_the_pipeline:
        outs = []
        for index, layer in enumerate((9, 18, 27)):
            piece = ops.slice(spread, [index * 4], [index * 4 + 4], [1], [2])
            piece.output(0).get_tensor().set_names({f"hidden_states.{layer}"})
            outs.append(ops.result(piece))
        encoder = ov.Model(outs, [ids], "text_encoder")
    else:
        spread.output(0).get_tensor().set_names({"prompt_embeds"})
        encoder = ov.Model([ops.result(spread)], [ids], "text_encoder")
    (root / "text_encoder").mkdir(parents=True)
    ov.save_model(encoder, root / "text_encoder" / "openvino_model.xml")

    hidden = ops.parameter([-1, -1, 4], ov.Type.f32, name="hidden_states")
    hidden.output(0).get_tensor().set_names({"hidden_states"})
    positions = []
    for name in ("img_ids", "txt_ids"):
        port = ops.parameter([-1, -1, 4] if for_the_pipeline else [-1, 4], ov.Type.f32, name=name)
        port.output(0).get_tensor().set_names({name})
        positions.append(port)
    flat = [ops.squeeze(port, ops.constant([0], ov.Type.i64)) if for_the_pipeline else port for port in positions]
    total = ops.add(ops.reduce_sum(flat[0], ops.constant([0, 1], ov.Type.i64)), ops.reduce_sum(flat[1], ops.constant([0, 1], ov.Type.i64)))
    out = ops.add(hidden, total)
    out.output(0).get_tensor().set_names({"out_sample"})
    (root / "transformer").mkdir()
    ov.save_model(ov.Model([ops.result(out)], [hidden, *positions], "transformer"), root / "transformer" / "openvino_model.xml")

    figures = {"bn_eps": 0.0001, "bn_running_mean_data": [0.5, -0.5], "bn_running_var_data": [3.0, 3.25]} if for_the_pipeline else {}
    for part in ("vae_decoder", "vae_encoder"):
        (root / part).mkdir()
        (root / part / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKLFlux2", **figures}), encoding="utf-8")
        (root / part / "openvino_model.bin").write_bytes(b"weights of the " + part.encode())
    if not for_the_pipeline:
        np.savez(root / "vae_bn_stats.npz", running_mean=np.array([0.5, -0.5], dtype=np.float32),
                 running_var=np.array([3.0, 3.25], dtype=np.float32), batch_norm_eps=np.float32(0.0001))
    (root / "model_index.json").write_text('{"_class_name": "Flux2KleinPipeline"}', encoding="utf-8")
    (root / "tokenizer").mkdir()
    (root / "tokenizer" / "tokenizer.json").write_bytes(b"x" * 2_000_000)  # large enough to be linked rather than copied
    return root


def test_the_hub_build_is_made_one_the_pipeline_opens_and_only_once(tmp_path, monkeypatch):
    pytest.importorskip("openvino")
    import numpy as np
    import openvino as ov

    source = _write_build(tmp_path / "hub")
    monkeypatch.setattr(klein, "PREPARED_ROOT", tmp_path / "prepared")
    assert klein.needs(source) == [klein.TEXT_ENCODER, klein.TRANSFORMER, klein.VAE]
    said = []
    made = klein.resolve(str(source), on_preparing=lambda: said.append("preparing"))
    assert said == ["preparing"] and made != str(source) and made.startswith(str(tmp_path / "prepared"))
    from pathlib import Path

    made = Path(made)
    assert klein.needs(made) == []  # nothing left to change
    core = ov.Core()
    # The text encoder: the same numbers, handed over as the three layers by name.
    before = core.compile_model(source / "text_encoder" / "openvino_model.xml", "CPU")
    after = core.compile_model(made / "text_encoder" / "openvino_model.xml", "CPU")
    tokens = np.array([[1, 2, 3]], dtype=np.int64)
    whole = before(tokens)[before.output(0)]
    pieces = after(tokens)
    assert [sorted(port.get_names()) for port in after.outputs] == [["hidden_states.9"], ["hidden_states.18"], ["hidden_states.27"]]
    for index, port in enumerate(after.outputs):
        assert np.array_equal(pieces[port], whole[:, :, index * 4:(index + 1) * 4])
    # The transformer: the same answer, with the positions taken the way the pipeline hands them over.
    hidden = np.ones((1, 5, 4), dtype=np.float32)
    image_ids, text_ids = np.full((5, 4), 2, dtype=np.float32), np.full((3, 4), 1, dtype=np.float32)
    old = core.compile_model(source / "transformer" / "openvino_model.xml", "CPU")
    new = core.compile_model(made / "transformer" / "openvino_model.xml", "CPU")
    assert [str(port.get_partial_shape()) for port in new.inputs if port.get_names() & {"img_ids", "txt_ids"}] == ["[?,?,4]", "[?,?,4]"]
    wanted = old({"hidden_states": hidden, "img_ids": image_ids, "txt_ids": text_ids})[old.output(0)]
    assert np.array_equal(new({"hidden_states": hidden, "img_ids": image_ids[None], "txt_ids": text_ids[None]})[new.output(0)], wanted)
    # The VAE's figures are where the pipeline reads them; what was not changed is the very same file.
    for part in ("vae_decoder", "vae_encoder"):
        config = json.loads((made / part / "config.json").read_text(encoding="utf-8"))
        assert config["bn_running_mean_data"] == [0.5, -0.5] and config["bn_running_var_data"] == [3.0, 3.25]
        assert config["bn_eps"] == pytest.approx(0.0001) and config["_class_name"] == "AutoencoderKLFlux2"
        assert (made / part / "openvino_model.bin").read_bytes() == (source / part / "openvino_model.bin").read_bytes()
    assert (made / "tokenizer" / "tokenizer.json").stat().st_size == 2_000_000 and (made / "model_index.json").exists()
    assert not made.with_name(made.name + ".part").exists()
    # Asked again, the copy that is there is the answer: nothing is prepared twice.
    assert klein.resolve(str(source), on_preparing=lambda: said.append("again")) == str(made) and said == ["preparing"]
    # A copy that was never finished is not a copy: it is made again.
    (made / "prepared.json").unlink()
    assert klein.resolve(str(source), on_preparing=lambda: said.append("again")) == str(made) and said == ["preparing", "again"]


def test_a_build_made_for_the_pipeline_is_used_as_it_stands(tmp_path, monkeypatch):
    pytest.importorskip("openvino")
    source = _write_build(tmp_path / "intel", for_the_pipeline=True)
    monkeypatch.setattr(klein, "PREPARED_ROOT", tmp_path / "prepared")
    said = []
    assert klein.needs(source) == []
    assert klein.resolve(str(source), on_preparing=lambda: said.append("preparing")) == str(source)
    assert said == [] and not (tmp_path / "prepared").exists()


def test_the_build_is_fetched_at_the_revision_it_was_tried_with_and_says_when_it_downloads(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    asked, said = [], []

    def snapshot_download(repo, revision=None, local_files_only=False, **kwargs):
        asked.append((repo, revision, local_files_only))
        if local_files_only:
            raise LocalEntryNotFoundError("not here yet")
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert klein._snapshot(lambda: said.append("downloading")) == tmp_path
    assert asked == [(klein.REPO, klein.REVISION, True), (klein.REPO, klein.REVISION, False)] and said == ["downloading"]
    assert len(klein.REVISION) == 40  # one commit, not a branch that moves


def test_the_inventory_lists_the_build_for_the_brick_at_that_same_revision(monkeypatch):
    spec = next(spec for spec in inventory.MODELS if spec.repo_id == klein.REPO)
    # Two bricks draw with it: the Page Agent from a description, this one from a photo.
    assert spec.demos == ("page-agent", "portrait-studio") and spec.revision == klein.REVISION and spec.engine == inventory.OPENVINO
    assert not [other for other in inventory.MODELS if "schnell" in (other.repo_id or "").lower()]  # the model it replaced is gone
    # Nothing else in the inventory is held at a revision: the others are Intel's builds or the model's own publisher's.
    assert [other.key for other in inventory.MODELS if other.revision] == [spec.key]
    import huggingface_hub

    asked = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **kwargs: asked.append((repo, kwargs.get("revision"))) or "here")
    inventory.download(spec)
    assert inventory.is_cached(spec)
    assert asked == [(klein.REPO, klein.REVISION), (klein.REPO, klein.REVISION)]


def test_the_page_agent_draws_with_the_same_build_and_the_model_it_replaced_is_nowhere():
    """One image model for the two bricks: the Page Agent's pictures come
    from the build Portrait Studio brought, found the same way."""
    import inspect

    from page_agent import images

    source = inspect.getsource(images.ImageMaker.__init__)
    assert "flux2_klein.resolve(" in source and not hasattr(images, "DEFAULT_REPO")
    wanted = {spec.key for spec in inventory.for_demo("page-agent")}
    assert "flux2-klein-ov" in wanted and not any("schnell" in key for key in wanted)
    assert "flux2-klein-ov" in {spec.key for spec in inventory.for_demo("portrait-studio")}


# --- the launcher's side ------------------------------------------------------------------


class _Stylist:
    """Stands in for the image model: remembers what it was asked, and
    hands back a plain picture."""

    made: list = []
    asked: list = []
    cancel_at: int | None = None

    def __init__(self, device, on_downloading=None, on_preparing=None, **kwargs):
        _Stylist.made.append(device)
        self.device = device
        if on_preparing:
            on_preparing()

    def restyle(self, picture, style, *, size, seed, on_step=None, stopped=None):
        from PIL import Image

        _Stylist.asked.append((picture.size, style.id, size, seed))
        cancelled = False
        for step in range(STEPS):
            if on_step:
                on_step(step + 1, STEPS)
            if _Stylist.cancel_at == step or (stopped and stopped()):
                cancelled = True
                break
        image = None if cancelled else Image.new("RGB", (size, size), (10, 200, 90))
        return Restyled(style=style.id, image=image, size=size, seed=seed, seconds=8.4, step_seconds=[2.0, 2.0, 2.0, 2.0],
                        cancelled=cancelled)


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher_app.events, "LOG_FILE", tmp_path / "events.log")
    monkeypatch.setattr(portrait_studio_runner, "Stylist", _Stylist)
    _Stylist.made, _Stylist.asked, _Stylist.cancel_at = [], [], None
    return portrait_studio_runner.PortraitStudioRunner()


def test_the_runner_draws_the_photo_in_the_style_and_hands_back_a_picture_with_what_it_took(runner):
    result = runner.restyle(device="GPU", data=_jpeg(160, 90), style_id="manga", size=512, seed=3)
    assert _Stylist.made == ["GPU"] and _Stylist.asked == [((512, 512), "manga", 512, 3)]  # the square in the middle, at the size asked
    assert result["style"] == "manga" and result["style_name"] == "Manga" and (result["size"], result["seed"]) == (512, 3)
    assert result["seconds"] == 8.4 and result["steps"] == STEPS and result["steps_per_second"] == 0.5
    assert result["cancelled"] is False and result["device"] == "GPU"
    head, body = result["image"].split(",", 1)
    from PIL import Image

    assert head == "data:image/jpeg;base64" and Image.open(io.BytesIO(base64.b64decode(body))).size == (512, 512)
    # The same chip again: the model is not loaded twice. Another chip: it is.
    runner.restyle(device="GPU", data=_jpeg(), style_id="oil", size=768, seed=0)
    runner.restyle(device="CPU", data=_jpeg(), style_id="oil", size=768, seed=0)
    assert _Stylist.made == ["GPU", "CPU"]
    assert launcher_app.loaded.info(runner) == {"engine": "openvino", "device": "CPU"}  # what the hardware panel lists, and can unload
    assert launcher_app.loaded.unload(runner, "portrait-studio") is True and launcher_app.loaded.info(runner) is None


def test_what_cannot_be_drawn_is_refused_before_the_model_is_loaded(runner):
    for wrong, says in (({"style_id": "cubist"}, "No style called"), ({"data": b"junk"}, "could not be read"),
                        ({"size": 640}, "512, 768, 1024")):
        asked = {"device": "GPU", "data": _jpeg(), "style_id": "manga", "size": 768, "seed": 0, **wrong}
        with pytest.raises(ValueError, match=says):
            runner.restyle(**asked)
    assert _Stylist.made == [] and _Stylist.asked == []


def test_a_portrait_stopped_part_way_comes_back_without_a_picture_and_a_second_request_is_told_to_wait(runner):
    _Stylist.cancel_at = 1
    stopped = runner.restyle(device="GPU", data=_jpeg(), style_id="clay", size=768, seed=0)
    assert stopped["cancelled"] is True and stopped["image"] is None
    _Stylist.cancel_at = None
    # While one is being drawn, another is refused rather than queued behind it.
    release, entered = threading.Event(), threading.Event()
    original = _Stylist.restyle

    def slow(self, *args, **kwargs):
        entered.set()
        release.wait(5)
        return original(self, *args, **kwargs)

    _Stylist.restyle = slow
    try:
        first = threading.Thread(target=lambda: runner.restyle(device="GPU", data=_jpeg(), style_id="clay", size=768, seed=0))
        first.start()
        assert entered.wait(5)
        with pytest.raises(Conflict, match="being drawn"):
            runner.restyle(device="GPU", data=_jpeg(), style_id="oil", size=768, seed=0)
        release.set()
        first.join(5)
    finally:
        _Stylist.restyle = original


def test_the_photo_a_page_sends_is_a_picture_or_is_refused():
    data = _jpeg()
    assert portrait_studio_runner.photo_bytes("data:image/jpeg;base64," + base64.b64encode(data).decode()) == data
    for wrong in ("", "hello", "data:text/html;base64,PGI+", "data:image/jpeg;base64", "data:image/jpeg;base64,@@@not base64@@@"):
        with pytest.raises(ValueError, match="did not arrive"):
            portrait_studio_runner.photo_bytes(wrong)


@pytest.fixture
def client(monkeypatch, runner):
    monkeypatch.setattr(launcher_app, "portrait_studio_runner", runner)
    monkeypatch.setattr(launcher_app, "resolve", lambda engine, device, **kwargs: (launcher_app.Engine.OPENVINO, device or "GPU"))
    return TestClient(launcher_app.app)


def test_the_brick_is_in_the_images_category_and_says_what_it_offers(client):
    demo = next(demo for demo in REGISTRY if demo.id == "portrait-studio")
    assert (demo.category, demo.status, demo.engines, demo.experimental) == ("Images", "available", ["openvino"], True)
    assert [other.id for other in REGISTRY if other.category == "Images"] == ["portrait-studio"]
    offered = client.get("/api/portrait-studio/devices").json()
    assert [style["id"] for style in offered["styles"]] == [style.id for style in styles.STYLES]
    assert all(style["name"] and style["hint"] for style in offered["styles"])
    assert offered["sizes"] == list(photo.SIZES) and offered["default_size"] == photo.DEFAULT_SIZE
    assert len(offered["samples"]) >= 3
    for sample in offered["samples"]:  # each a picture the page can show, of nobody
        assert sample["image_url"].startswith("/demo-assets/portraits/") and client.get(sample["image_url"]).status_code == 200
        assert "image model drew" in sample["next_step"]


def test_the_route_takes_a_photo_or_a_bundled_portrait_and_refuses_the_rest(client):
    photo_url = "data:image/jpeg;base64," + base64.b64encode(_jpeg(120, 80)).decode()
    drawn = client.post("/api/portrait-studio/restyle", json={"style": "cartoon", "photo": photo_url, "size": 512, "seed": 2}).json()
    assert drawn["style_name"] == "3D cartoon" and drawn["image"].startswith("data:image/jpeg;base64,") and drawn["seed"] == 2
    name = client.get("/api/portrait-studio/devices").json()["samples"][0]["name"]
    sampled = client.post("/api/portrait-studio/restyle", json={"style": "manga", "sample": name})
    assert sampled.status_code == 200 and _Stylist.asked[-1] == ((photo.DEFAULT_SIZE,) * 2, "manga", photo.DEFAULT_SIZE, 0)
    for wrong, says in (({"style": "manga"}, "did not arrive"), ({"style": "manga", "sample": "Nobody at all"}, "no bundled portrait"),
                        ({"style": "cubist", "photo": photo_url}, "No style called"),
                        ({"style": "manga", "photo": photo_url, "compute_device": "NPU"}, "does not run on the NPU"),
                        ({"style": "manga", "photo": photo_url, "size": 100}, "512, 768, 1024")):
        answer = client.post("/api/portrait-studio/restyle", json=wrong)
        assert answer.status_code == 400 and says in answer.json()["error"], wrong
    # The close button of the hardware panel reaches it: its model can be unloaded.
    assert client.post("/api/bricks/portrait-studio/stop").status_code == 200
