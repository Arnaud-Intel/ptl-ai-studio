"""A laptop with too little memory for the 30B coding model (2026-10-16): under 32 GB, and with no discrete GPU
that holds it by itself, the model is greyed out wherever it is offered -- the cards and panels, the routes that
load it, the models list, the compile list, the Auto Demo and the setup assistant. `PTL_MEMORY_GB` plays the
smaller laptop here."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from launcher import app as launcher_app
from launcher import autodemo_scenes, compile_ahead
from launcher.autodemo import Scene, Skip, Stand
from pantherlake_ai_core import engine, machine, models, prefetch
from pantherlake_ai_core.engine import GpuDevice

IGPU = GpuDevice("GPU.0", "Intel(R) Arc(TM) B390 GPU (iGPU)", None)
B60 = GpuDevice("GPU.1", "Intel(R) Arc(TM) Pro B60 Graphics (dGPU)", None, memory_bytes=24 * 1024**3)
A770 = GpuDevice("GPU.1", "Intel(R) Arc(TM) A770 Graphics (dGPU)", None, memory_bytes=16 * 1024**3)
CODER = models.BY_KEY["coder-30b-ov"]
CODING_DEMOS = {"code-review-assist", "html-creator", "page-agent"}


@pytest.fixture
def laptop(monkeypatch):
    """A laptop of so many GB with these GPUs."""

    def make(memory_gb: float, *gpus: GpuDevice) -> None:
        monkeypatch.setenv(machine.MEMORY_SWITCH, str(memory_gb))
        monkeypatch.setattr(engine, "list_gpu_devices", lambda: list(gpus or (IGPU,)))

    return make


def test_the_rule_memory_or_a_discrete_gpu_that_holds_the_model(laptop):
    assert CODER.needs_memory_gb == 32
    laptop(16)
    reason = machine.unfit_reason(32)
    assert "32 GB" in reason and "16 GB" in reason and "Arc Pro B60" in reason
    laptop(31.6)  # what Windows says of a 32 GB laptop: it has it
    assert machine.unfit_reason(32) is None
    laptop(24)
    assert machine.unfit_reason(32)
    laptop(16, IGPU, B60)  # the B60 holds it: where the app puts it, or on it by name -- not on the integrated GPU
    assert machine.unfit_reason(32) is None and machine.unfit_reason(32, "AUTO") is None and machine.unfit_reason(32, "GPU.1") is None
    assert "GPU.0 is not that GPU" in machine.unfit_reason(32, "GPU.0")
    laptop(16, IGPU, A770)  # 16 GB of its own is not enough for a 17 GB model
    assert machine.unfit_reason(32)
    laptop(16, IGPU, GpuDevice("GPU.1", "Intel(R) Arc(TM) Pro B60 Graphics (dGPU)", None))  # memory not reported: counted
    assert machine.unfit_reason(32) is None
    assert machine.unfit_reason(None) is None and machine.short_of(32, None) is False  # nothing needed, or nothing known
    assert machine.describe() == {
        "memory_gb": 16, "gpu_share_gb": 48, "gpu_share_percent": 300, "gpu_share_pending": False,
        "big_discrete_gpu": "GPU.1", "simulated": True,
    }  # the share every test plays (conftest), whatever the memory played here


def test_the_models_list_marks_it_and_everything_missing_leaves_it_out(laptop, monkeypatch):
    laptop(16)
    monkeypatch.setattr(models, "is_cached", lambda spec: False)
    downloader = prefetch.Prefetcher()
    monkeypatch.setattr(downloader, "_cache_state", lambda spec: (False, 0))
    monkeypatch.setattr(downloader, "_work", lambda chosen: None)  # nothing is fetched here
    status = downloader.status()
    marked = {entry["key"]: entry["unfit"] for entry in status["models"] if entry["unfit"]}
    assert set(marked) == {"coder-30b-ov"} and "16 GB" in marked["coder-30b-ov"]
    assert status["missing_count"] == status["total_count"] - 1 and status["unfit_count"] == 1
    assert status["machine"]["memory_gb"] == 16
    assert "coder-30b-ov" not in downloader.start()  # "download everything missing"
    downloader._thread.join(5)
    assert downloader.start(["coder-30b-ov"]) == ["coder-30b-ov"]  # asked for by name, it is fetched all the same
    downloader._thread.join(5)

    laptop(64)
    assert not any(entry["unfit"] for entry in downloader.status()["models"])


def test_the_compile_list_does_not_offer_it_for_a_chip_that_cannot_hold_it(laptop, monkeypatch):
    monkeypatch.setattr(compile_ahead, "prepared", lambda: {})
    monkeypatch.setattr(engine, "list_openvino_devices", lambda: ["CPU", "GPU.0", "GPU.1", "NPU"])
    monkeypatch.setattr(engine, "gpu_cache_refusal", lambda: None)
    laptop(16, IGPU, B60)
    items = {(item["key"], item["chip"]): item for item in compile_ahead.plan(is_cached=lambda spec: True)}
    assert items[("coder-30b-ov", "igpu")]["state"] == "unfit" and "16 GB" in items[("coder-30b-ov", "igpu")]["unfit"]
    assert items[("coder-30b-ov", "dgpu")]["state"] == "todo"  # the B60 holds it
    assert items[("vlm-7b-ov", "igpu")]["state"] == "todo"  # the other models are not touched


@pytest.fixture
def client(laptop, monkeypatch):
    laptop(16)
    monkeypatch.setattr(launcher_app, "list_openvino_devices", lambda: ["CPU", "GPU.0", "NPU"])
    monkeypatch.setattr(launcher_app, "list_gpu_devices", lambda: [IGPU])
    return TestClient(launcher_app.app)


def test_the_launcher_greys_it_out_and_refuses_it(client, monkeypatch):
    demos = {demo["id"]: demo for demo in client.get("/api/demos").json()}
    assert {demo_id for demo_id, demo in demos.items() if demo["unfit"]} == CODING_DEMOS
    assert client.get("/api/system/memory").json()["memory_gb"] == 16

    devices = client.get("/api/html-creator/devices").json()
    assert "16 GB" in devices["openvino_unfit"]  # the coding engine is greyed out; the portable one stays
    assert set(devices["openvino_unsupported"]) == {"CPU", "GPU.0", "NPU"}
    assert "openvino_unfit" not in client.get("/api/doc-qa/devices").json()  # a demo with no big model: untouched

    asked = []
    monkeypatch.setattr(launcher_app.html_creator_runner, "generate", lambda **kwargs: asked.append(kwargs))
    refused = client.post("/api/html-creator/generate", json={"prompt": "A bakery", "engine": "openvino"})
    assert refused.status_code == 400 and "portable engine" in refused.json()["error"] and asked == []
    monkeypatch.setattr(launcher_app.code_review_assist_runner, "review", lambda **kwargs: asked.append(kwargs))
    refused = client.post("/api/code-review-assist/review", json={"source": "diff_text", "diff_text": "x", "engine": "openvino"})
    assert refused.status_code == 400 and asked == []

    monkeypatch.setattr(launcher_app.page_agent_runner, "build", lambda **kwargs: asked.append(kwargs))
    refused = client.post("/api/page-agent/build", json={"request": "A page for a bakery."})
    assert refused.status_code == 400 and "32 GB" in refused.json()["error"] and asked == []


def test_a_launcher_on_a_big_enough_laptop_offers_it_as_before(laptop, monkeypatch):
    laptop(64)
    monkeypatch.setattr(launcher_app, "list_openvino_devices", lambda: ["CPU", "GPU.0", "NPU"])
    web = TestClient(launcher_app.app)
    assert not any(demo["unfit"] for demo in web.get("/api/demos").json())
    devices = web.get("/api/code-review-assist/devices").json()
    assert "openvino_unfit" not in devices and devices["openvino_unsupported"] == {}


def test_the_auto_demo_leaves_the_page_scene_out_and_plays_the_rest():
    samples = lambda demo: [{"name": "Neighbourhood bakery", "prompt": "A page for a bakery."}]  # noqa: E731
    small = Stand(npu=True, igpu="GPU.0", samples=samples, memory_gb=16)
    skipped = autodemo_scenes.page_agent(small, 1)
    assert isinstance(skipped, Skip) and "32 GB" in skipped.reason and "16 GB" in skipped.reason
    assert isinstance(autodemo_scenes.page_agent(Stand(npu=True, igpu="GPU.0", samples=samples, memory_gb=64), 1), Scene)
    held = Stand(npu=True, igpu="GPU.0", dgpu="GPU.1", big_gpu="GPU.1", samples=samples, memory_gb=16)
    assert isinstance(autodemo_scenes.page_agent(held, 1), Scene)  # the B60 holds the coding model
    left_out = Stand(npu=True, igpu="GPU.0", dgpu=None, big_gpu="GPU.1", samples=samples, memory_gb=16)
    assert isinstance(autodemo_scenes.page_agent(left_out, 1), Skip)  # "play as if it were not there"
    unknown = Stand(npu=True, igpu="GPU.0", samples=samples)  # a launcher that cannot tell: nothing held back
    assert isinstance(autodemo_scenes.page_agent(unknown, 1), Scene)


def test_the_setup_assistant_reads_the_studios_word_on_each_model():
    page = (Path(__file__).resolve().parents[1] / "setup" / "index.html").read_text(encoding="utf-8")
    for helper in ("function unfitModels", "function setUnfit", "function demoUnfit", "function demoOnCpuHere"):
        assert helper in page
    assert re.search(r"if \(setUnfit\(chosenSet\)\) chosenSet = \"auto-light\"", page)  # a set chosen before falls back
    script = (Path(__file__).resolve().parents[1] / "setup" / "assistant.ps1").read_text(encoding="utf-8")
    assert "needs 32 GB, or an Arc Pro B60" in script


def test_a_32_gb_laptop_whose_gpu_is_allowed_too_little(laptop, monkeypatch):
    """The memory is there and Windows lets the integrated GPU use 57% of it (18 GB): the coder needs about 20
    there while it writes (the user, 2026-10-23: it ran once raised to 75%). Greyed out like a small laptop's,
    with what to change -- and still downloaded, since a setting and a restart fix it."""
    assert CODER.needs_gpu_gb == 20
    laptop(31.6)
    monkeypatch.setenv(machine.GPU_SHARE_SWITCH, "18")
    reason = machine.blocked_reason(32, 20)
    assert machine.unfit_reason(32) is None and "18 GB" in reason and "75%" in reason and "Shared GPU/NPU Memory Override" in reason
    assert machine.share_reason(20, "CPU") is None  # not on the GPU, not its share
    monkeypatch.setenv(machine.GPU_SHARE_SWITCH, "23.6")  # 75% of a 32 GB laptop
    assert machine.blocked_reason(32, 20) is None
    monkeypatch.setenv(machine.GPU_SHARE_SWITCH, "18")
    laptop(31.6, IGPU, B60)  # the B60 holds it, whatever the integrated GPU is allowed
    assert machine.share_reason(20) is None and "GPU.0" not in (machine.share_reason(20, "GPU.1") or "")
    assert machine.share_reason(20, "GPU.0")
    assert machine.advised_share_percent(20, 31.6) == 75 and machine.advised_share_percent(28, 31.6) == 90
    assert models.unfit(CODER) is None  # the download list is not touched

    laptop(31.6)
    monkeypatch.setattr(launcher_app, "list_openvino_devices", lambda: ["CPU", "GPU.0", "NPU"])
    web = TestClient(launcher_app.app)
    demos = {demo["id"]: demo for demo in web.get("/api/demos").json()}
    assert {demo_id for demo_id, demo in demos.items() if demo["unfit"]} == CODING_DEMOS
    assert all(demos[demo_id]["unfit_fixable"] for demo_id in CODING_DEMOS)
    devices = web.get("/api/page-agent/devices").json()
    assert "75%" in devices["openvino_unfit"] and "integrated GPU" in devices["openvino_unsupported"]["GPU.0"]
    assert web.get("/api/system/memory").json()["gpu_share_gb"] == 18
    monkeypatch.setattr(launcher_app.page_agent_runner, "build", lambda **kwargs: None)
    refused = web.post("/api/page-agent/build", json={"request": "A page for a bakery."})
    assert refused.status_code == 400 and "Intel Graphics Software" in refused.json()["error"]

    samples = lambda demo: [{"name": "Neighbourhood bakery", "prompt": "A page for a bakery."}]  # noqa: E731
    skipped = autodemo_scenes.page_agent(Stand(npu=True, igpu="GPU.0", samples=samples, memory_gb=32, gpu_share_gb=18), 1)
    assert isinstance(skipped, Skip) and "75%" in skipped.reason
    raised = Stand(npu=True, igpu="GPU.0", samples=samples, memory_gb=32, gpu_share_gb=23.6)
    assert isinstance(autodemo_scenes.page_agent(raised, 1), Scene)


class _Holder:
    """A runner holding a model, the shape `loaded` expects."""

    def __init__(self):
        import threading

        self._session, self._engine, self._device, self._lock = object(), "openvino", "GPU.0", threading.Lock()


def test_a_demo_that_starts_on_a_tight_laptop_gets_the_memory_others_left_loaded(laptop, monkeypatch):
    """2026-10-23: the Page Agent's coder stayed loaded while the next demo started, and that one failed. On a
    laptop short of memory every model another demo holds idle is let go when a demo starts; one at work keeps
    its own; on a 64 GB laptop nothing changes."""
    from launcher import events

    page, ocr, busy = _Holder(), _Holder(), _Holder()
    monkeypatch.setattr(launcher_app, "_UNLOADABLE", {"page-agent": page, "screen-ocr": ocr, "html-creator": busy})
    busy._lock.acquire()  # in the middle of a request
    try:
        laptop(64)
        events.set_phase("doc-qa", "loading", "Loading model...")
        assert page._session is not None and ocr._session is not None

        laptop(31.6)
        events.set_phase("screen-ocr", "loading", "Loading model...")  # its own model is not let go
        assert ocr._session is not None and page._session is None and busy._session is not None
        events.set_phase("doc-qa", "running", "Answering...")
        assert ocr._session is None and busy._session is not None
    finally:
        busy._lock.release()
    logged = [entry for entry in events.recent_events() if entry["phase"] == "unloaded"]
    assert [entry["demo_id"] for entry in logged] == ["page-agent", "screen-ocr"]
    assert "for screen-ocr" in logged[0]["message"] and "page-agent" not in events.status_snapshot()


def test_the_share_is_what_intels_slider_set_and_a_change_waits_for_a_restart(laptop, monkeypatch):
    """Read where the slider writes it (2026-10-23: 57% on the XPS 14, 36.2 GB, as Intel's app said; DXGI said
    47.5). A change made since the laptop started is not in effect yet: where it matters, the page says to restart."""
    import sys

    if sys.platform == "win32":
        percent, pending = machine.gpu_share_setting()  # the real one: whatever this laptop is set to
        assert 7 <= percent <= 93 and pending in (True, False)
    monkeypatch.delenv(machine.GPU_SHARE_SWITCH)
    laptop(31.6)
    monkeypatch.setattr(machine, "gpu_share_setting", lambda: (57, False))
    assert round(machine.gpu_share_gb(), 1) == 18.0 and "set its Memory Limit to 75%" in machine.share_reason(20)
    monkeypatch.setattr(machine, "gpu_share_setting", lambda: (75, True))  # moved, not restarted yet
    assert "applies once the laptop restarts" in machine.share_reason(20)
    monkeypatch.setattr(machine, "gpu_share_setting", lambda: (75, False))
    assert machine.share_reason(20) is None and machine.describe()["gpu_share_percent"] == 75
    laptop(64)
    monkeypatch.setattr(machine, "gpu_share_setting", lambda: (75, True))
    assert machine.share_reason(20) is None  # 64 GB: what it was before was enough too


def test_every_mention_of_the_gpu_limit_links_to_how_to_change_it(laptop, monkeypatch):
    """The user, 2026-10-23: each place that says the GPU's limit is too low links to a page showing where it is
    set, made from their screenshot of Intel Graphics Software; "Needs 32 GB" is said only where the laptop has
    less. The page is served by the launcher, picture and all."""
    laptop(31.6)
    monkeypatch.setenv(machine.GPU_SHARE_SWITCH, "18")
    web = TestClient(launcher_app.app)
    page = web.get(machine.SHARE_HELP)
    assert page.status_code == 200 and "Shared GPU/NPU Memory Override" in page.text and "Memory Limit" in page.text
    assert web.get("/static/intel-gpu-memory-setting.webp").status_code == 200
    assert web.get("/api/system/memory").json()["gpu_share_pending"] is False

    samples = lambda demo: [{"name": "Neighbourhood bakery", "prompt": "A page for a bakery."}]  # noqa: E731
    stand = Stand(npu=True, igpu="GPU.0", samples=samples, memory_gb=32, gpu_share_gb=18)
    assert autodemo_scenes.page_agent(stand, 1).help == machine.SHARE_HELP
    assert autodemo_scenes.page_agent(Stand(npu=True, igpu="GPU.0", samples=samples, memory_gb=16), 1).help == ""

    script = (Path(launcher_app.STATIC_DIR) / "app.js").read_text(encoding="utf-8")
    assert f'const GPU_MEMORY_HELP = "{machine.SHARE_HELP}"' in script
    for place in ("GPU memory limit too low", "note.append(\" \", gpuMemoryHelpLink())", "entry.help ? ` ${gpuMemoryHelpHtml()}`",
                  "why.appendChild(gpuMemoryHelpLink())", "requirement.append(\" \", gpuMemoryHelpLink("):
        assert place in script, place
    html = (Path(launcher_app.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    assert re.search(r'id="landing-auto-need"[^>]*hidden', html) and re.search(r'id="autodemo-requirement"[^>]*hidden', html)


def test_the_setup_assistant_has_a_gpu_memory_step_for_32_gb_laptops():
    """The user, 2026-10-23: the first-launch helper walks a 32 GB laptop through the GPU's memory limit, with the
    same screenshot, and reads the setting through the Studio it has started."""
    page = (Path(__file__).resolve().parents[1] / "setup" / "index.html").read_text(encoding="utf-8")
    assert f"const CODER_GPU_GB = {CODER.needs_gpu_gb:g};" in page
    assert '{ id: "gpu", title: "GPU memory"' in page and "/studio/api/system/memory" in page
    assert "/static/intel-gpu-memory-setting.webp" in page and "Shared GPU/NPU Memory Override" in page
    script = (Path(__file__).resolve().parents[1] / "setup" / "assistant.ps1").read_text(encoding="utf-8")
    assert "'/studio/api/system/memory'" in script and "image/webp" in script
