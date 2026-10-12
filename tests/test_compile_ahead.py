"""Compiling the models for this laptop's chips ahead of a show: what is
compiled for which chip, the queue, and what the dialog is told. Nothing
here loads a model: the process that would is replaced."""
from __future__ import annotations

import json
import threading
import time

import pytest
from launcher import compile_ahead, compile_worker
from launcher.compile_ahead import DGPU, IGPU, NPU, Chip, Compiler
from launcher.errors import Conflict
from pantherlake_ai_core import engine, models

MACHINE = [Chip(NPU, "NPU", "NPU"), Chip(IGPU, "GPU.0", "Integrated GPU (Arc B390)"), Chip(DGPU, "GPU.1", "Arc Pro B60")]


@pytest.fixture
def laptop(monkeypatch, tmp_path):
    """A laptop with the three chips, every model downloaded, and a
    compiled-model cache of its own with nothing in it."""
    monkeypatch.setattr(engine, "OV_CACHE_DIR", tmp_path / "ov_cache")
    monkeypatch.setattr(engine, "openvino_version", lambda: "2026.4.1")
    monkeypatch.setattr(compile_ahead, "chips", lambda: list(MACHINE))
    monkeypatch.setattr(models, "is_cached", lambda spec: True)
    return tmp_path


def _settled(compiler: Compiler, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while compiler.running and time.monotonic() < deadline:
        time.sleep(0.02)
    return compiler.status()


def test_every_openvino_model_has_its_chips_and_a_way_to_load_it():
    """A model added to the inventory and to no table here would be the one
    that compiles in front of an audience."""
    openvino = {spec.key for spec in models.MODELS if spec.engine == models.OPENVINO}
    assert set(compile_ahead.CHIPS) == openvino == set(compile_worker.LOADERS)
    for key, kinds in compile_ahead.CHIPS.items():
        assert kinds and set(kinds) <= {NPU, IGPU, DGPU}, key
    # The large models never go to the NPU, and the NPU's own build of the 8B goes nowhere else.
    for key in ("coder-30b-ov", "vlm-7b-ov", "flux2-klein-ov", "notes-8b-ov"):
        assert NPU not in compile_ahead.CHIPS[key] and IGPU in compile_ahead.CHIPS[key], key
    assert compile_ahead.CHIPS["notes-8b-npu-ov"] == (NPU,)
    # What the app keeps on the NPU when the chip is left to it is compiled for the NPU.
    for key in ("whisper-medium-ov", "whisper-base-ov", "llm-1.5b-ov", "embed-ov", "yolo11s-ov"):
        assert NPU in compile_ahead.CHIPS[key], key


def test_the_plan_is_each_model_on_each_of_its_chips_that_this_machine_has(laptop, monkeypatch):
    items = compile_ahead.plan()
    assert len(items) == sum(len(kinds) for kinds in compile_ahead.CHIPS.values())
    assert all(item["state"] == "todo" for item in items)
    coder = [(item["chip"], item["device"], item["chip_name"]) for item in items if item["key"] == "coder-30b-ov"]
    assert coder == [(IGPU, "GPU.0", "Integrated GPU (Arc B390)"), (DGPU, "GPU.1", "Arc Pro B60")]

    # Without the dock: no B60, and the integrated GPU is plain "GPU" (it is the chip that is remembered, not its number).
    monkeypatch.setattr(compile_ahead, "chips", lambda: [MACHINE[0], Chip(IGPU, "GPU", "Integrated GPU (Arc B390)")])
    undocked = compile_ahead.plan()
    assert {item["chip"] for item in undocked} == {NPU, IGPU}
    assert next(item for item in undocked if item["key"] == "coder-30b-ov")["device"] == "GPU"

    # A model that is not downloaded cannot be compiled, and a chip that keeps nothing is not compiled for.
    monkeypatch.setattr(models, "is_cached", lambda spec: spec.key != "coder-30b-ov")
    monkeypatch.setattr(compile_ahead, "chips", lambda: [MACHINE[0], Chip(IGPU, "GPU.0", "Integrated GPU", keeps=False, why_not="12 GB free")])
    states = {(item["key"], item["chip"]): item["state"] for item in compile_ahead.plan()}
    assert states[("whisper-medium-ov", NPU)] == "todo" and states[("whisper-medium-ov", IGPU)] == "unkept"
    assert states[("coder-30b-ov", IGPU)] == "unkept"
    monkeypatch.setattr(compile_ahead, "chips", lambda: list(MACHINE))
    assert {(item["key"], item["chip"]): item["state"] for item in compile_ahead.plan()}[("coder-30b-ov", IGPU)] == "missing"


def test_a_run_compiles_what_is_to_do_in_order_and_writes_it_down(laptop):
    ran = []

    def run_item(key, device):
        ran.append((key, device))
        return 12.3

    compiler = Compiler(run_item)
    assert compiler.start([NPU]) == sum(1 for kinds in compile_ahead.CHIPS.values() if NPU in kinds)
    done = _settled(compiler)
    assert ran[0] == ("whisper-medium-ov", "NPU") and all(device == "NPU" for _, device in ran)
    assert done["running"] is False and done["done"] == done["total"] == len(ran)
    listed = json.loads((engine.ov_cache_dir() / compile_ahead.LEDGER).read_text(encoding="utf-8"))
    assert set(listed) == {f"{key}@npu" for key, _ in ran} and listed["whisper-medium-ov@npu"]["seconds"] == 12.3
    states = {(item["key"], item["chip"]): item for item in done["items"]}
    assert states[("whisper-medium-ov", NPU)]["state"] == "ready" and states[("whisper-medium-ov", NPU)]["seconds"] == 12.3
    assert states[("whisper-medium-ov", IGPU)]["state"] == "todo"  # not asked for
    # What is compiled is not compiled again: the next run takes the rest, and after it there is nothing to do.
    ran.clear()
    assert compiler.start() == len(done["items"]) - len(listed)
    _settled(compiler)
    assert not any(device == "NPU" for _, device in ran)
    with pytest.raises(Conflict, match="Nothing to compile"):
        compiler.start()


def test_the_list_of_what_was_compiled_belongs_to_one_openvino(laptop, monkeypatch):
    """A new OpenVINO starts the cache afresh, in a folder of its own: what
    was compiled for the old one is to compile again, and says so."""
    compiler = Compiler(lambda key, device: 1.0)
    compiler.start([NPU])
    assert _settled(compiler)["ready"] > 0
    monkeypatch.setattr(engine, "openvino_version", lambda: "2026.5.0")
    assert compile_ahead.prepared() == {} and all(item["state"] == "todo" for item in compile_ahead.plan())


def test_one_model_failing_does_not_stop_the_others_and_says_why(laptop):
    def run_item(key, device):
        if key == "whisper-base-ov":
            raise RuntimeError("the process compiling it ended with code 3221225477: no word from it")
        return 2.0

    compiler = Compiler(run_item)
    compiler.start([NPU])
    done = _settled(compiler)
    states = {item["key"]: item for item in done["items"] if item["chip"] == NPU}
    assert states["whisper-base-ov"]["state"] == "failed" and "3221225477" in states["whisper-base-ov"]["error"]
    assert all(item["state"] == "ready" for key, item in states.items() if key != "whisper-base-ov")
    assert "whisper-base-ov@npu" not in compile_ahead.prepared()
    # It is to compile again the next time, not struck off: it counts among what is left to do.
    assert done["todo"] == sum(1 for item in done["items"] if item["state"] in ("todo", "failed")) > 0
    assert compiler.start([NPU]) == 1


def test_stopping_ends_after_the_model_in_hand_and_leaves_the_rest_to_compile(laptop):
    going, release = threading.Event(), threading.Event()

    def run_item(key, device):
        going.set()
        release.wait(5)
        return 1.0

    compiler = Compiler(run_item)
    total = compiler.start([NPU])
    assert going.wait(3)
    live = compiler.status()
    assert live["running"] and live["current"] == {"key": "whisper-medium-ov", "chip": NPU} and live["total"] == total
    with pytest.raises(Conflict, match="already being compiled"):
        compiler.start([IGPU])
    compiler.stop()
    assert compiler.status()["stopped"] is True
    release.set()
    done = _settled(compiler)
    npu_items = [item for item in done["items"] if item["chip"] == NPU]
    assert [item["state"] for item in npu_items].count("ready") == 1  # the one in hand was finished: half a file is worse than none
    assert all(item["state"] in ("ready", "todo") for item in npu_items) and done["stopped"] is False


def test_it_waits_for_the_demos_and_refuses_a_chip_this_machine_has_not(laptop, monkeypatch):
    compiler = Compiler(lambda key, device: 1.0)
    with pytest.raises(Conflict, match="Stop the running demos first \\(Live Translation\\)"):
        compiler.start(busy=["Live Translation"])
    monkeypatch.setattr(compile_ahead, "chips", lambda: MACHINE[:2])
    with pytest.raises(ValueError, match="no such chip: dgpu"):
        compiler.start([DGPU])
    assert not compiler.running


def test_the_chips_are_named_as_the_app_names_them_and_a_gpu_that_keeps_nothing_says_why(monkeypatch):
    from pantherlake_ai_core.engine import GpuDevice

    monkeypatch.setattr(engine, "list_openvino_devices", lambda: ["CPU", "GPU.0", "GPU.1", "NPU"])
    monkeypatch.setattr(engine, "list_gpu_devices", lambda: [
        GpuDevice("GPU.0", "Intel(R) Arc(TM) B390 GPU (iGPU)", None), GpuDevice("GPU.1", "Intel(R) Arc(TM) Pro B60 Graphics (dGPU)", None)])
    monkeypatch.setattr(engine, "gpu_cache_refusal", lambda: None)
    assert [(chip.kind, chip.device, chip.name, chip.keeps) for chip in compile_ahead.chips()] == [
        (NPU, "NPU", "NPU", True), (IGPU, "GPU.0", "Integrated GPU (Arc B390)", True), (DGPU, "GPU.1", "Arc Pro B60", True)]
    monkeypatch.setattr(engine, "gpu_cache_refusal", lambda: "12 GB free on the disk, and it takes 20")
    kept = {chip.kind: (chip.keeps, chip.why_not) for chip in compile_ahead.chips()}
    assert kept[NPU] == (True, "") and kept[IGPU] == kept[DGPU] == (False, "12 GB free on the disk, and it takes 20")
    # A machine with one GPU and no NPU, or with nothing OpenVINO can use.
    monkeypatch.setattr(engine, "list_openvino_devices", lambda: ["CPU", "GPU"])
    monkeypatch.setattr(engine, "list_gpu_devices", lambda: [GpuDevice("GPU", "Intel(R) Arc(TM) 140V GPU (iGPU)", None)])
    assert [(chip.kind, chip.device) for chip in compile_ahead.chips()] == [(IGPU, "GPU")]
    monkeypatch.setattr(engine, "list_openvino_devices", lambda: [])
    monkeypatch.setattr(engine, "list_gpu_devices", lambda: [])
    assert compile_ahead.chips() == []


def test_the_worker_reports_in_one_line_whatever_happens(monkeypatch, capsys):
    loaded = []
    monkeypatch.setitem(compile_worker.LOADERS, "yolo11s-ov", lambda device: loaded.append(device))
    assert compile_worker.main(["yolo11s-ov", "NPU"]) == 0 and loaded == ["NPU"]
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["ok"] is True and report["seconds"] >= 0

    def broken(device):
        raise RuntimeError("Exception from src/inference: [GPU] out of\n memory")

    monkeypatch.setitem(compile_worker.LOADERS, "yolo11s-ov", broken)
    assert compile_worker.main(["yolo11s-ov", "GPU.0"]) == 1
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report == {"ok": False, "error": "RuntimeError: Exception from src/inference: [GPU] out of memory"}
    assert compile_worker.main(["no-such-model", "NPU"]) == 2 and json.loads(capsys.readouterr().out)["ok"] is False


def test_what_the_worker_process_says_is_read_and_a_silent_end_is_an_error(monkeypatch):
    import subprocess

    def ending(stdout="", stderr="", code=0):
        return lambda *args, **kwargs: subprocess.CompletedProcess(args, code, stdout, stderr)

    monkeypatch.setattr(subprocess, "run", ending('Fetching 20 files: 100%\n{"ok": true, "seconds": 198.5}\n'))
    assert compile_ahead._run_worker("whisper-medium-ov", "NPU") == 198.5
    monkeypatch.setattr(subprocess, "run", ending('{"ok": false, "error": "RuntimeError: no room"}\n', code=1))
    with pytest.raises(RuntimeError, match="no room"):
        compile_ahead._run_worker("coder-30b-ov", "GPU.0")
    # The NPU's compiler has ended its process without a word: said, with what the process left behind.
    monkeypatch.setattr(subprocess, "run", ending("", "some warning\nLLVM ERROR: unsupported operation\n", code=3221225477))
    with pytest.raises(RuntimeError, match="ended with code 3221225477: LLVM ERROR: unsupported operation"):
        compile_ahead._run_worker("flux2-klein-ov", "NPU")

    def too_long(*args, **kwargs):
        raise subprocess.TimeoutExpired(args, 1800)

    monkeypatch.setattr(subprocess, "run", too_long)
    with pytest.raises(RuntimeError, match="not finished after 30 minutes"):
        compile_ahead._run_worker("coder-30b-ov", "GPU.0")


def test_the_dialog_is_told_and_the_routes_refuse_sensibly(laptop, monkeypatch):
    from fastapi.testclient import TestClient
    from launcher import app as launcher_app
    from launcher import model_routes

    release = threading.Event()
    compiler = Compiler(lambda key, device: release.wait(5) and 1.0 or 1.0)
    monkeypatch.setattr(model_routes, "compiler", compiler)
    monkeypatch.setattr(model_routes.prefetcher, "_cache_state", lambda spec: (True, 100))
    monkeypatch.setattr(model_routes, "busy_demos", lambda: ["Document Q&A"])
    client = TestClient(launcher_app.app)

    told = client.get("/api/models").json()["compile"]
    assert [chip["kind"] for chip in told["chips"]] == [NPU, IGPU, DGPU] and told["running"] is False
    assert told["todo"] == len(told["items"]) and told["ready"] == 0
    assert {"key", "label", "chip", "chip_name", "device", "state", "seconds", "error"} <= set(told["items"][0])

    refused = client.post("/api/models/compile", json={"chips": [NPU]})
    assert refused.status_code == 409 and "Document Q&A" in refused.json()["error"]
    monkeypatch.setattr(model_routes, "busy_demos", lambda: [])
    assert client.post("/api/models/compile", json={"chips": ["tpu"]}).status_code == 400
    started = client.post("/api/models/compile", json={"chips": [NPU]})
    assert started.status_code == 202 and started.json()["started"] > 0
    try:
        assert client.get("/api/models").json()["compile"]["running"] is True
        assert client.post("/api/models/compile", json={}).status_code == 409  # one at a time
        assert client.post("/api/models/prefetch", json={}).status_code == 409  # nor a download beside it
        # An upgrade restarts the Studio, and would cut the compile in hand short.
        assert "Prepare models (compiling)" in launcher_app._busy_demos()
        assert client.post("/api/models/compile/stop").json() == {"stopping": True}
    finally:
        release.set()
    assert _settled(compiler)["running"] is False
    assert "Prepare models (compiling)" not in launcher_app._busy_demos()
    # And while models are being downloaded, nothing is compiled: what is not there cannot be.
    monkeypatch.setattr(type(model_routes.prefetcher), "running", property(lambda self: True))
    assert client.post("/api/models/compile", json={}).status_code == 409


def test_from_a_console_it_lists_and_compiles_the_same(laptop, monkeypatch, capsys):
    assert compile_ahead.main(["--list"]) == 0
    listed = capsys.readouterr().out
    assert listed.count("to compile") == sum(len(kinds) for kinds in compile_ahead.CHIPS.values())
    assert "Whisper medium" in listed and "Arc Pro B60" in listed
    ran = []
    monkeypatch.setattr(compile_ahead, "_run_worker", lambda key, device: ran.append((key, device)) or 3.0)
    monkeypatch.setattr(compile_ahead, "Compiler", lambda: Compiler(compile_ahead._run_worker))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    assert compile_ahead.main(["--chips", "dgpu"]) == 0
    said = capsys.readouterr().out
    assert ran and all(device == "GPU.1" for _, device in ran) and "compiled (3.0 s)" in said
    assert f"{len(ran)} of " in said
    assert compile_ahead.main(["--chips", "dgpu"]) == 0  # nothing left for that chip: said, and not an error
    assert "Nothing to compile" in capsys.readouterr().out


def test_the_dialog_has_what_its_script_fills_in():
    from pathlib import Path

    static = Path(compile_ahead.__file__).parent / "static"
    page, script = (static / "index.html").read_text(encoding="utf-8"), (static / "app.js").read_text(encoding="utf-8")
    for name in ("models-compile", "models-compile-summary", "models-compile-start", "models-compile-stop", "models-compile-chips"):
        assert f'id="{name}"' in page and f'el("{name}")' in script, name
    assert "<th>Compiled for</th>" in page and '"/api/models/compile"' in script and '"/api/models/compile/stop"' in script
