"""The engine/device rules every CLI and the launcher share -- with the
hardware probes monkeypatched, so the outcome doesn't depend on what the
test machine has installed."""
from __future__ import annotations

from pathlib import Path

import pytest

from pantherlake_ai_core import engine as engine_mod
from pantherlake_ai_core.engine import (
    Engine,
    GpuDevice,
    _luid_to_perfcounter_key,
    default_device,
    ov_config_for,
    preferred_large_model_device,
    resolve_engine,
)

IGPU = GpuDevice("GPU.0", "Intel(R) Arc(TM) B390 GPU (iGPU)", None)
DGPU = GpuDevice("GPU.1", "Intel(R) Arc(TM) Pro B60 Graphics (dGPU)", None)


def test_luid_decodes_to_the_windows_perfcounter_key():
    # Verified against real hardware: an Arc B390 iGPU's DEVICE_LUID.
    assert _luid_to_perfcounter_key("8df4090200000000") == "0x00000000_0x0209f48d"


@pytest.mark.parametrize("raw", ["", "zz", "8df40902", "8df409020000000000"])
def test_luid_rejects_malformed_input(raw):
    assert _luid_to_perfcounter_key(raw) is None


def test_resolve_engine_honours_an_explicit_choice():
    assert resolve_engine("portable") is Engine.PORTABLE
    assert resolve_engine(Engine.OPENVINO) is Engine.OPENVINO
    with pytest.raises(ValueError):
        resolve_engine("bogus")


def test_resolve_engine_prefers_openvino_only_when_it_sees_a_device(monkeypatch):
    monkeypatch.setattr(engine_mod, "list_openvino_devices", lambda: ["CPU", "NPU"])
    assert resolve_engine(None) is Engine.OPENVINO
    monkeypatch.setattr(engine_mod, "list_openvino_devices", lambda: [])
    assert resolve_engine(None) is Engine.PORTABLE


def test_default_device_per_engine():
    assert default_device(Engine.PORTABLE) == "cpu"
    assert default_device("openvino") == "AUTO"


def test_preferred_large_model_device_picks_the_discrete_gpu_then_the_integrated_one(monkeypatch):
    monkeypatch.setattr(engine_mod, "list_gpu_devices", lambda: [IGPU, DGPU])
    assert preferred_large_model_device() == "GPU.1"
    # The laptop on its own: the iGPU by name, not AUTO (BACKLOG R20).
    monkeypatch.setattr(engine_mod, "list_gpu_devices", lambda: [IGPU])
    assert preferred_large_model_device() == "GPU.0"
    monkeypatch.setattr(engine_mod, "list_gpu_devices", lambda: [])
    assert preferred_large_model_device() == "AUTO"


def test_the_npu_and_the_gpu_keep_what_they_compile_the_gpu_only_where_it_reads_it_back_right(monkeypatch, tmp_path):
    """Up to OpenVINO 2026.3 a GPU model read back from the cache answered
    wrong on this hardware -- YOLO11n-int8 found a scene full of objects on
    the load that compiled it and exactly one on every load from the cache,
    silently -- so the GPU got no CACHE_DIR at all. On 2026.4.1 every GPU
    model the app loads gives from the cache what it gives compiled, and
    the GPU has the cache: there, never on an older OpenVINO that a copy
    updated by hand may still run, and not when it is switched off."""
    monkeypatch.setattr(engine_mod, "OV_CACHE_DIR", tmp_path / "ov_cache")
    monkeypatch.setattr(engine_mod, "openvino_version", lambda: "2026.4.1")
    monkeypatch.delenv(engine_mod.GPU_CACHE_SWITCH, raising=False)
    monkeypatch.setattr(engine_mod, "_free_gb", lambda path: 400.0)
    for device in ("NPU", "npu", "GPU", "GPU.0", "GPU.1", "gpu.0"):
        cache_dir = Path(ov_config_for(device)["CACHE_DIR"])
        assert cache_dir.is_absolute() and cache_dir.is_dir()
        assert cache_dir == tmp_path / "ov_cache" / "2026.4.1"  # one folder per OpenVINO version
    for device in ("CPU", "AUTO"):  # the CPU loads fast without it, and AUTO may land on the CPU
        assert ov_config_for(device) == {}, f"{device} must not be cached"

    for older in ("2026.3.0", "2026.4.0", "2025.4.1"):
        monkeypatch.setattr(engine_mod, "openvino_version", lambda older=older: older)
        assert ov_config_for("GPU.0") == {} and ov_config_for("GPU") == {}, older
        assert "CACHE_DIR" in ov_config_for("NPU")  # the NPU's cache never had the fault
    for newer in ("2026.4.1", "2026.5.0", "2027.0.0"):
        monkeypatch.setattr(engine_mod, "openvino_version", lambda newer=newer: newer)
        assert "CACHE_DIR" in ov_config_for("GPU.1"), newer

    monkeypatch.setattr(engine_mod, "openvino_version", lambda: "2026.4.1")
    for word in ("0", "off", "OFF", " no ", "false"):  # the way out, should a model ever answer differently from the cache
        monkeypatch.setenv(engine_mod.GPU_CACHE_SWITCH, word)
        assert ov_config_for("GPU.0") == {} and "CACHE_DIR" in ov_config_for("NPU"), word
    for word in ("1", "on", ""):
        monkeypatch.setenv(engine_mod.GPU_CACHE_SWITCH, word)
        assert "CACHE_DIR" in ov_config_for("GPU.0"), word


def test_a_gpu_keeps_nothing_on_a_disk_that_is_nearly_full(monkeypatch, tmp_path):
    """What a GPU keeps is as large as the model: 15 GB for the 30B coder,
    32 GB for everything on one GPU. A disk that fills half-way through
    such a write is a demo that stops, so with little room the GPU compiles
    every time, as it did before it had the cache. The NPU's blobs are a
    few gigabytes in all and a compile there takes minutes: it keeps its."""
    monkeypatch.setattr(engine_mod, "OV_CACHE_DIR", tmp_path / "ov_cache")
    monkeypatch.setattr(engine_mod, "openvino_version", lambda: "2026.4.1")
    monkeypatch.delenv(engine_mod.GPU_CACHE_SWITCH, raising=False)
    for free, kept in ((5.0, False), (engine_mod.GPU_CACHE_FREE_GB - 0.1, False), (engine_mod.GPU_CACHE_FREE_GB, True),
                       (400.0, True), (None, True)):  # None: the drive would not say -- not a reason to go without
        monkeypatch.setattr(engine_mod, "_free_gb", lambda path, free=free: free)
        assert ("CACHE_DIR" in ov_config_for("GPU.0")) is kept, free
        assert "CACHE_DIR" in ov_config_for("NPU")
    assert engine_mod.GPU_CACHE_FREE_GB > 15  # more than the largest single model kept
    monkeypatch.undo()
    # The real thing answers for a folder that is not there yet: its drive is.
    assert engine_mod._free_gb(tmp_path / "not" / "made" / "yet") > 0


def test_what_an_older_openvino_compiled_is_removed_and_a_newer_ones_is_left(monkeypatch, tmp_path):
    """After the runtime is upgraded nothing can use what the old one
    compiled, and on the demo laptop that was 26 GB."""
    root = tmp_path / "ov_cache"
    monkeypatch.setattr(engine_mod, "OV_CACHE_DIR", root)
    monkeypatch.setattr(engine_mod, "openvino_version", lambda: "2026.4.1")
    assert engine_mod.clear_stale_ov_cache() == 0  # no cache yet: nothing to do, and nothing made
    assert not root.exists()
    for folder in ("2026.3.0", "2026.4.0", "2026.4.1", "2026.5.0", "none"):
        (root / folder).mkdir(parents=True)
        (root / folder / "model.blob").write_bytes(b"x" * 100)
    (root / "1234567890.blob").write_bytes(b"y" * 1000)  # from before the cache was kept per version
    # The NPU's larger blobs are read-only files, which Windows refuses to
    # delete as they are: the first run of this on the demo laptop left 27 GB.
    for kept_from_writing in (root / "1234567890.blob", root / "2026.3.0" / "model.blob"):
        kept_from_writing.chmod(0o444)
    assert engine_mod.clear_stale_ov_cache() == 1300
    assert sorted(entry.name for entry in root.iterdir()) == ["2026.4.1", "2026.5.0"]
    assert engine_mod.clear_stale_ov_cache() == 0
    monkeypatch.setattr(engine_mod, "openvino_version", lambda: None)  # no OpenVINO here: not this copy's to tidy
    assert engine_mod.clear_stale_ov_cache() == 0 and (root / "2026.4.1").exists()


def test_an_openvino_older_than_the_one_the_app_is_tried_on_is_noticed(monkeypatch):
    for installed, behind in (("2026.3.0", "2026.3.0"), ("2026.4.0", "2026.4.0"), (engine_mod.TESTED_OPENVINO, None),
                              ("2026.5.0", None), ("2027.0.0", None), (None, None)):
        monkeypatch.setattr(engine_mod, "openvino_version", lambda installed=installed: installed)
        assert engine_mod.openvino_behind() == behind
    # What the app is tried on is what the lock installs: the two cannot drift apart unnoticed.
    import re

    lock = (Path(__file__).resolve().parents[1] / "uv.lock").read_text(encoding="utf-8")
    assert re.search(r'name = "openvino"\nversion = "([^"]+)"', lock).group(1) == engine_mod.TESTED_OPENVINO
    assert re.search(r'name = "openvino-genai"\nversion = "([^"]+)"', lock).group(1).startswith(engine_mod.TESTED_OPENVINO)


def test_a_live_video_model_defaults_to_the_integrated_gpu(monkeypatch):
    """Not AUTO: on 30 identical street-camera frames AUTO took 33.6ms per
    detection against the iGPU's 7.9ms, for identical results. And not the
    discrete card either -- it carries a fixed ~18ms per-frame cost that
    doesn't shrink as the model grows, which is the data path rather than
    the silicon."""
    monkeypatch.setattr(
        engine_mod,
        "list_gpu_devices",
        lambda: [GpuDevice("GPU.0", "Intel(R) Arc(TM) B390 GPU (iGPU)", "luid0"),
                 GpuDevice("GPU.1", "Intel(R) Arc(TM) Pro B60 Graphics (dGPU)", "luid1")],
    )
    assert engine_mod.preferred_realtime_vision_device() == "GPU.0"


def test_the_discrete_card_is_used_when_it_is_the_only_one(monkeypatch):
    monkeypatch.setattr(
        engine_mod, "list_gpu_devices",
        lambda: [GpuDevice("GPU.0", "Intel(R) Arc(TM) Pro B60 Graphics (dGPU)", "luid1")],
    )
    assert engine_mod.preferred_realtime_vision_device() == "GPU.0"


def test_without_a_gpu_a_video_model_falls_back_to_auto(monkeypatch):
    monkeypatch.setattr(engine_mod, "list_gpu_devices", lambda: [])
    assert engine_mod.preferred_realtime_vision_device() == "AUTO"
