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


def test_preferred_large_model_device_picks_the_discrete_gpu(monkeypatch):
    monkeypatch.setattr(engine_mod, "list_gpu_devices", lambda: [IGPU, DGPU])
    assert preferred_large_model_device() == "GPU.1"
    monkeypatch.setattr(engine_mod, "list_gpu_devices", lambda: [IGPU])
    assert preferred_large_model_device() == "AUTO"
    monkeypatch.setattr(engine_mod, "list_gpu_devices", lambda: [])
    assert preferred_large_model_device() == "AUTO"


def test_ov_config_caches_compiled_models_for_gpu_and_npu_only(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_mod, "OV_CACHE_DIR", tmp_path / "ov_cache")
    for device in ("GPU", "GPU.1", "gpu.0", "NPU"):
        config = ov_config_for(device)
        cache_dir = Path(config["CACHE_DIR"])
        assert cache_dir.is_absolute() and cache_dir.is_dir()
    assert ov_config_for("CPU") == {}
    assert ov_config_for("AUTO") == {}
