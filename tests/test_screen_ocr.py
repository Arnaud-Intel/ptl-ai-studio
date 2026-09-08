"""screen-ocr's device resolution.

Its OpenVINO engine is a 7B vision-language model that cannot run on
OpenVINO's "AUTO" plugin -- it loads, then every generate() raises
"Accessing out-of-range dimension" -- and AUTO is what both the launcher
and the CLI default to. So the brick resolves AUTO itself.
"""
from __future__ import annotations

import pytest
from pantherlake_ai_core import engine as engine_mod
from pantherlake_ai_core.engine import GpuDevice
from screen_ocr.extractor_openvino import resolve_device

IGPU = GpuDevice("GPU.0", "Intel(R) Arc(TM) B390 GPU (iGPU)", None)
DGPU = GpuDevice("GPU.1", "Intel(R) Arc(TM) Pro B60 Graphics (dGPU)", None)


@pytest.mark.parametrize("device", ["CPU", "GPU", "GPU.0", "GPU.1", "NPU"])
def test_an_explicit_device_is_left_alone(device, monkeypatch):
    monkeypatch.setattr(engine_mod, "list_gpu_devices", lambda: [IGPU, DGPU])
    assert resolve_device(device) == device


def test_auto_prefers_the_discrete_gpu(monkeypatch):
    monkeypatch.setattr(engine_mod, "list_gpu_devices", lambda: [IGPU, DGPU])
    assert resolve_device("AUTO") == "GPU.1"


def test_auto_uses_the_only_gpu_when_there_is_one(monkeypatch):
    monkeypatch.setattr(engine_mod, "list_gpu_devices", lambda: [IGPU])
    assert resolve_device("AUTO") == "GPU.0"


def test_auto_falls_back_to_cpu_with_no_gpu(monkeypatch):
    monkeypatch.setattr(engine_mod, "list_gpu_devices", lambda: [])
    assert resolve_device("AUTO") == "CPU"


def test_auto_never_picks_the_npu_on_its_own(monkeypatch):
    # This model's NPU compile fails on this hardware (see the brick's
    # README), so it must never be chosen for the user.
    monkeypatch.setattr(engine_mod, "list_gpu_devices", lambda: [IGPU, DGPU])
    assert resolve_device("auto") != "NPU"
    assert resolve_device("AUTO") != "NPU"
