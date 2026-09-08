"""Attributing Windows "GPU Engine" counter instances to the NPU and to
each physical GPU -- on synthetic samples, so it runs anywhere."""
from __future__ import annotations

from pantherlake_ai_core import engine as engine_mod
from pantherlake_ai_core import telemetry
from pantherlake_ai_core.engine import GpuDevice

NPU = "0x00000000_0x0000abcd"
IGPU = "0x00000000_0x0209f48d"
DGPU = "0x00000000_0x0000beef"


def instance(pid: int, luid: str, engine: int, engine_type: str) -> str:
    # Get-Counter reports instance names lowercased, e.g.
    # pid_1234_luid_0x00000000_0x0000c6d3_phys_0_eng_0_engtype_3d
    return f"pid_{pid}_luid_{luid}_phys_0_eng_{engine}_engtype_{engine_type}"


SAMPLES = {
    instance(100, NPU, 0, "compute"): 12.0,
    instance(101, NPU, 0, "compute"): 3.0,
    instance(200, IGPU, 0, "3d"): 20.0,
    instance(200, IGPU, 1, "videodecode"): 5.0,
    instance(200, IGPU, 2, "copy"): 1.0,
    instance(300, DGPU, 0, "3d"): 50.0,
    instance(300, DGPU, 3, "compute"): 30.0,
}


def test_each_gpu_and_the_npu_are_attributed_by_luid(monkeypatch):
    monkeypatch.setattr(
        engine_mod, "list_gpu_devices", lambda: [GpuDevice("GPU.0", "iGPU", IGPU), GpuDevice("GPU.1", "dGPU", DGPU)]
    )
    gpus, npu_luid = telemetry._classify_luids(SAMPLES)
    assert npu_luid == NPU  # the only adapter whose engines are all "compute"
    assert {(g.id, g.name, g.percent) for g in gpus} == {("GPU.0", "iGPU", 26.0), ("GPU.1", "dGPU", 80.0)}
    assert telemetry._sum_for_luid(SAMPLES, npu_luid) == 15.0


def test_without_openvino_the_busiest_adapter_is_reported_as_the_gpu(monkeypatch):
    monkeypatch.setattr(engine_mod, "list_gpu_devices", lambda: [])
    gpus, npu_luid = telemetry._classify_luids(SAMPLES)
    assert npu_luid == NPU
    assert [(g.id, g.percent) for g in gpus] == [("GPU", 26.0)]  # the iGPU has the most engine types


def test_nothing_in_means_nothing_out():
    assert telemetry._classify_luids({}) == ([], None)
    assert telemetry._sum_for_luid(SAMPLES, None) is None


def test_a_luid_total_is_capped_at_100():
    samples = {instance(1, DGPU, 0, "3d"): 70.0, instance(2, DGPU, 1, "compute"): 55.0}
    assert telemetry._sum_for_luid(samples, DGPU) == 100.0
