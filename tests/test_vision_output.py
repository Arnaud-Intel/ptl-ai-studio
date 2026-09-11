import numpy as np
import pytest

from webcam_effects.capabilities import validate_device
from webcam_effects.matte import postprocess


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -1.0, 100.0])
def test_nonfinite_mask_fails_before_rendering(bad):
    mask = np.zeros((1, 1, 32, 32), dtype=np.float32)
    mask[0, 0, 5, 5] = bad
    with pytest.raises(RuntimeError, match="invalid mask"):
        postprocess(mask, 64, 48)


def test_valid_cpu_mask_remains_finite_and_bounded():
    mask = postprocess(np.full((1, 1, 32, 32), .5, dtype=np.float32), 64, 48)
    assert mask.shape == (48, 64)
    assert np.isfinite(mask).all() and np.allclose(mask, .5)


@pytest.mark.parametrize("device", ["GPU", "GPU.0", "gpu.1", "AUTO", "AUTO:GPU,CPU"])
def test_unqualified_gpu_is_rejected_before_model_loading(device):
    with pytest.raises(ValueError, match="temporarily unavailable"):
        validate_device(device)


def test_cpu_and_npu_are_not_blocked():
    validate_device("CPU")
    validate_device("NPU")
