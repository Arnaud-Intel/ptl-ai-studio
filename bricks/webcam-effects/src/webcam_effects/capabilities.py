"""Conservative device gate until the reported GPU numerical defect is qualified."""
GPU_REASON = "GPU segmentation is temporarily unavailable: invalid masks were reported. Choose CPU or NPU."


def validate_device(device: str) -> None:
    # AUTO may select the affected GPU too. Do not silently move an explicit workload.
    if device.upper() == "AUTO" or "GPU" in device.upper():
        raise ValueError(GPU_REASON)
