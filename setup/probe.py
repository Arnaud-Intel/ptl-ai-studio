"""What the freshly installed environment can do on this machine, as JSON.

Run by the setup assistant once the packages are in:

    uv run --no-sync python setup/probe.py out.json

It imports the heavy packages one at a time -- a missing Windows component
shows up here as "DLL load failed", on the package that needed it, rather
than later inside a demo -- then asks OpenVINO which chips it can use and
where the models will be kept. Nothing is downloaded and no model is loaded.
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

# In the order a failure is easiest to read: the small ones first.
PACKAGES = ("numpy", "cv2", "onnxruntime", "openvino", "openvino_genai", "torch", "launcher.app")


def main(out: str) -> int:
    report: dict = {"python": sys.version.split()[0], "packages": {}, "errors": {}, "devices": [], "gpus": [], "npu": None}
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            report["packages"][name] = str(getattr(module, "__version__", "") or "ok")
        except Exception as exc:  # an ImportError, or whatever a half-installed package raises
            report["errors"][name] = f"{type(exc).__name__}: {exc}"[:600]

    try:
        from pantherlake_ai_core.engine import list_gpu_devices, list_openvino_devices

        report["devices"] = list(list_openvino_devices())
        report["gpus"] = [{"id": gpu.id, "name": gpu.full_name} for gpu in list_gpu_devices()]
        if any(str(device).upper().startswith("NPU") for device in report["devices"]):
            import openvino as ov

            report["npu"] = str(ov.Core().get_property("NPU", "FULL_DEVICE_NAME"))
    except Exception as exc:
        report["errors"].setdefault("chips", f"{type(exc).__name__}: {exc}"[:600])

    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache = Path(HF_HUB_CACHE)
        cache.mkdir(parents=True, exist_ok=True)
        report["model_cache"] = str(cache)
        report["model_cache_free_gb"] = round(shutil.disk_usage(cache).free / 1024**3, 1)
    except Exception as exc:
        report["errors"].setdefault("model cache", f"{type(exc).__name__}: {exc}"[:600])

    Path(out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "probe.json"))
