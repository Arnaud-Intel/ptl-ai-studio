"""Compile the models for this laptop's chips before a show, so that no demo
makes its audience wait for it ("Prepare models", second half).

The first time a model runs on the NPU or on a GPU it is compiled for that
chip, and the result is kept (`engine.ov_config_for`): 5 s instead of 22 for
the 30B coder on the integrated GPU, and on the NPU the difference between
six seconds and more than three minutes for Whisper medium. A new OpenVINO
starts that cache afresh, so after an upgrade every first start is slow
again -- once, and now at a moment somebody chose.

What is compiled: each OpenVINO model that is downloaded, for the chips its
demos run it on (`CHIPS`), among the chips this machine has and that keep
what they compile. One at a time, each in a process of its own
(`compile_worker`): the chips are shared, and a compile that ends its
process must not end the Studio.

What has been compiled here is written beside the cache, in
`prepared.json`: OpenVINO names its files by a hash, and nothing else says
which model on which chip they are. A model compiled by a demo, not from
here, is not in that list; compiling it "again" reads it back in seconds
and writes it down.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from pantherlake_ai_core import engine, models, npu

from .errors import Conflict

NPU, IGPU, DGPU = "npu", "igpu", "dgpu"

# The chips each model is compiled for: where its demos run it when the chip
# is left to the app, and where the Auto Demo puts it. Small models go to the
# NPU and the integrated GPU and are never sent to a discrete GPU by
# themselves; the large ones go to whichever GPU is there, and never to the
# NPU. The 8B has a build of its own for the NPU.
CHIPS: dict[str, tuple[str, ...]] = {
    "whisper-medium-ov": (NPU, IGPU),
    "whisper-base-ov": (NPU, IGPU),
    "llm-1.5b-ov": (NPU, IGPU),
    "notes-8b-npu-ov": (NPU,),
    "notes-8b-ov": (IGPU, DGPU),
    "embed-ov": (NPU, IGPU),
    "yolo11s-ov": (NPU, IGPU),
    "vlm-7b-ov": (IGPU, DGPU),
    "flux2-klein-ov": (IGPU, DGPU),
    "coder-30b-ov": (IGPU, DGPU),
}
_TIMEOUT_SECONDS = 1800.0  # Whisper medium on the NPU takes 200 s; nothing here has taken ten minutes
LEDGER = "prepared.json"


@dataclass(frozen=True)
class Chip:
    kind: str  # npu | igpu | dgpu
    device: str  # what OpenVINO calls it today: "NPU", "GPU.0", or "GPU" with a single one
    name: str
    # Whether it keeps what it compiles; if not, compiling ahead buys nothing, and why.
    keeps: bool = True
    why_not: str = ""


def _gpu_name(full_name: str) -> str:
    name = full_name
    for noise in ("Intel(R) ", "(TM)", "(R)", " Graphics", "(iGPU)", "(dGPU)", " GPU"):
        name = name.replace(noise, "")
    return " ".join(name.split())


def chips() -> list[Chip]:
    """The chips of this machine that compile: the NPU, then the GPUs."""
    found: list[Chip] = []
    if any(npu.is_npu(device) for device in engine.list_openvino_devices()):
        lost = npu.lost()
        found.append(Chip(NPU, "NPU", "NPU", keeps=lost is None,
                          why_not="" if lost is None else "out of use until the Studio is started again"))
    refusal = engine.gpu_cache_refusal()
    for gpu in engine.list_gpu_devices():
        discrete = "dGPU" in gpu.full_name
        short = _gpu_name(gpu.full_name)
        found.append(Chip(DGPU if discrete else IGPU, gpu.id, short if discrete else f"Integrated GPU ({short})",
                          keeps=refusal is None, why_not=refusal or ""))
    return found


def _ledger_path() -> Path:
    return engine.ov_cache_dir() / LEDGER


def prepared() -> dict[str, dict]:
    """What was compiled from here for the OpenVINO installed now: {"model@chip kind": {"at", "seconds"}}."""
    try:
        listed = json.loads(_ledger_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return listed if isinstance(listed, dict) else {}


def _note(key: str, kind: str, seconds: float) -> None:
    listed = prepared()
    listed[f"{key}@{kind}"] = {"at": round(time.time()), "seconds": round(seconds, 1)}
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(listed, indent=1, sort_keys=True), encoding="utf-8")


def plan(is_cached: Callable[[models.ModelSpec], bool] | None = None) -> list[dict]:
    """Every model on every chip it could be compiled for here, in the
    inventory's order, the NPU first: what is done, what is to do, and what
    cannot be (the model is not downloaded, or the chip keeps nothing).
    `is_cached` answers "is this model downloaded" for a caller that has
    asked already; `models.is_cached` otherwise."""
    is_cached = is_cached or models.is_cached
    done = prepared()
    machine = chips()
    items = []
    for spec in models.MODELS:
        for kind in CHIPS.get(spec.key, ()):
            for chip in machine:
                if chip.kind != kind:
                    continue
                record = done.get(f"{spec.key}@{kind}")
                if record:
                    state = "ready"
                elif not chip.keeps:
                    state = "unkept"
                elif not is_cached(spec):
                    state = "missing"
                else:
                    state = "todo"
                items.append({
                    "key": spec.key, "label": spec.label, "chip": kind, "chip_name": chip.name, "device": chip.device,
                    "state": state, "seconds": (record or {}).get("seconds"), "error": None,
                })
    return items


def _run_worker(key: str, device: str) -> float:
    """Compile one model on one chip in a process of its own. Returns the
    seconds the load took; raises RuntimeError with the reason otherwise."""
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        done = subprocess.run(
            [sys.executable, "-m", "launcher.compile_worker", key, device],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_TIMEOUT_SECONDS, creationflags=flags, stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"not finished after {_TIMEOUT_SECONDS / 60:.0f} minutes, and stopped") from None
    report = None
    for line in reversed((done.stdout or "").strip().splitlines()):
        if line.startswith("{"):
            try:
                report = json.loads(line)
            except ValueError:
                continue
            break
    if report and report.get("ok"):
        return float(report.get("seconds") or 0.0)
    if report and report.get("error"):
        raise RuntimeError(report["error"])
    last = ((done.stderr or "").strip().splitlines() or ["no word from it"])[-1]
    raise RuntimeError(f"the process compiling it ended with code {done.returncode}: {last[:300]}")


class Compiler:
    """One compile at a time, in the order of `plan`."""

    def __init__(self, run_item: Callable[[str, str], float] = _run_worker) -> None:
        self._run_item = run_item
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._states: dict[tuple[str, str], dict] = {}  # (key, chip kind) -> {"state", "error"}, for the run in hand or just ended
        self._current: tuple[str, str] | None = None
        self._started_at: float | None = None
        self._item_started_at: float | None = None
        self._stopped = False

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self, is_cached: Callable[[models.ModelSpec], bool] | None = None) -> dict:
        items = plan(is_cached)
        with self._lock:
            states, current, stopped = dict(self._states), self._current, self._stopped
            started, item_started = self._started_at, self._item_started_at
        running = self.running
        for item in items:
            live = states.get((item["key"], item["chip"]))
            # What the run says wins over the list on disk, but for a model it has finished: the list has its seconds.
            if live and not (live["state"] == "ready" and item["state"] == "ready"):
                item["state"], item["error"] = live["state"], live.get("error")
        queued = [item for item in items if (item["key"], item["chip"]) in states]
        return {
            "chips": [asdict(chip) for chip in chips()],
            "items": items,
            "running": running,
            "stopped": stopped and running,
            "current": {"key": current[0], "chip": current[1]} if current and running else None,
            "done": sum(1 for item in queued if item["state"] in ("ready", "failed")),
            "total": len(queued),
            "elapsed_seconds": round(time.monotonic() - started) if running and started else None,
            "item_seconds": round(time.monotonic() - item_started) if running and item_started else None,
            # What a compile started now would take: what has not been compiled, and what failed the last time.
            "todo": sum(1 for item in items if item["state"] in ("todo", "failed")),
            "ready": sum(1 for item in items if item["state"] == "ready"),
        }

    def start(self, kinds: list[str] | None = None, busy: list[str] | None = None) -> int:
        """Compile what is to do on the chips named (all of them when none
        is). Returns how many models-on-a-chip that is."""
        if self.running:
            raise Conflict("Models are already being compiled.")
        if busy:
            raise Conflict(f"Stop the running demos first ({', '.join(busy)}): compiling uses the same chips.")
        known = {chip.kind for chip in chips()}
        unknown = [kind for kind in kinds or [] if kind not in known]
        if unknown:
            raise ValueError(f"This machine has no such chip: {', '.join(unknown)}.")
        chosen = [item for item in plan() if item["state"] == "todo" and (not kinds or item["chip"] in kinds)]
        if not chosen:
            raise Conflict("Nothing to compile: what is downloaded is compiled already for those chips.")
        self._stop.clear()
        with self._lock:
            self._states = {(item["key"], item["chip"]): {"state": "pending"} for item in chosen}
            self._current, self._stopped = None, False
            self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._work, args=(chosen,), daemon=True, name="model-compile")
        self._thread.start()
        return len(chosen)

    def stop(self) -> None:
        """Stops after the model in hand: a compile cut short can leave half
        a file in the cache, which is worse than none."""
        self._stop.set()
        with self._lock:
            self._stopped = True

    def _work(self, chosen: list[dict]) -> None:
        for item in chosen:
            where = (item["key"], item["chip"])
            if self._stop.is_set():
                with self._lock:
                    self._states.pop(where, None)  # back to "to compile"
                continue
            with self._lock:
                self._states[where] = {"state": "compiling"}
                self._current, self._item_started_at = where, time.monotonic()
            try:
                # On the NPU, in turn with the demos: one request at a time there (core's npu module).
                with npu.guard(item["device"]):
                    seconds = self._run_item(item["key"], item["device"])
                _note(item["key"], item["chip"], seconds)
                outcome = {"state": "ready"}
            except Exception as exc:  # one model failing must not stop the rest
                outcome = {"state": "failed", "error": " ".join(str(exc).split())[:400]}
            with self._lock:
                self._states[where] = outcome
                self._current = None


compiler = Compiler()


def main(argv: list[str] | None = None) -> int:
    """The same from a console, with the Studio stopped:

        uv run python -m launcher.compile_ahead            # compile what is to do
        uv run python -m launcher.compile_ahead --list     # only say what is and what is not
        uv run python -m launcher.compile_ahead --chips npu igpu
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m launcher.compile_ahead",
        description="Compile the downloaded models for this machine's chips, so that no demo compiles one in front of an audience.",
    )
    parser.add_argument("--list", action="store_true", help="Only show what is compiled and what is not.")
    parser.add_argument("--chips", nargs="*", choices=(NPU, IGPU, DGPU), help="The chips to compile for (default: all that keep what they compile).")
    args = parser.parse_args(argv)

    def show(items: list[dict]) -> None:
        words = {"ready": "compiled", "todo": "to compile", "missing": "not downloaded", "unkept": "the chip keeps nothing", "failed": "failed"}
        for item in items:
            took = f" ({item['seconds']} s)" if item["state"] == "ready" and item["seconds"] else ""
            print(f"  {item['label'][:44]:44s} {item['chip_name'][:28]:28s} {words.get(item['state'], item['state'])}{took}"
                  + (f": {item['error']}" if item["error"] else ""))

    for chip in chips():
        if not chip.keeps:
            print(f"{chip.name} keeps nothing of what it compiles: {chip.why_not}.")
    if args.list:
        show(plan())
        return 0
    local = Compiler()
    try:
        total = local.start(args.chips or None)
    except (Conflict, ValueError) as exc:
        print(exc)
        return 0 if "Nothing to compile" in str(exc) else 2
    print(f"Compiling {total} model(s), one at a time. Ctrl+C stops after the one in hand.")
    told: set[tuple[str, str]] = set()
    try:
        while local.running:
            time.sleep(1.0)
            status = local.status()
            for item in status["items"]:
                where = (item["key"], item["chip"])
                if where not in told and item["state"] in ("ready", "failed") and where in local._states:
                    told.add(where)
                    show([item])
    except KeyboardInterrupt:
        local.stop()
        print("Stopping after this model...")
        while local.running:
            time.sleep(0.5)
    final = local.status()
    print(f"{final['ready']} of {len(final['items'])} compiled.")
    return 1 if any(item["state"] == "failed" for item in final["items"]) else 0


if __name__ == "__main__":
    sys.exit(main())
