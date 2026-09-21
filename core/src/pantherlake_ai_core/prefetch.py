"""Download the demos' models ahead of a show, with the numbers that make
waiting bearable: what's missing, how big it is, how fast it's going, how
long is left (BACKLOG R10).

Used by the launcher's "Prepare models" button and by
`panther-lake-prefetch` on the command line. Progress is measured from the
cache growing on disk rather than from a download backend's callbacks, so
one mechanism covers every model here, whichever library fetches it.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import asdict, dataclass, field

from . import models
from .models import MODELS, ModelSpec

_SAMPLE_SECONDS = 0.5
_SPEED_SMOOTHING = 0.3  # weight of the newest sample: settles, but still reacts
_CACHED_TTL = 5.0  # seconds; the UI polls, and each check walks the Hub cache


@dataclass
class ModelStatus:
    key: str
    label: str
    demos: list[str]
    engine: str
    repo_id: str | None
    note: str
    cached: bool
    cached_bytes: int
    size_bytes: int | None = None  # what a full download costs, asked of the Hub
    state: str = "missing"  # ready | missing | pending | downloading | failed
    error: str | None = None


@dataclass
class _Run:
    keys: list[str]
    started_at: float
    current: str | None = None
    bytes_per_second: float = 0.0
    done_bytes: int = 0
    total_bytes: int = 0
    # What finished models added, plus where the one in flight has got to:
    # the disk can't show the latter, since one backend assembles the file
    # elsewhere and only puts it in place at the end.
    completed_bytes: int = 0
    current_bytes: int = 0
    stopped: bool = False
    errors: dict[str, str] = field(default_factory=dict)


class Prefetcher:
    """One download at a time: they share a disk and a link, so running two
    at once only makes both estimates wrong."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sizes: dict[str, int | None] = {}
        self._sizes_asked = False
        self._states: dict[str, str] = {}
        self._cached: dict[str, tuple[float, bool, int]] = {}
        self._run: _Run | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._sizes_done = threading.Event()

    # --- what the UI and the CLI read ---------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        with self._lock:
            run, sizes, states = self._run, dict(self._sizes), dict(self._states)
        entries = []
        for spec in MODELS:
            cached, on_disk = self._cache_state(spec)
            entries.append(
                ModelStatus(
                    key=spec.key, label=spec.label, demos=list(spec.demos), engine=spec.engine,
                    repo_id=spec.repo_id, note=spec.note, cached=cached, cached_bytes=on_disk,
                    size_bytes=sizes.get(spec.key),
                    state=states.get(spec.key) or ("ready" if cached else "missing"),
                    error=(run.errors.get(spec.key) if run else None),
                )
            )
        missing_bytes = sum(entry.size_bytes or 0 for entry in entries if not entry.cached)
        running = self.running
        # Sizes can arrive after a run starts, so the total is worked out
        # again here rather than frozen at the moment Download was pressed.
        run_total = (run.total_bytes or sum(sizes.get(key) or 0 for key in run.keys)) if run else 0
        remaining = max(run_total - run.done_bytes, 0) if running and run else missing_bytes
        speed = run.bytes_per_second if running and run else 0.0
        return {
            "models": [asdict(entry) for entry in entries],
            "ready_count": sum(1 for entry in entries if entry.cached),
            "total_count": len(entries),
            "missing_bytes": missing_bytes,
            "sizes_known": all(entry.size_bytes is not None for entry in entries if entry.repo_id),
            # Still being asked, as against asked and unanswerable (no network).
            "sizes_pending": self._sizes_asked and not self._sizes_done.is_set(),
            "running": running,
            "current": run.current if run else None,
            "done_bytes": run.done_bytes if run else 0,
            "run_total_bytes": run_total or None,
            "bytes_per_second": round(speed, 1),
            "eta_seconds": round(remaining / speed) if speed > 1 else None,
            "stopped": bool(run and run.stopped),
            "errors": dict(run.errors) if run else {},
        }

    def _cache_state(self, spec: ModelSpec) -> tuple[bool, int]:
        """(is it usable offline, bytes on disk) -- memoised briefly, since a
        poll asks this of every model and each answer walks the cache."""
        now = time.monotonic()
        hit = self._cached.get(spec.key)
        if hit and now - hit[0] < _CACHED_TTL:
            return hit[1], hit[2]
        answer = (models.is_cached(spec), models.cached_bytes(spec))
        self._cached[spec.key] = (now, *answer)
        return answer

    def sizes_in_background(self) -> None:
        """Ask the Hub what everything weighs, once, off the request path."""
        with self._lock:
            if self._sizes_asked:
                return
            self._sizes_asked = True

        def ask(spec: ModelSpec) -> None:
            size = models.remote_size(spec)
            with self._lock:
                self._sizes[spec.key] = size

        def run() -> None:
            # One HTTP call each, and a dozen of them: in parallel the dialog
            # fills in at once instead of a row at a time.
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=8, thread_name_prefix="model-size") as pool:
                list(pool.map(ask, MODELS))
            self._sizes_done.set()

        threading.Thread(target=run, daemon=True, name="model-sizes").start()

    # --- doing the work ------------------------------------------------------
    def start(self, keys: list[str] | None = None) -> list[str]:
        """Download `keys`, or everything missing. Returns what it will fetch."""
        if self.running:
            raise RuntimeError("A download is already running.")
        unknown = [key for key in keys or [] if key not in models.BY_KEY]
        if unknown:
            raise ValueError(f"Unknown model(s): {', '.join(unknown)}.")
        chosen = [models.BY_KEY[key] for key in keys] if keys else [s for s in MODELS if not models.is_cached(s)]
        if not chosen:
            raise RuntimeError("Every model is already downloaded.")
        self._stop.clear()
        self._finished.clear()
        self._cached.clear()
        with self._lock:
            self._states = {spec.key: "pending" for spec in chosen}
            self._run = _Run(
                keys=[spec.key for spec in chosen],
                started_at=time.monotonic(),
                total_bytes=sum((self._sizes.get(spec.key) or 0) - models.cached_bytes(spec) for spec in chosen),
            )
        self._thread = threading.Thread(target=self._work, args=(chosen,), daemon=True, name="model-prefetch")
        self._thread.start()
        return [spec.key for spec in chosen]

    def stop(self) -> None:
        """Stops after the model in flight: the Hub client has no mid-file
        cancel, and a partial blob resumes next time rather than being lost."""
        self._stop.set()
        with self._lock:
            if self._run:
                self._run.stopped = True

    def _work(self, chosen: list[ModelSpec]) -> None:
        start_bytes = {spec.key: models.cached_bytes(spec) for spec in chosen}
        watcher = threading.Thread(target=self._watch, args=(chosen, start_bytes), daemon=True, name="model-progress")
        watcher.start()
        try:
            for spec in chosen:
                if self._stop.is_set():
                    with self._lock:
                        self._states[spec.key] = "ready" if models.is_cached(spec) else "missing"
                    continue
                with self._lock:
                    self._states[spec.key] = "downloading"
                    self._run.current = spec.key
                state, error = self._download_with_one_retry(spec)
                with self._lock:
                    self._states[spec.key] = state
                    self._run.current = None
                    self._run.completed_bytes += max(
                        self._run.current_bytes, models.cached_bytes(spec) - start_bytes[spec.key], 0
                    )
                    self._run.current_bytes = 0
                    if error:
                        self._run.errors[spec.key] = error
        finally:
            self._finished.set()
            watcher.join(timeout=2.0)
            self._cached.clear()
            with self._lock:
                if self._run:
                    self._run.bytes_per_second = 0.0

    def _download_with_one_retry(self, spec: ModelSpec) -> tuple[str, str | None]:
        """One retry, because a dropped connection mid-show-prep is common and
        a resumed download costs seconds. A second failure is reported."""
        for attempt in (1, 2):
            try:
                models.download(spec, on_progress=self._note_progress)
                return "ready", None
            except Exception as exc:  # one model failing must not stop the rest
                if attempt == 2 or self._stop.is_set():
                    return "failed", " ".join(str(exc).split())[:300]
                time.sleep(2.0)
        return "failed", "unreachable"

    def _note_progress(self, position: int) -> None:
        """Absolute bytes for the model in flight, not an increment."""
        with self._lock:
            if self._run:
                self._run.current_bytes = position

    def _watch(self, chosen: list[ModelSpec], start_bytes: dict[str, int]) -> None:
        """Speed and progress: what the Hub client reported, or the cache
        growing on disk for the models other libraries fetch -- whichever
        is further along."""
        previous, last_sample = 0, time.monotonic()
        while not self._finished.wait(_SAMPLE_SECONDS):
            on_disk = sum(max(models.cached_bytes(spec) - start_bytes[spec.key], 0) for spec in chosen)
            with self._lock:
                reported = (self._run.completed_bytes + self._run.current_bytes) if self._run else 0
            # Whichever is further along: the client's own count, or the
            # cache for models a different library fetches.
            moved = max(on_disk, reported)
            now = time.monotonic()
            rate = max(moved - previous, 0) / max(now - last_sample, 1e-6)
            previous, last_sample = moved, now
            with self._lock:
                if self._run is None:
                    return
                self._run.done_bytes = moved
                smoothed = self._run.bytes_per_second
                self._run.bytes_per_second = rate if smoothed == 0 else smoothed + _SPEED_SMOOTHING * (rate - smoothed)


# --- command line -------------------------------------------------------------------


def humanize_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def humanize_seconds(seconds: float | None) -> str:
    if not seconds:
        return "--"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    return f"{minutes} min {secs:02d} s" if minutes else f"{secs} s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="panther-lake-prefetch",
        description="Download the demos' models before a show, so nothing downloads in front of an audience.",
    )
    parser.add_argument("keys", nargs="*", help="Model keys to fetch (default: everything missing).")
    parser.add_argument("--list", action="store_true", help="Only show what exists and what is missing.")
    parser.add_argument("--demo", help="Restrict to the models one demo needs, e.g. --demo screen-ocr.")
    args = parser.parse_args(argv)

    prefetcher = Prefetcher()
    prefetcher.sizes_in_background()
    for _ in range(20):  # sizes are one HTTP call each; show them if they arrive
        if prefetcher.status()["sizes_known"]:
            break
        time.sleep(0.5)

    status = prefetcher.status()
    wanted = [
        entry for entry in status["models"]
        if (not args.keys or entry["key"] in args.keys) and (not args.demo or args.demo in entry["demos"])
    ]
    if not wanted:
        print("No model matches that selection.", file=sys.stderr)
        return 2
    print(f"{'model':40s} {'size':>10s}  {'state':8s} used by")
    for entry in wanted:
        print(
            f"{entry['label'][:40]:40s} {humanize_bytes(entry['size_bytes']):>10s}  "
            f"{'ready' if entry['cached'] else 'missing':8s} {', '.join(entry['demos'])}"
        )
    missing = [entry for entry in wanted if not entry["cached"]]
    unknown = sum(1 for entry in missing if entry["repo_id"] and entry["size_bytes"] is None)
    known = humanize_bytes(sum(entry["size_bytes"] or 0 for entry in missing))
    print(
        f"\n{status['ready_count']}/{status['total_count']} models ready; {known} to download"
        + (f", plus {unknown} the Hub could not size." if unknown else ".")
    )
    if args.list or not missing:
        return 0

    prefetcher.start([entry["key"] for entry in missing])
    while prefetcher.running:
        time.sleep(1.0)
        live = prefetcher.status()
        label = models.BY_KEY[live["current"]].label if live["current"] else "finishing"
        print(
            f"\r{label[:36]:36s} {humanize_bytes(live['done_bytes']):>10s} of "
            f"{humanize_bytes(live['run_total_bytes']):<10s} {humanize_bytes(live['bytes_per_second'])}/s  "
            f"{humanize_seconds(live['eta_seconds'])} left    ",
            end="",
            flush=True,
        )
    print()
    done = prefetcher.status()
    for key, error in done["errors"].items():
        print(f"  {models.BY_KEY[key].label}: {error}", file=sys.stderr)
    print(f"{done['ready_count']}/{done['total_count']} models ready.")
    return 1 if done["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
