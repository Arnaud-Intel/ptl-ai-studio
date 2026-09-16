"""Upgrades a Panther Lake AI Studio checkout in place, then starts it again.

Started by the launcher (`updates.start_upgrade`) as a detached process on
the base Python in isolated mode. It must not run inside the project's
environment -- replacing that environment's packages is the point, and on
Windows a running process holds its DLLs open -- so it uses the standard
library only.

The steps, each logged to logs/upgrade.log: wait for the launcher's
processes to exit, fast-forward to origin/main, `uv sync`, record the
result in logs/upgrade-result.json, start the launcher again. A failed step
still ends with the launcher starting (on the old code, if the pull failed):
a demo machine that silently stays down is worse than one saying what broke.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

BRANCH = "main"
_WAIT_SECONDS = 90


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def wait_for_exit(pids: list[int], timeout: float) -> bool:
    """True once every pid has exited, False if one is still there at the deadline."""
    deadline = time.monotonic() + timeout
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        synchronize = 0x00100000
        for pid in pids:
            handle = kernel32.OpenProcess(synchronize, 0, pid)
            if not handle:
                continue  # already gone
            remaining = max(int((deadline - time.monotonic()) * 1000), 0)
            exited = kernel32.WaitForSingleObject(handle, remaining) == 0  # WAIT_OBJECT_0
            kernel32.CloseHandle(handle)
            if not exited:
                return False
        return True
    while time.monotonic() < deadline:
        if not any(_alive(pid) for pid in pids):
            return True
        time.sleep(0.25)
    return False


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # POSIX only: signal 0 just asks whether the process exists
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_step(command: list[str], cwd: Path, timeout: float) -> tuple[bool, str]:
    log("$ " + " ".join(command))
    try:
        done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(str(exc))
        return False, str(exc)
    output = (done.stdout + done.stderr).strip()
    if output:
        print(output, flush=True)
    return done.returncode == 0, output


def start_launcher(repo: Path, host: str, port: int) -> None:
    extra = ["--no-browser"]  # the page that asked for the upgrade reloads itself
    if host != "127.0.0.1":
        extra += ["--host", host]
    if port != 8765:
        extra += ["--port", str(port)]
    if os.name == "nt":
        # A console window of its own, as if start_launcher.bat had been
        # double-clicked: visible, and closable to stop it.
        subprocess.Popen(
            ["cmd", "/c", "start", "Panther Lake AI Studio", "/D", str(repo), str(repo / "start_launcher.bat"), *extra],
            cwd=repo,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    else:
        (repo / "logs").mkdir(exist_ok=True)
        with open(repo / "logs" / "launcher.log", "a", encoding="utf-8") as output:
            subprocess.Popen(
                ["uv", "run", "panther-lake-launcher", *extra],
                cwd=repo, stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            )
    log("Started the launcher again.")


def _tail(output: str, lines: int = 15) -> str:
    return "\n".join(output.strip().splitlines()[-lines:])[-2000:]


def _read_version(repo: Path) -> str | None:
    try:
        return (repo / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def upgrade(
    args: argparse.Namespace,
    *,
    runner: Callable[[list[str], Path, float], tuple[bool, str]] = run_step,
    waiter: Callable[[list[int], float], bool] = wait_for_exit,
    starter: Callable[[Path, str, int], None] = start_launcher,
) -> dict:
    repo = Path(args.repo)
    result = {"ok": False, "from": args.from_version, "to": None, "step": None, "error": None, "started_at": time.time()}
    log(f"Upgrading from v{args.from_version}; GitHub has v{args.to_version}.")
    if not waiter(args.wait_pids, _WAIT_SECONDS):
        log(f"The launcher's processes were still running after {_WAIT_SECONDS} s; carrying on anyway.")

    sync = ["uv", "sync"]
    for extra in args.extra:
        sync += ["--extra", extra]
    ok = True
    for step, command, timeout in (
        ("pull", ["git", "pull", "--ff-only", "origin", BRANCH], 300),
        ("sync", sync, 1800),
    ):
        ok, output = runner(command, repo, timeout)
        if not ok:
            result.update(step=step, error=_tail(output) or f"{command[0]} {command[1]} failed")
            break

    result.update(ok=ok, to=_read_version(repo), finished_at=time.time())
    (repo / "logs").mkdir(exist_ok=True)
    (repo / "logs" / "upgrade-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"Upgraded to v{result['to']}." if ok else f"Upgrade failed at the {result['step']} step.")
    starter(repo, args.host, args.port)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upgrade Panther Lake AI Studio in place and restart it.")
    p.add_argument("--repo", required=True)
    p.add_argument("--from-version", default="unknown")
    p.add_argument("--to-version", default="unknown")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--wait-pids", default="", help="Comma-separated launcher process ids to wait for.")
    p.add_argument("--extra", action="append", default=[], help="An extra to keep installed (repeatable).")
    args = p.parse_args(argv)
    args.wait_pids = [int(pid) for pid in args.wait_pids.split(",") if pid.strip()]
    return args


def main(argv: list[str] | None = None) -> int:
    return 0 if upgrade(parse_args(argv))["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
