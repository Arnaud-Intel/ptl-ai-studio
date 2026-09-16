"""Is a newer Panther Lake AI Studio on GitHub, and can this copy upgrade to
it in place?

Installs are git checkouts of the GitHub repo (see the README), so "newer"
is asked of git: fetch main, compare its VERSION file with this one, and list
what changed. An upgrade is a fast-forward pull plus `uv sync` -- and that
can't run inside the launcher on Windows, where a running process holds its
packages' DLLs open. So the launcher hands the work to `upgrade_helper.py`,
started as a detached process on the base Python (outside the project's
environment), and exits; the helper waits for it to be gone, upgrades, and
starts it again. Verified on the XPS 14: a helper started this way outlives
the `uv run` process that spawned it.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .errors import Conflict

REPO_ROOT = Path(__file__).resolve().parents[3]
BRANCH = "main"
REPO_URL = "https://github.com/Arnaud-Intel/ptl-ai-studio"
RAW_VERSION_URL = "https://raw.githubusercontent.com/Arnaud-Intel/ptl-ai-studio/main/VERSION"
HELPER = Path(__file__).resolve().with_name("upgrade_helper.py")
LOG_FILE = REPO_ROOT / "logs" / "upgrade.log"
RESULT_FILE = REPO_ROOT / "logs" / "upgrade-result.json"
# CI's automatic version bumps say nothing to a user about what changed.
_BUMP_PREFIX = "chore: bump version"


@dataclass
class UpdateStatus:
    local: str
    latest: str | None = None
    update_available: bool = False
    checked_at: float | None = None
    # Why the check itself failed (offline, git unusable); None when it worked.
    error: str | None = None
    can_upgrade: bool = False
    # Why this copy can't upgrade itself in place, when an update exists.
    blocked_reason: str | None = None
    changes: list[str] = field(default_factory=list)  # commit subjects, newest first
    repo_url: str = REPO_URL


class GitError(RuntimeError):
    pass


def _git(*args: str, timeout: float = 30) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitError(str(exc)) from exc
    if done.returncode != 0:
        raise GitError((done.stderr or done.stdout).strip() or f"git {args[0]} failed")
    return done.stdout.strip()


def _first_line(exc: BaseException) -> str:
    return (str(exc).strip().splitlines() or ["unknown error"])[0]


def version_key(version: str) -> tuple[int, ...]:
    """"0.2.10" sorts after "0.2.9", which comparing strings gets wrong."""
    return tuple(int(part) for part in re.findall(r"\d+", version))


def local_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def check() -> UpdateStatus:
    """Ask GitHub whether there's a newer version. Never raises: this runs at
    every start, and a laptop without a network isn't an error anyone needs
    a traceback for -- the status says what happened instead."""
    status = UpdateStatus(local=local_version(), checked_at=time.time())
    try:
        _git("rev-parse", "--is-inside-work-tree")
    except GitError:
        return _check_without_git(status)
    try:
        _git("fetch", "--quiet", "origin", BRANCH, timeout=60)
        status.latest = _git("show", f"origin/{BRANCH}:VERSION")
        behind = int(_git("rev-list", "--count", f"HEAD..origin/{BRANCH}"))
        status.update_available = behind > 0 and version_key(status.latest) > version_key(status.local)
        if status.update_available:
            subjects = _git("log", "--format=%s", f"HEAD..origin/{BRANCH}").splitlines()
            status.changes = [s for s in subjects if s and not s.startswith(_BUMP_PREFIX)]
    except (GitError, ValueError) as exc:
        status.update_available = False
        status.error = f"Couldn't check GitHub for updates: {_first_line(exc)}"
        return status
    if status.update_available:
        status.blocked_reason = _upgrade_blocker()
        status.can_upgrade = status.blocked_reason is None
    return status


def _check_without_git(status: UpdateStatus) -> UpdateStatus:
    """A copy downloaded as a zip can still be told a newer one exists, even
    though it can't upgrade itself."""
    try:
        with urllib.request.urlopen(RAW_VERSION_URL, timeout=10) as response:
            status.latest = response.read().decode("utf-8").strip()
    except (OSError, ValueError) as exc:
        status.error = f"Couldn't check GitHub for updates: {_first_line(exc)}"
        return status
    status.update_available = version_key(status.latest) > version_key(status.local)
    if status.update_available:
        status.blocked_reason = "This copy isn't a git checkout, so it can't upgrade itself: download the new version from GitHub."
    return status


def _upgrade_blocker() -> str | None:
    """Why a fast-forward pull plus `uv sync` could go wrong here, if it could."""
    try:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        if branch != BRANCH:
            return f"This copy is on branch '{branch}', not {BRANCH}: switch back to {BRANCH} to upgrade."
        ahead = int(_git("rev-list", "--count", f"origin/{BRANCH}..HEAD"))
        if ahead:
            return (
                f"This copy has {ahead} local commit(s) that aren't on GitHub, so it can't simply "
                "move forward: upgrade it by hand."
            )
        changed = _git("status", "--porcelain", "--untracked-files=no").splitlines()
        if changed:
            names = ", ".join(line[3:] for line in changed[:3]) + (", ..." if len(changed) > 3 else "")
            return f"Files tracked by git have local changes ({names}): commit or discard them first."
    except (GitError, ValueError) as exc:
        return f"Couldn't inspect this checkout: {_first_line(exc)}"
    if shutil.which("uv") is None:
        return "uv isn't on PATH, so the dependencies couldn't be updated: upgrade by hand."
    return None


# --- the launcher's view: checked once in the background, prompted once ----------

_lock = threading.Lock()
_status: UpdateStatus | None = None
_checking = False
# The popup is offered once per launcher start, not on every page load: a
# reload, or a second tab, shouldn't ask again.
_prompt_pending = True


def check_in_background() -> None:
    global _checking
    with _lock:
        if _checking:
            return
        _checking = True

    def run() -> None:
        global _status, _checking
        result = check()
        with _lock:
            _status = result
            _checking = False

    threading.Thread(target=run, daemon=True, name="update-check").start()


def refresh() -> UpdateStatus:
    global _status
    result = check()
    with _lock:
        _status = result
    return result


def snapshot() -> dict:
    with _lock:
        status, checking, prompt = _status, _checking, _prompt_pending
    data = asdict(status if status is not None else UpdateStatus(local=local_version()))
    data["checking"] = checking
    data["prompt"] = prompt and status is not None and status.update_available
    return data


def mark_prompted() -> None:
    global _prompt_pending
    with _lock:
        _prompt_pending = False


def last_result() -> dict | None:
    """What the helper recorded about the most recent upgrade, if any."""
    try:
        return json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- handing off to the helper ------------------------------------------------------


def launcher_pids() -> list[int]:
    """This process and the wrappers that started it -- uv's trampoline, the
    venv's python redirector. The helper waits for every one: any of them
    can still hold files `uv sync` has to replace."""
    import psutil

    me = psutil.Process()
    pids = [me.pid]
    for parent in me.parents():
        if parent.name().lower().startswith(("python", "panther-lake-launcher", "uv")):
            pids.append(parent.pid)
        else:
            break
    return pids


def installed_extras() -> list[str]:
    """The extras `uv sync` must keep: a plain sync removes them (see the README)."""
    return ["openvino"] if importlib.util.find_spec("openvino") is not None else []


def helper_command(*, host: str, port: int, from_version: str, to_version: str, pids: list[int], extras: list[str]) -> list[str]:
    # The base interpreter in isolated mode, never the project's own: the
    # helper replaces that environment, and imports nothing from it.
    python = getattr(sys, "_base_executable", None) or sys.executable
    command = [
        python, "-I", str(HELPER),
        "--repo", str(REPO_ROOT),
        "--from-version", from_version,
        "--to-version", to_version,
        "--host", host,
        "--port", str(port),
        "--wait-pids", ",".join(str(pid) for pid in pids),
    ]
    for extra in extras:
        command += ["--extra", extra]
    return command


def _spawn_detached(command: list[str]) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        kwargs = dict(stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True, cwd=str(REPO_ROOT))
        if sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            try:
                # Out of whatever job object the launcher's wrappers run in,
                # so closing them doesn't take the helper down too.
                subprocess.Popen(command, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kwargs)
            except OSError:
                subprocess.Popen(command, creationflags=flags, **kwargs)
        else:
            subprocess.Popen(command, start_new_session=True, **kwargs)


def start_upgrade(*, host: str, port: int, busy: list[str], spawn: Callable[[list[str]], None] | None = None) -> dict:
    """Check again, refuse with the reason if this copy can't upgrade now,
    otherwise start the helper. The caller then stops the launcher."""
    status = refresh()
    if status.error:
        raise Conflict(status.error)
    if not status.update_available:
        raise Conflict(f"Already up to date (v{status.local}).")
    if not status.can_upgrade:
        raise Conflict(status.blocked_reason or "This copy can't upgrade itself.")
    if busy:
        raise Conflict(f"Stop the running demos first ({', '.join(busy)}): upgrading restarts the launcher.")
    command = helper_command(
        host=host,
        port=port,
        from_version=status.local,
        to_version=status.latest or "",
        pids=launcher_pids(),
        extras=installed_extras(),
    )
    (spawn or _spawn_detached)(command)
    return {"status": "upgrading", "from": status.local, "to": status.latest}
