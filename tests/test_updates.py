"""The update check and the upgrade preflight, with git faked -- no network,
no repository state touched."""
from __future__ import annotations

import pytest

from launcher import updates
from launcher.errors import Conflict

BEHIND = {
    "rev-parse --is-inside-work-tree": "true",
    "fetch --quiet origin main": "",
    "show origin/main:VERSION": "0.2.47",
    "rev-list --count HEAD..origin/main": "3",
    "log --format=%s HEAD..origin/main": "chore: bump version to 0.2.47 [skip ci]\nAdd a version check\nchore: bump version to 0.2.46 [skip ci]",
    "rev-parse --abbrev-ref HEAD": "main",
    "rev-list --count origin/main..HEAD": "0",
    "status --porcelain --untracked-files=no": "",
}


def fake_git(monkeypatch, outputs):
    calls = []

    def git(*args, timeout=30):
        key = " ".join(args)
        calls.append(key)
        value = outputs.get(key, "")
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(updates, "_git", git)
    monkeypatch.setattr(updates, "local_version", lambda: "0.2.45")
    monkeypatch.setattr(updates.shutil, "which", lambda name: f"/bin/{name}")
    return calls


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    monkeypatch.setattr(updates, "_status", None)
    monkeypatch.setattr(updates, "_checking", False)
    monkeypatch.setattr(updates, "_prompt_pending", True)


def test_versions_compare_as_numbers():
    assert updates.version_key("0.2.10") > updates.version_key("0.2.9")


def test_behind_github_lists_what_changed_without_the_version_bumps(monkeypatch):
    fake_git(monkeypatch, BEHIND)
    status = updates.check()
    assert (status.local, status.latest, status.update_available, status.can_upgrade) == ("0.2.45", "0.2.47", True, True)
    assert status.changes == ["Add a version check"]
    assert status.error is None and status.blocked_reason is None


def test_up_to_date(monkeypatch):
    fake_git(monkeypatch, {**BEHIND, "show origin/main:VERSION": "0.2.45", "rev-list --count HEAD..origin/main": "0"})
    status = updates.check()
    assert not status.update_available and status.changes == [] and status.error is None


def test_offline_is_reported_not_raised(monkeypatch):
    fake_git(monkeypatch, {**BEHIND, "fetch --quiet origin main": updates.GitError("Could not resolve host: github.com")})
    status = updates.check()
    assert not status.update_available
    assert "Could not resolve host" in status.error


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"status --porcelain --untracked-files=no": " M launcher/src/launcher/app.py"}, "local changes (launcher/src/launcher/app.py)"),
        ({"rev-list --count origin/main..HEAD": "2"}, "2 local commit(s)"),
        ({"rev-parse --abbrev-ref HEAD": "feature"}, "branch 'feature'"),
    ],
)
def test_a_copy_that_cant_fast_forward_cleanly_says_why(monkeypatch, override, expected):
    fake_git(monkeypatch, {**BEHIND, **override})
    status = updates.check()
    assert status.update_available and not status.can_upgrade
    assert expected in status.blocked_reason


def test_without_uv_there_is_no_in_place_upgrade(monkeypatch):
    fake_git(monkeypatch, BEHIND)
    monkeypatch.setattr(updates.shutil, "which", lambda name: None)
    assert "uv isn't on PATH" in updates.check().blocked_reason


def test_the_prompt_is_offered_once_per_launcher_start(monkeypatch):
    fake_git(monkeypatch, BEHIND)
    updates.refresh()
    assert updates.snapshot()["prompt"] is True
    updates.mark_prompted()
    assert updates.snapshot()["prompt"] is False


def test_no_prompt_before_the_check_has_run():
    assert updates.snapshot()["prompt"] is False


def test_upgrade_refuses_while_demos_run(monkeypatch):
    fake_git(monkeypatch, BEHIND)
    with pytest.raises(Conflict, match="Live Speech Translation"):
        updates.start_upgrade(host="127.0.0.1", port=8765, busy=["Live Speech Translation"], spawn=lambda c: None)


def test_upgrade_refuses_a_blocked_copy(monkeypatch):
    fake_git(monkeypatch, {**BEHIND, "rev-list --count origin/main..HEAD": "1"})
    with pytest.raises(Conflict, match="local commit"):
        updates.start_upgrade(host="127.0.0.1", port=8765, busy=[], spawn=lambda c: None)


def test_upgrade_starts_the_helper_on_the_base_python(monkeypatch):
    fake_git(monkeypatch, BEHIND)
    monkeypatch.setattr(updates, "launcher_pids", lambda: [101, 102])
    monkeypatch.setattr(updates, "installed_extras", lambda: ["openvino"])
    spawned = []
    started = updates.start_upgrade(host="127.0.0.1", port=8766, busy=[], spawn=spawned.append)
    assert started == {"status": "upgrading", "from": "0.2.45", "to": "0.2.47"}
    (command,) = spawned
    assert command[1:3] == ["-I", str(updates.HELPER)]
    assert command[command.index("--wait-pids") + 1] == "101,102"
    assert command[command.index("--port") + 1] == "8766"
    assert command[-2:] == ["--extra", "openvino"]
