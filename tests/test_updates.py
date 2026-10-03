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
    # Oldest first, as `git log --reverse` gives it: subject, unit separator,
    # body, (unit separator, day -- left out here), record separator.
    "log --reverse --format=%s%x1f%b%x1f%cs%x1e HEAD..origin/main": (
        "Start offline\x1fuv re-syncs on every start,\nwhich fails offline.\n\nCo-Authored-By: x\x1e\n"
        "chore: bump version to 0.2.46 [skip ci]\x1f\x1e\n"
        "Add a version check\x1f\x1e\n"
        "chore: bump version to 0.2.47 [skip ci]\x1f\x1e\n"
    ),
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
    assert status.changes == ["Add a version check", "Start offline"]
    # ...and the same changes under the version that shipped each, newest first.
    assert status.changelog == [
        {"version": "0.2.47", "changes": [{"summary": "Add a version check", "details": ""}]},
        {
            "version": "0.2.46",
            "changes": [{"summary": "Start offline", "details": "uv re-syncs on every start, which fails offline."}],
        },
    ]
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


def test_changes_are_grouped_under_the_version_that_shipped_them():
    commits = [
        ("Fix A", "Why A.\n\nMore about A."),
        ("Fix B", ""),
        ("chore: bump version to 1.0.1 [skip ci]", ""),
        ("chore: bump version to 1.0.2 [skip ci]", ""),  # a bump with nothing before it says nothing
        ("Add C", "Because C."),
    ]
    assert updates.group_by_version(commits) == [
        {"version": None, "changes": [{"summary": "Add C", "details": "Because C."}]},  # pushed, not numbered yet
        {
            "version": "1.0.1",
            "changes": [{"summary": "Fix A", "details": "Why A."}, {"summary": "Fix B", "details": ""}],
        },
    ]


def test_a_version_carries_the_day_it_was_numbered():
    commits = [
        ("Fix A", "", "2026-10-01"),
        ("chore: bump version to 1.0.1 [skip ci]", "", "2026-10-02"),
        ("Add C", "", "2026-10-03"),
    ]
    assert [(v["version"], v["date"]) for v in updates.group_by_version(commits)] == [
        (None, "2026-10-03"),
        ("1.0.1", "2026-10-02"),
    ]


HISTORY_LOG = (
    "Initial commit\x1fWhy it exists.\n\x1f2026-08-21\x1e\n"
    "chore: bump version to 0.1.1 [skip ci]\x1f\x1f2026-08-21\x1e\n"
    "Add smart recall\x1f\x1f2026-08-22\x1e\n"
    "Fix its index\x1f\x1f2026-08-22\x1e\n"
    "chore: bump version to 0.1.2 [skip ci]\x1f\x1f2026-08-23\x1e\n"
)


def test_the_whole_history_is_read_from_this_copy(monkeypatch):
    calls = fake_git(monkeypatch, {"log --reverse --format=%s%x1f%b%x1f%cs%x1e HEAD": HISTORY_LOG})
    history = updates.history()
    assert history["available"] is True and history["reason"] is None
    assert [(v["version"], v["date"], [c["summary"] for c in v["changes"]]) for v in history["versions"]] == [
        ("0.1.2", "2026-08-23", ["Add smart recall", "Fix its index"]),
        ("0.1.1", "2026-08-21", ["Initial commit"]),
    ]
    assert history["versions"][1]["changes"][0]["details"] == "Why it exists."
    assert not any(call.startswith("fetch") for call in calls)  # nothing here needs the network


def test_a_copy_without_git_history_says_so_and_points_at_github(monkeypatch):
    fake_git(
        monkeypatch,
        {"log --reverse --format=%s%x1f%b%x1f%cs%x1e HEAD": updates.GitError("fatal: not a git repository")},
    )
    history = updates.history()
    assert history["available"] is False and history["versions"] == []
    assert "not a git repository" in history["reason"]
    assert history["repo_url"].startswith("https://github.com/")


def test_a_long_opening_paragraph_is_cut_short():
    details = updates.group_by_version([("X", "word " * 200)])[0]["changes"][0]["details"]
    assert len(details) <= 420 and details.endswith("…")


def test_the_result_of_an_upgrade_lists_the_versions_it_crossed(monkeypatch, tmp_path):
    result = tmp_path / "upgrade-result.json"
    result.write_text(
        '{"ok": true, "from": "0.2.45", "to": "0.2.47", "from_commit": "aaa", "to_commit": "bbb"}', encoding="utf-8"
    )
    monkeypatch.setattr(updates, "RESULT_FILE", result)
    log = BEHIND["log --reverse --format=%s%x1f%b%x1f%cs%x1e HEAD..origin/main"]
    fake_git(monkeypatch, {"log --reverse --format=%s%x1f%b%x1f%cs%x1e aaa..bbb": log})
    assert [version["version"] for version in updates.last_result()["changelog"]] == ["0.2.47", "0.2.46"]


def test_a_failed_upgrade_has_no_changelog(monkeypatch, tmp_path):
    result = tmp_path / "upgrade-result.json"
    result.write_text('{"ok": false, "from": "0.2.45", "to": "0.2.45", "step": "pull"}', encoding="utf-8")
    monkeypatch.setattr(updates, "RESULT_FILE", result)
    assert updates.last_result()["changelog"] == []


def test_an_upgrade_from_before_commits_were_recorded_still_gets_its_changelog(monkeypatch, tmp_path):
    """The helper that runs is the old version's; it wrote no commits."""
    result = tmp_path / "upgrade-result.json"
    result.write_text('{"ok": true, "from": "0.2.45", "to": "0.2.47"}', encoding="utf-8")
    monkeypatch.setattr(updates, "RESULT_FILE", result)
    log = BEHIND["log --reverse --format=%s%x1f%b%x1f%cs%x1e HEAD..origin/main"]
    calls = fake_git(
        monkeypatch,
        {
            "log -1 --format=%H --fixed-strings --grep=chore: bump version to 0.2.45 ": "c0ffee",
            "log --reverse --format=%s%x1f%b%x1f%cs%x1e c0ffee..HEAD": log,
        },
    )
    assert [version["version"] for version in updates.last_result()["changelog"]] == ["0.2.47", "0.2.46"]
    assert calls[-1].endswith("c0ffee..HEAD")
