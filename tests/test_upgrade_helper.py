"""The upgrade helper's sequence, with every command, wait and restart faked."""
from __future__ import annotations

import json

import pytest

from launcher import upgrade_helper as helper


@pytest.fixture(autouse=True)
def uv_by_name(monkeypatch):
    """The installer under its plain name, wherever this machine keeps it:
    the sequences below are told apart by the first word of each command."""
    monkeypatch.setattr(helper, "uv_command", lambda repo: "uv")


def _args(tmp_path):
    return helper.parse_args(
        ["--repo", str(tmp_path), "--from-version", "0.2.45", "--to-version", "0.2.46", "--wait-pids", "11, 22", "--extra", "openvino"]
    )


def _run(tmp_path, outcomes):
    commands, waited, started = [], [], []
    commits = ["aaa111", "bbb222"]  # where the checkout is before, then after

    def runner(command, cwd, timeout):
        commands.append(command)
        return outcomes.get(command[0], (True, "ok"))

    result = helper.upgrade(
        _args(tmp_path),
        runner=runner,
        waiter=lambda pids, timeout: waited.append(pids) or True,
        starter=lambda repo, host, port: started.append((repo, host, port)),
        head=lambda repo: commits.pop(0),
    )
    return result, commands, waited, started


def test_a_successful_upgrade_waits_pulls_syncs_records_and_restarts(tmp_path):
    (tmp_path / "VERSION").write_text("0.2.46\n", encoding="utf-8")
    result, commands, waited, started = _run(tmp_path, {})
    assert waited == [[11, 22]]
    assert commands == [["git", "pull", "--ff-only", "origin", "main"], ["uv", "sync", "--extra", "openvino"]]
    assert (result["ok"], result["from"], result["to"], result["step"]) == (True, "0.2.45", "0.2.46", None)
    assert json.loads((tmp_path / "logs" / "upgrade-result.json").read_text(encoding="utf-8"))["ok"] is True
    assert started == [(tmp_path, "127.0.0.1", 8765)]


def test_a_failed_pull_skips_the_sync_and_still_restarts_the_old_version(tmp_path):
    (tmp_path / "VERSION").write_text("0.2.45\n", encoding="utf-8")
    result, commands, _, started = _run(tmp_path, {"git": (False, "fatal: Not possible to fast-forward, aborting.")})
    assert len(commands) == 1
    assert (result["ok"], result["step"], result["to"]) == (False, "pull", "0.2.45")
    assert "fast-forward" in result["error"]
    assert len(started) == 1


def test_a_failed_sync_is_recorded_as_such(tmp_path):
    (tmp_path / "VERSION").write_text("0.2.46\n", encoding="utf-8")
    result, _, _, started = _run(tmp_path, {"uv": (False, "error: Failed to install openvino")})
    assert (result["ok"], result["step"]) == (False, "sync")
    assert len(started) == 1


def test_the_commits_before_and_after_are_recorded_for_the_changelog(tmp_path):
    (tmp_path / "VERSION").write_text("0.2.46\n", encoding="utf-8")
    result, _, _, _ = _run(tmp_path, {})
    assert (result["from_commit"], result["to_commit"]) == ("aaa111", "bbb222")
    saved = json.loads((tmp_path / "logs" / "upgrade-result.json").read_text(encoding="utf-8"))
    assert saved["from_commit"] == "aaa111" and saved["to_commit"] == "bbb222"


def test_the_installer_is_found_on_the_path_named_by_uv_or_kept_in_the_project(tmp_path, monkeypatch):
    """A laptop set up from nothing has uv in the project's own .tools folder
    and nowhere else: an upgrade must not ask for "uv" and find nothing."""
    import shutil

    monkeypatch.undo()  # the real lookup, not the fixture's stand-in
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.delenv("UV", raising=False)
    assert helper.uv_command(tmp_path) == "uv"  # none found: the step fails in the system's own words
    local = tmp_path / ".tools" / "uv" / ("uv.exe" if helper.os.name == "nt" else "uv")
    local.parent.mkdir(parents=True)
    local.write_bytes(b"")
    assert helper.uv_command(tmp_path) == str(local)
    named = tmp_path / "named-uv.exe"
    named.write_bytes(b"")
    monkeypatch.setenv("UV", str(named))  # what `uv run` tells the programs it starts
    assert helper.uv_command(tmp_path) == str(named)
    monkeypatch.setattr(shutil, "which", lambda name: "C:/tools/uv.exe")
    assert helper.uv_command(tmp_path) == "C:/tools/uv.exe"
