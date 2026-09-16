"""The upgrade helper's sequence, with every command, wait and restart faked."""
from __future__ import annotations

import json

from launcher import upgrade_helper as helper


def _args(tmp_path):
    return helper.parse_args(
        ["--repo", str(tmp_path), "--from-version", "0.2.45", "--to-version", "0.2.46", "--wait-pids", "11, 22", "--extra", "openvino"]
    )


def _run(tmp_path, outcomes):
    commands, waited, started = [], [], []

    def runner(command, cwd, timeout):
        commands.append(command)
        return outcomes.get(command[0], (True, "ok"))

    result = helper.upgrade(
        _args(tmp_path),
        runner=runner,
        waiter=lambda pids, timeout: waited.append(pids) or True,
        starter=lambda repo, host, port: started.append((repo, host, port)),
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
