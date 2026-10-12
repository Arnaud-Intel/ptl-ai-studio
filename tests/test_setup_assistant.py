"""The setup assistant: what a laptop with nothing installed is given to start from.

`first_launch.bat` opens a page served by PowerShell (`setup/assistant.ps1`,
`setup/index.html`) that walks through the installation. It has to work
before Python is there, so it is not Python; what can be checked from here
is that its pieces agree with the project they install -- the models and
demos the page names exist, the scripts are in the form Windows needs --
and, on Windows, its own PowerShell tests are run."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from launcher import registry
from pantherlake_ai_core import models

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "setup" / "index.html").read_text(encoding="utf-8")


def test_every_model_and_demo_the_page_offers_exists_in_the_studio():
    """The page offers sets of models by the keys of the Studio's own
    catalogue: a key renamed there must not leave a set that fetches less
    than it says."""
    keys = {spec.key for spec in models.MODELS}
    sets = re.search(r"const SETS = \[(.*?)\n\];", PAGE, re.S).group(1)
    named = set(re.findall(r'"([a-z0-9][a-z0-9.-]*-(?:ov|videos)|openvoice|wake-word|chatterbox|silero-vad|selfie-segmentation)"', sets))
    assert named and named <= keys, f"not in the catalogue: {sorted(named - keys)}"
    assert "PLANNER" in sets and {"notes-8b-npu-ov", "notes-8b-ov"} <= keys  # the page builder's planner, by the chip found
    for key in ("notes-8b-npu-ov", "notes-8b-ov"):
        assert f'"{key}"' in PAGE

    demos = dict(re.findall(r'"([a-z-]+)": "([^"]+)"', re.search(r"const DEMOS = \{(.*?)\n\};", PAGE, re.S).group(1)))
    available = {demo.id: demo.name for demo in registry.REGISTRY if demo.status == "available"}
    assert demos == available  # every demo, under the name its card has, and none that is not there


def test_each_set_is_what_it_says_it_is():
    """The Auto Demo's sets hold what its scenes load, no more: the models
    of the page builder are the difference between the two."""
    by_key = {spec.key: spec for spec in models.MODELS}

    def keys_of(set_id: str) -> list[str]:
        block = re.search(r'\{ id: "' + set_id + r'".*?keys: \[(.*?)\]', PAGE, re.S).group(1)
        return re.findall(r'"([^"]+)"', block)

    light, whole = keys_of("auto-light"), keys_of("auto")
    assert set(whole) - set(light) == {"flux2-klein-ov", "coder-30b-ov", "PLANNER"}  # what building a page takes
    for demo in ("doc-qa", "expense-extract", "smart-city-monitor", "video-commentary", "object-detection"):
        needed = {spec.key for spec in models.MODELS if demo in spec.demos and not spec.key.endswith("-portable")}
        assert needed <= set(light), f"{demo} would be missing {sorted(needed - set(light))}"
    small = keys_of("small")
    assert all(key in by_key for key in small) and "coder-30b-ov" not in small and "vlm-7b-ov" not in small  # small means small


@pytest.mark.parametrize("name", ["first_launch.bat", "start_launcher.bat", "stop_launcher.bat"])
def test_command_files_have_windows_line_endings_as_committed(name):
    """A ZIP downloaded from GitHub holds the files as they are committed,
    and cmd.exe can lose its place in a command file with bare line feeds."""
    raw = (ROOT / name).read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b""), f"{name} has bare line feeds"
    assert all(byte < 128 for byte in raw), f"{name} is not plain ASCII"
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.bat -text" in attributes  # kept byte for byte, by a clone and by a ZIP alike


@pytest.mark.parametrize("name", ["setup/assistant.ps1", "setup/get_uv.ps1"])
def test_the_scripts_windows_powershell_runs_are_plain_ascii(name):
    """Windows PowerShell 5.1 -- the one a fresh laptop has -- reads a script
    without a byte-order mark as ANSI: one accented letter and it is garbage."""
    raw = (ROOT / name).read_bytes()
    assert all(byte < 128 for byte in raw), f"{name} is not plain ASCII"
    assert b"`e" not in raw  # an escape PowerShell 5.1 does not know


def test_the_entry_points_find_each_other_and_an_installer_kept_in_the_project():
    first = (ROOT / "first_launch.bat").read_text(encoding="ascii")
    assert "setup\\assistant.ps1" in first and "first_launch.ps1" in first  # the page, and the console helper behind it
    assert 'if "%rc%"=="9" goto console' in first  # the page could not be served: the same steps, in the window
    assert "Extract All" in first  # started from inside the ZIP: said, not failed on
    start = (ROOT / "start_launcher.bat").read_text(encoding="ascii")
    assert ".tools\\uv\\uv.exe" in start and "first_launch.bat" in start
    assistant = (ROOT / "setup" / "assistant.ps1").read_text(encoding="ascii")
    assert ".tools\\uv" in assistant and "'sync', '--locked', '--extra', 'openvino'" in assistant
    # It never runs what a page sends it: an action is one of a fixed list.
    assert "Invoke-Expression" not in assistant and "iex " not in assistant.lower()
    console = (ROOT / "first_launch.ps1").read_text(encoding="utf-8")
    assert ".tools/uv/uv.exe" in console
    # An installer downloaded into the project is the laptop's, not the project's.
    assert ".tools/" in (ROOT / ".gitignore").read_text(encoding="utf-8").split()


def test_the_page_loads_nothing_from_the_web_and_carries_the_token():
    """It has to open on a laptop that has nothing, perhaps behind a network
    that lets little through: everything it shows is in the file or served
    beside it. And only the page the script served can give it orders."""
    assert "__SETUP_TOKEN__" in PAGE and '"X-Setup-Token": TOKEN' in PAGE
    for tag in re.findall(r"<(?:script|link|img)\b[^>]*>", PAGE):
        source = re.search(r'(?:src|href)="([^"]+)"', tag)
        assert source is None or source.group(1).startswith("/static/"), tag
    # What it links to is for the visitor to click, and never fetched: Intel's
    # drivers, and the addresses a piece of advice names (Microsoft's runtime).
    links = re.sub(r"https://www\.intel\.com/content/www/us/en/support/detect\.html|https://\$\{address\}", "", PAGE)
    assert "https://" not in links
    assert "aka\\.ms" in PAGE and 'rel="noopener"' in PAGE


def test_the_probe_reports_instead_of_failing(tmp_path, monkeypatch):
    """Run in the installed environment, it says which packages load and
    which chips are found. A package that does not load is a line of its
    report -- the point of it -- never a crash."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("setup_probe", ROOT / "setup" / "probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    monkeypatch.setattr(probe, "PACKAGES", ("json", "a_package_that_is_not_installed"))
    out = tmp_path / "probe.json"
    code = probe.main(str(out))
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["packages"] == {"json": json.__version__} and "a_package_that_is_not_installed" in report["errors"]
    assert code == 1  # something did not load: the assistant shows it
    assert isinstance(report["devices"], list) and "python" in report


def test_a_studio_that_was_stopped_is_not_started_again():
    """start_launcher.bat used to run the Studio a second time, "with the
    network", whenever the first run ended with an error -- which is also how
    a Studio stopped by stop_launcher.bat ends: it came back by itself in the
    window that had started it. Whether the network is needed is now asked
    before the Studio starts, with a command that does nothing; after that
    the Studio is started once, whichever way."""
    lines = [line.strip() for line in (ROOT / "start_launcher.bat").read_text(encoding="ascii").splitlines()]
    code = [line for line in lines if line and not line.upper().startswith(("REM", "ECHO"))]
    asked = code.index('"%PTL_UV%" run --offline %PTL_EXTRA% python -c "pass" >nul 2>&1')
    assert code[asked - 1] == 'set "PTL_OFFLINE=--offline"' and code[asked + 1] == 'if errorlevel 1 set "PTL_OFFLINE="'
    started = code.index('"%PTL_UV%" run %PTL_OFFLINE% %PTL_EXTRA% panther-lake-launcher %*')
    assert asked < started
    assert sum("panther-lake-launcher" in line for line in code) == 1  # offline or with the network, never both


def test_an_upgrade_can_replace_the_start_script_under_a_window_that_is_using_it():
    """cmd reads a command file by position, a line at a time. The window
    that started the Studio is still open when the upgrade replaces
    start_launcher.bat, and it carries on in the new file at the place it
    had reached in the old one: the first time the file grew, that was the
    middle of a sentence, run as a command, then "run first_launch.bat" and
    "Press any key". Two things keep that from happening. Everything from
    the Studio's start to the end of its window is one block, which cmd reads
    whole beforehand and never comes back from. And for the windows of the
    versions that had no such block, the places they come back to are all
    inside lines of colons -- labels, whatever the place -- ending on an exit."""
    raw = (ROOT / "start_launcher.bat").read_bytes()
    leave = raw.index(b"exit /b 0")
    for resume in (355, 1041, 2101, 2397):  # the byte after "timeout /t 10" in every start script released so far
        assert resume < leave and set(raw[resume:leave]) <= set(b":\r\n"), resume
    assert raw.startswith(b"@echo off\r\ngoto main\r\n:")  # and a new window jumps over them
    lines = [line.strip() for line in raw.decode("ascii").splitlines()]
    opened = lines.index("(")
    start = lines.index('"%PTL_UV%" run %PTL_OFFLINE% %PTL_EXTRA% panther-lake-launcher %*')
    assert start == opened + 1
    depth, closed = 0, None
    for index in range(opened, len(lines)):
        depth += lines[index].endswith("(") - (lines[index] == ")")
        if depth == 0:
            closed = index
            break
    block = lines[start:closed]
    assert block[-1] == "exit /b 0" and "timeout /t 10 >nul" in block  # the upgrade's wait is inside it, and it ends there
    assert not any(re.match(r"(goto|call\s+:)", line, re.I) for line in block)  # nothing in it goes back to the file
    printed = [line for line in block if line.lower().startswith("echo")]
    assert printed and not any("(" in line or ")" in line for line in printed)  # a bracket in a message would end the block


def test_a_copy_updated_by_hand_gets_the_openvino_its_version_is_tried_on():
    """uv re-syncs the environment at every start -- everything but the
    OpenVINO engine, which is an extra it has to be told about. A copy whose
    code was updated by hand (a git pull, a ZIP unpacked over it) kept the
    OpenVINO of the version before, with nothing saying so. The start
    script names the extra wherever OpenVINO is installed, in the check and
    in both starts, and never where it is not: a copy on the portable
    engines alone is not given an engine it did not ask for."""
    lines = [line.strip() for line in (ROOT / "start_launcher.bat").read_text(encoding="ascii").splitlines()]
    code = [line for line in lines if line and not line.upper().startswith(("REM", "ECHO"))]
    empty = code.index('set "PTL_EXTRA="')
    found = code.index('if exist "%~dp0.venv\\Lib\\site-packages\\openvino\\__init__.py" set "PTL_EXTRA=--extra openvino"')
    first_run = next(index for index, line in enumerate(code) if '"%PTL_UV%" run' in line)
    assert empty < found < first_run
    assert all("%PTL_EXTRA%" in line for line in code if '"%PTL_UV%" run' in line)
    # ...and the installation and the in-app upgrade install that same extra from the lock.
    assert "'sync', '--locked', '--extra', 'openvino'" in (ROOT / "setup" / "assistant.ps1").read_text(encoding="ascii")


@pytest.mark.skipif(os.name != "nt", reason="the assistant is Windows PowerShell")
def test_the_assistant_hands_over_to_the_console_when_it_cannot_start(tmp_path):
    """Stopped before its page is up -- here by a plan that is not there, on
    a laptop by rules that forbid what the page needs -- it has done nothing
    yet: exit code 9 is what sends first_launch.bat to the helper in the
    window, rather than leaving the visitor with an error and no way on."""
    done = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "setup" / "assistant.ps1"),
         "-NoBrowser", "-Plan", str(tmp_path / "not-there.json")],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 9, done.stdout[-800:] + done.stderr[-800:]
    assert "helper in this window" in done.stdout


@pytest.mark.skipif(os.name != "nt", reason="the assistant is Windows PowerShell")
@pytest.mark.parametrize("script", ["setup_assistant.Tests.ps1", "first_launch.Tests.ps1"])
def test_the_powershell_tests_pass(script):
    done = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "tests" / script)],
        capture_output=True, text=True, timeout=300,
    )
    assert done.returncode == 0 and "tests passed" in done.stdout, done.stdout[-1500:] + done.stderr[-1500:]
