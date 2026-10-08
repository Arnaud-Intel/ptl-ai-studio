"""The page agent (experimental): a planner, an image model and a coding
model on three chips, conducted and checked by plain code. Everything here
runs without a model: the plan's parser, who-works-where, the checks, the
repeated-picture repair, the conductor's order of work, and the routes."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from page_agent import checks, conductor
from page_agent import plan as planning
from page_agent.conductor import PageAgent, assign
from page_agent.repeats import give_repeats_their_own
from page_agent.types import Assignment
from pantherlake_ai_core.engine import GpuDevice

from launcher import app as launcher_app

IGPU = GpuDevice("GPU.0", "Intel(R) Arc(TM) B390 GPU (iGPU)", None)
DGPU = GpuDevice("GPU.1", "Intel(R) Arc(TM) Pro B60 Graphics (dGPU)", None)

# --- the plan ---------------------------------------------------------------------------

_ANSWER = """TITLE: Baked with Care
STYLE: warm and cozy | cream, golden, soft beige
SECTIONS: Welcome | About Us | Our Bakes | Visit
HERO: A golden sunrise over a rustic table of pastries, warm editorial photograph
PICTURE 2: A baker's hands dusting flour from a rolling pin, warm editorial photograph
PICTURE 3: A cozy dining nook with a view of the open kitchen, warm editorial photograph"""


def test_a_well_formed_plan_is_read_and_pictures_are_named_by_position():
    plan = planning.parse("a bakery page", _ANSWER)
    assert (plan.title, plan.sections) == ("Baked with Care", ["Welcome", "About Us", "Our Bakes", "Visit"])
    assert [(p.name, p.width, p.height) for p in plan.pictures] == [
        ("hero.jpg", 1024, 576), ("picture-2.jpg", 768, 512), ("picture-3.jpg", 768, 512),
    ]
    assert plan.pictures[1].prompt.startswith("A baker's hands") and plan.notes == []
    prompt = plan.page_prompt()
    assert prompt.startswith("a bakery page") and "Page title: Baked with Care" in prompt and "- Our Bakes" in prompt
    assert plan.closing().endswith("Start your reply with <!DOCTYPE html>.")  # or the coder answers "```" and stops


def test_a_small_models_liberties_with_the_format_are_tolerated():
    answer = (
        "<think>the user wants a page</think>\n"
        "**Title:** Ride the Alps\n"
        "- STYLE - alpine blue, white\n"
        "Sections: Bikes; Prices; Trails\n"
        "1. HERO: A trail at sunrise\n"
        "Picture two: A bike in a workshop\n"
        "IMAGE 3 (landscape): A trail at sunrise\n"  # the same sentence again: one picture, not two
        "PICTURE 4: one sentence describing a third picture\n"  # the instruction copied back: no picture
        "PICTURE 5: <A map on a wooden table>\n"
    )
    plan = planning.parse("bike rental", answer)
    assert plan.title == "Ride the Alps" and plan.style == "alpine blue, white" and plan.sections == ["Bikes", "Prices", "Trails"]
    assert [p.prompt for p in plan.pictures] == ["A trail at sunrise", "A bike in a workshop", "A map on a wooden table"]
    assert [p.name for p in plan.pictures] == ["hero.jpg", "picture-2.jpg", "picture-3.jpg"]


def test_a_plan_that_cannot_be_read_costs_quality_not_the_page():
    plan = planning.parse("a page for a florist", "Sure! Here is a wonderful idea for your page...")
    assert [p.name for p in plan.pictures] == ["hero.jpg"] and "a page for a florist" in plan.pictures[0].prompt
    assert len(plan.notes) == 2  # no picture, no title: both said
    assert planning.parse("x", "").pictures and planning.fallback("x", "the NPU is busy").notes == ["the NPU is busy"]
    many = "\n".join(f"PICTURE {n}: picture number {n}" for n in range(1, 9))
    assert len(planning.parse("x", many).pictures) == planning.MAX_PICTURES


# --- who works where --------------------------------------------------------------------


def test_with_a_discrete_gpu_the_pictures_and_the_page_get_a_chip_each():
    assert assign(["CPU", "GPU.0", "GPU.1", "NPU"], [IGPU, DGPU]) == Assignment("NPU", "GPU.0", "GPU.1", together=True)


def test_with_one_gpu_they_take_turns_on_it():
    assert assign(["CPU", "GPU.0", "NPU"], [IGPU]) == Assignment("NPU", "GPU.0", "GPU.0", together=False)


def test_without_an_npu_or_a_gpu_the_work_still_has_somewhere_to_go():
    assert assign(["CPU", "GPU.0"], [IGPU]).planner == "GPU.0"
    assert assign(["CPU", "GPU.0", "NPU"], [IGPU], npu_usable=False).planner == "GPU.0"  # reset by Windows this session
    assert assign(["CPU"], []) == Assignment("CPU", "CPU", "CPU", together=False)


def test_a_chip_given_by_name_is_used_as_given():
    chosen = assign(["CPU", "GPU.0", "GPU.1", "NPU"], [IGPU, DGPU], planner="CPU", images="GPU.1", page="GPU.1")
    assert chosen == Assignment("CPU", "GPU.1", "GPU.1", together=False)


# --- the checks -------------------------------------------------------------------------

_PAGE = (
    '<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width"><style>'
    ".hero{background-image:url('hero.jpg')}</style></head><body><h1>Bakery</h1>"
    '<img src="picture-2.jpg" alt="Hands dusting flour"><img src="picture-3.jpg" alt="A dining nook"></body></html>'
)


def test_a_sound_page_passes_every_check():
    plan = planning.parse("a bakery page", _ANSWER)
    verdict = checks.review(_PAGE, plan, ["hero.jpg", "picture-2.jpg", "picture-3.jpg"], truncated=False)
    assert all(check.passed for check in verdict) and checks.repair_note(verdict) is None


def test_the_faults_are_named_and_only_two_are_worth_writing_the_page_again():
    plan = planning.parse("a bakery page", _ANSWER)
    broken = _PAGE.replace("<h1>Bakery</h1>", '<link href="https://fonts.example/css" rel="stylesheet">').replace(
        '<meta name="viewport" content="width=device-width">', ""
    )
    verdict = {c.name: c for c in checks.review(broken, plan, ["hero.jpg"], truncated=True)}
    assert not any(c.passed for c in verdict.values() if c.name != checks.ONCE)
    assert "picture-2.jpg, picture-3.jpg" in verdict[checks.PICTURES].detail
    note = checks.repair_note(list(verdict.values()))
    assert "cut off" in note and "left pictures out" in note
    # A page with a remote font, no viewport and no heading is reported as it is: a second try is not the cure.
    only_cosmetic = checks.review(broken, plan, ["hero.jpg", "picture-2.jpg", "picture-3.jpg"], truncated=False)
    assert checks.repair_note(only_cosmetic) is None


# --- a picture shown several times is a page asking for more pictures -------------------


def test_repeated_pictures_get_files_of_their_own_described_by_the_page():
    written = (
        "<style>.hero{background:url('hero.jpg')}</style>"
        '<img src="hero.jpg" alt="Bikes on a ridge">'  # the background was its first use
        '<div><img src="picture-2.jpg" alt="A mountain bike"><h3>Alpine Explorer</h3></div>'
        '<div><img alt="A mountain bike" src="./img/picture-2.jpg"><h3>Valley <em>Cruiser</em></h3></div>'
        '<div><img src="picture-2.jpg"><h3>No alt text</h3></div>'  # nothing to draw from: left as it is
        '<img src="logo.png" alt="Not one of ours">'
    )
    patched, extras = give_repeats_their_own(written, ["hero.jpg", "picture-2.jpg"])
    assert extras == [("extra-1.jpg", "Bikes on a ridge"), ("extra-2.jpg", "Valley Cruiser: A mountain bike")]
    assert 'src="extra-1.jpg" alt="Bikes on a ridge"' in patched and 'src="extra-2.jpg"' in patched
    assert patched.count("picture-2.jpg") == 2 and "url('hero.jpg')" in patched and 'src="logo.png"' in patched


def test_a_page_that_repeats_nothing_is_left_alone_and_a_gallery_is_capped():
    assert give_repeats_their_own(_PAGE, ["hero.jpg", "picture-2.jpg", "picture-3.jpg"]) == (_PAGE, [])
    gallery = "".join(f'<img src="hero.jpg" alt="view {n}">' for n in range(12))
    assert len(give_repeats_their_own(gallery, ["hero.jpg"])[1]) == 6


# --- the conductor's order of work ------------------------------------------------------


class _Planner:
    device = "NPU"
    last_stats = None

    def answer(self, system_prompt, user_prompt, max_tokens=512, control=None):
        return _ANSWER


class _Artist:
    """Stands in for the image model: writes a small file where a picture is asked for."""

    made: list["_Artist"] = []

    def __init__(self, device, on_downloading=None):
        self.device, self.drawn = device, []
        _Artist.made.append(self)

    def draw(self, prompt, width, height, path, seed=0):
        self.drawn.append((path.name, prompt, seed))
        path.write_bytes(b"\xff\xd8 not really a picture \xff\xd9")
        return 0.01


class _Coder:
    """Stands in for the coding model: writes what the test tells it to, one page per call."""

    last_stats = None

    def __init__(self, pages):
        self.pages, self.asked = list(pages), []

    def count_tokens(self, text):
        return len(text.split())

    def prompt_budget(self, max_tokens):
        return 100_000

    def answer(self, system_prompt, user_prompt, max_tokens=512, control=None, sample=True):
        self.asked.append(user_prompt)
        return self.pages.pop(0)


@pytest.fixture
def studio(monkeypatch, tmp_path):
    """A PageAgent whose three models are stand-ins; returns it with the log of what was loaded."""
    loaded: list[tuple[str, str]] = []
    coder = _Coder([_PAGE])
    _Artist.made.clear()

    def create_llm(engine, *, device="AUTO", model_repo=None, **kwargs):
        loaded.append((model_repo, device))
        return _Planner() if "Coder" not in model_repo else coder

    monkeypatch.setattr(conductor, "create_llm", create_llm)
    monkeypatch.setattr("html_creator.session.create_llm", create_llm)
    monkeypatch.setattr("page_agent.images.ImageMaker", _Artist)
    return PageAgent(), coder, loaded, tmp_path


def _steps(log):
    return [(step, state) for step, state, _device, _detail in log if state in ("loading", "done")]


def test_on_one_gpu_the_pictures_come_first_and_the_models_never_share_it(studio):
    agent, coder, loaded, work_dir = studio
    log: list[tuple] = []
    result = agent.build(
        "a bakery page", assignment=Assignment("NPU", "GPU.0", "GPU.0", together=False), work_dir=work_dir,
        on_step=lambda *event: log.append(event),
    )
    assert _steps(log) == [
        ("plan", "loading"), ("plan", "done"), ("images", "loading"), ("images", "done"),
        ("page", "loading"), ("page", "done"), ("check", "done"),
    ]
    assert agent.loaded() == {"plan": "NPU", "page": "GPU.0"}  # the image model was let go before the coder loaded
    assert [name for name, _prompt, _seed in _Artist.made[0].drawn] == ["hero.jpg", "picture-2.jpg", "picture-3.jpg"]
    assert len({seed for _name, _prompt, seed in _Artist.made[0].drawn}) == 3  # no two pictures share a seed
    assert result.attempts == 1 and all(check.passed for check in result.checks)
    assert "data:image/jpeg;base64," in result.html and "picture-2.jpg" not in result.html  # embedded: one file
    assert "picture-2.jpg" in result.html_source  # ...and still readable as the model wrote it
    # The coder was told the pictures by name and how to begin, in that order.
    asked = coder.asked[0]
    assert asked.index("- hero.jpg (1024x576)") < asked.index("Start your reply with <!DOCTYPE html>.")
    assert loaded == [("OpenVINO/Qwen3-8B-int4-cw-ov", "NPU"), ("OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov", "GPU.0")]


def test_on_two_gpus_they_work_at_the_same_time_and_both_stay_loaded(studio, monkeypatch):
    agent, coder, _loaded, work_dir = studio
    drawing = threading.Event()
    original = _Artist.draw

    def slow_draw(self, prompt, width, height, path, seed=0):
        drawing.wait(5)  # the pictures are not ready until the page has been written...
        return original(self, prompt, width, height, path, seed)

    def answer(system_prompt, user_prompt, max_tokens=512, control=None, sample=True):
        drawing.set()  # ...which proves the page did not wait for them
        return _PAGE

    monkeypatch.setattr(_Artist, "draw", slow_draw)
    monkeypatch.setattr(coder, "answer", answer)
    result = agent.build(
        "a bakery page", assignment=Assignment("NPU", "GPU.0", "GPU.1", together=True), work_dir=work_dir
    )
    assert drawing.is_set() and len(result.pictures) == 3 and all(check.passed for check in result.checks)
    assert agent.loaded() == {"plan": "NPU", "images": "GPU.0", "page": "GPU.1"}


def test_a_page_cut_off_is_asked_for_once_more_with_the_fault_named(studio):
    agent, coder, _loaded, work_dir = studio
    coder.pages = [_PAGE[:200], _PAGE]  # the first one stops mid-way
    result = agent.build("a bakery page", assignment=Assignment("NPU", "GPU.0", "GPU.0", together=False), work_dir=work_dir)
    assert result.attempts == 2 and all(check.passed for check in result.checks)
    assert "cut off" in coder.asked[1] and "cut off" not in coder.asked[0]


def test_a_page_that_repeats_a_picture_gets_more_pictures_not_a_rewrite(studio):
    agent, coder, _loaded, work_dir = studio
    twice = _PAGE.replace("</body>", '<img src="picture-2.jpg" alt="Hands dusting flour"><h3>Brioche</h3></body>')
    coder.pages = [twice]
    seen: list[str] = []
    result = agent.build(
        "a bakery page", assignment=Assignment("NPU", "GPU.0", "GPU.0", together=False), work_dir=work_dir,
        on_picture=lambda picture: seen.append(picture.name),
    )
    assert result.attempts == 1 and len(coder.asked) == 1  # the page was written once
    assert seen == ["hero.jpg", "picture-2.jpg", "picture-3.jpg", "extra-1.jpg"]
    assert result.pictures[-1].prompt.startswith("Brioche: Hands dusting flour. Style: warm and cozy")
    assert all(check.passed for check in result.checks) and 'src="extra-1.jpg"' in result.html_source
    # In turn, still: the coder made room for the image model to come back.
    assert agent.loaded() == {"plan": "NPU", "images": "GPU.0"}


def test_a_planner_that_cannot_be_asked_does_not_cost_the_page(studio, monkeypatch):
    agent, _coder, _loaded, work_dir = studio

    def no_planner(engine, *, device="AUTO", model_repo=None, **kwargs):
        if "Coder" not in model_repo:
            raise RuntimeError("the NPU is not answering")
        return _Coder([_PAGE.replace("picture-2.jpg", "hero.jpg").replace("picture-3.jpg", "hero.jpg")])

    monkeypatch.setattr(conductor, "create_llm", no_planner)
    monkeypatch.setattr("html_creator.session.create_llm", no_planner)
    result = agent.build("a bakery page", assignment=Assignment("NPU", "GPU.0", "GPU.0", together=False), work_dir=work_dir)
    assert "the NPU is not answering" in result.plan.notes[0] and [p.name for p in result.plan.pictures] == ["hero.jpg"]
    assert result.html.startswith("<!DOCTYPE html>")


def test_leaving_flushes_and_ends_with_the_code_it_was_given(monkeypatch, capsys):
    from page_agent import leaving

    ended: list[int] = []
    # Off Windows, where ending the process outright is os._exit's job (on Windows it
    # would end this test run, which is the point of it and not something to try here).
    monkeypatch.setattr(leaving.sys, "platform", "linux")
    monkeypatch.setattr(leaving.os, "_exit", ended.append)
    print("the last line")
    leaving.leave_now(2)
    assert ended == [2] and "the last line" in capsys.readouterr().out


def test_only_openvino_and_only_a_request():
    from pantherlake_ai_core.engine import Engine

    with pytest.raises(ValueError, match="OpenVINO"):
        PageAgent(Engine.PORTABLE)
    with pytest.raises(ValueError, match="Describe the page"):
        PageAgent().build("  ", assignment=Assignment("NPU", "GPU.0", "GPU.0", False), work_dir=None)


# --- the launcher ------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher_app.events, "LOG_FILE", tmp_path / "events.log")
    monkeypatch.setattr(launcher_app, "list_openvino_devices", lambda: ["CPU", "GPU.0", "GPU.1", "NPU"])
    monkeypatch.setattr(launcher_app, "list_gpu_devices", lambda: [IGPU, DGPU])
    monkeypatch.setattr(launcher_app.npu, "lost", lambda: None)
    return TestClient(launcher_app.app)


def test_the_card_says_experimental_and_the_menus_are_told_what_auto_means(client):
    demo = next(d for d in client.get("/api/demos").json() if d["id"] == "page-agent")
    assert demo["experimental"] is True and demo["status"] == "available" and demo["engines"] == ["openvino"]
    assert all(d["experimental"] is False for d in client.get("/api/demos").json() if d["id"] != "page-agent")
    devices = client.get("/api/page-agent/devices").json()
    assert devices["auto_assignment"] == {"planner": "NPU", "images": "GPU.0", "page": "GPU.1", "together": True}
    assert len(devices["samples"]) >= 3 and all(sample["prompt"] for sample in devices["samples"])


def test_a_request_is_checked_before_any_model_is_loaded(client, monkeypatch):
    built: list = []
    monkeypatch.setattr(launcher_app.page_agent_runner, "build", lambda **kwargs: built.append(kwargs))
    assert client.post("/api/page-agent/build", json={"request": "  "}).status_code == 400
    refused = client.post("/api/page-agent/build", json={"request": "a page", "image_device": "NPU"})
    assert refused.status_code == 400 and "does not run on the NPU" in refused.json()["error"]
    assert client.post("/api/page-agent/build", json={"request": "a page", "page_device": "GPU.7"}).status_code == 400
    assert built == []


def test_a_build_reports_its_steps_plan_and_pictures_while_it_runs(client, monkeypatch, tmp_path):
    runner = launcher_app.page_agent_runner
    seen: dict = {}

    class _Session:
        def build(self, request, *, assignment, work_dir, on_step, on_plan, on_picture, on_downloading, control):
            on_step("plan", "running", "NPU", "")
            plan = planning.parse(request, _ANSWER)
            on_plan(plan)
            on_step("plan", "done", "NPU", "3 picture(s) planned")
            on_step("images", "running", "GPU.0", "picture 1 of 3")
            (work_dir / "hero.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            on_picture(SimpleNamespace(name="hero.jpg", width=1024, height=576, prompt="x", seconds=6.5))
            on_picture(SimpleNamespace(name="extra-1.jpg", width=768, height=512, prompt="Brioche", seconds=4.4))
            seen["during"] = runner.progress()
            seen["picture"] = client.get("/api/page-agent/picture/hero.jpg").status_code
            seen["not_drawn"] = client.get("/api/page-agent/picture/picture-2.jpg").status_code
            seen["not_ours"] = client.get("/api/page-agent/picture/..%2Fsecrets.txt").status_code
            seen["active"] = {(e["stage"], e["device"]) for e in launcher_app.activity.snapshot() if e["demo_id"] == "page-agent"}
            on_step("images", "done", "GPU.0", "3 picture(s) drawn")
            return SimpleNamespace(
                html="<html></html>", html_source=None, plan=plan, assignment=assignment, pictures=[], pictures_used=[],
                checks=[], attempts=1, seconds={"plan": 1.0, "images": 2.0, "page": 3.0, "total": 4.0}, cancelled=False,
                stats=None, planner_stats=None,
            )

    monkeypatch.setattr(runner, "_session", _Session())
    monkeypatch.setattr(runner, "_engine", "openvino")
    answer = client.post("/api/page-agent/build", json={"request": "a bakery page"})
    assert answer.status_code == 200 and answer.json()["assignment"]["together"] is True

    during = seen["during"]
    assert during["running"] is True and during["plan"]["title"] == "Baked with Care"
    assert [(s["id"], s["state"], s["device"]) for s in during["steps"]] == [
        ("plan", "done", "NPU"), ("images", "running", "GPU.0"), ("page", "pending", "GPU.1"), ("check", "pending", "CPU"),
    ]
    pictures = {p["name"]: p for p in during["plan"]["pictures"]}
    assert pictures["hero.jpg"]["ready"] and not pictures["picture-2.jpg"]["ready"] and pictures["extra-1.jpg"]["extra"]
    assert (seen["picture"], seen["not_drawn"], seen["not_ours"]) == (200, 404, 404)
    # The hardware panel shows the step under its chip, and the conductor under the CPU.
    assert seen["active"] == {("images", "GPU.0"), ("conductor", "CPU")}

    after = client.get("/api/page-agent/progress").json()
    assert after["running"] is False and launcher_app.activity.snapshot() == []
    assert client.get("/api/status").json().get("page-agent") is None


def test_the_hardware_panel_is_told_each_chip_the_agent_holds_a_model_on(client, monkeypatch):
    runner = launcher_app.page_agent_runner
    assert [h for h in client.get("/api/telemetry").json()["loaded"] if h["demo_id"] == "page-agent"] == []
    monkeypatch.setattr(
        runner, "_session", SimpleNamespace(loaded=lambda: {"plan": "NPU", "images": "GPU.0", "page": "GPU.1"})
    )
    held = [h for h in client.get("/api/telemetry").json()["loaded"] if h["demo_id"] == "page-agent"]
    assert [(h["device"], h["stage_label"]) for h in held] == [
        ("NPU", "planner"), ("GPU.0", "image model"), ("GPU.1", "coding model"),
    ]


def test_two_builds_at_once_are_refused(client, monkeypatch):
    runner = launcher_app.page_agent_runner
    assert runner._lock.acquire(blocking=False)
    try:
        refused = client.post("/api/page-agent/build", json={"request": "a page"})
        assert refused.status_code == 409 and "being built already" in refused.json()["error"]
    finally:
        runner._lock.release()
