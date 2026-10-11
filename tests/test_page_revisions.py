"""The page a brick made is kept by the launcher so that parts of it can be
changed (launcher.page_revisions, and the /api/<brick>/revision routes of
HTML Creator and the Page Agent). No model here: a stand-in writes each part
again, and what is checked is which parts it is asked for, in which order,
what becomes of the page, and what is refused before it is asked anything."""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from html_creator import revise
from html_creator.types import Revised
from launcher import app as launcher_app
from launcher import html_creator_runner, page_agent_runner
from launcher.errors import Conflict
from launcher.page_revisions import PageRevisions

PAGE = """<!DOCTYPE html>
<html><head><title>Bakery</title><style>.card { padding: 1rem; }</style></head>
<body>
<header><h1>Fournil</h1></header>
<main><section class="breads"><div class="card"><h3>Loaf</h3><p>6.50</p></div><div class="card"><h3>Rye</h3><p>7.00</p></div></section></main>
<footer><p>4 rue des Fours</p></footer>
</body></html>"""
# The numbers the parts carry: header 1, h1 2, main 3, section 4, card 5, h3 6, p 7, card 8, h3 9, p 10, footer 11, p 12.


def write(log=None, fail=()):
    """A stand-in for the model: it puts the comment where the part's words were."""
    def one(source, change, pictures):
        part, comment = int(change["part"]), change["comment"]
        if log is not None:
            log.append(part)
        found = revise.find(source, part)
        before = source[found.start:found.end]
        if part in fail:
            return Revised(source=source, number=part, comment=comment, before=before, problem="the model wrote no element")
        after = f"<{found.tag}>{comment}</{found.tag}>"
        return Revised(source=revise.replaced(source, found, after), number=part, comment=comment, before=before, after=after,
                       plan=f"said {comment}", seconds=1.5)
    return one


def test_the_kept_page_is_shown_with_its_numbers_only_in_the_copy_to_point_at():
    revisions = PageRevisions()
    assert revisions.state() == {"available": False, "version": 0, "versions": 0, "html": "", "marked": "", "history": [], "can_undo": False}
    with pytest.raises(Conflict, match="no page to change yet"):
        revisions.source
    revisions.start(PAGE)
    state = revisions.state()
    assert state["available"] and state["version"] == 0 and state["html"] == PAGE and not state["can_undo"]
    assert f'<footer {revise.MARK}="11">' in state["marked"] and revise.MARK not in state["html"]
    assert "first_html" not in state and revisions.state(first=True)["first_html"] == PAGE


def test_changes_are_made_from_the_end_of_the_page_to_its_beginning_and_told_in_the_order_given():
    revisions, asked = PageRevisions(), []
    revisions.start(PAGE)
    results = revisions.change(0, [{"part": 2, "comment": "Bread"}, {"part": 12, "comment": "Closed on Mondays"},
                                   {"part": 8, "comment": "Brioche"}], write(asked))
    assert asked == [12, 8, 2]  # a change further down leaves the numbers before it as they were pointed at
    assert [(r["part"], r["done"], r["plan"]) for r in results] == [(2, True, "said Bread"), (12, True, "said Closed on Mondays"), (8, True, "said Brioche")]
    state = revisions.state()
    assert state["version"] == 1 and state["can_undo"] and state["history"][0]["changes"] == results
    page = state["html"]
    assert "<h1>Bread</h1>" in page and "<div>Brioche</div>" in page and "<p>Closed on Mondays</p>" in page
    assert "<h3>Loaf</h3><p>6.50</p>" in page and "Rye" not in page  # the first card was nobody's business
    revisions.undo()
    assert revisions.state()["html"] == PAGE and revisions.version == 0
    with pytest.raises(Conflict, match="nothing to undo"):
        revisions.undo()


def test_a_changed_part_is_found_again_in_the_page_before_and_in_the_page_after():
    """To outline a change on both sides of a comparison: its number in the
    page it was pointed at in, and its number in the page it left -- which
    a part written again before it, with more parts in it or fewer, moved."""
    revisions = PageRevisions()
    revisions.start(PAGE)

    def grows(source, change, pictures):
        found = revise.find(source, int(change["part"]))
        after = "<div class=\"card\"><h3>Loaf</h3><p>6.50</p><p>Baked twice a day</p><span>New</span></div>" if found.number == 5 else "<p>Closed on Mondays</p>"
        return Revised(source=revise.replaced(source, found, after), number=found.number, comment=change["comment"],
                       before=source[found.start:found.end], after=after)

    results = revisions.change(0, [{"part": 5, "comment": "two more lines", "label": "div.card"},
                                   {"part": 12, "comment": "closed on Mondays", "label": "p"}], grows)
    assert [(r["part"], r["part_after"], r["label"]) for r in results] == [(5, 5, "div.card"), (12, 14, "p")]
    assert results[0]["before"] == "<div class=\"card\"><h3>Loaf</h3><p>6.50</p></div>" and "Baked twice a day" in results[0]["after"]
    after = revisions.source
    assert after[revise.find(after, 14).start:revise.find(after, 14).end] == "<p>Closed on Mondays</p>"
    # A part that could not be written is where it was, under a new number: it can be shown, and asked for again.
    results = revisions.change(1, [{"part": 14, "comment": "open"}, {"part": 2, "comment": "Bread"},
                                   {"part": 5, "comment": "as it was"}], lambda source, change, pictures: (
        Revised(source=source, number=14, comment="open", problem="no answer") if int(change["part"]) == 14
        else Revised(source=revise.replaced(source, revise.find(source, 5), "<div class=\"card\"><h3>Loaf</h3></div>"), number=5,
                     comment=change["comment"]) if int(change["part"]) == 5 else write()(source, change, pictures)))
    assert [(r["part"], r["part_after"], r["done"]) for r in results] == [(14, 11, False), (2, 2, True), (5, 5, True)]


def test_an_earlier_version_can_be_looked_at_and_gone_back_to():
    revisions = PageRevisions()
    revisions.start(PAGE)
    revisions.change(0, [{"part": 2, "comment": "Bread"}], write())
    revisions.change(1, [{"part": 12, "comment": "Closed"}], write())
    revisions.change(2, [{"part": 6, "comment": "Miche"}], write())
    first = revisions.version_state(0)
    assert first["html"] == PAGE and revise.MARK in first["marked"] and "Bread" in revisions.version_state(1)["html"]
    with pytest.raises(ValueError, match="no version 7"):
        revisions.version_state(7)
    revisions.restore(1)  # the two versions after it are dropped
    assert revisions.version == 1 and "<h1>Bread</h1>" in revisions.source and "Closed" not in revisions.source
    assert [entry["version"] for entry in revisions.state()["history"]] == [1]
    for wrong in (1, 5, -1):  # the version in hand, one that never was
        with pytest.raises(ValueError, match="no earlier version"):
            revisions.restore(wrong)


def test_a_page_being_changed_cannot_be_taken_back_from_under_the_change():
    revisions = PageRevisions()
    revisions.start(PAGE)
    revisions.change(0, [{"part": 2, "comment": "Bread"}], write())
    seen = []

    def meanwhile(source, change, pictures):
        for attempt in (revisions.undo, lambda: revisions.change(1, [{"part": 6, "comment": "x"}], write())):
            with pytest.raises(Conflict, match="being changed"):
                attempt()
        seen.append(True)
        return write()(source, change, pictures)

    revisions.change(1, [{"part": 12, "comment": "Closed"}], meanwhile)
    assert seen and revisions.version == 2
    revisions.undo()  # and afterwards it can
    assert revisions.version == 1


def test_a_change_the_model_could_not_make_leaves_that_part_and_keeps_the_others():
    revisions = PageRevisions()
    revisions.start(PAGE)
    results = revisions.change(0, [{"part": 2, "comment": "Bread"}, {"part": 11, "comment": "x"}], write(fail={11}))
    assert [r["done"] for r in results] == [True, False] and results[1]["problem"] == "the model wrote no element"
    assert revisions.version == 1 and "4 rue des Fours" in revisions.source and "<h1>Bread</h1>" in revisions.source
    # None could be made: the page is the version it was.
    assert [r["done"] for r in revisions.change(1, [{"part": 11, "comment": "x"}], write(fail={11}))] == [False]
    assert revisions.version == 1


def test_a_batch_that_was_stopped_leaves_the_parts_not_reached_as_they_were():
    revisions, asked = PageRevisions(), []
    revisions.start(PAGE)
    results = revisions.change(0, [{"part": 2, "comment": "Bread"}, {"part": 12, "comment": "Closed"}], write(asked),
                               stopped=lambda: bool(asked))  # stopped during the first one written
    assert asked == [12] and [r["done"] for r in results] == [False, True]
    assert "stopped before" in results[0]["problem"] and "<h1>Fournil</h1>" in revisions.source


@pytest.mark.parametrize("version,changes,error,says", [
    (3, [{"part": 2, "comment": "x"}], Conflict, "changed since these parts were pointed at"),
    (0, [], ValueError, "Point at a part"),
    (0, [{"part": 2, "comment": "x"}, {"part": 2, "comment": "y"}], ValueError, "commented twice"),
    (0, [{"part": 2, "comment": "  "}], ValueError, "Part 2 has no comment"),
    (0, [{"part": 40, "comment": "x"}], ValueError, "no part 40"),
    (0, [{"part": 4, "comment": "x"}, {"part": 9, "comment": "y"}], ValueError, "Part 9 is inside part 4"),
])
def test_what_cannot_be_done_is_refused_before_the_model_is_asked(version, changes, error, says):
    revisions, asked = PageRevisions(), []
    revisions.start(PAGE)
    with pytest.raises(error, match=says):
        revisions.change(version, changes, write(asked))
    assert asked == [] and revisions.version == 0


# ----------------------------------------------------------------- through the launcher


class _Session:
    """HTML Creator's session and the Page Agent's, as far as a change goes."""

    asked: list = []

    def __init__(self, *args, **kwargs):
        pass

    def loaded(self):
        return {}

    def revise(self, source, part, comment, *, seen=None, pictures=None, on_ready=None, device=None, **kwargs):
        if on_ready:
            on_ready()
        _Session.asked.append((part, comment, seen, device))
        return write()(source, {"part": part, "comment": comment}, pictures)

    def redraw(self, picture, comment, *, assignment, work_dir, name, **kwargs):
        _Session.asked.append(("picture", picture.name, comment, name))
        return SimpleNamespace(name=name, path=work_dir / name, width=picture.width, height=picture.height,
                               prompt=f"{picture.caption}, {comment}", seconds=7.0)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher_app.events, "LOG_FILE", tmp_path / "events.log")
    _Session.asked = []
    monkeypatch.setattr(html_creator_runner, "HtmlCreatorSession", _Session)
    monkeypatch.setattr(page_agent_runner, "PageAgent", _Session)
    creator, agent = html_creator_runner.HtmlCreatorRunner(), page_agent_runner.PageAgentRunner()
    monkeypatch.setattr(launcher_app, "html_creator_runner", creator)
    monkeypatch.setattr(launcher_app, "page_agent_runner", agent)
    return TestClient(launcher_app.app), creator, agent


def test_html_creator_changes_the_parts_pointed_at_and_can_take_it_back(client):
    api, creator, _agent = client
    assert api.get("/api/html-creator/revision").json()["available"] is False
    refused = api.post("/api/html-creator/revision/change", json={"version": 0, "changes": [{"part": 2, "comment": "Bread"}]})
    assert refused.status_code == 409 and "generate one first" in refused.json()["error"]

    creator.revisions.start(PAGE)
    creator._made_with = ("openvino", "GPU")  # what a generation leaves behind
    done = api.post("/api/html-creator/revision/change", json={"version": 0, "changes": [
        {"part": 2, "comment": "Bread", "seen": []}, {"part": 11, "comment": "Closed on Mondays", "seen": [12]}]})
    body = done.json()
    assert done.status_code == 200 and body["version"] == 1 and body["can_undo"]
    assert [(r["part"], r["done"]) for r in body["results"]] == [(2, True), (11, True)]
    assert "<h1>Bread</h1>" in body["html"] and revise.MARK not in body["html"] and revise.MARK in body["marked"]
    assert _Session.asked == [(11, "Closed on Mondays", [12], None), (2, "Bread", [], None)]

    # Pointed at on the page as it was: the page has moved on.
    stale = api.post("/api/html-creator/revision/change", json={"version": 0, "changes": [{"part": 2, "comment": "x"}]})
    assert stale.status_code == 409 and "changed since" in stale.json()["error"]
    nested = api.post("/api/html-creator/revision/change", json={"version": 1, "changes": [
        {"part": 4, "comment": "x"}, {"part": 5, "comment": "y"}]})
    assert nested.status_code == 400 and "is inside part 4" in nested.json()["error"]

    back = api.post("/api/html-creator/revision/undo").json()
    assert back["version"] == 0 and back["html"] == PAGE
    assert api.post("/api/html-creator/revision/undo").status_code == 409
    assert api.get("/api/html-creator/revision?first=1").json()["first_html"] == PAGE
    # Any version can be fetched to be looked at, and gone back to.
    creator.revisions.change(0, [{"part": 2, "comment": "Pain", "label": "h1"}], write())
    creator.revisions.change(1, [{"part": 12, "comment": "Fermé"}], write())
    assert api.get("/api/html-creator/revision/version/0").json()["html"] == PAGE
    assert api.get("/api/html-creator/revision/version/9").status_code == 400
    restored = api.post("/api/html-creator/revision/restore", json={"version": 1}).json()
    assert restored["version"] == 1 and "<h1>Pain</h1>" in restored["html"] and "Fermé" not in restored["html"]
    assert restored["history"][-1]["changes"][0]["label"] == "h1" and restored["history"][-1]["changes"][0]["part_after"] == 2
    assert api.post("/api/html-creator/revision/restore", json={"version": 1}).status_code == 400
    assert api.get("/api/doc-qa/revision").status_code == 400  # a brick that makes no page


def test_the_page_agent_changes_a_part_on_the_chip_that_wrote_the_page_and_checks_the_page_again(client):
    from page_agent import plan as planning

    api, _creator, agent = client
    assert api.post("/api/page-agent/revision/change", json={"version": 0, "changes": [{"part": 2, "comment": "x"}]}).status_code == 409
    agent._built = SimpleNamespace(plan=planning.fallback("A page for a bakery", ""), assignment=SimpleNamespace(page="GPU.1"))
    agent.revisions.start(PAGE)
    before = api.get("/api/page-agent/revision").json()
    assert before["available"] and [check["name"] for check in before["checks"]]  # the build's checks, of the page as it is
    done = api.post("/api/page-agent/revision/change", json={"version": 0, "changes": [{"part": 6, "comment": "Country loaf"}]})
    body = done.json()
    assert done.status_code == 200 and body["version"] == 1 and "<h3>Country loaf</h3>" in body["html"]
    assert _Session.asked == [(6, "Country loaf", None, "GPU.1")] and "checks" in body
    assert agent._device == "GPU.1"  # what the hardware panel lists the coding model under


def test_the_workspace_in_the_browser_asks_only_for_what_the_launcher_serves():
    static = Path(launcher_app.__file__).parent / "static"
    script = (static / "app.js").read_text(encoding="utf-8")
    page = (static / "index.html").read_text(encoding="utf-8")
    served = set(launcher_app.app.openapi()["paths"])
    asked = set(re.findall(r"/api/\$\{(?:this\.)?brick\}(/revision(?:/[a-z]+)*)", script))
    assert {"/revision", "/revision/change", "/revision/restore", "/revision/version"} <= asked
    for path in asked:
        assert any(route == "/api/{demo_id}" + path or route.startswith("/api/{demo_id}" + path + "/{") for route in served), path
    for brick in ("htmlc", "agent"):  # what opens it, in each of the two bricks
        assert f'id="{brick}-change"' in page and f'el("{brick}-change")' in script


def test_pictures_to_draw_again_come_after_every_part_to_write_again():
    """A picture drawn again changes no number, and where the two models
    share a chip they then change places once and not at every turn."""
    revisions, asked = PageRevisions(), []
    revisions.start(PAGE)
    revisions.change(0, [{"part": 12, "comment": "warmer", "picture": "oven.jpg"}, {"part": 2, "comment": "Bread"},
                         {"part": 9, "comment": "Seigle"}], write(asked))
    assert asked == [9, 2, 12]


def test_a_comment_on_a_picture_of_a_built_page_goes_to_the_image_model(client, tmp_path):
    from html_creator import pictures as picture_kit
    from page_agent import plan as planning

    api, creator, agent = client
    page = PAGE.replace("<header><h1>Fournil</h1></header>", '<header class="hero"><h1>Fournil</h1></header>').replace(
        ".card { padding: 1rem; }", '.card { padding: 1rem; } .hero { background-image: url("hero.jpg"); }').replace(
        "<h3>Loaf</h3>", '<img src="loaf.jpg" alt="a loaf"><h3>Loaf</h3>')
    for name in ("hero.jpg", "loaf.jpg", "loaf-r2.jpg"):
        (tmp_path / name).write_bytes(b"not really a picture")
    agent._built = SimpleNamespace(plan=planning.fallback("A page for a bakery", ""), assignment=SimpleNamespace(page="GPU", images="GPU", planner="NPU"))
    agent._work_dir = tmp_path
    agent.revisions.start(page, [picture_kit.Picture("hero.jpg", tmp_path / "hero.jpg", "image/jpeg", 1024, 576, "a bakery at dawn"),
                                 picture_kit.Picture("loaf.jpg", tmp_path / "loaf.jpg", "image/jpeg", 768, 512, "a country loaf")])
    state = api.get("/api/page-agent/revision").json()
    # The parts that show a picture: the header, by its class's rule, and the <img> itself.
    assert state["pictures"] == {"1": "hero.jpg", "6": "loaf.jpg"}

    done = api.post("/api/page-agent/revision/change", json={"version": 0, "changes": [
        {"part": 6, "comment": "on a wooden board", "picture": "loaf.jpg"}]})
    body = done.json()
    assert done.status_code == 200 and body["version"] == 1
    assert _Session.asked == [("picture", "loaf.jpg", "on a wooden board", "loaf-r2.jpg")]  # a name of its own: the old one is kept
    assert 'src="loaf-r2.jpg"' in body["html_source"] and "loaf.jpg" not in body["html_source"]
    assert body["results"][0]["picture"] == "loaf.jpg" and "a country loaf, on a wooden board" in body["results"][0]["plan"]
    assert body["pictures"] == {"1": "hero.jpg", "6": "loaf-r2.jpg"}
    assert body["html"].count("data:image/jpeg") == 2  # both pictures are in the page that is shown
    # Taken back, the page shows the picture it had: its file was never touched.
    assert 'src="loaf.jpg"' in api.post("/api/page-agent/revision/undo").json()["html_source"]

    # HTML Creator's pictures are files somebody gave it: nothing there draws.
    creator.revisions.start(page)
    creator._made_with = ("openvino", "GPU")
    refused = api.post("/api/html-creator/revision/change", json={"version": 0, "changes": [
        {"part": 6, "comment": "warmer", "picture": "loaf.jpg"}]})
    assert refused.status_code == 400 and "no image model here" in refused.json()["error"]
