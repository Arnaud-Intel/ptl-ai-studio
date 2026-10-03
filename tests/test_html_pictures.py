"""Pictures for a generated page: what the model is told, and how the file
names it writes become the pictures themselves -- with no model loaded."""
from __future__ import annotations

import base64

import pytest
from html_creator import pictures
from html_creator import session as html_session
from html_creator.session import HtmlCreatorSession
from pantherlake_ai_core.engine import Engine

SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 800"><rect width="1200" height="800"/></svg>'


@pytest.fixture
def kit(tmp_path):
    (tmp_path / "hero-ridge.svg").write_text(SVG, encoding="utf-8")
    (tmp_path / "fjord.svg").write_text(SVG.replace("1200 800", "600 400"), encoding="utf-8")
    (tmp_path / "notes.md").write_text("not a picture", encoding="utf-8")
    (tmp_path / "captions.txt").write_text(
        "# what each shows\nhero-ridge.svg: aurora over a ridge\nFJORD.svg: a fjord at sunrise\n", encoding="utf-8"
    )
    return tmp_path


def test_a_folder_becomes_named_sized_captioned_pictures(kit):
    offered, notes = pictures.load(str(kit))
    assert notes == []
    assert [(p.name, p.width, p.height, p.caption) for p in offered] == [
        ("fjord.svg", 600, 400, "a fjord at sunrise"),  # captions match whatever the case
        ("hero-ridge.svg", 1200, 800, "aurora over a ridge"),
    ]


def test_the_model_is_told_the_files_that_exist(kit):
    offered, _ = pictures.load(str(kit))
    told = pictures.manifest(offered)
    assert "- hero-ridge.svg (1200x800): aurora over a ridge" in told
    assert "the only images that exist" in told and "notes.md" not in told


def test_a_folder_with_no_pictures_or_no_folder_says_so(tmp_path):
    with pytest.raises(ValueError, match="No pictures"):
        pictures.load(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        pictures.load(str(tmp_path / "missing"))


def test_only_so_many_pictures_are_offered_and_odd_names_are_left_out(tmp_path):
    for i in range(pictures.MAX_PICTURES + 2):
        (tmp_path / f"shot-{i:02d}.svg").write_text(SVG, encoding="utf-8")
    (tmp_path / "it's mine.svg").write_text(SVG, encoding="utf-8")
    offered, notes = pictures.load(str(tmp_path))
    assert len(offered) == pictures.MAX_PICTURES
    assert any("it's mine.svg was left out" in note for note in notes)
    assert any(f"first {pictures.MAX_PICTURES}" in note for note in notes)


def test_an_oversized_picture_that_cannot_be_scaled_is_left_out(tmp_path, monkeypatch):
    monkeypatch.setattr(pictures, "MAX_BYTES", 200)
    (tmp_path / "small.svg").write_text(SVG, encoding="utf-8")
    (tmp_path / "huge.svg").write_text(SVG + " " * 400, encoding="utf-8")
    offered, notes = pictures.load(str(tmp_path))
    assert [p.name for p in offered] == ["small.svg"]
    assert "huge.svg was left out" in notes[0]


@pytest.mark.parametrize(
    "written",
    [
        '<img src="hero-ridge.svg" alt="x">',
        "<img src='./hero-ridge.svg'>",
        '<img src="images/hero-ridge.svg">',
        '<img src="/assets/img/hero-ridge.svg">',
        '<img src="https://example.com/pics/hero-ridge.svg">',
        "<style>.hero{background:url(hero-ridge.svg) center/cover}</style>",
        '<style>.hero{background-image: url("hero-ridge.svg")}</style>',
        "<script>const shots = ['hero-ridge.svg', 'other.png'];</script>",
        "<script>const shot = `hero-ridge.svg`;</script>",
        '<img srcset="hero-ridge.svg 2x">',
    ],
)
def test_every_way_a_page_names_a_picture_gets_the_picture(kit, written):
    offered, _ = pictures.load(str(kit))
    page, used = pictures.embed(written, offered)
    assert used == ["hero-ridge.svg"]
    assert "hero-ridge.svg" not in page and "example.com" not in page and "images/" not in page
    encoded = page.split("base64,")[1].split('"')[0].split("'")[0].split(")")[0].split("`")[0].split(" ")[0]
    assert base64.b64decode(encoded).decode("utf-8") == SVG


def test_text_about_a_picture_is_not_a_reference_to_it(kit):
    offered, _ = pictures.load(str(kit))
    for written in (
        '<img src="x.png" alt="hero-ridge.svg">',
        "<p>Photo: hero-ridge.svg</p>".replace(" hero", "&nbsp;hero"),
        '<img src="my-hero-ridge.svg.bak">',
    ):
        assert pictures.embed(written, offered) == (written, [])


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.asked = None
        self.last_stats = None

    def answer(self, system_prompt, user_prompt, max_tokens=512, control=None, sample=True):
        self.asked = (system_prompt, user_prompt)
        self.sampled = sample
        return self.reply


def _session(monkeypatch, reply, loads=None):
    llm = _FakeLLM(reply)

    def create_llm(*args, **kwargs):
        if loads is not None:
            loads.append(1)
        return llm

    monkeypatch.setattr(html_session, "create_llm", create_llm)
    return HtmlCreatorSession(Engine.OPENVINO, compute_device="GPU"), llm


def test_a_page_with_pictures_is_told_about_them_and_comes_back_with_them_inside(kit, monkeypatch):
    written = '<!DOCTYPE html><html><body><img src="hero-ridge.svg" alt="Aurora"></body></html>'
    session, llm = _session(monkeypatch, written)
    result = session.generate(prompt="a travel page", pictures=str(kit))
    system_prompt, user_prompt = llm.asked
    assert "PICTURES list" in system_prompt
    assert user_prompt.startswith("a travel page") and "- fjord.svg (600x400): a fjord at sunrise" in user_prompt
    assert 'src="data:image/svg+xml;base64,' in result.html and result.html_truncated is False
    assert result.html_source == written  # what a person can read
    assert (result.pictures_offered, result.pictures_used) == (2, ["hero-ridge.svg"])


def test_a_page_without_pictures_is_asked_and_returned_as_before(monkeypatch):
    written = "<!DOCTYPE html><html><body>Hello</body></html>"
    session, llm = _session(monkeypatch, written)
    result = session.generate(prompt="a bakery")
    assert "PICTURES" not in llm.asked[0] and llm.asked[1] == "a bakery"
    assert result.html == written and result.html_source is None and result.pictures_offered == 0


def test_the_model_is_kept_between_pages_unless_asked_for_the_same_page_every_time(monkeypatch):
    loads = []
    session, llm = _session(monkeypatch, "<!DOCTYPE html><html></html>", loads)
    result = session.generate(prompt="a bakery")
    session.generate(prompt="a bakery")
    assert len(loads) == 1 and llm.sampled is True and result.repeatable is False  # off by default

    result = session.generate(prompt="a bakery", repeatable=True)
    session.generate(prompt="a bakery", repeatable=True)
    assert len(loads) == 3  # a fresh model for each of the two
    assert llm.sampled is False and result.repeatable is True


def test_the_sandbox_rules_reach_the_model(monkeypatch):
    session, llm = _session(monkeypatch, "<!DOCTYPE html><html></html>")
    session.generate(prompt="a game")
    assert "localStorage" in llm.asked[0] and "alert()" in llm.asked[0]
