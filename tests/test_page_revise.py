"""Changing one part of a generated page (html_creator.revise): the part is
found in the page's source, written again by the model, and put back with
every other character of the page left as it was. No model is loaded here:
what is checked is the finding, the putting back, and what an answer is
refused or asked again for."""
from __future__ import annotations

import pytest
from html_creator import pictures as picture_kit
from html_creator import revise
from html_creator import session as creator
from html_creator.session import HtmlCreatorSession
from pantherlake_ai_core.engine import Engine

PAGE = """<!DOCTYPE html>
<html><head><title>Atelier Quay</title>
<style>
:root { --green: #2a4d3c; }
body { font-family: serif; }
.hero h1 { font-size: 3rem; }
.card, .tile { padding: 1rem; }
.card:hover { animation: pop 1s; }
.footer { color: red; }
@media (max-width: 600px) { .card { padding: 0; } .footer { display: none; } }
@keyframes pop { from { opacity: 0 } to { opacity: 1 } }
@keyframes spin { from { opacity: 0 } to { opacity: 1 } }
ul li { margin: 0; }
</style></head>
<body>
<header class="hero"><h1>A quieter base</h1><p>First line<br>second</header>
<main>
  <section id="rooms"><div class="card"><img src="a.jpg" alt="x"><h2>Suite, 180 euros</h2><ul><li>one<li>two</ul></div>
  <div class="card"><svg viewBox="0 0 1 1"><path d="M0 0"/><circle r="1"/></svg><p>Open paragraph<div>block</div></div></section>
</main>
<footer class="footer">Rue de la Paix</footer>
<script>
var tag = "<div>not a tag</div>";
count(document.getElementById('rooms'), 3);
</script>
</body></html>"""


def numbered(source=PAGE):
    return {part.number: part for part in revise.parts(source) if part.number is not None}


def text_of(number, source=PAGE):
    part = numbered(source)[number]
    return source[part.start:part.end]


def test_every_part_that_can_be_seen_is_numbered_and_ends_where_the_browser_ends_it():
    parts = numbered()
    assert [parts[n].tag for n in sorted(parts)] == [
        "header", "h1", "p", "main", "section", "div", "img", "h2", "ul", "li", "li", "div", "svg", "path", "circle", "p", "div", "footer"]
    assert text_of(3) == "<p>First line<br>second"  # a paragraph nobody closed ends with what holds it
    assert text_of(10) == "<li>one" and text_of(11) == "<li>two"  # an item ends where the next begins
    assert text_of(16) == "<p>Open paragraph"  # and a paragraph where a block begins
    assert text_of(14) == '<path d="M0 0"/>' and text_of(13).endswith("</svg>")  # in a drawing, "/>" closes
    assert text_of(7) == '<img src="a.jpg" alt="x">'
    assert text_of(18) == '<footer class="footer">Rue de la Paix</footer>'
    # Nothing is made of what a script holds in a string, and a script is not pointed at.
    assert all(part.tag not in ("script", "style", "br") for part in parts.values())
    assert revise.inside(PAGE, 5) == list(range(6, 18))


def test_the_marked_copy_differs_from_the_page_by_its_numbers_alone():
    import re

    marked = revise.marked(PAGE)
    assert re.sub(rf' {revise.MARK}="\d+"', "", marked) == PAGE
    assert f'<section {revise.MARK}="5" id="rooms">' in marked and f'<li {revise.MARK}="11">two' in marked
    assert revise.MARK not in PAGE


def test_a_part_is_replaced_and_not_another_character_moves():
    part = numbered()[6]
    new = revise.replaced(PAGE, part, '<div class="card">NEW</div>')
    assert new == PAGE[:part.start] + '<div class="card">NEW</div>' + PAGE[part.end:]
    with pytest.raises(ValueError, match="no part 99"):
        revise.find(PAGE, 99)


def test_what_the_browser_saw_inside_a_part_has_to_be_what_the_source_has_inside_it():
    assert revise.same_inside(PAGE, 5, list(range(6, 18)))
    assert revise.same_inside(PAGE, 5, None)  # nothing to compare with: asked from a tool
    assert not revise.same_inside(PAGE, 5, [6, 7, 8])  # the browser closed the section earlier than the source does


def test_only_the_rules_that_apply_to_the_part_are_shown_to_the_model():
    rules = revise.styles_for(PAGE, text_of(6))
    assert ".card, .tile { padding: 1rem; }" in rules and "ul li { margin: 0; }" in rules
    assert ":root { --green: #2a4d3c; }" in rules and "body { font-family: serif; }" in rules  # what applies everywhere
    assert "@media (max-width: 600px) {\n  .card { padding: 0; }\n}" in rules  # the part's rule in it, and not the footer's
    assert ".footer" not in rules and ".hero" not in rules
    assert "@keyframes pop" in rules and "@keyframes spin" not in rules  # the animation its rules name


def test_the_model_is_told_where_the_part_is_what_the_page_says_and_which_script_touches_it():
    asked = revise.request(PAGE, numbered()[5], "  three   rooms\n instead of two ", "PICTURES -- a.jpg, b.jpg")
    assert "THE PAGE: Atelier Quay" in asked and "WHERE THE PART IS: main > section#rooms" in asked
    assert "- header.hero -- A quieter base First line second" in asked and "- footer.footer -- Rue de la Paix" in asked
    assert "A quieter base First line second [THE PART] Rue de la Paix" in asked  # the rest of the page, in words
    assert "count(document.getElementById('rooms'), 3);" in asked and "not a tag" in asked  # with the line before it
    assert "PICTURES -- a.jpg, b.jpg" in asked
    assert "WHAT TO CHANGE: three rooms instead of two" in asked
    assert asked.rstrip().endswith("then the <section> element, from <section to </section>.")
    assert "THE PART (a <section> element):\n" + text_of(5) in asked


@pytest.mark.parametrize("written,element,css,script,problem", [
    ('<!-- I change: the title -->\n<div class="card"><h2>Room</h2></div>\n<style>.card h2 { color: red; }</style>\n'
     "<script>go();</script>", '<div class="card"><h2>Room</h2></div>', ".card h2 { color: red; }", "go();", ""),
    ('```html\n<div class="card">x</div>\n```', '<div class="card">x</div>', "", "", ""),
    ('<div class="card">x</div>\nThis adds the room.', '<div class="card">x</div>', "", "", ""),  # what follows is left out
    ("<section>x</section>", "", "", "", "the model wrote a <section> where the part is a <div>"),
    ('<div class="card"><h2>cut', "", "", "", "the model's answer was cut off before the part ended"),
    ("Sure! Here it is", "", "", "", "the model wrote no element"),
])
def test_an_answer_is_one_element_with_perhaps_rules_and_a_script(written, element, css, script, problem):
    answer = revise.read_answer(written, "div")
    assert (answer.element, answer.css, answer.script, answer.problem) == (element, css, script, problem)
    if "I change:" in written:
        assert answer.plan == "the title"


def test_the_rules_and_the_script_go_where_they_have_the_last_word_and_repeats_are_dropped():
    part = numbered()[6]
    answer = revise.Answer(
        element='<div class="card">NEW</div>',
        css=".card, .tile { padding: 1rem; }\n.card { padding: 2rem; }", script="again();",
    )
    page = revise.changed(PAGE, part, answer)
    assert '<div class="card">NEW</div>' in page
    added = page[page.index("/* added with a change to one part */"):page.index("</style>")]
    assert ".card {\n  padding: 2rem;\n}" in added and ".tile" not in added  # the rule the page had is not said twice
    assert page.index("ul li { margin: 0; }") < page.index("padding: 2rem")  # after the page's own: it wins
    assert page.rstrip().endswith("again();\n</script>\n</body></html>") and page.index("count(") < page.index("again();")
    assert revise.changed(PAGE, part, revise.Answer(element=text_of(6))) == PAGE  # nothing changed, nothing moves


def test_what_is_wrong_with_an_answer_is_seen_in_its_own_text():
    part, pictures = numbered()[6], ["a.jpg", "b.jpg"]
    fine = revise.Answer(element='<div class="card"><img src="b.jpg" alt="y"><a href="#rooms">Rooms</a></div>')
    assert revise.faults(PAGE, part, fine, pictures) == []
    wrong = revise.Answer(
        element='<div class="card"><a class="btn" href="#suites">See</a><img src="img/c.jpg" alt="z">'
                '<p class="note" style="color: red">styled where it stands</p></div>',
        css=".card.open p { display: block; }", script="document.getElementById('price').textContent = '1';",
    )
    found = revise.faults(PAGE, part, wrong, pictures)
    assert any('the class "btn" has no style rule' in fault for fault in found)
    assert not any('"note"' in fault for fault in found)  # it carries its own style
    assert any("the link to #suites goes nowhere" in fault for fault in found)
    assert any("an element with the id price" in fault for fault in found)
    assert any("the rule for .open never applies" in fault for fault in found)
    assert any("the picture c.jpg does not exist" in fault for fault in found)
    assert revise.faults_note(found).count("\n- ") == len(found)
    # A script that gives the class is what makes its rule apply.
    opens = revise.Answer(element=text_of(6), css=".card.open p { display: block; }", script="card.classList.toggle('open');")
    assert revise.faults(PAGE, part, opens) == []


def test_a_badge_placed_against_nothing_gets_the_part_to_be_placed_against():
    part = numbered()[6]
    badge = '<span style="position: absolute; top: 10px">Popular</span>'
    fixed = revise.anchored(PAGE, part, revise.Answer(element=f'<div class="card">{badge}</div>'))
    assert fixed.element == f'<div class="card" style="position: relative">{badge}</div>' and fixed.notes
    merged = revise.anchored(PAGE, part, revise.Answer(element=f'<div class="card" style="color: red">{badge}</div>'))
    assert merged.element.startswith('<div class="card" style="position: relative; color: red">')
    # Left alone: nothing placed, the part already placed by its tag, or by a rule of its own.
    for element, css in ((text_of(6), ""), (f'<div class="card" style="position: sticky">{badge}</div>', ""),
                         (f'<div class="card">{badge}</div>', ".card { position: relative; }")):
        answer = revise.anchored(PAGE, part, revise.Answer(element=element, css=css))
        assert answer.element == element and not answer.notes


class _Model:
    """Writes what it is told to, in turn, and keeps what it was asked."""

    last_stats = None

    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    def count_tokens(self, text):
        return len(text.split())

    def answer(self, system, request, *, max_tokens, control=None, begin=None, **kwargs):
        self.asked.append((request, begin, max_tokens))
        return self.answers.pop(0)


def revising(monkeypatch, *answers):
    session = HtmlCreatorSession(Engine.OPENVINO, compute_device="GPU")
    session._llm = _Model(*answers)
    return session, session._llm


def test_one_part_is_written_again_and_the_rest_of_the_page_is_the_same(monkeypatch):
    new = '<div class="card"><img src="a.jpg" alt="x"><h2>Suite, 190 euros</h2><ul><li>one<li>two</ul></div>'
    session, model = revising(monkeypatch, f"<!-- I change: the price -->\n{new}")
    done = session.revise(PAGE, 6, "190 euros", seen=[7, 8, 9, 10, 11])
    part = numbered()[6]
    assert done.source == PAGE[:part.start] + new + PAGE[part.end:] and not done.problem
    assert (done.before, done.after, done.plan, done.attempts) == (text_of(6), new, "the price", 1)
    assert model.asked[0][1] == revise.BEGIN and "WHAT TO CHANGE: 190 euros" in model.asked[0][0]
    assert 700 <= model.asked[0][2] <= creator._REVISE_MAX_TOKENS


def test_an_answer_that_is_another_element_is_asked_again_on_the_parts_own_tag(monkeypatch):
    session, model = revising(monkeypatch, "<!-- I change: it --><section>x</section>", '<div class="card">y</div>')
    done = session.revise(PAGE, 6, "shorter")
    assert done.after == '<div class="card">y</div>' and done.attempts == 2 and not done.problem
    assert [begin for _request, begin, _room in model.asked] == [revise.BEGIN, "<div"]

    session, model = revising(monkeypatch, "no", "still no")
    failed = session.revise(PAGE, 6, "shorter")
    assert failed.problem == "the model wrote no element" and failed.source == PAGE and failed.after == ""


def test_an_answer_with_faults_is_told_them_once_and_the_better_one_is_kept(monkeypatch):
    faulty = '<div class="card"><a class="btn" href="#suites">See</a></div>'
    better = '<div class="card"><a href="#rooms">See</a></div>'
    session, model = revising(monkeypatch, faulty, better)
    done = session.revise(PAGE, 6, "a link to the rooms")
    assert done.after == better and done.attempts == 2 and done.notes == []
    assert "YOUR FIRST ANSWER HAD THESE FAULTS" in model.asked[1][0] and "#suites goes nowhere" in model.asked[1][0]
    assert "write its rule in the <style> element" in model.asked[1][0]  # what is wrong, and what to do about it

    session, model = revising(monkeypatch, faulty, faulty)  # no better the second time: kept, and said
    kept = session.revise(PAGE, 6, "a link to the rooms")
    assert kept.after == faulty and len(kept.notes) == 2 and all(note.startswith("to check: ") for note in kept.notes)


def test_a_fault_the_part_already_had_is_not_asked_about_again(monkeypatch):
    """A footer whose links went nowhere before anybody touched it: seen on a
    page the Page Agent built (2026-10-11), where a line added to it was
    asked for twice, 21 s instead of 8, for a fault that was not the change's."""
    page = PAGE.replace('<footer class="footer">Rue de la Paix</footer>',
                        '<footer class="footer"><a href="#story">Our story</a> Rue de la Paix</footer>')
    footer = next(n for n, part in numbered(page).items() if part.tag == "footer")
    kept = '<footer class="footer"><a href="#story">Our story</a> Rue de la Paix. Closed on Mondays.</footer>'
    session, model = revising(monkeypatch, kept)
    done = session.revise(page, footer, "say we close on Mondays")
    assert done.after == kept and done.attempts == 1 and done.notes == []
    # A link the change itself brings is another matter.
    worse = kept.replace("Closed on Mondays.", '<a href="#hours">Hours</a>')
    session, model = revising(monkeypatch, worse, worse)
    assert session.revise(page, footer, "link to the hours").attempts == 2


def test_a_part_the_browser_shows_otherwise_or_an_empty_comment_is_refused_before_the_model_is_asked(monkeypatch):
    session, model = revising(monkeypatch)
    with pytest.raises(ValueError, match="Say what to change"):
        session.revise(PAGE, 6, "   ")
    with pytest.raises(ValueError, match="point at the part around it"):
        session.revise(PAGE, 5, "shorter", seen=[6, 7])
    with pytest.raises(ValueError, match="no part 40"):
        session.revise(PAGE, 40, "shorter")
    assert model.asked == []


def test_the_pictures_a_page_may_use_are_named_to_the_model_and_checked_in_its_answer(monkeypatch, tmp_path):
    offered = [picture_kit.Picture("a.jpg", tmp_path / "a.jpg", "image/jpeg", caption="the suite"),
               picture_kit.Picture("b.jpg", tmp_path / "b.jpg", "image/jpeg", caption="the loft")]
    wrong = '<div class="card"><img src="loft.jpg" alt="the loft"></div>'
    right = '<div class="card"><img src="b.jpg" alt="the loft"></div>'
    session, model = revising(monkeypatch, wrong, right)
    done = session.revise(PAGE, 6, "show the loft", pictures=offered)
    assert done.after == right and "- b.jpg: the loft" in model.asked[0][0]
    assert "the picture loft.jpg does not exist" in model.asked[1][0]
