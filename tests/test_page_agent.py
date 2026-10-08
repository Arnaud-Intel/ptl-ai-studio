"""The page agent (experimental): a planner, an image model and a coding
model on three chips, conducted and checked by plain code. Everything here
runs without a model: the plan's parser, the art direction, who-works-where,
the checks, the repeated-picture repair, the loop watch, the conductor's
order of work, and the routes."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from page_agent import art_direction, checks, conductor, runaway
from page_agent import plan as planning
from page_agent.conductor import PageAgent, assign
from page_agent.repeats import give_repeats_their_own
from page_agent.types import Assignment, DrawnPicture, PictureStats
from pantherlake_ai_core.engine import GpuDevice
from pantherlake_ai_core.types import GenerationControl, GenerationStats

from launcher import app as launcher_app

IGPU = GpuDevice("GPU.0", "Intel(R) Arc(TM) B390 GPU (iGPU)", None)
DGPU = GpuDevice("GPU.1", "Intel(R) Arc(TM) Pro B60 Graphics (dGPU)", None)

# --- the plan ---------------------------------------------------------------------------

# The form the planner is asked for: each thing the page presents, then its photograph.
_BRIEF = """NAME: Col Bleu Bikes
HEADLINE: Rent it. Ride it. Repeat.
STYLE: crisp and outdoorsy, glacier blue, spruce green, a tight modern sans
SECTIONS: Bikes | Trails | Workshop
HERO PHOTO: A mountain bike on a wooden rack in an alpine meadow, golden hour
THING 1: Trail hardtail
PHOTO 1: A hardtail bike on a rocky trail, spruce trees in the distance, morning mist
THING 2: Enduro full-suspension
PHOTO 2: A full-suspension bike with mud on the chain, midday sun
THING 3: Electric mountain bike
PHOTO 3: An electric mountain bike among pine trees and wildflowers, late afternoon light
STORY PHOTO: A mechanic truing a wheel at a workbench, seen from the side
DETAIL PHOTO: A close-up of a helmet strap being fastened, shallow focus"""

# An older and shorter form, which still has to be read: three pictures, none of them for a card.
_ANSWER = """TITLE: Baked with Care
STYLE: warm and cozy | cream, golden, soft beige
SECTIONS: Welcome | About Us | Our Bakes | Visit
HERO: A golden sunrise over a rustic table of pastries, warm editorial photograph
PICTURE 2: A baker's hands dusting flour from a rolling pin, warm editorial photograph
PICTURE 3: A cozy dining nook with a view of the open kitchen, warm editorial photograph"""


def test_a_plan_names_the_page_and_gives_every_picture_a_place():
    plan = planning.parse("a bike rental page", _BRIEF)
    assert (plan.title, plan.headline) == ("Col Bleu Bikes", "Rent it. Ride it. Repeat.")
    assert plan.sections == ["Bikes", "Trails", "Workshop"]
    assert plan.offers == ["Trail hardtail", "Enduro full-suspension", "Electric mountain bike"]
    assert [(p.name, p.role, p.width, p.height) for p in plan.pictures] == [
        ("hero.jpg", "hero", 1024, 576),
        ("picture-2.jpg", "offer", 768, 512), ("picture-3.jpg", "offer", 768, 512), ("picture-4.jpg", "offer", 768, 512),
        ("picture-5.jpg", "story", 640, 768), ("picture-6.jpg", "closing", 1024, 576),
    ]
    assert plan.pictures[2].prompt.startswith("A full-suspension bike") and plan.notes == [] and plan.copied == 0


def test_the_coder_is_told_the_request_first_then_the_plan_then_how_the_studio_builds_a_page():
    prompt = planning.parse("a bike rental page", _BRIEF).page_prompt()
    assert prompt.startswith("a bike rental page")
    for said in ("Name: Col Bleu Bikes", "Headline: Rent it. Ride it. Repeat.", "On offer: Trail hardtail | Enduro"):
        assert said in prompt
    assert prompt.index("the request wins") < prompt.index("HOW THE STUDIO BUILDS IT")
    # Every picture has its place, by file name, and the cards are named in the order of their pictures.
    assert "Trail hardtail, Enduro full-suspension, Electric mountain bike" in prompt
    assert "picture-2.jpg, picture-3.jpg and picture-4.jpg on top" in prompt and '<img src="picture-5.jpg">' in prompt
    # The rows and bands the model got wrong when they were only described are rules now: shown to it, and
    # said to be in the page already. The two pictures that sit behind text are placed by the rule that names them.
    assert "these rules are in the page already" in prompt and "Do not write them again" in prompt
    assert ".figures { display: grid;" in prompt and ".split { display: grid;" in prompt
    assert 'url("hero.jpg") center / cover; }' in prompt and ".dark *, .hero *, .closing * { color: #fff; }" in prompt
    assert '.closing { text-align: center; background: linear-gradient(rgba(6,10,18,.72), rgba(6,10,18,.72)), url("picture-6.jpg")' in prompt
    assert "document.documentElement.classList.add('js')" in prompt  # nothing stays hidden if the script does not run
    assert planning.parse("x", _BRIEF).closing().endswith("Start your reply with <!DOCTYPE html>.")


def test_the_direction_only_asks_for_pictures_the_page_was_given():
    three = planning.parse("a bakery page", _ANSWER)
    assert [(p.name, p.role, p.width, p.height) for p in three.pictures] == [
        ("hero.jpg", "hero", 1024, 576), ("picture-2.jpg", "feature", 768, 512), ("picture-3.jpg", "feature", 768, 512),
    ]
    direction = art_direction.for_plan(three.pictures, three.offers)
    assert "three .card without pictures" in direction and ".closing { text-align: center; background: var(--dark); }" in direction
    assert '<img src="picture-2.jpg">' in direction and '<img src="picture-3.jpg">' in direction
    assert "picture-4.jpg" not in direction and "picture-6.jpg" not in direction
    # With no picture at all there is still a page to describe, and nothing in it names a file.
    assert "background: var(--dark); }\n.hero h1" in art_direction.for_plan([]) and ".jpg" not in art_direction.for_plan([])


def test_the_studios_layout_rules_go_into_every_page_ahead_of_its_own():
    plan = planning.parse("a bike rental page", _BRIEF)
    page = (
        "<!DOCTYPE html><html><head><style>\n:root { --accent: #f60; }\nh2 { color: red; }\n</style></head><body>"
        '<section class="hero"><h1>Rent it</h1></section><section class="band closing"><h2>Book</h2></section></body></html>'
    )
    styled = art_direction.with_stylesheet(page, plan.pictures)
    ours, theirs = styled.index(".figures { display: grid;"), styled.index("h2 { color: red; }")
    assert styled.index("<style>") < ours < theirs  # the page's own rules come after, and so may overrule
    assert styled.index("--accent: #c2410c") < styled.index("--accent: #f60")  # as its colours overrule the defaults
    assert 'url("hero.jpg") center / cover' in styled and 'url("picture-6.jpg") center / cover' in styled
    assert art_direction.with_stylesheet(styled, plan.pictures) == styled  # once
    # A file the stylesheet names is a picture the page shows: with no element for it, the rule names none.
    plain = art_direction.with_stylesheet(page.replace('class="hero"', 'class="hero-banner"'), plan.pictures)
    assert "hero.jpg" not in plain and ".hero { min-height: 88vh;" in plain and "picture-6.jpg" in plain
    # A page that wrote no stylesheet is given one.
    bare = art_direction.with_stylesheet("<html><head><title>x</title></head><body><p>hello</p></body></html>", [])
    assert bare.index("<style>") < bare.index("</head>") and ".wrap { max-width: 1120px;" in bare
    assert art_direction.with_stylesheet("<p>hello</p>", []).startswith("<style>")


def test_a_small_models_liberties_with_the_format_are_tolerated():
    answer = (
        "<think>the user wants a page</think>\n"
        "**Title:** Ride the Alps\n"
        "- STYLE - alpine blue, white\n"
        "Sections: Bikes; Prices; Trails\n"
        "1. HERO: A trail at sunrise\n"
        "Picture two: A bike in a workshop\n"
        "IMAGE 3 (landscape): A trail at sunrise\n"  # the same sentence again: one picture, not two
        "PICTURE 4: a photograph of THING 3\n"  # the instruction copied back: no picture
        "PICTURE \u79d1\u5b66: <A mechanic at a bench>\n"  # a garbled number: the picture after the one before
    )
    plan = planning.parse("bike rental", answer)
    assert plan.title == "Ride the Alps" and plan.style == "alpine blue, white" and plan.sections == ["Bikes", "Prices", "Trails"]
    assert [(p.name, p.prompt) for p in plan.pictures] == [
        ("hero.jpg", "A trail at sunrise"), ("picture-2.jpg", "A bike in a workshop"), ("picture-5.jpg", "A mechanic at a bench"),
    ]
    # Short of three pictures for the cards, the cards go without and the picture there is sits beside text.
    assert [p.role for p in plan.pictures] == ["hero", "feature", "story"]


def test_what_an_image_model_cannot_write_is_taken_out_of_a_picture():
    clean = planning.without_writing
    assert clean("A hardtail bike with a rack on a gravel path, morning light") == "A hardtail bike with a rack on a gravel path, morning light"
    assert clean("A clerk at a desk with a tablet and a map, beside a bike rack") == "A clerk at a desk, beside a bike rack"
    assert clean("A singer walking past a stage with a banner reading 'Les Heures Bleues'") == "A singer walking past a stage"
    assert clean("A coffee cup with a steamy top, golden light, a wall lamp glowing at 8:07") == "A coffee cup with a steamy top, golden light"
    assert clean('A cabin called "Maison des Pins" between tall pines') == "A cabin between tall pines"
    assert clean("A potter's workshop at dawn, light through the bakers' window") == "A potter's workshop at dawn, light through the bakers' window"
    # A picture *of* writing leaves nothing to photograph.
    assert clean("A close-up of a ticket with the festival's dates") == "" and clean("A tablet with a savings estimator") == ""


def test_a_card_whose_picture_cannot_be_used_gets_one_from_its_own_name():
    answer = _BRIEF.replace("PHOTO 2: A full-suspension bike with mud on the chain, midday sun", "PHOTO 2: A price list on a chalkboard")
    plan = planning.parse("a bike rental page", answer)
    assert [p.role for p in plan.pictures[1:4]] == ["offer"] * 3  # the three cards stay alike
    assert plan.pictures[2].prompt == "Enduro full-suspension, editorial photograph, soft natural light"
    assert "were of writing" in plan.notes[0]
    # The same sentence for two pictures is one picture: the other keeps its place and the gap is filled the same way.
    twice = _BRIEF.replace("PHOTO 3: An electric mountain bike among pine trees and wildflowers, late afternoon light",
                           "PHOTO 3: A full-suspension bike with mud on the chain, midday sun")
    plan = planning.parse("a bike rental page", twice)
    assert plan.pictures[3].prompt.startswith("Electric mountain bike, editorial") and plan.pictures[4].role == "story"


def test_the_prompts_own_example_sent_back_is_not_a_plan():
    example = planning.SYSTEM_PROMPT.split("never its content:\n")[1]
    plan = planning.parse("an architecture studio", example)
    assert plan.copied == 9 and plan.offers == []  # six photographs and three things, none of them ours
    assert [p.name for p in plan.pictures] == ["hero.jpg"] and "an architecture studio" in plan.pictures[0].prompt
    assert any("own example" in note for note in plan.notes)


def test_a_plan_that_cannot_be_read_costs_quality_not_the_page():
    plan = planning.parse("a page for a florist", "Sure! Here is a wonderful idea for your page...")
    assert [p.name for p in plan.pictures] == ["hero.jpg"] and "a page for a florist" in plan.pictures[0].prompt
    assert len(plan.notes) == 2  # no picture, no name: both said
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
    assert not any(c.passed for c in verdict.values() if c.name not in (checks.ONCE, checks.FIGURES))
    assert "picture-2.jpg, picture-3.jpg" in verdict[checks.PICTURES].detail
    note = checks.repair_note(list(verdict.values()))
    assert "cut off" in note and "left pictures out" in note
    # Stopped for repeating itself, the page is told that, which is not the same advice as "make it shorter".
    assert "round in circles" in checks.repair_note(list(verdict.values()), ran_away=True)
    # A page with a remote font, no viewport and no heading is reported as it is: a second try is not the cure.
    only_cosmetic = checks.review(broken, plan, ["hero.jpg", "picture-2.jpg", "picture-3.jpg"], truncated=False)
    assert checks.repair_note(only_cosmetic) is None


def test_a_page_with_a_picture_to_spare_is_shown_not_written_again():
    plan = planning.parse("a bike rental page", _BRIEF)
    names = [p.name for p in plan.pictures]
    five = {c.name: c for c in checks.review(_PAGE, plan, names[:5], truncated=False)}
    assert five[checks.PICTURES].passed and five[checks.PICTURES].detail == "5 of 6, not placed: picture-6.jpg"
    four = checks.review(_PAGE, plan, names[:4], truncated=False)
    assert not {c.name: c for c in four}[checks.PICTURES].passed and checks.repair_note(four) is None  # said, and shown
    two = checks.review(_PAGE, plan, names[:2], truncated=False)
    assert "left pictures out" in checks.repair_note(two)  # most of its pictures missing: that is another page
    headless = checks.review(_PAGE, plan, names[1:], truncated=False)
    assert "left pictures out" in checks.repair_note(headless)  # and so is a page without the one it opens on


def test_a_picture_behind_an_element_is_one_place_however_many_rules_say_so():
    plan = planning.parse("a bakery page", _ANSWER)
    names = ["hero.jpg", "picture-2.jpg", "picture-3.jpg"]
    twice_in_css = _PAGE.replace("</style>", ".hero{background:url(hero.jpg) center/cover}</style>")
    assert {c.name: c for c in checks.review(twice_in_css, plan, names, truncated=False)}[checks.ONCE].passed
    and_in_an_img = twice_in_css.replace("</body>", '<img src="hero.jpg" alt="Again"></body>')
    once = {c.name: c for c in checks.review(and_in_an_img, plan, names, truncated=False)}[checks.ONCE]
    assert not once.passed and "hero.jpg x2" in once.detail


def test_the_page_is_checked_for_the_figures_the_request_gave():
    request = "Three bikes: hardtail (EUR 39 / 210), enduro (EUR 65 / 350). 140 km of trails, 1,400 m of climb, first lift at 8:00."
    plan = planning.parse(request, _ANSWER)
    names = [p.name for p in plan.pictures]

    def kept(body: str):
        page = _PAGE.replace("<h1>Bakery</h1>", f"<h1>Bikes</h1><p>{body}</p>")
        return {c.name: c for c in checks.review(page, plan, names, truncated=False)}[checks.FIGURES]

    assert kept("39 a day, 210 a week; 65 and 350; 140 km; 1400 m; from 8:00").detail == "all 7"  # 1,400 written 1400
    most = kept("From &euro;39 a day or 210 a week, 65 for the enduro. 140 km of trails, up to 1,400 m of climb.")
    assert most.passed and most.detail == "5 of 7, not found: 350, 8:00"  # two thirds of them is enough to pass
    few = kept("Bikes from 39 a day.")
    assert not few.passed and few.retry is False  # said, and shown: it is not a reason to write the page again
    none = planning.parse("a page for a florist", _ANSWER)
    assert {c.name: c for c in checks.review(_PAGE, none, names, truncated=False)}[checks.FIGURES].detail == "the request gave none"


# --- a picture shown several times is a page asking for more pictures -------------------


def test_repeated_pictures_get_files_of_their_own_described_by_the_page():
    written = (
        "<style>.hero{background:url('hero.jpg')}</style>"
        '<img src="hero.jpg" alt="Bikes on a ridge">'  # the background was its first use
        '<div><img src="picture-2.jpg" alt="A mountain bike"><h3>Alpine Explorer</h3></div>'
        '<div><img alt="A mountain bike" src="./img/picture-2.jpg"><h3>Valley <em>Cruiser</em></h3></div>'
        '<div><img src="picture-2.jpg"><h3>No alt text</h3></div>'  # nothing to draw from: left as it is
    )
    patched, extras = give_repeats_their_own(written, ["hero.jpg", "picture-2.jpg"])
    assert extras == [("extra-1.jpg", "Bikes on a ridge"), ("extra-2.jpg", "Valley Cruiser: A mountain bike")]
    assert 'src="extra-1.jpg" alt="Bikes on a ridge"' in patched and 'src="extra-2.jpg"' in patched
    assert patched.count("picture-2.jpg") == 2 and "url('hero.jpg')" in patched


def test_a_picture_nobody_drew_is_drawn_from_what_the_page_says_it_shows():
    written = (
        '<img src="hero.jpg" alt="A flower shop">'
        '<div class="card"><img src="https://placehold.co/300x200?text=Sophia+R." alt="Sophia R., owner"><h3>Sophia R.</h3></div>'
        '<div class="card"><img src="team/james.png" alt="A florist"><h3>James T.</h3></div>'
        '<img src="https://cdn.example/banner.jpg">'  # points nowhere and says nothing: a page reads better without it
        '<img src="data:image/svg+xml,%3Csvg/%3E" alt="A drawing of its own">'  # the picture itself: not a file
    )
    patched, extras = give_repeats_their_own(written, ["hero.jpg"])
    assert extras == [("extra-1.jpg", "Sophia R., owner"), ("extra-2.jpg", "James T.: A florist")]
    assert "placehold.co" not in patched and "james.png" not in patched and "cdn.example" not in patched
    assert patched.count("<img") == 4 and 'src="hero.jpg"' in patched and "data:image/svg+xml" in patched
    # Past the limit a picture nobody drew is taken out too: a broken image is worse than none.
    many = "".join(f'<img src="https://placehold.co/{n}" alt="Portrait number {n}">' for n in range(9))
    patched, extras = give_repeats_their_own(many, ["hero.jpg"])
    assert len(extras) == 6 and patched.count("<img") == 6


def test_a_page_that_repeats_nothing_is_left_alone_and_a_gallery_is_capped():
    assert give_repeats_their_own(_PAGE, ["hero.jpg", "picture-2.jpg", "picture-3.jpg"]) == (_PAGE, [])
    gallery = "".join(f'<img src="hero.jpg" alt="view {n}">' for n in range(12))
    assert len(give_repeats_their_own(gallery, ["hero.jpg"])[1]) == 6


# --- a page that goes round in circles is stopped ---------------------------------------

_LOOP = (
    "    .section:last-child .btn {\n      color: white;\n    }\n\n"
    "    .section:last-child .btn:hover {\n      background-color: white;\n      color: var(--very-dark);\n    }\n\n"
)


def _fed(watch, text, piece=40):
    """Feeds `text` to the watch the way a model would, and says where it asked to stop."""
    for start in range(0, len(text), piece):
        watch.control.on_text(text[start:start + piece])
        if watch.control.should_stop():
            return start + piece
    return None


def test_a_loop_is_stopped_within_a_few_thousand_characters_and_a_page_is_not():
    cards = "".join(
        f'<div class="card reveal"><img src="picture-{n}.jpg" alt="Bread number {n}"><div class="body"><h3>Loaf {n}</h3>'
        f"<p>Baked at {n + 2} in the morning with flour number {n * 7} and nothing else, sold by {n + 9}.</p></div></div>\n"
        for n in range(1, 40)
    )
    sound = runaway.Watch()
    assert _fed(sound, cards) is None and sound.ran_away is False  # alike, but each says something
    seen: list[str] = []
    looping = runaway.Watch(GenerationControl(on_text=seen.append))
    stopped_at = _fed(looping, cards[:4000] + _LOOP * 200)
    assert looping.ran_away and 4000 < stopped_at < 4000 + 2 * runaway.WINDOW
    assert "".join(seen) == (cards[:4000] + _LOOP * 200)[:stopped_at]  # whoever follows the page saw all of it


def test_the_watch_still_stops_when_asked_to():
    asked = runaway.Watch(GenerationControl(should_stop=lambda: True))
    asked.control.on_text("<html>")
    assert asked.control.should_stop() is True and asked.ran_away is False


# --- how fast each model works, in its own unit -----------------------------------------


def test_an_image_models_speed_is_counted_in_pictures_and_in_steps(tmp_path):
    def drawn(name, width, height, seconds, denoise):
        return DrawnPicture(name, tmp_path / name, width, height, "x", seconds, steps=4, denoise_seconds=denoise)

    stats = PictureStats.of([drawn("hero.jpg", 1024, 576, 7.5, 5.0), drawn("picture-2.jpg", 768, 512, 4.5, 3.0)], "GPU.0")
    assert (stats.device, stats.pictures, stats.seconds, stats.megapixels, stats.steps) == ("GPU.0", 2, 12.0, 0.98, 8)
    assert stats.images_per_minute == 10.0  # two in twelve seconds
    assert stats.steps_per_second == 1.0  # eight steps in the eight seconds the runtime counted for them
    # Where the runtime's counters cannot be read there is no steps figure, and the rest stands.
    uncounted = PictureStats.of([drawn("hero.jpg", 1024, 576, 6.0, None)], "CPU")
    assert uncounted.steps_per_second is None and uncounted.images_per_minute == 10.0
    assert PictureStats.of([], "GPU.0") is None


def test_a_build_says_how_fast_each_model_worked_and_what_loading_them_took(studio):
    agent, coder, _loaded, work_dir = studio
    coder.pages = [_PAGE, _PAGE]
    told: list[tuple] = []
    both = Assignment("NPU", "GPU.0", "GPU.1", together=True)
    result = agent.build("a bakery page", assignment=both, work_dir=work_dir, on_metric=lambda *said: told.append(said))
    # Each step in its own unit; the pictures' figure moves with every picture and is final at the last.
    assert ("plan", 19.2, "tok/s", True) in told and ("page", 62.5, "tok/s", True) in told
    pictures = [said for said in told if said[0] == "images"]
    assert [said[2] for said in pictures] == ["images/min"] * 4 and [said[3] for said in pictures] == [False, False, True, True]
    assert told[-1] == ("images", result.picture_stats.images_per_minute, "images/min", True)  # all of them, extras included
    stats = result.picture_stats
    assert (stats.device, stats.pictures, stats.steps, stats.megapixels) == ("GPU.0", 3, 12, 1.38)
    assert stats.steps_per_second == 500.0  # twelve steps in the 0.024 s the stand-in says they took
    assert result.planner_stats.tokens_per_second == 19.2 and result.stats.tokens_per_second == 62.5
    # The first build loads three models and says what each took; with two GPUs the next one loads none.
    assert set(result.loads) == {"plan", "images", "page"}
    again = agent.build("a bakery page", assignment=both, work_dir=work_dir)
    assert again.loads == {} and again.picture_stats.pictures == 3


# --- the conductor's order of work ------------------------------------------------------


class _Planner:
    device = "NPU"
    last_stats = GenerationStats("NPU", tokens=240, seconds=12.5, tokens_per_second=19.2)

    asked: list[bool] = []

    def answer(self, system_prompt, user_prompt, max_tokens=512, control=None, sample=True):
        _Planner.asked.append(sample)
        return _ANSWER


class _Artist:
    """Stands in for the image model: writes a small file where a picture is asked for."""

    made: list["_Artist"] = []
    # As the real one: four denoising steps a picture, and what they took by the runtime's own count.
    last_steps, last_denoise_seconds = 4, 0.008

    def __init__(self, device, on_downloading=None):
        self.device, self.drawn = device, []
        _Artist.made.append(self)

    def draw(self, prompt, width, height, path, seed=0):
        self.drawn.append((path.name, prompt, seed))
        path.write_bytes(b"\xff\xd8 not really a picture \xff\xd9")
        return 0.01


class _Coder:
    """Stands in for the coding model: writes what the test tells it to, one page per call."""

    last_stats = GenerationStats("GPU.0", tokens=4000, seconds=64.0, tokens_per_second=62.5, first_token_seconds=1.2)

    def __init__(self, pages):
        self.pages, self.asked, self.room = list(pages), [], []

    def count_tokens(self, text):
        return len(text.split())

    def prompt_budget(self, max_tokens):
        return 100_000

    def answer(self, system_prompt, user_prompt, max_tokens=512, control=None, sample=True, begin=None, temperature=None):
        self.asked.append(user_prompt)
        self.room.append((max_tokens, begin, temperature))
        page = self.pages.pop(0)
        if control is not None and control.on_text is not None:
            # As a model would: piece by piece, until told to stop.
            for start in range(0, len(page), 200):
                control.on_text(page[start:start + 200])
                if control.should_stop is not None and control.should_stop():
                    return page[:start + 200]
        return page


@pytest.fixture
def studio(monkeypatch, tmp_path):
    """A PageAgent whose three models are stand-ins; returns it with the log of what was loaded."""
    loaded: list[tuple[str, str]] = []
    coder = _Coder([_PAGE])
    _Artist.made.clear()
    _Planner.asked.clear()

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
    assert ".figures { display: grid;" in result.html and "the studio's layout rules" in result.html_source
    # The coder was told the pictures by name and how to begin, in that order.
    asked = coder.asked[0]
    assert asked.index("- hero.jpg (1024x576)") < asked.index("Start your reply with <!DOCTYPE html>.")
    assert asked.index("HOW THE STUDIO BUILDS IT") < asked.index("- hero.jpg (1024x576)")
    # ...and its answer was begun for it, with room for a full page, at the coding model's own temperature;
    # the planner took its most likely words.
    assert coder.room == [(8192, "<!DOCTYPE html>\n", 0.7)] and _Planner.asked == [False]
    assert loaded == [("OpenVINO/Qwen3-8B-int4-cw-ov", "NPU"), ("OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov", "GPU.0")]


def test_on_two_gpus_they_work_at_the_same_time_and_both_stay_loaded(studio, monkeypatch):
    agent, coder, _loaded, work_dir = studio
    drawing = threading.Event()
    original = _Artist.draw

    def slow_draw(self, prompt, width, height, path, seed=0):
        drawing.wait(5)  # the pictures are not ready until the page has been written...
        return original(self, prompt, width, height, path, seed)

    def answer(system_prompt, user_prompt, max_tokens=512, control=None, sample=True, begin=None, temperature=None):
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


def test_a_picture_the_stylesheet_puts_behind_the_hero_counts_as_placed(studio):
    agent, coder, _loaded, work_dir = studio
    # The page leaves the hero's picture to the studio's rule, as it is told to.
    coder.pages = [_PAGE.replace(".hero{background-image:url('hero.jpg')}", "").replace("<h1>Bakery</h1>", '<section class="hero"><h1>Bakery</h1></section>')]
    log: list[tuple] = []
    result = agent.build(
        "a bakery page", assignment=Assignment("NPU", "GPU.0", "GPU.0", together=False), work_dir=work_dir,
        on_step=lambda *event: log.append(event),
    )
    assert result.attempts == 1 and all(check.passed for check in result.checks)
    assert result.pictures_used == ["hero.jpg", "picture-2.jpg", "picture-3.jpg"]
    assert ("page", "done", "GPU.0", "3 of 3 picture(s) placed") in log and "hero.jpg" not in result.html


def test_a_page_going_round_in_circles_is_stopped_and_asked_for_again(studio):
    agent, coder, _loaded, work_dir = studio
    looping = _PAGE.replace("</style>", _LOOP * 400 + "</style>")
    coder.pages = [looping, _PAGE]
    log: list[tuple] = []
    result = agent.build(
        "a bakery page", assignment=Assignment("NPU", "GPU.0", "GPU.0", together=False), work_dir=work_dir,
        on_step=lambda *event: log.append(event),
    )
    assert result.attempts == 2 and all(check.passed for check in result.checks) and result.cancelled is False
    assert ("page", "done", "GPU.0", "stopped: it was repeating itself") in log
    assert "round in circles" in coder.asked[1] and "cut off" not in coder.asked[1]


def test_a_planner_that_sends_its_example_back_is_asked_once_more(studio, monkeypatch):
    agent, _coder, _loaded, work_dir = studio
    example = planning.SYSTEM_PROMPT.split("never its content:\n")[1]
    answers = [example, _ANSWER]

    def answer(self, system_prompt, user_prompt, max_tokens=512, control=None, sample=True):
        _Planner.asked.append(sample)
        return answers.pop(0)

    monkeypatch.setattr(_Planner, "answer", answer)
    result = agent.build("a bakery page", assignment=Assignment("NPU", "GPU.0", "GPU.0", together=False), work_dir=work_dir)
    assert _Planner.asked == [False, True]  # asked the same way it would say the same thing again
    assert result.plan.title == "Baked with Care" and len(result.plan.pictures) == 3


def test_a_page_asking_the_network_for_a_picture_gets_one_drawn_instead(studio):
    agent, coder, _loaded, work_dir = studio
    coder.pages = [_PAGE.replace("</body>", '<img src="https://placehold.co/300x200?text=Ines" alt="Ines at the oven">'
                                            '<img src="https://placehold.co/80x80"></body>')]
    result = agent.build("a bakery page", assignment=Assignment("NPU", "GPU.0", "GPU.0", together=False), work_dir=work_dir)
    assert result.attempts == 1 and all(check.passed for check in result.checks)  # self-contained again
    assert result.pictures[-1].prompt == "Ines at the oven, editorial photograph, soft natural light"
    assert "placehold.co" not in result.html and 'src="extra-1.jpg"' in result.html_source


def test_an_img_pointing_nowhere_with_nothing_to_draw_from_is_taken_out(studio):
    agent, coder, _loaded, work_dir = studio
    coder.pages = [_PAGE.replace("</body>", '<img src="https://placehold.co/80x80"></body>')]
    result = agent.build("a bakery page", assignment=Assignment("NPU", "GPU.0", "GPU.0", together=False), work_dir=work_dir)
    assert "placehold.co" not in result.html and len(result.pictures) == 3 and all(check.passed for check in result.checks)


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
    assert result.pictures[-1].prompt == "Brioche: Hands dusting flour, editorial photograph, soft natural light"
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
        def build(self, request, *, assignment, work_dir, on_step, on_plan, on_picture, on_downloading, control, on_metric):
            on_step("plan", "running", "NPU", "")
            plan = planning.parse(request, _ANSWER)
            on_plan(plan)
            on_metric("plan", 19.2, "tok/s", True)
            on_step("plan", "done", "NPU", "3 picture(s) planned")
            on_step("images", "running", "GPU.0", "picture 1 of 3")
            (work_dir / "hero.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            on_picture(SimpleNamespace(name="hero.jpg", width=1024, height=576, prompt="x", seconds=6.5))
            on_metric("images", 9.23, "images/min", False)
            on_picture(SimpleNamespace(name="extra-1.jpg", width=768, height=512, prompt="Brioche", seconds=4.4))
            on_step("page", "running", "GPU.1", "")
            # The coding model writing: its rate is counted from its tokens as they come.
            control.on_text("<!DOCTYPE html>")
            control.on_tokens(20)
            time.sleep(0.05)
            control.on_tokens(20)
            seen["during"] = runner.progress()
            seen["metrics"] = {
                (m["stage"], m["unit"], m["sticky"]) for m in launcher_app.metrics.snapshot() if m["demo_id"] == "page-agent"
            }
            seen["written_so_far"] = client.get("/api/bricks/page-agent/partial?stage=page").json()["active"]
            on_metric("page", 61.4, "tok/s", True)
            on_step("page", "done", "GPU.1", "3 of 3 picture(s) placed")
            seen["picture"] = client.get("/api/page-agent/picture/hero.jpg").status_code
            seen["not_drawn"] = client.get("/api/page-agent/picture/picture-2.jpg").status_code
            seen["not_ours"] = client.get("/api/page-agent/picture/..%2Fsecrets.txt").status_code
            seen["active"] = {(e["stage"], e["device"]) for e in launcher_app.activity.snapshot() if e["demo_id"] == "page-agent"}
            on_step("images", "done", "GPU.0", "3 picture(s) drawn")
            return SimpleNamespace(
                html="<html></html>", html_source=None, plan=plan, assignment=assignment, pictures=[], pictures_used=[],
                checks=[], attempts=1, seconds={"plan": 1.0, "images": 2.0, "page": 3.0, "total": 4.0}, cancelled=False,
                stats=None, planner_stats=None, loads={"images": 27.4},
                picture_stats=PictureStats("GPU.0", pictures=3, seconds=19.5, megapixels=1.38, steps=12,
                                           images_per_minute=9.23, steps_per_second=0.92),
            )

    monkeypatch.setattr(runner, "_session", _Session())
    monkeypatch.setattr(runner, "_engine", "openvino")
    answer = client.post("/api/page-agent/build", json={"request": "a bakery page"})
    assert answer.status_code == 200 and answer.json()["assignment"]["together"] is True

    during = seen["during"]
    assert during["running"] is True and during["plan"]["title"] == "Baked with Care"
    assert [(s["id"], s["state"], s["device"]) for s in during["steps"]] == [
        ("plan", "done", "NPU"), ("images", "running", "GPU.0"), ("page", "running", "GPU.1"), ("check", "pending", "CPU"),
    ]
    # Each step with its own figure in its own unit -- the page's counted from its tokens while it is written.
    rates = {s["id"]: s["rate"] for s in during["steps"]}
    assert rates["plan"] == {"value": 19.2, "unit": "tok/s"} and rates["images"] == {"value": 9.23, "unit": "images/min"}
    assert rates["page"]["unit"] == "tok/s" and rates["page"]["value"] > 0 and rates["check"] is None
    # ...and each under its own stage, which is what puts it under its own chip in the hardware panel:
    # the planner's to stay, the two at work to follow the work.
    assert seen["metrics"] == {("plan", "tok/s", True), ("images", "images/min", False), ("page", "tok/s", False)}
    assert seen["written_so_far"] is True  # the page being written is followed under the page step's own stage
    pictures = {p["name"]: p for p in during["plan"]["pictures"]}
    assert pictures["hero.jpg"]["ready"] and not pictures["picture-2.jpg"]["ready"] and pictures["extra-1.jpg"]["extra"]
    assert (seen["picture"], seen["not_drawn"], seen["not_ours"]) == (200, 404, 404)
    # The hardware panel shows the step under its chip, and the conductor under the CPU.
    assert seen["active"] == {("images", "GPU.0"), ("conductor", "CPU")}  # the page was done by then
    # What the build returns says how fast each model worked and what loading took.
    built = answer.json()
    assert built["picture_stats"]["images_per_minute"] == 9.23 and built["picture_stats"]["steps_per_second"] == 0.92
    assert built["loads"] == {"images": 27.4} and built["plan"]["headline"] == "" and built["plan"]["offers"] == []

    after = client.get("/api/page-agent/progress").json()
    assert after["running"] is False and launcher_app.activity.snapshot() == []
    # A step's last word stays for as long as its model is loaded; a figure that was only on its way goes.
    left = {(m["stage"], m["value"]) for m in launcher_app.metrics.snapshot() if m["demo_id"] == "page-agent"}
    assert left == {("plan", 19.2), ("page", 61.4)}
    assert {s["id"]: s["rate"] and s["rate"]["value"] for s in after["steps"]}["page"] == 61.4
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
