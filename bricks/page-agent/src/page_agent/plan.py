"""The plan a small model writes from the user's request: what the page is
called, how it should look, what it presents, and which pictures it needs.

The planner is the smallest model in the chain and the least reliable, so
its job is narrow and its answer is a handful of labelled lines rather than
JSON -- a missing bracket loses a whole JSON reply, a malformed line loses
one line. Whatever cannot be read is replaced by a plain default made from
the request itself, so a bad plan costs quality, never the page.

What the format has taught (Qwen3-8B on the NPU, 2026-10-08):

- A line to fill in such as "IMAGE name shape: ..." comes back with the
  words "name shape" still in it, so the model chooses nothing but the
  description: file names and sizes are given out here, by the line's label.
- Asked to "describe a picture" for a page whose request went into colours,
  type and prices, it described the page: "a trail map with EUR 39 / 210 in
  chalk white", "in a tight modern sans", a clock showing the opening hour.
  Asked for a photograph for a photographer who has not read the request,
  with one example of the form, it described scenes. The example is about
  something no sample is about, and is told apart from an answer below.
- Three photographs "of the three main things" were of anything nearby --
  one bike and two trails for three bikes -- until each thing was named on
  the line just above its photograph (THING 1, PHOTO 1, THING 2...). Since
  then they have matched in the nine plans tried, loosely in one (three
  coffees shown as beans, a cup and a jug).
- It still asks for writing now and then (a ticket, a banner with the
  band's name, a clock that "reads 8:07"), and an image model cannot write:
  those parts are taken out of the description here, and a picture that is
  of writing is replaced or not drawn.
- When it draws among the likely words it sometimes repeats itself (one
  sentence for two pictures, once in nine plans) or sends the example back
  as its answer (once in nine). The conductor has it take the most likely
  word instead; what still slips through is dropped here, and each picture
  keeps the place its label gave it instead of sliding into the gap.
- A request in French had its pictures described in French whatever the
  prompt said, which the image model reads less well. The one French request
  tried since the changes above came back with English photographs; one is
  not a rule.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import art_direction
from .art_direction import CLOSING, FEATURE, HERO, OFFER, STORY

# The pictures of a page, by the label of their line, and what each is for.
# A real page is carried by its pictures -- one behind the headline, one on
# each card of what is on offer, one beside the story, one behind the last
# call to action -- and three were not enough: the page filled its cards with
# whatever it had (a map on a bike's card), or with one picture three times.
# Sizes are multiples of 64, which every image model here accepts.
HERO_SIZE = (1024, 576)
BODY_SIZE = (768, 512)
TALL_SIZE = (640, 768)
MAX_PICTURES = 6
_SLOT_ROLES = (HERO, OFFER, OFFER, OFFER, STORY, CLOSING)
_OFFER_SLOTS = (1, 2, 3)
_SIZES = {HERO: HERO_SIZE, OFFER: BODY_SIZE, STORY: TALL_SIZE, CLOSING: HERO_SIZE, FEATURE: BODY_SIZE}

# The example the planner is shown. Its subject is one no sample is about, and
# its picture lines are kept apart so that one sent back as an answer -- it
# happens: an architecture studio was once planned six photographs of pottery
# -- is recognised and left out.
_EXAMPLE_PICTURES = (
    "A small harbour at first light with dinghies pulled up on the slipway, calm water, low sun",
    "Two children in life jackets steering a small dinghy, spray in the air, bright morning",
    "A catamaran flying one hull across open water, seen from a chase boat, hard midday light",
    "A wooden keelboat heeling under full sail near a rocky headland, late afternoon",
    "An instructor coiling a rope on the pontoon beside a rack of masts, overcast light",
    "A close-up of wet hands tying a knot around a steel cleat, shallow focus",
)
_EXAMPLE_THINGS = ("Dinghy course for children", "Catamaran course", "Keelboat cruising")
_EXAMPLE = (
    "NAME: Les Voiles de Kervoal",
    "HEADLINE: Learn the wind in a week",
    "STYLE: fresh and bright, navy, sail white, buoy red, a clean rounded sans",
    "SECTIONS: Courses | The boats | The team | Come and sail",
    "HERO PHOTO: " + _EXAMPLE_PICTURES[0],
    *(
        line
        for number, thing in enumerate(_EXAMPLE_THINGS, 1)
        for line in (f"THING {number}: {thing}", f"PHOTO {number}: {_EXAMPLE_PICTURES[number]}")
    ),
    "STORY PHOTO: " + _EXAMPLE_PICTURES[4],
    "DETAIL PHOTO: " + _EXAMPLE_PICTURES[5],
)

SYSTEM_PROMPT = (
    "You are the art director of a small web studio. From a request for a web page -- one line, or a full brief -- "
    "you write what the studio works from: a designer will build the page, and a photographer will take its six "
    "photographs.\n\n"
    "Reply with exactly these thirteen lines and nothing else, in plain text:\n"
    "NAME: the name of the business, product or event the page is for\n"
    "HEADLINE: the big line at the top of the page, six words at most\n"
    "STYLE: the mood, three or four colours by name, the feel of the type\n"
    "SECTIONS: the page's sections in order, separated by \" | \"\n"
    "HERO PHOTO: the wide photograph at the top of the page\n"
    "THING 1: the first of the three main things the page presents, in a few words\n"
    "PHOTO 1: a photograph of THING 1\n"
    "THING 2: the second of them\n"
    "PHOTO 2: a photograph of THING 2\n"
    "THING 3: the third of them\n"
    "PHOTO 3: a photograph of THING 3\n"
    "STORY PHOTO: a photograph of the people or the place behind it all, at work\n"
    "DETAIL PHOTO: a close-up photograph of a detail that sets the mood\n\n"
    "Keep what the request decides: its name, its headline, its colours, the things it lists. Invent what it leaves "
    "open, a name included. The three things are what the page is there to show -- three products, services, "
    "places or people, in the order the request gives them. Write NAME, HEADLINE, STYLE, SECTIONS and the THING "
    "lines in the language of the request.\n\n"
    "The six PHOTO lines are for the photographer, who has not read the request and only reads English. Each is "
    "one English sentence that says what is in front of the camera: the subject, where it is, the light. Six "
    "different subjects, never the same sentence twice. A photograph has no writing in it: never mention a name, a "
    "price, a date, a number, a sign, a banner, a map, a label, a ticket, a card, a screen, a logo, a quote, or the "
    "colours and the type of the page.\n\n"
    "An example of the form, for a sailing school -- copy the form, never its content:\n"
    + "\n".join(_EXAMPLE)
)

_START = r"^[\s>*#\-\d.)]*\**\s*"
_VALUE = r"\s*\**\s*[:\-]\s*\**\s*<?(.+?)>?\s*\**\s*$"
_TITLE = re.compile(_START + "(?:NAME|BRAND|TITLE)" + _VALUE, re.IGNORECASE)
_HEADLINE = re.compile(_START + "HEADLINE" + _VALUE, re.IGNORECASE)
_STYLE = re.compile(_START + "STYLE" + _VALUE, re.IGNORECASE)
_SECTIONS = re.compile(_START + "SECTIONS?" + _VALUE, re.IGNORECASE)
_OFFERS = re.compile(_START + "OFFERS?" + _VALUE, re.IGNORECASE)
_THING = re.compile(_START + r"THING\b([^:]{0,20}):\s*\**\s*<?(.+?)>?\s*\**\s*$", re.IGNORECASE)
# HERO PHOTO, PHOTO 2, STORY PHOTO, and what a model writes instead (PICTURE 2, IMAGE 3, "Picture two"): the first
# word and any number before the colon say which picture it is.
_PICTURE = re.compile(
    _START + r"(HERO|STORY|TEAM|DETAIL|PHOTO|PICTURE|IMAGE)\b([^:]{0,40}):\s*\**\s*<?(.+?)>?\s*\**\s*$", re.IGNORECASE
)
_THINKING = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)

# What an image model draws as gibberish: writing, and the things that carry it.
_WRITING = re.compile(
    r"\b(?:text|typography|lettering|letters|words?|large type|quotes?|captions?|headline|signs?|signage|banners?|posters?|"
    r"labels?|labell?ed|logos?|maps?|tickets?|cards?|menus?|screens?|tablets?|laptops?|charts?|prices?|calendars?|"
    r"stickers?|chalkboards?|blackboards?|clocks?)\b",
    re.IGNORECASE,
)
# Words in quotation marks, with what introduces them ("a banner reading 'Les Heures Bleues'"); an apostrophe is not one.
_QUOTED = re.compile(
    r"""\s*(?:\b(?:reading|saying|that says|named|called))?\s*(?:"[^"]{1,80}"|“[^”]{1,80}”|(?<!\w)['‘](?=\w)[^'’]{1,80}['’](?!\w))""",
    re.IGNORECASE,
)
_CLAUSE = re.compile(r"(,\s*|\s+with\s+|\s+beside\s+|\s+next to\s+)", re.IGNORECASE)


@dataclass
class PictureSpec:
    name: str  # the file name the page refers to, e.g. "hero.jpg"
    width: int
    height: int
    prompt: str  # what the image model is asked for -- and what the page is told the picture shows
    role: str = FEATURE  # where the page is asked to put it (see art_direction.py)


@dataclass
class PagePlan:
    request: str
    title: str = ""  # the name on the page: the business, the product, the event
    headline: str = ""
    style: str = ""
    sections: list[str] = field(default_factory=list)
    offers: list[str] = field(default_factory=list)  # the three things the page presents; pictures 2 to 4 show them
    pictures: list[PictureSpec] = field(default_factory=list)
    # What had to be made up, or left out, because of what the planner's answer was.
    notes: list[str] = field(default_factory=list)
    copied: int = 0  # picture lines that were the prompt's example sent back: a plan worth asking for again

    def page_prompt(self) -> str:
        """What the coding model is asked for: the user's own words first,
        then what the plan adds, then how the studio builds a page. The
        pictures are described to it separately, by file name."""
        lines = [self.request.strip()]
        added = []
        if self.title:
            added.append(f"Name: {self.title}")
        if self.headline:
            added.append(f"Headline: {self.headline}")
        if self.style:
            added.append(f"Look and feel: {self.style}")
        if self.offers:
            added.append("On offer: " + " | ".join(self.offers))
        if self.sections:
            added.append("Sections, in this order:")
            added += [f"- {section}" for section in self.sections]
        if added:
            lines += ["", "THE STUDIO'S NOTES -- where the request above says otherwise, the request wins:"] + added
        lines += ["", art_direction.for_plan(self.pictures, self.offers)]
        return "\n".join(lines)

    def closing(self) -> str:
        """The last line of the request, after the list of pictures -- where
        it is followed. Left to itself the coding model answered a request
        of this shape with an opening code fence and nothing else -- "```",
        then the end, on a fresh model, sampled or not (2026-10-08). Told
        how to begin, it wrote the page four times out of four."""
        return (
            "Everything the request lists is on the page, with the names, the prices and the figures it gives. "
            "Give every <img> an alt text that says what it shows. Start your reply with <!DOCTYPE html>."
        )


def _spec(slot: int, prompt: str, role: str = "") -> PictureSpec:
    role = role or (HERO if slot == 0 else FEATURE)
    name = "hero.jpg" if slot == 0 else f"picture-{slot + 1}.jpg"
    return PictureSpec(name, *_SIZES[role], prompt, role)


def default_pictures(request: str) -> list[PictureSpec]:
    """One wide picture of what the request is about, for when the planner
    gave none that could be read."""
    subject = " ".join(request.split())[:300]
    return [_spec(0, f"{subject}, editorial photograph, soft natural light")]


def _is_the_form(prompt: str) -> bool:
    """A line of the format sent back as its own answer ("a photograph of
    the second thing in OFFER"): the model filled nothing in."""
    said = prompt.lower().rstrip(".")
    return said.startswith((
        "one sentence", "a photograph of the", "a photograph of thing", "a close-up photograph of a detail",
        "the wide photograph", "the first of the three", "the second of them", "the third of them",
    ))


def _is_the_example(prompt: str) -> bool:
    return prompt.lower().rstrip(".") in (line.lower() for line in _EXAMPLE_PICTURES + _EXAMPLE_THINGS)


def without_writing(prompt: str) -> str:
    """`prompt` without the parts that ask for writing -- a quoted name, "with
    a banner", "beside a map" -- or "" when its subject is itself a piece of
    writing (a ticket, a sign), which leaves nothing to photograph."""
    parts = _CLAUSE.split(_QUOTED.sub("", prompt))
    if _WRITING.search(parts[0]):
        return ""
    kept = parts[0]
    for joint, clause in zip(parts[1::2], parts[2::2]):
        # A figure in a detail is a figure somebody would have to write ("a clock on the wall reads 8:07").
        if not _WRITING.search(clause) and not re.search(r"\d", clause):
            kept += joint + clause
    return kept.strip(" ,")


def _slot(word: str, label: str, last: int) -> int:
    """Which picture a line is. HERO PHOTO is the first, PHOTO 1 to 3 show
    the three things, STORY PHOTO and DETAIL PHOTO are the last two; a model
    that numbers its pictures instead means the fifth by "PICTURE 5"; and a
    label that says nothing ("Picture two", a number the model garbled) is
    the one after the line before."""
    word, number = word.upper(), re.search(r"\d+", label)
    named = {"HERO": 0, "STORY": 4, "TEAM": 4, "DETAIL": 5}
    if word in named:
        return named[word]
    if number and word == "PHOTO" and 1 <= int(number.group()) <= 3:
        return int(number.group())
    if number and word != "PHOTO" and 1 <= int(number.group()) <= MAX_PICTURES:
        return int(number.group()) - 1
    return last + 1


def parse(request: str, answer: str) -> PagePlan:
    """The plan in `answer`, as far as it can be read; never raises."""
    plan = PagePlan(request=request)
    prompts: dict[int, str] = {}
    things: list[str] = []
    unwritable, copied, slot = 0, 0, -1
    for line in _THINKING.sub("", answer or "").splitlines():
        if not line.strip():
            continue
        thing = _THING.match(line)
        if thing:
            named = thing.group(2).strip()
            if _is_the_example(named):
                copied += 1
            elif not _is_the_form(named) and named.lower() not in (t.lower() for t in things):
                things.append(named)
            continue
        picture = _PICTURE.match(line)
        if picture:
            slot = _slot(picture.group(1), picture.group(2), slot)
            prompt = picture.group(3).strip()
            # The same sentence twice is one picture, and a line still
            # holding the instruction is none. Either leaves its place empty.
            if not prompt or prompt.lower() in (p.lower() for p in prompts.values()) or _is_the_form(prompt):
                continue
            if _is_the_example(prompt):
                copied += 1
                continue
            if slot in prompts or slot >= MAX_PICTURES:
                continue
            prompt = without_writing(prompt)
            if prompt:
                prompts[slot] = prompt
            else:
                unwritable += 1
            continue
        for pattern, attribute in ((_TITLE, "title"), (_HEADLINE, "headline"), (_STYLE, "style")):
            found = pattern.match(line)
            if found and not getattr(plan, attribute):
                setattr(plan, attribute, found.group(1).strip())
                break
        else:
            for pattern, attribute in ((_SECTIONS, "sections"), (_OFFERS, "offers")):
                found = pattern.match(line)
                if found and not getattr(plan, attribute):
                    parts = re.split(r"\s*\|\s*|\s*;\s*", found.group(1))
                    setattr(plan, attribute, [part.strip(" -") for part in parts if len(part.strip(" -")) > 1])

    plan.offers = (things or plan.offers)[:3]
    # A card of the offer whose picture could not be used is given one made
    # from the offer's own words, so that the three cards stay alike.
    if len(plan.offers) == 3 and 0 < sum(slot in prompts for slot in _OFFER_SLOTS) < 3:
        for slot, thing in zip(_OFFER_SLOTS, plan.offers):
            fallback = without_writing(thing)
            if slot not in prompts and fallback:
                prompts[slot] = f"{fallback}, editorial photograph, soft natural light"
    # The three cards get a picture each or none: short of three, the ones
    # there are go beside text instead.
    cards = all(slot in prompts for slot in _OFFER_SLOTS)
    plan.pictures = [
        _spec(slot, prompt, _SLOT_ROLES[slot] if cards or slot not in _OFFER_SLOTS else FEATURE)
        for slot, prompt in sorted(prompts.items())
    ]
    if plan.pictures and plan.pictures[0].role != HERO:
        # No picture was given for the top of the page: the first one there is goes there.
        first = plan.pictures[0]
        plan.pictures[0] = PictureSpec(first.name, *HERO_SIZE, first.prompt, HERO)

    plan.copied = copied
    if copied:
        # What came with the example goes with it: a page for an architect is not called after a sailing school.
        theirs = {line.split(": ", 1)[1].lower() for line in _EXAMPLE}
        for attribute in ("title", "headline", "style"):
            if getattr(plan, attribute).lower() in theirs:
                setattr(plan, attribute, "")
        if " | ".join(plan.sections).lower() in theirs:
            plan.sections = []
        plan.notes.append(f"The planner sent back {copied} line(s) of its own example, which were left out.")
    if unwritable:
        plan.notes.append(f"{unwritable} planned picture(s) were of writing, which an image model cannot draw, and were left out.")
    if not plan.pictures:
        plan.pictures = default_pictures(request)
        plan.notes.append("The plan named no picture that could be read, so one was made from the request itself.")
    if not plan.title:
        plan.notes.append("The plan gave no name; the page is written from the request alone.")
    return plan


def fallback(request: str, reason: str) -> PagePlan:
    """The plan when the planner could not be asked at all."""
    return PagePlan(request=request, pictures=default_pictures(request), notes=[reason])
