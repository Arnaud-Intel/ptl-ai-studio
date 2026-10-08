"""The plan a small model writes from the user's one-line request: what the
page is, how it should look, and which pictures it needs.

The planner is the smallest model in the chain and the least reliable, so
its job is narrow and its answer is a handful of labelled lines rather than
JSON -- a missing bracket loses a whole JSON reply, a malformed line loses
one line. Whatever cannot be read is replaced by a plain default made from
the request itself, so a bad plan costs quality, never the page.

Two things the first version of the format taught (Qwen3-8B on the NPU,
2026-10-08): a line to fill in such as "IMAGE name shape: ..." comes back
with the words "name shape" still in it, so the model chooses nothing but
the description -- file names and sizes are given out here, by position;
and a request in French got its pictures described in French, which the
image model reads badly, so those lines are asked for in English.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# The pictures of a page, by position: the first is the wide one at the top,
# the others sit in the body. Sizes are multiples of 64, which every image
# model here accepts.
HERO_SIZE = (1024, 576)
BODY_SIZE = (768, 512)
MAX_PICTURES = 4

SYSTEM_PROMPT = (
    "You are the art director of a small web studio. From a short request for "
    "a web page you write the brief the studio works from: a designer will "
    "build the page from it and an image generator will draw its pictures.\n\n"
    "Reply with exactly these six lines and nothing else, in plain text:\n"
    "TITLE: the page's title, a few words\n"
    "STYLE: the mood, two to four colours by name, the feel of the type\n"
    "SECTIONS: the page's sections in order, separated by \" | \"\n"
    "HERO: one sentence describing the main picture, a wide one for the top of the page\n"
    "PICTURE 2: one sentence describing a second picture\n"
    "PICTURE 3: one sentence describing a third picture\n\n"
    "Write TITLE and SECTIONS in the language of the request. Write HERO and "
    "the PICTURE lines in English whatever the language of the request, "
    "because the image generator only reads English. Each of them says what "
    "the picture shows -- the subject, the setting, the light -- and ends with "
    "its visual style, the same style in all three. The three pictures show "
    "three different things. Pictures show things, places and people; never "
    "ask for text, letters, numbers, logos, charts or screenshots in a "
    "picture, because the image generator cannot write."
)

_START = r"^[\s>*#\-\d.)]*\**\s*"
_VALUE = r"\s*\**\s*[:\-]\s*\**\s*<?(.+?)>?\s*\**\s*$"
_TITLE = re.compile(_START + "TITLE" + _VALUE, re.IGNORECASE)
_STYLE = re.compile(_START + "STYLE" + _VALUE, re.IGNORECASE)
_SECTIONS = re.compile(_START + "SECTIONS?" + _VALUE, re.IGNORECASE)
# HERO, PICTURE 2, IMAGE 3, "Picture two" ...: whatever follows the word up to the colon is not needed.
_PICTURE = re.compile(_START + r"(?:HERO|PICTURE|IMAGE)\b[^:]{0,40}:\s*\**\s*<?(.+?)>?\s*\**\s*$", re.IGNORECASE)
_THINKING = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


@dataclass
class PictureSpec:
    name: str  # the file name the page refers to, e.g. "hero.jpg"
    width: int
    height: int
    prompt: str  # what the image model is asked for -- and what the page is told the picture shows


@dataclass
class PagePlan:
    request: str
    title: str = ""
    style: str = ""
    sections: list[str] = field(default_factory=list)
    pictures: list[PictureSpec] = field(default_factory=list)
    # What had to be made up because the planner's answer could not be read.
    notes: list[str] = field(default_factory=list)

    def page_prompt(self) -> str:
        """What the coding model is asked for: the user's own words first,
        then what the plan adds. The pictures are described to it separately,
        by file name."""
        lines = [self.request.strip()]
        if self.title:
            lines.append(f"Page title: {self.title}")
        if self.style:
            lines.append(f"Look and feel: {self.style}")
        if self.sections:
            lines.append("Sections, in this order:")
            lines += [f"- {section}" for section in self.sections]
        return "\n".join(lines)

    def closing(self) -> str:
        """The last line of the request, after the list of pictures -- where
        it is followed. Left to itself the coding model answered a request
        of this shape with an opening code fence and nothing else -- "```",
        then the end, on a fresh model, sampled or not (2026-10-08). Told
        how to begin, it wrote the page four times out of four."""
        return "Give every <img> an alt text that says what it shows. Start your reply with <!DOCTYPE html>."


def _spec(position: int, prompt: str) -> PictureSpec:
    if position == 0:
        return PictureSpec("hero.jpg", *HERO_SIZE, prompt)
    return PictureSpec(f"picture-{position + 1}.jpg", *BODY_SIZE, prompt)


def default_pictures(request: str) -> list[PictureSpec]:
    """One wide picture of what the request is about, for when the planner
    gave none that could be read."""
    subject = " ".join(request.split())[:300]
    return [_spec(0, f"{subject}, editorial photograph, soft natural light")]


def parse(request: str, answer: str) -> PagePlan:
    """The plan in `answer`, as far as it can be read; never raises."""
    plan = PagePlan(request=request)
    prompts: list[str] = []
    for line in _THINKING.sub("", answer or "").splitlines():
        if not line.strip():
            continue
        picture = _PICTURE.match(line)
        if picture:
            prompt = picture.group(1).strip()
            # The same sentence twice is one picture, and a line still holding
            # the instruction ("one sentence describing...") is none.
            if prompt and prompt.lower() not in (p.lower() for p in prompts) and not prompt.lower().startswith("one sentence"):
                prompts.append(prompt)
            continue
        for pattern, attribute in ((_TITLE, "title"), (_STYLE, "style")):
            found = pattern.match(line)
            if found and not getattr(plan, attribute):
                setattr(plan, attribute, found.group(1).strip())
                break
        else:
            sections = _SECTIONS.match(line)
            if sections and not plan.sections:
                parts = re.split(r"\s*\|\s*|\s*;\s*", sections.group(1))
                plan.sections = [part.strip(" -") for part in parts if len(part.strip(" -")) > 1]

    plan.pictures = [_spec(position, prompt) for position, prompt in enumerate(prompts[:MAX_PICTURES])]
    if not plan.pictures:
        plan.pictures = default_pictures(request)
        plan.notes.append("The plan named no picture that could be read, so one was made from the request itself.")
    if not plan.title:
        plan.notes.append("The plan gave no title; the page is written from the request alone.")
    return plan


def fallback(request: str, reason: str) -> PagePlan:
    """The plan when the planner could not be asked at all."""
    return PagePlan(request=request, pictures=default_pictures(request), notes=[reason])
