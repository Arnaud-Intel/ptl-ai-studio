"""How the studio builds a page: what the conductor adds to every request
before the coding model reads it, and the layout rules it puts into every
page afterwards. Plain text, no model.

Left to itself the coding model answers a request for a page with the same
page every time: a dimmed picture under a centred headline, then section
after section of a centred heading over three equal cards, one background
colour throughout, one system font. Nothing in it is wrong and nothing in
it looks like somebody designed it (2026-10-08, three samples). What a
designed page has that it lacks can be said, and is said here: a hero that
fills the first screen with a headline large enough to be the first thing
seen, a different layout for every section, backgrounds that alternate,
figures in very large type, a picture in the place it was drawn for.

How to say it took five attempts to learn.

- Advice is ignored and a specification is followed. "Give every section its
  own layout, never the same block twice" changed nothing; a numbered list
  of the sections with what each contains was built as written.
- Where a layout was described in words -- "a band of figures separated by
  thin rules", "numbered steps side by side", "all of it white" -- it came
  out wrong on every other page: figures stacked in a column, four steps as
  three and one, a dark headline on a dark photograph. So the CSS those
  rows and bands hang on, some twenty rules, was written out for the model
  to copy.
- Copying is not what a model drawing its words freely does best: two pages
  in thirteen left out the rule that sizes the section headings, one the
  closing band's. The rules are now the brick's own stylesheet, put at the
  top of every page's <style> once it is written (`with_stylesheet`), and
  the model is shown them and told they are there. It mostly writes them
  out again all the same, which costs a few seconds and harms nothing; what
  matters is that a rule it forgets is no longer missing. The rest is the
  model's: the palette, the type, the navigation, every component, the
  copy, the scripts -- and it may overrule any of these rules, since its
  own come after.
- A picture is placed by the rule that names its file. Told in a sentence
  where each belonged, the page put the closing picture beside the story
  and the first picture at both ends.
- Whatever hides content until a script reveals it is written so that the
  page is whole when the script does not run.
"""
from __future__ import annotations

import re

# What a picture is for. The plan gives each of its pictures one of these, by
# the place of its line (see plan.py); the page is told where each belongs,
# by file name.
HERO, OFFER, STORY, CLOSING, FEATURE = "hero", "offer", "story", "closing", "feature"

_PALETTE = """\
HOW THE STUDIO BUILDS IT -- a real site that somebody designed, not a template.

Colours and type
- Seven CSS variables on :root, taken from the look asked for, and nothing but them in the rest of the CSS: --bg (the page), --surface (a second, slightly different tone for alternate sections and cards), --ink, --muted, --accent, --accent-dark, --dark (one very dark colour, for bands and the footer).
- Headings in a display face: Georgia, "Iowan Old Style", "Palatino Linotype", serif -- or, when the look is modern or sporty, "Segoe UI", "Helvetica Neue", Arial, sans-serif at weight 800 with letter-spacing -0.02em. Text in a system sans, 1.0625rem, line-height 1.65, in --muted, no line longer than 62ch."""

# Colours for a page that defined none, or not all seven: the page's own :root comes later and wins.
_DEFAULTS = (
    ":root { --bg: #faf9f6; --surface: #ffffff; --ink: #14181f; --muted: #5c6470; --accent: #c2410c; "
    "--accent-dark: #9a3412; --dark: #0b0f17; }"
)

_RULES = """\
.wrap { max-width: 1120px; margin: 0 auto; padding: 0 24px; }
section { padding: clamp(4.5rem, 9vw, 7.5rem) 0; }
section > .split, section > .grid, section > .figures, section.split, section.grid, section.figures { box-sizing: content-box; max-width: 1072px; margin-left: auto; margin-right: auto; padding-left: 24px; padding-right: 24px; }
h2 { font-size: clamp(1.9rem, 4vw, 3rem); line-height: 1.1; margin: 0 0 2.5rem; color: var(--ink); }
.eyebrow { display: block; font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); margin-bottom: .8rem; }
.grid { display: grid; gap: 28px; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); }
.split { display: grid; gap: clamp(2rem, 6vw, 5rem); grid-template-columns: 5fr 6fr; align-items: center; }
.split img { display: block; width: 100%; aspect-ratio: 5 / 6; object-fit: cover; border-radius: 22px; }
.figures { display: grid; gap: 24px; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); text-align: center; }
.figures strong { display: block; font-size: clamp(2.4rem, 5vw, 3.8rem); line-height: 1.1; color: var(--ink); }
.card { background: var(--surface); border-radius: 18px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,.06), 0 12px 32px rgba(0,0,0,.08); transition: transform .25s; }
.card:hover { transform: translateY(-4px); }
.card img { display: block; width: 100%; aspect-ratio: 3 / 2; object-fit: cover; }
.card .body { padding: 1.5rem; }
.btn { display: inline-block; padding: .9rem 1.6rem; border-radius: 999px; font-weight: 600; text-decoration: none; background: var(--accent); color: #fff; border: 2px solid var(--accent); transition: background .2s, transform .2s; }
.btn:hover { background: var(--accent-dark); border-color: var(--accent-dark); transform: translateY(-2px); }
.btn.ghost { background: transparent; color: var(--ink); border-color: currentColor; }
.dark { background: var(--dark); }
.dark *, .hero *, .closing * { color: #fff; }
@media (max-width: 720px) { .split { grid-template-columns: 1fr; } nav .links { display: none; } }"""

_USING_THEM = (
    "Sections alternate between --bg and --surface (a .card on a --surface section takes --bg), with .dark for the "
    "bands, which hold no .card: never three sections in a row on one colour. Every section's content sits in a "
    ".wrap, the navigation's and the footer's too, and every section heading is an h2, left-aligned under an .eyebrow."
)

_NAVIGATION = (
    "Navigation, sticky at the top, translucent with backdrop-filter: blur(12px): the name as a wordmark on the left, "
    "three or four links to sections of the page in a .links, one small .btn on the right."
)
_FIGURES = (
    "Key figures, straight under the hero: a .figures row of three or four <div>, one per figure, each holding the "
    "figure in a <strong> in the display face and under it its label in a <span> (small, uppercase, --muted)."
)
_REST = (
    "Then the request's remaining parts, each a section of its own under a heading that names what it holds, in the "
    "form that suits it and no two alike: for a process, a .grid of steps with a large numeral in --accent over each "
    "step's title and one line of text, as many columns as steps; for prices or hours, a table with a header row and "
    "thin rules between rows; for plans or tickets, a .grid of price cards with the recommended one outlined in "
    "--accent; for things that have a level, a .grid of .card with a small coloured tag each; tabs or a slider where "
    "the request asks for something to try, working, with their script."
)
_QUOTE = (
    "If the request asks for one, or the page has fewer than seven sections without it: one sentence from a customer "
    "as a pull quote in a .dark band, centred, in the display face at clamp(1.6rem, 3.4vw, 2.6rem) and at most 30ch "
    "wide, with the person's name and town below in small type."
)
_FAQ = (
    "If the request asks for them: questions and answers, three or four, as <details> elements one under the other "
    "(not a grid) in a column at most 760px wide, the question in <summary> at weight 600 with 1.1rem of padding "
    "above and below, a thin rule between two questions."
)
_FOOTER = (
    "Footer, .dark: the name with one line about it, three short columns of links or facts in a .grid, then a line "
    "of small print."
)

_CRAFT = """\
Craft
- A button is an <a class="btn"> to a section of the page; a second one beside it is .btn.ghost.
- Motion, lightly, and written so that the page is whole when no script runs: the script's first line is document.documentElement.classList.add('js'); the CSS hides only ".js .reveal:not(.in)" (opacity 0, translateY(18px), transition .7s); an IntersectionObserver adds the class "in" to each .reveal as it comes into view, once. A figure that is a whole number is written <span class="count" data-to="140">140</span> with its unit outside the span, and the script counts each .count up from 0 to its data-to once, when it comes into view.
- The copy is the business talking: specific names, prices, hours and places, short sentences, sentence case. No "Welcome to", no "Lorem ipsum", no exclamation marks, no emoji. A button says what it does ("Book a bike"), never "Learn more".
- A contact form, if the request wants one, has three fields at most; submitting it only shows a line of thanks in the page.
- Nothing scrolls sideways at 380px wide.
- Write each rule once, without comments, and style the parts of the page with the shared classes rather than with a rule per place."""

# How the rules are recognised in a page that already has them.
_MARK = "/* the studio's layout rules */"


def _hero_rule(name: str | None) -> str:
    behind = (
        "linear-gradient(to top, rgba(6,10,18,.88), rgba(6,10,18,.4) 55%, rgba(6,10,18,.12)), "
        f'url("{name}") center / cover'
        if name
        else "var(--dark)"
    )
    return (
        f".hero {{ min-height: 88vh; display: flex; align-items: flex-end; padding: 0 0 4rem; background: {behind}; }}\n"
        ".hero h1 { font-size: clamp(2.8rem, 7vw, 5.4rem); line-height: 1.02; max-width: 14ch; margin: 0 0 1rem; }"
    )


def _closing_rule(name: str | None) -> str:
    if name:
        return (
            ".closing { text-align: center; background: linear-gradient(rgba(6,10,18,.72), rgba(6,10,18,.72)), "
            f'url("{name}") center / cover; }}'
        )
    return ".closing { text-align: center; background: var(--dark); }"


def _places(pictures: list) -> tuple[str | None, list[str], list[str], str | None]:
    """Where each picture goes: behind the headline, on the three cards,
    beside text, behind the last call to action."""
    by_role: dict[str, list[str]] = {}
    for picture in pictures:
        by_role.setdefault(getattr(picture, "role", FEATURE), []).append(picture.name)
    on_cards = by_role.get(OFFER, [])
    beside_text = by_role.get(STORY, []) + by_role.get(FEATURE, [])
    if len(on_cards) != 3:  # a picture each or none
        on_cards, beside_text = [], on_cards + beside_text
    return (by_role.get(HERO) or [None])[0], on_cards, beside_text, (by_role.get(CLOSING) or [None])[0]


def _rules(hero: str | None, closing: str | None) -> str:
    """The layout rules as CSS: the same for every page but for the two that
    name a file."""
    return "\n".join([_RULES, _hero_rule(hero), _closing_rule(closing)])


def _has(written: str, css_class: str) -> bool:
    return bool(re.search(r"""class\s*=\s*["'][^"']*(?<![\w-])""" + css_class + r"""(?![\w-])""", written, re.IGNORECASE))


def with_stylesheet(written: str, pictures: list) -> str:
    """`written` with the studio's rules at the top of its stylesheet, ahead
    of the page's own rules, which may overrule them. A page with no
    stylesheet at all is given one. The two rules that put a picture behind
    an element name its file only if the page has that element: a file the
    stylesheet names is a picture the page shows."""
    if _MARK in written:
        return written
    hero, _cards, _beside, closing = _places(pictures)
    ours = _rules(hero if _has(written, "hero") else None, closing if _has(written, "closing") else None)
    block = f"\n{_MARK}\n{_DEFAULTS}\n{ours}\n"
    opening = re.search(r"<style\b[^>]*>", written, re.IGNORECASE)
    if opening:
        return written[: opening.end()] + block + written[opening.end():]
    sheet = f"<style>{block}</style>\n"
    for anchor in (r"</head>", r"<body\b"):
        found = re.search(anchor, written, re.IGNORECASE)
        if found:
            return written[: found.start()] + sheet + written[found.start():]
    return sheet + written


def _offer(names: list[str], things: list[str]) -> str:
    what = f" -- {', '.join(things)} --" if len(things) == 3 else ""
    if len(names) == 3:
        return (
            f"The offer: a .grid of three .card{what} with {names[0]}, {names[1]} and {names[2]} on top, in that "
            "order, one <img> each. Each card is about what its picture shows; its .body holds a name in an h3, two "
            "lines of text, and a price or a tag in --accent at weight 700."
        )
    return (
        f"The offer: a .grid of three .card{what} without pictures: in each .body a large numeral or a short rule in "
        "--accent, a name in an h3, two lines of text, and a price or a tag in --accent at weight 700."
    )


def _split(name: str, first: bool) -> str:
    order = "the picture first" if first else "the text first, the picture on the right"
    return (
        f'A .split, {order}: <img src="{name}"> on one side; on the other an .eyebrow, an h2, two short paragraphs '
        "and a signature line or three short facts."
    )


def for_plan(pictures: list, things: list[str] | None = None) -> str:
    """The direction for a page that has `pictures` to place (anything with
    a `name` and a `role`) and presents `things`. Each picture is given its
    place by file name, so what the text asks for is always something the
    page was given."""
    hero, on_cards, beside_text, closing = _places(pictures)
    hero_part = (
        "Hero, a <section class=\"hero\">" + (f" (the rule above puts {hero} behind it)" if hero else "") + ": its "
        ".wrap holds an .eyebrow, the headline in an h1, one sentence, a .btn and a .btn.ghost."
    )
    closing_part = (
        "Closing band, a <section class=\"closing\">"
        + (f" (the rule above puts {closing} behind it, and no other picture goes there)" if closing else "")
        + ": one line in an h2, one .btn."
    )
    parts = [_NAVIGATION, hero_part, _FIGURES, _offer(on_cards, list(things or []))]
    parts += [_split(name, index % 2 == 0) for index, name in enumerate(beside_text)]
    parts += [_REST, _QUOTE, _FAQ, closing_part, _FOOTER]
    numbered = "\n".join(f"{number}. {part}" for number, part in enumerate(parts, 1))
    return (
        f"{_PALETTE}\n\n"
        "Layout -- these rules are in the page already: they are put at the top of your <style> before the page is "
        "shown. Do not write them again; write the rest of the stylesheet (the seven variables on :root, the fonts, "
        f"the navigation, the footer, whatever else the page needs) and build the page from their classes:\n"
        f"{_rules(hero, closing)}\n{_USING_THEM}\n\n"
        "The page, top to bottom. What the request asks for is all there, in the order it gives; the parts below "
        f"are the forms to give it:\n{numbered}\n\n{_CRAFT}"
    )
