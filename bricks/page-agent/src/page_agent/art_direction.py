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
- What a reader would see go wrong is not left to the model (2026-10-25,
  five pages looked at): a few rules come after its own, for a phone
  (`_LAST_WORD`), and the motion and a last look at the colours are the
  studio's script (`_SCRIPT`, `with_script`). The model counted "1.5 TB/s"
  up as 1500, and put white cards in a dark band whose rule made every
  text in it white.
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
- Buttons are white text on --accent, so --accent is deep enough to carry it: a light colour the look asks for (sun-yellow, mint, lime) is used for details on --dark, and --accent is a deeper shade of it.
- {headings} Text in a system sans, 1.0625rem, line-height 1.65, in --muted, no line longer than 62ch."""

_SERIF = 'Georgia, "Iowan Old Style", "Palatino Linotype", serif'
_SANS = '"Segoe UI", "Helvetica Neue", Arial, sans-serif'
_HEADINGS = {
    "serif": f"Headings in a display face: {_SERIF}.",
    "sans": f"Headings in {_SANS} at weight 800 with letter-spacing -0.02em.",
    "": (
        f"Headings in a display face: {_SERIF} -- or, when the look is modern or sporty, {_SANS} at weight 800 "
        "with letter-spacing -0.02em."
    ),
}

# Colours for a page that defined none, or not all seven: the page's own :root comes later and wins.
_DEFAULTS = (
    ":root { --bg: #faf9f6; --surface: #ffffff; --ink: #14181f; --muted: #5c6470; --accent: #c2410c; "
    "--accent-dark: #9a3412; --dark: #0b0f17; }"
)

_RULES = """\
.wrap { max-width: 1120px; margin: 0 auto; padding: 0 24px; }
section { padding: clamp(4.5rem, 9vw, 7.5rem) 0; }
section > .split, section > .grid, section > .figures, section.split, section.grid, section.figures { box-sizing: content-box; max-width: 1072px; margin-left: auto; margin-right: auto; padding-left: 24px; padding-right: 24px; }
section:is(.split, .grid, .figures):has(> .wrap:only-child) { display: block; max-width: none; padding-left: 0; padding-right: 0; }
section:is(.split, .grid, .figures) > .wrap:not(:only-child) { margin: 0; padding: 0; max-width: none; }
h2 { font-size: clamp(1.9rem, 4vw, 3rem); line-height: 1.1; margin: 0 0 2.5rem; color: var(--ink); }
.eyebrow { display: block; font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: var(--accent); margin-bottom: .8rem; }
.grid { display: grid; gap: 28px; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); }
.split { display: grid; gap: clamp(2rem, 6vw, 5rem); grid-template-columns: 5fr 6fr; align-items: center; }
.split img { display: block; width: 100%; aspect-ratio: 5 / 6; object-fit: cover; border-radius: 22px; }
.figures { display: grid; gap: 24px; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); text-align: center; }
.figures strong { display: block; font-size: clamp(2.4rem, 5vw, 3.8rem); line-height: 1.1; color: var(--ink); }
.step-num { display: block; font-size: clamp(2.4rem, 4vw, 3.2rem); font-weight: 800; line-height: 1; color: var(--accent); margin-bottom: .8rem; }
.card { background: var(--surface); color: var(--ink); border-radius: 18px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,.06), 0 12px 32px rgba(0,0,0,.08); transition: transform .25s; }
.card:hover { transform: translateY(-4px); }
.card img { display: block; width: 100%; aspect-ratio: 3 / 2; object-fit: cover; }
.card .body { padding: 1.5rem; }
.btn { display: inline-block; padding: .9rem 1.6rem; border-radius: 999px; font-weight: 600; text-decoration: none; background: var(--accent); color: #fff; border: 2px solid var(--accent); transition: background .2s, transform .2s; }
.btn:hover { background: var(--accent-dark); border-color: var(--accent-dark); transform: translateY(-2px); }
.btn.ghost { background: transparent; color: inherit; border-color: currentColor; }
.dark { background: var(--dark); }
.dark, .hero, .closing { color: #fff; }
:is(.dark, .hero, .closing) :where(h1, h2, h3, h4, p, li, span, strong, small, blockquote, figcaption, cite, dt, dd, th, td, label) { color: inherit; }
.quote { text-align: center; }
.quote blockquote { font-size: clamp(1.6rem, 3.4vw, 2.6rem); line-height: 1.25; max-width: 30ch; margin: 0 auto 1.5rem; }
.js .reveal:not(.in) { opacity: 0; transform: translateY(18px); }
.reveal { transition: opacity .7s, transform .7s; }
@media (max-width: 720px) { .split { grid-template-columns: 1fr; } nav .links { display: none; } }"""

# After the page's own rules, where they hold whatever it wrote. A .reveal the script has shown is shown, whatever
# the page's own rules say of .reveal. The rest keeps the page readable on a phone: each was a page that scrolled
# sideways at 380px (2026-10-25, five pages of five) -- the navigation's links shown again by the page's own
# `nav .links { display: flex }`, a grid given three fixed columns, a mock app window of 840px -- and a form field
# of width 100% with its padding outside it.
_LAST_WORD = """\
.reveal.in { opacity: 1; transform: none; }
input, select, textarea { box-sizing: border-box; max-width: 100%; }
@media (max-width: 720px) {
nav .links { display: none; }
.split, .grid, [class*="grid"]:not(.figures) { grid-template-columns: 1fr; }
.figures { grid-template-columns: 1fr 1fr; }
table { display: block; overflow-x: auto; }
img, svg, video, canvas { max-width: 100%; }
body { overflow-x: clip; }
}"""

# The page's motion, and a last look at its colours, written once here rather than by the model at every page: a
# model left to count figures up wrote 1.5 TB/s as 1500 (to make it a whole number), started 1,240 at "0" in the
# HTML and stepped it up by one every 20 ms -- 25 s (2026-10-25). Here a figure is counted up from its own text,
# decimals and thousands separators kept, in a second; a year is left alone. Without a script the page shows the
# figures as written and every .reveal in place.
#
# The colours: text the page put on a background of nearly its own colour -- white on a white card in a dark band,
# white on a sun-yellow button -- is given the page's ink or white, whichever reads better. Text over a picture or
# a gradient is not judged.
_SCRIPT = r"""
(function () {
  var root = document.documentElement;
  root.classList.add('js');
  var reveal = [].slice.call(document.querySelectorAll('.reveal'));
  if ('IntersectionObserver' in window) {
    var shown = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); shown.unobserve(e.target); } });
    }, { threshold: 0.12 });
    reveal.forEach(function (el) { shown.observe(el); });
  } else {
    reveal.forEach(function (el) { el.classList.add('in'); });
  }
  // A figure keeps what it says: the page's own counter, if it wrote one anyway, is undone as it writes
  // (it made 99.95% into 9,995 by keeping the digits only).
  function figure(el) {
    var html = el.innerHTML, found = /(^|>)([^<]*?)(\d[\d,]*(?:\.\d+)?)/.exec(html), shown = html;
    function put(markup) { el.innerHTML = markup; shown = el.innerHTML; }
    new MutationObserver(function () { if (el.innerHTML !== shown) put(shown); })
      .observe(el, { childList: true, characterData: true, subtree: true });
    if (!found || /^(19|20)\d\d$/.test(found[3])) return null;
    var token = found[3], at = found.index + found[1].length + found[2].length;
    var decimals = (token.split('.')[1] || '').length, grouped = token.indexOf(',') >= 0;
    var target = parseFloat(token.replace(/,/g, '')), start = null;
    function show(value) {
      var written = grouped ? value.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) : value.toFixed(decimals);
      put(html.slice(0, at) + written + html.slice(at + token.length));
    }
    function step(now) {
      if (start === null) start = now;
      var t = Math.min(1, (now - start) / 1100);
      if (t < 1) { show(target * (1 - Math.pow(1 - t, 3))); requestAnimationFrame(step); } else { put(html); }
    }
    return function () { requestAnimationFrame(step); };
  }
  var figures = [].slice.call(document.querySelectorAll('.figures strong'));
  var counters = figures.map(figure);
  if ('IntersectionObserver' in window) {
    var counted = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        counted.unobserve(e.target);
        var start = counters[figures.indexOf(e.target)];
        if (start) start();
      });
    }, { threshold: 0.5 });
    figures.forEach(function (el) { counted.observe(el); });
  }

  function rgb(c) { var m = /rgba?\(([^)]+)\)/.exec(c || ''); if (!m) return null; var p = m[1].split(',').map(parseFloat); return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 }; }
  function lum(c) { return [c.r, c.g, c.b].map(function (v) { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }).reduce(function (s, v, i) { return s + v * [0.2126, 0.7152, 0.0722][i]; }, 0); }
  function contrast(a, b) { var x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); }
  function behind(el) {
    for (var e = el; e && e.nodeType === 1; e = e.parentElement) {
      var s = getComputedStyle(e);
      if (s.backgroundImage && s.backgroundImage !== 'none') return null;
      var c = rgb(s.backgroundColor);
      if (c && c.a > 0.5) return c;
    }
    return { r: 255, g: 255, b: 255, a: 1 };  // nothing painted behind it all: the browser's white
  }
  var probe = document.createElement('i');
  probe.style.color = 'var(--ink)';
  document.body.appendChild(probe);
  var ink = rgb(getComputedStyle(probe).color) || { r: 20, g: 24, b: 31, a: 1 };
  probe.remove();
  var white = { r: 255, g: 255, b: 255, a: 1 }, black = { r: 17, g: 17, b: 17, a: 1 };
  [].slice.call(document.body.querySelectorAll('*')).forEach(function (el) {
    var own = [].some.call(el.childNodes, function (n) { return n.nodeType === 3 && n.nodeValue.trim(); });
    if (!own || el.closest('script, style')) return;
    var colour = rgb(getComputedStyle(el).color), ground = behind(el);
    if (!colour || !ground || contrast(colour, ground) >= 2.5) return;
    var best = [ink, white, black].sort(function (a, b) { return contrast(b, ground) - contrast(a, ground); })[0];
    el.style.color = 'rgb(' + best.r + ',' + best.g + ',' + best.b + ')';
  });
})();
"""

_USING_THEM = (
    "Sections alternate between --bg and --surface (a .card on a --surface section takes --bg), with .dark for the "
    "bands: never three sections in a row on one colour. A .dark band holds text, figures or a quote and nothing "
    "with a light background of its own -- no .card, no price card, no form, no tool to try: those go on a --bg or "
    "--surface section. Every section is written the same way, the navigation's and the footer's content in a "
    ".wrap too:\n"
    '<section id="..."><div class="wrap"><span class="eyebrow">...</span><h2>...</h2> ... '
    '<div class="grid">...</div></div></section>\n'
    "-- every section heading an h2, left-aligned under an .eyebrow, and .grid, .split and .figures on a <div> "
    "inside the .wrap, never on the <section>."
)

_NAVIGATION = (
    "Navigation, sticky at the top, translucent with backdrop-filter: blur(12px): the name as a wordmark on the left, "
    "three or four links to sections of the page in a .links, one small .btn on the right."
)
_FIGURES = (
    "Key figures, straight under the hero: a .figures row of three or four <div>, one per figure, each holding the "
    "figure in a <strong> in the display face, written exactly as the request gives it -- its decimals, its "
    "thousands separator, its unit -- and under it its label in a <span> (small, uppercase, --muted). Where the "
    "request gives no figures, three or four that fit what it is about."
)
_REST = (
    "Then the request's remaining parts, each a section of its own under a heading that names what it holds, in the "
    "form that suits it and no two alike: for a process, a .grid of steps, each a <div> with its numeral written out "
    'in a <b class="step-num">01</b> over the step\'s title and one line of text, as many columns as steps; for prices, '
    "hours or specifications, a table with a header row and thin rules between rows; for plans or tickets, a .grid "
    "of price cards with the recommended one outlined in --accent; for things that have a level, a .grid of .card "
    "with a small coloured tag each; tabs or a slider where the request asks for something to try, working, with "
    "their script, every element it writes to present in the HTML."
)
_QUOTE = (
    'If the request asks for one, or the page has fewer than seven sections without it: a <section class="dark '
    'quote"> whose .wrap holds one sentence from a customer in a <blockquote> in the display face, and the person\'s '
    "name and town below it in a small <p>."
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
- Motion is the studio's own script, added to the page after yours: it shows each element with the class "reveal" as it comes into view (give it to the content of the sections below the hero) and counts up the figures in .figures. Write no script for either, and no rule that hides anything; your own <script> is for what the request asks to try.
- The copy is the business talking: specific names, prices, hours and places, short sentences, sentence case. No "Welcome to", no "Lorem ipsum", no exclamation marks, no emoji. A button says what it does ("Book a bike"), never "Learn more".
- A contact form, if the request wants one, has three fields at most; submitting it only shows a line of thanks in the page.
- Nothing scrolls sideways at 380px wide.
- Write each rule once, without comments, and style the parts of the page with the shared classes rather than with a rule per place."""

# How the rules are recognised in a page that already has them.
_MARK = "/* the studio's layout rules */"
_LAST_MARK = "/* the studio's last rules */"
_SCRIPT_MARK = "the studio's script"


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


def face(look: str) -> str:
    """The kind of type the request or the plan asks for: "serif", "sans", or
    "" when it says neither. The heading face was left to the model with
    both offered, and it took the serif for "a tight modern sans" and "a
    geometric sans" alike (2026-10-25); the body text is a sans whatever the
    look, so a serif named is the headings'."""
    if re.search(r"(?<!sans-)(?<!sans )\bserif", look, re.IGNORECASE):
        return "serif"
    return "sans" if re.search(r"\bsans\b", look, re.IGNORECASE) else ""


def _rules(hero: str | None, closing: str | None, kind: str = "") -> str:
    """The layout rules as CSS: the same for every page but for the two that
    name a file, and the headings' face where the look decides it."""
    rules = [_RULES, _hero_rule(hero), _closing_rule(closing)]
    if kind == "serif":
        rules.append(f"h1, h2, h3, .figures strong {{ font-family: {_SERIF}; }}")
    elif kind == "sans":
        rules.append(f"h1, h2, h3, .figures strong {{ font-family: {_SANS}; font-weight: 800; letter-spacing: -0.02em; }}")
    return "\n".join(rules)


def _has(written: str, css_class: str) -> bool:
    return bool(re.search(r"""class\s*=\s*["'][^"']*(?<![\w-])""" + css_class + r"""(?![\w-])""", written, re.IGNORECASE))


def with_stylesheet(written: str, pictures: list, look: str = "") -> str:
    """`written` with the studio's rules at the top of its stylesheet, ahead
    of the page's own rules, which may overrule them, and the few that keep
    it readable on a phone at the end, where they hold. A page with no
    stylesheet at all is given one. The two rules that put a picture behind
    an element name its file only if the page has that element: a file the
    stylesheet names is a picture the page shows."""
    if _MARK in written:
        return written
    hero, _cards, _beside, closing = _places(pictures)
    ours = _rules(hero if _has(written, "hero") else None, closing if _has(written, "closing") else None, face(look))
    block = f"\n{_MARK}\n{_DEFAULTS}\n{ours}\n"
    last = f"\n{_LAST_MARK}\n{_LAST_WORD}\n"
    opening = re.search(r"<style\b[^>]*>", written, re.IGNORECASE)
    if opening:
        closing_tag = re.search(r"</style\s*>", written[opening.end():], re.IGNORECASE)
        end = opening.end() + closing_tag.start() if closing_tag else len(written)
        return written[: opening.end()] + block + written[opening.end(): end] + last + written[end:]
    sheet = f"<style>{block}{last}</style>\n"
    for anchor in (r"</head>", r"<body\b"):
        found = re.search(anchor, written, re.IGNORECASE)
        if found:
            return written[: found.start()] + sheet + written[found.start():]
    return sheet + written


def _figure_written_out(found: re.Match) -> str:
    value = found.group(2)
    if "," not in value and "." not in value and len(value) > 3:
        value = f"{int(value):,}"
    return found.group(1) + value + found.group(3)


def with_script(written: str) -> str:
    """`written` with the studio's script at the end of its body: the
    reveals, the figures counted up, the last look at the colours. A figure
    the page wrote as "0" for a counter of its own to fill in
    (`<span data-to="1240">0</span>`) is written out first: it is what the
    page says without a script, and what the script counts up to."""
    written = re.sub(
        r"""(<[^>]*\bdata-to\s*=\s*["']?([\d.,]*\d)["']?[^>]*>)\s*0\s*(</)""", _figure_written_out, written,
        flags=re.IGNORECASE,
    )
    if _SCRIPT_MARK in written:
        return written
    script = f"<script>/* {_SCRIPT_MARK} */{_SCRIPT}</script>\n"
    found = None
    for found in re.finditer(r"</body\s*>", written, re.IGNORECASE):
        pass
    if found is None:
        return written + "\n" + script
    return written[: found.start()] + script + written[found.start():]


def _offer(names: list[str], things: list[str]) -> str:
    what = f" -- {', '.join(things)} --" if len(things) == 3 else ""
    if len(names) == 3:
        return (
            f"The offer, on a --bg or --surface section under an .eyebrow and an h2 in the page's own words: a .grid "
            f"of three .card{what} with {names[0]}, {names[1]} and {names[2]} on top, in that order, one <img> each. Each card "
            "is about what its picture shows; its .body holds a name in an h3, two lines of text, and a price or a "
            "tag in --accent at weight 700."
        )
    return (
        f"The offer, on a --bg or --surface section under an .eyebrow and an h2 in the page's own words: a .grid of "
        f"three .card{what} without pictures: in each .body a large numeral or a short rule in --accent, a name in an h3, "
        "two lines of text, and a price or a tag in --accent at weight 700."
    )


def _split(name: str, first: bool) -> str:
    order = "the picture first" if first else "the text first, the picture on the right"
    return (
        f'A .split, {order}: <img src="{name}"> on one side; on the other an .eyebrow, an h2, two short paragraphs '
        "and a signature line or three short facts."
    )


def for_plan(pictures: list, things: list[str] | None = None, look: str = "") -> str:
    """The direction for a page that has `pictures` to place (anything with
    a `name` and a `role`), presents `things` and looks as `look` says (the
    request and the plan's STYLE: what it says of the type). Each picture
    is given its place by file name, so what the text asks for is always
    something the page was given."""
    hero, on_cards, beside_text, closing = _places(pictures)
    kind = face(look)
    hero_part = (
        "Hero, a <section class=\"hero\">" + (f" (the rule above puts {hero} behind it)" if hero else "") + ": its "
        ".wrap holds an .eyebrow, the headline in an h1 -- the request's own headline, word for word, where it gives "
        "one -- one sentence, a .btn and a .btn.ghost."
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
        f"{_PALETTE.replace('{headings}', _HEADINGS[kind])}\n\n"
        "Layout -- these rules are in the page already: they are put at the top of your <style> before the page is "
        "shown. Do not write them again; write the rest of the stylesheet (the seven variables on :root, the fonts, "
        f"the navigation, the footer, whatever else the page needs) and build the page from their classes:\n"
        f"{_rules(hero, closing, kind)}\n{_USING_THEM}\n\n"
        "The page, top to bottom. What the request asks for is all there, in the order it gives; the parts below "
        f"are the forms to give it:\n{numbered}\n\n{_CRAFT}"
    )
