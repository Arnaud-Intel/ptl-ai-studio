"""Changing one part of a page that is already written.

Until now a page that was nearly right was written again from nothing: a
minute of work, and everything that was right moved too (a page is not the
same twice). This module is what lets one part be written again while the
rest stays as it was, character for character. It finds the parts of a page
in its source, tells the browser which is which, picks out the few style
rules that apply to a part, and puts the new part where the old one was.

How a part is pointed at. The page is shown in a sandboxed frame, which the
launcher cannot look into. So the frame is given a copy of the page in which
every element of the body carries a number (`marked`), and a small script
that says which number was clicked. The number is written into the opening
tag itself: whatever the browser makes of the page -- and it does not always
make of it what a plain reading of the source would -- a part is its opening
tag, and that is in one place. Where the part ends in the source is worked
out here, by `parts`; `same_inside` checks it against what the browser saw
before anything is replaced.

The copy with the numbers is for the frame only. The page that is kept, and
the one that is downloaded, never carry them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

MARK = "data-part"

_VOID = frozenset("area base br col embed hr img input link meta param source track wbr".split())
# Inside these, a tag ending in "/>" is closed by it; outside, a browser reads
# "<div/>" as an opening tag and so do we.
_FOREIGN = frozenset(("svg", "math"))
# Not for pointing at: nothing to see, or nothing that could be written again
# by itself.
_UNMARKED = frozenset("script style template noscript br wbr source track param".split())
# An opening tag that ends an element left open, and how far up to look for
# it: `<li>` ends the `<li>` before it, but not one of a list further out.
_ENDS = {
    "li": (("li",), ("ul", "ol", "menu")),
    "dt": (("dt", "dd"), ("dl",)),
    "dd": (("dt", "dd"), ("dl",)),
    "tr": (("tr", "td", "th"), ("table", "tbody", "thead", "tfoot")),
    "td": (("td", "th"), ("tr", "table")),
    "th": (("td", "th"), ("tr", "table")),
    "tbody": (("tbody", "thead", "tfoot", "tr", "td", "th"), ("table",)),
    "tfoot": (("tbody", "thead", "tfoot", "tr", "td", "th"), ("table",)),
    "option": (("option",), ("select", "datalist", "optgroup")),
}
# These end a paragraph left open.
_BLOCKS = frozenset(
    "address article aside blockquote details div dl fieldset figcaption figure footer form h1 h2 h3 h4 h5 h6 header "
    "hr main menu nav ol p pre section table ul".split()
)


@dataclass
class Part:
    """One element of a page, as it stands in the page's source."""

    index: int  # among every element of the page, in the order their tags open
    tag: str
    start: int  # where its opening tag begins
    opened: int  # where its opening tag ends
    end: int = -1  # where the element ends: after its closing tag, or where something else ended it
    parent: int | None = None
    in_body: bool = False
    # The number it carries in the marked copy; None for what is not marked.
    number: int | None = None
    classes: tuple[str, ...] = ()
    ident: str = ""
    styled: bool = False  # it carries a style attribute of its own


class _Reader(HTMLParser):
    """Reads a page's source and notes where each element begins and ends."""

    def __init__(self, source: str):
        super().__init__(convert_charrefs=True)
        self.source = source
        self.parts: list[Part] = []
        self._open: list[Part] = []
        self._lines = [0]
        for match in re.finditer("\n", source):
            self._lines.append(match.end())
        self.feed(source)
        self.close()
        for part in self._open:  # never closed: it runs to the end of what was read
            part.end = len(source)

    def _here(self) -> int:
        line, column = self.getpos()
        return self._lines[line - 1] + column

    def _close(self, depth: int, at: int) -> None:
        """Everything open from `depth` up ends at `at`."""
        for part in self._open[depth:]:
            part.end = at
        del self._open[depth:]

    def _end_what_this_ends(self, tag: str, at: int) -> None:
        names = [part.tag for part in self._open]
        if tag in _BLOCKS and names and names[-1] == "p":
            self._close(len(names) - 1, at)
            return
        ended, within = _ENDS.get(tag, ((), ()))
        for depth in range(len(names) - 1, -1, -1):
            if names[depth] in ended:
                self._close(depth, at)
                return
            if names[depth] in within:
                return

    def _begin(self, tag: str, attrs, closed: bool) -> None:
        start = self._here()
        self._end_what_this_ends(tag, start)
        attributes = {name: value or "" for name, value in attrs}
        part = Part(
            index=len(self.parts), tag=tag, start=start, opened=start + len(self.get_starttag_text()),
            parent=self._open[-1].index if self._open else None,
            in_body=any(above.tag == "body" for above in self._open),
            classes=tuple(attributes.get("class", "").split()), ident=attributes.get("id", ""),
            styled=bool(attributes.get("style", "").strip()),
        )
        self.parts.append(part)
        if closed or tag in _VOID:
            part.end = part.opened
        else:
            self._open.append(part)

    def handle_starttag(self, tag, attrs):
        self._begin(tag, attrs, closed=False)

    def handle_startendtag(self, tag, attrs):
        foreign = tag in _FOREIGN or any(above.tag in _FOREIGN for above in self._open)
        self._begin(tag, attrs, closed=foreign)

    def handle_endtag(self, tag):
        start = self._here()
        after = self.source.find(">", start) + 1 or len(self.source)
        for depth in range(len(self._open) - 1, -1, -1):
            if self._open[depth].tag == tag:
                self._close(depth + 1, start)  # what it leaves open inside ends where it does
                self._open.pop().end = after
                return
        # A closing tag for nothing that is open: a browser passes over it too.


def parts(source: str) -> list[Part]:
    """Every element of the page, in the order their tags open. Those a
    person can point at -- inside the body, and something to see -- are
    numbered from 1."""
    found = _Reader(source).parts
    has_body = any(part.tag == "body" for part in found)
    number = 0
    for part in found:
        inside = part.in_body if has_body else part.tag not in ("html", "head", "title", "meta", "link")
        if inside and part.tag not in _UNMARKED:
            number += 1
            part.number = number
    return found


def marked(source: str) -> str:
    """The page with its number written into every part that can be pointed
    at: the copy the frame is shown, and nothing else is."""
    out, last = [], 0
    for part in parts(source):
        if part.number is None:
            continue
        name = re.match(r"<[^\s/>]+", source[part.start:part.opened])
        at = part.start + name.end()
        out.append(source[last:at])
        out.append(f' {MARK}="{part.number}"')
        last = at
    out.append(source[last:])
    return "".join(out)


def find(source: str, number: int) -> Part:
    for part in parts(source):
        if part.number == number:
            return part
    raise ValueError(f"The page has no part {number}: it may have changed since it was pointed at.")


def inside(source: str, number: int) -> list[int]:
    """The numbers of the parts that lie within part `number` in the source."""
    found = parts(source)
    part = next((p for p in found if p.number == number), None)
    if part is None:
        raise ValueError(f"The page has no part {number}: it may have changed since it was pointed at.")
    return [p.number for p in found if p.number is not None and part.start < p.start < part.end]


def same_inside(source: str, number: int, seen: list[int] | None) -> bool:
    """Whether the browser found within the part what the source has within
    it. They differ when the page's HTML is not what it looks like -- a tag
    never closed, a table the browser rearranged -- and replacing the part
    would then replace something else."""
    return seen is None or sorted(set(seen)) == inside(source, number)


def where(source: str, part: Part) -> str:
    """Where the part sits, in the words of a stylesheet: `main > section.rooms > div.card`."""
    by_index = {p.index: p for p in parts(source)}
    names = []
    current: Part | None = by_index.get(part.index)
    while current is not None and current.tag not in ("body", "html"):
        label = current.tag + (f"#{current.ident}" if current.ident else "") + "".join(f".{name}" for name in current.classes[:2])
        names.append(label)
        current = by_index.get(current.parent) if current.parent is not None else None
    return " > ".join(reversed(names))


def replaced(source: str, part: Part, new: str) -> str:
    """The page with `new` where the part was, and not another character changed."""
    return source[:part.start] + new + source[part.end:]


# ----------------------------------------------------------------- the style rules of a part

_STYLE_BLOCK = re.compile(r"<style\b[^>]*>(.*?)</style\s*>", re.IGNORECASE | re.DOTALL)
_SCRIPT_BLOCK = re.compile(r"<script\b[^>]*>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL)
_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_NAME = r"-?[_a-zA-Z][\w-]*"
# What always applies, whatever the part: the page's variables and its base type.
_EVERYWHERE = re.compile(r"^(:root|html|body|\*|::selection)\b|^\*$")


def _rules(css: str) -> list[tuple[str, str]]:
    """The stylesheet's rules, outermost only, as (what comes before the
    brace, what is between the braces)."""
    rules, depth, start, opened = [], 0, 0, 0
    quote = ""
    for at, char in enumerate(css):
        if quote:
            if char == quote and css[at - 1] != "\\":
                quote = ""
        elif char in "\"'":
            quote = char
        elif char == "{":
            if depth == 0:
                opened = at
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                rules.append((css[start:opened].strip(), css[opened + 1:at].strip()))
                start = at + 1
        elif char == ";" and depth == 0:
            start = at + 1  # an @import or @charset: a statement, not a rule
    return rules


def _applies(selectors: str, classes: set[str], idents: set[str], tags: set[str]) -> bool:
    for selector in selectors.split(","):
        selector = selector.strip()
        if not selector:
            continue
        if _EVERYWHERE.match(selector):
            return True
        named = re.findall(rf"([.#])({_NAME})", re.sub(r"\[[^\]]*\]|\([^)]*\)", "", selector))
        if named:
            if any((name in classes) if kind == "." else (name in idents) for kind, name in named):
                return True
            continue
        last = re.split(r"[\s>+~]+", selector)[-1]
        tag = re.match(r"[a-zA-Z][\w-]*", last)
        if tag and tag.group(0).lower() in tags:
            return True
    return False


def _kept(css: str, classes: set[str], idents: set[str], tags: set[str]) -> list[str]:
    kept = []
    for before, body in _rules(css):
        if before.startswith("@"):
            kind = before.split()[0].lower()
            if kind in ("@media", "@supports", "@container", "@layer"):
                inner = _kept(body, classes, idents, tags)
                if inner:
                    kept.append(before + " {\n  " + "\n  ".join(inner) + "\n}")
            continue  # @keyframes and @font-face are looked at afterwards, by name
        if _applies(before, classes, idents, tags):
            kept.append(f"{before} {{ {' '.join(body.split())} }}")
    return kept


def styles_for(source: str, fragment: str) -> str:
    """The rules of the page's stylesheet that apply to `fragment`: those
    that name one of its classes or ids, those for its kinds of element, and
    the page's variables and base rules. A page's whole stylesheet is as long
    as the page; the part's own rules are a few lines."""
    css = _COMMENT.sub("", "\n".join(_STYLE_BLOCK.findall(source)))
    inner = _Reader(fragment).parts
    classes = {name for part in inner for name in part.classes}
    idents = {part.ident for part in inner if part.ident}
    tags = {part.tag for part in inner}
    kept = _kept(css, classes, idents, tags)
    used = "\n".join(kept)
    for before, body in _rules(css):  # the animations those rules name
        if before.lower().startswith("@keyframes"):
            name = before.split()[-1]
            if re.search(rf"\b{re.escape(name)}\b", used):
                kept.append(f"{before} {{ {' '.join(body.split())} }}")
    return "\n".join(kept)


def with_rules(source: str, css: str) -> str:
    """The page with `css` added at the end of its stylesheet, where it has
    the last word."""
    css = css.strip()
    if not css:
        return source
    blocks = list(_STYLE_BLOCK.finditer(source))
    if blocks:
        at = blocks[-1].end(1)
        return source[:at].rstrip() + "\n\n/* added with a change to one part */\n" + css + "\n" + source[at:]
    head = re.search(r"</head\s*>", source, re.IGNORECASE)
    at = head.start() if head else 0
    return source[:at] + "<style>\n" + css + "\n</style>\n" + source[at:]


# ----------------------------------------------------------------- what the model is asked, and what it answers

# What follows was tried on two pages HTML Creator had written, with eleven
# changes (2026-10-11, Qwen3-Coder-30B on the integrated GPU). Given the part,
# its rules and the comment and nothing else, the model wrote a usable part
# every time, in 2 to 13 s -- and three of the eleven made the page worse:
# a price row filled with invented prices, in dollars (the rooms' prices are
# in another section, which it had not been shown); answers hidden behind a
# class nothing ever set (asked to make them open on a click, it had no way
# to hand back a script); a picture called by a name no file has. Hence the
# words of the rest of the page, the list of pictures, and the script.
SYSTEM_PROMPT = (
    "You change one part of an HTML page that already exists. You are given that part, the style rules of the page "
    "that apply to it, what the rest of the page says, and what to change. Write the part again with the change made.\n"
    "Your reply has this form and nothing else:\n"
    "1. An HTML comment of one line saying what you change.\n"
    "2. ONE HTML element: the part itself, with the same tag name, from its opening tag to its closing tag.\n"
    "3. Only if the change cannot be made without it, ONE <style> element with new rules.\n"
    "4. Only if the change cannot work without it, ONE <script> element. It runs after the page's own script.\n"
    "Rules:\n"
    "- Make the change that is asked and no other. Keep the wording, the class names, the ids, the pictures and the "
    "links that the change does not concern.\n"
    "- Take names, prices and dates from what the page says. Invent only what the page does not give.\n"
    "- Prefer the classes the page already has. To restyle the part, write its rule again with the new values: the "
    "later rule wins. Never repeat a rule you leave as it is.\n"
    "- To put things side by side, wrap them in a new <div> and make that div a flex row or a grid. Do not turn a "
    "container that also holds a heading into a row.\n"
    "- Do not use position: absolute or fixed, unless the part's own opening tag also gets "
    "style=\"position: relative\".\n"
    "- A rule that waits for a class needs the attribute or the script that sets that class.\n"
    "- The script only makes this part work. It never creates, moves or removes anything outside the part, and it "
    "uses the page's own functions when the page's script already does the same thing for the part.\n"
    "- A link or a button to another place on the page uses a selector from the list of the page's blocks.\n"
    "- No external file, font or image, and no picture that is not in the list of pictures.\n"
    "Raw HTML only: no markdown fence, nothing after the last element."
)
# How the answer is begun for the model: the line of plan first. Begun with
# the part's own opening, it wrote the part at once; asked to say first what
# it would change, it made fewer changes that do nothing. (Begun this way it
# once wrote a <div> for a <section>: `html_creator.session` asks again then,
# with the answer begun on the tag itself.)
BEGIN = "<!-- I change: "
# The words of the rest of the page that the model is shown, at most.
_ELSEWHERE_LIMIT = 3500
_SCRIPT_LINES = 40
_TAG = re.compile(r"<[^>]+>")


def _words(markup: str) -> str:
    import html as entities

    markup = re.sub(r"<(script|style)\b.*?</\1\s*>", " ", markup, flags=re.IGNORECASE | re.DOTALL)
    return " ".join(entities.unescape(_TAG.sub(" ", markup)).split())


def said_elsewhere(source: str, part: Part) -> str:
    """What the page says outside the part, as plain words: where the model
    finds the names and the prices it must not contradict."""
    body = re.search(r"<body\b[^>]*>", source, re.IGNORECASE)
    start = body.end() if body else 0
    words = _words(source[start:part.start] + " [THE PART] " + source[part.end:])
    if len(words) > _ELSEWHERE_LIMIT:
        # Around the part first: what stands beside it matters most.
        at = words.find("[THE PART]")
        begin = max(0, min(at - _ELSEWHERE_LIMIT // 2, len(words) - _ELSEWHERE_LIMIT))
        words = ("... " if begin else "") + words[begin:begin + _ELSEWHERE_LIMIT] + (" ..." if begin + _ELSEWHERE_LIMIT < len(words) else "")
    return words


def _label(part: Part) -> str:
    return part.tag + (f"#{part.ident}" if part.ident else "") + "".join(f".{name}" for name in part.classes[:2])


def blocks(source: str) -> list[str]:
    """The page's large blocks in order, each as a selector and its first
    words: what a link from one part to another can aim at."""
    found = parts(source)
    by_index = {part.index: part for part in found}
    lines = []
    for part in found:
        above = by_index.get(part.parent) if part.parent is not None else None
        if part.number is None or above is None:
            continue
        outer = above.tag == "body" or (above.tag == "main" and above.parent is not None and by_index[above.parent].tag == "body")
        if outer and part.tag != "main":
            words = _words(source[part.start:part.end])
            lines.append(_label(part) + (f" -- {words[:60]}" if words else ""))
    return lines[:20]


def script_for(source: str, fragment: str) -> str:
    """The lines of the page's own script that name something of the part --
    one of its ids, its classes -- each with the line before and the line
    after. A page that counts a number up from its script has to be asked
    for one more number in the same way."""
    inner = _Reader(fragment).parts
    names = {part.ident for part in inner if part.ident} | {name for part in inner for name in part.classes}
    names = {name for name in names if len(name) >= 4}
    if not names:
        return ""
    lines = "\n".join(_SCRIPT_BLOCK.findall(source)).splitlines()
    wanted = set()
    for at, line in enumerate(lines):
        if any(re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", line) for name in names):
            wanted.update((at - 1, at, at + 1))
    kept, last = [], None
    for at in sorted(index for index in wanted if 0 <= index < len(lines)):
        if not lines[at].strip():
            continue
        if last is not None and at > last + 1:
            kept.append("...")
        kept.append(lines[at].rstrip())
        last = at
        if len(kept) >= _SCRIPT_LINES:
            kept.append("...")
            break
    return "\n".join(kept)


def request(source: str, part: Part, comment: str, pictures: str = "") -> str:
    """What the model is told. `pictures`, if the page came with any: the
    list of the files that exist, as the page's own request gave it."""
    fragment = source[part.start:part.end]
    title = re.search(r"<title[^>]*>(.*?)</title\s*>", source, re.IGNORECASE | re.DOTALL)
    rules = styles_for(source, fragment)
    script = script_for(source, fragment)
    outline = blocks(source)
    return "\n\n".join(
        piece for piece in (
            f"THE PAGE: {' '.join(title.group(1).split())}" if title else "",
            "THE PAGE'S BLOCKS, IN ORDER:\n" + "\n".join(f"- {line}" for line in outline) if outline else "",
            "WHAT THE REST OF THE PAGE SAYS:\n" + said_elsewhere(source, part),
            pictures.strip(),
            f"WHERE THE PART IS: {where(source, part)}",
            "STYLE RULES OF THE PAGE THAT APPLY TO THE PART:\n" + rules if rules else "",
            "THE PAGE'S OWN SCRIPT, WHERE IT NAMES THE PART (an element you add needs the same treatment, in your <script>):\n"
            + script if script else "",
            f"THE PART (a <{part.tag}> element):\n" + fragment,
            "WHAT TO CHANGE: " + " ".join(comment.split()),
            f"Reply with the one-line comment, then the <{part.tag}> element"
            + ("." if part.tag in _VOID else f", from <{part.tag} to </{part.tag}>."),
        ) if piece
    )


@dataclass
class Answer:
    element: str = ""  # the part as written again
    css: str = ""  # the style rules it came with
    script: str = ""  # the script it came with
    plan: str = ""  # what the model said it would change
    problem: str = ""  # why it cannot be used, if it cannot
    notes: list[str] = field(default_factory=list)


_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n|\n```\s*$")
_FIRST_COMMENT = re.compile(r"^\s*<!--(.*?)-->", re.DOTALL)


def read_answer(text: str, tag: str) -> Answer:
    """The element, the style rules and the script in what the model wrote,
    or why what it wrote cannot take the part's place."""
    text = _FENCE.sub("", text.strip()).strip()
    answer = Answer()
    plan = _FIRST_COMMENT.match(text)
    if plan:
        answer.plan = " ".join(plan.group(1).split()).removeprefix("I change:").strip()
    reader = _Reader(text)
    top = [part for part in reader.parts if part.parent is None]
    elements = [part for part in top if part.tag not in ("style", "script")]
    beside = lambda name: "".join(text[part.start:part.end] for part in top if part.tag == name)  # noqa: E731
    answer.css = "\n".join(block.strip() for block in _STYLE_BLOCK.findall(beside("style")) if block.strip())
    answer.script = "\n".join(block.strip() for block in _SCRIPT_BLOCK.findall(beside("script")) if block.strip())
    if not elements:
        answer.problem = "the model wrote no element"
        return answer
    first = elements[0]
    if first.tag != tag:
        answer.problem = f"the model wrote a <{first.tag}> where the part is a <{tag}>"
        return answer
    closing = re.search(rf"</{re.escape(tag)}\s*>\s*$", text[first.start:first.end], re.IGNORECASE)
    if tag not in _VOID and not closing:
        answer.problem = "the model's answer was cut off before the part ended"
        return answer
    if len(elements) > 1:
        answer.notes.append(f"the model wrote {len(elements)} elements; the first one was kept")
    answer.element = text[first.start:first.end]
    return answer


def _same_rule(one: str, other: str) -> bool:
    return " ".join(one.split()) == " ".join(other.split())


def new_rules(source: str, css: str) -> str:
    """`css` without the rules the page already has, word for word: asked not
    to repeat them, the model repeats some all the same."""
    have = [f"{before}{{{body}}}" for before, body in _rules(_COMMENT.sub("", "\n".join(_STYLE_BLOCK.findall(source))))]
    kept = []
    for before, body in _rules(_COMMENT.sub("", css)):
        rule = f"{before}{{{body}}}"
        if not any(_same_rule(rule, existing) for existing in have):
            kept.append(f"{before} {{\n  {body}\n}}")
    return "\n".join(kept)


def with_script(source: str, script: str) -> str:
    """The page with `script` added after its own, at the end of the body."""
    script = script.strip()
    if not script:
        return source
    added = "<script>\n/* added with a change to one part */\n" + script + "\n</script>\n"
    ends = [match.start() for match in re.finditer(r"</body\s*>", source, re.IGNORECASE)]
    at = ends[-1] if ends else len(source)
    return source[:at] + added + source[at:]


def changed(source: str, part: Part, answer: Answer) -> str:
    """The page with the answer in the part's place: the part, and the rules
    and the script it came with. Nothing else moves."""
    page = replaced(source, part, answer.element)
    return with_script(with_rules(page, new_rules(source, answer.css)), answer.script)


# ----------------------------------------------------------------- what an answer is checked for

# The same fourteen changes, a third time (2026-10-11): every one came back as
# a part that could be used, and four of them were not finished. A "button"
# that was a link with a class no rule dresses, to an id nothing has; a badge
# placed with position: absolute and nothing for it to be placed against, so
# that it sat in the corner of the page -- three times of three, whatever the
# instructions said; a fourth number added to a band whose script counts the
# other three up, and left at 0. What can be seen in the text of an answer is
# looked for here. A fault is told to the model, once, and what is left is
# told to the person.
_IMAGE_FILE = re.compile(r"""(?:src=|url\()\s*["']?([^"')\s>]+\.(?:png|jpe?g|webp|gif|svg))""", re.IGNORECASE)
_ID = re.compile(r"""\bid\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


def _classes(markup: str) -> set[str]:
    return {name for part in _Reader(markup).parts for name in part.classes}


def _selectors(css: str) -> list[str]:
    found = []
    for before, body in _rules(_COMMENT.sub("", css)):
        if before.startswith("@"):
            if before.split()[0].lower() in ("@media", "@supports", "@container", "@layer"):
                found.extend(_selectors(body))
            continue
        found.append(before)
    return found


def faults(source: str, part: Part, answer: Answer, pictures: list[str] | None = None) -> list[str]:
    """What is wrong with `answer` that its own text shows: each as a
    sentence the model can be told, and a person after it."""
    found = []
    page = replaced(source, part, answer.element)
    page_css = "\n".join(_STYLE_BLOCK.findall(page)) + "\n" + answer.css
    ruled = set(re.findall(rf"\.({_NAME})", " ".join(_selectors(page_css))))
    before = _classes(source[part.start:part.end])
    # A class on an element that is styled where it stands needs no rule.
    after = {name for piece in _Reader(answer.element).parts if not piece.styled for name in piece.classes}
    for name in sorted(after - before - ruled):
        found.append(
            f'the class "{name}" has no style rule: write its rule in the <style> element after the part, '
            "or style the element where it stands with a style attribute"
        )
    ids = set(_ID.findall(page))
    every_class = _classes(page)
    for target in sorted(set(re.findall(r"""href\s*=\s*["']#([^"']+)["']""", answer.element)) - ids):
        aim = f".{target}" if target in every_class else "a selector from the list of the page's blocks"
        found.append(
            f"the link to #{target} goes nowhere: no element of the page has that id. Scroll to the block instead, "
            f"with href=\"#\" and onclick=\"document.querySelector('{aim if aim.startswith('.') else 'SELECTOR'}')"
            f".scrollIntoView({{behavior:'smooth'}}); return false;\"" + ("" if aim.startswith(".") else f", SELECTOR being {aim}")
        )
    script = answer.script + " " + " ".join(re.findall(r"""\bon\w+\s*=\s*"([^"]*)""", answer.element))
    for target in sorted(set(re.findall(r"""getElementById\(\s*["']([^"']+)["']\s*\)""", script)) - ids):
        found.append(f"the script looks for an element with the id {target}, and the page has none")
    everything = page + " " + answer.script
    for selector in _selectors(answer.css):
        for name in re.findall(rf"\.({_NAME})", selector):
            given = re.search(rf"""["'\s.]{re.escape(name)}["'\s]""", everything.replace(answer.css, ""))
            if not given:
                found.append(f"the rule for .{name} never applies: nothing on the page is given that class, and no script gives it")
    if pictures is not None:
        for name in sorted({name.rsplit("/", 1)[-1] for name in _IMAGE_FILE.findall(answer.element + answer.css)} - set(pictures)):
            found.append(f"the picture {name} does not exist: only the pictures in the list do")
    return found


def faults_note(found: list[str]) -> str:
    """What the model is told when it is asked once more."""
    return (
        "YOUR FIRST ANSWER HAD THESE FAULTS. Write the whole answer again without them:\n"
        + "\n".join(f"- {fault}" for fault in found)
    )


_ABSOLUTE = re.compile(r"position\s*:\s*absolute", re.IGNORECASE)
_POSITIONED = re.compile(r"position\s*:\s*(relative|absolute|fixed|sticky)", re.IGNORECASE)


def anchored(source: str, part: Part, answer: Answer) -> Answer:
    """`answer`, with the part made the thing its absolutely placed pieces
    are placed against, when the model placed some and gave them nothing.

    Put right here and not asked for again: asked three times not to do it,
    the model did it three times, and what is missing is always the same
    four words on the part's own opening tag."""
    opening = re.match(r"<[^>]*>", answer.element)
    inside = answer.element[opening.end():] if opening else ""
    if not opening or not (_ABSOLUTE.search(inside) or _ABSOLUTE.search(answer.css)):
        return answer
    tag = opening.group(0)
    root = _Reader(answer.element).parts[0]
    known = "\n".join(_STYLE_BLOCK.findall(source)) + "\n" + answer.css
    for before, body in _rules(_COMMENT.sub("", known)):
        mine = [name for name in re.findall(rf"\.({_NAME})", before.split(",")[-1].split()[-1] if before.strip() else "")]
        if mine and set(mine) <= set(root.classes) and _POSITIONED.search(body):
            return answer  # the page already places it
    if _POSITIONED.search(tag):
        return answer
    style = re.search(r"""\bstyle\s*=\s*(["'])""", tag)
    if style:
        new_tag = tag[:style.end()] + "position: relative; " + tag[style.end():]
    else:
        new_tag = tag[:-1].rstrip("/").rstrip() + ' style="position: relative">'
    answer.element = new_tag + answer.element[opening.end():]
    answer.notes.append("the part was given position: relative, for what the change places inside it")
    return answer
