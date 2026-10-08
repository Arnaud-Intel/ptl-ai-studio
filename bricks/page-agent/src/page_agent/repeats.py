"""A page that shows one picture several times, or a picture nobody drew,
is asking for more pictures.

The plan has six, each with its place; a page with three trail cards as
well, or a team of three, has more places than that, and the coding model
fills them with what it has -- a planned picture again -- or with what it
knows from other pages: a placeholder service's address
(`https://placehold.co/300x200?text=Sophia+R.`), which is a broken image on
a machine that is offline and a request to the network on one that is not.
Asking it to write the page again without them takes a minute and loses the
cards their pictures. Drawing the missing ones takes seconds each and is
what the page was reaching for: every such `<img>` gets a file name of its
own, and its alt text -- the page's own words for what belongs there --
becomes the description the image model draws from.
"""
from __future__ import annotations

import re

_IMG = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_SRC = re.compile(r"""(\bsrc\s*=\s*)(["'])(.*?)\2""", re.IGNORECASE)
_ALT = re.compile(r"""\balt\s*=\s*(["'])(.*?)\1""", re.IGNORECASE | re.DOTALL)
_URL = r"""url\(\s*["']?[^"')]*{name}["']?\s*\)"""
_HEADING = re.compile(r"<h[2-6]\b[^>]*>(.*?)</h[2-6]>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
# How far past an <img> its card's heading is looked for.
_NEARBY = 500

# More than this and the page is a gallery, which is another request.
MAX_EXTRA = 6


def give_repeats_their_own(written: str, names: list[str], limit: int = MAX_EXTRA) -> tuple[str, list[tuple[str, str]]]:
    """`written` with every `<img>` that repeats one of `names`, or points
    at a file that is none of them, pointed at a new file, and the new files
    as (file name, what the page says it shows).

    A picture's first use stays as it is -- a CSS background counts as one.
    Three cards given one picture tend to be given one alt text too, so the
    heading that follows the `<img>` (the card's title) goes in front of
    it: that is what tells the three apart. An `<img>` without alt text
    gives nothing to draw from: a repeat is left as it is, and one pointing
    at a file that does not exist is taken out, since a page reads better
    without a picture than with a broken one. So is one past the limit."""
    used = {name for name in names if re.search(_URL.format(name=re.escape(name)), written, re.IGNORECASE)}
    extras: list[tuple[str, str]] = []

    def visit(tag: re.Match) -> str:
        text = tag.group(0)
        source = _SRC.search(text)
        if not source or source.group(3).strip().lower().startswith("data:"):
            return text  # nothing it points at, or the picture itself
        name = source.group(3).split("?")[0].split("#")[0].replace("\\", "/").rsplit("/", 1)[-1]
        ours = name in names
        if ours and name not in used:
            used.add(name)
            return text
        alt = _ALT.search(text)
        description = " ".join(alt.group(2).split()) if alt else ""
        if not description or len(extras) >= limit:
            return text if ours else ""
        # The card's own heading: the first one after the picture, and before
        # the next picture -- past that it is another card's.
        following = _IMG.search(written, tag.end())
        until = min(tag.end() + _NEARBY, following.start() if following else len(written))
        heading = _HEADING.search(written, tag.end(), until)
        title = " ".join(_TAG.sub("", heading.group(1)).split()) if heading else ""
        if title and title.lower() not in description.lower():
            description = f"{title}: {description}"
        new_name = f"extra-{len(extras) + 1}.jpg"
        extras.append((new_name, description))
        return text[: source.start(3)] + new_name + text[source.end(3):]

    return _IMG.sub(visit, written), extras
