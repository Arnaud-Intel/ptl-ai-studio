"""Pictures a generated page may use.

The model writes text: it cannot see a picture, and it cannot write one.
What it can do is place pictures it has been told about. This module reads
a folder of image files, describes them to the model by file name (with a
caption, when the folder has a `captions.txt`), and afterwards swaps every
file name the page references for the picture itself, as a `data:` URI --
so the page is still one self-contained file.
"""
from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
}
CAPTIONS_FILE = "captions.txt"
# More than a page places, and each one costs prompt tokens.
MAX_PICTURES = 12
# A photo straight off a phone is several megabytes, and a page may use one
# three times (a card, a lightbox, a background): larger files are scaled
# down before they are embedded.
MAX_BYTES = 1_500_000
MAX_SIDE = 1600
# A name the model can copy exactly and that can sit inside an attribute.
_SAFE_NAME = re.compile(r"^[\w][\w .()-]*$")


@dataclass
class Picture:
    name: str
    path: Path
    mime: str
    width: int | None = None
    height: int | None = None
    caption: str = ""


def _captions(folder: Path) -> dict[str, str]:
    """`captions.txt`, if the folder has one: `file name: what it shows`
    per line. Lines starting with # are comments."""
    try:
        lines = (folder / CAPTIONS_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    captions = {}
    for line in lines:
        name, colon, caption = line.partition(":")
        if colon and not line.lstrip().startswith("#"):
            captions[name.strip().lower()] = caption.strip()
    return captions


def _pillow():
    try:
        from PIL import Image
    except ImportError:
        return None
    return Image


def _svg_size(path: Path) -> tuple[int | None, int | None]:
    head = path.read_text(encoding="utf-8", errors="replace")[:2000]
    box = re.search(r'viewBox="\s*[-\d.]+[\s,]+[-\d.]+[\s,]+([\d.]+)[\s,]+([\d.]+)', head)
    if box:
        return round(float(box.group(1))), round(float(box.group(2)))
    width, height = re.search(r'\bwidth="([\d.]+)', head), re.search(r'\bheight="([\d.]+)', head)
    if width and height:
        return round(float(width.group(1))), round(float(height.group(1)))
    return None, None


def _size(path: Path) -> tuple[int | None, int | None]:
    if path.suffix.lower() == ".svg":
        return _svg_size(path)
    image = _pillow()
    if image is None:
        return None, None
    try:
        with image.open(path) as opened:
            return opened.size
    except OSError:
        return None, None


def load(folder: str) -> tuple[list[Picture], list[str]]:
    """The pictures in `folder` (not its subfolders) in name order, and a
    note for each thing left out and why."""
    resolved = Path(folder).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Not a folder: {resolved}")
    captions = _captions(resolved)
    files = sorted(
        (p for p in resolved.iterdir() if p.is_file() and p.suffix.lower() in MIME_TYPES), key=lambda p: p.name.lower()
    )
    if not files:
        raise ValueError(f"No pictures (png, jpg, webp, gif or svg) in '{resolved}'.")

    pictures: list[Picture] = []
    notes: list[str] = []
    for path in files:
        if not _SAFE_NAME.match(path.name):
            notes.append(f"{path.name} was left out: rename it using letters, digits, dashes and dots only.")
            continue
        if path.stat().st_size > MAX_BYTES and (path.suffix.lower() in (".svg", ".gif") or _pillow() is None):
            notes.append(
                f"{path.name} was left out: it is larger than {MAX_BYTES / 1_000_000:.1f} MB and could not be scaled down."
            )
            continue
        if len(pictures) == MAX_PICTURES:
            notes.append(f"Only the first {MAX_PICTURES} pictures were offered; the folder has {len(files)}.")
            break
        width, height = _size(path)
        pictures.append(
            Picture(path.name, path, MIME_TYPES[path.suffix.lower()], width, height, captions.get(path.name.lower(), ""))
        )
    if not pictures:
        raise ValueError(f"None of the pictures in '{resolved}' can be used. " + " ".join(notes))
    return pictures, notes


def manifest(pictures: list[Picture]) -> str:
    """What the model is told: the files that exist, how big, what they show."""
    lines = [
        "PICTURES -- these files are the only images that exist. Place them with "
        '<img src="FILE" alt="..."> or CSS background-image: url("FILE"), using the exact file name '
        "with nothing in front of it:"
    ]
    for picture in pictures:
        size = f" ({picture.width}x{picture.height})" if picture.width and picture.height else ""
        caption = f": {picture.caption}" if picture.caption else ""
        lines.append(f"- {picture.name}{size}{caption}")
    return "\n".join(lines)


def _data_uri(picture: Picture) -> str:
    data = picture.path.read_bytes()
    mime = picture.mime
    if len(data) > MAX_BYTES:
        image = _pillow()
        with image.open(io.BytesIO(data)) as opened:
            opened.thumbnail((MAX_SIDE, MAX_SIDE))
            buffer = io.BytesIO()
            if opened.mode in ("RGBA", "LA", "P"):
                opened.convert("RGBA").save(buffer, format="PNG", optimize=True)
                mime = "image/png"
            else:
                opened.convert("RGB").save(buffer, format="JPEG", quality=85)
                mime = "image/jpeg"
        data = buffer.getvalue()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _reference(name: str) -> re.Pattern:
    """`name` where a page refers to the file: in a quoted attribute or
    string, in url(...), in a srcset -- with whatever path the model put in
    front of it. Not in an alt or title text, and not in the middle of a
    longer name."""
    return re.compile(
        r"""(?<!alt=["'])(?<!title=["'])(?<=["'`(=\s,])"""
        r"(?:https?://[\w.:-]+)?(?:\.{0,2}/)?(?:[\w.-]+/)*"
        + re.escape(name)
        + r"""(?=["'`)\s,>]|$)"""
    )


def referenced(html: str, names: list[str]) -> list[str]:
    """Those of `names` the page refers to as files, in the order given."""
    return [name for name in names if _reference(name).search(html)]


def embed(html: str, pictures: list[Picture]) -> tuple[str, list[str]]:
    """Put each referenced picture into the page itself. Returns the page
    and the names of the pictures it used."""
    used = []
    for picture in pictures:
        pattern = _reference(picture.name)
        if not pattern.search(html):
            continue
        uri = _data_uri(picture)
        html = pattern.sub(lambda _match: uri, html)
        used.append(picture.name)
    return html, used
