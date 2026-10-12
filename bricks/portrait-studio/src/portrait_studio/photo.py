"""The photo, made into what the model is given: upright, square, at the
size the picture will be drawn at.

Square because a portrait is a face in the middle of a frame: a webcam's
wide picture loses its two sides, a phone's tall one its top and bottom.
The model draws in multiples of 16 pixels.
"""
from __future__ import annotations

import io

SIZES = (512, 768, 1024)
DEFAULT_SIZE = 768
# A photo larger than this is not a photo somebody took to try a demo; and
# decoding is refused before it is attempted on anything a lot larger still.
MAX_BYTES = 20 * 1024 * 1024
_MAX_PIXELS = 60_000_000


def read(data: bytes):
    """The picture in `data` (JPEG, PNG, WebP...), upright. ValueError if
    it is not a picture, or not one of a reasonable size."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not data:
        raise ValueError("There is no photo: take one, upload one, or pick a sample.")
    if len(data) > MAX_BYTES:
        raise ValueError(f"That photo is {len(data) / 1e6:.0f} MB: use one under {MAX_BYTES // (1024 * 1024)} MB.")
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > _MAX_PIXELS:
            raise ValueError(f"That photo is {image.width} x {image.height} pixels: use a smaller one.")
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("That file could not be read as a picture.") from exc
    # A phone says which way up its picture is beside the picture, not in it.
    return ImageOps.exif_transpose(image).convert("RGB")


def square(image, size: int = DEFAULT_SIZE):
    """The middle of the picture, as a square of `size` pixels."""
    from PIL import Image

    if size not in SIZES:
        raise ValueError(f"The picture is drawn at {', '.join(map(str, SIZES))} pixels, not {size}.")
    side = min(image.width, image.height)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)
