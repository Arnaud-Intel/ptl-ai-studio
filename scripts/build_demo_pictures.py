"""Rebuild the fictional pictures the HTML Creator's samples place on a page.

Drawn here, as SVG, from seeded random numbers: no downloaded photos, no
real places, brands or people, and the same files every run. Standard
library only. Outputs stay under sample-data/pictures.
"""
from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "sample-data" / "pictures"


def n(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def linear(name: str, stops, x1=0, y1=0, x2=0, y2=1, units: str | None = None) -> str:
    body = "".join(
        f'<stop offset="{offset}" stop-color="{color}"' + (f' stop-opacity="{alpha}"' if alpha is not None else "") + "/>"
        for offset, color, alpha in stops
    )
    space = f' gradientUnits="{units}"' if units else ""
    return f'<linearGradient id="{name}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"{space}>{body}</linearGradient>'


def radial(name: str, stops) -> str:
    body = "".join(f'<stop offset="{offset}" stop-color="{color}" stop-opacity="{alpha}"/>' for offset, color, alpha in stops)
    return f'<radialGradient id="{name}">{body}</radialGradient>'


def blur(name: str, amount: float) -> str:
    return f'<filter id="{name}" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="{amount}"/></filter>'


def ridge(rng: random.Random, x0: float, x1: float, base: float, amp: float, rough: float = 0.52, detail: int = 6):
    """A skyline from x0 to x1 around `base`: midpoint displacement."""
    heights = [rng.uniform(-amp, amp) * 0.3, rng.uniform(-amp, amp) * 0.3]
    scale = amp
    for _ in range(detail):
        finer = []
        for a, b in zip(heights, heights[1:]):
            finer += [a, (a + b) / 2 + rng.uniform(-scale, scale)]
        finer.append(heights[-1])
        heights = finer
        scale *= rough
    last = len(heights) - 1
    return [(x0 + (x1 - x0) * i / last, base + h) for i, h in enumerate(heights)]


def land(points, floor: float, fill: str, extra: str = "") -> str:
    """The area under a skyline, down to `floor`."""
    line = " ".join(f"L{n(x)},{n(y)}" for x, y in points)
    return f'<path d="M{n(points[0][0])},{n(floor)} {line} L{n(points[-1][0])},{n(floor)} Z" fill="{fill}"{extra}/>'


def peaks(rng: random.Random, summits, jitter: float = 14, steps: int = 5):
    """A skyline through the given summits and valleys, roughened between them."""
    points = []
    for (xa, ya), (xb, yb) in zip(summits, summits[1:]):
        for i in range(steps):
            t = i / steps
            wobble = 0 if i == 0 else rng.uniform(-jitter, jitter)
            points.append((xa + (xb - xa) * t, ya + (yb - ya) * t + wobble))
    points.append(summits[-1])
    return points


def height_at(points, x: float) -> float:
    for (xa, ya), (xb, yb) in zip(points, points[1:]):
        if xa <= x <= xb:
            return ya + (yb - ya) * (x - xa) / (xb - xa or 1)
    return points[-1][1]


def stars(rng: random.Random, width: float, height: float, count: int) -> str:
    return "".join(
        f'<circle cx="{n(rng.uniform(0, width))}" cy="{n(rng.uniform(0, height))}" '
        f'r="{rng.choice([0.7, 0.9, 1.1, 1.7])}" fill="#fff" opacity="{rng.uniform(0.3, 0.95):.2f}"/>'
        for _ in range(count)
    )


def pine(x: float, base: float, height: float, fill: str) -> str:
    width = height * 0.42
    top = base - height
    tiers = "".join(
        f'<polygon points="{n(x)},{n(top + a * height)} {n(x + half * width)},{n(top + b * height)} {n(x - half * width)},{n(top + b * height)}"/>'
        for a, b, half in ((0, 0.4, 0.3), (0.2, 0.66, 0.41), (0.42, 0.9, 0.5))
    )
    trunk = f'<rect x="{n(x - width * 0.04)}" y="{n(base - height * 0.12)}" width="{n(width * 0.08)}" height="{n(height * 0.12)}"/>'
    return f'<g fill="{fill}">{trunk}{tiers}</g>'


def pines(rng: random.Random, skyline, count: int, low: float, high: float, fill: str, x0=None, x1=None) -> str:
    x0 = skyline[0][0] if x0 is None else x0
    x1 = skyline[-1][0] if x1 is None else x1
    spots = sorted(rng.uniform(x0, x1) for _ in range(count))
    return "".join(pine(x, height_at(skyline, x) + 6, rng.uniform(low, high), fill) for x in spots)


def glow(x: float, y: float, r: float, gradient: str) -> str:
    return f'<circle cx="{n(x)}" cy="{n(y)}" r="{n(r)}" fill="url(#{gradient})"/>'


def shimmer(rng: random.Random, x: float, top: float, bottom: float, spread: float, color: str, count: int = 26) -> str:
    """A light's reflection on water: short streaks, wider and fainter with distance from it."""
    out = []
    for i in range(count):
        t = i / (count - 1)
        y = top + (bottom - top) * t ** 1.4
        half = (10 + spread * t) * rng.uniform(0.5, 1)
        out.append(
            f'<rect x="{n(x - half + rng.uniform(-8, 8) * t)}" y="{n(y)}" width="{n(half * 2)}" height="{n(1.6 + 3 * t)}" '
            f'rx="2" fill="{color}" opacity="{0.75 * (1 - t) + 0.12:.2f}"/>'
        )
    return "".join(out)


def ripples(rng: random.Random, width: float, top: float, bottom: float, color: str, count: int = 40) -> str:
    out = []
    for _ in range(count):
        t = rng.random()
        y = top + (bottom - top) * t
        length = rng.uniform(30, 150) * (0.4 + t)
        out.append(
            f'<rect x="{n(rng.uniform(0, width))}" y="{n(y)}" width="{n(length)}" height="{n(1 + 2 * t)}" rx="1.5" '
            f'fill="{color}" opacity="{rng.uniform(0.08, 0.26):.2f}"/>'
        )
    return "".join(out)


def clouds(rng: random.Random, spots, fill: str = "#fff", filt: str = "soft") -> str:
    out = []
    for x, y, scale in spots:
        puffs = "".join(
            f'<ellipse cx="{n(x + rng.uniform(-70, 70) * scale)}" cy="{n(y + rng.uniform(-10, 10) * scale)}" '
            f'rx="{n(rng.uniform(40, 80) * scale)}" ry="{n(rng.uniform(14, 26) * scale)}"/>'
            for _ in range(6)
        )
        out.append(f'<g fill="{fill}" opacity="0.85" filter="url(#{filt})">{puffs}</g>')
    return "".join(out)


def picture(width: int, height: int, title: str, defs: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{title}"><title>{title}</title><defs>{defs}</defs>'
        f'<g clip-path="url(#frame)">{body}</g></svg>\n'
    )


def frame(width: int, height: int) -> str:
    return f'<clipPath id="frame"><rect width="{width}" height="{height}"/></clipPath>'


# ---------------------------------------------------------------------------- the pictures


def aurora_ridge() -> str:
    w, h = 1600, 900
    rng = random.Random(11)
    defs = frame(w, h) + linear("sky", [(0, "#040a1c", None), (0.5, "#0a2244", None), (0.8, "#11475a", None), (1, "#1c6b66", None)])
    defs += linear("green", [(0, "#4dffa6", 0), (0.3, "#4dffa6", 1), (0.7, "#1fe0a8", 0.7), (1, "#1fe0a8", 0)])
    defs += linear("violet", [(0, "#b48cff", 0), (0.4, "#9d7bff", 0.7), (1, "#5aa9ff", 0)])
    defs += blur("haze", 26) + blur("soft", 9)
    defs += radial("moon", [(0, "#fffbe6", 0.55), (1, "#fffbe6", 0)]) + radial("lamp", [(0, "#ffc46b", 0.9), (1, "#ffc46b", 0)])
    defs += linear("mist", [(0, "#9be7d8", 0), (0.5, "#9be7d8", 0.22), (1, "#9be7d8", 0)])
    body = f'<rect width="{w}" height="{h}" fill="url(#sky)"/>' + stars(rng, w, 520, 230)
    body += glow(1330, 140, 120, "moon") + '<circle cx="1330" cy="140" r="27" fill="#f6f2dc"/><circle cx="1342" cy="132" r="24" fill="#0a1f40"/>'
    for path, gradient, width, opacity in (
        ("M-80,360 C240,150 520,430 860,250 S1380,120 1700,300", "green", 150, 1),
        ("M-80,250 C300,120 700,300 1000,170 S1500,90 1700,190", "violet", 110, 0.7),
        ("M-60,450 C320,330 640,470 980,350 S1420,300 1700,420", "green", 90, 0.65),
    ):
        body += f'<path d="{path}" fill="none" stroke="url(#{gradient})" stroke-width="{width}" stroke-linecap="round" opacity="{opacity}" filter="url(#haze)"/>'
    far = ridge(rng, 0, w, 560, 95)
    body += land(far, h, "#143f5f")
    body += f'<rect y="520" width="{w}" height="180" fill="url(#mist)"/>'
    middle = ridge(rng, 0, w, 650, 125)
    body += land(middle, h, "#0d2c48")
    near = ridge(rng, 0, w, 760, 80, rough=0.45)
    body += land(near, h, "#081b30") + pines(rng, near, 46, 40, 86, "#061425")
    x = 1120
    y = height_at(near, x) + 4
    body += glow(x, y - 14, 90, "lamp")
    body += f'<polygon points="{x - 30},{n(y)} {x},{n(y - 40)} {x + 30},{n(y)}" fill="#ffb454"/>'
    body += f'<polygon points="{x - 7},{n(y)} {x},{n(y - 22)} {x + 7},{n(y)}" fill="#7a3f12"/>'
    front = ridge(rng, 0, w, 850, 46, rough=0.4)
    body += land(front, h, "#030b17") + pines(rng, front, 26, 70, 150, "#020812")
    return picture(w, h, "Aurora over a mountain ridge at night, a lit tent on the slope", defs, body)


def fjord_morning() -> str:
    w, h = 1200, 800
    rng = random.Random(23)
    water = 500
    defs = frame(w, h) + linear("sky", [(0, "#27477c", None), (0.42, "#c9788a", None), (0.58, "#f6b47a", None), (0.66, "#ffe1ad", None), (1, "#ffe1ad", None)])
    defs += linear("sea", [(0, "#f7c78f", None), (0.18, "#c98b8f", None), (0.5, "#35618f", None), (1, "#142f52", None)])
    defs += radial("sun", [(0, "#fff6cf", 0.95), (0.35, "#ffd98e", 0.5), (1, "#ffd98e", 0)]) + blur("soft", 7)
    defs += linear("left", [(0, "#3b4a6e", None), (1, "#1c2742", None)]) + linear("right", [(0, "#4a5a80", None), (1, "#222f4f", None)])
    body = f'<rect width="{w}" height="{h}" fill="url(#sky)"/>'
    body += glow(610, 470, 260, "sun") + '<circle cx="610" cy="470" r="44" fill="#fff4c9"/>'
    body += clouds(rng, [(260, 150, 1.3), (880, 110, 1.6), (1040, 250, 0.9)], fill="#ffd9c2")
    body += land(ridge(rng, 0, w, 440, 55), water, "#8f7ea8", ' opacity="0.9"')

    def wall(x0, x1, top, steep, gradient, flip=False):
        points = []
        for x, y in ridge(rng, x0, x1, 0, 38, rough=0.55):
            t = (x - x0) / (x1 - x0)
            t = 1 - t if flip else t
            points.append((x, top + (water - top) * t ** steep + y * (1 - t * 0.7)))
        return land(points, water, f"url(#{gradient})")

    body += wall(0, 520, 130, 1.5, "left") + wall(700, w, 170, 1.4, "right", flip=True)
    body += wall(0, 330, 250, 1.2, "left") + wall(880, w, 300, 1.3, "right", flip=True)
    body += f'<rect y="{water}" width="{w}" height="{h - water}" fill="url(#sea)"/>'
    body += shimmer(rng, 610, water + 4, h, 90, "#fff1c4") + ripples(rng, w, water + 20, h, "#ffffff")
    x, y = 842, water
    body += f'<rect x="{x}" y="{y - 26}" width="40" height="26" fill="#b83a34"/><polygon points="{x - 5},{y - 26} {x + 20},{y - 44} {x + 45},{y - 26}" fill="#2a2231"/>'
    body += f'<rect x="{x + 15}" y="{y - 15}" width="9" height="9" fill="#ffd98e"/><rect x="{x + 40}" y="{y - 3}" width="46" height="3" fill="#3a2b2b"/>'
    body += f'<rect x="{x}" y="{y + 2}" width="40" height="16" fill="#b83a34" opacity="0.28"/>'
    for bx, by in ((420, 250), (446, 238), (470, 258), (720, 300)):
        body += f'<path d="M{bx},{by} q7,-7 14,0 q7,-7 14,0" fill="none" stroke="#2a2f4a" stroke-width="2" stroke-linecap="round"/>'
    return picture(w, h, "A fjord at sunrise, a red cabin on the shore", defs, body)


def forest_trail() -> str:
    w, h = 1200, 800
    rng = random.Random(37)
    defs = frame(w, h) + linear("air", [(0, "#eef6e6", None), (0.45, "#b9d8bd", None), (1, "#6fa283", None)])
    defs += radial("sun", [(0, "#fffbe0", 0.95), (1, "#fffbe0", 0)]) + blur("soft", 12)
    defs += linear("fog", [(0, "#ffffff", 0), (0.5, "#ffffff", 0.5), (1, "#ffffff", 0)])
    defs += linear("floor", [(0, "#5f9a72", None), (1, "#2c5a40", None)]) + linear("path", [(0, "#f3e8cf", None), (1, "#cdb68f", None)])
    body = f'<rect width="{w}" height="{h}" fill="url(#air)"/>' + glow(250, 150, 330, "sun")
    for x in (120, 260, 430):
        body += f'<polygon points="250,150 {x + 380},800 {x + 560},800" fill="#fffbe0" opacity="0.10"/>'
    for base, low, high, fill, count, fog in (
        (410, 70, 120, "#8fb79b", 60, 380),
        (500, 120, 190, "#5d9272", 44, 470),
        (600, 200, 290, "#34694c", 30, 560),
    ):
        skyline = ridge(rng, 0, w, base, 16, rough=0.4, detail=4)
        body += land(skyline, h, fill) + pines(rng, skyline, count, low, high, fill)
        body += f'<rect y="{fog}" width="{w}" height="110" fill="url(#fog)" filter="url(#soft)"/>'
    body += f'<rect y="600" width="{w}" height="200" fill="url(#floor)"/>'
    body += '<path d="M430,800 C500,720 640,690 612,640 C596,612 600,602 604,596 L622,596 C640,612 652,632 690,672 C760,744 742,760 800,800 Z" fill="url(#path)"/>'
    ground = [(0, 760), (w, 760)]
    body += pines(rng, ground, 7, 380, 560, "#173a2b", 0, 330) + pines(rng, ground, 7, 380, 560, "#173a2b", 880, w)
    body += f'<rect y="700" width="{w}" height="100" fill="#0f2a1f" opacity="0.35"/>'
    return picture(w, h, "A sandy trail into a misty pine forest", defs, body)


def glacier_lake() -> str:
    w, h = 1200, 800
    rng = random.Random(41)
    shore = 470
    defs = frame(w, h) + linear("sky", [(0, "#5fb4ea", None), (0.6, "#bfe6fb", None), (1, "#e9f8ff", None)]) + blur("soft", 6)
    defs += linear("peak", [(0, "#ffffff", None), (0.45, "#dcecf8", None), (1, "#5f84ab", None)], 0, 110, 0, shore, "userSpaceOnUse")
    defs += linear("slope", [(0, "#f2f9ff", None), (0.5, "#9dbbd6", None), (1, "#466c92", None)], 0, 260, 0, shore, "userSpaceOnUse")
    defs += linear("lake", [(0, "#56d6cf", None), (0.4, "#22a6b3", None), (1, "#0d5873", None)])
    defs += linear("ice", [(0, "#ffffff", None), (1, "#bfe3f5", None)])
    body = f'<rect width="{w}" height="{h}" fill="url(#sky)"/>' + clouds(rng, [(200, 120, 1.5), (760, 90, 1.2), (1010, 190, 1.7)])
    high = [(0, 330), (150, 240), (240, 320), (330, 150), (430, 300), (520, 215), (585, 290), (650, 110), (735, 275),
            (810, 190), (885, 300), (965, 140), (1045, 315), (1125, 250), (w, 340)]
    body += land(peaks(rng, high), shore, "url(#peak)")
    # The glacier comes down the saddle between two summits and spreads at the lake.
    body += '<path d="M700,262 C716,320 668,372 590,470 L880,470 C812,410 778,340 768,262 Z" fill="url(#ice)"/>'
    for start, bend, end in ((716, 690, 640), (734, 720, 720), (750, 760, 800), (760, 790, 850)):
        body += f'<path d="M{start},285 Q{bend},380 {end},468" fill="none" stroke="#8fc4e0" stroke-width="2" opacity="0.75"/>'
    low = [(0, 400), (120, 370), (230, 420), (360, 380), (480, 440), (590, 470)]
    body += land(peaks(rng, low, jitter=9), shore, "url(#slope)")
    low = [(880, 470), (960, 420), (1060, 445), (1140, 390), (w, 420)]
    body += land(peaks(rng, low, jitter=9), shore, "url(#slope)")
    body += f'<rect y="{shore}" width="{w}" height="{h - shore}" fill="url(#lake)"/>'
    body += f'<rect y="{shore}" width="{w}" height="70" fill="#ffffff" opacity="0.16"/>' + ripples(rng, w, shore + 16, h, "#ffffff", 55)
    for x, y, r in ((300, 560, 26), (350, 574, 14), (880, 610, 34), (640, 520, 12)):
        body += f'<polygon points="{x - r},{y} {x - r * 0.4},{y - r * 0.8} {x + r * 0.3},{y - r} {x + r},{y}" fill="#f4fbff"/>'
        body += f'<polygon points="{x - r},{y} {x + r},{y} {x + r * 0.5},{y + r * 0.35} {x - r * 0.6},{y + r * 0.3}" fill="#bfe3f5" opacity="0.6"/>'
    bank = ridge(rng, 0, 420, 730, 26, rough=0.45, detail=4)
    body += land(bank, h, "#26363f") + pines(rng, bank, 9, 110, 210, "#1c2b33")
    body += land(ridge(rng, 300, w, 792, 14, rough=0.4, detail=4), h, "#1a272e")
    return picture(w, h, "A turquoise lake below a glacier and snowy peaks", defs, body)


def lighthouse() -> str:
    w, h = 1200, 800
    rng = random.Random(53)
    horizon = 500
    defs = frame(w, h) + linear("sky", [(0, "#241a4a", None), (0.38, "#76397a", None), (0.52, "#e7745b", None), (0.625, "#ffd28c", None), (1, "#ffd28c", None)])
    defs += linear("sea", [(0, "#f3a66d", None), (0.2, "#8a5a8f", None), (0.55, "#33457f", None), (1, "#141f4a", None)])
    defs += radial("sun", [(0, "#fff0c2", 0.95), (0.4, "#ffc37a", 0.45), (1, "#ffc37a", 0)]) + blur("soft", 8)
    defs += linear("beam", [(0, "#fff3b0", 0.75), (1, "#fff3b0", 0)], 1, 0, 0, 0) + linear("rock", [(0, "#3a2c55", None), (1, "#191330", None)])
    body = f'<rect width="{w}" height="{h}" fill="url(#sky)"/>' + stars(rng, w, 240, 70)
    body += glow(300, horizon, 240, "sun") + f'<circle cx="300" cy="{horizon}" r="56" fill="#ffe7ae"/>'
    body += clouds(rng, [(560, 330, 1.4), (150, 260, 1.1), (980, 220, 1.2)], fill="#f29a7e")
    body += f'<rect y="{horizon}" width="{w}" height="{h - horizon}" fill="url(#sea)"/>'
    body += shimmer(rng, 300, horizon + 4, h, 110, "#ffe2a1") + ripples(rng, w, horizon + 16, h, "#ffffff", 46)
    cliff = [(x, y + 300 * max(0.0, (760 - x) / 190) ** 1.6) for x, y in ridge(rng, 570, w, 505, 14, rough=0.5, detail=5)]
    body += land(cliff, h, "url(#rock)")
    x = 900
    y = height_at(cliff, x)
    lamp = y - 232
    body += f'<polygon points="{x},{lamp + 14} 60,{lamp - 70} 60,{lamp + 150}" fill="url(#beam)" filter="url(#soft)"/>'
    body += f'<polygon points="{x - 30},{n(y)} {x - 19},{n(y - 210)} {x + 19},{n(y - 210)} {x + 30},{n(y)}" fill="#f6efe4"/>'
    for top, bottom in ((0.18, 0.34), (0.56, 0.72)):
        half_top, half_bottom = 30 - 11 * (1 - top), 30 - 11 * (1 - bottom)
        body += (
            f'<polygon points="{n(x - half_top)},{n(y - 210 * (1 - top))} {n(x + half_top)},{n(y - 210 * (1 - top))} '
            f'{n(x + half_bottom)},{n(y - 210 * (1 - bottom))} {n(x - half_bottom)},{n(y - 210 * (1 - bottom))}" fill="#cf4458"/>'
        )
    body += f'<rect x="{x - 25}" y="{n(y - 216)}" width="50" height="7" fill="#221a36"/>'
    body += f'<rect x="{x - 16}" y="{n(y - 244)}" width="32" height="28" fill="#2c2444"/><rect x="{x - 11}" y="{n(y - 240)}" width="22" height="20" fill="#fff3b0"/>'
    body += f'<polygon points="{x - 21},{n(y - 244)} {x},{n(y - 268)} {x + 21},{n(y - 244)}" fill="#cf4458"/>'
    body += glow(x, lamp + 14, 60, "sun")
    hx = 985
    hy = height_at(cliff, hx + 30)
    body += f'<rect x="{hx}" y="{n(hy - 34)}" width="64" height="36" fill="#efe6d8"/><polygon points="{hx - 6},{n(hy - 34)} {hx + 32},{n(hy - 58)} {hx + 70},{n(hy - 34)}" fill="#7d2f45"/>'
    body += f'<rect x="{hx + 12}" y="{n(hy - 24)}" width="12" height="12" fill="#ffd98e"/><rect x="{hx + 40}" y="{n(hy - 24)}" width="12" height="24" fill="#4b3349"/>'
    for i in range(5):
        wx = 560 + i * 26
        body += f'<path d="M{wx},{760 - i * 9} q16,-12 32,0" fill="none" stroke="#ffffff" stroke-width="3" opacity="0.5" stroke-linecap="round"/>'
    return picture(w, h, "A lighthouse on a cliff at dusk, its beam over the sea", defs, body)


def plateau_cabin() -> str:
    w, h = 1200, 800
    rng = random.Random(67)
    defs = frame(w, h) + linear("sky", [(0, "#0b1736", None), (0.55, "#27406e", None), (1, "#6076aa", None)])
    defs += radial("moon", [(0, "#fffbe6", 0.6), (1, "#fffbe6", 0)]) + radial("lamp", [(0, "#ffc857", 0.85), (1, "#ffc857", 0)])
    defs += linear("far", [(0, "#c9d6ee", None), (1, "#8196c2", None)]) + linear("mid", [(0, "#e3ebf8", None), (1, "#9fb3d8", None)])
    defs += linear("near", [(0, "#f7faff", None), (1, "#b9c9e6", None)]) + blur("soft", 7)
    body = f'<rect width="{w}" height="{h}" fill="url(#sky)"/>' + stars(rng, w, 430, 170)
    body += glow(930, 150, 150, "moon") + '<circle cx="930" cy="150" r="42" fill="#f8f4e3"/>'
    for cx, cy, r in ((918, 140, 9), (944, 164, 6), (936, 132, 4)):
        body += f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#e2dcc4" opacity="0.7"/>'
    body += land(ridge(rng, 0, w, 430, 70, rough=0.55), h, "#3d5283")
    body += '<path d="M0,520 C200,440 420,470 640,500 S1020,430 1200,480 L1200,800 L0,800 Z" fill="url(#far)"/>'
    body += '<path d="M0,600 C240,540 470,570 700,590 S1040,560 1200,600 L1200,800 L0,800 Z" fill="url(#mid)"/>'
    skyline = [(0, 560), (300, 548), (560, 574), (w, 590)]
    body += pines(rng, skyline, 5, 90, 150, "#1d3440", 60, 330) + pines(rng, skyline, 4, 80, 130, "#1d3440", 900, 1150)
    x, y = 520, 600
    body += glow(x + 78, y - 34, 120, "lamp")
    for i in range(7):
        body += f'<circle cx="{n(x + 104 + i * 9 + rng.uniform(-4, 4))}" cy="{n(y - 132 - i * 30)}" r="{n(9 + i * 5)}" fill="#dfe6f5" opacity="{0.42 - i * 0.05:.2f}" filter="url(#soft)"/>'
    body += f'<rect x="{x + 96}" y="{y - 124}" width="16" height="34" fill="#3b2a22"/>'
    body += f'<rect x="{x}" y="{y - 70}" width="140" height="70" fill="#5e3d2b"/>'
    for i in range(1, 5):
        body += f'<rect x="{x}" y="{y - 70 + i * 14}" width="140" height="2" fill="#472c1e"/>'
    body += f'<polygon points="{x - 14},{y - 68} {x + 70},{y - 118} {x + 154},{y - 68}" fill="#3a2419"/>'
    body += f'<polygon points="{x - 14},{y - 68} {x + 70},{y - 118} {x + 154},{y - 68} {x + 142},{y - 72} {x + 70},{y - 106} {x - 2},{y - 72}" fill="#f7faff"/>'
    body += f'<rect x="{x + 60}" y="{y - 48}" width="38" height="30" fill="#ffc857"/><rect x="{x + 78}" y="{y - 48}" width="2" height="30" fill="#5e3d2b"/><rect x="{x + 60}" y="{y - 34}" width="38" height="2" fill="#5e3d2b"/>'
    body += f'<rect x="{x + 16}" y="{y - 46}" width="26" height="46" fill="#3a2419"/>'
    body += '<path d="M0,690 C260,630 520,650 760,676 S1060,650 1200,690 L1200,800 L0,800 Z" fill="url(#near)"/>'
    body += f'<path d="M{x + 30},{y + 4} C{x + 10},{y + 60} {x - 120},{y + 110} {x - 260},{y + 200}" fill="none" stroke="#9fb3d8" stroke-width="5" stroke-dasharray="3 16" stroke-linecap="round"/>'
    return picture(w, h, "A wooden cabin with a lit window on a snowy plateau at night", defs, body)


def midnight_bay() -> str:
    w, h = 1200, 800
    rng = random.Random(79)
    horizon = 420
    defs = frame(w, h) + linear("sky", [(0, "#43527c", None), (0.3, "#c97c7a", None), (0.45, "#f7a868", None), (0.525, "#ffe09a", None), (1, "#ffe09a", None)])
    defs += linear("sea", [(0, "#ffd68c", None), (0.22, "#ea9069", None), (0.6, "#50507f", None), (1, "#232a55", None)])
    defs += radial("sun", [(0, "#fffbe0", 1), (0.3, "#ffe09a", 0.6), (1, "#ffe09a", 0)]) + blur("soft", 7)
    body = f'<rect width="{w}" height="{h}" fill="url(#sky)"/>' + glow(800, horizon - 26, 280, "sun")
    body += f'<circle cx="800" cy="{horizon - 26}" r="48" fill="#fffbe3"/>' + clouds(rng, [(300, 170, 1.5), (1020, 130, 1.2)], fill="#f7b48d")
    for x0, x1, amp, fill in ((0, 430, 34, "#5a4a78"), (520, 700, 16, "#6d5a86"), (940, w, 46, "#4a3f6b")):
        isle = [(x, min(y, horizon)) for x, y in ridge(rng, x0, x1, horizon - amp * 0.7, amp, rough=0.5, detail=5)]
        isle[0], isle[-1] = (isle[0][0], horizon), (isle[-1][0], horizon)
        body += land(isle, horizon, fill)
    body += f'<rect y="{horizon}" width="{w}" height="{h - horizon}" fill="url(#sea)"/>'
    body += shimmer(rng, 800, horizon + 4, h, 130, "#fff3c2", 32) + ripples(rng, w, horizon + 14, h, "#ffffff", 50)
    for x, y, hull, scale in ((400, 640, "#d9463f", 1.6), (660, 540, "#f0bb3c", 1.05)):
        s = scale
        boat = f'<path d="M{n(x - 78 * s)},{y} Q{x},{n(y - 16 * s)} {n(x + 78 * s)},{y} Q{x},{n(y + 13 * s)} {n(x - 78 * s)},{y} Z" fill="{hull}"/>'
        boat += f'<path d="M{n(x - 16 * s)},{y} L{n(x - 9 * s)},{n(y - 34 * s)} L{n(x + 9 * s)},{n(y - 34 * s)} L{n(x + 14 * s)},{y} Z" fill="#26304f"/>'
        boat += f'<circle cx="{x}" cy="{n(y - 44 * s)}" r="{n(9 * s)}" fill="#26304f"/>'
        boat += f'<line x1="{n(x - 58 * s)}" y1="{n(y - 46 * s)}" x2="{n(x + 56 * s)}" y2="{n(y + 4 * s)}" stroke="#26304f" stroke-width="{n(3.2 * s)}" stroke-linecap="round"/>'
        boat += f'<ellipse cx="{n(x - 60 * s)}" cy="{n(y - 47 * s)}" rx="{n(9 * s)}" ry="{n(4.5 * s)}" fill="#26304f" transform="rotate(24 {n(x - 60 * s)} {n(y - 47 * s)})"/>'
        boat += f'<ellipse cx="{x}" cy="{n(y + 16 * s)}" rx="{n(74 * s)}" ry="{n(7 * s)}" fill="{hull}" opacity="0.22"/>'
        body += boat
    rocks = ridge(rng, 0, 380, 760, 30, rough=0.5, detail=4)
    body += land(rocks, h, "#1b1c3a")
    return picture(w, h, "Two kayaks on a calm bay under the midnight sun", defs, body)


NORDLYS = {
    "hero-aurora-ridge.svg": (aurora_ridge, "wide night panorama: green aurora over mountain ridges, a small lit tent; best as the full-width hero background"),
    "fjord-morning.svg": (fjord_morning, "a fjord at sunrise with steep rock walls and a red cabin on the shore"),
    "forest-trail.svg": (forest_trail, "a sandy trail leading into a misty green pine forest"),
    "glacier-lake.svg": (glacier_lake, "a turquoise lake below a glacier and snowy peaks, daylight"),
    "lighthouse-dusk.svg": (lighthouse, "a red-and-white lighthouse on a cliff at dusk, its beam over the sea"),
    "plateau-cabin.svg": (plateau_cabin, "a wooden cabin with a lit window on a snowy plateau under a full moon"),
    "midnight-bay.svg": (midnight_bay, "two sea kayaks on a calm golden bay under the midnight sun"),
}


def build(folder: str, pictures: dict, intro: str) -> None:
    target = ROOT / folder
    target.mkdir(parents=True, exist_ok=True)
    lines = [f"# {intro}", "# One line per picture: file name, a colon, what it shows.", ""]
    for name, (draw, caption) in pictures.items():
        (target / name).write_text(draw(), encoding="utf-8", newline="\n")
        lines.append(f"{name}: {caption}")
    (target / "captions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(pictures)} pictures to {target}")


if __name__ == "__main__":
    build("nordlys-trails", NORDLYS, "FICTIONAL DEMO PICTURES for the invented travel brand Nordlys Trails. Drawn by scripts/build_demo_pictures.py; no real place.")
