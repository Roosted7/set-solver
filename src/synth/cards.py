"""Procedural Set card renderer.

Draws any of the 81 cards at any resolution with a randomised "deck style" (ink colours,
line widths, stripe density, symbol proportions), so the models never depend on one
particular deck scan or resolution.

Card id convention (same as the web app): id = number*27 + color*9 + shape*3 + fill
  number 0..2 -> 1,2,3     color 0..2 -> red, green, purple
  shape  0..2 -> diamond, oval, squiggle     fill 0..2 -> open, solid, striped
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np

CARD_ASPECT = 57 / 89  # width / height (portrait)
NUMBER = ['1', '2', '3']
COLOR = ['red', 'green', 'purple']
SHAPE = ['diamond', 'oval', 'squiggle']
FILL = ['open', 'solid', 'striped']


def attrs(cid: int):
    return cid // 27, (cid // 9) % 3, (cid // 3) % 3, cid % 3


def card_name(cid: int) -> str:
    n, c, s, f = attrs(cid)
    return f'{NUMBER[n]} {FILL[f]} {COLOR[c]} {SHAPE[s]}'


def _hsv_jitter(rgb, dh, ds, dv, rng):
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV).astype(np.float32)[0, 0]
    hsv[0] = (hsv[0] + rng.uniform(-dh, dh)) % 180
    hsv[1] = np.clip(hsv[1] * rng.uniform(1 - ds, 1 + ds), 0, 255)
    hsv[2] = np.clip(hsv[2] * rng.uniform(1 - dv, 1 + dv), 0, 255)
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0, 0])


@dataclass
class DeckStyle:
    inks: list            # 3 RGB tuples
    paper: tuple          # RGB
    line_w: float         # outline width as fraction of symbol height
    stripe_gap: float     # stripe spacing as fraction of symbol height
    stripe_w: float       # stripe width as fraction of gap
    sym_w: float          # symbol width as fraction of card width
    sym_h: float          # symbol height as fraction of symbol width
    gap: float            # vertical gap between symbols as fraction of symbol height
    squiggle_var: float
    corner_r: float       # card corner radius as fraction of width
    border: tuple | None  # optional thin border colour
    stripe_alpha: float = 1.0  # faint, printed-looking stripes

    @staticmethod
    def random(rng: random.Random) -> 'DeckStyle':
        base = [(225, 30, 45), (0, 150, 70), (95, 40, 145)]
        # some decks: orange-ish red, teal-ish green, blue-ish purple
        inks = [_hsv_jitter(base[0], 6, 0.25, 0.2, rng), _hsv_jitter(base[1], 10, 0.3, 0.3, rng), _hsv_jitter(base[2], 10, 0.3, 0.35, rng)]
        paper = _hsv_jitter((248, 247, 242), 10, 1.0, 0.04, rng)
        return DeckStyle(
            inks=inks, paper=paper,
            line_w=rng.uniform(0.03, 0.08), stripe_gap=rng.choice([rng.uniform(0.03, 0.07), rng.uniform(0.06, 0.14)]), stripe_w=rng.uniform(0.12, 0.5),
            sym_w=rng.uniform(0.52, 0.74), sym_h=rng.uniform(0.38, 0.52), gap=rng.uniform(0.2, 0.5),
            squiggle_var=rng.uniform(0, 1), corner_r=rng.uniform(0.05, 0.1),
            border=None if rng.random() < 0.8 else _hsv_jitter((200, 200, 200), 180, 1, 0.2, rng),
            stripe_alpha=1.0 if rng.random() < 0.5 else rng.uniform(0.35, 0.9),
        )


# Squiggle outlines traced from a real deck (width-normalised, centred, 64 points each)
SQUIGGLE_TEMPLATES = [
    np.array([[0.417, -0.243], [0.38, -0.228], [0.345, -0.208], [0.311, -0.186], [0.275, -0.168], [0.239, -0.151], [0.2, -0.141], [0.16, -0.134], [0.116, -0.134], [0.077, -0.143], [0.037, -0.151], [-0.0, -0.164], [-0.039, -0.177], [-0.076, -0.191], [-0.116, -0.2], [-0.155, -0.208], [-0.197, -0.213], [-0.236, -0.213], [-0.278, -0.211], [-0.317, -0.2], [-0.353, -0.182], [-0.387, -0.16], [-0.417, -0.134], [-0.443, -0.101], [-0.462, -0.066], [-0.478, -0.029], [-0.491, 0.008], [-0.498, 0.049], [-0.5, 0.091], [-0.496, 0.133], [-0.487, 0.172], [-0.466, 0.207], [-0.435, 0.23], [-0.396, 0.239], [-0.359, 0.225], [-0.324, 0.204], [-0.29, 0.182], [-0.254, 0.164], [-0.217, 0.149], [-0.177, 0.143], [-0.137, 0.134], [-0.096, 0.138], [-0.056, 0.147], [-0.018, 0.16], [0.019, 0.173], [0.057, 0.186], [0.099, 0.191], [0.14, 0.195], [0.183, 0.195], [0.225, 0.191], [0.264, 0.182], [0.302, 0.169], [0.338, 0.151], [0.372, 0.13], [0.404, 0.103], [0.43, 0.073], [0.452, 0.039], [0.469, 0.003], [0.482, -0.035], [0.491, -0.075], [0.496, -0.116], [0.493, -0.158], [0.478, -0.195], [0.454, -0.228]]),
    np.array([[-0.309, -0.246], [-0.343, -0.229], [-0.383, -0.22], [-0.413, -0.193], [-0.441, -0.166], [-0.467, -0.135], [-0.483, -0.098], [-0.491, -0.058], [-0.496, -0.016], [-0.496, 0.025], [-0.491, 0.067], [-0.476, 0.104], [-0.456, 0.137], [-0.427, 0.166], [-0.393, 0.185], [-0.357, 0.181], [-0.319, 0.167], [-0.287, 0.146], [-0.245, 0.141], [-0.207, 0.131], [-0.169, 0.128], [-0.129, 0.13], [-0.089, 0.139], [-0.05, 0.152], [-0.016, 0.174], [0.02, 0.193], [0.059, 0.207], [0.099, 0.215], [0.139, 0.224], [0.179, 0.233], [0.22, 0.24], [0.263, 0.241], [0.302, 0.228], [0.34, 0.223], [0.373, 0.203], [0.404, 0.18], [0.435, 0.156], [0.461, 0.128], [0.474, 0.09], [0.487, 0.052], [0.492, 0.01], [0.493, -0.03], [0.491, -0.071], [0.472, -0.104], [0.452, -0.137], [0.417, -0.159], [0.377, -0.159], [0.336, -0.15], [0.302, -0.133], [0.264, -0.127], [0.232, -0.107], [0.191, -0.107], [0.151, -0.107], [0.112, -0.108], [0.074, -0.124], [0.036, -0.137], [-0.002, -0.152], [-0.039, -0.167], [-0.07, -0.189], [-0.106, -0.207], [-0.144, -0.22], [-0.184, -0.228], [-0.225, -0.237], [-0.265, -0.245]]),
]


def _shape_poly(shape: int, w: float, h: float, var: float, n: int = 64) -> np.ndarray:
    """Symbol outline centred at 0, horizontal (width w along x)."""
    if shape == 0:  # diamond
        return np.array([[-w / 2, 0], [0, -h / 2], [w / 2, 0], [0, h / 2]], np.float64)
    if shape == 1:  # oval / stadium
        r = h / 2
        pts = []
        for a in np.linspace(-math.pi / 2, math.pi / 2, n // 2):
            pts.append([w / 2 - r + r * math.cos(a), r * math.sin(a)])
        for a in np.linspace(math.pi / 2, 3 * math.pi / 2, n // 2):
            pts.append([-w / 2 + r + r * math.cos(a), r * math.sin(a)])
        return np.array(pts)
    if var < 0.75:  # traced real outline, slightly varied
        tpl = SQUIGGLE_TEMPLATES[int(var * 100) % len(SQUIGGLE_TEMPLATES)].copy()
        tpl[:, 1] *= (h / w) / (tpl[:, 1].max() - tpl[:, 1].min())   # match requested height
        k = (var * 7.31) % 1 - 0.5
        tpl[:, 1] *= 1 + 0.25 * k * np.cos(math.pi * tpl[:, 0] * 2)  # thicker / thinner waist
        if int(var * 1000) % 10 == 0:
            tpl[:, 0] *= -1                                          # some decks: mirrored
        return tpl * w
    # squiggle: a thick, slanted "~" of nearly constant width with rounded ends (point-symmetric)
    t = np.linspace(-1, 1, n)
    amp = h * (0.13 + 0.08 * var)
    centre = -amp * np.sin(math.pi * t)                         # left half low, right half high
    p = 6 + 6 * var
    thick = h * (0.30 + 0.06 * var) * np.power(np.clip(1 - np.abs(t) ** p, 0, 1), 0.5)
    x = t * w / 2
    top = np.stack([x, centre - thick], 1)
    bot = np.stack([x, centre + thick], 1)[::-1]
    pts = np.concatenate([top, bot])
    return pts


def render_card(cid: int, style: DeckStyle, width: int, rng: random.Random) -> np.ndarray:
    """RGBA card image, portrait, `width` px wide. Drawn at 2x and downsampled for anti-aliasing."""
    n, c, s, f = attrs(cid)
    ss = 2
    W = width * ss
    H = int(round(W / CARD_ASPECT))
    img = np.zeros((H, W, 4), np.uint8)
    # paper with rounded corners
    r = int(style.corner_r * W)
    mask = np.zeros((H, W), np.uint8)
    cv2.rectangle(mask, (r, 0), (W - 1 - r, H - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (W - 1, H - 1 - r), 255, -1)
    for cx, cy in [(r, r), (W - 1 - r, r), (r, H - 1 - r), (W - 1 - r, H - 1 - r)]:
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    img[..., :3] = style.paper
    img[..., 3] = mask
    if style.border is not None:
        cv2.rectangle(img, (int(W * 0.04), int(W * 0.04)), (W - int(W * 0.04), H - int(W * 0.04)), (*style.border, 255), max(1, W // 120))
    ink = style.inks[c]
    sw = style.sym_w * W * rng.uniform(0.97, 1.03)
    sh = style.sym_h * sw
    k = n + 1
    gap = style.gap * sh
    total = k * sh + (k - 1) * gap
    lw = max(1.0, style.line_w * sh)
    S = 16  # fixed-point shift for sub-pixel drawing
    for i in range(k):
        cy = H / 2 - total / 2 + sh / 2 + i * (sh + gap)
        poly = _shape_poly(s, sw, sh, style.squiggle_var) + [W / 2, cy]
        p = np.round(poly * S).astype(np.int32)
        if f == 1:
            cv2.fillPoly(img, [p], (*ink, 255), cv2.LINE_AA, shift=4)
        elif f == 2:
            sym = np.zeros((H, W), np.uint8)
            cv2.fillPoly(sym, [p], 255, cv2.LINE_AA, shift=4)
            stripes = np.zeros((H, W), np.uint8)
            g = style.stripe_gap * sh
            x0 = W / 2 - sw / 2
            j = 0
            while x0 + j * g < W / 2 + sw / 2:
                x = x0 + j * g
                cv2.line(stripes, (int(x * S), 0), (int(x * S), (H - 1) * S), 255, max(1, int(round(g * style.stripe_w))), cv2.LINE_AA, shift=4)
                j += 1
            m = (sym.astype(np.float32) * stripes.astype(np.float32) / 255 / 255)[..., None] * style.stripe_alpha
            img[..., :3] = (img[..., :3] * (1 - m) + np.array(ink) * m).astype(np.uint8)
        cv2.polylines(img, [p], True, (*ink, 255), int(round(lw)), cv2.LINE_AA, shift=4)
    img[..., 3] = mask
    return cv2.resize(img, (width, int(round(width / CARD_ASPECT))), interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    import sys
    rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    st = DeckStyle.random(rng)
    tiles = [render_card(i, st, 120, rng) for i in range(81)]
    h = tiles[0].shape[0]
    rows = [np.concatenate(tiles[r * 9:(r + 1) * 9], 1) for r in range(9)]
    sheet = np.concatenate(rows, 0)
    bg = np.full(sheet.shape[:2] + (3,), 60, np.uint8)
    a = sheet[..., 3:4] / 255
    out = (sheet[..., :3] * a + bg * (1 - a)).astype(np.uint8)
    cv2.imwrite(sys.argv[2] if len(sys.argv) > 2 else 'sheet.png', out[..., ::-1])
