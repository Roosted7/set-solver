"""Synthetic table scenes for training the v3 detector and classifier.

A scene = a table texture with distractors (plates, papers, books, card backs), a layout of
face-up Set cards (grids, touching, slightly overlapping, scattered), seen through a random
pinhole camera (top-down to ~65 degrees tilt, any rotation, wide to normal lens), with hands,
shadows, glare, white balance, blur, noise, resolution loss and JPEG.

Each scene returns the image plus, per card, its id, its 4 projected corners
(card top-left, top-right, bottom-right, bottom-left) and its visible fraction.
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np

from .cards import CARD_ASPECT, DeckStyle, render_card

CARD_W = 200                       # card width on the world canvas (px)
CARD_H = int(round(CARD_W / CARD_ASPECT))


# ---------------------------------------------------------------- textures
def _noise(h, w, scale, rng):
    sm = np.random.default_rng(rng.randrange(1 << 30)).random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_CUBIC)


def _rand_color(rng, sat=(0, 1), val=(0, 1)):
    hsv = np.uint8([[[rng.randrange(180), int(255 * rng.uniform(*sat)), int(255 * rng.uniform(*val))]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32)


def texture(h, w, rng: random.Random) -> np.ndarray:
    kind = rng.choices(['wood', 'cloth', 'solid', 'stone', 'pattern'], [0.4, 0.15, 0.15, 0.15, 0.15])[0]
    if kind == 'wood':
        base = _rand_color(rng, (0.35, 0.8), (0.2, 0.75))
        if rng.random() < 0.3:
            base = _rand_color(rng, (0.1, 0.4), (0.5, 0.9))  # light wood / painted
        y = np.arange(h, dtype=np.float32)[:, None]
        warp = _noise(h, w, rng.choice([64, 128, 256]), rng) * rng.uniform(10, 60)
        freq = rng.uniform(0.04, 0.2)
        grain = 0.5 + 0.5 * np.sin((y + warp) * freq + _noise(h, w, 16, rng) * 3)
        fine = _noise(h, w, 2, rng)
        t = 0.75 + 0.2 * grain + 0.08 * fine
        img = base[None, None] * t[..., None]
        if rng.random() < 0.5:
            img = np.rot90(img).copy() if h == w else cv2.rotate(cv2.resize(img, (h, w)), cv2.ROTATE_90_CLOCKWISE)
    elif kind == 'cloth':
        base = _rand_color(rng, (0.2, 0.9), (0.15, 0.8))
        t = 0.85 + 0.15 * _noise(h, w, 2, rng) + 0.1 * _noise(h, w, 40, rng)
        img = base[None, None] * t[..., None]
    elif kind == 'solid':
        base = _rand_color(rng, (0, 0.6), (0.1, 0.95))
        img = base[None, None] * (0.92 + 0.1 * _noise(h, w, 100, rng))[..., None]
    elif kind == 'stone':
        base = _rand_color(rng, (0, 0.3), (0.3, 0.9))
        t = 0.8 + 0.25 * _noise(h, w, 8, rng) + 0.15 * _noise(h, w, 50, rng)
        img = base[None, None] * t[..., None]
    else:  # tiles / checks / stripes
        a, b = _rand_color(rng, (0, 1), (0.1, 0.9)), _rand_color(rng, (0, 1), (0.1, 0.9))
        s = rng.randrange(30, 300)
        yy, xx = np.mgrid[0:h, 0:w]
        m = (((xx // s) + (yy // s)) % 2).astype(np.float32) if rng.random() < 0.5 else ((xx // s) % 2).astype(np.float32)
        img = a * m[..., None] + b * (1 - m[..., None])
        img *= (0.9 + 0.1 * _noise(h, w, 4, rng))[..., None]
    return np.clip(img, 0, 255).astype(np.uint8) if img.shape[:2] == (h, w) else cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (w, h))


def _paste(dst, src_rgba, cx, cy, angle, scale=1.0):
    """Alpha-composite src rotated by angle (deg) centred at (cx,cy). Returns pasted corner quad and mask."""
    h, w = src_rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += [cx - w / 2, cy - h / 2]
    H, W = dst.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64) @ M.T
    x0, y0 = max(0, int(corners[:, 0].min()) - 2), max(0, int(corners[:, 1].min()) - 2)
    x1, y1 = min(W, int(corners[:, 0].max()) + 3), min(H, int(corners[:, 1].max()) + 3)
    full = np.zeros((H, W), np.uint8)
    if x1 <= x0 or y1 <= y0:
        return corners, full
    M2 = M.copy(); M2[:, 2] -= [x0, y0]
    warped = cv2.warpAffine(src_rgba, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = warped[..., 3:4].astype(np.float32) / 255
    roi = dst[y0:y1, x0:x1]
    roi[...] = (roi * (1 - a) + warped[..., :3] * a).astype(np.uint8)
    full[y0:y1, x0:x1] = warped[..., 3]
    return corners, full


def distractor(rng: random.Random, kind=None) -> np.ndarray:
    """RGBA distractor object: plate, paper sheet, book, card back stack, phone, coaster..."""
    kind = kind or rng.choice(['plate', 'paper', 'book', 'backs', 'phone', 'blob', 'cardlike'])
    if kind == 'plate':
        d = rng.randrange(300, 900)
        img = np.zeros((d, d, 4), np.uint8)
        img[..., :3] = texture(d, d, rng)
        if rng.random() < 0.6:  # busy illustration
            for _ in range(rng.randrange(20, 80)):
                c = tuple(int(v) for v in _rand_color(rng, (0.2, 1), (0.2, 1)))
                p = (rng.randrange(d), rng.randrange(d))
                cv2.ellipse(img, p, (rng.randrange(5, d // 6), rng.randrange(5, d // 8)), rng.randrange(180), 0, 360, (*c, 255), -1, cv2.LINE_AA)
        mask = np.zeros((d, d), np.uint8)
        cv2.circle(mask, (d // 2, d // 2), d // 2 - 2, 255, -1, cv2.LINE_AA)
        img[..., 3] = mask
        return img
    if kind in ('paper', 'cardlike'):
        if kind == 'paper':
            w, h = rng.randrange(300, 700), rng.randrange(400, 900)
        else:  # white rectangles roughly card sized: the hardest negatives (notes, receipts, other cards)
            w = int(CARD_W * rng.uniform(0.6, 1.4)); h = int(w * rng.uniform(0.5, 2.0))
        img = np.zeros((h, w, 4), np.uint8)
        img[..., :3] = _rand_color(rng, (0, 0.15), (0.75, 1))
        if kind == 'paper' or rng.random() < 0.7:
            if rng.random() < 0.5:  # grid (sudoku etc.)
                s = rng.randrange(15, 60)
                for x in range(rng.randrange(5, 30), w, s): cv2.line(img, (x, 0), (x, h), (40, 40, 40, 255), rng.choice([1, 2, 3]))
                for y in range(rng.randrange(5, 30), h, s): cv2.line(img, (0, y), (w, y), (40, 40, 40, 255), rng.choice([1, 2, 3]))
            else:  # text-ish lines / logos
                for y in range(rng.randrange(10, 40), h - 10, rng.randrange(12, 30)):
                    x0 = rng.randrange(5, 40); cv2.line(img, (x0, y), (rng.randrange(x0 + 10, w), y), tuple(int(v) for v in _rand_color(rng, (0, 1), (0, 0.6))) + (255,), rng.randrange(2, 7))
        img[..., 3] = 255
        return img
    if kind == 'book':
        w, h = rng.randrange(300, 800), rng.randrange(400, 1000)
        img = np.zeros((h, w, 4), np.uint8)
        img[..., :3] = texture(h, w, rng)
        cv2.rectangle(img, (w // 8, h // 6), (w - w // 8, h // 3), tuple(int(v) for v in _rand_color(rng)) + (255,), -1)
        img[..., 3] = 255
        return img
    if kind == 'backs':  # face-down Set cards / deck box: should NOT be detected
        w, h = CARD_W, CARD_H
        img = np.zeros((h, w, 4), np.uint8)
        col = _rand_color(rng, (0.4, 0.9), (0.25, 0.7))
        img[..., :3] = col
        for _ in range(rng.randrange(3, 12)):
            c = tuple(int(v) for v in _rand_color(rng, (0, 1), (0.3, 1)))
            cv2.putText(img, rng.choice(['SET', 'S', 'E', 'T', '*']), (rng.randrange(0, w - 40), rng.randrange(30, h)), cv2.FONT_HERSHEY_DUPLEX, rng.uniform(0.8, 2.5), (*c, 255), rng.randrange(2, 6))
        r = int(w * 0.08)
        mask = np.zeros((h, w), np.uint8)
        cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1); cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
        for cx, cy in [(r, r), (w - r, r), (r, h - r), (w - r, h - r)]: cv2.circle(mask, (cx, cy), r, 255, -1)
        img[..., 3] = mask
        if rng.random() < 0.6:  # a stack: thick edge
            pad = rng.randrange(4, 20)
            big = np.zeros((h + pad, w + pad, 4), np.uint8)
            big[pad:, pad:] = img; big[pad:, pad:, :3] = (240, 240, 235)
            big[:h, :w] = np.where(img[..., 3:4] > 0, img, big[:h, :w])
            img = big
        return img
    if kind == 'phone':
        w, h = rng.randrange(150, 260), rng.randrange(300, 520)
        img = np.zeros((h, w, 4), np.uint8); img[..., :3] = _rand_color(rng, (0, 0.3), (0, 0.3)); img[..., 3] = 255
        cv2.rectangle(img, (8, 8), (w - 8, h - 8), tuple(int(v) for v in _rand_color(rng)) + (255,), -1)
        return img
    d = rng.randrange(80, 400)
    img = np.zeros((d, d, 4), np.uint8); img[..., :3] = _rand_color(rng)
    mask = np.zeros((d, d), np.uint8)
    pts = np.array([[rng.randrange(d), rng.randrange(d)] for _ in range(rng.randrange(3, 8))], np.int32)
    cv2.fillPoly(mask, [cv2.convexHull(pts)], 255, cv2.LINE_AA)
    img[..., 3] = mask
    return img


# ---------------------------------------------------------------- layout
def layout(n, rng: random.Random):
    """Card centres + rotations on the world plane, in card-widths. Mostly grids like real play."""
    mode = rng.choices(['grid', 'tight', 'scatter', 'rows'], [0.55, 0.2, 0.1, 0.15])[0]
    landscape = rng.random() < 0.15
    cw, ch = (CARD_H, CARD_W) if landscape else (CARD_W, CARD_H)
    base_rot = 90 if landscape else 0
    out = []
    if mode in ('grid', 'tight', 'rows'):
        rows = rng.choice([3, 3, 3, 4, 2]) if n > 4 else 1
        cols = math.ceil(n / rows)
        gx = cw * (rng.uniform(-0.04, 0.08) if mode == 'tight' else rng.uniform(0.08, 0.6))
        gy = ch * (rng.uniform(-0.04, 0.08) if mode == 'tight' else rng.uniform(0.05, 0.5))
        jit = rng.uniform(0, 0.12) * cw
        rj = rng.uniform(0, 10)
        slots = [(r, c) for r in range(rows) for c in range(cols)]
        rng.shuffle(slots)
        for r, c in sorted(slots[:n]):
            x = c * (cw + gx) + rng.gauss(0, jit) if mode != 'rows' else c * (cw + gx) + rng.uniform(-0.2, 0.2) * cw
            y = r * (ch + gy) + rng.gauss(0, jit)
            out.append((x, y, base_rot + rng.gauss(0, rj) + (180 if rng.random() < 0.5 else 0)))
    else:
        for _ in range(n):
            for _try in range(30):
                x, y = rng.uniform(0, 5 * cw), rng.uniform(0, 4 * ch)
                if all(math.hypot(x - a, y - b) > 0.8 * cw for a, b, _ in out): break
            out.append((x, y, rng.uniform(0, 360)))
    xs, ys = [p[0] for p in out], [p[1] for p in out]
    mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return [(x - mx, y - my, a) for x, y, a in out]


# ---------------------------------------------------------------- camera
def camera_homography(world_center, out_w, out_h, card_px, rng: random.Random):
    """Homography from world canvas to image. card_px = desired card width in image at the centre."""
    tilt = math.radians(rng.choices([rng.uniform(0, 20), rng.uniform(15, 45), rng.uniform(40, 68)], [0.35, 0.45, 0.2])[0])
    yaw = rng.uniform(0, 2 * math.pi)
    roll = math.radians(rng.gauss(0, 6))
    fov = math.radians(rng.uniform(55, 100))
    f = (max(out_w, out_h) / 2) / math.tan(fov / 2)
    # distance so a card at the centre spans card_px
    dist = f * CARD_W / card_px
    # camera looks at the world point world_center (+ small offset), from direction given by tilt/yaw
    tx, ty = world_center
    tx += rng.uniform(-0.6, 0.6) * CARD_W * 3; ty += rng.uniform(-0.5, 0.5) * CARD_H * 2
    C = np.array([tx + dist * math.sin(tilt) * math.cos(yaw), ty + dist * math.sin(tilt) * math.sin(yaw), -dist * math.cos(tilt)])
    target = np.array([tx, ty, 0.0])
    z = target - C; z /= np.linalg.norm(z)
    up = np.array([0, 0, -1.0]) if tilt > 0.05 else np.array([-math.sin(yaw), math.cos(yaw), 0])
    x = np.cross(z, up); n = np.linalg.norm(x)
    if n < 1e-6: x = np.array([1.0, 0, 0])
    else: x /= n
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # world -> cam rows
    cr, sr = math.cos(roll), math.sin(roll)
    R = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @ R
    t = -R @ C
    K = np.array([[f, 0, out_w / 2 + rng.uniform(-0.1, 0.1) * out_w], [0, f, out_h / 2 + rng.uniform(-0.1, 0.1) * out_h], [0, 0, 1]])
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return Hm / Hm[2, 2]


def project(Hm, pts):
    p = np.column_stack([pts, np.ones(len(pts))]) @ Hm.T
    return p[:, :2] / p[:, 2:3], p[:, 2]


# ---------------------------------------------------------------- hands / photometric
def draw_hand(img, rng: random.Random, scale):
    """Crude arm + hand: skin-coloured polygons with shading, entering from an image edge."""
    H, W = img.shape[:2]
    skin = np.array(_skin(rng), np.float32)
    mask = np.zeros((H, W), np.float32)
    edge = rng.randrange(4)
    tip = np.array([rng.uniform(0.15, 0.85) * W, rng.uniform(0.15, 0.85) * H])
    start = [np.array([rng.uniform(0, W), -50]), np.array([rng.uniform(0, W), H + 50]), np.array([-50, rng.uniform(0, H)]), np.array([W + 50, rng.uniform(0, H)])][edge]
    d = tip - start; L = np.linalg.norm(d) + 1e-6; u = d / L; nrm = np.array([-u[1], u[0]])
    aw = scale * rng.uniform(0.9, 1.6)
    arm = np.array([start + nrm * aw * 0.6, tip - u * aw * 0.6 + nrm * aw * 0.55, tip - u * aw * 0.6 - nrm * aw * 0.55, start - nrm * aw * 0.6])
    cv2.fillPoly(mask, [arm.astype(np.int32)], 1.0, cv2.LINE_AA)
    cv2.ellipse(mask, tuple(int(v) for v in (tip - u * aw * 0.3)), (int(aw * 0.75), int(aw * 0.6)), math.degrees(math.atan2(u[1], u[0])), 0, 360, 1.0, -1, cv2.LINE_AA)
    for k in range(rng.randrange(0, 5)):  # fingers
        a = math.atan2(u[1], u[0]) + rng.uniform(-0.9, 0.9)
        fl = aw * rng.uniform(0.6, 1.3)
        p0 = tip - u * aw * 0.2 + nrm * aw * rng.uniform(-0.4, 0.4)
        p1 = p0 + fl * np.array([math.cos(a), math.sin(a)])
        cv2.line(mask, tuple(int(v) for v in p0), tuple(int(v) for v in p1), 1.0, max(2, int(aw * 0.22)), cv2.LINE_AA)
    if rng.random() < 0.3:  # holding a card (dealing)
        pass
    shade = 0.75 + 0.35 * _noise(H, W, 30, rng)
    m = cv2.GaussianBlur(mask, (0, 0), 1.2)[..., None]
    img[...] = np.clip(img * (1 - m) + (skin * shade[..., None]) * m, 0, 255).astype(np.uint8)
    # soft shadow next to the hand
    sh = cv2.GaussianBlur(np.roll(mask, (int(aw * 0.3), int(aw * 0.3)), (0, 1)), (0, 0), aw * 0.4)
    img[...] = (img * (1 - 0.35 * np.clip(sh - mask, 0, 1)[..., None])).astype(np.uint8)
    return mask


def _skin(rng):
    return tuple(int(v) for v in rng.choice([(224, 172, 140), (198, 134, 100), (160, 105, 75), (110, 70, 50), (235, 190, 160)])) if rng.random() < 0.85 else tuple(int(v) for v in _rand_color(rng))


def photometric(img, rng: random.Random):
    H, W = img.shape[:2]
    f = img.astype(np.float32)
    # lighting gradient / vignetting
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    g = 1 + rng.uniform(-0.35, 0.35) * ((xx / W - 0.5) * math.cos(rng.uniform(0, 6.3)) + (yy / H - 0.5) * math.sin(rng.uniform(0, 6.3)))
    r2 = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W * W / 4 + H * H / 4)
    g *= 1 - rng.uniform(0, 0.4) * r2
    f *= g[..., None]
    # soft shadows
    for _ in range(rng.choices([0, 1, 2], [0.5, 0.35, 0.15])[0]):
        m = np.zeros((H, W), np.float32)
        pts = np.array([[rng.uniform(-0.2, 1.2) * W, rng.uniform(-0.2, 1.2) * H] for _ in range(rng.randrange(3, 7))], np.int32)
        cv2.fillPoly(m, [cv2.convexHull(pts)], 1.0)
        m = cv2.GaussianBlur(m, (0, 0), rng.uniform(5, 60))
        f *= (1 - rng.uniform(0.2, 0.55) * m)[..., None]
    # glare
    for _ in range(rng.choices([0, 1, 2], [0.6, 0.3, 0.1])[0]):
        m = np.zeros((H, W), np.float32)
        cv2.ellipse(m, (rng.randrange(W), rng.randrange(H)), (rng.randrange(10, W // 4), rng.randrange(5, H // 6)), rng.randrange(180), 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), rng.uniform(5, 40))
        f += (m * rng.uniform(60, 220))[..., None]
    # white balance: warm (tungsten) to cool, plus exposure and gamma
    temp = rng.choices([rng.uniform(-1, -0.3), rng.uniform(-0.3, 0.3), rng.uniform(0.3, 1)], [0.35, 0.45, 0.2])[0]  # -1 warm, +1 cool
    wb = np.array([1 - 0.35 * temp, 1 - 0.05 * abs(temp), 1 + 0.45 * temp]) if temp > 0 else np.array([1 + 0.25 * -temp, 1 + 0.02 * -temp, 1 - 0.5 * -temp])
    wb *= rng.uniform(0.92, 1.08, ) if False else np.array([rng.uniform(0.95, 1.05) for _ in range(3)])
    f *= wb * rng.uniform(0.6, 1.35)
    f = np.clip(f, 0, 255)
    f = 255 * (f / 255) ** rng.uniform(0.75, 1.35)
    # saturation
    gray = f.mean(2, keepdims=True)
    f = gray + (f - gray) * rng.uniform(0.6, 1.4)
    img = np.clip(f, 0, 255).astype(np.uint8)
    # blur: defocus or motion
    r = rng.random()
    if r < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 2.2))
    elif r < 0.5:
        k = rng.randrange(3, 13); ker = np.zeros((k, k), np.float32); ker[k // 2] = 1
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        ker = cv2.warpAffine(ker, M, (k, k)); ker /= ker.sum()
        img = cv2.filter2D(img, -1, ker)
    # resolution loss (video / digital zoom / messaging app)
    if rng.random() < 0.6:
        s = rng.uniform(0.35, 0.9)
        small = cv2.resize(img, (max(16, int(W * s)), max(16, int(H * s))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (W, H), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    # sensor noise
    if rng.random() < 0.7:
        img = np.clip(img.astype(np.float32) + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(1, 9), img.shape), 0, 255).astype(np.uint8)
    # jpeg
    if rng.random() < 0.8:
        ok, buf = cv2.imencode('.jpg', img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, rng.randrange(25, 95)])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return img


# ---------------------------------------------------------------- scene
def make_scene(rng: random.Random, out_size=None):
    """Returns (rgb image, cards) where cards = [{id, quad(4x2 image px), vis}]."""
    if out_size is None:
        out_size = rng.choice([(960, 540), (960, 720), (720, 960), (540, 960), (800, 800), (1024, 576)])
    out_w, out_h = out_size
    n = rng.choices([12, rng.randrange(3, 12), rng.randrange(13, 19), rng.randrange(1, 4)], [0.45, 0.3, 0.2, 0.05])[0]
    pos = layout(n, rng)
    span_x = max(abs(p[0]) for p in pos) * 2 + CARD_H * 2
    span_y = max(abs(p[1]) for p in pos) * 2 + CARD_H * 2
    WW = int(max(span_x, span_y) * 1.5) + 600
    WW = min(WW, 3600)
    world = cv2.resize(texture(WW // 2, WW // 2, rng), (WW, WW), interpolation=cv2.INTER_LINEAR)
    occ = np.zeros((WW, WW), np.uint8)  # what covers what, on the table plane (card index + 1)
    cx0 = cy0 = WW / 2
    # distractors around the layout (some may be partly under the cards' area edge)
    for _ in range(rng.choices([0, 1, 2, 3, 5], [0.15, 0.25, 0.25, 0.2, 0.15])[0]):
        d = distractor(rng)
        ang = rng.uniform(0, 360)
        for _try in range(10):
            x, y = rng.uniform(0, WW), rng.uniform(0, WW)
            if abs(x - cx0) > span_x / 2 + d.shape[1] / 3 or abs(y - cy0) > span_y / 2 + d.shape[0] / 3: break
        _paste(world, d, x, y, ang)
    table = world.copy()  # without cards: used to extend the table beyond the canvas
    # cards
    style = DeckStyle.random(rng)
    ids = rng.sample(range(81), n)
    cards = []
    order = list(range(n))
    if rng.random() < 0.15:  # a pile: random paint order -> overlaps resolved by order
        rng.shuffle(order)
    for k in order:
        x, y, a = pos[k]
        img = render_card(ids[k], style, CARD_W, rng)
        # subtle drop shadow
        sh = np.zeros_like(img); sh[..., 3] = (img[..., 3] * rng.uniform(0.15, 0.4)).astype(np.uint8)
        sh = cv2.GaussianBlur(sh, (0, 0), 4)
        _paste(world, sh, cx0 + x + 4, cy0 + y + 5, -a)
        quad, m = _paste(world, img, cx0 + x, cy0 + y, -a)
        occ[m > 128] = k + 1
        cards.append({'id': ids[k], 'wquad': quad, 'k': k})
    # camera
    card_px = rng.choices([rng.uniform(25, 50), rng.uniform(50, 110), rng.uniform(110, 220)], [0.25, 0.6, 0.15])[0] * out_w / 960
    Hm = camera_homography((cx0, cy0), out_w, out_h, card_px, rng)
    bg = cv2.resize(texture(out_h // 2, out_w // 2, rng), (out_w, out_h))
    # the table usually fills the view (mirror the texture); sometimes its edge is visible
    edge = rng.random() < 0.25
    img = cv2.warpPerspective(world, Hm, (out_w, out_h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full((WW, WW), 255, np.uint8), Hm, (out_w, out_h), flags=cv2.INTER_NEAREST)
    outside = bg if edge else cv2.warpPerspective(table, Hm, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    img = np.where(valid[..., None] > 0, img, outside)
    occ_img = cv2.warpPerspective(occ, Hm, (out_w, out_h), flags=cv2.INTER_NEAREST)
    # hands
    hand = np.zeros((out_h, out_w), np.float32)
    if rng.random() < 0.3:
        for _ in range(rng.choice([1, 1, 2])):
            hand = np.maximum(hand, draw_hand(img, rng, card_px * rng.uniform(0.6, 1.1)))
    img = photometric(img, rng)
    out = []
    for c in cards:
        q, z = project(Hm, c['wquad'])
        if (z <= 0).any():
            continue
        full = abs(cv2.contourArea(q.astype(np.float32)))
        if full < 30:
            continue
        own = (occ_img == c['k'] + 1) & (hand < 0.5)
        vis = float(own.sum()) / full
        out.append({'id': c['id'], 'quad': q.tolist(), 'vis': min(1.0, vis), 'area': full})
    return img, out


if __name__ == '__main__':
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    for i in range(int(sys.argv[2]) if len(sys.argv) > 2 else 6):
        rng = random.Random(seed * 1000 + i)
        img, cards = make_scene(rng)
        vis = img.copy()
        for c in cards:
            col = (0, 255, 0) if c['vis'] > 0.6 else (255, 0, 0)
            cv2.polylines(vis, [np.int32(c['quad'])], True, col, 1)
        cv2.imwrite(f'{sys.argv[3] if len(sys.argv) > 3 else "."}/scene_{i}.jpg', vis[..., ::-1])
