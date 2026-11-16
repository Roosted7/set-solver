"""Card crop geometry shared by training and the web app (docs/app.js: cropCard()).

A card is described by a rotated box (cx, cy, w, h, angle) as produced by the OBB detector.
The crop puts the card's long side vertical and scales it into CW x CH pixels with a small
margin. The 180-degree ambiguity is left to the classifier (trained with both orientations).
"""
import math

import cv2
import numpy as np

CW, CH = 96, 144       # classifier input (width, height)
MARGIN = 1.06          # crop is this much larger than the card box


def quad_to_obb(quad):
    (cx, cy), (w, h), ang = cv2.minAreaRect(np.float32(quad))
    return cx, cy, w, h, math.radians(ang)


def crop_affine(cx, cy, w, h, ang, margin=MARGIN):
    """2x3 affine mapping the crop (CW x CH) to image coordinates... inverted for warpAffine."""
    ux, uy = math.cos(ang), math.sin(ang)        # direction of side w
    vx, vy = -uy, ux                             # direction of side h
    if w > h:                                    # make "down" the long side
        w, h = h, w
        ux, uy, vx, vy = vx, vy, -ux, -uy
    sw, sh = w * margin, h * margin
    # crop (0,0) -> top-left, (CW,0) -> top-right, (0,CH) -> bottom-left
    tl = (cx - ux * sw / 2 - vx * sh / 2, cy - uy * sw / 2 - vy * sh / 2)
    tr = (tl[0] + ux * sw, tl[1] + uy * sw)
    bl = (tl[0] + vx * sh, tl[1] + vy * sh)
    src = np.float32([tl, tr, bl])
    dst = np.float32([[0, 0], [CW, 0], [0, CH]])
    return cv2.getAffineTransform(src, dst), max(sw / CW, sh / CH)


def crop_card(img, obb, margin=MARGIN, area=True):
    """img: HxWx3 uint8. Returns CH x CW x 3 crop. area=True pre-shrinks large cards (like the browser does)."""
    cx, cy, w, h, ang = obb
    M, scale = crop_affine(cx, cy, w, h, ang, margin)
    if area and scale > 1.5:
        f = 1 / scale * 1.2
        small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        M2, _ = crop_affine(cx * f, cy * f, w * f, h * f, ang, margin)
        return cv2.warpAffine(small, M2, (CW, CH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.warpAffine(img, M, (CW, CH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
