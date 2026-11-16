"""Python mirror of the web app's detector + classifier, for offline evaluation on videos.

    python scripts/v3/pipeline.py VIDEO --every 15 --out preds.json [--models docs/models]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.synth.crops import CH, CW, crop_card  # noqa: E402
from src.synth.cards import card_name  # noqa: E402

DET, CONF, NMS = 640, 0.35, 0.5


def rot_iou(a, b):
    ra = ((a[0], a[1]), (a[2], a[3]), math.degrees(a[4]))
    rb = ((b[0], b[1]), (b[2], b[3]), math.degrees(b[4]))
    r, pts = cv2.rotatedRectangleIntersection(ra, rb)
    if r == cv2.INTERSECT_NONE or pts is None:
        return 0.0
    inter = cv2.contourArea(pts)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


class Pipeline:
    def __init__(self, models=Path('docs/models')):
        prov = ['CPUExecutionProvider']
        self.det = ort.InferenceSession(str(models / 'detector_v3.onnx'), providers=prov)
        self.cls = ort.InferenceSession(str(models / 'classifier_v3.onnx'), providers=prov)

    def detect(self, rgb, region=None):
        H, W = rgb.shape[:2]
        rx, ry, rw, rh = region or (0, 0, W, H)
        sub = rgb[int(ry):int(ry + rh), int(rx):int(rx + rw)]
        r = min(DET / rw, DET / rh); nw, nh = round(rw * r), round(rh * r)
        px, py = (DET - nw) // 2, (DET - nh) // 2
        can = np.full((DET, DET, 3), 114, np.uint8)
        can[py:py + nh, px:px + nw] = cv2.resize(sub, (nw, nh), interpolation=cv2.INTER_AREA)
        o = self.det.run(None, {self.det.get_inputs()[0].name: can.transpose(2, 0, 1)[None].astype(np.float32) / 255})[0][0]
        if o.shape[0] > o.shape[1]:
            o = o.T
        keep = o[4] >= CONF
        boxes = [((cx - px) / r + rx, (cy - py) / r + ry, w / r, h / r, a, s) for cx, cy, w, h, s, a in o[[0, 1, 2, 3, 4, -1]][:, keep].T]
        boxes.sort(key=lambda b: -b[5])
        out = []
        for b in boxes:
            if all(rot_iou(b, k) < NMS for k in out):
                out.append(b)
        return out

    def classify(self, rgb, boxes):
        if not boxes:
            return []
        x = np.stack([crop_card(rgb, b[:5]) for b in boxes]).transpose(0, 3, 1, 2).astype(np.float32) / 255
        L = self.cls.run(None, {'images': x})[0]
        res = []
        for l in L:
            p = [np.exp(l[h * 3:h * 3 + 3] - l[h * 3:h * 3 + 3].max()) for h in range(4)]
            p = [q / q.sum() for q in p]
            idx = [int(q.argmax()) for q in p]
            cid = idx[0] * 27 + idx[1] * 9 + idx[2] * 3 + idx[3]
            res.append({'id': cid, 'name': card_name(cid), 'cert': float(min(q.max() for q in p)), 'readable': float(1 / (1 + np.exp(-l[12])))})
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--every', type=int, default=15)
    ap.add_argument('--out', default='preds.json')
    ap.add_argument('--models', type=Path, default=Path('docs/models'))
    a = ap.parse_args()
    P = Pipeline(a.models)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rows, fi = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if fi % a.every == 0:
            rgb = bgr[..., ::-1]
            boxes = P.detect(rgb)
            cls = P.classify(rgb, boxes)
            rows.append({'t': round(fi / fps, 3), 'cards': [{'obb': [round(float(v), 2) for v in b[:5]], 'score': round(float(b[5]), 3), **c} for b, c in zip(boxes, cls)]})
        fi += 1
    Path(a.out).write_text(json.dumps(rows))
    print(len(rows), 'frames, mean cards', np.mean([len(r['cards']) for r in rows]))


if __name__ == '__main__':
    main()
