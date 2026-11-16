"""Classifier accuracy on labelled real video frames (truth.json from label_video.py export).

    python scripts/v3/eval_real.py data/videos/game_hq.mp4 data/real/game_hq --from 36 \
        --model docs/models/classifier_v3.onnx [--old docs/models/classifier.onnx] [--train-until 36]

Reports per-attribute and whole-card accuracy per frame, separately for cards that also occur in the
training part of the video (t < --train-until) and for cards that never do.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.synth.crops import crop_card  # noqa: E402

ATTR = ['number', 'color', 'shape', 'fill']
attrs = lambda c: (c // 27, (c // 9) % 3, (c // 3) % 3, c % 3)


def run_v3(sess, rgb, obbs):
    x = np.stack([crop_card(rgb, o) for o in obbs]).transpose(0, 3, 1, 2).astype(np.float32) / 255
    L = sess.run(None, {'images': x})[0]
    return [tuple(int(l[h * 3:h * 3 + 3].argmax()) for h in range(4)) for l in L], [float(1 / (1 + np.exp(-l[12]))) for l in L]


def run_old(sess, rgb, obbs):
    """v1/v2 classifier: axis-aligned box crop squashed to 224x224."""
    xs = []
    for cx, cy, w, h, a in obbs:
        pts = cv2.boxPoints(((cx, cy), (w, h), np.degrees(a)))
        x1, y1 = np.maximum(0, pts.min(0)).astype(int); x2, y2 = pts.max(0).astype(int)
        xs.append(cv2.resize(rgb[y1:y2, x1:x2], (224, 224)).transpose(2, 0, 1).astype(np.float32) / 255)
    L = sess.run(None, {'images': np.stack(xs)})[0]
    # old heads: number | color | shape | fill, same index order as v3
    return [tuple(int(l[h * 3:h * 3 + 3].argmax()) for h in range(4)) for l in L], [1.0] * len(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video'); ap.add_argument('labeldir', type=Path)
    ap.add_argument('--model', default='docs/models/classifier_v3.onnx')
    ap.add_argument('--old', default=None)
    ap.add_argument('--from', dest='start', type=float, default=36)
    ap.add_argument('--train-until', type=float, default=36)
    a = ap.parse_args()
    truth = json.loads((a.labeldir / 'truth.json').read_text())
    seen_ids = {c['id'] for t, cs in truth.items() if float(t) < a.train_until for c in cs}
    models = {'v3': (ort.InferenceSession(a.model, providers=['CPUExecutionProvider']), run_v3)}
    if a.old:
        models['old'] = (ort.InferenceSession(a.old, providers=['CPUExecutionProvider']), run_old)
    want = {round(float(t) * 1000): cs for t, cs in truth.items() if float(t) >= a.start}
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS)
    stats = {m: defaultdict(lambda: np.zeros(6)) for m in models}  # group -> [n, num, col, shp, fill, all]
    errors = defaultdict(lambda: defaultdict(int))
    fi = 0
    while want:
        ok, bgr = cap.read()
        if not ok: break
        key = round(fi / fps * 1000); fi += 1
        hit = next((k for k in (key - 1, key, key + 1) if k in want), None)
        if hit is None: continue
        cs = want.pop(hit)
        rgb = bgr[..., ::-1]
        for m, (sess, fn) in models.items():
            preds, readable = fn(sess, rgb, [c['obb'] for c in cs])
            for c, p in zip(cs, preds):
                g = 'cards also in training part' if c['id'] in seen_ids else 'cards NOT in training part'
                t = attrs(c['id'])
                ok4 = [p[h] == t[h] for h in range(4)]
                for grp in (g, 'all'):
                    stats[m][grp] += [1, *ok4, all(ok4)]
                if not all(ok4):
                    errors[m][(c['id'], p)] += 1
    for m in models:
        print(f'== {m} ({a.model if m == "v3" else a.old})')
        for g, s in stats[m].items():
            n = s[0]
            print(f'  {g:30s} n={int(n):5d}  ' + '  '.join(f'{k} {v / n:.3f}' for k, v in zip(ATTR + ['card'], s[1:])))
        top = sorted(errors[m].items(), key=lambda kv: -kv[1])[:8]
        names = lambda c: ' '.join(str(v) for v in c)
        for (cid, p), k in top:
            print(f'    {k:4d}x truth {attrs(cid)} -> pred {p}')


if __name__ == '__main__':
    main()
