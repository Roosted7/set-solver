"""Generate the v3 synthetic dataset.

    python scripts/v3/gen_dataset.py --out data/synth_v3 --train 12000 --val 600

Writes
  <out>/images/{train,val}/NNNNNN.jpg
  <out>/labels/{train,val}/NNNNNN.txt   YOLO-OBB: "0 x1 y1 x2 y2 x3 y3 x4 y4" (normalised)
  <out>/meta/{train,val}/NNNNNN.json    per card: id, quad (px), vis   (used by the classifier)
  <out>/data.yaml
"""
import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import cv2
import numpy as np

cv2.setNumThreads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.synth.boards import make_scene  # noqa: E402

MIN_VIS = 0.45      # label a card for the detector when at least this much of it is visible
MIN_SIDE = 8        # px; smaller cards are not labelled


def work(args):
    out, split, i, seed = args
    if (out / 'meta' / split / f'{i:06d}.json').exists():
        return 0
    rng = random.Random(seed)
    img, cards = make_scene(rng)
    h, w = img.shape[:2]
    lines = []
    for c in cards:
        q = np.array(c['quad'])
        side = min(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[1] - q[2]))
        inside = ((q[:, 0] > -0.05 * w) & (q[:, 0] < 1.05 * w) & (q[:, 1] > -0.05 * h) & (q[:, 1] < 1.05 * h)).mean()
        c['label'] = bool(c['vis'] >= MIN_VIS and side >= MIN_SIDE)
        if c['label']:
            qn = np.clip(q / [w, h], 0, 1)
            lines.append('0 ' + ' '.join(f'{v:.5f}' for v in qn.reshape(-1)))
        c['inside'] = float(inside)
    name = f'{i:06d}'
    cv2.imwrite(str(out / 'images' / split / f'{name}.jpg'), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / 'labels' / split / f'{name}.txt').write_text('\n'.join(lines) + ('\n' if lines else ''))
    (out / 'meta' / split / f'{name}.json').write_text(json.dumps({'w': w, 'h': h, 'cards': cards}))
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / 'data/synth_v3')
    ap.add_argument('--train', type=int, default=12000)
    ap.add_argument('--val', type=int, default=600)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    jobs = []
    for split, n, off in (('train', a.train, 0), ('val', a.val, 10_000_000)):
        for d in ('images', 'labels', 'meta'):
            (a.out / d / split).mkdir(parents=True, exist_ok=True)
        jobs += [(a.out, split, i, a.seed * 100_000_000 + off + i) for i in range(n)]
    with Pool(a.workers) as p:
        tot = 0
        for k, n in enumerate(p.imap_unordered(work, jobs, chunksize=8)):
            tot += n
            if k % 500 == 0:
                print(f'{k}/{len(jobs)} scenes, {tot} cards', flush=True)
    (a.out / 'data.yaml').write_text(f'path: {a.out}\ntrain: images/train\nval: images/val\nnames:\n  0: card\n')
    print('done', tot, 'cards')


if __name__ == '__main__':
    main()
