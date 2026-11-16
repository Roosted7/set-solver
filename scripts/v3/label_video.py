"""Turn a game video into labelled real data, with one label per physical card instead of per frame.

1) python scripts/v3/pipeline.py VIDEO --every 6 --out preds.json        (detector + classifier per frame)
2) python scripts/v3/label_video.py tracks VIDEO preds.json OUTDIR
      links detections over time into card tracks; writes OUTDIR/tracks.json and a contact sheet
      OUTDIR/sheet_*.jpg with one picture per track and the classifier's guess
3) write OUTDIR/labels.json: {"<track id>": "1 open green diamond" | "x" (not a card / skip), ...}
   A card swapped in the same spot (one track, two cards): [[from_s, to_s, "name"], [from_s, to_s, "name"]]
   (tracks you don't list keep the classifier's guess only if you pass --trust-guess)
4) python scripts/v3/label_video.py export VIDEO OUTDIR [--until SECONDS] [--from SECONDS]
      writes OUTDIR/crops/<card id>/*.jpg (classifier training/test crops, same geometry as the app)
      and OUTDIR/truth.json: per sampled frame, the visible cards with their true ids (for evaluation)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.synth.cards import COLOR, FILL, NUMBER, SHAPE, card_name  # noqa: E402
from src.synth.crops import crop_card  # noqa: E402


def name_to_id(s):
    n, f, c, sh = s.split()
    return NUMBER.index(n) * 27 + COLOR.index(c) * 9 + SHAPE.index(sh) * 3 + FILL.index(f)


def build_tracks(rows, max_gap=1.0):
    tracks = []  # {id, dets:[(t, obb, guess)]}
    nid = 0
    for r in rows:
        t = r['t']
        cards = [c for c in r['cards'] if c['readable'] > 0.3]
        live = [tr for tr in tracks if t - tr['dets'][-1][0] <= max_gap]
        pairs = []
        for i, c in enumerate(cards):
            for tr in live:
                o = tr['dets'][-1][1]
                d = np.hypot(c['obb'][0] - o[0], c['obb'][1] - o[1]) / min(o[2], o[3])
                if d < 0.45:
                    pairs.append((d, i, tr['id']))
        pairs.sort()
        used_c, used_t = set(), set()
        for d, i, tid in pairs:
            if i in used_c or tid in used_t:
                continue
            used_c.add(i); used_t.add(tid)
            next(tr for tr in tracks if tr['id'] == tid)['dets'].append((t, cards[i]['obb'], cards[i]['name']))
        for i, c in enumerate(cards):
            if i not in used_c:
                tracks.append({'id': nid, 'dets': [(t, c['obb'], c['name'])]}); nid += 1
    return tracks


def frame_at(cap, fps, t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, im = cap.read()
    return im[..., ::-1] if ok else None


def cmd_tracks(a):
    rows = json.loads(Path(a.preds).read_text())
    tracks = [tr for tr in build_tracks(rows) if len(tr['dets']) >= a.min_len]
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    (out / 'tracks.json').write_text(json.dumps(tracks))
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS)
    tiles = []
    for tr in tracks:
        t, obb, _ = tr['dets'][len(tr['dets']) // 2]
        im = frame_at(cap, fps, t)
        crop = cv2.resize(crop_card(im, obb, margin=1.02), (144, 216))[..., ::-1].copy()
        guesses = [g for _, _, g in tr['dets']]
        g = max(set(guesses), key=guesses.count)
        cv2.rectangle(crop, (0, 0), (144, 34), (0, 0, 0), -1)
        cv2.putText(crop, f"#{tr['id']} {tr['dets'][0][0]:.0f}-{tr['dets'][-1][0]:.0f}s", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(crop, g, (3, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
        tiles.append(crop)
    per = 40
    for s in range(0, len(tiles), per):
        chunk = tiles[s:s + per]
        while len(chunk) % 10: chunk.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / f'sheet_{s // per}.jpg'), np.vstack([np.hstack(chunk[i:i + 10]) for i in range(0, len(chunk), 10)]))
    print(len(tracks), 'tracks')


def cmd_export(a):
    out = Path(a.outdir)
    tracks = json.loads((out / 'tracks.json').read_text())
    labels = json.loads((out / 'labels.json').read_text())
    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS)
    truth = {}
    crops_dir = out / 'crops'
    n = 0
    for tr in tracks:
        lab = labels.get(str(tr['id']))
        if lab is None or lab == 'x':
            continue
        for k, (t, obb, _) in enumerate(tr['dets']):
            name = lab if isinstance(lab, str) else next((n for t0, t1, n in lab if t0 <= t < t1), 'x')
            if name == 'x':
                continue
            cid = name_to_id(name)
            truth.setdefault(f'{t:.3f}', []).append({'id': cid, 'obb': obb, 'track': tr['id']})
            if not a.truth_only and a.start <= t < a.until and k % a.crop_every == 0:
                im = frame_at(cap, fps, t)
                d = crops_dir / str(cid); d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"tr{tr['id']}_{t:.2f}.jpg"), crop_card(im, obb)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                n += 1
    (out / 'truth.json').write_text(json.dumps(truth))
    print(n, 'crops,', len(truth), 'frames with truth')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('tracks'); p.add_argument('video'); p.add_argument('preds'); p.add_argument('outdir'); p.add_argument('--min-len', type=int, default=4)
    p = sp.add_parser('export'); p.add_argument('video'); p.add_argument('outdir')
    p.add_argument('--from', dest='start', type=float, default=0); p.add_argument('--until', type=float, default=1e9); p.add_argument('--crop-every', type=int, default=2)
    p.add_argument('--truth-only', action='store_true')
    a = ap.parse_args()
    {'tracks': cmd_tracks, 'export': cmd_export}[a.cmd](a)
