"""Train the v3 card classifier: MobileNetV4-conv-small, 96x144 crops, 4 attribute heads + "readable card" head.

    python scripts/v3/train_classifier.py --data data/synth_v3 --epochs 12

Crops are cut on the fly from the synthetic scenes with the same rotated-box geometry as the
web app (src/synth/crops.py), with random box jitter so detector imprecision is covered.
Negatives (table, clutter, hands, two half cards, mostly covered cards) teach the extra head
to reject anything that is not a fully visible card.
Optional real crops: --real DIR with DIR/<card id>/*.jpg (already cropped with crop_card).
"""
import argparse
import os
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.synth.crops import CH, CW, crop_card, quad_to_obb  # noqa: E402

cv2.setNumThreads(1)
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('HF_HOME', str(ROOT / 'data/cache/huggingface'))  # pretrained weights stay in the project
READ_VIS, NEG_VIS, MIN_SIDE = 0.9, 0.6, 14


def jitter_obb(obb, rng, s=1.0):
    cx, cy, w, h, a = obb
    m = max(w, h)
    return (cx + rng.gauss(0, 0.03 * s) * m, cy + rng.gauss(0, 0.03 * s) * m,
            w * math.exp(rng.gauss(0, 0.05 * s)), h * math.exp(rng.gauss(0, 0.05 * s)), a + rng.gauss(0, 0.05 * s))


class Scenes(Dataset):
    def __init__(self, root: Path, split: str, neg_frac=0.3, real=None, train=True):
        self.root, self.split, self.train = root, split, train
        self.items = []
        for f in sorted((root / 'meta' / split).glob('*.json')):
            m = json.loads(f.read_text())
            name = f.stem
            for k, c in enumerate(m['cards']):
                q = np.array(c['quad'])
                side = min(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[1] - q[2]))
                if c.get('inside', 1) < 1 or side < MIN_SIDE:
                    continue
                if c['vis'] >= READ_VIS:
                    self.items.append(('card', name, k))
                elif c['vis'] < NEG_VIS:
                    self.items.append(('hidden', name, k))
        n_neg = int(len(self.items) * neg_frac)
        names = sorted({it[1] for it in self.items})
        self.items += [('neg', random.Random(i).choice(names), -1) for i in range(n_neg)]
        self.real = []
        if real:
            for d in Path(real).iterdir():
                if d.is_dir() and d.name.isdigit():
                    self.real += [(p, int(d.name)) for p in d.glob('*.jpg')]
            self.items += [('real', i, -1) for i in range(len(self.real))] * (3 if train else 1)
        self.cache = {}
        print(split, {k: sum(1 for i in self.items if i[0] == k) for k in ('card', 'hidden', 'neg', 'real')})

    def __len__(self):
        return len(self.items)

    def _scene(self, name):
        img = cv2.imread(str(self.root / 'images' / self.split / f'{name}.jpg'))[..., ::-1]
        meta = json.loads((self.root / 'meta' / self.split / f'{name}.json').read_text())
        return img, meta

    def __getitem__(self, i, scene=None):
        kind, name, k = self.items[i]
        rng = random.Random() if self.train else random.Random(i)
        attrs, valid = [-100] * 4, 0.0
        if kind == 'real':
            p, cid = self.real[name]
            crop = cv2.imread(str(p))[..., ::-1]
            crop = cv2.resize(crop, (CW, CH), interpolation=cv2.INTER_AREA)
            attrs, valid = [cid // 27, (cid // 9) % 3, (cid // 3) % 3, cid % 3], 1.0
        else:
            img, meta = scene or self._scene(name)
            cards = meta['cards']
            if kind in ('card', 'hidden'):
                c = cards[k]
                obb = jitter_obb(quad_to_obb(c['quad']), rng, 1.0 if self.train else 0.5)
                cid = c['id']
                if kind == 'card':
                    attrs, valid = [cid // 27, (cid // 9) % 3, (cid // 3) % 3, cid % 3], 1.0
            else:
                obb = self._negative(cards, img.shape, rng)
            crop = crop_card(img, obb, area=rng.random() < 0.7)
        if self.train:
            crop = self._augment(crop, rng)
        elif rng.random() < 0.5:
            crop = crop[::-1, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float() / 255
        return x, torch.tensor(attrs), torch.tensor(valid)

    def _negative(self, cards, shape, rng):
        H, W = shape[:2]
        sizes = [quad_to_obb(c['quad']) for c in cards] or [(0, 0, 60, 90, 0)]
        ref = rng.choice(sizes)
        r = rng.random()
        if r < 0.35 and len(cards) >= 2:  # straddling two neighbouring cards
            a = rng.choice(cards)
            ca = np.mean(a['quad'], 0)
            others = sorted(cards, key=lambda c: np.linalg.norm(np.mean(c['quad'], 0) - ca))[1:3]
            if others:
                cb = np.mean(rng.choice(others)['quad'], 0)
                t = rng.uniform(0.35, 0.65)
                cx, cy = ca * (1 - t) + cb * t
                return (cx, cy, ref[2] * rng.uniform(0.8, 1.3), ref[3] * rng.uniform(0.8, 1.3), ref[4] + rng.gauss(0, 0.2))
        if r < 0.5 and cards:  # a card with a big offset (partial card + table)
            c = quad_to_obb(rng.choice(cards)['quad'])
            m = max(c[2], c[3])
            ang = rng.uniform(0, 6.28)
            d = rng.uniform(0.35, 0.7) * m
            return (c[0] + d * math.cos(ang), c[1] + d * math.sin(ang), c[2], c[3], c[4])
        for _ in range(20):  # random spot away from cards
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            if all(np.linalg.norm(np.mean(c['quad'], 0) - (cx, cy)) > 0.6 * max(ref[2], ref[3]) for c in cards):
                break
        s = rng.uniform(0.5, 2.0)
        return (cx, cy, ref[2] * s * rng.uniform(0.8, 1.2), ref[3] * s * rng.uniform(0.8, 1.2), rng.uniform(0, 6.28))

    def _augment(self, crop, rng):
        if rng.random() < 0.5:
            crop = crop[::-1, ::-1]
        crop = np.ascontiguousarray(crop)
        if rng.random() < 0.35:  # simulate a smaller / softer card
            f = rng.uniform(0.35, 0.8)
            crop = cv2.resize(cv2.resize(crop, (max(8, int(CW * f)), max(8, int(CH * f))), interpolation=cv2.INTER_AREA), (CW, CH), interpolation=cv2.INTER_LINEAR)
        f = crop.astype(np.float32)
        if rng.random() < 0.8:  # extra colour cast / exposure
            f *= np.array([rng.uniform(0.85, 1.15) for _ in range(3)]) * rng.uniform(0.8, 1.2)
        if rng.random() < 0.3:
            f += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 8), f.shape)
        return np.clip(f, 0, 255).astype(np.uint8)


class SceneGroups(Dataset):
    """Groups the crops of one scene so each JPEG is decoded once (the loader was CPU-bound)."""

    def __init__(self, ds: Scenes):
        self.ds = ds
        groups = {}
        for i, (kind, name, _) in enumerate(ds.items):
            groups.setdefault(('real', i) if kind == 'real' else ('scene', name), []).append(i)
        self.groups = list(groups.values())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, g):
        idx = self.groups[g]
        kind, name, _ = self.ds.items[idx[0]]
        scene = None if kind == 'real' else self.ds._scene(name)
        xs, As, Vs = zip(*(self.ds.__getitem__(i, scene) for i in idx))
        return torch.stack(xs), torch.stack(As), torch.stack(Vs)


def collate(batch):
    return tuple(torch.cat(t) for t in zip(*batch))


class CardNet(nn.Module):
    def __init__(self, backbone='mobilenetv4_conv_small', pretrained=True):
        super().__init__()
        self.body = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        with torch.no_grad():
            self.body.eval(); d = self.body(torch.zeros(1, 3, CH, CW)).shape[1]; self.body.train()
        self.head = nn.Linear(d, 13)   # number(3) color(3) shape(3) fill(3) readable(1)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer('temp', torch.ones(5))  # per-head temperature (calibration)

    def forward(self, x):  # x: RGB 0..1
        z = self.head(self.body((x - self.mean) / self.std))
        t = torch.cat([self.temp[:4].repeat_interleave(3), self.temp[4:]])
        return z / t


def losses(z, attrs, valid):
    l = 0
    for h in range(4):
        w = 2.0 if h == 3 else 1.0
        if (attrs[:, h] >= 0).any():
            l = l + w * F.cross_entropy(z[:, h * 3:h * 3 + 3], attrs[:, h], ignore_index=-100)
    return l + F.binary_cross_entropy_with_logits(z[:, 12], valid)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    zs, As, Vs = [], [], []
    for x, a, v in loader:
        zs.append(model(x.to(dev)).float().cpu()); As.append(a); Vs.append(v)
    z, a, v = torch.cat(zs), torch.cat(As), torch.cat(Vs)
    m = a[:, 0] >= 0
    acc = [(z[m, h * 3:h * 3 + 3].argmax(1) == a[m, h]).float().mean().item() for h in range(4)]
    allok = torch.stack([(z[m, h * 3:h * 3 + 3].argmax(1) == a[m, h]) for h in range(4)]).all(0).float().mean().item()
    vacc = ((z[:, 12] > 0).float() == v).float().mean().item()
    return {'number': acc[0], 'color': acc[1], 'shape': acc[2], 'fill': acc[3], 'card': allok, 'readable': vacc}, (z, a, v)


def calibrate(model, z, a, v):
    """Fit one temperature per head on validation logits (they are already divided by the current temps)."""
    m = a[:, 0] >= 0
    temps = []
    for h in range(5):
        logT = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
        def closure():
            opt.zero_grad()
            if h < 4:
                loss = F.cross_entropy(z[m, h * 3:h * 3 + 3] / logT.exp(), a[m, h])
            else:
                loss = F.binary_cross_entropy_with_logits(z[:, 12] / logT.exp(), v)
            loss.backward(); return loss
        opt.step(closure)
        temps.append(logT.exp().item())
    model.temp *= torch.tensor(temps, device=model.temp.device)
    return temps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / 'data/synth_v3')
    ap.add_argument('--real', type=Path, default=None)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch', type=int, default=192)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--out', type=Path, default=ROOT / 'runs/cls_v3')
    ap.add_argument('--backbone', default='mobilenetv4_conv_small')
    ap.add_argument('--init', type=Path, default=None, help='start from these weights (fine-tuning)')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    tr = Scenes(a.data, 'train', real=a.real, train=True)
    va = Scenes(a.data, 'val', train=False)
    # batches are groups of scenes (~11 crops each); --batch is the approximate number of crops
    tl = DataLoader(SceneGroups(tr), max(1, a.batch // 11), shuffle=True, num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True, collate_fn=collate)
    vl = DataLoader(SceneGroups(va), 16, shuffle=False, num_workers=4, collate_fn=collate)
    model = CardNet(a.backbone, pretrained=a.init is None).to(dev)
    if a.init:
        sd = torch.load(a.init, map_location=dev); sd['temp'] = torch.ones(5)
        model.load_state_dict(sd)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.02)
    steps = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    best = 0
    for ep in range(a.epochs):
        model.train()
        tot = 0
        for i, (x, at, v) in enumerate(tl):
            x, at, v = x.to(dev, non_blocking=True), at.to(dev), v.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = model(x)
            loss = losses(z.float(), at, v)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item()
            if i % 200 == 0:
                print(f'ep {ep} it {i}/{len(tl)} loss {tot / (i + 1):.4f}', flush=True)
        res, _ = evaluate(model, vl, dev)
        print(f'epoch {ep}: ' + ' '.join(f'{k} {v:.4f}' for k, v in res.items()), flush=True)
        if res['card'] + res['readable'] > best:
            best = res['card'] + res['readable']
            torch.save(model.state_dict(), a.out / 'best.pt')
    model.load_state_dict(torch.load(a.out / 'best.pt'))
    res, (z, at, v) = evaluate(model, vl, dev)
    temps = calibrate(model, z, at, v)
    print('temperatures', temps)
    torch.save(model.state_dict(), a.out / 'best_calibrated.pt')
    (a.out / 'result.json').write_text(json.dumps({'val': res, 'temps': temps}, indent=1))


if __name__ == '__main__':
    main()
