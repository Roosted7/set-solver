// Card classifier: MobileNetV4-small on 96x144 crops cut straight out of the frame along the
// detector's rotated box (same geometry as src/synth/crops.py used in training).
// Output per crop: 12 attribute logits (number|color|shape|fill, calibrated) + 1 "readable card" logit.

import { ort, models } from './runtime.js';

export const CW = 96, CH = 144, MARGIN = 1.06;
const atlas = document.createElement('canvas');
const actx = atlas.getContext('2d', { willReadFrequently: true });
export const clsStats = { msPerCrop: 4, ms: 0 };

// affine (as canvas transform) mapping source pixels into crop k of the atlas
function cropTransform(b, k) {
  let w = b.w, h = b.h, ux = Math.cos(b.a), uy = Math.sin(b.a), vx = -uy, vy = ux;
  if (w > h) { [w, h] = [h, w]; [ux, uy, vx, vy] = [vx, vy, -ux, -uy]; }
  const sw = w * MARGIN, sh = h * MARGIN;
  const tlx = b.cx - ux * sw / 2 - vx * sh / 2, tly = b.cy - uy * sw / 2 - vy * sh / 2;
  // source = tl + u * (ex) + v * (ey), with ex = U*sw/CW, ey = V*sh/CH
  const a00 = ux * sw / CW, a10 = uy * sw / CW, a01 = vx * sh / CH, a11 = vy * sh / CH;
  const det = a00 * a11 - a01 * a10;
  const i00 = a11 / det, i01 = -a01 / det, i10 = -a10 / det, i11 = a00 / det;
  const e = -(i00 * tlx + i01 * tly), f = -(i10 * tlx + i11 * tly) + k * CH;
  return [i00, i10, i01, i11, e, f];
}

// boxes: [{cx,cy,w,h,a}]  ->  [{lp: Float64Array(12) log-probs, readable: 0..1}]
export async function classify(src, boxes) {
  if (!boxes.length) return [];
  const t0 = performance.now(), n = boxes.length;
  if (atlas.width !== CW || atlas.height < n * CH) { atlas.width = CW; atlas.height = Math.max(n, 8) * CH; }
  actx.imageSmoothingEnabled = true; actx.imageSmoothingQuality = 'high';
  for (let k = 0; k < n; k++) {
    actx.save();
    actx.setTransform(1, 0, 0, 1, 0, 0);
    actx.beginPath(); actx.rect(0, k * CH, CW, CH); actx.clip();
    actx.fillStyle = '#777'; actx.fillRect(0, k * CH, CW, CH);
    actx.setTransform(...cropTransform(boxes[k], k));
    actx.drawImage(src, 0, 0);
    actx.restore();
  }
  const d = actx.getImageData(0, 0, CW, n * CH).data, px = CW * CH;
  const input = new Float32Array(n * 3 * px);
  for (let k = 0; k < n; k++) {
    const o = k * 3 * px, s = k * px * 4;
    for (let i = 0; i < px; i++) { const j = s + i * 4; input[o + i] = d[j] / 255; input[o + px + i] = d[j + 1] / 255; input[o + 2 * px + i] = d[j + 2] / 255; }
  }
  const cls = models.cls;
  const out = await cls.run({ [cls.inputNames[0]]: new ort.Tensor('float32', input, [n, 3, CH, CW]) });
  const L = out[cls.outputNames[0]].data, K = L.length / n;
  const ms = performance.now() - t0;
  clsStats.ms = ms; clsStats.msPerCrop = clsStats.msPerCrop * 0.8 + (ms / n) * 0.2;
  const res = [];
  for (let k = 0; k < n; k++) {
    const lp = new Float64Array(12);
    for (let h = 0; h < 4; h++) {
      const o = k * K + h * 3, m = Math.max(L[o], L[o + 1], L[o + 2]);
      const lse = m + Math.log(Math.exp(L[o] - m) + Math.exp(L[o + 1] - m) + Math.exp(L[o + 2] - m));
      for (let c = 0; c < 3; c++) lp[h * 3 + c] = L[o + c] - lse;
    }
    res.push({ lp, readable: 1 / (1 + Math.exp(-L[k * K + 12])) });
  }
  return res;
}

// debug: last atlas as data URL
export const atlasURL = () => atlas.toDataURL();
