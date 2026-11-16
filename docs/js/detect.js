// Card detector: YOLO26n-OBB (rotated boxes), always at 640 px.
// Input is a letterboxed region of the frame; output rows are cx, cy, w, h, score, angle.

import { ort, models } from './runtime.js';
import { rotIoU } from './geometry.js';

export const DET = 640;
const CONF = 0.35, NMS_IOU = 0.5;
const canvas = Object.assign(document.createElement('canvas'), { width: DET, height: DET });
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const input = new Float32Array(3 * DET * DET);

export const detStats = { ms: 0 };

// region = [x, y, w, h] in source pixels (default: whole frame)
export async function detect(src, W, H, region) {
  const t0 = performance.now();
  const [rx, ry, rw, rh] = region || [0, 0, W, H];
  const r = Math.min(DET / rw, DET / rh), nw = Math.round(rw * r), nh = Math.round(rh * r);
  const px = Math.floor((DET - nw) / 2), py = Math.floor((DET - nh) / 2);
  ctx.fillStyle = 'rgb(114,114,114)'; ctx.fillRect(0, 0, DET, DET);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, rx, ry, rw, rh, px, py, nw, nh);
  const d = ctx.getImageData(0, 0, DET, DET).data, n = DET * DET;
  for (let i = 0, j = 0; i < n; i++, j += 4) { input[i] = d[j] / 255; input[n + i] = d[j + 1] / 255; input[2 * n + i] = d[j + 2] / 255; }
  const det = models.det;
  const out = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, DET, DET]) });
  const t = out[det.outputNames[0]], o = t.data;
  // [1, 6, N] (raw head) or [1, N, 6+] (end-to-end export)
  const rowsFirst = t.dims[1] > t.dims[2], N = rowsFirst ? t.dims[1] : t.dims[2], K = rowsFirst ? t.dims[2] : t.dims[1];
  const at = rowsFirst ? (i, k) => o[i * K + k] : (i, k) => o[k * N + i];
  const angIdx = K - 1;
  const boxes = [];
  for (let i = 0; i < N; i++) {
    const sc = at(i, 4);
    if (sc < CONF) continue;
    boxes.push({ cx: (at(i, 0) - px) / r + rx, cy: (at(i, 1) - py) / r + ry, w: at(i, 2) / r, h: at(i, 3) / r, a: at(i, angIdx), score: sc });
  }
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of boxes) if (keep.every(k => rotIoU(k, b) < NMS_IOU)) keep.push(b);
  // a box mostly outside the searched region is cut off by its edge: unreliable
  const inside = keep.filter(b => b.cx > rx && b.cx < rx + rw && b.cy > ry && b.cy < ry + rh);
  detStats.ms = detStats.ms ? detStats.ms * 0.8 + (performance.now() - t0) * 0.2 : performance.now() - t0;
  return inside;
}
