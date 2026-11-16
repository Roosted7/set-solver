// SET! — on-device Set solver.
// Models: YOLO11n card detector + MobileNetV3-small 4-head attribute classifier,
// both from wangtian24/set-solver, converted to ONNX and run with onnxruntime-web.
//
// Pipeline per frame
//   1. cheap motion check on a 64x48 thumbnail -> skip the detector when nothing changed
//   2. detector on a region of interest around the cards (auto 320/480/640)
//   3. tracker: match boxes to card tracks, measure per-card motion (camera shake removed)
//   4. classify only cards that need it (new / uncertain / not re-checked for a while),
//      after finding the card's 4 corners and warping it flat (handles player-angle views)
//   5. each card accumulates evidence (log-probabilities per attribute) over frames
//   6. call SET only when the table is settled: no moving cards, nothing occluded, all certain

const ORT_VER = '1.31.1';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
const ort = await import(ORT_BASE + 'ort.webgpu.min.mjs');
ort.env.wasm.wasmPaths = ORT_BASE;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;

// ---------- constants ----------
const DET_CONF = 0.4, NMS_IOU = 0.6, CLS = 224;
const LOGP_FLOOR = Math.log(0.03);   // one bad frame can't outvote many good ones
const CERT_LIVE = 0.985;             // posterior needed per attribute to trust a card (live: several readings)
const CERT_STILL = 0.9;              // a single photo only gets one reading
let CERT = CERT_LIVE;
const RECHECK_MS = 2500;             // re-verify certain cards this often
const STATIC_THR = 0.05;             // motion (fraction of card width) still counted as "resting"
const FRAME_BUDGET_MS = 110;         // target time per processing step
const NUMBER = ['1', '2', '3'];
const COLOR = ['red', 'green', 'purple'];
const SHAPE = ['diamond', 'oval', 'squiggle'];
const FILL = ['open', 'solid', 'striped'];
const COLOR_HEX = { red: '#ff5a5a', green: '#3ddc78', purple: '#bb77ff' };
const SHAPE_SYM = { diamond: '◆', oval: '⬮', squiggle: '∿' };
const SET_COLORS = ['#ffd400', '#00e5ff', '#ff4fd8', '#8cff3c', '#ff8a00', '#9d7bff', '#ffffff'];
const attrsOf = id => [Math.floor(id / 27), Math.floor(id / 9) % 3, Math.floor(id / 3) % 3, id % 3];
const cardName = id => { const [n, c, s, f] = attrsOf(id); return `${NUMBER[n]} ${FILL[f]} ${COLOR[c]} ${SHAPE_SYM[SHAPE[s]]}`; };
const isSet = (a, b, c) => { const A = attrsOf(a), B = attrsOf(b), C = attrsOf(c); return A.every((v, i) => (v + B[i] + C[i]) % 3 === 0); };
const median = a => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---------- DOM + settings ----------
const $ = id => document.getElementById(id);
const video = $('video'), still = $('still'), overlay = $('overlay'), octx = overlay.getContext('2d'), stage = $('stage');
const statusEl = $('status'), statsEl = $('stats'), banner = $('banner');
const SETTINGS = ['minCards', 'detSize', 'camRes', 'showLabels', 'soundMode', 'rectify', 'debug'];
const settings = new Proxy({}, { get: (_, k) => $(k).value });
for (const id of SETTINGS) {
  try { const v = localStorage.getItem('set.' + id); if (v !== null && [...$(id).options].some(o => o.value === v)) $(id).value = v; } catch {}
  $(id).addEventListener('change', () => { try { localStorage.setItem('set.' + id, $(id).value); } catch {} });
}

// ---------- models ----------
const sessions = {};
let backend = '?';
async function makeSession(url) {
  const eps = ('gpu' in navigator) ? ['webgpu', 'wasm'] : ['wasm'];
  for (const ep of eps) {
    try { const s = await ort.InferenceSession.create(url, { executionProviders: [ep], graphOptimizationLevel: 'all' }); backend = ep; return s; }
    catch (e) { console.warn('EP failed', ep, e); }
  }
  throw new Error('Could not load ' + url);
}
const getDetector = size => (sessions['det' + size] ||= makeSession(`models/detector_${size}.onnx`));
const getClassifier = () => (sessions.cls ||= makeSession('models/classifier.onnx'));
const modelsReady = Promise.all([getDetector(640), getClassifier()]).then(() => {
  $('loadMsg').textContent = `Models loaded (${backend}).`;
  $('backendInfo').textContent = `Backend: ${backend}${ort.env.wasm.numThreads > 1 ? `, ${ort.env.wasm.numThreads} threads` : ''}. Everything runs on this device.`;
}).catch(e => { $('err').textContent = 'Model load failed: ' + e.message; throw e; });

// ---------- canvases ----------
const frameCanvas = document.createElement('canvas'), frameCtx = frameCanvas.getContext('2d');
const detCanvas = document.createElement('canvas'), detCtx = detCanvas.getContext('2d', { willReadFrequently: true });
const thumbCanvas = Object.assign(document.createElement('canvas'), { width: 64, height: 48 });
const thumbCtx = thumbCanvas.getContext('2d', { willReadFrequently: true });

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]), x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return u > 0 ? inter / u : 0;
}

// ---------- detection ----------
const detTimes = {};   // per size: EMA in ms
let autoSize = 640, autoRuns = {}, stillMode = false;
function currentDetSize() { return stillMode ? 640 : settings.detSize === 'auto' ? autoSize : +settings.detSize; }
// auto: 640 copes best with small and angled cards; step down only when it's too slow on this phone
function updateAutoSize(S, ms) {
  autoRuns[S] = (autoRuns[S] || 0) + 1;
  if (autoRuns[S] <= 3) return;                       // ignore warm-up runs
  detTimes[S] = detTimes[S] ? detTimes[S] * 0.85 + ms * 0.15 : ms;
  if (settings.detSize !== 'auto' || autoRuns[S] < 8) return;
  const t = detTimes[S];
  if (S === 640 && t > 110) { autoSize = 480; getDetector(480); }
  else if (S === 480 && t > 140) { autoSize = 320; getDetector(320); }
  else if (S === 480 && t < 55 && !(detTimes[640] > 110)) autoSize = 640;
  else if (S === 320 && t < 45 && !(detTimes[480] > 140)) autoSize = 480;
}

async function detect(src, W, H, roi) {
  const S = currentDetSize();
  const det = await getDetector(S);
  const t0 = performance.now();
  detCanvas.width = detCanvas.height = S;
  const [rx, ry, rw, rh] = roi || [0, 0, W, H];
  const r = Math.min(S / rw, S / rh), nw = Math.round(rw * r), nh = Math.round(rh * r);
  const px = Math.floor((S - nw) / 2), py = Math.floor((S - nh) / 2);
  detCtx.fillStyle = 'rgb(114,114,114)'; detCtx.fillRect(0, 0, S, S);
  detCtx.drawImage(src, rx, ry, rw, rh, px, py, nw, nh);
  const d0 = detCtx.getImageData(0, 0, S, S).data, n = S * S, input = new Float32Array(3 * n);
  for (let i = 0, j = 0; i < n; i++, j += 4) { input[i] = d0[j] / 255; input[n + i] = d0[j + 1] / 255; input[2 * n + i] = d0[j + 2] / 255; }
  const out = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, S, S]) });
  const t = out[det.outputNames[0]], d = t.data, N = t.dims[2];
  const boxes = [];
  for (let i = 0; i < N; i++) {
    const sc = d[4 * N + i];
    if (sc < DET_CONF) continue;
    const cx = (d[i] - px) / r + rx, cy = (d[N + i] - py) / r + ry, w = d[2 * N + i] / r, h = d[3 * N + i] / r;
    boxes.push({ box: [Math.max(0, cx - w / 2), Math.max(0, cy - h / 2), Math.min(W, cx + w / 2), Math.min(H, cy + h / 2)], score: sc });
  }
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of boxes) if (keep.every(k => iou(k.box, b.box) < NMS_IOU)) keep.push(b);
  updateAutoSize(S, performance.now() - t0);
  lastDetMs = performance.now() - t0;
  return dropStraddlers(splitMerged(keep));
}

// In a tight grid the detector sometimes adds a box straddling two neighbouring cards.
// Such a box is almost entirely covered by other boxes; a real card (even one partly
// under another card) is not.
function dropStraddlers(boxes) {
  const order = [...boxes].sort((a, b) => a.score - b.score);
  let keep = [...boxes];
  for (const b of order) {
    const others = keep.filter(o => o !== b), a = (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]);
    let cov = 0;
    for (const o of others) cov += Math.max(0, Math.min(b.box[2], o.box[2]) - Math.max(b.box[0], o.box[0])) * Math.max(0, Math.min(b.box[3], o.box[3]) - Math.max(b.box[1], o.box[1]));
    if (cov / a > 0.6 && others.filter(o => iou(o.box, b.box) > 0.15).length >= 2) keep = others;
  }
  return keep;
}

// Touching cards are sometimes detected as one box twice the size: split those, drop other oversized boxes.
function splitMerged(boxes) {
  if (boxes.length < 3) return boxes;
  const mw = median(boxes.map(b => b.box[2] - b.box[0])), mh = median(boxes.map(b => b.box[3] - b.box[1]));
  const out = [];
  for (const b of boxes) {
    const [x1, y1, x2, y2] = b.box, fw = (x2 - x1) / mw, fh = (y2 - y1) / mh;
    const near1 = v => v > 0.75 && v < 1.3, near2 = v => v > 1.7 && v < 2.5;
    if (near2(fw) && near1(fh)) { const xm = (x1 + x2) / 2; out.push({ ...b, box: [x1, y1, xm, y2] }, { ...b, box: [xm, y1, x2, y2] }); }
    else if (near2(fh) && near1(fw)) { const ym = (y1 + y2) / 2; out.push({ ...b, box: [x1, y1, x2, ym] }, { ...b, box: [x1, ym, x2, y2] }); }
    else if (fw * fh <= 2.2) out.push(b);
  }
  return out;
}

// ---------- region of interest ----------
// The detector was trained on photos where the cards fill most of the frame, so with a
// wide-angle lens we detect on a crop around the cards (and search centre crops when lost).
const SEARCH = [1, 0.7, 0.5, 0.36];
let roi = null, missFrames = 0, searchIdx = 0, lastRoi = null, lastDetMs = 0;
const centreRoi = (W, H, f) => [W * (1 - f) / 2, H * (1 - f) / 2, W * f, H * f];
function roiAround(boxes, W, H, margin) {
  const mw = median(boxes.map(b => b.box[2] - b.box[0])), mh = median(boxes.map(b => b.box[3] - b.box[1]));
  let x1 = Math.min(...boxes.map(b => b.box[0])) - margin * mw, y1 = Math.min(...boxes.map(b => b.box[1])) - margin * mh;
  let x2 = Math.max(...boxes.map(b => b.box[2])) + margin * mw, y2 = Math.max(...boxes.map(b => b.box[3])) + margin * mh;
  const minW = 4.5 * mw, minH = 4.5 * mh;
  if (x2 - x1 < minW) { const c = (x1 + x2) / 2; x1 = c - minW / 2; x2 = c + minW / 2; }
  if (y2 - y1 < minH) { const c = (y1 + y2) / 2; y1 = c - minH / 2; y2 = c + minH / 2; }
  x1 = Math.max(0, x1); y1 = Math.max(0, y1); x2 = Math.min(W, x2); y2 = Math.min(H, y2);
  return [x1, y1, x2 - x1, y2 - y1];
}
function grow([x, y, w, h], W, H, f) {
  const nw = Math.min(W, w * f), nh = Math.min(H, h * f);
  return [Math.max(0, Math.min(W - nw, x - (nw - w) / 2)), Math.max(0, Math.min(H - nh, y - (nh - h) / 2)), nw, nh];
}
async function detectTracked(src, W, H, now) {
  const wide = roi && detCount % 6 === 0;   // periodically look wider for newly dealt cards
  const r = roi ? (wide ? grow(roi, W, H, 1.6) : roi) : centreRoi(W, H, SEARCH[searchIdx]);
  const dets = await detect(src, W, H, r);
  lastRoi = r;
  // search area covers this frame's detections AND every card we're still tracking,
  // so a card missed in one frame stays inside the area for the next one
  const known = dets.concat(tracks.filter(t => now - t.seenAt < 2500).map(t => ({ box: t.box })));
  if (dets.length) { missFrames = 0; roi = roiAround(known, W, H, 1.3); }
  else if (++missFrames >= 2) { roi = null; searchIdx = (searchIdx + 1) % SEARCH.length; missFrames = 0; }
  return dets;
}
async function detectStill(src, W, H) {
  let best = [];
  for (const f of SEARCH) { const d = await detect(src, W, H, centreRoi(W, H, f)); if (d.length > best.length) best = d; if (best.length >= 9) break; }
  if (best.length) { const r = roiAround(best, W, H, 1.3); const d = await detect(src, W, H, r); if (d.length >= best.length) best = d; lastRoi = r; }
  return best;
}

// ---------- card rectification ----------
// Find the white card's 4 corners inside the detection box and warp it onto a square.
// The classifier was trained on whole card scans resized to a square, so this matches its
// training data regardless of viewing angle, rotation, or neighbouring cards.
function cardQuad(img, rw, rh, inner) {
  const step = Math.max(1, Math.ceil(Math.max(rw, rh) / 128));
  const mw = Math.floor(rw / step), mh = Math.floor(rh / step), n = mw * mh;
  const score = new Uint8Array(n), hist = new Uint32Array(256), d = img.data;
  for (let y = 0; y < mh; y++) for (let x = 0; x < mw; x++) {
    const j = ((y * step) * rw + x * step) * 4, v = Math.min(d[j], d[j + 1], d[j + 2]);
    score[y * mw + x] = v; hist[v]++;
  }
  // Otsu threshold on min(R,G,B): card paper is bright in all channels, table and symbols are not
  let sum = 0; for (let i = 0; i < 256; i++) sum += i * hist[i];
  let sB = 0, wB = 0, best = 0, thr = 128;
  for (let i = 0; i < 256; i++) {
    wB += hist[i]; if (!wB) continue; const wF = n - wB; if (!wF) break;
    sB += i * hist[i]; const mB = sB / wB, mF = (sum - sB) / wF, v = wB * wF * (mB - mF) ** 2;
    if (v > best) { best = v; thr = i; }
  }
  const [ix1, iy1, ix2, iy2] = inner.map(v => v / step);
  const fg = new Uint8Array(n);
  for (let i = 0; i < n; i++) fg[i] = score[i] > thr ? 1 : 0;
  // connected components; keep the largest one that overlaps the central part of the box
  const lab = new Int32Array(n), stack = new Int32Array(n);
  const cx1 = ix1 + (ix2 - ix1) * 0.25, cx2 = ix2 - (ix2 - ix1) * 0.25, cy1 = iy1 + (iy2 - iy1) * 0.25, cy2 = iy2 - (iy2 - iy1) * 0.25;
  let bestLab = 0, bestArea = 0, nl = 0;
  for (let s = 0; s < n; s++) {
    if (!fg[s] || lab[s]) continue;
    nl++; let sp = 0, area = 0, central = false; stack[sp++] = s; lab[s] = nl;
    while (sp) {
      const p = stack[--sp], x = p % mw, y = (p / mw) | 0; area++;
      if (x >= cx1 && x <= cx2 && y >= cy1 && y <= cy2) central = true;
      if (x > 0 && fg[p - 1] && !lab[p - 1]) { lab[p - 1] = nl; stack[sp++] = p - 1; }
      if (x < mw - 1 && fg[p + 1] && !lab[p + 1]) { lab[p + 1] = nl; stack[sp++] = p + 1; }
      if (y > 0 && fg[p - mw] && !lab[p - mw]) { lab[p - mw] = nl; stack[sp++] = p - mw; }
      if (y < mh - 1 && fg[p + mw] && !lab[p + mw]) { lab[p + mw] = nl; stack[sp++] = p + mw; }
    }
    if (central && area > bestArea) { bestArea = area; bestLab = nl; }
  }
  if (!bestLab) return null;
  // row extremes -> convex hull
  const pts = [];
  for (let y = 0; y < mh; y++) {
    let lo = -1, hi = -1;
    for (let x = 0; x < mw; x++) if (lab[y * mw + x] === bestLab) { if (lo < 0) lo = x; hi = x; }
    if (lo >= 0) { pts.push([lo, y], [hi + 1, y], [lo, y + 1], [hi + 1, y + 1]); }
  }
  let hull = convexHull(pts);
  if (hull.length < 4) return null;
  // simplify to 4 corners by repeatedly dropping the vertex that loses the least area
  while (hull.length > 4) {
    let bi = 0, ba = Infinity;
    for (let i = 0; i < hull.length; i++) {
      const a = hull[(i + hull.length - 1) % hull.length], b = hull[i], c = hull[(i + 1) % hull.length];
      const ar = Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]));
      if (ar < ba) { ba = ar; bi = i; }
    }
    hull.splice(bi, 1);
  }
  const quad = hull.map(([x, y]) => [x * step, y * step]);
  // sanity checks: plausible size relative to the detection box, not degenerate
  const qa = polyArea(quad), ba2 = (inner[2] - inner[0]) * (inner[3] - inner[1]);
  if (qa < 0.3 * ba2 || qa > 1.2 * ba2) return null;
  for (let i = 0; i < 4; i++) { const a = quad[i], b = quad[(i + 1) % 4]; if (Math.hypot(a[0] - b[0], a[1] - b[1]) < 0.25 * Math.sqrt(qa)) return null; }
  return quad;
}
function convexHull(p) {
  p.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
const polyArea = q => Math.abs(q.reduce((s, a, i) => { const b = q[(i + 1) % q.length]; return s + a[0] * b[1] - b[0] * a[1]; }, 0)) / 2;

// homography mapping unit-square corners (0,0),(1,0),(1,1),(0,1) -> quad
function squareToQuad(q) {
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = q;
  const dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3, dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
  const den = dx1 * dy2 - dx2 * dy1;
  const g = (dx3 * dy2 - dx2 * dy3) / den, h = (dx1 * dy3 - dx3 * dy1) / den;
  return [x1 - x0 + g * x1, x3 - x0 + h * x3, x0, y1 - y0 + g * y1, y3 - y0 + h * y3, y0, g, h];
}

// Warp quad -> square into out (CHW, CLS x CLS). Rendered at `detail` px then upsampled,
// which mimics the low-resolution card photos the classifier was trained on.
function warpInto(img, rw, rh, quad, out, off, detail) {
  const [a, b, c, d, e, f, g, h] = squareToQuad(quad), D = img.data, px = CLS * CLS;
  const small = new Float32Array(detail * detail * 3);
  for (let y = 0; y < detail; y++) for (let x = 0; x < detail; x++) {
    const u = (x + 0.5) / detail, v = (y + 0.5) / detail, w = g * u + h * v + 1;
    let sx = (a * u + b * v + c) / w - 0.5, sy = (d * u + e * v + f) / w - 0.5;
    sx = Math.max(0, Math.min(rw - 1.001, sx)); sy = Math.max(0, Math.min(rh - 1.001, sy));
    const x0 = sx | 0, y0 = sy | 0, fx = sx - x0, fy = sy - y0, i00 = (y0 * rw + x0) * 4, i10 = i00 + 4, i01 = i00 + rw * 4, i11 = i01 + 4;
    const o = (y * detail + x) * 3;
    for (let k = 0; k < 3; k++)
      small[o + k] = ((D[i00 + k] * (1 - fx) + D[i10 + k] * fx) * (1 - fy) + (D[i01 + k] * (1 - fx) + D[i11 + k] * fx) * fy) / 255;
  }
  const sc = detail / CLS;
  for (let y = 0; y < CLS; y++) {
    let sy = (y + 0.5) * sc - 0.5; sy = Math.max(0, Math.min(detail - 1.001, sy));
    const y0 = sy | 0, fy = sy - y0;
    for (let x = 0; x < CLS; x++) {
      let sx = (x + 0.5) * sc - 0.5; sx = Math.max(0, Math.min(detail - 1.001, sx));
      const x0 = sx | 0, fx = sx - x0, i00 = (y0 * detail + x0) * 3, i10 = i00 + 3, i01 = i00 + detail * 3, i11 = i01 + 3;
      for (let k = 0; k < 3; k++)
        out[off + k * px + y * CLS + x] = (small[i00 + k] * (1 - fx) + small[i10 + k] * fx) * (1 - fy) + (small[i01 + k] * (1 - fx) + small[i11 + k] * fx) * fy;
    }
  }
}

// Order corners clockwise starting with the top-left-most one.
function orderQuad(q) {
  const cx = q.reduce((s, p) => s + p[0], 0) / 4, cy = q.reduce((s, p) => s + p[1], 0) / 4;
  const s = [...q].sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
  let k = 0; for (let i = 1; i < 4; i++) if (s[i][0] + s[i][1] < s[k][0] + s[k][1]) k = i;
  return [0, 1, 2, 3].map(i => s[(k + i) % 4]);
}

// Classify a list of {box, views} jobs. views: 1 or 2 (second = card rotated 180°).
let clsMsPerView = 15;
async function classify(src, jobs) {
  if (!jobs.length) return [];
  const cls = await getClassifier();
  const t0 = performance.now();
  const nViews = jobs.reduce((s, j) => s + j.views, 0), px = CLS * CLS;
  const input = new Float32Array(nViews * 3 * px);
  const detail = settings.debug === 'sharp' ? CLS : 112;
  let v = 0;
  for (const job of jobs) {
    const [bx1, by1, bx2, by2] = job.box, bw = bx2 - bx1, bh = by2 - by1;
    const rx = Math.max(0, Math.floor(bx1 - bw * 0.08)), ry = Math.max(0, Math.floor(by1 - bh * 0.08));
    const rw = Math.max(2, Math.min(src.width, Math.ceil(bx2 + bw * 0.08)) - rx), rh = Math.max(2, Math.min(src.height, Math.ceil(by2 + bh * 0.08)) - ry);
    const img = src.getContext('2d').getImageData(rx, ry, rw, rh);
    const inner = [bx1 - rx, by1 - ry, bx2 - rx, by2 - ry];
    let quad = settings.rectify === 'on' ? cardQuad(img, rw, rh, inner) : null;
    job.rectified = !!quad;
    if (!quad) quad = [[inner[0], inner[1]], [inner[2], inner[1]], [inner[2], inner[3]], [inner[0], inner[3]]];
    quad = orderQuad(quad);
    job.quad = quad.map(([x, y]) => [x + rx, y + ry]);
    warpInto(img, rw, rh, quad, input, v++ * 3 * px, detail);
    if (job.views > 1) warpInto(img, rw, rh, [quad[2], quad[3], quad[0], quad[1]], input, v++ * 3 * px, detail);
  }
  const out = await cls.run({ [cls.inputNames[0]]: new ort.Tensor('float32', input, [nViews, 3, CLS, CLS]) });
  const L = out[cls.outputNames[0]].data;
  clsMsPerView = clsMsPerView * 0.8 + ((performance.now() - t0) / nViews) * 0.2;
  stats.crops += nViews;
  // per job: averaged log-probabilities (12 values)
  v = 0;
  return jobs.map(job => {
    const lp = new Float64Array(12);
    for (let r = 0; r < job.views; r++, v++) for (let h = 0; h < 4; h++) {
      const o = v * 12 + h * 3, m = Math.max(L[o], L[o + 1], L[o + 2]);
      const lse = m + Math.log(Math.exp(L[o] - m) + Math.exp(L[o + 1] - m) + Math.exp(L[o + 2] - m));
      for (let k = 0; k < 3; k++) lp[h * 3 + k] += (L[o + k] - lse) / job.views;
    }
    return lp;
  });
}

// ---------- tracks: one per physical card, accumulating evidence ----------
let tracks = [], nextTid = 1, detCount = 0;
const stats = { frames: 0, dets: 0, crops: 0, skipped: 0 };
function newTrack(box, now) {
  return { tid: nextTid++, box, born: now, nDet: 1, seenAt: now, seen: true, cx: (box[0] + box[2]) / 2, cy: (box[1] + box[3]) / 2,
    seenDet: detCount, motion: 1, staticSince: 0, lp: new Float64Array(12), nObs: 0, cert: 0, label: 0, lastCls: 0, verified: false, disagree: 0, quad: null, rectified: false };
}
function posterior(t) {
  // per attribute: index of best value and its probability
  const idx = [], p = [];
  for (let h = 0; h < 4; h++) {
    const o = h * 3, m = Math.max(t.lp[o], t.lp[o + 1], t.lp[o + 2]);
    const e = [0, 1, 2].map(k => Math.exp(t.lp[o + k] - m)), s = e[0] + e[1] + e[2];
    const k = e.indexOf(Math.max(...e)); idx.push(k); p.push(e[k] / s);
  }
  return { label: idx[0] * 27 + idx[1] * 9 + idx[2] * 3 + idx[3], cert: t.nObs ? Math.min(...p) : 0 };
}
function addEvidence(t, lp) {
  const obs = posterior({ lp, nObs: 1 }), cur = posterior(t);
  if (t.nObs >= 2 && cur.cert > 0.9 && obs.cert > 0.85 && obs.label !== cur.label) {
    // a confident reading that contradicts what we know: occlusion, glare, or a new card in this spot
    t.verified = false;
    if (++t.disagree >= 2) { t.lp = new Float64Array(12); t.nObs = 0; t.disagree = 0; }
  } else if (obs.label === cur.label || t.nObs === 0) { t.disagree = 0; t.verified = true; }
  for (let i = 0; i < 12; i++) t.lp[i] += Math.max(LOGP_FLOOR, lp[i]);
  t.nObs++;
  Object.assign(t, posterior(t));
}

function updateTracks(dets, now) {
  detCount++;
  for (const t of tracks) t.seen = false;
  // greedy IoU matching, then centre-distance matching for what's left
  const pairs = [];
  dets.forEach((d, i) => tracks.forEach((t, j) => { const v = iou(d.box, t.box); if (v > 0.2) pairs.push([v, i, j]); }));
  pairs.sort((a, b) => b[0] - a[0]);
  const dT = new Array(dets.length).fill(null), usedT = new Set();
  for (const [, i, j] of pairs) if (!dT[i] && !usedT.has(j)) { dT[i] = tracks[j]; usedT.add(j); }
  dets.forEach((d, i) => {
    if (dT[i]) return;
    const cx = (d.box[0] + d.box[2]) / 2, cy = (d.box[1] + d.box[3]) / 2, cw = d.box[2] - d.box[0];
    let best = null, bd = 0.6 * cw;
    tracks.forEach((t, j) => { if (usedT.has(j)) return; const dd = Math.hypot(t.cx - cx, t.cy - cy); if (dd < bd) { bd = dd; best = j; } });
    if (best !== null) { dT[i] = tracks[best]; usedT.add(best); }
  });
  // camera shake: remove the median shift, what's left is the card's own motion
  const shifts = dets.map((d, i) => dT[i] ? [(d.box[0] + d.box[2]) / 2 - dT[i].cx, (d.box[1] + d.box[3]) / 2 - dT[i].cy] : null).filter(Boolean);
  const gx = shifts.length ? median(shifts.map(s => s[0])) : 0, gy = shifts.length ? median(shifts.map(s => s[1])) : 0;
  dets.forEach((d, i) => {
    const cx = (d.box[0] + d.box[2]) / 2, cy = (d.box[1] + d.box[3]) / 2, cw = d.box[2] - d.box[0];
    let t = dT[i];
    if (!t) { t = newTrack(d.box, now); tracks.push(t); }
    else {
      t.motion = Math.hypot(cx - t.cx - gx, cy - t.cy - gy) / cw;
      // a jump in size also counts as motion (card lifted, picked up, or box flickering)
      t.motion = Math.max(t.motion, Math.abs(Math.log(((d.box[2] - d.box[0]) * (d.box[3] - d.box[1])) / ((t.box[2] - t.box[0]) * (t.box[3] - t.box[1])))) / 2);
      if (t.motion < STATIC_THR) { if (!t.staticSince) t.staticSince = now; } else { t.staticSince = 0; }
      if (detCount - t.seenDet > 2) t.verified = false;   // was hidden for a while: re-check before trusting
      t.nDet++;
    }
    t.box = d.box; t.cx = cx; t.cy = cy; t.seen = true; t.seenAt = now; t.seenDet = detCount;
  });
  // keep hidden cards for a while (hand passing over); certain cards longer
  tracks = tracks.filter(t => t.seen || now - t.seenAt < (t.cert >= CERT ? 2000 : 600));
  // a hidden track overlapping a visible one is a duplicate
  tracks = tracks.filter(t => t.seen || !tracks.some(o => o.seen && iou(o.box, t.box) > 0.4));
}
const isSettled = (t, now) => t.seen && t.nDet >= 3 && t.staticSince && now - t.staticSince >= 250;

// Which cards to classify this step, in priority order, within the time budget.
function schedule(now, isStill) {
  const live = tracks.filter(t => t.seen && (isStill || (t.nDet >= 2 && t.motion < STATIC_THR * 2)));
  const need = live.filter(t => t.cert < CERT || !t.verified).sort((a, b) => a.nObs - b.nObs || a.lastCls - b.lastCls);
  const recheck = live.filter(t => !need.includes(t) && now - t.lastCls > RECHECK_MS).sort((a, b) => a.lastCls - b.lastCls);
  const budget = isStill ? 99 : Math.max(4, Math.min(16, Math.floor((FRAME_BUDGET_MS - lastDetMs) / Math.max(3, clsMsPerView))));
  const jobs = [];
  let used = 0;
  for (const t of need) { const v = t.nObs < 4 ? 2 : 1; if (used + v > budget && jobs.length) break; jobs.push({ t, box: t.box, views: v }); used += v; }
  for (const t of recheck) { if (used + 1 > budget) break; jobs.push({ t, box: t.box, views: 1 }); used++; }
  return jobs;
}

// ---------- calling sets ----------
let state = 'idle';               // idle | scanning | found
let found = null;                 // { sets: [[tid,tid,tid],...], labels: Map tid->label, at, sel }
let candidateKey = '', candidateStreak = 0, noSetStreak = 0, emptySince = 0, lastEvalDet = -1;

function setsAmong(cards) {
  const out = [];
  for (let i = 0; i < cards.length; i++) for (let j = i + 1; j < cards.length; j++) for (let k = j + 1; k < cards.length; k++)
    if (isSet(cards[i].label, cards[j].label, cards[k].label)) out.push([cards[i], cards[j], cards[k]]);
  return out;
}

function evaluate(now, isStill) {
  if (state === 'found') {
    // keep showing the announced sets while their cards are still there with the same reading
    const byTid = new Map(tracks.map(t => [t.tid, t]));
    const alive = found.sets.filter(s => s.every(tid => { const t = byTid.get(tid); return t && t.label === found.labels.get(tid); }));
    found.visible = alive;
    const settledNow = tracks.filter(t => isSettled(t, now) && t.born > found.at).length;
    if (settledNow >= 1 && alive.length < found.sets.length) { state = 'scanning'; found = null; candidateStreak = 0; return evaluate(now, isStill); }
    if (alive.length) emptySince = 0;
    else if (!emptySince) emptySince = now;
    if (!isStill && !alive.length && now - emptySince > 1000 && now - found.at > 2500) { state = 'scanning'; found = null; candidateStreak = 0; }
    else { setStatus(alive.length ? `SET!${alive.length > 1 ? `  ${alive.length} sets` : ''}` : 'SET! (cards taken)'); renderSetList(); return; }
  }
  if (state !== 'scanning') return;
  renderSetList();
  const live = tracks.filter(t => t.seen), hidden = tracks.filter(t => !t.seen);
  const min = +settings.minCards;
  if (live.length < min) { setStatus(live.length ? `Waiting for cards… ${live.length}/${min}` : 'Looking for cards…'); candidateStreak = noSetStreak = 0; return; }
  if (!isStill && (hidden.length || live.some(t => !isSettled(t, now)))) { setStatus('Waiting for the table to settle…'); candidateStreak = 0; return; }
  const unread = live.filter(t => t.cert < CERT || !t.verified);
  if (unread.length) { setStatus(`Reading cards… ${live.length - unread.length}/${live.length}`); candidateStreak = 0; return; }
  // the same card twice means one of them is misread: look again
  const byLabel = new Map();
  for (const t of live) { if (byLabel.has(t.label)) { t.verified = false; byLabel.get(t.label).verified = false; } else byLabel.set(t.label, t); }
  if (live.some(t => !t.verified)) { setStatus('Double-checking…'); candidateStreak = 0; return; }
  const sets = setsAmong(live);
  if (sets.length) {
    const key = sets.map(s => s.map(t => t.tid).sort((a, b) => a - b).join('.')).sort().join('|');
    if (detCount !== lastEvalDet) { candidateStreak = key === candidateKey ? candidateStreak + 1 : 1; lastEvalDet = detCount; }
    candidateKey = key;
    if (candidateStreak >= 2 || isStill) announce(sets, now); else setStatus('Checking…');
    noSetStreak = 0;
  } else {
    noSetStreak++;
    setStatus(noSetStreak >= 3 || isStill ? `No set in these ${live.length} cards — deal 3 more` : 'Checking…');
  }
}

function announce(sets, now) {
  state = 'found';
  found = { sets: sets.map(s => s.map(t => t.tid)), labels: new Map(tracks.map(t => [t.tid, t.label])), at: now, sel: 0, visible: null };
  found.visible = found.sets;
  banner.textContent = sets.length > 1 ? `SET! ×${sets.length}` : 'SET!';
  banner.classList.add('show'); setTimeout(() => banner.classList.remove('show'), 1300);
  playSound();
  if (navigator.vibrate) try { navigator.vibrate(120); } catch {}
  setStatus(`SET!${sets.length > 1 ? `  ${sets.length} sets` : ''}`);
  renderSetList();
}
function setStatus(s) { if (statusEl.textContent !== s) statusEl.textContent = s; }

function renderSetList() {
  const el = $('setList');
  const sets = state === 'found' && found ? found.visible || [] : [];
  const byTid = new Map(tracks.map(t => [t.tid, t]));
  const html = sets.map((s, i) => {
    const sel = found.sets.indexOf(s) === found.sel;
    const names = s.map(tid => { const l = found.labels.get(tid); return `<span style="color:${COLOR_HEX[COLOR[attrsOf(l)[1]]]}">${cardName(l)}</span>`; }).join('');
    return `<button class="setItem${sel ? ' sel' : ''}" data-i="${found.sets.indexOf(s)}"><i style="background:${SET_COLORS[found.sets.indexOf(s) % SET_COLORS.length]}"></i><div class="names">${names}</div></button>`;
  }).join('');
  if (el.dataset.h !== html) { el.innerHTML = html; el.dataset.h = html; }
  $('nextBtn').style.display = state === 'found' ? '' : 'none';
  void byTid;
}
$('setList').addEventListener('click', e => {
  const b = e.target.closest('.setItem'); if (!b || !found) return;
  found.sel = +b.dataset.i; renderSetList(); draw();
});
stage.addEventListener('click', () => {
  if (state !== 'found' || !found.visible || found.visible.length < 2) return;
  const idx = found.visible.map(s => found.sets.indexOf(s)), k = idx.indexOf(found.sel);
  found.sel = idx[(k + 1) % idx.length]; renderSetList(); draw();
});
$('nextBtn').addEventListener('click', () => { if (state === 'found') { state = running ? 'scanning' : 'idle'; found = null; candidateStreak = 0; renderSetList(); draw(); } });

// ---------- sound ----------
let audioCtx = null, customAudio = null;
function unlockAudio() {
  try { audioCtx ||= new (window.AudioContext || window.webkitAudioContext)(); audioCtx.resume(); } catch {}
  if (customAudio) { customAudio.muted = true; customAudio.play().then(() => { customAudio.pause(); customAudio.muted = false; customAudio.currentTime = 0; }).catch(() => {}); }
  if ('speechSynthesis' in window) { const u = new SpeechSynthesisUtterance(' '); u.volume = 0; speechSynthesis.speak(u); }
}
function beep() {
  if (!audioCtx) return;
  const t = audioCtx.currentTime;
  [880, 1320].forEach((f, i) => {
    const o = audioCtx.createOscillator(), g = audioCtx.createGain();
    o.frequency.value = f; o.type = 'square';
    g.gain.setValueAtTime(0.0001, t + i * 0.12); g.gain.exponentialRampToValueAtTime(0.25, t + i * 0.12 + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + i * 0.12 + 0.18);
    o.connect(g).connect(audioCtx.destination); o.start(t + i * 0.12); o.stop(t + i * 0.12 + 0.2);
  });
}
function playSound() {
  const m = settings.soundMode;
  if (m === 'off') return;
  if (m === 'file' && customAudio) { customAudio.currentTime = 0; customAudio.play().catch(beep); return; }
  if (m === 'voice' && 'speechSynthesis' in window) {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance('Set!'); u.rate = 1.1; u.pitch = 1.2; speechSynthesis.speak(u); return;
  }
  beep();
}
const syncSoundRow = () => { $('soundFileRow').style.display = settings.soundMode === 'file' ? '' : 'none'; };
$('soundMode').addEventListener('change', syncSoundRow); syncSoundRow();
$('soundFile').addEventListener('change', e => { const f = e.target.files[0]; if (f) { customAudio = new Audio(URL.createObjectURL(f)); customAudio.preload = 'auto'; } });

// ---------- drawing ----------
let srcW = 1, srcH = 1;
function draw() {
  const dpr = window.devicePixelRatio || 1, cw = stage.clientWidth, ch = stage.clientHeight;
  if (overlay.width !== Math.round(cw * dpr) || overlay.height !== Math.round(ch * dpr)) { overlay.width = Math.round(cw * dpr); overlay.height = Math.round(ch * dpr); }
  octx.setTransform(dpr, 0, 0, dpr, 0, 0); octx.clearRect(0, 0, cw, ch);
  const s = Math.min(cw / srcW, ch / srcH), ox = (cw - srcW * s) / 2, oy = (ch - srcH * s) / 2;
  const P = (x, y) => [ox + x * s, oy + y * s];
  const debug = settings.debug !== 'off';
  if (lastRoi && debug) {
    const [x, y] = P(lastRoi[0], lastRoi[1]);
    octx.setLineDash([6, 6]); octx.lineWidth = 1; octx.strokeStyle = '#ffffff50'; octx.strokeRect(x, y, lastRoi[2] * s, lastRoi[3] * s); octx.setLineDash([]);
  }
  // which sets each card belongs to (for nested coloured outlines)
  const member = new Map();
  if (state === 'found' && found?.visible) {
    const order = [...found.visible].sort((a, b) => (found.sets.indexOf(a) === found.sel) - (found.sets.indexOf(b) === found.sel));
    for (const st of order) for (const tid of st) { if (!member.has(tid)) member.set(tid, []); member.get(tid).push(found.sets.indexOf(st)); }
  }
  for (const t of tracks) {
    const [x1, y1] = P(t.box[0], t.box[1]), [x2, y2] = P(t.box[2], t.box[3]), w = x2 - x1, h = y2 - y1;
    const sets = member.get(t.tid) || [];
    octx.setLineDash(t.seen ? [] : [4, 4]);
    if (!sets.length) {
      octx.lineWidth = 1.5; octx.strokeStyle = t.cert >= CERT && t.verified ? '#ffffffa0' : '#ffffff45';
      roundRect(x1, y1, w, h, Math.min(w, h) * 0.08); octx.stroke();
    }
    sets.forEach((si, k) => {
      const inset = k * 5, sel = si === found.sel;
      octx.lineWidth = sel ? 6 : 3.5; octx.strokeStyle = SET_COLORS[si % SET_COLORS.length];
      if (sel) { octx.shadowColor = octx.strokeStyle; octx.shadowBlur = 16; }
      roundRect(x1 - 3 + inset, y1 - 3 + inset, w + 6 - 2 * inset, h + 6 - 2 * inset, Math.min(w, h) * 0.08); octx.stroke();
      octx.shadowBlur = 0;
    });
    octx.setLineDash([]);
    if (debug && t.quad) {
      octx.beginPath(); t.quad.forEach(([x, y], i) => { const [a, b] = P(x, y); i ? octx.lineTo(a, b) : octx.moveTo(a, b); }); octx.closePath();
      octx.lineWidth = 1; octx.strokeStyle = t.rectified ? '#00ffaa' : '#ff9900'; octx.stroke();
    }
    const fs = Math.min(15, w / 6.5);
    if (settings.showLabels === '1' && t.nObs && fs >= 8) {
      octx.font = `600 ${fs}px system-ui, sans-serif`;
      const txt = cardName(t.label) + (t.cert >= CERT && t.verified ? '' : ' ?'), col = COLOR_HEX[COLOR[attrsOf(t.label)[1]]];
      const tw = Math.min(octx.measureText(txt).width + 10, w + 10);
      octx.fillStyle = '#000000b0'; octx.fillRect(x1, y2 + 2, tw, fs + 6);
      octx.fillStyle = col; octx.fillText(txt, x1 + 5, y2 + fs + 2, w);
    }
  }
}
function roundRect(x, y, w, h, r) {
  octx.beginPath(); octx.moveTo(x + r, y); octx.arcTo(x + w, y, x + w, y + h, r); octx.arcTo(x + w, y + h, x, y + h, r);
  octx.arcTo(x, y + h, x, y, r); octx.arcTo(x, y, x + w, y, r); octx.closePath();
}

// ---------- main loop ----------
let running = false, loopToken = 0, lastThumb = null, lastDetAt = 0, fps = 0;
function thumbDiff() {
  thumbCtx.drawImage(frameCanvas, 0, 0, 64, 48);
  const d = thumbCtx.getImageData(0, 0, 64, 48).data, g = new Uint8Array(64 * 48);
  for (let i = 0; i < g.length; i++) g[i] = (d[i * 4] + 2 * d[i * 4 + 1] + d[i * 4 + 2]) >> 2;
  let diff = 255;
  if (lastThumb) { let s = 0; for (let i = 0; i < g.length; i++) s += Math.abs(g[i] - lastThumb[i]); diff = s / g.length; }
  return { g, diff };
}

async function step(src, W, H, isStill) {
  const now = performance.now();
  stats.frames++;
  srcW = W; srcH = H;
  let didDet = false;
  stillMode = isStill; CERT = isStill ? CERT_STILL : CERT_LIVE;
  if (isStill) {
    await getDetector(640);
    tracks = []; const dets = await detectStill(src, W, H);
    for (const d of dets) tracks.push(newTrack(d.box, now));
    didDet = true;
  } else {
    const { g, diff } = thumbDiff();
    // run the detector unless the picture hasn't changed and every card is settled
    const busy = tracks.some(t => !t.seen || t.nDet < 3 || !t.staticSince);
    if (diff > 2.5 || busy || now - lastDetAt > 1000 || !tracks.length) {
      const dets = await detectTracked(src, W, H, now);
      updateTracks(dets, now); lastThumb = g; lastDetAt = now; didDet = true; stats.dets++;
    } else stats.skipped++;
  }
  const jobs = schedule(now, isStill);
  if (jobs.length) {
    const res = await classify(src, jobs);
    jobs.forEach((j, i) => { addEvidence(j.t, res[i]); j.t.lastCls = now; j.t.quad = j.quad; j.t.rectified = j.rectified; });
  }
  evaluate(now, isStill);
  draw();
  window.__last = {
    state, cards: tracks.map(t => ({ tid: t.tid, box: t.box.map(Math.round), name: t.nObs ? cardName(t.label) : null, cert: +t.cert.toFixed(3), seen: t.seen, rect: t.rectified })),
    sets: (found?.visible || []).map(s => s.map(tid => cardName(found.labels.get(tid)))), stats: { ...stats }, detSize: currentDetSize(),
  };
  return { didDet, jobs: jobs.length };
}

async function loop(token) {
  while (running && token === loopToken) {
    if (video.readyState < 2 || !video.videoWidth) { await sleep(50); continue; }
    const W = video.videoWidth, H = video.videoHeight;
    if (frameCanvas.width !== W || frameCanvas.height !== H) { frameCanvas.width = W; frameCanvas.height = H; }
    frameCtx.drawImage(video, 0, 0, W, H);
    const t0 = performance.now();
    try {
      const r = await step(frameCanvas, W, H, false);
      const dt = performance.now() - t0;
      if (r.didDet || r.jobs) fps = fps ? fps * 0.8 + (1000 / dt) * 0.2 : 1000 / dt;
      statsEl.textContent = `${tracks.filter(t => t.seen).length} cards · ${fps.toFixed(1)} fps\ndet ${currentDetSize()}${settings.detSize === 'auto' ? ' auto' : ''} · ${r.didDet ? lastDetMs.toFixed(0) + ' ms' : 'idle'} · cls ${r.jobs}`;
      if (!r.didDet && !r.jobs) await sleep(60);
    } catch (e) { console.error(e); setStatus('Error: ' + e.message); await sleep(500); }
    await sleep(0);
  }
}

// ---------- camera ----------
let stream = null, track = null, cameras = [], camId = null, torchOn = false;
try { camId = localStorage.getItem('set.camId'); } catch {}
function lensLabel(cam, i) {
  const l = cam.label.toLowerCase();
  if (/ultra/.test(l)) return '0.5×';
  if (/tele/.test(l)) return 'Tele';
  if (/triple|dual/.test(l)) return 'Auto';
  if (/^(back camera|rückkamera|camera achter)$/i.test(cam.label.trim())) return '1×';
  return `Lens ${i + 1}`;
}
async function listCameras() {
  const all = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
  const back = all.filter(d => !/front|user|face|voor/i.test(d.label));
  cameras = back.length ? back : all;
  const sel = $('camSelect'); sel.innerHTML = '';
  all.forEach(d => { const o = document.createElement('option'); o.value = d.deviceId; o.textContent = d.label || ('camera ' + d.deviceId.slice(0, 6)); sel.appendChild(o); });
  const lb = $('lensButtons'); lb.innerHTML = '';
  if (cameras.length > 1) cameras.forEach((c, i) => {
    const b = document.createElement('button'); b.textContent = lensLabel(c, i); b.title = c.label; b.dataset.id = c.deviceId;
    b.onclick = () => openCamera(c.deviceId).catch(e => setStatus(e.message)); lb.appendChild(b);
  });
}
function markActiveCamera() {
  const id = track && track.getSettings().deviceId;
  $('camSelect').value = id || '';
  for (const b of $('lensButtons').children) b.classList.toggle('on', b.dataset.id === id);
}
async function openCamera(deviceId) {
  if (stream) stream.getTracks().forEach(t => t.stop());
  torchOn = false;
  const h = +settings.camRes, base = { width: { ideal: Math.round(h * 16 / 9) }, height: { ideal: h }, frameRate: { ideal: 30 } };
  const c = deviceId ? { ...base, deviceId: { exact: deviceId } } : { ...base, facingMode: { ideal: 'environment' } };
  stream = await navigator.mediaDevices.getUserMedia({ video: c, audio: false });
  track = stream.getVideoTracks()[0];
  video.srcObject = stream;
  await video.play().catch(() => {});
  camId = track.getSettings().deviceId;
  try { localStorage.setItem('set.camId', camId); } catch {}
  setupTrackControls(); markActiveCamera();
  tracks = []; roi = null; lastThumb = null;
}
function setupTrackControls() {
  const caps = track.getCapabilities ? track.getCapabilities() : {};
  if (caps.focusMode && caps.focusMode.includes('continuous')) track.applyConstraints({ advanced: [{ focusMode: 'continuous' }] }).catch(() => {});
  $('torchBtn').style.display = caps.torch ? '' : 'none';
  $('torchBtn').classList.remove('on');
  const zr = $('zoomRow'), z = $('zoom');
  if (caps.zoom && caps.zoom.max > caps.zoom.min) {
    zr.style.display = 'flex';
    z.min = caps.zoom.min; z.max = Math.min(caps.zoom.max, 4); z.step = caps.zoom.step || 0.1;
    const cur = track.getSettings().zoom ?? 1, start = caps.zoom.min < 1 ? caps.zoom.min : cur;
    z.value = start; $('zoomVal').textContent = (+start).toFixed(1) + '×';
    if (start !== cur) track.applyConstraints({ advanced: [{ zoom: start }] }).catch(() => {});
  } else zr.style.display = 'none';
  $('camInfo').textContent = (() => { const s = track.getSettings(); return `${s.width}×${s.height}`; })();
}
$('zoom').addEventListener('input', e => {
  const v = +e.target.value; $('zoomVal').textContent = v.toFixed(1) + '×';
  track?.applyConstraints({ advanced: [{ zoom: v }] }).catch(() => {});
});
$('torchBtn').addEventListener('click', async () => {
  if (!track) return;
  torchOn = !torchOn;
  try { await track.applyConstraints({ advanced: [{ torch: torchOn }] }); }
  catch { torchOn = false; setStatus('Flash not available on this lens'); }
  $('torchBtn').classList.toggle('on', torchOn);
});
$('camSelect').addEventListener('change', e => openCamera(e.target.value).catch(err => setStatus(err.message)));
$('camRes').addEventListener('change', () => stream && openCamera(camId).catch(err => setStatus(err.message)));

let wakeLock = null;
async function keepAwake() { try { wakeLock = await navigator.wakeLock?.request('screen'); } catch {} }
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && running) keepAwake(); });

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera API not available. Open this page over https:// (or localhost).');
  await openCamera(camId);
  await listCameras();
  let chosen = null; try { chosen = localStorage.getItem('set.chosenOnce'); } catch {}
  if (!chosen) {
    const uw = cameras.find(c => /ultra/i.test(c.label));
    if (uw && uw.deviceId !== camId) await openCamera(uw.deviceId);
    try { localStorage.setItem('set.chosenOnce', '1'); } catch {}
  }
  markActiveCamera();
}
function startScan() {
  unlockAudio();
  still.style.display = 'none'; video.style.display = '';
  state = 'scanning'; found = null; candidateStreak = 0; tracks = []; roi = null; lastThumb = null;
  running = true; loopToken++; loop(loopToken);
  $('startBtn').textContent = 'Stop'; $('startBtn').classList.add('running');
  keepAwake(); renderSetList();
}
function stopScan() {
  running = false; if (state === 'scanning') state = 'idle';
  $('startBtn').textContent = 'Start'; $('startBtn').classList.remove('running');
  if (state !== 'found') setStatus('Paused');
  wakeLock?.release?.().catch(() => {}); wakeLock = null;
}
$('startBtn').addEventListener('click', async () => {
  if (running) return stopScan();
  try { if (!stream) await startCamera(); await modelsReady; startScan(); } catch (e) { setStatus(e.message); }
});
$('gearBtn').addEventListener('click', () => $('settings').classList.toggle('open'));
$('closeSettings').addEventListener('click', () => $('settings').classList.remove('open'));
$('detSize').addEventListener('change', () => { if (settings.detSize !== 'auto') getDetector(+settings.detSize); });
$('introBtn').addEventListener('click', async () => {
  try {
    $('introBtn').disabled = true; $('introBtn').textContent = 'Starting…';
    await startCamera(); await modelsReady;
    $('intro').style.display = 'none'; startScan();
  } catch (e) {
    $('err').textContent = (e.name === 'NotAllowedError' ? 'Camera permission was denied.' : e.message) + '\n\nThe camera only works on https:// pages (or localhost).';
    $('introBtn').disabled = false; $('introBtn').textContent = 'Start camera';
  }
});

// ---------- photo mode ----------
async function runOnPhoto(file) {
  stopScan(); unlockAudio();
  const img = await createImageBitmap(file, { imageOrientation: 'from-image' });
  still.width = img.width; still.height = img.height;
  still.getContext('2d', { willReadFrequently: true }).drawImage(img, 0, 0);
  still.style.display = 'block'; video.style.display = 'none';
  $('intro').style.display = 'none';
  await modelsReady;
  state = 'scanning'; found = null;
  const t0 = performance.now();
  await step(still, img.width, img.height, true);
  statsEl.textContent = `${tracks.length} cards · ${(performance.now() - t0).toFixed(0)} ms`;
}
const photoPicker = Object.assign(document.createElement('input'), { type: 'file', accept: 'image/*' });
photoPicker.addEventListener('change', e => e.target.files[0] && runOnPhoto(e.target.files[0]));
$('introPhoto').addEventListener('click', e => { e.preventDefault(); photoPicker.click(); });
$('photoFile').addEventListener('change', e => { e.target.files[0] && runOnPhoto(e.target.files[0]); $('settings').classList.remove('open'); });
window.addEventListener('resize', () => draw());
new ResizeObserver(() => draw()).observe(stage);
