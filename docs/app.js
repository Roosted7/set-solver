// SET! — on-device Set solver.
// Models: YOLO11n card detector + MobileNetV3-small 4-head attribute classifier,
// both from wangtian24/set-solver (MIT), converted to ONNX and run with onnxruntime-web.

const ORT_VER = '1.31.1';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
const ort = await import(ORT_BASE + 'ort.webgpu.min.mjs');
ort.env.wasm.wasmPaths = ORT_BASE;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;

// ---------- constants ----------
const DET_CONF = 0.5, NMS_IOU = 0.5, CLS_CONF = 0.8, CLS_SIZE = 224;
const NUMBER = ['1', '2', '3'];
const COLOR = ['red', 'green', 'purple'];
const SHAPE = ['diamond', 'oval', 'squiggle'];
const FILL = ['empty', 'solid', 'striped'];
const COLOR_HEX = { red: '#ff4d4d', green: '#36d16b', purple: '#b56cff' };
const SHAPE_SYM = { diamond: '◆', oval: '⬮', squiggle: '∿' };
const FILL_SYM = { empty: 'open', solid: 'solid', striped: 'striped' };
const attrsOf = id => [Math.floor(id / 27), Math.floor(id / 9) % 3, Math.floor(id / 3) % 3, id % 3];
const cardName = id => { const [n, c, s, f] = attrsOf(id); return `${NUMBER[n]} ${FILL_SYM[FILL[f]]} ${COLOR[c]} ${SHAPE_SYM[SHAPE[s]]}`; };
const isSet = (a, b, c) => { const A = attrsOf(a), B = attrsOf(b), C = attrsOf(c); return A.every((v, i) => (v + B[i] + C[i]) % 3 === 0); };

// ---------- DOM ----------
const $ = id => document.getElementById(id);
const video = $('video'), still = $('still'), overlay = $('overlay'), octx = overlay.getContext('2d');
const statusEl = $('status'), statsEl = $('stats'), banner = $('banner');
const settings = {
  get minCards() { return +$('minCards').value; },
  get detSize() { return +$('detSize').value; },
  get showLabels() { return $('showLabels').value === '1'; },
  get soundMode() { return $('soundMode').value; },
};
// remember settings (best effort)
for (const id of ['minCards', 'detSize', 'showLabels', 'soundMode']) {
  try { const v = localStorage.getItem('set.' + id); if (v) $(id).value = v; } catch {}
  $(id).addEventListener('change', () => { try { localStorage.setItem('set.' + id, $(id).value); } catch {} });
}

// ---------- models ----------
const sessions = {};
let backend = '?';
async function makeSession(url) {
  const eps = ('gpu' in navigator) ? ['webgpu', 'wasm'] : ['wasm'];
  for (const ep of eps) {
    try {
      const s = await ort.InferenceSession.create(url, { executionProviders: [ep], graphOptimizationLevel: 'all' });
      backend = ep; return s;
    } catch (e) { console.warn('EP failed', ep, e); }
  }
  throw new Error('Could not load ' + url);
}
async function getDetector(size) {
  if (!sessions['det' + size]) sessions['det' + size] = makeSession(`models/detector_${size}.onnx`);
  return sessions['det' + size];
}
async function getClassifier() {
  if (!sessions.cls) sessions.cls = makeSession('models/classifier.onnx');
  return sessions.cls;
}
const modelsReady = Promise.all([getDetector(settings.detSize), getClassifier()]).then(() => {
  $('loadMsg').textContent = `Models loaded (${backend}).`;
  $('backendInfo').textContent = `Inference backend: ${backend}${ort.env.wasm.numThreads > 1 ? `, ${ort.env.wasm.numThreads} threads` : ''}. All processing is on this device.`;
}).catch(e => { $('err').textContent = 'Model load failed: ' + e.message; throw e; });

// ---------- image helpers ----------
const frameCanvas = document.createElement('canvas');
const frameCtx = frameCanvas.getContext('2d', { willReadFrequently: false });
const detCanvas = document.createElement('canvas');
const detCtx = detCanvas.getContext('2d', { willReadFrequently: true });
const cropCanvas = document.createElement('canvas');
cropCanvas.width = cropCanvas.height = CLS_SIZE;
const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });
cropCtx.imageSmoothingQuality = 'high';

function toCHW(imgData, out, offset) {
  const d = imgData.data, n = imgData.width * imgData.height;
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[offset + i] = d[j] / 255;
    out[offset + n + i] = d[j + 1] / 255;
    out[offset + 2 * n + i] = d[j + 2] / 255;
  }
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]), x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}

// ---------- detection ----------
async function detect(src, W, H, roi) {
  const S = settings.detSize;
  const det = await getDetector(S);
  detCanvas.width = detCanvas.height = S;
  const [rx, ry, rw, rh] = roi || [0, 0, W, H];
  const r = Math.min(S / rw, S / rh), nw = Math.round(rw * r), nh = Math.round(rh * r);
  const dx = Math.floor((S - nw) / 2) - rx * r, dy = Math.floor((S - nh) / 2) - ry * r;
  detCtx.fillStyle = 'rgb(114,114,114)'; detCtx.fillRect(0, 0, S, S);
  detCtx.drawImage(src, rx, ry, rw, rh, Math.floor((S - nw) / 2), Math.floor((S - nh) / 2), nw, nh);
  const input = new Float32Array(3 * S * S);
  toCHW(detCtx.getImageData(0, 0, S, S), input, 0);
  const out = await det.run({ [det.inputNames[0]]: new ort.Tensor('float32', input, [1, 3, S, S]) });
  const t = out[det.outputNames[0]], d = t.data, N = t.dims[2];
  let boxes = [];
  for (let i = 0; i < N; i++) {
    const sc = d[4 * N + i];
    if (sc < DET_CONF) continue;
    const cx = d[i], cy = d[N + i], w = d[2 * N + i], h = d[3 * N + i];
    boxes.push({
      box: [(cx - w / 2 - dx) / r, (cy - h / 2 - dy) / r, (cx + w / 2 - dx) / r, (cy + h / 2 - dy) / r].map((v, k) => Math.max(0, Math.min(k % 2 ? H : W, v))),
      score: sc,
    });
  }
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of boxes) if (keep.every(k => iou(k.box, b.box) < NMS_IOU)) keep.push(b);
  // drop boxes that merged two cards (same rule as the original repo)
  if (keep.length >= 3) {
    const areas = keep.map(b => (b.box[2] - b.box[0]) * (b.box[3] - b.box[1])).sort((a, b) => a - b);
    const med = areas[Math.floor(areas.length / 2)];
    return keep.filter(b => (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) <= med * 2.2);
  }
  return keep;
}

// ---------- classification ----------
async function classify(src, boxes) {
  if (!boxes.length) return [];
  const cls = await getClassifier();
  const n = boxes.length, px = CLS_SIZE * CLS_SIZE;
  const input = new Float32Array(n * 3 * px);
  boxes.forEach((b, i) => {
    const [x1, y1, x2, y2] = b.box.map(Math.round);
    cropCtx.drawImage(src, x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1), 0, 0, CLS_SIZE, CLS_SIZE);
    toCHW(cropCtx.getImageData(0, 0, CLS_SIZE, CLS_SIZE), input, i * 3 * px);
  });
  const out = await cls.run({ [cls.inputNames[0]]: new ort.Tensor('float32', input, [n, 3, CLS_SIZE, CLS_SIZE]) });
  const L = out[cls.outputNames[0]].data;
  const res = [];
  for (let i = 0; i < n; i++) {
    const idx = [], conf = [];
    for (let h = 0; h < 4; h++) {
      const o = i * 12 + h * 3, m = Math.max(L[o], L[o + 1], L[o + 2]);
      const e = [0, 1, 2].map(k => Math.exp(L[o + k] - m)), s = e[0] + e[1] + e[2];
      const k = e.indexOf(Math.max(...e)); idx.push(k); conf.push(e[k] / s);
    }
    res.push({ id: idx[0] * 27 + idx[1] * 9 + idx[2] * 3 + idx[3], conf: Math.min(...conf) });
  }
  return res;
}

// ---------- tracking + voting ----------
let tracks = [], nextTrackId = 1, frameNo = 0;
const HIST = 6;
function stableLabel(t) {
  const counts = {};
  let valid = 0;
  for (const v of t.hist) if (v >= 0) { counts[v] = (counts[v] || 0) + 1; valid++; }
  let best = -1, bc = 0;
  for (const k in counts) if (counts[k] > bc) { bc = counts[k]; best = +k; }
  return bc >= 2 && bc * 2 > valid ? best : -1;
}
function matchTracks(dets) {
  const used = new Set();
  const pairs = [];
  dets.forEach((d, i) => tracks.forEach((t, j) => { const v = iou(d.box, t.box); if (v > 0.25) pairs.push([v, i, j]); }));
  pairs.sort((a, b) => b[0] - a[0]);
  const detTrack = new Array(dets.length).fill(null), takenT = new Set();
  for (const [, i, j] of pairs) if (!used.has(i) && !takenT.has(j)) { used.add(i); takenT.add(j); detTrack[i] = tracks[j]; }
  dets.forEach((d, i) => {
    if (!detTrack[i]) { const t = { tid: nextTrackId++, box: d.box, hist: [], lastSeen: frameNo }; tracks.push(t); detTrack[i] = t; }
    detTrack[i].box = d.box; detTrack[i].lastSeen = frameNo;
  });
  tracks = tracks.filter(t => frameNo - t.lastSeen <= 3);
  return detTrack;
}

// ---------- set logic / UI state ----------
let state = 'idle';            // idle | scanning | found
let found = null;              // {ids:[a,b,c], at:time}
let candidateKey = '', candidateStreak = 0, noSetStreak = 0, missingSince = 0;
let allSets = [], setIdx = 0;

function findSets(cards) {
  const sets = [];
  for (let i = 0; i < cards.length; i++) for (let j = i + 1; j < cards.length; j++) for (let k = j + 1; k < cards.length; k++)
    if (isSet(cards[i].label, cards[j].label, cards[k].label)) sets.push([cards[i], cards[j], cards[k]]);
  return sets;
}

function evaluate(live, isStill) {
  const labeled = live.filter(t => t.label >= 0);
  // de-duplicate identical labels (a misread can't be the same card twice)
  const seen = new Set(), cards = [];
  for (const t of labeled) if (!seen.has(t.label)) { seen.add(t.label); cards.push(t); }
  const enough = live.length >= settings.minCards;
  const allRead = labeled.length === live.length;
  allSets = findSets(cards);

  if (state === 'found') {
    const present = found.ids.filter(id => cards.some(c => c.label === id)).length;
    if (present < 3) { if (!missingSince) missingSince = performance.now(); }
    else missingSince = 0;
    if (missingSince && performance.now() - missingSince > 1200) { state = 'scanning'; found = null; missingSince = 0; candidateStreak = 0; }
  }
  if (state === 'scanning') {
    if (!enough) { setStatus(live.length ? `Waiting for cards… ${live.length}/${settings.minCards}` : 'Looking for cards…'); candidateStreak = 0; noSetStreak = 0; }
    else if (!allRead && !isStill) setStatus(`Reading cards… ${labeled.length}/${live.length}`);
    else if (allSets.length) {
      const key = allSets[0].map(c => c.label).sort((a, b) => a - b).join(',');
      candidateStreak = key === candidateKey ? candidateStreak + 1 : 1; candidateKey = key;
      if (candidateStreak >= 2 || isStill) {
        setIdx = 0; announce(allSets[0]);
      } else setStatus('Checking…');
      noSetStreak = 0;
    } else {
      noSetStreak++;
      setStatus(noSetStreak >= 3 || isStill ? `No set among ${cards.length} cards — deal 3 more` : 'Checking…');
    }
  }
  if (state === 'found') setStatus(`SET! ${allSets.length > 1 ? `(tap for next of ${allSets.length})` : ''}`);
}

function announce(set) {
  state = 'found';
  found = { ids: set.map(c => c.label), at: performance.now() };
  banner.classList.add('show'); setTimeout(() => banner.classList.remove('show'), 1100);
  playSound();
  if (navigator.vibrate) try { navigator.vibrate(120); } catch {}
}

function setStatus(s) { if (statusEl.textContent !== s) statusEl.textContent = s; }

// tap to cycle through sets
$('stage').addEventListener('click', () => {
  if (state !== 'found' || allSets.length < 2) return;
  setIdx = (setIdx + 1) % allSets.length;
  found = { ids: allSets[setIdx].map(c => c.label), at: performance.now() };
  draw(lastLive, lastSrcW, lastSrcH);
});

// ---------- sound ----------
let audioCtx = null, customAudio = null;
function unlockAudio() {
  try { audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)(); audioCtx.resume(); } catch {}
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
    const u = new SpeechSynthesisUtterance('Set!'); u.rate = 1.1; u.pitch = 1.2; u.volume = 1;
    speechSynthesis.speak(u); return;
  }
  beep();
}
$('soundMode').addEventListener('change', () => { $('soundFileRow').style.display = settings.soundMode === 'file' ? '' : 'none'; });
$('soundFileRow').style.display = settings.soundMode === 'file' ? '' : 'none';
$('soundFile').addEventListener('change', e => {
  const f = e.target.files[0]; if (!f) return;
  customAudio = new Audio(URL.createObjectURL(f)); customAudio.preload = 'auto';
});

// ---------- drawing ----------
let lastLive = [], lastSrcW = 1, lastSrcH = 1;
function draw(live, W, H) {
  lastLive = live; lastSrcW = W; lastSrcH = H;
  const dpr = window.devicePixelRatio || 1;
  const cw = overlay.clientWidth, ch = overlay.clientHeight;
  if (overlay.width !== Math.round(cw * dpr) || overlay.height !== Math.round(ch * dpr)) { overlay.width = Math.round(cw * dpr); overlay.height = Math.round(ch * dpr); }
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, cw, ch);
  const s = Math.min(cw / W, ch / H), ox = (cw - W * s) / 2, oy = (ch - H * s) / 2;
  const map = b => [ox + b[0] * s, oy + b[1] * s, (b[2] - b[0]) * s, (b[3] - b[1]) * s];
  if (lastRoi && running) {
    const [x, y, w, h] = map([lastRoi[0], lastRoi[1], lastRoi[0] + lastRoi[2], lastRoi[1] + lastRoi[3]]);
    octx.setLineDash([6, 6]); octx.lineWidth = 1; octx.strokeStyle = '#ffffff40'; octx.strokeRect(x, y, w, h); octx.setLineDash([]);
  }
  const inSet = new Set(state === 'found' && found ? found.ids : []);
  for (const t of live) {
    const [x, y, w, h] = map(t.box);
    const hit = t.label >= 0 && inSet.has(t.label);
    octx.lineWidth = hit ? 6 : 2;
    octx.strokeStyle = hit ? '#ffd400' : (t.label >= 0 ? '#ffffffb0' : '#ffffff55');
    if (hit) { octx.shadowColor = '#ffd400'; octx.shadowBlur = 18; }
    roundRect(x, y, w, h, Math.min(w, h) * 0.08); octx.stroke();
    octx.shadowBlur = 0;
    // label scaled to the card; hidden when cards are too small on screen to fit one
    const fs = Math.min(15, w / 6.5);
    if (settings.showLabels && t.label >= 0 && fs >= 8) {
      octx.font = `600 ${fs}px system-ui, sans-serif`;
      const txt = cardName(t.label), col = COLOR_HEX[COLOR[attrsOf(t.label)[1]]];
      const tw = octx.measureText(txt).width + 10;
      octx.fillStyle = '#000000b0'; octx.fillRect(x, y + h + 2, Math.min(tw, w + 10), fs + 6);
      octx.fillStyle = col; octx.fillText(txt, x + 5, y + h + fs + 2, w);
    }
  }
}
function roundRect(x, y, w, h, r) {
  octx.beginPath(); octx.moveTo(x + r, y); octx.arcTo(x + w, y, x + w, y + h, r); octx.arcTo(x + w, y + h, x, y + h, r);
  octx.arcTo(x, y + h, x, y, r); octx.arcTo(x, y, x + w, y, r); octx.closePath();
}

// ---------- region of interest ----------
// The detector was trained on photos where the cards fill most of the frame, so with a
// wide-angle lens we detect on a crop around the cards (and search centre crops when lost).
const SEARCH = [1, 0.7, 0.5, 0.36];
let roi = null, missFrames = 0, searchIdx = 0;
const centreRoi = (W, H, f) => [W * (1 - f) / 2, H * (1 - f) / 2, W * f, H * f];
function roiAround(boxes, W, H, margin) {
  const ws = boxes.map(b => b.box[2] - b.box[0]).sort((a, b) => a - b), hs = boxes.map(b => b.box[3] - b.box[1]).sort((a, b) => a - b);
  const mw = ws[ws.length >> 1], mh = hs[hs.length >> 1];
  let x1 = Math.min(...boxes.map(b => b.box[0])) - margin * mw, y1 = Math.min(...boxes.map(b => b.box[1])) - margin * mh;
  let x2 = Math.max(...boxes.map(b => b.box[2])) + margin * mw, y2 = Math.max(...boxes.map(b => b.box[3])) + margin * mh;
  // never zoom in further than ~4 card widths, so the cards stay a "normal" size for the detector
  const minW = 4.5 * mw, minH = 4.5 * mh;
  if (x2 - x1 < minW) { const c = (x1 + x2) / 2; x1 = c - minW / 2; x2 = c + minW / 2; }
  if (y2 - y1 < minH) { const c = (y1 + y2) / 2; y1 = c - minH / 2; y2 = c + minH / 2; }
  x1 = Math.max(0, x1); y1 = Math.max(0, y1); x2 = Math.min(W, x2); y2 = Math.min(H, y2);
  return [x1, y1, x2 - x1, y2 - y1];
}
async function detectTracked(src, W, H) {
  // every 6th frame look in a wider window so newly dealt cards further away are found
  const wide = roi && frameNo % 6 === 0;
  const r = roi ? (wide ? grow(roi, W, H, 1.6) : roi) : centreRoi(W, H, SEARCH[searchIdx]);
  const dets = await detect(src, W, H, r);
  lastRoi = r;
  if (dets.length) { missFrames = 0; roi = roiAround(dets, W, H, 1.3); }
  else if (++missFrames >= 2) { roi = null; searchIdx = (searchIdx + 1) % SEARCH.length; missFrames = 0; }
  return dets;
}
function grow([x, y, w, h], W, H, f) {
  const nw = Math.min(W, w * f), nh = Math.min(H, h * f);
  const nx = Math.max(0, Math.min(W - nw, x - (nw - w) / 2)), ny = Math.max(0, Math.min(H - nh, y - (nh - h) / 2));
  return [nx, ny, nw, nh];
}
async function detectStill(src, W, H) {
  let best = [];
  for (const f of SEARCH) { const d = await detect(src, W, H, centreRoi(W, H, f)); if (d.length > best.length) best = d; if (best.length >= 9) break; }
  if (best.length) { const r = roiAround(best, W, H, 1.3); const d = await detect(src, W, H, r); if (d.length >= best.length) best = d; lastRoi = r; }
  return best;
}
let lastRoi = null;

// ---------- pipeline ----------
async function processFrame(src, W, H, isStill) {
  frameNo++;
  const t0 = performance.now();
  const dets = isStill ? await detectStill(src, W, H) : await detectTracked(src, W, H);
  const t1 = performance.now();
  let live;
  if (isStill) {
    const res = await classify(src, dets);
    live = dets.map((d, i) => ({ box: d.box, label: res[i].conf >= CLS_CONF ? res[i].id : -1, conf: res[i].conf }));
  } else {
    const dt = matchTracks(dets);
    // re-classify unlocked tracks every frame, locked ones every 4th frame
    const todo = dets.map((d, i) => i).filter(i => { const t = dt[i]; return stableLabel(t) < 0 || t.hist.length < 3 || (frameNo + t.tid) % 4 === 0; });
    const res = await classify(src, todo.map(i => dets[i]));
    todo.forEach((i, k) => { const t = dt[i]; t.hist.push(res[k].conf >= CLS_CONF ? res[k].id : -1); if (t.hist.length > HIST) t.hist.shift(); });
    live = dt.map(t => ({ box: t.box, label: stableLabel(t) }));
  }
  const t2 = performance.now();
  evaluate(live, isStill);
  draw(live, W, H);
  window.__last = { cards: live.map(c => ({ box: c.box.map(Math.round), label: c.label, name: c.label >= 0 ? cardName(c.label) : null })), sets: allSets.map(s => s.map(c => cardName(c.label))), state, ms: { det: t1 - t0, cls: t2 - t1 } };
  return { det: t1 - t0, cls: t2 - t1 };
}

let running = false, loopToken = 0, fpsAvg = 0;
async function loop(token) {
  while (running && token === loopToken) {
    if (video.readyState < 2 || !video.videoWidth) { await sleep(50); continue; }
    const W = video.videoWidth, H = video.videoHeight;
    if (frameCanvas.width !== W || frameCanvas.height !== H) { frameCanvas.width = W; frameCanvas.height = H; }
    frameCtx.drawImage(video, 0, 0, W, H);
    const t0 = performance.now();
    try {
      const ms = await processFrame(frameCanvas, W, H, false);
      const dt = performance.now() - t0;
      fpsAvg = fpsAvg ? fpsAvg * 0.8 + (1000 / dt) * 0.2 : 1000 / dt;
      statsEl.textContent = `${lastLive.length} cards · ${fpsAvg.toFixed(1)} fps\ndet ${ms.det.toFixed(0)} · cls ${ms.cls.toFixed(0)} ms`;
      statsEl.style.whiteSpace = 'pre';
    } catch (e) { console.error(e); setStatus('Error: ' + e.message); await sleep(500); }
    await sleep(0);
  }
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---------- camera ----------
let stream = null, track = null, cameras = [], camId = null, torchOn = false;
try { camId = localStorage.getItem('set.camId'); } catch {}

function lensLabel(cam, i, backCount) {
  const l = cam.label.toLowerCase();
  if (/ultra/.test(l)) return '0.5×';
  if (/tele/.test(l)) return 'Tele';
  if (/triple|dual/.test(l)) return 'Auto';
  if (/^back camera$|^rückkamera$|^camera achter$/i.test(cam.label.trim())) return '1×';
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
    const b = document.createElement('button'); b.textContent = lensLabel(c, i, cameras.length); b.title = c.label; b.dataset.id = c.deviceId;
    b.onclick = () => openCamera(c.deviceId); lb.appendChild(b);
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
  const base = { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } };
  const c = deviceId ? { ...base, deviceId: { exact: deviceId } } : { ...base, facingMode: { ideal: 'environment' } };
  stream = await navigator.mediaDevices.getUserMedia({ video: c, audio: false });
  track = stream.getVideoTracks()[0];
  video.srcObject = stream;
  await video.play().catch(() => {});
  camId = track.getSettings().deviceId;
  try { localStorage.setItem('set.camId', camId); } catch {}
  setupTrackControls();
  markActiveCamera();
  tracks = []; roi = null;
}

function setupTrackControls() {
  const caps = track.getCapabilities ? track.getCapabilities() : {};
  // continuous autofocus where offered
  if (caps.focusMode && caps.focusMode.includes('continuous')) track.applyConstraints({ advanced: [{ focusMode: 'continuous' }] }).catch(() => {});
  // flash
  $('torchBtn').style.display = caps.torch ? '' : 'none';
  $('torchBtn').classList.remove('on');
  // zoom (on some Android phones zoom < 1 switches to the ultra-wide lens)
  const zr = $('zoomRow'), z = $('zoom');
  if (caps.zoom && caps.zoom.max > caps.zoom.min) {
    zr.style.display = 'flex';
    z.min = caps.zoom.min; z.max = Math.min(caps.zoom.max, 4); z.step = caps.zoom.step || 0.1;
    const cur = track.getSettings().zoom ?? 1;
    const start = caps.zoom.min < 1 ? caps.zoom.min : cur;
    z.value = start; $('zoomVal').textContent = (+start).toFixed(1) + '×';
    if (start !== cur) track.applyConstraints({ advanced: [{ zoom: start }] }).catch(() => {});
  } else zr.style.display = 'none';
}
$('zoom').addEventListener('input', e => {
  const v = +e.target.value; $('zoomVal').textContent = v.toFixed(1) + '×';
  track && track.applyConstraints({ advanced: [{ zoom: v }] }).catch(() => {});
});
$('torchBtn').addEventListener('click', async () => {
  if (!track) return;
  torchOn = !torchOn;
  try { await track.applyConstraints({ advanced: [{ torch: torchOn }] }); }
  catch (e) { torchOn = false; setStatus('Flash not available on this lens'); }
  $('torchBtn').classList.toggle('on', torchOn);
});
$('camSelect').addEventListener('change', e => openCamera(e.target.value).catch(err => setStatus(err.message)));

let wakeLock = null;
async function keepAwake() { try { wakeLock = await navigator.wakeLock?.request('screen'); } catch {} }
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && running) keepAwake(); });

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera API not available. Open this page over https:// (or localhost).');
  await openCamera(camId);
  await listCameras();
  // first run: prefer the ultra-wide lens if the browser exposes it by name
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
  state = 'scanning'; found = null; candidateStreak = 0; tracks = []; roi = null;
  running = true; loopToken++; loop(loopToken);
  $('startBtn').textContent = 'Stop'; $('startBtn').classList.add('running');
  keepAwake();
}
function stopScan() {
  running = false; state = 'idle';
  $('startBtn').textContent = 'Start'; $('startBtn').classList.remove('running');
  setStatus('Paused'); wakeLock?.release?.().catch(() => {}); wakeLock = null;
}
$('startBtn').addEventListener('click', async () => {
  if (running) return stopScan();
  try { if (!stream) await startCamera(); await modelsReady; startScan(); } catch (e) { setStatus(e.message); }
});
$('gearBtn').addEventListener('click', () => $('settings').classList.toggle('open'));
$('detSize').addEventListener('change', () => getDetector(settings.detSize));

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
  still.getContext('2d').drawImage(img, 0, 0);
  // display the photo with object-fit: contain semantics
  still.style.display = 'block'; still.style.objectFit = 'contain'; video.style.display = 'none';
  $('intro').style.display = 'none';
  await modelsReady;
  state = 'scanning'; found = null;
  const ms = await processFrame(still, img.width, img.height, true);
  statsEl.textContent = `${lastLive.length} cards\ndet ${ms.det.toFixed(0)} · cls ${ms.cls.toFixed(0)} ms`; statsEl.style.whiteSpace = 'pre';
}
const photoPicker = Object.assign(document.createElement('input'), { type: 'file', accept: 'image/*' });
photoPicker.addEventListener('change', e => e.target.files[0] && runOnPhoto(e.target.files[0]));
$('introPhoto').addEventListener('click', e => { e.preventDefault(); photoPicker.click(); });
$('photoFile').addEventListener('change', e => { e.target.files[0] && runOnPhoto(e.target.files[0]); $('settings').classList.remove('open'); });
window.addEventListener('resize', () => draw(lastLive, lastSrcW, lastSrcH));
window.__runOnPhoto = runOnPhoto;
