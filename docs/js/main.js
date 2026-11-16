// SET! v3 — main loop.
//
// Per frame:
//   1. detect cards (YOLO26n-OBB at 640). Alternates between the whole frame and a region around
//      the known cards (sharper for far/small cards). The whole frame is searched every other frame,
//      so the search area can never lock onto a wrong spot.
//   2. update card tracks (per card: seen / hidden / moving).
//   3. read the cards that need it most (new, uncovered, moved, uncertain, then a slow round-robin),
//      as many as fit in the time budget, batched in one classifier run.
//   4. call every set whose three cards are known; keep it up until one of its cards goes.

import { modelsReady, backendText } from './runtime.js';
import { detect, detStats } from './detect.js';
import { classify, clsStats } from './classify.js';
import { tracks, resetTracks, updateTracks, addReading, pickReads, isLive, setStepMs } from './tracker.js';
import { game, evaluate, resetGame, dismissShown } from './sets.js';
import { view, draw, renderSetList, setupSetList, cardName } from './draw.js';
import { unlockAudio, playSound, setupSoundSettings } from './sound.js';
import { cam, startCamera, setupCameraControls } from './camera.js';
import { longSide, aabb, median } from './geometry.js';

const $ = id => document.getElementById(id);
const video = $('video'), still = $('still'), statusEl = $('status'), statsEl = $('stats'), banner = $('banner');
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ---------- settings ----------
const SETTINGS = ['minCards', 'camRes', 'showLabels', 'soundMode', 'debug'];
for (const id of SETTINGS) {
  try { const v = localStorage.getItem('set.' + id); if (v !== null && [...$(id).options].some(o => o.value === v)) $(id).value = v; } catch {}
  $(id).addEventListener('change', () => { try { localStorage.setItem('set.' + id, $(id).value); } catch {} syncView(); });
}
function syncView() { view.debug = $('debug').value !== 'off'; view.labels = $('showLabels').value === '1'; }
syncView();

modelsReady.then(() => {
  $('loadMsg').textContent = 'Models loaded.';
  $('backendInfo').textContent = `Runs on this device (${backendText()}).`;
}).catch(e => { $('err').textContent = 'Model load failed: ' + e.message; });

// ---------- detection region ----------
let stepNo = 0;
function chooseRegion(W, H, now) {
  const live = tracks.filter(t => isLive(t, now));
  if (stepNo % 2 === 0 || live.length < 3) return null;   // whole frame
  const L = median(live.map(t => longSide(t.box)));
  const bb = live.map(t => aabb(t.box));
  let x1 = Math.min(...bb.map(b => b[0])) - 1.2 * L, y1 = Math.min(...bb.map(b => b[1])) - 1.2 * L;
  let x2 = Math.max(...bb.map(b => b[2])) + 1.2 * L, y2 = Math.max(...bb.map(b => b[3])) + 1.2 * L;
  x1 = Math.max(0, x1); y1 = Math.max(0, y1); x2 = Math.min(W, x2); y2 = Math.min(H, y2);
  if ((x2 - x1) * (y2 - y1) > 0.6 * W * H) return null;   // hardly smaller than the frame: not worth it
  return [x1, y1, x2 - x1, y2 - y1];
}

// ---------- one processing step ----------
let budget = 8, callCount = 0;
const frameCanvas = document.createElement('canvas'), frameCtx = frameCanvas.getContext('2d');

async function step(src, W, H, isStill) {
  const now = performance.now();
  stepNo++;
  view.srcW = W; view.srcH = H;
  let nRead = 0;
  if (isStill) {
    resetTracks(); resetGame();
    const full = await detect(src, W, H);
    updateTracks(full, now);
    for (const t of tracks) t.hits = 2;
    const reads = tracks.filter(t => t.seen);
    const res = await classify(src, reads.map(t => t.box));
    reads.forEach((t, i) => addReading(t, res[i], now, true));
    for (const t of tracks) t.seenAt = performance.now();
    nRead = reads.length;
  } else {
    const region = chooseRegion(W, H, now);
    view.region = region || [0, 0, W, H];
    const dets = await detect(src, W, H, region);
    updateTracks(dets, now, region);
    const reads = pickReads(now, budget);
    if (reads.length) {
      const res = await classify(src, reads.map(t => t.box));
      reads.forEach((t, i) => addReading(t, res[i], now));
    }
    nRead = reads.length;
  }
  const { newSets } = evaluate(performance.now(), +$('minCards').value);
  if (newSets) { announce(); callCount++; }
  setStatus(game.status);
  draw(); renderSetList();
  const vis = game.shown.filter(s => !s.lostAt);
  window.__last = {
    state: vis.length ? 'found' : 'scanning', status: game.status, calls: callCount,
    cards: tracks.filter(t => t.hits >= 2).map(t => ({ tid: t.tid, name: t.label >= 0 ? cardName(t.label) : null, cert: +t.cert.toFixed(3), known: t.known, seen: t.seen, box: [t.box.cx, t.box.cy, t.box.w, t.box.h, t.box.a].map(v => +v.toFixed(2)) })),
    sets: vis.map(s => s.labels.map(cardName)), nRead, budget, detMs: +detStats.ms.toFixed(1), clsMs: +clsStats.ms.toFixed(1), region: view.region,
  };
  return nRead;
}

function announce() {
  const n = game.shown.filter(s => !s.lostAt).length;
  banner.textContent = n > 1 ? `SET! ×${n}` : 'SET!';
  banner.classList.add('show'); setTimeout(() => banner.classList.remove('show'), 1200);
  playSound($('soundMode').value);
  if (navigator.vibrate) try { navigator.vibrate(120); } catch {}
}
function setStatus(s) { if (statusEl.textContent !== s) statusEl.textContent = s; }

// ---------- loop ----------
let running = false, loopToken = 0, fps = 0;
async function loop(token) {
  while (running && token === loopToken) {
    if (video.readyState < 2 || !video.videoWidth) { await sleep(50); continue; }
    const W = video.videoWidth, H = video.videoHeight;
    if (frameCanvas.width !== W || frameCanvas.height !== H) { frameCanvas.width = W; frameCanvas.height = H; }
    frameCtx.drawImage(video, 0, 0, W, H);
    const t0 = performance.now();
    try {
      const n = await step(frameCanvas, W, H, false);
      const dt = performance.now() - t0;
      setStepMs(dt);
      // reading a card is cheap next to detecting: allow reading to take up to half the detector's time
      // (at least 40 ms), i.e. read everything that needs it on a fast phone, a few cards on a slow one
      budget = Math.max(3, Math.min(16, Math.floor(Math.max(40, 0.5 * detStats.ms) / Math.max(1, clsStats.msPerCrop))));
      fps = fps ? fps * 0.85 + (1000 / dt) * 0.15 : 1000 / dt;
      const live = tracks.filter(t => isLive(t, performance.now())).length;
      statsEl.textContent = `${live} cards · ${fps.toFixed(1)} fps\ndet ${detStats.ms.toFixed(0)} ms · read ${n} (${clsStats.msPerCrop.toFixed(1)} ms/card)`;
    } catch (e) { console.error(e); setStatus('Error: ' + e.message); await sleep(500); }
    await sleep(0);
  }
}

let wakeLock = null;
async function keepAwake() { try { wakeLock = await navigator.wakeLock?.request('screen'); } catch {} }
document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible' && running) keepAwake(); });

function startScan() {
  unlockAudio();
  still.style.display = 'none'; video.style.display = '';
  resetTracks(); resetGame();
  running = true; loopToken++; loop(loopToken);
  $('startBtn').textContent = 'Stop'; $('startBtn').classList.add('running');
  keepAwake(); renderSetList();
}
function stopScan() {
  running = false;
  $('startBtn').textContent = 'Start'; $('startBtn').classList.remove('running');
  setStatus('Paused');
  wakeLock?.release?.().catch(() => {}); wakeLock = null;
}
const showErr = e => setStatus(e.message);
const onCameraOpened = () => { resetTracks(); };

$('startBtn').addEventListener('click', async () => {
  if (running) return stopScan();
  try { if (!cam.stream) await startCamera(showErr, onCameraOpened); await modelsReady; startScan(); } catch (e) { setStatus(e.message); }
});
$('nextBtn').addEventListener('click', () => { dismissShown(); renderSetList(); draw(); });
$('gearBtn').addEventListener('click', () => $('settings').classList.toggle('open'));
$('closeSettings').addEventListener('click', () => $('settings').classList.remove('open'));
$('introBtn').addEventListener('click', async () => {
  try {
    $('introBtn').disabled = true; $('introBtn').textContent = 'Starting…';
    await startCamera(showErr, onCameraOpened); await modelsReady;
    $('intro').style.display = 'none'; startScan();
  } catch (e) {
    $('err').textContent = (e.name === 'NotAllowedError' ? 'Camera permission was denied.' : e.message) + '\n\nThe camera only works on https:// pages (or localhost).';
    $('introBtn').disabled = false; $('introBtn').textContent = 'Start camera';
  }
});
setupSetList(); setupSoundSettings(); setupCameraControls(showErr);

// ---------- photo mode ----------
async function runOnPhoto(file) {
  stopScan(); unlockAudio();
  const img = await createImageBitmap(file, { imageOrientation: 'from-image' });
  still.width = img.width; still.height = img.height;
  still.getContext('2d').drawImage(img, 0, 0);
  still.style.display = 'block'; video.style.display = 'none';
  $('intro').style.display = 'none';
  await modelsReady;
  const t0 = performance.now();
  await step(still, img.width, img.height, true);
  statsEl.textContent = `${tracks.length} cards · ${(performance.now() - t0).toFixed(0)} ms`;
}
const photoPicker = Object.assign(document.createElement('input'), { type: 'file', accept: 'image/*' });
photoPicker.addEventListener('change', e => e.target.files[0] && runOnPhoto(e.target.files[0]));
$('introPhoto').addEventListener('click', e => { e.preventDefault(); photoPicker.click(); });
$('photoFile').addEventListener('change', e => { e.target.files[0] && runOnPhoto(e.target.files[0]); $('settings').classList.remove('open'); });
