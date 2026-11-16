// One track per physical card. Each track decides for itself whether it is known;
// nothing about one card blocks another.
//
// Evidence: per attribute, log-probabilities summed with forgetting (DECAY), so a few readings
// smooth out noise but a swapped card is relearned within ~2 readings.
// Known: certain enough AND the last two readings agree AND it has been read since it was last
// hidden or moved (so a card that was covered or replaced is never trusted on old evidence).

import { median, shortSide, rotIoU } from './geometry.js';

const DECAY = 0.55, LP_FLOOR = Math.log(0.02);
const CERT = 0.9;              // per-attribute posterior needed
const READABLE = 0.5;          // "readable card" probability needed for a reading to count
const MOVING = 0.12;           // motion per detection (fraction of the card's short side) counted as moving
const HIDE_STALE_MS = 350;     // hidden longer than this -> needs a fresh reading
const DROP_MS = 1500;          // forget a card not seen for this long

export const attrsOf = id => [Math.floor(id / 27), Math.floor(id / 9) % 3, Math.floor(id / 3) % 3, id % 3];

let nextId = 1, stepMs = 100;
export let tracks = [];
export function resetTracks() { tracks = []; }

function newTrack(b, now) {
  return { tid: nextId++, box: b, born: now, seenAt: now, seen: true, hits: 1, motion: 0, movedSinceRead: 0,
    lp: new Float64Array(12), reads: [], unreadable: 0, label: -1, cert: 0, fresh: false, lastCls: 0, known: false };
}

function posterior(lp) {
  const idx = [], p = [];
  for (let h = 0; h < 4; h++) {
    const o = h * 3, m = Math.max(lp[o], lp[o + 1], lp[o + 2]);
    const e = [0, 1, 2].map(k => Math.exp(lp[o + k] - m)), s = e[0] + e[1] + e[2];
    const k = e.indexOf(Math.max(...e)); idx.push(k); p.push(e[k] / s);
  }
  return { label: idx[0] * 27 + idx[1] * 9 + idx[2] * 3 + idx[3], cert: Math.min(...p) };
}

function refresh(t) {
  const last2 = t.reads.slice(-2);
  t.known = t.reads.length >= 2 && t.cert >= CERT && t.fresh && last2.every(l => l === t.label) && t.unreadable === 0;
}

export function addReading(t, r, now, still = false) {
  t.lastCls = now; t.movedSinceRead = 0;
  if (r.readable < READABLE) { t.unreadable++; t.known = false; return; }
  t.unreadable = 0;
  const one = posterior(r.lp);
  for (let i = 0; i < 12; i++) t.lp[i] = t.lp[i] * DECAY + Math.max(LP_FLOOR, r.lp[i]);
  t.reads.push(one.label); if (t.reads.length > 4) t.reads.shift();
  Object.assign(t, posterior(t.lp));
  t.fresh = true;
  if (still) { t.known = t.cert >= 0.7; return; }
  refresh(t);
}

// Match detections to tracks by centre distance (relative to card size), greedy nearest first.
// region: the part of the frame that was searched; cards outside it keep their state.
export function updateTracks(dets, now, region) {
  const inRegion = t => !region || (t.box.cx > region[0] && t.box.cx < region[0] + region[2] && t.box.cy > region[1] && t.box.cy < region[1] + region[3]);
  for (const t of tracks) if (inRegion(t)) t.seen = false;
  const pairs = [];
  dets.forEach((d, i) => tracks.forEach((t, j) => {
    const dist = Math.hypot(d.cx - t.box.cx, d.cy - t.box.cy) / shortSide(t.box);
    if (dist < 0.6) pairs.push([dist, i, j]);
  }));
  pairs.sort((a, b) => a[0] - b[0]);
  const dT = new Array(dets.length).fill(null), used = new Set();
  for (const [, i, j] of pairs) if (!dT[i] && !used.has(j)) { dT[i] = tracks[j]; used.add(j); }
  // camera shake: remove the median shift; what's left is the card's own motion
  const sh = dets.map((d, i) => dT[i] ? [d.cx - dT[i].box.cx, d.cy - dT[i].box.cy] : null).filter(Boolean);
  const gx = sh.length >= 3 ? median(sh.map(s => s[0])) : 0, gy = sh.length >= 3 ? median(sh.map(s => s[1])) : 0;
  dets.forEach((d, i) => {
    let t = dT[i];
    if (!t) { tracks.push(newTrack(d, now)); return; }
    const s = shortSide(t.box);
    t.motion = Math.hypot(d.cx - t.box.cx - gx, d.cy - t.box.cy - gy) / s;
    t.movedSinceRead += t.motion;
    if (now - t.seenAt > Math.max(HIDE_STALE_MS, 1.5 * stepMs) || t.movedSinceRead > 0.35) { t.fresh = false; t.known = false; }
    t.box = d; t.seen = true; t.seenAt = now; t.hits++;
  });
  tracks = tracks.filter(t => t.seen || now - t.seenAt < DROP_MS);
  // single-frame detections that never came back are noise
  tracks = tracks.filter(t => t.hits >= 2 || now - t.born < 600);
  // a hidden track overlapping a visible one is the same card seen again under a new track
  tracks = tracks.filter(t => t.seen || !tracks.some(o => o.seen && o !== t && rotIoU(o.box, t.box) > 0.3));
}

export const isMoving = t => t.motion > MOVING;
// "recently seen" scales with how long a processing step takes on this device
export const setStepMs = ms => { stepMs = stepMs * 0.8 + ms * 0.2; };
export const liveWindow = () => Math.max(400, 2.5 * stepMs);
export const isLive = (t, now) => t.hits >= 2 && now - t.seenAt < liveWindow();

// Which tracks to read now, most useful first, within `budget` crops.
export function pickReads(now, budget) {
  const cand = tracks.filter(t => t.seen && t.hits >= 1 && !isMoving(t));
  const prio = t => !t.reads.length ? 0 : !t.fresh ? 1 : !t.known ? 2 : 3;
  cand.sort((a, b) => prio(a) - prio(b) || a.lastCls - b.lastCls);
  const out = [];
  for (const t of cand) {
    if (out.length >= budget) break;
    if (prio(t) === 3 && now - t.lastCls < 700) continue;   // known cards: re-check about every 0.7 s
    out.push(t);
  }
  return out;
}
