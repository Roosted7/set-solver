// Overlay drawing and the list of sets beside the camera.
import { tracks, attrsOf } from './tracker.js';
import { corners } from './geometry.js';
import { game } from './sets.js';

const $ = id => document.getElementById(id);
const overlay = $('overlay'), octx = overlay.getContext('2d'), stage = $('stage');
const NUMBER = ['1', '2', '3'], COLOR = ['red', 'green', 'purple'], SHAPE = ['diamond', 'oval', 'squiggle'], FILL = ['open', 'solid', 'striped'];
const COLOR_HEX = { red: '#ff5a5a', green: '#3ddc78', purple: '#bb77ff' };
const SHAPE_SYM = { diamond: '◆', oval: '⬮', squiggle: '∿' };
export const SET_COLORS = ['#ffd400', '#00e5ff', '#ff4fd8', '#8cff3c', '#ff8a00', '#9d7bff', '#ffffff'];
export const cardName = id => { const [n, c, s, f] = attrsOf(id); return `${NUMBER[n]} ${FILL[f]} ${COLOR[c]} ${SHAPE_SYM[SHAPE[s]]}`; };
export const cardColor = id => COLOR_HEX[COLOR[attrsOf(id)[1]]];

export const view = { srcW: 1, srcH: 1, region: null, sel: null, debug: false, labels: true };

function polyPath(pts, inset = 0) {
  const cx = pts.reduce((s, p) => s + p[0], 0) / 4, cy = pts.reduce((s, p) => s + p[1], 0) / 4;
  octx.beginPath();
  pts.forEach(([x, y], i) => {
    const d = Math.hypot(x - cx, y - cy) || 1, k = (d - inset) / d;
    const X = cx + (x - cx) * k, Y = cy + (y - cy) * k;
    i ? octx.lineTo(X, Y) : octx.moveTo(X, Y);
  });
  octx.closePath();
}

export function draw() {
  const dpr = window.devicePixelRatio || 1, cw = stage.clientWidth, ch = stage.clientHeight;
  if (overlay.width !== Math.round(cw * dpr) || overlay.height !== Math.round(ch * dpr)) { overlay.width = Math.round(cw * dpr); overlay.height = Math.round(ch * dpr); }
  octx.setTransform(dpr, 0, 0, dpr, 0, 0); octx.clearRect(0, 0, cw, ch);
  const s = Math.min(cw / view.srcW, ch / view.srcH), ox = (cw - view.srcW * s) / 2, oy = (ch - view.srcH * s) / 2;
  const P = ([x, y]) => [ox + x * s, oy + y * s];
  if (view.debug && view.region) {
    const [x, y] = P(view.region);
    octx.setLineDash([6, 6]); octx.lineWidth = 1; octx.strokeStyle = '#ffffff60'; octx.strokeRect(x, y, view.region[2] * s, view.region[3] * s); octx.setLineDash([]);
  }
  const visible = game.shown.filter(x => !x.lostAt);
  const member = new Map();
  for (const st of visible) for (const tid of st.tids) { if (!member.has(tid)) member.set(tid, []); member.get(tid).push(st); }
  for (const t of tracks) {
    if (t.hits < 2) continue;
    const pts = corners(t.box).map(P);
    const sets = member.get(t.tid) || [];
    octx.setLineDash(t.seen ? [] : [4, 4]);
    if (!sets.length) {
      octx.lineWidth = 1.5;
      octx.strokeStyle = t.unreadable ? '#ff666655' : t.known ? '#ffffffa0' : '#ffffff45';
      polyPath(pts); octx.stroke();
    }
    sets.forEach((st, k) => {
      const sel = view.sel === st.key || (view.sel == null && k === 0 && st === visible[0]);
      octx.lineWidth = sel ? 6 : 3.5; octx.strokeStyle = SET_COLORS[st.color % SET_COLORS.length];
      if (sel) { octx.shadowColor = octx.strokeStyle; octx.shadowBlur = 16; }
      polyPath(pts, -3 + k * 5); octx.stroke(); octx.shadowBlur = 0;
    });
    octx.setLineDash([]);
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const x1 = Math.min(...xs), y2 = Math.max(...ys), w = Math.max(...xs) - x1;
    const fs = Math.min(15, w / 6.5);
    if (view.labels && t.label >= 0 && fs >= 8) {
      octx.font = `600 ${fs}px system-ui, sans-serif`;
      const txt = cardName(t.label) + (t.known ? '' : ' ?');
      const tw = Math.min(octx.measureText(txt).width + 10, w + 10);
      octx.fillStyle = '#000000b0'; octx.fillRect(x1, y2 + 2, tw, fs + 6);
      octx.fillStyle = cardColor(t.label); octx.fillText(txt, x1 + 5, y2 + fs + 2, w);
    }
    if (view.debug) {
      octx.font = '10px system-ui'; octx.fillStyle = '#fff';
      octx.fillText(`${t.tid} c${t.cert.toFixed(2)} r${t.reads.length}${t.fresh ? '' : ' stale'}`, x1, Math.min(...ys) - 3);
    }
  }
}

export function renderSetList() {
  const el = $('setList');
  const visible = game.shown.filter(x => !x.lostAt);
  const html = visible.map(st => {
    const names = st.labels.map(l => `<span style="color:${cardColor(l)}">${cardName(l)}</span>`).join('');
    return `<button class="setItem${view.sel === st.key ? ' sel' : ''}" data-k="${st.key}"><i style="background:${SET_COLORS[st.color % SET_COLORS.length]}"></i><div class="names">${names}</div></button>`;
  }).join('');
  if (el.dataset.h !== html) { el.innerHTML = html; el.dataset.h = html; }
  $('nextBtn').style.display = visible.length ? '' : 'none';
}

export function setupSetList() {
  $('setList').addEventListener('click', e => { const b = e.target.closest('.setItem'); if (!b) return; view.sel = b.dataset.k; renderSetList(); draw(); });
  stage.addEventListener('click', () => {
    const vis = game.shown.filter(x => !x.lostAt); if (vis.length < 2) return;
    const k = vis.findIndex(x => x.key === view.sel);
    view.sel = vis[(k + 1) % vis.length].key; renderSetList(); draw();
  });
  window.addEventListener('resize', draw);
  new ResizeObserver(draw).observe(stage);
}
