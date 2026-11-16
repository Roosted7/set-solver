// Small geometry helpers for rotated boxes {cx, cy, w, h, a} (a in radians, w along angle a).

export const median = a => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };

export function corners(b) {
  const c = Math.cos(b.a), s = Math.sin(b.a), hw = b.w / 2, hh = b.h / 2;
  return [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([x, y]) => [b.cx + x * c - y * s, b.cy + x * s + y * c]);
}

export const shortSide = b => Math.min(b.w, b.h);
export const longSide = b => Math.max(b.w, b.h);

export function aabb(b) {
  const p = corners(b), xs = p.map(q => q[0]), ys = p.map(q => q[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

const polyArea = p => Math.abs(p.reduce((s, a, i) => { const b = p[(i + 1) % p.length]; return s + a[0] * b[1] - b[0] * a[1]; }, 0)) / 2;

// Sutherland-Hodgman clip of convex polygon p by convex polygon q (both clockwise or both ccw)
function clip(p, q) {
  let out = p;
  const sgn = Math.sign((q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[1][1] - q[0][1]) * (q[2][0] - q[0][0])) || 1;
  for (let i = 0; i < q.length && out.length; i++) {
    const a = q[i], b = q[(i + 1) % q.length], inp = out; out = [];
    const side = v => sgn * ((b[0] - a[0]) * (v[1] - a[1]) - (b[1] - a[1]) * (v[0] - a[0]));
    for (let j = 0; j < inp.length; j++) {
      const P = inp[j], Q = inp[(j + 1) % inp.length], sp = side(P), sq = side(Q);
      if (sp >= 0) out.push(P);
      if ((sp >= 0) !== (sq >= 0)) { const t = sp / (sp - sq); out.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]); }
    }
  }
  return out;
}

export function rotIoU(a, b) {
  if (Math.hypot(a.cx - b.cx, a.cy - b.cy) > (longSide(a) + longSide(b)) / 2) return 0;
  const pa = corners(a), pb = corners(b), inter = clip(pa, pb);
  const ia = inter.length >= 3 ? polyArea(inter) : 0;
  return ia / (a.w * a.h + b.w * b.h - ia);
}
