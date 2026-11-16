// Calling sets. Works per set, not per table: a set is called as soon as its own three cards
// are known, whatever happens with the other cards (a hand, a misread neighbour, a card being dealt).
// Called sets stay up until one of their cards is taken or changes.

import { tracks, attrsOf, isLive, isMoving, liveWindow } from './tracker.js';

const isSet = (a, b, c) => { const A = attrsOf(a), B = attrsOf(b), C = attrsOf(c); return A.every((v, i) => (v + B[i] + C[i]) % 3 === 0); };
const GONE_MS = 600;   // a set disappears this long after one of its cards stopped being seen/known

export const game = {
  shown: [],          // [{key, tids:[..], labels:[..], color, at, lostAt}]
  dismissed: new Set(),
  noSetSince: 0,
  status: '',
  colorIdx: 0,
  recent: new Map(),   // card labels of a set -> when it was last on screen (no second "Set!" while it's being picked up)
};

export function resetGame() { game.shown = []; game.dismissed.clear(); game.recent.clear(); game.noSetSince = 0; }

// returns {newSets: number} so the caller can play a sound
export function evaluate(now, minNoSet) {
  const live = tracks.filter(t => isLive(t, now));
  const known = live.filter(t => t.known);
  // the same card can't be on the table twice: keep the more certain reading
  const byLabel = new Map();
  for (const t of known) { const o = byLabel.get(t.label); if (!o || t.cert > o.cert) byLabel.set(t.label, t); }
  const cards = [...byLabel.values()].filter(t => !isMoving(t));
  const found = [];
  for (let i = 0; i < cards.length; i++) for (let j = i + 1; j < cards.length; j++) for (let k = j + 1; k < cards.length; k++)
    if (isSet(cards[i].label, cards[j].label, cards[k].label)) found.push([cards[i], cards[j], cards[k]]);
  const byTid = new Map(tracks.map(t => [t.tid, t]));
  // update shown sets: keep while all their cards exist with the same labels
  for (const s of game.shown) {
    const ok = s.tids.every((tid, i) => { const t = byTid.get(tid); return t && t.label === s.labels[i] && now - t.seenAt < liveWindow(); });
    if (ok) s.lostAt = 0; else if (!s.lostAt) s.lostAt = now;
  }
  game.shown = game.shown.filter(s => !s.lostAt || now - s.lostAt < GONE_MS);
  let newSets = 0;
  for (const st of found) {
    const tids = st.map(t => t.tid).sort((a, b) => a - b);
    const key = tids.join('.') + ':' + tids.map(id => byTid.get(id).label).join('.');
    if (game.shown.some(s => s.key === key) || game.dismissed.has(key)) continue;
    const labels = tids.map(id => byTid.get(id).label), lkey = [...labels].sort((a, b) => a - b).join('.');
    const seenRecently = now - (game.recent.get(lkey) ?? -1e9) < 4000;
    game.shown.push({ key, tids, labels, color: game.colorIdx++, at: now, lostAt: 0 });
    if (!seenRecently) newSets++;   // same three cards just disappeared (e.g. being picked up): show, but don't call again
  }
  for (const st of game.shown) if (!st.lostAt) game.recent.set([...st.labels].sort((a, b) => a - b).join('.'), now);
  // status
  const visible = game.shown.filter(s => !s.lostAt);
  const unread = live.filter(t => !t.known).length;
  if (visible.length) { game.status = `SET!${visible.length > 1 ? `  ${visible.length} sets` : ''}`; game.noSetSince = 0; }
  else if (!live.length) { game.status = 'Looking for cards…'; game.noSetSince = 0; }
  else if (unread) { game.status = `Reading cards… ${live.length - unread}/${live.length}`; game.noSetSince = 0; }
  else {
    game.noSetSince ||= now;
    game.status = live.length >= minNoSet && now - game.noSetSince > 1200 ? `No set in these ${live.length} cards — deal 3 more` : `${live.length} cards, no set yet`;
  }
  return { newSets, visible };
}

export function dismissShown() { for (const s of game.shown) game.dismissed.add(s.key); game.shown = []; }
