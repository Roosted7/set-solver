"""Score a browser replay log (scratch/harness/run.mjs output) against the labelled video truth.

    python scripts/v3/score_replay.py replay.json data/real/game_hq/truth.json [--video-len 72.7]

truth.json holds, every few frames, the visible cards with their real ids. From it we know which sets
were on the table when. For every call the app made we check if it was a real set, and for every set
that was on the table for at least --min-dur seconds we measure how long it took until it was called.
"""
import argparse
import json
from itertools import combinations

NUMBER = ['1', '2', '3']; COLOR = ['red', 'green', 'purple']; SHAPE = {'◆': 0, '⬮': 1, '∿': 2}; FILL = ['open', 'solid', 'striped']


def parse(name):  # "1 open purple ◆" -> id
    n, f, c, s = name.split()
    return NUMBER.index(n) * 27 + COLOR.index(c) * 9 + SHAPE[s] * 3 + FILL.index(f)


def is_set(a, b, c):
    A = [(x // 27, (x // 9) % 3, (x // 3) % 3, x % 3) for x in (a, b, c)]
    return all((A[0][i] + A[1][i] + A[2][i]) % 3 == 0 for i in range(4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('replay'); ap.add_argument('truth')
    ap.add_argument('--video-len', type=float, default=72.7)
    ap.add_argument('--min-dur', type=float, default=1.5)
    a = ap.parse_args()
    truth = sorted((float(t), {c['id'] for c in cs}) for t, cs in json.load(open(a.truth)).items())
    tt = [t for t, _ in truth]

    def table_at(t, win=0.6):
        """cards on the table around t (union over +-win s: a hand briefly hiding a card doesn't remove it)"""
        near = [ids for tt_, ids in truth if abs(tt_ - t) <= win]
        return set().union(*near) if near else None

    # true set episodes
    episodes = {}  # frozenset ids -> list of [start, end]
    for t, ids in truth:
        for s in combinations(sorted(ids), 3):
            if is_set(*s):
                ep = episodes.setdefault(frozenset(s), [])
                if ep and t - ep[-1][1] < 0.6: ep[-1][1] = t
                else: ep.append([t, t])
    log = json.load(open(a.replay))
    calls, prev, prev_calls = [], set(), 0
    for x in log:
        t = x['t'] / 1000
        if t > a.video_len: break
        cur = {frozenset(parse(n) for n in s) for s in x.get('sets', [])}
        announced = x.get('calls', prev_calls + 1) > prev_calls   # v3 logs its "Set!" count; older apps don't
        if announced:
            for s in cur - prev:
                calls.append((t, s))
        prev, prev_calls = cur, x.get('calls', prev_calls)
    good = bad = 0
    bad_list = []
    for t, s in calls:
        tab = table_at(t)
        ok = len(s) == 3 and is_set(*s) and (tab is None or s <= tab)
        if ok: good += 1
        else: bad += 1; bad_list.append((round(t, 1), sorted(s)))
    lat, missed, missed_list = [], 0, []
    for s, eps in episodes.items():
        for st, en in eps:
            if en - st < a.min_dur: continue
            hit = [t for t, c in calls if c == s and st - 0.5 <= t <= en + 0.5]
            if hit: lat.append(max(0.0, min(hit) - st))
            else: missed += 1; missed_list.append((round(st, 1), round(en, 1), sorted(s)))
    n_ep = len(lat) + missed
    print(f'calls: {len(calls)}  correct {good}  wrong {bad}')
    print(f'sets on the table >= {a.min_dur}s: {n_ep}  called {len(lat)}  missed {missed}')
    if lat:
        lat.sort()
        print(f'time to call: median {lat[len(lat) // 2]:.2f}s  mean {sum(lat) / len(lat):.2f}s  max {lat[-1]:.2f}s')
    for b in bad_list[:10]: print('  wrong call', b)
    for m in missed_list[:10]: print('  missed set (from, to, cards)', m)
    steps = [b['t'] - a_['t'] for a_, b in zip(log, log[1:])]
    print(f'steps {len(log)}, median step {sorted(steps)[len(steps) // 2]:.0f} ms')


if __name__ == '__main__':
    main()
