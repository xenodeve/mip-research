"""#307 loop share: over a trailing window of W tokens (digit runs read as one), the share whose K-gram was already seen in the last P.
One new gram moves the share by 1/W, so it does not restart, unlike the run of cover_sim.py (round 8 div_big #1: run reset every ~20k).
For every stored answer: the highest share in any full window, split by right / wrong / cut. python share_sim.py K P W"""
import json
import sys
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIGITS = frozenset(range(15, 25))
K, P, W = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])


def collapse(tokens):
    c, prev = [], False
    for t in tokens:
        d = t in DIGITS
        if not (d and prev):
            c.append(-1 if d else t)
        prev = d
    return c


def best_share(tokens):
    c = collapse(tokens)
    seen, win, hits, best, first = {}, deque(), 0, 0.0, None
    for i in range(K - 1, len(c)):
        g = tuple(c[i - K + 1:i + 1])
        j = seen.get(g)
        h = 1 if j is not None and j >= i - P else 0
        seen[g] = i
        win.append(h)
        hits += h
        if len(win) > W:
            hits -= win.popleft()
        if len(win) == W:
            s = hits / W
            if s > best:
                best = s
    return best


out = {'right': [], 'wrong': [], 'cut': []}
for arm in sorted((HERE / 'out').iterdir()):
    ids, rec = arm / 'ids.jsonl', arm / 'records.json'
    if not (ids.exists() and rec.exists()):
        continue
    recs = {(r['id'], r['sample']): r for r in json.loads(rec.read_text(encoding='utf-8'))}
    for line in ids.open(encoding='utf-8'):
        row = json.loads(line)
        r = recs.get((row['id'], row['sample']))
        if r is None:
            continue
        g = 'cut' if r['finish'] != 'stop' else 'right' if r['right'] else 'wrong'
        out[g].append((round(best_share(row['out_ids']), 4), len(row['out_ids']), arm.name, row['id'][3:22], row['sample']))
for g, v in out.items():
    v.sort(reverse=True)
    print(g, len(v), 'top:', v[:8])
long_cut = sorted(v for v in out['cut'] if v[1] > 150000)
print('cut past 150k tokens:', len(long_cut), [x[0] for x in long_cut])
