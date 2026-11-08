"""#307 shape loop: the counting check with each run of digits collapsed to one token, so `(Digit 5243: 7)` and `(Digit 5244: 2)` are one
round. For every stored answer of the named arms, read every 1024 tokens over the whole reply: the longest shape cycle at that point
(period <= 128 collapsed tokens), in rounds and in tokens. Per answer: the most rounds it ever reached, where it first reached each cap,
and how the run ended. A cap is safe when no answer that finished right ever reaches it.
python shape_sim.py ARM [ARM...]  -> shape_sim.json and a table."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPS = (64, 96, 128, 192, 256, 384, 512)
LOOK = 128 * 600   # collapsed tokens read back


def collapse(t):
    out, pos = [], []
    prev_digit = False
    for i, x in enumerate(t):
        d = 15 <= x <= 24
        if d and prev_digit:
            pos[-1] = i
            continue
        out.append(15 if d else x)
        pos.append(i)
        prev_digit = d
    return out, pos


def best_rounds(c):
    """(rounds, period) of the longest exact cycle ending at the tail of collapsed c, period <= 128."""
    n = len(c)
    best = (0, 0)
    for p in range(1, min(128, n - 1) + 1):
        if c[-1] != c[-1 - p]:
            continue
        j = n - 1
        stop = max(p, n - LOOK)
        while j >= stop and c[j] == c[j - p]:
            j -= 1
        rounds = (n - 1 - j) // p + 1
        if rounds > best[0]:
            best = (rounds, p)
    return best


def scan(job):
    key, ids = job
    c, pos = collapse(ids)
    most, first = 0, {}
    k = 0
    for i in range(1023, len(ids), 1024):
        while k < len(pos) and pos[k] <= i:
            k += 1
        r, p = best_rounds(c[:k])
        most = max(most, r)
        for cap in CAPS:
            if r >= cap and cap not in first:
                first[cap] = i + 1
    return key, most, first


def main():
    meta, jobs = {}, []
    for arm in sys.argv[1:]:
        recs = {(r['id'], r['sample']): r for r in json.loads((HERE / 'out' / arm / 'records.json').read_text(encoding='utf-8'))}
        for line in (HERE / 'out' / arm / 'ids.jsonl').read_text(encoding='utf-8').splitlines():
            d = json.loads(line)
            r = recs[(d['id'], d['sample'])]
            key = f"{arm}|{d['id']}|{d['sample']}"
            meta[key] = dict(finish=r['finish'], right=r['right'], tokens=len(d['out_ids']))
            jobs.append((key, d['out_ids']))
    jobs.sort(key=lambda j: -len(j[1]))
    res = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for key, most, first in ex.map(scan, jobs, chunksize=1):
            res[key] = dict(most=most, first=first)
    (HERE / 'shape_sim.json').write_text(json.dumps(dict(meta=meta, res=res)), encoding='utf-8')
    groups = {'right': lambda m: m['finish'] == 'stop' and m['right'], 'finished wrong': lambda m: m['finish'] == 'stop' and not m['right'],
              'cut (stuck/length)': lambda m: m['finish'] != 'stop'}
    for name, f in groups.items():
        ks = [k for k, m in meta.items() if f(m)]
        print(f'{name}: {len(ks)} answers')
        for cap in CAPS:
            hit = [k for k in ks if str(cap) in map(str, res[k]['first'])]
            saved = sum(meta[k]['tokens'] - res[k]['first'][cap] for k in hit)
            print(f'   cap {cap:>3} rounds: reached by {len(hit)}, tokens after it {saved:,}')
        top = sorted(ks, key=lambda k: -res[k]['most'])[:8]
        print('   most rounds:', [(k, res[k]['most']) for k in top])


if __name__ == '__main__':
    main()
