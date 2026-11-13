"""round10_read.py: round 10's short run read (4B NVFP4 W4A8 · llama.cpp+MIP · PC layer, both cards), docs/plans/2026-11-13-round10.md.

Every (row, sample) of arms A, B, CH, by the rule and by the API reader's corrected verdict (`triage.json`); pairs A-B, A-CH, B-CH with the
one-sided sign test; the same on the rows where `--question-line` puts nothing (a question that is not one fixed computation), so a pass
cannot come from software-computed values alone; every right -> wrong pair; round 9's General on the same rows as totals (not pairs).
  python scripts/pc/round10_read.py > results/round10/short-read.txt
"""
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding='utf-8')

W = Path('C:/Users/xenod/mip-work/a100')
CARDS = {'5060': 'r10-q-5060', '4070': 'r10-q-4070'}
ARMS = ('a', 'b', 'ch')


def sign_p(wins, losses):
    """One-sided sign test: P(X >= wins) for X ~ Bin(wins + losses, 1/2)."""
    n = wins + losses
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n if n else 1.0


def load():
    got = {}   # (arm, id, sample) -> dict(rule, corrected, tokens, card, ql)
    for card, d in CARDS.items():
        for arm in ARMS:
            base = W / 'pc' / d / arm
            tri = {(t['id'], t['sample']): t for t in json.loads((base / 'triage.json').read_text(encoding='utf-8'))}
            for r in json.loads((base / 'records.json').read_text(encoding='utf-8')):
                k = (r['id'], r.get('sample', 0))
                t = tri.get(k)
                corrected = (t['verdict'] or {}).get('verdict') == 'same' if t and t.get('verdict') else r['right']
                ql = any(e.get('kind') == 'question line' or str(e.get('what', '')).startswith('question line') for e in r['events'])
                got[(arm,) + k] = dict(rule=r['right'], corrected=corrected, tokens=r['tokens'], card=card, ql=ql)
    return got


def main():
    from tools.igc.question_read import read_exact
    rows = {}
    for f in ('r10q-a-rows.jsonl', 'r10q-b-rows.jsonl'):
        for line in (W / f).read_text(encoding='utf-8').splitlines():
            if line.strip():
                x = json.loads(line)
                rows[x['id']] = x
    got = load()
    # rows the question line reached: it put or steered its line in an answer of B or CH (read_exact also reads `simplify_surd`
    # questions, whose surd value the line does not put)
    fixed = {k[1] for k, g in got.items() if g['ql']}
    assert all(read_exact(rows[i]['question']) is not None for i in fixed)
    keys = sorted({k[1:] for k in got})
    print(f'{len(rows)} rows, {len(keys)} (row, sample) keys; rows that are one fixed computation (the question line): {len(fixed)}')
    for score in ('rule', 'corrected'):
        print(f'\n== {score} ==')
        for arm in ARMS:
            vals = [got[(arm,) + k][score] for k in keys if (arm,) + k in got]
            print(f'  {arm}: {sum(vals)} / {len(vals)}')
        for sub, name in ((keys, 'every pair'), ([k for k in keys if k[0] not in fixed], 'pairs the question line does not reach')):
            print(f'  {name} ({len(sub)}):')
            for x, y in (('a', 'b'), ('a', 'ch'), ('b', 'ch')):
                w = [k for k in sub if (x,) + k in got and (y,) + k in got]
                win = [k for k in w if got[(y,) + k][score] and not got[(x,) + k][score]]
                lose = [k for k in w if got[(x,) + k][score] and not got[(y,) + k][score]]
                print(f'    {x} vs {y}: {y} only right {len(win)}, {x} only right {len(lose)}, p = {sign_p(len(win), len(lose)):.3g}')
                for k in lose:
                    print(f'      {x} right -> {y} wrong: {k[0]}#{k[1]}')
    print('\n== per answer (rule / corrected, tokens) ==')
    for k in keys:
        cells = []
        for arm in ARMS:
            g = got.get((arm,) + k)
            cells.append(f"{arm} {'.' if g is None else ('R' if g['rule'] else 'w') + ('R' if g['corrected'] else 'w') + ' ' + str(g['tokens'])}")
        print(f"  {k[0]}#{k[1]} {'QL' if k[0] in fixed else '--'} | " + ' | '.join(cells))
    # round 9's General on the same rows: totals, not pairs (another server build, -np and pool)
    r9 = {}
    for h in ('r9a-gen', 'r9a3-gen', 'r9b-gen'):
        for r in json.loads((W / 'pc' / h / 'records.json').read_text(encoding='utf-8')):
            if r['id'] in rows:
                r9[(r['id'], r.get('sample', 0))] = r['right']
    print(f'\nround 9 General on these rows (rule, totals): {sum(r9.values())} / {len(r9)}; round 10 A on the same keys: '
          f"{sum(got[('a',) + k]['rule'] for k in r9 if ('a',) + k in got)} / {sum(1 for k in r9 if ('a',) + k in got)}")


if __name__ == '__main__':
    main()
