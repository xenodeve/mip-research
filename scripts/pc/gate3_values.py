"""gate3_values.py OUT: gate 3 of the token channel (docs/plans/2026-11-12-mip-token-channel-plan.md, task 7). CPU only.

For every `steered value` event of a --channel run (records.json and ids.jsonl in OUT): (1) the value checked again independently: an
inline value against mathexpr's exact evaluation of its expression, an approximation by results/round10/analyze.py's own check, a
question value against question_read; (2) the model's text at the event's position starts with the steered text (what the model
wrote is what was steered); (3) whether the steering finished, was dropped, or the request ended in the middle. Split by answers that
finished right and wrong. Writes OUT/gate3.txt.  python scripts/pc/gate3_values.py <run dir>
"""
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'results' / 'round10'))


def value_right(e, question):
    from tools.igc.mathexpr import evaluate
    kind = e.get('what')
    if kind == 'approx value':
        from analyze import approx_value_right
        return approx_value_right(e)
    if kind == 'question value':
        from tools.igc.question_read import read_exact
        got = read_exact(question or '')
        return got is not None and evaluate(e['value']) == Fraction(got[1])
    want = evaluate(e['expr'])
    return want is not None and evaluate(e['value']) == want


def main():
    out = Path(sys.argv[1])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('C:/Users/xenod/mip-work/q4b-hf')
    rows = {}
    for line in Path(json.loads((out / 'arm.json').read_text(encoding='utf-8'))['rows']).read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r['id']] = r
    ids = {}
    for line in (out / 'ids.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip():
            x = json.loads(line)
            ids[(x['id'], x.get('sample', 0))] = x['out_ids']
    c, bad = Counter(), []
    for rec in json.loads((out / 'records.json').read_text(encoding='utf-8')):
        final = 'right' if rec['right'] else 'wrong'
        out_ids = ids.get((rec['id'], rec.get('sample', 0)), [])
        for e in rec.get('events', []):
            if e.get('kind') != 'steered value':
                continue
            state = 'done' if e.get('done') else 'dropped' if e.get('dropped') else 'open'
            c[(final, state)] += 1
            ok = value_right(e, rows.get(rec['id'], {}).get('question'))
            c[(final, 'value right' if ok else 'VALUE WRONG')] += 1
            written = tok.decode(out_ids[e['at']:e['at'] + 40]) if out_ids else ''
            want = (' ' + e['value']) if e.get('what') != 'question value' else e['expr']
            wrote = written.startswith(want) or written.lstrip().startswith(e['value'])
            c[(final, 'written as steered' if wrote else 'TEXT DIFFERS')] += 1
            if not ok or (state == 'done' and not wrote):
                bad.append(f"{rec['id']}#{rec.get('sample', 0)} {final} {state} {e.get('what')}: {e['expr'][:60]!r} = {e['value']!r}; "
                           f"text {written[:40]!r}")
    lines = [f'gate 3 (token channel): {out}', ' '.join(f'{k[0]}/{k[1]}: {n}' for k, n in sorted(c.items()))]
    wrong = sum(n for k, n in c.items() if k[1] == 'VALUE WRONG')
    lines.append('PASS: every steered value right' if wrong == 0 else f'{wrong} VALUES WRONG (listed)')
    lines += bad[:60]
    (out / 'gate3.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:3]))


if __name__ == '__main__':
    main()
