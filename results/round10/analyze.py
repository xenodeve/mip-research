"""analyze.py: round 10 on the PC layer (4B NVFP4 W4A8 · llama.cpp · PC layer), arms A (round 9's flags), B (+ --question-quote
--approx-put --answer-rewind 2 --letter-count), C (B + the Sandbox, MATH rows only). Reports, per AGENTS.md:
1. per arm: right by the rule (the form-noted ones apart), the API reader, answers whose final value software put (must be 0);
2. paired A vs B (and B vs C on C's rows): both, A only, B only, sign test;
3. every touch of each new mechanism: was the value right (checked again independently), and did the answer move between arms;
4. round 9's 25 wrong General answers, one by one: A, B, C and which mechanisms acted.
python results/round10/analyze.py [--pc DIR] > report.txt
"""
import argparse, json, math, re, sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding='utf-8')
from tools.igc.select_rows import right  # noqa: E402
from tools.igc.standard_score import form_note  # noqa: E402

W = Path('C:/Users/xenod/mip-work/a100')
NEW = ('question quote', 'approx value', 'letter count', 'answer rewind', 'verify gate', 'loop hint')
SOFTWARE_FINAL = ('exact answer',)   # a mechanism that writes the final value; none is on in round 10
ROUND9_WRONG = ['dm_div_big-1395#0', 'dm_div_big-1395#1', 'dm_div_big-5070#0', 'dm_div_big-5070#1', 'dm_div_big-4138#0',
                'dm_div_big-4138#1', 'dm_div_big-9991#0', 'dm_div_big-9991#1', 'dm_div_big-2783#0', 'dm_mul_div_longer-1505#0',
                'dm_mul_div_longer-5942#1', 'dm_mul_div_longer-9991#1', 'dm_mixed_longer-9951#0', 'dm_simplify_surd-3480#0',
                'dm_simplify_surd-3480#1', 'dm_swr_p_sequence-3480#0', 'dm_swr_p_sequence-9951#1', 'stdmath-45fa3ef2#0',
                'stdmath-fbf10ed4#1', 'stdmath-e3263989#0', 'stdmath-e3263989#1', 'stdmath-97b18f58#0', 'stdmath-97b18f58#1',
                'stdmath-370853d3#0', 'stdmath-370853d3#1']


def sign_p(a, b):
    n = a + b
    return 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n if n else 1.0


def load(pc, arm, rows):
    out = {}
    for half in ('a', 'b'):
        d = pc / f'r10{half}-{arm}'
        if not (d / 'records.json').exists():
            continue
        verdicts = json.loads((d / 'verdicts.json').read_text(encoding='utf-8')) if (d / 'verdicts.json').exists() else {}
        for rec in json.loads((d / 'records.json').read_text(encoding='utf-8')):
            if rec['id'] not in rows:
                continue
            row = rows[rec['id']]
            key = f"{rec['id']}#{rec['sample']}"
            note = form_note(rec['text'], row['answer']) if row['suite'] in ('stdmath', 'arith') else None
            out[key] = dict(right=right(row, rec['text']), note=note, api=(verdicts.get(key) or {}).get('verdict'),
                            events=rec.get('events', []), tokens=rec['tokens'], finish=rec['finish'])
    return out


def approx_value_right(event):
    """The put after `\\approx` checked again: the expression evaluated in mathexpr's wide mode, compared with the put as a Decimal."""
    from decimal import Decimal
    from tools.igc.mathexpr import evaluate, wide
    with wide():
        v = evaluate(event.get('expr', ''))
    if v is None:
        return None
    m = re.fullmatch(r'(-?[\d.]+)(?: \\times 10\^\{(-?\d+)\})?', event['value'])
    put = Fraction(m.group(1)) * (Fraction(10) ** int(m.group(2)) if m.group(2) else 1)
    return v == 0 and put == 0 or v != 0 and abs(put - v) <= abs(v) * Fraction(1, 10 ** 9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pc', default=str(W / 'pc'))
    args = ap.parse_args()
    pc = Path(args.pc)
    rows = {}
    for line in (W / 'r10-rows.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r['id']] = r
    arms = {a: load(pc, a, rows) for a in ('a', 'b', 'c')}

    print('1. Per arm')
    for a, res in arms.items():
        if not res:
            continue
        ok = sum(v['right'] for v in res.values())
        noted = sum(bool(v['note']) for v in res.values())
        api = sum(v['api'] == 'same' for v in res.values())
        soft = sum(any(e['kind'] in SOFTWARE_FINAL for e in v['events']) for v in res.values())
        stuck = sum(v['finish'] != 'stop' for v in res.values())
        print(f'   {a.upper()}: {len(res)} answers, right {ok} (of them form-noted {noted}), API reader {api}, '
              f'final value put by software {soft}, not finished {stuck}, tokens {sum(v["tokens"] for v in res.values()):,}')
        for k, v in res.items():
            if v['note']:
                print(f'      noted {k}: {v["note"]}')

    print('\n2. Paired')
    for x, y in (('a', 'b'), ('b', 'c')):
        keys = sorted(set(arms[x]) & set(arms[y]))
        if not keys:
            continue
        both = sum(arms[x][k]['right'] and arms[y][k]['right'] for k in keys)
        xo = [k for k in keys if arms[x][k]['right'] and not arms[y][k]['right']]
        yo = [k for k in keys if arms[y][k]['right'] and not arms[x][k]['right']]
        print(f'   {x.upper()} vs {y.upper()} on {len(keys)}: both {both}, {x.upper()} only {len(xo)}, {y.upper()} only {len(yo)}, '
              f'sign test p = {sign_p(len(xo), len(yo)):.3g}')
        print(f'      {x.upper()} only: {xo}')
        print(f'      {y.upper()} only: {yo}')

    print('\n3. Touches of the new mechanisms (arms B and C)')
    for a in ('b', 'c'):
        res = arms[a]
        if not res:
            continue
        count = Counter()
        moved = defaultdict(Counter)
        approx_checked = Counter()
        for k, v in res.items():
            kinds = {e['kind'] for e in v['events']}
            for e in v['events']:
                if e['kind'] in NEW:
                    count[e['kind']] += 1
                if e['kind'] == 'approx value':
                    approx_checked[approx_value_right(e)] += 1
            base = arms['a'].get(k) if a == 'b' else arms['b'].get(k)
            for kind in kinds & set(NEW):
                if base is None:
                    moved[kind]['no pair'] += 1
                else:
                    moved[kind][('right' if base['right'] else 'wrong') + ' -> ' + ('right' if v['right'] else 'wrong')] += 1
        print(f'   {a.upper()}: touches {dict(count)}')
        print(f'      approx values checked again: right {approx_checked[True]}, wrong {approx_checked[False]}, unread {approx_checked[None]}')
        for kind, c in moved.items():
            print(f'      answers with {kind}: {dict(c)} (against arm {"A" if a == "b" else "B"} on the same key)')

    print("\n4. Round 9's 25 wrong General answers")
    print('   key                               A        B        C        acted in B / C')
    for key in ROUND9_WRONG:
        cells = []
        for a in ('a', 'b', 'c'):
            v = arms[a].get(key)
            cells.append('-' if v is None else ('right*' if v['note'] else 'right') if v['right'] else 'wrong')
        acted = ' / '.join(','.join(sorted({e['kind'] for e in arms[a][key]['events'] if e['kind'] in NEW})) or '-' if key in arms[a] else '-'
                           for a in ('b', 'c'))
        print(f'   {key:34s} {cells[0]:8s} {cells[1]:8s} {cells[2]:8s} {acted}')
    print('   (right* = right with a form note)')


if __name__ == '__main__':
    main()
