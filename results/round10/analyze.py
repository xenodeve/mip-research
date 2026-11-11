"""analyze.py: round 10 on the PC layer (4B NVFP4 W4A8 · llama.cpp · PC layer), arms A (round 9's flags), B (+ --question-quote
--approx-put --answer-rewind 2 --letter-count), C (B + the Sandbox, MATH rows only). Reports, per AGENTS.md:
1. per arm: right by the rule (the form-noted ones apart), the API reader, the corrected score, answers whose final value software put
   (must be 0), finishes by kind;
2. paired A vs B (and B vs C on C's rows): both, A only, B only, one-sided sign test (H1: the later arm is right more often) on the rule
   score and on the corrected score (the PC-layer gate wants p < 0.05 on both);
3. every touch of each new mechanism: was the value it put right (checked again here, not by the code that put it), split by answers
   that finished right and wrong; then whether the answer moved against the other arm, and where the two arms' texts first part
   (a pair that parts before the first touch is path noise, not the mechanism's doing);
4. round 9's 25 wrong General answers, one by one: A, B, C and which mechanisms acted.
python results/round10/analyze.py [--pc DIR] [--halves a,b] [--rows FILE ...] > report.txt
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
# the kinds tir4b writes (Rollout.inject's second argument, rewind_answer's event); checked against the records at load
NEW = ('question quote', 'approx value', 'letter count', 'answer rewind', 'verify gate', 'loop hint')
VALUE_PUTS = ('question quote', 'approx value', 'letter count')   # the ones that put a value that can be right or wrong
SOFTWARE_FINAL = ('exact answer',)   # a mechanism that writes the final value; none is on in round 10
ROUND9_WRONG = ['dm_div_big-1395#0', 'dm_div_big-1395#1', 'dm_div_big-5070#0', 'dm_div_big-5070#1', 'dm_div_big-4138#0',
                'dm_div_big-4138#1', 'dm_div_big-9991#0', 'dm_div_big-9991#1', 'dm_div_big-2783#0', 'dm_mul_div_longer-1505#0',
                'dm_mul_div_longer-5942#1', 'dm_mul_div_longer-9991#1', 'dm_mixed_longer-9951#0', 'dm_simplify_surd-3480#0',
                'dm_simplify_surd-3480#1', 'dm_swr_p_sequence-3480#0', 'dm_swr_p_sequence-9951#1', 'stdmath-45fa3ef2#0',
                'stdmath-fbf10ed4#1', 'stdmath-e3263989#0', 'stdmath-e3263989#1', 'stdmath-97b18f58#0', 'stdmath-97b18f58#1',
                'stdmath-370853d3#0', 'stdmath-370853d3#1']
HAND = {}   # filled from --hand
ROWS = ('r10a-rows.jsonl', 'r10b-rows.jsonl', 'r10a-math-rows.jsonl', 'r10b-math-rows.jsonl')


def sign_p(worse, better):
    """One-sided sign test: P(X >= better) for X ~ Bin(worse + better, 1/2), H1: the later arm wins more of the split pairs."""
    n = worse + better
    return sum(math.comb(n, i) for i in range(better, n + 1)) / 2 ** n if n else 1.0


def load(pc, halves, arm, rows, unknown):
    out = {}
    for half in halves:
        d = pc / f'r10{half}-{arm}'
        if not (d / 'records.json').exists():
            continue
        verdicts = json.loads((d / 'verdicts.json').read_text(encoding='utf-8')) if (d / 'verdicts.json').exists() else {}
        ids = {}
        if (d / 'ids.jsonl').exists():
            for line in (d / 'ids.jsonl').read_text(encoding='utf-8').splitlines():
                if line.strip():
                    x = json.loads(line)
                    ids[f"{x['id']}#{x['sample']}"] = x['out_ids']
        for rec in json.loads((d / 'records.json').read_text(encoding='utf-8')):
            if rec['id'] not in rows:
                unknown.append(f"{d.name}/{rec['id']}")
                continue
            row = rows[rec['id']]
            key = f"{rec['id']}#{rec['sample']}"
            note = form_note(rec['text'], row['answer']) if row['suite'] in ('stdmath', 'arith') else None
            ok = right(row, rec['text'])
            api = (verdicts.get(key) or {}).get('verdict')
            # the corrected score: a hand verdict (--hand) where one was read, else the API reader's, else the rule's; the reader
            # is wrong too at times (round 9 dm_simplify_surd-3480#0: `9216 if the sum is 0` read as right)
            fixed = HAND[key] if key in HAND else ok if api is None else api == 'same'
            out[key] = dict(right=ok, note=note, api=api, fixed=fixed, events=rec.get('events', []), tokens=rec['tokens'],
                            finish=rec['finish'], text=rec['text'], row=row, out_ids=ids.get(key))
    return out


def approx_value_right(event):
    """The put after `\\approx` checked again: the expression evaluated in mathexpr's wide mode, compared with the put as a Decimal
    to the 10 significant digits it gives."""
    from tools.igc.mathexpr import evaluate, wide
    with wide():
        v = evaluate(event.get('expr', ''))
    m = re.fullmatch(r'(-?[\d.]+)(?: \\times 10\^\{(-?\d+)\})?', event.get('value', ''))
    if v is None or m is None:
        return None
    scale = Fraction(10) ** int(m.group(2)) if m.group(2) else Fraction(1)
    put = Fraction(m.group(1)) * scale
    places = len(m.group(1).split('.')[1]) if '.' in m.group(1) else 0
    # right when it is the value rounded to the places it gives: within half a unit of its last place
    return abs(put - v) <= Fraction(1, 2) * Fraction(1, 10 ** places) * scale


_LETTER_LINE = re.compile(r'The string (\w+) has (\d+) letters: ([^\n]*?)\.\n')


def letter_count_right(event, text, question):
    """The letter line checked again: the string stands in the question as a word, and the line's total and per-letter counts are the
    string's own (counted here with str.count, not with letters.py's Counter)."""
    s = event.get('string', '')
    m = _LETTER_LINE.search(text)
    if not s or m is None or m.group(1) != s or not re.search(r'(?<![A-Za-z])' + re.escape(s) + r'(?![A-Za-z])', question):
        return None if m is None else False
    if int(m.group(2)) != len(s):
        return False
    for part in m.group(3).split(', '):
        ch, n = part.rsplit(' ', 1)
        if s.count(ch) != int(n):
            return False
    return {ch for ch in s} == {p.rsplit(' ', 1)[0] for p in m.group(3).split(', ')}


_QUOTE_LINE = re.compile(r'The question writes: \$(.*?)\$\.\n')


def quote_value_right(expr, gold):
    """A quote is the question's computation, so its value is the gold answer: evaluated exactly here and compared with gold as a
    number. A moved bracket or a lost sign changes the value, which a check of the digits in order does not see. None when either side
    cannot be read as a number (a surd quote, a gold that is not a number)."""
    from tools.igc.mathexpr import evaluate
    from tools.igc.question_read import _plain
    try:
        v = evaluate(_plain(expr))
        g = Fraction(str(gold).replace(' ', ''))
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    return None if v is None else Fraction(v) == g


def question_quote_right(text, gold):
    """Every quote in the text checked again by its value against gold (quote_value_right); None when no quote can be read."""
    verdicts = [quote_value_right(x, gold) for x in _QUOTE_LINE.findall(text)]
    if not verdicts or all(v is None for v in verdicts):
        return None
    return all(v is not False for v in verdicts)


def first_part(a, b):
    """The first token index where two answers' outputs differ (None when either has no ids)."""
    if a is None or b is None:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pc', default=str(W / 'pc'))
    ap.add_argument('--halves', default='a,b', help='the TAGs the halves ran under (pc_round10.sh TAG), e.g. a,b or smoke')
    ap.add_argument('--rows', nargs='*', default=[str(W / r) for r in ROWS])
    ap.add_argument('--hand', default='', help='JSON {"<id>#<sample>": true|false}: verdicts read by hand where rule and reader differ')
    args = ap.parse_args()
    pc = Path(args.pc)
    halves = args.halves.split(',')
    rows = {}
    for f in args.rows:
        for line in Path(f).read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r['id']] = r
    unknown = []
    if args.hand:
        HAND.update(json.loads(Path(args.hand).read_text(encoding='utf-8')))
    arms = {a: load(pc, halves, a, rows, unknown) for a in ('a', 'b', 'c')}
    if unknown:
        print(f'!! {len(unknown)} records whose row is in none of the row files (left out): {unknown[:10]}')
    seen = Counter(e['kind'] for res in arms.values() for v in res.values() for e in v['events'])
    print(f'event kinds in the records: {dict(seen)}')
    for kind in NEW:
        if kind not in seen:
            print(f'   (no "{kind}" event in any record)')

    print('\n1. Per arm')
    for a, res in arms.items():
        if not res:
            continue
        ok = sum(v['right'] for v in res.values())
        noted = sum(bool(v['note']) for v in res.values())
        api = sum(v['api'] == 'same' for v in res.values())
        unread = sum(v['api'] is None for v in res.values())
        fixed = sum(v['fixed'] for v in res.values())
        soft = sum(any(e['kind'] in SOFTWARE_FINAL for e in v['events']) for v in res.values())
        finishes = Counter(v['finish'] for v in res.values())
        print(f'   {a.upper()}: {len(res)} answers, rule {ok} (form-noted {noted}), API reader {api} ({unread} not read), corrected {fixed}, '
              f'final value put by software {soft}, finishes {dict(finishes)}, tokens {sum(v["tokens"] for v in res.values()):,}')
        for k, v in res.items():
            if v['note']:
                print(f'      noted {k}: {v["note"]}')
            if v['api'] is not None and (v['api'] == 'same') != v['right']:
                print(f'      rule and reader differ {k}: rule {"right" if v["right"] else "wrong"}, reader {v["api"]} (read it)')

    print('\n2. Paired (one-sided sign test, H1: the later arm right more often)')
    for x, y in (('a', 'b'), ('b', 'c')):
        keys = sorted(set(arms[x]) & set(arms[y]))
        if not keys:
            continue
        for score in ('right', 'fixed'):
            both = sum(arms[x][k][score] and arms[y][k][score] for k in keys)
            xo = [k for k in keys if arms[x][k][score] and not arms[y][k][score]]
            yo = [k for k in keys if arms[y][k][score] and not arms[x][k][score]]
            name = 'rule' if score == 'right' else 'corrected'
            print(f'   {x.upper()} vs {y.upper()} on {len(keys)}, {name}: both {both}, {x.upper()} only {len(xo)}, {y.upper()} only {len(yo)}, '
                  f'p = {sign_p(len(xo), len(yo)):.3g}')
            print(f'      {x.upper()} only: {xo}')
            print(f'      {y.upper()} only: {yo}')

    print('\n3. Touches of the new mechanisms (arms B and C)')
    for a in ('b', 'c'):
        res = arms[a]
        if not res:
            continue
        base_arm = 'a' if a == 'b' else 'b'
        count = Counter()
        checked = defaultdict(Counter)   # kind -> (value verdict, final) -> n
        moved = defaultdict(Counter)
        noise = Counter()
        for k, v in res.items():
            final = 'right' if v['right'] else 'wrong'
            kinds = {e['kind'] for e in v['events']} & set(NEW)
            for e in v['events']:
                if e['kind'] not in NEW:
                    continue
                count[e['kind']] += 1
                if e['kind'] == 'approx value':
                    verdict = approx_value_right(e)
                elif e['kind'] == 'letter count':
                    verdict = letter_count_right(e, v['text'], v['row']['question'])
                elif e['kind'] == 'question quote':
                    verdict = question_quote_right(v['text'], v['row']['answer'])
                else:
                    continue
                checked[e['kind']][({True: 'value right', False: 'VALUE WRONG', None: 'not checkable'}[verdict], final)] += 1
            base = arms[base_arm].get(k)
            touches = [e['at'] for e in v['events'] if e['kind'] in NEW]
            for kind in kinds:
                if base is None:
                    moved[kind]['no pair'] += 1
                else:
                    moved[kind][('right' if base['right'] else 'wrong') + ' -> ' + final] += 1
            if base is not None and touches:
                part = first_part(base['out_ids'], v['out_ids'])
                if part is None:
                    noise['no ids'] += 1
                elif part < min(touches):
                    noise['parted before the first touch (path noise)'] += 1
                else:
                    noise['parted at or after the first touch'] += 1
        print(f'   {a.upper()}: touches {dict(count)}')
        for kind, c in checked.items():
            print(f'      {kind}, value checked again (by final answer): {dict(c)}')
        for kind, c in moved.items():
            print(f'      answers with {kind}: {dict(c)} (against arm {base_arm.upper()} on the same key)')
        print(f'      where the touched answers part from arm {base_arm.upper()}: {dict(noise)}')

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
