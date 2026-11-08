"""#307 failures of the best config on the A100 arms: (1) every inline put re-checked by an independent evaluator (Python's own Fraction
arithmetic on a normalised copy, not mathexpr); (2) every close and what the answer ended as; (3) answers right in the same round's
no-closer arm (loop-gen) and wrong with the closers, and the reverse. python fail_audit.py ARM [ARM...]"""
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent  # a folder with out/<arm>/ A100 records (records.json, ids.jsonl); not in this repo
sys.stdout.reconfigure(encoding='utf-8')


def plain(expr):
    """A copy Python can evaluate, or None when it is not plain arithmetic (roots, mod, functions, words)."""
    s = expr.strip().strip('$').strip()
    s = re.sub(r'\\left|\\right|\\,|\\!|\\;|\s', '', s)
    for a, b in ((r'\times', '*'), (r'\cdot', '*'), (r'\div', '/'), ('×', '*'), ('·', '*'), ('÷', '/'), ('−', '-'), ('^', '**'),
                 ('{', '('), ('}', ')'), ('[', '('), (']', ')')):
        s = s.replace(a, b)
    s = re.sub(r'\\d?frac\(([^()]*)\)\(([^()]*)\)', r'((\1)/(\2))', s)
    s = re.sub(r'(?<=\d),(?=\d{3}\b)', '', s)
    if not re.fullmatch(r'[\d.+\-*/()]+', s) or '**' in s and re.search(r'\*\*\(?-?\d{3,}', s):
        return None
    s = re.sub(r'(\d+(?:\.\d+)?)', r'F("\1")', s)
    return s


def value_of(text):
    t = text.strip().replace('...', '')
    m = re.fullmatch(r'(-?\d+)\s*(?:\\text\{\s*rem\s*\}|rem)\s*(\d+)', t)
    if m:
        return ('rem', int(m.group(1)), int(m.group(2)))
    try:
        return Fraction(t.replace(',', ''))
    except (ValueError, ZeroDivisionError):
        return None


def check_put(expr, value):
    s = plain(expr)
    v = value_of(value)
    if s is None or v is None:
        return 'not plain'
    try:
        got = eval(s, {'F': Fraction})
    except (ZeroDivisionError, SyntaxError, TypeError, OverflowError, ValueError):
        return 'not plain'
    if isinstance(v, tuple):
        nums = [int(x) for x in re.findall(r'\d+', expr)]
        if len(nums) != 2 or nums[1] == 0:
            return 'not plain'
        return 'agree' if divmod(nums[0], nums[1]) == v[1:] else 'DIFFER'
    if value.strip().endswith('...'):
        return 'agree' if abs(got - v) < Fraction(1, 10 ** (len(value.split('.')[-1].rstrip('.')) - 0)) * 10 else 'DIFFER'
    return 'agree' if got == v else 'DIFFER'


CLOSES = ('loop close', 'answer close')
for arm in sys.argv[1:]:
    recs = json.loads((HERE / 'out' / arm / 'records.json').read_text(encoding='utf-8'))
    puts, differ = Counter(), []
    acts = Counter()
    for r in recs:
        for e in r['events']:
            if e['kind'] == 'inline value':
                c = check_put(e.get('expr', ''), e.get('value', ''))
                puts[c] += 1
                if c == 'DIFFER':
                    differ.append((r['id'], r['sample'], e.get('expr'), e.get('value')))
        kinds = {e['kind'] for e in r['events']}
        for k in CLOSES:
            if k in kinds:
                acts[(k, 'right' if r['right'] else 'wrong' if r['finish'] == 'stop' else 'cut')] += 1
        for k, n in (('early cycle', r.get('early_cycles', 0)), ('shape cap', r.get('shape_loops', 0)), ('cover cap', r.get('cover_loops', 0))):
            if n:
                acts[(k, 'right' if r['right'] else 'wrong' if r['finish'] == 'stop' else 'cut')] += 1
    total = sum(puts.values())
    print(f'== {arm}: {len(recs)} answers, {sum(r["right"] for r in recs)} right; puts {total}: {dict(puts)}')
    for d in differ[:12]:
        print('   DIFFER', d)
    for k in sorted(acts):
        print('   acted', k, acts[k])
    base = {(r['id'], r['sample']): r for r in json.loads((HERE / 'out' / 'loop-gen' / 'records.json').read_text(encoding='utf-8'))}
    lost = [(r['id'], r['sample']) for r in recs if (r['id'], r['sample']) in base and base[(r['id'], r['sample'])]['right'] and not r['right']]
    won = [(r['id'], r['sample']) for r in recs if (r['id'], r['sample']) in base and not base[(r['id'], r['sample'])]['right'] and r['right']]
    print(f'   vs loop-gen (same round 4 rows and seeds, not token-paired): right->wrong {len(lost)} {lost}  wrong->right {len(won)}')
