"""gate3_values.py OUT: gate 3 of the token channel (docs/plans/2026-11-12-mip-token-channel-plan.md, task 7). CPU only.

For every `steered value` event of a --channel run (records.json and ids.jsonl in OUT): (1) the value checked again independently: an
inline value against mathexpr's exact evaluation of its expression, an approximation by results/round10/analyze.py's own check, a
question value against question_read; (2) the model's text at the event's position starts with the steered text (what the model
wrote is what was steered); (3) whether the steering finished, was dropped, or the request ended in the middle. Split by answers that
finished right and wrong. Writes OUT/gate3.txt.  python scripts/pc/gate3_values.py <run dir>
"""
import json
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'results' / 'round10'))


REM = re.compile(r'^\s*(-?\d+)\s*(?:\\text\{\s*rem\s*\}|rem)\s*(\d+)\s*$')
DIV = re.compile(r'^\s*\(?\s*(-?\d+)\s*\)?\s*(?:\\div|/|÷|:)\s*\(?\s*(-?\d+)\s*\)?\s*$')
GCD = re.compile(r'^\s*\\?gcd\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*$')
CUT = re.compile(r'^\s*(-?)(\d+)\.(\d+)\.\.\.\s*$')


def _sym(text):
    """A LaTeX value or expression as a sympy number, or None: \sqrt{a}, \frac{a}{b}, \cdot, \times, implicit products, \pi, |x|,
    \sin \cos \tan (an angle in degrees with ^\circ, in radians otherwise)."""
    import sympy
    t = text.replace('$', '').strip()
    t = re.sub(r'\^\s*\{?\s*\\circ\s*\}?', '*pi/180', t)
    t = re.sub(r'\|([^|]*)\|', r'Abs(\1)', t)
    for _ in range(4):
        t = re.sub(r'\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}', r'((\1)/(\2))', t)
        t = re.sub(r'\\sqrt\{([^{}]*)\}', r'sqrt(\1)', t)
    t = re.sub(r'\\(sin|cos|tan)\b', r'\1', t).replace(r'\pi', 'pi')
    t = t.replace(r'\cdot', '*').replace(r'\times', '*').replace(r'\div', '/').replace(r'\left', '').replace(r'\right', '').replace('{', '(').replace('}', ')')
    if '\\' in t:
        return None
    try:
        from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application,
                                                convert_xor)
        return parse_expr(t, transformations=standard_transformations + (implicit_multiplication_application, convert_xor),
                          local_dict={'sqrt': sympy.sqrt, 'P': sympy.ff, 'pi': sympy.pi, 'sin': sympy.sin, 'cos': sympy.cos,
                                      'tan': sympy.tan, 'Abs': sympy.Abs}, evaluate=True)
    except Exception:
        return None


def _one_right(expr, value):
    """True / False for one expression and one value by sympy, None when either side cannot be read. A trig function left
    unevaluated (`sin(45)`: an angle with no unit, true in degrees and false in radians) is not decided here."""
    import sympy
    a = _sym(expr)
    if a is None or a.free_symbols or a.atoms(sympy.sin, sympy.cos, sympy.tan):
        return None
    m = CUT.match(value)
    if m:   # `2.0345...`: the value cut to its places
        places = len(m.group(3))
        want = sympy.N(a, 40)
        cut = sympy.Integer(int(abs(want) * 10 ** places)) / 10 ** places * (-1 if want < 0 else 1)
        return cut == sympy.Rational(f'{m.group(1)}{m.group(2)}.{m.group(3)}')
    b = _sym(value)
    if b is None or b.free_symbols:
        return None
    return sympy.simplify(a - b) == 0


def _parts(text):
    """`(a, b, c)` as its parts at the top level, or None when it is not a tuple."""
    t = text.replace('$', '').strip()
    if not (t.startswith('(') and t.endswith(')')):
        return None
    parts, depth, cur = [], 0, ''
    for ch in t[1:-1]:
        depth += ch in '({['
        depth -= ch in ')}]'
        if ch == ',' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    parts.append(cur)
    return parts if len(parts) > 1 and depth == 0 else None


def _surd_right(e):
    """True / False by sympy for a value sympy reads (roots, pi, trig, plus-minus, tuples), None when either side cannot be read.
    `a \pm b` is both signs, on both sides alike; a tuple is right when every part is."""
    expr, value = e['expr'], e['value']
    pe, pv = _parts(expr), _parts(value)
    if pe is not None or pv is not None:
        if pe is None or pv is None or len(pe) != len(pv):
            return None
        got = [_one_right(x, y) for x, y in zip(pe, pv)]
        return None if None in got else all(got)
    if r'\pm' in expr or r'\pm' in value:
        got = [_one_right(expr.replace(r'\pm', s), value.replace(r'\pm', s)) for s in '+-']
        return None if None in got else all(got)
    return _one_right(expr, value)


def value_right(e, question):
    """True / False / None (a form this check cannot read). Each General Style form by its own meaning: `q rem r` is a quotient and a
    remainder (n = q d + r, 0 <= r < d); `x.dddd...` is the value cut to its places; anything else equal in value."""
    from tools.igc.mathexpr import evaluate
    kind = e.get('what')
    if ' + ' in (kind or ''):   # lines steered together at the start: each by its own kind
        parts = zip(kind.split(' + '), e['expr'].split(' | '), e['value'].split(' | '))
        got = [value_right(dict(what=k, expr=x, value=v), question) for k, x, v in parts]
        return False if False in got else None if None in got else True
    if kind == 'letter count':
        return letters_right(e['expr'], e['value'], question)
    if kind == 'question quote':   # the question's own words: verbatim in the question, and the line is that text
        return e['value'] == e['expr'] and re.sub(r'\s', '', e['expr']) in re.sub(r'\s', '', question or '')
    if kind == 'approx value':
        from analyze import approx_value_right
        return approx_value_right(e)
    if kind == 'question value':
        from tools.igc.question_read import read_exact
        got = read_exact(question or '')
        return None if got is None else evaluate(e['value']) == Fraction(got[1])
    value = e['value']
    g = GCD.match(e['expr'])
    if g:
        from math import gcd
        return value.strip().lstrip('-').isdigit() and int(value) == gcd(int(g.group(1)), int(g.group(2)))
    m = REM.match(value)
    if m:
        d = DIV.match(re.sub(r'^.*\$', '', e['expr']))
        if not d:
            return None
        n, k = int(d.group(1)), int(d.group(2))
        q, r = int(m.group(1)), int(m.group(2))
        return k != 0 and n == q * k + r and 0 <= r < abs(k)
    if re.search(r'\\(sqrt|pi|sin|cos|tan|pm)\b|\||^\s*P\(|^\s*\(.*,', e['expr']) or re.search(r'\\(sqrt|pi|pm)\b', value):
        return _surd_right(e)
    want = evaluate(e['expr'])
    if want is None:   # a value too long to be a step (`1.01^{30}`), put cut: read in mathexpr's wide mode
        from tools.igc.mathexpr import wide
        with wide():
            want = evaluate(e['expr'])
    if want is None:
        return None
    m = CUT.match(value)
    if m:
        places = len(m.group(3))
        sign = -1 if want < 0 else 1
        cut = Fraction(int(abs(want) * 10 ** places), 10 ** places) * sign
        return cut == Fraction(evaluate(f'{m.group(1)}{m.group(2)}.{m.group(3)}'))
    got = evaluate(value)
    return None if got is None else got == want


LETTERS = re.compile(r'^The string (\S+) has (\d+) letters: (.*)\.$')


def letters_right(string, line, question):
    """A letter-count line counted again with str.count (not tools/igc/letters.py): the string is the question's, its length and
    every letter's count right, every letter of it named."""
    m = LETTERS.match(line.strip())
    if not m or m.group(1) != string or string not in (question or ''):
        return False
    counts = dict(x.rsplit(' ', 1) for x in m.group(3).split(', '))
    return int(m.group(2)) == len(string) and set(counts) == set(string) and all(int(n) == string.count(k) for k, n in counts.items())


def written_want(e):
    """The text the model should have written at the event's position."""
    from tools.igc.question_quote import QUOTE
    out = []
    for k, x, v in zip(e['what'].split(' + '), e['expr'].split(' | '), e['value'].split(' | ')):
        out.append(f'{x} = {v}\n' if k == 'question value' else v + '\n' if k == 'letter count'
                   else QUOTE.format(expr=x) if k == 'question quote' else ' ' + v)
    return ''.join(out).rstrip('\n')


MISSED = re.compile(r'no plan for slot (\d+) seq (\d+) in')


def missed_asks(log):
    """(slot, seq) of every ask the shim gave up waiting for: a plan for it was never applied, whatever the event says."""
    return {(int(a), int(b)) for a, b in MISSED.findall(log)}


def steer_state(e, missed, closes):
    """missed / cut / done / dropped / open. `cut`: the value sits at or past a loop close of the driver, which cut the looping text
    there and wrote the close and the answer in its place; the event is kept, but nothing was steered at that place of the text."""
    if (e.get('slot'), e.get('seq')) in missed:
        return 'missed'
    if any(e['at'] >= c for c in closes):
        return 'cut'
    return 'done' if e.get('done') else 'dropped' if e.get('dropped') else 'open'


def main():
    out = Path(sys.argv[1])
    log = out.parent / 'llama.log'
    if not log.exists():
        log = out.parent / 'shim_misses.log'   # the copy in results/ keeps only the shim's miss lines
    missed = missed_asks(log.read_text(encoding='utf-8', errors='replace')) if log.exists() else set()
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
    c, bad, per = Counter(), [], Counter()
    for rec in json.loads((out / 'records.json').read_text(encoding='utf-8')):
        final = 'right' if rec['right'] else 'wrong'
        out_ids = ids.get((rec['id'], rec.get('sample', 0)), [])
        closes = [e['at'] for e in rec.get('events', []) if e.get('kind') == 'loop close']
        for e in rec.get('events', []):
            if e.get('kind') != 'steered value':
                continue
            state = steer_state(e, missed, closes)
            c[(final, state)] += 1
            ok = value_right(e, rows.get(rec['id'], {}).get('question'))
            c[(final, 'value right' if ok else 'unchecked' if ok is None else 'VALUE WRONG')] += 1
            # a question line can be long and written a digit a token: 200 tokens, not 40, so the check never cuts it short
            written = tok.decode(out_ids[e['at']:e['at'] + 200]) if out_ids else ''
            want = written_want(e)
            wrote = written.startswith(want) or written.lstrip().startswith(want.lstrip())
            if state != 'cut':   # past a close the text there is the close and the answer, not the looping text the value was in
                c[(final, 'written as steered' if wrote else 'TEXT DIFFERS')] += 1
            what = e.get('what') or '?'   # each mechanism apart (developer, 2026-11-13)
            per[(what, 'plans')] += 1
            per[(what, state)] += 1
            per[(what, 'value right' if ok else 'unchecked' if ok is None else 'VALUE WRONG')] += 1
            per[(what, f'on {final} answers')] += 1
            if state != 'cut':
                per[(what, 'written as steered' if wrote else 'TEXT DIFFERS')] += 1
            if ok is False or (state == 'done' and not wrote):
                bad.append(f"{rec['id']}#{rec.get('sample', 0)} {final} {state} {e.get('what')}: {e['expr'][:60]!r} = {e['value']!r}; "
                           f"text {written[:40]!r}")
    lines = [f'gate 3 (token channel): {out}', ' '.join(f'{k[0]}/{k[1]}: {n}' for k, n in sorted(c.items()))]
    wrong = sum(n for k, n in c.items() if k[1] == 'VALUE WRONG')
    lines.append('PASS: every steered value right' if wrong == 0 else f'{wrong} VALUES WRONG (listed)')
    lines.append('per mechanism:')
    for what in sorted({k[0] for k in per}):
        lines.append(f'  {what}: ' + ', '.join(f'{k[1]} {n}' for k, n in sorted(per.items()) if k[0] == what))
    lines += bad[:60]
    (out / 'gate3.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:3]))


if __name__ == '__main__':
    main()
