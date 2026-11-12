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
CUT = re.compile(r'^\s*(-?)(\d+)\.(\d+)\.\.\.\s*$')


def _sym(text):
    """A LaTeX value or expression with surds as a sympy number, or None: \sqrt{a}, \frac{a}{b}, \cdot, \times, implicit products."""
    import sympy
    t = text.replace('$', '').strip()
    for _ in range(4):
        t = re.sub(r'\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}', r'((\1)/(\2))', t)
        t = re.sub(r'\\sqrt\{([^{}]*)\}', r'sqrt(\1)', t)
    t = t.replace(r'\cdot', '*').replace(r'\times', '*').replace(r'\left', '').replace(r'\right', '').replace('{', '(').replace('}', ')')
    if '\\' in t:
        return None
    try:
        from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication_application,
                                                convert_xor)
        return parse_expr(t, transformations=standard_transformations + (implicit_multiplication_application, convert_xor),
                          local_dict={'sqrt': sympy.sqrt, 'P': sympy.ff}, evaluate=True)
    except Exception:
        return None


def _surd_right(e):
    """True / False by sympy for a value with a root in it, None when either side cannot be read."""
    import sympy
    a, b = _sym(e['expr']), _sym(e['value'])
    if a is None or b is None or a.free_symbols or b.free_symbols:
        return None
    return sympy.simplify(a - b) == 0


def value_right(e, question):
    """True / False / None (a form this check cannot read). Each General Style form by its own meaning: `q rem r` is a quotient and a
    remainder (n = q d + r, 0 <= r < d); `x.dddd...` is the value cut to its places; anything else equal in value."""
    from tools.igc.mathexpr import evaluate
    kind = e.get('what')
    if kind == 'approx value':
        from analyze import approx_value_right
        return approx_value_right(e)
    if kind == 'question value':
        from tools.igc.question_read import read_exact
        got = read_exact(question or '')
        return None if got is None else evaluate(e['value']) == Fraction(got[1])
    value = e['value']
    m = REM.match(value)
    if m:
        d = DIV.match(re.sub(r'^.*\$', '', e['expr']))
        if not d:
            return None
        n, k = int(d.group(1)), int(d.group(2))
        q, r = int(m.group(1)), int(m.group(2))
        return k != 0 and n == q * k + r and 0 <= r < abs(k)
    if r'\sqrt' in e['expr'] or r'\sqrt' in value or re.match(r'^\s*P\(', e['expr']):
        return _surd_right(e)
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


MISSED = re.compile(r'no plan for slot (\d+) seq (\d+) in')


def missed_asks(log):
    """(slot, seq) of every ask the shim gave up waiting for: a plan for it was never applied, whatever the event says."""
    return {(int(a), int(b)) for a, b in MISSED.findall(log)}


def main():
    out = Path(sys.argv[1])
    log = out.parent / 'llama.log'
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
    c, bad = Counter(), []
    for rec in json.loads((out / 'records.json').read_text(encoding='utf-8')):
        final = 'right' if rec['right'] else 'wrong'
        out_ids = ids.get((rec['id'], rec.get('sample', 0)), [])
        for e in rec.get('events', []):
            if e.get('kind') != 'steered value':
                continue
            state = 'missed' if (e.get('slot'), e.get('seq')) in missed else \
                'done' if e.get('done') else 'dropped' if e.get('dropped') else 'open'
            c[(final, state)] += 1
            ok = value_right(e, rows.get(rec['id'], {}).get('question'))
            c[(final, 'value right' if ok else 'unchecked' if ok is None else 'VALUE WRONG')] += 1
            written = tok.decode(out_ids[e['at']:e['at'] + 40]) if out_ids else ''
            want = (' ' + e['value']) if e.get('what') != 'question value' else e['expr']
            wrote = written.startswith(want) or written.lstrip().startswith(e['value'])
            c[(final, 'written as steered' if wrote else 'TEXT DIFFERS')] += 1
            if ok is False or (state == 'done' and not wrote):
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
