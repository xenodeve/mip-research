"""approx_scan.py: how often the model's own value after `\\approx` is wrong, where what stands before it is a computation software can
do (round 9, 4B NVFP4 W4A8 · llama.cpp · PC layer). The gate before building a put at `\\approx`.

For every `X \\approx D` (D a plain decimal): X is the longest tail of the text before `\\approx` (on its line, after the last `=` or
`\\approx`) that exactexpr reads to a value; the model's D is right when it is the value rounded or cut at D's places (one unit in the
last place either way). Prints counts per arm and the wrong ones.

python results/exact-answer-r9/approx_scan.py <pc dir>
"""
import json, math, re, sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.igc.exactexpr import NotExact, evaluate as exact  # noqa: E402

APPROX = re.compile(r'(?:\\approx|≈)\s*\$?\s*(-?\d[\d,]*\.\d+|-?\d[\d,]*)(?![\d.])')


def value_of(expr):
    try:
        v = exact(expr)
    except (NotExact, ArithmeticError, ValueError, OverflowError, RecursionError, KeyError, IndexError, TypeError):
        return None
    if v is None or v.has_pi():
        return None
    return v


def as_float(v):
    q = v.rational()
    if q is not None:
        return float(q)
    return sum(float(c) * math.sqrt(r) for (r, _), c in v.t.items())


def expression_before(text):
    line = re.split(r'\n|\$\$|\\\[|=|\\approx|≈|:', text)[-1]
    line = line.lstrip(' $(')
    words = line.split(' ')
    for i in range(len(words)):
        cand = ' '.join(words[i:]).strip().strip('$').strip()
        if not cand or not re.search(r'\d', cand) or not re.search(r'[-+*/^×÷]|\\times|\\cdot|\\frac|\\sqrt|\^', cand):
            continue
        if re.search(r'[A-Za-z]', re.sub(r'\\(?:times|cdot|frac|dfrac|sqrt|left|right)', '', cand)):
            continue
        v = value_of(cand)
        if v is not None:
            return cand, v
    return None


def main():
    pc = Path(sys.argv[1])
    for arm in ('gen', 'ctrl'):
        seen = computable = wrong = 0
        bad = []
        for half in ('a', 'b'):
            d = pc / f'r9{half}-{arm}'
            for rec in json.loads((d / 'records.json').read_text(encoding='utf-8')):
                text = rec['text']
                for m in APPROX.finditer(text):
                    seen += 1
                    got = expression_before(text[max(0, m.start() - 300):m.start()])
                    if got is None:
                        continue
                    computable += 1
                    written = m.group(1).replace(',', '')
                    places = len(written.split('.')[1]) if '.' in written else 0
                    true = as_float(got[1])
                    unit = 10 ** -places
                    if abs(float(written) - true) > unit * 1.0001 + 1e-12 * abs(true):
                        wrong += 1
                        bad.append((f"{rec['id']}#{rec['sample']}", got[0][:60], written, f'{true:.10g}'))
        print(arm, '`\\approx` with a decimal', seen, '| computation before it', computable, '| model value wrong', wrong)
        for b in bad[:40]:
            print('   ', *b)


if __name__ == '__main__':
    main()
