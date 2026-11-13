"""long_gate.py: the cut put of a long exact value (tir4b._long_found) replayed on stored text. For every `=` in the given record dirs whose
expression the normal read refuses and the wide read takes, the value it would put against what the model wrote next. Classes:
  agree            the model's decimal is the value cut or rounded at its own places (within one unit of its last place)
  slip             the model's decimal is another value: a model slip, or a misread (each one is listed, to be read by hand)
  expression next  the model wrote an expression or a word after `=`, not a number
Prefilter (CPU): only an `=` whose line has a `^`, or three or more decimals, or a number of 15 or more digits; a long value without
any of these is not counted. Old puts cannot change: _long_found runs only where the normal read and the tuple and plus-minus readers
found nothing, which this script checks at every count.
python scripts/pc/long_gate.py <records dir> ... > out.txt
"""
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding='utf-8')
from tools.igc import tir4b  # noqa: E402
from tools.igc.unified_detect import expression_before_with_mathbackend  # noqa: E402

EQ = re.compile(r'(?<![=<>!\\])=(?!=)')
AFTER = re.compile(r'\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)')
OPERATOR = re.compile(r'\s*(?:\^|\\times|\\cdot|\*|/|\\div|\+|-\s*\d|\()')   # the number starts an expression
DEC = re.compile(r'\d+\.\d+')


def wanted(line):
    return '^' in line or len(DEC.findall(line)) >= 3 or re.search(r'\d{15}', line) is not None


def classify(put, written):
    true_cut = Fraction(put.rstrip('.').rstrip('.'))
    w = Fraction(written.replace(',', ''))
    places = len(written.split('.')[1]) if '.' in written else 0
    # the put is cut at its 10 digits; the model's value at fewer places is right when it is within one unit of its last place
    return 'agree' if abs(w - true_cut) <= Fraction(1, 10 ** places) else 'slip'


def main():
    counts = {'seen': 0, 'put': 0, 'agree': 0, 'slip': 0, 'expression next': 0, 'in thinking': 0}
    questions, slips = set(), []
    for d in sys.argv[1:]:
        try:
            recs = json.loads((Path(d) / 'records.json').read_text(encoding='utf-8'))
        except Exception as e:
            print('SKIP', d, e)
            continue
        for rec in recs:
            text = rec.get('text') or ''
            close = text.find('</think>')
            for m in EQ.finditer(text):
                start = text.rfind('\n', 0, m.start()) + 1
                if not wanted(text[start:m.start()]):
                    continue
                counts['seen'] += 1
                before = text[max(0, m.start() - 400):m.start()]
                if expression_before_with_mathbackend(before) is not None:
                    continue   # the normal read takes it: _long_found is never reached there
                long = tir4b._long_found(before)
                if long is None:
                    continue
                if tir4b.inline_found(before, use_mathbackend=True) != long:
                    continue   # a tuple or plus-minus reader took it first
                counts['put'] += 1
                counts['in thinking'] += close < 0 or m.start() < close
                questions.add(rec['id'])
                a = AFTER.match(text, m.end())
                if not a or OPERATOR.match(text, a.end()):
                    counts['expression next'] += 1
                    continue
                cls = classify(long[1], a.group(1))
                counts[cls] += 1
                if cls == 'slip':
                    line = text[start:m.end() + 40].replace('\n', ' | ')
                    slips.append((f"{Path(d).parent.name}/{Path(d).name}:{rec['id']}#{rec.get('sample', 0)}", long[0][:60], long[1],
                                  a.group(1)[:30], line[-200:]))
    counts['distinct questions'] = len(questions)
    print(counts)
    for s in slips:
        print('SLIP', s[0], '| read', s[1], '=', s[2], '| model', s[3], '|', s[4])


if __name__ == '__main__':
    main()
