"""approx_gate.py: --approx-put replayed on stored text (P11 gate). For every `\\approx` / `≈` in the answers of the given record dirs, what
approx_found reads before it and the value it would put, against the decimal the model wrote. Classes:
  agree      the model's decimal is the put value rounded or cut at its own places (the reader read what the model computed)
  sci        the model wrote `d \\times 10^{k}` and its mantissa agrees once the power is read
  estimate   the model wrote 1-2 significant digits within 10 %
  off        anything else: a model slip, or a misread (each one is listed to be read by hand)
python results/exact-answer-r9/approx_gate.py <records dir> ... > out.txt
"""
import json, math, re, sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding='utf-8')
from tools.igc.approx_put import approx_found  # noqa: E402

MARK = re.compile(r'\\approx|≈')
AFTER = re.compile(r'\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)(\s*\\times\s*10\^\{?(-?\d+)\}?)?')


def value_of(spelled):
    m = re.fullmatch(r'(-?[\d.]+)(?: \\times 10\^\{(-?\d+)\})?', spelled)
    return Fraction(m.group(1)) * (Fraction(10) ** int(m.group(2)) if m.group(2) else 1)


def classify(true, written, power):
    w = Fraction(written.replace(',', ''))
    if power is not None:
        w *= Fraction(10) ** int(power)
    places = len(written.split('.')[1]) if '.' in written else 0
    unit = Fraction(1, 10 ** places) * (Fraction(10) ** int(power) if power is not None else 1)
    if abs(w - true) <= unit:
        return 'sci' if power is not None else 'agree'
    digits = len(written.replace(',', '').replace('.', '').lstrip('-0').rstrip('0'))
    if digits <= 2 and true != 0 and abs(w - true) <= abs(true) / 10:
        return 'estimate'
    return 'off'


def main():
    counts = {'seen': 0, 'read': 0, 'agree': 0, 'sci': 0, 'estimate': 0, 'off': 0, 'no number after': 0}
    off = []
    for d in sys.argv[1:]:
        recs = json.loads((Path(d) / 'records.json').read_text(encoding='utf-8'))
        for rec in recs:
            text = rec['text']
            for m in MARK.finditer(text):
                counts['seen'] += 1
                got = approx_found(text[max(0, m.start() - 400):m.start()])
                if got is None:
                    continue
                counts['read'] += 1
                a = AFTER.match(text, m.end())
                if not a:
                    counts['no number after'] += 1
                    continue
                cls = classify(value_of(got[1]), a.group(1), a.group(3))
                counts[cls] += 1
                if cls == 'off':
                    line = text[max(0, m.start() - 120):m.end() + 40].replace('\n', ' | ')
                    off.append((f"{Path(d).name}:{rec['id']}#{rec['sample']}", got[0][:60], got[1], a.group(0).strip()[:30], line))
    print(counts)
    for o in off:
        print('OFF', o[0], '| read', o[1], '=', o[2], '| model', o[3], '|', o[4])


if __name__ == '__main__':
    main()
