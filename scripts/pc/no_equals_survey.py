"""no_equals_survey.py: where the model writes a value software can compute without an `=` before it (token channel: MIP reads every
token, so a value need not wait for `=`). CPU only, stored text.

For every `<expression> <link> <number>` in the thinking of the given record dirs, with a link that states the expression's value
(`is`, `are`, `gives`, `equals`, `makes`, `yields`, `becomes`, `we get`, `->`, `\\to`, `=>`, `\\Rightarrow`), mathexpr reads the
expression and the model's number is compared with it: agree (the value, or it cut or rounded at the places it shows), slip, or the
expression unread. Long division in words (`13 into 92 is 7`) and remainders (`Remainder 5`, `rem 5`) are counted as their own forms.
Counts per link, by answers that finished right and wrong, with distinct questions; slips listed for reading by hand.
python scripts/pc/no_equals_survey.py <records dir> ... > out.txt
"""
import json
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding='utf-8')
from tools.igc.mathexpr import expression_before  # noqa: E402

LINK = re.compile(r'\s*(?:\\\)|\$)?\s*(?P<link>is|are|gives|equals|makes|yields|becomes|we get|->|\\to|=>|\\Rightarrow|→)\s+\$?\s*'
                  r'(?P<num>-?\d[\d,]*(?:\.\d+)?)(?![\d.]*\s*(?:\^|\\times|\\cdot|\*|/|\\div|\+|\())')
INTO = re.compile(r'(?<![\d.])(\d+) (?:goes )?into (\d+) (?:goes |is |=\s*)(\d+)')
REM = re.compile(r'(?<![\d.])(\d+)\s*(?:/|\\div|÷|divided by)\s*(\d+)[^\n]{0,40}?(?:remainder|rem\.?|R)\s*(?:of\s*|is\s*|=\s*)?(\d+)', re.I)


def agree(value, written):
    w = Fraction(written.replace(',', ''))
    places = len(written.split('.')[1]) if '.' in written else 0
    return abs(w - value) < Fraction(1, 10 ** places) if places else w == value


def main():
    c = defaultdict(Counter)
    questions = defaultdict(set)
    slips = []
    for d in sys.argv[1:]:
        p = Path(d) / 'records.json'
        if not p.exists():
            continue
        for rec in json.loads(p.read_text(encoding='utf-8')):
            text = rec.get('text') or ''
            close = text.find('</think>')
            think = text if close < 0 else text[:close]
            final = 'right' if rec.get('right') else 'wrong'
            for m in re.finditer(r'(?=\s*(?:\\\)|\$)?\s*(?:is|are|gives|equals|makes|yields|becomes|we get|->|\\to|=>|\\Rightarrow|→)\s)', think):
                lm = LINK.match(think, m.start())
                if not lm:
                    continue
                link = lm.group('link')
                line = think[think.rfind('\n', 0, m.start()) + 1:m.start()]
                got = expression_before(line)
                if got is None:
                    continue
                expr, value = got
                if not re.search(r'[-+*/^×÷·]|\\(?:times|cdot|div|frac|sqrt)', expr):
                    continue   # a lone number (`x is 5`) is no computation
                key = f'link {link}'
                c[key]['read'] += 1
                ok = agree(Fraction(value), lm.group('num'))
                c[key][f'{"agree" if ok else "slip"} on {final}'] += 1
                questions[key].add(rec['id'])
                if not ok:
                    slips.append(f"{Path(d).name}:{rec['id']}#{rec.get('sample', 0)} {final} | {expr[:60]} {link} {lm.group('num')} "
                                 f"(true {value}) | {line[-80:]}")
            for m in INTO.finditer(think):
                d_, n, q = map(int, m.groups())
                ok = d_ != 0 and n // d_ == q
                c['long division in words'][f'{"agree" if ok else "slip"} on {final}'] += 1
                questions['long division in words'].add(rec['id'])
                if not ok:
                    slips.append(f"{Path(d).name}:{rec['id']}#{rec.get('sample', 0)} {final} | {m.group(0)} (true {n // d_ if d_ else '-'})")
            for m in REM.finditer(think):
                n, d_, r = map(int, m.groups())
                ok = d_ != 0 and n % d_ == r
                c['remainder'][f'{"agree" if ok else "slip"} on {final}'] += 1
                questions['remainder'].add(rec['id'])
                if not ok:
                    slips.append(f"{Path(d).name}:{rec['id']}#{rec.get('sample', 0)} {final} | {m.group(0)[:80]} (true {n % d_ if d_ else '-'})")
    for k in sorted(c):
        print(k, dict(c[k]), 'distinct questions', len(questions[k]))
    for s in slips:
        print('SLIP', s)


if __name__ == '__main__':
    main()
