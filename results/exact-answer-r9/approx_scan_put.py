"""approx_scan_put.py: as approx_scan.py, but X is read by the reader the `=` put uses (unified_detect.expression_before_with_mathbackend,
with General Style's refusals via tir4b.inline_found): what a put at `\approx` would actually see. Counts per arm, wrong ones listed.

python results/exact-answer-r9/approx_scan_put.py <pc dir>
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.igc import tir4b  # noqa: E402
from tools.igc.unified_detect import expression_before_with_mathbackend as raw  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from approx_scan import APPROX  # noqa: E402


def main():
    pc = Path(sys.argv[1])
    for arm in ('gen', 'ctrl'):
        seen = read = wrong = 0
        bad = []
        for half in ('a', 'b'):
            d = pc / f'r9{half}-{arm}'
            for rec in json.loads((d / 'records.json').read_text(encoding='utf-8')):
                text = rec['text']
                for m in APPROX.finditer(text):
                    seen += 1
                    before = text[max(0, m.start() - 600):m.start()]
                    put = tir4b.inline_found(before, use_mathbackend=True)
                    got = raw(before)
                    if put is None or got is None or not hasattr(got[1], 'denominator'):
                        continue
                    read += 1
                    written = m.group(1).replace(',', '')
                    places = len(written.split('.')[1]) if '.' in written else 0
                    unit = 10 ** -places
                    true = float(got[1])
                    if abs(float(written) - true) > unit * 1.0001 + 1e-12 * abs(true):
                        wrong += 1
                        bad.append((f"{rec['id']}#{rec['sample']}", got[0][:50], 'model', written, 'put', put[1][:24]))
        print(arm, 'approx with a decimal', seen, '| read by the = reader', read, '| model value wrong', wrong)
        for b in bad[:25]:
            print('   ', *b)


if __name__ == '__main__':
    main()
