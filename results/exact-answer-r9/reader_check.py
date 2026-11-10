"""reader_check.py: question_read.read_exact on every row of the given row files, against gold (select_rows.right on a box holding the
value). A read that is not gold is a misread, and --exact-answer would write it into the answer: the count must be 0.

python results/exact-answer-r9/reader_check.py <rows> [<rows> ...]
"""
import json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.igc import tir4b  # noqa: E402
from tools.igc.question_read import read_exact  # noqa: E402
from tools.igc.select_rows import right  # noqa: E402


def main():
    seen, read, wrong = Counter(), Counter(), []
    for f in sys.argv[1:]:
        for line in Path(f).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row['id'].rsplit('-', 1)[0]
            seen[kind] += 1
            got = read_exact(row.get('question') or '')
            if got is None:
                continue
            read[kind] += 1
            text = '</think>\n\n\\boxed{' + tir4b.spelled(got[1]) + '}<|im_end|>'
            if not right(row, text):
                wrong.append((row['id'], got[0][:80], tir4b.spelled(got[1]), row['answer']))
    for kind in sorted(seen):
        print(f'{kind}: read {read[kind]} of {seen[kind]}')
    print('reads', sum(read.values()), 'of', sum(seen.values()), '| not gold', len(wrong))
    for w in wrong:
        print('  NOT GOLD', *w)


if __name__ == '__main__':
    main()
