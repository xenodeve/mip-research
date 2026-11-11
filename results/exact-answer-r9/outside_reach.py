"""outside_reach.py: the widened question reader on sets it was never written from: MATH-500 (500) and GSM8K test (1,319). Every read
is listed with gold; a read whose value is not gold is a false read (the box rule would write a wrong answer).

python results/exact-answer-r9/outside_reach.py <MATH-500 test.jsonl> <gsm8k test parquet>
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.igc.question_read import read_exact  # noqa: E402
from tools.igc.standard_score import math_right  # noqa: E402
from tools.igc.tir4b import spelled  # noqa: E402


def check(name, items):
    reads = false = 0
    for q, gold in items:
        got = read_exact(q)
        if got is None:
            continue
        reads += 1
        ok = math_right('</think>\n\n\\boxed{' + spelled(got[1]) + '}', gold)
        false += not ok
        print(f'  {name} {"ok   " if ok else "FALSE"} gold {gold[:30]!s:32} read {got[0][:60]!s:62} = {spelled(got[1])[:30]}  | {q[:90]!r}')
    print(f'{name}: {len(items)} questions, read {reads}, not gold {false}')


def main():
    math = [json.loads(x) for x in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if x.strip()]
    check('MATH-500', [(r['problem'], r['answer']) for r in math])
    import polars as pl
    gsm = pl.read_parquet(sys.argv[2])
    check('GSM8K', [(q, a.split('####')[-1].strip().replace(',', '')) for q, a in zip(gsm['question'], gsm['answer'])])


if __name__ == '__main__':
    main()
