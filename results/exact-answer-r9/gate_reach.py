"""gate_reach.py: where --verify-gate's case (an equation given, all its solutions asked) is read: MATH-500, GSM8K test, stored rows.
Every read is printed so it can be checked that each is an all-solutions question.
python results/exact-answer-r9/gate_reach.py <MATH-500 test.jsonl> <gsm8k parquet>"""
import glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding='utf-8')
from tools.igc.tir4b import _ALL_SOLUTIONS  # noqa: E402


def due(q):
    q = q.split('\n\nPut the final answer')[0]
    return '=' in q and _ALL_SOLUTIONS.search(q) is not None


math = [json.loads(x) for x in Path(sys.argv[1]).read_text(encoding='utf-8').splitlines() if x.strip()]
import polars as pl  # noqa: E402
gsm = list(pl.read_parquet(sys.argv[2])['question'])
stored = {}
for f in glob.glob('C:/Users/xenod/mip-work/a100/*-rows.jsonl'):
    for line in open(f, encoding='utf-8'):
        if line.strip():
            r = json.loads(line)
            stored[r['id']] = r.get('question') or ''
for name, qs in (('MATH-500', [r['problem'] for r in math]), ('GSM8K', gsm), ('stored', list(stored.values()))):
    hits = [q for q in qs if due(q)]
    print(f'{name}: {len(qs)} questions, {len(hits)} read')
    for q in hits:
        print('   ', repr(q[:150]))
