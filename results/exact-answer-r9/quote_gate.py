"""quote_gate.py: where --question-quote's copy trigger fires on stored text (round 9 and A100 traces). Every fire is printed with the
model's line and the question's expression, to be read by hand: each must be a copy that says something else than the question.
python results/exact-answer-r9/quote_gate.py <rows jsonl> -- <records dir> ..."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding='utf-8')
from tools.igc.question_read import read_exact  # noqa: E402
from tools.igc.question_quote import copy_differs, numbers_of  # noqa: E402

cut = sys.argv.index('--')
rows = {}
for f in sys.argv[1:cut]:
    for line in Path(f).read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r['id']] = r
answers = fires = 0
seen_q = 0
for d in sys.argv[cut + 1:]:
    for rec in json.loads((Path(d) / 'records.json').read_text(encoding='utf-8')):
        row = rows.get(rec['id'])
        if not row or not row.get('question'):
            continue
        got = read_exact(row['question'])
        if got is None:
            continue
        seen_q += 1
        nums = numbers_of(got[0])
        think = rec['text'].split('</think>')[0]
        for line in think.split('\n'):
            if copy_differs(line, nums, got[1]):
                fires += 1
                print(f"FIRE {Path(d).name}:{rec['id']}#{rec['sample']} right={rec.get('right')}\n   question {got[0]}\n   model    {line.strip()[:200]}")
                break
print('answers with a one-computation question', seen_q, '| answers where it fires', fires)
