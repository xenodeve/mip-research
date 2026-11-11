"""rescore_forms.py: round 9 (240 pairs, 4B NVFP4 W4A8 · llama.cpp · PC layer) by the rule with the form rule of 2026-11-11: a correctly
rounded or cut decimal of a gold with no finite decimal is right, noted. Prints totals per arm and every noted answer with what the
model boxed and the gold.
python results/exact-answer-r9/rescore_forms.py"""
import json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding='utf-8')
from tools.igc.select_rows import right  # noqa: E402
from tools.igc.standard_score import form_note  # noqa: E402

W = Path('C:/Users/xenod/mip-work/a100')
rows = {}
for f in ('frozen9a-rows.jsonl', 'frozen9b-rows.jsonl', 'frozen9a-move-rows.jsonl'):
    for line in (W / f).read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r['id']] = r
res, notes = {}, {}
for arm in ('gen', 'ctrl'):
    res[arm], notes[arm] = {}, []
    for tag in ('a', 'b', 'a3'):
        for rec in json.loads((W / 'pc' / f'r9{tag}-{arm}' / 'records.json').read_text(encoding='utf-8')):
            row = rows[rec['id']]
            key = (rec['id'], rec['sample'])
            res[arm][key] = right(row, rec['text'])
            note = form_note(rec['text'], row['answer']) if row['suite'] in ('stdmath', 'arith') else None
            if note:
                notes[arm].append((f"{rec['id']}#{rec['sample']}", note))
keys = set(res['gen']) & set(res['ctrl'])
g = sum(res['gen'][k] for k in keys)
c = sum(res['ctrl'][k] for k in keys)
go = sum(res['gen'][k] and not res['ctrl'][k] for k in keys)
co = sum(res['ctrl'][k] and not res['gen'][k] for k in keys)
n = go + co
p = 2 * sum(math.comb(n, i) for i in range(min(go, co) + 1)) / 2 ** n
print(f'pairs {len(keys)} | General {g} (noted {len(notes["gen"])}) | control {c} (noted {len(notes["ctrl"])}) | General alone {go}, control alone {co}, p = {p:.1e}')
for arm in ('gen', 'ctrl'):
    for key, note in notes[arm]:
        print(f'  {arm:4s} {key:32s} {note}')
