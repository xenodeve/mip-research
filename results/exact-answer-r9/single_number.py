"""How often the model cites the prompt's "single number" while choosing the answer's form, round 9 both arms, and how those answers
scored (rule with the form rule). Also which prompt lines the rows carry."""
import json, re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, 'D:/Github/mip-research')
sys.stdout.reconfigure(encoding='utf-8')
from tools.igc.select_rows import right
from tools.igc.standard_score import form_note

W = Path('C:/Users/xenod/mip-work/a100')
rows = {}
for f in ('frozen9a-rows.jsonl', 'frozen9b-rows.jsonl', 'frozen9a-move-rows.jsonl'):
    for line in (W / f).read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r['id']] = r
tails = Counter(r['question'].strip().split('\n')[-1] for r in rows.values())
print('prompt lines the rows end with:')
for k, v in tails.most_common():
    print(f'   {v:3d}  {k}')
CITE = re.compile(r'single number', re.I)
FORM = re.compile(r'decimal|integer (?:quotient|part)|round', re.I)
for arm in ('gen', 'ctrl'):
    tab = Counter()
    ex = []
    for tag in ('a', 'b', 'a3'):
        for rec in json.loads((W / 'pc' / f'r9{tag}-{arm}' / 'records.json').read_text(encoding='utf-8')):
            row = rows[rec['id']]
            text = rec['text']
            cites = [m.start() for m in CITE.finditer(text)]
            # a citation near a form choice: within 300 characters of `decimal`, `integer quotient/part` or `round`
            near = any(FORM.search(text[max(0, i - 300):i + 300]) for i in cites)
            if not near:
                continue
            ok = right(row, text)
            note = form_note(text, row['answer']) if row['suite'] in ('stdmath', 'arith') else None
            mod = rec['id'].rsplit('-', 1)[0]
            tab[(mod, 'right' if ok and not note else 'right, noted' if note else 'wrong')] += 1
            if not ok:
                ex.append(f"{rec['id']}#{rec['sample']}")
    print(f'\n{arm}: answers citing "single number" next to a form choice, by module and score')
    for k, v in sorted(tab.items()):
        print(f'   {k[0]:22s} {k[1]:13s} {v}')
    print('   wrong ones:', ex)
