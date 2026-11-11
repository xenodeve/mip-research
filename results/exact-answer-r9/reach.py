"""Where does read_exact (the trigger of --exact-answer) fire, by row family, across every stored row file? Reads and agreement with gold."""
import json, sys, glob
from collections import defaultdict
sys.path.insert(0, 'D:/Github/mip-research')
from tools.igc.question_read import read_exact
from tools.igc.select_rows import right
from tools.igc.tir4b import spelled, prompt_text

seen = {}
for f in glob.glob('C:/Users/xenod/mip-work/a100/*-rows.jsonl'):
    for l in open(f, encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            seen[r['id']] = r
fam = defaultdict(lambda: [0, 0, 0])   # rows, read, read and gold
for rid, r in seen.items():
    k = rid.split('-')[0]
    if k.startswith('dm'):
        k = 'deepmind:' + rid.rsplit('-', 1)[0]
    fam[k][0] += 1
    try:
        got = read_exact(prompt_text(r))
    except Exception:
        got = None
    if got is None:
        continue
    fam[k][1] += 1
    fam[k][2] += bool(right(r, '</think>\n\n\\boxed{' + spelled(got[1]) + '}'))
tot = [0, 0, 0]
for k in sorted(fam):
    print(f'{k:40s} rows {fam[k][0]:4d} read {fam[k][1]:4d} gold {fam[k][2]:4d}')
    for i in range(3):
        tot[i] += fam[k][i]
print('all', tot)
