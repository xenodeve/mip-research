"""The situation behind --exact-answer is "the whole question is one computation". Does read_exact see that situation in any wording,
or only in DeepMind's? Each of the 80 read questions is re-asked in other common wordings (fixed list, written before the run, not
adjusted after); counts per wording how many are read, and to the same value."""
import json, sys, glob, re
sys.path.insert(0, 'D:/Github/mip-research')
from tools.igc.question_read import read_exact
from tools.igc.tir4b import prompt_text

WORDINGS = [
    'What is {e}?',
    'Compute {e}.',
    'Evaluate {e}.',
    'Work out {e}.',
    'Find the value of {e}.',
    '{e} = ?',
    'Can you calculate {e} for me?',
    'Please simplify {e}.',
    'What do you get when you compute {e}?',
    '{e}',
]
rows = {}
for f in glob.glob('C:/Users/xenod/mip-work/a100/*-rows.jsonl'):
    for l in open(f, encoding='utf-8'):
        if l.strip():
            r = json.loads(l)
            rows[r['id']] = r
base = []
for rid, r in rows.items():
    q = prompt_text(r)
    got = read_exact(q)
    if got is not None:
        base.append((rid, q, got))
print('read in the original wording:', len(base))
print('example original:', base[0][1][:160].replace('\n', ' '), '| expr:', base[0][2][0][:80])
for w in WORDINGS:
    n = same = 0
    miss = []
    for rid, q, got in base:
        g = read_exact(w.format(e=got[0]))
        if g is not None:
            n += 1
            same += g[1] == got[1]
        else:
            miss.append(rid)
    print(f'{w:42s} read {n:3d}/{len(base)} same value {same:3d}  e.g. missed {miss[:2]}')
