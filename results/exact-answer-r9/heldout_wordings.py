"""heldout_wordings.py: the widened question reader on wordings it was NOT written from (list fixed 2026-11-11 before the run; not
the 10 of phrasing.py, not the tests). Three checks on the 80 questions the reader reads in their own wording:
1. new request wordings (should read, same value);
2. the same expressions written in LaTeX (should read, same value);
3. wordings that change what is asked (should read nothing).
Then reach over every stored row: reads per family and agreement with gold (a read that disagrees with gold is a false read).

python results/exact-answer-r9/heldout_wordings.py
"""
import glob, json, re, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.igc.question_read import read_exact  # noqa: E402
from tools.igc.select_rows import right  # noqa: E402
from tools.igc.tir4b import prompt_text, spelled  # noqa: E402

SHOULD_READ = [
    'Would you work out {e} please?',
    'I want to know what {e} is.',
    'Help me compute {e}.',
    'Determine the value of {e}.',
    'Give me the result of {e}, thanks.',
    'OK so what does {e} equal?',
    'Calculate the following expression: {e}',
    'Solve: {e}',
]
SHOULD_NOT = [
    'What is {e} to three significant figures?',
    'Express {e} as a percentage.',
    'What is {e} in hexadecimal?',
    'Is {e} an integer?',
    'What is the floor of {e}?',
    'What is {e} squared?',
    'Find {e} and then add 5.',
    'What is the absolute value of {e}?',
    'Estimate {e}.',
    'What is {e} modulo 3?',
]


def latex(e):
    """The DeepMind expression in LaTeX: `*` as `\\times`, `a/b` of two plain numbers as `\\frac{a}{b}`, `sqrt(n)` as `\\sqrt{n}`."""
    e = re.sub(r'sqrt\((\d+)\)', r'\\sqrt{\1}', e)
    e = re.sub(r'\((-?\d+)\)/\((-?\d+)\)', r'\\frac{\1}{\2}', e)
    e = re.sub(r'(?<![\d)])(\d+)/(\d+)(?![\d(])', r'\\frac{\1}{\2}', e)
    e = e.replace('**', '^').replace('*', ' \\times ')
    return '$' + e + '$'


def main():
    rows = {}
    for f in glob.glob('C:/Users/xenod/mip-work/a100/*-rows.jsonl'):
        for line in open(f, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                rows[r['id']] = r
    base = [(rid, read_exact(prompt_text(r))) for rid, r in rows.items() if rid.startswith('dm_')]
    base = [(rid, got) for rid, got in base if got is not None]
    print('read in their own wording:', len(base))
    print('1. should read')
    for w in SHOULD_READ:
        got = [read_exact(w.format(e=g[0])) for _, g in base]
        print(f'   {w:45s} read {sum(x is not None for x in got):3d}/{len(base)}  same value {sum(x is not None and x[1] == g[1] for x, (_, g) in zip(got, base)):3d}')
    got = [read_exact('Compute ' + latex(g[0]) + '.') for _, g in base]
    print(f'2. LaTeX `Compute $...$.`                       read {sum(x is not None for x in got):3d}/{len(base)}  same value {sum(x is not None and x[1] == g[1] for x, (_, g) in zip(got, base)):3d}')
    miss = [latex(g[0]) for x, (_, g) in zip(got, base) if x is None][:3]
    if miss:
        print('      e.g. missed', miss)
    print('3. should read nothing')
    for w in SHOULD_NOT:
        got = [read_exact(w.format(e=g[0])) for _, g in base]
        print(f'   {w:45s} read {sum(x is not None for x in got):3d}/{len(base)}')
    print('4. reach over every stored row')
    fam = defaultdict(lambda: [0, 0, 0])
    false = []
    for rid, r in rows.items():
        k = rid.rsplit('-', 1)[0] if rid.startswith('dm') else rid.split('-')[0]
        fam[k][0] += 1
        got = read_exact(prompt_text(r))
        if got is None:
            continue
        fam[k][1] += 1
        ok = right(r, '</think>\n\n\\boxed{' + spelled(got[1]) + '}')
        fam[k][2] += bool(ok)
        if not ok:
            false.append((rid, got[0][:60], str(r.get('answer'))[:30]))
    for k in sorted(fam):
        print(f'   {k:28s} rows {fam[k][0]:4d} read {fam[k][1]:4d} equal to gold {fam[k][2]:4d}')
    print('   reads that disagree with gold:', false[:10])


if __name__ == '__main__':
    main()
