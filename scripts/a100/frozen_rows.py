"""Build a frozen row set on this PC: questions no rule, threshold or closer was written or tuned from.

Same make-up as round 1's `fresh-rows.jsonl` (8 DeepMind mathematics modules x 10, MATH-500 level 5 x 40), so the result can be set
beside round 1, but drawn with another seed and with every question id that appears anywhere in the stored work removed:
- DeepMind: the seed-0 rows the old dm arms ran, and every `dm_<module>-<index>` id found in SEEN files;
- MATH-500: every `stdmath-<hash>` id found in SEEN files.

Round 9 needs this: round 1's 120 rows hold all 24 loop rows the closers (`--shape-cap`, `--cover-cap`, A, B, V7) were tuned on.

python -I scripts/a100/frozen_rows.py --dm-tar PATH --math500 PATH --seen DIR [DIR ...] --seed 2 --out ROWS.jsonl
The answers go into the rows file: keep it out of this public repository (WORK dir, see README)."""
import argparse
import json
import random
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.igc.external_score import question_id  # noqa: E402
from tools.igc.standard_score import normalize_math  # noqa: E402

SUFFIX = '\n\nGive the final answer as a single number in \\boxed{} at the end.'
SUFFIX_ANY = '\n\nPut the final answer in \\boxed{} at the end.'
MODULES = {
    'dm_mixed_longer': 'extrapolate/arithmetic__mixed_longer',
    'dm_mul_div_longer': 'extrapolate/arithmetic__mul_div_multiple_longer',
    'dm_div_big': 'extrapolate/arithmetic__div_big',
    'dm_simplify_surd': 'interpolate/arithmetic__simplify_surd',
    'dm_div_remainder': 'interpolate/numbers__div_remainder',
    'dm_poly_evaluate': 'interpolate/polynomials__evaluate',
    'dm_conversion': 'interpolate/measurement__conversion',
    'dm_swr_p_sequence': 'interpolate/probability__swr_p_sequence',
}
NUMERIC = {'dm_mixed_longer', 'dm_mul_div_longer', 'dm_div_big', 'dm_div_remainder', 'dm_poly_evaluate', 'dm_conversion'}
USED_SEED0 = {'extrapolate/arithmetic__mul_div_multiple_longer', 'extrapolate/arithmetic__div_big'}
ID = re.compile(r'(?:stdmath-[0-9a-f]{8}|dm_[a-z_]+-\d+)')
TEXT = {'.json', '.jsonl', '.txt', '.md', '.log', '.py', '.sh', '.s'}


def seen_ids(dirs):
    seen = set()
    for d in dirs:
        for p in Path(d).rglob('*'):
            if p.is_file() and p.suffix in TEXT and p.stat().st_size < 400_000_000:
                seen.update(ID.findall(p.read_text(encoding='utf-8', errors='replace')))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dm-tar', required=True)
    ap.add_argument('--math500', required=True, help='HuggingFaceH4/MATH-500 test.jsonl')
    ap.add_argument('--seen', nargs='+', required=True)
    ap.add_argument('--seed', type=int, default=2)
    ap.add_argument('--per-module', type=int, default=10)
    ap.add_argument('--math', type=int, default=40)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    seen = seen_ids(a.seen)
    print('seen ids', len(seen), dict(Counter(i.rsplit('-', 1)[0] for i in seen)))

    want = {f'mathematics_dataset-v1.0/{m}.txt': name for name, m in MODULES.items()}
    texts = {}
    with tarfile.open(a.dm_tar) as t:
        for member in t:
            if member.name in want:
                texts[want[member.name]] = t.extractfile(member).read().decode('utf-8')
    assert set(texts) == set(MODULES), set(MODULES) - set(texts)
    rows = []
    for name, module in MODULES.items():
        lines = texts[name].splitlines()
        pairs = [(lines[i].strip(), lines[i + 1].strip()) for i in range(0, len(lines) - 1, 2)]
        used = set(random.Random(0).sample(range(len(pairs)), min(40, len(pairs)))) if module in USED_SEED0 else set()
        used |= {int(i.rsplit('-', 1)[1]) for i in seen if i.rsplit('-', 1)[0] == name}
        pool = [i for i in range(len(pairs)) if i not in used]
        for i in sorted(random.Random(a.seed).sample(pool, a.per_module)):
            rows.append(dict(id=f'{name}-{i}', suite='arith', group=name,
                             question=pairs[i][0] + (SUFFIX if name in NUMERIC else SUFFIX_ANY), answer=pairs[i][1]))

    m5 = [json.loads(line) for line in open(a.math500, encoding='utf-8')]
    m5 = [r for r in m5 if r['level'] == 5]
    free = [r for r in m5 if question_id('stdmath', r['problem']) not in seen]
    print('MATH-500 level 5', len(m5), 'unused', len(free))
    assert len(free) >= a.math, (len(free), a.math)
    random.Random(a.seed).shuffle(free)
    for r in free[:a.math]:
        rows.append(dict(id=question_id('stdmath', r['problem']), suite='stdmath', group='math500-l5', question=r['problem'],
                         answer=normalize_math(str(r['answer']))))

    ids = [r['id'] for r in rows]
    assert len(set(ids)) == len(ids), 'duplicate ids'
    assert not set(ids) & seen, sorted(set(ids) & seen)
    Path(a.out).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print('rows', len(rows), dict(Counter(r['group'] for r in rows)), '-> ', a.out)


if __name__ == '__main__':
    main()
