"""fuzz_corpus.py: a differential-fuzz corpus for the rational ops (design "How 100% is made", 2: fuzzing stays on as a standing gate).
Random operands of 1 to 20 digits as ints, Fractions, 'p/q' strings and decimal strings, with signs, around the device's 9- and 18-digit
edges in particular; Python's canonical output frozen beside each, in the corpus format of harness.py.
python results/cuda-mathbackend/fuzz_corpus.py --seed 2 --n 4000
"""
import argparse
import json
import platform
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.igc.mathbackend import canon  # noqa: E402

OPS = ['numbers.add', 'numbers.sub', 'numbers.mul', 'numbers.div']


def number(rng):
    digits = rng.choice([1, 2, 5, 8, 9, 10, 12, 17, 18, 19, 20] + list(range(1, 21)))
    n = rng.randrange(10 ** (digits - 1) if digits > 1 else 0, 10 ** digits)
    return -n if rng.random() < 0.4 else n


def operand(rng):
    kind = rng.random()
    if kind < 0.3:
        return number(rng)
    if kind < 0.55:
        d = abs(number(rng)) or 1
        return Fraction(number(rng), d)
    if kind < 0.8:
        d = abs(number(rng))
        return f'{number(rng)}/{d}'   # d may be 0: a refusal, and the device's fast path must hand it on
    i, f = abs(number(rng)), abs(number(rng))
    sign = rng.choice(['', '-', '+'])
    return f'{sign}{i if rng.random() < 0.8 else ""}.{f if rng.random() < 0.9 else ""}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=2)
    ap.add_argument('--n', type=int, default=4000)
    ap.add_argument('--out', default=str(Path(__file__).with_name('corpus')))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for _ in range(a.n):
        op = rng.choice(OPS)
        args = (operand(rng), operand(rng))
        rows.append(dict(op=op, args=canon.encode(list(args)).decode(), out=canon.call(op, args).decode()))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    head = dict(corpus=1, module='numbers', kind='fuzz', seed=a.seed, n=a.n, ops=OPS, cases=len(rows), python=platform.python_version(),
                commit=commit, python_bugs=[])
    path = Path(a.out) / f'numbers-fuzz-s{a.seed}.jsonl'
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(head) + '\n')
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{path}: {len(rows)} cases')


if __name__ == '__main__':
    main()
