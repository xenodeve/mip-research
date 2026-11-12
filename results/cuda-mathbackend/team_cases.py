"""team_cases.py: a corpus for the work the device splits across a block of threads (MIP_TEAM), at the places a split can go wrong.

- carries that ripple across a thread's segment: products and sums of all-ones limbs, (2^(32k) - 1) * (2^(32j) - 1), (10^k - 1)^2,
  x - (x - 1), 2^(32k) - 1 + 1;
- sizes on both sides of every threshold: products of 1 to 400 limbs by 1 to 400 limbs, unbalanced (1 x 300, 300 x 2);
- long decimal strings (digits to limbs) and long results (limbs to decimal), a decimal string with thousands of places (10^k as the
  denominator), fractions whose parts are all nines;
- big parts with a one-limb common factor (the division by a small number).
Run the corpus with MIP_TEAM_WORK=1 as well, so every case goes through the team whatever its size.
python results/cuda-mathbackend/team_cases.py  (writes corpus/numbers-team-s5.jsonl)
"""
import json
import platform
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.igc.mathbackend import canon  # noqa: E402

B = 1 << 32


def build(seed=5):
    rng = random.Random(seed)
    ops = ['numbers.add', 'numbers.sub', 'numbers.mul', 'numbers.div']
    cases = []
    for k, j in ((1, 1), (2, 3), (16, 16), (33, 31), (64, 64), (128, 5), (200, 200), (400, 400), (1000, 3)):
        x, y = B ** k - 1, B ** j - 1
        cases.append(('numbers.mul', (x, y)))
        cases.append(('numbers.add', (x, 1)))
        cases.append(('numbers.sub', (x + 1, 1)))
        cases.append(('numbers.sub', (x, x - 1)))
        cases.append(('numbers.add', (x, y)))
    for k in (9, 18, 19, 100, 300, 1000, 2000, 3000):
        x = 10 ** k - 1
        cases.append(('numbers.mul', (x, x)))
        cases.append(('numbers.mul', (str(x), str(x))))
        cases.append(('numbers.add', (f'{x}/{x + 2}', 1)))
        cases.append(('numbers.div', (x, 10 ** (k // 2) - 1)))
    for na in (1, 2, 3, 8, 15, 16, 17, 31, 32, 33, 64, 100, 255, 256, 257, 400):
        for nb in (1, 2, 7, 64, 300):
            a = rng.randrange(B ** (na - 1), B ** na)
            b = rng.randrange(B ** (nb - 1), B ** nb)
            cases.append(('numbers.mul', (a * rng.choice([1, -1]), b)))
            if (na + nb) % 3 == 0:
                cases.append((rng.choice(ops), (Fraction(a, rng.randrange(1, B)), Fraction(b, rng.randrange(1, B)))))
    for digits in (20, 200, 1000, 2500, 3990):
        x = rng.randrange(10 ** (digits - 1), 10 ** digits)
        cases.append(('numbers.add', (str(x), '1')))
        cases.append(('numbers.mul', (str(x), f'{rng.randrange(1, 10 ** 9)}')))
        p = rng.randrange(1, digits)
        s = str(x)
        cases.append(('numbers.add', (s[:p] + '.' + s[p:], '1/3')))
        cases.append(('numbers.sub', ('0.' + s[:-1], 0)))
        cases.append(('numbers.mul', ('-' + s[:digits // 2] + '/' + s[digits // 2:], 7)))
    for digits in (100, 1000, 3000):
        g = rng.randrange(2, B)
        x, y = rng.randrange(10 ** (digits - 1), 10 ** digits), rng.randrange(1, B)
        cases.append(('numbers.div', (x * g, y * g)))
        cases.append(('numbers.add', (Fraction(x, 6 * g), Fraction(5, 6 * g))))
    rows = []
    for op, args in cases:
        enc = canon.encode(list(args)).decode()
        rows.append(dict(op=op, args=enc, out=canon.call(op, args).decode()))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    head = dict(corpus=1, module='numbers', kind='team', seed=seed, ops=ops, cases=len(rows), python=platform.python_version(),
                commit=commit, python_bugs=[])
    path = Path(__file__).with_name('corpus') / f'numbers-team-s{seed}.jsonl'
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(head) + '\n')
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{path}: {len(rows)} cases')


if __name__ == '__main__':
    build()
