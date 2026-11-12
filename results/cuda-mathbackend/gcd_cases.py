"""gcd_cases.py: a corpus for the reduction's gcd with multi-limb parts (Lehmer's algorithm and its full Euclid step).

Cases, each a call whose reduction needs gcd(numerator, denominator) with both parts above one limb:
- consecutive Fibonacci numbers F(k+1)/F(k) (every Euclid quotient is 1: the most Lehmer steps per bit) and k*F(j+1)/k*F(j);
- u much longer than v (Lehmer finds no cofactors: the full step x mod y), 'u/v' with v of 3 to 12 limbs and u of 100 to 3000 digits;
- parts made of all-ones limbs, (2^a - 1)/(2^b - 1) (gcd 2^gcd(a,b) - 1), and 10^a - 1 over 10^b - 1;
- a common factor of many limbs, k*x / k*y with k of 20 to 1200 digits, and a sum a/b + c/d that reduces by a big gcd;
- sizes around the two-limb end of the loop (parts of 60 to 70 bits).
Expected outputs are Python's. python results/cuda-mathbackend/gcd_cases.py  (writes corpus/numbers-gcd-s4.jsonl)
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


def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def build(seed=4):
    rng = random.Random(seed)
    ops = ['numbers.add', 'numbers.sub', 'numbers.mul', 'numbers.div']
    cases = []
    one = lambda: rng.choice([1, 0, '-1/3', Fraction(7, 5)])   # noqa: E731
    for k in (70, 100, 200, 500, 1000, 3000, 9000):
        f1, f0 = fib(k + 1), fib(k)
        cases.append((rng.choice(ops), (f'{f1}/{f0}', one())))
        m = rng.randrange(10 ** 20, 10 ** 21)
        cases.append((rng.choice(ops), (f'{m * f1}/{m * f0}', one())))
        cases.append(('numbers.div', (Fraction(f1), Fraction(f0))))
    for vl in (3, 5, 8, 12):
        for ud in (100, 400, 1500, 3000):
            v = rng.randrange(1 << (32 * vl - 32), 1 << (32 * vl))
            u = rng.randrange(10 ** (ud - 1), 10 ** ud)
            g = rng.randrange(1 << 40, 1 << 70)
            cases.append((rng.choice(ops), (f'{u * g}/{v * g}', one())))
            cases.append(('numbers.mul', (Fraction(u * g), Fraction(1, v * g))))
    for a, b in ((96, 64), (320, 192), (1024, 768), (2000, 1500), (4096, 4064), (3000, 2999)):
        x, y = (1 << a) - 1, (1 << b) - 1
        cases.append(('numbers.div', (x, y)))
        cases.append(('numbers.mul', (Fraction(x, y), 1)))
    for a, b in ((40, 30), (300, 120), (1000, 750), (2400, 1600)):
        cases.append(('numbers.div', (10 ** a - 1, 10 ** b - 1)))
    for kd in (20, 60, 200, 600, 1200):
        k = rng.randrange(10 ** (kd - 1), 10 ** kd)
        x, y = rng.randrange(10 ** 30, 10 ** 60), rng.randrange(10 ** 30, 10 ** 60)
        cases.append((rng.choice(ops), (f'{k * x}/{k * y}', one())))
        cases.append(('numbers.div', (k * x, k * y)))
        d = rng.randrange(10 ** (kd - 1), 10 ** kd)
        cases.append(('numbers.add', (Fraction(1, 3 * d), Fraction(2, 3 * d))))
        cases.append(('numbers.sub', (Fraction(5, 7 * d), Fraction(-2, 7 * d))))
    for bits in range(60, 71, 2):
        x, y = rng.randrange(1 << (bits - 1), 1 << bits), rng.randrange(1 << (bits - 1), 1 << bits)
        g = rng.randrange(1 << 20, 1 << 30)
        cases.append(('numbers.div', (x * g, y * g)))
        cases.append(('numbers.add', (Fraction(x, y * g), Fraction(1, g))))
    rows = []
    for op, args in cases:
        enc = canon.encode(list(args)).decode()
        if len(enc) > 60000:
            continue
        rows.append(dict(op=op, args=enc, out=canon.call(op, args).decode()))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    head = dict(corpus=1, module='numbers', kind='gcd', seed=seed, ops=ops, cases=len(rows), python=platform.python_version(),
                commit=commit, python_bugs=[])
    path = Path(__file__).with_name('corpus') / f'numbers-gcd-s{seed}.jsonl'
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(head) + '\n')
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{path}: {len(rows)} cases')


if __name__ == '__main__':
    build()
