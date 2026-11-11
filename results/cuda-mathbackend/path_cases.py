"""path_cases.py: a corpus built to drive the device's full path through its rare branches, and a census of which path each case took.

The random corpora reach Knuth D only when a gcd is >= 2^32 and its add-back step almost never (about 2 in 2^32 per step). Here:
- unreduced 'k*a/k*b' strings with k of 10 to 400 digits (the gcd is k: multi-limb division in the reduction);
- a/b + c/b and x/y * y/x with big parts (the sum or product reduces by a big gcd);
- add-back vectors found by running the device's own Knuth D (mirrored below, 32-bit limbs, normalised) over structured operands
  (limbs of 0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF) and keeping the (u, v) pairs where qhat was one too large; each becomes the
  string f'{u}/{v * m}' with m coprime to u / v, so rat_reduce divides u by the gcd v.
Every case's expected output is Python's; `census` runs the DLL and counts the path bits it reports (1 register, 2 full, 4 Knuth D,
8 add-back, 16 timed out).
python results/cuda-mathbackend/path_cases.py build      (writes corpus/numbers-paths-s3.jsonl)
python results/cuda-mathbackend/path_cases.py census corpus/numbers-paths-s3.jsonl
"""
import ctypes
import itertools
import json
import math
import platform
import random
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.igc.mathbackend import canon  # noqa: E402

B = 1 << 32


def limbs(x):
    out = []
    while x:
        out.append(x & (B - 1))
        x >>= 32
    return out


def knuth_addback(u, v):
    """Mirror of mag_div: True when some step adds back (qhat one too large after the two-limb correction)."""
    U, V = limbs(u), limbs(v)
    n = len(V)
    if n < 2 or len(U) < n:
        return False
    s = 32 - V[-1].bit_length()
    vn = limbs(v << s)
    un = limbs(u << s) + [0]
    while len(un) < len(U) + 1:
        un.append(0)
    m = len(U) - n
    for j in range(m, -1, -1):
        num = (un[j + n] << 32) | un[j + n - 1]
        qhat, rhat = divmod(num, vn[n - 1])
        while qhat >= B or qhat * vn[n - 2] > ((rhat << 32) | un[j + n - 2]):
            qhat -= 1
            rhat += vn[n - 1]
            if rhat >= B:
                break
        part = sum(x << (32 * k) for k, x in enumerate(un[j:j + n + 1]))
        prod = qhat * sum(x << (32 * k) for k, x in enumerate(vn))
        if part < prod:
            return True
        rest = part - prod
        for k in range(n + 1):
            un[j + k] = (rest >> (32 * k)) & (B - 1)
    return False


def addback_pairs(rng, want=40):
    pats = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFE, 0x80000001]
    found = []
    for nv in (2, 3):
        for vl in itertools.product(pats, repeat=nv):
            if vl[-1] == 0:
                continue
            v = sum(x << (32 * k) for k, x in enumerate(vl))
            for _ in range(60):
                ql = [rng.choice(pats) for _ in range(rng.randint(1, 3))]
                q = sum(x << (32 * k) for k, x in enumerate(ql)) or 1
                u = q * v
                if knuth_addback(u, v):
                    found.append((u, v))
                    if len(found) >= want:
                        return found
    return found


def build(seed=3):
    rng = random.Random(seed)
    cases = []
    ops = ['numbers.add', 'numbers.sub', 'numbers.mul', 'numbers.div']
    for kd in (10, 20, 40, 80, 200, 400):
        for _ in range(6):
            k = rng.randrange(10 ** (kd - 1), 10 ** kd)
            a, b = rng.randrange(1, 10 ** 30), rng.randrange(1, 10 ** 30)
            cases.append((rng.choice(ops), (f'{k * a}/{k * b}', rng.choice([1, '-3/7', Fraction(5, 11)]))))
            big = rng.randrange(10 ** (kd - 1), 10 ** kd)
            cases.append(('numbers.add', (Fraction(1, big), Fraction(big - 1, big))))
            cases.append(('numbers.sub', (Fraction(big + 1, big), Fraction(1, big))))
            x, y = rng.randrange(10 ** (kd - 1), 10 ** kd), rng.randrange(10 ** (kd - 1), 10 ** kd)
            cases.append(('numbers.mul', (Fraction(x, y), Fraction(y, x))))
            cases.append(('numbers.div', (Fraction(x, y), Fraction(x, y))))
    pairs = addback_pairs(rng)
    for u, v in pairs:
        m = 3
        while math.gcd(u // v, m) != 1:
            m += 2
        s = f'{u}/{v * m}'
        if len(s) <= 4000:
            cases.append((rng.choice(ops), (s, 1)))
    rows = [dict(op=op, args=canon.encode(list(args)).decode(), out=canon.call(op, args).decode()) for op, args in cases]
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    head = dict(corpus=1, module='numbers', kind='paths', seed=seed, ops=ops, cases=len(rows), addback_pairs=len(pairs),
                python=platform.python_version(), commit=commit, python_bugs=[])
    path = Path(__file__).with_name('corpus') / f'numbers-paths-s{seed}.jsonl'
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(head) + '\n')
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{path}: {len(rows)} cases, {len(pairs)} add-back pairs found by the mirror')


def census(corpus):
    from tools.igc.mathbackend import cuda_backend, harness
    head, rows = harness.load(corpus)
    lib = cuda_backend._load()
    v = (ctypes.c_uint64 * 8)()
    bits, agree, by_path = Counter(), 0, Counter()
    for r in rows:
        got = cuda_backend.call(r['op'], r['args'].encode())
        lib.mip_diag(v)
        p = int(v[6])
        ok = got == r['out'].encode()
        agree += ok
        for b, name in ((1, 'register'), (2, 'full'), (4, 'Knuth D'), (8, 'add-back'), (16, 'timed out')):
            if p & b:
                bits[name] += 1
                by_path[(name, ok)] += 1
    print(f'{Path(corpus).name}: {agree}/{len(rows)} agree; cases per path: {dict(bits)}')
    print('   per path, (path, agrees): ' + ', '.join(f'{k}: {n}' for k, n in sorted(by_path.items())))
    cuda_backend.close()


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build()
    else:
        census(sys.argv[2])
