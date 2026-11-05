"""Edge cases for every operation of tools/igc/mathbackend/counting.py, with oracles that share no code with the backend (#298).

sim_counting.py draws random cases; this file adds the corners a random draw misses: empty and zero inputs, one element, the largest input the operation allows,
the first input past its bound (the backend must refuse), and malformed input (BadInput). It is called by sim_counting.g_edges, after all the random
generators, so the random cases keep the seeds they always had. The list does not depend on the seed or on n.

Oracles, named in each case: `brute` (plain loops or recursion written here, no itertools shortcut that the backend also uses where it matters), `closed form`
(a formula derived differently from the backend's), `sympy`, `certificate`, `expected refusal`, `expected BadInput`. Where the only feasible oracle at the bound is the
backend's own formula, the case says so in its name.
"""
import itertools
import math
import random
from fractions import Fraction

import sympy
from sympy import Rational

from tools.igc.mathbackend import sim_counting as S
from tools.igc.mathbackend.sim_counting import BadIn, Refuse, mk

F = Fraction
_CASES = []


def add(op, args, val, name, **kw):
    _CASES.append(mk('counting.' + op, args, val, name, **kw))


def refuse(op, args, why='just past the bound', **kw):
    _CASES.append(mk('counting.' + op, args, Refuse(why), f'expected refusal ({why})', **kw))


def bad(op, args, why='malformed input', **kw):
    _CASES.append(mk('counting.' + op, args, BadIn(why), f'expected BadInput ({why})', **kw))


# ---------------------------------------------------------------- oracle helpers (plain loops, no backend code)

def fact(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def binom(n, k):
    """C(n, k), 0 outside 0 <= k <= n (math.comb, only used for oracle formulas that differ from the backend's)."""
    return math.comb(n, k) if 0 <= k <= n else 0


def der_alt(n):
    """Derangements n! * sum (-1)^k / k! as an exact integer, summing n!/k! from k = n down."""
    t, acc = 1, 0
    for k in range(n, -1, -1):
        acc += (-1) ** k * t
        t *= k if k else 1
    return acc


def double_fact_odd(m):
    """(2m-1)!! = 1*3*5*...*(2m-1)"""
    r = 1
    for i in range(1, 2 * m, 2):
        r *= i
    return r


def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def lucas(n):
    a, b = 2, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def stars_brute(n, k, lows, ups):
    """Solutions of x_1 + ... + x_k = n with lows[i] <= x_i <= ups[i] (None = no upper bound), by recursion."""
    def go(i, left):
        if i == k:
            return int(left == 0)
        hi = left - sum(lows[i + 1:])
        if ups[i] is not None:
            hi = min(hi, ups[i])
        return sum(go(i + 1, left - x) for x in range(lows[i], hi + 1))
    return go(0, n)


def bell_triangle(n):
    row = [1]
    for _ in range(n):
        new = [row[-1]]
        for x in row:
            new.append(new[-1] + x)
        row = new
    return row[0]


def perms_of(m):
    """all permutations of range(m) as tuples, built by recursion"""
    if m == 0:
        return [()]
    out = []
    for p in perms_of(m - 1):
        for pos in range(m):
            out.append(p[:pos] + (m - 1,) + p[pos:])
    return out


def tw_brute(m, k, balls, boxes, rule):
    """Twelvefold way by listing every function from m balls to k boxes and counting the distinct classes."""
    seen = set()
    for f in itertools.product(range(k), repeat=m):
        occ = tuple(f.count(b) for b in range(k))
        if rule == 'injective' and any(o > 1 for o in occ):
            continue
        if rule == 'surjective' and any(o < 1 for o in occ):
            continue
        if balls == 'labelled' and boxes == 'labelled':
            key = f
        elif balls == 'identical' and boxes == 'labelled':
            key = occ
        elif balls == 'labelled':
            key = frozenset(frozenset(j for j in range(m) if f[j] == b) for b in range(k) if b in f)
        else:
            key = tuple(sorted(occ))
        seen.add(key)
    return len(seen)


def pc_brute(parts_list, n, kw):
    """Partitions of n (non-increasing tuples) that satisfy the keyword options, by definition."""
    out = 0
    for p in parts_list:
        if 'parts' in kw and any(x not in kw['parts'] for x in p):
            continue
        if kw.get('distinct') and len(set(p)) != len(p):
            continue
        if 'num_parts' in kw and len(p) != kw['num_parts']:
            continue
        if 'max_parts' in kw and len(p) > kw['max_parts']:
            continue
        if 'max_part' in kw and any(x > kw['max_part'] for x in p):
            continue
        if 'min_part' in kw and any(x < kw['min_part'] for x in p):
            continue
        out += 1
    return out


def comps_brute(n, k, allowed):
    """Compositions of n (ordered), into k parts (k None: any number), every part satisfying allowed(part)."""
    def go(left, cnt):
        if left == 0 and (k is None or cnt == k):
            tail = 1
        else:
            tail = 0
        if k is not None and cnt == k:
            return tail
        return tail + sum(go(left - p, cnt + 1) for p in range(1, left + 1) if allowed(p))
    return go(n, 0)


# ---------------------------------------------------------------- 1 stars_and_bars

def e_stars_and_bars():
    P = 'stars_and_bars'
    table = [(0, 0, None, 0), (1, 0, None, 0), (-1, 0, None, 0), (-1, 3, None, 0), (0, 3, None, 0), (0, 1, None, 0), (5, 1, None, 0), (5, 1, 4, 0), (5, 1, 5, 0),
             (0, 3, 0, 0), (3, 3, 1, 0), (3, 2, 1, 0), (6, 3, None, [1, 2, 3]), (5, 3, None, [1, 2, 3]), (7, 3, None, [1, 2, 3]), (4, 2, 2, 3), (4, 2, [None, 1], 0),
             (0, 2, None, -1), (2, 3, None, -1), (5, 3, [2, None, 0], [0, 1, 0]), (10, 4, [3, 3, 3, 3], 0), (12, 4, 3, 0), (13, 4, 3, 0), (0, 0, 3, 0), (2, 0, None, []),
             (6, 3, 2, 2), (5, 3, 2, 2), (8, 4, [None, None, None, None], 1), (0, 4, 0, 0), (1, 4, 0, 0), (9, 1, 9, 9), (9, 1, 8, 0), (-3, 2, None, -2), (-5, 2, None, -2)]
    for n, k, up, low in table:
        lows = [low] * k if isinstance(low, int) else list(low)
        ups = [None] * k if up is None else ([up] * k if isinstance(up, int) else list(up))
        add(P, (n, k), stars_brute(n, k, lows, ups), 'brute recursion', upper=up, lower=low)
    add(P, (10 ** 6, 3), (10 ** 6 + 1) * (10 ** 6 + 2) // 2, 'closed form (n+1)(n+2)/2, the largest n allowed')
    add(P, (0, 10 ** 6), 1, 'closed form (all zeros), the largest k allowed')
    add(P, (1, 10 ** 6), 10 ** 6, 'closed form (which variable is 1), the largest k allowed')
    add(P, (2, 10 ** 6), 10 ** 6 + (10 ** 6 * (10 ** 6 - 1)) // 2, 'closed form (one variable 2, or two variables 1)')
    add(P, (10 ** 6, 1), 1, 'closed form')
    add(P, (99, 100), 100, 'closed form (0/1 variables: choose the one that is 0)', upper=1)
    add(P, (10 ** 6, 1), 1, 'closed form (one variable with an upper bound, large)', upper=10 ** 6)
    add(P, (10 ** 6, 1), 0, 'closed form (upper bound below the target)', upper=10 ** 6 - 1)
    add(P, (3, 4), 20, 'brute recursion (lower bound as a list of length k)', lower=[0, 0, 0, 0])
    refuse(P, (10 ** 6 + 1, 2))
    refuse(P, (0, 10 ** 6 + 1))
    refuse(P, (1000, 10001), 'work bound (n+1)*k > 1e7', upper=1)
    refuse(P, (10 ** 7, 1), 'work bound (n+1)*k > 1e7', upper=10 ** 7)
    bad(P, (True, 3))
    bad(P, (3, -1))
    bad(P, (3, 2), 'upper list of the wrong length', upper=[1, 2, 3])
    bad(P, (3, 2), 'float bound', upper=1.5)


# ---------------------------------------------------------------- 2 derangements, 3 rencontres, 4 surjections

def e_derangements():
    P = 'derangements'
    for n in range(0, 9):
        add(P, (n,), sum(1 for p in perms_of(n) if all(p[i] != i for i in range(n))), 'brute over all permutations')
    for n in (9, 10, 100, 1000, 2500, 5000):
        add(P, (n,), der_alt(n), 'closed form n! sum (-1)^k/k! as an integer alternating sum' + (', the largest n allowed' if n == 5000 else ''))
    refuse(P, (5001,))
    bad(P, (-1,))
    bad(P, (True,))
    bad(P, (3.0,))
    bad(P, ('4',))


def e_rencontres():
    P = 'rencontres'
    for n in range(0, 8):
        ps = perms_of(n)
        fixed = [sum(1 for i in range(n) if p[i] == i) for p in ps]
        for k in range(0, 9):
            add(P, (n, k), fixed.count(k), 'brute over all permutations')
    for n, k in ((5000, 5000), (5000, 4999), (5000, 4998), (5000, 4997), (5000, 5001), (5000, 0), (5000, 1), (60, 3), (60, 60), (60, 59)):
        add(P, (n, k), binom(n, k) * der_alt(n - k) if k <= n else 0, 'closed form C(n,k) * D(n-k), D by the alternating sum' + (', n = 5000 is the largest allowed' if n == 5000 else ''))
    refuse(P, (5001, 0))
    bad(P, (-1, 0))
    bad(P, (3, -1))


def e_surjections():
    P = 'surjections'
    for n in range(0, 7):
        for k in range(0, 7):
            add(P, (n, k), sum(1 for f in itertools.product(range(k), repeat=n) if len(set(f)) == k), 'brute over all functions')
    for n in (7, 20, 100):
        add(P, (n, 1), 1, 'closed form')
        add(P, (n, 2), 2 ** n - 2, 'closed form 2^n - 2')
        add(P, (n, 3), 3 ** n - 3 * 2 ** n + 3, 'closed form 3^n - 3 2^n + 3')
        add(P, (n, n), fact(n), 'closed form n!')
        add(P, (n, n + 1), 0, 'k > n')
        add(P, (n, 0), 0, 'closed form (no function onto the empty set)')
    add(P, (5000, 1), 1, 'closed form, the largest n allowed')
    add(P, (5000, 2), 2 ** 5000 - 2, 'closed form, the largest n allowed')
    add(P, (5000, 3), 3 ** 5000 - 3 * 2 ** 5000 + 3, 'closed form, the largest n allowed')
    add(P, (2000, 2000), fact(2000), 'closed form n!, the largest k allowed')
    add(P, (1999, 2000), 0, 'k > n, k = 2000')
    add(P, (2000, 1999), binom(2000, 2) * fact(1999), 'closed form (one box gets two balls)')
    add(P, (5000, 0), 0, 'closed form (n > 0 balls onto no box)')
    refuse(P, (5001, 3))
    refuse(P, (10, 2001))
    bad(P, (-1, 2))
    bad(P, (2, -1))


# ---------------------------------------------------------------- 5 set_partitions

def e_set_partitions():
    P = 'set_partitions'
    lists = {m: S.set_partitions_list(m) for m in range(0, 8)}
    for m in range(0, 8):
        add(P, (m,), len(lists[m]), 'brute listing of all set partitions')
        for k in range(0, 9):
            add(P, (m, k), sum(1 for p in lists[m] if len(p) == k), 'brute listing of all set partitions')
        for sizes in ([1], [2], [1, 2], [3], [2, 3], [4], [1, 2, 3], []):
            for k in (None, 0, 1, 2, 3, 4):
                add(P, (m, k), sum(1 for p in lists[m] if all(len(b) in sizes for b in p) and (k is None or len(p) == k)), 'brute listing of all set partitions', block_sizes=sizes)
    for n, v in ((1500, bell_triangle(1500)), (1000, bell_triangle(1000))):
        add(P, (n,), v, 'closed form: Bell number from the Bell triangle (plain additions)' + (', the largest n allowed' if n == 1500 else ''))
    add(P, (1500, 1), 1, 'closed form, the largest n allowed')
    add(P, (1500, 2), 2 ** 1499 - 1, 'closed form 2^(n-1) - 1')
    add(P, (1500, 3), (3 ** 1500 - 3 * 2 ** 1500 + 3) // 6, 'closed form (3^n - 3 2^n + 3)/6')
    add(P, (1500, 1500), 1, 'closed form')
    add(P, (1500, 1499), binom(1500, 2), 'closed form C(n,2)')
    add(P, (1500, 1501), 0, 'k > n')
    add(P, (1500, 0), 0, 'closed form')
    add(P, (300,), bell_triangle(300), 'closed form: Bell number from the Bell triangle')
    add(P, (300,), sum(binom(300, 2 * j) * double_fact_odd(j) for j in range(151)), 'closed form: involution number sum C(n,2j) (2j-1)!!', block_sizes=[1, 2])
    add(P, (5, 10 ** 6), 0, 'k > n (returns before any work)', block_sizes=[1])
    add(P, (0, None), 1, 'the empty set has one partition (no blocks)', block_sizes=[2])
    refuse(P, (1501,))
    refuse(P, (1500,), 'work bound (n+1)*(k+1)*sizes > 1e7', block_sizes=[1, 2, 3, 4, 5])
    bad(P, (4,), 'a block size of 0', block_sizes=[0, 1])
    bad(P, (4,), 'a negative block size', block_sizes=[-1])
    bad(P, (-1,))
    bad(P, (3, -1))


# ---------------------------------------------------------------- 6 twelvefold_way

def e_twelvefold():
    P = 'twelvefold_way'
    for balls in ('labelled', 'identical'):
        for boxes in ('labelled', 'identical'):
            for rule in ('any', 'injective', 'surjective'):
                for n in range(0, 6):
                    for k in range(0, 6):
                        add(P, (n, k, balls, boxes, rule), tw_brute(n, k, balls, boxes, rule), 'brute enumeration of all functions, classes by canonical form')
    add(P, (3000, 3000, 'labelled', 'labelled', 'any'), 3000 ** 3000, 'closed form k^n, the largest n and k allowed')
    add(P, (3000, 3000, 'labelled', 'labelled', 'injective'), fact(3000), 'closed form n!')
    add(P, (1500, 1500, 'labelled', 'labelled', 'surjective'), fact(1500), 'closed form n!')
    add(P, (3000, 3000, 'identical', 'labelled', 'any'), binom(5999, 3000), 'closed form C(n+k-1, n)')
    add(P, (3000, 3000, 'identical', 'labelled', 'surjective'), 1, 'closed form C(n-1, k-1)')
    add(P, (3000, 3000, 'identical', 'labelled', 'injective'), 1, 'closed form C(k, n)')
    add(P, (1500, 3, 'labelled', 'identical', 'any'), 1 + (2 ** 1499 - 1) + (3 ** 1500 - 3 * 2 ** 1500 + 3) // 6, 'closed form S(n,1)+S(n,2)+S(n,3), the largest n allowed')
    add(P, (1500, 1500, 'labelled', 'identical', 'surjective'), 1, 'closed form S(n,n)')
    add(P, (3000, 3000, 'identical', 'identical', 'injective'), 1, 'closed form')
    add(P, (3000, 2, 'identical', 'identical', 'any'), 3000 // 2 + 1, 'closed form (parts of size <= 2)')
    add(P, (3000, 3000, 'identical', 'identical', 'surjective'), 1, 'closed form (n parts of size 1)')
    add(P, (1000, 1000, 'identical', 'identical', 'any'), int(S.npartitions(1000)), 'sympy partition number')
    add(P, (3000, 3000, 'identical', 'identical', 'any'), int(S.npartitions(3000)), 'sympy partition number, the largest n allowed')
    add(P, (5, 3000, 'labelled', 'labelled', 'surjective'), 0, 'closed form (more boxes than balls), the largest k allowed')
    add(P, (1500, 2500, 'labelled', 'labelled', 'surjective'), 0, 'closed form (more boxes than balls)')
    add(P, (1500, 2500, 'labelled', 'identical', 'surjective'), 0, 'closed form (more boxes than balls)')
    refuse(P, (3001, 2))
    refuse(P, (2, 3001))
    refuse(P, (1501, 2, 'labelled', 'identical', 'any'))
    refuse(P, (1501, 2, 'labelled', 'identical', 'surjective'))
    bad(P, (3, 3, 'labelled', 'identical', 'sometimes'))
    bad(P, (3, 3, 'distinct'))
    bad(P, (-1, 3))
    bad(P, (3, True))


# ---------------------------------------------------------------- 7 permutations_with_cycle_type

def cycle_hist(m):
    h = {}
    for p in perms_of(m):
        seen, cyc = [False] * m, []
        for s in range(m):
            if not seen[s]:
                L, j = 0, s
                while not seen[j]:
                    seen[j] = True
                    j = p[j]
                    L += 1
                cyc.append(L)
        key = tuple(sorted(cyc))
        h[key] = h.get(key, 0) + 1
    return h


def e_cycle_type():
    P = 'permutations_with_cycle_type'
    add(P, ([],), 1, 'the empty permutation')
    for m in range(1, 7):
        for key, cnt in cycle_hist(m).items():
            add(P, (list(key),), cnt, 'brute over all permutations')
            add(P, (list(reversed(key)),), cnt, 'brute over all permutations (order of the list does not matter)')
    add(P, ([2, 1, 2],), 15, 'brute over all permutations (unsorted list)')
    add(P, ([2000],), fact(1999), 'closed form (n-1)!, the largest n allowed')
    add(P, ([1] * 2000,), 1, 'closed form (the identity), the largest n allowed')
    add(P, ([2] * 1000,), double_fact_odd(1000), 'closed form (n-1)!! for fixed-point-free involutions, the largest n allowed')
    add(P, ([1000, 1000],), fact(2000) // (1000 * 1000 * 2), 'closed form n!/(l^m m!)')
    add(P, ([1999, 1],), fact(2000) // 1999, 'closed form n!/(l^m m!)')
    refuse(P, ([2001],))
    refuse(P, ([1] * 2001,))
    bad(P, ([0],))
    bad(P, ([-1, 2],))
    bad(P, ([1.5],))
    bad(P, ('abc',))
    bad(P, ([True],))


# ---------------------------------------------------------------- 8 eulerian, 9 mahonian

def eul_formula(n, k):
    return sum((-1) ** j * binom(n + 1, j) * (k + 1 - j) ** n for j in range(k + 2))


def e_eulerian():
    P = 'eulerian'
    for n in range(0, 9):
        hist = {}
        for p in perms_of(n):
            d = sum(1 for i in range(n - 1) if p[i] > p[i + 1])
            hist[d] = hist.get(d, 0) + 1
        for k in range(0, 10):
            add(P, (n, k), hist.get(k, 0), 'brute over all permutations')
    for k in (0, 1, 2, 3, 100, 499, 500, 997, 998, 999, 1000, 1001):
        add(P, (1000, k), eul_formula(1000, k) if k < 1000 else 0, 'closed form sum (-1)^j C(n+1,j) (k+1-j)^n, the largest n allowed')
    add(P, (999, 998), 1, 'closed form (the single permutation with n-1 descents)')
    refuse(P, (1001, 0))
    bad(P, (-1, 0))
    bad(P, (3, -1))


def maho_pent(n, k):
    """Permutations of n with k inversions for k <= n: coefficient of q^k in prod_{i>=1}(1-q^i) * (1-q)^-n, with Euler's pentagonal numbers."""
    total, j = 0, 0
    while True:
        done = True
        for jj in ((0,) if j == 0 else (j, -j)):
            g = jj * (3 * jj - 1) // 2
            if g <= k:
                done = False
                total += (-1) ** (abs(jj)) * binom(n - 1 + k - g, n - 1)
        if done and j > 0:
            break
        j += 1
    return total


def maho_sympy(n):
    q = sympy.symbols('q')
    poly = sympy.Poly(1, q)
    for i in range(1, n + 1):
        poly = poly * sympy.Poly(sum(q ** j for j in range(i)), q)
    return {m[0]: int(c) for m, c in poly.terms()}


def e_mahonian():
    P = 'mahonian'
    for n in range(0, 8):
        hist = {}
        for p in perms_of(n):
            d = sum(1 for a in range(n) for b in range(a + 1, n) if p[a] > p[b])
            hist[d] = hist.get(d, 0) + 1
        for k in range(0, n * (n - 1) // 2 + 3):
            add(P, (n, k), hist.get(k, 0), 'brute over all permutations')
    co = maho_sympy(40)
    top = 40 * 39 // 2
    for k in (0, 1, 2, 3, 20, 100, 389, 390, 391, 700, top - 1, top, top + 1, top + 1000):
        add(P, (40, k), co.get(k, 0), 'sympy product of (1+q+..+q^(i-1))')
    for n, k in ((1000, 0), (1000, 1), (1000, 2), (1000, 10), (1000, 500), (1000, 999), (1000, 1000), (2000, 100), (2000, 1000)):
        add(P, (n, k), maho_pent(n, k), 'closed form: Euler pentagonal numbers times (1-q)^-n' + (', the largest n allowed' if n == 2000 else ''))
    add(P, (50, 1), 49, 'closed form n-1')
    add(P, (50, 2), maho_sympy(50)[2], 'sympy product of (1+q+..+q^(i-1))')
    refuse(P, (2000, 5000), 'work bound n*(k+1) > 1e7')
    refuse(P, (2001, 0))
    refuse(P, (2000, 1999000), 'work bound n*(k+1) > 1e7')
    bad(P, (-1, 0))
    bad(P, (3, -1))


# ---------------------------------------------------------------- 10 anagram_count, 11 count_subsets_with_sum

def e_anagram():
    P = 'anagram_count'
    words = ['', 'a', 'aa', 'ab', 'aab', 'abab', 'aabb', 'aabbc', 'abcabc', 'aaabbbc', 'abcdefg', 'aaaaaab', 'abbbbbb', 'aabbccdd', 'aaaabbbb', 'a1a1', 'éeé', 'ISSISSIPPI']
    for w in words:
        perms = set(itertools.permutations(w)) if len(w) <= 8 else None
        if perms is None:
            continue
        add(P, (w,), len(perms), 'brute set of permutations')
        add(P, (w, True), sum(1 for p in perms if all(p[j] != p[j + 1] for j in range(len(p) - 1))), 'brute set of permutations', )
    add(P, ('a' * 400,), 1, 'closed form, the longest word allowed')
    add(P, ('a' * 400, True), 0, 'closed form (equal letters are always adjacent), the longest word allowed')
    add(P, ('ab' * 200,), binom(400, 200), 'closed form C(400,200), the longest word allowed')
    add(P, ('ab' * 200, True), 2, 'closed form (only abab.. and baba..), the longest word allowed')
    add(P, ('a' * 399 + 'b', True), 0, 'closed form')
    add(P, ('abcdefghijkl',), fact(12), 'closed form n!')
    add(P, ('abcdefghijkl', True), fact(12), 'closed form n! (all letters distinct: no two equal)')
    add(P, ('aabbccddeeff',), fact(12) // 2 ** 6, 'closed form n!/prod m_i!')
    add(P, ('aaaaabbbbbccccc',), fact(15) // fact(5) ** 3, 'closed form n!/prod m_i!')
    add(P, ('a' * 30 + 'b' * 30 + 'c' * 30, ), fact(90) // fact(30) ** 3, 'closed form n!/prod m_i!')
    add(P, ('ab', True), 2, 'brute')
    add(P, ('abc' * 3, True), sum(1 for p in set(itertools.permutations('abcabcabc')) if all(p[j] != p[j + 1] for j in range(8))), 'brute set of permutations')
    refuse(P, ('a' * 401,))
    refuse(P, ('abcdefghijklmnopqrst' * 2 + 'a',), 'work bound states*letters^2 > 1e7', no_adjacent_equal=True)
    refuse(P, ('abcdefghijklmnopqrstuvwxyz',), 'work bound states*letters^2 > 1e7', no_adjacent_equal=True)
    bad(P, (123,))
    bad(P, (None,))


def e_subset_sum():
    P = 'count_subsets_with_sum'

    def brute(vals, target=None, mod=None, rem=0, size=None):
        c = 0
        for m in range(1 << len(vals)):
            sub = [vals[j] for j in range(len(vals)) if m >> j & 1]
            if size is not None and len(sub) != size:
                continue
            s = sum(sub)
            if (s == target) if mod is None else (s % mod == rem % mod):
                c += 1
        return c
    lists = [[], [0], [5], [-5], [0, 0, 0], [1, 1, 1, 1], [1, -1], [2, 4, 6], [-2, 3, 5, -7, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [3, 3, 3], [0, 1, 0, 1]]
    for vals in lists:
        for t in (-7, -1, 0, 1, 2, 3, 5, 6, 10, 55):
            for size in (None, 0, 1, 2, len(vals), len(vals) + 1):
                add(P, (vals,), brute(vals, target=t, size=size), 'brute subsets', target=t, size=size)
        for md in (1, 2, 3, 5, 7):
            for rem in (0, 1, -1, 7, 12):
                add(P, (vals,), brute(vals, mod=md, rem=rem), 'brute subsets', mod=md, rem=rem)
                add(P, (vals,), brute(vals, mod=md, rem=rem, size=2), 'brute subsets', mod=md, rem=rem, size=2)
    add(P, ([0] * 3000,), 2 ** 3000, 'closed form 2^n, the most values allowed', target=0)
    add(P, ([0] * 3000,), 0, 'closed form', target=1)
    add(P, ([0] * 3000,), 2 ** 3000, 'closed form 2^n, the most values allowed', mod=1)
    add(P, ([7] * 3000,), 1, 'closed form (only the empty subset sums to 0)', target=0, size=0)
    add(P, ([1] * 60,), binom(60, 30), 'closed form C(n, k)', target=30)
    add(P, ([1] * 60,), binom(60, 30), 'closed form C(n, k)', target=30, size=30)
    add(P, ([1] * 60,), 0, 'closed form (size and sum disagree)', target=30, size=29)
    refuse(P, ([0] * 3001,), target=0)
    bad(P, ([1, 2],), 'neither target nor mod')
    bad(P, ([1, 2],), 'both target and mod', target=1, mod=2)
    bad(P, ([1, 2],), 'mod 0', mod=0)
    bad(P, ([1, 2],), 'negative size', target=1, size=-1)
    bad(P, ([1, True],), 'a bool value', target=1)
    bad(P, ([1, 2.5],), 'a float value', target=1)
    bad(P, (5,), 'values not a list', target=1)


# ---------------------------------------------------------------- 12 partition_count, 13 compositions_count

def e_partition_count():
    P = 'partition_count'
    allp = {m: S.partitions_brute(m) for m in range(0, 14)}
    rng = random.Random(1201)
    for m in range(0, 14):
        add(P, (m,), len(allp[m]), 'brute enumeration of all partitions')
    opts_parts = [None, [], [1], [2, 3], [1, 4, 5], [3, 3, 7], [0 + 9]]
    opts_dist = [False, True]
    opts_cnt = [None, ('num_parts', 0), ('num_parts', 1), ('num_parts', 2), ('num_parts', 4), ('max_parts', 0), ('max_parts', 1), ('max_parts', 3), ('max_parts', 20)]
    opts_max = [None, 0, 1, 2, 5, 40]
    opts_min = [None, 0, 1, 2, 3, 50]
    combos = list(itertools.product(range(0, 14), opts_parts, opts_dist, opts_cnt, opts_max, opts_min))
    for m, parts, dist, cnt, mx, mn in rng.sample(combos, 700):
        kw = {}
        if parts is not None:
            kw['parts'] = parts
        if dist:
            kw['distinct'] = True
        if cnt is not None:
            kw[cnt[0]] = cnt[1]
        if mx is not None:
            kw['max_part'] = mx
        if mn is not None:
            kw['min_part'] = mn
        add(P, (m,), pc_brute(allp[m], m, kw), 'brute enumeration of all partitions, options by definition', **kw)
    # every single option alone, on every n 0..13 (the random sample above may miss a corner)
    for m in range(0, 14):
        for kw in ({'num_parts': 0}, {'num_parts': m}, {'num_parts': m + 1}, {'max_parts': 0}, {'max_parts': m}, {'max_part': 0}, {'max_part': m}, {'max_part': m + 1},
                   {'min_part': 0}, {'min_part': m}, {'min_part': m + 1}, {'distinct': True}, {'parts': []}, {'parts': [m or 1]}, {'parts': [m + 1]}, {'parts': [1], 'distinct': True},
                   {'distinct': True, 'num_parts': 2}, {'distinct': True, 'max_parts': 2}, {'min_part': 2, 'max_part': 4}, {'min_part': 4, 'max_part': 2}):
            add(P, (m,), pc_brute(allp[m], m, kw), 'brute enumeration of all partitions, options by definition', **kw)
    add(P, (3000,), int(S.npartitions(3000)), 'sympy partition number, the largest n allowed')
    add(P, (2000,), int(S.npartitions(2000)), 'sympy partition number')
    odd_dp = [0] * 3001
    odd_dp[0] = 1
    for p in range(1, 3001, 2):
        for s in range(p, 3001):
            odd_dp[s] += odd_dp[s - p]
    add(P, (3000,), odd_dp[3000], 'closed form: distinct parts = odd parts (Euler), plain DP over the odd parts', distinct=True)
    add(P, (3000,), 3000 // 2 + 1, 'closed form (parts <= 2)', max_part=2)
    add(P, (3000,), 1, 'closed form (n parts of size 1)', num_parts=3000)
    add(P, (3000,), 0, 'closed form (more parts than n)', num_parts=3001)
    add(P, (3000,), 1, 'closed form (one part)', num_parts=1)
    add(P, (3000,), 1500, 'closed form (two parts: floor(n/2))', num_parts=2)
    add(P, (3000,), 1, 'closed form (all ones)', max_part=1)
    add(P, (3000,), 0, 'closed form', parts=[])
    add(P, (3000,), 1, 'closed form (n = a multiple of the only part)', parts=[3000])
    add(P, (0,), 1, 'closed form (the empty partition)', parts=[])
    add(P, (0,), 1, 'closed form (the empty partition)', num_parts=0)
    add(P, (0,), 0, 'closed form (no partition of 0 into one part)', num_parts=1)
    add(P, (1,), 0, 'closed form', num_parts=0)
    refuse(P, (3001,))
    refuse(P, (3000,), 'work bound sizes*n*parts > 1e7', num_parts=2, parts=list(range(1, 3001)))
    refuse(P, (3000,), 'work bound sizes*n*parts > 1e7', max_parts=5, distinct=True)
    bad(P, (5,), 'num_parts and max_parts together', num_parts=2, max_parts=3)
    bad(P, (5,), 'a part of size 0', parts=[0, 1])
    bad(P, (5,), 'a negative part', parts=[-1])
    bad(P, (-1,))
    bad(P, (5,), 'negative max_part', max_part=-1)


def e_compositions():
    P = 'compositions_count'
    for n in range(0, 11):
        add(P, (n,), comps_brute(n, None, lambda p: True), 'brute recursion')
        for k in range(0, 7):
            add(P, (n, k), comps_brute(n, k, lambda p: True), 'brute recursion')
        for parts in ([], [1], [2], [1, 2], [2, 3], [3, 3], [1, 5, 9], [11, 12]):
            add(P, (n,), comps_brute(n, None, lambda p, parts=parts: p in parts), 'brute recursion', parts=parts)
            for k in (0, 1, 2, 3):
                add(P, (n, k), comps_brute(n, k, lambda p, parts=parts: p in parts), 'brute recursion', parts=parts)
        for lo, hi in ((1, 3), (2, 2), (3, 2), (2, None), (4, 9)):
            add(P, (n,), comps_brute(n, None, lambda p, lo=lo, hi=hi: p >= lo and (hi is None or p <= hi)), 'brute recursion', min_part=lo, max_part=hi)
            for k in (1, 2, 3):
                add(P, (n, k), comps_brute(n, k, lambda p, lo=lo, hi=hi: p >= lo and (hi is None or p <= hi)), 'brute recursion', min_part=lo, max_part=hi)
    for n in range(0, 7):          # parts of size 0 are allowed when the number of parts is fixed: weak compositions
        for k in range(0, 4):
            add(P, (n, k), stars_brute(n, k, [0] * k, [None] * k), 'brute recursion (weak compositions)', min_part=0)
    add(P, (200000, 1), 1, 'closed form, the largest n allowed')
    add(P, (200000, 2), 199999, 'closed form n-1, the largest n allowed')
    add(P, (200000, 2), 200001, 'closed form n+1 (weak compositions)', min_part=0)
    add(P, (200000,), 1, 'closed form (only parts of size 1)', parts=[1])
    add(P, (50000,), fib(50001), 'closed form Fibonacci F(n+1)', parts=[1, 2])
    add(P, (200000,), 0, 'closed form (parts bigger than n)', parts=[200001, 300000])
    add(P, (1000, 100), binom(999, 99), 'closed form C(n-1, k-1)')
    add(P, (1000, 100), binom(1099, 99), 'closed form C(n+k-1, k-1) (weak)', min_part=0)
    add(P, (30,), 2 ** 29, 'closed form 2^(n-1)')
    add(P, (0,), 1, 'closed form (the empty composition)', parts=[])
    refuse(P, (200001,))
    refuse(P, (200000,), 'work bound n*sizes > 1e7')
    refuse(P, (100000, 100000), 'k bound', parts=[1])
    refuse(P, (3000, 90000), 'work bound k*n*sizes > 1e7', parts=[1, 2])
    bad(P, (5,), 'part size 0 without k', min_part=0)
    bad(P, (5,), 'a part of size 0', parts=[0])
    bad(P, (-1,))
    bad(P, (5, -1))


# ---------------------------------------------------------------- 14 standard_young_tableaux, 15 semistandard_young_tableaux

def frobenius_syt(shape):
    """Standard Young tableaux by the Frobenius determinant formula n! det[1/(l_i - i + j)!] (a determinant, not hook lengths)."""
    l = len(shape)
    n = sum(shape)
    M = sympy.Matrix(l, l, lambda i, j: (Rational(1, math.factorial(shape[i] - i + j)) if shape[i] - i + j >= 0 else 0))
    return int(M.det() * math.factorial(n))


def e_syt():
    P = 'standard_young_tableaux'
    for sh in S.shapes_upto(9):
        add(P, (list(sh),), S.syt_rec(tuple(sh)), 'brute recurrence (remove the largest entry from a corner)')
    for sh in ([5, 4, 3, 2, 1], [8, 6, 4, 2, 1], [10, 8, 5, 3, 1], [7, 7, 7, 7], [12, 3, 3, 1, 1, 1], [20, 1], [6, 5, 5, 4, 3, 2, 2, 1]):
        add(P, (sh,), frobenius_syt(sh), 'closed form: Frobenius determinant')
    add(P, ([2000],), 1, 'closed form (one row), the most cells allowed')
    add(P, ([1] * 2000,), 1, 'closed form (one column), the most cells allowed')
    add(P, ([1000, 1000],), math.comb(2000, 1000) // 1001, 'closed form Catalan(1000), the most cells allowed')
    add(P, ([1999, 1],), 1999, 'closed form n-1 (the shape (n-1, 1))')
    add(P, ([1000] + [1] * 1000,), math.comb(1999, 999), 'closed form C(a+b-1, a-1) for the hook (a, 1^b), the most cells allowed')
    add(P, ([1500, 500],), math.comb(2000, 500) - math.comb(2000, 499), 'closed form (two rows): C(n, b) - C(n, b-1)')
    refuse(P, ([2001],))
    refuse(P, ([1001, 1000],))
    refuse(P, ([1] * 2001,))
    bad(P, ([],))
    bad(P, ([0],))
    bad(P, ([2, 3],), 'rows must not grow')
    bad(P, ([2, 0],))
    bad(P, ([True],))
    bad(P, ('abc',))


def e_ssyt():
    P = 'semistandard_young_tableaux'
    for sh in S.shapes_upto(6):
        for k in range(0, 7):
            add(P, (list(sh), k), S.ssyt_rec(tuple(sh), k), 'brute recurrence (remove a horizontal strip)')
    for sh in ([4, 3, 2, 1], [5, 5, 2], [3, 3, 3], [6, 1, 1], [2, 2, 2, 2, 2]):
        for k in (1, 2, 4, 5, 8, 9):
            add(P, (sh, k), S.weyl_dim(sh, k) if k >= len(sh) else 0, 'closed form Weyl dimension formula (0 when k < number of rows)')
    add(P, ([2000], 3), math.comb(2002, 2), 'closed form C(n+k-1, k-1), the most cells allowed')
    add(P, ([2000], 1), 1, 'closed form')
    add(P, ([1] * 2000, 2000), 1, 'closed form C(k, m) for a column, the most cells allowed')
    add(P, ([1] * 2000, 2001), 2001, 'closed form C(k, m) for a column')
    add(P, ([1] * 2000, 1999), 0, 'closed form C(k, m) for a column (too few values)')
    add(P, ([1], 10 ** 6), 10 ** 6, 'closed form, the largest k allowed')
    add(P, ([2], 10 ** 6), math.comb(10 ** 6 + 1, 2), 'closed form C(n+k-1, k-1), the largest k allowed')
    add(P, ([1, 1], 10 ** 6), math.comb(10 ** 6, 2), 'closed form C(k, 2) for a column, the largest k allowed')
    add(P, ([2, 1], 1), 0, 'closed form (two rows need two values)')
    add(P, ([3], 0), 0, 'closed form (no values)')
    refuse(P, ([1], 10 ** 6 + 1))
    refuse(P, ([2001], 2))
    bad(P, ([], 3))
    bad(P, ([2, 3], 3))
    bad(P, ([2], -1))
    bad(P, ([2], 1.5))


# ---------------------------------------------------------------- 16 necklace_count, 17 orbit_count_colorings

def necklace_burnside(n, k):
    """necklaces = (1/n) sum over the n rotations r of k^gcd(r, n), a plain loop (the backend groups by divisors and phi)."""
    if n == 0:
        return 1
    return sum(k ** math.gcd(r, n) for r in range(n)) // n


def bracelet_closed(n, k):
    nk = necklace_burnside(n, k)
    if n == 0:
        return 1
    if n % 2:
        refl = k ** ((n + 1) // 2)
    else:
        refl = (k ** (n // 2) + k ** (n // 2 + 1)) // 2
    return (nk + refl) // 2


def necklace_counts_burnside(counts):
    n = sum(counts)
    if n == 0:
        return 1
    tot = 0
    for r in range(n):
        g = math.gcd(r, n)
        L = n // g
        if all(c % L == 0 for c in counts):
            m, t = g, 1
            for c in counts:
                t *= math.comb(m, c // L)
                m -= c // L
            tot += t
    return tot // n


def e_necklace():
    P = 'necklace_count'
    for n in range(0, 9):
        for k in range(0, 4):
            for br in (False, True):
                v = (1 if n == 0 else (S.necklace_brute(n, k, br) if k > 0 else 0))
                add(P, (n, k), v, 'brute canonical form (1 when n = 0, 0 with no colours)', bracelet=br)
    for counts in ([0], [1], [3], [0, 3, 0], [1, 1], [2, 1], [1, 1, 1], [2, 2], [3, 3], [2, 2, 2], [4, 2], [1, 0, 2], [0, 0], [6], [3, 2, 1], [2, 2, 2, 2], [5, 1, 1], [4, 4]):
        for br in (False, True):
            n = sum(counts)
            if n == 0:
                add(P, (), 1, 'closed form (no beads: one necklace)', counts=counts, bracelet=br)
            else:
                add(P, (), S.necklace_brute(n, len(counts), br, counts), 'brute canonical form', counts=counts, bracelet=br)
    for n, k in ((5000, 2), (5000, 3), (4999, 2), (4096, 2), (5000, 10 ** 6), (2310, 7)):
        add(P, (n, k), necklace_burnside(n, k), 'Burnside over all n rotations with a plain gcd loop' + (', the largest n allowed' if n >= 4999 else ''))
        add(P, (n, k), bracelet_closed(n, k), 'closed form: necklaces plus the reflection term k^((n+1)/2) or (k^(n/2)+k^(n/2+1))/2' + (', the largest n allowed' if n >= 4999 else ''), bracelet=True)
    add(P, (), necklace_counts_burnside([2500, 2500]), 'Burnside over all n rotations with a plain gcd loop (multinomial of the fixed content)', counts=[2500, 2500])
    add(P, (), necklace_counts_burnside([1000, 1000, 1000]), 'Burnside over all n rotations with a plain gcd loop (multinomial of the fixed content)', counts=[1000, 1000, 1000])
    add(P, (), necklace_counts_burnside([5000]), 'Burnside over all n rotations with a plain gcd loop (one colour)', counts=[5000])
    add(P, (), necklace_counts_burnside([1] * 8), 'Burnside over all n rotations with a plain gcd loop (8 colours, the most allowed)', counts=[1] * 8)
    add(P, (), 5040, 'closed form (n-1)! (8 distinct beads on a necklace)', counts=[1] * 8)
    add(P, (), 2520, 'closed form (n-1)!/2 for 8 distinct beads on a bracelet', counts=[1] * 8, bracelet=True)
    add(P, (0, 0), 1, 'the empty necklace')
    add(P, (0, 5), 1, 'the empty necklace')
    add(P, (3, 0), 0, 'no colours')
    add(P, (1, 7), 7, 'closed form k')
    add(P, (1, 7), 7, 'closed form k', bracelet=True)
    add(P, (2, 5), 15, 'closed form k(k+1)/2')
    add(P, (2, 5), 15, 'closed form k(k+1)/2', bracelet=True)
    refuse(P, (5001, 2))
    refuse(P, (3, 10 ** 6 + 1))
    refuse(P, (), 'more than 8 colours with fixed counts', counts=[1] * 9)
    refuse(P, (), 'work bound states*cycles*colours > 1e7', counts=[2500, 2500], bracelet=True)
    bad(P, (), 'no arguments')
    bad(P, (3,), 'k missing')
    bad(P, (), 'empty counts', counts=[])
    bad(P, (), 'negative count', counts=[2, -1])
    bad(P, (-1, 2))


def cyc(m):
    return [(j + 1) % m for j in range(m)]


def dih(m):
    return [[(j + 1) % m for j in range(m)], [(-j) % m for j in range(m)]]


def sym_gens(m):
    return [[(j + 1) % m for j in range(m)], [1, 0] + list(range(2, m))]


def e_orbit():
    P = 'orbit_count_colorings'
    add(P, (0, [[]], 3), 1, 'closed form (no points: one empty colouring)')
    add(P, (0, [[]], None, [0]), 1, 'closed form (no points)')
    add(P, (1, [[0]], 5), 5, 'closed form')
    add(P, (3, [[0, 1, 2]], 0), 0, 'closed form (no colours)')
    add(P, (3, [[0, 1, 2]], 2), 8, 'closed form k^n (the trivial group)')
    add(P, (4, [[0, 1, 2, 3], [0, 1, 2, 3]], 3), 81, 'closed form k^n (duplicate identity generators)')
    for m in (2, 3, 4, 5, 6):
        for k in (1, 2, 3):
            cols = list(itertools.product(range(k), repeat=m))
            add(P, (m, [cyc(m)], k), S.orbits_by_union_find(m, [cyc(m)], cols), 'brute orbits by union-find (cyclic group)')
            add(P, (m, dih(m), k), S.orbits_by_union_find(m, dih(m), cols), 'brute orbits by union-find (dihedral group)')
            add(P, (m, sym_gens(m), k), math.comb(m + k - 1, m), 'closed form C(n+k-1, n) (the symmetric group: a multiset of colours)')
    for m in (3, 4, 5, 6):
        add(P, (m, sym_gens(m), None, [m - 1, 1]), 1, 'closed form (symmetric group, fixed content: one orbit)')
        add(P, (m, sym_gens(m), None, [m]), 1, 'closed form (symmetric group, fixed content: one orbit)')
        add(P, (m, [cyc(m)], None, [m - 1, 1]), 1, 'closed form (a single odd bead on a cycle: one necklace)')
    add(P, (8, [cyc(8)], None, [4, 4]), S.necklace_brute(8, 2, False, [4, 4]), 'brute canonical form (necklace with fixed content)')
    add(P, (8, S.cube_vertex_gens(), 2), 23, 'known value (cube vertices, rotations, 2 colours)')
    add(P, (6, S.cube_face_gens(), 2), 10, 'known value (cube faces, rotations, 2 colours)')
    add(P, (6, S.cube_face_gens(), 3), 57, 'known value (cube faces, rotations, 3 colours)')
    add(P, (8, S.cube_vertex_gens(), 3), S.orbits_by_union_find(8, S.cube_vertex_gens(), list(itertools.product(range(3), repeat=8))), 'brute orbits by union-find (cube vertices, 3 colours)')
    add(P, (8, sym_gens(8), 3), math.comb(10, 8), 'closed form C(n+k-1, n) (S_8 has 40320 elements)')
    add(P, (64, [cyc(64)], 2), necklace_burnside(64, 2), 'Burnside with a plain gcd loop (the most points allowed)')
    add(P, (64, dih(64), 2), bracelet_closed(64, 2), 'closed form bracelet count (the most points allowed)')
    add(P, (64, [list(range(64))], 3), 3 ** 64, 'closed form k^n (the trivial group, the most points allowed)')
    refuse(P, (65, [list(range(65))], 2))
    refuse(P, (9, sym_gens(9), 2), 'group has more than 200000 elements (9! = 362880)')
    bad(P, (3, [[0, 0, 1]], 2), 'not a permutation')
    bad(P, (3, [[0, 1]], 2), 'wrong length')
    bad(P, (3, [], 2), 'no generators')
    bad(P, (3, [[0, 1, 2]], 2, [1, 2]), 'both k and counts')
    bad(P, (3, [[0, 1, 2]]), 'neither k nor counts')
    bad(P, (3, [[0, 1, 2]], None, [1, 1]), 'counts do not add up to n_points')
    bad(P, (3, [[0, 1, 2]], -1))
    refuse(P, (9, [list(range(9))], None, [1] * 9), 'more than 8 colours with fixed counts')


# ---------------------------------------------------------------- 18 lattice_paths

def lp_brute(start, end, steps, forb, region, constraint):
    x0, y0, x1, y1 = region
    memo = {}

    def ok(x, y):
        if not (x0 <= x <= x1 and y0 <= y <= y1) or (x, y) in forb:
            return False
        return {None: True, 'y<=x': y <= x, 'y<x': y < x, 'y>=x': y >= x, 'y>x': y > x}[constraint]

    def go(p):
        if p in memo:
            return memo[p]
        if not ok(*p):
            r = 0
        elif p == end:
            r = 1
        else:
            r = sum(go((p[0] + dx, p[1] + dy)) for dx, dy in steps)
        memo[p] = r
        return r
    return go(start)


def e_lattice():
    P = 'lattice_paths'
    D = ((1, 0), (0, 1))

    def case(end, start=(0, 0), steps=D, forb=(), region=None, constraint=None, name='brute memoised depth-first enumeration'):
        if region is None:
            eff = (min(start[0], end[0]), min(start[1], end[1]), max(start[0], end[0]), max(start[1], end[1]))
        else:
            eff = region
        add(P, (end,), lp_brute(start, end, steps, set(forb), eff, constraint), name, start=start, steps=list(steps), forbidden=list(forb), region=region, constraint=constraint)
    case((0, 0))
    case((0, 0), forb=[(0, 0)])
    case((3, 3), forb=[(3, 3)])
    case((3, 3), forb=[(0, 0)])
    case((3, 3), forb=[(1, 0), (0, 1)])
    case((3, 3), forb=[(1, 1), (2, 2), (3, 1)])
    case((-1, 0))
    case((0, -1))
    case((-2, -2), start=(-5, -5))
    case((2, 2), start=(-2, -2))
    case((4, 4), region=(0, 0, 3, 4))
    case((4, 4), region=(1, 1, 4, 4))
    case((4, 4), region=(0, 0, 4, 3))
    case((4, 4), region=(5, 5, 6, 6))
    for c in (None, 'y<=x', 'y<x', 'y>=x', 'y>x'):
        for end in ((0, 0), (1, 1), (4, 4), (4, 2), (2, 4), (5, 0), (0, 5)):
            case(end, constraint=c)
        case((4, 4), start=(0, 1), constraint=c)
        case((4, 4), start=(1, 0), constraint=c)
        case((3, 3), forb=[(2, 2)], constraint=c)
    case((3, 3), steps=((1, 1),))
    case((3, 2), steps=((1, 1),))
    case((4, 6), steps=((2, 0), (0, 3)))
    case((5, 0), steps=((1, 0),))
    case((0, 5), steps=((0, 1),))
    case((1, 5), steps=((0, 1),))
    case((3, 3), steps=((1, 0), (0, 1), (1, 1)))
    case((6, 0), steps=((1, 1), (1, 0), (1, -1)), region=(0, 0, 6, 10), name='brute memoised depth-first enumeration (Motzkin paths: 51)')
    case((6, 0), steps=((1, 1), (1, -1)), region=(0, 0, 6, 3))
    case((6, 0), steps=((1, 1), (1, -1)), region=(0, 0, 6, 2))
    case((0, 3), steps=((1, 1), (-1, 1), (1, 0)), region=(-3, 0, 3, 3), name='brute memoised depth-first enumeration (steps ordered by y)')
    case((2, 2), steps=((1, 1), (-1, 1), (1, 0), (0, 1)), region=(-1, 0, 3, 4), name='brute memoised depth-first enumeration (steps ordered by y)')
    case((3, 3), forb=[(1, 1), (1, 1)])
    add(P, ((6, 0),), 51, 'known value (Motzkin number M6)', steps=[(1, 1), (1, 0), (1, -1)], region=(0, 0, 6, 10))
    add(P, ((500, 500),), math.comb(1000, 500), 'closed form C(a+b, a)')
    add(P, ((0, 800),), 1, 'closed form (a straight line)')
    add(P, ((1000, 1000),), 0, 'closed form (the target is outside the region)', region=(0, 0, 999, 999))
    add(P, ((400, 400),), math.comb(800, 400) // 401, 'closed form Catalan(400) (paths never above the diagonal)', constraint='y<=x')
    add(P, ((400, 400),), math.comb(800, 400) // 401, 'closed form Catalan(400) (paths never below the diagonal)', constraint='y>=x')
    add(P, ((400, 399),), math.comb(799, 399) - math.comb(799, 400), 'closed form (ballot number: strictly below the diagonal needs y < x)', constraint='y<x')
    refuse(P, ((1, 1),), 'region has more than 4e6 cells', region=(0, 0, 2000, 2000))
    refuse(P, ((1, 0),), 'region has more than 4e6 cells', region=(0, 0, 4000001, 0))
    refuse(P, ((1, 1),), 'steps not ordered: a cycle is possible', steps=[(1, 0), (-1, 0)])
    refuse(P, ((1, 1),), 'a negative step needs a region', steps=[(1, 0), (0, 1), (1, -1)])
    bad(P, ((1, 1),), 'a zero step', steps=[(0, 0)])
    bad(P, ((1, 1),), 'no steps', steps=[])
    bad(P, ((1, 1),), 'unknown constraint', constraint='x<y')
    bad(P, ((1, 1),), 'region of the wrong length', region=(0, 0, 3))
    bad(P, ((1,),), 'end is not a pair')
    bad(P, ((1, 1),), 'start is not a pair', start=(0, 0, 0))
    add(P, ((3, 3),), 0, 'an empty region (x1 < x0) has no path', region=(5, 0, 2, 9))


# ---------------------------------------------------------------- 19 catalan .. 24 narayana

def catalan_rec(n):
    """C_n by the ratio recurrence C_{k+1} = C_k * 2(2k+1) / (k+2), plain integers."""
    c = 1
    for k in range(n):
        c = c * 2 * (2 * k + 1) // (k + 2)
    return c


CAT = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786, 208012, 742900, 2674440]


def e_catalan():
    P = 'catalan'
    for n, v in enumerate(CAT):
        add(P, (n,), v, 'known values (OEIS A000108)')
    for n in (1000, 20000, 99999, 100000):
        add(P, (n,), catalan_rec(n), 'closed form: ratio recurrence C(k+1) = C(k) 2(2k+1)/(k+2)' + (', the largest n allowed' if n == 100000 else ''))
    refuse(P, (100001,))
    bad(P, (-1,))
    bad(P, (2.0,))


def dyck_height_hist(n):
    hist = {}
    for seq in itertools.product((1, -1), repeat=2 * n):
        s, top, ok = 0, 0, True
        for x in seq:
            s += x
            if s < 0:
                ok = False
                break
            top = max(top, s)
        if ok and s == 0:
            hist[top] = hist.get(top, 0) + 1
    return hist


def e_dyck():
    P = 'dyck_paths'
    for n in range(0, 10):
        hist = dyck_height_hist(n)
        add(P, (n,), sum(hist.values()), 'brute sequences')
        for h in range(0, 12):
            add(P, (n, h), sum(c for t, c in hist.items() if t <= h), 'brute sequences')
    add(P, (100000, 1), 1, 'closed form (only UDUD..)')
    add(P, (100000, 0), 0, 'closed form (no step allowed)')
    add(P, (100000, 2), 2 ** 99999, 'closed form 2^(n-1) for height <= 2, the largest n allowed')
    add(P, (5000, 3), fib(9999), 'closed form F(2n-1) for height <= 3')
    add(P, (1000, 1000), catalan_rec(1000), 'closed form: ratio recurrence (max_height >= n is no limit)')
    add(P, (1000, 5000), catalan_rec(1000), 'closed form: ratio recurrence (max_height above n)')
    add(P, (100000,), catalan_rec(100000), 'closed form: ratio recurrence, the largest n allowed')
    add(P, (0, 0), 1, 'closed form (the empty path)')
    add(P, (0,), 1, 'closed form (the empty path)')
    refuse(P, (100001,))
    refuse(P, (100000, 50), 'work bound n*max_height > 1e7')
    refuse(P, (3000, 3000), 'work bound n*max_height > 1e7')
    bad(P, (-1,))
    bad(P, (3, -1))


def e_ballot():
    P = 'ballot'
    for a in range(0, 8):
        for b in range(0, 8):
            for strict in (True, False):
                if a + b == 0 and strict:
                    continue
                t = 0
                for seq in itertools.product((1, -1), repeat=a + b):
                    if seq.count(1) != a:
                        continue
                    s, ok = 0, True
                    for x in seq:
                        s += x
                        if s < 0 or (strict and s == 0):
                            ok = False
                            break
                    t += ok
                add(P, (a, b, strict), t, 'brute sequences')
    bad(P, (0, 0, True), 'no votes: strictly ahead is undefined')
    bad(P, (0, 0))
    add(P, (0, 0, False), 1, 'closed form (the empty count)')
    add(P, (200000, 0), 1, 'closed form (A always leads), the most votes allowed')
    add(P, (200000, 0, False), 1, 'closed form, the most votes allowed')
    add(P, (199999, 1), 199998, 'closed form a-1 (the one B vote must come after two A votes)')
    add(P, (199999, 1, False), 199999, 'closed form a (the one B vote must come after one A vote)')
    add(P, (100000, 100000, False), catalan_rec(100000), 'closed form: Catalan by the ratio recurrence, the most votes allowed')
    add(P, (100000, 100000), 0, 'closed form (a tie is never strictly ahead)')
    add(P, (3, 4), 0, 'closed form (b > a)')
    add(P, (3, 4, False), 0, 'closed form (b > a)')
    add(P, (10, 5), (10 - 5) * math.comb(15, 5) // 15, 'closed form (a-b)/(a+b) C(a+b, a)')
    add(P, (10, 9, False), math.comb(19, 9) - math.comb(19, 8), 'closed form C(a+b, b) - C(a+b, b-1)')
    refuse(P, (200001, 0))
    refuse(P, (0, 200001))
    bad(P, (-1, 0))
    bad(P, (3, -1))
    bad(P, (True, 1))


def e_motzkin():
    P = 'motzkin'
    for n in range(0, 12):
        t = 0
        for seq in itertools.product((1, 0, -1), repeat=n):
            s, ok = 0, True
            for x in seq:
                s += x
                if s < 0:
                    ok = False
                    break
            t += ok and s == 0
        add(P, (n,), t, 'brute step sequences')
    for n in (100, 1000, 3000):
        v = sum(math.comb(n, 2 * k) * (math.comb(2 * k, k) // (k + 1)) for k in range(n // 2 + 1))
        add(P, (n,), v, 'closed form sum C(n,2k) Catalan(k)' + (', the largest n allowed' if n == 3000 else ''))
    refuse(P, (3001,))
    bad(P, (-1,))


def e_schroeder():
    P = 'schroeder'
    big = [1, 2, 6, 22, 90, 394, 1806, 8558, 41586, 206098]
    small = [1, 1, 3, 11, 45, 197, 903, 4279, 20793, 103049]
    for n in range(10):
        add(P, (n,), big[n], 'known values (OEIS A006318)')
        add(P, (n, True), small[n], 'known values (OEIS A001003)')
        if n <= 7:
            add(P, (n,), S.schroeder_brute(n), 'brute step sequences')
    for n in (500, 3000):
        v = int(sum(Fraction(math.comb(n, k) * math.comb(n + k, k), k + 1) for k in range(n + 1)))
        add(P, (n,), v, 'closed form sum C(n,k) C(n+k,k)/(k+1)' + (', the largest n allowed' if n == 3000 else ''))
        add(P, (n, True), v // 2, 'closed form: the little number is half of the large one' + (', the largest n allowed' if n == 3000 else ''))
    refuse(P, (3001,))
    bad(P, (-1,))


def e_narayana():
    P = 'narayana'
    for m in range(0, 9):
        hist = {}
        for seq in itertools.product((1, -1), repeat=2 * m):
            s, ok = 0, True
            for x in seq:
                s += x
                if s < 0:
                    ok = False
                    break
            if ok and s == 0:
                peaks = sum(1 for j in range(2 * m - 1) if seq[j] == 1 and seq[j + 1] == -1)
                hist[peaks] = hist.get(peaks, 0) + 1
        for k in range(0, 11):
            add(P, (m, k), hist.get(k, 0), 'brute Dyck words counting peaks')
    n = 100000
    for k in (0, 1, 2, 3, 50000, 50001, 99999, 100000, 100001):
        v = 0 if k < 1 or k > n else math.comb(n - 1, k - 1) * math.comb(n, k - 1) // k
        add(P, (n, k), v, 'closed form (1/k) C(n-1,k-1) C(n,k-1), the largest n allowed')
    refuse(P, (100001, 3))
    bad(P, (-1, 0))
    bad(P, (3, -1))


# ---------------------------------------------------------------- 25 strings_with_patterns, 26 transfer_matrix_walks

def sp_brute(alpha, length, avoid, contain):
    t = 0
    for tup in itertools.product(alpha, repeat=length):
        s = ''.join(tup)
        if all(p not in s for p in avoid) and all(c in s for c in contain):
            t += 1
    return t


def e_strings():
    P = 'strings_with_patterns'
    rng = random.Random(2501)
    for L in range(0, 5):
        add(P, ('ab', L), 2 ** L, 'closed form')
        add(P, (1, L), 1, 'closed form (one letter)')
    add(P, (1, 0), 1, 'the empty string', avoid=['0'])
    add(P, (1, 3), 0, 'closed form (the only letter is avoided)', avoid=['0'])
    add(P, ('a', 3), 1, 'closed form (one letter)', contain=['aa'])
    add(P, ('a', 1), 0, 'closed form (too short)', contain=['aa'])
    add(P, ('ab', 0), 0, 'closed form (the empty string contains nothing)', contain=['a'])
    add(P, ('ab', 4), 0, 'closed form (a required pattern outside the alphabet)', contain=['c'])
    add(P, ('ab', 4), 16, 'closed form (an avoided pattern outside the alphabet is no constraint)', avoid=['c'])
    add(P, ('ab', 4), 16 - 1, 'brute', avoid=['aaaa'])
    add(P, ('ab', 5), sp_brute('ab', 5, ['aa', 'aa'], []), 'brute (a repeated pattern)', avoid=['aa', 'aa'])
    add(P, ('ab', 5), sp_brute('ab', 5, ['a', 'aa'], []), 'brute (one avoided pattern inside another)', avoid=['a', 'aa'])
    add(P, ('ab', 5), sp_brute('ab', 5, [], ['ab', 'ab']), 'brute (a repeated required pattern)', contain=['ab', 'ab'])
    add(P, ('ab', 5), sp_brute('ab', 5, [], ['ab', 'b']), 'brute (a required pattern inside another)', contain=['ab', 'b'])
    add(P, ('abc', 6), sp_brute('abc', 6, ['ab', 'ba'], ['c', 'cc']), 'brute', avoid=['ab', 'ba'], contain=['c', 'cc'])
    pool = ['a', 'b', 'c', 'aa', 'ab', 'ba', 'bb', 'ac', 'ca', 'cc', 'aba', 'bab', 'aab', 'abc', 'cab', 'bca', 'aaa', 'abab', 'x', 'ax', '0']
    for _ in range(400):
        alpha = rng.choice(['ab', 'abc', 'ba', 'a', 'abcd'])
        L = rng.randint(0, 7 if len(alpha) < 4 else 5)
        avoid = [rng.choice(pool) for _ in range(rng.choice([0, 0, 1, 2, 3]))]
        contain = [rng.choice(pool) for _ in range(rng.choice([0, 0, 1, 2, 3]))]
        add(P, (alpha, L), sp_brute(alpha, L, avoid, contain), 'brute enumeration of all strings', avoid=avoid, contain=contain)
    ten = ['aab', 'abb', 'aba', 'bab', 'bba', 'abab', 'bbb', 'aaab', 'baa', 'bbaa']
    add(P, ('ab', 9), sp_brute('ab', 9, [], ten), 'brute (ten required patterns, the most allowed)', contain=ten)
    add(P, ('ab', 5), 0, 'brute (ten required patterns, string too short)', contain=ten)
    add(P, ('ab', 20000), fib(20002), 'closed form F(n+2) (binary strings with no 11)', avoid=['bb'])
    add(P, (1, 10 ** 6), 1, 'closed form (one letter), the longest length allowed')
    add(P, (1, 10 ** 6), 1, 'closed form (one letter), the longest length allowed', contain=['0'])
    add(P, (36, 1), 36, 'closed form (36 letters, the largest alphabet)')
    add(P, (36, 2), 36 * 36 - 36, 'closed form (36 letters, avoid the doubled letters)', avoid=[ch * 2 for ch in '0123456789abcdefghijklmnopqrstuvwxyz'])
    refuse(P, ('ab', 10 ** 6 + 1))
    refuse(P, ('ab', 3), 'more than 10 required patterns', contain=[str(i) for i in range(11)])
    refuse(P, ('ab', 3), 'more than 10 required patterns', contain=['a'] * 11)
    refuse(P, (36, 10 ** 6), 'work bound states*2^contain*alphabet*length > 1e7', avoid=['01'])
    bad(P, ('ab', 3), 'an empty pattern', avoid=[''])
    bad(P, ('ab', 3), 'an empty required pattern', contain=[''])
    bad(P, ('ab', 3), 'a pattern that is not a string', avoid=[5])
    bad(P, (0, 3), 'alphabet size 0')
    bad(P, (37, 3), 'alphabet size 37')
    bad(P, ('aa', 3), 'repeated letters in the alphabet')
    bad(P, ('', 3), 'empty alphabet')
    bad(P, ('ab', -1))


def tm_brute(A, L, st, en):
    m = len(A)
    tot = 0
    for a in (range(m) if st is None else [st]):
        def go(pos, left):
            if left == 0:
                return 1 if (en is None or pos == en) else 0
            return sum(A[pos][j] * go(j, left - 1) for j in range(m))
        tot += go(a, L)
    return tot


def e_transfer():
    P = 'transfer_matrix_walks'
    rng = random.Random(2601)
    for A in ([[0]], [[1]], [[2]], [[1, 1], [1, 0]], [[0, 1], [1, 0]], [[1, 1, 0], [0, 1, 1], [1, 0, 1]], [[0, 0], [0, 0]], [[-1, 2], [3, -4]], [[1, 0], [0, 1]]):
        m = len(A)
        for L in range(0, 6):
            for st in [None] + list(range(m)):
                for en in [None] + list(range(m)):
                    add(P, (A, L, st, en), tm_brute(A, L, st, en), 'brute depth-first walk enumeration')
    for _ in range(40):
        m = rng.randint(1, 4)
        A = [[rng.randint(-2, 3) for _ in range(m)] for _ in range(m)]
        L = rng.randint(0, 6)
        st, en = rng.choice([None, rng.randrange(m)]), rng.choice([None, rng.randrange(m)])
        add(P, (A, L, st, en), tm_brute(A, L, st, en), 'brute depth-first walk enumeration')
    Fh = [[F(1, 2), F(1, 3)], [F(1, 4), 1]]
    add(P, (Fh, 3, 0, 1), tm_brute(Fh, 3, 0, 1), 'brute depth-first walk enumeration (Fraction entries)')
    add(P, ([['1/2']], 3), F(1, 8), 'closed form (1/2)^3 (entry given as text)')
    add(P, ([[1, 1], [1, 0]], 30, 0, 0), fib(31), 'closed form Fibonacci F(L+1)')
    add(P, ([[1, 1], [1, 0]], 3000, 0, 0), fib(3001), 'closed form Fibonacci F(L+1), the direct branch')
    add(P, ([[1, 0], [0, 1]], 10 ** 6, 0, 0), 1, 'closed form (identity matrix), the longest walk allowed, the direct branch')
    ident3 = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
    add(P, (ident3, 10 ** 6), 3, 'closed form (identity matrix), the longest walk allowed, squaring branch')
    shift3 = [[1 if j == (i + 1) % 3 else 0 for j in range(3)] for i in range(3)]
    add(P, (shift3, 10 ** 6, 0, 10 ** 6 % 3), 1, 'closed form (cyclic shift: one walk), the longest walk allowed, squaring branch')
    add(P, (shift3, 10 ** 6, 0, (10 ** 6 + 1) % 3), 0, 'closed form (cyclic shift), squaring branch')
    cyc80 = [[1 if j == (i + 1) % 80 else 0 for j in range(80)] for i in range(80)]
    add(P, (cyc80, 700, 5, (5 + 700) % 80), 1, 'closed form (cyclic shift on 80 states, the largest matrix), squaring branch')
    add(P, (cyc80, 700, 5), 1, 'closed form (cyclic shift on 80 states), squaring branch')
    diag = [[(1 + (i % 2)) if i == j else 0 for j in range(80)] for i in range(80)]
    add(P, (diag, 700), 40 + 40 * 2 ** 700, 'closed form (diagonal matrix: sum of d_i^L), squaring branch')
    add(P, (diag, 700, 1, 1), 2 ** 700, 'closed form (diagonal matrix), squaring branch')
    add(P, (diag, 0), 80, 'closed form (length 0: every start counts once)')
    refuse(P, ([[0] * 81 for _ in range(81)], 1))
    refuse(P, ([[1]], 10 ** 6 + 1))
    bad(P, ([], 3), 'empty matrix')
    bad(P, ([[1, 2]], 3), 'not square')
    bad(P, ([[1, 2], [3]], 3), 'ragged')
    bad(P, ([[1]], 3, 1), 'start out of range')
    bad(P, ([[1]], 3, None, -1), 'end out of range')
    bad(P, ([[1]], 3, True), 'a bool start')
    bad(P, ([[1]], -1), 'negative length')
    bad(P, ([[1.5]], 2), 'a float entry')


# ---------------------------------------------------------------- 27 grid_colorings, 28 domino_tilings

def e_grid():
    P = 'grid_colorings'
    add(P, (0, 0, 3), 1, 'closed form (empty grid)')
    add(P, (0, 5, 3), 1, 'closed form (empty grid)')
    add(P, (5, 0, 3), 1, 'closed form (empty grid)')
    add(P, (1, 1, 0), 0, 'closed form (0 colours)')
    add(P, (1, 1, 1), 1, 'closed form (1 cell, 1 colour)')
    add(P, (1, 1, 5), 5, 'closed form (1 cell, k colours)')
    add(P, (1, 4, 3), 3 * 2 ** 3, 'closed form path k(k-1)^(c-1)')
    add(P, (2, 2, 2), 2, 'closed form cycle C_4 with 2 colours')
    add(P, (2, 2, 3), (3 - 1) ** 4 + (-1) ** 4 * (3 - 1), 'closed form cycle C_4 with 3 colours')
    refuse(P, (30, 30, 4))
    bad(P, (-1, 2, 2))
    bad(P, (2, -1, 2))
    bad(P, (2, 2, -1))
    bad(P, (2, 2, 2), 'invalid allowed table', allowed='bad')


def e_domino():
    P = 'domino_tilings'
    add(P, (0, 0), 1, 'closed form (empty grid)')
    add(P, (1, 1), 0, 'closed form (odd area)')
    add(P, (3, 3), 0, 'closed form (odd area)')
    add(P, (2, 1), 1, 'closed form')
    add(P, (2, 2), 2, 'closed form F_3')
    add(P, (2, 3), 3, 'closed form F_4')
    add(P, (2, 4), 5, 'closed form F_5')
    add(P, (2, 5), 8, 'closed form F_6')
    add(P, (2, 2), 0, 'brute (diagonal cells blocked, impossible)', blocked=[(0, 0), (1, 1)])
    add(P, (2, 2), 1, 'brute (top row blocked, 1 domino remaining)', blocked=[(0, 0), (0, 1)])
    add(P, (8, 8), 12988816, 'known constant (Fischer 1961, 8x8 chessboard)')
    refuse(P, (30, 30))
    bad(P, (-1, 2))
    bad(P, (2, -1))
    bad(P, (2, 2), 'blocked cell out of bounds', blocked=[(0, 5)])


# ---------------------------------------------------------------- 29 count_digit_constraints

def e_digits():
    P = 'count_digit_constraints'
    add(P, (1, 0), 0, 'closed form (empty range)')
    add(P, (5, 5), 1, 'closed form (single integer)')
    add(P, (1, 9), 9, 'closed form digits 1..9')
    add(P, (1, 10), 10, 'closed form distinct digits', distinct=True)
    add(P, (1, 99), 90, 'closed form no adjacent equal', no_adjacent_equal=True)
    add(P, (10, 99), 45, 'closed form nondecreasing digits', nondecreasing=True)
    add(P, (10, 99), 36, 'closed form strictly increasing digits', increasing=True)
    add(P, (1, 100), 20, 'closed form divisible by 5', mod=5, rem=0)
    add(P, (1, 15, 2), 7, 'closed form binary integers divisible by 2', mod=2, rem=0)
    refuse(P, (1, 10 ** 70))
    refuse(P, (1, 10, 17), 'base > 16')
    bad(P, (1, 10, 1), 'base < 2')
    bad(P, (-1, 10), 'lo < 0')


# ---------------------------------------------------------------- 30 spanning_trees, 31 labelled_trees, 32 rooted_forests

def e_spanning():
    P = 'spanning_trees'
    add(P, (0, []), 0, 'closed form (0 vertices)')
    add(P, (1, []), 1, 'closed form (1 vertex, empty tree)')
    add(P, (2, [(0, 1)]), 1, 'closed form K_2')
    add(P, (3, [(0, 1), (1, 2), (0, 2)]), 3, 'closed form K_3 (Cayley 3^1)')
    add(P, (4, [(0, 1), (1, 2), (2, 3), (3, 0)]), 4, 'closed form C_4 (cycle of 4)')
    add(P, (5, [(i, (i + 1) % 5) for i in range(5)]), 5, 'closed form C_5 (cycle of 5)')
    add(P, (3, [(0, 1)]), 0, 'closed form (disconnected graph)')
    add(P, (4, [(u, v) for u in range(4) for v in range(u + 1, 4)]), 16, 'closed form K_4 (Cayley 4^2)')
    refuse(P, (81, []))
    bad(P, (-1, []))
    bad(P, (3, [(0, 3)]), 'edge out of range')
    bad(P, (3, [(0, 0)]), 'self loop')


def e_labelled():
    P = 'labelled_trees'
    add(P, (0,), 0, 'closed form (0 vertices)')
    add(P, (1,), 1, 'closed form (1 vertex)')
    add(P, (2,), 1, 'closed form (2 vertices)')
    add(P, (3,), 3, 'closed form 3^(3-2)')
    add(P, (4,), 16, 'closed form 4^(4-2)')
    add(P, (5,), 125, 'closed form 5^(5-2)')
    add(P, (4,), 2, 'closed form degree sequence [1, 1, 2, 2]', degrees=[1, 1, 2, 2])
    add(P, (4,), 0, 'closed form impossible degree sequence', degrees=[1, 1, 1, 1])
    add(P, (100000,), 100000 ** 99998, 'closed form Cayley bound')
    refuse(P, (100001,))
    bad(P, (-1,))
    bad(P, (4,), 'degrees length mismatch', degrees=[1, 2])


def e_forests():
    P = 'rooted_forests'
    add(P, (3, 0), 0, 'closed form k=0')
    add(P, (3, 4), 0, 'closed form k > n')
    add(P, (3, 1), 3, 'closed form k=1: 1 * 3^(3-1-1) = 3')
    add(P, (3, 3), 1, 'closed form k=n: 1')
    add(P, (4, 2), 8, 'closed form 2 * 4^(4-2-1) = 8')
    add(P, (100000, 1), 100000 ** 99998, 'closed form n=100000 bound')
    refuse(P, (100001, 1))
    bad(P, (-1, 2))
    bad(P, (4, -1))


# ---------------------------------------------------------------- 33 perfect_matchings, 34 matchings_count, 35 permutations_allowed

def e_perfect_matchings():
    P = 'perfect_matchings'
    add(P, (0,), 1, 'closed form n=0 complete graph')
    add(P, (1,), 0, 'closed form odd n')
    add(P, (3,), 0, 'closed form odd n')
    add(P, (2,), 1, 'closed form K_2: 1!! = 1')
    add(P, (4,), 3, 'closed form K_4: 3!! = 3')
    add(P, (6,), 15, 'closed form K_6: 5!! = 15')
    add(P, (0, []), 1, 'closed form n=0 given edges')
    add(P, (2, [(0, 1)]), 1, 'closed form single edge')
    add(P, (4, [(0, 1), (2, 3)]), 1, 'closed form two disjoint edges')
    add(P, (4, [(0, 1), (1, 2), (2, 3), (3, 0)]), 2, 'closed form C_4')
    add(P, (4, []), 0, 'closed form no edges')
    refuse(P, (2001,))
    refuse(P, (24, [(0, 1)]))
    bad(P, (-1,))
    bad(P, (4, [(0, 4)]))


def e_matchings():
    P = 'matchings_count'
    add(P, (0, []), 1, 'closed form n=0 (empty matching)')
    add(P, (2, [(0, 1)]), 2, 'closed form K_2 (empty + 1 edge)')
    add(P, (2, [(0, 1)], 0), 1, 'closed form size 0')
    add(P, (2, [(0, 1)], 1), 1, 'closed form size 1')
    add(P, (2, [(0, 1)], 2), 0, 'closed form size 2 impossible')
    add(P, (3, [(0, 1), (1, 2), (0, 2)]), 4, 'closed form K_3 (empty + 3 single edges)')
    add(P, (4, [(0, 1), (2, 3)]), 4, 'closed form two independent edges (1+2+1)')
    refuse(P, (24, []))
    bad(P, (-1, []))
    bad(P, (2, [(0, 1)], -1))


def e_perm_allowed():
    P = 'permutations_allowed'
    add(P, ([],), 1, 'closed form empty matrix')
    add(P, ([[1]],), 1, 'closed form 1x1 ones')
    add(P, ([[0]],), 0, 'closed form 1x1 zeros')
    add(P, ([[1, 1], [1, 1]],), 2, 'closed form 2x2 all ones (2!)')
    add(P, ([[1, 0], [0, 1]],), 1, 'closed form 2x2 identity')
    add(P, ([[0, 1], [1, 0]],), 1, 'closed form 2x2 derangement')
    add(P, ([[1, 1, 1], [1, 1, 1], [1, 1, 1]],), 6, 'closed form 3x3 all ones (3!)')
    add(P, ([[0, 1, 1], [1, 0, 1], [1, 1, 0]],), 2, 'closed form 3x3 derangements (!3 = 2)')
    refuse(P, ([[1] * 21 for _ in range(21)],))
    bad(P, ([[1, 0]],), 'non-square matrix')
    bad(P, ('not a matrix',))


# ---------------------------------------------------------------- 36 independent_sets, 37 chromatic_polynomial, 38 colorings_count, 39 graph_automorphisms

def e_independent():
    P = 'independent_sets'
    add(P, (0, []), 1, 'closed form n=0')
    add(P, (1, []), 2, 'closed form n=1 (empty + {0})')
    add(P, (3, []), 8, 'closed form empty graph (2^3)')
    add(P, (3, [(0, 1), (1, 2), (0, 2)]), 4, 'closed form K_3 (empty + 3 singletons)')
    add(P, (3, [(0, 1), (1, 2), (0, 2)], 0), 1, 'closed form K_3 size 0')
    add(P, (3, [(0, 1), (1, 2), (0, 2)], 1), 3, 'closed form K_3 size 1')
    add(P, (3, [(0, 1), (1, 2), (0, 2)], 2), 0, 'closed form K_3 size 2')
    add(P, (3, [(0, 1), (1, 2), (0, 2)], 'poly'), [1, 3], 'closed form K_3 independence poly')
    add(P, (4, [(0, 1), (1, 2), (2, 3)]), 8, 'closed form P_4 (Fibonacci F_6 = 8)')
    refuse(P, (50, []))
    bad(P, (-1, []))
    bad(P, (3, [(0, 3)]))


def e_chromatic():
    P = 'chromatic_polynomial'
    add(P, (0, []), [1], 'closed form n=0')
    add(P, (1, []), [0, 1], 'closed form n=1 (k)')
    add(P, (2, []), [0, 0, 1], 'closed form n=2 (k^2)')
    add(P, (3, []), [0, 0, 0, 1], 'closed form n=3 no-edge (k^3)')
    add(P, (2, [(0, 1)]), [0, -1, 1], 'closed form K_2 (k^2 - k)')
    add(P, (3, [(0, 1)]), [0, 0, -1, 1], 'closed form 3 vertices 1 edge (k^3 - k^2)')
    add(P, (3, [(0, 1), (1, 2)]), [0, 1, -2, 1], 'closed form P_3 (k^3 - 2k^2 + k)')
    add(P, (3, [(0, 1), (1, 2), (0, 2)]), [0, 2, -3, 1], 'closed form K_3 (k^3 - 3k^2 + 2k)')
    add(P, (4, [(0, 1), (1, 2), (2, 3), (3, 0)]), [0, -3, 6, -4, 1], 'closed form C_4 (k^4 - 4k^3 + 6k^2 - 3k)')
    refuse(P, (14, []))
    bad(P, (-1, []))
    bad(P, (3, [(0, 3)]))
    bad(P, (3, [(0, 0)]), 'self loop')


def e_colorings():
    P = 'colorings_count'
    add(P, (0, [], 3), 1, 'closed form n=0')
    add(P, (1, [], 0), 0, 'closed form 0 colours')
    add(P, (1, [], 3), 3, 'closed form 1 vertex, 3 colours')
    add(P, (3, [], 2), 8, 'closed form empty graph on 3 vertices (2^3)')
    add(P, (3, [(0, 1), (1, 2), (0, 2)], 2), 0, 'closed form K_3 with 2 colours')
    add(P, (3, [(0, 1), (1, 2), (0, 2)], 3), 6, 'closed form K_3 with 3 colours (3!)')
    add(P, (4, [(0, 1), (1, 2), (2, 3), (3, 0)], 2), 2, 'closed form C_4 with 2 colours')
    add(P, (4, [(0, 1), (1, 2), (2, 3)], 3), 3 * 2 ** 3, 'closed form tree on 4 vertices')
    refuse(P, (100001, [], 2))
    bad(P, (-1, [], 2))
    bad(P, (3, [], -1))


def e_automorphisms():
    P = 'graph_automorphisms'
    add(P, (0, []), 1, 'closed form n=0')
    add(P, (1, []), 1, 'closed form n=1')
    add(P, (2, []), 2, 'closed form empty graph K_2 complement (2!)')
    add(P, (2, [(0, 1)]), 2, 'closed form K_2 (2!)')
    add(P, (3, [(0, 1), (1, 2), (0, 2)]), 6, 'closed form K_3 (S_3 = 6)')
    add(P, (4, [(0, 1), (1, 2), (2, 3), (3, 0)]), 8, 'closed form C_4 (D_4 = 8)')
    add(P, (4, [(0, 1), (1, 2), (2, 3)]), 2, 'closed form P_4 (reflection only = 2)')
    refuse(P, (17, []))
    bad(P, (-1, []))
    bad(P, (3, [(0, 3)]))


# ---------------------------------------------------------------- 40 union_size, 41 count_divisible, 42 count_coprime

def e_union():
    P = 'union_size'
    add(P, ({},), 0, 'closed form empty dict')
    add(P, ({(0,): 10},), 10, 'closed form single set')
    add(P, ({(0,): 10, (1,): 8, (0, 1): 3},), 15, 'closed form 10 + 8 - 3 = 15')
    add(P, ({(0,): 10, (1,): 10, (2,): 10, (0, 1): 2, (0, 2): 2, (1, 2): 2, (0, 1, 2): 1},), 25, 'closed form 30 - 6 + 1 = 25')
    bad(P, ('not a dict',))
    bad(P, ({(): 5},), 'empty key tuple')
    bad(P, ({(0, 0): 5},), 'repeated index in key')


def e_divisible():
    P = 'count_divisible'
    add(P, (10, [2]), 5, 'closed form 10//2')
    add(P, (0, [2]), 0, 'closed form n=0, lo=1')
    add(P, (10, [2, 3]), 7, 'closed form 5 + 3 - 1 = 7')
    add(P, (100, [2, 3, 5]), 74, 'closed form known example')
    add(P, (100, [2, 3, 5]), 26, 'closed form none divisible', exactly=0)
    add(P, (100, [2, 3, 5]), 3, 'closed form divisible by all three (lcm 30)', exactly=3)
    add(P, (5, [2, 3]), 0, 'closed form exactly 3 impossible', exactly=3)
    add(P, (10, [2]), 3, 'closed form lo=5', lo=5)
    refuse(P, (100, list(range(1, 22))))
    bad(P, (10, [0]), 'divisor 0')
    bad(P, (10, [-2]), 'negative divisor')


def e_coprime():
    P = 'count_coprime'
    add(P, (10, 1), 10, 'closed form coprime to 1')
    add(P, (0, 5), 0, 'closed form n=0, lo=1')
    add(P, (12, 12), 4, 'closed form phi(12) = 4')
    add(P, (100, 30), 26, 'closed form known example')
    add(P, (10, 5), 4, 'closed form lo=5', lo=5)
    refuse(P, (100, 10 ** 13 + 1))
    bad(P, (10, 0), 'm=0')
    bad(P, (10, -5), 'm < 0')


# ---------------------------------------------------------------- 43 brute_count, 44 predicate_histogram, 45 prob_uniform, 46 expectation_uniform

def e_brute():
    P = 'brute_count'
    add(P, (['1..6', '1..6'], 'x0 + x1 == 7'), 6, 'brute sum 7 of two dice')
    add(P, (['0..0'], 'x0 == 0'), 1, 'brute single element')
    add(P, (['1..5'], 'x0 > 10'), 0, 'brute no matching elements')
    add(P, ('1..4', 'x0 < x1'), 6, 'brute combinations of 2 from 4', space='combinations', r=2)
    refuse(P, (['1..10000', '1..10000'], 'x0 < x1'))
    bad(P, (['1..6'], 'syntax error !!!'))


def e_hist():
    P = 'predicate_histogram'
    add(P, (['1..10'], ['x0 % 2 == 0']), {0: 5, 1: 5}, 'hist single predicate')
    add(P, (['1..30'], ['x0 % 2 == 0', 'x0 % 3 == 0']), {0: 10, 1: 15, 2: 5}, 'hist two predicates')
    bad(P, (['1..10'], []), 'empty predicates list')
    bad(P, (['1..10'], ['x0 == 1'] * 13), 'more than 12 predicates')


def e_prob():
    P = 'prob_uniform'
    add(P, (['1..6'], 'x0 % 2 == 0'), F(1, 2), 'prob fair die even')
    add(P, (['1..6', '1..6'], 'x0 + x1 == 7'), F(1, 6), 'prob sum 7')
    add(P, (['1..6', '1..6'], 'x0 + x1 == 7'), F(1, 6), 'conditional prob', given='x0 == 3')
    bad(P, (['1..6'], 'x0 == 1'), 'impossible given condition', given='x0 > 10')


def e_expect():
    P = 'expectation_uniform'
    add(P, (['1..6'], 'x0'), {'mean': F(7, 2), 'variance': F(35, 12), 'count': 6}, 'expectation fair die')
    add(P, (['0..0'], 'x0'), {'mean': F(0), 'variance': F(0), 'count': 1}, 'expectation constant 0')
    bad(P, (['1..0'], 'x0'), 'empty space')


# ---------------------------------------------------------------- 47 dice_sum_distribution .. 50 dist_stats

def e_dice_dist():
    P = 'dice_sum_distribution'
    add(P, (0,), {0: 1}, 'closed form 0 dice')
    add(P, (1, 4), {1: 1, 2: 1, 3: 1, 4: 1}, 'closed form 1 die 4 sides')
    add(P, (2, 2), {2: 1, 3: 2, 4: 1}, 'closed form 2 dice 2 sides')
    add(P, (2,), {0: 1, 1: 2, 2: 1}, 'closed form custom faces [0, 1]', faces=[0, 1])
    refuse(P, (5001, 6))
    bad(P, (-1, 6))
    bad(P, (2,), 'empty faces list', faces=[])


def e_dice_prob():
    P = 'dice_sum_probability'
    add(P, (1, 6, 3), F(1, 6), 'closed form 1 die total 3')
    add(P, (2, 6, 7), F(1, 6), 'closed form 2 dice total 7')
    add(P, (2, 6, 2, 4), F(6, 36), 'closed form 2 dice total 2..4')
    bad(P, (-1, 6, 3))


def e_dice_ext():
    P = 'dice_extreme_distribution'
    add(P, (1, 6), {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}, 'closed form 1 die max')
    add(P, (1, 6), {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}, 'closed form 1 die min', stat='min')
    add(P, (2, 3), {1: 1, 2: 3, 3: 5}, 'closed form 2 dice 3 sides max')
    add(P, (2, 3), {1: 5, 2: 3, 3: 1}, 'closed form 2 dice 3 sides min', stat='min')
    refuse(P, (5001, 6))
    bad(P, (2, 6), 'invalid stat', stat='median')


def e_dist_stats():
    P = 'dist_stats'
    add(P, ([(1, 1)],), {'mean': F(1), 'variance': F(0), 'total_weight': F(1)}, 'stats single point')
    add(P, ([(1, 1), (2, 1), (3, 1)],), {'mean': F(2), 'variance': F(2, 3), 'total_weight': F(3)}, 'stats uniform 1..3')
    bad(P, ([],), 'empty pairs')
    bad(P, ([(1, -1)],), 'negative weight')
    bad(P, ([(1, 0)],), 'zero total weight')


# ---------------------------------------------------------------- 51 bayes_posterior .. 55 geometric_distribution

def e_bayes():
    P = 'bayes_posterior'
    add(P, ([F(1, 2), F(1, 2)], [F(1, 4), F(1, 8)]), [F(2, 3), F(1, 3)], 'bayes 2 hypotheses')
    add(P, ([1], [F(1, 2)]), [F(1)], 'bayes 1 hypothesis')
    bad(P, ([F(1, 3), F(1, 3)], [1, 1]), 'priors do not sum to 1')
    bad(P, ([F(1, 2), F(1, 2)], [0, 0]), 'evidence probability 0')
    bad(P, ([F(1, 2)], [F(1, 2), F(1, 2)]), 'length mismatch')


def e_binom():
    P = 'binomial_probability'
    add(P, (0, F(1, 2), 0), F(1), 'closed form n=0')
    add(P, (5, 0, 0), F(1), 'closed form p=0, X=0')
    add(P, (5, 0, 1), F(0), 'closed form p=0, X=1')
    add(P, (5, 1, 5), F(1), 'closed form p=1, X=n')
    add(P, (4, F(1, 2), 2), F(3, 8), 'closed form 4 flips 2 heads')
    add(P, (4, F(1, 2), 0, 4), F(1), 'closed form full range')
    refuse(P, (20001, F(1, 2), 5))
    bad(P, (-1, F(1, 2), 0))
    bad(P, (5, F(3, 2), 0), 'p > 1')


def e_hyper():
    P = 'hypergeometric_probability'
    add(P, (0, 0, 0, 0), F(1), 'closed form N=0')
    add(P, (10, 5, 0, 0), F(1), 'closed form draw 0')
    add(P, (10, 0, 5, 0), F(1), 'closed form 0 marked items')
    add(P, (10, 10, 5, 5), F(1), 'closed form all marked items')
    add(P, (4, 2, 2, 1), F(4, 6), 'closed form 2 from 4 with 2 marked')
    refuse(P, (10 ** 6 + 1, 5, 5, 2))
    bad(P, (10, 11, 5, 2), 'K > N')
    bad(P, (10, 5, 11, 2), 'n > N')


def e_mhyper():
    P = 'multivariate_hypergeometric'
    add(P, ([2, 2], [0, 0]), F(1), 'closed form draw 0')
    add(P, ([2, 2], [1, 1]), F(2, 3), 'closed form draw 1 each')
    add(P, ([2, 2], [3, 0]), F(0), 'closed form draw > count')
    add(P, ([5], [3]), F(1), 'closed form single category')
    refuse(P, ([10 ** 6 + 1], [1]))
    bad(P, ([2, 2], [1]), 'length mismatch')
    bad(P, ([], []), 'empty list')


def e_geom():
    P = 'geometric_distribution'
    add(P, (1, 1), {'pmf': F(1), 'cdf': F(1), 'tail': F(0), 'mean': F(1), 'variance': F(0)}, 'closed form p=1')
    add(P, (F(1, 2), 0), {'pmf': F(0), 'cdf': F(0), 'tail': F(1), 'mean': F(2), 'variance': F(2)}, 'closed form n=0')
    add(P, (F(1, 2), 1), {'pmf': F(1, 2), 'cdf': F(1, 2), 'tail': F(1, 2), 'mean': F(2), 'variance': F(2)}, 'closed form n=1')
    add(P, (F(1, 2), 2), {'pmf': F(1, 4), 'cdf': F(3, 4), 'tail': F(1, 4), 'mean': F(2), 'variance': F(2)}, 'closed form n=2')
    refuse(P, (F(1, 2), 10 ** 6 + 1))
    bad(P, (0, 1), 'p=0')
    bad(P, (F(3, 2), 1), 'p > 1')


# ---------------------------------------------------------------- 56 coupon_collector .. 59 birthday_probability

def e_coupon():
    P = 'coupon_collector'
    add(P, (0,), {'mean': F(0), 'variance': F(0)}, 'closed form 0 coupons')
    add(P, (1,), {'mean': F(1), 'variance': F(0)}, 'closed form 1 coupon')
    add(P, (2,), {'mean': F(3), 'variance': F(2)}, 'closed form 2 coupons')
    add(P, (2, 1), {'mean': F(1), 'variance': F(0)}, 'closed form collect 1 of 2')
    add(P, (), {'mean': F(3), 'variance': F(2)}, 'closed form probs [1/2, 1/2]', probs=[F(1, 2), F(1, 2)])
    refuse(P, (100001,))
    refuse(P, (), 'more than 16 probs', probs=[F(1, 17)] * 17)
    bad(P, (-1,))
    bad(P, (), 'probs sum != 1', probs=[F(1, 2)])


def e_coupon_prob():
    P = 'coupon_collector_probability'
    add(P, (0, 0), F(1), 'closed form n=0')
    add(P, (1, 1), F(1), 'closed form 1 coupon 1 draw')
    add(P, (2, 1), F(0), 'closed form 2 coupons 1 draw')
    add(P, (2, 2), F(1, 2), 'closed form 2 coupons 2 draws')
    add(P, (2, 3), F(3, 4), 'closed form 2 coupons 3 draws')
    refuse(P, (2001, 10))
    refuse(P, (10, 20001))
    bad(P, (-1, 5))


def e_gambler():
    P = 'gamblers_ruin'
    add(P, (0, 5, F(1, 2)), {'win': F(0), 'ruin': F(1), 'expected_duration': F(0)}, 'closed form i=0')
    add(P, (5, 5, F(1, 2)), {'win': F(1), 'ruin': F(0), 'expected_duration': F(0)}, 'closed form i=N')
    add(P, (1, 2, F(1, 2)), {'win': F(1, 2), 'ruin': F(1, 2), 'expected_duration': F(1)}, 'closed form symmetric 1 of 2')
    add(P, (1, 3, F(1, 2)), {'win': F(1, 3), 'ruin': F(2, 3), 'expected_duration': F(2)}, 'closed form symmetric 1 of 3')
    refuse(P, (1, 2001, F(1, 2)))
    bad(P, (3, 2, F(1, 2)), 'i > N')
    bad(P, (1, 5, 0), 'p <= 0')
    bad(P, (1, 5, 1), 'p >= 1')


def e_birthday():
    P = 'birthday_probability'
    add(P, (0, 365), {'distinct': F(1), 'collision': F(0)}, 'closed form n=0')
    add(P, (1, 365), {'distinct': F(1), 'collision': F(0)}, 'closed form n=1')
    add(P, (2, 365), {'distinct': F(364, 365), 'collision': F(1, 365)}, 'closed form n=2')
    add(P, (366, 365), {'distinct': F(0), 'collision': F(1)}, 'closed form pigeonhole n > days')
    refuse(P, (10 ** 5 + 1, 365))
    bad(P, (5, 0), 'days=0')
    bad(P, (-1, 365))


# ---------------------------------------------------------------- 60 markov_absorption .. 63 markov_distribution

def e_absorption():
    P = 'markov_absorption'
    M = [[1, 0, 0], [F(1, 2), 0, F(1, 2)], [0, 0, 1]]
    add(P, (M,), {'absorbing': [0, 2], 'transient': [1], 'probabilities': {1: {0: F(1, 2), 2: F(1, 2)}}, 'expected_steps': {1: F(1)}}, 'markov simple absorption')
    refuse(P, ([[1 if i == j else 0 for j in range(61)] for i in range(61)],))
    bad(P, ([[0.5, 0.5], [0.5, 0.5]],), 'float matrix')
    bad(P, ([[F(1, 2)], [1]],), 'non-square matrix')


def e_hitting():
    P = 'markov_hitting_time'
    M = [[F(1, 2), F(1, 2)], [0, 1]]
    add(P, (M, [1], 0), F(2), 'hitting time geometric mean 2')
    add(P, (M, [1], 1), F(0), 'hitting time already at target')
    add(P, (M, [1]), {0: F(2), 1: F(0)}, 'hitting time all states')
    refuse(P, ([[1 if i == j else 0 for j in range(61)] for i in range(61)], [0]))
    bad(P, (M, [1], -1), 'negative start')


def e_stationary():
    P = 'markov_stationary'
    M2 = [[F(1, 2), F(1, 2)], [F(1, 2), F(1, 2)]]
    add(P, (M2,), [F(1, 2), F(1, 2)], 'stationary symmetric 2 states')
    M3 = [[F(1, 3), F(1, 3), F(1, 3)], [F(1, 3), F(1, 3), F(1, 3)], [F(1, 3), F(1, 3), F(1, 3)]]
    add(P, (M3,), [F(1, 3), F(1, 3), F(1, 3)], 'stationary uniform 3 states')
    refuse(P, ([[F(1, 61)] * 61 for _ in range(61)],))
    bad(P, ([[F(1, 2), F(1, 3)], [0, 1]],), 'rows do not sum to 1')


def e_mdist():
    P = 'markov_distribution'
    M = [[0, 1], [1, 0]]
    add(P, (M, 0, 0), [1, 0], 'closed form 0 steps')
    add(P, (M, 0, 1), [0, 1], 'closed form 1 step')
    add(P, (M, 0, 2), [1, 0], 'closed form 2 steps')
    add(P, (M, [F(1, 2), F(1, 2)], 1), [F(1, 2), F(1, 2)], 'stationary start')
    refuse(P, ([[F(1, 2), F(1, 2)], [F(1, 2), F(1, 2)]], 0, 10 ** 8))
    bad(P, (M, 2, 1), 'start state out of range')
    bad(P, (M, 0, -1), 'negative steps')


# ---------------------------------------------------------------- 64 pattern_race .. 66 walk_distribution

def e_pattern_race():
    P = 'pattern_race'
    add(P, (['H', 'T'],), {'probabilities': [F(1, 2), F(1, 2)], 'expected_length': F(1)}, 'pattern race symmetric 1 letter')
    add(P, (['HH', 'TT'],), {'probabilities': [F(1, 2), F(1, 2)], 'expected_length': F(3)}, 'pattern race symmetric 2 letters')
    refuse(P, ([],), 'empty patterns list')
    refuse(P, (['H' * 35, 'T' * 35],))
    bad(P, (['H', 'T'], 'HT', [F(1, 2)]), 'probs length mismatch')


def e_wait():
    P = 'expected_wait_pattern'
    add(P, ('H',), 2, 'wait 1 letter')
    add(P, ('HH',), 6, 'wait 2 letters (2^2 + 2)')
    add(P, ('HT',), 4, 'wait 2 letters no overlap (2^2)')
    add(P, ('0', '012'), 3, 'wait 1 letter alphabet size 3')
    bad(P, ('H' * 1001,), 'pattern longer than 1000')
    bad(P, ('',), 'empty pattern')
    bad(P, ('A', 'BC'), 'pattern character outside alphabet')


def e_walk():
    P = 'walk_distribution'
    mv = [(1, 1), (-1, 1)]
    add(P, (mv, 0), {0: F(1)}, 'walk 0 steps')
    add(P, (mv, 1), {-1: F(1, 2), 1: F(1, 2)}, 'walk 1 step')
    add(P, (mv, 2), {-2: F(1, 4), 0: F(1, 2), 2: F(1, 4)}, 'walk 2 steps')
    add(P, (mv, 2), {-1: F(1, 2), 0: F(1, 4), 2: F(1, 4)}, 'walk absorbing at -1', absorbing=[-1])
    bad(P, ([], 1), 'empty moves')
    bad(P, ([(1, -1)], 1), 'negative weight')


# ---------------------------------------------------------------- 67 chords_cross .. 70 matching_crossings

def e_chords_cross():
    P = 'chords_cross'
    add(P, (1, 3, 2, 4), True, 'chords cross')
    add(P, (1, 2, 3, 4), False, 'chords do not cross')
    add(P, (1, 2, 2, 3), False, 'chords sharing endpoint do not cross')
    add(P, (1, 4, 2, 3), False, 'nested chords do not cross')
    bad(P, (1.5, 2, 3, 4), 'float position')
    bad(P, ('a', 2, 3, 4), 'string position')


def e_crossing_pairs():
    P = 'crossing_pairs'
    add(P, ([],), 0, 'empty chord list')
    add(P, ([(1, 2)],), 0, 'single chord')
    add(P, ([(1, 3), (2, 4)],), 1, 'one crossing pair')
    add(P, ([(1, 2), (3, 4)],), 0, 'disjoint chords')
    add(P, ([(1, 4), (2, 5), (3, 6)],), 3, 'three mutually crossing chords')
    refuse(P, ([(i, i + 1) for i in range(2001)],))
    bad(P, ([(1,)],), 'chord is not a pair')


def e_chord_prob():
    P = 'chord_crossing_probability'
    add(P, ('four_points',), F(1, 3), 'four points model')
    add(P, ('four_points', 10), F(1, 3), 'four points model with n_points')
    add(P, ('two_distinct_chords', 3), F(0, 1), 'two distinct chords 3 points')
    add(P, ('two_distinct_chords', 4), F(1, 15), 'two distinct chords 4 points')
    add(P, ('two_distinct_chords', 6), F(1, 7), 'two distinct chords 6 points')
    add(P, ('independent_chords', 4), F(1, 18), 'independent chords 4 points')
    add(P, ('independent_points', 4), F(1, 32), 'independent points 4 points')
    refuse(P, ('two_distinct_chords', 10 ** 6 + 1))
    bad(P, ('unknown_model',))
    bad(P, ('two_distinct_chords', 1), 'need at least two chords')


def e_matching_crossings():
    P = 'matching_crossings'
    add(P, (0,), {'matchings': 1, 'non_crossing': 1, 'probability_non_crossing': F(1), 'expected_crossings': F(0)}, 'm=0 points')
    add(P, (1,), {'matchings': 1, 'non_crossing': 1, 'probability_non_crossing': F(1), 'expected_crossings': F(0)}, 'm=1 chord')
    add(P, (2,), {'matchings': 3, 'non_crossing': 2, 'probability_non_crossing': F(2, 3), 'expected_crossings': F(1, 3)}, 'm=2 chords')
    add(P, (3,), {'matchings': 15, 'non_crossing': 5, 'probability_non_crossing': F(1, 3), 'expected_crossings': F(1)}, 'm=3 chords')
    refuse(P, (2001,))
    bad(P, (-1,))


def edge_cases():
    """All edge cases, in a fixed order (built once)."""
    if not _CASES:
        for name, fn in sorted(globals().items()):
            if name.startswith('e_') and callable(fn):
                fn()
    return list(_CASES)
