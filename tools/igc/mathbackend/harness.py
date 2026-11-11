"""harness.py: the gate the CUDAMathBackend passes before an operation counts as ported (docs/plans/2026-11-11-cuda-mathbackend-design.md,
"How 100% is made and proven").

1. A frozen corpus: each case is an operation, its arguments (canonical bytes) and Python's canonical output (a value or a refusal
   class), generated once from the sim_* generators with a seed and written to a versioned JSONL file with the Python version and
   the commit it came from. A seed change makes a new corpus.
2. A comparator: a backend is any callable (name, args_bytes) -> output bytes; the report per operation counts cases, agreements,
   refusals matched and disagreements (with the first examples). An operation is ported only at 100 % on every case.
3. The null backend: Python itself through decode -> run -> encode, which must read 100 % (it shakes out encoding faults before any
   device code exists); and wrong backends, each of which must read below 100 % on the operations it touches (a harness that never
   failed has not been tested).

python -m tools.igc.mathbackend.harness build numbers --ops numbers.add,numbers.gcd --seed 1 --n 60 --out results/cuda-mathbackend/corpus
python -m tools.igc.mathbackend.harness check results/cuda-mathbackend/corpus/numbers-s1.jsonl --backend python
"""
import argparse
import importlib
import json
import platform
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from tools.igc.mathbackend import canon

ROOT = Path(__file__).resolve().parents[3]


# --- the corpus -------------------------------------------------------------------------------------------------------------------------

def cases(module, ops=None, seed=1, n=60):
    """(op, args) of the sim generators for `module`, in a fixed order; the generator's own oracle is dropped (the corpus holds Python's
    output: the oracle already checked Python against SymPy and brute force, mip-research#17)."""
    sim = importlib.import_module(f'tools.igc.mathbackend.sim_{module}')
    names = sorted(sim.GEN) if ops is None else list(ops)
    for name in names:
        rng = random.Random(f'{seed}:{name}')
        for case in sim.GEN[name](rng, n):
            yield name, tuple(case[0])
        for case in getattr(sim, 'BOUNDS', {}).get(name, []):   # the bound cases the sim file names
            yield name, tuple(case[0])
        for args in boundary(name):
            yield name, tuple(args)


def boundary(name):
    """The adversarial boundary set (design "How 100% is made", 2): the cases seeded generators under-sample, built on purpose.
    For the rational operations: Python's Fraction(str) grammar at every edge (Unicode digits and whitespace, underscores, signs,
    '.5' and '5.', a zero denominator, e/n anywhere, the 4000-character cap), and integers across limb boundaries."""
    if name not in ('numbers.add', 'numbers.sub', 'numbers.mul', 'numbers.div'):
        return []
    strings = ['0', '-0', '+7', '-0/5', '.5', '5.', '-.5', '1_000', '1__000', '_1', '1_', '1_000/2_0', '1/0', '0/0', '1 / 2', ' 1/2 ',
               '  3/4 　', '١٢/٣', '१.२', '１２', '1/-2', '--1', '+-1', '.', '-', '', '   ',
               '1e3', '1E3', 'nan', 'inf', 'Infinity', '1.5e0', '0x10', '1/2/3', '1.2.3', '١_٢', '1._5', '1.5_', '00012/0004',
               '7' * 4000, '7' * 4001, ' ' + '7' * 3999 + ' ', '1/' + '9' * 3998, '1.' + '0' * 3998]
    # the device's register path takes parts of at most 9 digits: both sides of that edge (and of 18), as ints and as strings
    strings += ['9' * 9, '9' * 10, '-' + '9' * 9, '9' * 9 + '/' + '9' * 8 + '7', '9' * 10 + '/7', '.' + '9' * 9, '9' * 4 + '.' + '9' * 5,
                '9' * 5 + '.' + '9' * 5, '1/' + '9' * 9, '1/' + '9' * 10]
    strings += ['9' * 18, '9' * 19, '-' + '9' * 18, '9' * 18 + '/' + '9' * 17 + '7', '9' * 19 + '/7', '.' + '9' * 18, '9' * 9 + '.' + '9' * 9,
                '9' * 10 + '.' + '9' * 9, '000000000000000000001', '-.5', '+.5', '5./2']
    ints = [0, 1, -1, 2 ** 32 - 1, 2 ** 32, -(2 ** 32), 2 ** 64 + 1, 10 ** 9, 10 ** 9 - 1, -(10 ** 18), 2 ** 1000 - 1, 3 ** 700,
            10 ** 18 - 1, -(10 ** 18 - 1), 10 ** 18, 2 ** 63 - 1, -(2 ** 63), 999999999, -999999999, 1000000000,
            # past the device's 47 KB shared arena, inside its limb bound: the call must move to the global arena, not refuse
            # (2026-11-11: a 20,000-digit product came back BadInput)
            7 ** 5900, -(3 ** 12000), 10 ** 19999 + 7]
    from fractions import Fraction
    other = [Fraction(-1, 2 ** 64), Fraction(2 ** 64 - 1, 2 ** 32 + 1), True, False, None, 1.5, [1], (1, 2),
             Fraction(10 ** 18 - 1, 10 ** 18 - 3), Fraction(-(10 ** 18 - 1), 7), Fraction(1, 10 ** 18 - 1),
             Fraction(999999999, 999999997), Fraction(-999999999, 1000000000), Fraction(1, 999999999)]
    out = []
    for s in strings:
        out += [(s, 1), (1, s)]
    for i in ints:
        for j in (ints[3], ints[7], Fraction(1, 3), '-5/6'):
            out.append((i, j))
    for x in other:
        out += [(x, 1), (1, x)]
    return out


def build(module, out_dir, ops=None, seed=1, n=60):
    """Write the frozen corpus file and return its path. A case whose Python output is not a value or a refusal (another exception, a
    type canon cannot encode) is listed in the header's `python_bugs`, not frozen: it is fixed in Python first."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{module}-s{seed}.jsonl'
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ''
    rows, bugs = [], []
    for name, args in cases(module, ops, seed, n):
        try:
            args_b = canon.encode(list(args))
            out_b = canon.call(name, args)
        except Exception as e:   # noqa: BLE001  a Python bug or an unencodable type: reported, never frozen
            bugs.append(dict(op=name, args=repr(args)[:200], error=f'{type(e).__name__}: {e}'[:200]))
            continue
        rows.append(dict(op=name, args=args_b.decode('utf-8'), out=out_b.decode('utf-8')))
    head = dict(corpus=1, module=module, seed=seed, n=n, ops=sorted({r['op'] for r in rows}), cases=len(rows),
                python=platform.python_version(), commit=commit, python_bugs=bugs)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(head, ensure_ascii=False) + '\n')
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return path


def load(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    head = json.loads(lines[0])
    return head, [json.loads(x) for x in lines[1:] if x.strip()]


# --- backends ---------------------------------------------------------------------------------------------------------------------------

def python_backend(name, args_b):
    """The null backend: decode the arguments, run Python, encode."""
    return canon.call(name, tuple(canon.decode(args_b)))


def _mutate(out_b, how):
    """A wrong backend's output: Python's, damaged one way. Returns None when the damage does not apply to this output."""
    t = json.loads(out_b)

    def walk(t):
        tag = t[0]
        if how == 'off_by_one' and tag == 'i':
            return ['i', str(int(t[1]) + 1)], True
        if how == 'unreduced' and tag == 'q':
            return ['q', str(2 * int(t[1])), str(2 * int(t[2]))], True
        if how == 'decimal_exponent' and tag == 'd':
            return ['d', t[1] + '0' if '.' in t[1] and 'E' not in t[1] else t[1] + '.0'], True
        if how == 'reorder' and tag in ('l', 't') and len(t[1]) > 1 and t[1] != t[1][::-1]:
            return [tag, t[1][::-1]], True
        if tag in ('l', 't', 'm', 'e'):
            items, done = [], False
            for y in t[1]:
                if done:
                    items.append(y)
                    continue
                if tag == 'm':
                    v, done = walk(y[1])
                    items.append([y[0], v])
                else:
                    v, done = walk(y)
                    items.append(v)
            return [tag] + [items], done
        if tag == 'r':
            p, done = walk(t[1])
            return (['r', p, t[2], t[3]], True) if done else (t, False)
        return t, False
    if how == 'wrong_refusal':
        if t[0] != 'x':
            return None
        return canon.encode_tree(['x', 'BadInput' if t[1] == 'NotCovered' else 'NotCovered'])
    if how == 'refuse_answer':
        return None if t[0] == 'x' else canon.encode_tree(['x', 'NotCovered'])
    new, done = walk(t)
    return canon.encode_tree(new) if done else None


WRONG = ('off_by_one', 'unreduced', 'decimal_exponent', 'reorder', 'wrong_refusal', 'refuse_answer')


def wrong_backend(how):
    """A backend that answers as Python does, damaged `how` wherever the damage applies (elsewhere it answers right)."""
    def backend(name, args_b):
        out = python_backend(name, args_b)
        bad = _mutate(out, how)
        return out if bad is None else bad
    backend.__name__ = f'wrong:{how}'
    return backend


# --- the comparator ---------------------------------------------------------------------------------------------------------------------

def check(corpus_rows, backend, examples=3):
    """Per operation: cases, agree, refusals (in the corpus) matched, disagree, and the first disagreeing examples."""
    rep = defaultdict(lambda: dict(cases=0, agree=0, refusals=0, refusals_matched=0, disagree=0, examples=[]))
    for r in corpus_rows:
        want = r['out'].encode('utf-8')
        try:
            got = backend(r['op'], r['args'].encode('utf-8'))
        except Exception as e:   # noqa: BLE001  a crash is a disagreement, never a skip
            got = f'crash {type(e).__name__}: {e}'.encode('utf-8')
        c = rep[r['op']]
        c['cases'] += 1
        refusal = r['out'].startswith('["x"')
        c['refusals'] += refusal
        if got == want:
            c['agree'] += 1
            c['refusals_matched'] += refusal
        else:
            c['disagree'] += 1
            if len(c['examples']) < examples:
                c['examples'].append(dict(args=r['args'][:160], want=r['out'][:160], got=got.decode('utf-8', 'replace')[:160]))
    return dict(rep)


def all_agree(report):
    return all(c['disagree'] == 0 for c in report.values())


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('module')
    b.add_argument('--ops', default='')
    b.add_argument('--seed', type=int, default=1)
    b.add_argument('--n', type=int, default=60)
    b.add_argument('--out', default=str(ROOT / 'results' / 'cuda-mathbackend' / 'corpus'))
    c = sub.add_parser('check')
    c.add_argument('corpus')
    c.add_argument('--backend', default='python', help="python | wrong:<how> | cuda (the C ABI, once built)")
    a = ap.parse_args(argv)
    if a.cmd == 'build':
        path = build(a.module, a.out, [x for x in a.ops.split(',') if x] or None, a.seed, a.n)
        head, rows = load(path)
        print(f'{path}: {head["cases"]} cases, {len(head["ops"])} ops, python_bugs {len(head["python_bugs"])}')
        return 0
    head, rows = load(a.corpus)
    if a.backend == 'python':
        backend = python_backend
    elif a.backend.startswith('wrong:'):
        backend = wrong_backend(a.backend.split(':', 1)[1])
    elif a.backend == 'cuda':
        from tools.igc.mathbackend import cuda_backend
        backend = cuda_backend.call
    else:
        raise SystemExit(f'unknown backend {a.backend}')
    rep = check(rows, backend)
    for name, r in sorted(rep.items()):
        flag = 'OK ' if r['disagree'] == 0 else 'BAD'
        print(f'{flag} {name:36s} {r["agree"]}/{r["cases"]} (refusals {r["refusals_matched"]}/{r["refusals"]})')
        for e in r['examples']:
            print(f'      args {e["args"]}  want {e["want"]}  got {e["got"]}')
    print(f'{sum(r["agree"] for r in rep.values())}/{sum(r["cases"] for r in rep.values())} agree; '
          f'{"ALL OPS AT 100 %" if all_agree(rep) else "NOT 100 %"}')
    return 0 if all_agree(rep) else 1


if __name__ == '__main__':
    sys.exit(main())
