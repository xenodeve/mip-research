"""Mutation test of tools/igc/mathbackend/numbers.py against sim_numbers.py (no model, no network).

    python tools/mutation_mathbackend_numbers.py [--seed 1] [--n 60] [--per-op 6] [--want 2] [--jobs 8] [--only numbers.gcd ...]

For every registered operation it builds mutants of the operation's own function (and of the private helpers that function calls, transitively; the argument-check helpers
`_int _rat _small _pos _nonneg _bits` are not mutated on their own, their call sites in the operation are). A mutant is one change: an arithmetic operator, a comparison, and/or,
a `not`, or an integer constant (+1). The mutants of one operation are tried in a fixed pseudo-random order, each on a COPY of the package under build-mathbackend/mut/ (the
real files are never touched): the copy is run through `sim_numbers --only <op>` and the mutant is KILLED when the simulation reports at least one disagreement for that operation,
SURVIVED when it reports none, and a TIMEOUT (a mutant that hangs) is set aside and another one is tried. It stops for an operation after `--want` kills or `--per-op` decided
mutants. The report (build-mathbackend/mutation_numbers.json) lists every mutant tried with its source line and the verdict, so a survivor can be read and judged (an equivalent
mutant, or a missing case).
"""
import argparse
import ast
import concurrent.futures
import copy
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'tools' / 'igc' / 'mathbackend'
WORK = ROOT / 'build-mathbackend' / 'mut'
CHECKS = {'_int', '_rat', '_small', '_pos', '_nonneg', '_bits'}

BINOPS = {
    ast.Add: [ast.Sub, ast.Mult],
    ast.Sub: [ast.Add, ast.Mult],
    ast.Mult: [ast.Add, ast.Sub],
    ast.FloorDiv: [ast.Mod, ast.Add],
    ast.Mod: [ast.FloorDiv, ast.Add],
    ast.Pow: [ast.Mult, ast.Add],
}
CMPOPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In}


def module_functions(tree):
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def op_table(tree):
    """{op name: function name} from the @op('numbers.x', ...) decorators."""
    out = {}
    for fn in module_functions(tree).values():
        for d in fn.decorator_list:
            if isinstance(d, ast.Call) and getattr(d.func, 'id', '') == 'op' and d.args and isinstance(d.args[0], ast.Constant):
                out[d.args[0].value] = fn.name
    return out


def called_private(fn, funcs):
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    return {x for x in names if x in funcs and x.startswith('_') and x not in CHECKS and x != fn.name}


def helper_closure(fn, funcs):
    seen, todo = set(), [fn]
    while todo:
        cur = todo.pop()
        for h in called_private(cur, funcs):
            if h not in seen:
                seen.add(h)
                todo.append(funcs[h])
    return sorted(seen)


def sites(fn):
    """Mutation sites inside one function, as (index of the node in ast.walk order, description, node kind). The docstring and the decorator are not touched."""
    out = []
    body_nodes = []
    for stmt in fn.body:
        if isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, 'value', None), ast.Constant) and isinstance(stmt.value.value, str):
            continue
        body_nodes.extend(ast.walk(stmt))
    for node in body_nodes:
        if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
            for target_op in BINOPS[type(node.op)]:
                out.append((node, f'{type(node.op).__name__}->{target_op.__name__}'))
        elif isinstance(node, ast.Compare):
            for i, o in enumerate(node.ops):
                if type(o) in CMPOPS:
                    out.append((node, f'cmp#{i} {type(o).__name__}->{CMPOPS[type(o)].__name__}'))
        elif isinstance(node, ast.BoolOp):
            out.append((node, 'And<->Or'))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            out.append((node, 'drop not'))
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            out.append((node, f'const {node.value}->{node.value + 1 if node.value >= 1 else 1}'))
        elif isinstance(node, ast.Return) and node.value is not None:
            out.append((node, 'return + 1'))
            out.append((node, 'return - 1'))
    return out


def apply(node, desc):
    if isinstance(node, ast.BinOp):
        target_name = desc.split('->')[1]
        node.op = getattr(ast, target_name)()
    elif isinstance(node, ast.Compare):
        i = int(desc.split('#')[1].split(' ')[0])
        node.ops[i] = CMPOPS[type(node.ops[i])]()
    elif isinstance(node, ast.BoolOp):
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif isinstance(node, ast.UnaryOp):
        node.op = ast.UAdd()
    elif isinstance(node, ast.Constant):
        node.value = node.value + 1 if node.value >= 1 else 1
    elif isinstance(node, ast.Return):
        if desc == 'return + 1':
            node.value = ast.BinOp(left=node.value, op=ast.Add(), right=ast.Constant(value=1))
        elif desc == 'return - 1':
            node.value = ast.BinOp(left=node.value, op=ast.Sub(), right=ast.Constant(value=1))


def make_mutants(source):
    tree = ast.parse(source)
    funcs = module_functions(tree)
    table = op_table(tree)
    out = {}
    for opname, fname in table.items():
        own = [(fname, i) for i in range(len(sites(funcs[fname])))]
        helpers = [(h, i) for h in helper_closure(funcs[fname], funcs) for i in range(len(sites(funcs[h])))]
        rng = random.Random('mut:' + opname)
        rng.shuffle(own)
        rng.shuffle(helpers)
        order = []
        a, b = list(own), list(helpers)
        while a or b:                       # own sites first, a helper site after every two of them
            for _ in range(2):
                if a:
                    order.append(a.pop())
            if b:
                order.append(b.pop())
        out[opname] = order
    return tree, funcs, out


def build_mutant(source, fname, idx):
    tree = ast.parse(source)
    fn = module_functions(tree)[fname]
    node, desc = sites(fn)[idx]
    line = node.lineno
    apply(node, desc)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), desc, line


def run_one(job):
    opname, fname, idx, seed, n, timeout = job
    source = (PKG / 'numbers.py').read_text(encoding='utf-8')
    mutated, desc, line = build_mutant(source, fname, idx)
    d = WORK / f'{opname.split(".")[1]}_{fname}_{idx}'
    if d.exists():
        shutil.rmtree(d)
    pk = d / 'tools' / 'igc' / 'mathbackend'
    pk.mkdir(parents=True)
    for f in PKG.glob('*.py'):
        shutil.copy(f, pk / f.name)
    (pk / 'numbers.py').write_text(mutated, encoding='utf-8')
    env = dict(os.environ, PYTHONPATH=str(d), PYTHONDONTWRITEBYTECODE='1')
    t0 = time.time()
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    try:
        r = subprocess.run([sys.executable, '-W', 'ignore', '-m', 'tools.igc.mathbackend.sim_numbers', '--seed', str(seed), '--n', str(n), '--only', opname],
                           cwd=str(d), env=env, capture_output=True, text=True, timeout=timeout, creationflags=flags)
    except subprocess.TimeoutExpired:
        shutil.rmtree(d, ignore_errors=True)
        return dict(op=opname, func=fname, line=line, change=desc, verdict='timeout', seconds=round(time.time() - t0, 1))
    rep_path = d / 'build-mathbackend' / 'numbers.json'
    verdict, dis, example = 'error', 0, (r.stderr or r.stdout)[-300:]
    if rep_path.exists():
        rep = json.loads(rep_path.read_text(encoding='utf-8'))
        rec = rep['ops'].get(opname)
        if rec:
            dis = rec['disagree']
            verdict = 'killed' if dis else 'survived'
            example = rec['disagree_examples'][0]['detail'][:120] if rec['disagree_examples'] else ''
    shutil.rmtree(d, ignore_errors=True)
    return dict(op=opname, func=fname, line=line, change=desc, verdict=verdict, disagree=dis, example=example, seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--per-op', type=int, default=8)
    ap.add_argument('--want', type=int, default=2)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=240)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    source = (PKG / 'numbers.py').read_text(encoding='utf-8')
    _, _, plans = make_mutants(source)
    ops = [o for o in sorted(plans) if not a.only or o in a.only]
    WORK.mkdir(parents=True, exist_ok=True)
    results = {o: [] for o in ops}
    done = {o: False for o in ops}
    nxt = {o: 0 for o in ops}
    pending = {}
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        def submit(o, k):                       # k = how many mutants to have in flight for this op
            while nxt[o] < len(plans[o]) and len([1 for f, oo in pending.items() if oo == o]) < k:
                fname, idx = plans[o][nxt[o]]
                nxt[o] += 1
                pending[ex.submit(run_one, (o, fname, idx, a.seed, a.n, a.timeout))] = o
        for o in ops:
            submit(o, a.want)
        while pending:
            fin, _ = concurrent.futures.wait(list(pending), return_when=concurrent.futures.FIRST_COMPLETED)
            for f in fin:
                o = pending.pop(f)
                res = f.result()
                results[o].append(res)
                print(f"{o:38s} {res['verdict']:9s} {res['func']}:{res['line']} {res['change']}  {res.get('example', '')[:70]}", flush=True)
                kills = sum(1 for r in results[o] if r['verdict'] == 'killed')
                decided = sum(1 for r in results[o] if r['verdict'] in ('killed', 'survived'))
                if kills < a.want and decided < a.per_op:
                    submit(o, 1)
    table = {}
    for o in ops:
        rs = results[o]
        table[o] = dict(tried=len(rs), killed=sum(r['verdict'] == 'killed' for r in rs), survived=sum(r['verdict'] == 'survived' for r in rs),
                        timeout=sum(r['verdict'] == 'timeout' for r in rs), error=sum(r['verdict'] == 'error' for r in rs), mutants=rs)
    out = ROOT / 'build-mathbackend' / 'mutation_numbers.json'
    out.write_text(json.dumps(table, indent=1), encoding='utf-8')
    print(f"\n{'operation':38s} tried killed survived timeout error")
    for o, t in table.items():
        flag = '' if t['killed'] >= a.want else '   <-- fewer than wanted kills'
        print(f"{o:38s} {t['tried']:5d} {t['killed']:6d} {t['survived']:8d} {t['timeout']:7d} {t['error']:5d}{flag}")
    short = [o for o, t in table.items() if t['killed'] < a.want]
    print(f"\n{len(ops)} operations, {sum(t['killed'] for t in table.values())} mutants killed, {sum(t['survived'] for t in table.values())} survived, "
          f"{sum(t['timeout'] for t in table.values())} timeouts; operations with fewer than {a.want} kills: {len(short)}")
    return 1 if short else 0


if __name__ == '__main__':
    sys.exit(main())
