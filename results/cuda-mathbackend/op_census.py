"""op_census.py: which MathBackend operations the model's text reaches, with what argument sizes, and Python's time per call: the
baseline that orders the CUDAMathBackend port and is the denominator of its speed test (docs/plans/2026-11-11-cuda-mathbackend-design.md).

Round 9's General traces (4B NVFP4 W4A8 · llama.cpp · PC layer, r9a-gen, r9a3-gen, r9b-gen, 240 answers) are replayed under round 10's
arm B (sim_replay, --recorded), the way tir4b would act on them, with every registered operation wrapped: each call is counted with the
bit length of its integer arguments (numerators and denominators for a Fraction) and its wall time in Python. CPU only; no GPU.
python -u results/cuda-mathbackend/op_census.py > results/cuda-mathbackend/op_census.txt   (REPLAY_WORKERS=N, default 8)
"""
import json, os, sys, time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'round10'))
sys.stdout.reconfigure(encoding='utf-8')

import replay_gate as G  # noqa: E402  (rows, halves, arm B, the tokenizer set-up)

_CALLS = None


def _bits(x):
    if isinstance(x, bool):
        return 1
    if isinstance(x, int):
        return abs(x).bit_length()
    if isinstance(x, Fraction):
        return max(abs(x.numerator).bit_length(), x.denominator.bit_length())
    if isinstance(x, (list, tuple)):
        return max((_bits(y) for y in x), default=0)
    return 0


def _init():
    """Load the tokenizer and rows (as the gate does), then wrap every registered operation."""
    global _CALLS
    G._init()
    from tools.igc import mathbackend
    mathbackend.load_all()
    _CALLS = defaultdict(lambda: dict(n=0, seconds=0.0, bits=Counter(), refused=0))

    def wrap(name, fn):
        def counted(*a, **kw):
            c = _CALLS[name]
            c['n'] += 1
            c['bits'][min(4096, 1 << max(0, max((_bits(x) for x in list(a) + list(kw.values())), default=0) - 1).bit_length())] += 1
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            except (mathbackend.NotCovered, mathbackend.BadInput):
                c['refused'] += 1
                raise
            finally:
                c['seconds'] += time.perf_counter() - t
        return counted
    for name, entry in mathbackend._REG.items():
        entry['fn'] = wrap(name, entry['fn'])


def job(half, part, parts):
    G.job('census', G.ARM_B, half, part, parts)
    return {k: dict(v, bits=dict(v['bits'])) for k, v in _CALLS.items()}


def main():
    from concurrent.futures import ProcessPoolExecutor
    parts = int(os.environ.get('REPLAY_PARTS', '4'))
    jobs = [(h, p, parts if h != 'r9a3-gen' else 1) for h in G.HALVES for p in range(parts if h != 'r9a3-gen' else 1)]
    total = defaultdict(lambda: dict(n=0, seconds=0.0, bits=Counter(), refused=0))
    with ProcessPoolExecutor(int(os.environ.get('REPLAY_WORKERS', '8')), initializer=_init) as pool:
        for got in pool.map(job, *zip(*jobs)):
            for k, v in got.items():
                t = total[k]
                t['n'] += v['n']
                t['seconds'] += v['seconds']
                t['refused'] += v['refused']
                t['bits'].update(v['bits'])
    print(f'arm B over {", ".join(G.HALVES)}: {sum(v["n"] for v in total.values())} calls to {len(total)} operations')
    print('operation                              calls  refused  mean us   arg bits (upper bound: calls)')
    for k, v in sorted(total.items(), key=lambda kv: -kv[1]['n']):
        mean = 1e6 * v['seconds'] / v['n'] if v['n'] else 0
        bits = ', '.join(f'{b}: {n}' for b, n in sorted(v['bits'].items()))
        print(f'{k:38s} {v["n"]:6d} {v["refused"]:8d} {mean:8.1f}   {bits}')
    json.dump({k: dict(v, bits=dict(v['bits'])) for k, v in total.items()},
              open(Path(__file__).with_name('op_census.json'), 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
