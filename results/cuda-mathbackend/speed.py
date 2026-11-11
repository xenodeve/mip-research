"""speed.py: the CUDAMathBackend against the Python MathBackend on the same calls (design: "Speed", measured per call).

For every case of a frozen corpus whose op the DLL ports: Python's time for the call (decode excluded: run() on Python values) and the
device's time through the C ABI (canonical bytes in and out, the queue round trip or the launch, the copy) in the mode set by
MIP_CUDA_RESIDENT. Each call repeats R times, warm; the median per op and per argument size is reported, with the agreement checked
on every call. One process, one card (CUDA_VISIBLE_DEVICES); run it with no other engine on that card unless that is the question.
python results/cuda-mathbackend/speed.py results/cuda-mathbackend/corpus/numbers-s1.jsonl [--repeat 50]
"""
import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding='utf-8')
from tools.igc.mathbackend import canon, cuda_backend, harness  # noqa: E402
from tools.igc.mathbackend import run as py_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus')
    ap.add_argument('--repeat', type=int, default=50)
    a = ap.parse_args()
    head, rows = harness.load(a.corpus)
    ported = set(cuda_backend.ops())
    rows = [r for r in rows if r['op'] in ported]
    py_t, dev_t, bad = defaultdict(list), defaultdict(list), 0
    for r in rows:
        args_b = r['args'].encode()
        args = tuple(canon.decode(args_b))
        if cuda_backend.call(r['op'], args_b) != r['out'].encode():
            bad += 1
            continue
        size = max(len(x) for x in r['args'].split('"')) // 10 * 10   # digits of the longest argument, to the ten
        for _ in range(a.repeat):
            t = time.perf_counter()
            try:
                py_run(r['op'], *args)
            except Exception:   # noqa: BLE001  refusals are calls too
                pass
            py_t[(r['op'], size)].append(time.perf_counter() - t)
            t = time.perf_counter()
            cuda_backend.call(r['op'], args_b)
            dev_t[(r['op'], size)].append(time.perf_counter() - t)
    print(f'{len(rows)} cases of {sorted(ported)}; {bad} disagreed and were not timed; launches {cuda_backend.launches()}')
    print('op               digits   cases   Python us (median)   device us (median)   device / Python')
    for key in sorted(py_t):
        p, d = statistics.median(py_t[key]) * 1e6, statistics.median(dev_t[key]) * 1e6
        print(f'{key[0]:16s} {key[1]:6d} {len(py_t[key]) // a.repeat:7d} {p:20.1f} {d:20.1f} {d / p:16.1f}x')
    allp = statistics.median(x for v in py_t.values() for x in v) * 1e6
    alld = statistics.median(x for v in dev_t.values() for x in v) * 1e6
    print(f'all calls: Python {allp:.1f} us, device {alld:.1f} us median')
    cuda_backend.close()


if __name__ == '__main__':
    main()
