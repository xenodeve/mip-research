"""sparse.py: latency of calls that come one at a time with idle gaps between them (the way tir4b calls: rarely), so the GPU has time
to drop to its idle clock (P8, ~195 MHz on the 5060 Ti) between calls. numbers.add 1 + 1, --n calls each --gap-ms apart; Python's
MathBackend timed on the same rhythm for comparison.
python results/cuda-mathbackend/sparse.py --n 200 --gap-ms 50
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.igc.mathbackend import canon, cuda_backend, run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=200)
ap.add_argument('--gap-ms', type=float, default=50)
a = ap.parse_args()
b = canon.encode([1, 1])
cuda_backend.call('numbers.add', b)
dev, py, devt = [], [], []
for _ in range(a.n):
    time.sleep(a.gap_ms / 1000)
    t = time.perf_counter()
    cuda_backend.call('numbers.add', b)
    dev.append((time.perf_counter() - t) * 1e6)
    devt.append(cuda_backend.last_device_us())
    time.sleep(a.gap_ms / 1000)
    t = time.perf_counter()
    run('numbers.add', 1, 1)
    py.append((time.perf_counter() - t) * 1e6)
q = lambda xs, f: sorted(xs)[int(f * (len(xs) - 1))]
print(f'gap {a.gap_ms} ms, {a.n} calls: device median {statistics.median(dev):.1f} us (p10 {q(dev, .1):.1f}, p90 {q(dev, .9):.1f}), '
      f'its own work {statistics.median(devt):.1f} us; Python median {statistics.median(py):.1f} us')
cuda_backend.close()
