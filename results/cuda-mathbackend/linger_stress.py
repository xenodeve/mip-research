"""linger_stress.py: mode 2 with lingering under requests that arrive around the window's end (the Dekker race). Each call is
add(i, 1) with a new i, so an answer that belongs to another request is caught; the gap before each call is drawn uniformly from 0 to
--gap-us (a busy wait). Prints the wrong answers (must be 0), the launches and the calls a lingering kernel served.
python results/cuda-mathbackend/linger_stress.py --n 20000 --gap-us 120   (MIP_CUDA_RESIDENT=2, MIP_LINGER_US=50)
"""
import argparse
import ctypes
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.igc.mathbackend import canon, cuda_backend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=20000)
ap.add_argument('--gap-us', type=float, default=120)
a = ap.parse_args()
lib = cuda_backend._load()
lib.mip_skipped.restype = ctypes.c_uint64
rng = random.Random(7)
wrong = 0
t_all = time.perf_counter()
for i in range(a.n):
    end = time.perf_counter() + rng.uniform(0, a.gap_us) * 1e-6
    while time.perf_counter() < end:
        pass
    x = rng.randrange(-10 ** 17, 10 ** 17)
    got = cuda_backend.call('numbers.add', canon.encode([x, 1]))
    if got != canon.encode(x + 1):
        wrong += 1
        if wrong <= 5:
            print(f'WRONG call {i}: add({x}, 1) -> {got[:60]}')
print(f'{a.n} calls, gaps 0-{a.gap_us} us: wrong {wrong}; launches {cuda_backend.launches()}, served by a lingering kernel '
      f'{lib.mip_skipped()}; {time.perf_counter() - t_all:.1f} s')
cuda_backend.close()
