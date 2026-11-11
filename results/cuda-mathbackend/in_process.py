"""in_process.py: MIP inside the engine's process (the module's end shape), with a stand-in engine: mip_engine_start runs back-to-back
kernels that fill every SM for --step-us each, on their own stream of the same CUDA context. Measured: the latency of numbers.add 1 + 1
every --gap-ms while the engine runs, and the engine's steps per second with and without the calls (its slowdown).
python results/cuda-mathbackend/in_process.py --step-us 50 --gap-ms 5 --n 1000   (MIP_CUDA_RESIDENT, MIP_LINGER_US as usual)
"""
import argparse
import ctypes
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.igc.mathbackend import canon, cuda_backend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--step-us', type=float, default=50)
ap.add_argument('--gap-ms', type=float, default=5)
ap.add_argument('--n', type=int, default=1000)
a = ap.parse_args()
lib = cuda_backend._load()
lib.mip_engine_start.argtypes = [ctypes.c_double, ctypes.c_int]
lib.mip_engine_stop.restype = ctypes.c_double
b = canon.encode([1, 1])
cuda_backend.call('numbers.add', b)
time.sleep(1.5)   # a resident worker or a lingering kernel from that call has ended: 'alone' is alone

assert lib.mip_engine_start(a.step_us, 0) == 0
time.sleep(3)
alone = lib.mip_engine_stop()

assert lib.mip_engine_start(a.step_us, 0) == 0
time.sleep(1)
lat, seen, done, work = [], [], [], []
v = (ctypes.c_uint64 * 8)()
for _ in range(a.n):
    if a.gap_ms:
        time.sleep(a.gap_ms / 1000)
    t = time.perf_counter()
    got = cuda_backend.call('numbers.add', b)
    lat.append((time.perf_counter() - t) * 1e6)
    lib.mip_diag(v)
    seen.append(v[4] / 1000)
    done.append(v[5] / 1000)
    work.append(cuda_backend.last_device_us())
    assert got == b'["i","2"]'
beside = lib.mip_engine_stop()
q = lambda xs, f: sorted(xs)[int(f * (len(xs) - 1))]
print(f'step {a.step_us} us, gap {a.gap_ms} ms: latency median {statistics.median(lat):.1f} us (p10 {q(lat, .1):.1f}, p90 {q(lat, .9):.1f}, '
      f'max {max(lat):.0f}); engine {alone:.0f} steps/s alone, {beside:.0f} with the calls ({100 * (beside / alone - 1):+.1f} %)')
print(f'   inside the DLL (resident mode): request -> seen by the device {statistics.median(seen):.1f} us, -> answer back '
      f'{statistics.median(done):.1f} us; the work {statistics.median(work):.1f} us')
cuda_backend.close()
