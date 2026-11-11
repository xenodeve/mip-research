"""ping.py: the round trip of the smallest call (numbers.add 1 + 1) through the C ABI, repeated, to tune the transport apart from the
work. Prints the median and 10th / 90th percentile of the wall time per call and of the device's own time, and the launches.
python results/cuda-mathbackend/ping.py [--n 2000]   (MIP_CUDA_RESIDENT, MIP_SPIN, MIP_NANOSLEEP as the DLL reads them)
"""
import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.igc.mathbackend import canon, cuda_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000)
    a = ap.parse_args()
    args = canon.encode([1, 1])
    assert cuda_backend.call('numbers.add', args) == b'["i","2"]'
    import ctypes
    lib = cuda_backend._load()
    lib.mip_skipped.restype = ctypes.c_uint64
    v = (ctypes.c_uint64 * 8)()
    wall, dev, seen, done = [], [], [], []
    for _ in range(a.n):
        t = time.perf_counter()
        cuda_backend.call('numbers.add', args)
        wall.append((time.perf_counter() - t) * 1e6)
        dev.append(cuda_backend.last_device_us())
        lib.mip_diag(v)
        seen.append(v[4] / 1000)
        done.append(v[5] / 1000)
    print('(mode 2: "seen" is the launch call returning)')
    print(f'host clock, median: request written -> seen flag back {statistics.median(seen):.1f} us; -> answer back '
          f'{statistics.median(done):.1f} us')
    khz = v[3] or 1
    print(f'diag (last call): polls {v[0]} over {v[1] / khz * 1000:.0f} us -> {v[1] / max(1, v[0]) / khz * 1000:.2f} us per poll; '
          f'device saw the request {v[2] / khz * 1000:.1f} us after its previous answer')
    noop = []
    for _ in range(a.n):
        t = time.perf_counter()
        cuda_backend.call('numbers.none', args)
        noop.append((time.perf_counter() - t) * 1e6)
    print(f'wrapper alone (an op the DLL refuses before any launch): median {statistics.median(noop):.1f} us')
    wall.sort()
    q = lambda xs, f: xs[int(f * (len(xs) - 1))]
    env = {k: os.environ.get(k, '-') for k in ('MIP_CUDA_RESIDENT', 'MIP_SPIN', 'MIP_LINGER_SLEEP', 'MIP_LINGER_US')}
    print(f'{env} wall us median {statistics.median(wall):.1f} p10 {q(wall, .1):.1f} p90 {q(wall, .9):.1f}; '
          f'device us median {statistics.median(dev):.1f}; launches {cuda_backend.launches()}, '
          f'served by a lingering kernel {lib.mip_skipped() if hasattr(lib, "mip_skipped") else "-"}')
    cuda_backend.close()


if __name__ == '__main__':
    main()
