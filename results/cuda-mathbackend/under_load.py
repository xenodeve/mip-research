"""under_load.py: latency of sparse CUDAMathBackend calls while llama-server decodes on the same card (the PC layer's real condition:
the GPU is awake, and our kernel waits for the other process's context to yield its time slice). Four threads keep a constant decode
load (as impact.py); calls numbers.add 1 + 1 every --gap-ms; Python's MathBackend on the same rhythm.
python results/cuda-mathbackend/under_load.py --server http://127.0.0.1:8099 --n 300 --gap-ms 1
"""
import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import impact  # noqa: E402  (its load threads)
from tools.igc.mathbackend import canon, cuda_backend, run  # noqa: E402
import json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--server', default='http://127.0.0.1:8099')
ap.add_argument('--n', type=int, default=300)
ap.add_argument('--gap-ms', type=float, default=1)
ap.add_argument('--empty', action='store_true', help='time an empty flag-only kernel (mip_floor) instead of a call: the bare floor')
a = ap.parse_args()
import ctypes, os  # noqa: E402
if os.environ.get('MIP_GPU_PRIORITY'):   # Windows' GPU scheduling class for this process: 2 normal, 3 above normal, 4 high, 5 realtime
    cls = int(os.environ['MIP_GPU_PRIORITY'])
    k32, gdi = ctypes.windll.kernel32, ctypes.windll.gdi32
    k32.GetCurrentProcess.restype = ctypes.c_void_p   # a 64-bit pseudo-handle: the default int return truncates it
    gdi.D3DKMTSetProcessSchedulingPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_int]
    gdi.D3DKMTSetProcessSchedulingPriorityClass.restype = ctypes.c_long
    st = gdi.D3DKMTSetProcessSchedulingPriorityClass(k32.GetCurrentProcess(), cls) & 0xFFFFFFFF
    print(f'GPU scheduling priority class {cls}: NTSTATUS {st:#x}', flush=True)
row = json.loads(open('C:/Users/xenod/mip-work/a100/pc9-dev-rows.jsonl', encoding='utf-8').readline())
prompt = '<|im_start|>user\n' + row['question'] + '<|im_end|>\n<|im_start|>assistant\n<think>\n'
ts = [threading.Thread(target=impact.load, args=(a.server, prompt, 1 + k), daemon=True) for k in range(4)]
for t in ts:
    t.start()
time.sleep(15)
b = canon.encode([1, 1])
cuda_backend.call('numbers.add', b)
lib = cuda_backend._load()
lib.mip_floor.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
fo = (ctypes.c_double * 2)()
dv = (ctypes.c_uint64 * 8)()
inside = []
t_calls = time.monotonic()
dev, py = [], []
for _ in range(a.n):
    time.sleep(a.gap_ms / 1000)
    if a.empty:
        lib.mip_floor(1, 0, fo)
        dev.append(fo[0])   # measured inside the DLL
    else:
        t = time.perf_counter()
        cuda_backend.call('numbers.add', b)
        dev.append((time.perf_counter() - t) * 1e6)
        lib.mip_diag(dv)
        inside.append(dv[5] / 1000)   # mode 2: from the request to the flag back, inside the DLL
    time.sleep(a.gap_ms / 1000)
    t = time.perf_counter()
    run('numbers.add', 1, 1)
    py.append((time.perf_counter() - t) * 1e6)
toks = sum(n for t, n in impact.DONE if t >= t_calls)
rate = toks / (time.monotonic() - t_calls)
impact.STOP.set()
q = lambda xs, f: sorted(xs)[int(f * (len(xs) - 1))]
print(f'{"EMPTY kernel" if a.empty else "numbers.add"} under decode load, gap {a.gap_ms} ms, {a.n} calls: device median {statistics.median(dev):.1f} us (p10 {q(dev, .1):.1f}, '
      f'p90 {q(dev, .9):.1f}, max {max(dev):.0f}); Python median {statistics.median(py):.1f} us; decode {rate:.0f} tok/s meanwhile')
if inside:
    print(f'   inside the DLL: median {statistics.median(inside):.1f} us (p10 {q(inside, .1):.1f})')
cuda_backend.close()
