"""every case of a corpus with a short time guard: each call must come back (a value that agrees, or DEVICE_FAILED), never hang or
come back wrong"""
import sys, time
sys.path.insert(0, r'D:\Github\mip-research')
from tools.igc.mathbackend import cuda_backend, harness
head, rows = harness.load(sys.argv[1])
ok = failed = wrong = 0
t0 = time.time()
for r in rows:
    got = cuda_backend.call(r['op'], r['args'].encode())
    if got == r['out'].encode(): ok += 1
    elif got.startswith(b'device:DEVICE_FAILED'): failed += 1
    else: wrong += 1; print('WRONG', r['op'], r['args'][:80], got[:80])
print(f'{sys.argv[1]}: agree {ok}, DEVICE_FAILED {failed}, wrong {wrong}, {time.time() - t0:.1f} s')
