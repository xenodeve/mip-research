"""floor.py: the least a launch per call costs on this machine, measured inside the DLL (mip_floor): an empty kernel that only raises the
mapped flag, through the runtime API and through the driver API's cuLaunchKernel; beside it the real call (numbers.add 1 + 1) from C
through the same path, for the gap between them. python results/cuda-mathbackend/floor.py   (MIP_CUDA_RESIDENT=2)
"""
import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.igc.mathbackend import cuda_backend  # noqa: E402

lib = cuda_backend._load()
lib.mip_floor.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
out = (ctypes.c_double * 2)()
for rep in range(2):
    for api, name in ((0, 'runtime cudaLaunchKernel'), (1, 'driver cuLaunchKernel')):
        assert lib.mip_floor(5000, api, out) == 0
        print(f'empty kernel, {name}: round trip median {out[0]:.2f} us, of it the launch call {out[1]:.2f} us')
cuda_backend.close()
