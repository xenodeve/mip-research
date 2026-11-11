"""cuda_backend.py: the CUDAMathBackend's C ABI (cuda/mathbackend/mipmath.h) from Python, as a harness backend:
call(name, args_bytes) -> canonical output bytes, the same contract as harness.python_backend.

The DLL is build-cuda/mipmath.dll (cuda/mathbackend/build.cmd). Environment: MIP_CUDA_DEVICE (CUDA index, default 0 after
CUDA_VISIBLE_DEVICES), MIP_CUDA_RESIDENT (1: the resident worker, relaunched every ~1 s; 0: one launch per call).
A status that is not a value or a refusal comes back as `device:<status>` bytes, which never equal a canonical encoding, so the
harness counts it as a disagreement.
"""
import ctypes
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATUS = {0: 'OK', 1: 'NOT_COVERED', 2: 'BAD_INPUT', 3: 'DEVICE_FAILED', 4: 'UNKNOWN_OP', 5: 'HOST_ERROR'}
_lib = None
_out = None


def _load():
    global _lib, _out
    if _lib is not None:
        return _lib
    path = os.environ.get('MIP_CUDA_DLL', str(ROOT / 'build-cuda' / 'mipmath.dll'))
    os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64')
    os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin')
    lib = ctypes.CDLL(path)
    lib.mip_open.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.mip_call.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_size_t)]
    lib.mip_ops.restype = ctypes.c_char_p
    lib.mip_launches.restype = ctypes.c_uint64
    lib.mip_last_device_us.restype = ctypes.c_double
    st = lib.mip_open(int(os.environ.get('MIP_CUDA_DEVICE', '0')), int(os.environ.get('MIP_CUDA_RESIDENT', '1')))
    if st != 0:
        raise RuntimeError(f'mip_open: {STATUS.get(st, st)}')
    _out = ctypes.create_string_buffer(1 << 20)
    _lib = lib
    return lib


def ops():
    return [x for x in _load().mip_ops().decode().split('\n') if x]


def launches():
    return _load().mip_launches()


_names = {}            # op name -> its encoded bytes, made once
_n = ctypes.c_size_t(0)
_n_ref = None
_call = None
_out_len = 0


def call(name, args_b):
    """The wrapper keeps per call only what must change: the op's bytes are cached, the length cell and its reference are made once,
    and `args_b` goes to ctypes as it is (a bytes object needs no copy)."""
    global _n_ref, _call, _out_len
    if _call is None:
        lib = _load()
        _call, _n_ref, _out_len = lib.mip_call, ctypes.byref(_n), len(_out)
    nb = _names.get(name)
    if nb is None:
        nb = _names[name] = name.encode()
    if type(args_b) is not bytes:
        args_b = bytes(args_b)
    st = _call(nb, args_b, len(args_b), _out, _out_len, _n_ref)
    if st < 3:
        # only the bytes written: `_out.raw` copies the whole 1 MB buffer, which was ~310 us of every call in the first speed runs
        return ctypes.string_at(_out, _n.value)
    return f'device:{STATUS.get(st, st)}'.encode()


def last_device_us():
    return _load().mip_last_device_us()


def close():
    global _lib
    if _lib is not None:
        _lib.mip_close()
        _lib = None
