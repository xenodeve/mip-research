/* mipmath.h: the C ABI of the CUDAMathBackend (docs/plans/2026-11-11-cuda-mathbackend-design.md).
 *
 * A call names an operation ("numbers.add") and passes its arguments as the canonical bytes of tools/igc/mathbackend/canon.py (a JSON
 * list of tagged values, numbers as decimal digits). The answer is canonical bytes too; equality with Python is a byte comparison.
 *
 * The work runs on the device in a resident worker kernel that polls a request slot in mapped host memory; each launch of the worker
 * ends itself after about MIP_WORKER_SECONDS and the host launches the next before the following call (Windows' TDR watchdog kills a
 * kernel after 2 s under WDDM). The host parses no value and does no arithmetic: it copies bytes in and out.
 */
#ifndef MIPMATH_H
#define MIPMATH_H
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define MIP_API __declspec(dllexport)
#else
#define MIP_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum mip_status {
    MIP_OK = 0,              /* out holds the value's canonical bytes */
    MIP_NOT_COVERED = 1,     /* out holds ["x","NotCovered"] */
    MIP_BAD_INPUT = 2,       /* out holds ["x","BadInput"] */
    MIP_DEVICE_FAILED = 3,   /* no answer: device memory, a fault, a step cap, an output past MIP_OUT_MAX; never a value */
    MIP_UNKNOWN_OP = 4,      /* the operation is not ported (yet): the caller has no device answer */
    MIP_HOST_ERROR = 5       /* mip_open not called, a CUDA error on the host, arguments past MIP_ARGS_MAX */
};

#define MIP_ARGS_MAX (1u << 20)   /* bytes of canonical arguments per call */
#define MIP_OUT_MAX (1u << 20)    /* bytes of canonical output per call */

/* device: CUDA device index; resident: 1 for the resident worker, 0 to launch one kernel per call (the comparison mode) */
MIP_API int mip_open(int device, int resident);
MIP_API void mip_close(void);
MIP_API int mip_call(const char *op, const uint8_t *args, size_t args_len, uint8_t *out, size_t out_cap, size_t *out_len);
/* the operations this build ports, one per line, NUL-terminated */
MIP_API const char *mip_ops(void);
/* launches of the worker so far (resident mode), to check the relaunch path ran */
MIP_API uint64_t mip_launches(void);

#ifdef __cplusplus
}
#endif
#endif
