// mipmath.cu: the CUDAMathBackend, first slice: numbers.add / sub / mul / div on exact rationals
// (docs/plans/2026-11-11-cuda-mathbackend-design.md). Byte identity with tools/igc/mathbackend (Python 3.11.9) is the gate:
// tools/igc/mathbackend/harness.py check <corpus> --backend cuda.
//
// Everything below runs on the device, in one thread per call: the JSON of the arguments, the strings Python's Fraction() accepts
// (Unicode digits and whitespace from unicode_tables.h, made by gen_tables.py), the big integers (32-bit limbs), the gcd, and the
// decimal digits of the answer. The host copies bytes and launches; it reads no value.
#include "mipmath.h"
#include "unicode_tables.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>

// ---- limits --------------------------------------------------------------------------------------------------------------------------
// MIP_LIMBS caps one integer at 8192 limbs (262,144 bits). Python's add/mul/... are "any size": a bound in bits both sides keep is open
// question 3 of the design; past this cap the device answers NOT_COVERED, which the corpus check shows as a disagreement until Python
// refuses at the same bound.
#define MIP_LIMBS 8192
#define ARENA_WORDS (MIP_LIMBS * 64)
#define SH_WORDS 11008   // the shared-memory arena of the worker: 43 KB; with the team's 4 KB of segment maps, under the 48 KB a
                         // block gets without opting in
#define MIP_STR_MAX 4000          // numbers._rat: a string longer than 4000 characters (after strip) is BadInput
#define MIP_WORKER_SECONDS 1.0

enum { OP_ADD = 1, OP_SUB, OP_MUL, OP_DIV };

struct Slot {
    volatile int32_t state;       // 0 empty, 1 request, 2 answered, 3 quit
    volatile uint32_t oplen;      // beside state in one 8-byte word: op in the low 8 bits, the argument length above (mode 2)
    volatile int32_t seen;        // diagnostics: the worker sets it the moment it reads a request
    volatile int32_t lingering;   // mode 2 with MIP_LINGER_US: 1 while a kernel waits for the next request after answering
    volatile uint32_t seq;        // the request's number (mode 2): a kernel launched for an older request does not use its params
    int32_t op;
    int32_t status;
    uint32_t args_len;
    uint32_t out_len;
    uint64_t dev_cycles;   // clock cycles serve() took on the device: the work, apart from the round trip
    uint64_t polls;        // diagnostics: polls of the flag since this worker launched, and the cycles they spanned
    uint64_t poll_cycles;
    uint64_t wait_cycles;  // cycles from the previous answer to seeing this request (the device side of the gap)
    uint32_t path;         // the paths the last call took (g_path bits)
    alignas(32) uint8_t args[MIP_ARGS_MAX];   // one 32-byte sector holds a short request: one PCIe read for it
    uint8_t out[MIP_OUT_MAX];
};


// ---- the time guard and the path record ----------------------------------------------------------------------------------------------
// A kernel that runs past ~2 s is killed by Windows' TDR watchdog and the driver resets for every program on the machine (a game on the
// other card included). The long loops (schoolbook multiply, long division, gcd, digit conversions) check a deadline set when a call
// starts; past it the call ends as DEVICE_FAILED, never as a value. It is a clock, so it is not the input-defined step cap the design
// asks for (open question 3); it only keeps the machine safe until that cap exists. One block serves a call at a time, so the
// state is plain device globals: thread 0 sets the deadline and owns g_path; the team's workers only read the deadline and may set
// g_timed_out (always to 1).
#define MIP_CALL_SECONDS 0.4
__device__ long long g_deadline = 0x7fffffffffffffffLL;
__device__ long long g_call_cycles = 0x7fffffffffffffffLL;   // set at open: MIP_CALL_SECONDS of the rated clock
__device__ int g_timed_out = 0;
__device__ unsigned g_path = 0;   // bits of the paths this call took: 1 register, 2 full, 4 Knuth D, 8 its add-back, 16 timed out,
                                  // 32 a Lehmer step, 64 a full Euclid step inside Lehmer, 128 a team multiply, 256 a carry
                                  // that crossed a team segment boundary in the fixup, 512 a team conversion (digits to limbs
                                  // or limbs to decimal), 1024 a Lehmer step applied by the team

__device__ __forceinline__ int past_deadline() {
    if (clock64() > g_deadline) {
        g_timed_out = 1;   // any thread may set it; serve_req (thread 0) turns it into path bit 16
        return 1;
    }
    return 0;
}

// ---- the team: a block of threads for the long loops of the full path ---------------------------------------------------------------
// Every kernel runs as one block of MIP_TEAM threads (1: the single thread of before; otherwise at least 64). Warp 0 is the master:
// thread 0 runs everything the single thread ran (the polling, the register path, the parsing, every decision) and its 31 lanes only
// keep it company at the dispatch barrier. Warps 1 and up are the workers: they wait at the dispatch barrier until thread 0 hands them a
// job (a long loop split by limbs) through `tm`, do it, and meet thread 0 at the barrier again; when thread 0 is done with the kernel it
// hands them QUIT. Thread 0 never runs job code: dispatching is two barriers, so no call sits inside its inlined serial code (a call
// there made the compiler save registers around it and spill the whole path, 2026-11-12). A job's own barriers are `barrier.sync 2` over
// the workers alone (a whole number of warps, as that form needs); the dispatch barrier is the non-aligned `barrier.sync 1` over the
// block (thread 0 reaches it from a different place than its warp-mates; on sm_70 and later a barrier is counted per thread). Rules that
// keep it from hanging: thread 0 leaves every kernel through one exit that sends QUIT; a job's barriers are the same count on every worker
// whatever happens inside it (the time guard shortens loops, it never skips a barrier); a job carries values and arena pointers only,
// never a reference to thread 0's locals (they live in its own local memory). Only thread 0 writes g_path.
enum { J_QUIT = 0, J_MUL, J_TODEC, J_FROMDEC, J_TODEC_REG, J_FROMDEC_REG, J_LIN };
#define TEN9 1000000000u
__device__ unsigned long long g_tcarry[2 * 1024];   // a job's carry out of each thread's segment (two words each)
__device__ int g_tk[1025];                          // a job's segment starts
__device__ long long g_team_work = 2048;            // MIP_TEAM_WORK: the least work (limb steps) worth a thread; 1 sends every case to the team
__shared__ int sh_n;                                // a pass job's current length, set by the thread that owns the top segment
__shared__ unsigned long long sh_ms[2 * 256];       // a pass job's segment maps: S mod d and W^len mod d, then the incoming remainders
__shared__ unsigned char sh_nz[256];                // a register pass job: segment t is not zero
__shared__ int sh_stop;                             // a register pass job: the number is zero (or the time guard ran out)
#define PASS_SEGS 256

// the least limbs in one thread's segment of a pass job (a pass is a few barriers: a segment shorter than this is not worth one)
__device__ __forceinline__ int team_seg() { long long g = g_team_work / 128; return g < 4 ? 4 : g > 64 ? 64 : (int)g; }   // >= 4: a pass by 2^32 can take two chunks off the top

// the workers of this block: every warp but the master's (0 without a team)
__device__ __forceinline__ int team_workers() { return blockDim.x >= 64 ? (int)blockDim.x - 32 : 0; }

// workers for a job of `work` limb steps: none past the team, none with less than g_team_work (0 or 1: thread 0 does it alone)
__device__ __forceinline__ int team_threads(long long work) {
    long long p = work / g_team_work, w = team_workers();
    if (p > w) p = w;
    return (int)p;
}
struct Team {
    int job;
    int P;                // workers taking part in this job (<= team_workers())
    long long a0, a1, a2, a3, a4, a5, a6, a7, d;   // the job's arguments: arena pointers as integers, lengths, divisors, cofactors
};
__shared__ Team tm;

__device__ __forceinline__ void team_bar() { asm volatile("barrier.sync 1;" ::: "memory"); }
__device__ __forceinline__ void team_wbar() { asm volatile("barrier.sync 2, %0;" :: "r"((int)blockDim.x - 32) : "memory"); }

__device__ __noinline__ void job_do(int job, int w);

// thread 0: run `job` on P workers (tm's arguments already set; 2 <= P <= team_workers()) and return when they are done
__device__ __forceinline__ void team_run(int job, int P) {
    tm.P = P;
    tm.job = job;
    team_bar();   // the workers start
    team_bar();   // the workers are done
}

// every thread but 0: wait for jobs until QUIT; the workers (warps 1 and up) do them, warp 0's other lanes only keep the count
__device__ __noinline__ void team_serve() {
    for (;;) {
        team_bar();
        int j = tm.job;
        if (j == J_QUIT) return;
        if (threadIdx.x >= 32) job_do(j, threadIdx.x - 32);
        team_bar();
    }
}

__device__ __forceinline__ void team_quit() {
    if (blockDim.x < 64) return;
    tm.job = J_QUIT;
    team_bar();
}

// Schoolbook multiply by columns: column k of r = a * b is the sum of a[i] * b[k - i]. The columns are split into tm.P ranges of equal
// work (column k has min(k + 1, n, m, n + m - 1 - k) products); each thread sums its columns into a 128-bit accumulator, keeps the low
// limb and carries the rest to its next column, and leaves its last carry in g_tcarry for thread 0's fixup (mul_fixup).
// C(k): the products in columns 0 .. k - 1.
__device__ __forceinline__ unsigned long long mul_work_before(long long k, long long n, long long m) {
    unsigned long long A = k <= n ? (unsigned long long)(k * (k + 1) / 2) : (unsigned long long)(n * (n + 1) / 2 + (k - n) * n);
    unsigned long long Bm = k <= m ? 0ull : (unsigned long long)((k - m) * (k - m + 1) / 2);
    return A - Bm;
}

// the first column whose products before it reach `target`
__device__ __forceinline__ int mul_col_start(unsigned long long target, long long n, long long m) {
    long long lo = 0, hi = n + m - 1;
    while (lo < hi) {
        long long mid = (lo + hi) / 2;
        if (mul_work_before(mid, n, m) >= target) hi = mid; else lo = mid + 1;
    }
    return (int)lo;
}

__device__ __noinline__ void job_mul(int t) {
    uint32_t *r = (uint32_t *)tm.a0;
    const uint32_t *a = (const uint32_t *)tm.a1, *b = (const uint32_t *)tm.a2;
    long long n = tm.a3, m = tm.a4;
    int P = tm.P, K = (int)(n + m - 1);
    unsigned long long W = (unsigned long long)n * (unsigned long long)m;
    int k0 = t == 0 ? 0 : mul_col_start(W * (unsigned long long)t / (unsigned long long)P, n, m);
    int k1 = t + 1 == P ? K : mul_col_start(W * (unsigned long long)(t + 1) / (unsigned long long)P, n, m);
    unsigned long long lo = 0, hi = 0;   // the running 128-bit accumulator
    for (int k = k0; k < k1; k++) {
        if (((k - k0) & 63) == 63 && past_deadline()) break;   // the barrier after the job still comes
        int i0 = k - (int)m + 1 > 0 ? k - (int)m + 1 : 0, i1 = k < n - 1 ? k : (int)n - 1;
        for (int i = i0; i <= i1; i++) {
            unsigned long long pr = (unsigned long long)a[i] * b[k - i];
            lo += pr;
            hi += lo < pr;
        }
        r[k] = (uint32_t)lo;
        lo = (lo >> 32) | (hi << 32);
        hi >>= 32;
    }
    g_tk[t] = k0;
    g_tcarry[2 * t] = lo;
    g_tcarry[2 * t + 1] = hi;
}

// ---- conversions between limbs and decimal, by passes split across the team ----------------------------------------------------------
// A pass divides the whole number by a small divisor d: limbs (base 2^32) by 10^9 to print (TODEC), base-10^9 chunks by 2^32 to read.
// Each thread owns a segment; the remainder a segment passes down is (r_in * W^len + S) mod d, an affine map of the remainder it gets
// from above. Phase A: each thread loads its segment (into registers when it has at most PASS_REG words: the loads go out together
// instead of one dependent trip each) and leaves S mod d and W^len mod d in shared memory. Phase B: worker 0 runs the maps from the top,
// hands each segment its incoming remainder and writes the bottom one out (one output digit). Phase C: each thread divides its own
// segment and stores it; the thread of the top segment sets the next length. Three barriers a pass, whatever happens in it: every thread
// decides the segments from the same length (sh_n), so every thread meets every barrier, and the time guard ends the job by setting that
// length to 0 at a barrier, never between them. When the number gets too short for two segments the job ends and thread 0 finishes
// alone. (Measured 2026-11-12: a shuffle scan of the maps by warp 0 was 3-6x slower than thread 0's walk over shared memory.)
#define PASS_REG 32

__device__ __forceinline__ uint32_t pow_mod_1e9(uint32_t b, int e) {   // b^e mod 10^9, b < 10^9
    unsigned long long r = 1, x = b;
    while (e) { if (e & 1) r = r * x % TEN9; x = x * x % TEN9; e >>= 1; }
    return (uint32_t)r;
}
__device__ __forceinline__ uint32_t pow_u32(uint32_t b, int e) {   // b^e mod 2^32
    uint32_t r = 1;
    while (e) { if (e & 1) r *= b; b *= b; e >>= 1; }
    return r;
}

// the register job's segment length for n words, or 0 when n words do not fit in the block's registers (the memory job then)
__device__ __forceinline__ int pass_reg_len(int n) {
    int T = team_workers() < PASS_SEGS ? team_workers() : PASS_SEGS;
    int L = g_team_work == 1 ? 2 : (int)ceilf(sqrtf(0.45f * (float)n));
    if (L < 4 && g_team_work != 1) L = 4;
    if (L > PASS_REG) L = PASS_REG;
    if ((n + L - 1) / L > T) L = (n + T - 1) / T;   // more segments than threads: longer segments
    return L <= PASS_REG ? L : 0;
}

// one word into the running remainder (S mod d by Horner, from the top word down)
template <int TODEC>
__device__ __forceinline__ unsigned long long pass_mod(unsigned long long m, uint32_t w) {
    return TODEC ? ((m << 32) | w) % TEN9 : (uint32_t)(m * TEN9 + w);
}
// one word divided, with the remainder r carried down
template <int TODEC>
__device__ __forceinline__ uint32_t pass_div(unsigned long long &r, uint32_t w) {
    if (TODEC) {
        unsigned long long cur = (r << 32) | w;
        r = cur % TEN9;
        return (uint32_t)(cur / TEN9);
    }
    unsigned long long cur = r * TEN9 + w;   // r < 2^32: below 2^62
    r = cur & 0xFFFFFFFFull;
    return (uint32_t)(cur >> 32);
}

// a[0 .. n) (consumed) divided by d over and over, the remainders to out[0 ..), least significant first; tm: a0 a, a1 out, a2 n.
// Leaves in tm.a3 the digits written and in tm.a2 the words left (thread 0 finishes them alone).
template <int TODEC>
__device__ __noinline__ void job_pass(int t) {
    uint32_t *a = (uint32_t *)tm.a0, *out = (uint32_t *)tm.a1;
    int nout = 0, seg = team_seg();
    if (t == 0) sh_n = (int)tm.a2;
    team_wbar();
    for (int pass = 0;; pass++) {
        int n = sh_n;
        int P = n / seg;
        if (P > tm.P) P = tm.P;
        if (P > PASS_SEGS) P = PASS_SEGS;
        if (P < 2) break;   // every thread saw the same n: they all leave here
        int lo = (int)((long long)t * n / P), hi = (int)((long long)(t + 1) * n / P), len = hi - lo;
        uint32_t buf[PASS_REG];
        int reg = len <= PASS_REG;
        if (t < P) {   // A
            unsigned long long m = 0;
            if (reg) {
#pragma unroll
                for (int k = 0; k < PASS_REG; k++)
                    if (k < len) buf[k] = a[lo + k];
#pragma unroll
                for (int k = PASS_REG - 1; k >= 0; k--)
                    if (k < len) m = pass_mod<TODEC>(m, buf[k]);
            } else {
                for (int i = hi - 1; i >= lo; i--) m = pass_mod<TODEC>(m, a[i]);
            }
            sh_ms[2 * t] = m;
            sh_ms[2 * t + 1] = TODEC ? pow_mod_1e9(294967296u, len) : pow_u32(TEN9, len);   // 2^32 mod 10^9 = 294967296
        }
        team_wbar();
        if (t == 0) {   // B: the remainders from the top
            unsigned long long r = 0;
            for (int q = P - 1; q >= 0; q--) {
                unsigned long long m = sh_ms[2 * q], pw = sh_ms[2 * q + 1];
                sh_ms[2 * q] = r;
                r = TODEC ? (r * pw + m) % TEN9 : (uint32_t)(r * pw + m);
            }
            out[nout++] = (uint32_t)r;
            if ((pass & 63) == 63) past_deadline();
        }
        team_wbar();
        if (t < P) {   // C
            unsigned long long r = sh_ms[2 * t];
            if (reg) {
#pragma unroll
                for (int k = PASS_REG - 1; k >= 0; k--)
                    if (k < len) buf[k] = pass_div<TODEC>(r, buf[k]);
#pragma unroll
                for (int k = 0; k < PASS_REG; k++)
                    if (k < len) a[lo + k] = buf[k];
            } else {
                for (int i = hi - 1; i >= lo; i--) a[i] = pass_div<TODEC>(r, a[i]);
            }
            if (t == P - 1) {
                int k = n;
                while (k > lo && a[k - 1] == 0) k--;
                sh_n = g_timed_out ? 0 : k;
            }
        }
        team_wbar();
    }
    if (t == 0) { tm.a3 = nout; tm.a2 = g_timed_out ? 0 : sh_n; }
}

// The same passes with every segment held in its thread's registers for the whole job (at most PASS_REG words a segment, fixed
// boundaries): a pass is thread 0's walk over the maps (B) and each thread dividing its registers and computing its next S mod d in the
// same loop (C); two barriers and no memory traffic but the maps. A segment whose top words have run out just holds zeros. Worker 0 stops
// when every segment is zero, so nothing is left to finish alone. Segments of about sqrt(0.45 n) words balance B (one step a segment)
// against C (two steps a word); MIP_TEAM_WORK=1 forces segments of 2 words so the tests cross many boundaries.
template <int TODEC>
__device__ __noinline__ void job_pass_reg(int t) {
    uint32_t *a = (uint32_t *)tm.a0, *out = (uint32_t *)tm.a1;
    int n = (int)tm.a2, L = (int)tm.a4, P = (n + L - 1) / L;
    uint32_t buf[PASS_REG];
    int lo = t * L, len = t < P ? (n - lo < L ? n - lo : L) : 0;
    if (t < P) {
        unsigned long long m = 0;
        uint32_t nz = 0;
#pragma unroll
        for (int k = 0; k < PASS_REG; k++)
            if (k < len) { buf[k] = a[lo + k]; nz |= buf[k]; }
#pragma unroll
        for (int k = PASS_REG - 1; k >= 0; k--)
            if (k < len) m = pass_mod<TODEC>(m, buf[k]);
        sh_ms[2 * t] = m;
        sh_ms[2 * t + 1] = TODEC ? pow_mod_1e9(294967296u, len) : pow_u32(TEN9, len);
        sh_nz[t] = nz != 0;
    }
    int nout = 0, top = P - 1;
    team_wbar();
    for (int pass = 0;; pass++) {
        if (t < 32) {   // B, by worker warp 0: the top segment that is not zero (it only moves down), then the remainders from the top
            const unsigned full = 0xFFFFFFFFu;
            while (top >= 0 && !sh_nz[top]) top--;
            int stop = top < 0 || ((pass & 63) == 63 && t == 0 && past_deadline());
            stop = __shfl_sync(full, stop, 0);
            if (!stop) {
                int S = top + 1;   // segments from S - 1 down to 0
                if (S <= 32) {   // few: worker 0 alone
                    if (t == 0) {
                        unsigned long long r = 0;
                        for (int q = top; q >= 0; q--) {
                            unsigned long long m = sh_ms[2 * q], pw = sh_ms[2 * q + 1];
                            sh_ms[2 * q] = r;
                            r = TODEC ? (r * pw + m) % TEN9 : (uint32_t)(r * pw + m);
                        }
                        out[nout] = (uint32_t)r;
                    }
                } else {   // many: each lane composes its share of the maps, a shuffle scan composes the shares
                    int c = (S + 31) / 32, u0 = t * c, u1 = u0 + c < S ? u0 + c : S;   // u counts from the top: segment top - u
                    unsigned long long cp = 1, cm = 0;
                    for (int u = u0; u < u1; u++) {
                        int q = top - u;
                        unsigned long long m = sh_ms[2 * q], pw = sh_ms[2 * q + 1];
                        cm = TODEC ? (cm * pw + m) % TEN9 : (uint32_t)(cm * pw + m);
                        cp = TODEC ? cp * pw % TEN9 : (uint32_t)(cp * pw);
                    }
                    for (int off = 1; off < 32; off <<= 1) {   // earlier lanes (higher segments) first: (A then B) = (A.p B.p, A.m B.p + B.m)
                        unsigned long long op = __shfl_up_sync(full, cp, off), om = __shfl_up_sync(full, cm, off);
                        if (t >= off) {
                            cm = TODEC ? (om * cp + cm) % TEN9 : (uint32_t)(om * cp + cm);
                            cp = TODEC ? op * cp % TEN9 : (uint32_t)(op * cp);
                        }
                    }
                    unsigned long long r = __shfl_up_sync(full, cm, 1);
                    if (t == 0) r = 0;
                    for (int u = u0; u < u1; u++) {
                        int q = top - u;
                        unsigned long long m = sh_ms[2 * q], pw = sh_ms[2 * q + 1];
                        sh_ms[2 * q] = r;
                        r = TODEC ? (r * pw + m) % TEN9 : (uint32_t)(r * pw + m);
                        if (q == 0) out[nout] = (uint32_t)r;
                    }
                }
                nout++;
            }
            if (t == 0) sh_stop = stop;
        }
        team_wbar();
        if (sh_stop) break;   // every thread reads it after the same barrier
        if (t < P) {   // C: divide, then the next S mod d of the quotient
            unsigned long long r = sh_ms[2 * t], m = 0;
            uint32_t nz = 0;
#pragma unroll
            for (int k = PASS_REG - 1; k >= 0; k--)
                if (k < len) { buf[k] = pass_div<TODEC>(r, buf[k]); nz |= buf[k]; }
#pragma unroll
            for (int k = PASS_REG - 1; k >= 0; k--)
                if (k < len) m = pass_mod<TODEC>(m, buf[k]);
            sh_ms[2 * t] = m;
            sh_nz[t] = nz != 0;
        }
        team_wbar();
    }
    if (t == 0) { tm.a3 = nout; tm.a2 = 0; }
}

// Lehmer's step by the team: t1 = A x + B y and t2 = C x + D y (|A|, |B|, |C|, |D| < 2^29, |x| >= |y|), each worker over its own
// limbs of x with its own signed carries; the carries out of each segment go to g_tcarry and thread 0 ripples them (lin_fixup).
// tm: a0 t1, a1 t2, a2 x, a3 y, a4 x.n | y.n << 32, a5 A, a6 B, a7 C; D in tm.d.
__device__ __noinline__ void job_lin(int t) {
    uint32_t *t1 = (uint32_t *)tm.a0, *t2 = (uint32_t *)tm.a1;
    const uint32_t *x = (const uint32_t *)tm.a2, *y = (const uint32_t *)tm.a3;
    int n = (int)(tm.a4 & 0xFFFFFFFF), yn = (int)(tm.a4 >> 32), P = tm.P;
    long long A = tm.a5, B = tm.a6, C = tm.a7, D = tm.d;
    int lo = (int)((long long)t * n / P), hi = (int)((long long)(t + 1) * n / P);
    long long c1 = 0, c2 = 0;
    for (int i = lo; i < hi; i++) {
        long long xi = x[i], yi = i < yn ? y[i] : 0;
        long long v1 = A * xi + B * yi + c1, v2 = C * xi + D * yi + c2;   // |v| < 2^62
        t1[i] = (uint32_t)v1;
        t2[i] = (uint32_t)v2;
        c1 = v1 >> 32;
        c2 = v2 >> 32;
    }
    g_tk[t] = lo;
    g_tcarry[2 * t] = (unsigned long long)c1;
    g_tcarry[2 * t + 1] = (unsigned long long)c2;
}

__device__ __noinline__ void job_do(int job, int t) {   // t: the worker's index
    if (job == J_TODEC) { job_pass<1>(t); return; }       // every thread of the block: the passes are counted together
    if (job == J_FROMDEC) { job_pass<0>(t); return; }
    if (job == J_TODEC_REG) { job_pass_reg<1>(t); return; }
    if (job == J_FROMDEC_REG) { job_pass_reg<0>(t); return; }
    if (t >= tm.P) return;
    if (job == J_MUL) job_mul(t);
    else if (job == J_LIN) job_lin(t);
}

// ---- big integers (sign and magnitude, little-endian 32-bit limbs, no leading zero limbs; zero has n == 0) ------------------------------
struct Big {
    uint32_t *w;
    int n;
    int neg;
    int cap;
};

struct Ctx {
    uint32_t *arena;
    int cap;    // words in this arena (shared or global)
    int used;
    int fail;   // a device failure: arena exhausted (never a value)
};

__device__ Big big_new(Ctx &c, int cap) {
    Big b;
    b.n = 0;
    b.neg = 0;
    b.cap = cap;
    if (c.used + cap > c.cap) {
        c.fail = 1;
        b.w = c.arena;
        b.cap = 0;
        return b;
    }
    b.w = c.arena + c.used;
    c.used += cap;
    return b;
}

__device__ void big_trim(Big &b) {
    while (b.n > 0 && b.w[b.n - 1] == 0) b.n--;
    if (b.n == 0) b.neg = 0;
}

__device__ int mag_cmp(const Big &a, const Big &b) {
    if (a.n != b.n) return a.n < b.n ? -1 : 1;
    for (int i = a.n - 1; i >= 0; i--)
        if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1;
    return 0;
}

// r = |a| + |b|
__device__ void mag_add(Big &r, const Big &a, const Big &b) {
    int n = a.n > b.n ? a.n : b.n;
    uint64_t carry = 0;
    for (int i = 0; i < n; i++) {
        uint64_t s = carry + (i < a.n ? a.w[i] : 0) + (i < b.n ? b.w[i] : 0);
        r.w[i] = (uint32_t)s;
        carry = s >> 32;
    }
    r.n = n;
    if (carry) r.w[r.n++] = (uint32_t)carry;
    big_trim(r);
}

// r = |a| - |b|, |a| >= |b|
__device__ void mag_sub(Big &r, const Big &a, const Big &b) {
    int64_t borrow = 0;
    for (int i = 0; i < a.n; i++) {
        int64_t s = (int64_t)a.w[i] - (i < b.n ? b.w[i] : 0) - borrow;
        borrow = s < 0;
        r.w[i] = (uint32_t)(s + (borrow ? ((int64_t)1 << 32) : 0));
    }
    r.n = a.n;
    big_trim(r);
}

// r = a + b (signed); r has room for max(n)+1 limbs
__device__ void big_add(Big &r, const Big &a, const Big &b) {
    if (a.neg == b.neg) {
        mag_add(r, a, b);
        r.neg = r.n ? a.neg : 0;
    } else if (mag_cmp(a, b) >= 0) {
        mag_sub(r, a, b);
        r.neg = r.n ? a.neg : 0;
    } else {
        mag_sub(r, b, a);
        r.neg = r.n ? b.neg : 0;
    }
}

// r = a * b (r distinct from a and b; room for a.n + b.n limbs)
__device__ void big_mul(Big &r, const Big &a, const Big &b) {
    if (a.n == 0 || b.n == 0) {
        r.n = 0;
        r.neg = 0;
        return;
    }
    int P = team_threads((long long)a.n * b.n);
    if (P > 1) {
        tm.a0 = (long long)r.w; tm.a1 = (long long)a.w; tm.a2 = (long long)b.w; tm.a3 = a.n; tm.a4 = b.n;
        team_run(J_MUL, P);
        g_path |= 128;
        // the fixup: each segment's carry goes into the next segment's first columns, rippling as far as it must
        int K = a.n + b.n - 1;
        unsigned long long lo = 0, hi = 0;
        for (int t = 0; t < P; t++) {
            int k = g_tk[t], kend = t + 1 < P ? g_tk[t + 1] : K;
            while ((lo | hi) && k < kend) {
                unsigned long long x = lo + r.w[k];
                hi += x < lo;
                r.w[k] = (uint32_t)x;
                lo = (x >> 32) | (hi << 32);
                hi >>= 32;
                if (k > g_tk[t]) g_path |= 256;   // the carry went past the segment's first column
                k++;
            }
            unsigned long long cl = g_tcarry[2 * t], ch = g_tcarry[2 * t + 1];
            lo += cl;
            hi += ch + (lo < cl);
        }
        // the carry past the last column: the product's top limb (a * b < 2^(32 (n + m)), so it fits)
        r.w[K] = (uint32_t)lo;
        r.n = a.n + b.n;
        r.neg = a.neg != b.neg;
        big_trim(r);
        return;
    }
    for (int i = 0; i < a.n + b.n; i++) r.w[i] = 0;
    for (int i = 0; i < a.n; i++) {
        if ((i & 63) == 0 && past_deadline()) { r.n = 0; return; }
        uint64_t carry = 0, ai = a.w[i];
        for (int j = 0; j < b.n; j++) {
            uint64_t t = ai * b.w[j] + r.w[i + j] + carry;
            r.w[i + j] = (uint32_t)t;
            carry = t >> 32;
        }
        r.w[i + b.n] = (uint32_t)carry;
    }
    r.n = a.n + b.n;
    r.neg = a.neg != b.neg;
    big_trim(r);
}

// a = a * m + add, in place (room for one more limb)
__device__ void mag_muladd_small(Big &a, uint32_t m, uint32_t add) {
    uint64_t carry = add;
    for (int i = 0; i < a.n; i++) {
        uint64_t t = (uint64_t)a.w[i] * m + carry;
        a.w[i] = (uint32_t)t;
        carry = t >> 32;
    }
    if (carry) a.w[a.n++] = (uint32_t)carry;
}

// a = a / d in place; returns the remainder
__device__ uint32_t mag_divmod_small(Big &a, uint32_t d) {
    uint64_t rem = 0;
    for (int i = a.n - 1; i >= 0; i--) {
        uint64_t cur = (rem << 32) | a.w[i];
        a.w[i] = (uint32_t)(cur / d);
        rem = cur % d;
    }
    big_trim(a);
    return (uint32_t)rem;
}

__device__ int clz32(uint32_t x) { return __clz(x); }

// q = |u| / |v|, exact or floor of magnitudes (Knuth D); q has room for u.n - v.n + 1 limbs; v nonzero
__device__ void mag_div(Ctx &c, Big &q, const Big &u, const Big &v) {
    if (mag_cmp(u, v) < 0) {
        q.n = 0;
        q.neg = 0;
        return;
    }
    if (v.n == 1) {
        for (int i = 0; i < u.n; i++) q.w[i] = u.w[i];
        q.n = u.n;
        mag_divmod_small(q, v.w[0]);
        return;
    }
    int s = clz32(v.w[v.n - 1]);
    Big vn = big_new(c, v.n), un = big_new(c, u.n + 1);
    if (c.fail) return;
    for (int i = v.n - 1; i > 0; i--) vn.w[i] = (v.w[i] << s) | (s ? (uint32_t)((uint64_t)v.w[i - 1] >> (32 - s)) : 0);
    vn.w[0] = v.w[0] << s;
    un.w[u.n] = s ? (uint32_t)((uint64_t)u.w[u.n - 1] >> (32 - s)) : 0;
    for (int i = u.n - 1; i > 0; i--) un.w[i] = (u.w[i] << s) | (s ? (uint32_t)((uint64_t)u.w[i - 1] >> (32 - s)) : 0);
    un.w[0] = u.w[0] << s;
    int n = v.n, m = u.n - v.n;
    const uint64_t B = (uint64_t)1 << 32;
    g_path |= 4;
    for (int j = m; j >= 0; j--) {
        if ((j & 63) == 0 && past_deadline()) { c.fail = 1; return; }
        uint64_t num = ((uint64_t)un.w[j + n] << 32) | un.w[j + n - 1];
        uint64_t qhat = num / vn.w[n - 1], rhat = num % vn.w[n - 1];
        while (qhat >= B || qhat * vn.w[n - 2] > ((rhat << 32) | un.w[j + n - 2])) {
            qhat--;
            rhat += vn.w[n - 1];
            if (rhat >= B) break;
        }
        int64_t borrow = 0;
        uint64_t carry = 0;
        for (int i = 0; i < n; i++) {
            uint64_t p = qhat * vn.w[i] + carry;
            carry = p >> 32;
            int64_t t = (int64_t)un.w[i + j] - borrow - (int64_t)(uint32_t)p;
            un.w[i + j] = (uint32_t)t;
            borrow = t < 0 ? 1 : 0;
        }
        int64_t t = (int64_t)un.w[j + n] - borrow - (int64_t)carry;
        un.w[j + n] = (uint32_t)t;
        if (t < 0) {   // added back: qhat was one too large
            g_path |= 8;
            qhat--;
            uint64_t cc = 0;
            for (int i = 0; i < n; i++) {
                uint64_t s2 = (uint64_t)un.w[i + j] + vn.w[i] + cc;
                un.w[i + j] = (uint32_t)s2;
                cc = s2 >> 32;
            }
            un.w[j + n] += (uint32_t)cc;
        }
        q.w[j] = (uint32_t)qhat;
    }
    q.n = m + 1;
    q.neg = 0;
    big_trim(q);
}

__device__ void big_copy(Big &r, const Big &a) {
    for (int i = 0; i < a.n; i++) r.w[i] = a.w[i];
    r.n = a.n;
    r.neg = a.neg;
}

// shift a magnitude right by k bits in place
__device__ void mag_shr(Big &a, int k) {
    int limbs = k / 32, bits = k % 32;
    if (limbs >= a.n) {
        a.n = 0;
        return;
    }
    for (int i = 0; i < a.n - limbs; i++) {
        uint64_t lo = a.w[i + limbs], hi = i + limbs + 1 < a.n ? a.w[i + limbs + 1] : 0;
        a.w[i] = (uint32_t)(((hi << 32) | lo) >> bits);
    }
    a.n -= limbs;
    big_trim(a);
}

__device__ int mag_ctz(const Big &a) {
    int k = 0;
    for (int i = 0; i < a.n; i++) {
        if (a.w[i]) return k + __ffs(a.w[i]) - 1;
        k += 32;
    }
    return k;
}

__device__ void mag_shl(Big &a, int k) {
    int limbs = k / 32, bits = k % 32;
    if (a.n == 0) return;
    a.w[a.n + limbs] = 0;
    for (int i = a.n - 1; i >= 0; i--) {
        uint64_t v = (uint64_t)a.w[i] << bits;
        a.w[i + limbs + 1] |= (uint32_t)(v >> 32);
        a.w[i + limbs] = (uint32_t)v;
    }
    for (int i = 0; i < limbs; i++) a.w[i] = 0;
    a.n += limbs + 1;
    big_trim(a);
}

// r = |u| mod |v| (v nonzero; r distinct from u and v, room for u.n limbs); the quotient's scratch is given back to the arena
__device__ void mag_mod(Ctx &c, Big &r, const Big &u, const Big &v) {
    if (mag_cmp(u, v) < 0) { big_copy(r, u); r.neg = 0; return; }
    if (v.n == 1) {
        big_copy(r, u);
        r.neg = 0;
        uint32_t m = mag_divmod_small(r, v.w[0]);
        r.w[0] = m;
        r.n = m ? 1 : 0;
        return;
    }
    int mark = c.used;
    Big q = big_new(c, u.n - v.n + 2), p = big_new(c, u.n + 2);
    if (c.fail) return;
    mag_div(c, q, u, v);
    if (c.fail) return;
    big_mul(p, q, v);   // q * v <= u: at most u.n limbs
    mag_sub(r, u, p);
    c.used = mark;
}

// r = A x + B y with |A|, |B| < 2^29 (|x| >= |y|); returns 0 when the result is negative or has no room (never a value then)
__device__ int mag_lin(Big &r, const Big &x, long long A, const Big &y, long long B) {
    long long cy = 0;
    int n = x.n;
    for (int i = 0; i < n; i++) {
        long long v = A * (long long)x.w[i] + B * (long long)(i < y.n ? y.w[i] : 0u) + cy;   // |v| < 2^62
        r.w[i] = (uint32_t)v;
        cy = v >> 32;   // arithmetic: the floor
    }
    while (cy > 0) {
        if (n >= r.cap) return 0;
        r.w[n++] = (uint32_t)cy;
        cy >>= 32;
    }
    if (cy < 0) return 0;
    r.n = n;
    r.neg = 0;
    big_trim(r);
    return 1;
}

__device__ int mag_bits(const Big &a) { return a.n ? 32 * (a.n - 1) + 32 - __clz(a.w[a.n - 1]) : 0; }

// the bits [k, k + 64) of a magnitude, k >= 0
__device__ unsigned long long mag_bits_at(const Big &a, int k) {
    int l = k / 32, s = k % 32;
    unsigned long long lo = l < a.n ? a.w[l] : 0, mid = l + 1 < a.n ? a.w[l + 1] : 0, hi = l + 2 < a.n ? a.w[l + 2] : 0;
    unsigned long long low64 = (mid << 32) | lo;
    return s ? (low64 >> s) | (hi << (64 - s)) : low64;
}

// g = gcd(|a|, |b|): Lehmer's algorithm (Knuth 4.5.2, Algorithm L) on the top 29 bits, so each cofactor stays below 2^29 and a step
// costs one pass over the numbers instead of the binary gcd's pass per bit; a step that finds no cofactors is one full Euclid step
// (x mod y); once y fits in two limbs the rest is 64-bit Euclid. g has room for max(a.n, b.n) + 1 limbs.
__device__ void mag_gcd(Ctx &c, Big &g, const Big &a, const Big &b) {
    if (a.n == 0) { big_copy(g, b); g.neg = 0; return; }
    if (b.n == 0) { big_copy(g, a); g.neg = 0; return; }
    int cap = (a.n > b.n ? a.n : b.n) + 2;
    Big x = big_new(c, cap), y = big_new(c, cap), t1 = big_new(c, cap), t2 = big_new(c, cap);
    if (c.fail) return;
    if (mag_cmp(a, b) >= 0) { big_copy(x, a); big_copy(y, b); } else { big_copy(x, b); big_copy(y, a); }
    x.neg = y.neg = 0;
    int turns = 0;
    while (y.n > 2) {   // x >= y here
        if ((++turns & 15) == 0 && past_deadline()) { c.fail = 1; return; }
        int k = mag_bits(x) - 29;   // y.n > 2: x has more than 64 bits, k > 35
        long long uh = (long long)(mag_bits_at(x, k) & 0x1FFFFFFFull), vh = (long long)(mag_bits_at(y, k) & 0x1FFFFFFFull);
        // y <= x, so y >> k < 2^29 as well: the masks only drop bits that are zero
        long long A = 1, B = 0, C = 0, D = 1;
        for (;;) {
            if (vh + C == 0 || vh + D == 0) break;
            long long q = (uh + A) / (vh + C);
            if (q != (uh + B) / (vh + D)) break;
            long long T = A - q * C; A = C; C = T;
            T = B - q * D; B = D; D = T;
            T = uh - q * vh; uh = vh; vh = T;
        }
        if (B == 0) {   // no cofactors: one full step, (x, y) = (y, x mod y)
            g_path |= 64;
            mag_mod(c, t1, x, y);
            if (c.fail) return;
            Big t = x; x = y; y = t1; t1 = t;
        } else {
            g_path |= 32;
            int P = team_threads((long long)x.n * 32);
            if (P > x.n) P = x.n;
            if (P > 1) {   // the team applies the step; thread 0 ripples each segment's carries into the next
                tm.a0 = (long long)t1.w; tm.a1 = (long long)t2.w; tm.a2 = (long long)x.w; tm.a3 = (long long)y.w;
                tm.a4 = (long long)(unsigned)x.n | ((long long)y.n << 32);
                tm.a5 = A; tm.a6 = B; tm.a7 = C; tm.d = D;
                team_run(J_LIN, P);
                g_path |= 1024;
                for (int o = 0; o < 2; o++) {
                    uint32_t *w = o ? t2.w : t1.w;
                    long long cy = 0;
                    for (int s2 = 0; s2 < P; s2++) {
                        int k = g_tk[s2], kend = s2 + 1 < P ? g_tk[s2 + 1] : x.n;
                        while (cy && k < kend) {
                            long long v = (long long)w[k] + cy;
                            w[k] = (uint32_t)v;
                            cy = v >> 32;
                            k++;
                        }
                        cy += (long long)g_tcarry[2 * s2 + o];
                    }
                    Big &r = o ? t2 : t1;
                    int nn = x.n;
                    while (cy > 0) {   // never for a true Lehmer step (the result is below x); kept so a carry is never lost
                        if (nn >= r.cap) { c.fail = 1; return; }
                        r.w[nn++] = (uint32_t)cy;
                        cy >>= 32;
                    }
                    if (cy < 0) { c.fail = 1; return; }   // a negative result: never a value
                    r.n = nn;
                    r.neg = 0;
                    big_trim(r);
                }
            } else if (!mag_lin(t1, x, A, y, B) || !mag_lin(t2, x, C, y, D)) { c.fail = 1; return; }
            Big t = x; x = t1; t1 = t;
            t = y; y = t2; t2 = t;
            if (mag_cmp(x, y) < 0) { t = x; x = y; y = t; }
        }
    }
    unsigned long long u0, v0;
    if (y.n == 0) {
        big_copy(g, x);
        g.neg = 0;
        return;
    }
    mag_mod(c, t1, x, y);   // y fits in 64 bits: so does x mod y
    if (c.fail) return;
    v0 = y.w[0] | (y.n > 1 ? (unsigned long long)y.w[1] << 32 : 0);
    u0 = t1.n ? (t1.w[0] | (t1.n > 1 ? (unsigned long long)t1.w[1] << 32 : 0)) : 0;
    while (u0) { unsigned long long m = v0 % u0; v0 = u0; u0 = m; }
    g.w[0] = (uint32_t)v0;
    g.w[1] = (uint32_t)(v0 >> 32);
    g.n = 2;
    g.neg = 0;
    big_trim(g);
}

// ---- rationals: n / d, d > 0, reduced -------------------------------------------------------------------------------------------------
struct Rat {
    Big n, d;
};

__device__ void rat_reduce(Ctx &c, Rat &r, int cap) {
    if (r.n.n == 0) {   // zero is 0/1
        r.d.n = 1;
        r.d.w[0] = 1;
        r.d.neg = 0;
        return;
    }
    if (r.d.n == 1 && r.d.w[0] == 1) return;   // an integer: nothing to reduce (a binary gcd with 1 walked every bit of n)
    if (r.d.n == 1) {   // a one-limb denominator: gcd(n, d) = gcd(d, n mod d), one pass over n and a 32-bit Euclid
        Big t = big_new(c, r.n.n + 1);
        if (c.fail) return;
        big_copy(t, r.n);
        uint32_t d0 = r.d.w[0], x = d0, y = mag_divmod_small(t, d0);   // y = n mod d
        while (y) { uint32_t m = x % y; x = y; y = m; }
        uint32_t g0 = x;   // gcd(d, n mod d); d itself when d divides n
        if (g0 == 1) return;
        int neg = r.n.neg;
        mag_divmod_small(r.n, g0);
        r.n.neg = r.n.n ? neg : 0;
        r.d.w[0] /= g0;
        return;
    }
    Big g = big_new(c, cap);
    if (c.fail) return;
    mag_gcd(c, g, r.n, r.d);
    if (c.fail) return;
    if (g.n == 1 && g.w[0] == 1) return;
    Big qn = big_new(c, cap), qd = big_new(c, cap);
    if (c.fail) return;
    int neg = r.n.neg;
    mag_div(c, qn, r.n, g);
    mag_div(c, qd, r.d, g);
    qn.neg = qn.n ? neg : 0;
    r.n = qn;
    r.d = qd;
}

// ---- text: UTF-8, JSON, Python's Fraction(str) --------------------------------------------------------------------------------------
__device__ int utf8_next(const uint8_t *s, int len, int &i, uint32_t &cp) {
    if (i >= len) return 0;
    uint8_t b = s[i];
    int k = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : 4;
    if (i + k > len) return -1;
    cp = k == 1 ? b : k == 2 ? (b & 0x1F) : k == 3 ? (b & 0x0F) : (b & 0x07);
    for (int t = 1; t < k; t++) cp = (cp << 6) | (s[i + t] & 0x3F);
    i += k;
    return 1;
}

__device__ int nd_digit(uint32_t cp) {   // re's \d and int(): the digit's value, or -1
    if (cp < 0x80) return cp >= '0' && cp <= '9' ? (int)(cp - '0') : -1;   // ASCII first: no other run starts below 0x80
    for (int k = 0; k < MIP_ND_ZEROS; k++)
        if (cp >= mip_nd_zero[k] && cp < mip_nd_zero[k] + 10) return (int)(cp - mip_nd_zero[k]);
    return -1;
}

__device__ int is_space(uint32_t cp) {
    if (cp < 0x80) return cp == ' ' || (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x1F);   // str.isspace() on ASCII
    for (int k = 0; k < MIP_SPACES; k++)
        if (mip_space[k] == cp) return 1;
    return 0;
}

// Big from a run of digit values (0..9). A long run (from about 128 limbs: below it the passes' barriers cost more than they
// save, measured 2026-11-12) goes to base-10^9 chunks, then the team divides the chunks by 2^32 a pass at a time
// (each pass one limb, J_FROMDEC) and thread 0 finishes the last few chunks alone; a short run is read 9 digits at a time by thread 0.
__device__ void big_from_digits(Ctx &c, Big &r, const uint8_t *dig, int n) {
    r.n = 0;
    r.neg = 0;
    int k = (n + 8) / 9;
    if (team_workers() && k / team_seg() >= 2 && team_threads((long long)k * k / 4) > 1 && c.used + k + 1 <= c.cap) {
        uint32_t *d = c.arena + c.used;
        for (int j = 0; j < k; j++) {   // chunk j: the digits [n - 9 (j + 1), n - 9 j), least significant chunk first
            int end = n - 9 * j, beg = end - 9 < 0 ? 0 : end - 9;
            uint32_t v = 0;
            for (int i = beg; i < end; i++) v = v * 10 + dig[i];
            d[j] = v;
        }
        while (k && d[k - 1] == 0) k--;
        int nl = 0;
        if (k / team_seg() >= 2) {
            tm.a0 = (long long)d; tm.a1 = (long long)r.w; tm.a2 = k;
            int L = pass_reg_len(k);
            tm.a4 = L;
            team_run(L ? J_FROMDEC_REG : J_FROMDEC, team_workers());
            nl = (int)tm.a3;
            k = (int)tm.a2;
            g_path |= 512;
        }
        while (k) {   // the rest alone: the same passes, one thread
            if ((nl & 63) == 63 && past_deadline()) { r.n = 0; return; }
            unsigned long long rem = 0;
            for (int i = k - 1; i >= 0; i--) {
                unsigned long long cur = rem * TEN9 + d[i];
                d[i] = (uint32_t)(cur >> 32);
                rem = cur & 0xFFFFFFFFull;
            }
            if (nl >= r.cap) { c.fail = 1; r.n = 0; return; }
            r.w[nl++] = (uint32_t)rem;
            while (k && d[k - 1] == 0) k--;
        }
        r.n = nl;
        big_trim(r);
        return;
    }
    int i = 0, first = n % 9 ? n % 9 : 9;
    while (i < n) {
        if ((i & 511) == 0 && past_deadline()) { r.n = 0; return; }
        int take = i == 0 ? first : 9;
        uint32_t chunk = 0, mul = 1;
        for (int k2 = 0; k2 < take; k2++) {
            chunk = chunk * 10 + dig[i + k2];
            mul *= 10;
        }
        mag_muladd_small(r, mul, chunk);
        if (r.n == 0 && chunk) r.w[r.n++] = chunk;
        i += take;
    }
    big_trim(r);
}

// a JSON string starting at s[i] == '"'; its unescaped UTF-8 bytes go to dst; returns the length or -1
__device__ int json_string(const uint8_t *s, int len, int &i, uint8_t *dst, int cap) {
    if (i >= len || s[i] != '"') return -1;
    i++;
    int o = 0;
    while (i < len && s[i] != '"') {
        uint32_t cp;
        if (s[i] == '\\') {
            if (i + 1 >= len) return -1;
            uint8_t e = s[i + 1];
            i += 2;
            if (e == 'u') {
                if (i + 4 > len) return -1;
                cp = 0;
                for (int k = 0; k < 4; k++) {
                    uint8_t h = s[i + k];
                    cp = cp * 16 + (h <= '9' ? h - '0' : (h | 32) - 'a' + 10);
                }
                i += 4;
                if (cp >= 0xD800 && cp < 0xDC00 && i + 6 <= len && s[i] == '\\' && s[i + 1] == 'u') {
                    uint32_t lo = 0;
                    for (int k = 0; k < 4; k++) {
                        uint8_t h = s[i + 2 + k];
                        lo = lo * 16 + (h <= '9' ? h - '0' : (h | 32) - 'a' + 10);
                    }
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    i += 6;
                }
            } else {
                cp = e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e == 'b' ? '\b' : e == 'f' ? '\f' : e;
            }
            int need = cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
            if (o + need > cap) return -1;
            if (need == 1) dst[o++] = (uint8_t)cp;
            else if (need == 2) { dst[o++] = 0xC0 | (cp >> 6); dst[o++] = 0x80 | (cp & 0x3F); }
            else if (need == 3) { dst[o++] = 0xE0 | (cp >> 12); dst[o++] = 0x80 | ((cp >> 6) & 0x3F); dst[o++] = 0x80 | (cp & 0x3F); }
            else { dst[o++] = 0xF0 | (cp >> 18); dst[o++] = 0x80 | ((cp >> 12) & 0x3F); dst[o++] = 0x80 | ((cp >> 6) & 0x3F); dst[o++] = 0x80 | (cp & 0x3F); }
        } else {
            if (o >= cap) return -1;
            dst[o++] = s[i++];
        }
    }
    if (i >= len) return -1;
    i++;
    return o;
}

// Fraction(s) as numbers._rat reads a string (Python 3.11.9): strip; more than 4000 characters, or an e/E/n/N anywhere, is BadInput;
// then Fraction's grammar: [sign] (digits[_digits]* | empty) ( '/' digits[_digits]* | '.' (empty | digits[_digits]*) )?, the
// lookahead asking for a digit or '.digit' first. A zero denominator is BadInput.
// Returns MIP_OK or MIP_BAD_INPUT; r gets the unreduced value.
__device__ int rat_from_str(Ctx &c, Rat &r, const uint8_t *s, int len) {
    // decode, strip; the code points go to the arena (a 16 KB local array would overflow the device stack)
    if (c.used + MIP_STR_MAX + 1 > c.cap) { c.fail = 1; return MIP_DEVICE_FAILED; }
    uint32_t *cps = c.arena + c.used;
    int i = 0, n = 0, start = -1, end = -1;
    uint32_t cp;
    // first pass: count code points and find the stripped range (on code point indices)
    {
        int j = 0;
        int k = 0;
        while (utf8_next(s, len, j, cp) == 1) {
            if (!is_space(cp)) {
                if (start < 0) start = k;
                end = k;
            }
            k++;
        }
        n = k;
    }
    if (start < 0) return MIP_BAD_INPUT;   // empty after strip: Fraction('') fails
    int m = end - start + 1;
    if (m > MIP_STR_MAX) return MIP_BAD_INPUT;
    c.used += m + 1;
    {
        int j = 0, k = 0, o = 0;
        while (utf8_next(s, len, j, cp) == 1) {
            if (k >= start && k <= end) cps[o++] = cp;
            k++;
        }
    }
    (void)i;
    (void)n;
    for (int k = 0; k < m; k++)
        if (cps[k] == 'e' || cps[k] == 'E' || cps[k] == 'n' || cps[k] == 'N') return MIP_BAD_INPUT;
    int p = 0, neg = 0;
    if (p < m && (cps[p] == '-' || cps[p] == '+')) { neg = cps[p] == '-'; p++; }
    // lookahead: a digit, or '.' then a digit
    if (!(p < m && (nd_digit(cps[p]) >= 0 || (cps[p] == '.' && p + 1 < m && nd_digit(cps[p + 1]) >= 0)))) return MIP_BAD_INPUT;
    uint8_t *num = (uint8_t *)(c.arena + c.used);   // digit values, borrowed from the arena's free space
    int need = (3 * MIP_STR_MAX + 3) / 4 + 4;
    if (c.used + need > c.cap) { c.fail = 1; return MIP_DEVICE_FAILED; }
    c.used += need;
    uint8_t *den = num + MIP_STR_MAX, *dec = den + MIP_STR_MAX;
    // digits[_digits]* or empty; returns count, -1 on a bad underscore
    auto group = [&](uint8_t *out, int &cnt) -> int {
        cnt = 0;
        int last_us = 0, any = 0;
        while (p < m) {
            int d = nd_digit(cps[p]);
            if (d >= 0) { out[cnt++] = (uint8_t)d; any = 1; last_us = 0; p++; continue; }
            if (cps[p] == '_' && any && !last_us && p + 1 < m && nd_digit(cps[p + 1]) >= 0) { last_us = 1; p++; continue; }
            break;
        }
        return 0;
    };
    int nn = 0, nd = 0, ndec = 0, kind = 0;   // kind 0 integer, 1 slash, 2 decimal
    group(num, nn);
    if (p < m && cps[p] == '/') {
        p++;
        group(den, nd);
        if (nd == 0) return MIP_BAD_INPUT;
        kind = 1;
    } else if (p < m && cps[p] == '.') {
        p++;
        group(dec, ndec);
        kind = 2;
    }
    if (p != m) return MIP_BAD_INPUT;
    int cap = MIP_STR_MAX / 9 + 4;
    r.n = big_new(c, 2 * cap + 4);
    r.d = big_new(c, 2 * cap + 4);
    if (c.fail) return MIP_DEVICE_FAILED;
    if (kind == 2) {   // numerator * 10^k + decimal digits over 10^k: one run of digits
        for (int k = 0; k < ndec; k++) num[nn + k] = dec[k];
        big_from_digits(c, r.n, num, nn + ndec);
        dec[0] = 1;   // 10^ndec as the digit run 1 0 0 ... (ndec + 1 <= MIP_STR_MAX digits): read like any other run
        for (int k = 1; k <= ndec; k++) dec[k] = 0;
        big_from_digits(c, r.d, dec, ndec + 1);
    } else {
        big_from_digits(c, r.n, num, nn);
        if (kind == 1) {
            big_from_digits(c, r.d, den, nd);
            if (r.d.n == 0) return MIP_BAD_INPUT;   // Fraction(x, 0): ZeroDivisionError
        } else {
            r.d.n = 1;
            r.d.w[0] = 1;
        }
    }
    r.d.neg = 0;
    r.n.neg = r.n.n ? neg : 0;
    return MIP_OK;
}

// an ASCII decimal integer string as canon writes it (optional '-', digits)
__device__ int big_from_ascii(Ctx &c, Big &r, const uint8_t *s, int len) {
    int neg = 0, i = 0;
    if (len && s[0] == '-') { neg = 1; i = 1; }
    int digits = len - i;
    if (digits <= 0) return MIP_BAD_INPUT;
    int limbs = digits / 9 + 2;
    if (limbs > MIP_LIMBS) return MIP_NOT_COVERED;   // past the device's bound (design open question 3)
    r = big_new(c, 2 * limbs + 4);
    uint8_t *dig = (uint8_t *)(c.arena + c.used);
    if (c.used + digits / 4 + 2 > c.cap) { c.fail = 1; return MIP_DEVICE_FAILED; }
    if (c.fail) return MIP_DEVICE_FAILED;
    c.used += digits / 4 + 2;   // the digit values are held: the conversion takes its chunks from the arena after them
    for (int k = 0; k < digits; k++) {
        uint8_t ch = s[i + k];
        if (ch < '0' || ch > '9') return MIP_BAD_INPUT;
        dig[k] = ch - '0';
    }
    big_from_digits(c, r, dig, digits);
    r.neg = r.n ? neg : 0;
    return MIP_OK;
}

// ---- output ----------------------------------------------------------------------------------------------------------------------------
struct Out {
    uint8_t *p;
    uint32_t n, cap;
    int over;
};

__device__ __forceinline__ void put(Out &o, const char *s) {
    while (*s) {
        if (o.n >= o.cap) { o.over = 1; return; }
        o.p[o.n++] = (uint8_t)*s++;
    }
}

// the decimal digits of a, written into o (a is consumed)
__device__ void put_big(Ctx &c, Out &o, Big a) {
    if (a.neg) put(o, "-");
    if (a.n == 0) { put(o, "0"); return; }
    uint32_t *chunks = c.arena + c.used;   // base-10^9 digits, least significant first
    int nc = 0;
    if (team_workers() && a.n / team_seg() >= 2 && team_threads((long long)a.n * a.n / 4) > 1) {
        if (c.used + 2 * a.n + 4 > c.cap) { c.fail = 1; return; }   // chunks <= 1.08 a.n + 1
        tm.a0 = (long long)a.w; tm.a1 = (long long)chunks; tm.a2 = a.n;
        int L = pass_reg_len(a.n);
        tm.a4 = L;
        team_run(L ? J_TODEC_REG : J_TODEC, team_workers());
        nc = (int)tm.a3;
        a.n = (int)tm.a2;
        g_path |= 512;
        if (g_timed_out) { c.fail = 1; return; }
    }
    while (a.n) {
        if ((nc & 63) == 0 && past_deadline()) { c.fail = 1; return; }
        if (c.used + nc + 1 > c.cap) { c.fail = 1; return; }
        chunks[nc++] = mag_divmod_small(a, 1000000000u);
    }
    char buf[12];
    for (int k = nc - 1; k >= 0; k--) {
        uint32_t v = chunks[k];
        int len = 0;
        char t[10];
        do { t[len++] = (char)('0' + v % 10); v /= 10; } while (v);
        int pad = k == nc - 1 ? 0 : 9 - len;
        int b = 0;
        for (int z = 0; z < pad; z++) buf[b++] = '0';
        for (int z = len - 1; z >= 0; z--) buf[b++] = t[z];
        buf[b] = 0;
        put(o, buf);
    }
}

__device__ void put_rat(Ctx &c, Out &o, Rat &r) {
    if (r.d.n == 1 && r.d.w[0] == 1) {
        put(o, "[\"i\",\"");
        put_big(c, o, r.n);
        put(o, "\"]");
    } else {
        put(o, "[\"q\",\"");
        put_big(c, o, r.n);
        put(o, "\",\"");
        put_big(c, o, r.d);
        put(o, "\"]");
    }
}

// ---- arguments ---------------------------------------------------------------------------------------------------------------------
// The canonical arguments: ["l",[item, item, ...]]. read_rat parses one item as numbers._rat reads a value: int and Fraction pass,
// a string goes through Fraction(str), bool and every other type is BadInput. On return i is past the item.
__device__ __forceinline__ int skip_ws(const uint8_t *s, int len, int i) {
    while (i < len && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) i++;
    return i;
}

__device__ int skip_value(const uint8_t *s, int len, int &i) {   // any JSON value
    int depth = 0;
    do {
        if (i >= len) return -1;
        uint8_t ch = s[i];
        if (ch == '"') {
            i++;
            while (i < len && s[i] != '"') i += s[i] == '\\' ? 2 : 1;
            i++;
        } else {
            if (ch == '[' || ch == '{') depth++;
            if (ch == ']' || ch == '}') depth--;
            i++;
        }
        i = skip_ws(s, len, i);
    } while (depth > 0);
    return 0;
}

__device__ int read_rat(Ctx &c, Rat &r, const uint8_t *s, int len, int &i) {
    i = skip_ws(s, len, i);
    if (i >= len || s[i] != '[') return MIP_BAD_INPUT;
    int item = i;
    i++;
    uint8_t tag[4];
    int tl = json_string(s, len, i, tag, 4);
    if (tl != 1) { i = item; skip_value(s, len, i); return MIP_BAD_INPUT; }
    if (tag[0] == 'i' || tag[0] == 'q' || tag[0] == 's') {
        i = skip_ws(s, len, i);
        if (s[i] != ',') return MIP_BAD_INPUT;
        i = skip_ws(s, len, i + 1);
        uint8_t *buf = (uint8_t *)(c.arena + c.used);
        int room = (c.cap - c.used) * 4 / 2;
        int bl = json_string(s, len, i, buf, room);
        if (bl < 0) { c.fail = 1; return MIP_DEVICE_FAILED; }   // no room: a resource failure, retried on the global arena, never a refusal
        c.used += (bl + 7) / 4;
        int st;
        if (tag[0] == 's') {
            st = rat_from_str(c, r, buf, bl);
        } else {
            st = big_from_ascii(c, r.n, buf, bl);
            if (st == MIP_OK) {
                if (tag[0] == 'i') {
                    r.d = big_new(c, 2);
                    r.d.n = 1;
                    r.d.w[0] = 1;
                } else {
                    i = skip_ws(s, len, i);
                    if (s[i] != ',') return MIP_BAD_INPUT;
                    i = skip_ws(s, len, i + 1);
                    uint8_t *b2 = (uint8_t *)(c.arena + c.used);
                    int l2 = json_string(s, len, i, b2, (c.cap - c.used) * 4 / 2);
                    if (l2 < 0) { c.fail = 1; return MIP_DEVICE_FAILED; }
                    c.used += (l2 + 7) / 4;
                    st = big_from_ascii(c, r.d, b2, l2);
                }
            }
        }
        i = skip_ws(s, len, i);
        if (i < len && s[i] == ']') i++;
        if (c.fail) return MIP_DEVICE_FAILED;
        return st;
    }
    i = item;
    skip_value(s, len, i);
    return MIP_BAD_INPUT;   // bool, float, None, Decimal, list, ...: numbers._rat refuses them
}

// ---- the operations ----------------------------------------------------------------------------------------------------------------
// Python's argument order: each argument is converted in turn, so a refusal comes from the first bad argument; for div, _rat(b) and the
// zero check come before _rat(a). Both conversions raise BadInput, so the class is the same whichever is first, except where a device
// bound (NOT_COVERED) meets a BadInput: then Python's order is followed.
// ---- the small fast path -------------------------------------------------------------------------------------------------------------
// Most calls the model's text makes are small. When both arguments are an int, a Fraction or a plain ASCII string ([sign] digits,
// optionally '/' digits or '.' digits) whose parts have at most 18 digits, the call runs in registers with two-word (128-bit)
// integers and no arena. Anything else (Unicode, whitespace, underscores, a zero denominator or divisor, a malformed string, a longer
// number, another type) is not judged here: the call goes to the full path, which holds Python's rules and its order of refusals. So the
// fast path only answers calls the full path would answer with a value. Parts below 10^18 < 2^60 keep each product below 2^120 and a
// sum of two below 2^121. (MSVC, nvcc's host compiler here, has no __int128: U2 is two 64-bit words, products by __umul64hi.)
#define FAST_DIGITS 18

__device__ __forceinline__ int fast_digits(const uint8_t *s, int a, int b, long long &v) {   // s[a..b): 1 to FAST_DIGITS ASCII digits
    if (b - a < 1 || b - a > FAST_DIGITS) return 0;
    // two 32-bit runs of at most 9 digits each (a 64-bit multiply per digit costs several instructions here), joined once
    int mid = b - a > 9 ? b - 9 : a;
    unsigned hi = 0, lo = 0;
    for (int k = a; k < mid; k++) {
        unsigned d = (unsigned)s[k] - '0';
        if (d > 9) return 0;
        hi = hi * 10 + d;
    }
    for (int k = mid; k < b; k++) {
        unsigned d = (unsigned)s[k] - '0';
        if (d > 9) return 0;
        lo = lo * 10 + d;
    }
    v = (long long)hi * 1000000000LL + lo;
    return 1;
}

// the end of a JSON string's bytes starting at p, if it holds no escape; -1 otherwise
__device__ __forceinline__ int fast_string_end(const uint8_t *s, int len, int p) {
    int q = p;
    while (q < len && s[q] != '"' && s[q] != '\\') q++;
    return q < len && s[q] == '"' ? q : -1;
}

// one argument item at s[i] as n / d, d > 0 (not reduced); i moves past the item. 0: not a fast argument.
// canon.encode writes no whitespace, so the register path reads none: a request with any goes to the full path, which reads JSON
__device__ __forceinline__ int fast_arg(const uint8_t *s, int len, int &i, long long &n, long long &d) {
    if (i + 6 > len || s[i] != '[' || s[i + 1] != '"' || s[i + 3] != '"' || s[i + 4] != ',' || s[i + 5] != '"') return 0;
    uint8_t tag = s[i + 2];
    int p = i + 6;
    int q = fast_string_end(s, len, p);
    if (q < 0) return 0;
    long long v, w;
    if (tag == 'i' || tag == 'q') {
        int neg = p < q && s[p] == '-';
        if (!fast_digits(s, p + neg, q, v)) return 0;
        n = neg ? -v : v;
        d = 1;
        if (tag == 'q') {
            if (q + 2 >= len || s[q + 1] != ',' || s[q + 2] != '"') return 0;
            int t = fast_string_end(s, len, q + 3);
            if (t < 0 || !fast_digits(s, q + 3, t, w) || w == 0) return 0;
            d = w;
            q = t;
        }
    } else if (tag == 's') {
        int a = p, neg = 0;
        if (a < q && (s[a] == '-' || s[a] == '+')) { neg = s[a] == '-'; a++; }
        int slash = -1, dot = -1;
        for (int k = a; k < q; k++) {
            if (s[k] == '/' && slash < 0 && dot < 0) slash = k;
            else if (s[k] == '.' && dot < 0 && slash < 0) dot = k;
            else if (s[k] < '0' || s[k] > '9') return 0;
        }
        if (slash >= 0) {
            if (!fast_digits(s, a, slash, v) || !fast_digits(s, slash + 1, q, w) || w == 0) return 0;
            n = v;
            d = w;
        } else if (dot >= 0) {   // '5.', '.5', '1.25': the digits of both sides over 10^(fraction digits)
            int ni = dot - a, nf = q - dot - 1;
            if (ni + nf < 1 || ni + nf > FAST_DIGITS) return 0;
            long long x = 0, scale = 1;
            for (int k = a; k < dot; k++) x = x * 10 + (s[k] - '0');
            for (int k = dot + 1; k < q; k++) { x = x * 10 + (s[k] - '0'); scale *= 10; }
            n = x;
            d = scale;
        } else {
            if (!fast_digits(s, a, q, v)) return 0;
            n = v;
            d = 1;
        }
        if (neg) n = -n;
    } else {
        return 0;
    }
    int e = q + 1;
    if (e >= len || s[e] != ']') return 0;
    i = e + 1;
    return 1;
}

struct U2 {
    unsigned long long lo, hi;
};

__device__ U2 u2(unsigned long long x) { U2 r; r.lo = x; r.hi = 0; return r; }
__device__ int u2_zero(U2 x) { return !x.lo && !x.hi; }
__device__ int u2_cmp(U2 x, U2 y) { return x.hi != y.hi ? (x.hi < y.hi ? -1 : 1) : x.lo != y.lo ? (x.lo < y.lo ? -1 : 1) : 0; }
__device__ U2 u2_mul(unsigned long long a, unsigned long long b) { U2 r; r.lo = a * b; r.hi = __umul64hi(a, b); return r; }
__device__ U2 u2_add(U2 x, U2 y) { U2 r; r.lo = x.lo + y.lo; r.hi = x.hi + y.hi + (r.lo < x.lo); return r; }
__device__ U2 u2_sub(U2 x, U2 y) { U2 r; r.lo = x.lo - y.lo; r.hi = x.hi - y.hi - (x.lo < y.lo); return r; }   // x >= y
__device__ U2 u2_shr(U2 x, int k) {   // 0 <= k < 128
    U2 r;
    if (k == 0) return x;
    if (k >= 64) { r.lo = x.hi >> (k - 64); r.hi = 0; return r; }
    r.lo = (x.lo >> k) | (x.hi << (64 - k));
    r.hi = x.hi >> k;
    return r;
}
__device__ U2 u2_shl(U2 x, int k) {   // 0 <= k < 128
    U2 r;
    if (k == 0) return x;
    if (k >= 64) { r.hi = x.lo << (k - 64); r.lo = 0; return r; }
    r.hi = (x.hi << k) | (x.lo >> (64 - k));
    r.lo = x.lo << k;
    return r;
}
__device__ int u2_ctz(U2 x) { return x.lo ? __ffsll((long long)x.lo) - 1 : 64 + __ffsll((long long)x.hi) - 1; }   // x nonzero

__device__ int u2_bits(U2 x) { return x.hi ? 128 - __clzll((long long)x.hi) : x.lo ? 64 - __clzll((long long)x.lo) : 0; }

// q = u / v, r = u % v (v nonzero). One word each: the hardware divide. Otherwise shift-subtract over the bits u has above v's top only.
__device__ __forceinline__ void u2_divmod(U2 u, U2 v, U2 &q, U2 &r) {
    if (!u.hi && !v.hi) {
        q = u2(u.lo / v.lo);
        r = u2(u.lo % v.lo);
        return;
    }
    q = u2(0);
    r = u2(0);
    int top = u2_bits(u) - 1;
    for (int i = top; i >= 0; i--) {
        r = u2_shl(r, 1);
        r.lo |= (i >= 64 ? u.hi >> (i - 64) : u.lo >> i) & 1;
        if (u2_cmp(r, v) >= 0) {
            r = u2_sub(r, v);
            if (i >= 64) q.hi |= 1ull << (i - 64);
            else q.lo |= 1ull << i;
        }
    }
}

__device__ __forceinline__ U2 u2_gcd(U2 a, U2 b) {   // binary gcd, both nonzero
    int sh = u2_ctz(U2{a.lo | b.lo, a.hi | b.hi});
    a = u2_shr(a, u2_ctz(a));
    while (!u2_zero(b)) {
        if (!a.hi && !b.hi && a.lo <= 0xFFFFFFFFull && b.lo <= 0xFFFFFFFFull) {   // 32 bits each: finish in 32-bit registers
            unsigned x = (unsigned)a.lo, y = (unsigned)b.lo;
            while (y) {
                y >>= __ffs((int)y) - 1;
                if (x > y) { unsigned t = x; x = y; y = t; }
                y -= x;
            }
            return u2_shl(u2(x), sh);
        }
        if (!a.hi && !b.hi) {   // one word each: finish in 64-bit registers
            unsigned long long x = a.lo, y = b.lo;
            while (y) {
                y >>= __ffsll((long long)y) - 1;
                if (x > y) { unsigned long long t = x; x = y; y = t; }
                y -= x;
            }
            return u2_shl(u2(x), sh);
        }
        b = u2_shr(b, u2_ctz(b));
        if (u2_cmp(a, b) > 0) { U2 t = a; a = b; b = t; }
        b = u2_sub(b, a);
    }
    return u2_shl(a, sh);
}

// a value below 2^32 printed with 32-bit division: the GPU has no 64-bit divide instruction (each 64-bit `/` or `%` is a long software
// routine), and most answers of the register path are this small
__device__ __forceinline__ void put_u32(Out &o, unsigned x) {
    char t[12], buf[12];
    int m = 0;
    do { t[m++] = (char)('0' + x % 10u); x /= 10u; } while (x);
    for (int z = 0; z < m; z++) buf[z] = t[m - 1 - z];
    buf[m] = 0;
    put(o, buf);
}

__device__ __forceinline__ void put_u2(Out &o, U2 v) {   // decimal, in chunks of 18 digits
    if (!v.hi && v.lo <= 0xFFFFFFFFull) {
        put_u32(o, (unsigned)v.lo);
        return;
    }
    const U2 base = u2(1000000000000000000ull);
    unsigned long long chunk[3];
    int n = 0;
    do {
        U2 q, r;
        u2_divmod(v, base, q, r);
        chunk[n++] = r.lo;
        v = q;
    } while (!u2_zero(v));
    char buf[24];
    for (int k = n - 1; k >= 0; k--) {
        unsigned long long x = chunk[k];
        char t[20];
        int m = 0;
        do { t[m++] = (char)('0' + x % 10); x /= 10; } while (x);
        int b = 0;
        for (int z = k == n - 1 ? 0 : 18 - m; z > 0; z--) buf[b++] = '0';
        for (int z = m - 1; z >= 0; z--) buf[b++] = t[z];
        buf[b] = 0;
        put(o, buf);
    }
}

// a signed two-word value: magnitude and sign
struct S2 {
    U2 m;
    int neg;
};

__device__ __forceinline__ S2 s2_mul(long long a, long long b) {   // |a|, |b| < 2^63
    S2 r;
    r.m = u2_mul((unsigned long long)(a < 0 ? -a : a), (unsigned long long)(b < 0 ? -b : b));
    r.neg = !u2_zero(r.m) && ((a < 0) != (b < 0));
    return r;
}

__device__ __forceinline__ S2 s2_add(S2 x, S2 y) {
    S2 r;
    if (x.neg == y.neg) { r.m = u2_add(x.m, y.m); r.neg = x.neg; }
    else if (u2_cmp(x.m, y.m) >= 0) { r.m = u2_sub(x.m, y.m); r.neg = x.neg; }
    else { r.m = u2_sub(y.m, x.m); r.neg = y.neg; }
    if (u2_zero(r.m)) r.neg = 0;
    return r;
}

// 1: answered into o; 0: not a fast call (o untouched)
__device__ __forceinline__ int process_fast(int op, const uint8_t *s, int len, Out &o) {
    // ["l",[ <arg> , <arg> ]] exactly, no whitespace (canon.encode's form)
    if (len < 10 || s[0] != '[' || s[1] != '"' || s[2] != 'l' || s[3] != '"' || s[4] != ',' || s[5] != '[') return 0;
    long long an, ad, bn, bd;
    int ii = 0;
    // two integers, the commonest request: ["l",[["i","<A>"],["i","<B>"]]]. The fixed 12-byte head is three word compares (s is the
    // 4-aligned local copy); then A's digits, the fixed 9 bytes between, B's digits and the fixed tail. Any mismatch falls through to
    // the general reading below, which accepts the same forms and more.
    {
        const uint32_t *w = (const uint32_t *)s;
        if (len >= 26 && w[1] == 0x225B5B2Cu && w[2] == 0x222C2269u) {   // ',[["'  'i","'
            // each integer read in one pass: its value accumulates while its end is found (1 to 18 digits, an optional '-')
            int p = 12, na = s[p] == '-', ea = p + na;
            unsigned long long va = 0;
            while (ea < len && ea - p - na < 19) {
                unsigned d = (unsigned)s[ea] - '0';
                if (d > 9) break;
                va = va * 10 + d;
                ea++;
            }
            int da = ea - p - na;
            int q = ea + 9;   // '"],["i","'
            if (da >= 1 && da <= 18 && q < len && s[ea] == '"' && s[ea + 1] == ']' && s[ea + 2] == ',' && s[ea + 3] == '[' &&
                s[ea + 4] == '"' && s[ea + 5] == 'i' && s[ea + 6] == '"' && s[ea + 7] == ',' && s[ea + 8] == '"') {
                int nb = s[q] == '-', eb = q + nb;
                unsigned long long vb = 0;
                while (eb < len && eb - q - nb < 19) {
                    unsigned d = (unsigned)s[eb] - '0';
                    if (d > 9) break;
                    vb = vb * 10 + d;
                    eb++;
                }
                int db = eb - q - nb;
                if (db >= 1 && db <= 18 && eb + 4 == len && s[eb] == '"' && s[eb + 1] == ']' && s[eb + 2] == ']' && s[eb + 3] == ']') {
                    an = na ? -(long long)va : (long long)va;
                    bn = nb ? -(long long)vb : (long long)vb;
                    ad = bd = 1;
                    ii = 1;
                }
            }
        }
    }
    if (!ii) {
        int i = 6;
        if (!fast_arg(s, len, i, an, ad)) return 0;
        if (i >= len || s[i] != ',') return 0;
        if (!fast_arg(s, len, ++i, bn, bd)) return 0;
        if (i + 2 != len || s[i] != ']' || s[i + 1] != ']') return 0;
    }
    // two integers below 2^31 in magnitude: the answer in native 64-bit (a sum or product stays below 2^62), a quotient reduced in 32 bits
    const long long LIM = 0x7FFFFFFFLL;
    if (ad == 1 && bd == 1 && an <= LIM && an >= -LIM && bn <= LIM && bn >= -LIM && (op != OP_DIV || bn != 0)) {
        long long num, den = 1;
        if (op == OP_ADD) num = an + bn;
        else if (op == OP_SUB) num = an - bn;
        else if (op == OP_MUL) num = an * bn;
        else {
            unsigned x = (unsigned)(an < 0 ? -an : an), y = (unsigned)(bn < 0 ? -bn : bn);
            unsigned g = y;   // gcd(x, y), Euclid in 32 bits (gcd(0, y) = y)
            for (unsigned a2 = x, b2 = y; b2;) { unsigned t = a2 % b2; a2 = b2; b2 = t; g = a2; }
            num = (long long)(x / g) * ((an < 0) != (bn < 0) && x ? -1 : 1);
            den = y / g;
        }
        unsigned long long mag = (unsigned long long)(num < 0 ? -num : num);
        // one pass straight into the output: the fixed parts as known bytes, each number's digits written from its last one back
        // (its length counted first), with one bound check for the whole answer instead of one per byte
        int lm = 1, ld = 1;
        for (unsigned long long t = mag; t >= 10; t /= 10) lm++;
        for (unsigned t = (unsigned)den; t >= 10; t /= 10) ld++;
        int need = 6 + (num < 0) + lm + (den != 1 ? 3 + ld : 0) + 2;
        if (o.n + need > o.cap) { o.over = 1; return 0; }
        uint8_t *w = o.p + o.n;
        w[0] = '['; w[1] = '"'; w[2] = den == 1 ? 'i' : 'q'; w[3] = '"'; w[4] = ','; w[5] = '"';
        int k = 6;
        if (num < 0) w[k++] = '-';
        if (mag <= 0xFFFFFFFFull) {   // 32-bit digits where they fit (no 64-bit divide on the GPU)
            unsigned v = (unsigned)mag;
            for (int z = k + lm - 1; z >= k; z--) { w[z] = (uint8_t)('0' + v % 10u); v /= 10u; }
        } else {
            unsigned long long v = mag;
            for (int z = k + lm - 1; z >= k; z--) { w[z] = (uint8_t)('0' + v % 10ull); v /= 10ull; }
        }
        k += lm;
        if (den != 1) {
            w[k++] = '"'; w[k++] = ','; w[k++] = '"';
            unsigned v = (unsigned)den;
            for (int z = k + ld - 1; z >= k; z--) { w[z] = (uint8_t)('0' + v % 10u); v /= 10u; }
            k += ld;
        }
        w[k++] = '"'; w[k++] = ']';
        o.n += k;
        return 1;
    }
    S2 N, D;
    if (op == OP_ADD || op == OP_SUB) {
        S2 t = s2_mul(bn, ad);
        if (op == OP_SUB && !u2_zero(t.m)) t.neg = !t.neg;
        N = s2_add(s2_mul(an, bd), t);
        D = s2_mul(ad, bd);
    } else if (op == OP_MUL) {
        N = s2_mul(an, bn);
        D = s2_mul(ad, bd);
    } else {
        if (bn == 0) return 0;   // division by zero: the full path refuses it
        N = s2_mul(an, bd);
        D = s2_mul(ad, bn);
        if (D.neg) {   // the sign goes to the numerator
            D.neg = 0;
            if (!u2_zero(N.m)) N.neg = !N.neg;
        }
    }
    if (u2_zero(N.m)) {
        put(o, "[\"i\",\"0\"]");
        return !o.over;
    }
    U2 g = u2_gcd(N.m, D.m), r;
    if (!N.m.hi && !D.m.hi && !g.hi && N.m.lo <= 0xFFFFFFFFull && D.m.lo <= 0xFFFFFFFFull) {   // all below 2^32: 32-bit divides
        N.m = u2((unsigned)N.m.lo / (unsigned)g.lo);
        D.m = u2((unsigned)D.m.lo / (unsigned)g.lo);
    } else {
        u2_divmod(N.m, g, N.m, r);
        u2_divmod(D.m, g, D.m, r);
    }
    int whole = D.m.lo == 1 && D.m.hi == 0;
    put(o, whole ? (N.neg ? "[\"i\",\"-" : "[\"i\",\"") : (N.neg ? "[\"q\",\"-" : "[\"q\",\""));
    put_u2(o, N.m);
    if (!whole) {
        put(o, "\",\"");
        put_u2(o, D.m);
    }
    put(o, "\"]");
    return !o.over;
}

__device__ int process(Ctx &c, int op, const uint8_t *s, int len, Out &o) {
    int i = skip_ws(s, len, 0);
    // ["l",[ ... ]]
    if (i >= len || s[i] != '[') return MIP_BAD_INPUT;
    i++;
    uint8_t tag[4];
    if (json_string(s, len, i, tag, 4) != 1 || tag[0] != 'l') return MIP_BAD_INPUT;
    i = skip_ws(s, len, i);
    if (s[i] != ',') return MIP_BAD_INPUT;
    i = skip_ws(s, len, i + 1);
    if (s[i] != '[') return MIP_BAD_INPUT;
    i++;
    Rat a, b;
    int ia = i;
    int sa = read_rat(c, a, s, len, i);
    if (c.fail || sa == MIP_DEVICE_FAILED) return MIP_DEVICE_FAILED;   // never read on past a failed argument
    i = skip_ws(s, len, i);
    if (s[i] != ',') return MIP_BAD_INPUT;   // not two arguments: a TypeError in Python, never in the corpus
    i++;
    int sb = read_rat(c, b, s, len, i);
    if (c.fail || sb == MIP_DEVICE_FAILED) return MIP_DEVICE_FAILED;
    (void)ia;
    if (op == OP_DIV) {
        if (sb != MIP_OK) return sb;
        if (b.n.n == 0) return MIP_BAD_INPUT;   // division by zero
        if (sa != MIP_OK) return sa;
    } else {
        if (sa != MIP_OK) return sa;
        if (sb != MIP_OK) return sb;
    }
    int cap = a.n.n + a.d.n + b.n.n + b.d.n + 8;
    if (cap > 4 * MIP_LIMBS) return MIP_NOT_COVERED;
    Rat r;
    r.n = big_new(c, cap);
    r.d = big_new(c, cap);
    Big t1 = big_new(c, cap), t2 = big_new(c, cap);
    if (c.fail) return MIP_DEVICE_FAILED;
    if (op == OP_ADD || op == OP_SUB) {
        big_mul(t1, a.n, b.d);
        big_mul(t2, b.n, a.d);
        if (op == OP_SUB) t2.neg = t2.n ? !t2.neg : 0;
        big_add(r.n, t1, t2);
        big_mul(r.d, a.d, b.d);
    } else if (op == OP_MUL) {
        big_mul(r.n, a.n, b.n);
        big_mul(r.d, a.d, b.d);
    } else {   // div: (a.n * b.d) / (a.d * b.n), the sign moved to the numerator
        big_mul(r.n, a.n, b.d);
        big_mul(r.d, a.d, b.n);
        if (r.d.neg) {
            r.d.neg = 0;
            r.n.neg = r.n.n ? !r.n.neg : 0;
        }
    }
    rat_reduce(c, r, cap);
    if (c.fail) return MIP_DEVICE_FAILED;
    put_rat(c, o, r);
    if (c.fail || o.over) return MIP_DEVICE_FAILED;
    return MIP_OK;
}

__device__ void serve_inner(Slot *slot, uint32_t *arena, uint32_t *sh, int op, uint32_t len);

__device__ __noinline__ void serve_req(Slot *slot, uint32_t *arena, uint32_t *sh, int op, uint32_t len) {
    long long t = clock64();
    g_deadline = t + g_call_cycles;
    g_timed_out = 0;
    g_path = 0;
    serve_inner(slot, arena, sh, op, len);
    if (g_timed_out) {   // whatever the call left is not an answer
        g_path |= 16;
        slot->status = MIP_DEVICE_FAILED;
        slot->out_len = 0;
    }
    slot->path = g_path;
    slot->dev_cycles = (uint64_t)(clock64() - t);
}

// the header read from the slot (modes 0 and 1, and mode 2's fallbacks)
__device__ __noinline__ void serve(Slot *slot, uint32_t *arena, uint32_t *sh) { serve_req(slot, arena, sh, slot->op, slot->args_len); }

__device__ void serve_inner(Slot *slot, uint32_t *arena, uint32_t *sh, int op, uint32_t len_arg) {
    Out o;
    o.p = slot->out;
    o.n = 0;
    o.cap = MIP_OUT_MAX;
    o.over = 0;
    if (len_arg <= 256) {   // small arguments: a local copy (L1-cached), then the register path
        uint32_t loc[64];
        uint32_t w = (len_arg + 3) / 4;
        const uint32_t *src0 = (const uint32_t *)slot->args;
        // plain loads: the L2 fetches a 32-byte sector a PCIe read; ld.cv loads went one round trip a word (5.4 -> 21 us, 2026-11-11)
        for (uint32_t k = 0; k < w; k++) loc[k] = src0[k];
        Out lo;
        lo.p = slot->out;   // byte writes: posted, they do not stall (a local buffer written in words measured no faster)
        lo.n = 0;
        lo.cap = MIP_OUT_MAX;
        lo.over = 0;
        if (process_fast(op, (const uint8_t *)loc, (int)len_arg, lo)) {
            g_path |= 1;
            slot->status = MIP_OK;
            slot->out_len = lo.n;
            return;
        }
    }
    // The full path. Its working memory goes to the block's shared memory first (tens of cycles a word, against hundreds in global
    // memory, and one thread waits on every access); a call that runs out of it is done again from the start on the global arena.
    // The arguments are copied in once, in 4-byte words (in the resident mode the slot is mapped host memory: a read there crosses PCIe).
    uint32_t len = len_arg, words = (len + 3) / 4;
    int st = MIP_DEVICE_FAILED;
    g_path |= 2;
    for (int attempt = sh ? 0 : 1; attempt < 2; attempt++) {
        uint32_t *ar = attempt == 0 ? sh : arena;
        int cap = attempt == 0 ? SH_WORDS : ARENA_WORDS;
        if (words + 64 > (uint32_t)cap) continue;
        Ctx c;
        c.arena = ar;
        c.cap = cap;
        c.fail = 0;
        const uint32_t *src = (const uint32_t *)slot->args;
        for (uint32_t k = 0; k < words; k++) ar[k] = src[k];
        c.used = words + 1;
        o.n = 0;
        o.over = 0;
        st = process(c, op, (const uint8_t *)ar, (int)len, o);
        if (!(attempt == 0 && c.fail)) break;   // only a shared arena that ran out is worth a second try
        st = MIP_DEVICE_FAILED;
    }
    if (st == MIP_NOT_COVERED || st == MIP_BAD_INPUT) {
        o.n = 0;
        o.over = 0;
        put(o, st == MIP_NOT_COVERED ? "[\"x\",\"NotCovered\"]" : "[\"x\",\"BadInput\"]");
    } else if (st != MIP_OK) {
        o.n = 0;
    }
    slot->status = st;
    slot->out_len = o.n;
}

__device__ int g_poll_mode = 1;   // 1: flag read with ld.cv and written with a system atomic; 0: plain volatile (MIP_POLL)

// The resident worker: polls the slot until about `budget` clock cycles have passed, then ends (the host launches the next).
__device__ __forceinline__ void worker_body(Slot *slot, uint32_t *arena, long long budget, int sleep_ns) {
    extern __shared__ uint32_t sh_arena[];
    long long t0 = clock64(), last = t0;
    unsigned long long polls = 0;
    for (;;) {
        // read the flag past every cache (ld.global.cv): a cached copy of mapped host memory can stay stale for a long time
        int st = g_poll_mode ? __ldcv((const int *)&slot->state) : slot->state;
        polls++;
        if (st == 1) {
            slot->seen = 1;   // a volatile store: system atomics on mapped host memory are not supported here (HostNativeAtomicSupported 0)
            long long seen = clock64();
            slot->polls = polls;
            slot->poll_cycles = (uint64_t)(seen - t0);
            slot->wait_cycles = (uint64_t)(seen - last);
            __threadfence_system();
            serve(slot, arena, sh_arena);
            last = clock64();
            __threadfence_system();
            // a call does not reset the budget: the worker still ends on time
            if (g_poll_mode) slot->state = 2;   // volatile store after the system fence (no host-memory atomics on this card)
            else slot->state = 2;
        } else if (st == 3) {
            return;
        }
        if (clock64() - t0 > budget) return;
        if (sleep_ns) __nanosleep(sleep_ns);   // MIP_NANOSLEEP: between polls; 0 polls flat out
    }
}

__global__ void worker(Slot *slot, uint32_t *arena, long long budget, int sleep_ns) {
    if (threadIdx.x) { if (blockDim.x >= 64) team_serve(); return; }
    worker_body(slot, arena, budget, sleep_ns);
    team_quit();
}

// One launch per call (the comparison mode): the slot is in device memory.
__device__ __forceinline__ void once_body(Slot *slot, uint32_t *arena) {
    extern __shared__ uint32_t sh_arena[];
    serve(slot, arena, sh_arena);
}

__global__ void once(Slot *slot, uint32_t *arena) {
    if (threadIdx.x) { if (blockDim.x >= 64) team_serve(); return; }
    once_body(slot, arena);
    team_quit();
}

// Mode 2 linger (MIP_LINGER_US): after answering, a kernel keeps polling the slot for a short while, so a request that follows soon is
// served without a launch (the resident worker's latency) while a quiet engine gets its GPU back as soon as the window closes. A kernel
// that ends clears `lingering`, fences, then looks at the flag once more; the host sets the flag, fences, then reads `lingering`
// (Dekker): either the host sees the kernel gone and launches, or the kernel sees the request and serves it. Kernels on one stream run
// one after another, and a kernel serves only a flag that reads 1, so a request is served once.
__device__ long long g_linger_cycles = 0;
__device__ unsigned g_linger_sleep_ns = 100;

__device__ __noinline__ void serve_flag(Slot *slot, uint32_t *arena, uint32_t *sh) {
    __threadfence_system();
    serve(slot, arena, sh);
    __threadfence_system();
    slot->state = 2;   // volatile store after __threadfence_system(): the answer is visible before the flag
}

__device__ __noinline__ void linger(Slot *slot, uint32_t *arena, uint32_t *sh) {
    if (g_linger_cycles <= 0) return;
    slot->lingering = 1;
    __threadfence_system();
    long long end = clock64() + g_linger_cycles;
    for (;;) {
        unsigned long long bell = __ldcv((const unsigned long long *)&slot->state);
        int st = (int)(unsigned)bell;
        if (st == 1) {   // the same read brought op and length: no PCIe read for the header
            unsigned ol = (unsigned)(bell >> 32);
            __threadfence_system();
            serve_req(slot, arena, sh, (int)(ol & 0xFF), ol >> 8);
            __threadfence_system();
            slot->state = 2;
            end = clock64() + g_linger_cycles;
            continue;
        }
        if (st == 3 || clock64() > end) break;
        if (g_linger_sleep_ns) __nanosleep(g_linger_sleep_ns);   // MIP_LINGER_SLEEP: between polls of the linger window
    }
    slot->lingering = 0;
    __threadfence_system();
    if (__ldcv((const int *)&slot->state) == 1) serve_flag(slot, arena, sh);
}

// One launch per call, zero copy (mode 2): the slot is mapped host memory; the kernel reads the request there, answers there and raises
// the slot's flag itself, so the host needs no copy and no synchronising driver call: it launches, then polls the flag.
__device__ __forceinline__ void once_mapped_body(Slot *slot, uint32_t *arena) {
    extern __shared__ uint32_t sh_arena[];
    if (__ldcv((const int *)&slot->state) == 1) serve_flag(slot, arena, sh_arena);   // else a lingering kernel already served it
    linger(slot, arena, sh_arena);
}

__global__ void once_mapped(Slot *slot, uint32_t *arena) {
    if (threadIdx.x) { if (blockDim.x >= 64) team_serve(); return; }
    once_mapped_body(slot, arena);
    team_quit();
}

// Mode 2 with small arguments: they travel in the launch's own parameters (the constant bank), so the register path reads no byte over
// PCIe; a call the register path does not take falls back to serve(), which reads the copy in the mapped slot.
template <int W>
struct ParamArgsT {
    int32_t op;
    int32_t len;
    uint32_t seq;
    uint32_t w[W];   // up to 4*W bytes of canonical arguments
};
typedef ParamArgsT<64> ParamArgs;    // 256 bytes
// (a 64-byte block for short requests measured no faster to launch: 13.8-14.0 us either way, 2026-11-11)

__device__ int g_param_smem = 0;   // 1: once_param launches with the shared arena (MIP_PARAM_SMEM); 0: none, a faster start

__device__ int g_diag = 1;

template <int W>
__device__ __forceinline__ void once_param_body(Slot *slot, uint32_t *arena, const ParamArgsT<W> &pa) {
    extern __shared__ uint32_t sh_arena[];
    uint32_t *sh = g_param_smem ? sh_arena : nullptr;
    // only with lingering can a kernel find its request taken or replaced; without it the two PCIe reads are skipped
    if (g_linger_cycles > 0) {
        if (__ldcv((const int *)&slot->state) != 1) {   // already served (by a lingering kernel): only linger
            linger(slot, arena, sh);
            return;
        }
        if (slot->seq != pa.seq) {   // a newer request than this launch's parameters: serve it from the slot
            serve_flag(slot, arena, sh);
            linger(slot, arena, sh);
            return;
        }
    }
    long long t = clock64();
    uint32_t loc[64];
    int words = (pa.len + 3) / 4;
    for (int k = 0; k < words; k++) loc[k] = pa.w[k];
    // the answer is built in a local buffer and written to the mapped slot in 4-byte words: one PCIe write a word, not one a byte
    uint32_t lout[32];
    Out o;
    o.p = (uint8_t *)lout;
    o.n = 0;
    o.cap = sizeof(lout);
    o.over = 0;
    if (process_fast(pa.op, (const uint8_t *)loc, pa.len, o)) {
        uint32_t *dst = (uint32_t *)slot->out;
        for (uint32_t k = 0; k < (o.n + 3) / 4; k++) dst[k] = lout[k];
        slot->status = MIP_OK;
        slot->out_len = o.n;
        if (g_diag) { slot->path = 1; slot->dev_cycles = (uint64_t)(clock64() - t); }   // diagnostics: two PCIe writes (MIP_DIAG=0 drops them)
    } else {
        serve(slot, arena, sh);   // no shared memory asked: the full path uses the global arena
    }
    __threadfence_system();
    slot->state = 2;   // volatile store after __threadfence_system(): the answer is visible before the flag
    linger(slot, arena, sh);
}

template <int W>
__global__ void once_param(Slot *slot, uint32_t *arena, ParamArgsT<W> pa) {
    if (threadIdx.x) { if (blockDim.x >= 64) team_serve(); return; }
    once_param_body(slot, arena, pa);
    team_quit();
}

// ---- host ----------------------------------------------------------------------------------------------------------------------------
static std::mutex g_lock;
static int g_open = 0, g_resident = 1;   // the mode: 0 a launch per call with copies, 1 the resident worker, 2 a launch per call, zero copy
static cudaGraphExec_t g_graph_exec = nullptr;   // mode 2 with MIP_GRAPH=1: the launch captured once as a CUDA graph
static int g_use_graph = 0;
static int g_param_args = 1;   // mode 2: small arguments in the launch parameters (MIP_PARAM=0 turns it off)
static int g_param_smem_host = 0;   // MIP_PARAM_SMEM: the parameter kernel asks for the shared arena too
static uint32_t g_seq = 0;
static uint64_t g_skipped = 0;   // mode 2 calls a lingering kernel served without a launch
static Slot *g_host_slot = nullptr, *g_dev_slot = nullptr;   // resident: mapped pinned memory; per call: device memory
static uint32_t *g_arena = nullptr;
static cudaStream_t g_stream;
static cudaEvent_t g_done;
static long long g_budget = 0;
static int g_running = 0;
static std::chrono::steady_clock::time_point g_launched;   // when the running worker was launched
static int g_clock_khz = 1;
static double g_us_seen = 0, g_us_done = 0;   // diagnostics of the last resident call, host clock
static int g_spin = 2000;   // polls before the host starts yielding the core (MIP_SPIN)
static int g_sleep_ns = 500;
static int g_team = 1;   // MIP_TEAM: threads in every kernel's block (default set at open, by the mode)   // the worker's sleep between polls (MIP_NANOSLEEP)
static std::atomic<uint64_t> g_launches{0};

static int launch_worker() {
    worker<<<1, g_team, SH_WORDS * sizeof(uint32_t), g_stream>>>(g_dev_slot, g_arena, g_budget, g_sleep_ns);
    if (cudaGetLastError() != cudaSuccess) return -1;
    cudaEventRecord(g_done, g_stream);
    g_running = 1;
    g_launched = std::chrono::steady_clock::now();
    g_launches++;
    return 0;
}

// The worker ends itself after MIP_WORKER_SECONDS of device clock. Before 0.8 of that has passed on the host clock it is alive without
// asking the driver: a cudaEventQuery per poll costs a driver call under WDDM, which the first speed runs paid on every poll.
static int worker_alive() {
    if (!g_running) return 0;
    double age = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_launched).count();
    if (age < 0.8 * MIP_WORKER_SECONDS) return 1;
    if (cudaEventQuery(g_done) == cudaSuccess) g_running = 0;
    return g_running;
}

extern "C" MIP_API int mip_open(int device, int resident) {
    std::lock_guard<std::mutex> l(g_lock);
    if (g_open) return MIP_OK;
    if (cudaSetDevice(device) != cudaSuccess) return MIP_HOST_ERROR;
    // how a synchronising driver call waits (mode 0 only; modes 1 and 2 poll the mapped flag): MIP_SYNC spin | yield | block
    unsigned sync = cudaDeviceScheduleAuto;
    if (resident == 0) sync = cudaDeviceScheduleBlockingSync;
    if (const char *e = getenv("MIP_SYNC")) sync = !strcmp(e, "spin") ? cudaDeviceScheduleSpin : !strcmp(e, "yield") ? cudaDeviceScheduleYield : cudaDeviceScheduleBlockingSync;
    cudaSetDeviceFlags(cudaDeviceMapHost | sync);
    if (const char *e = getenv("MIP_GRAPH")) g_use_graph = atoi(e);
    if (const char *e = getenv("MIP_PARAM")) g_param_args = atoi(e);
    if (const char *e = getenv("MIP_DIAG")) { int v = atoi(e); cudaMemcpyToSymbol(g_diag, &v, sizeof v); }
    if (const char *e = getenv("MIP_LINGER_SLEEP")) { unsigned ns = (unsigned)atoi(e); cudaMemcpyToSymbol(g_linger_sleep_ns, &ns, sizeof ns); }
    if (const char *e = getenv("MIP_LINGER_US")) {
        int cr = 0;
        cudaDeviceGetAttribute(&cr, cudaDevAttrClockRate, device);
        long long lc = (long long)(cr * 1000.0 * atof(e) * 1e-6);
        cudaMemcpyToSymbol(g_linger_cycles, &lc, sizeof lc);
    }
    if (const char *e = getenv("MIP_PARAM_SMEM")) { g_param_smem_host = atoi(e); cudaMemcpyToSymbol(g_param_smem, &g_param_smem_host, sizeof(int)); }
    // the team: 256 threads where a block lives only for its call (modes 0 and 2); none for the resident worker (mode 1), whose block
    // would hold an SM's registers away from an engine in the same process for as long as it runs (MIP_TEAM overrides)
    g_team = resident == 1 ? 1 : 256;
    if (const char *e = getenv("MIP_TEAM")) { g_team = atoi(e); if (g_team < 1) g_team = 1; if (g_team > 1024) g_team = 1024; }
    if (const char *e = getenv("MIP_TEAM_WORK")) { long long v = atoll(e); if (v < 1) v = 1; cudaMemcpyToSymbol(g_team_work, &v, sizeof v); }
    if (const char *e = getenv("MIP_SPIN")) g_spin = atoi(e);
    if (const char *e = getenv("MIP_NANOSLEEP")) g_sleep_ns = atoi(e);
    if (const char *e = getenv("MIP_POLL")) { int m = atoi(e); cudaMemcpyToSymbol(g_poll_mode, &m, sizeof m); }
    {   // a block larger than a kernel's limit does not launch at all: cap the team at the smallest limit, in whole warps
        cudaFuncAttributes fa;
        int lim = 1024;
        if (cudaFuncGetAttributes(&fa, worker) == cudaSuccess && fa.maxThreadsPerBlock < lim) lim = fa.maxThreadsPerBlock;
        if (cudaFuncGetAttributes(&fa, once) == cudaSuccess && fa.maxThreadsPerBlock < lim) lim = fa.maxThreadsPerBlock;
        if (cudaFuncGetAttributes(&fa, once_mapped) == cudaSuccess && fa.maxThreadsPerBlock < lim) lim = fa.maxThreadsPerBlock;
        if (cudaFuncGetAttributes(&fa, once_param<64>) == cudaSuccess && fa.maxThreadsPerBlock < lim) lim = fa.maxThreadsPerBlock;
        if (g_team > lim) g_team = lim;
        if (g_team > 32) g_team -= g_team % 32;
        if (g_team > 1 && g_team < 64) g_team = 64;   // a team is the master's warp and at least one warp of workers
    }
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device);
    g_clock_khz = clock_khz ? clock_khz : 1;
    g_budget = (long long)(clock_khz * 1000.0 * MIP_WORKER_SECONDS);
    {   // MIP_CALL_US: a shorter guard, for the test that the guard ends every path (the team's jobs included) without a hang
        double sec = MIP_CALL_SECONDS;
        if (const char *e = getenv("MIP_CALL_US")) { double v = atof(e) / 1e6; if (v > 0 && v < sec) sec = v; }
        long long cc = (long long)(clock_khz * 1000.0 * sec);
        cudaMemcpyToSymbol(g_call_cycles, &cc, sizeof cc);
    }
    if (cudaMalloc(&g_arena, sizeof(uint32_t) * ARENA_WORDS) != cudaSuccess) return MIP_HOST_ERROR;
    int lo, hi;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    { const char *e = getenv("MIP_STREAM_HIGH"); cudaStreamCreateWithPriority(&g_stream, cudaStreamNonBlocking, e && atoi(e) ? hi : lo); }   // lowest by default: an engine in this process goes first
    cudaEventCreateWithFlags(&g_done, cudaEventDisableTiming);
    g_resident = resident;
    if (resident) {   // modes 1 and 2: the slot in mapped host memory
        if (cudaHostAlloc(&g_host_slot, sizeof(Slot), cudaHostAllocMapped) != cudaSuccess) return MIP_HOST_ERROR;
        cudaHostGetDevicePointer(&g_dev_slot, g_host_slot, 0);
        g_host_slot->state = 0;
        if (resident == 2 && g_use_graph) {   // capture the one launch; every call replays it (same slot, same arena)
            cudaGraph_t graph;
            if (cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return MIP_HOST_ERROR;
            once_mapped<<<1, g_team, SH_WORDS * sizeof(uint32_t), g_stream>>>(g_dev_slot, g_arena);
            if (cudaStreamEndCapture(g_stream, &graph) != cudaSuccess) return MIP_HOST_ERROR;
            if (cudaGraphInstantiate(&g_graph_exec, graph, 0) != cudaSuccess) return MIP_HOST_ERROR;
            cudaGraphDestroy(graph);
            cudaGraphUpload(g_graph_exec, g_stream);
        }
    } else {
        if (cudaHostAlloc(&g_host_slot, sizeof(Slot), 0) != cudaSuccess) return MIP_HOST_ERROR;
        if (cudaMalloc(&g_dev_slot, sizeof(Slot)) != cudaSuccess) return MIP_HOST_ERROR;
    }
    g_open = 1;
    return MIP_OK;
}

extern "C" MIP_API void mip_close(void) {
    std::lock_guard<std::mutex> l(g_lock);
    if (!g_open) return;
    if (g_resident) {
        g_host_slot->state = 3;
        cudaStreamSynchronize(g_stream);
        if (g_graph_exec) { cudaGraphExecDestroy(g_graph_exec); g_graph_exec = nullptr; }
        cudaFreeHost(g_host_slot);
    } else {
        cudaFree(g_dev_slot);
        cudaFreeHost(g_host_slot);
    }
    cudaFree(g_arena);
    cudaEventDestroy(g_done);
    cudaStreamDestroy(g_stream);
    g_open = 0;
    g_running = 0;
}

static int op_id(const char *op) {
    if (!strcmp(op, "numbers.add")) return OP_ADD;
    if (!strcmp(op, "numbers.sub")) return OP_SUB;
    if (!strcmp(op, "numbers.mul")) return OP_MUL;
    if (!strcmp(op, "numbers.div")) return OP_DIV;
    return 0;
}

extern "C" MIP_API const char *mip_ops(void) { return "numbers.add\nnumbers.sub\nnumbers.mul\nnumbers.div\n"; }

extern "C" MIP_API uint64_t mip_launches(void) { return g_launches.load(); }

// mode 2: calls a lingering kernel served without a launch
extern "C" MIP_API uint64_t mip_skipped(void) { return g_skipped; }

// microseconds the last call spent in serve() on the device (clock64 over the rated clock: an estimate under boost clocks)
extern "C" MIP_API void mip_diag(uint64_t *v) {   // polls, poll cycles, wait cycles, clock kHz
    v[0] = g_host_slot->polls; v[1] = g_host_slot->poll_cycles; v[2] = g_host_slot->wait_cycles; v[3] = (uint64_t)g_clock_khz;
    v[4] = (uint64_t)(g_us_seen * 1000); v[5] = (uint64_t)(g_us_done * 1000);
    v[6] = g_host_slot->path;   // ns: request written -> seen flag back, -> answer back
}

// ---- the floor of a launch per call --------------------------------------------------------------------------------------------------
// An empty kernel that only raises the flag, launched n times the way mode 2 launches (launch, then poll the mapped flag), through the
// runtime API (api 0) or the driver API's cuLaunchKernel on the same function (api 1). The median of n round trips, measured inside the
// DLL (no Python), is the least a launch per call can cost on this machine; out[0] the median round trip, out[1] the median time of the
// launch call alone, in microseconds.
// ---- an engine in the same process (a stand-in) -------------------------------------------------------------------------------------
// The module's end shape is MIP inside the engine's process, sharing its CUDA context. To measure that without an engine: a host thread
// launches back-to-back "decode step" kernels on their own stream of this context, each a grid that fills every SM for about
// `step_us` (blocks spin on the clock), the way a decode step keeps the card busy; mip_call meanwhile goes through its usual path.
// mip_engine_stop returns the steps completed per second, so the engine's slowdown is measured beside our latency.
__global__ void engine_step(long long cycles, unsigned *sink) {
    long long t = clock64();
    unsigned x = threadIdx.x;
    while (clock64() - t < cycles) x = x * 1664525u + 1013904223u;
    if (x == 0x12345678u) atomicAdd(sink, 1u);   // keeps the loop from being removed
}

static std::thread g_engine;
static std::atomic<int> g_engine_run{0};
static std::atomic<uint64_t> g_engine_steps{0};
static std::chrono::steady_clock::time_point g_engine_t0;

extern "C" MIP_API int mip_engine_start(double step_us, int high_priority) {
    if (!g_open || g_engine_run.load()) return MIP_HOST_ERROR;
    int dev = 0, sms = 0, khz = 0, lo, hi;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev);
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    long long cycles = (long long)(khz * 1000.0 * step_us * 1e-6);
    g_engine_run = 1;
    g_engine_steps = 0;
    g_engine_t0 = std::chrono::steady_clock::now();
    g_engine = std::thread([=]() {
        cudaSetDevice(dev);
        cudaStream_t st;
        cudaStreamCreateWithPriority(&st, cudaStreamNonBlocking, high_priority ? hi : lo);
        unsigned *sink;
        cudaMalloc(&sink, sizeof(unsigned));
        while (g_engine_run.load()) {
            for (int k = 0; k < 8; k++) engine_step<<<sms * 4, 256, 0, st>>>(cycles, sink);   // 4 blocks of 256 per SM: every SM busy
            cudaStreamSynchronize(st);
            g_engine_steps += 8;
        }
        cudaFree(sink);
        cudaStreamDestroy(st);
    });
    return MIP_OK;
}

extern "C" MIP_API double mip_engine_stop(void) {
    g_engine_run = 0;
    if (g_engine.joinable()) g_engine.join();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_engine_t0).count();
    return g_engine_steps.load() / s;
}

__global__ void flag_only(Slot *slot) {
    __threadfence_system();
    slot->state = 2;
}

#include <cuda.h>
#include <algorithm>
#include <vector>

extern "C" MIP_API int mip_floor(int n, int api, double *out) {
    std::lock_guard<std::mutex> l(g_lock);
    if (!g_open || g_resident != 2) return MIP_HOST_ERROR;
    Slot *h = g_host_slot;
    CUfunction f = nullptr;
    if (api == 1 && cudaGetFuncBySymbol((cudaFunction_t *)&f, (const void *)flag_only) != cudaSuccess) return MIP_HOST_ERROR;
    std::vector<double> rt(n), lc(n);
    for (int i = 0; i < n; i++) {
        h->state = 1;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        auto t0 = std::chrono::steady_clock::now();
        if (api == 1) {
            void *args[] = {&g_dev_slot};
            if (cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, (CUstream)g_stream, args, nullptr) != CUDA_SUCCESS) return MIP_HOST_ERROR;
        } else {
            flag_only<<<1, 1, 0, g_stream>>>(g_dev_slot);
        }
        auto t1 = std::chrono::steady_clock::now();
        while (h->state != 2) {
        }
        auto t2 = std::chrono::steady_clock::now();
        lc[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        rt[i] = std::chrono::duration<double, std::micro>(t2 - t0).count();
        h->state = 0;
    }
    std::sort(rt.begin(), rt.end());
    std::sort(lc.begin(), lc.end());
    out[0] = rt[n / 2];
    out[1] = lc[n / 2];
    return MIP_OK;
}

extern "C" MIP_API double mip_last_device_us(void) { return g_host_slot ? g_host_slot->dev_cycles / (g_clock_khz / 1000.0) : 0.0; }

extern "C" MIP_API int mip_call(const char *op, const uint8_t *args, size_t args_len, uint8_t *out, size_t out_cap, size_t *out_len) {
    std::lock_guard<std::mutex> l(g_lock);
    *out_len = 0;
    if (!g_open) return MIP_HOST_ERROR;
    int id = op_id(op);
    if (!id) return MIP_UNKNOWN_OP;
    if (args_len > MIP_ARGS_MAX) return MIP_HOST_ERROR;
    Slot *h = g_host_slot;
    memcpy(h->args, args, args_len);
    h->args_len = (uint32_t)args_len;
    h->op = id;
    if (g_resident == 2) {
        // launch (or replay the graph), then poll the flag the kernel raises: no copy, no synchronising call
        h->seq = ++g_seq;
        h->oplen = (uint32_t)id | ((uint32_t)args_len << 8);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        auto t0 = std::chrono::steady_clock::now();
        // state and oplen in one aligned 8-byte store (the slot is page-aligned; state is its first word)
        *(volatile uint64_t *)&h->state = ((uint64_t)h->oplen << 32) | 1u;
        std::atomic_thread_fence(std::memory_order_seq_cst);   // Dekker with the kernel's exit: the flag, then `lingering`
        cudaError_t e = cudaSuccess;
        int launched = !h->lingering;   // read once
        if (!launched) {
            g_skipped++;   // a kernel is waiting: no launch
        } else if (g_graph_exec) {
            e = cudaGraphLaunch(g_graph_exec, g_stream);
        } else if (args_len <= sizeof(((ParamArgs *)0)->w) && g_param_args) {
            ParamArgs pa;
            pa.op = id;
            pa.len = (int32_t)args_len;
            pa.seq = g_seq;
            memcpy(pa.w, args, args_len);
            once_param<64><<<1, g_team, g_param_smem_host ? SH_WORDS * sizeof(uint32_t) : 0, g_stream>>>(g_dev_slot, g_arena, pa);
            e = cudaGetLastError();
        } else {
            once_mapped<<<1, g_team, SH_WORDS * sizeof(uint32_t), g_stream>>>(g_dev_slot, g_arena);
            e = cudaGetLastError();
        }
        if (e != cudaSuccess) return MIP_HOST_ERROR;
        if (launched) g_launches++;
        // diagnostics: the launch call's own time, and the time until the flag came back (host clock)
        g_us_seen = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        int spins = 0;
        while (h->state != 2) {
            if (++spins > g_spin) {
                std::this_thread::yield();
                // a kernel that ended without raising the flag failed: ask the driver now and then, not on every poll
                if ((spins & 1023) == 0) {
                    cudaError_t q = cudaStreamQuery(g_stream);
                    if (q == cudaSuccess && h->state == 1) {   // no kernel left and the request not taken: launch one
                        once_mapped<<<1, g_team, SH_WORDS * sizeof(uint32_t), g_stream>>>(g_dev_slot, g_arena);
                        g_launches++;
                    } else if (q != cudaSuccess && q != cudaErrorNotReady) {
                        return MIP_DEVICE_FAILED;
                    }
                }
            }
        }
        g_us_done = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        h->state = 0;
    } else if (g_resident) {
        h->seen = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        auto t0 = std::chrono::steady_clock::now();
        h->state = 1;
        if (!worker_alive() && launch_worker()) return MIP_HOST_ERROR;
        int spins = 0;
        int marked = 0;
        while (h->state != 2) {
            if (!marked && h->seen) {
                g_us_seen = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                marked = 1;
            }
            // poll a little (a small call answers in microseconds), then give the core away between polls
            if (++spins > g_spin) std::this_thread::yield();
            if (!worker_alive()) {   // the worker ended (its budget) before it took the request: launch the next
                if (h->state == 2) break;
                if (cudaGetLastError() != cudaSuccess) return MIP_DEVICE_FAILED;
                if (launch_worker()) return MIP_HOST_ERROR;
            }
        }
        g_us_done = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (!marked) g_us_seen = -1;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        h->state = 0;
    } else {
        size_t head = offsetof(Slot, args) + args_len;
        if (cudaMemcpyAsync(g_dev_slot, h, head, cudaMemcpyHostToDevice, g_stream) != cudaSuccess) return MIP_HOST_ERROR;
        once<<<1, g_team, SH_WORDS * sizeof(uint32_t), g_stream>>>(g_dev_slot, g_arena);
        g_launches++;
        if (cudaMemcpyAsync(&h->status, &g_dev_slot->status, offsetof(Slot, args) - offsetof(Slot, status), cudaMemcpyDeviceToHost, g_stream) != cudaSuccess)
            return MIP_HOST_ERROR;
        if (cudaStreamSynchronize(g_stream) != cudaSuccess) return MIP_DEVICE_FAILED;
        if (h->out_len && cudaMemcpy(h->out, g_dev_slot->out, h->out_len, cudaMemcpyDeviceToHost) != cudaSuccess) return MIP_HOST_ERROR;
    }
    int st = h->status;
    if (st == MIP_OK || st == MIP_NOT_COVERED || st == MIP_BAD_INPUT) {
        if (h->out_len > out_cap) return MIP_HOST_ERROR;
        memcpy(out, h->out, h->out_len);
        *out_len = h->out_len;
    }
    return st;
}
