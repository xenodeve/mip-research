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
#define SH_WORDS 12032   // the shared-memory arena of the worker: 47 KB, under the 48 KB a block gets without opting in
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
// asks for (open question 3); it only keeps the machine safe until that cap exists. One device thread serves a call at a time, so the
// state is plain device globals.
#define MIP_CALL_SECONDS 0.4
__device__ long long g_deadline = 0x7fffffffffffffffLL;
__device__ long long g_call_cycles = 0x7fffffffffffffffLL;   // set at open: MIP_CALL_SECONDS of the rated clock
__device__ int g_timed_out = 0;
__device__ unsigned g_path = 0;   // bits of the paths this call took: 1 register, 2 full, 4 Knuth D, 8 its add-back, 16 timed out

__device__ __forceinline__ int past_deadline() {
    if (clock64() > g_deadline) {
        g_timed_out = 1;
        g_path |= 16;
        return 1;
    }
    return 0;
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

// g = gcd(|a|, |b|), binary gcd (Stein); g has room for max(a.n, b.n) + 1 limbs
__device__ void mag_gcd(Ctx &c, Big &g, const Big &a, const Big &b) {
    if (a.n == 0) { big_copy(g, b); g.neg = 0; return; }
    if (b.n == 0) { big_copy(g, a); g.neg = 0; return; }
    int cap = (a.n > b.n ? a.n : b.n) + 2;
    Big x = big_new(c, cap), y = big_new(c, cap);
    if (c.fail) return;
    big_copy(x, a); x.neg = 0;
    big_copy(y, b); y.neg = 0;
    int zx = mag_ctz(x), zy = mag_ctz(y), shift = zx < zy ? zx : zy;
    mag_shr(x, zx);
    mag_shr(y, zy);
    int turns = 0;
    while (y.n) {
        if ((++turns & 255) == 0 && past_deadline()) { c.fail = 1; return; }
        int cmp = mag_cmp(x, y);
        if (cmp > 0) { Big t = x; x = y; y = t; }
        mag_sub(y, y, x);
        if (y.n) mag_shr(y, mag_ctz(y));
    }
    big_copy(g, x);
    for (int i = g.n; i < g.cap; i++) g.w[i] = 0;
    if (g.n + shift / 32 + 2 > g.cap) { c.fail = 1; return; }
    mag_shl(g, shift);
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

// Big from a run of digit values (0..9), 9 at a time
__device__ void big_from_digits(Big &r, const uint8_t *dig, int n) {
    r.n = 0;
    r.neg = 0;
    int i = 0, first = n % 9 ? n % 9 : 9;
    while (i < n) {
        if ((i & 511) == 0 && past_deadline()) { r.n = 0; return; }
        int take = i == 0 ? first : 9;
        uint32_t chunk = 0, mul = 1;
        for (int k = 0; k < take; k++) {
            chunk = chunk * 10 + dig[i + k];
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
        big_from_digits(r.n, num, nn + ndec);
        r.d.n = 1;
        r.d.w[0] = 1;
        for (int k = 0; k < ndec; k++) mag_muladd_small(r.d, 10, 0);
    } else {
        big_from_digits(r.n, num, nn);
        if (kind == 1) {
            big_from_digits(r.d, den, nd);
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
    for (int k = 0; k < digits; k++) {
        uint8_t ch = s[i + k];
        if (ch < '0' || ch > '9') return MIP_BAD_INPUT;
        dig[k] = ch - '0';
    }
    big_from_digits(r, dig, digits);
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
__global__ void worker(Slot *slot, uint32_t *arena, long long budget, int sleep_ns) {
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

// One launch per call (the comparison mode): the slot is in device memory.
__global__ void once(Slot *slot, uint32_t *arena) {
    extern __shared__ uint32_t sh_arena[];
    serve(slot, arena, sh_arena);
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
__global__ void once_mapped(Slot *slot, uint32_t *arena) {
    extern __shared__ uint32_t sh_arena[];
    if (__ldcv((const int *)&slot->state) == 1) serve_flag(slot, arena, sh_arena);   // else a lingering kernel already served it
    linger(slot, arena, sh_arena);
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
__global__ void once_param(Slot *slot, uint32_t *arena, ParamArgsT<W> pa) {
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
static int g_sleep_ns = 500;   // the worker's sleep between polls (MIP_NANOSLEEP)
static std::atomic<uint64_t> g_launches{0};

static int launch_worker() {
    worker<<<1, 1, SH_WORDS * sizeof(uint32_t), g_stream>>>(g_dev_slot, g_arena, g_budget, g_sleep_ns);
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
    if (const char *e = getenv("MIP_SPIN")) g_spin = atoi(e);
    if (const char *e = getenv("MIP_NANOSLEEP")) g_sleep_ns = atoi(e);
    if (const char *e = getenv("MIP_POLL")) { int m = atoi(e); cudaMemcpyToSymbol(g_poll_mode, &m, sizeof m); }
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device);
    g_clock_khz = clock_khz ? clock_khz : 1;
    g_budget = (long long)(clock_khz * 1000.0 * MIP_WORKER_SECONDS);
    { long long cc = (long long)(clock_khz * 1000.0 * MIP_CALL_SECONDS); cudaMemcpyToSymbol(g_call_cycles, &cc, sizeof cc); }
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
            once_mapped<<<1, 1, SH_WORDS * sizeof(uint32_t), g_stream>>>(g_dev_slot, g_arena);
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
            once_param<64><<<1, 1, g_param_smem_host ? SH_WORDS * sizeof(uint32_t) : 0, g_stream>>>(g_dev_slot, g_arena, pa);
            e = cudaGetLastError();
        } else {
            once_mapped<<<1, 1, SH_WORDS * sizeof(uint32_t), g_stream>>>(g_dev_slot, g_arena);
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
                        once_mapped<<<1, 1, SH_WORDS * sizeof(uint32_t), g_stream>>>(g_dev_slot, g_arena);
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
        once<<<1, 1, SH_WORDS * sizeof(uint32_t), g_stream>>>(g_dev_slot, g_arena);
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
