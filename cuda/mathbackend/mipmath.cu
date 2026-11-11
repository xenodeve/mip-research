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

// ---- limits --------------------------------------------------------------------------------------------------------------------------
// MIP_LIMBS caps one integer at 8192 limbs (262,144 bits). Python's add/mul/... are "any size": a bound in bits both sides keep is open
// question 3 of the design; past this cap the device answers NOT_COVERED, which the corpus check shows as a disagreement until Python
// refuses at the same bound.
#define MIP_LIMBS 8192
#define ARENA_WORDS (MIP_LIMBS * 64)
#define MIP_STR_MAX 4000          // numbers._rat: a string longer than 4000 characters (after strip) is BadInput
#define MIP_WORKER_SECONDS 1.0

enum { OP_ADD = 1, OP_SUB, OP_MUL, OP_DIV };

struct Slot {
    volatile int32_t state;       // 0 empty, 1 request, 2 answered, 3 quit
    int32_t op;
    int32_t status;
    uint32_t args_len;
    uint32_t out_len;
    uint8_t args[MIP_ARGS_MAX];
    uint8_t out[MIP_OUT_MAX];
};

// ---- big integers (sign and magnitude, little-endian 32-bit limbs, no leading zero limbs; zero has n == 0) ------------------------------
struct Big {
    uint32_t *w;
    int n;
    int neg;
    int cap;
};

struct Ctx {
    uint32_t *arena;
    int used;
    int fail;   // a device failure: arena exhausted (never a value)
};

__device__ Big big_new(Ctx &c, int cap) {
    Big b;
    b.n = 0;
    b.neg = 0;
    b.cap = cap;
    if (c.used + cap > ARENA_WORDS) {
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
    for (int j = m; j >= 0; j--) {
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
    while (y.n) {
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
    if (c.used + MIP_STR_MAX + 1 > ARENA_WORDS) { c.fail = 1; return MIP_DEVICE_FAILED; }
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
    if (c.used + need > ARENA_WORDS) { c.fail = 1; return MIP_DEVICE_FAILED; }
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
    if (c.used + digits / 4 + 2 > ARENA_WORDS) { c.fail = 1; return MIP_DEVICE_FAILED; }
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

__device__ void put(Out &o, const char *s) {
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
        if (c.used + nc + 1 > ARENA_WORDS) { c.fail = 1; return; }
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
__device__ int skip_ws(const uint8_t *s, int len, int i) {
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
        int room = (ARENA_WORDS - c.used) * 4 / 2;
        int bl = json_string(s, len, i, buf, room);
        if (bl < 0) return MIP_DEVICE_FAILED;
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
                    int l2 = json_string(s, len, i, b2, (ARENA_WORDS - c.used) * 4 / 2);
                    if (l2 < 0) return MIP_DEVICE_FAILED;
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
// optionally '/' digits or '.' digits) whose parts have at most 9 digits, the call runs in registers with 64-bit integers and no
// arena. Anything else (Unicode, whitespace, underscores, a zero denominator or divisor, a malformed string, a longer number, another
// type) is not judged here: the call goes to the full path, which holds Python's rules and its order of refusals. So the fast path only
// answers calls the full path would answer with a value. Parts below 10^9 keep each product below 10^18 and a sum of two below
// 2 * 10^18 < 2^63. (MSVC, nvcc's host compiler here, has no __int128, so 18-digit parts would need a two-word type.)
typedef long long i128;
typedef unsigned long long u128;
#define FAST_DIGITS 9

__device__ int fast_digits(const uint8_t *s, int a, int b, long long &v) {   // s[a..b): 1 to FAST_DIGITS ASCII digits
    if (b - a < 1 || b - a > FAST_DIGITS) return 0;
    long long x = 0;
    for (int k = a; k < b; k++) {
        if (s[k] < '0' || s[k] > '9') return 0;
        x = x * 10 + (s[k] - '0');
    }
    v = x;
    return 1;
}

// the end of a JSON string's bytes starting at p, if it holds no escape; -1 otherwise
__device__ int fast_string_end(const uint8_t *s, int len, int p) {
    int q = p;
    while (q < len && s[q] != '"' && s[q] != '\\') q++;
    return q < len && s[q] == '"' ? q : -1;
}

// one argument item at s[i] as n / d, d > 0 (not reduced); i moves past the item. 0: not a fast argument.
__device__ int fast_arg(const uint8_t *s, int len, int &i, long long &n, long long &d) {
    i = skip_ws(s, len, i);
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
    int e = skip_ws(s, len, q + 1);
    if (e >= len || s[e] != ']') return 0;
    i = e + 1;
    return 1;
}

__device__ int ctz128(u128 x) { return __ffsll((long long)x) - 1; }   // x nonzero

__device__ u128 gcd128(u128 a, u128 b) {   // binary gcd, both nonzero
    int sh = ctz128(a | b);
    a >>= ctz128(a);
    while (b) {
        b >>= ctz128(b);
        if (a > b) { u128 t = a; a = b; b = t; }
        b -= a;
    }
    return a << sh;
}

__device__ void put_u128(Out &o, u128 v) {
    char t[24], buf[24];
    int n = 0;
    do { t[n++] = (char)('0' + (int)(v % 10)); v /= 10; } while (v);
    for (int k = 0; k < n; k++) buf[k] = t[n - 1 - k];
    buf[n] = 0;
    put(o, buf);
}

// 1: answered into o; 0: not a fast call (o untouched)
__device__ int process_fast(int op, const uint8_t *s, int len, Out &o) {
    int i = skip_ws(s, len, 0);
    if (i + 4 > len || s[i] != '[' || s[i + 1] != '"' || s[i + 2] != 'l' || s[i + 3] != '"') return 0;
    i = skip_ws(s, len, i + 4);
    if (i >= len || s[i] != ',') return 0;
    i = skip_ws(s, len, i + 1);
    if (i >= len || s[i] != '[') return 0;
    i++;
    long long an, ad, bn, bd;
    if (!fast_arg(s, len, i, an, ad)) return 0;
    i = skip_ws(s, len, i);
    if (i >= len || s[i] != ',') return 0;
    if (!fast_arg(s, len, ++i, bn, bd)) return 0;
    i = skip_ws(s, len, i);
    if (i >= len || s[i] != ']') return 0;
    i = skip_ws(s, len, i + 1);
    if (i >= len || s[i] != ']' || skip_ws(s, len, i + 1) != len) return 0;
    i128 N, D;
    if (op == OP_ADD) { N = (i128)an * bd + (i128)bn * ad; D = (i128)ad * bd; }
    else if (op == OP_SUB) { N = (i128)an * bd - (i128)bn * ad; D = (i128)ad * bd; }
    else if (op == OP_MUL) { N = (i128)an * bn; D = (i128)ad * bd; }
    else {
        if (bn == 0) return 0;   // division by zero: the full path refuses it
        N = (i128)an * bd;
        D = (i128)ad * bn;
        if (D < 0) { N = -N; D = -D; }
    }
    int neg = N < 0;
    u128 un = neg ? (u128)(-N) : (u128)N, ud = (u128)D;
    if (un == 0) {
        put(o, "[\"i\",\"0\"]");
        return !o.over;
    }
    u128 g = gcd128(un, ud);
    un /= g;
    ud /= g;
    put(o, ud == 1 ? (neg ? "[\"i\",\"-" : "[\"i\",\"") : (neg ? "[\"q\",\"-" : "[\"q\",\""));
    put_u128(o, un);
    if (ud != 1) {
        put(o, "\",\"");
        put_u128(o, ud);
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
    if (c.fail) return MIP_DEVICE_FAILED;
    i = skip_ws(s, len, i);
    if (s[i] != ',') return MIP_BAD_INPUT;   // not two arguments: a TypeError in Python, never in the corpus
    i++;
    int sb = read_rat(c, b, s, len, i);
    if (c.fail) return MIP_DEVICE_FAILED;
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

__device__ void serve(Slot *slot, uint32_t *arena) {
    Ctx c;
    c.arena = arena;
    c.used = 0;
    c.fail = 0;
    Out o;
    o.p = slot->out;
    o.n = 0;
    o.cap = MIP_OUT_MAX;
    o.over = 0;
    if (slot->args_len <= 256) {   // small arguments: a local copy (L1-cached), then the register path
        uint32_t loc[64];
        uint32_t w = (slot->args_len + 3) / 4;
        const uint32_t *src0 = (const uint32_t *)slot->args;
        for (uint32_t k = 0; k < w; k++) loc[k] = src0[k];
        if (process_fast(slot->op, (const uint8_t *)loc, (int)slot->args_len, o)) {
            slot->status = MIP_OK;
            slot->out_len = o.n;
            return;
        }
        o.n = 0;
        o.over = 0;
    }
    // the arguments are read many times while parsing: copy them once into device memory in 4-byte words (in the resident mode the
    // slot is mapped host memory, and every byte read there is a PCIe transaction)
    uint32_t len = slot->args_len, words = (len + 3) / 4;
    if (words + 16 > ARENA_WORDS) { slot->status = MIP_DEVICE_FAILED; slot->out_len = 0; return; }
    const uint32_t *src = (const uint32_t *)slot->args;
    for (uint32_t k = 0; k < words; k++) arena[k] = src[k];
    c.used = words + 1;
    int st = process(c, slot->op, (const uint8_t *)arena, (int)len, o);
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

// The resident worker: polls the slot until about `budget` clock cycles have passed, then ends (the host launches the next).
__global__ void worker(Slot *slot, uint32_t *arena, long long budget) {
    long long t0 = clock64();
    for (;;) {
        int st = slot->state;
        if (st == 1) {
            __threadfence_system();
            serve(slot, arena);
            __threadfence_system();
            slot->state = 2;   // a call does not reset the budget: the worker still ends on time
        } else if (st == 3) {
            return;
        }
        if (clock64() - t0 > budget) return;
        __nanosleep(500);
    }
}

// One launch per call (the comparison mode): the slot is in device memory.
__global__ void once(Slot *slot, uint32_t *arena) { serve(slot, arena); }

// ---- host ----------------------------------------------------------------------------------------------------------------------------
static std::mutex g_lock;
static int g_open = 0, g_resident = 1;
static Slot *g_host_slot = nullptr, *g_dev_slot = nullptr;   // resident: mapped pinned memory; per call: device memory
static uint32_t *g_arena = nullptr;
static cudaStream_t g_stream;
static cudaEvent_t g_done;
static long long g_budget = 0;
static int g_running = 0;
static int g_spin = 2000;   // polls before the host starts yielding the core (MIP_SPIN)
static std::atomic<uint64_t> g_launches{0};

static int launch_worker() {
    worker<<<1, 1, 0, g_stream>>>(g_dev_slot, g_arena, g_budget);
    if (cudaGetLastError() != cudaSuccess) return -1;
    cudaEventRecord(g_done, g_stream);
    g_running = 1;
    g_launches++;
    return 0;
}

static int worker_alive() {
    if (!g_running) return 0;
    if (cudaEventQuery(g_done) == cudaSuccess) g_running = 0;
    return g_running;
}

extern "C" MIP_API int mip_open(int device, int resident) {
    std::lock_guard<std::mutex> l(g_lock);
    if (g_open) return MIP_OK;
    if (cudaSetDevice(device) != cudaSuccess) return MIP_HOST_ERROR;
    // one launch per call: the host blocks in the driver while the kernel runs instead of spinning
    cudaSetDeviceFlags(cudaDeviceMapHost | (resident ? cudaDeviceScheduleAuto : cudaDeviceScheduleBlockingSync));
    if (const char *e = getenv("MIP_SPIN")) g_spin = atoi(e);
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device);
    g_budget = (long long)(clock_khz * 1000.0 * MIP_WORKER_SECONDS);
    if (cudaMalloc(&g_arena, sizeof(uint32_t) * ARENA_WORDS) != cudaSuccess) return MIP_HOST_ERROR;
    int lo, hi;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    cudaStreamCreateWithPriority(&g_stream, cudaStreamNonBlocking, lo);   // the lowest priority: the engine's work goes first
    cudaEventCreateWithFlags(&g_done, cudaEventDisableTiming);
    g_resident = resident;
    if (resident) {
        if (cudaHostAlloc(&g_host_slot, sizeof(Slot), cudaHostAllocMapped) != cudaSuccess) return MIP_HOST_ERROR;
        cudaHostGetDevicePointer(&g_dev_slot, g_host_slot, 0);
        g_host_slot->state = 0;
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
    if (g_resident) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        h->state = 1;
        if (!worker_alive() && launch_worker()) return MIP_HOST_ERROR;
        int spins = 0;
        while (h->state != 2) {
            // poll a little (a small call answers in microseconds), then give the core away between polls
            if (++spins > g_spin) std::this_thread::yield();
            if (!worker_alive()) {   // the worker ended (its budget) before it took the request: launch the next
                if (h->state == 2) break;
                if (cudaGetLastError() != cudaSuccess) return MIP_DEVICE_FAILED;
                if (launch_worker()) return MIP_HOST_ERROR;
            }
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        h->state = 0;
    } else {
        size_t head = offsetof(Slot, args) + args_len;
        if (cudaMemcpyAsync(g_dev_slot, h, head, cudaMemcpyHostToDevice, g_stream) != cudaSuccess) return MIP_HOST_ERROR;
        once<<<1, 1, 0, g_stream>>>(g_dev_slot, g_arena);
        g_launches++;
        if (cudaMemcpyAsync(&h->status, &g_dev_slot->status, sizeof(int32_t) * 3, cudaMemcpyDeviceToHost, g_stream) != cudaSuccess)
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
