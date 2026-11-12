# Compute

What produces the exact value MIP puts after a `=`. There are two separate compute paths, and they share no code:

- **The best config (tir4b, Qwen3.5-4B, on vLLM or llama.cpp alike) computes on the CPU, in Python:** `mathexpr` reads the expression, `exactexpr` and the
  `mathbackend` registry compute it, all with the standard library's `int`, `Fraction` and `Decimal`. Nothing in this path runs on a GPU.
- **Strata's engine (`xeno/mip-sandbox`, Qwen3.8 Flash-Next / Gyro-M) computes on the GPU, with three typed CUDA executors** (integer
  `+ - ×`; integer power and, at the ABI only, floor division, mod and gcd; rational and decimal `+ - × ÷`). The engine has no MathBackend: of
  its 387 operations, none runs in the engine.

This page also lists the offline oracles that check both paths, the audit of every value put, and the compiled-circuit executor
(Qwenputer) that the typed CUDA executor replaced for arithmetic.

| Mechanism | What it does (one line) | Runs on | In best config? | Best measured result | Status |
|---|---|---|---|---|---|
| [MathBackend](#mathbackend) | 387 exact operations in 6 modules, each bounded, refusing rather than guessing | tir4b (vLLM/llama-server), CPU Python | Yes, through the inline bridge (`--mathbackend`) | 387 / 387 operations, 31,064 cases, 0 disagreements with SymPy / mpmath / NetworkX / brute force (re-run 2026-11-08: same totals) | in best config |
| [Inline bridge: mathexpr + exactexpr](#inline-bridge-mathexpr--exactexpr) | Reads the expression before `=` and computes it exactly: rationals, roots kept as roots, exact trig and logs, sums, `a^b mod m`, named number functions | tir4b, CPU Python | Yes | 0.18 ms per `=`; 3,896 new values agree with the model's next value on 6 arms it was not written from | in best config |
| [Audit of every value put](#audit-of-every-value-put) | Re-computes every put with an independent evaluator after the run | offline tool | Checks the best config | 0 wrong of 39,046 checked puts (A100, best-config arms); 0 false of 41,176 on the frozen H3 set | in best config (as its gate) |
| [CUDAMathBackend](#cudamathbackend) | The MathBackend's `numbers.add/sub/mul/div` on the GPU behind a C ABI, byte-identical to Python | mip-research `cuda/mathbackend` (branch `cuda-mathbackend`), any process via the DLL | No (development) | fuzz 4,000 / 4,000, paths 220 / 220 on both cards; 10.3 us a call, 5.8-6.5 us back to back; decode beside it -0.1 to -0.2 % (mode 2) | development |
| [ABI1: integer `+ - ×` on CUDA](#abi1-integer-----on-cuda) | One CUDA thread per request, up to 62-digit signed integers, no float | Strata engine (xeno/mip-sandbox, not in main) | No | 7 tests pass incl. 256 held-out cases; 0.066 / 0.093 / 0.085 ms median per call on 3 tasks | engine only |
| [ABI2: floor division, mod, gcd, power on CUDA](#abi2-floor-division-mod-gcd-power-on-cuda) | Integer primitives with explicit domain/overflow statuses | Strata engine (xeno/mip-sandbox, not in main) | No | 1,271 / 1,271 records byte-exact (GPU sanitizers clean) | engine only (only power is reached from text) |
| [ABI3: rationals and decimals on CUDA](#abi3-rationals-and-decimals-on-cuda) | Exact `+ - × ÷` on fractions and `+ - ×` on decimals, reduced result | Strata engine (xeno/mip-sandbox, not in main) | No | 1,403 / 1,403 records byte-exact (GPU sanitizers clean) | engine only |
| [Engine inline execution](#engine-inline-execution---igc-inline-exec) | Parses `A op B =` from token ids and runs it on the three executors inside the decode loop | Strata engine (xeno/mip-sandbox, not in main) | No | BigBench 3/5-digit `+ - ×`: 240/240 both arms; 5-digit multiplication 146 against 485 tokens | engine only |
| [Controlled runtime connection](#controlled-runtime-connection) | A typed request, matched by prefix, runs on CUDA and feeds the native output mapper | Strata engine (diagnostic) | No | 30 / 30 runtime results match the checked bytes | built, measured, kept as option |
| [Selective dispatch](#selective-dispatch) | Turns learned readouts into a bounded exact request on the device, or a valueless failure | Strata engine (diagnostic) | No | 64 fixture cases pass, sanitizers clean | built, not measured yet (no model run) |
| [Qwenputer (compiled circuit)](#qwenputer-compiled-circuit) | Arithmetic firmware compiled into sparse FFN/GDN tensors, run on CUDA | Strata engine (xeno/mip-sandbox, not in main) | No | 64 / 64 held-out cases, 183,052 steps exact; about 5 s per call | engine only (replaced for arithmetic by ABI1-3) |
| [Offline oracles](#offline-oracles) | `integer_extended_oracle.py`, `rational_oracle.py`, `sandbox_oracle.py`: independent references for the CUDA corpora and the sandbox | offline tool | No | see each executor; sandbox: 569 / 569 dev programs, 78 / 82 HumanEval match, 0 mismatches | built, measured, kept as option |

## MathBackend

**What it is.** A registry of exact operations in `../tools/igc/mathbackend/` (`numbers` 100, `geometry` 86, `algebra` 70,
`counting` 70, `search` 43, `symbolic` 18; 387 in all, counted from `registry()` on the current code). Each operation is registered with
`@op(name, kind, bound=...)` (`../tools/igc/mathbackend/__init__.py`) and called as `run('numbers.gcd', 12, 18)`.

**How it computes.** Standard library only: `int`, `fractions.Fraction`, `decimal.Decimal`, and `Surd`, an exact `p + q*sqrt(d)`. SymPy,
mpmath and NetworkX are imported only by the `sim_*.py` and test files, as the oracle. An input past an operation's bound raises
`NotCovered`; a malformed or out-of-domain input raises `BadInput`. The caller then puts nothing. The contract is
`Strata-xeno xeno/mip-sandbox: docs/plans/2026-11-05-mip-math-backend-coverage.md`.

**What the best config reaches.** The inline bridge calls the registry by name for 18 operations: gcd, lcm, totient, modular inverse,
binomial, falling factorial, Catalan, derangements, Bell, partition count, Stirling numbers of both kinds, Fibonacci, Lucas, Möbius, number
and sum of divisors, and one determinant (`algebra.mat_det`) (`../tools/igc/unified_detect.py`, `_CALLS`, `_GATED_OP`). `C(n,k)`, `P(n,k)`,
`\sigma`, `\tau`, `\phi`, `\mu` are read only when the text before names the function, and never on a line about points. The other
operations are reachable through the Python API only (for example from a sandbox program), not from the model's text.

**Measured.**
- **Against independent libraries:** `TOTAL 387/387, 31,064 cases, 0 disagreements` against SymPy, mpmath, NetworkX or brute force, with
  every operation covered by at least one case (mip-research#17). Unit tests: mathbackend 259 OK. [^sim-rerun]
- **Scrutiny (mip-research#17, three read-only reviewers):** 18 hand-written mutants all killed; `mat_det` against SymPy on 3,000 random
  cases; `max_flow` / `shortest_path` / `mst` / `assignment` against NetworkX or brute force on 5,200 cases; Surd on 3,000 operations. Every
  wrong value found there was fixed or turned into a refusal. [^scrutiny]
- **Bounds now in place (mip-research#17):** `isolate_real_roots` on Mignotte a=10^40 in 0.32 s; `mat_det` refuses past
  n³(nB)^1.5 > 1.2e11 in 0.01 s; `_squarefree` on a 14-digit prime in 0.002 s with 0 differences from trial division over 8,512 inputs;
  `max_flow` on 4,000 fractional paths in 0.02 s. [^open-bounds]

**Where it lives / to use it:** `../tools/igc/mathbackend/`; on in tir4b with `--inline-calc --mathbackend`.

## Inline bridge: mathexpr + exactexpr

**How it works.** At each `=` the model writes, tir4b cuts the text before it to the longest expression that ends there
(`../tools/igc/mathexpr.py`: ASCII, LaTeX at any depth, Unicode) and computes it as a `Fraction`. With `--mathbackend`,
`unified_detect.expression_before_with_mathbackend` first replaces each named function call by its MathBackend value, then hands the text
to `../tools/igc/exactexpr.py`, which computes in Q(√2, √3, ...): square roots kept as roots (`\sqrt{1674} = 3\sqrt{186}`), trig at multiples
of 15° and 36°, logs that come out (`\log_8 \frac{1}{4} = -2/3`), `\sum` / `\prod` over a range, `a^b mod m` by squaring, a product with no
sign (`4(117)`), `\binom`, `!!`, base-b numerals, repeating decimals. A value is exact or there is none: a free symbol, a word, π outside
an angle, a root of a root, a cube root that does not come out, or a value past `mathexpr.LIMIT` is refused. A function the text defined
(`f(2)` after `f(y) = ...`) is not read. [^fdef]

**Measured** (mip-research#17, stored Qwen3.5-4B traces, all bf16 but pooled across two engines: **4B bf16 · vLLM · A100** arms and
**4B bf16 GGUF · llama.cpp · 5060 Ti** arms (pc4, pc5); compared with what the model wrote next):

| Traces | `=` | Values, mathexpr alone | Values, with the bridge | New values: agree / no value after / differ |
|---|---|---|---|---|
| 6 control arms the rules were not written from (dm-base, base, pc4-base, lb1-base, a100 ctrlb, pc5-ctrl) | 129,964 | 28,202 | 32,464 | 3,896 / 252 / 114 |

The 114 that differ are decimals the model rounded or wanted (`2400 / 144 = 16.66`), slips of the model where the software's value is the
right one (`97_{16} = 143`, right 151), and functions the text had redefined. **Time:** 22.8 s for the 129,964 `=`, **0.18 ms per `=`**,
for the whole host bridge (reading, computing and refusing). [^cpu-time]

**Where it lives / to use it:** `../tools/igc/mathexpr.py`, `../tools/igc/exactexpr.py`, `../tools/igc/unified_detect.py`; on in tir4b with
`--inline-calc --mathbackend`.

## Audit of every value put

Two checks, each independent of MIP's own reader.

**Live, best config on the A100** (`../scripts/fail_audit.py`). Every `inline value` event in an arm's `records.json` is re-computed by
Python's `eval` over `Fraction` on a normalised copy of the expression (`\times` → `*`, `\frac{a}{b}` → `(a)/(b)`, thousands commas
dropped), then compared with the value put; a `q rem r` put is checked with `divmod`; a cut decimal `...` is checked to its last digit.
Expressions that are not plain arithmetic (roots, mod, functions, words) are counted `not plain` and not checked.

Re-run for this page on the A100 records of mip-research#19 (the stored `records.json` of each arm), Qwen3.5-4B bf16,
vLLM (loop-lcacv7: the 24 looping questions × 2 samples; loop-shape: 12 answers):

| Arm | Answers | Puts checked, agree | DIFFER | Not plain |
|---|---|---|---|---|
| loop-lcacv7 (inline + A + B + V7) | 48 | 19,406 | 0 | 1,084 |
| loop-shape (with the shape cap) | 12 | 19,640 | 0 | 124 |
| **Best-config arms** | 60 | **39,046** | **0** | 1,208 |

[^four-arms]

**Frozen set H3** (mip-research#19). 1,526 stored traces of models and sources no rule was written from (Qwen3-235B-A22B on OpenThoughts4
math 303, Gemini thinking on s1K-1.1 about 1,000, DeepSeek-R1 on s1K-1.1 non-math 123 and on OpenThoughts science 100), replayed through the
General Style at code `1d573ae3`, measured once before anyone read H3: **0 false puts of 41,176** (289 unchecked: values past the exact field
with no float, puts the checker cannot read) (mip-research#19, the H3 results table). The bug-gate comment of the same issue, with cut
decimals also checked against an independent float, gives 0 of 41,176 on H3, 0 of 39,264 on the seven 4B development arms and 0 of 48,158 on
H1+H2. [^truth-script]

**Where it lives / to use it:** `../scripts/fail_audit.py ARM...` over a folder `out/<arm>/records.json` (the A100 records are not in this
repo).

## CUDAMathBackend

**What it is.** The MathBackend's operations on the GPU, behind a C ABI that takes and returns the canonical bytes of
`../tools/igc/mathbackend/canon.py`, so equality with Python is a byte comparison. Ported: `numbers.add`, `sub`, `mul`, `div`; the other
383 return `UNKNOWN_OP`. **How to build, call and configure it, with the measured cost of each mode:
[`../cuda/mathbackend/README.md`](../cuda/mathbackend/README.md).** Design and every measurement:
[`plans/2026-11-11-cuda-mathbackend-design.md`](plans/2026-11-11-cuda-mathbackend-design.md).

**Measured** (RTX 5060 Ti and RTX 4070 SUPER, CUDA 13.3, Python 3.11.9 as the reference): `numbers-fuzz-s2` 4,000 / 4,000 and
`numbers-paths-s3` 220 / 220 byte-identical in every mode on both cards (the paths corpus drives Knuth division and its add-back step);
`numbers-s1` 1,816 / 1,816 (RTX 5060 Ti; the 4070 SUPER ran it at 1,800 before the 2026-11-12 changes); `numbers-gcd-s4` 101 / 101 and
`numbers-team-s5` 214 / 214 (RTX 5060 Ti, modes 1 and 2, with and without every case forced through the team); compute-sanitizer 0
errors (before the team; with it, synccheck and memcheck 0 errors). Big numbers run on a block of 256 threads (multiply, decimal
conversions, Lehmer's gcd steps): 20,000-digit integers 14 ms a call (one thread: 122 ms), still far slower than Python (11-23 us).
In a live run (tir4b General flags, 12 answers in flight, 2026-11-12) the pause of an inline put is 218 ms (4B NVFP4 W4A8 · llama.cpp
· RTX 4070 SUPER) and 273 ms (4B NVFP4 W4A8 · llama.cpp · RTX 5060 Ti) at the median, almost all the engine's stop and resume; MIP's host
work is 2.5-3.1 ms a put, 0.20-0.25 ms per generated token, and the CUDAMathBackend across processes adds 0.3-0.5 ms a put rather than
saving any (design, "The pause of an inline put"; [mip-research#25](https://github.com/xenodeve/mip-research/issues/25)). A call costs 10.3 us inside the DLL on an idle card (an empty kernel: 8.7), 5.8-6.5 us back to back with the
linger; beside llama-server decoding in another process ~175 us (as an empty kernel there), with decode unchanged (-0.1 to -0.2 %).

**Where it lives / to use it:** mode 2 (`MIP_CUDA_RESIDENT=2`) beside an engine in another process; mode 1 only inside the engine's own
process. **It is MIP's main computer; the Python MathBackend is the reference and the fallback, developed beside it** (the developer,
2026-11-12). `MIP_EXPR_DEVICE=cuda` sends the text readers' `+ − × ÷` (`../tools/igc/mathexpr.py`, `_arith`) through it, which keeps
that work off the CPU the developer uses; both arms of the token channel's development run used it, and every value they steered
checked right ([inject](inject.md#token-channel---channel)).

## ABI1: integer `+ - ×` on CUDA

**How it works.** `Strata-xeno xeno/mip-sandbox: include/strata/core/igc_exact.hpp` and `Strata-xeno xeno/mip-sandbox: src/kernels/igc_exact.cu`. A 160-byte request
holds two canonical signed decimal integers of at most 62 digits, the operation, an invocation id and a position; an 88-byte result holds
the value or a status. Up to 64 requests per launch, one CUDA thread per request, on an explicit created stream. The kernel validates
version, reserved fields, digit range and canonical form; malformed, unsupported and overflow (a result over 62 digits) return an empty
value, distinct from a numeric zero. No float, no atomics, no host arithmetic, no per-call allocation.

**Measured.**
- `Strata-xeno xeno/mip-sandbox: docs/reports/2026-10-29-igc-i1a.md`: **7 tests pass**, including **256 frozen held-out signed cases** against independent Python
  integers; memcheck / initcheck / synccheck 0 errors and racecheck 0 hazards on a 64-record sequence. RTX 5060 Ti.
- `Strata-xeno xeno/mip-sandbox: docs/reports/2026-10-29-igc-component-benchmark.md` (2026-10-29, RTX 5060 Ti, CUDA 13.3, driver 621.01, HIGH priority, warm, two ABBA
  blocks, 4 calls per cell; the interval includes H2D copy, the kernel, the result copy and the wait; parsing excluded):

| Task | Exact CUDA, ms min / median / max | Qwenputer, ms median |
|---|---|---|
| `17*23 = 391` | 0.0469 / **0.06635** / 0.0886 | 5,079.0054 |
| `-123+456 = 333` | 0.0726 / **0.09265** / 0.1502 | 6,018.5617 |
| `1000-1 = 999` | 0.0475 / **0.08490** / 0.1234 | 4,946.5520 |

Fresh process, `17*23`: first call after setup 0.83-0.87 ms; explicit device buffers 248 bytes. [^cuda-time]

**Where it lives / to use it:** opt-in test target `STRATA_BUILD_IGC_EXACT_TESTS=ON` (`igc_exact_run`); in the model only through
[engine inline execution](#engine-inline-execution---igc-inline-exec).

## ABI2: floor division, mod, gcd, power on CUDA

**How it works.** `Strata-xeno xeno/mip-sandbox: include/strata/core/igc_integer_extended.hpp`, `Strata-xeno xeno/mip-sandbox: src/kernels/igc_integer_extended.cu`. Same 160/88-byte records and
62-digit operands as ABI1, with its own version tag (each ABI refuses the other's records). Floor division rounds toward −∞; mod takes the
divisor's sign (`a = b*q + r`, `|r| < |b|`); `gcd` is non-negative with `gcd(0,0) = 0`; `0**0 = 1`; a negative power of a non-unit base is
unsupported; division or mod by zero is a domain error. Euclid is capped at 498 steps, powers at exponent 206 overflow, decimal long
division uses at most nine trial subtractions per digit. Values never wrap or clip.

**Measured** (`Strata-xeno xeno/mip-sandbox: docs/reports/2026-10-29-igc-integer-extended.md`, Windows, CUDA 13.3.73, driver 621.01). Corpus02, 1,271 requests built by
`../tools/igc/integer_extended_oracle.py` from Python's `divmod`, `math.gcd` and `pow` and compared after the native run:

| Mode | Result |
|---|---|
| RTX 5060 Ti, direct batch 1 and 64; graph batch 1 ×2; graph batch 64 ×64 | 1,271 / 1,271 each, exact bytes |
| memcheck / initcheck / synccheck / racecheck, graph batch 64 ×2 | 1,271 / 1,271, 0 errors / 0 hazards |
| RTX 4070 SUPER, graph batch 64 ×2 | 1,271 / 1,271, exact bytes |

Values and failure statuses are both compared byte for byte (corpus01, also 1,271 requests and passing all regular modes, is 957 values
and 314 malformed / unsupported / overflow / domain records; the report does not give corpus02's split). [^abi2-note]

**Where it lives / to use it:** opt-in `STRATA_BUILD_IGC_INTEGER_EXTENDED_TESTS=ON`. From the model's text only **power** reaches it
(`Strata-xeno xeno/mip-sandbox: include/strata/program/igc_learned_diagnostic.hpp:578-589`); floor division, mod and gcd would need the engine's token parser (`Strata-xeno xeno/mip-sandbox: include/strata/core/igc_inline.hpp`) to emit them.

## ABI3: rationals and decimals on CUDA

**How it works.** `Strata-xeno xeno/mip-sandbox: include/strata/core/igc_rational_exact.hpp`, `Strata-xeno xeno/mip-sandbox: src/kernels/igc_rational_exact.cu`. A 304-byte request and 152-byte
result. Rationals (domain 2) take signed, unreduced numerators and denominators; decimals (domain 3) are a coefficient times 10^-scale,
scale 0..61. Exact `+ - × ÷` on rationals and `+ - ×` on decimals; the result is a reduced fraction with a positive denominator, zero as
`0/1`. Cross products use 126-digit work buffers and are reduced before the final 62-digit check; Euclid capped at 1,010 steps. No
rounding: a decimal result is an exact fraction, and rounding would need an explicit mode that does not exist.

**Measured** (`Strata-xeno xeno/mip-sandbox: docs/reports/2026-10-29-igc-rational-exact.md`, Windows, CUDA 13.3.73, driver 621.01). Frozen corpus of 1,403 records from
`../tools/igc/rational_oracle.py` (Python `Fraction`): 850 values, 375 overflow, 12 domain errors, 157 malformed, 9 unsupported:

| Mode | Result |
|---|---|
| RTX 5060 Ti, direct batch 1 and 64; graph batch 1 ×2; graph batch 64 ×8 | 1,403 / 1,403 each |
| memcheck / initcheck / synccheck / racecheck, graph batch 64 ×2 | 1,403 / 1,403, 0 errors / 0 hazards |
| RTX 4070 SUPER, graph batch 64 ×2 | 1,403 / 1,403 |

Examples: `1/3 + 1/6 = 1/2`, `(-2/3) * (9/-4) = 3/2`, decimal `0.1 + 0.2 = 3/10`. [^abi3-note]

**Where it lives / to use it:** opt-in `STRATA_BUILD_IGC_RATIONAL_TESTS=ON`; in the model through engine inline execution for any
expression with a decimal or a division.

## Engine inline execution (`--igc-inline-exec`)

**How it works.** `Strata-xeno xeno/mip-sandbox: include/strata/core/igc_inline.hpp` parses `A op B =` (and longer expressions, in their own precedence, as a postfix
program) straight from the Gyro-M pack's token ids: each digit is one token, `\times`, `\cdot`, `\div` are two. Parsing is on the host;
the value always comes from the device. In `Strata-xeno xeno/mip-sandbox: include/strata/program/igc_learned_diagnostic.hpp:563-630` each step of the program is copied
to the device, run, copied back and synchronised (`cudaStreamSynchronize`):

- all integers, no division: `+ - ×` on ABI1 (`:592`), integer power on ABI2 (`:585`);
- any decimal or division: ABI3 (`:617`); a fractional power is left to the model.

The value's tokens are then forced into the decode, with a source note inside the thinking (`--igc-no-note` drops it).

**Measured** (Strata-xeno#276, comment `igc:ladder-rung1-20261031`; Qwen3.8 Gyro-M on the D2x profile, engine `a4b6e4521ee5c0ec`, same session, scorer
`../tools/igc/external_score.py`):

| Suite | Computation on | Off |
|---|---|---|
| BigBench 3-digit and 5-digit add / sub / mul, 40 each | 240 / 240 | 240 / 240 |
| Tokens, 5-digit multiplication (total per case) | 146 | 485 |
| Tokens, 3-digit multiplication | 68 | 121 |

[^engine-row]

**Where it lives / to use it:** a CUDA build of xeno/mip-sandbox with `-DSTRATA_BUILD_IGC_HEAD_PROBE=ON` (it defines
`STRATA_IGC_NATIVE_DIAGNOSTIC`), run with `--igc-learned-diagnostic-bundle PATH --igc-inline-exec` (`Strata-xeno xeno/mip-sandbox: src/program/generate.cpp:2819`,
`:3870-3880`; the inline path lives in the learned-diagnostic object, so it needs a bundle). Not in main.

## Controlled runtime connection

**How it works.** A version-2 fixture holds the canonical prefix SHA256 and a 160-byte ABI1 request (operands and operation, no answer).
At the matched request's first decode window the engine runs one ABI1 execution and broadcasts the device result to the verifier's rows;
the learned native output mapper and Q8 head read it on the same stream, with no host readback
(`Strata-xeno xeno/mip-sandbox: docs/plans/2026-10-29-igc-controlled-runtime-executor.md`, `Strata-xeno xeno/mip-sandbox: docs/reports/2026-10-29-igc-runtime-output-connection.md`).

**Measured.** All 30 typed runtime results match the earlier independently checked ABI1 bytes; broadcast tests clean under all four
sanitizers. On the Hub Gyro-M 262K profile, 30 known questions at temperature 0 / max 96 tokens, the runtime-on arm gave 24 / 24 training
and 6 / 6 development questions right against 23 / 24 and 6 / 6 with it off. [^runtime-note]

**Where it lives / to use it:** diagnostic build, `--igc-output-diagnostic-artifact` and `--igc-output-diagnostic-fixture`; the operands
come from the fixture, not from the model.

## Selective dispatch

**How it works.** `SelectionInputs` reads two learned readouts and a readiness head on the device; only when readiness holds, both readouts
are finite and agree on all nine fields, and the operands are canonical one- or two-digit integers does it write a fresh 160-byte ABI1
request. Every other case writes an inactive record that the executor turns into a valueless failure, so a failed dispatch never becomes a
numeric zero (`Strata-xeno xeno/mip-sandbox: src/kernels/igc_dispatch.cu`, `Strata-xeno xeno/mip-sandbox: docs/reports/2026-10-29-igc-selective-dispatch.md`).

**Measured.** 64 fixture cases pass, normal and graph A/B/A, hard-off byte preservation; memcheck / initcheck / synccheck 0 errors,
racecheck 0 hazards. A component check only: no model run. [^dispatch-note]

**Where it lives / to use it:** internal API of the diagnostic build; needs the learned readouts (see the learned catalog).

## Qwenputer (compiled circuit)

**How it works.** The pinned upstream Qwenputer (divagr18/qwenputer `ca13310e`) compiles a RISC-V calculator firmware into a sparse
FFN/GDN tensor circuit. Strata reads it as a data-only `STQPUT01` artifact (`Strata-xeno xeno/mip-sandbox: docs/QWENPUTER_FORMAT.md`: 64-byte header, sorted
sections, SHA256 per section, 256 MiB limit), validates it on the CPU before any GPU allocation (`Strata-xeno xeno/mip-sandbox: src/core/qwenputer_artifact.cpp`), and
runs it on CUDA as a WHILE graph (`Strata-xeno xeno/mip-sandbox: src/kernels/qwenputer.cu`, `Strata-xeno xeno/mip-sandbox: src/kernels/qwenputer_runner.cu`). No host calculator or opcode emulator supplies the
predictions.

**Measured** (`Strata-xeno xeno/mip-sandbox: docs/reports/2026-10-28-qwenputer-p3-progress.md`, `Strata-xeno xeno/mip-sandbox: docs/reports/2026-10-28-qwenputer-p4-progress.md`, RTX 5060 Ti):
- P3, supplied-trace replay: 79 cases, 74,704 steps, 0 residual / state / prediction mismatches.
- P4, autonomous from a text prefix: 79 development cases, 74,624 steps pass; **frozen held-out 64 / 64, 183,052 steps**, repeated with the
  final binary with identical outputs and state hashes.
- Cost per call: about 5 s (`17*23`: 5,079 ms median, 1,543 circuit evaluations), against 0.066 ms for ABI1 on the same task
  (component benchmark above). This is why arithmetic moved to the typed executors. [^qwenputer-note]

**Where it lives / to use it:** `STRATA_BUILD_QWENPUTER_HOST_TESTS`, `Strata-xeno xeno/mip-sandbox: tests/qwenputer/`; the decoder hooks (P5) are not built, so the model
cannot call it.

## Offline oracles

- `../tools/igc/integer_extended_oracle.py` and `../tools/igc/rational_oracle.py`: build the ABI2 / ABI3 request corpora and their expected
  results from Python big integers and `Fraction`, and compare them with the native output **after** the native run. The native process
  receives only `requests.bin`; no expected value enters it. Results under ABI2 and ABI3 above.
- `../tools/igc/sandbox_oracle.py`: runs each program under `python -I -B -X utf8` and under the sandbox binary; the two must agree on
  stdout and on the class of an uncaught error. Against CPython 3.11.9: **569 / 569 development programs**; HumanEval canonical solutions,
  odd tasks held out and run once on a frozen header: **78 of 82 match, 2 refused, 2 at the step limit, 0 mismatches** (commit `8197cb5`).
  Details in the sandbox catalog.

## Other files in this area

- `../tools/igc/contest_math.py`: AIME 2025 / 2026, HMMT Feb 2025 and CMIMC 2025 rows from MathArena, scored by reading the last `\boxed{}`
  with `mathexpr` and requiring an exact match. A benchmark loader, not a compute path (bench catalog).
- `../tools/igc/cuda_head.py`: a CUDA forward with a late-head STE backward for the learned native head. It computes no arithmetic
  value (learned catalog).

[^sim-rerun]: Re-run for this page on 2026-11-08 on the current code (`python -m tools.igc.mathbackend.sim`, defaults seed 1,
    n 60 per generator; SymPy 1.14.0, mpmath 1.3.0, NetworkX 3.6.1): `TOTAL 387/387, 31064 cases, 30384 answered, 30384 agree, 20 refused,
    517 expected, 0 disagree`. The 20 refusals are numeric limits, integrals and series that do not converge to the digits asked, reported
    as refusals, not values. "0 disagreements" says nothing about an input no generator produces.
[^scrutiny]: Fixed in commits `a48a0570`, `e2285304`, `67ca97d5` (xeno/mip-sandbox). Among the wrong values found and fixed: thousands
    separators split into arguments (`gcd(1,000, 25)` gave 1), Greek letters read as number functions, a determinant taken as the value of
    the whole text, `Vmatrix` read as a determinant. Two operations written after single contest questions were removed
    (`algebra.triangle_wave_intersections`, `geometry.polar_triangle_chain`), and the problem-named operations are out of the inline path.
    The defect of mip-research#14 (`P(2, 3) =` and `C(4, 6) =` put 0; a restatement of a dodecagon question put the gold 315) no longer
    reproduces: all three texts return no value on the current code (checked 2026-11-08).
[^open-bounds]: Still open in mip-research#17: `canon('(x+y+z+w)^200')` takes 15-20 s before it refuses; `count_real_roots`,
    `sturm_sequence` and `poly_gcd` have no work budget (dense degree 200 at 20 bits, about 3-5 s). Neither is reachable from the model's
    text. Conventions a model may not share are documented, not changed: `stirling1` unsigned by default, `sigma(n, k)` takes n first,
    `falling_factorial(5, 7) = 0`.
[^fdef]: Reading a function the text defined was built and then taken out: a model redefines `f` or defines it by pieces, and reading its
    latest `f(x) = ...` put a wrong value at 11 places of the stored control answers, with no rule that told them apart
    (`exactexpr.py` docstring, Strata-xeno#306 = mip-research#18). Also measured and not built: expanding a product of sums (the model wrote
    the expanded form next in 141 of 394 cases) and substituting a value set on the same line (62 agree, 259 differ) (mip-research#17).
[^cpu-time]: 0.18 ms is the host bridge over every `=` of stored traces, including the ones it refuses; it is not one operation's time and
    is not comparable with the CUDA per-call time below. On the set the rules were made on (a100_262k control, 27,627 `=`), 11 new values
    differ, 1 of them a real wrong value (`w(0)`, a function the text defined; that reading is now removed, see [^fdef]).
[^four-arms]: The same script over four loop arms of round 7 (`loop-lc`, `loop-lcac`, `loop-lcacv7`, `loop-shape`,
    156 answers) gives 51,015 checked, 0 DIFFER, 2,850 not plain; that sum includes the two arms without V7 and the shape cap. Over the
    General-Style development arms the script flags 7 puts (`fresh-gen` 4, `fresh-qwen` 2, `v2-gen` 1), all of the form
    `1.18 \times 10^9 / 4 \times 10^3 = 295000`: the checker's copy reads the line left to right, the put follows the scientific-notation
    grouping, and the put equals `(1.18·10^9)/(4·10^3)` exactly in all 7 (checked with `Fraction`). These arms are not the best config.
    1,208 puts of the best-config arms (roots, mod, functions) are outside what this re-check evaluates.
[^truth-script]: The H3 gate is the issue's `truth_audit.py`, which is in neither mip-research nor xeno/mip-sandbox; the 41,176 is quoted
    from mip-research#19 and was not re-run here. It covers the General Style replay, not a live best-config run.
[^cuda-time]: Four observations per cell; "max" is the sample maximum, not a tail estimate. It is a standalone component measurement:
    operands arrive already typed, so parsing, the language model and the injection are outside the interval. No learned mapper, model or
    MTP took part. Not a CPU-against-GPU comparison with the 0.18 ms of the bridge.
[^abi2-note]: A first sandboxed memcheck failed to attach (`No attachable process found`) and is not counted; the run in the native
    execution context is the one reported. HIP, SYCL and Linux are not run. No latency was measured for ABI2.
[^abi3-note]: Functional qualification, not a speed result. Decimal division, scales above 61, mixed formats and rational powers, floor,
    mod and gcd are unsupported. HIP, SYCL, Linux and device-fault injection are not run.
[^engine-row]: This run used an engine build before the parser fix `af41da64`. The same comment found the parser delivering a wrong value
    on two rows (`math-24`: `a+b=-1+5=` delivered 6, right 4; `gsm8k-82`: `4 \times 125 - 2 =` delivered 123) because a sign merged into the
    token before a number and `\times` was unknown; the fix refuses after 1,010 binding tokens and makes `\times`, `\cdot`, `\div`, `·`,
    `−` real operators. On the same build MATH-500 integers went 55 with against 59 without. The word-number sidecar was on and fired 0
    times. A later rung on GSM8K + MATH (Strata-xeno#276, comment `igc:ladder-rung2-scorer-20261101`) found 16 more wrong delivered
    values of 957 (13 a number written against a letter, `d12^2=` read as `12^2`; 2 a number after `√`; 1 a list read as thousands); the
    fix is commit `1be41906`, and later engine runs use it. No BigBench rerun on a fixed build was found. The
    "0 wrong values" of the audit above is a tir4b claim; no equivalent audit of the current engine path was found. Runs on this PC do
    not repeat token for token, so the arms are compared as totals. Engine results with time (5-digit multiplication 2.80 / 2.82 against 6.25 / 6.61 s per answer in an ABBA,
    Strata-xeno#276) are on the inject and benchmarks pages.
[^runtime-note]: Raw-token parity against the build without the hook failed, including on an unmatched Thai control; the same binary
    already varied before the hook, so the cause is not established. Operands come from an oracle fixture, so this is not learned
    extraction and not a quality or speed result.
[^dispatch-note]: The selection policy was motivated by 130 / 288 correct agreed development tuples; that is not a guarantee on unseen
    tasks, and the sealed acceptance set was not opened.
[^qwenputer-note]: WHILE-graph memcheck / synccheck / racecheck are blocked by an installed-toolchain instrumentation failure (initcheck
    passes); this debt is open. Of the 29 math cases, 21 are solved expressions and 8 are explicit firmware errors or refusals. P5
    (ownership and state) is partial and the decoder is not integrated. HIP / SYCL are not validated.
