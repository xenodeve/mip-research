# Design: CUDAMathBackend, the MathBackend on CUDA Graph (2026-11-11)

## Intent (the developer's words, 2026-11-11)

- "Move our software's processing onto the GPU entirely, without disturbing the CPU." Strata-xeno decodes with the CPU too (the CPU pool
  computes experts), and MIP must not compete for it.
- "Reduce latency during decode as well. If the engine decodes very fast, MIP makes it slower, because MIP on the CPU is the critical
  path." Speed is wanted anyway.
- "The port to CUDA must match 100% first; develop further only once it is confirmed complete, the same as Python." Python is the
  oracle that tests the new CUDA Graph code; the new code is called **CUDAMathBackend**.
- Agreed: it lives in **mip-research**, beside the Python MathBackend.

Standing decisions that apply (memory and `AGENTS.md`): every typed operation picks its backend, GPU first, the host only where the GPU
cannot yet do it, and says which is which; both paths are kept and compared on one measure; permissively licensed code (MIT, BSD,
Apache-2.0, ISC, BSL-1.0, zlib, PSF) may be copied with its notice kept in `third_party/mathbackend/README.md`; copyleft is never
copied; **linking a library as a dependency needs a conversation with the developer first**.

## What exists

- **Python MathBackend** (`tools/igc/mathbackend/`): 387 operations in 6 modules: `numbers` 100, `geometry` 86, `algebra` 70,
  `counting` 70, `search` 43, `symbolic` 18. Standard library only: `int` (unbounded), `fractions.Fraction`, `decimal.Decimal`, and
  `Surd` (exact `p + q*sqrt(d)`). Each op is registered with `@op(name, kind, bound)`, called as `run(name, *args)`, and refuses with
  `NotCovered` (past its bound) or `BadInput` (outside its domain). Bounds as written: 55 ops "any size", 21 "degree <= 2000", others
  such as "n up to 2,000,000 bits", "digits <= 2000", "<= 1e6 values", "p prime < 3.3e24". Source use: 94 ops use `Fraction`, 22
  `Decimal`, 21 `Surd`, 47 dictionaries/graphs/heaps, 25 return lists, 7 return strings, 3 touch floats.
- **Oracle evidence:** 387/387 ops, 31,064 cases, 0 disagreements against SymPy, mpmath, NetworkX or brute force (`sim_*.py`,
  mip-research#17); unit tests 259 OK.
- **CUDA in the Strata engine** (`xeno/mip-sandbox`, not in main): ABI1 integer `+ - ×` (one thread per request, up to 62-digit
  integers, 0.066-0.093 ms median per call), ABI2 floor division, mod, gcd, power, ABI3 rationals and decimals `+ - × ÷`, each
  byte-exact against its oracle with sanitizers clean (`docs/compute.md`). None of the 387 MathBackend ops runs on a GPU today.
- **Toolchain:** CUDA 13.3; RTX 5060 Ti (sm_120) and RTX 4070 SUPER (sm_89, the display card, its VRAM headroom rule applies); Windows
  now, Linux too.

## Goals

1. **CUDAMathBackend**: every one of the 387 operations runs on the GPU, issued as CUDA Graphs, behind a C ABI that any engine can call.
2. **100% agreement with the Python MathBackend**, case by case, on values **and** on refusals: where Python returns a value, CUDA
   returns the same value in the same type and form; where Python raises `NotCovered` or `BadInput`, CUDA refuses with the same class.
3. **No CPU work in the computation.** The host only enqueues (or, in a later phase, the engine's own graph contains the nodes); no
   host-side arithmetic, parsing of values, or fallback in the GPU path.
4. A verification harness that proves goal 2 and stays in the repository as the gate for every later change.

## Phases (order decided by the developer)

1. **Port** the 387 ops, one module at a time, to 100% agreement. No new behaviour.
2. **Confirm**: the full harness at 100% on every op, sanitizers clean, then the developer's confirmation that it is complete.
3. **Develop further** (each its own design): speed; inside the decode loop of Strata-xeno (Qwen3.6-35B-A3B first); and moving MIP's
   other software, the text readers (`mathexpr`, `question_read`, the `=` detector) and the loop checks, onto the GPU as well (the
   developer: "all of our software").

## Non-goals during phases 1-2

- No change to any operation's logic, bound or output form on the CUDA side alone. A bug found in Python during the port is fixed in
  Python first (with its test and oracle case), then ported; the two never differ on purpose.
- No new operations; no performance work beyond what is needed to pass the harness in reasonable time.
- Not yet the text readers or the loop checks (phase 3), and not yet the engine integration.

## Anti-goals (a failure even if it is fast)

- Any case where CUDA and Python differ, in value, type, form or refusal.
- CUDA returning a value where Python refuses (an unverified value put into a model's reasoning).
- Floating point standing in for exact arithmetic anywhere Python is exact.
- CPU work hidden inside the "GPU" path (host loops, host bignum, a silent host fallback).

## How, as decided so far

- **Location:** `cuda/mathbackend/` in mip-research (my call on the folder name): CUDA C++ sources, a C ABI header, a build for sm_89 and
  sm_120, and the harness under `tools/` beside the Python oracle.
- **Interface (my call, the builder may refine):** a C ABI that takes an operation id and its arguments in a canonical byte encoding and
  returns a status (`OK`, `NOT_COVERED`, `BAD_INPUT`) and the result in the same canonical encoding. Python defines the encoding for every
  value type it returns (int, Fraction, Decimal with its exponent, Surd, lists, tuples, matrices, strings, dicts), so equality is a byte
  comparison of the two encodings.
- **CUDA Graph:** the work of an operation is issued as a captured graph replayed with new parameters. The rule is that the host never
  drives an iteration, not that every loop is a graph node: a data-dependent loop may be a graph conditional node (CUDA 12.4+) or one
  kernel that loops on the device (recursion, memoisation, DPLL and Pollard rho are not practical as conditional nodes).
- **Launch: a resident kernel, relaunched every ~1 s** (the developer, 2026-11-11). A worker kernel stays on the device and polls a
  request queue in mapped host memory, so a call costs a PCIe round trip instead of a launch (the engine's ABI1-3 measured
  0.066-0.093 ms per launched call). Each launch of the worker ends itself after about one second and the host launches the next at
  once, so no kernel outlives Windows' TDR watchdog (2 s under WDDM; every GeForce card runs WDDM); the registry is not touched. To
  measure before it is relied on: the call latency against a launch per call and against Python; and, beside llama-server on the same
  card, how much decode slows, since two processes' contexts time-share the GPU on Windows (no MPS). Inside an engine's own process
  the worker runs on a low-priority stream of that context instead, which is the module's end shape.
- **Values cross the ABI as digits, not limbs** (proposed 2026-11-11, for the developer). Turning a decimal string into limbs is bignum arithmetic, so it runs on the device
  (goal 3); the canonical encoding carries integers as sign + decimal digits, and the device parses and prints them.
- **A fourth status, `DEVICE_FAILED`** (proposed 2026-11-11, for the developer), for what is not the op's answer: out of device memory, a kernel fault, or a device loop that
  reached its step cap. The step cap is a function of the input (a count of iterations fixed per op), never a clock, so it is the same
  on both cards. The caller treats `DEVICE_FAILED` as "no value": nothing is put, the event is logged, and there is no silent host
  fallback (anti-goal). The harness always counts it as a disagreement: Python has no such status, so identity of refusals means a
  `DEVICE_FAILED` on any corpus case fails the gate.
- **Exact arithmetic on the device:** multi-limb integers, fractions as reduced pairs, decimals with Python's `decimal` semantics for the
  context each op sets (precision, rounding), and `Surd`.
- **Verification harness:** Python generates the cases (the 31,064 existing ones, plus new cases at and past each op's bound), runs both
  backends, and compares the canonical bytes; it reports per-op coverage (cases, agreements, refusals matched) and must read 100% on every
  op. `compute-sanitizer` memcheck and racecheck clean. Results go in `docs/compute.md` with the build named.

## Left to the builder

- Kernel design, limb size, memory layout, graph granularity (one graph per op, per family, or per call shape).
- The order of modules within phase 1 (a suggestion: `numbers` first, since most ops the model's text reaches are there, then
  `counting`, `algebra`, `geometry`, `search`, `symbolic`).
- How to reuse ABI1-3's code and tests from the Strata engine.
- Test layout and the CI-style command that runs the harness.

## Calls made after the builder check (mine; the developer can overrule)

- **Corpus:** the cases are generated once from the `sim_*.py` seeds and frozen with Python's canonical outputs as a versioned corpus
  file; a seed change makes a new corpus, never a silent one.
- **Reference Python:** pinned to the version on this PC (3.11), recorded in the corpus; an exception other than `NotCovered` /
  `BadInput` (`ValueError` past 4,300 digits, `InvalidOperation`, `ZeroDivisionError`, `RecursionError`, `MemoryError`) is a Python bug,
  fixed in Python first (its test and its corpus case), never a third ABI status.
- **Floating point inside Python ops** (`complex_roots` pre-filter, `geometry` pre-filters, a `math.log` bound, `_det`'s float cost
  estimate): made exact in Python first, so no float decides an output or a refusal on either side.

## How "100%" is made and proven (calls after review, 2026-11-11)

1. **The harness comes before any kernel, and is proven with a null backend.** First: the frozen corpus (cases, Python's canonical
   outputs, refusal class), a canonical byte encoding for every return type, a comparator with per-op coverage. Then Python itself is
   run through the C ABI's encode -> decode path as a "null backend" and must read 100%; this shakes out encoding and normalisation
   faults before any CUDA exists. **The harness must also be shown to fail:** deliberately wrong backends (a value off by one, the
   wrong refusal class, an unreduced `Fraction`, a reordered list, a `Decimal` with another exponent, a refusal where Python answers)
   each run through it and must read below 100 % on the ops they touch. A harness that only ever passed has not been tested.
2. **What 100% covers, in writing:** identity on the frozen corpus **plus an adversarial boundary set**, not a proof over all inputs;
   differential fuzzing (new random cases, Python against CUDA) stays on as a standing gate. The boundary set is built on purpose, since
   seeded generators under-sample it: each op's bound and bound ± 1; zero, negatives and sign edges; carries across limb boundaries
   (multiples of 2^32 and 2^64); Decimal exponent extremes; degenerate geometry (collinear, zero area, coincident points).
3. **Normalisation rules the encoding assumes, written per type:** `Fraction` reduced with the sign on the numerator; `Surd` in its
   canonical form; for each of the 22 Decimal ops, the exact context it sets (precision, rounding, Emax/Emin) is frozen and ported, not
   general `decimal`; `int` -> `str` and Python's 4,300-digit limit for the 7 string-returning ops; list and dict order; negative zero.
4. **Both cards:** the corpus must pass on sm_89 and sm_120 alike; no atomics or reduction orders that can change a result.
5. **Measure the critical path before ordering the ops** (AGENTS.md: timeline first, no guessing). One served run of Strata-xeno with
   `STRATA_TIMELINE` gives MIP's host time per decode round (`parse_inline`, the ABI calls); stored traces give which ops the model's
   text reaches. Phase 1's order follows those numbers. **The acceptance measure for the developer's goal is fixed now:** host time per
   decode round that MIP causes (target: the graph launch alone) and tokens per second with MIP on against off, ABBA on the served
   profile, beside the 100% gate. Phase 1 can be 100% correct and change nothing the developer asked for unless this is measured.
   **The decision rule, fixed before the measurement:** the timeline splits MIP's host time per decode round into MathBackend calls
   and the rest (the text readers, the loop checks). If MathBackend calls are a small part of it, porting them first does not reach
   the developer's goal, and phase 3's text readers move ahead of phase 1's tail; the share that decides it is the developer's to set
   (proposed: under a third of MIP's host time). The measure also counts the two costs a GPU path adds: decode slowed by MIP's kernels
   on the same card, and the sync that reads a value back before it is put into the stream. One served run answers this, so it is
   proposed **before** the design is approved.
6. **A change to Python is a change to the reference live rounds use.** Making the float pre-filters exact and adding bounds (open
   question 3) changes Python's own outputs and refusals, so it passes its own gate first: the 31,064 oracle cases again, and a replay
   of the stored traces showing no put is lost or changed (`sim_replay`), before the corpus is frozen from it.

## Open, for the developer

1. **Ops that take text** (`symbolic` 18, and the ops whose argument is an expression or a callable). Proposal: phase 1 ports the ops
   whose arguments are numbers only, with the excluded ops named in a list in the repository; the text ops move to phase 3 with the
   text readers. **If taken,** goal 1 and phases 1-2 read "every op whose arguments are numbers (N of 387, list in
   `cuda/mathbackend/EXCLUDED`)", "complete" in phase 2 means that set at 100 %, and the excluded ops stay host-only and are labelled
   so, as the standing decision "the host only where the GPU cannot yet, and says which" requires; goal 3 applies to the ported set.
2. **Ops with more than one right answer** (SAT, CSP, shortest path, LP, about a third of `search`). Proposal A: two harness modes,
   identity (same bytes) where Python's answer is unique, and validity (the answer checked against the problem by Python) where it is
   not. Proposal B: identity everywhere, by porting the same algorithm single-threaded, so the device returns the answer Python does.
   B keeps "100 % the same as Python" literal; A allows parallel search.
3. **Ops "of any size"** (55). Proposal: give each a bound in Python first, written as a function of the input (bits, digits, a count
   of values or of search steps), never as time or memory, since a clock or a memory limit refuses differently on each card and on
   Python and breaks refusal identity. The sizes are the developer's to set; the device's memory then follows from the largest input
   the bounds allow. The new bounds pass the gate in point 6 above.

## Progress (2026-11-11)

**Build:** `cuda/mathbackend/mipmath.cu` on branch `cuda-mathbackend` (the commit of this section), CUDA 13.3, sm_89 + sm_120; RTX 5060 Ti alone, no
llama-server on the card; Python 3.11.9. Ported: `numbers.add`, `sub`, `mul`, `div` (exact rationals; Python's `Fraction(str)` grammar
with Unicode digits and whitespace on the device), one device thread per call.

Correct per path (`results/cuda-mathbackend/path_cases.py census`: the device reports which paths each call took):

| Corpus | Agree | Register path | Full path | Knuth D (multi-limb division) | Its add-back step |
|---|---|---|---|---|---|
| `numbers-paths-s3` (220, built to force the rare branches; add-back vectors found by mirroring the device's Knuth D) | **220 / 220** | 24 | 196 | 194 | **29** |
| `numbers-fuzz-s2` (4,000 random, 1-20 digits) | **4,000 / 4,000** | 2,182 | 1,818 | 35 | 0 |
| `numbers-s1` (1,816 seeded + boundary) | **1,800 / 1,816** | 945 | 823 | 6 | 0 |

The 16 of `numbers-s1` are the boundary cases with a 20,000-digit integer: the device answers `DEVICE_FAILED` (never a value), because
one thread's quadratic arithmetic does not finish within the call's time guard. Before the guard the same input came back as
**BadInput** where Python answers: a string longer than the shared-memory arena failed to copy, the failure was not flagged, and the
parser read on into a refusal (fixed: a resource failure now retries on the global arena and never turns into a refusal; the boundary
set gained 5,000- and 20,000-digit integers). Both modes agree; the 4070 (sm_89) and `compute-sanitizer` have not run yet.

**Time guard.** Windows resets the driver for the whole machine when a kernel runs past ~2 s. Every long loop checks a deadline set
when the call starts (`MIP_CALL_SECONDS` 0.4 of the rated clock) and ends the call as `DEVICE_FAILED`; checked with 20,000- and
50,000-digit products (~0.41-0.45 s, then the card serves the next call). A clock is not the input-defined cap the design asks for
(open question 3): it keeps the machine safe until that cap exists, and the cap, or faster big-integer kernels, decides the 16 above.

**Speed per call** (`results/cuda-mathbackend/speed.py`, resident worker, RTX 5060 Ti alone at Idle priority, Python 3.11.9 in the same
process, medians; `speed7-*.txt`). Correct on every run: `numbers-s1` 1,768 / 1,768 and the fuzz corpus `numbers-fuzz-s2` 4,000 / 4,000,
both modes.

| Arguments | Python | Device, round trip | Of it, computing on the device |
|---|---|---|---|
| parts of at most 9 digits (register path) | 6.5-11.5 us | 14.9-16.8 us | 7.1-7.7 us |
| 10-18 digits (register path, two-word arithmetic) | 8.7-12.0 us | 18.9-21.7 us | 9.3-12.6 us |
| 19-40 characters (full path) | 7.7-13.5 us | 51-70 us | 39-64 us |
| 300 digits (full path) | 9-11 us | 0.46-1.0 ms | 0.38-0.85 ms |
| 4,000-character strings (full path) | 102 us | 70 ms | 76 ms |
| all `numbers-s1` cases | 10.5 us | 18.4 us | 8.7 us |

How it got there (`results/cuda-mathbackend/ping.py`, `numbers.add 1 + 1`): the round trip itself is **9.5 us** from writing the request
to reading the answer in the DLL (3.1 us until the device's seen-flag comes back); the device polls the flag every 1.4 us. The first
speed tables (440 us per call) were **the Python wrapper**, not the device: `ctypes` `.raw` copied the whole 1 MB output buffer on every
call (~310 us); it now reads only the bytes written. A launch per call costs 248 us (the WDDM driver), so the resident worker stays.
The register path (the device's own time from 16.7 to 7.4 us for small calls) uses native 64-bit operations whenever the two-word
values fit one word. Moving the full path's working memory to shared memory changed little: its cost is not the arena (to profile:
register spills to local memory, the parsing).

Next: the full path for 19-200 digits (profile it first), a warp per call for big integers, several calls per pass, then the other
`numbers` ops; then the measure the design fixed, host time per decode round beside llama-server.

### The acceptance measure, first half: decode beside the worker (2026-11-11)

`results/cuda-mathbackend/impact.py`: one llama-server session on the RTX 5060 Ti (4B NVFP4 W4A8 · llama.cpp, `-np 4`, 4 constant
256-token decodes, no `--prio`), and in another process the MathBackend DLL on the same card, in 60 s windows ordered
none / keep / busy / busy / keep / none (keep: one call every 0.5 s; busy: calls back to back). `impact-ns500.txt`, `impact-ns0.txt`,
`impact-perlaunch.txt`:

| Mode | none | keep | busy |
|---|---|---|---|
| resident worker, `MIP_NANOSLEEP` 500 | 233 tok/s | 130 (**-44 %**) | 112 (-52 %) |
| resident worker, `MIP_NANOSLEEP` 0 | 237 | 121 (**-49 %**) | 116 (-51 %) |
| one launch per call | 223 | 223 (**-0.1 %**) | 205 (-8.3 %, at ~1,230 calls/s) |

A resident worker in another process halves the engine's decode even while it only polls: Windows time-shares the GPU between the two
processes' contexts (no MPS under WDDM), and a polling kernel always wants its share; sleeping between polls does not change that.
A launch per call costs nothing while idle and 8 % under a flood of calls, at 248 us a call. So, beside an engine in another process
(the PC layer, tir4b over llama-server), the launch-per-call mode is the one to use; the resident worker belongs inside the engine's
own context, on a low-priority stream, where it shares the card without time-slicing. The developer's choice of the resident worker
(2026-11-11) was made before this measurement and is the developer's to revisit.

### What the model's text reaches (2026-11-11)

Read in the code: tir4b's inline values evaluate arithmetic in `mathexpr` (Python `Fraction`) and call `mathbackend.run()` only when the
text names a function (`\gcd`, `\lcm`, `totient`, `comb`, `modinv`, `mat_det`, ...): `numbers.add` and the like are not on that path.
Counted on about a tenth of round 9's General traces (32 answers, General flags, `sim_replay --recorded`): `run()` **1** call
(`numbers.falling_factorial`), `mathexpr.evaluate` **32,335** calls. The counts are calls in tir4b on the PC layer, not time in Strata's
C++ decode loop: that half of the measure is the `STRATA_TIMELINE` run, still to do. By the decision rule above, the text readers (phase
3) come before the rest of phase 1: open question 1 (text input) is now the one that decides the port's order, not question 3.

### Latency, launch per call without a resident kernel (2026-11-11)

Mode 2 (`MIP_CUDA_RESIDENT=2`): the slot is mapped host memory, each call is one launch, and the kernel raises the slot's flag itself,
so the host makes no copy and no synchronising driver call; arguments of up to 256 bytes travel in the launch parameters
(`once_param`), so the register path reads nothing over PCIe. `results/cuda-mathbackend/ping.py`, `numbers.add 1 + 1`, RTX 5060 Ti:

| Mode | Median per call | Device's own time |
|---|---|---|
| launch per call with copies, blocking sync (mode 0, before) | 240 us | 3.0 us |
| mode 0, spinning sync | 114 us | 3.0 us |
| mode 2 | 19-21 us | 5.2 us |
| mode 2 as a CUDA graph | 20.6 us | 5.2 us |
| **mode 2, arguments in the launch parameters** | **15.2-16.0 us** | **2.5 us** |
| resident worker (mode 1), for reference | 12 us | 4.7 us |

Where mode 2's 15.5 us goes (DLL clock): the Python wrapper 1.1 us, the launch call 5 us (the WDDM driver), launch to flag 8 us
(the GPU starting the kernel, 2.5 us of work, the flag over PCIe). Neither shared memory nor word-wise output writes moved it.
Correct in mode 2 on every corpus (`numbers-fuzz-s2` 4,000 / 4,000, `numbers-paths-s3` 220 / 220, `numbers-s1` 1,800 / 1,816 with the
16 time-guard cases). Beside llama-server (`impact-mode2.txt`): none 232.7 tok/s, keep -0.2 %, busy -8.0 % at ~5,200 calls/s.
Mode 2 is the PC layer's mode: the latency of the resident worker within 4 us, at no cost to decode while idle.

**Linger** (`MIP_LINGER_US`): after answering, the mode-2 kernel keeps polling the slot for the window, so a request that follows soon
needs no launch; the host sets the flag, fences, reads `lingering`, and launches only when no kernel waits (the kernel clears
`lingering`, fences, and reads the flag once more: Dekker, so a request is neither lost nor served twice; a request number keeps a
kernel from answering with an older launch's parameters).

| | Back-to-back median | Launches for 5,000 calls | Beside llama-server: keep / busy |
|---|---|---|---|
| mode 2, no linger | 16.5 us | 5,001 | -0.2 % / -8.0 % (~5,200 calls/s) |
| **mode 2, linger 50 us** | **10.1 us** | 12 | +5.0 % (noise) / -28.4 % (~22,000 calls/s) |
| mode 2, linger 200 us | 10.2 us | 6 | |

`linger_stress.py`: 60,000 calls `add(i, 1)` with gaps drawn from 0-30, 0-120 and 0-400 us around the window's end: **0 wrong answers**;
a lingering kernel served 91 %, 19 % and 4 % of them. All three corpora as before with the linger on. The window costs nothing while
calls are sparse and, during a dense burst, holds the GPU like the resident worker; it is a knob for bursty callers, off by default.

Tried and dropped (same session, the committed build re-measured beside each): the lingering kernel reading small arguments in 16-byte
vectors and writing the answer through a local buffer in 4-byte words: 10.1 -> 11.2 us (device 5.2 -> 6.5 us). Posted byte writes to
mapped memory do not stall the thread; the local buffer's round trip does.

**Further levers, each A/B'd twice on the 5060 Ti (`ping.py`, 5,000 calls; same session):**

| Lever | Before | After | Kept |
|---|---|---|---|
| Python wrapper: the op's bytes cached, the length cell made once, no `bytes()` copy | wrapper 1.1 us, single 15.5 | **0.8 us, single 13.8 us** | yes |
| doorbell: `op` and length in the same 8-byte word as the flag the lingering kernel polls (one PCIe read fewer) | linger 10.0 us, device 5.2 | **8.7-9.1 us, device 3.9-4.2** | yes |
| linger poll sleep 100 ns -> 0 (`MIP_LINGER_SLEEP`) | 10.2 / 10.3 | 10.0 / 10.1 | knob, default 100 |
| host spin 1,000,000 -> 2,000 polls before yielding | 10.0 | 9.9-10.0 | no change |
| 16 uncached (`ld.cv`) loads of the arguments in flight | device 5.4 us | **21 us**: each uncached read of mapped memory is its own round trip | dropped |
| request aligned to a 32-byte sector | 10.0 / 13.9 | 10.0 / 13.9 | harmless, kept |
| answer through a local buffer in 4-byte words | 8.9-9.1 | 9.0-9.2 | dropped |
| a 64-byte parameter block instead of 268 bytes | single 13.8-13.9 | 13.9-14.0 | dropped |

After them: a single call **13.8 us** (the launch call ~5 us and the kernel's start ~5 us are WDDM's), back to back with the linger
**8.7-9.1 us**. `linger_stress.py` 60,000 racing calls: 0 wrong; every corpus as before, linger on and off.
