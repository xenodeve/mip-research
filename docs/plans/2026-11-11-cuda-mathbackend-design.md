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

**Build:** `cuda/mathbackend/mipmath.cu` at `02f2587` (branch `cuda-mathbackend`), CUDA 13.3, sm_89 + sm_120; RTX 5060 Ti alone, no
llama-server on the card; Python 3.11.9. Ported: `numbers.add`, `sub`, `mul`, `div` (exact rationals; Python's `Fraction(str)` grammar
with Unicode digits and whitespace on the device), one device thread per call.

| Check | Result |
|---|---|
| Corpus `results/cuda-mathbackend/corpus/numbers-s1.jsonl` (1,416 cases: 800 seeded, 616 boundary) | **1,416 / 1,416 byte-identical**, refusals 269 / 269, in both modes (resident worker, one launch per call) |
| The answer came from the device | an unported op returns `device:UNKNOWN_OP`; the process holds the 5060 Ti; a 2^200 product matches |
| Null backend / wrong backends | 100 % / each caught (`tools/test_mathbackend_harness.py`) |

**Speed per call** (`results/cuda-mathbackend/speed.py`, 30 repeats, median; `speed-resident1.txt`, `speed-resident0.txt`):

| | Python | Device, resident worker | Device, one launch per call |
|---|---|---|---|
| all 1,416 cases | 24.8 us | 440 us (18x slower) | 656 us (Python 37.0 in that run) |
| `add`, small arguments | 14-19 us | 294-384 us | 454-510 us |
| 4,000-character strings | 192-194 us | 71,900 us (370x slower) | |

Why, read from the code, not yet profiled: one device thread does everything, and every limb, digit and code point lives in global
memory (hundreds of cycles per dependent access, nothing to hide it behind); each character of a string is matched against the 66
digit runs and 29 spaces in constant memory; strings become integers digit-chunk by digit-chunk (quadratic). In the resident mode the
host also spins on the slot while it waits, so the CPU is not yet freed either. The latency floor of a mapped-memory round trip is a
few microseconds, so a single small call can at best come near Python; the gain the design is after (no CPU work, host time per decode
round) needs the next steps: an ASCII fast path and registers or shared memory for small values, a warp per call for big ones, many
calls per pass (one slot per warp), and a host that waits without spinning. Then measure again, per call, batched, and host time per
decode round.
