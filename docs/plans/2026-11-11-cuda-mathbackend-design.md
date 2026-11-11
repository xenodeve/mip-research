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
- **CUDA Graph:** the work of an operation is issued as a captured graph replayed with new parameters; data-dependent loops and branches
  use graph conditional nodes (CUDA 12.4+), so the host does not drive iterations.
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

- **Graph shape:** an operation may be one kernel inside a graph that loops on the device (recursion, memoisation, DPLL and Pollard rho
  are not practical as conditional nodes). The rule is that the host never drives an iteration, not that every loop is a graph node.
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
   faults before any CUDA exists.
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

## Open, for the developer

- *(the three questions below, answered in the conversation)*
