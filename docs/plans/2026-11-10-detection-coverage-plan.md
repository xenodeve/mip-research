# Plan: detection, every detector, every case it is for (2026-11-10)

**The developer, 2026-11-10:** test detection from every detector we have, covering as many cases as possible. Detectors cover each
other's blind spots and are used together. **A detector that does not fire on a case outside its condition is not a bad detector;
the case is just not its condition**, the way the question reader is for questions that state a computation. And widen each
detector's scope where it can be widened.

So each detector is judged **inside its own condition**, and the set is judged by what **all of them together** catch. This plan
says how, gives the numbers measured today, and lists every detector with its status, its gap and its next step.

**Build of every number here:** 4B NVFP4 W4A8 · llama.cpp · PC layer, round 9 (232 of 240 pairs; a2 pending), unless a row names
another build. Measured offline on stored records, no GPU.

## 1. How a detector is measured

Every detector gets three numbers, each with its denominator:

1. **Coverage inside its condition:** caught / should have caught, on the cases it is for.
2. **Precision:** right fires / fires, from a hand-read sample (stratified: one fire per answer first, so a few long answers cannot
   make the figure).
3. **Silence outside its condition:** fires on cases it is not for. Zero for a detector that **acts**.

Detectors are of two kinds, the contract `UnifiedDetector` already names ([detect](../detect.md#unifieddetector-fusion)):

- **Acts:** it puts text into the stream (the `=` reader, the question reader, the hand-value check, the loop kinds). It must be
  exact: a false fire is a wrong put, which the General Style forbids.
- **Arms only:** it raises a signal that lets an exact reader act sooner or look harder (probes, readiness, the MoE router, the
  drafter's draft). It may be imprecise, because nothing is put on its word alone.

## 2. What should be caught (the events)

| Event | Example | Detectors whose condition it is |
|---|---|---|
| E1 a calculation written with `=` | `$2928 / 28 =` | `=` reader (+ MathBackend reader) |
| E2 a calculation written with `\approx` or `\equiv` | `$73388 / 36903 \approx 1.988653$`, `$71451 \equiv 2 \pmod{13}$` | hand-value check (after the line); **no stop yet** |
| E3 a calculation in words or in the head, no relation written | `which gives 58` | probes, readiness, drafter (arm); `=` reader cannot |
| E4 the question copied or set up wrong | `dm_mul_div_longer-1505#0`: 331 right values on a wrong setup | question reader |
| E5 a hand value that is wrong | `$(-8)(-9) - 7(2) = 58 - 14 = 44$` | hand-value check |
| E6 a loop, by kind (A-F) | `Wait, $72 - 14 = 58$.` x12 | loop shapes + loop kind classifier |
| E7 a calculation about to start | 2 tokens before the expression | probes, MoE router, drafter |
| E8 a re-check about to start | `Wait,` coming | drafter's draft, probes |
| E9 the model ready to close its thinking | `</think>` near the top of the distribution | ThinkBrake reading (`n_probs`), drafter |

A detector is scored only on its own events. The **union matrix** (event x detector: which fired, which missed, which fired
wrongly) is built from hand-labelled samples per event, the way the 69 loop kinds were, and it is what shows the blind spots.

## 3. Measured today

### `=` reader (E1), in the arm

Points at `=` with an expression the arm's reader (`--mathbackend`) computes: **14,609 + 17,352** in the two General halves;
software put the value at **98 %** (14,278 and 17,062). The rest, **331 + 290**, were left to the model. Not split by cause yet; the
records carry `echo_skips` (329 / 357) and `joins_held` (12,049 / 13,951), so the split is a measurement to make, not a number here.
(`coverage.py --mathbackend`; without the flag the tool undercounts: roots, pi and quotients are not points to the plain reader.)

### Hand-value check (E5, and E2 after the fact), off in the arm

`mathexpr.wrong_relations` run over the model's own lines finds **334** relations it calls wrong in General (64 answers) and **428**
in control; **0** were corrected, because `--check-hand` was off. One of the 334 answers is `stdmath-fbf10ed4#1`, the answer the
close hurt: its cross product was wrong four times before the close (`$(-8)(-9) - 7(2) = 58 - 14 = 44$`).

**Read, one flag from each of the 64 answers** (the rest, 270, not read):

| Read as | n | What it is |
|---|---|---|
| a real wrong value | **51** | 14 a chain that adds a term to a total already reached (`$7^2 + 24^2 = 625 + 576 = 1201$`); 12 decimals wrong in the digits (`\approx 1.988653`, true 1.988673); 10 residues wrong (`$71451 \equiv 2 \pmod{13}$`, true 3); 5 a rough value of the wrong size (`$41815 \div 5934 \approx ... \approx 700$`); 10 other arithmetic (`$\lfloor 10/2 \rfloor + \lfloor 10/4 \rfloor + \lfloor 10/8 \rfloor = 8 + 2 + 1 = 11$`) |
| **not an error** | **12** | 7 the long-multiplication carry (`$8 \times 3 = 24 + 2 = 26$`: product plus carry, true under that reading); 5 the parser splitting a chain wrongly (`$2.25(65) - 144 = 2.25(64) + 2.25 - 144$`, true) |
| asked, not stated | 1 | `= 17 - 128 + 81$?` |

**Precision 51 of 64 (80 %), read one per answer.** Because the check acts, the 12 are 12 wrong puts: the carry notation and the
chain split must be fixed **before** `--check-hand` runs anywhere (a gate, not a backlog item).

**Where the 334 sit:** 37 on `\equiv` lines and 79 on `\approx` lines in General (91 and 66 in control): **116 of 334 are relations
the `=` stop never stops at**. That is the blind spot of the `=` reader that the hand-value check covers, and a scope to widen
(section 5).

### Question reader (E4), off in the arm

On frozen9 it reads **30 of 120** questions, every one to gold's exact value: all 10 of `dm_div_big`, `dm_mixed_longer` and
`dm_mul_div_longer`. The other 90 are outside its condition (`dm_conversion`, `dm_div_remainder`, `dm_poly_evaluate`,
`dm_simplify_surd`, `dm_swr_p_sequence`, and the 40 MATH level-5 word problems), so its silence there is correct. **It was off in
round 9 with no reason recorded**: built for arithmetic mode (thinking off, 160 / 160, 4 of 4 copy errors caught in pc9), it was not
carried into the General arm. Queued below.

### Loop shapes and loop kinds (E6)

Shapes: the loop checks of [anti-loop](../antiloop.md), on in the arm. Kinds: the v1 classifier agrees with the hand labels on 55 of
69 ([decide layer 4a](2026-11-10-decide-layer-design.md)).

## 4. Every detector, with its status in the arm

Status is one of three: **on**; **off, with the measured reason**; **queued, with its slot**. A built mechanism with none of the
three is the gap that let the question reader drop out unnoticed.

| Detector | Acts / arms | Condition (its cases) | Runs on | Status in round 9's General arm | Today | Gap and scope | Next |
|---|---|---|---|---|---|---|---|
| `=` reader + MathBackend reader | acts | E1 | tir4b | **on** | 98 % of 31,961 points put | `\approx`, `\equiv` and words are not stops | split the 621 misses by cause; stop at `\approx` / `\equiv` (5) |
| Question reader (`--question-value`) | acts | E4: a question that states one computation | tir4b | **queued, slot 1** (no reason recorded for off) | 30 / 120 frozen9, all gold | 5 DeepMind module forms and LaTeX expressions it does not read | on, on development rows; widen (5) |
| Hand-value check (`--check-hand`) | acts | E5, and E2 after the line | tir4b | **queued, slot 2, after the gate** | 334 flags, 0 corrected; 51 / 64 real | 12 / 64 false: carry notation, chain split | fix both with tests, re-read, then on |
| Loop shapes (repeat, counting, line-back, V7, shape cap, cover cap) | acts (close) | E6 | tir4b | **on** | replay 0 of 915 / 1,078 right touched | shape only, not kind | with the kind classifier |
| Loop kind classifier (decide layer v1) | acts (settle) | E6 by kind | tir4b, offline | **queued** (settle probes running) | 55 / 69 | 8 misses fall back to close | per-kind pairing from the probes |
| ThinkBrake reading (`n_probs`) | arms (close timing) | E9 | llama-server | **queued** (settle probe 2) | none yet | new | the probe's `brake` arm |
| Engine parser `parse_inline`, interception | acts | E1 inside the decode loop | Strata engine | **off: engine only**; not on llama-server | 48 / 48 word problems (Flash-Next) | not on the 4B | with the move into the engine |
| Residual probes | arms | E3, E7, E8 | HF / engine | **off: needs the hidden state**, which llama-server and vLLM do not expose | 89 % at k = 2 (4B, offline) | not live | with the move into the engine |
| Readiness classifier | arms | E7 | Strata engine | **off: engine only, diagnostic** | 111 / 112 (Gyro-M) | not on the 4B | with the move into the engine |
| MoE router | arms | E7 | engine | **off: the 4B is dense** | 83 % at k = 2 (Qwen3.6, offline) | not live | with Qwen3.6-35B-A3B |
| Drafter's draft (MTP) as a reading | arms | E7, E8, E9 | engine | **off: not built** (the engine only drops drafts after a put) | none | new | build with the move into the engine |
| `UnifiedDetector` fusion | the rule between the two kinds | all | offline | **off: built, not measured** | none | | when two arming signals are live |
| Sidecar (natural-language firing) | acts | E3 in words | Strata engine | **off: no result** | none | | after the engine-side detectors |
| NL levels 0 / 1 / 2 | (labelling tool) | E3 labels | offline | n/a: it labels data, it does not detect live | GSM8K 29 / 30 / 41 % | | use it to label E3 for the union matrix |

## 5. Widening the scope

Each widening is justified by the notation, not by how often a form appears in frozen9; frozen9's modules **are** the test set.
Each is validated on cases frozen9 did not supply: DeepMind modules frozen9 did not draw from (the dataset is on the Hub) and frozen10.

- **Question reader:** the remainder (`the remainder when A is divided by B`), polynomial evaluation (`Let f(x) = ...; what is
  f(3)?`), unit conversion (exact unit ratios), surd expressions (`sqrt(...)` in the question), the letter string's counts for
  `swr_p_sequence` (a count is a fact: `12 letters, d 3, e 9`; it does not answer the probability), and LaTeX expressions in a
  question (`Compute $\frac{...}{...}$`).
- **`=` reader:** stop at `\approx` (the General Style's cut decimal) and `\equiv ... \pmod m` (the residue): 116 of the 334 wrong
  hand values sit there.
- **Hand-value check:** the carry reading (`a \times b = p + c = q` in a column), the chain split, `rem` forms, symbolic identities
  (both sides equal as polynomials), values of a function the question defines.
- **Loop kinds:** the 8 classifier misses (algebra, function values, letter strings) are the same widenings as the check's.

## 6. The queue, CPU before GPU

1. **CPU:** the cause split of the `=` reader's 621 misses; the carry and chain-split fixes with tests, then a re-read of a fresh
   stratified sample of the check's flags; the union matrix on round 9's traces (E1, E2, E4, E5, E6; hand-labelled samples per event).
2. **PC layer, development rows** (frozen9 after round 9's verdict), **one detector per run** against the General arm, rule + API
   reader, the three questions per touch: the question reader first (it needs no fix), the hand-value check second (after its gate).
3. **Widenings**, each with tests from the notation, validated off frozen9.
4. **Engine-side detectors** (probes, readiness, drafter) with the move into the engine; the router with Qwen3.6-35B-A3B.

Nothing here runs on the GPU while the settle probes run.
