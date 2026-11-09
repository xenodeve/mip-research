# Plan: MIP's decide layer, v1 (2026-11-10)

The developer, 2026-11-09: MIP's components (MIP Antiloop, MIP MathBackend, MIP Sandbox) should make the engine **intervene by the
situation, with judgment**: the right component at the right moment, and nothing when nothing helps. Today each component acts on
its own trigger: MathBackend at `=`, Antiloop at a repeat, Sandbox at a block. Nothing reads the situation as a whole. This plan
designs the layer that does, from a measurement of round 9's closes. It is a plan, not a protocol spec and not code: the developer
said no spec or module refactor while the methods still change ([README](../../README.md#what-mip-is-and-where-it-stands)).

**Build:** every number here is **4B NVFP4 W4A8 · llama.cpp · PC layer**, round 9, **232 of 240 pairs** (queue a2, 8 pairs, still
pending), General arm (`--loop-close --answer-close --early-cycle`, shape cap 1024, cover cap 32768, inline values General Style,
`--loop-value` off) against the control arm on the same rows and seeds. Right = right by the rule or by the API reader.

## 1. What was measured

Every General answer that had a close: **69 answers** (94 closes: 66 loop closes in the thinking, 28 answer closes). Each was read
by hand at the close point (the thinking's last lines, the answer, the boxed value) and given one class for what the model was
repeating when the close came. The control arm's result on the same key gives the outcome: **helped** (General right, control
wrong), **both right**, **hurt** (General wrong, control right), **both wrong**. Scripts and the per-case text:
`r9_closes.py`, `r9_labels.py`, `r9_prevent.py` (session scratchpad, copied to `results/decide-layer-r9/`).

| Class: what the model repeated | helped | both right | hurt | both wrong | all |
|---|---|---|---|---|---|
| **A** an equation that is already true (`Wait, $72 - 14 = 58$.` x12) | 7 | 9 | 2 | 3 | 21 |
| **B** how the question's input reads (``what if the `-2` is `-2`?``, a string's letters, "three quarters") | 14 | 6 | 0 | 3 | 23 |
| **C** the form of a value already found (`I'll write 1/10.` / `I'll write 0.1.`), or a settled answer | 5 | 4 | 0 | 2 | 11 |
| **D** a common factor, by trial division (`$1237763 / 6491 = 190 \text{ rem } 4473$`, ...) | 0 | 0 | 1 | 4 | 5 |
| **E** a run of one token (`000...`, `Wait.`, `I'll write the response.`) | 1 | 2 | 0 | 0 | 3 |
| **F** a contradiction in a hard problem it cannot resolve | 1 | 1 | 0 | 4 | 6 |
| **all** | **28** | **22** | **3** | **16** | **69** |

**By close pattern:**

| Closes in the answer | helped | both right | hurt | both wrong | all |
|---|---|---|---|---|---|
| loop close only | 19 | 18 | **0** | 4 | 41 |
| loop close, then the answer looped and got an answer close | 9 | 2 | **3** | 11 | 25 |
| answer close only | 0 | 2 | 0 | 1 | 3 |

## 2. What it shows

1. **60 of 69 loops were doubts software can settle** (A 21, B 23, C 11, D 5). The model was not lost: it had the equation, the
   question, or the value, and kept checking it. Only F (6) is reasoning the software cannot help with, and E (3) is a token run.
2. **The close is safe when the thinking close ends the loop; it is weak when the answer loops again.** Loop close only: 19 helped,
   0 hurt. When the answer looped again after the thinking was closed: 9 helped, **3 hurt**, 11 both wrong. All 3 hurt answers are
   in that pattern: `stdmath-fbf10ed4#1` (A: `72 - 14 = 58` is a true step of a cross product that was not finished, boxed 58),
   `dm_mul_div_longer-9991#1` (A: a true sub-step, then a doubt loop in the answer, boxed -136/3), `dm_mixed_longer-9951#0` (D: the
   trial division hit `4109 / 587 = 7.000...` and ran zeros; the answer then made its own error). A second loop after a thinking
   close means that close came too early, and the answer close then boxes whatever is on the page.
3. **9 of the 16 both-wrong answers had the exact value in hand** and lost it to the form: all are `dm_div_big` (`5070#0/#1`,
   `1395#0/#1`, `2783#0`, `4138#0/#1`, `9991#0/#1`), where the gold is the fraction in lowest terms and the model wrote a rounded
   decimal, an integer quotient, or both forms (hedged). The other 7: F 4, B 3 (misread input).
4. **The prevention side is not shown.** Of the 73 keys where control ended `stuck`, General ended 29 right **with no close at all**
   (and 44 with a close). One might read that as inline values keeping the model out of the loop, but in only 5 of the 29 does a
   General put on the very equation control was repeating come first (`dm_mixed_longer-1391#0` `4 \times 22.875 = 91.5`,
   `dm_div_remainder-1391#0`, `dm_mixed_longer-9951#1`, `dm_simplify_surd-926#0`, `-9937#1`; a number match flags 8, and reading
   them leaves 5: `stdmath-9f7f1563#0/#1` and `dm_conversion-9951#0` share numbers only); in 17 control's repeated line has no
   numbers at all (`I will write the response.`, `Yes.`, a reading doubt). Same seed, but the paths part at the first put, so this
   is paired by question, not by trajectory: the 29 are a count, not a mechanism.

## 3. The design

The decide layer sits in MIP's core between "collect" and the connected software. It acts at the **same moment the loop is detected
today** (the repeat check, early cycle, shape cap, cover cap): no new trigger, so it adds no new place to act. At that moment it
classifies the loop and picks the action: **settle first, close second.**

### Signals it reads (all from text MIP already has)

| Signal | How it is read | Class it serves |
|---|---|---|
| the loop's repeated unit | exists: `loop_start` / the exact cycle | all |
| the unit parses as an equation, and MathBackend says it is true | `mathexpr` + MathBackend, exact | A |
| the unit is about the question's input (a number, a sign, a bracket, a string, a phrase) | **start from the question:** find the question's numbers, strings and bracketed terms that nearly match what the loop repeats. A verbatim-substring test misses the misreads: in all 3 B both-wrong rows the model quoted the input wrongly (`vgcgp l cggp gvp` for `vgcgp lcgpgvp`), so its quote is not in the question | B |
| two forms of one value alternate (fraction / decimal, integer / exact) | both parse and are equal, or one is the other rounded | C |
| one dividend over a run of divisors | the same left operand, divisors changing, `/` or `\text{rem}` | D |
| the phase (thinking or answer), and whether a thinking close already happened in this answer | exists (`phase`, `events`) | the pattern in 2.2 |
| the question's own exact value, when it states a computation | exists: `question_read` (160 / 160 DeepMind rows) | A, C |

### Action per class

| Class | Action | Why it is true under every reading (not tuned to the set) |
|---|---|---|
| **B** (design first: largest, 0 hurt) | Put the question's text the model is doubting, verbatim: `The question writes: <quote>.` Close only if the loop resumes. | A verbatim quote of the input is exact; it decides nothing for the model. |
| **A** | Put a confirm line on the looped equation (`Checked exactly: <equation>.`); **no close** while the looped value is not the question's own value (`question_read`) or the answer's final quantity; close only if the loop resumes. **On rows with no stated computation** (`question_read` reads none on MATH-style rows; 0 of 568 on LiveBench) the intermediate-step test has no signal yet, so there v1 is confirm, then close when the loop resumes: today's close one beat later. Both A-hurt rows are of this kind (`stdmath-fbf10ed4#1`, `dm_mul_div_longer-9991#1`: a sub-step, not the question's value); whether the beat matters is measured on the next round's closes. | A true equation is true; the confirm says only that. |
| **C** | Put the exact value in its exact form (lowest-terms fraction, exact surd) once; then close as today. | The exact form is equal to every rounding of it; a rounded decimal is not equal to the value. |
| **D** | Put the result of the search: `gcd(a, b) = g`, or the fraction in lowest terms. | A gcd is a gcd: MathBackend computes it in microseconds. |
| **E** | Close, as today. | Nothing to settle. |
| **F** | Close, as today. The ceiling: 4 of 6 wrong in both arms. | Nothing software can settle. |
| unclassified | As today. | Silence: no settle line unless the check above is exact. |

**Precedence.** (1) Within one loop: settle before close. (2) A second loop in the answer after a thinking close (pattern 2.2)
gets the settle of its class before the answer close, never the answer close alone. (3) An inline value at `=` is never blocked by
this layer; the layer acts only at loop detection.

**What it does not do.** It never writes the answer for the model: the boxed value stays the model's (software computes, the model
decides). It adds no trigger, so it cannot act on an answer that never loops.

### Known risk before it is built

`--loop-value` (D in `antiloop.md`) already puts the question's exact value before a close. On the A100 (round 5) its value was
right in 12 of 20 answers and **the model did not adopt it** on the two division rows: it restarted the long division. So a settle
line can be read and ignored; that is what the PC run must measure (per touch: value right, needed, helped), not assume. Every new
put form (the confirm line, the quote, the gcd line) is also judged by agreement with the model's next value, in and out of sample,
as the inline forms were, before it goes into a round.

## 4. The gate, in order

1. **Classifier in replay:** on the stored round-9 traces, the classifier reproduces the 69 hand labels below (coverage per class,
   with the denominator; for B the hard part is the 3 misread rows, which a substring test misses), and on the General answers with no close it touches **0** (it has no trigger there; the replay checks it).
2. **Each put form** checked for agreement with the model's next value (in and out of sample).
3. **PC layer paired run** (`4B NVFP4 W4A8 · llama.cpp · PC layer`) on **development rows** after round 9's verdict (frozen9 then
   becomes development rows), both arms in one round, rule + API reader, the three questions per touch. Whether settling beats
   closing is **unmeasured until this run**.
4. **Validation on rows frozen9 did not supply** (frozen10), before any A100 round. The six classes were read off frozen9, the test
   set; each action above is justified by the notation, and a rule whose only support is how often a class appeared here is left out.

## 4a. Simulation of v1 (2026-11-10, no model, no GPU)

The classifier and settle lines are `tools/igc/loop_class.py` (tests `tools/test_loop_class.py`, written from each class's definition
with made-up loops, before the run); the run is `python -m tools.igc.decide_sim <pc> results/decide-layer-r9/labels.json --rows ...`,
output `results/decide-layer-r9/sim-v1.json`. Rules were written once and measured once; nothing below was tuned to the labels.

**Classifier against the 69 hand labels: 55 of 69 agree.**

| hand \ classifier | A | B | C | D | E | F | of |
|---|---|---|---|---|---|---|---|
| A | 16 | 1 | 0 | 0 | 0 | 4 | 21 |
| B | 2 | 18 | 1 | 0 | 0 | 2 | 23 |
| C | 0 | 0 | 11 | 0 | 0 | 0 | 11 |
| D | 0 | 0 | 0 | 3 | 1 | 1 | 5 |
| E | 0 | 0 | 0 | 0 | 3 | 0 | 3 |
| F | 0 | 2 | 0 | 0 | 0 | 4 | 6 |

- **No disagreement puts a false line.** Every settle line is exact by construction: an A line is a relation MathBackend computes as
  true, a B line quotes the question, a C or D line carries the question's exact value. Where a C or D line carries a value it is
  gold in **10 of 10** (labelled) and **4 of 4** (control).
- **The misses fall back to today's close (silence):** 4 A rows the software cannot evaluate (`x^2 - x + 1 = 0` algebra,
  `f(1, 3) = 0`, a `27 / 3 = 9 rem 0` trace), 2 B rows on a letter string with no term shared with the question
  (`dm_swr_p_sequence-9951#0/#1`), and `dm_div_big-1395#0` (a search over years, not a division). `dm_mixed_longer-9951#0` (a trial
  division that ran into zeros) is read as E, close as today.
- **Harmless confusions:** 2 F rows read as B (a geometry doubt; the line quotes the question), 2 B rows read as A (a true
  `2 / (-10) = -0.2` in a reading-doubt loop).
- **Settle line available:** 49 of 69 labelled loops (A 18, B 21, C 7 of 12, D 3); the other 20 close as today. On control's 73
  stuck loops (an arm with no closes): A 6, B 21, C 11, D 1, E 8, F 26; a settle line on 31 of 73.
- **The rows that matter:** of the 9 both-wrong `dm_div_big` answers with the value in hand, 5 get the exact value (C or D), 3 a
  confirm (A), 1 none. Of the 3 hurt answers, `stdmath-fbf10ed4#1` and `dm_mul_div_longer-9991#1` get a confirm (A), and
  `dm_mixed_longer-9951#0` closes as today (E).

**What the simulation cannot say:** whether the model reads a settle line and stops looping. The C lines on `dm_div_big` repeat
the question's own fraction (`(-73388)/(36903) = -73388/36903`): the model's doubt there was decimal against fraction, the very case
where `--loop-value` was right and not adopted. Next, before a full round: a **settle probe** like the cue probe (`antiloop.md`,
"Cue probe"): the 69 stored prefixes, each continued with today's close and with the settle line, same seeds, on its own
llama-server (never a round's), rule + API reader. That needs the GPU.

## 5. Deferred, with the reason

- **MIP Adapter with no decode** (MIP picks an external tool and its arguments itself): needs this layer to work on the internal
  tools first.
- **A learned controller:** the learned work is paused, and a rule can say why it acted.
- **Detecting unfinished work from structure** (a cross product with one component left): no signal for it yet beyond
  "the looped value is not the question's value"; measured later on the closes of the next round.
- **The 29 answers General ended with no close:** what kept them out of a loop is not shown (2.4); a trajectory-level check needs
  paired continuations, not this data.

## Appendix: the 69 labels

| Answer | Class | Closes | Outcome | Control |
|---|---|---|---|---|
| `dm_mixed_longer-5053#0` | A | loop | helped | stuck |
| `dm_mixed_longer-5053#1` | A | loop | helped | stuck |
| `dm_div_remainder-5053#0` | A | loop | helped | stuck |
| `dm_mul_div_longer-5942#0` | A | loop | helped | stuck |
| `dm_mul_div_longer-5942#1` | A | loop+answer | helped | stuck |
| `dm_div_big-5942#0` | A | loop | helped | stuck |
| `stdmath-ed48028c#1` | A | loop | helped | stuck |
| `dm_mixed_longer-926#0` | A | loop+answer | both right | stop |
| `dm_mixed_longer-926#1` | A | answer | both right | stop |
| `dm_mul_div_longer-5070#1` | A | loop | both right | stop |
| `dm_div_big-928#1` | A | loop | both right | stop |
| `dm_div_remainder-926#0` | A | loop | both right | stop |
| `stdmath-eaaaaeb1#1` | A | loop | both right | stop |
| `dm_mixed_longer-5920#1` | A | loop+answer | both right | stop |
| `stdmath-ed48028c#0` | A | loop | both right | stop |
| `stdmath-5a258a23#0` | A | loop | both right | stop |
| `dm_mul_div_longer-9991#1` | A | loop+answer | hurt | stop |
| `stdmath-fbf10ed4#1` | A | loop+answer | hurt | stop |
| `dm_div_big-5070#1` | A | loop+answer | both wrong | stuck |
| `dm_div_big-9991#0` | A | loop+answer | both wrong | stuck |
| `dm_div_big-9991#1` | A | loop+answer | both wrong | stuck |
| `dm_simplify_surd-3480#0` | B | loop+answer | helped | stuck |
| `dm_simplify_surd-9937#0` | B | loop+answer | helped | stuck |
| `dm_swr_p_sequence-3480#1` | B | loop+answer | helped | stuck |
| `dm_swr_p_sequence-5053#0` | B | loop | helped | stuck |
| `dm_mul_div_longer-1395#0` | B | loop | helped | stuck |
| `dm_mul_div_longer-1395#1` | B | loop | helped | stuck |
| `dm_div_big-5942#1` | B | loop | helped | stuck |
| `dm_simplify_surd-1391#0` | B | loop+answer | helped | stuck |
| `dm_simplify_surd-1391#1` | B | loop | helped | stuck |
| `dm_simplify_surd-5920#0` | B | loop | helped | stuck |
| `dm_simplify_surd-5920#1` | B | loop | helped | stuck |
| `dm_conversion-1391#0` | B | loop | helped | stuck |
| `dm_conversion-1391#1` | B | loop | helped | stuck |
| `dm_swr_p_sequence-9951#0` | B | loop | helped | stuck |
| `dm_simplify_surd-926#1` | B | loop | both right | stop |
| `dm_simplify_surd-1501#1` | B | loop | both right | stop |
| `dm_simplify_surd-5053#1` | B | loop | both right | stop |
| `dm_simplify_surd-2773#0` | B | loop | both right | stop |
| `dm_simplify_surd-9951#0` | B | loop | both right | stop |
| `dm_simplify_surd-9951#1` | B | loop | both right | stop |
| `dm_simplify_surd-3480#1` | B | loop+answer | both wrong | stuck |
| `dm_swr_p_sequence-3480#0` | B | loop+answer | both wrong | stuck |
| `dm_swr_p_sequence-9951#1` | B | loop | both wrong | stuck |
| `dm_div_big-1505#0` | C | loop+answer | helped | stuck |
| `dm_div_big-1505#1` | C | loop | helped | stuck |
| `stdmath-9b21d63d#0` | C | loop | helped | stuck |
| `dm_div_big-2783#1` | C | loop+answer | helped | stuck |
| `stdmath-5dd40b1a#1` | C | loop | helped | stuck |
| `dm_div_big-3491#0` | C | loop | both right | stop |
| `dm_div_big-9977#0` | C | answer | both right | stop |
| `dm_swr_p_sequence-4125#0` | C | loop | both right | stop |
| `dm_swr_p_sequence-4125#1` | C | loop | both right | stop |
| `dm_div_big-1395#1` | C | loop | both wrong | stuck |
| `dm_div_big-4138#0` | C | loop | both wrong | stuck |
| `dm_div_big-5070#0` | D | loop+answer | both wrong | stuck |
| `dm_div_big-1395#0` | D | loop | both wrong | stuck |
| `dm_div_big-2783#0` | D | loop+answer | both wrong | stuck |
| `dm_div_big-4138#1` | D | loop+answer | both wrong | stuck |
| `dm_mixed_longer-9951#0` | D | loop+answer | hurt | stop |
| `dm_simplify_surd-2773#1` | E | loop+answer | helped | stuck |
| `stdmath-6fe51888#0` | E | loop | both right | stop |
| `stdmath-5e2eb2d9#0` | E | loop | both right | stop |
| `stdmath-0b7036eb#0` | F | loop+answer | helped | stuck |
| `stdmath-245aadb1#0` | F | loop | both right | stop |
| `stdmath-370853d3#0` | F | loop+answer | both wrong | stuck |
| `stdmath-370853d3#1` | F | loop+answer | both wrong | stuck |
| `stdmath-97b18f58#0` | F | loop+answer | both wrong | stuck |
| `stdmath-97b18f58#1` | F | answer | both wrong | stuck |
