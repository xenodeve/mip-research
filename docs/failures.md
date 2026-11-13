# Failure patterns: what made the model wrong, and what fixes it

Every way an answer or a mechanism went wrong that was read by hand or recorded in the docs, one row per pattern: what the model did, where it was seen (with the build),
what fixes it or is expected to, and **whether the fix has been tested**. A pattern stays here after it is fixed, so the next model
or round can be checked against it. The developer's rule (2026-11-10): a pattern that can occur gets fixed, however rare.

[ภาษาไทย](#ภาษาไทย)

## How a pattern is fixed: the situation, not the test set

The developer (2026-11-11): a test set is a probe that **finds a situation in which the model fails**; the fix targets that situation,
which can occur in any set of the same kind or in real use. A fix that moves the test set's answers but is shaped to the set's wording,
module or gold format is tuning, not a fix. So, for every pattern:

1. **State the situation in words** with no module name, set wording or row id ("the whole question is one computation", "the model
   closes on a value that is a step of the asked quantity"). If it cannot be written that way, it is not a pattern yet.
2. **Build the detector from that statement**, not from the rows that showed it.
3. **Test the detector where it was not written from:** a list of wordings or sets fixed before the run, reported as reach per family
   (read / should read, and right when read). A detector that sees the situation only in the test's wording is a test-set fix.
4. **Count evidence by distinct questions,** not answers: one question x 2 seeds is one piece of evidence ("one question" is said).
5. **Measure the gain on rows the fix was not built from** (frozen10 and another family), with 0 right -> wrong.

**First check by this rule (2026-11-11), `--exact-answer`'s detector** (`question_read.read_exact`; script
`results/exact-answer-r9/phrasing.py`): the fix's logic states the situation generally, but its detector was bound to the test's
wording. On the 80 questions it reads, re-asked in 10 wordings fixed before the run:

| Wording | Read |
|---|---|
| `What is {e}?`, `Compute {e}.`, `Evaluate {e}.`, `Work out {e}.`, `Find the value of {e}.`, `{e}` | 80 / 80 each, same value |
| `{e} = ?`, `Can you calculate {e} for me?`, `Please simplify {e}.`, `What do you get when you compute {e}?` | **0 / 80 each** |

Across all 370 stored rows it fires on 4 DeepMind modules only (80 of 80 gold) and on 0 of MATH (162), LiveBench (32) and contest (16)
rows; those families hold few one-computation questions, so silence there is expected, but the wording table shows the detector would
miss the same situation asked another way. **Fixed (D19, 2026-11-11):** `read_exact` was widened from the situation's statement: around one expression may stand only the
grammar of asking (courtesy, the asking verb, "the value/result of", "equals", `= ?`), a word list written from English request
grammar; any other word (round, decimal, binary, remainder, percent, squared, a unit, a variable, `and`) reads nothing; LaTeX
(`\times`, `\cdot`, `\div`, `\frac`, `\sqrt`, `$`) is read. Tests in `tools/test_igc_question_read.py` (`TheSituation`), written before
the measurement. Measured on lists fixed before the run and not used to write it (`results/exact-answer-r9/heldout_wordings.py`,
`outside_reach.py`):

| Check | Result |
|---|---|
| 8 new request wordings (`Would you work out {e} please?`, `Solve: {e}`, ...) | 80 / 80 each, same value |
| the same expressions in LaTeX (`Compute $... \times \frac{a}{b}$.`) | 80 / 80, same value |
| 10 wordings that change what is asked (`to three significant figures`, `in hexadecimal`, `squared`, `modulo 3`, `Estimate`, ...) | **0 / 80** each |
| MATH-500 (500, never used here) | 6 read, **6 equal to gold** (`Simplify $\sqrt{242}$.`, `What is the value of $513^2 - 487^2$?`, ...) |
| GSM8K test (1,319 word problems) | 0 read |
| stored rows (370) | 81 read (80 DeepMind + 1 MATH), all equal to gold |
| round 9 replay of `--exact-answer` | unchanged: 24 acts, 16 wrong -> right, 0 right -> wrong |

Not yet: a live run of `--exact-answer` on questions outside DeepMind's wording. **Queued** with the 6 MATH-500 questions it reads, inside the next live round ([plan](plans/2026-11-10-round9-fixes.md#queued-for-the-next-live-round-2026-11-11)).

**Status words:**
- **fixed, live** = the fix ran with the model and the pattern's answers came out right.
- **fixed, replay** = the fix acts correctly on stored text; no live run yet.
- **fix planned** = a mechanism is designed or exists elsewhere, not yet tried on this pattern.
- **no fix yet** = nothing in the plans reaches it.
- **ceiling** = the information is not in what the model reads. (P10 carried this label until 2026-11-11; the question turned out
  solvable from its text, so none of round 9's answers is a ceiling now.)

**Source of the round 9 rows:** all 25 General answers that the rule scores wrong in round 9 (240 pairs, **4B NVFP4 W4A8 · llama.cpp ·
PC layer**, General Style + closers), each read by hand (plan:
[2026-11-10-round9-fixes](plans/2026-11-10-round9-fixes.md)): P1-P5 15, P7 2, P8-P10 6, P11 1, P12 1 = 25.

## Summary

| # | Pattern | Seen (answers) | Fix | Tested? | Status |
|---|---|---|---|---|---|
| P1 | Right value, wrong form: a rounded decimal or the integer part boxed instead of the exact fraction | round 9: 6 | **the developer, 2026-11-11: counted right with a note that the form is not gold's, the answer and the gold written in the note** (`standard_score.form_note`): a decimal rounded or cut at its own places, 3 or more significant digits, of a gold with no finite decimal. The integer part, `0`, a lost sign or wrong digits stay wrong (values, not forms) | round 9 re-scored (`results/exact-answer-r9/rescore_forms.py`): General 215 -> 218, control 161 (0 noted); the 3 noted: 1395#0 `-1.9886`, 1395#1 `-1.9887` for `-73388/36903`, 5070#1 `0.00456226` for `5647/1237763`; still wrong: 4138#0 (`-10710884`, integer part; the model: *"Given the request for a 'single number' ... I will provide the integer quotient"*, the words of **our** prompt, not the question's; reached by `--question-quote` at its thinking loop's close, the cause itself (the prompt) left as the developer chose), 4138#1 (sign lost), 5070#0 (`0.004562376`, wrong digits) | fixed in the scorer, both arms alike |
| P2 | Two values offered, or a hedge (`275 or -275`) | round 9: 3 | `--answer-rewind`: the looping answer is cut out and resampled, the model writes its own value (free continuation, arXiv 2604.06613) | not live yet | built; round 10 |
| P3 | A wrong value written during a loop, then boxed at the close | round 9: 3 (`9951#0`, `9991#1`, `2783#0` boxed `0`) | `--answer-rewind`, as P2 | not live yet | built; round 10 |
| P4 | The question's precedence misread (`(-25)/X*1/(-4)` as `-25/(X/(-4))`) | round 9: 1 | `--question-quote`: when the model's copy of the question's expression reads to another value, the question's words are put back (RE2, arXiv 2309.06275; number-copy errors, arXiv 2502.08680) | stored text: fires on 15 of 442 answers, all 15 read by hand as real mis-copies (a moved bracket, `/` as `\cdot`, a term moved) | built; round 10 |
| P5 | `simplify_surd` loops counting parentheses | round 9: 2 (3480 x2; 9937#0 now right by the scorer fix) | `--question-quote` at the thinking loop's close: the question's words, then the close (settle and close at once: 32 vs 31 of 49) | not live yet | built; round 10 |
| P6 | Minus lost after the answer-close cue: the cue ended in `{` and Qwen writes `{-` as one token | cue probe (2026-11-09): 22 of 28 gold-negative values | the cue ends before `{` | cue probe: minus on gold-negative values 41 -> 100 of 112 | fixed, live (probe) |
| P7 | Letters of the question's string miscounted or misquoted | round 9: 2 (`swr_p_sequence`); a general weakness of tokenized models (arXiv 2505.14172) | `--letter-count` (`tools/igc/letters.py`): the string the question makes the object of a letters question, counted, put at the start | offline: 14 of 14 fixed wordings right; MATH-500 1 read (ELLIPSE), GSM8K 0 of 1,319, stored rows 10 (all raw-string letter questions); the first rule read 6 prose words (`in full`), fixed by structure | built, offline; live next round |
| P8 | A convention chosen wrong, extra roots kept (arccot) | round 9: 2 (`e3263989`) | `--verify-gate` (Sandbox, PC box): an equation given and all its solutions asked, the first close opens a program in which the model checks each candidate; the model chooses the convention | reach: MATH-500 11 questions (all-solutions or sums of all solutions), GSM8K 0 of 1,319; box smoke: sympy runs, the repository path refused | built; live in round 10 arm C |
| P9 | A hard maximum guessed after a very long search (71k tokens) | round 9: 2 (`97b18f58`) | `--loop-hint` (Sandbox, PC box): a thinking loop is cut and a program opens to test the idea, up to 3 times, then the close | built earlier (#12); never measured on the PC | live in round 10 arm C |
| P10 | The setup is left open ("as shown"); the model lists layouts in a loop and never tests one | round 9: 2 (`370853d3`) | `--loop-hint` (Sandbox, arm C): a program tests the layouts against the stated tangency; `--answer-rewind` (arm B) only resamples. First labelled a ceiling, wrongly: the text gives r = 14/3 | not live yet | built; round 10 arm C |
| P11 | The model's own value after `\approx` is wrong (`(1.01)^{30} \approx 1.3448888`) | round 9: 1; general (MATH 401, MedCalc audit) | `--approx-put` (`tools/igc/approx_put.py`): stop at `\approx`/`≈`, put X to 10 significant digits; mathexpr's wide mode reads large powers; one token after it may not carry `\times`/scale words on | replay of every stored `\approx`: round 9 3,242 read, model's own value agrees 2,648, a slip 336 (12 %), an expression next 167; A100 bf16 2,037 / 1,916 / 53 / 144; 34 possible cut pieces read by hand, 0 misreads. At `=`, an exact value too long to be a step (`1.01^{30} =`) is put cut to the same 10 digits, `1.347848915...` (`--long-value`, `tir4b._long_found`, `scripts/pc/long_gate.py`): 86 on 2 questions, agree 22, expression next 50, slip 14, 0 misreads | built, offline; live in the next channel run |
| P12 | The answer close boxes an unfinished step (`72 - 14 = 58`, a cross-product component) | round 9: 1 (`fbf10ed4#1`); the 3 hurt L7 answers all after a forced cue | `--answer-rewind N`: an answer loop is cut out whole and resampled with another seed, up to N times, before the cue; nothing put (forced extraction fails, arXiv 2604.06613); not when the question is one computation | round 9 reach: 8 answers (5 questions): 1 hurt by the close, 5 wrong both arms, **2 the close helped (could be lost)** | built, offline; live next round |
| P13 | The loop runs as long as before even when the closing value is fixed | targeted run: `2783#0` 70,772 tokens, `5070#1` 67,665 | earlier loop signals exist (arXiv 2601.05693, 2510.10103, 2509.26522) | not measured | measure where the tokens go first |
| P14 | A put value read and not adopted: the model restarts the long division | A100 round 5 (`--loop-value`): 2 of 2 division rows | the box rule (`--exact-answer`) instead of a line before the close | P1 live run | replaced by P1's fix |
| P15 | Patterns MIP caused itself (put forms that started loops) | see the section below | each fixed in the put | replay gates | fixed |
| P16 | The rule scorer read a right answer as wrong (or a hedge as right) | round 9: 6 | the scorer rules | re-score of round 9 | fixed |
| L1-L8 | Loops, by the shape the detectors see | A100 rounds, round 9 | [Anti-loop](antiloop.md) detectors + closes | see [Loops](#loops) | mostly in best config |
| LA-LF | Loops, by what the model repeats (round 9's 69 closes) | round 9 | decide layer: one pairing per kind | simulation + settle probe | fix planned |
| D1-D19 | Detection: a calculation missed, misread, or a false call | see [Detection](#detection) | | | mixed |
| I1-I17 | Inject: a put form, note or closer that misled the model or was ignored | see [Inject](#inject) | | | mixed |
| C1-C5 | Compute: a wrong or slow value from the software | see [Compute](#compute) | | | mixed |
| S1-S17 | Sandbox and tools: blocks not opened, programs that mislead, box faults | see [Sandbox and tools](#sandbox-and-tools) | | | mixed |
| T1-T9 | Learned / latent heads that did not generalize | see [Learned and latent](#learned-and-latent) | | | mostly paused |
| M1-M20 | Measurement errors that made a result look better or worse than it was | see [Measurement](#measurement-and-instrument-errors) | | | fixed or withdrawn |

## Round 9, one by one

> **2026-11-11:** `--exact-answer`, named below as the fix of P1-P5, is withdrawn as a model fix (it wrote the answer for the model; [ERRATA](ERRATA.md) MR-35 to MR-37). The summary table above gives the replacements; the sections below keep the history.

### P1. Right value, wrong form

- **What happened:** the model had the exact fraction on the page, then boxed a decimal (`-1.9886`, `0.004562376`,
  `10710884.384615...`) or the integer quotient (`-10710884`). The gold is the fraction in lowest terms.
- **Seen:** `dm_div_big` 1395 x2, 5070 x2, 4138 x2. Control lost the same rows to `stuck`.
- **Fix:** `--exact-answer` ([inject](inject.md#--exact-answer)): when `question_read` reads the whole question as one computation,
  the last box gets the exact value, at the answer-close cue or over the last box.
- **Tested:** replay on round 9: 24 acts, 16 wrong -> right, 0 right -> wrong. Targeted live run (2026-11-11, code `6c9331e`, 11 rows
  x 2 seeds): **22 of 22 right** against 7 of 22 for round 9's stored answers; all 9 `div_big` answers wrong in round 9 are right,
  each with an act. Not yet paired (next: the paired live test).

### P2. Two values, or a hedge

- **What happened:** `275 or -275` (`dm_mul_div_longer-5942#1`), both a fraction and a decimal offered (`div_big-9991` x2).
- **Fix and test:** as P1, at the cue. Live: `9991` x2 right. `5942#1` was scored right by the old rule, so it was not in the
  targeted run; replay right.

### P3. A wrong value from a loop, boxed at the close

- **What happened:** the answer looped, and the close boxed the value on the page: `4694/15` for 313 (`mixed_longer-9951#0`, a trial
  division that ran into zeros), `-136/3` for -136 (`mul_div_longer-9991#1`, a doubt loop on a true sub-step), `0` for
  `-171/2221234` (`div_big-2783#0`).
- **Fix:** `--exact-answer` puts the question's exact value after the cue.
- **Tested:** `2783#0` right live with an act. `9951#0` and `9991#1` came out right live **with no act**: the model took another path,
  so this run does not show the fix on them. The paired test has to.

### P4. Precedence misread

- **What happened:** `dm_mul_div_longer-1505#0`: `(-25)/X*1/(-4)` read as `-25/(X/(-4))`, boxed -12.375 (live run: -33); gold -198.
- **Fix:** `--exact-answer` widened: any last box whose value is not the exact value gets it, integers included (the developer:
  when the question is the computation, its value is the answer).
- **Tested:** live: the act wrote `-198` over `-33`; right.

### P5. Surd simplification loops

- **What happened:** `dm_simplify_surd` 3480 x2 looped counting parentheses in `(sqrt(...)...)**2`; the close boxed `9216` and
  `2633616 - 103680\sqrt{3}` (gold `884016 - 103680\sqrt{3}`).
- **Fix:** `question_read.read_exact` reads `Simplify <expression with square roots>` and computes `a + b\sqrt{n}` exactly (84 of 379
  stored rows read, 84 of 84 gold); `--exact-answer` then covers the module.
- **Tested:** live: 9937 x2 right with an act at the cue. 3480 x2 right with no act, no loop this time, 31k tokens against 13-17k in
  round 9.

### P7. Letters miscounted

- **What happened:** `dm_swr_p_sequence` 3480#0 (`1/66` for `1/78`) and 9951#1 (`7/55` for `56/495`): the model counted the letters of
  the question's string wrong from the start. On other rows of this module the model quoted the string wrongly
  (`vgcgp l cggp gvp` for `vgcgp lcgpgvp`, [decide layer](plans/2026-11-10-decide-layer-design.md)).
- **Expected fix:** a count by software. A Sandbox program helps only if the string it counts is the question's own: if the model
  copies its misquote into the code, the program counts the misquote. So the count reads the string from the question text. A reader
  for this one sentence shape would be tuned to DeepMind and is not used. The decide layer's B line (quote the question) missed both
  rows in simulation: no term of the loop is shared with the question.
- **Tested:** no.

### P8-P10. MATH level 5 reasoning

- `stdmath-e3263989` x2: the answer `3-2\sqrt{2}, 3+2\sqrt{2}, -3+2\sqrt{2}, ...` keeps roots that fail the arccot convention (gold
  `3\pm2\sqrt{2}`). **Expected:** a Sandbox check that substitutes each candidate into the equation numerically drops the false ones.
  The hypothesis-loop rules (`--igc-sandbox-verify`, `-expect`, `-why`) exist in the Strata engine only and were measured on
  LiveCodeBench, not on math ([sandbox](sandbox.md#hypothesis-loop-rules-draft-call-expect-why-gate-pass-close)).
- `stdmath-97b18f58` x2: a hard maximum, guessed (64 and 4 for 0), one answer 70,974 tokens. **Expected:** a brute-force numeric search
  from the statement as an oracle (`--igc-sandbox-brute`, engine only, never measured).
- `stdmath-370853d3` x2 (re-read 2026-11-11; first labelled a ceiling, wrongly): the text does not fix the layout ("as shown"), but it can be solved from the text by testing layouts against the tangency it states: the semicircle on QA has radius 7, the smaller one on QB at B has radius r, and the centres 7 + r apart give √(49 + (14 − r)²) = 7 + r, r = 14/3 (gold). The model listed layouts in a loop (`If they are tangent to the radii... If they are tangent to the arc...`) and never computed one; the answer close boxed 7. **The case:** a setup the text
  leaves open, tried case by case without a check. **Mechanism:** `--loop-hint` in arm C (the loop is cut and a program opens to test the
  layouts); `--answer-rewind` in arm B reaches it too but only resamples.

### P11. The model's own `\approx` value

- **What happened:** `stdmath-45fa3ef2#0`: `(1.01)^{30} \approx 1.3448888` (true 1.3478489), so 32277 for 32348.
- **Measured** (`results/exact-answer-r9/approx_scan.py`): General wrote 3,391 `\approx` with a decimal; the computation before it is
  readable on 1,952, and the model's value is off on 316. The 316 mix real slips, scientific notation read as its mantissa, and
  deliberate estimates (`\approx 34000`). The `=` put's forms are wrong after `\approx`, and `(1.01)^{30}` is not read at all.
- **Fix:** `--approx-put` puts X to 10 significant digits after `\approx` (table above). **The same case at `=`** (2026-11-13): an exact
  value too long to be written as a step (a decimal to a power: `1.01^{30}` is a 61-digit fraction, past mathexpr's power and size
  limits) was refused at `=`, so the model's own digits stood (`1.01^{30} = 1.3448888213...`, miss1 B). `--long-value`
  (`tir4b._long_found`) reads it
  in mathexpr's wide mode, only where every other reader found nothing, and puts it cut toward zero to the same 10 significant digits,
  marked `...` (`1.01^{30} = 1.347848915...`), true read as the value cut to its places. Not an integer, and only for
  1e-4 <= |value| < 1e9. Gate (`scripts/pc/long_gate.py`, every stored record, `results/token-channel/long-gate.txt`): 86 such `=`
  on 2 distinct questions; the model's next number agrees 22, an expression next 50, a slip 14, all 14 read by hand as model slips
  (`1.01^{25} = 1.28235` for 1.2824320, `24000(1.01)^{30} = 32277.33` for 32348.37), 0 misreads; the other question is a runaway
  fraction chain (`dmx_div_big-3350`, pc9-handoff), read right. No existing put changes. Off in miss1 (both `--approx-put` and the
  `=` put postdate dev2's flags); on in the next channel run.

### P12. The close boxed an unfinished step

- **What happened:** `stdmath-fbf10ed4#1`: the model looped doubting a true `72 - 14 = 58`, one component of a cross product; the
  answer close boxed 58 (gold 7). Control got it right.
- **Why the planned fix is not enough:** the decide layer v1 reads this as class A (a true equation) and puts a confirm, then closes
  as today, one beat later. Detecting unfinished work from the structure (a cross product with a component left) is deferred in the
  [decide layer plan](plans/2026-11-10-decide-layer-design.md#5-deferred-with-the-reason): no signal yet.
- **Needed:** a signal that the boxed value is a step, not the answer: the question asks for one quantity and the loop's value is a
  sub-result of it. **No fix yet.**

### P13. Long loops even with the right closing value

- `div_big-2783#0` 70,772 tokens, `div_big-5070#1` 67,665 in the targeted run. `--exact-answer` fixes the value at the close, not the
  loop's length. The settle probe (2026-11-10): settle line and close together 32 of 49, today's close 31, fewer tokens.

## The DeepMind rows' prompt line (decided 2026-11-11)

**What it is.** Every DeepMind row of rounds 1-9 ends with `Give the final answer as a single number in \boxed{} at the end.`
(`scripts/a100/frozen_rows.py`, `SUFFIX`, added 2026-11-08). It is **our** line: the DeepMind Mathematics dataset (Saxton et al.,
ICLR 2019) gives the question alone (`Divide -73388 by 36903.`) and an exact answer. The MATH rows end with
`Put the final answer in \boxed{} at the end.` (`SUFFIX_ANY`), the line math evaluations commonly use.

**What it did** (round 9, `results/exact-answer-r9/single_number.py`): answers that cite "single number" while
choosing between a decimal, an integer quotient and rounding: General 21 (11 right, 3 right with a form note, **7 wrong**), control 23
(9 right, **14 wrong**), nearly all `div_big`. `div_big-4138#0`: *"Given the request for a 'single number' ... I will provide the integer
quotient."* The line makes the model doubt the form, and part of the class-C loops the closers end come from it.

**Decision (the developer, 2026-11-11):**
- Changing the line to ask for an exact value is not done: it would steer the form toward gold's and move further from the dataset.
- **Round 10 keeps the current line**, so it compares with round 9 directly.
- **frozen10 onward: the DeepMind rows get the same neutral line as the MATH rows** (`Put the final answer in \boxed{} at the end.`):
  no form asked, the model chooses, and the scorer's form rule (a correct decimal of a gold with no finite decimal is right, noted)
  applies to every arm alike.
- **What may be claimed:** "DeepMind Mathematics questions with our answer-format line, scored by our rule and the API reader". Rounds
  1-9 used our own line; no DeepMind number here is a leaderboard number, and the DeepMind numbers before and after frozen10 are not
  compared directly.

## Loops

A loop makes the model wrong two ways: with no closer it runs until the stuck stop and ends with **no answer** (control: 73 of 240
answers in round 9 ended `stuck`), and with a closer it can be **closed on the wrong value** (P3, P12). The developer's principle
(2026-11-10): loops are of several kinds and each kind gets its own pairing, never one closer for all. Two views: the shape the
detectors see (L), and what the model repeats (LA-LF). Detail: [Anti-loop](antiloop.md), [decide layer](plans/2026-11-10-decide-layer-design.md).

### By shape: what the detectors see

| # | Shape | Example | Detector + action | Tested | Status |
|---|---|---|---|---|---|
| L1 | Exact cycle: the same text again and again | `Wait, $72 - 14 = 58$.` x12; a repeating decimal on one line (`-5511170.918918...`) | repeat check (20 rounds, 8 past 512 tokens); V7 early cycle (6 rounds over 500+ tokens, at every line end); then loop close A | A100: 197 guard stops classified, the widest 6 read, all real loops. V7 replay: 0 of 915 right answers touched, 212 of 243 loops caught sooner. Loop rows 25 -> 37 of 48 with A | in best config |
| L2 | Counting loop: the same shape with changing numbers (hand counting) | digit-masked cycle of 64 rounds | counting check; `--guard-progress` lets real work go (trial division by one prime after another, a search over x) | replay: changes exactly the 3 rows still working; every other stop kept | in best config |
| L3 | Line back with small rewording | spelling the prompt's letters 13-25 times; `So ∠AFD = 80` x17 | line-back check (a 12+ token line seen 16 times) | A100: 23 stops, the 12 widest read, none cut early | in best config |
| L4 | Long division digit by digit, numbers changing length (no exact cycle) | `div_big-2208`: 0 right in 28 earlier answers | shape cap: digit runs read as one token, close at 1,024 rounds | A100 round 8: closed at 55k model tokens | in best config |
| L5 | Long division where zero digits break the cycle | round 8 `div_big` #1: shape rounds never past 46, 196k tokens to the context end | cover cap: 97 % of the last 32,768 tokens repeat a 12-gram seen in the 2,048 before | replay: 0 of 1,078 right touched; 12 of 13 answers past 150k stopped at 40k-114k | in best config |
| L6 | A loop in the answer after the thinking was closed | 3 of 15 closed thinking loops carried the doubt into the answer (A100 loop-lc) | answer close B: cut to one round, then the final-answer cue | A100: cut answers 3 -> 1; replay 0 of 370 right touched | in best config |
| L7 | **The close came too early:** loop close, then the answer loops again, and B boxes what is on the page | round 9: 25 answers with this pattern, **3 hurt** (`fbf10ed4#1`, `9991#1`, `9951#0`), 11 both wrong; 0 hurt with the loop close alone (41) | `--exact-answer` at the cue (when the question is one computation); decide layer: the class's settle before the answer close | `--exact-answer`: live, see P3. Decide layer: simulation only | fixed for one-computation questions; open for the rest (P12) |
| L8 | Echo: the model checks the same expression again and again after a put | 100 times `-1/2` (`livebench-652451e2`) | echo guard (the 3rd put in a row within 64 tokens skipped) | stored traces: echo in 7 of 48 answers | in best config |

### By what the model repeats: round 9's 69 closes

Every General answer with a close in round 9 (232 pairs), read by hand at the close; the outcome against control on the same key.

| # | What the model repeated | Example | Helped / both right / hurt / both wrong | Planned pairing | Status |
|---|---|---|---|---|---|
| LA | An equation that is already true | `Wait, $72 - 14 = 58$.` | 7 / 9 / **2** / 3 (21) | confirm line `Checked exactly: <equation>.`, no close while the value is not the question's | simulation; P12 shows the confirm alone may not be enough |
| LB | How the question's input reads | ``what if the `-2` is `-2`?``, a string's letters, "three quarters" | 14 / 6 / 0 / 3 (23) | quote the question verbatim: `The question writes: <quote>.` | simulation: missed the 2 `swr` rows (the model's misquote shares no term with the question, P7) |
| LC | The form of a value already found | `I'll write 1/10.` / `I'll write 0.1.` | 5 / 4 / 0 / 2 (11) | the exact value in its exact form, once, then close | **covered by `--exact-answer`** on one-computation questions (P1) |
| LD | A common factor by trial division | `$1237763 / 6491 = 190 \text{ rem } 4473$`, ... | 0 / 0 / **1** / 4 (5) | the result of the search: `gcd(a, b) = g`, or the fraction in lowest terms | simulation; on `div_big` covered by `--exact-answer` |
| LE | A run of one token | `000...`, `Wait.`, `I'll write the response.` | 1 / 2 / 0 / 0 (3) | close, as today | in best config |
| LF | A contradiction in a hard problem it cannot resolve | MATH level 5 | 1 / 1 / 0 / 4 (6) | close, as today; Sandbox / oracle (P8-P9) | ceiling of the close |

- **Control's 73 stuck loops**, read the same way: A 6, B 21, C 11, D 1, E 8, F 26. General ended 29 of those keys right with no close at
  all; what kept them out of the loop is not shown (paired by question, not by path).
- **Classifier in simulation** (`tools/igc/loop_class.py`): 55 of 69 agree with the hand labels; every settle line it would write is
  exact by construction; a settle line is available on 49 of 69.
- **Settle probe** (2026-11-10, the 49 stored prefixes, on its own llama-server): a settle line alone ended 7 of 49 doubt loops;
  "settle, then close only if the loop resumes" 16; settle and close at once **32**, against today's close 31, with fewer tokens. So
  a settle line does not end a loop by itself; it goes with the close. The gain over today's close is not shown yet.

## Detection

Where a calculation is, and what it is. Builds as the source states; "engine" = Flash-Next (Gyro-M) · Strata engine.
Detail: [Detection](detect.md), [detection coverage plan](plans/2026-11-10-detection-coverage-plan.md).

| # | Pattern | Example / seen | Fix | Tested | Status |
|---|---|---|---|---|---|
| D1 | Question reader off in round 9, no reason recorded | it reads 30 of 120 frozen9 questions, all equal to gold (4B NVFP4 · PC layer) | turn it on (coverage plan slot 1); `read_exact` widened it (P5) | reader check 84/84 | fix planned |
| D2 | The `=` reader leaves points to the model, cause not split | 331 + 290 of 14,609 + 17,352 points not put (98 % put) | split the 621 by cause (echo skips, join holds, unread forms) | no | fix planned |
| D3 | No stop at `\approx` or `\equiv` | 116 of the 334 hand values `--check-hand` calls wrong sit on such lines (37 `\equiv` + 79 `\approx`) | stop at both (detection side of P11) | no | fix planned |
| D4 | Forms the reader cannot read | H3: 1,052 of 31,996 (unbalanced tails 342, roots 95, `...` decimals 89, huge powers 87 by design, fractions 70, spaced factors 52, fractional powers 52, pi in degrees 52) | none named | — | open |
| D5 | Point names read as functions | `P(2, 3) =`, `C(4, 6) =` got `0`; a polygon regex put gold 315 after the question's own wording | `4dc5e546`: bare `C(`/`P(` not read, `P(n,k)` with k > n is a point | replay: no value on those texts | fixed, replay |
| D6 | Ops written from single target questions | `polar_triangle_chain` wrong in most of its domain; `triangle_wave_intersections` | both deleted, the rest moved off the inline path; the "WON" claims withdrawn | — | fixed |
| D7 | MathBackend reader misreads | `gcd(1,000, 25)` = 1 (thousands split); Greek letters as functions; a determinant taken as the whole text's value; `Vmatrix` | `a48a0570`, `e2285304`, `67ca97d5` | 387 ops, 31,064 cases, 0 disagree | fixed, measured |
| D8 | A function the text defined was read | the latest `f(x) = ...` put a wrong value at 11 places of stored control answers (`w(0)`) | the reading removed | — | fixed |
| D9 | Engine parser ignored a scaling word | `half of 68 + 10 =` read as `68 + 10`, 78 put (gold 54); 2 of 4 runs did not recover (engine) | `scales_next` | six rounds all right, none wrote 78 | fixed, measured (small) |
| D10 | Engine parser walked across a sentence end | `909. 59,237*900,000 =` refused; the model's wrong value stood | `. ` ends the walk | no number | fixed, not measured |
| D11 | Engine parser: sign merged into the token, `\times` unknown | `a+b=-1+5=` put 6; `4 \times 125 - 2 =` put 123; MATH-500 integers 55 on vs 59 off | `af41da64` (operators added, refuse after 1,010 binding tokens) | no rerun | fixed, not measured |
| D12 | Engine parser: a number against a letter | on16: 16 wrong of 957 puts, 13 of them `d12^2=` read as `12^2` | `1be41906` | class not re-counted | fixed |
| D13 | Percent read as arithmetic | `100 - 30%` is 70 to a reader, 99.7 to arithmetic | percent only as a factor | the ABBA build lacked it | fixed, not measured |
| D14 | A name as an operand ends the parse and the run sticks | `gsm8k-34`: the model multiplied with a name `C` (engine, GSM-Hard) | none | — | **open** |
| D15 | `--check-hand` false corrections | replay: 6 of 6 corrections false (tails of longer expressions, parity, `\to`, degrees); H3 found more forms, each now a test; flags 432 -> 303 | rules with tests | a count, not precision | fixed, replay (unconfirmed) |
| D16 | `--check-hand` false flags still open | 12 of 64 sampled flags not errors: 7 long-multiplication carry (`$8 \times 3 = 24 + 2 = 26$`), 5 chain splits; precision 51/64; 270 unread | carry and chain-split rules, a gate before it runs | no | fix planned |
| D17 | Latent signals see wrong values, loops and doomed traces badly | wrong value AUROC 0.705 (4B) / 0.668 (Qwen3.6); loops: 24 onsets, router 11/24 at k=2; on a MIP arm probe onsets fall to 80 / 59 / 50 % at k = 2/8/32 | none | — | open |
| D18 | Offline precision did not predict live firing (sidecar, `UnifiedDetector`) | sidecar fired 3 times live, one before a wrong answer; "0 destructive false alarms" contradicted by #14 | sidecar off; fusion acts only with the exact parse | — | withdrawn |
| D19 | The question reader saw "one computation" only in the test's wording | 0 / 80 for `{e} = ?`, `Can you calculate {e} for me?`, `Please simplify {e}.`, `What do you get when you compute {e}?` (80 / 80 for 6 other wordings) | `read_exact` widened from the situation's statement: request grammar around one expression, LaTeX read, any other word reads nothing | held out: 8 new wordings 80/80, LaTeX 80/80, 10 meaning-changing wordings 0/80; MATH-500 6 reads, 6 gold; GSM8K 0 reads of 1,319; round 9 replay unchanged (16 / 0) | fixed, offline |

## Inject

How a value or note goes in, and how the model takes it. Detail: [Inject](inject.md), [settle adoption](plans/2026-11-10-settle-adoption-plan.md).

| # | Pattern | Example / seen | Fix | Tested | Status |
|---|---|---|---|---|---|
| I1 | The ` (exact)` note draws a re-check (4B) | re-check word within 300 characters 41 -> 65 % and 35 -> 57 %; first word `wait` in 42/126 with the mark vs 12/126 without (4B bf16 · A100) | `--inline-plain` | measured | fixed, in best config |
| I2 | The opposite on Gyro-M: an unmarked value is distrusted | `= 2939885984? Let me compute carefully.` looped to the limit (engine, one unpaired run) | decision: the plain value for every model | unconfirmed | superseded |
| I3 | The model writes ` (exact)` after its own wrong value | engine | none: the note is no proof of source | — | open |
| I4 | A digit joined onto a put | `-5` then the model's `75` read as `-575`; `8^{34} = ...` then zeros | `--no-digit-join` | General 92/21,398 -> 0/17,485; Qwen profile 234/35,319 -> 0/15,282 | fixed, in best config |
| I5 | The model took a delivered sum's value as its first term | `5*8387270 + 8 = 41936358 (exact) + 8 = 41936366` (engine) | `--igc-sum-note 1` | 13 events in 10/24 runs -> 0/24; tir4b has no sum note | fixed (engine) |
| I6 | A system hint copied into a new loop | "marked values come from a calculator": re-check sentences 290 -> 333 (Gyro-M, 60 cases) | hint rejected | measured | fixed (not adopted) |
| I7 | `stop checking` ends the thinking with work left | 10 of 294 runs ended on the repeat note, 5 with a step to go (engine) | wordings 1 and 2 built | queued, not run | fix planned |
| I8 | The third-repeat close cuts the thinking mid-work | 61 of 294 runs ended on it, 6 wrong on answerable questions; `math-3`, `math-119`; lost LCB row 3364 (engine) | none | same-exe A/B stopped at 9 of 26 | open (unconfirmed) |
| I9 | Values put in the middle of the model's own method | `--inline-calc` alone, thinking off: 81/105 vs base 85 (4B · L4) | the General Style, arithmetic mode | measured | superseded |
| I10 | The General Style alone raises re-checking and loops | re-check words +14 %; 197/240 vs control 209, every wrong answer a loop (4B bf16 · A100) | the loop closers | measured (round 9: 215 vs 161) | fixed by the closers |
| I11 | A correction read and redone by hand | `--check-hand` corrected 22 values on 17 rows; the model redid the work after each; 116 vs base 121 (4B · L4) | none | measured | open (not in best config) |
| I12 | Old model profiles write false forms | residue guess held on the 4B (5/6), failed on R1/QwQ (22/61) | profiles left unused | measured | fixed |
| I13 | Soft logit override does nothing | +8: 19/48 vs 21/48, 2.8x wall time (4B bf16 · A100) | none | measured | kept as option |
| I14 | Re-check hold-down harms | with loop-close: cut answers 3 -> 9, tokens per right answer +43 % | dropped | measured | dropped |
| I15 | Wording does not end a doubt loop | after `Checked exactly: $2928 / 28 = 732/7$ holds.` the model wrote `Let me re-check $2928 / 28$.`; voice line 6/49 (4B NVFP4 · PC layer) | the line goes with the close | settle_close 32 vs close 31, one seed | fix planned (more seeds) |
| I16 | ThinkBrake never fires in a doubt loop | 47 loops: `</think>` never within 1.0 nat of the top token | none | measured | ceiling |
| I17 | Closers' own misses not in the loop section | V7 closed 2 right answers early (`stdmath-d93ff30b#0`, `1adf1706#0`, unpaired); shape cap's only act ended wrong (sign lost); cover cap has no live run | — | — | open (unconfirmed) |

## Compute

What computes the value. Detail: [Compute](compute.md).

| # | Pattern | Example / seen | Fix | Status |
|---|---|---|---|---|
| C1 | Wrong values in MathBackend ops behind an earlier "0 disagreements" claim | found later; two ops deleted | fixed or refused; now 387 ops, 31,064 cases, 0 disagree | corrected |
| C2 | Ops with no work budget | `canon('(x+y+z+w)^200')` 15-20 s before refusing; `count_real_roots`, `sturm_sequence`, `poly_gcd` | none (not reachable from the text) | open |
| C3 | Conventions a model may not share | `stirling1` unsigned, `sigma(n, k)` n first, `falling_factorial(5, 7) = 0` | documented, not changed | by design |
| C4 | The engine path has no value audit | "0 wrong values" is a tir4b claim only | none | open |
| C5 | Qwenputer too slow, not integrated | about 5 s per call (`17*23` 5,079 ms) vs 0.066 ms for ABI1; 8 of 29 math cases firmware errors or refusals | replaced by ABI1-3 for arithmetic | replaced |

## Sandbox and tools

Code run inside the reasoning. Detail: [Sandbox and tools](sandbox.md).

| # | Pattern | Example / seen | Fix | Tested | Status |
|---|---|---|---|---|---|
| S1 | The model rarely opens a block itself | Flash-Next 2 of 450 runs; 4B 2 in 82 rows; `--native-tool` 0 calls in 9 | forced block (built); 4B fine-tune (built) | fine-tune: no result (no CU) | open |
| S2 | Spontaneous calls come after the thinking, not in it | 203 of 205 (greedy); sampled 0 inside the thinking (GPQA 0/27, HLE 0/64) | none | — | open |
| S3 | Drafts written, never called | most of 15 blocks "printed nothing" | `kNothingPrinted`, `--igc-sandbox-call`, `auto_call` | see S4 | fixed (mechanism) |
| S4 | A forced `print(` gives a blind constant | HumanEvalFix: 12 of 22 forced calls printed a constant | none | — | open |
| S5 | Close after a print fires on a wrong program | 1 wrong answer from the close; 13 runs printed nothing (41 % of the arm's tokens) | none | +4 right, p 0.29 | kept as option (unconfirmed) |
| S6 | The close loses on coding | LCB hard 20, paired, A100: plain 17 vs block + close 13-15 | ruled out for coding | measured | ruled out |
| S7 | Forced blocks too long on the 4B | 40 of 82 too long (reasoning in comments); 58 vs 56 (4B bf16 · A100) | none | measured | open |
| S8 | The hypothesis loop never completes | "a real mismatch, then a pass" 0 in every arm; 3017 `expect` used a made-up value and the why-line called a right program wrong | none | measured | open |
| S9 | An unfinished brute force read as a wrong approach | returned `None`; brute gave a value 3 times in 181 blocks | `c1460ad2` (`kStubNote`) | no arm | fixed, not measured |
| S10 | Tool calls built on a made-up premise | GPQA: 26 calls on an invented sign convention, right -> wrong; SuperGPQA `b6b43614` | none | — | open |
| S11 | A hint cuts the reading of the question | the line hint lost 5 rows on MMLU-Pro (11 vs 16); no hint moved wrong -> right | tuning stopped by a rule set before | measured | stopped |
| S12 | A trust line ignored | `--trust-line cort`: the 4B re-checked a printed -530 for 12,000 tokens | none | — | open |
| S13 | The kernel handoff showed old prints | a first block moving to the kernel | `8569a1a2` | — | fixed |
| S14 | A dead kernel written to (past 120 s) | — | `04cc1b26` | — | fixed, not measured |
| S15 | Box bug failed blocks | `_Stays` had no `poll()`: 20 of 45 blocks failed in `pc5` | `3aed1e9b` | — | fixed |
| S16 | Box isolation gaps | deny ACE does nothing against an AppContainer on 26300; disk limit is a poll (43-143 MiB at a 32 MiB limit); can read `C:\Windows` and HKLM | none | — | open |
| S17 | Names lost in the served reasoning tool | names do not reach the next round unless the turn moved to the kernel | engine side of Strata-xeno#291 | never run | fix planned |

## Learned and latent

Heads on the hidden state; the latent path is paused. Detail: [Learned and latent](learned.md).

| # | Pattern | Example / seen | Status |
|---|---|---|---|
| T1 | Input heads do not generalize to new templates | 0/12 exact operand tuples on template-held-out cases; 16/32 false calls on each new axis | gate never passed |
| T2 | Adam overshoot | loss 2.1e-5 -> 1.03 at epoch 70 | fixed (cosine LR) |
| T3 | The value is not only at the literal digit | literal value at the attention argmax lowered tuples 50 -> 39, 77 -> 29 | rejected |
| T4 | Retained head: digit-3 errors and false calls | all 96 active-digit errors on digit 3; 112/384 quote-copy false-call candidates | open |
| T5 | Warm start forgets | 22 right legacy tuples turned wrong | open |
| T6 | Readiness misses a new family | `en-exact-result` 0/12 selected; more data made it worse (110/112) | open |
| T7 | Bundle v2 does not generalize | 0/120 in-scope calls, 16/40 false calls on GSM8K | withdrawn |
| T8 | Residual write is not selective | rewrote right values at up to 59/60 control points | not adopted |
| T9 | No whole value at `=` in the residual | digit 3 reads 0.15 at L28 (digits 1-2: 0.99 / 0.89) | ceiling |

## Measurement and instrument errors

Errors in how a result was measured; each one changed a number that was quoted. Full list with ids: [ERRATA](ERRATA.md).

| # | Pattern | Example | Status |
|---|---|---|---|
| M1 | An edit that never applied | `on-07` repeated `on-06` | withdrawn |
| M2 | False determinism | "engine repeats 27/27": 48 of 49 text pairs diverge at T 0 | withdrawn |
| M3 | "Paired" comparisons that were path noise | vLLM A100 control token-identical on 2/104 | corrected |
| M4 | Unpaired, different-exe or different-session arms | PC block-2048 5/6 reversed to 13/20 when paired | corrected |
| M5 | Greedy wins matched by sampled controls | GPQA 11/13 vs 6/13: sampled no-tool controls turn most misses right | withdrawn |
| M6 | Regression to the mean | rows picked as misses turn right on any new path | rule: fresh paired baseline |
| M7 | Scorer faults | GSM8K 25/40, MATH 45 % (`I hope it is correct.`), every root answer wrong (160 -> 173 of 213), LCB `pow` | fixed |
| M8 | Audit blind to a class | `value_audit.py` skipped text after a letter: on16 16 wrong of 957 unseen | fixed (`1be41906`) |
| M9 | Audit false flags | 7 puts `1.18 \times 10^9 / 4 \times 10^3 = 295000` flagged, all exact; "0 wrong of 51,015" corrected to 39,046 | corrected |
| M10 | Over-counted failure review | third-repeat closes matched on thinking that ended by itself | corrected |
| M11 | Labeller faults | rule pointer 46/411 -> 21/293; sidecar "65 % exact" while all-`none` scores 69 % | corrected |
| M12 | Standard error on the wrong unit | computations counted as independent; per row: logit8 - ctrl = -0.9 [-6.0, +4.9] | corrected |
| M13 | Small-sample AUROC inflated by one trace | digit confidence 0.984 -> 0.705 | corrected |
| M14 | Token caps that cut answers | the 20,000 cap cut 24 of 48 in control | rule: full context every test |
| M15 | An arm run without execution | `--hint-calc` without `--execute`: outputs made up | withdrawn |
| M16 | The bench stuck check stopped a plain arm early | a 2-token cycle stopped before the engine could close | noted |
| M17 | Candidate-close replay claims | replay `min_tokens=1500` vs live 0, one of three paths | withdrawn |
| M18 | Development-set tuning presented as results | pc9 156/160: rules set reading the same rows; base 114-121 | rule: no test tuning, frozen sets |
| M19 | Wrong gold | GSM-Hard: 20 cases wrong in both arms | noted |
| M20 | A gate too short | the 73-token numeric gate missed drift at 2,048 positions | corrected |

## Patterns MIP itself caused (fixed in the put)

| Pattern | Seen | Fix |
|---|---|---|
| A put spelled `9/2` for `0.9 \times 5` broke the model's decimal sum into a loop | DeepMind 6890 | `spelled()`: a terminating decimal is put as a decimal, the model's own spelling |
| A put on `= 15/13 (exact)` after `= 15/13` repeated: `= 15/13 = 15/13 = ...` until `stuck` | #306 | nothing new is put; the value is left to the model and the loop checks |
| The same expression put back to back many times (100 times `-1/2` on `livebench-652451e2`) | #13 | echo guard: the 3rd put in a row within 64 tokens is skipped |
| A nudge that cut 64 tokens of a line-back loop left the loop and ended `stuck` | #304 | nudge dropped; the close acts |
| The answer-close cue ending in `{` lost minus signs (P6) | cue probe | cue ends before `{` |

## Rule scorer errors (fixed)

| Answer | Read as | Fix |
|---|---|---|
| `dm_div_big-9977#0`, `dm_simplify_surd-9937#0` | a box asked about and rejected (`\boxed{70468}? No`) taken as the answer | a box followed by `?` on its line is not the answer (#24) |
| `dm_mul_div_longer-5942#1` | `275 or -275` counted right | two different values in the last box line: no answer |
| `stdmath-2aff3e30#0` | `0.2` for `1/5` in a vector counted wrong | vectors compared entry by entry |
| `stdmath-9f7f1563#0` | `3<\lambda\le4` for `(3,4]` counted wrong | an inequality read as an interval |
| `stdmath-bf7ece2b` x2 (a3) | `\{-2, 1-\sqrt{5}, 1+\sqrt{5}\}` for `\{1\pm\sqrt{5},-2\}` counted wrong | set braces stripped before the values are compared (`2fcfbb9`) |
| `dm_simplify_surd-3480#0` | the API reader counted `9216 if the sum is 0` right | an API reader error; the rule is right |

## ภาษาไทย

### หลักการแก้: แก้ที่สถานการณ์ ไม่ใช่แก้เข้าหาชุดทดสอบ

ผู้พัฒนา (2026-11-11): ชุดทดสอบเป็นเครื่องมือหาว่า**สถานการณ์ไหนที่โมเดลพลาด** การแก้ต้องตั้งเป้าที่สถานการณ์นั้น ซึ่งเกิดได้กับชุดอื่นที่เป็นแบบเดียวกันหรือตอนใช้งานจริง
ถ้าแก้แล้วคะแนนชุดทดสอบขึ้นเพราะไปจับถ้อยคำ, module หรือรูปเฉลยของชุดนั้น ถือว่าเป็นการ tune ไม่ใช่การแก้

1. เขียนสถานการณ์เป็นคำพูดโดยไม่อ้างชื่อ module ถ้อยคำของชุด หรือชื่อข้อ ถ้าเขียนแบบนี้ไม่ได้ ก็ยังไม่ใช่รูปแบบความผิด
2. สร้างตัวตรวจจากคำอธิบายสถานการณ์นั้น ไม่ใช่จากข้อที่เจอ
3. ทดสอบตัวตรวจกับถ้อยคำหรือชุดที่ไม่ได้ใช้ตอนสร้าง (กำหนดรายการไว้ก่อนรัน) แล้วรายงานแยกตามตระกูล
4. นับหลักฐานเป็นจำนวนโจทย์ ไม่ใช่จำนวนคำตอบ โจทย์ข้อเดียว × 2 seed นับเป็นหลักฐานชิ้นเดียว
5. วัดผลกับข้อที่ไม่ได้ใช้ตอนสร้าง (frozen10 และอีกตระกูลหนึ่ง) ต้องไม่มีข้อที่ถูกแล้วกลายเป็นผิด

**ตรวจครั้งแรกตามหลักนี้:** ตรรกะของ `--exact-answer` เป็นระดับสถานการณ์ แต่ตัวอ่านโจทย์ที่ใช้เริ่มการทำงานผูกกับถ้อยคำของชุดทดสอบ
ลองถาม 80 ข้อเดิมด้วย 10 ถ้อยคำ อ่านได้ 6 แบบ (80/80) แต่อ่านไม่ได้เลย (0/80) 4 แบบ คือ `{e} = ?`, `Can you calculate {e} for me?`, `Please simplify {e}.`, `What do you get when you compute {e}?`
**แก้แล้ว (D19):** ขยายตัวอ่านจากคำอธิบายสถานการณ์ คือรอบนิพจน์มีได้แค่คำขอให้คำนวณ คำอื่นที่อาจเปลี่ยนสิ่งที่ถาม (ปัด, ฐานสอง, เศษ, ยกกำลังสอง, หน่วย, ตัวแปร) จะไม่อ่าน และอ่าน LaTeX ได้ ทดสอบกับรายการที่กำหนดไว้ก่อนรันและไม่ได้ใช้ตอนเขียน: ถ้อยคำใหม่ 8 แบบอ่านได้ 80/80, LaTeX 80/80, ถ้อยคำที่เปลี่ยนความหมาย 10 แบบอ่าน 0/80, MATH-500 อ่านได้ 6 ข้อตรงเฉลยทั้ง 6, GSM8K อ่าน 0 จาก 1,319, replay รอบ 9 เท่าเดิม

หน้านี้บันทึกทุกรูปแบบที่ทำให้โมเดลตอบผิดซึ่งอ่านด้วยมือแล้ว ว่าเกิดอะไร เจอที่ไหน แก้ยังไง หรือคาดว่าวิธีไหนจะช่วย และทดสอบแล้วหรือยัง
รูปแบบที่แก้แล้วยังอยู่ในหน้านี้ เพื่อใช้ตรวจโมเดลหรือรอบถัดไป ตามกฎของผู้พัฒนา: รูปแบบที่เกิดขึ้นได้ต้องแก้ทุกแบบ ไม่ว่าจะเจอน้อยแค่ไหน

| # | รูปแบบ | เจอ | วิธีแก้ | ทดสอบแล้ว? | สถานะ |
|---|---|---|---|---|---|
| P1 | ค่าถูกแต่ตอบเป็นทศนิยมหรือส่วนจำนวนเต็มแทนเศษส่วนเป๊ะ | รอบ 9: 6 | ผู้พัฒนา (2026-11-11): นับถูกพร้อมหมายเหตุว่ารูปไม่ตรง โดยเขียนคำตอบกับเฉลยไว้ในหมายเหตุ ใช้กับทศนิยมที่ปัดหรือตัดถูกของเฉลยที่เป็นทศนิยมจบไม่ได้ ส่วนจำนวนเต็ม เครื่องหมายหาย หรือหลักผิดยังนับผิด | รอบ 9: General 215 -> 218 (หมายเหตุ 3), control 161 | แก้ในตัวตรวจคะแนน ทุกแขนเท่ากัน |
| P2 | ตอบสองค่า หรือลังเล | รอบ 9: 3 | `--answer-rewind` ตัด loop แล้วให้โมเดลเขียนเอง | ยังไม่รันจริง | สร้างแล้ว รอบ 10 |
| P3 | ค่าผิดจาก loop ถูกใส่กล่องตอนปิด | รอบ 9: 3 | `--answer-rewind` | ยังไม่รันจริง | สร้างแล้ว รอบ 10 |
| P4 | อ่านลำดับการคำนวณในโจทย์ผิด | รอบ 9: 1 | `--question-quote` ยกข้อความในโจทย์มาให้เมื่อสำเนาของโมเดลต่างจากโจทย์ | ทำงาน 15 จาก 442 คำตอบ คัดลอกผิดจริงทั้ง 15 | สร้างแล้ว รอบ 10 |
| P5 | `simplify_surd` วนกับการนับวงเล็บ | รอบ 9: 2 | `--question-quote` ตอนปิด loop | ยังไม่รันจริง | สร้างแล้ว รอบ 10 |
| P6 | เครื่องหมายลบหายหลังประโยคปิด (`{-` เป็น token เดียว) | cue probe | ประโยคปิดจบก่อน `{` | 41 → 100 จาก 112 | แก้แล้ว |
| P7 | นับตัวอักษรในสตริงของโจทย์ผิด | รอบ 9: 2 | `--letter-count` นับจากสตริงที่โจทย์ถามถึง ใส่ตอนเริ่ม | offline ผ่าน 14/14, GSM8K อ่าน 0 | สร้างแล้ว รอรันจริง |
| P8 | เลือกนิยามผิด เก็บรากเกินมา (arccot) | รอบ 9: 2 | `--verify-gate` เปิดโปรแกรมให้โมเดลตรวจคำตอบแต่ละตัวในสมการเดิมก่อนปิดการคิด | MATH-500 ทำงาน 11 ข้อ, GSM8K 0 | สร้างแล้ว รันในแขน C |
| P9 | เดาค่าสูงสุดหลังค้นนานมาก | รอบ 9: 2 | `--loop-hint` ตัด loop แล้วเปิดโปรแกรมให้ทดสอบความคิด | ยังไม่เคยวัดบน PC | รันในแขน C |
| P10 | โจทย์ไม่ระบุการจัดวาง ("as shown") โมเดลวนไล่ทีละแบบแต่ไม่คำนวณตรวจ | รอบ 9: 2 | `--loop-hint` (Sandbox แขน C) ให้โปรแกรมตรวจแต่ละการจัดวาง; เดิมติดป้ายว่าเพดาน ซึ่งผิด เพราะแก้จากข้อความได้ r = 14/3 | ยังไม่รันจริง | สร้างแล้ว รอบ 10 แขน C |
| P11 | ค่าที่โมเดลเขียนหลัง `\approx` ผิด | รอบ 9: 1 | `--approx-put` ใส่ค่า 10 หลัก กันไม่ให้ `\times`/หน่วยต่อท้าย | replay: โมเดลพลาด 12%, อ่านผิด 0; ที่ `=` ค่าที่ยาวเกินจะเขียนเป็นขั้นได้ (`1.01^{30}`) ใส่แบบตัด 10 หลัก `1.347848915...`: 86 จุด 2 ข้อ โมเดลพลาด 14 อ่านผิด 0 | สร้างแล้ว เปิดใน channel run ถัดไป |
| P12 | ตัวปิดเอาค่ากลางทางของงานที่ยังไม่เสร็จไปตอบ | รอบ 9: 1 | `--answer-rewind` ตัด loop ทั้งก้อนแล้วสุ่มเขียนต่อใหม่ก่อนจะบังคับคำตอบ | ทำงานกับ 8 คำตอบในรอบ 9 มี 2 ข้อที่อาจเสีย | สร้างแล้ว รอรันจริง |
| P13 | loop ยังยาวเท่าเดิมแม้ค่าตอนปิดถูก | รอบรันเฉพาะข้อ: 70k token | มีสัญญาณจับ loop เร็วขึ้นในงานวิจัย | ยังไม่ได้วัด | วัดก่อนว่า token หายไปช่วงไหน |
| P14 | โมเดลอ่านค่าที่ใส่ให้แต่ไม่ใช้ เริ่มหารยาวใหม่ | A100 รอบ 5 | ใช้กฎแก้กล่องแทน | ผลรันจริงของ P1 | ถูกแทนด้วย P1 |
| P15 | รูปแบบที่ MIP ทำให้เกิดเอง | ดูตาราง | แก้ที่วิธีใส่ค่า | replay | แก้แล้ว |
| P16 | ตัวตรวจคะแนนด้วยกฎตัดผิด | รอบ 9: 6 | แก้กฎ | ตรวจรอบ 9 ใหม่ | แก้แล้ว |

### การวน (loop)

loop ทำให้ตอบผิดได้สองทาง ถ้าไม่มีตัวปิดจะวนจนหมดแล้วจบแบบไม่มีคำตอบ (control รอบ 9 จบ `stuck` 73 จาก 240) ถ้ามีตัวปิดก็อาจปิดด้วยค่าผิด (P3, P12)
หลักของผู้พัฒนา: loop มีหลายแบบ แต่ละแบบต้องจับคู่วิธีของมันเอง ไม่ใช้ตัวปิดตัวเดียวกับทุกแบบ

**ตามรูปร่างที่ตัวตรวจเห็น**

| # | รูปร่าง | ตัวตรวจและสิ่งที่ทำ | ผลทดสอบ | สถานะ |
|---|---|---|---|---|
| L1 | ข้อความเดิมซ้ำเป๊ะ | repeat check, V7 early cycle แล้วปิดด้วย A | V7 ไม่แตะข้อที่ถูก 0/915, จับเร็วขึ้น 212/243; A100 loop rows 25 → 37 จาก 48 | อยู่ใน best config |
| L2 | วนนับเลข รูปเดิมแต่ตัวเลขเปลี่ยน | counting check + `--guard-progress` ปล่อยงานที่คืบหน้าจริง | replay เปลี่ยนแค่ 3 ข้อที่ยังทำงานอยู่ | อยู่ใน best config |
| L3 | บรรทัดเดิมกลับมาโดยเปลี่ยนคำนิดหน่อย | line-back check | 23 ครั้ง ไม่ตัดก่อนเวลาเลย | อยู่ใน best config |
| L4 | หารยาวทีละหลัก ตัวเลขยาวไม่เท่ากัน | shape cap 1024 | A100 ปิดได้ที่ 55k token | อยู่ใน best config |
| L5 | หารยาวที่หลัก 0 ทำให้รอบไม่ซ้ำเป๊ะ | cover cap 32768 | ไม่แตะข้อที่ถูก 0/1,078, หยุดได้ 12/13 | อยู่ใน best config |
| L6 | วนต่อในคำตอบหลังปิดการคิดแล้ว | answer close B | A100 ข้อที่ถูกตัดลดจาก 3 เหลือ 1 | อยู่ใน best config |
| L7 | **ปิดเร็วเกินไป:** ปิดการคิดแล้ววนต่อในคำตอบ B เลยเอาค่าที่อยู่บนหน้าไปตอบ | `--exact-answer` (โจทย์คำนวณเดียว); decide layer | รอบ 9: รูปแบบนี้ 25 ข้อ ทำให้ผิด 3; ถ้าปิดการคิดอย่างเดียวไม่ทำให้ผิดเลย (41) | แก้แล้วสำหรับโจทย์คำนวณเดียว ที่เหลือยังเปิด |
| L8 | echo: ตรวจนิพจน์เดิมซ้ำหลังได้ค่า | echo guard | พบใน 7 จาก 48 คำตอบ | อยู่ใน best config |

**ตามสิ่งที่โมเดลวนซ้ำ** (69 ครั้งที่ปิดในรอบ 9)

| # | สิ่งที่วน | ช่วย / ถูกทั้งคู่ / ทำให้ผิด / ผิดทั้งคู่ | วิธีที่วางไว้ | สถานะ |
|---|---|---|---|---|
| LA | สมการที่ถูกอยู่แล้ว | 7 / 9 / **2** / 3 | บรรทัดยืนยันค่า | จำลองแล้ว อาจไม่พอ (P12) |
| LB | การอ่านข้อมูลในโจทย์ | 14 / 6 / 0 / 3 | ยกข้อความในโจทย์มาให้ตรงตัว | จำลองแล้ว จับข้อ swr ไม่ได้ (P7) |
| LC | รูปของค่าที่หาได้แล้ว (เศษส่วนหรือทศนิยม) | 5 / 4 / 0 / 2 | ใส่ค่าเป๊ะครั้งเดียวแล้วปิด | **`--exact-answer` ครอบคลุมแล้ว** |
| LD | หาตัวประกอบร่วมด้วยการหารลองทีละตัว | 0 / 0 / **1** / 4 | ใส่ผล gcd ให้ | จำลองแล้ว; `div_big` ครอบคลุมด้วย `--exact-answer` |
| LE | token เดิมซ้ำ (`000...`, `Wait.`) | 1 / 2 / 0 / 0 | ปิดเหมือนเดิม | อยู่ใน best config |
| LF | ขัดแย้งในโจทย์ยากที่แก้ไม่ได้ | 1 / 1 / 0 / 4 | ปิดเหมือนเดิม; Sandbox / oracle | เพดานของตัวปิด |

settle probe: บรรทัด settle อย่างเดียวหยุด loop ได้ 7 จาก 49, settle แล้วปิดทันทีได้ 32 เทียบกับปิดแบบปัจจุบัน 31 ยังไม่เห็นว่าดีกว่า

### หมวดอื่น (รายละเอียดอยู่ในตารางภาษาอังกฤษด้านบน)

| หมวด | ข้อ | ที่ยังเปิดหรือรอแก้ |
|---|---|---|
| **Detection** (หาตำแหน่งการคำนวณ) | D1-D18 | ตัวอ่านโจทย์ยังไม่ได้เปิดใน General (D1), 621 จุดที่ไม่ได้ใส่ค่ายังไม่ได้แยกสาเหตุ (D2), ไม่หยุดที่ `\approx`/`\equiv` (D3), รูปที่ยังอ่านไม่ได้ 1,052 จุด (D4), ชื่อตัวแปรเป็นตัวถูกดำเนินการแล้วรันค้าง (D14), `--check-hand` ยังเตือนผิดเรื่องตัวทดกับการแยกสาย (D16), สัญญาณแฝงจับค่าผิดได้ไม่ดี (D17) |
| **Inject** (ใส่ค่าและคำบอก) | I1-I17 | โมเดลเขียน `(exact)` ต่อท้ายค่าผิดของตัวเอง (I3), `stop checking` และการปิดครั้งที่สามตัดงานกลางทาง (I7, I8), โมเดลทำการแก้ไขซ้ำด้วยมือ (I11), คำพูดอย่างเดียวหยุด loop ไม่ได้ (I15), V7 ปิดข้อถูกเร็วเกินไป 2 ข้อ (I17) |
| **Compute** (คำนวณค่า) | C1-C5 | op ที่ไม่มีงบเวลา (C2), เส้นทางของ engine ยังไม่มีการตรวจค่า (C4) |
| **Sandbox** | S1-S17 | โมเดลไม่เปิดบล็อกเอง (S1), เรียกเครื่องมือหลังคิดจบแล้ว (S2), `print(` ที่บังคับได้ค่าคงที่ (S4), hypothesis loop ไม่เคยครบวง (S8), เรียกเครื่องมือบนสมมติฐานที่แต่งขึ้น (S10), กล่องแยกยังมีช่องโหว่ (S16) |
| **Learned / latent** | T1-T9 | หยุดพักอยู่ หัวที่เรียนแล้วใช้กับรูปแบบโจทย์ใหม่ไม่ได้ |
| **การวัดผลผิด** | M1-M20 | แก้หรือถอนหมดแล้ว และกลายเป็นกฎ: จับคู่ในรอบเดียวกัน, ใช้บริบทเต็ม, ไม่ปรับกฎให้เข้ากับชุดทดสอบ, ใช้ชุดแช่แข็ง |

### คำสั่งรูปคำตอบของข้อ DeepMind (ตัดสิน 2026-11-11)

ข้อ DeepMind ในรอบ 1-9 ลงท้ายด้วย `Give the final answer as a single number in \boxed{} at the end.` ซึ่งเป็นข้อความที่**เราเขียนเอง** ชุดข้อมูลจริงมีแค่โจทย์กับคำตอบเป๊ะ
ในรอบ 9 มีคำตอบที่อ้างคำว่า "single number" ตอนเลือกรูปคำตอบ General 21 ข้อ (ผิด 7) control 23 ข้อ (ผิด 14) เกือบทั้งหมดเป็น `div_big`
ผู้พัฒนาตัดสิน: ไม่เปลี่ยนเป็น "exact" เพราะจะชี้นำให้ตรงกับเฉลยและห่างจากชุดข้อมูลมากขึ้น รอบ 10 ใช้คำสั่งเดิมเพื่อเทียบกับรอบ 9 ได้ตรง
ตั้งแต่ frozen10 ข้อ DeepMind ใช้คำสั่งกลางเดียวกับข้อ MATH (`Put the final answer in \boxed{} at the end.`) ไม่ระบุรูป ให้โมเดลเลือกเอง แล้วใช้กฎหมายเหตุของตัวตรวจกับทุกแขน
สิ่งที่อ้างได้: "โจทย์จาก DeepMind Mathematics พร้อมคำสั่งรูปคำตอบของเรา ตรวจด้วยกฎของเราและ API" ไม่ใช่ตัวเลข leaderboard และห้ามเทียบตัวเลข DeepMind ก่อนและหลัง frozen10 ตรง ๆ
