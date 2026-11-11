# Round 9's failures and the fixes for them (2026-11-10)

> **Current state (2026-11-11):** `--exact-answer` is withdrawn (ERRATA MR-35 to MR-37); where the tables below name it as a fix
> (failures 1-3, "Queued" item 1, step 4a), read [Changed: `--exact-answer` withdrawn](#changed-2026-11-11---exact-answer-withdrawn).
> What round 10 runs, and what blocks its launch, is in [Round 10 as it runs](#round-10-as-it-runs-2026-11-11) at the end.

**Build:** 4B NVFP4 W4A8 · llama.cpp · PC layer, round 9 (half A `c093b2f`, half B `255ec4e`), General arm, 232 of 240 pairs.
Every wrong answer of the General arm was read by hand: 23 wrong by both the rule and the API reader, and 6 more where the two
readers disagree. The control arm's 75 wrong answers are almost all loops that ran out (`stuck`), which the closers already answer.

## The failures, by kind

| # | Kind | Answers | n | Fix |
|---|---|---|---|---|
| 1 | Right value, wrong form: the exact fraction written, then a decimal or the integer part boxed; or the old cue cut the answer | `dm_div_big` 1395×2, 2783#0, 4138×2, 5070×2, 9991×2; `dm_mixed_longer-9951#0`; `dm_mul_div_longer-9991#1`; (9977#0, see 8) | 12 | `--exact-answer` (built, `9beabf0`): replay 12 wrong -> right, 0 right -> wrong |
| 2 | The model read the question's precedence wrong (-12.375 for -198); `question_read` reads -198, but the box rule only acts on a shortening of a non-integer value | `dm_mul_div_longer-1505#0` | 1 | **`--exact-answer`, widened:** when `question_read` reads the whole question as one computation, the last box holds the exact value whatever the model wrote, integer values included (the developer: when the question is the computation, its value is the answer) |
| 3 | `simplify_surd` loops on parenthesis counting; `question_read` reads none of the 10 surd questions (`sqrt`, `**`) | `dm_simplify_surd` 3480×2, 9937#0 | 3 | **`question_read`, widened:** read `Simplify <expression with square roots>` and compute it exactly as `a + b*sqrt(n)`, so the module falls under `--exact-answer` |
| 4 | No put after `\approx`: `(1.01)^{30} \approx 1.3448888` (true 1.3478489); both arms wrong | `stdmath-45fa3ef2#0` | 1 | **Changed 2026-11-11:** stop at `\approx` as at `=` and put the value to 10 significant digits (scientific form when very small or large): a more precise value is a true approximation under every reading, deliberate estimates (`\approx 34000`) included, so no estimate rule is needed. First the reader for a decimal base to an integer power (D4: `(1.01)^{30}` is read by neither evaluator). Published work shows the case is general (MATH 401, MedCalc audit); ours to test: 0 false reads in replay of every stored `\approx`, round 9 and the A100 traces |
| 5 | The answer close boxed an unfinished step: the model looped doubting a right `72 - 14 = 58` and the cue took 58 | `stdmath-fbf10ed4#1` | 1 | **Changed 2026-11-11:** the forced final-answer cue is the failing mechanism (detection-extraction gap, arXiv 2604.06613: forced extraction fails on 42 % of prefixes that free continuation answers; round 9: all 3 hurt answers came after a forced cue). Fix: when the answer loops, do not force the answer; cut the loop and let the model continue (with a confirm line for a true equation), and force the cue only as the last resort. Ours to test: which continuation ends the loop on the 4B (the settle probe saw free writing run to the cap), probe on the 25 L7 answers + the A100 answer-close answers |
| 6 | Letters of the question's string miscounted | `dm_swr_p_sequence` 3480#0, 9951#1 | 2 | **Changed 2026-11-11:** in. The case is general (The Strawberry Problem, arXiv 2505.14172: character-level tasks fail from tokenization), not DeepMind's. Fix: software reads the string the question quotes, in any wording, and puts the letter count when the model starts counting; the count of a quoted string is exact. Ours to test: the reader on wordings and string tasks it was not written from, 0 false reads |
| 7 | Reasoning, not computation: arccot convention, a figure the model cannot see, a hard maximum guessed | `stdmath` e3263989×2, 370853d3×2, 97b18f58×2 | 6 | Not a put. arccot: checking candidates against the equation is the right structure (arXiv 2608.28725), but the branch has two conventions, so software cannot decide it under every reading: a Sandbox check by the model (A100). The maximum: a numeric search is evidence, not a value: brute-force oracle in the Sandbox (A100). The figure (re-read 2026-11-11): the text leaves the layout open but fixes it by tangency (r = 14/3 by testing layouts): the Sandbox (`--loop-hint`, arm C) |
| 8 | Rule scorer | `dm_div_big-9977#0` and `dm_simplify_surd-9937#0` (a `? No` box, #24); `dm_mul_div_longer-5942#1` ("275 or -275" counted right); `stdmath-2aff3e30#0` (`0.2` for `1/5` counted wrong); `stdmath-9f7f1563#0` (`3<\lambda\le4` for `(3,4]`) | 5 | Fix the rule; re-score round 9. Read `dm_simplify_surd-3480#0` by hand: the API reader says right, the last box is `9216 if the sum is 0` |

If 1-4 and 8 hold, up to about 17 of the 23 can move; replay has to show it.

## Steps

1. **Round 9's last 8 pairs (a3)**, on round 9's code (`c093b2f`, worktree), the 4 moved MATH-500 L5 rows x 2 seeds, both arms on one card,
   `-np 4` (the 4070 SUPER: the 5060 Ti is disabled). Runs while steps 2-4 are written (GPU vs CPU).
2. **Code, test first** (`tools/test_igc_tir4b.py`, `tools/test_question_read.py`, the scorer's tests):
   - `--exact-answer` box rule for any value (2).
   - `question_read` for surd simplification (3), checked on all 10 frozen9 `simplify_surd` rows against gold.
   - `--inline-calc` at `\approx` (4).
   - the rule scorer (8).
3. **Replay on round 9**, both arms: wrong -> right, right -> wrong (must be 0), false puts (must be 0). Report each touch with the
   three questions (value right, needed, helped).
4. **Paired live test** on the PC layer: `div_big`, `mixed_longer`, `mul_div_longer`, `simplify_surd` (40 rows x 2 seeds): round 9's
   flags against round 9's flags + the fixes, one card per arm, rule + API reader. These rows are development rows now; the measuring
   round is frozen10.
5. **frozen10** with the fixes, if step 4 shows them working.

## Progress (2026-11-10)

| Step | State | Result |
|---|---|---|
| 1. a3 | running on the **4070 SUPER** (the 5060 Ti is disabled in Device Manager, code 22; not re-enabled without the developer) | a2's rows were on the 4070 too, so the pairs stay one card |
| 2. scorer (8) | done, `96cf412` | round 9 re-scored: General 204 -> **207** by the rule (API 208), control 157 unchanged. 4 wrong -> right (9977#0, 9937#0, 2aff3e30#0, 9f7f1563#0), 1 right -> wrong (5942#1, a hedge). The one disagreement left, `dm_simplify_surd-3480#0`, is the API reader's: the last box is `9216 if the sum is 0` |
| 2. `--exact-answer` any value (2) | done, `96cf412` | replay 19 acts, 13 wrong -> right, 0 right -> wrong |
| 2. surd questions (3) | done, `ce92287` | `read_exact`: 84 of 379 stored rows read, 84 of 84 gold, 0 wrong. Replay 24 acts, **16 wrong -> right, 0 right -> wrong**; control 0 acts |
| 2. `\approx` (4) | **measured, not built: the design does not hold** | see below |
| 1. a3 (done 2026-11-10) | done on the 4070 | General 8 of 8, control 4 of 8 (4 `stuck`). Round 9, 240 pairs: **215 vs 161** by the rule (API **216 vs 161**); with the form rule of 2026-11-11, **218 vs 161** (3 noted), General alone 59, control alone 5, p = 9e-13 |
| scorer: set braces | done, `2fcfbb9` | `\{-2, 1-\sqrt{5}, 1+\sqrt{5}\}` for `\{1\pm\sqrt{5},-2\}` (a3 `stdmath-bf7ece2b` x2) was scored wrong |
| 4a. targeted live run (2026-11-11) | done, `6c9331e` | the 11 rows the replay moved x 2 seeds, round 9's flags + `--exact-answer`: **22 of 22 right** (round 9: 7 of 22); 13 acts, 13 values right; all 9 `div_big` answers wrong in round 9 right with an act; 4 others right with no act (sample path). Loops as long as before. [inject.md](../inject.md#--exact-answer), #19 |
| 4. paired live test | **queued: runs inside the next round, not on its own** (developer 2026-11-11: test live together with another round so no time is lost) | 40 rows x 2 seeds (`div_big`, `mixed_longer`, `mul_div_longer`, `simplify_surd`), both arms, rule + API reader. **Add the 6 MATH-500 questions the widened reader reads** (D19: `Simplify $\sqrt{242}$.`, `What is the value of $513^2 - 487^2$?`, ... list from `results/exact-answer-r9/outside_reach.py`): the first live run of `--exact-answer` outside DeepMind's wording. Before it: measure why the 5060 Ti ran ~12 tok/s with ~30k-token answers in flight (hypothesis: each request resends the whole answer for ~18 tokens) |

**`\approx`: what the gate showed** (`results/exact-answer-r9/approx_scan.py`, `approx_scan_put.py`, round 9 stored text). General wrote
3,391 `\approx` with a decimal after it; the `=` reader reads the computation before 1,952 of them, and the model's value is off by more
than one unit in its last place on 316 (control: 739 read, 171 off). The 316 are three different things, and a put cannot treat them
alike:
- real slips (`5647/1237763 \approx 0.0045623755`, true `0.00456226...`, many times in `div_big-5070`);
- scientific notation, which the scan reads as its mantissa only (`3/2372493604 \approx 1.2645 \times 10^{-9}`): not wrong;
- deliberate estimates (`33869 \times 143 \approx 34000`, `2.2 \times 10^8 / 1.57 \times 10^7 \approx 14`): a magnitude, not a value.

And the `=` put's forms are wrong after `\approx`: it would put `0 \text{ rem } 3` or `0.0000...` (the long-division forms). The
motivating case is not even read: the reader refuses `(1.01)^{30}`. So a put at `\approx` needs its own design: a decimal form with
enough significant digits, the decimal power read, and a rule for estimates; first a scan that reads `\times 10^{k}` and splits the 316
by hand. Not in the paired test.

## Paused (2026-11-10, the developer)

The developer paused this work after the summary. Where it stands:
- a3 was at General 6 of 8, control 5 of 8 records (`a100/pc/r9a3-*`); `scripts/pc/pc_round9.sh` resumes it with `RESUME=1`.
- Waiting on the developer: re-enable the 5060 Ti (Device Manager, code 22) or run the paired test on the 4070 alone (~9 h vs
  ~4.5-5 h); `swr_p_sequence` by Sandbox or a sentence reader.
- Pick-up order: finish a3 -> round 9's final totals (rule + API) to #19 and the profile README 232 -> 240 (PR xenodeve/xenodeve#7)
  -> the paired live test (step 4) -> frozen10.


## Queued for the next live round (2026-11-11)

The developer: write it down now, and test live together with the next round so no time is lost. Everything below rides one round, both arms, rule + API reader, the three questions per touch:

1. `--exact-answer` with the widened reader (D19) on the 40 development rows x 2 seeds **and** the 6 MATH-500 questions it reads (never used here: the first live run outside DeepMind's wording).
2. P11 (put at `\approx`), P12 (no forced answer cue in an answer loop) and P7 (letter count of a quoted string), each that passes its offline gate by then, designed from the case's statement ([failures](../failures.md#how-a-pattern-is-fixed-the-situation-not-the-test-set)); see the research table below.
3. P8 and P9 stay out: not puts (a convention, a numeric search); they go to the Sandbox work on the A100. P13 waits for the measurement of where a long loop's tokens go.

## Research behind the remaining fixes (2026-11-11)

Searched before building, so a case already shown in published work is not re-proved here. What the literature settles, and what it
does not (our reader's correctness and whether the 4B adopts a put are still ours to measure). Abstracts and summaries read, not
every full paper.

| Fix | What published work shows | What it settles | Still ours to test |
|---|---|---|---|
| P11 put after `\approx` | MATH 401 (arXiv 2304.02015): persistent weakness on decimal exponentiation and large numbers. MedCalc-Bench audit (arXiv 2603.02222): models often get exponentiations and logs of decimals slightly wrong, and a ~0.7 % error compounds past a ±5 % band. Calculator / SymPy checking of a model's own calculations (Language hooks, arXiv 2412.05967). A silently wrong tool costs up to 47 % (arXiv 2406.19228) | the case is general, not a test artefact; software should compute it; a false put is worse than none | the reader on `(1.01)^{30}` (D4) and 0 false reads in replay |
| P12 close on a step | The detection-extraction gap (arXiv 2604.06613): forcing an answer out of a prefix fails on 42 % of prefixes whose free continuation recovers it; forced early exits lose 41-62 points; free continuation as the exit gains 1-5 | **forcing the final-answer cue is the failing mechanism**, matching round 9's L7 (3 hurt, all after a forced cue); the fix should let the model continue rather than force a value | which form of "continue" ends the loop on the 4B (the settle probe saw free writing run to the cap: 16 vs 31) |
| P13 long loops | Circular reasoning (arXiv 2601.05693): loops are preceded by a surge of reflective sentences (`But`, `Wait`); REFRAIN (arXiv 2510.10103) and Word Salad Chopper detect redundant repetition earlier; entropy after `</think>` (arXiv 2509.26522) flags when more thinking stops paying, 12-22 % fewer tokens, no accuracy loss | earlier loop detection is possible from signals we do not read yet | first measure where 2783#0's 70k tokens went |
| P7 letters of a string | The Strawberry Problem (arXiv 2505.14172, EMNLP 2025): character-level tasks fail from tokenization and emerge late in training; letter-count errors grow with word length and letter frequency (arXiv 2412.18626) | the case is a general weakness of tokenized models, not DeepMind's; a count by software is exact | a reader that finds the quoted string in any wording, 0 false reads |
| P8 candidate solutions | Step-level verification work treats squaring and similar steps as candidate-generating and requires each candidate checked against the original equation (arXiv 2608.28725); self-verification by the model is unreliable (arXiv 2512.02304) | checking candidates is the right structure, and software should do it | the arccot branch has two conventions, so a put is not true under every reading: stays a Sandbox check by the model |

Sources: arXiv [2304.02015](https://arxiv.org/abs/2304.02015), [2603.02222](https://arxiv.org/abs/2603.02222), [2412.05967](https://arxiv.org/abs/2412.05967),
[2406.19228](https://arxiv.org/abs/2406.19228), [2604.06613](https://arxiv.org/abs/2604.06613), [2601.05693](https://arxiv.org/abs/2601.05693),
[2510.10103](https://arxiv.org/abs/2510.10103), [2509.26522](https://arxiv.org/abs/2509.26522), [2505.14172](https://arxiv.org/abs/2505.14172),
[2412.18626](https://arxiv.org/abs/2412.18626), [2608.28725](https://arxiv.org/abs/2608.28725), [2512.02304](https://arxiv.org/abs/2512.02304).

## Built for the next live round (2026-11-11)

| Fix | Flag / file | Offline gate | Result |
|---|---|---|---|
| P11 | `--approx-put`, `tools/igc/approx_put.py`, `mathexpr.wide()` | `results/exact-answer-r9/approx_gate.py`, every stored `\approx` | round 9: 3,242 read; agrees with the model 2,648, model slip 336, estimate 39, scientific 34, an expression next 167. A100 bf16 (not used to build): 2,037 with a number next, 1,916 agree, 53 slips, 144 an expression next. 34 possible cut pieces read by hand: 0 misreads. After the put one token may not be `\times`, `\cdot`, `*` or a scale word (74 scientific values and 1 `million` would have read `4228161 \times 10^{6}`) |
| P12 | `--answer-rewind N`, `Rollout.rewind_answer`, `loop_rewind` | round 9 reach | 8 answers (5 questions): 1 the close hurt, 5 wrong in both arms, 2 the close helped. The live round decides; the 2 are watched |
| P7 | `--letter-count`, `tools/igc/letters.py` | `results/exact-answer-r9/letters_heldout.py` | 8 string wordings 8/8, 6 non-string letter questions 0/6; MATH-500 1 read (ELLIPSE, right), GSM8K 0 of 1,319; stored rows 10 (raw-string draws). The first rule (any word after `in`/`from`) read 6 prose words; fixed by requiring the phrase to end at the string |
| all | `tools/test_igc_round10_fixes.py`, `tools/test_igc_tir4b.py` `Round10Fixes`; `sim_replay` | 891 tests pass; replay of r9b-gen and r9a-ctrl with the new flags runs, `vllm_refused` 0 |

**Added 2026-11-11: the Sandbox on the PC** (the developer: the Sandbox has to be tested on the PC before the A100 anyway). Arm C = arm B
+ `--execute --box --loop-hint --verify-gate`, on the round's 23 MATH rows (12 + 11 per card, `--live 1`): P8 by `--verify-gate` (new,
`tir4b.GATE_OPEN`, the engine's `--igc-sandbox-gate` ported; reach `results/exact-answer-r9/gate_reach.py`: MATH-500 11, GSM8K 0), P9 by
`--loop-hint` (built in #12). Blocks run in the AppContainer box (D1; smoke 2026-11-11: start 2.2 s, sympy runs, `D:/Github` refused).
P10 (the figure) is not a ceiling after all: the text gives r = 14/3 by testing layouts; `--loop-hint` in arm C reaches it.

## Changed 2026-11-11: `--exact-answer` withdrawn

`--exact-answer` wrote the answer for the model (ERRATA MR-35 to MR-37), so it is out of round 10. Its cases get mechanisms that leave
the answer to the model:

| Case | Replacement | Support |
|---|---|---|
| P1 right value, wrong form (6) | none yet: the model chooses the form. **The developer decides**: the prompt asks for the exact value in every arm, or "value right, form not gold's" is reported apart | rule scorers' form strictness changes scores widely (arXiv 2604.22597; Math-Verify) |
| P2, P3 a hedge or a loop's value boxed (6) | `--answer-rewind 2`, now for every question | detection-extraction gap (arXiv 2604.06613) |
| P4 a mis-copied question (1) | `--question-quote` (`tools/igc/question_quote.py`): the model's copy holds the question's numbers in order but reads to another value -> the question's words, verbatim | RE2 (arXiv 2309.06275, EMNLP 2024); number-copy errors (GSM-Ranges, arXiv 2502.08680). Stored text: 15 fires in 442 answers, all real mis-copies |
| P5 a surd loop on brackets (2) | `--question-quote` at the thinking loop's close | settle and close at once, 32 vs 31 of 49 |

Round 10's arm B is now `--question-quote --approx-put --answer-rewind 2 --letter-count`; arm C adds the Sandbox.

## Coverage of round 9's wrong General answers (2026-11-11, after the form rule)

25 wrong by the old rule -> 3 counted right with a form note (1395 x2, 5070#1) -> 22 wrong. Every one is reached by a mechanism of round 10,
checked against its stored text (the trigger reads it, or the answer has the loop / close the mechanism acts on):

| Answers | Mechanism | Arm |
|---|---|---|
| div_big 5070#0 | `--approx-put` (its own wrong `\approx` digits) + `--answer-rewind` | B, C |
| div_big 4138#1, 9991 x2, 2783#0; mul_div 5942#1, 9991#1; mixed 9951#0; stdmath fbf10ed4#1 | `--answer-rewind` | B, C |
| div_big 4138#0 | `--question-quote` at its thinking loop's close; its cause is the prompt's "single number" (kept, the developer's choice) | B, C |
| mul_div 1505#0 | `--question-quote` (its copy moved a bracket; the trigger fires on the stored line) | B, C |
| surd 3480 x2 | `--question-quote` at the loop's close + `--answer-rewind` | B, C |
| swr 3480#0, 9951#1 | `--letter-count` (both strings read) | B, C |
| stdmath 45fa3ef2#0 | `--approx-put` (`(1.01)^{30}` read) | B, C |
| stdmath e3263989 x2 | `--verify-gate` | C |
| stdmath 97b18f58 x2 | `--loop-hint` (+ `--answer-rewind` in B) | C |
| stdmath 370853d3 x2 | `--loop-hint` (layouts testable; first called a ceiling, wrongly) | C |

Reached is not fixed: each mechanism that leaves the answer to the model works only if the model uses it, which round 10 measures.
Why two were first called uncovered: they were labelled once (4138#0 "a deliberate choice", 370853d3 "a ceiling") and the label was
carried on without checking which later mechanism reaches them, nor trying to solve the ceiling question from its text.

## Prompt line: the DeepMind rows' prompt line (decided 2026-11-11)

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

## Round 10 as it runs (2026-11-11)

**What it is.** A development run, not a measurement: the rows include the ones the fixes were built from, so a gain here says the
mechanisms work as built and the model uses them, and nothing about a fresh set. The measurement is frozen10 (both arms, PC layer and
A100). Build: **4B NVFP4 W4A8 · llama.cpp · PC layer**, `build-w4a8` (sm_89, `c9f3203a`), `qwen35-4b-nvfp4-axion.gguf`,
`-c 262144 -kvu -fa on -ctk q8_0 -ctv q8_0 --no-cache-idle-slots`, T 0.6, top-p 0.95, top-k 20, seeds 1-2, full context.

| | Rows (`mip-work/a100`) | Answers |
|---|---|---|
| A: round 9's General flags | `r10a-rows` 37 (5060 Ti) + `r10b-rows` 36 (4070 SUPER): DeepMind `div_big`, `mixed_longer`, `mul_div_longer`, `simplify_surd`, `swr_p_sequence` 10 each; MATH-500 L1-L5 23 (L5 10) | 146 |
| B: A + `--question-quote --approx-put --answer-rewind 2 --letter-count` | the same | 146 |
| C: B + `--execute --box --loop-hint --verify-gate` | `r10a-math-rows` 12 + `r10b-math-rows` 11 | 46 |

Flags: `--inline-calc --mathbackend --inline-plain --inline-style general` put software's value at `=` as a plain number;
`--no-digit-join` keeps the model from gluing digits to a put; `--guard-progress` puts nothing on a line already looping;
`--loop-close` closes a looping thinking with `</think>`; `--answer-close` cuts a looping answer and asks for the final answer;
`--early-cycle` catches an exact cycle early; `--shape-cap 1024 --cover-cap 32768` bound the loop-shape detectors; `--question-quote`
puts the question's own expression when the model's copy of it reads to another value, and at a thinking loop's close;
`--approx-put` puts the value to 10 significant digits after `\approx`; `--answer-rewind 2` cuts a looping answer out and lets the
model write on with another seed, at most twice, before the forced cue; `--letter-count` puts the letter counts of a string the
question writes out; `--execute --box` runs the model's code blocks in the AppContainer box; `--loop-hint` invites a program at a
thinking loop; `--verify-gate` opens a block to check every candidate against the original equation, once, when the question asks
for all solutions.

**What counts (proposed 2026-11-11, for the developer to set before the run).** The fixes go to frozen10 when, on the 146 pairs A vs B: (1) B is right more often than A,
one-sided sign test p < 0.05 on the rule score and on the API-corrected score both; (2) no value put by B's mechanisms is wrong
(each checked again by `analyze.py`); (3) B loses no answer A got right that the touch caused (a pair that parts before the first
touch is path noise; one that parts at a touch and goes right -> wrong is read by hand and must be understood). Arm C reaches 3
distinct questions (`e3263989`, `97b18f58`, `370853d3`, 6 answers) inside its 46: it is a working check of the Sandbox on the PC,
read answer by answer, not a score.

**The simulation gate** (`results/round10/replay_gate.py`, round 9's 240 General traces replayed under arm B, each mechanism alone
and all four together; the replay stops at the first new act):

| Mechanism | First acts (alone) | On answers that finished right / wrong | Values wrong | `vllm_refused` |
|---|---|---|---|---|
| `--question-quote` | 42 | 31 / 11 | 0 (28 quotes evaluated to gold; 14 not checkable: a surd quote or a gold that is not a number) | 0 |
| `--approx-put` | 138 | 119 / 19 | 0 (each re-evaluated exactly, within half a unit of its last place) | 0 |
| `--letter-count` | 10 | 8 / 2 | 0 (counted again with `str.count`) | 0 |
| `--answer-rewind 2` | 0: not measurable by replay | 28 answers had an answer close in round 9: 10 / 18 | puts no value | 0 |

The rewind cannot be replayed: when round 9's answer close acted it cut the loop out of the text (`cut_tokens` 400-5,600), so the stored
traces hold no loop for `loop_rewind` to find (0 of the 28 at the close's position). Its reach is the 28 answers whose answer close
acted, 10 of them right, and whether the rewind acts there in place of the close is a question only the live run answers; the 10 right
ones are read one by one in round 10's paired split.

Acts on right answers put true text (the question's words, a value to the places it gives, a count) and decide nothing; whether they
cost a right answer is what the live run's paired split (3) answers.

**Speed: llama-server's idle-slot cache, found in the smoke.** The smoke on the 5060 Ti fell from ~60 to 2-4 tok/s per slot (6-9 tok/s
total), GPU at 3-4 %, the server on one CPU core. Cause, read in `tools/server/server-context.cpp` (llama.cpp `1deefcca3`):
`--cache-idle-slots` is on by default, and under `-kvu` every new task saves each idle slot to the RAM prompt cache and clears it
(`TAG_IDLE_SLOT_CLEAR`). tir4b sends a request every ~18 tokens, so an answer's slot is often idle when another answer's request
arrives; its next request finds its cache gone, takes a slot by LRU (2,068 of 2,293 selections) and reloads its state over PCIe,
2 s median and growing with the context, with the server's main loop blocked meanwhile (4,250 s of an 88-minute run). Five answers
in flight on `-np 4` made it worse. Fixed: `--no-cache-idle-slots` on every PC launcher, and `pc_round10.sh` starts `N/2*2 + 1`
slots so every answer in flight keeps its own (slots cost no VRAM under `-kvu`). The same smoke resumed on a fresh server with the
flag and four answers on four slots: 107 tok/s total over its 16 minutes (153 in the first 3), per slot 30-45 tok/s with no fall as the
context grew to 30k, 3,523 of 3,527 selections by LCP, GPU 63-91 %. Both changed at once (fresh server, four in flight, the flag); the
config round 10 uses (`pc_round10.sh`: five slots, five in flight) has not run yet: a 5-minute smoke through the launcher comes first. Round 9 had it milder, with four answers on four slots: LRU on
7 % of selections in r9b, 17 % in r9a2 (whose per-slot speed fell from 12 to 4 tok/s); its timing is not comparable to round 10's.
The fixed sweep (`docs/benchmarks.md`, speed of the PC layer): 5060 Ti np 4 / 6 / 8 = 124 / 144 / 154 tok/s total, 37 / 29 / 23 per slot;
4070 SUPER np 4 / 6 = 132 / 159, 39 / 31. Server flags beyond these (`-cram 0`, `--prio 2`, `--backend-sampling`, MTP and n-gram
speculative decoding) are under test (`scripts/pc/flag_test.sh`, each speculative config gated by `scripts/pc/spec_probe.py`).
With `-c 524288 -np 5` the 5060 Ti uses 14.8 of 16.3 GB (measured); the 4070 SUPER cannot, so it would split the config: the developer's
call. The KV pool has not overflowed so far: no `purging slot` and no failed decode in any log of round 9 or the smoke.

**Expected time:** about 5-7 h with both cards, from round 9's tokens on these rows (~1.7M per arm on the DeepMind and repeated MATH
rows, ~0.5M for the 17 new MATH rows, C ~0.7M) at the fixed smoke's ~107 tok/s per card, arm C one answer at a time on each card beside A and B; the
longest answers (60-70k tokens in round 9) set the tail.

**The smoke** (`a100/pc/r10smoke-*`, `r10smoke4070-c`): A and B answered `dm_swr_p_sequence-3480` with no error, `--letter-count` put
the counts, `--approx-put` put 3 values (all right). Arm C on the 4070 SUPER answered `stdmath-e3263989` twice: the gate opened
(`verify gate`), the box ran the checks (`output` by the gate, 1.1-3.0 s), both answers boxed four solutions and are wrong (gold
`3\pm2\sqrt{2}`). In seed 0 the gate's own check printed `sin = -0.333` for the two negative candidates, and the model still kept them;
in seed 1 the model's check used an arccot branch under which the negatives pass. Seed 0 also put two code fragments it quoted from its own earlier code in ```` ```python ```` fences ("Let's check `cot_inv`
implementation in code."), which the harness ran as blocks: `IndentationError`, true output, not a cut by the harness. This is round 9's P8 as written: the branch convention is the model's to decide, and the output
showed it the answer it did not take.
