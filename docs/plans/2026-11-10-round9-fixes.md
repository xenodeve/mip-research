# Round 9's failures and the fixes for them (2026-11-10)

**Build:** 4B NVFP4 W4A8 · llama.cpp · PC layer, round 9 (half A `c093b2f`, half B `255ec4e`), General arm, 232 of 240 pairs.
Every wrong answer of the General arm was read by hand: 23 wrong by both the rule and the API reader, and 6 more where the two
readers disagree. The control arm's 75 wrong answers are almost all loops that ran out (`stuck`), which the closers already answer.

## The failures, by kind

| # | Kind | Answers | n | Fix |
|---|---|---|---|---|
| 1 | Right value, wrong form: the exact fraction written, then a decimal or the integer part boxed; or the old cue cut the answer | `dm_div_big` 1395×2, 2783#0, 4138×2, 5070×2, 9991×2; `dm_mixed_longer-9951#0`; `dm_mul_div_longer-9991#1`; (9977#0, see 8) | 12 | `--exact-answer` (built, `9beabf0`): replay 12 wrong -> right, 0 right -> wrong |
| 2 | The model read the question's precedence wrong (-12.375 for -198); `question_read` reads -198, but the box rule only acts on a shortening of a non-integer value | `dm_mul_div_longer-1505#0` | 1 | **`--exact-answer`, widened:** when `question_read` reads the whole question as one computation, the last box holds the exact value whatever the model wrote, integer values included (the developer: when the question is the computation, its value is the answer) |
| 3 | `simplify_surd` loops on parenthesis counting; `question_read` reads none of the 10 surd questions (`sqrt`, `**`) | `dm_simplify_surd` 3480×2, 9937#0 | 3 | **`question_read`, widened:** read `Simplify <expression with square roots>` and compute it exactly as `a + b*sqrt(n)`, so the module falls under `--exact-answer` |
| 4 | No put after `\approx`: `(1.01)^{30} \approx 1.3448888` (true 1.3478489); both arms wrong | `stdmath-45fa3ef2#0` | 1 | **`--inline-calc`, widened:** after `expr \approx` where `expr` is computable, put the value rounded correctly to the places the model goes on to show (true under every reading). Replay gate first: 0 false puts; how often the model's own `\approx` value is wrong |
| 5 | The answer close boxed an unfinished step: the model looped doubting a right `72 - 14 = 58` and the cue took 58 | `stdmath-fbf10ed4#1` | 1 | Not this step. Settle before close (decide layer); the settle probes have not shown a gain yet |
| 6 | Letters of the question's string miscounted | `dm_swr_p_sequence` 3480#0, 9951#1 | 2 | Not this step. A reader for this sentence shape is tuning to DeepMind; the general route is the Sandbox counting |
| 7 | Reasoning, not computation: arccot convention, a figure the model cannot see, a hard maximum guessed | `stdmath` e3263989×2, 370853d3×2, 97b18f58×2 | 6 | Not this step: the hypothesis loop / oracle (A100) |
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
| 1. a3 (done 2026-11-10) | done on the 4070 | General 8 of 8, control 4 of 8 (4 `stuck`). Round 9, 240 pairs: **215 vs 161** by the rule (API **216 vs 161**), General alone 59, control alone 5, p = 9e-13 |
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
2. Whatever of P11 (`pprox` put) and P12 (close on a step of the asked quantity) passes its offline gate by then, each designed from the case's statement ([failures](../failures.md#how-a-pattern-is-fixed-the-situation-not-the-test-set)).
3. P7 and P8 stay out: their evidence is one module / one question.

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
