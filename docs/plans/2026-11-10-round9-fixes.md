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
