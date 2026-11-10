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

1. **Round 9's last 8 pairs (a3)**, on round 9's code (`c093b2f`, worktree), the 4 moved MATH-500 L5 rows x 2 seeds, both arms on the
   5060 Ti, `-np 4`. Runs while steps 2-4 are written (GPU vs CPU).
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
