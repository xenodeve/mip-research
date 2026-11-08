# Anti-loop: loop detectors and closers

A small reasoning model at full context can run the same text until the context ends and write no answer. This category is every
check that decides "the text is looping" and every action taken then: close the thinking, close the answer, cut back and nudge, or
stop. It covers the `tir4b` harness (vLLM / `llama-server`, Qwen3.5-4B) and the Strata engine's own checks on Strata-xeno branch
`xeno/mip-sandbox` (run on this PC). The thresholds of the tir4b loop checks (line-back, counting guard, V7, shape cap, cover cap)
were taken from stored traces so that no stored answer that finished right crosses them; the engine's own checks have their scans
quoted beside each rule.

## Summary

| Mechanism | What it does (one line) | Runs on | In best config? | Best measured result | Status |
|---|---|---|---|---|---|
| Repeat check + stuck stop | Exact cycle of 20 rounds (8 past 512 tokens) ends the answer `stuck`, or hands it to a closer | tir4b (vLLM/llama-server) | yes (always on) | 167 of 197 A100 guard stops; the 6 widest read, all real loops | in best config |
| Counting check + `--guard-progress` | Digit-masked cycle of 64 rounds; with the guard, rounds that are at least 7/8 new go on | tir4b (vLLM/llama-server) | yes | Guard changes exactly the 3 rows that were still working; every other stop kept | in best config |
| Line-back check | One line of 12+ tokens seen 16 times | tir4b; also Strata engine | yes | 23 A100 stops, 12 widest read: none cut early | in best config |
| A: loop close (`--loop-close`) | Puts `</think>` on a thinking loop so the model answers | tir4b (vLLM) | yes | A100: 25 -> 37 of 48 (+13 / -1) on loop rows | in best config |
| B: answer close (`--answer-close`) | Answer loop cut to one round, then `The final answer is $\boxed{` and one value | tir4b (vLLM) | yes | A100: cut answers 3 -> 1; replay 0 acts on 370 right | in best config |
| V7: early cycle (`--early-cycle`) | Exact cycle of 6 rounds (3 past 512) spanning 500+ tokens, at every line end | tir4b (vLLM) | yes | Replay: 0 of 915 right touched, 212 of 243 loops sooner, 624k tokens | in best config |
| Shape cap (`--shape-cap 1024`) | Exact cycle with each digit run read as one token, 1,024 rounds | tir4b (vLLM) | yes | A100: closed the long division at 55k model tokens | in best config |
| Cover cap (`--cover-cap 32768`) | 97 % of the last 32,768 tokens repeat a 12-gram of the 2,048 before | tir4b (vLLM) | yes | Replay: 0 of 1,078 right touched, 12 of 13 runs past 150k stopped | in best config |
| Echo guard | 3rd value in a row for one expression, each within 64 tokens, left to the model | tir4b (vLLM) | yes (part of `--inline-calc`) | Stored traces: an echo in 7 of 48 answers; A100 loop-lcacv7: 55 skips in 6 answers | in best config |
| D: loop value (`--loop-value`) | Puts the question's own exact computation before a close | tir4b (vLLM) | no | A100: value right 12 of 20 times, adopted on none of the 2 division rows | built, measured, kept as option |
| Loop nudge (`--candidate-close`, no candidate) | Cuts an exact cycle to one round and writes `Wait, let me rethink...` | tir4b (vLLM) | no | Replay: acts on 30 cut and 18 right answers | built, measured, dropped from every arm |
| CandidateAnswerTracker + candidate close | Closes on a final answer confirmed by a program's output (needs `--execute`) | tir4b (vLLM) | no | Replay: 0 closes, 0 regressions (no program in the traces) | built, not measured yet |
| Engine exact-cycle close (`in_cycle`, period to 4,096) | Thinking tail is one exact cycle: close | Strata engine (xeno/mip-sandbox, not in main) | no | Threshold scan of 12,164 stored runs | engine only |
| Engine counting / line / value-count closes | Masked 64-round cycle, line back 16 times, 256 values: close | Strata engine (xeno/mip-sandbox, not in main) | no | Threshold scans of 7,444 / 7,437 / 2,471 stored runs | engine only |
| Engine third-repeat close (`--igc-close-repeat 3`) | The 3rd sight of one expression closes the thinking | Strata engine (xeno/mip-sandbox, not in main) | no | This PC, 49 Qs: removing it +43.7 % tokens, 40 vs 37 right | engine only |
| Engine repeat note (`--igc-repeat-note`) | 2nd sight of an expression gets ` (already confirmed, stop checking)` | Strata engine (xeno/mip-sandbox, not in main) | no | Wording variants built, not compared | engine only |
| Engine repeated-program close | The same sandbox program N times closes the thinking | Strata engine (xeno/mip-sandbox, not in main) | no | none read | engine only |
| Sandbox close after a print (`--igc-sandbox-close`) | Thinking closed once a program printed a result | Strata engine (xeno/mip-sandbox, not in main) | no | This PC, 49 math Qs: 0.508x tokens, 77 vs 73 of 98 right | engine only |

The best config is `--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress --loop-close
--answer-close --early-cycle --shape-cap 1024 --cover-cap 32768` (README). The A100 loop rounds below are Qwen3.5-4B bf16 on vLLM 0.33,
Colab A100 80 GB, T 0.6, top-p 0.95, top-k 20, full 262,144-token context, on the 24 loop rows x 2 samples (rows where some rerun arm
ended `stuck`), scored by the rule scorer with the API reader beside it.

### The loop rows on the A100 (24 rows x 2 = 48 answers)

| Arm | Closers | Right | No answer (`stuck` / `length`) | Tokens |
|---|---|---|---|---|
| Plain model (`fresh-ctrl`, another run) | none | 24 | 21 | not compared |
| Inline values only (`loop-gen`) | none | 25 | 23 | 754k |
| `loop-lc` | A | 37 | 3 | 724k |
| `loop-lcac` | A + B | 37 | 1 | 861k |
| `loop-lcacv7` | A + B + V7 | **37** | **1** | 937k |

Read from `records.json` of each arm in the results backup (`a100/out/<arm>`), plain model restricted to the same 48 (id, sample)
pairs. `loop-gen`, `loop-lc` and `loop-lcac` ran in one round; `loop-lcacv7` is round 7 with the same rows and seeds. [^pairing] [^tokens]
Shape cap and cover cap are not in these 48: shape cap ran on 3 rows (round 8), cover cap has only the replay gate. [^round9]

## Repeat check and the stuck stop

- **Trigger:** every request of the thinking ends after at most `CHUNK` = 1,024 tokens, and every answer request, then
  `looping_now()` reads the thinking since the last nudge (or the answer alone) ([`../tools/igc/tir4b.py`](../tools/igc/tir4b.py),
  `looping_now`, line 1119).
- **Check:** `repeating()` ([`../tools/igc/external_score.py`](../tools/igc/external_score.py):49): the tail is one exact cycle of
  period up to 4,096 tokens, 20 rounds for a period up to 512, 8 rounds past it.
- **Action:** `on_loop()` (tir4b.py:1164): a closer if one is on (A in the thinking, B in the answer), else `done('stuck')`: the answer
  ends with no answer. This is the same guard in every arm, MIP or not, so a loop cannot run to the end of the context alone.
- **Where the threshold came from:** the engine's own rule, "past 512 tokens eight rounds are enough, as in the engine: a longer cycle
  once ran to the end of the context, 31 minutes, because this stopped looking at 512" (external_score.py docstring).
- **Measured:** all 197 guard stops in the 8 A100 arms were classified by distinct lines among the last 64. Exact cycle: 167 stops,
  134 at 16 of 64 or fewer; all 6 at 40 of 64 or more were read: 4 a repeating decimal on one line (`-5511170.918918...`), 2 cycles of
  847 and 1,217 tokens, all real loops (mip-research#19, loop round, simulation first). [^stuckrounds]

## Counting check and `--guard-progress`

- **Check:** `counting()` (external_score.py:62): with digit tokens masked, an exact cycle of period up to 128 tokens and 64 rounds.
  Hand counting changes its numbers, so the exact cycle check misses it.
- **`--guard-progress`:** when the masked cycle's rounds, unmasked, are at least 7/8 distinct, it is work going on (trial division by
  one prime after another, a search over x) and is let go. A cycle that comes back to the same rounds is still stopped.
- **Where the threshold came from:** 3 of the 38 rows the A100 General arm ended `stuck` were stopped by `counting` with all 64 of
  their last lines different (`div_big-2208`, `div_big-7765` dividing past the square root; `surd-1033` searching over x)
  (mip-research#19, "The 38 stuck rows").
- **Replay result** (`which_guard.py` over every stored arm): it changes exactly those 3 General-arm rows; in all 9 older arms every
  stuck stop was an exact cycle or a line-back and stays stopped; the check can only turn a stop into no stop (mip-research#19, "Two
  fixes from the General arm").
- **A100:** every loop-round arm ran with it; no arm without it on the same round. [^progress]

## Line-back check

- **Check:** `line_back()` (external_score.py:82): one line of at least 12 tokens seen 16 times over the window. Read at a line end
  (token 198 or 271) or every 4,096 tokens.
- **Where the threshold came from:** the engine's rule (Strata-xeno xeno/mip-sandbox: include/strata/core/igc_inline.hpp:121-124): of
  7,437 stored runs, a 12-token line came back 16 times in 3 that ended by themselves (one right, 2,535 tokens from its end) and in all
  4 that ran to the end of the context, at 11,202 to 128,741 tokens.
- **Measured on the A100 stops:** 23 line-back stops; the 12 at 40 of 64 distinct lines or more were all read: 10 circle with small
  rewording (spelling the prompt's letters 13-25 times, `So ∠AFD = 80` 17 times), 2 the endless division of 6681/10943. None was cut
  early (mip-research#19). [^lineearly]

## A: loop close (`--loop-close`)

- **What it does:** on a loop in the thinking, `LOOP_CLOSE` = `\n</think>\n\n` is put (tir4b.py:28) and the phase becomes the answer,
  so the model writes its answer from the work it has done.
- **Replay gate** (`sim_replay --recorded` of the General and Qwen rerun arms, 104 answers each): acts on 28 of 30 cut answers
  (General) and 22 of 23 (Qwen), on 0 of 73 and 0 of 78 right answers. It acts exactly where the guard ended the answer `stuck`.
  On 538 answers of 7 arms: 112 cut acted on, 0 of 370 right (mip-research#19).
- **A100** (loop round, same round): 37 of 48 against 25 for inline values only, +13 / -1; cut answers 23 -> 3. In `loop-lcacv7`
  A closed 21 answers and 13 of them ended right (records). Tokens per right answer 30,180 -> 19,570 (-35 %) (mip-research#19, loop
  round, section 4). [^aloss]
- **By loop shape** (all loop points of loop-gen and loop-lc, read): re-reading the prompt 9 closed, 8 right after; answer known but
  cannot stop 5, 4 right. [^doubt]

## B: answer close (`--answer-close`)

- **What it does:** a loop in the answer after the thinking is closed (tir4b.py:1215 `close_answer`): an exact cycle is cut back to
  one round (`loop_start`), then `\n\nThe final answer is $\boxed{` is put and the model writes one value, at most
  `ANSWER_CLOSE_TOKENS` = 64 tokens, to a line end. Nothing else acts after the cue.
- **Why:** loop-lc closed 15 thinking loops; 12 answered right; the 3 others carried the same doubt into the answer and looped there.
- **Replay gate** (7 arms, 538 answers, current code): acts on 5 of 120 cut answers, 0 of 370 right, 0 of 15 finished wrong;
  `vllm_refused` 0 in every arm (mip-research#19, "`--answer-close`").
- **A100:** `loop-lcac` 37 of 48, cut answers 3 -> 1 against loop-lc. It turns `stuck` into a finished answer, not into a right one.
  [^bright]

## V7: early cycle (`--early-cycle`)

- **What it does:** at every line end of the thinking (tir4b.py:1138 `early_cut`), `early_cycle()` (tir4b.py:1565) asks whether the
  tail is one exact cycle of `EARLY_ROUNDS` = 6 rounds (`EARLY_ROUNDS_LONG` = 3 past 512 tokens) that spans `EARLY_SPAN` = 500 tokens
  or more. On a hit the tokens after that line end are dropped and `on_loop()` runs (A closes). The span keeps short echoes out.
- **Where the threshold came from:** detector variants replayed over the 10 stored A100 arms with no closer (915 right answers, 243
  cut in the thinking). 6 rounds alone fired on 4 right answers; adding the 500-token span gave V7 (mip-research#19, section 6):

  | Variant | Fires on right (of 915) | On finished wrong | Caught before the run's stop | Tokens earlier |
  |---|---|---|---|---|
  | The runner's checks (modelled) | 0 | 0 | 119 / 243 | 403k |
  | **V7** | **0** | **0** | **212 / 243** | **624k** |

- **Replay gate** before the A100 (`sim/cmb-abv7-*`): 0 acts on right answers in 6 of 7 arms; in loop-lc 14 acts, each the same loop the
  run closed later; `vllm_refused` 0 (mip-research#19 round 7; RESUME.md).
- **A100** (round 7, `loop-lcacv7`): 37 of 48, the same as A + B; V7 fired on 17 of 48 answers; no answer ended `stuck`. [^v7]

## Shape cap (`--shape-cap 1024`)

- **What it does:** `shape_rounds()` (tir4b.py:1495) collapses each run of digit tokens to one token, then counts the rounds of the
  longest exact cycle (period up to `SHAPE_PERIOD` = 128) ending at the tail, so `(Digit 5243: 7)` and `(Digit 5244: 2)` are one round,
  even when the numbers change length. At the cap the thinking (A) or answer (B) is closed, progress or not.
- **Where the threshold came from** (`shape_sim.txt`, stored traces of the A100 arms): the longest run in an answer that finished
  right is 805 rounds (`v2t-qwen dm_mixed_longer-9325`, then 540); the four that ran to the context end reached 1,874-4,267. Cap 1,024.
- **Replay gate** (`sim/cmb-abvs-*`): 0 acts on right answers outside arms that already close loops; `vllm_refused` 0 (RESUME.md).
  [^shapegate]
- **A100** (round 8, A + B + V7 + shape cap, 3 loop rows x 4): 8 of 12 right (div_big-2208 1/4, simplify_surd-1931 4/4,
  conversion-4179 3/4). Its one act: div_big-2208 #0, long division digit by digit, closed at 55k model tokens (shape rounds
  1,019 -> 1,024); no act on the other two rows. div_big-2208 had 0 right in 28 earlier answers across 13 arms (mip-research#19
  round 8). [^shape]

## Cover cap (`--cover-cap 32768`)

- **What it does:** `Cover` (tir4b.py:1523) is fed as the text grows. Digit runs read as one token; for each position, whether its
  12-gram (`COVER_GRAM`) was seen in the last 2,048 (`COVER_BACK`); the share of such positions over the last W = 32,768 is the score,
  0.97 (`COVER_SHARE`) or more is a loop. One new gram moves the share by 1/W, so one odd round does not break it.
- **Why:** a long division where every quotient digit 0 makes a shorter round breaks the exact cycle that shape cap needs (round 8
  div_big #1: shape rounds never past 46, 196k model tokens to the context end).
- **Where the threshold came from** (`share_sim.py 12 2048 32768`, stored traces of 16 A100 arms): right answers reach 0.92 at most
  (two that looped long and then came right), the next 0.35; all 10 answers past 150k tokens in a loop 0.999-1.0.
- **Replay gate** (`sim_gate.py cov2 cover_cap=32768`, every arm with its own flags): 0 acts on 1,078 right answers; `vllm_refused` 0;
  answers past 150k tokens stopped 12 of 13, at 40k-114k (loop-shape div_big #1 at 64.5k instead of the context end; loop-lcacv7
  div_big #0 at 61.5k instead of the run's own close at 145k) (mip-research#19, "`--cover-cap`"). [^cover]

## Echo guard

- **What it does:** with `--inline-calc`, `echoing()` (tir4b.py:1107): when software's value for one expression would be the 3rd
  (`ECHO_RUN`) back to back, each within `ECHO_GAP` = 64 model tokens of the last, the value is not put and the model writes it.
  It never calls `on_loop()`: whether the thinking loops is for the loop checks. Counted in the record as `echo_skips`.
- **Where it came from:** a count over the whole answer is no loop: on the 48 stored `inlineclose` traces an echo occurs in 7 of 48
  answers; `livebench-652451e2` has 100 on `-1/2`; the 4 right answers among the 7 have 1-6 each (mip-research#13, fixed in
  `4dc5e546`).
- **A100 (counted from records):** `loop-lcacv7` 6 answers with an echo skip (5 right), 55 skips; `fresh-gen` 27 answers, 188 skips.
  [^echo]

## D: loop value (`--loop-value`)

- **What it does:** before a loop is closed (A or B), when `tools.igc.question_read` reads a computation from the question, puts
  `Computed exactly: $<expr> = <value>$` on its own line, once (tir4b.py:1199).
- **Replay:** acts on 31 cut and 2 finished-wrong answers, on 1 right answer (gold put); the value is gold in all 34
  (mip-research#19, section 3).
- **A100** (round 5, A + B + D, 5 rows x 4): the value was put in 12 of 20 answers, always right; the model did not adopt it on the
  two division rows. Off by default. **To use:** add `--loop-value`. [^d]

## Loop nudge (`--candidate-close` with no live candidate)

- **What it does:** in `on_loop()` with a tracker and no live candidate, at most `LOOP_NUDGES` = 3 times: an exact cycle
  (`loop_start`, at least 4 rounds) is cut back to one round and `\nWait, let me rethink this derivation carefully.\n` is put; the
  thinking's checks then read only from the nudge on (`loop_floor`). Any other loop, or one past the nudges, is closed by A.
- **Replay** (538 answers of 7 arms): acts on 30 cut answers and on **18 right** ones: checking at every line end fires on transient
  repetition. Dropped from every arm. **Where:** tir4b.py:1179-1192; **to use:** `--loop-close --candidate-close` without `--execute`.
  [^nudge]

## CandidateAnswerTracker and the candidate close

- **What it does:** `CandidateAnswerTracker` ([`../tools/igc/unified_detect.py`](../tools/igc/unified_detect.py):576) records final
  answers the model writes (`\boxed{v}`, `the final answer is v`) after `CANDIDATE_MIN_TOKENS` = 1,500 tokens. A written candidate is
  always `METHOD_ALGEBRAIC`; a program's printed value is a second method and counts only when it equals a final answer already
  written. At a line end that starts a re-check (`Wait`, `Hmm`, `Let me double-check`), `CROSS_METHOD_CONFIRMED` closes the thinking
  (tir4b.py:1354 `check_candidate_line`).
- **So it needs `--execute`:** with no program run nothing can confirm a candidate.
- **Replay:** stored 262k arms `inlineclose` and `ctrl` with the live rule: 0 closes, 0 regressions (no program ran in them)
  (mip-research#15, fixed in `4dc5e546`); `--candidate-close --execute` on the A100 rerun arms: 0 acts on cut, 0 on right
  (mip-research#19). **Not measured live.** **To use:** `--loop-close --candidate-close --execute` on an arm whose model writes
  programs. [^cand]

## Strata engine: exact-cycle, counting, line and value-count closes

All in `IgcLearnedDiagnostic` (Strata-xeno xeno/mip-sandbox: include/strata/program/igc_learned_diagnostic.hpp:424-441, 646-650),
checked every 16 thinking tokens over a tail of at most 16,384 tokens, with the rules in
Strata-xeno xeno/mip-sandbox: include/strata/core/igc_inline.hpp:107-195. Each puts `</think>` and logs its kind.

| Close | Rule | Where the threshold came from (code comment) |
|---|---|---|
| `loop close` | `in_cycle(tail, 4096, 8)`: exact cycle, period to 4,096; span `max(clamp(8 x period, 64, 1024), 3 x period)` | 1,410 runs: no text repeated six times ever left; 7,073 runs: none left a 300-char block after three rounds; 1,472 runs long enough: no 1,000-char block left after three rounds by itself (mip-research#6) |
| `counting close` | digits masked, period to 128, 64 rounds | 7,444 stored runs: fired on 5 that ended by themselves, all wrong, none right |
| `line repeat close` | a 12-token line back 16 times | 7,437 stored runs (see line-back above) |
| `too many values close` | 256 inline values in one thinking | 2,471 runs that took a value and ended by themselves: one took more than 256 and was wrong; a right one took 78 |

- **Why 4,096:** `HumanEval/39` in `tl-onhg2` ran to 261,906 tokens in one exact cycle of about 3,630 tokens (61 rounds, primes 41 to
  997); periods to 512 could not see it. Scan of all 12,164 stored runs (`hg_left.py`): 42 stretches of a 1,000+ char block three times
  in a row, 39 run to the end of the text, the 3 that stop short are none the model leaving a loop (mip-research#6). **Status:** engine
  only, unit tests pass. **To use:** the Strata engine with `--igc-inline-exec`. [^incycle]

## Strata engine: third-repeat close (`--igc-close-repeat`)

- **What it does:** with `--igc-inline-exec`, each delivered value is keyed by its program; when one expression has been seen
  `close_at - 1` times before, `after_value()` returns `.` `\n` `</think>` instead of a note (Strata-xeno xeno/mip-sandbox:
  include/strata/core/igc_inline.hpp:201-202; Strata-xeno xeno/mip-sandbox: include/strata/program/igc_learned_diagnostic.hpp:630-645).
  Default 3, 0 never. **To use:** on by default with the engine's `--igc-inline-exec`; `--igc-close-repeat 0` turns it off.
- **Measured** (Strata engine on this PC, 49 questions, one run each; `onc0` close, `onc1` no close; Strata-xeno#276 comment
  "The third-repeat close stays") [^model]:

  | | Close | No close |
  |---|---|---|
  | Right, after reading | 40 | 37 |
  | Tokens | 117,495 | 168,882 |
  | Stuck in a loop | 0 | 1 |

  Removing it added 43.7 % tokens (more on 36 questions, fewer on 13); no question went from wrong to right when it
  was removed. The arms ran on different exes and parted before the close on most questions: totals only, unconfirmed until the
  same-exe test runs. **Status:** engine only, default on with `--igc-inline-exec`. [^close3]

## Strata engine: repeat note (`--igc-repeat-note`)

- **What it does:** the second sight of an expression gets ` (already confirmed, stop checking)` (mode 0, default), ` (already
  confirmed)` (1) or ` (already confirmed, next step)` (2) after the value (Strata-xeno xeno/mip-sandbox:
  include/strata/core/igc_inline.hpp:196-208). Meant to stop the re-check that
  becomes a loop.
- **Measured:** only as part of the whole note; the three wordings are built and not compared. **To use:** `--igc-repeat-note 1|2`.
  [^note]

## Strata engine: repeated-program close and sandbox close after a print

- **Repeated program:** with the sandbox, the same program seen `close_repeat` times closes the thinking ("a loop of whole blocks is
  far longer than in_cycle reaches, 2567 tokens on one question", Strata-xeno xeno/mip-sandbox:
  include/strata/program/igc_learned_diagnostic.hpp:464-469). At that commit it uses the same `--igc-close-repeat` value, so 0 turns it
  off. No measurement of it alone was found. [^progclose]
- **Close after a print** (`--igc-sandbox-close`, with `--igc-sandbox-read N` to let the model read the result N tokens first, and
  `--igc-sandbox-close-line`): once a program printed and no check failed, `</think>` is put (Strata-xeno xeno/mip-sandbox:
  include/strata/program/igc_learned_diagnostic.hpp:528-536). **To use:** `--igc-sandbox --igc-sandbox-after 512 --igc-sandbox-close`.
- **Measured** (Strata engine on this PC [^model], exe `b324679801c45fdc`, 49 questions: 29 GSM, 20 MATH; 2 rounds of 3 arms in rotation, same session;
  arm `--igc-sandbox --igc-sandbox-after 512 --igc-sandbox-close`; mip-research#3): tokens 125,608 against 247,162 plain (**0.508x**,
  lower on 41 questions, sign-flip p below 0.00001); right 77 against 73 of 98 (p 0.29, no measured gain: the close did not lower
  accuracy); stuck 0 against 2. **Status:** engine only, for math. [^sbclose]

## Notes

[^pairing]: vLLM on the A100 does not repeat an answer across runs (mip-research#21), so arms are compared as totals, and 1-2 answers
    is within run-to-run variation. The plain model is another run (`fresh-ctrl`), not part of the loop round. `loop-lcacv7` vs
    `loop-lcac`: answers changed both ways (2 gained, 2 lost).
[^tokens]: Totals are not a saving: the closers let more answers finish, and one answer that runs to the context end weighs 210-262k.
    In `loop-lcac` that answer is conversion-4179#1 (a counting loop `--guard-progress` let go); in `loop-lcacv7` div_big-2208#1 (digit
    division whose numbers change length). Neither is a loop V7 can see. Unpaired runs: no token claim.
[^round9]: The best config with every closer against a plain control, 120 fresh rows x 2 in one round (round 9), is ready but not run:
    the A100 was refused for lack of compute units (mip-research#19, RESUME.md).
[^stuckrounds]: In the Strata bench (`external_bench`) the stuck stop wants 20 rounds checked every 64 tokens, which can stop a cycle of
    up to 3 tokens before the engine's own close (64 tokens at least); `math-2ff342ec` ended `stuck` this way twice in the 49-question
    arms of mip-research#3. `--stuck-rounds 40` leaves such loops to the engine (mip-research#6).
[^progress]: Trade-off: a row that truly counts forever now runs to 262,144 tokens. On the A100 this happened once: `loop-lcac`
    conversion-4179#1 (`check ... base 5815. No.`) ran 262k because every round had a new number. Shape cap was built for this.
[^lineearly]: line-back cannot be made earlier: lines come back in right answers (variant V3 with line-back at 5 fired on 221 of 915
    right answers in the replay).
[^aloss]: The one loss, stdmath-0ada1f63#0, finished right in loop-gen and looped in loop-lc; the paths parted before the loop, so the
    close did not cut a right answer. Of A's 21 closes in loop-lcacv7, 8 ended wrong.
[^doubt]: A fixes none of the genuine doubt loops: re-checking a doubted value 3 closed, 0 right; endless division or repeating
    decimal 2 closed, 0 right (0 of 5). Word Salad Chopper (arXiv 2511.00536) reports the same for a forced answer after a loop.
[^bright]: B acted 7 times in loop-lcac and was right once (`\boxed{7}`); 3 div_big answers boxed a decimal instead of the fraction,
    3 boxed a value the model still doubted. In loop-lcacv7 B acted on 6 answers and none was right. B acts only on an answer already
    looping.
[^v7]: 2 answers right with inline values only were wrong with A + B + V7: MATH level-5 problems (stdmath-d93ff30b#0,
    stdmath-1adf1706#0) where V7 closed the thinking at 8,433 and 6,582 tokens, and the model had finished right at 42,974 and 31,518
    tokens without the closers (loop-gen). Paths are not paired, so this is not a measured cost; of the 17 answers V7 acted on, 6 ended
    wrong. Open: a longer V7 span for long MATH derivations.
[^shapegate]: The gate's wording is "0 acts on right answers outside arms that already close loops"; acts inside closing arms were not
    broken down in the notes read.
[^shape]: 3 rows x 4 samples, unpaired with earlier arms, so the 1 of 4 on div_big-2208 is not a measured gain. After the close the
    model went back to dividing in the answer and B forced `0.610502` (sign lost). div_big-2208 #1 escaped it: zero quotient digits
    break the exact cycle (shape rounds never past 46), which is why cover cap exists.
[^cover]: No A100 run with cover cap yet (round 9 blocked). The 13th long answer parts from its record at an inline value, so the replay
    cannot follow it. A first version that counted the run of repeated 12-grams failed: it
    restarted every ~20k tokens, max 19,714, below the longest right answer (22,668).
[^echo]: It replaced a rule that treated the 3rd use of one expression anywhere in the answer as a loop; that rule would have cut or
    nudged 41 of 48 stored answers and made the stored `inlineclose` 41/48 describe code that no longer exists. That 41/48 is
    withdrawn (mip-research#13).
[^d]: D gains nothing on doubt loops: on div_big-2208 and -4192 (8 answers) the model restarted the long division in the answer and B
    forced a decimal, 3 times with the sign dropped; on mixed_longer-6219#0 the put was -254 and the model halved it to -127. It costs
    tokens (div_big-2208: 46k-135k per answer). 5 vs 6 of 10 on samples 0-1 is within noise. Open, not built: let B offer the value
    already computed from the question.
[^nudge]: Before `4dc5e546` the nudge cut the last 64 tokens whatever they were and could not clear a line-back loop (line_back counted
    the whole history), so three nudges fired on three lines and the answer ended `stuck` (mip-research#16). Fixed: it cuts only an
    exact cycle and the checks read from the nudge on. The 18-right replay figure is the reason it is dropped. Suppress-recheck
    (`--suppress-recheck 60`), tested in the same round, is dropped too: 20 of 48 alone and cut answers 3 -> 9 with A on the A100.
[^cand]: Before `4dc5e546` the method was a prose keyword check (`substituting`, `verify:`) and two runs of the same derivation
    confirmed each other (mip-research#15); the 4,131 tokens the old rule saved in replay came from those keyword "confirmations".
    Under the target-7 flags without `--execute` the close cannot fire now. Any earlier result of the candidate close
    (including "0 regressions on 96 traces") is withdrawn: that replay covered one of three closing paths.
[^incycle]: The fix to 4,096 is not measured live yet. The other long run of that task (`tl-onhp1`, `For x=N, 325. Good.` with N rising
    to 66,317, 145,880 tokens, answered right) has no exact repeat and is out of its reach.
[^model]: The comments these engine numbers come from do not name the model. The engine runs of mip-research#3 on this PC are on Gyro-M
    (Qwen3.8-Flash-Next), which is most likely the model here; sampling is not stated in the comments read.
[^progclose]: An older form closed on a program repeated 3 times even under `--close-repeat 0`, and that close ended 2 rows lost in
    old arms (mip-research#8).
[^close3]: `onc0` and `onc1` are different exes and 48 of 49 pairs diverged before the close, so this is not a paired result. Harm read
    by hand: by text at least 61 of 294 runs (`tl-onpa/b/c`) ended on this close, 44 right, 17 wrong; 6 wrong runs on answerable
    questions closed mid-thought (a reading under test, a new sub-case) (mip-research#5). `gsm8k-0` in `onc0` opens with a total it
    never added. The planned test on one exe over repeated rounds (`--igc-close-repeat 3` vs `0`, 26 questions, `tl-otra/otrb`) has
    no result in the sources read.
[^note]: 10 of 294 runs ended right after the note; 5 had work left, and `693ebe9d` (`5^5 ÷ 5^4 - 5^3 + 5^2 · 5^1`) stopped after its
    first term twice and answered -95 for 5 (mip-research#5). Whether `stop checking` causes this is not measured.
[^sbclose]: Seconds 0.55x plain in that rotation are not a speed claim (no ABBA). The close's own harm: one wrong program
    (`43e87f6b`, printed `0 6 42`) ended the thinking and the reply said 0. On code the close fails: LiveCodeBench hard, 20 rows, A100,
    plain 17 right (684,271 tokens) against 14 (block at 512, close on a print), 15 (read 256 tokens, then close), 13 (block at 2,048);
    14 long HumanEval tasks on this PC: no gain, 36 of 37 forced blocks printed nothing. Developer's direction: a close that cuts
    reasoning short is out for coding; the math result must clear the same bar on a paired machine (mip-research#3).
