# Benchmarks and method

Every set MIP was measured on, every solid result with its exact setup, and the tools that ran, scored and paired those runs.
Two models carry the results: **Qwen3.5-4B** (bf16 on vLLM on a Colab A100, or on llama-server on this PC's RTX 5060 Ti, through
`tir4b`) and **Qwen3.8-Flash-Next** (the Gyro-M Swift 1.5 IQ2_XS pack on Strata's own engine, on this PC's D2x placement or on one
Colab A100). A result is only comparable with another one from the same model, machine, sampling and round; the method section says why.

## Summary

| Mechanism | What it does (one line) | Runs on | In best config? | Best measured result | Status |
|---|---|---|---|---|---|
| `tir4b run` (vLLM) | Runs Qwen3.5-4B with every MIP rule in Python, stopping the request at each `=` to put the value | tir4b (vLLM) | yes, it is the best config's runner | Loop rows 24 x 2: 37/48 with the closers, against 24/48 for the plain model [^loop48] | in best config |
| `tir4b --server` (llama-server) | The same rules over llama-server's `/completion`, for the 5060 Ti on Windows | tir4b (llama-server) | no | DeepMind 160 arithmetic mode: 156/160 against 118 | built, measured, kept as option |
| `external_bench` + `external_score` | Gyro-M through the engine's direct protocol, fixed first-N cases, one scorer for every arm | Strata engine (xeno/mip-sandbox, not in main) | no (engine side) | GSM-Hard 200: 177 vs 178 right at 571 vs 731 tokens | engine only |
| `sim_replay --recorded` + `sim_gate` / `gate_read` | Replays stored answers through tir4b's state machine with a mechanism added; the gate before any A100 round | offline tool | gate for every best-config part | cover-cap: 0 acts on 1,078 right answers, 12/13 long loops stopped | in best config |
| `answer_check` (API reader) | A second reader beside the rule; every disagreement is read by hand | offline tool (this PC only) | used on every arm | Agreed with the rule on 48/48 answers of round 7 | in best config |
| `standard_rows` / `standard_score` | Published few-shot prompts (Minerva MATH, BBH, MMLU-Pro, SuperGPQA) and their scoring | offline tool + Strata engine | no | MATH 4-shot, thinking off: 47/51 on Gyro-M | built, measured, kept as option |
| `livebench` / `contest_math` | LiveBench rows and its own scoring rules, ported; newer contest rows (AIME 2025/26, HMMT, CMIMC) | offline tool | no | 48 `lb1-rows`: ctrl 21 / 25, inline 21 (see results) | built, measured, kept as option |
| `lcb_score` | LiveCodeBench public and private tests under CPython, with LCB's own header, on a throwaway VM | offline tool (VM only) | no | LCB hard 20 paired: MIP values 18 vs plain 17 at 0.40x tokens | built, measured, kept as option |
| `select_rows` / `arm_plan` / `arm_pair` | Picks the rows an arm can change, reuses rows a rule cannot touch, stops at the first loss, compares two arms row by row | offline tool | no | Used for the Flash-Next A100 plan stages | built, measured, kept as option |
| `coverage` | caught / should-catch per mechanism from an arm's `ids.jsonl` | offline tool | used to report | Round 1: software computed 19,546 of 19,636 `=` (99.5 %) in the 162 answers right in all three arms | in best config |
| `fail_audit` / `put_reject` / `share_sim` | Audits behind the reported numbers: every put re-checked by `Fraction`, doubt after a put, the loop-share threshold | offline tool | used to report | share threshold: right answers reach 0.92, loops past 150k reach 0.999-1.0 | in best config |
| `natural_problems` | 48 word problems with long numbers (not a benchmark) | offline tool + Strata engine | no | 48/48 both arms, 249 vs 457 tokens | engine only |
| Colab parity + `STRATA_IGC_BINDING_CACHE` | The engine queue moved to a Colab A100 after a parity test fixed before the run | Strata engine (xeno/mip-sandbox) | no | A100 repeats itself: 49/49 identical replies in both arms | engine only |
| Batch slots (`--slots`) | MIP state per engine, not per slot | Strata engine | no | — | spec only |
| Qwen3.6-35B-A3B NVFP4 target | The MIP target model on Strata's q36 path | Strata engine (xeno/q36-nvfp4) | no | No MIP run; router sees 83 % of computation starts 2 tokens ahead | built, not measured yet |
| Qwen3.5-4B engine port | The 4B on Strata as `qwen35dense` (qwen4exp shape, Q8_0) | Strata engine (xeno/mip-sandbox, not in main) | no | No MIP run; decode 82.4-82.5 tok/s on the 5060 Ti | built, not measured yet |

**Where each tool outside the best config lives, and what it takes to use it**
- `tir4b --server`: [`tir4b.py`](../tools/igc/tir4b.py); a llama-server on 127.0.0.1 with the Qwen3.5-4B bf16 GGUF, the other flags unchanged.
- `external_bench`, `natural_problems`, Colab parity: [`external_bench.py`](../tools/igc/external_bench.py) driving `strata.exe` from Strata-xeno xeno/mip-sandbox (not in main); needs the Gyro-M pack, the MIP bundle and `--igc-result-bridge --igc-inline-exec`.
- `standard_rows`, `standard_score`, `livebench`, `contest_math`, `lcb_score`: in `tools/igc/`; need the fetched data directory, and `lcb_score` only on a throwaway VM.
- `select_rows`, `arm_plan`, `arm_pair`: in `tools/igc/`; need a base arm's `records.json` and engine log, and row reuse only on a machine that repeats itself.
- Batch slots, the 4B port, Qwen3.6: see their sections below.

## Master results table

Only the latest state of each line of work. Arms are compared within one round on one machine; where that is not the case the row says so.

### Qwen3.5-4B, tir4b on vLLM 0.33, Colab A100 80 GB (bf16, T 0.6, top-p 0.95, top-k 20, 262,144-token context)

| Set | n | Arms | Result | Source |
|---|---|---|---|---|
| Fresh 120: DeepMind 80 (8 modules x 10: mixed_longer, mul_div_longer, div_big, simplify_surd, div_remainder, poly_evaluate, conversion, swr_p_sequence) + MATH-500 level 5, 40 | 120 x 2 samples, seed 1 | control / General Style / Qwen profile | right 209 / 197 / 203 of 240; loops (`stuck`) 31 / 43 / 32; finished-but-wrong 0 / 0 / 3 | mip-research#19, "A100 result" and "A100 result 2" |
| Same 120, tokens of the same right answer | 170 pairs right in both arms, same seed | control vs General | 0.97 overall; poly_evaluate 0.67 (17 pairs), simplify_surd 0.71 (3), conversion 0.89 (18), swr_p_sequence 0.89 (12), div_big 0.96 (8), div_remainder 0.97 (18), mul_div_longer 1.01, mixed_longer 1.13, MATH-500 L5 0.98 (59) [^ordinary] | recomputed from `fresh-gen` / `fresh-ctrl` records |
| Rerun of the 52 rows any arm got wrong | 104 answers | control / General / Qwen, with `--no-digit-join --guard-progress` (control `--guard-progress` only) | right, round 1 -> rerun: 73 -> 71 / 61 -> 73 / 67 -> 79; loops 31 -> 30 / 43 -> 29 / 32 -> 22; digit joins after a put 92 of 21,398 -> 0 of 17,485 (General) [^unpaired] | mip-research#19, "Rerun of the wrong rows" |
| Loop rows: the 24 rows any rerun arm ended `stuck` | 24 x 2 | plain (round 1 control) / inline General (`loop-gen`) / + loop close / + answer close / + V7 early cycle | right 24 / 25 / 37 / 37 / 37 of 48; cut 21 / 23 / 3 / 1 / 1; DeepMind 17 rows: 17 / 17 / — / — / 27 of 34; MATH L5 7 rows: 7 / 8 / — / — / 10 of 14 [^loop48] | mip-research#19, "Loop round" §1 and "A100 round 7"; split recomputed from records |
| Same loop rows, median tokens of right answers | 24 x 2 | plain vs A+B+V7 | MATH L5 15,260 -> 9,090; DeepMind 6,472 -> 8,371 [^median] | recomputed from records |
| 3 hardest loop rows (div_big-2208, simplify_surd-1931, conversion-4179) | 3 x 4 | A+B+V7 + `--shape-cap 1024` | 8/12 right (1/4, 4/4, 3/4), 1 ran to the context end | mip-research#19, "Round 8" |
| Rows where the question gives a value (D, `--loop-value`) | 5 x 4 | A+B+D | 0 `stuck` (loop-gen 7/10); D's value right in 12 of 20, model did not adopt it | mip-research#19, "A100 round 5" |
| Held-out reasoning: AIME 2025 30, HMMT Feb 2025 30, MATH long 22 | 82, sample 0 | Base vs Base + MIP + Sandbox (forced block at 512) | 58 vs 56 right; 0.931x tokens; model opened a block itself 2 times in 82 rows | mip-research#12, "Base vs Base + MIP + Sandbox" |
| 36 reasoning rows (24 Base misses + 12 Base right), seed 2 | 10 paired rows | `--inline-calc` vs control | 2 vs 5 right; dropped | mip-research#12, "Round 1 verdict" |

### Qwen3.5-4B, tir4b on vLLM 0.33, Colab A100 40 GB, 20,000-token cap

| Set | n | Arms | Result | Source |
|---|---|---|---|---|
| `lb1-rows`: 32 LiveBench math + 16 AIME/SMC (`livebench.py`, `contest_math.py`) | 48, T 0.6, seed 2 | ctrl / ctrlb (repeat) / inline / logit8 / inline + `--suppress-recheck 60` | right 21 / 25 / 21 / 19 / 19; re-check word within 300 chars after a value 38.5 / 34.5 / 47.2 / 37.6 / 16.1 %; inline - ctrlb +12.7 points [+7.6, +17.0] [^cap20k] | Strata-xeno#298, "ctrlb ... is in" and "inline + hold-down" |

### Qwen3.5-4B, tir4b on llama-server, this PC's RTX 5060 Ti (bf16 GGUF, 16 slots)

| Set | n | Arms | Result | Source |
|---|---|---|---|---|
| DeepMind arithmetic (Qwenputer's 160 rows: add_or_sub_big, div_big, mul_big, mul_div_multiple_longer, 40 each, seed 0) | 160, greedy, no thinking, 4,096 cap, system "Do every arithmetic step exactly." | base / arithmetic mode (pc9: `--whole-hint --whole-plain --hand-off --question-value` with the fraction close) / ablation: whole step only, no hand-off or question value | right 118 / **156** / 138; tokens vs base 1.00 / **0.05** / 0.11 [^dm160] | Strata-xeno#298, "pc9: 156/160"; mip-research#12, "PC run: whole-plain vs base" |
| Same 160, question reader offline | 160 | `question_read.py` | 160/160 equal to gold's exact value | Strata-xeno#298, "Catch point 1" |

### Qwen3.8-Flash-Next (Gyro-M), Strata engine, this PC (D2x: RTX 5060 Ti + RTX 4070 SUPER + CPU pool), thinking on

The decoding of each run is as its source states; the 49-question rotation's source does not state it.

| Set | n | Arms | Result | Source |
|---|---|---|---|---|
| BIG-bench arithmetic 3- and 5-digit multiply, ABBA (on, off, off, on) | 40 per task | `--igc-result-bridge --igc-inline-exec` on vs off | 5-digit x: right 39, 39 / 40, 39; tokens 137, 142 / 491, 522; seconds 2.80, 2.82 / 6.25, 6.61. 3-digit x: 40 all; tokens 74, 72 / 118, 120. 3- and 5-digit + and -: 40/40 in all four arms | Strata-xeno#276, "ABBA: inline execution on and off" |
| MATH-500, integer answers, same ABBA | 60 | same | right 58, 58 / 58, 57; tokens 854, 874 / 1,035, 1,085; seconds 15.7, 15.3 / 16.6, 17.5 | same |
| GSM8K, same ABBA | 100 | same | right 94, 93 / 95, 94; tokens 290, 298 / 297, 298; seconds 6.14, 5.92 / 5.46, 5.51 (+10 % time) [^gsmcost] | same |
| GSM8K 300 + MATH-500 integers 150 | 450 | on (on16) vs off (off15) | 433 vs 433 after reading every wrong answer; each right alone in 8 | Strata-xeno#276, "Rung 2 scores corrected" |
| The 30 limit and loop cases of that run, run to the end of the context | 30 | on vs off | 23 vs 26 right; tokens per case GSM 2,052 / 2,036, MATH 4,197 / 7,697 [^limit30] | Strata-xeno#276, "Limit and loop cases run to the end" |
| GSM-Hard v2, first 200 positive-integer cases | 200 | on vs off | 177 vs 178 after reading disagreements; 571 vs 731 tokens per case; decode 1,942 vs 2,363 s [^gsmhard] | Strata-xeno#276, "GSM-Hard 200" |
| Word problems with long numbers (`natural_problems.py`) | 48 | on vs off | 48 / 48 right; 249 vs 457 tokens; 4.21 vs 6.58 s [^natural] | Strata-xeno#276, "Sidecar in the engine, and a parser bug" |
| Two-digit + - x (EleutherAI arithmetic), one call at answer start | 40 per task | token-result bridge vs off | 120/120 both; tokens 14.0 / 35.4, 13.1 / 32.9, 16.4 / 54.8 [^bridge] | Strata-xeno#276, "Token-result bridge" |
| 49 questions (29 GSM8K, 20 MATH), 2 rounds in rotation | 98 runs | plain vs `--igc-sandbox --igc-sandbox-after 512 --igc-sandbox-close` | 73 vs 77 right; 247,162 vs 125,608 tokens (0.508x), lower on 41 of 49 questions [^q49] | mip-research#3, "Program then close on 49 questions" |
| HumanEval 0-39 | 38 scoreable | plain vs `--igc-sandbox` | 38/38 both [^humaneval] | mip-research#3, "First coding runs with thinking" |
| HumanEvalFix | 40 | plain vs forced block, call and close | 40/40 both; dropped as a test | mip-research#3, "The forced call, first run" |

### Qwen3.8-Flash-Next (Gyro-M), Strata engine, Colab A100 80 GB (one card, every expert in VRAM)

| Set | n | Arms | Result | Source |
|---|---|---|---|---|
| Parity set: the 49 questions above | 2 rounds per arm | plain / sandbox with close | right 78 (plain) and 76 (close) against 73 and 76 on this PC; tokens 239,028 / 129,930 (0.544); round 1 and round 2 identical on 49/49 | Strata-xeno#283, "Result on the A100" |
| MATH, 4-shot (Minerva), thinking off | first 51 of a shuffled 5,000 | MIP off | 47/51 (92.2 %) | Strata-xeno#283, "A scorer fault" |
| Calculation subsets, thinking off, MIP off, no cap | MATH 93, MMLU-Pro 60, SuperGPQA 60 | first pass | 90 (96.8 %), 56 (93.3 %), 41 (68.3 %) [^hard] | mip-research#3, "Hard suites, first pass" |
| LiveCodeBench hard, paired, LCB tests, 6 s limit | 20, greedy | plain / MIP values + loop checks + repeat close / sandbox closes at 512, 512 + read 256, 2,048 | 17 / **18** / 14 / 15 / 13 right; 684,271 / 271,515 (0.40x) / 394,054 / 396,800 / 222,358 tokens | mip-research#3, "First setting to beat plain" and "Every close lost" |
| LiveCodeBench hard, 10 chosen rows | 10, greedy | no MIP (`lon`) / MIP no sandbox / tool route (`lt2`) | 7 / 8 / 8 right; tool route 0.62x tokens on the 7 rows right in both | mip-research#8, "`lt2`" |
| LiveCodeBench, 30 new hard rows | 29 scored | tool route (`lm1`) | 28/29 after the checker fix; no pair (the paired arm's VM was lost) | mip-research#10 |
| GSM8K, seeded | 60, greedy | no MIP vs tool route | 60 / 60; 24,254 vs 15,044 tokens (0.62x) | mip-research#8, "Full plan, stage 1" |
| MATH level 5, seeded | 60, greedy | no MIP vs tool route | 57 vs 59; 183,394 vs 145,697 (0.79x) | mip-research#8, "Full plan, stages 2 and 3" |
| Sampled OFF/ON pairs (b1 MIP off, no tool; a1 MIP + tool route + sandbox), T 0.6 seed 1 | MATH long 22, GPQA Diamond 13, SuperGPQA calc 29, HLE 10 | OFF vs ON | right 21/21, 12/11, 11/10, 4/4; tokens ON/OFF 0.81, 0.73, 0.71, 1.05; calls inside the thinking 0 in every set [^toolroute] | mip-research#8, "Sampled OFF against ON" and the SuperGPQA and HLE pairs |
| MMLU-Pro long rows, tool route + hints | 19, greedy | no tool / line / wait / stop hint | 16 / 11 / 14 / 15 right; per-row median 1 / 0.37 / 0.59 / 0.81x | mip-research#8, "mps" |

### Offline sets (no generation)

| Set | n | What | Result | Source |
|---|---|---|---|---|
| H3, the frozen fresh set (Qwen3-235B-A22B OpenThoughts4 math 303, Gemini thinking s1K-1.1 ~1,000, DeepSeek-R1 s1K-1.1 non-math 123, DeepSeek-R1 OpenThoughts science 100) | 1,526 traces | General Style replayed, code 1d573ae3, measured once | `=` caught 94.4 % (30,194 / 31,996); false puts 0 of 41,176; model's next value = the put 31,621 / 32,812 | mip-research#19, "H3, the frozen fresh set" |
| Stored A100 traces, 16 arms | 1,078 right answers | `--cover-cap 32768` replay gate | 0 acts on right answers; 12 of 13 answers past 150k stopped at 40k-114k; `vllm_refused` 0 | mip-research#19, "`--cover-cap`" |
| Stored traces, 10 arms with no closer | 915 right, 243 cut | V7 early cycle replay | 0 fires on right answers; 212 / 243 loops caught earlier, 624k tokens earlier | mip-research#19, "Loop round" §6 |
| 46 control traces of the 4B (1.16 M tokens), read by both models | 3,176 computation onsets | detectors, 2 tokens ahead, alarms at 1 % of positions | residual probe 89 % (4B) / 87 % (Qwen3.6-35B-A3B); Qwen3.6 MoE router 83 %; text alone 53 % | Strata-xeno#298, "Detection coverage, 46 traces" |

## Sets

### DeepMind arithmetic (Qwenputer's 160 rows)
Four sets of 40 rows, seed 0, from Qwenputer's protocol: greedy, thinking off, system line "Do every arithmetic step exactly.", 4,096-token
cap. Run on this PC through `tir4b --server` because vLLM does not run on Windows. Arms add one layer at a time:
`--whole-hint --whole-plain` (software computes the whole step), `--hand-off` (cut a line that only restates the value and lead into the box),
`--question-value` (read the computation from the question). The arithmetic-mode command is in the README.

### Fresh 120 (the round-1 rows of the best-config work)
80 DeepMind rows (8 calculation modules x 10, seed 1) and 40 MATH-500 level 5 rows that no rule was written from; 2 samples per row,
the same seeds in every arm (`seed_of(1, id, sample)`). Scored by the rule (`math_answer`) beside the API reader, keyed by sample.
The 24 loop rows are the subset any rerun arm ended `stuck`; they carry the best-config result.

### `lb1-rows` (LiveBench + contest, 48)
32 LiveBench math questions and 16 AIME 2024/25 or SMC questions, scored with LiveBench's own rules ported in
[`livebench.py`](../tools/igc/livebench.py) (AMPS_Hard compared with sympy; an undecided answer scores 0 and is counted as `unverified`)
and [`contest_math.py`](../tools/igc/contest_math.py) (last `\boxed{}` read exactly). Used for the H-trust arms on the 4B.

### Flash-Next external suites
GSM8K, MATH-500 (integer answers only), BIG-bench arithmetic and GSM-Hard v2 (positive-integer cases) as fixed first-N cases through
[`external_bench.py`](../tools/igc/external_bench.py), scored by [`external_score.py`](../tools/igc/external_score.py). The A100 plan
added seeded rows from [`standard_rows.py`](../tools/igc/standard_rows.py) `plan`: GSM8K 60, MATH L5 60, MMLU-Pro calculation 60,
SuperGPQA calculation 60 (of 8,910 passing the rule), GPQA Diamond 60 (Physics + Chemistry), LiveCodeBench v6 60. Published base scores
(GSM8K 93.29, MATH 72.78, BBH 90.87, 4-shot CoT on base weights, arXiv 2608.30320 Table 11) are kept apart: our arms run the post-trained
model with thinking on, on plain prompts, and are a different protocol. GPQA and HLE questions and golds stay off the tracker.

### Qwen3.6-35B-A3B NVFP4 (Strata-xeno#298) and the Qwen3.5-4B engine port (Strata-xeno#297)
No MIP benchmark has run on either yet. What is measured:
- Qwen3.6 on `xeno/q36-nvfp4`, this PC, one session: strata-q36 (5060 + RAM, chunk 2048) prompt 454 / 553 tok/s at 2K / 8K and decode 86.4 /
  55.1, against llama.cpp on both cards with MTP 2 at 2,954-2,986 / 3,445-3,470 and 167-169 / 141-144. With the opt-in prompt expert split
  `STRATA_Q36_PREFILL_SPLIT=1` the issue reports an 8K prompt at 1,627 tok/s against 552 without it, decode unchanged (another session). Its router and residual probe are in the detection table above.
- Qwen3.5-4B on Strata (`qwen35dense`, Q8_0, stage 3 46209ef5), 5060 Ti alone, high priority, 6 runs per arm: prompt 2,283.6-2,320.6 tok/s at
  2K and 3,388.4-3,405.3 at 8K; decode 82.4-82.5 tok/s after a short prompt. HF gate on a Colab L4: top-1 0.90 / 0.98 / 1.00 and greedy
  48/48 on short code / math / Thai prompts [^port4b].
- Where it lives: Strata-xeno xeno/mip-sandbox (4B port: `tools/qwen35_to_qwen4exp.py`, `src/prefill/dense_prefill.cpp`); xeno/q36-nvfp4
  (Qwen3.6). To use for MIP: a sandbox-only IGC mode for the 4B (port stage 4), and MIP wired into q36.

## Method

### `tir4b` (the 4B harness)
[`tir4b.py`](../tools/igc/tir4b.py) rebuilds the engine's rules in Python for Qwen3.5-4B. On vLLM it sends one request, stops at every `=`
whose expression it reads, appends the value and sends the next request; every closer acts between requests. Flags that set the run:
`--model --rows --out --samples --seed --ids` (`ids.jsonl` holds every token and every injected span, which the replay and the audits read),
`--max-tokens 0` (the default: the whole 262,144-token context, the rule for every arm since 2026-11-06), `--memory`.
`--server <url>` (tir4b.py:1989) drives llama-server instead (tir4b.py:1688); `--greedy` and `--no-think` (tir4b.py:1998-1999) are for the
Qwenputer protocol. Each request is checked against vLLM's own limits before it is sent (`vllm_refusal`, tir4b.py:200).
Because each put ends a request, tir4b's tokens per second are not the method's speed [^tokens].

### `sim_replay --recorded` and the replay gate
[`sim_replay.py`](../tools/igc/sim_replay.py) drives tir4b's real `Rollout` with a stored answer as the model and reports every value,
close and nudge an arm would make, and where. `--recorded` (sim_replay.py:242) follows an MIP arm's own trace past its puts, so the
replay can run on the arms that already had MIP; `--rows` (:243) gives each record its question. [`sim_gate.py`](../scripts/sim_gate.py)
replays every stored A100 arm with its own flags (read from `arm.json`) plus the new mechanism, 4 at a time; [`gate_read.py`](../scripts/gate_read.py)
counts acts on right / wrong / cut answers and sets each act on a finished answer against the run's own later close. The rule since
round 5: a mechanism goes on the A100 only after 0 acts on right answers and `vllm_refused` 0 in this gate, then a one-row smoke.
Thresholds come from the same traces: [`share_sim.py`](../scripts/share_sim.py) (`12 2048 32768`) gave the cover-cap level, `shape_sim`
the shape-cap level (right answers reach 805 rounds, wasters 1,874-4,267).

### `answer_check` (the API reader)
[`answer_check.py`](../tools/igc/answer_check.py) asks an OpenAI-compatible model (`ANSWER_CHECK_BASE_URL`, `_MODEL`, `_API_KEY` from a
local `.env`, answer_check.py:73-77) whether a reply settles on the gold. Verdicts are keyed by question id and sample (answer_check.py:34).
It runs on this PC only, never on a VM where model code runs. Agree right: done; disagree: a person reads the case and says who erred.
On Flash-Next the reader (DeepSeek V4 Flash) agreed with 67 hand-labelled cases. Reading the disagreements found and fixed these scorer
faults: `r'\boxed'` never read (43e6b2eb), the Minerva closing line demanded (b1ff415d, MATH 45 % -> 92.2 %), the later of the final-answer
sentence and the box (70b821b4), a root and a set in any order and `±` (bdc0da33, a741a187), a letter hedge before the final `Answer:` (6a29c80d).

### Scorers by suite
- [`external_score.py`](../tools/igc/external_score.py): one scorer for every Flash-Next external arm, plus a census of what the model computes.
- [`standard_score.py`](../tools/igc/standard_score.py): the published few-shot extraction (MATH: Lewkowycz et al. 2022 appendix D; BBH:
  the text after `the answer is`). Strings only, so MATH is a lower bound.
- [`lcb_score.py`](../tools/igc/lcb_score.py): LiveCodeBench's own `import_string` header and comparison, a timed-out case retried alone,
  run on a throwaway VM only (model code runs). After the `pow` fix `lm1` moved 25 -> 28 of 29 and `lnt` 10 -> 11 of 20; all other arms
  unchanged (mip-research#10).
- [`livebench.py`](../tools/igc/livebench.py), [`contest_math.py`](../tools/igc/contest_math.py): above.

### Row selection and arm planning
- [`select_rows.py`](../tools/igc/select_rows.py): the rows a first pass got wrong and a feature can reach, plus a seeded share of right rows.
- [`arm_plan.py`](../tools/igc/arm_plan.py): on a machine that repeats itself, a base row on which no new rule can fire is reused token for
  token (`--reuse-from`, `--reuse-rules`); rows run shortest first and the arm stops at its first loss (`--short-first`, `--stop-on-loss`).
- [`arm_pair.py`](../tools/igc/arm_pair.py): two arms of one set row by row (both / only base / only full / neither), finishes, tokens and
  what the software did.
- [`coverage.py`](../tools/igc/coverage.py): each line is caught / should-catch with its denominator, never a firing count.

### Pairing limits
| Runner | Repeats itself? | Measured | Consequence |
|---|---|---|---|
| Strata engine, this PC (D2x, adaptive swaps, MTP drafts, CPU pool) | no | 1 of 49 and 0 of 49 replies identical between two rounds; swaps off: 0 of 8 | Compare totals over repeated rounds in rotation, not single paired runs |
| Strata engine, Colab A100 80 GB (one card, every expert resident) | yes | 49 of 49 identical in both arms, sandbox included | One round is a paired A/B; rows a rule cannot touch can be reused |
| tir4b on vLLM, Colab A100 (batched) | no, across runs | control reran 2 of 104 token-identical; first difference median token 547 | Arms in different runs are not token-paired; ±1-2 answers is noise. Options left to the developer: `VLLM_BATCH_INVARIANT=1`, both arms in one launch, or repeated rounds |
| Sampled arms (T 0.6), any machine | no, by design | — | Comparisons rest on the spread over seeds; then arms on two A100s may be compared on right answers and tokens, not seconds |

Seconds from Colab are never a speed claim: one card and no D2x placement. Strata-xeno#283 measured A100 VRAM read 1,756 GB/s against
425 (5060 Ti) and 470 (4070 SUPER).

### Colab parity (Strata-xeno#283)
The criteria were fixed before the run on the 49 questions, as a band around this PC's own spread (right 69-77 plain, 72-80 close, tokens
within 15 %, close / plain 0.43-0.59). A100: 78 plain (one above the band, because both A100 rounds are the same run: 39 + 39), 76 close,
239,028 / 129,930 tokens, ratio 0.544, Linux unit tests pass. The queue moved on that result; the L4 was dropped (27 tok/s, not reproducible).
Boot on a Colab CPU spent 620 s hashing the 92 GB model for the MIP binding (0.2 GiB/s without SHA instructions); `STRATA_IGC_BINDING_CACHE`
(f9607519) keys the hash by path, size and write time and cut the next boot to 18 s.

### Logging which mechanism acted (mip-research#11)
The inline value, the mismatch and match lines and several closes were written into the stream with no log line, so a record could not say
what acted. Fix: `log_inject(why, from)` at all 24 sites that set or extend `bridge_tokens_` prints `strata igc inject: <why> at=<token> ids=<...>`
(Strata-xeno xeno/mip-sandbox: include/strata/program/igc_learned_diagnostic.hpp); the bench cuts `engine.log` per row into `records[].events`;
`ran` gains `turn ok seconds restarted code_chars shown_chars`; `report.json` gains `argv git bundle_sha256 tokenizer_sha256`, so a pair can
prove its arms differ only in the flag under test.

### Batch slots (Strata-xeno#284)
`--batch N` decodes through `batch_step`, and every MIP use sits in the solo `GEN` path; the state (`bridge_tokens_`, `thinking_tail_`,
the sandbox session) is one object per engine (Strata-xeno xeno/mip-sandbox: src/program/generate.cpp, `batch_step` and the solo GEN path).
The test queue does not use slots: MIP-off alone could not either, since the arms would take different decode paths. Slot work stopped; the
code stays opt-in. What it would take: per-slot MIP state, and a refusal of `--igc-* --batch` until then.

### Pack tokenizer bytes (Strata-xeno#293)
`tools/strata_tokenizer.py:325` and `:328` (Strata-xeno, as the issue cites them) write `merges.txt` and `tokenizer.json` with `write_text()`, so a Windows pack has CRLF and a
Linux pack LF. The MIP binding hashes raw bytes (Strata-xeno xeno/mip-sandbox: src/core/qwenputer_binding.cpp:120-137), so `bundle-v6` binds
to Windows packs only. Tokens do not change. In place: convert the VM's two files to CRLF and check the digests. Fix (not done): write every pack
file with `newline="\n"` together with a re-bind of the bundle.

### Rules the method follows
Full context in every arm (no `--max-tokens`), sampled at T 0.6 rather than greedy for rule A/Bs (greedy keeps one premise and its misses
regress on any path), the replay gate before any A100 round, the API reader beside the rule on every arm, and coverage reported as
caught / should-catch.

[^ordinary]: Counted from the stored `right` field of the round-1 records (an older rule that marked 16 right surd and set answers wrong, so simplify_surd has only 3 pairs); finished (`stop`) in both arms; ratio of summed tokens. Right answers in that round, after the fixed rule: General 197, control 209. With thinking on, no mechanism alone has raised accuracy on this broad set; what moves is tokens.
[^unpaired]: vLLM does not repeat an answer across runs (mip-research#21): control's own change (-2) is the noise floor of this table. One round each.
[^loop48]: The 24 rows were chosen because some rerun arm looped on them, and the plain column is the round-1 control, another run; only `loop-gen` and the closer arms ran in one round on the same rows and seeds. A+B and A+B+V7 differ in which answers are right (2 gained, 2 lost), within vLLM noise. Answer close turned 1 of 7 acts into a right answer; V7 fired on 17 of 48 answers. The round with every closer including `--cover-cap 32768` against a plain control on 120 fresh rows is ready and not run (compute units).
[^median]: Medians over different sets of right answers (7 vs 10 on MATH L5, 17 vs 27 on DeepMind): the arithmetic median rises because 10 more long answers now finish. Totals over all 48 answers: plain 631,522, A+B+V7 936,996 tokens, since one answer that runs to the context end weighs 210-262k.
[^cap20k]: 20,000-token cap and an A100 40 GB, run before the full-context rule; texts identical between arms on 0 of 48, so no arm is shown to change the number right (ctrl vs ctrlb already differ by 4). The re-check rate is the result. A later 262k-context pair on these rows (`inlineclose` 41/48 vs `ctrl` 33/48) was withdrawn and is not used.
[^dm160]: The plain model hits the 4,096 cap on many answers. The hand-off rules were set while reading these same 160 rows, so 156 is a development result. A variant that led into the box and counted a rounded decimal as a restatement scored lower and was reverted. One base-right row (5409) is still wrong in pc9. On LiveBench the question reader finds a stated computation in 0 of 568 questions, so this mode applies only where the question is a calculation. Not verified here: the README's 261,731 -> 12,062 token totals and the per-answer medians and seconds; the issue gives 0.05x.
[^gsmcost]: The 0.5-0.6 s per GSM8K case came from a delivered value running one token per round (191 of 2,991 rounds, timeline of 25 cases); ceiling of the fix 5.3 %, not built. This build had neither the binding-token parser fixes (af41da64) nor the letter-adjacency refusal (1be41906) that came later.
[^limit30]: The 30 were picked as the hardest cases (every limit or loop of the 450). Off finished all 30 by itself with room, so most limit hits had been long thinking, not loops. The third-repeat close, on in these runs, was later found to end the thinking mid-work (math-3, math-119; mip-research#5).
[^gsmhard]: 20 cases have the same wrong value in both arms; 5 computed by hand show the gold is wrong, not the model. The model's own large products in off had no error, so there was no wrong sum to repair: what moves is tokens and time.
[^natural]: Not a benchmark: a fixed set from `natural_problems.py`. The run had the sidecar on; it fired 0 times, so the result belongs to the parser path. Sidecar results are withdrawn.
[^bridge]: The bridge text is fixed and replaces the model's wording; operands up to two digits; not an ABBA.
[^q49]: Right +4 net is p 0.29, no measured gain; the run answers that the close did not lower accuracy. Token drop p < 0.00001 over 49 questions. One wrong answer came from a wrong program that the close trusted. The close stays off by default (it changes outputs; replies get shorter). On LiveCodeBench every close setting lost right answers (table above).
[^humaneval]: Tasks 32 and 38 use `random` in their tests and cannot be scored in the interpreter. The token difference is one task (39).
[^hard]: First pass of a suite, not a MIP comparison: the misses decide whether a suite has room. MATH's 3 misses are right values in another form; SuperGPQA's 19 misses were the first real room. Not comparable with Table 11.
[^toolroute]: Flash-Next on the Strata engine (engine 6a3c527) on two A100s, not Qwen3.5-4B. One seed. The GPQA 13 rows are a greedy control's 7 misses + 6 right rows, the SuperGPQA 29 its 19 misses + 10 right rows (`select_rows.py`); rows picked as misses turn right on any new path by regression to the mean. HLE: the rule gave ON 5/10; the reader and a hand reading give 4/10 (a hedged exact-answer line). No call ever came inside the thinking, so the saving comes from the tool route's prompt and the thinking itself. On the greedy GPQA 13 rows a tool arm first looked like 11 vs 6; a sampled no-tool, MIP-off arm got 6 of the same 7 misses right, so those wins were the path, not the method.
[^tokens]: Inside Strata's engine the put happens in the decode loop and costs no request (Strata-xeno xeno/mip-sandbox: include/strata/core/igc_inline.hpp); that path has not been timed with the best-config rules. Other results withdrawn and not used anywhere here: 43/48, 0/96 regressions, `inlineclose` 41/48, the sidecar results, the latent (learned) bundle results, the "one-token windows" result (Strata-xeno#276), and the 166 tok/s prompt figure on the A100 (Strata-xeno#283).
[^port4b]: Long prompts drift: KL 0.103-0.125 at 2,048 positions on stage 3 (0.239-0.256 on stage 2), far past quantisation noise; being localised.
