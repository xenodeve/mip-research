# mip-research

**MIP (Model Inline Processing): software computes, the model reasons.** While a model thinks, MIP reads the expression it writes
before `=`, computes it exactly and puts the value after the `=`, and it closes the loops a small model cannot leave, so the model
still answers. No weight is changed.

This repo holds the mechanisms, the harness and the measurements. [Strata-xeno](https://github.com/xenodeve/Strata-xeno) runs MIP in
its engine and server for Qwen3.8-Flash-Next and Qwen3.6-35B-A3B-NVFP4. The results below are measured on **Qwen3.5-4B** with the
current **General Style** (forms true under every reading, no per-model profile).

## At a glance

- **Arithmetic: 95 % fewer tokens and more right answers.** When software takes the whole calculation, the 4B gets **156 of 160**
  DeepMind arithmetic questions right against 118, with **0.05x the tokens** (261,731 -> 12,062) and answers **85 % faster**
  ([Arithmetic mode](#arithmetic-mode-95--fewer-tokens)).
- **On the target model in Strata's engine.** Qwen3.8-Flash-Next with inline values: 5-digit multiplication **-72 % tokens and
  -56 % time**, GSM-Hard -22 % tokens and -18 % time, MATH -18 % tokens, with the same right answers
  ([Flash-Next in Strata](#on-the-target-model-qwen38-flash-next-in-strata)).
- **About 30 % fewer tokens on hard reasoning sets** with the tool route, at the same accuracy (MATH long 0.81x, GPQA 0.73x,
  SuperGPQA 0.71x) ([Tool route](#tool-route-hard-reasoning-sets)).
- **+13 right answers where a 4B model gets stuck.** On questions where the plain model loops: 24 -> **37 of 48** right, and
  answers with no answer at all fall from 21 to **1** ([Results](#results)).
- **No fine-tuning, no second model.** The plain checkpoint with its own sampling; MIP only reads the text and adds tokens to it.
- **Never a wrong number.** **0 wrong values in 39,046** puts, each re-checked by an independent evaluator. The math is exact
  (387 operations, no floating point) and refuses rather than guesses.
- **Ends runaway thinking.** Loops that ran 200k-262k tokens to the end of the context stop at 40k-114k, and right answers on
  MATH level 5 are **40 % shorter**.
- **Fewer tokens where the calculation is fixed.** The same right answer, with and without MIP: polynomial evaluation **-33 %**,
  surds -29 %, unit conversion and sequences -11 % ([Ordinary math](#ordinary-math)).
- **Stays out of the way.** It acts only after an `=` the model wrote, only on exact arithmetic, and every loop threshold sits above
  all right answers: the newest closer touched **0 of 1,078** right answers.

Contents: [Arithmetic mode](#arithmetic-mode-95--fewer-tokens) · [Flash-Next in Strata](#on-the-target-model-qwen38-flash-next-in-strata) ·
[Tool route](#tool-route-hard-reasoning-sets) · [Best config](#best-config) · [Results](#results) ·
[Seeing a calculation coming](#seeing-a-calculation-coming) ·
[Tokens](#tokens) · [Ordinary math](#ordinary-math) · [Detection points](#detection-points) · [How a value is put](#how-a-value-is-put) ·
[Loops it prevents](#loops-it-prevents) · [MathBackend](#mathbackend) · [Staying out of the way](#staying-out-of-the-way) ·
[No fine-tuning](#no-fine-tuning) · [Accuracy audit](#accuracy-audit) · [Next](#next) · [Repo](#repo) · [ภาษาไทย](#ภาษาไทย)

## Arithmetic mode: 95 % fewer tokens

For a question whose work is one fixed calculation, software takes the whole step:

```sh
python tools/igc/tir4b.py run --model <Qwen3.5-4B> --server http://127.0.0.1:8099 --rows dm-rows.jsonl --no-think --greedy \
  --system "Do every arithmetic step exactly." --inline-calc --inline-plain --whole-hint --whole-plain --question-value --hand-off
```

1. **The whole step.** The reply opens asking for the whole calculation as one plain expression in the question's order
   (`--whole-hint --whole-plain`); software computes it exactly, sign and uneven quotient included.
2. **The question's own computation.** Software also reads the computation from the question, in symbols or in words
   (`less than`, `take away`, `product of` ...). On these 160 questions it reads all 160 exactly, so a number the model copies
   wrong is caught (`--question-value`).
3. **The hand-off.** A line that only restates software's value is cut, and the model is led to its answer with
   `So the final answer is \boxed{` (`--hand-off`). The model writes the answer.

DeepMind arithmetic, Qwenputer's 160 questions (40 each of big addition and subtraction, big division, big multiplication, long
multiply-divide chains), Qwen3.5-4B, thinking off, greedy, llama-server on an RTX 5060 Ti:

| | Plain model | Arithmetic mode |
|---|---|---|
| Right | 118 / 160 | **156 / 160** |
| Tokens, all answers | 261,731 | **12,062 (0.046x)** |
| Median tokens per answer | 1,191 | **38** |
| Median seconds per answer | 61 | **8** [^time-a] |

| Kind | Right, plain -> MIP | Tokens with MIP / without |
|---|---|---|
| Big multiplication | 21 -> **40** | **0.018** |
| Big addition and subtraction | 33 -> 39 | 0.044 |
| Long multiply-divide chains | 35 -> **40** | 0.047 |
| Big division | 29 -> 37 | 0.072 |

On the 117 answers right in both arms: **0.042x the tokens (-96 %) and 0.15x the seconds (-85 %)**, the same answer written
without grinding the digits by hand. 39 answers gained. [^cap]

## On the target model: Qwen3.8-Flash-Next in Strata

The same idea inside Strata's own engine (`--igc-result-bridge --igc-inline-exec`): the engine reads `A op B =` as the model writes
it and puts the exact value in the decode loop. Qwen3.8-Flash-Next (Gyro-M, Swift 1.5 IQ2_XS) on this PC, greedy, thinking on.

| Set | Right, MIP on / off | Tokens per answer, on / off | Time per answer, on / off |
|---|---|---|---|
| 5-digit multiplication (40, two rounds each, ABBA) | 39, 39 / 40, 39 | **137, 142 / 491, 522 (-72 %)** | **2.80, 2.82 / 6.25, 6.61 s (-56 %)** |
| 3-digit multiplication (40, ABBA) | 40 / 40 | 74, 72 / 118, 120 (-39 %) | |
| MATH-500 integer answers (60, ABBA) | 58, 58 / 58, 57 | 854, 874 / 1,035, 1,085 (-18 %) | 15.7, 15.3 / 16.6, 17.5 s (-8 %) |
| GSM-Hard (first 200 positive-integer cases) | 177 / 178 | **571 / 731 (-22 %)** | decode 1,942 / 2,363 s in total (-18 %) |
| Word problems with long numbers (48) | 48 / 48 | 249 / 457 (-46 %) | 4.21 / 6.58 s (-36 %) |
| Two-digit + - x, one call at the answer start (120) | 120 / 120 | 14.0 / 35.4, 13.1 / 32.9, 16.4 / 54.8 | 0.91 / 0.98, 0.79 / 0.85, 0.88 / 1.12 s |

The answers stay the same and the work the model would do by hand disappears: the larger the numbers, the larger the cut.
BigBench 3- and 5-digit arithmetic (240 questions): 240/240 with and without, 5-digit multiplication 146 against 485 tokens. [^flash]

## Tool route: hard reasoning sets

The reasoning-tool route (a Python sandbox the model may call, with its prompt line), Qwen3.5-4B on the A100, sampled at T 0.6,
the API reader beside the rule:

| Set | Right, without / with | Tokens with / without |
|---|---|---|
| MATH, long problems (22) | 21 / 21 | **0.81** (167,743 -> 135,772) |
| GPQA (13) | 12 / 11 | **0.73** |
| SuperGPQA, calculation questions (29) | 11 / 10 | **0.71** |

About 30 % fewer tokens at the same accuracy, within one answer per set. [^tool]

## Best config

For reasoning with thinking on, where calculations sit inside a longer argument:

```sh
python tools/igc/tir4b.py run --model <Qwen3.5-4B> --rows rows.jsonl --samples 2 --seed 1 --ids --out out/arm \
  --inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress \
  --loop-close --answer-close --early-cycle --shape-cap 1024 --cover-cap 32768
```

| Part | Flags |
|---|---|
| Inline values (General Style) | `--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress` |
| A: loop close | `--loop-close` |
| B: answer close | `--answer-close` |
| V7: early cycle | `--early-cycle` |
| Shape cap | `--shape-cap 1024` |
| Cover cap | `--cover-cap 32768` [^cover] |

Qwen3.5-4B, bf16, vLLM 0.33 on a Colab A100 80 GB, T 0.6, top-p 0.95, top-k 20, the full 262,144-token context.

## Results

24 questions where the plain model loops, 2 answers each: 17 DeepMind arithmetic questions (sequences, surds, big divisions, mixed
expressions, unit conversion) and 7 MATH-500 level-5 problems. Scored by the rule, with an API reader beside it and every
disagreement read by hand.

| Arm | Arithmetic | MATH L5 | All | No answer |
|---|---|---|---|---|
| Plain model | 17/34 | 7/14 | 24/48 | 21 |
| Inline values only | 17/34 | 8/14 | 25/48 | 23 |
| **Best config** | **27/34** | **10/14** | **37/48** | **1** |

**+13 right answers over the plain model (24 -> 37)**, +12 over inline values alone, and answers that ended with no answer fall
from 21 to 1. On the three hardest questions with shape cap: 8 of 12 right, none running to the end of the context. [^pairing]

## Tokens

- **Right answers on MATH level 5 are 40 % shorter:** median 15,260 tokens for the plain model, 9,090 with the best config.
- **Runaway thinking stops early.** Shape cap closes a long division at 55k tokens where such loops run past 200k; cover cap stops
  every loop that ran past 150k at 40k-114k.
- **20 answers that had no answer now finish** with one. [^tokens]

## Ordinary math

120 ordinary questions (DeepMind arithmetic and MATH-500 level 5), 2 answers each, inline values against the plain model in one
round with the same seeds. Only answers right in both arms are counted, so each pair is the same right answer written with and
without software's values.

| Kind of question | Pairs | Tokens with MIP / without |
|---|---|---|
| Polynomial evaluation | 17 | **0.67** |
| Surds | 3 | 0.71 |
| Unit conversion | 18 | 0.89 |
| Sequences | 12 | 0.89 |
| Big divisions | 8 | 0.96 |
| Division with remainder | 18 | 0.97 |

Where the work is a fixed calculation the model would otherwise grind by hand, the right answer is up to a third shorter. [^ordinary]

## Seeing a calculation coming

Before a model writes a calculation, its hidden state already shows it. Over 46 stored traces (1.16 M tokens, 3 folds by trace,
alarms set at 1 % of positions), the share of the 3,176 places where a calculation starts that each signal sees **2 tokens ahead**:

| Signal | Seen ahead |
|---|---|
| Probe on the residual stream, Qwen3.5-4B | **89 %** (2,813) |
| Probe on the residual stream, Qwen3.6-35B-A3B | **87 %** (2,769) |
| The MoE router of Qwen3.6, a signal that costs nothing | **83 %** (2,639) |
| The text alone | 53 % (1,690) |

The start of a re-check (4,419 places) is seen by the next-token mass in 78 % (4B) and 70 % (Qwen3.6). These signals let MIP
prepare a value before the `=` instead of after it. [^detect]

## Detection points

MIP reads the model's own text at five points. The prompt, the sampling and the weights stay the model's own.

| # | Where | When | What it may do |
|---|---|---|---|
| 1 | The token `=` | every request stops at an `=` token, in the thinking and in the answer, so MIP reads before the model's next token | put the exact value |
| 2 | The token after a put that ends in a digit (and a `.` right after it) | one token | hold digit tokens down, so the model's next digit cannot glue onto the put (`--no-digit-join`) |
| 3 | Every line end in the thinking | as tokens arrive | V7: close the thinking on a short exact cycle |
| 4 | The thinking since the last close, or the whole answer | every 1,024 tokens and at each request end | the repeat check, shape cap and cover cap: close the thinking (A) or the answer (B) |
| 5 | The answer after B's cue | to a line end or 64 tokens | one value is read and the answer ends |

## How a value is put

1. The text before the `=` is cut to the longest arithmetic expression that ends there, read in every form a model writes
   (`tools/igc/mathexpr.py`: ASCII, LaTeX at any depth, Unicode; `unified_detect` adds calls such as `\gcd(12, 18)`, `\binom{7}{3}`).
2. MathBackend computes it exactly.
3. The value is spelled in the General Style and checked against the refusals ([Staying out of the way](#staying-out-of-the-way)).
4. ` value` is appended to the model's text as plain text and the next request goes on from it: to the model it is the next words
   on the page. If any step says no, nothing is put and the model writes its own value.

**The General Style: a value true however the `=` is read.**

| The model wrote | Put | Why this form |
|---|---|---|
| `12 \times 7 =` | `84` | an integer as itself |
| `9/2 =` | `4.5` | a terminating decimal, as the model spells it |
| `6/26 =` | `3/13` | a quotient that reduces: the lowest terms are the step |
| `34 \div 4 =` in a long division | `8 \text{ rem } 2` | true as a quotient and as a division |
| `1302 / 9 =` where the text works in decimals | `144.6666...` | cut and marked as cut, never a false equality |
| `7^5 =` in work mod 11 | the residue | only when the model's own last `=` there linked a value to its residue |

## Loops it prevents

| Loop | Example | Caught by |
|---|---|---|
| An exact cycle | the same re-check paragraph again and again | the repeat check (20 rounds, 8 past 512 tokens); V7 sooner (6 rounds, 3 past 512, spanning 500+ tokens) |
| Counting | `check 101. No. check 102. No.` | the counting check, digits masked; `--guard-progress` lets it go when nearly every round is new (trial division by prime after prime) |
| A line that keeps coming back | `So the answer is 24?` between other lines | the line-back check |
| A cycle whose numbers change length | `$31090 / 10943 = 2.84...$ (Digit 5244: 2)` | shape cap: each digit run read as one token, an exact cycle of 1,024 rounds |
| A cycle with odd rounds in it | long division where each zero digit makes a shorter round | cover cap: 97 % of the last 32,768 tokens repeat a 12-token stretch seen in the 2,048 before |
| The doubt carried into the answer | `-5511170.918918...` without end, a recount of `19+1` | B: the cycle cut to one round, `The final answer is $\boxed{` put, one value of at most 64 tokens |
| The same value put again and again | `= 15/13 = 15/13 = ...` | the third put in a row of one expression is left to the model |

A loop in the thinking is closed by A: `</think>` is put and the model answers from what it has worked out. Of 21 thinking loops A
closed, 13 then answered right.

## MathBackend

`tools/igc/mathbackend/` is a registry of **387 operations in six modules**: numbers 100 (gcd, lcm, phi, modular inverse, digits,
divisors, primes), geometry 86, algebra 70, counting 70 (C, P, Catalan, derangements), search 43, symbolic 18.

- **Exact only.** Python's standard library: `Fraction` for rationals, `Surd` for `p + q*sqrt(d)`, `Decimal` where a decimal is asked
  for. No floating point stands behind a put; a root or power that is not exact is refused, never rounded.
- **Bounded.** Each operation declares its bound (`@op(name, kind, bound=...)`). Past it, or outside its domain, it raises
  `NotCovered` or `BadInput`, and the caller does not guess: no put.
- **Checked against independent libraries.** Each module has a `sim_*.py` that draws inputs and compares every operation with SymPy,
  mpmath or NetworkX, which the backend itself never imports (`tools/test_mathbackend_*.py`): 387 of 387 operations, 31,064
  cases, 0 disagreements.
- **Fast.** 0.18 ms per `=` on the CPU. Its values match what the model went on to write in 3,896 places on 6 arms it was not built
  from, and many of the rest are slips the model made and software got right.
- **On the GPU too.** The engine's CUDA executor is byte-exact on 1,271 of 1,271 integer cases (floor division, mod, gcd, pow) and
  1,403 of 1,403 rational and decimal cases (`0.1 + 0.2 -> 3/10`), at a median 0.066-0.093 ms per call.

## Staying out of the way

MIP acts only where it is sure the value is right and the model is about to need it.

- **Only after an `=` the model wrote, only on exact arithmetic.** A symbol, a word, a mixed number (`2\frac{1}{3}`), an inexact root,
  or a number too big to be a step: no put. On 120 questions it put values at 19,546 of the 19,636 readable `=`; no other text is touched.
- **No put where the right value is not what the model means:**
  - a number alone (`m=4,512=8*64` is m=4 and 512);
  - nothing new (the value would repeat the expression);
  - an operand of a longer expression still being written;
  - a borrowing column of a written subtraction (`0-9 = 1`), a count of coins;
  - an operator the question defines otherwise (`a * b = a - b`);
  - a congruence with no modulus named;
  - the third put in a row of one expression.
- **Closers act only on a loop, and every threshold sits above the right answers.** Shape cap: right answers reach 805 rounds, the
  cap is 1,024. Cover cap: right answers reach a share of 0.92, the cap is 0.97. Each closer passes a replay of every stored A100 trace
  before it runs on the A100: no act on a right answer, or every act read and understood, and no request vLLM would refuse.
- **The model decides.** A put is a value on the page, not an instruction; a close hands the model `</think>` or the final-answer
  cue, and the model writes the answer.

## No fine-tuning

MIP changes no weight. It reads the text stream at inference time and appends tokens to it. The model is the plain checkpoint with
its model card's sampling. The digit set and the `=` and line-end tokens are read from the tokenizer, so the same mechanisms run on
another model's tokenizer.

## Accuracy audit

The best config's A100 records, counted with `scripts/fail_audit.py` and `scripts/put_reject.py`:

| Check | Result | How it is counted |
|---|---|---|
| Wrong values put, frozen test set | **0 of 41,176** puts | 1,526 stored traces set aside before the General Style was written; it puts a value at 94.4 % of the readable `=` there (30,194 of 31,996), and the model's own next value matches the put 31,621 of 32,812 times |
| Digits glued onto a put | **0** (92 before `--no-digit-join`) | the model's next token read as a continuation of the number, over 17,485 puts |
| Wrong values put | **0 of 39,046** puts | every put re-computed by Python's `Fraction` on a normalised copy, independent of MIP's reader [^recheck] |
| The model doubting a value | **4.1 %** of software's values, against **15.2 %** of its own | a doubt word (`wait`, `no,`, `wrong`, `actually`, `let me recheck`) within the next 40 tokens; the cases read are doubts of a right value |
| Loop close: tokens per right answer | **-35 %** (30,180 -> 19,570) | loop rows, the same round with and without A |
| V7 on right answers | **0 of 915** | replay; it catches 212 of 243 loops sooner, 624k tokens earlier |
| Cover cap on right answers | **0 of 1,078** | every stored A100 trace replayed with cover cap; it stops 12 of the 13 loops past 150k tokens, the 13th at 58k when fed directly |
| Answers turned right | **14** | answer by answer against the inline-only arm of the same round [^lost] |

## Next

- The best config with every closer on a broad set against the plain model, both arms in one round (#19).
- The thresholds on Qwen3.8-Flash-Next and Qwen3.6-35B-A3B-NVFP4, and the put inside Strata's decode loop. [^next]

## Repo

| Path | What |
|---|---|
| `tools/igc/tir4b.py` | the harness: the model in vLLM or llama-server, inline values, the arithmetic mode and every closer |
| `tools/igc/sim_replay.py` | replays stored traces with a mechanism added: where it acts, on how many cut answers, whether on a right one, what vLLM would refuse |
| `tools/igc/mathbackend/`, `tools/igc/mathexpr.py` | the exact computation and the reader |
| `tools/igc/question_read.py` | the computation a question states, read by software |
| `tools/igc/answer_check.py` | the API reader beside the rule (its key stays on the machine that runs it) |
| `tools/qwenputer/` | Native Qwenputer: computation compiled for the engine |
| `scripts/` | the audits and the replay gate behind the numbers here |

The engine side of MIP lives in Strata-xeno (`include/strata/core/igc_*.hpp`, `src/kernels/qwenputer.cu`);
`tools/test_igc_adapter_artifact.py` checks the schema header against a Strata-xeno checkout named by `STRATA_ROOT`.
`tir4b --execute/--box` needs Strata-xeno on `PYTHONPATH`. Issue numbers from before the split are mapped in #22.

```sh
python -m unittest discover -s tools -p "test_*.py" -t .     # 923 tests; 78 skip without a GPU, a model or STRATA_ROOT
```

[^time-a]: Seconds per answer with 16 answers in flight, both arms on the same PC; not an ABBA timing.
[^cap]: The plain model ran with a 4,096-token cap and 20 of its answers reached it. One answer right without MIP was wrong with it
    (`add_or_sub_big-5409`). The hand-off rules were set while reading these same 160 questions, and two identical plain runs on this
    PC gave 114 and 118, so a few rows move between runs. This mode is for calculation questions with thinking off; on LiveBench
    the question reader finds a stated computation in 0 of 568 questions, so it applies where the question is a calculation.
[^flash]: Greedy runs on this PC, which does not repeat itself token for token, so the arms are compared as totals, not per
    question. These runs used the engine build before two later parser fixes (binding tokens, letter adjacency). On GSM8K the tokens
    were equal and the time per token was about 5 % higher with MIP on. No accuracy gain was measured on these sets: the model's own
    arithmetic was already right, and what moves is tokens and time.
[^tool]: One sampled answer per question, the arm without the tool on one A100 account and the arm with it on another. HLE (10):
    4 / 4 right at 1.05x tokens. Almost every call came after the thinking, so most of the saving comes from the tool route's prompt
    line rather than from the calls themselves.
[^detect]: These detectors read; they do not act yet. A put still waits for the `=` and an exact parse. The probes need the hidden
    state, which the engine has and llama-server and vLLM do not expose.
[^cover]: Cover cap is measured by replaying the stored A100 traces; its own A100 round is the next one (#19).
[^pairing]: The best config and the inline-only arm ran in one round on the same questions and seeds; the plain model is another run.
    vLLM does not repeat an answer across runs (#21), so a difference of 1-2 answers is within run-to-run variation.
[^tokens]: Totals over all answers: plain 631k, inline only 754k, best config 936k, because more answers finish and one answer that
    runs to the context end weighs 210-262k. On arithmetic the median right answer is 8,371 tokens with the best config against 6,472,
    since 10 more of them now finish. In this harness the best config runs at 25.7 tokens/s per answer against 37.9: `tir4b` ends a vLLM
    request at every `=` and starts the next after the put; inside Strata's engine the put happens in the decode loop
    (`include/strata/core/igc_inline.hpp`), not timed yet.
[^ordinary]: Other kinds in the same comparison: multiply-divide chains 1.01, long mixed expressions 1.13 (the model checks a value it
    was given), MATH-500 level 5 0.98; all 170 pairs 0.97 (fewer tokens in 93). Right answers in that round, inline values alone
    with no closer: 197 of 240 against 209 for the plain model; every miss in both arms was a loop, which is what the closers are for.
    With thinking on, no mechanism alone has raised accuracy on a broad set yet; what it moves is tokens.
[^recheck]: 1,208 more puts (roots, mod, functions) are outside what this re-check can evaluate.
[^lost]: 2 answers right in the inline-only arm were wrong with the best config, both MATH level-5 problems where V7 closed the
    thinking at 8,433 and 6,582 tokens and the model had finished right at 42,974 and 31,518 without the closers. Of the closes, these
    ended in a wrong answer: A 8 of 21, V7 6 of 17, B 6 of 6 (B acts only on an answer already looping).
[^next]: Also open: a longer V7 span for long MATH derivations, and the answer after a close (the model can go back to dividing).

---

## ภาษาไทย

**MIP (Model Inline Processing): ให้ software คำนวณ ให้ model ใช้เหตุผล** ระหว่างที่ model คิด MIP จะอ่านนิพจน์ที่ model เขียนก่อน `=`
คำนวณให้ตรงเป๊ะ แล้วใส่ค่าต่อท้าย `=` และยังปิด loop ที่ model เล็กออกเองไม่ได้ ให้ model ยังตอบออกมาได้ โดยไม่แตะน้ำหนักของ model เลย

repo นี้เก็บกลไก เครื่องมือทดลอง และผลวัด ส่วน [Strata-xeno](https://github.com/xenodeve/Strata-xeno) เป็นที่ที่เอา MIP ไปรันจริงใน engine กับ server
ให้ Qwen3.8-Flash-Next และ Qwen3.6-35B-A3B-NVFP4 ผลทั้งหมดในนี้วัดบน **Qwen3.5-4B** ด้วย **General Style** ตัวล่าสุด คือรูปแบบค่าที่ถูกไม่ว่าจะอ่านแบบไหน ไม่มี profile แยกตาม model

### สรุปสั้น ๆ

- **เลขคณิต: token ลดลง 95% และตอบถูกมากขึ้น:** พอให้ software คำนวณทั้งขั้น model 4B ตอบถูก **156 จาก 160** ข้อเลขคณิต DeepMind จากเดิม 118
  ใช้ token **0.05 เท่า** (261,731 เหลือ 12,062) และตอบ**เร็วขึ้น 85%**
- **บน model เป้าหมายใน engine ของ Strata:** Qwen3.8-Flash-Next ที่เปิดตัวเขียนค่า คูณเลข 5 หลักใช้ **token ลดลง 72% และเวลาลดลง 56%**
  GSM-Hard token ลดลง 22% เวลาลดลง 18% และ MATH token ลดลง 18% โดยคำตอบที่ถูกยังเท่าเดิม
- **โจทย์เหตุผลยาก ๆ ใช้ token น้อยลงประมาณ 30%** เมื่อเปิด tool route และตอบถูกเท่าเดิม (MATH long 0.81 เท่า, GPQA 0.73 เท่า, SuperGPQA 0.71 เท่า)
- **ตอบถูกเพิ่ม 13 คำตอบ ในโจทย์ที่ model 4B วนติด:** จาก 24 เป็น **37 จาก 48** และคำตอบที่ไม่มีคำตอบเลยลดจาก 21 เหลือ **1**
- **ไม่ต้อง fine-tune ไม่ต้องใช้ model ตัวที่สอง:** ใช้ checkpoint ปกติกับค่าการสุ่มของมันเอง MIP แค่อ่านข้อความแล้วต่อ token ลงไป
- **ไม่เคยใส่เลขผิด:** ใส่ค่า 39,046 ครั้ง **ผิด 0 ครั้ง** ตรวจซ้ำทุกค่าด้วยตัวคำนวณอิสระ คำนวณแบบตรงเป๊ะ (387 operation ไม่มี floating point) ถ้าไม่แน่ใจจะไม่ใส่ ไม่เดา
- **หยุดการคิดที่วนไม่จบ:** loop ที่เคยวิ่งไป 200k-262k token จนเต็ม context หยุดได้ที่ 40k-114k และคำตอบที่ถูกบน MATH ระดับ 5 **สั้นลง 40%**
- **ใช้ token น้อยลงในโจทย์ที่คำนวณตายตัว:** คำตอบที่ถูกคำตอบเดียวกัน ตอนมีและไม่มี MIP แทนค่าในพหุนามลด **33%** จัดรูปรากลด 29% แปลงหน่วยกับลำดับลด 11%
- **ไม่ไปยุ่งกับงานอื่น:** ใส่ค่าเฉพาะหลัง `=` ที่ model เขียนเอง และเฉพาะเลขคณิตที่คำนวณได้ตรงเป๊ะ เกณฑ์ของตัวปิด loop ตั้งไว้สูงกว่าคำตอบที่ถูกทุกตัว
  ตัวปิด loop ตัวล่าสุดแตะคำตอบที่ถูก **0 จาก 1,078**

### โหมดเลขคณิต: token ลดลง 95%

ถ้าโจทย์ทั้งข้อเป็นการคำนวณแบบตายตัว software จะรับไปทั้งขั้น (ดูคำสั่งในหัวข้อ Arithmetic mode ข้างบน)
1. **ทั้งขั้น:** ตอนเริ่มตอบ ให้ model เขียนการคำนวณทั้งหมดเป็นนิพจน์เดียว เรียงตามลำดับของโจทย์ แล้ว software คำนวณให้ตรงเป๊ะ ทั้งเครื่องหมายและการหารที่ไม่ลงตัว
2. **การคำนวณที่โจทย์เขียนมาเอง:** software อ่านการคำนวณจากตัวโจทย์ด้วย ไม่ว่าจะเขียนเป็นสัญลักษณ์หรือเป็นคำ (`less than`, `take away`, `product of` ...)
   ใน 160 ข้อนี้อ่านถูกครบทั้ง 160 ถ้า model ลอกตัวเลขจากโจทย์ผิดก็จับได้
3. **การส่งต่อ (hand-off):** บรรทัดที่แค่พูดค่าของ software ซ้ำจะถูกตัด แล้วพา model ไปที่ `So the final answer is \boxed{` จากนั้น model เขียนคำตอบเอง

ทดสอบบนเลขคณิต DeepMind 160 ข้อของ Qwenputer มี 4 ประเภท ประเภทละ 40 ข้อ คือ บวกลบเลขใหญ่ หารเลขใหญ่ คูณเลขใหญ่ และคูณหารต่อกันยาว ๆ
ใช้ Qwen3.5-4B ปิด thinking, greedy, llama-server บน RTX 5060 Ti

| | model เปล่า | โหมดเลขคณิต |
|---|---|---|
| ตอบถูก | 118 / 160 | **156 / 160** |
| token รวม | 261,731 | **12,062 (0.046 เท่า)** |
| มัธยฐาน token ต่อคำตอบ | 1,191 | **38** |
| มัธยฐานวินาทีต่อคำตอบ | 61 | **8** [^th-time-a] |

| ประเภท | ตอบถูก เปล่า -> MIP | token มี MIP / ไม่มี |
|---|---|---|
| คูณเลขใหญ่ | 21 -> **40** | **0.018** |
| บวกลบเลขใหญ่ | 33 -> 39 | 0.044 |
| คูณหารต่อกันยาว ๆ | 35 -> **40** | 0.047 |
| หารเลขใหญ่ | 29 -> 37 | 0.072 |

117 คำตอบที่ถูกทั้งสองฝั่ง: **token เหลือ 0.042 เท่า (ลด 96%) และเวลาเหลือ 0.15 เท่า (ลด 85%)** เป็นคำตอบเดียวกัน แค่ไม่ต้องไล่คิดตัวเลขเองทีละหลัก
และมีคำตอบที่เปลี่ยนจากผิดเป็นถูก 39 คำตอบ [^th-cap]

### บน model เป้าหมาย: Qwen3.8-Flash-Next ใน Strata

แนวคิดเดียวกันนี้ทำงานอยู่ใน engine ของ Strata เอง (`--igc-result-bridge --igc-inline-exec`) engine อ่าน `A op B =` ตอนที่ model เขียน
แล้วใส่ค่าที่ตรงเป๊ะลงไปใน decode loop ทดสอบกับ Qwen3.8-Flash-Next (Gyro-M, Swift 1.5 IQ2_XS) บนเครื่องนี้ แบบ greedy และเปิด thinking

| ชุดทดสอบ | ตอบถูก เปิด / ปิด MIP | token ต่อคำตอบ เปิด / ปิด | เวลาต่อคำตอบ เปิด / ปิด |
|---|---|---|---|
| คูณเลข 5 หลัก (40 ข้อ อย่างละ 2 รอบ ABBA) | 39, 39 / 40, 39 | **137, 142 / 491, 522 (ลด 72%)** | **2.80, 2.82 / 6.25, 6.61 วินาที (ลด 56%)** |
| คูณเลข 3 หลัก (40 ข้อ ABBA) | 40 / 40 | 74, 72 / 118, 120 (ลด 39%) | |
| MATH-500 ข้อที่ตอบเป็นจำนวนเต็ม (60 ข้อ ABBA) | 58, 58 / 58, 57 | 854, 874 / 1,035, 1,085 (ลด 18%) | 15.7, 15.3 / 16.6, 17.5 วินาที (ลด 8%) |
| GSM-Hard (200 ข้อแรกที่ตอบเป็นจำนวนเต็มบวก) | 177 / 178 | **571 / 731 (ลด 22%)** | เวลา decode รวม 1,942 / 2,363 วินาที (ลด 18%) |
| โจทย์ประยุกต์ที่มีตัวเลขยาว (48 ข้อ) | 48 / 48 | 249 / 457 (ลด 46%) | 4.21 / 6.58 วินาที (ลด 36%) |
| บวก ลบ คูณ เลข 2 หลัก เรียกครั้งเดียวตอนเริ่มตอบ (120 ข้อ) | 120 / 120 | 14.0 / 35.4, 13.1 / 32.9, 16.4 / 54.8 | 0.91 / 0.98, 0.79 / 0.85, 0.88 / 1.12 วินาที |

คำตอบยังเหมือนเดิม แต่งานที่ model ต้องคิดเลขเองหายไป ตัวเลขยิ่งยาว ส่วนที่ลดได้ก็ยิ่งมาก
เลขคณิต BigBench 3 และ 5 หลัก (240 ข้อ) ได้ 240/240 ทั้งตอนเปิดและปิด และคูณเลข 5 หลักใช้ 146 token เทียบกับ 485 [^th-flash]

### tool route: โจทย์เหตุผลยาก ๆ

เปิด tool route คือมี sandbox Python ที่ model เรียกใช้ได้ พร้อมบรรทัด prompt ของมัน ทดสอบกับ Qwen3.5-4B บน A100 แบบสุ่ม T 0.6 และมี API reader ตรวจคู่กับกฎ

| ชุดทดสอบ | ตอบถูก ไม่เปิด / เปิด | token เปิด / ไม่เปิด |
|---|---|---|
| MATH ข้อยาว (22 ข้อ) | 21 / 21 | **0.81** (167,743 เหลือ 135,772) |
| GPQA (13 ข้อ) | 12 / 11 | **0.73** |
| SuperGPQA ข้อคำนวณ (29 ข้อ) | 11 / 10 | **0.71** |

ใช้ token น้อยลงประมาณ 30% และตอบถูกเท่าเดิม ต่างกันไม่เกิน 1 คำตอบต่อชุด [^th-tool]

### best config

ใช้กับงานที่ต้องใช้เหตุผลตอนเปิด thinking ซึ่งการคำนวณแทรกอยู่ในเหตุผลยาว ๆ ใช้ flag ตามคำสั่งในหัวข้อ Best config ข้างบน ประกอบด้วย
- **ตัวเขียนค่า (General Style)**
- **A: loop close**
- **B: answer close**
- **V7: early cycle**
- **shape-cap**
- **cover-cap** [^th-cover]

วัดบน Qwen3.5-4B, bf16, vLLM 0.33, Colab A100 80 GB, T 0.6, top-p 0.95, top-k 20, context เต็ม 262,144 token

### ผลการทดสอบ

ใช้ 24 ข้อที่ model เปล่าวนติด ข้อละ 2 คำตอบ แบ่งเป็นเลขคณิต DeepMind 17 ข้อ (ลำดับ ราก หารเลขใหญ่ นิพจน์ผสม แปลงหน่วย) และ MATH-500 ระดับ 5 อีก 7 ข้อ
ตรวจคำตอบด้วยกฎ มี API reader ตรวจคู่กัน และข้อไหนที่สองฝั่งให้ผลไม่ตรงกันก็อ่านเองทุกข้อ

| แขน | เลขคณิต | MATH L5 | รวม | ไม่มีคำตอบ |
|---|---|---|---|---|
| model เปล่า | 17/34 | 7/14 | 24/48 | 21 |
| ใส่ค่าอย่างเดียว | 17/34 | 8/14 | 25/48 | 23 |
| **best config** | **27/34** | **10/14** | **37/48** | **1** |

- **ตอบถูกเพิ่ม 13 คำตอบจาก model เปล่า (24 เป็น 37)** และเพิ่ม 12 คำตอบจากแบบใส่ค่าอย่างเดียว
- **คำตอบที่จบแบบไม่มีคำตอบ** ลดจาก 21 เหลือ 1
- **3 ข้อที่ยากที่สุดเมื่อเพิ่ม shape-cap:** ถูก 8 จาก 12 และไม่มีคำตอบไหนวิ่งจนเต็ม context [^th-pairing]

### token

- **คำตอบที่ถูกบน MATH ระดับ 5 สั้นลง 40%:** มัธยฐาน 15,260 token ของ model เปล่า เหลือ 9,090 เมื่อใช้ best config
- **หยุดการคิดที่วนไม่จบได้เร็ว:** shape-cap ปิดการหารยาวได้ที่ 55k แทนที่จะเกิน 200k ส่วน cover-cap หยุด loop ที่เคยเกิน 150k ได้ทุกตัว ที่ 40k-114k
- **คำตอบที่เคยไม่มีคำตอบ 20 คำตอบ ตอนนี้ตอบออกมาได้** [^th-tokens]

### โจทย์คณิตทั่วไป

ใช้โจทย์คณิตทั่วไป 120 ข้อ (เลขคณิต DeepMind และ MATH-500 ระดับ 5) ข้อละ 2 คำตอบ เทียบแบบใส่ค่ากับ model เปล่าในรอบเดียวกัน seed เดียวกัน
นับเฉพาะคำตอบที่ถูกทั้งสองฝั่ง แต่ละคู่จึงเป็นคำตอบที่ถูกคำตอบเดียวกัน ตอนที่มีค่าจาก software กับตอนที่ไม่มี

| ประเภทโจทย์ | คู่ | token มี MIP / ไม่มี |
|---|---|---|
| แทนค่าในพหุนาม | 17 | **0.67** |
| จัดรูปราก | 3 | 0.71 |
| แปลงหน่วย | 18 | 0.89 |
| ลำดับ | 12 | 0.89 |
| หารเลขใหญ่ | 8 | 0.96 |
| หารมีเศษ | 18 | 0.97 |

ถ้าเป็นงานคำนวณตายตัวที่ปกติ model ต้องคิดเองด้วยมือ คำตอบที่ถูกจะสั้นลงได้ถึงหนึ่งในสาม [^th-ordinary]

### เห็นการคำนวณก่อนที่จะมาถึง

ก่อน model จะเขียนการคำนวณ hidden state ของมันก็บอกไว้แล้ว วัดจาก trace ที่เก็บไว้ 46 ชุด (1.16 ล้าน token แบ่ง 3 fold ตาม trace ตั้งให้เตือนที่ 1% ของตำแหน่ง)
ดูว่าแต่ละสัญญาณเห็นจุดเริ่มการคำนวณ 3,176 จุด **ล่วงหน้า 2 token** ได้กี่เปอร์เซ็นต์

| สัญญาณ | เห็นล่วงหน้า |
|---|---|
| probe ที่อ่าน residual stream ของ Qwen3.5-4B | **89%** (2,813) |
| probe ที่อ่าน residual stream ของ Qwen3.6-35B-A3B | **87%** (2,769) |
| router ของ MoE ใน Qwen3.6 ซึ่งได้มาฟรีไม่มีต้นทุน | **83%** (2,639) |
| ข้อความอย่างเดียว | 53% (1,690) |

จุดที่ model เริ่มตรวจซ้ำ (4,419 จุด) next-token mass เห็นได้ 78% ใน 4B และ 70% ใน Qwen3.6 สัญญาณพวกนี้ทำให้ MIP เตรียมค่าไว้ก่อนถึง `=` ได้ ไม่ต้องรอหลัง `=` [^th-detect]

### จุดที่ตรวจจับ

MIP อ่านข้อความที่ model เขียนเองอยู่ 5 จุด prompt, การสุ่ม และน้ำหนักของ model ยังเหมือนเดิมทั้งหมด

1. **token `=`:** ทุก request หยุดที่ `=` ทั้งตอนคิดและตอนตอบ MIP จึงได้อ่านก่อนที่ model จะเขียน token ถัดไป จุดนี้ใส่ค่าที่ตรงเป๊ะ
2. **token ถัดจากค่าที่ใส่ ถ้าค่านั้นลงท้ายด้วยตัวเลข** (รวม `.` ที่ตามมาทันที): กดตัวเลขไว้ 1 token ตัวเลขที่ model เขียนต่อจะได้ไม่ไปติดกับค่าที่ใส่
3. **ทุกครั้งที่จบบรรทัดตอนคิด:** V7 ปิดการคิดเมื่อเจอ cycle สั้นที่ซ้ำตรงตัว
4. **การคิดทั้งหมดตั้งแต่ครั้งล่าสุดที่ปิด หรือส่วนคำตอบทั้งหมด:** ตรวจทุก 1,024 token และทุกครั้งที่ request จบ ด้วยตัวตรวจการวน shape-cap และ cover-cap ถ้าเจอก็ปิดการคิด (A) หรือปิดคำตอบ (B)
5. **ส่วนคำตอบหลังประโยคบังคับตอบของ B:** อ่านค่าเดียวแล้วจบคำตอบ

### ใส่ค่ายังไง

1. ตัดข้อความก่อน `=` ให้เหลือนิพจน์เลขที่ยาวที่สุดที่จบตรงนั้น อ่านได้ทุกรูปแบบที่ model เขียน (ASCII, LaTeX ซ้อนกี่ชั้นก็ได้, Unicode และการเรียกฟังก์ชันอย่าง `\gcd(12, 18)`, `\binom{7}{3}`)
2. MathBackend คำนวณให้ตรงเป๊ะ
3. เขียนค่าในรูปแบบ General Style แล้วเช็กกับเงื่อนไขที่ไม่ให้ใส่
4. ต่อ ` value` ท้ายข้อความของ model เป็นข้อความธรรมดา แล้ว request ถัดไปก็เดินต่อจากตรงนั้น สำหรับ model ค่าที่ใส่ให้ก็คือคำถัดไปบนหน้ากระดาษ
   ถ้าขั้นไหนบอกว่าไม่ใส่ ก็ไม่ใส่อะไรเลย model จะเขียนค่าเอง

**General Style คือค่าที่ถูกไม่ว่าจะอ่าน `=` แบบไหน**
- `12 \times 7 =` ใส่ `84`
- `9/2 =` ใส่ `4.5` เป็นทศนิยมที่จบได้ แบบเดียวกับที่ model เขียน
- `6/26 =` ใส่ `3/13` เพราะการย่อเศษส่วนก็คือขั้นนั้นพอดี
- `34 \div 4 =` ระหว่างหารยาว ใส่ `8 \text{ rem } 2` ถูกทั้งในแง่ผลหารและในแง่การหาร
- `1302 / 9 =` ตอนที่ model กำลังทำทศนิยม ใส่ `144.6666...` ตัดทศนิยมแล้วมีเครื่องหมายบอกไว้ ไม่เป็นสมการเท็จ
- ระหว่างทำ mod ใส่เศษเหลือ เฉพาะเมื่อ `=` ตัวล่าสุดที่ model เขียนเองบนบรรทัดนั้นหมายถึงเศษเหลือ

### loop ที่ป้องกันได้

- **cycle ที่ซ้ำตรงตัว:** ตัวตรวจการวน (20 รอบ หรือ 8 รอบเมื่อยาวเกิน 512 token) และ V7 ที่จับได้เร็วกว่า (6 รอบ หรือ 3 รอบ ยาว 500 token ขึ้นไป)
- **การนับเลข** เช่น `check 101. No. check 102. No.`: ตัวตรวจการนับที่ปิดตัวเลขทิ้ง ส่วน `--guard-progress` ปล่อยให้วิ่งต่อถ้าแต่ละรอบเป็นของใหม่เกือบหมด เช่น ลองหารด้วยจำนวนเฉพาะไปทีละตัว
- **บรรทัดที่วนกลับมาเรื่อย ๆ:** ตัวตรวจ line-back
- **cycle ที่ตัวเลขยาวไม่เท่ากัน:** shape-cap มองตัวเลขทั้งก้อนเป็น token เดียว แล้วปิดเมื่อเจอ cycle ที่ซ้ำตรงตัวครบ 1,024 รอบ
- **cycle ที่มีรอบแปลก ๆ แทรก** เช่น หารยาวที่ทุกหลัก 0 ทำให้รอบนั้นสั้นลง: cover-cap ปิดเมื่อ 97% ของ 32,768 token ล่าสุด ซ้ำท่อน 12 token ที่เคยเห็นใน 2,048 token ก่อนหน้า
- **ความสงสัยเดิมที่พาไปวนในส่วนคำตอบ** เช่น ทศนิยมไม่รู้จบ: B ตัด cycle เหลือรอบเดียว ใส่ `The final answer is $\boxed{` แล้วให้ model เขียนค่าเดียว ไม่เกิน 64 token
- **ค่าเดิมถูกใส่ซ้ำไปซ้ำมา:** ครั้งที่ 3 ติดกันของนิพจน์เดียวกันจะปล่อยให้ model เขียนเอง

loop ในส่วนคิดจะถูกปิดด้วย A คือใส่ `</think>` แล้วให้ model ตอบจากสิ่งที่คิดมาแล้ว A ปิดไป 21 ครั้ง และตอบถูก 13 ครั้ง

### MathBackend

`tools/igc/mathbackend/` เป็นที่รวม **387 operation แบ่งเป็น 6 module**:
- numbers 100 (gcd, lcm, phi, inverse mod, หลักตัวเลข, ตัวหาร, จำนวนเฉพาะ)
- geometry 86
- algebra 70
- counting 70 (C, P, Catalan, derangement)
- search 43
- symbolic 18

หลักการทำงาน
- **คำนวณแบบตรงเป๊ะเท่านั้น:** ใช้แค่ standard library ของ Python ได้แก่ `Fraction`, `Surd` (`p + q*sqrt(d)`) และ `Decimal` ไม่มี floating point อยู่เบื้องหลังค่าที่ใส่ ถ้าราก ยกกำลัง ได้ค่าไม่ลงตัวก็ไม่ใส่ ไม่ปัดเศษ
- **มีขอบเขตชัด:** ทุก operation ประกาศขอบเขตของตัวเอง ถ้าเกินขอบเขตหรืออยู่นอกโดเมน จะโยน `NotCovered` หรือ `BadInput` แล้วไม่เดา ผลคือไม่ใส่ค่า
- **ตรวจกับ library อิสระ:** ทุก module มี `sim_*.py` ที่สุ่มค่าเข้าไป แล้วเทียบทุก operation กับ SymPy, mpmath หรือ NetworkX ซึ่งตัว backend เองไม่ได้ import
  ผลคือ 387 จาก 387 operation, 31,064 กรณี ไม่ตรงกัน 0 กรณี
- **เร็ว:** 0.18 ms ต่อ `=` บน CPU ค่าที่ให้ตรงกับที่ model เขียนต่อ 3,896 จุด บน 6 arm ที่ไม่ได้ใช้ตอนสร้าง ส่วนที่ไม่ตรงหลายจุดเป็น model คิดพลาดแต่ software ถูก
- **บน GPU ก็ทำได้:** ตัวคำนวณ CUDA ของ engine ตรงทุก byte ทั้งจำนวนเต็ม 1,271 จาก 1,271 กรณี (หารปัดลง mod gcd ยกกำลัง)
  และเศษส่วนกับทศนิยม 1,403 จาก 1,403 กรณี (`0.1 + 0.2 -> 3/10`) ใช้เวลามัธยฐาน 0.066-0.093 ms ต่อครั้ง

### ไม่กระทบการทำงานอื่น

MIP แทรกเฉพาะตอนที่มั่นใจว่าค่าถูก และ model กำลังจะต้องใช้ค่านั้นจริง ๆ
- **ใส่ค่าเฉพาะหลัง `=` ที่ model เขียนเอง และเฉพาะเลขคณิตที่คำนวณได้ตรงเป๊ะ:** ถ้ามีตัวแปร มีคำพูด มีจำนวนคละ มีรากที่ไม่ลงตัว หรือเลขใหญ่เกินกว่าจะเป็นขั้นหนึ่ง ก็ไม่ใส่
  ใน 120 ข้อ ใส่ค่าไป 19,546 จาก 19,636 จุด `=` ที่อ่านได้ ข้อความส่วนอื่นไม่ถูกแตะเลย
- **ถึงเลขจะถูก ก็ไม่ใส่ ถ้าค่านั้นไม่ใช่สิ่งที่ model หมายถึง:**
  - ตัวเลขเดี่ยว ๆ
  - ค่าที่ซ้ำกับนิพจน์ ไม่มีอะไรใหม่
  - ตัวเลขที่เป็นส่วนหนึ่งของนิพจน์ที่ยังเขียนไม่จบ
  - หลักที่ต้องยืมในการลบแบบตั้งหลัก
  - การนับเหรียญ
  - operator ที่โจทย์นิยามไว้เป็นอย่างอื่น
  - congruence ที่ไม่ได้บอก modulus
  - การใส่ค่าครั้งที่ 3 ติดกันของนิพจน์เดียวกัน
- **ตัวปิด loop ทำงานเฉพาะตอนที่วนจริง และเกณฑ์ตั้งไว้สูงกว่าคำตอบที่ถูกทุกตัว:**
  - shape-cap: คำตอบที่ถูกได้สูงสุด 805 รอบ เกณฑ์อยู่ที่ 1,024
  - cover-cap: คำตอบที่ถูกได้สูงสุด 0.92 เกณฑ์อยู่ที่ 0.97
  - ทุกตัวต้องผ่าน replay ของ trace จาก A100 ทั้งหมดก่อนขึ้น A100 คือต้องไม่แตะคำตอบที่ถูก หรือถ้าแตะต้องอ่านแล้วเข้าใจทุกครั้ง และต้องไม่มี request ที่ vLLM จะไม่รับ
- **model เป็นคนตัดสินใจ:** ค่าที่ใส่เป็นแค่ตัวเลขบนหน้ากระดาษ ไม่ใช่คำสั่ง ส่วนการปิด loop ก็แค่ยื่น `</think>` หรือประโยคบังคับตอบให้ แล้ว model เขียนคำตอบเอง

### ไม่ต้อง fine-tune

MIP ไม่ได้แก้น้ำหนักของ model เลย ทำงานกับข้อความตอน inference อ่านสิ่งที่ model เขียน แล้วต่อ token ลงไป model ที่ใช้คือ checkpoint ปกติ
ใช้ค่าการสุ่มตาม model card ส่วนชุดตัวเลข token `=` และ token จบบรรทัด ได้มาจาก tokenizer กลไกเดียวกันจึงใช้กับ tokenizer ของ model อื่นได้

### ตรวจความแม่นยำ

นับจาก record บน A100 ของ best config ด้วย `scripts/fail_audit.py` และ `scripts/put_reject.py`

| ตรวจอะไร | ผล | นับยังไง |
|---|---|---|
| ใส่ค่าผิด ชุดทดสอบที่แช่แข็งไว้ | **0 จาก 41,176** ครั้ง | trace 1,526 ชุดที่กันไว้ก่อนเขียน General Style ใส่ค่าได้ 94.4% ของ `=` ที่อ่านได้ (30,194 จาก 31,996) และค่าที่ model เขียนต่อตรงกับค่าที่ใส่ 31,621 จาก 32,812 ครั้ง |
| ตัวเลขไปติดกับค่าที่ใส่ | **0** (ก่อนมี `--no-digit-join` มี 92) | token ถัดไปของ model ถูกอ่านเป็นตัวเลขต่อกัน นับจาก 17,485 ครั้งที่ใส่ |
| ใส่ค่าผิด | **0 จาก 39,046** ครั้ง | คำนวณซ้ำทุกค่าด้วย `Fraction` ของ Python ไม่ได้ใช้ตัวอ่านของ MIP เอง [^th-recheck] |
| model สงสัยค่าที่ได้ | ค่าจาก software **4.1%** ส่วนค่าที่ model คิดเอง **15.2%** | นับคำอย่าง `wait`, `no,`, `wrong`, `actually` ใน 40 token ถัดไป กรณีที่อ่านดูเป็นการสงสัยค่าที่ถูกอยู่แล้ว |
| loop close: token ต่อคำตอบที่ถูก | **ลด 35%** (30,180 เหลือ 19,570) | ชุดข้อที่วน รอบเดียวกัน มีและไม่มี A |
| V7 แตะคำตอบที่ถูก | **0 จาก 915** | replay จับ loop ได้เร็วขึ้น 212 จาก 243 เร็วกว่าเดิมรวม 624k token |
| cover-cap แตะคำตอบที่ถูก | **0 จาก 1,078** | replay trace จาก A100 ทั้งหมด หยุด loop ที่เกิน 150k ได้ 12 จาก 13 ส่วนตัวที่ 13 จับได้ที่ 58k เมื่อป้อนเข้าตัวจับตรง ๆ |
| คำตอบที่เปลี่ยนเป็นถูก | **14** | เทียบทีละคำตอบกับแขนใส่ค่าอย่างเดียวในรอบเดียวกัน [^th-lost] |

### ขั้นต่อไป

- **best config ที่มีตัวปิด loop ครบ บนชุดกว้าง:** เทียบกับ model เปล่า สองแขนรันในรอบเดียวกัน (#19)
- **ตั้งเกณฑ์บน model เป้าหมาย:** Qwen3.8-Flash-Next และ Qwen3.6-35B-A3B-NVFP4
- **ย้ายการใส่ค่าเข้าไปใน decode loop ของ Strata** [^th-next]

### repo

ส่วน engine ของ MIP อยู่ที่ Strata-xeno (`include/strata/core/igc_*.hpp`, `src/kernels/qwenputer.cu`) test จะเทียบ header กับ Strata-xeno ให้ เมื่อตั้ง `STRATA_ROOT` ไว้
ส่วน `tir4b --execute/--box` ต้องมี Strata-xeno อยู่ใน `PYTHONPATH` และเลข issue ก่อนแยก repo ดูเทียบได้ที่ #22

[^th-time-a]: หมายเหตุ: วินาทีต่อคำตอบ ตอนที่รันพร้อมกัน 16 คำตอบ ทั้งสองฝั่งรันบนเครื่องเดียวกัน แต่ไม่ได้จับเวลาแบบ ABBA
[^th-cap]: หมายเหตุ: model เปล่ารันโดยมีเพดาน 4,096 token และมี 20 คำตอบที่ชนเพดาน คำตอบที่ถูกตอนไม่มี MIP แต่ผิดตอนมี MIP มี 1 คำตอบ (`add_or_sub_big-5409`)
    กฎ hand-off ตั้งขึ้นระหว่างอ่าน 160 ข้อชุดเดียวกันนี้ และ model เปล่ารันซ้ำ 2 รอบบนเครื่องนี้ได้ 114 กับ 118 ผลจึงขยับได้ไม่กี่ข้อระหว่างรอบ
    โหมดนี้ใช้กับโจทย์คำนวณตอนปิด thinking บน LiveBench ตัวอ่านโจทย์เจอการคำนวณที่โจทย์เขียนไว้ 0 จาก 568 ข้อ จึงใช้ได้เฉพาะเมื่อตัวโจทย์เป็นการคำนวณ
[^th-flash]: หมายเหตุ: รันแบบ greedy บนเครื่องนี้ ซึ่งให้ token ซ้ำเดิมทุกตัวไม่ได้ จึงเทียบเป็นยอดรวม ไม่ได้เทียบทีละข้อ รอบนี้ใช้ engine รุ่นก่อนแก้ parser อีก 2 จุด (binding token กับตัวอักษรที่ติดกัน)
    บน GSM8K token เท่าเดิม แต่เวลาต่อ token สูงขึ้นประมาณ 5% เมื่อเปิด MIP ชุดเหล่านี้ไม่มีความแม่นที่เพิ่มขึ้น เพราะ model คิดเลขเองถูกอยู่แล้ว ที่ขยับคือ token กับเวลา
[^th-tool]: หมายเหตุ: ข้อละ 1 คำตอบแบบสุ่ม แขนไม่เปิดรันบน A100 บัญชีหนึ่ง แขนเปิดรันอีกบัญชีหนึ่ง HLE (10 ข้อ) ได้ 4 / 4 ที่ 1.05 เท่า
    การเรียก tool เกือบทั้งหมดเกิดหลังการคิดจบแล้ว token ที่ลดได้ส่วนใหญ่จึงมาจากบรรทัด prompt ของ tool route มากกว่าจากการเรียก tool
[^th-detect]: หมายเหตุ: ตัวตรวจจับเหล่านี้แค่อ่าน ยังไม่ได้ลงมือทำอะไร การใส่ค่ายังรอ `=` และการ parse ที่ตรงเป๊ะ probe ต้องใช้ hidden state ซึ่ง engine มี แต่ llama-server กับ vLLM ไม่เปิดให้อ่าน
[^th-cover]: หมายเหตุ: cover-cap วัดจากการ replay trace ที่เก็บจาก A100 ส่วนรอบที่รันบน A100 เองคือรอบถัดไป (#19)
[^th-pairing]: หมายเหตุ: best config กับแขนใส่ค่าอย่างเดียวรันในรอบเดียวกัน ข้อเดียวกัน seed เดียวกัน ส่วน model เปล่ามาจากอีกรอบ
    vLLM ให้คำตอบซ้ำเดิมข้ามรอบไม่ได้ (#21) ต่างกัน 1-2 คำตอบจึงอยู่ในช่วงที่ผลแต่ละรอบแกว่งได้
[^th-tokens]: หมายเหตุ: token รวมทุกคำตอบ model เปล่า 631k ใส่ค่าอย่างเดียว 754k best config 936k เพราะมีคำตอบที่คิดจนจบมากขึ้น และคำตอบที่วิ่งจนเต็ม context
    คำตอบเดียวกินไป 210-262k ส่วนเลขคณิต มัธยฐานของคำตอบที่ถูกอยู่ที่ 8,371 token เทียบกับ 6,472 เพราะมีอีก 10 คำตอบที่ตอนนี้คิดจนจบได้
    ในเครื่องมือทดสอบ best config ได้ 25.7 token/s ต่อคำตอบ ส่วน model เปล่าได้ 37.9 เพราะ `tir4b` ต้องจบ request ทุกครั้งที่เจอ `=` แล้วเริ่ม request ใหม่หลังใส่ค่า
    ใน engine ของ Strata การใส่ค่าเกิดอยู่ใน decode loop (`include/strata/core/igc_inline.hpp`) และยังไม่ได้จับเวลา
[^th-ordinary]: หมายเหตุ: ประเภทอื่นในการเทียบชุดเดียวกัน คูณหารต่อกัน 1.01 นิพจน์ผสมยาว 1.13 (model ตรวจค่าที่ได้รับซ้ำ) MATH-500 ระดับ 5 0.98
    รวมทั้ง 170 คู่ 0.97 (ใช้ token น้อยลง 93 คู่) คะแนนในรอบนั้น แบบใส่ค่าอย่างเดียวไม่มีตัวปิด loop ได้ 197 จาก 240 ส่วน model เปล่าได้ 209
    ทุกข้อที่ผิดในทั้งสองแขนเกิดจากการวน ซึ่งเป็นสิ่งที่ตัวปิด loop มีไว้แก้ และตอนเปิด thinking ยังไม่มีกลไกไหนที่เพิ่มความแม่นบนชุดกว้างได้ด้วยตัวเอง สิ่งที่ขยับคือ token
[^th-recheck]: หมายเหตุ: อีก 1,208 ครั้ง (ราก mod ฟังก์ชัน) อยู่นอกขอบเขตที่วิธีตรวจซ้ำนี้คำนวณได้
[^th-lost]: หมายเหตุ: มี 2 คำตอบที่ถูกในแขนใส่ค่าอย่างเดียว แต่ผิดเมื่อใช้ best config ทั้งสองเป็นโจทย์ MATH ระดับ 5 ที่ V7 ปิดการคิดที่ 8,433 และ 6,582 token
    ส่วนตอนไม่มีตัวปิด loop model คิดจนตอบถูกที่ 42,974 และ 31,518 token
    การปิดที่จบด้วยคำตอบผิด: A 8 จาก 21, V7 6 จาก 17, B 6 จาก 6 (B ทำงานเฉพาะกับคำตอบที่วนอยู่แล้ว)
[^th-next]: หมายเหตุ: เรื่องที่ยังเปิดอยู่อีก 2 เรื่อง คือ ช่วงของ V7 ที่ยาวขึ้นสำหรับโจทย์ MATH ที่ต้องคิดยาว และคำตอบหลังการปิด loop (model อาจกลับไปหารต่อ)
