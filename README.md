# mip-research

**MIP (Model Inline Processing): software computes, the model reasons.** While an LLM reasons, MIP intercepts arithmetic expressions at `=` ([Detection points](#detection-points)), injects exact computed values inline ([How a value is put](#how-a-value-is-put)) via a 387-operation zero-float CPU/CUDA engine ([MathBackend](#mathbackend)), and closes repetitive loops that trap smaller models ([Loops it prevents](#loops-it-prevents))—requiring **zero weight changes and no second model** ([No fine-tuning](#no-fine-tuning)). Across benchmarks, MIP cuts tokens by **95%** and boosts accuracy from **118 to 156/160** on single-step calculations ([Arithmetic mode](#arithmetic-mode-95--fewer-tokens)), speeds up 5-digit multiplication by **56% (-72% tokens)** inside [Strata-xeno](https://github.com/xenodeve/Strata-xeno)'s native decode loop on **Qwen3.8-Flash-Next** ([Flash-Next in Strata](#on-the-target-model-qwen38-flash-next-in-strata)), saves **~30% tokens** on hard reasoning via sandbox calls ([Tool route](#tool-route-hard-reasoning-sets)), rescues **+13 looping problems** (`24 -> 37/48`) while cutting unanswered timeouts from **21 to 1** ([Best config](#best-config) · [Results](#results)), shortens MATH Level 5 solutions by **40%** ([Tokens](#tokens)) and routine CoT math by up to **33%** ([Ordinary math](#ordinary-math)), predicts upcoming calculations **2 tokens ahead** at **83%–89%** recall ([Seeing a calculation coming](#seeing-a-calculation-coming)), and maintains **0 wrong values** across **41,176** frozen + **39,046** live insertions with **0 false loop closes** on **1,078** correct answers ([Staying out of the way](#staying-out-of-the-way) · [Accuracy audit](#accuracy-audit) · [Next](#next) · [Repo](#repo) · [ภาษาไทย](#ภาษาไทย)).

## At a glance

| Capability / Benchmark | Key Result | Impact & Guarantee | Section |
|---|---|---|---|
| **Single-step arithmetic** (DeepMind 160, Qwen3.5-4B) | **156 / 160** correct (vs. 118) | **-95% tokens** (`0.046x`: 261,731 -> 12,062) · **85% faster** (`61s -> 8s`) | [Arithmetic mode](#arithmetic-mode-95--fewer-tokens) |
| **Native decode-loop execution** (Qwen3.8-Flash-Next in Strata) | **-72% tokens** · **-56% time** on 5-digit mult | GSM-Hard **-22% tokens / -18% time** · MATH **-18% tokens** · long-number word problems **-46% tokens** at matched accuracy | [Flash-Next in Strata](#on-the-target-model-qwen38-flash-next-in-strata) |
| **Hard reasoning tool route** (MATH long, GPQA, SuperGPQA) | **~30% fewer tokens** (`0.71x–0.81x`) | Same accuracy (within 1 answer per set) on Qwen3.5-4B | [Tool route](#tool-route-hard-reasoning-sets) |
| **Loop recovery on hard problems** (48 loop-prone runs) | **24 -> 37 / 48** correct (**+13**) | Unanswered timeouts drop from **21 to 1** · Best config combines inline eval + loop closers | [Best config](#best-config) · [Results](#results) |
| **Runaway thinking & token control** | **-40% tokens** on MATH Level 5 | Cuts 200k–262k context-limit loops down to **40k–114k** (`15,260 -> 9,090` median tokens on MATH L5) | [Tokens](#tokens) · [Loops it prevents](#loops-it-prevents) |
| **Routine math inside CoT** (120 ordinary problems) | Up to **-33% tokens** on identical answers | Polynomials **-33%** (`0.67x`) · Surds **-29%** (`0.71x`) · Units & sequences **-11%** (`0.89x`) | [Ordinary math](#ordinary-math) |
| **Predictive lookahead** (2 tokens before calculation starts) | **89%** (4B) / **87%** (35B) via residual probe | **83%** recall at **zero compute cost** from Qwen3.6's MoE router (vs. 53% from text alone) | [Seeing a calculation coming](#seeing-a-calculation-coming) |
| **Exact CPU & CUDA MathBackend** (387 operations) | **0 wrong values** in **41,176** + **39,046** puts | `0.18 ms` CPU / `0.066–0.093 ms` CUDA · **0 disagreements** in 31,064 cross-checked test cases | [Detection points](#detection-points) · [How a value is put](#how-a-value-is-put) · [MathBackend](#mathbackend) |
| **Non-intrusive guardrails** (1,526 frozen + A100 traces) | **0 / 1,078** (`cover-cap`) · **0 / 915** (`V7`) false closes | Model doubts MIP values only **4.1%** of the time (vs. **15.2%** for its own) · Tokenizer-portable | [Staying out of the way](#staying-out-of-the-way) · [No fine-tuning](#no-fine-tuning) · [Accuracy audit](#accuracy-audit) |

Everything else that was built and measured, including the engine-side mechanisms not in the best config: **[the catalog](docs/INDEX.md)**.

## Arithmetic mode: 95 % fewer tokens

| Metric (DeepMind 160, Qwen3.5-4B, `--no-think`, greedy, RTX 5060 Ti) | Plain model | Arithmetic mode |
|---|---|---|
| Right | 118 / 160 | **156 / 160 (+39 gained, -1 lost)** |
| Tokens, all answers | 261,731 | **12,062 (0.046x)** |
| Median tokens per answer | 1,191 | **38** |
| Median seconds per answer | 61 | **8 (-85%)** [^time-a] |
| Matched right answers (117 right in both arms) | 1.00x tokens / 1.00x time | **0.042x tokens (-96%) / 0.15x seconds (-85%)** [^cap] |

| Kind (40 questions each) | Right, plain -> MIP | Tokens with MIP / without |
|---|---|---|
| Big multiplication | 21 -> **40** | **0.018** |
| Big addition and subtraction | 33 -> **39** | 0.044 |
| Long multiply-divide chains | 35 -> **40** | 0.047 |
| Big division | 29 -> **37** | 0.072 |

| Step | Flags | What software does |
|---|---|---|
| **1. The whole step** | `--whole-hint --whole-plain` | Opens by asking for the whole calculation as one plain expression in the question's order; computes it exactly (sign and uneven quotient included). |
| **2. Question's own computation** | `--question-value` | Reads the computation directly from the question in symbols or words (`less than`, `take away`, `product of`...). Parses **160 / 160** exactly, catching any number the model copies wrong. |
| **3. The hand-off** | `--hand-off` | Cuts any line that only restates software's value and leads the model to `So the final answer is \boxed{` so the model writes the final answer. |

```sh
python tools/igc/tir4b.py run --model <Qwen3.5-4B> --server http://127.0.0.1:8099 --rows dm-rows.jsonl --no-think --greedy \
  --system "Do every arithmetic step exactly." --inline-calc --inline-plain --whole-hint --whole-plain --question-value --hand-off
```

## On the target model: Qwen3.8-Flash-Next in Strata

| Set (Qwen3.8-Flash-Next, Gyro-M Swift 1.5 IQ2_XS, greedy, thinking on, `--igc-result-bridge --igc-inline-exec`) | Right, MIP on / off | Tokens per answer, on / off | Time per answer, on / off |
|---|---|---|---|
| **5-digit multiplication** (40, two rounds each, ABBA) | 39, 39 / 40, 39 | **137, 142 / 491, 522 (-72 %)** | **2.80, 2.82 / 6.25, 6.61 s (-56 %)** |
| **Word problems with long numbers** (48) | 48 / 48 | **249 / 457 (-46 %)** | **4.21 / 6.58 s (-36 %)** |
| **3-digit multiplication** (40, ABBA) | 40 / 40 | **74, 72 / 118, 120 (-39 %)** | — |
| **GSM-Hard** (first 200 positive-integer cases) | 177 / 178 | **571 / 731 (-22 %)** | **decode 1,942 / 2,363 s total (-18 %)** |
| **MATH-500 integer answers** (60, ABBA) | 58, 58 / 58, 57 | **854, 874 / 1,035, 1,085 (-18 %)** | **15.7, 15.3 / 16.6, 17.5 s (-8 %)** |
| **Two-digit `+ - x`**, one call at answer start (120) | 120 / 120 | 14.0 / 35.4, 13.1 / 32.9, 16.4 / 54.8 | 0.91 / 0.98, 0.79 / 0.85, 0.88 / 1.12 s |
| **BigBench 3- and 5-digit arithmetic** (240) | 240 / 240 | **146 / 485** (on 5-digit mult) | — |

Strata's engine reads `A op B =` as the model writes it and injects the exact value directly inside the decode loop: accuracy holds steady while manual digit calculation disappears (the larger the numbers, the larger the cut). [^flash]

## Tool route: hard reasoning sets

| Set (Qwen3.5-4B on A100, T 0.6, rule + API reader) | Right, without / with tool | Tokens with / without tool |
|---|---|---|
| **SuperGPQA, calculation questions** (29) | 11 / 10 | **0.71 (-29 %)** |
| **GPQA** (13) | 12 / 11 | **0.73 (-27 %)** |
| **MATH, long problems** (22) | 21 / 21 | **0.81 (-19 %)** (167,743 -> 135,772) |

A callable Python sandbox (with its prompt line) saves **~30% tokens** on hard reasoning sets at the same accuracy (within 1 answer per set). [^tool]

## Best config

| Part (Qwen3.5-4B, bf16, vLLM 0.33, Colab A100 80 GB, T 0.6, top-p 0.95, top-k 20, 262,144-token context) | Flags | Role in CoT Reasoning (`thinking` on) |
|---|---|---|
| **Inline values (General Style)** | `--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress` | Exact inline values after `=`, prevents digit gluing, and guards prime-trial progress |
| **A: loop close** | `--loop-close` | Closes `<think>` on a thinking loop so the model writes its answer |
| **B: answer close** | `--answer-close` | Cuts answer-stage loops to 1 round and cues `The final answer is $\boxed{` |
| **V7: early cycle** | `--early-cycle` | Closes `<think>` early on short exact cycles at line ends |
| **Shape cap** | `--shape-cap 1024` | Masks digit runs to close 1,024-round structural cycles with changing number lengths |
| **Cover cap** | `--cover-cap 32768` [^cover] | Closes irregular loops when 97% of the last 32,768 tokens repeat 12-token stretches from the 2,048 before |

```sh
python tools/igc/tir4b.py run --model <Qwen3.5-4B> --rows rows.jsonl --samples 2 --seed 1 --ids --out out/arm \
  --inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress \
  --loop-close --answer-close --early-cycle --shape-cap 1024 --cover-cap 32768
```

## Results

| Arm (24 looping questions × 2 samples = 48 answers; rule + API reader + manual review) | Arithmetic (17 Qs / 34 runs) | MATH L5 (7 Qs / 14 runs) | All (48 runs) | No answer |
|---|---|---|---|---|
| Plain model | 17 / 34 | 7 / 14 | 24 / 48 | 21 |
| Inline values only | 17 / 34 | 8 / 14 | 25 / 48 | 23 |
| **Best config** | **27 / 34** | **10 / 14** | **37 / 48 (+13)** | **1 (-20)** |
| **3 hardest questions with `shape-cap`** | — | — | **8 / 12** | **0** hitting context end [^pairing] |

## Tokens

| Metric | Plain model / Without closers | With Best config | Impact |
|---|---|---|---|
| **MATH Level 5 median tokens** (right answers) | 15,260 | **9,090** | **40 % shorter** right answers |
| **Long-division loops** (`shape-cap`) | > 200,000 (to context end) | **55,000** | Stops runaway long division early |
| **Loops running past 150k tokens** (`cover-cap`) | 150,000 – 262,144 | **40,000 – 114,000** | Stops every >150k loop well before context limit |
| **Previously unanswered runs** | 21 with no answer | **1 with no answer** | **20 answers that had no answer now finish** [^tokens] |

## Ordinary math

| Kind of question (120 ordinary questions × 2 samples, matched right pairs, same seed) | Pairs | Tokens with MIP / without | Token cut |
|---|---|---|---|
| **Polynomial evaluation** | 17 | **0.67** | **-33 %** |
| **Surds** | 3 | **0.71** | **-29 %** |
| **Unit conversion** | 18 | **0.89** | **-11 %** |
| **Sequences** | 12 | **0.89** | **-11 %** |
| **Big divisions** | 8 | **0.96** | **-4 %** |
| **Division with remainder** | 18 | **0.97** | **-3 %** |

Only answers right in both arms are counted, isolating pure token savings on the exact same right answer. [^ordinary]

## Seeing a calculation coming

| Signal (46 traces, 1.16 M tokens, 3 folds by trace, alarms at 1 % of positions) | Calculation start seen **2 tokens ahead** (3,176 places) | Re-check start seen ahead (4,419 places, next-token mass) |
|---|---|---|
| **Probe on the residual stream, Qwen3.5-4B** | **89 %** (2,813) | **78 %** |
| **Probe on the residual stream, Qwen3.6-35B-A3B** | **87 %** (2,769) | **70 %** |
| **The MoE router of Qwen3.6** *(costs nothing)* | **83 %** (2,639) | — |
| **The text alone** | 53 % (1,690) | — |

Hidden-state signals let MIP prepare a value before `=` arrives instead of waiting until after it. [^detect]

## Detection points

| # | Where MIP reads | When | What it may do |
|---|---|---|---|
| **1** | **The token `=`** | Every request stops at `=` (in thinking and in answer) before the model's next token | Put the exact value |
| **2** | **Token after a digit-ending put** (and trailing `.`) | 1 token | Hold digit tokens down so the model's next digit cannot glue onto the put (`--no-digit-join`) |
| **3** | **Every line end in thinking** | As tokens arrive | **V7:** Close thinking on a short exact cycle |
| **4** | **Thinking since last close, or whole answer** | Every 1,024 tokens and at each request end | Repeat check, `shape-cap`, and `cover-cap`: close thinking (**A**) or answer (**B**) |
| **5** | **Answer after B's cue** | To a line end or 64 tokens | Read one value and end the answer |

## How a value is put

| Step | Stage | Mechanism |
|---|---|---|
| **1** | **Read expression** | Cuts text before `=` to the longest arithmetic expression ending there (`tools/igc/mathexpr.py`: ASCII, LaTeX at any depth, Unicode; `unified_detect`: `\gcd(12, 18)`, `\binom{7}{3}`). |
| **2** | **Compute** | `MathBackend` computes the exact result. |
| **3** | **Format & check** | Spells the value in **General Style** and checks against refusals ([Staying out of the way](#staying-out-of-the-way)). |
| **4** | **Append** | Appends ` value` as plain text and resumes decoding; if any step refuses, nothing is put and the model writes its own value. |

| The model wrote | Put (General Style) | Why this form is true under every reading of `=` |
|---|---|---|
| `12 \times 7 =` | `84` | An integer as itself |
| `9/2 =` | `4.5` | A terminating decimal, as the model spells it |
| `6/26 =` | `3/13` | A quotient that reduces: lowest terms are the natural step |
| `34 \div 4 =` (in long division) | `8 \text{ rem } 2` | True both as a quotient and as a division step |
| `1302 / 9 =` (in decimal work) | `144.6666...` | Cut and marked with `...`, never a false equality |
| `7^5 =` (in work mod 11) | The residue | Used only when the model's own last `=` there linked a value to its residue |

## Loops it prevents

| Loop | Example | Caught by |
|---|---|---|
| **An exact cycle** | The same re-check paragraph again and again | **Repeat check** (20 rounds, 8 past 512 tokens); **V7** sooner (6 rounds, 3 past 512, spanning 500+ tokens) |
| **Counting** | `check 101. No. check 102. No.` | **Counting check** (digits masked); `--guard-progress` lets it go when nearly every round is new (trial division by prime after prime) |
| **A line that keeps coming back** | `So the answer is 24?` between other lines | **Line-back check** |
| **A cycle whose numbers change length** | `$31090 / 10943 = 2.84...$ (Digit 5244: 2)` | **Shape cap:** each digit run read as one token, an exact cycle of **1,024 rounds** |
| **A cycle with odd rounds in it** | Long division where each zero digit makes a shorter round | **Cover cap:** **97 %** of the last 32,768 tokens repeat a 12-token stretch seen in the 2,048 before |
| **Doubt carried into the answer** | `-5511170.918918...` without end, or recounting `19+1` | **Closer B:** cuts cycle to 1 round, puts `The final answer is $\boxed{`, reads 1 value (<= 64 tokens) |
| **Same value put again and again** | `= 15/13 = 15/13 = ...` | **3rd put in a row** of one expression is left to the model |
| **Thinking loop recovery (Closer A)** | Any loop inside `<think>` | **Closer A** puts `</think>` so the model answers from existing work (**13 of 21** closed loops then answered right) |

## MathBackend

| Aspect | Specification | Verification & Performance |
|---|---|---|
| **387 operations in 6 modules** (`tools/igc/mathbackend/`) | `numbers` 100 (gcd, lcm, phi, mod inverse, digits, divisors, primes) · `geometry` 86 · `algebra` 70 · `counting` 70 (C, P, Catalan, derangements) · `search` 43 · `symbolic` 18 | Registry of exact symbolic & numeric operations |
| **Exact only** | Python stdlib `Fraction` (rationals), `Surd` (`p + q*sqrt(d)`), `Decimal` (when decimal requested) | Zero floating point; inexact roots or powers are refused, never rounded |
| **Bounded** | Each operation declares `@op(name, kind, bound=...)` | Past its bound or domain, raises `NotCovered` or `BadInput` -> no put, zero guessing |
| **Checked against independent libraries** | `sim_*.py` compares every op against `SymPy`, `mpmath`, or `NetworkX` (`tools/test_mathbackend_*.py`) | **387 / 387 operations · 31,064 cases · 0 disagreements** |
| **Fast on CPU** | **0.18 ms per `=`** | Matches what the model went on to write in **3,896 places** on 6 unseen arms (most differences are model slips that software got right) |
| **Byte-exact on GPU (CUDA)** | **0.066 – 0.093 ms median per call** | **1,271 / 1,271** integer cases (`//`, `mod`, `gcd`, `pow`) and **1,403 / 1,403** rational/decimal cases (`0.1 + 0.2 -> 3/10`) |

## Staying out of the way

| Guardrail | Rule | Measured Behavior |
|---|---|---|
| **Strict trigger** | Only after a model-written `=`, only on exact arithmetic (skips symbols, words, mixed numbers `2\frac{1}{3}`, inexact roots, oversized numbers) | Put values at **19,546 of 19,636** readable `=` across 120 questions; no other text touched |
| **Semantic refusals** (when literal value != model intent) | No put on: standalone numbers (`m=4,512=8*64`), no-op repeats, partial operands still being written, subtraction borrowing columns (`0-9 = 1`), coin counts, redefined operators (`a * b = a - b`), congruences without modulus, or 3rd consecutive repeat | Never overwrites non-standard or partial notation |
| **Loop thresholds above all right answers** | `shape-cap`: right answers reach **805 rounds** (cap is **1,024**) · `cover-cap`: right answers reach **0.92** (cap is **0.97**) | Every closer passes full A100 trace replay: **0 acts on right answers** and **0 requests refused by vLLM** |
| **The model decides** | A put is plain text on the page (not an instruction); a close only hands `</think>` or the final-answer cue | The model writes the final answer itself |

## No fine-tuning

| Feature | Detail |
|---|---|
| **Zero weight changes** | Reads the inference text stream and appends tokens; uses the plain checkpoint with its model-card sampling |
| **Tokenizer-agnostic** | Reads digit tokens, `=`, and line-end tokens directly from the tokenizer, running unchanged on another model's tokenizer |

## Accuracy audit

| Check (`scripts/fail_audit.py`, `scripts/put_reject.py`) | Result | How it is counted |
|---|---|---|
| **Wrong values put, frozen test set** | **0 of 41,176** puts | 1,526 stored traces set aside before General Style was written; puts at **94.4 %** of readable `=` (30,194 / 31,996), matching the model's next value **31,621 of 32,812** times |
| **Digits glued onto a put** | **0** *(92 before `--no-digit-join`)* | Model's next token read as a continuation of the number across 17,485 puts |
| **Wrong values put (live A100)** | **0 of 39,046** puts | Every put re-computed by Python's `Fraction` on a normalized copy, independent of MIP's reader [^recheck] |
| **The model doubting a value** | **4.1 %** (software's) vs. **15.2 %** (model's own) | Doubt words (`wait`, `no,`, `wrong`, `actually`, `let me recheck`) within the next 40 tokens (inspected cases are doubts of a right value) |
| **Loop close: tokens per right answer** | **-35 %** (30,180 -> 19,570) | Loop rows in the same round, with and without Closer A |
| **V7 on right answers** | **0 of 915** | Replay; catches **212 of 243** loops sooner, **624k tokens** earlier |
| **Cover cap on right answers** | **0 of 1,078** | All stored A100 traces replayed; stops **12 of 13** loops past 150k tokens (the 13th at 58k when fed directly) |
| **Answers turned right** | **14** | Answer-by-answer against the inline-only arm of the same round [^lost] |

## Next

| Item | Target |
|---|---|
| **Broad unified round (#19)** | Best config with every closer on a broad set against the plain model, both arms in one round |
| **Target models & decode loop** | Thresholds on **Qwen3.8-Flash-Next** and **Qwen3.6-35B-A3B-NVFP4**, and inline put inside Strata's decode loop [^next] |

## Repo

| Path | What it holds |
|---|---|
| `tools/igc/tir4b.py` | Harness: runs the model in vLLM or `llama-server`, inline values, arithmetic mode, and every closer (`--execute/--box` requires Strata-xeno on `PYTHONPATH`) |
| `tools/igc/sim_replay.py` | Replays stored traces with a mechanism added: where it acts, on how many cut answers, whether on a right one, and what vLLM would refuse |
| `tools/igc/mathbackend/`, `tools/igc/mathexpr.py` | Exact computation engine and expression reader |
| `tools/igc/question_read.py` | Reads the computation stated in a question |
| `tools/igc/answer_check.py` | API reader beside the deterministic rule (key stays local) |
| `tools/qwenputer/` | Native Qwenputer: computation compiled for the engine |
| `scripts/` | Audits and replay gates behind all reported numbers |
| [`docs/`](docs/INDEX.md) | The full catalog: every mechanism built or measured, by category (detection, inject, anti-loop, compute, sandbox, learned), engine-only ones included, and every benchmark with its setup |
| `results/arithmetic-mode/` | The records behind the arithmetic-mode numbers (base and pc9) |
| [Strata-xeno](https://github.com/xenodeve/Strata-xeno) (`include/strata/core/igc_*.hpp`, `src/kernels/qwenputer.cu`) | Engine-side MIP implementation (`tools/test_igc_adapter_artifact.py` checks schema headers against `STRATA_ROOT`; pre-split issues mapped in #22) |

```sh
python -m unittest discover -s tools -p "test_*.py" -t .     # 923 tests; 78 skip without a GPU, a model or STRATA_ROOT
```

[^time-a]: Seconds per answer with 16 answers in flight, both arms on the same PC; not an ABBA timing.
[^cap]: The plain model ran with a 4,096-token cap and 20 of its answers reached it. One answer right without MIP was wrong with it (`add_or_sub_big-5409`). The hand-off rules were set while reading these same 160 questions, and two identical plain runs on this PC gave 114 and 118, so a few rows move between runs. This mode is for calculation questions with thinking off; on LiveBench the question reader finds a stated computation in 0 of 568 questions, so it applies where the question is a calculation.
[^flash]: Greedy runs on this PC, which does not repeat itself token for token, so the arms are compared as totals, not per question. These runs used the engine build before two later parser fixes (binding tokens, letter adjacency). On GSM8K the tokens were equal and the time per token was about 5 % higher with MIP on. No accuracy gain was measured on these sets: the model's own arithmetic was already right, and what moves is tokens and time.
[^tool]: One sampled answer per question, the arm without the tool on one A100 account and the arm with it on another. HLE (10): 4 / 4 right at 1.05x tokens. Almost every call came after the thinking, so most of the saving comes from the tool route's prompt line rather than from the calls themselves.
[^detect]: These detectors read; they do not act yet. A put still waits for the `=` and an exact parse. The probes need the hidden state, which the engine has and llama-server and vLLM do not expose.
[^cover]: Cover cap is measured by replaying the stored A100 traces; its own A100 round is the next one (#19).
[^pairing]: The best config and the inline-only arm ran in one round on the same questions and seeds; the plain model is another run. vLLM does not repeat an answer across runs (#21), so a difference of 1-2 answers is within run-to-run variation.
[^tokens]: Totals over all answers: plain 631k, inline only 754k, best config 936k, because more answers finish and one answer that runs to the context end weighs 210-262k. On arithmetic the median right answer is 8,371 tokens with the best config against 6,472, since 10 more of them now finish. In this harness the best config runs at 25.7 tokens/s per answer against 37.9: `tir4b` ends a vLLM request at every `=` and starts the next after the put; inside Strata's engine the put happens in the decode loop (`include/strata/core/igc_inline.hpp`), not timed yet.
[^ordinary]: Other kinds in the same comparison: multiply-divide chains 1.01, long mixed expressions 1.13 (the model checks a value it was given), MATH-500 level 5 0.98; all 170 pairs 0.97 (fewer tokens in 93). Right answers in that round, inline values alone with no closer: 197 of 240 against 209 for the plain model; every miss in both arms was a loop, which is what the closers are for. With thinking on, no mechanism alone has raised accuracy on a broad set yet; what it moves is tokens.
[^recheck]: 1,208 more puts (roots, mod, functions) are outside what this re-check can evaluate.
[^lost]: 2 answers right in the inline-only arm were wrong with the best config, both MATH level-5 problems where V7 closed the thinking at 8,433 and 6,582 tokens and the model had finished right at 42,974 and 31,518 without the closers. Of the closes, these ended in a wrong answer: A 8 of 21, V7 6 of 17, B 6 of 6 (B acts only on an answer already looping).
[^next]: Also open: a longer V7 span for long MATH derivations, and the answer after a close (the model can go back to dividing).

---

## ภาษาไทย

**MIP (Model Inline Processing): ให้ซอฟต์แวร์คิดเลข ให้โมเดลคิดเหตุผล** ระหว่างที่โมเดลกำลังคิด MIP จะดักอ่านสมการหน้าเครื่องหมาย `=` ([จุดที่ตรวจจับ](#จุดที่ตรวจจับ)) แล้วคำนวณค่าที่แม่นยำเป๊ะด้วยคลังคณิตศาสตร์ 387 ฟังก์ชันแบบไร้ floating point ทั้งบน CPU และ CUDA ([MathBackend](#mathbackend-1)) เพื่อเติมคำตอบต่อท้าย `=` ทันทีในระดับเสี้ยววินาที ([ใส่ค่ายังไง](#ใส่ค่ายังไง)) พร้อมตัดจบ loop ที่โมเดลขนาดเล็กชอบคิดวนจนตัน ([loop ที่ป้องกันได้](#loop-ที่ป้องกันได้)) โดย**ไม่ต้องแก้ weight ไม่ต้อง fine-tune และไม่ต้องใช้โมเดลตัวที่สอง** ([ไม่ต้อง fine-tune](#ไม่ต้อง-fine-tune)) ผลลัพธ์คือ: ลด token ลง **95%** (ตอบเร็วขึ้น **85%**) และเพิ่มความแม่นยำจาก **118 เป็น 156/160** ในงานคิดเลขก้อนเดียวจบ ([โหมดเลขคณิต](#โหมดเลขคณิต-token-ลดลง-95)), ลดเวลาคูณเลข 5 หลักลง **56% (token ลด 72%)** เมื่อรันใน decode loop ของ [Strata-xeno](https://github.com/xenodeve/Strata-xeno) บน **Qwen3.8-Flash-Next** ([บน model เป้าหมาย](#บน-model-เป้าหมาย-qwen38-flash-next-ใน-strata)), ประหยัด token ลง **~30%** บนโจทย์เหตุผลระดับยากด้วย Python sandbox ([tool route](#tool-route-โจทย์เหตุผลยาก-ๆ)), กู้คืนคำตอบในโจทย์ที่ชอบติด loop ได้เพิ่ม **+13 ข้อ** (`24 -> 37/48`) และลดเคสไร้คำตอบจาก **21 เหลือ 1** ([best config](#best-config-1) · [ผลการทดสอบ](#ผลการทดสอบ)), ย่อคำตอบที่ถูกบน MATH ระดับ 5 ให้**สั้นลง 40%** ([token](#token)) และโจทย์คณิตทั่วไปสั้นลงสูงสุด **33%** ([โจทย์คณิตทั่วไป](#โจทย์คณิตทั่วไป)), พยากรณ์จังหวะเริ่มคิดเลขล่วงหน้าได้ถึง **2 token** ด้วยความแม่นยำ **83%–89%** ([เห็นการคำนวณก่อนที่จะมาถึง](#เห็นการคำนวณก่อนที่จะมาถึง)) และการันตี**เติมเลขผิด 0 ครั้ง** จากการเติมค่ากว่า **41,176 + 39,046 ครั้ง** โดยไม่ไปแตะคำตอบที่ถูกเลยแม้แต่ข้อเดียว (**0 จาก 1,078**) ([ไม่กระทบการทำงานอื่น](#ไม่กระทบการทำงานอื่น) · [ตรวจความแม่นยำ](#ตรวจความแม่นยำ) · [ขั้นต่อไป](#ขั้นต่อไป) · [repo](#repo-1))

### สรุปสั้น ๆ

| จุดเด่น / ชุดทดสอบ | ผลลัพธ์สำคัญ | ประสิทธิภาพและความแม่นยำที่ได้ | หัวข้อ |
|---|---|---|---|
| **โหมดเลขคณิตก้อนเดียวจบ** (DeepMind 160 ข้อ, Qwen3.5-4B) | ตอบถูก **156 / 160** (จากเดิม 118) | **token ลดลง 95%** (`0.046x`: 261,731 เหลือ 12,062) · **เร็วขึ้น 85%** (`61s -> 8s`) | [โหมดเลขคณิต](#โหมดเลขคณิต-token-ลดลง-95) |
| **รันใน Decode Loop ของ Strata** (Qwen3.8-Flash-Next) | คูณเลข 5 หลัก **token ลด 72% · เวลาลด 56%** | GSM-Hard **token ลด 22% / เวลาลด 18%** · MATH **token ลด 18%** · โจทย์เลขยาว **ลด 46%** โดยความแม่นยำเท่าเดิม | [บน model เป้าหมาย](#บน-model-เป้าหมาย-qwen38-flash-next-ใน-strata) |
| **Tool Route บนโจทย์เหตุผลยาก ๆ** (MATH ข้อยาว, GPQA, SuperGPQA) | **token ลดลง ~30%** (`0.71x–0.81x`) | ความแม่นยำเท่าเดิม (ต่างกันไม่เกิน 1 ข้อต่อชุด) บน Qwen3.5-4B | [tool route](#tool-route-โจทย์เหตุผลยาก-ๆ) |
| **แก้โจทย์ที่โมเดล 4B ชอบคิดวน** (48 รอบที่โมเดลปกติติด loop) | ตอบถูกเพิ่มจาก **24 เป็น 37 / 48 (+13)** | คำตอบที่จบแบบไม่มีคำตอบเลยลดจาก **21 เหลือ 1** ด้วยชุดตั้งค่า Best Config | [best config](#best-config-1) · [ผลการทดสอบ](#ผลการทดสอบ) |
| **หยุดการคิดที่วนไม่จบ & ลด Token** | คำตอบที่ถูกบน MATH ระดับ 5 **สั้นลง 40%** | หยุด loop ที่เคยวิ่ง 200k–262k token จนเต็ม context ได้ที่ **40k–114k** (มัธยฐาน MATH L5 ลดจาก `15,260 เหลือ 9,090`) | [token](#token) · [loop ที่ป้องกันได้](#loop-ที่ป้องกันได้) |
| **ลด Token ในโจทย์คณิตทั่วไป** (เทียบคำตอบที่ถูกเหมือนกัน 120 ข้อ) | สั้นลงสูงสุด **33%** | แทนค่าในพหุนาม **-33%** (`0.67x`) · จัดรูปราก **-29%** (`0.71x`) · แปลงหน่วยกับลำดับ **-11%** (`0.89x`) | [โจทย์คณิตทั่วไป](#โจทย์คณิตทั่วไป) |
| **เห็นการคำนวณล่วงหน้า 2 Token** (ก่อนโมเดลเริ่มเขียนสมการ) | แม่นยำ **89%** (4B) / **87%** (35B) ด้วย probe | อ่านจาก MoE router ของ Qwen3.6 ได้ฟรีโดยไม่มีต้นทุน แม่นยำ **83%** (ดูข้อความอย่างเดียวได้ 53%) | [เห็นการคำนวณก่อนที่จะมาถึง](#เห็นการคำนวณก่อนที่จะมาถึง) |
| **MathBackend แม่นยำ 100% ทั้ง CPU & GPU** (387 ฟังก์ชัน) | **ใส่เลขผิด 0 ครั้ง** จาก **41,176** และ **39,046** ครั้ง | เร็ว `0.18 ms` บน CPU / `0.066–0.093 ms` บน CUDA · เทียบผล 31,064 กรณีกับไลบรารีอิสระ **ตรงกัน 100%** | [จุดที่ตรวจจับ](#จุดที่ตรวจจับ) · [ใส่ค่ายังไง](#ใส่ค่ายังไง) · [MathBackend](#mathbackend-1) |
| **ไม่กระทบงานอื่น & ไม่ต้อง Fine-tune** | แตะคำตอบที่ถูก **0 / 1,078** (`cover-cap`) และ **0 / 915** (`V7`) | โมเดลสงสัยค่าจากซอฟต์แวร์เพียง **4.1%** (เทียบกับคิดเอง **15.2%**) · ใช้กับ tokenizer อื่นได้ทันที | [ไม่กระทบการทำงานอื่น](#ไม่กระทบการทำงานอื่น) · [ไม่ต้อง fine-tune](#ไม่ต้อง-fine-tune) · [ตรวจความแม่นยำ](#ตรวจความแม่นยำ) |

กลไกอื่นทั้งหมดที่สร้างและวัดแล้ว รวมถึงตัวที่อยู่ฝั่ง engine และไม่ได้อยู่ใน best config: **[แคตตาล็อก](docs/INDEX.md#ภาษาไทย)**

### โหมดเลขคณิต: token ลดลง 95%

| ตัวชี้วัด (เลขคณิต DeepMind 160 ข้อ, Qwen3.5-4B, ปิด thinking, greedy, RTX 5060 Ti) | โมเดลปกติ | โหมดเลขคณิต (MIP) |
|---|---|---|
| ตอบถูก | 118 / 160 | **156 / 160 (+39 ข้อที่เปลี่ยนจากผิดเป็นถูก, -1 ข้อ)** |
| token รวมทุกคำตอบ | 261,731 | **12,062 (0.046 เท่า)** |
| มัธยฐาน token ต่อคำตอบ | 1,191 | **38** |
| มัธยฐานวินาทีต่อคำตอบ | 61 | **8 (เร็วขึ้น 85%)** [^th-time-a] |
| เทียบเฉพาะ 117 คำตอบที่ถูกทั้งสองฝั่ง | 1.00x token / 1.00x เวลา | **token เหลือ 0.042 เท่า (-96%) / เวลาเหลือ 0.15 เท่า (-85%)** [^th-cap] |

| ประเภทโจทย์ (ประเภทละ 40 ข้อ) | ตอบถูก (โมเดลปกติ -> MIP) | สัดส่วน token (มี MIP / ไม่มี) |
|---|---|---|
| คูณเลขใหญ่ | 21 -> **40** | **0.018** |
| บวกลบเลขใหญ่ | 33 -> **39** | 0.044 |
| คูณหารต่อกันยาว ๆ | 35 -> **40** | 0.047 |
| หารเลขใหญ่ | 29 -> **37** | 0.072 |

| ขั้นตอน (เมื่อโจทย์ทั้งข้อเป็นการคำนวณตายตัว — ดูคำสั่งรันในหัวข้อ [Arithmetic mode](#arithmetic-mode-95--fewer-tokens)) | Flag | การทำงาน |
|---|---|---|
| **1. คำนวณรวบทั้งขั้น** | `--whole-hint --whole-plain` | ให้โมเดลเขียนการคำนวณทั้งหมดเป็นนิพจน์เดียวเรียงตามลำดับโจทย์ แล้วซอฟต์แวร์คำนวณให้ตรงเป๊ะทั้งเครื่องหมายและการหารที่ไม่ลงตัว |
| **2. อ่านสมการจากตัวโจทย์เอง** | `--question-value` | ซอฟต์แวร์อ่านการคำนวณจากโจทย์โดยตรงทั้งแบบสัญลักษณ์และคำพูด (`less than`, `take away`, `product of`...) อ่านถูกครบ **160 / 160 ข้อ** ถ้าโมเดลลอกเลขผิดก็จับได้ทันที |
| **3. การส่งต่อ (Hand-off)** | `--hand-off` | ตัดบรรทัดที่แค่พูดค่าของซอฟต์แวร์ซ้ำ แล้วพาโมเดลไปที่ `So the final answer is \boxed{` เพื่อให้โมเดลเขียนคำตอบสุดท้ายเอง |

### บน model เป้าหมาย: Qwen3.8-Flash-Next ใน Strata

| ชุดทดสอบ (Qwen3.8-Flash-Next, Gyro-M Swift 1.5 IQ2_XS, greedy, เปิด thinking, `--igc-result-bridge --igc-inline-exec`) | ตอบถูก เปิด / ปิด MIP | token ต่อคำตอบ เปิด / ปิด | เวลาต่อคำตอบ เปิด / ปิด |
|---|---|---|---|
| **คูณเลข 5 หลัก** (40 ข้อ อย่างละ 2 รอบ ABBA) | 39, 39 / 40, 39 | **137, 142 / 491, 522 (ลด 72%)** | **2.80, 2.82 / 6.25, 6.61 วินาที (ลด 56%)** |
| **โจทย์ประยุกต์ที่มีตัวเลขยาว** (48 ข้อ) | 48 / 48 | **249 / 457 (ลด 46%)** | **4.21 / 6.58 วินาที (ลด 36%)** |
| **คูณเลข 3 หลัก** (40 ข้อ ABBA) | 40 / 40 | **74, 72 / 118, 120 (ลด 39%)** | — |
| **GSM-Hard** (200 ข้อแรกที่ตอบเป็นจำนวนเต็มบวก) | 177 / 178 | **571 / 731 (ลด 22%)** | **เวลา decode รวม 1,942 / 2,363 วินาที (ลด 18%)** |
| **MATH-500 ข้อที่ตอบเป็นจำนวนเต็ม** (60 ข้อ ABBA) | 58, 58 / 58, 57 | **854, 874 / 1,035, 1,085 (ลด 18%)** | **15.7, 15.3 / 16.6, 17.5 วินาที (ลด 8%)** |
| **บวก ลบ คูณ เลข 2 หลัก** เรียกครั้งเดียวตอนเริ่มตอบ (120 ข้อ) | 120 / 120 | 14.0 / 35.4, 13.1 / 32.9, 16.4 / 54.8 | 0.91 / 0.98, 0.79 / 0.85, 0.88 / 1.12 วินาที |
| **เลขคณิต BigBench 3 และ 5 หลัก** (240 ข้อ) | 240 / 240 | **146 / 485** (สำหรับคูณเลข 5 หลัก) | — |

ตัว engine ของ Strata อ่าน `A op B =` ทันทีที่โมเดลเขียนแล้วเติมค่าที่ตรงเป๊ะลงไปใน decode loop โดยตรง: คำตอบยังคงเดิมแต่งานทดเลขด้วยมือหายไป ยิ่งตัวเลขยาวก็ยิ่งลดได้มาก [^th-flash]

### tool route: โจทย์เหตุผลยาก ๆ

| ชุดทดสอบ (Qwen3.5-4B บน A100, สุ่มที่ T 0.6, ตรวจด้วยกฎคู่กับ API reader) | ตอบถูก ไม่เปิด / เปิด | token เปิด / ไม่เปิด |
|---|---|---|
| **SuperGPQA ข้อคำนวณ** (29 ข้อ) | 11 / 10 | **0.71 (ลด 29%)** |
| **GPQA** (13 ข้อ) | 12 / 11 | **0.73 (ลด 27%)** |
| **MATH ข้อยาว** (22 ข้อ) | 21 / 21 | **0.81 (ลด 19%)** (167,743 เหลือ 135,772) |

การเปิด Python sandbox ให้โมเดลเรียกใช้ (พร้อมบรรทัด prompt) ช่วยลด token ลงประมาณ **30%** โดยตอบถูกเท่าเดิม (ต่างกันไม่เกิน 1 คำตอบต่อชุด) [^th-tool]

### best config

| ส่วนประกอบ (Qwen3.5-4B, bf16, vLLM 0.33, Colab A100 80 GB, T 0.6, top-p 0.95, top-k 20, context 262,144 token) | Flags (ดูคำสั่งรันในหัวข้อ [Best config](#best-config)) | หน้าที่ในงานคิดเหตุผลยาว ๆ (เปิด `thinking`) |
|---|---|---|
| **ตัวเขียนค่า (General Style)** | `--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress` | เติมค่าที่คำนวณตรงเป๊ะหลัง `=` ป้องกันตัวเลขไหลติดกัน และเปิดทางให้การไล่หารด้วยจำนวนเฉพาะทำงานต่อได้ |
| **A: loop close** | `--loop-close` | ปิด `</think>` เมื่อคิดวน เพื่อให้โมเดลสรุปคำตอบจากสิ่งที่คิดมาแล้ว |
| **B: answer close** | `--answer-close` | ตัด loop ที่ลามเข้าส่วนคำตอบให้เหลือรอบเดียว แล้วส่งเข้า `\boxed{` |
| **V7: early cycle** | `--early-cycle` | ปิด `<think>` ทันทีเมื่อเจอ cycle สั้นที่ซ้ำตรงตัวตอนจบบรรทัด |
| **shape-cap** | `--shape-cap 1024` | มองตัวเลขติดกันทั้งก้อนเป็น token เดียว แล้วปิดเมื่อวนซ้ำครบ 1,024 รอบ |
| **cover-cap** | `--cover-cap 32768` [^th-cover] | ปิดเมื่อ 97% ของ 32,768 token ล่าสุดซ้ำท่อน 12 token ที่เคยเห็นใน 2,048 token ก่อนหน้า |

### ผลการทดสอบ

| รูปแบบการทดสอบ (โจทย์ที่โมเดลปกติวนติด 24 ข้อ × 2 คำตอบ = 48 รอบ; กฎ + API reader + คนอ่านเช็กข้อที่ไม่ตรง) | เลขคณิต DeepMind (17 ข้อ / 34 รอบ) | MATH L5 (7 ข้อ / 14 รอบ) | รวมทั้งหมด (48 รอบ) | ไม่มีคำตอบ |
|---|---|---|---|---|
| โมเดลปกติ | 17 / 34 | 7 / 14 | 24 / 48 | 21 |
| ใส่ค่าอย่างเดียว | 17 / 34 | 8 / 14 | 25 / 48 | 23 |
| **best config** | **27 / 34** | **10 / 14** | **37 / 48 (+13)** | **1 (-20)** |
| **3 ข้อที่ยากที่สุดเมื่อเพิ่ม `shape-cap`** | — | — | **8 / 12** | **0** (ไม่มีรอบไหนวิ่งจนเต็ม context) [^th-pairing] |

### token

| ตัวชี้วัด | โมเดลปกติ / ไม่เปิดใช้ตัวปิด loop | เมื่อใช้ best config | ผลลัพธ์ที่ได้ |
|---|---|---|---|
| **มัธยฐาน token ของคำตอบที่ถูกบน MATH ระดับ 5** | 15,260 | **9,090** | คำตอบที่ถูก **สั้นลง 40%** |
| **การตั้งหารยาวที่วนไม่จบ** (`shape-cap`) | เกิน 200,000 (วิ่งจนเต็ม context) | **55,000** | ปิดการหารยาวที่วนซ้ำได้ตั้งแต่เนิ่น ๆ |
| **loop ที่เคยวิ่งเกิน 150k token** (`cover-cap`) | 150,000 – 262,144 | **40,000 – 114,000** | หยุดได้ครบทุกตัวก่อนชนเพดาน context |
| **คำตอบที่เคยจบแบบไม่มีคำตอบ** | ไม่มีคำตอบ 21 รอบ | **ไม่มีคำตอบ 1 รอบ** | **20 คำตอบที่เคยไม่มีคำตอบ ตอนนี้ตอบออกมาได้** [^th-tokens] |

### โจทย์คณิตทั่วไป

| ประเภทโจทย์ (โจทย์คณิตทั่วไป 120 ข้อ × 2 คำตอบ, นับเฉพาะคู่ที่ถูกทั้งสองฝั่งด้วย seed เดียวกัน) | จำนวนคู่ | token มี MIP / ไม่มี | ส่วนที่สั้นลง |
|---|---|---|---|
| **แทนค่าในพหุนาม** | 17 | **0.67** | **-33%** |
| **จัดรูปราก** | 3 | **0.71** | **-29%** |
| **แปลงหน่วย** | 18 | **0.89** | **-11%** |
| **ลำดับ** | 12 | **0.89** | **-11%** |
| **หารเลขใหญ่** | 8 | **0.96** | **-4%** |
| **หารมีเศษ** | 18 | **0.97** | **-3%** |

แต่ละคู่คือคำตอบที่ถูกคำตอบเดียวกัน จึงสะท้อนการประหยัด token ล้วน ๆ ในงานคำนวณตายตัวที่ปกติโมเดลต้องทดเลขด้วยมือ [^th-ordinary]

### เห็นการคำนวณก่อนที่จะมาถึง

| สัญญาณ (จาก trace 46 ชุด, 1.16 ล้าน token, แบ่ง 3 fold ตาม trace, ตั้งเตือนที่ 1% ของตำแหน่ง) | เห็นจุดเริ่มคำนวณ **ล่วงหน้า 2 token** (3,176 จุด) | เห็นจุดเริ่มตรวจซ้ำล่วงหน้า (4,419 จุด ด้วย next-token mass) |
|---|---|---|
| **probe ที่อ่าน residual stream ของ Qwen3.5-4B** | **89%** (2,813) | **78%** |
| **probe ที่อ่าน residual stream ของ Qwen3.6-35B-A3B** | **87%** (2,769) | **70%** |
| **router ของ MoE ใน Qwen3.6** *(ได้มาฟรีไม่มีต้นทุน)* | **83%** (2,639) | — |
| **ข้อความอย่างเดียว** | 53% (1,690) | — |

สัญญาณจาก hidden state ทำให้ MIP เตรียมค่าไว้ก่อนถึง `=` ได้โดยไม่ต้องรอหลัง `=` [^th-detect]

### จุดที่ตรวจจับ

| # | จุดที่ MIP อ่านข้อความของโมเดล | จังหวะเวลา | สิ่งที่ MIP ทำ (โดยไม่แตะ prompt, ค่าการสุ่ม หรือน้ำหนักโมเดล) |
|---|---|---|---|
| **1** | **token `=`** | ทุก request หยุดที่ `=` ทั้งตอนคิดและตอนตอบ ก่อนโมเดลเขียน token ถัดไป | ใส่ค่าที่คำนวณได้ตรงเป๊ะ |
| **2** | **token ถัดจากค่าที่ใส่ ถ้าลงท้ายด้วยตัวเลข** (รวม `.` ที่ตามมาทันที) | 1 token | กดตัวเลขไว้ 1 จังหวะ (`--no-digit-join`) ไม่ให้ตัวเลขที่โมเดลเขียนต่อไหลไปติดกับค่าที่ใส่ |
| **3** | **ทุกครั้งที่จบบรรทัดตอนคิด** | ทันทีที่ token มาถึง | **V7:** ปิดการคิดเมื่อเจอ cycle สั้นที่ซ้ำตรงตัว |
| **4** | **การคิดตั้งแต่ครั้งล่าสุดที่ปิด หรือส่วนคำตอบทั้งหมด** | ทุก 1,024 token และทุกครั้งที่จบ request | ตัวตรวจการวน, `shape-cap` และ `cover-cap`: ปิดการคิด (**A**) หรือปิดคำตอบ (**B**) |
| **5** | **ส่วนคำตอบหลังประโยคบังคับตอบของ B** | จนจบบรรทัด หรือไม่เกิน 64 token | อ่านค่าเดียวแล้วจบคำตอบ |

### ใส่ค่ายังไง

| ขั้นตอน | การทำงาน | รายละเอียด |
|---|---|---|
| **1. ตัดนิพจน์** | ตัดข้อความก่อน `=` ให้เหลือนิพจน์เลขที่ยาวที่สุดที่จบตรงนั้น | อ่านได้ทุกรูปแบบที่โมเดลเขียน (`mathexpr.py`: ASCII, LaTeX ซ้อนกี่ชั้นก็ได้, Unicode และ `unified_detect`: `\gcd(12, 18)`, `\binom{7}{3}`) |
| **2. คำนวณ** | `MathBackend` คำนวณให้ตรงเป๊ะ | ถ้าขั้นไหนไม่เข้าเกณฑ์ก็จะไม่ใส่อะไรเลย แล้วปล่อยให้โมเดลเขียนค่าเอง |
| **3. จัดรูปแบบ** | เขียนค่าในรูปแบบ **General Style** แล้วเช็กกับเงื่อนไขที่ไม่ให้ใส่ | ดูเงื่อนไขทั้งหมดในหัวข้อ [ไม่กระทบการทำงานอื่น](#ไม่กระทบการทำงานอื่น) |
| **4. ต่อข้อความ** | ต่อ ` value` ท้ายข้อความของโมเดลเป็นข้อความธรรมดา | request ถัดไปเดินต่อจากตรงนั้นทันที สำหรับโมเดลแล้วค่าที่ใส่ก็คือคำถัดไปบนหน้ากระดาษ |

| สิ่งที่โมเดลเขียน | ค่าที่ใส่ (General Style) | เหตุผลที่ใช้รูปแบบนี้ (ถูกต้องเสมอไม่ว่าจะอ่าน `=` แบบไหน) |
|---|---|---|
| `12 \times 7 =` | `84` | จำนวนเต็มปกติ ใส่เป็นตัวมันเอง |
| `9/2 =` | `4.5` | ทศนิยมที่จบได้ เขียนแบบเดียวกับที่โมเดลเขียน |
| `6/26 =` | `3/13` | เศษส่วนที่ทอนได้ การย่อเป็นเศษส่วนอย่างต่ำคือขั้นถัดไปพอดี |
| `34 \div 4 =` (ระหว่างหารยาว) | `8 \text{ rem } 2` | ถูกทั้งในแง่ผลหารและในแง่การหาร |
| `1302 / 9 =` (ตอนกำลังทำทศนิยม) | `144.6666...` | ตัดทศนิยมแล้วใส่ `...` บอกไว้ชัดเจน ไม่เป็นสมการเท็จ |
| `7^5 =` (ระหว่างทำ mod 11) | เศษเหลือ | ใส่เฉพาะเมื่อ `=` ตัวล่าสุดที่โมเดลเขียนเองบนบรรทัดนั้นหมายถึงเศษเหลือ |

### loop ที่ป้องกันได้

| รูปแบบ Loop | ตัวอย่าง | ตัวจับและเกณฑ์การปิด |
|---|---|---|
| **cycle ที่ซ้ำตรงตัว** | ย่อหน้าตรวจคำตอบเดิมซ้ำไปซ้ำมา | **ตัวตรวจการวน** (20 รอบ หรือ 8 รอบเมื่อยาวเกิน 512 token) และ **V7** ที่จับได้เร็วกว่า (6 รอบ หรือ 3 รอบที่ยาว 500+ token) |
| **การนับเลข** | `check 101. No. check 102. No.` | **ตัวตรวจการนับ** (ปิดตัวเลขทิ้ง); `--guard-progress` ปล่อยให้วิ่งต่อถ้าแต่ละรอบเป็นของใหม่เกือบหมด (เช่น ไล่หารด้วยจำนวนเฉพาะทีละตัว) |
| **บรรทัดที่วนกลับมาเรื่อย ๆ** | `So the answer is 24?` แทรกอยู่ระหว่างบรรทัดอื่น | **ตัวตรวจ line-back** |
| **cycle ที่ตัวเลขยาวไม่เท่ากัน** | `$31090 / 10943 = 2.84...$ (Digit 5244: 2)` | **shape-cap:** มองตัวเลขทั้งก้อนเป็น token เดียว แล้วปิดเมื่อเจอ cycle ที่ซ้ำตรงตัวครบ **1,024 รอบ** |
| **cycle ที่มีรอบแปลก ๆ แทรก** | หารยาวที่ทุกหลัก 0 ทำให้รอบนั้นสั้นลง | **cover-cap:** ปิดเมื่อ **97%** ของ 32,768 token ล่าสุด ซ้ำท่อน 12 token ที่เคยเห็นใน 2,048 token ก่อนหน้า |
| **ความสงสัยเดิมที่พาไปวนในส่วนคำตอบ** | ทศนิยมไม่รู้จบ `-5511170.918918...` หรือนับ `19+1` ซ้ำ | **B:** ตัด cycle เหลือรอบเดียว ใส่ `The final answer is $\boxed{` แล้วให้โมเดลเขียนค่าเดียวไม่เกิน 64 token |
| **เติมค่าเดิมซ้ำไปซ้ำมา** | `= 15/13 = 15/13 = ...` | **ครั้งที่ 3 ติดกัน**ของนิพจน์เดียวกันจะปล่อยให้โมเดลเขียนเอง |
| **การปิด loop ในส่วนคิด (A)** | loop ทุกประเภทข้างต้นที่เกิดใน `<think>` | **A** ใส่ `</think>` ให้โมเดลตอบจากสิ่งที่คิดมาแล้ว (จาก 21 ครั้งที่ A ปิด โมเดลตอบถูก **13 ครั้ง**) |

### MathBackend

| คุณสมบัติ | รายละเอียด | ผลการตรวจสอบและความเร็ว |
|---|---|---|
| **387 operation ใน 6 module** (`tools/igc/mathbackend/`) | `numbers` 100 (gcd, lcm, phi, inverse mod, หลักตัวเลข, ตัวหาร, จำนวนเฉพาะ) · `geometry` 86 · `algebra` 70 · `counting` 70 (C, P, Catalan, derangement) · `search` 43 · `symbolic` 18 | คลังคำนวณคณิตศาสตร์แบบแม่นยำ |
| **คำนวณแบบตรงเป๊ะเท่านั้น** | ใช้แค่ standard library ของ Python: `Fraction`, `Surd` (`p + q*sqrt(d)`) และ `Decimal` | ไม่มี floating point อยู่เบื้องหลัง ถ้ารากหรือเลขยกกำลังไม่ลงตัวจะไม่ใส่และไม่ปัดเศษ |
| **มีขอบเขตชัด** | ทุก operation ประกาศขอบเขตตัวเอง (`@op(name, kind, bound=...)`) | ถ้าเกินขอบเขตหรืออยู่นอกโดเมนจะโยน `NotCovered` หรือ `BadInput` แล้วไม่ใส่ค่า (ไม่เดา) |
| **ตรวจกับ library อิสระ** | ทุก module มี `sim_*.py` สุ่มค่าไปเทียบกับ `SymPy`, `mpmath` หรือ `NetworkX` (ซึ่ง backend ไม่ได้ import) | **387 / 387 operation · 31,064 กรณี · ไม่ตรงกัน 0 กรณี** |
| **ความเร็วบน CPU** | **0.18 ms ต่อ `=`** | ค่าที่ให้ตรงกับที่โมเดลเขียนต่อ **3,896 จุด** บน 6 การทดลองที่ไม่ได้ใช้ตอนสร้าง (ส่วนที่ไม่ตรงหลายจุดคือโมเดลคิดพลาดแต่ซอฟต์แวร์คิดถูก) |
| **คำนวณบน GPU (CUDA)** | เวลามัธยฐาน **0.066 – 0.093 ms ต่อครั้ง** | **ตรงทุก byte** ทั้งจำนวนเต็ม **1,271 / 1,271 กรณี** (หารปัดลง, `mod`, `gcd`, ยกกำลัง) และเศษส่วนกับทศนิยม **1,403 / 1,403 กรณี** (`0.1 + 0.2 -> 3/10`) |

### ไม่กระทบการทำงานอื่น

| หลักการป้องกัน | เงื่อนไข / เกณฑ์ที่ใช้ | ผลวัดจากการใช้งานจริง |
|---|---|---|
| **ใส่ค่าเฉพาะจุดที่ชัดเจน** | เฉพาะหลัง `=` ที่โมเดลเขียนเอง และเฉพาะเลขคณิตที่คำนวณได้ตรงเป๊ะ (ถ้ามีตัวแปร คำพูด จำนวนคละ `2\frac{1}{3}` รากที่ไม่ลงตัว หรือเลขใหญ่เกินขั้นหนึ่ง จะไม่ใส่) | ใน 120 ข้อ ใส่ค่าไป **19,546 จาก 19,636** จุด `=` ที่อ่านได้ โดยไม่แตะข้อความส่วนอื่นเลย |
| **ไม่ใส่ถ้าไม่ใช่สิ่งที่โมเดลหมายถึง** | ไม่ใส่เมื่อเป็น: ตัวเลขเดี่ยว ๆ (`m=4,512=8*64`), ค่าซ้ำนิพจน์เดิมไม่มีอะไรใหม่, ตัวเลขที่เป็นส่วนหนึ่งของนิพจน์ที่ยังเขียนไม่จบ, หลักที่ต้องยืมในการลบแบบตั้งหลัก (`0-9 = 1`), การนับเหรียญ, operator ที่โจทย์นิยามเป็นอย่างอื่น (`a * b = a - b`), congruence ที่ไม่บอก modulus, หรือการใส่ค่าครั้งที่ 3 ติดกันของนิพจน์เดียวกัน | ป้องกันการตีความสมการผิดบริบท |
| **เกณฑ์ตัวปิด loop สูงกว่าคำตอบที่ถูกทุกตัว** | `shape-cap`: คำตอบที่ถูกได้สูงสุด **805 รอบ** (เกณฑ์อยู่ที่ **1,024**) · `cover-cap`: คำตอบที่ถูกได้สูงสุด **0.92** (เกณฑ์อยู่ที่ **0.97**) | ทุกตัวผ่าน replay ของ trace จาก A100 ทั้งหมดก่อนรันจริง: ไม่แตะคำตอบที่ถูก และไม่มี request ที่ vLLM ปฏิเสธ |
| **โมเดลเป็นคนตัดสินใจ** | ค่าที่ใส่เป็นแค่ตัวเลขบนหน้ากระดาษ ไม่ใช่คำสั่ง ส่วนการปิด loop แค่ยื่น `</think>` หรือประโยคบังคับตอบให้ | โมเดลเป็นคนเขียนคำตอบสุดท้ายเองเสมอ |

### ไม่ต้อง fine-tune

| ด้าน | การทำงาน |
|---|---|
| **ไม่แก้น้ำหนักโมเดล** | ทำงานกับข้อความตอน inference โดยอ่านสิ่งที่โมเดลเขียนแล้วต่อ token ลงไป ใช้ checkpoint ปกติและค่าการสุ่มตาม model card |
| **เปลี่ยนโมเดลได้ทันที** | ดึงชุดตัวเลข token `=` และ token จบบรรทัดมาจาก tokenizer โดยตรง กลไกเดียวกันจึงนำไปใช้กับ tokenizer ของโมเดลอื่นได้เลย |

### ตรวจความแม่นยำ

| ตรวจอะไร (`scripts/fail_audit.py`, `scripts/put_reject.py`) | ผล | นับยังไง |
|---|---|---|
| **ใส่ค่าผิด (ชุดทดสอบที่แช่แข็งไว้)** | **0 จาก 41,176** ครั้ง | วัดจาก trace 1,526 ชุดที่กันไว้ก่อนเขียน General Style ใส่ค่าได้ **94.4%** ของ `=` ที่อ่านได้ (30,194 / 31,996) และค่าที่โมเดลเขียนต่อตรงกับค่าที่ใส่ **31,621 จาก 32,812** ครั้ง |
| **ตัวเลขไปติดกับค่าที่ใส่** | **0** *(ก่อนมี `--no-digit-join` มี 92)* | ตรวจว่า token ถัดไปของโมเดลกลายเป็นตัวเลขต่อกันหรือไม่ นับจาก 17,485 ครั้งที่ใส่ |
| **ใส่ค่าผิด (บน A100 ของ best config)** | **0 จาก 39,046** ครั้ง | คำนวณซ้ำทุกค่าด้วย `Fraction` ของ Python โดยไม่ได้ใช้ตัวอ่านของ MIP เอง [^th-recheck] |
| **โมเดลสงสัยค่าที่ได้** | ค่าจากซอฟต์แวร์ **4.1%** vs. คิดเอง **15.2%** | นับคำอย่าง `wait`, `no,`, `wrong`, `actually`, `let me recheck` ใน 40 token ถัดไป (กรณีที่อ่านดูเป็นการสงสัยค่าที่ถูกอยู่แล้ว) |
| **loop close: token ต่อคำตอบที่ถูก** | **ลด 35%** (30,180 เหลือ 19,570) | ชุดข้อที่วนในรอบเดียวกัน เทียบระหว่างมีและไม่มีตัวตัดจบ A |
| **V7 แตะคำตอบที่ถูก** | **0 จาก 915** | วัดด้วย replay; จับ loop ได้เร็วขึ้น **212 จาก 243** เร็วกว่าเดิมรวม **624k token** |
| **cover-cap แตะคำตอบที่ถูก** | **0 จาก 1,078** | replay trace จาก A100 ทั้งหมด; หยุด loop ที่เกิน 150k ได้ **12 จาก 13** (ส่วนตัวที่ 13 จับได้ที่ 58k เมื่อป้อนเข้าตัวจับตรง ๆ) |
| **คำตอบที่เปลี่ยนเป็นถูก** | **14** | เทียบทีละคำตอบกับฝั่งใส่ค่าอย่างเดียวในรอบเดียวกัน [^th-lost] |

### ขั้นต่อไป

| แผนงาน | รายละเอียด |
|---|---|
| **รันเทียบชุดใหญ่ในรอบเดียว (#19)** | ทดสอบ best config ที่มีตัวปิด loop ครบทุกตัวบนชุดข้อสอบกว้าง เทียบกับโมเดลปกติพร้อมกันทั้งสองฝั่งในรอบเดียวกัน |
| **ตั้งเกณฑ์บนโมเดลเป้าหมาย & ผสานเข้า Decode Loop** | ตั้งเกณฑ์บน **Qwen3.8-Flash-Next** และ **Qwen3.6-35B-A3B-NVFP4** พร้อมย้ายการใส่ค่าเข้าไปใน decode loop ของ Strata [^th-next] |

### repo

| พาธ | หน้าที่ |
|---|---|
| `tools/igc/tir4b.py` | เครื่องมือทดสอบหลัก (harness): รันโมเดลใน vLLM หรือ `llama-server`, ระบบใส่ค่า, โหมดเลขคณิต และตัวปิด loop ทุกตัว (`--execute/--box` ต้องมี Strata-xeno ใน `PYTHONPATH`) |
| `tools/igc/sim_replay.py` | จำลอง replay บน trace ที่เก็บไว้เมื่อเพิ่มกลไกใหม่: ดูจุดที่ทำงาน จำนวนคำตอบที่ตัดจบ การแตะคำตอบที่ถูก และเช็กคำขอที่ vLLM จะปฏิเสธ |
| `tools/igc/mathbackend/`, `tools/igc/mathexpr.py` | ระบบคำนวณคณิตศาสตร์แบบตรงเป๊ะและตัวอ่านนิพจน์ |
| `tools/igc/question_read.py` | อ่านการคำนวณที่ตัวโจทย์ระบุไว้ |
| `tools/igc/answer_check.py` | ตัวตรวจผ่าน API คู่กับกฎตรวจคำตอบ (เก็บ key ไว้บนเครื่องที่รัน) |
| `tools/qwenputer/` | Native Qwenputer: การคำนวณที่คอมไพล์สำหรับรันใน engine |
| `scripts/` | สคริปต์ตรวจสอบและ replay gate ที่อยู่เบื้องหลังตัวเลขทั้งหมดในหน้านี้ |
| [`docs/`](docs/INDEX.md) | แคตตาล็อกเต็ม: ทุกกลไกที่สร้างหรือวัดแล้ว แยกตามหมวด (detect ใส่ค่า ปิด loop ตัวคำนวณ sandbox learned) รวมตัวที่มีเฉพาะใน engine และ benchmark ทุกชุดพร้อม setup |
| `results/arithmetic-mode/` | record ที่อยู่เบื้องหลังตัวเลขของโหมดเลขคณิต (base และ pc9) |
| [Strata-xeno](https://github.com/xenodeve/Strata-xeno) (`include/strata/core/igc_*.hpp`, `src/kernels/qwenputer.cu`) | ส่วน engine ของ MIP (สคริปต์ test จะเทียบ schema header กับ `STRATA_ROOT` ให้; เลข issue ก่อนแยก repo ดูเทียบได้ที่ #22) |

[^th-time-a]: หมายเหตุ: วินาทีต่อคำตอบ ตอนที่รันพร้อมกัน 16 คำตอบ ทั้งสองฝั่งรันบนเครื่องเดียวกัน แต่ไม่ได้จับเวลาแบบ ABBA
[^th-cap]: หมายเหตุ: โมเดลปกติรันโดยมีเพดาน 4,096 token และมี 20 คำตอบที่ชนเพดาน คำตอบที่ถูกตอนไม่มี MIP แต่ผิดตอนมี MIP มี 1 คำตอบ (`add_or_sub_big-5409`) กฎ hand-off ตั้งขึ้นระหว่างอ่าน 160 ข้อชุดเดียวกันนี้ และโมเดลปกติรันซ้ำ 2 รอบบนเครื่องนี้ได้ 114 กับ 118 ผลจึงขยับได้ไม่กี่ข้อระหว่างรอบ โหมดนี้ใช้กับโจทย์คำนวณตอนปิด thinking บน LiveBench ตัวอ่านโจทย์เจอการคำนวณที่โจทย์เขียนไว้ 0 จาก 568 ข้อ จึงใช้ได้เฉพาะเมื่อตัวโจทย์เป็นการคำนวณ
[^th-flash]: หมายเหตุ: รันแบบ greedy บนเครื่องนี้ ซึ่งให้ token ซ้ำเดิมทุกตัวไม่ได้ จึงเทียบเป็นยอดรวม ไม่ได้เทียบทีละข้อ รอบนี้ใช้ engine รุ่นก่อนแก้ parser อีก 2 จุด (binding token กับตัวอักษรที่ติดกัน) บน GSM8K token เท่าเดิม แต่เวลาต่อ token สูงขึ้นประมาณ 5% เมื่อเปิด MIP ชุดเหล่านี้ไม่มีความแม่นยำที่เพิ่มขึ้น เพราะโมเดลคิดเลขเองถูกอยู่แล้ว ที่ขยับคือ token กับเวลา
[^th-tool]: หมายเหตุ: ข้อละ 1 คำตอบแบบสุ่ม ฝั่งไม่เปิดรันบน A100 บัญชีหนึ่ง ฝั่งเปิดรันอีกบัญชีหนึ่ง HLE (10 ข้อ) ได้ 4 / 4 ที่ 1.05 เท่า การเรียก tool เกือบทั้งหมดเกิดหลังการคิดจบแล้ว token ที่ลดได้ส่วนใหญ่จึงมาจากบรรทัด prompt ของ tool route มากกว่าจากการเรียก tool
[^th-detect]: หมายเหตุ: ตัวตรวจจับเหล่านี้แค่อ่าน ยังไม่ได้ลงมือทำอะไร การใส่ค่ายังรอ `=` และการ parse ที่ตรงเป๊ะ probe ต้องใช้ hidden state ซึ่ง engine มี แต่ `llama-server` กับ vLLM ไม่เปิดให้อ่าน
[^th-cover]: หมายเหตุ: `cover-cap` วัดจากการ replay trace ที่เก็บจาก A100 ส่วนรอบที่รันบน A100 เองคือรอบถัดไป (#19)
[^th-pairing]: หมายเหตุ: best config กับฝั่งใส่ค่าอย่างเดียวรันในรอบเดียวกัน ข้อเดียวกัน seed เดียวกัน ส่วนโมเดลปกติมาจากอีกรอบ เนื่องจาก vLLM ให้คำตอบซ้ำเดิมข้ามรอบไม่ได้ (#21) ต่างกัน 1–2 คำตอบจึงอยู่ในช่วงที่ผลแต่ละรอบแกว่งได้
[^th-tokens]: หมายเหตุ: token รวมทุกคำตอบ โมเดลปกติ 631k ใส่ค่าอย่างเดียว 754k best config 936k เพราะมีคำตอบที่คิดจนจบมากขึ้น และคำตอบที่วิ่งจนเต็ม context คำตอบเดียวกินไป 210k–262k ส่วนเลขคณิต มัธยฐานของคำตอบที่ถูกอยู่ที่ 8,371 token (best config) เทียบกับ 6,472 token เพราะมีอีก 10 คำตอบที่ตอนนี้คิดจนจบได้ ในเครื่องมือทดสอบ best config ได้ 25.7 token/s ต่อคำตอบ ส่วนโมเดลปกติได้ 37.9 เพราะ `tir4b` ต้องจบ request ทุกครั้งที่เจอ `=` แล้วเริ่ม request ใหม่หลังใส่ค่า ใน engine ของ Strata การใส่ค่าเกิดอยู่ใน decode loop (`include/strata/core/igc_inline.hpp`) และยังไม่ได้จับเวลา
[^th-ordinary]: หมายเหตุ: ประเภทอื่นในการเทียบชุดเดียวกัน คูณหารต่อกัน 1.01 นิพจน์ผสมยาว 1.13 (โมเดลตรวจค่าที่ได้รับซ้ำ) MATH-500 ระดับ 5 0.98 รวมทั้ง 170 คู่ 0.97 (ใช้ token น้อยลง 93 คู่) คะแนนในรอบนั้น แบบใส่ค่าอย่างเดียวไม่มีตัวปิด loop ได้ 197 จาก 240 ส่วนโมเดลปกติได้ 209 ทุกข้อที่ผิดในทั้งสองฝั่งเกิดจากการวน ซึ่งเป็นสิ่งที่ตัวปิด loop มีไว้แก้ และตอนเปิด thinking ยังไม่มีกลไกไหนที่เพิ่มความแม่นยำบนชุดกว้างได้ด้วยตัวเอง สิ่งที่ขยับคือ token
[^th-recheck]: หมายเหตุ: อีก 1,208 ครั้ง (ราก mod ฟังก์ชัน) อยู่นอกขอบเขตที่วิธีตรวจซ้ำนี้คำนวณได้
[^th-lost]: หมายเหตุ: มี 2 คำตอบที่ถูกในฝั่งใส่ค่าอย่างเดียว แต่ผิดเมื่อใช้ best config ทั้งสองเป็นโจทย์ MATH ระดับ 5 ที่ V7 ปิดการคิดที่ 8,433 และ 6,582 token ส่วนตอนไม่มีตัวปิด loop โมเดลคิดจนตอบถูกที่ 42,974 และ 31,518 token การปิดที่จบด้วยคำตอบผิด: A 8 จาก 21, V7 6 จาก 17, B 6 จาก 6 (B ทำงานเฉพาะกับคำตอบที่วนอยู่แล้ว)
[^th-next]: หมายเหตุ: เรื่องที่ยังเปิดอยู่อีก 2 เรื่อง คือ ช่วงของ V7 ที่ยาวขึ้นสำหรับโจทย์ MATH ที่ต้องคิดยาว และคำตอบหลังการปิด loop (โมเดลอาจกลับไปหารต่อ)
