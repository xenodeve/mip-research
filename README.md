# mip-research

**MIP (Model Inline Processing): software computes, the model reasons.** While a model thinks, MIP reads the expression it writes
before `=`, computes it exactly and puts the value after the `=`, and it closes the loops a small model cannot leave, so the model
still answers. No weight is changed.

This repo holds the mechanisms, the harness and the measurements. [Strata-xeno](https://github.com/xenodeve/Strata-xeno) runs MIP in
its engine and server for Qwen3.8-Flash-Next and Qwen3.6-35B-A3B-NVFP4. Everything below is the current **General Style** (forms true
under every reading, no per-model profile), measured on **Qwen3.5-4B** (bf16, vLLM 0.33, Colab A100 80 GB, T 0.6, top-p 0.95,
top-k 20, the full 262,144-token context).

## At a glance

- **Arithmetic: 95 % fewer tokens and more right answers.** When software takes the whole calculation, the 4B gets **156 of 160**
  DeepMind arithmetic questions right against 118, with **0.05x the tokens** (261,731 -> 12,062) and answers 85 % faster
  ([Arithmetic mode](#arithmetic-mode-95--fewer-tokens)).
- **+13 right answers where a 4B model gets stuck.** On 24 questions where the plain model loops: 24 -> **37 of 48** right, and
  answers with no answer at all fall from 21 to **1**.
- **No fine-tuning, no second model.** The plain checkpoint, its own sampling; MIP only reads the text and adds tokens to it.
- **Never a wrong number.** **0 wrong values in 39,046** puts, each re-checked by an independent evaluator. The math is exact
  (387 operations, no floating point) and refuses rather than guesses.
- **Ends runaway thinking.** Loops that ran 200k-262k tokens to the end of the context are stopped at 40k-114k; right answers on
  MATH level 5 are **40 % shorter**.
- **Fewer tokens where the calculation is fixed.** On answers right both with and without MIP: polynomial evaluation **-33 %**,
  surds -29 %, unit conversion and sequences -11 %; over all ordinary math the saving is small (-3 %) ([Ordinary math](#ordinary-math-answers-right-in-both-arms)).
- **Stays out of the way.** It acts only after an `=` the model wrote, only on exact arithmetic, and every loop threshold sits above
  all right answers: the newest closer touched **0 of 1,078** right answers in replay.
- **Failures are counted, not hidden:** 2 of 48 right answers were lost to a close that came too early ([Failures](#failures)).

Contents: [Arithmetic mode](#arithmetic-mode-95--fewer-tokens) · [Best config](#best-config) · [Results](#results) · [Tokens and time](#tokens-and-time) · [Ordinary math](#ordinary-math-answers-right-in-both-arms) · [Detection points](#detection-points) ·
[How a value is put](#how-a-value-is-put) · [Loops it prevents](#loops-it-prevents) · [MathBackend](#mathbackend) ·
[Staying out of the way](#staying-out-of-the-way) · [No fine-tuning](#no-fine-tuning) · [Failures](#failures) ·
[Not measured yet](#not-measured-yet) · [Repo](#repo) · [ภาษาไทย](#ภาษาไทย)

## Arithmetic mode: 95 % fewer tokens

For a question whose work is one fixed calculation, MIP lets software take the whole step instead of each `=`:

```sh
python tools/igc/tir4b.py run --model <Qwen3.5-4B> --server http://127.0.0.1:8099 --rows dm-rows.jsonl --no-think --greedy \
  --system "Do every arithmetic step exactly." --inline-calc --inline-plain --whole-hint --whole-plain --question-value --hand-off
```

1. **The whole step.** The reply opens asking for the whole calculation as one plain expression in the question's order
   (`--whole-hint --whole-plain`); software computes it exactly, sign and uneven quotient included.
2. **The question's own computation.** Software also reads the computation from the question itself, in symbols or in words
   (`less than`, `take away`, `product of` ...): on these 160 questions it reads all 160 exactly, so a number the model copies wrong
   is caught (`--question-value`).
3. **The hand-off.** A line that only restates software's value (`= v`, or a long decimal expansion) is cut, and the model is led
   to its answer with `So the final answer is \boxed{` (`--hand-off`). Software never writes the answer; the model does.

DeepMind arithmetic, Qwenputer's 160 questions (40 each of big addition and subtraction, big division, big multiplication, long
multiply-divide chains), Qwen3.5-4B, thinking off, greedy, llama-server on an RTX 5060 Ti, one answer each:

| | Plain model | Arithmetic mode |
|---|---|---|
| Right | 118 / 160 | **156 / 160** |
| Tokens, all answers | 261,731 | **12,062 (0.046x)** |
| Median tokens per answer | 1,191 | **38** |
| Median seconds per answer | 61 | **8** |

| Kind | Right, plain -> MIP | Tokens with MIP / without |
|---|---|---|
| Big multiplication | 21 -> **40** | **0.018** |
| Big addition and subtraction | 33 -> 39 | 0.044 |
| Long multiply-divide chains | 35 -> **40** | 0.047 |
| Big division | 29 -> 37 | 0.072 |

- **On the 117 answers right in both arms: 0.042x the tokens (-96 %) and 0.15x the seconds (-85 %),** the same answer written
  without grinding the digits by hand. Seconds are per answer with 16 answers in flight, both arms on the same PC, not an ABBA.
- **39 answers gained, 1 lost** (`add_or_sub_big-5409`).
- The plain model has a 4,096-token cap here, and 20 of its 160 answers reach it; those count as wrong, which favours the MIP arm
  on accuracy. The token cut on answers right in both arms does not depend on the cap.
- This mode answers a calculation question with thinking off. The best config below is for reasoning with thinking on, where the
  calculations sit inside a longer argument.

## Best config

```sh
python tools/igc/tir4b.py run --model <Qwen3.5-4B> --rows rows.jsonl --samples 2 --seed 1 --ids --out out/arm \
  --inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress \
  --loop-close --answer-close --early-cycle --shape-cap 1024 --cover-cap 32768
```

| Part | Flags | Measured on |
|---|---|---|
| Inline values (General Style) | `--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress` | A100 |
| A: loop close | `--loop-close` | A100 |
| B: answer close | `--answer-close` | A100 |
| V7: early cycle | `--early-cycle` | A100 |
| Shape cap | `--shape-cap 1024` | A100 (3 questions) |
| Cover cap | `--cover-cap 32768` | replay of stored A100 traces |

## Results

24 questions where the plain model loops (48 answers, 2 samples each): 17 DeepMind arithmetic questions (34 answers: sequences,
surds, big divisions, mixed expressions, unit conversion) and 7 MATH-500 level-5 problems (14 answers). Right answers are scored
by the rule, with an API reader beside it and every disagreement read by hand.

| Arm | Arithmetic | MATH L5 | All | No answer (cut) |
|---|---|---|---|---|
| Plain model, no MIP | 17/34 | 7/14 | 24/48 | 21 |
| Inline values only | 17/34 | 8/14 | 25/48 | 23 |
| **Best config** (inline + A + B + V7) | **27/34** | **10/14** | **37/48** | **1** |

**+13 right answers over the plain model (24 -> 37 of 48)** and +12 over inline values alone, and the answers that ended with
no answer fall from 21 to 1. On the three hardest of these questions with shape cap added (12 answers): 8 right and none run to the
end of the context. The best config and the inline-only arm ran in one round on the same questions and seeds; the plain model is
another run. vLLM does not repeat an answer across runs (#21), so a difference of 1-2 answers is noise.

## Tokens and time

| Arm | Tokens, all answers | Median tokens per answer | Median tokens of a right answer |
|---|---|---|---|
| Plain model | 631k | 10,096 | 7,144 |
| Inline values only | 754k | 11,360 | 10,221 |
| **Best config** | 936k | 10,961 | 8,829 |

By kind, the best config's median right answer is 8,371 tokens on arithmetic and 9,090 on MATH L5 (plain model 6,472 and 15,260):
**on MATH the right answers are 40 % shorter**; on arithmetic they are longer, because 10 more arithmetic answers now finish where
the plain model looped and was cut.

- **The total rises because more answers finish.** The best config turns 20 cut answers into finished ones, and a finished answer
  spends tokens on its answer. One answer that runs to the context end costs 210-262k tokens, a quarter of an arm's total, so totals
  move with those few answers.
- **What the closers save is the run to the end.** Shape cap closed a long division at 55k tokens where such loops run past 200k; in
  replay, cover cap stops every loop that ran past 150k at 40k-114k.
- **Time.** In this harness the best config ran at 25.7 tokens/s per answer against 37.9 for the plain model (vLLM, many answers in
  one batch). That is the harness: `tir4b` ends a vLLM request at every `=` and starts the next after the put, about 20,000 extra
  requests an arm. In Strata's engine the put happens inside the decode loop (`include/strata/core/igc_inline.hpp`); its time there
  is not measured yet.

## Ordinary math: answers right in both arms

120 ordinary questions (DeepMind arithmetic and MATH-500 level 5), 2 answers each, inline values against the plain model in one
round with the same seeds. Only the 170 answer pairs right in both arms are counted, so the tokens compare the same right answer
written with and without software's values. The closers do not act on these answers.

| Kind of question | Pairs | Tokens with MIP / without |
|---|---|---|
| Polynomial evaluation | 17 | **0.67** |
| Surds | 3 | 0.71 |
| Unit conversion | 18 | 0.89 |
| Sequences | 12 | 0.89 |
| Big divisions | 8 | 0.96 |
| Division with remainder | 18 | 0.97 |
| Multiply-divide chains | 18 | 1.01 |
| Long mixed expressions | 17 | 1.13 |
| MATH-500 level 5 | 59 | 0.98 |
| **All** | **170** | **0.97** (median per pair 0.96; fewer tokens in 93 of 170) |

Where the work is a fixed calculation the model would otherwise grind by hand, the right answer is up to a third shorter. Over
all ordinary math the saving is small, and on long mixed expressions the answer is longer: the model still checks a value it was
given. Time in this harness is not comparable (the harness pauses at every `=`, [Tokens and time](#tokens-and-time)).

## Detection points

MIP reads the model's own text at five points. The prompt, the sampling and the weights stay the model's own.

| # | Where | When | What it may do |
|---|---|---|---|
| 1 | The token `=` | every request stops at an `=` token, in the thinking and in the answer, so MIP reads before the model's next token | put the exact value |
| 2 | The token after a put that ends in a digit (and a `.` right after it) | one token | hold digit tokens down, so the model's next digit cannot glue onto the put (`--no-digit-join`) |
| 3 | Every line end in the thinking | as tokens arrive | V7: close the thinking on a short exact cycle |
| 4 | The thinking since the last close, or the whole answer | every 1,024 tokens and at each request end | the repeat check, shape cap and cover cap: close the thinking (A) or the answer (B) |
| 5 | The answer after B's cue | to a line end or 64 tokens | nothing: one value is read and the answer ends |

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
closed in the best config, 13 then answered right.

## MathBackend

`tools/igc/mathbackend/` is a registry of **387 operations in six modules**: numbers 100 (gcd, lcm, phi, modular inverse, digits,
divisors, primes), geometry 86, algebra 70, counting 70 (C, P, Catalan, derangements), search 43, symbolic 18.

- **Exact only.** Python's standard library: `Fraction` for rationals, `Surd` for `p + q*sqrt(d)`, `Decimal` where a decimal is asked
  for. No floating point stands behind a put; a root or power that is not exact is refused, never rounded.
- **Bounded.** Each operation declares its bound (`@op(name, kind, bound=...)`). Past it, or outside its domain, it raises
  `NotCovered` or `BadInput`, and the caller must not guess: no put.
- **Checked against independent libraries.** Each module has a `sim_*.py` that draws inputs and compares every operation with SymPy,
  mpmath or NetworkX, which the backend itself never imports (`tools/test_mathbackend_*.py`).

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
another model's tokenizer; the thresholds are checked again on each model.

## Failures

The best config's A100 records (`loop-lcacv7`, 48 answers, and `loop-shape` with shape cap, 12 answers), counted with
`scripts/fail_audit.py` and `scripts/put_reject.py`:

| Failure | Count | How it is counted |
|---|---|---|
| A wrong value put | **0 of 39,046** puts re-checked | every put re-computed by Python's `Fraction` on a normalised copy, independent of MIP's reader; 1,208 more (roots, mod, functions) cannot be re-checked this way |
| A put the model pushed away | **4.1 %** (832 of 20,490) | a rejection word (`wait`, `no,`, `wrong`, `actually`, `let me recheck`) within the next 40 tokens. After the model's **own** `= value` with no MIP: 15.2 %. The cases read are the model doubting a right value (`10856 / 5192 = 23/11? No, wait.`) |
| A right answer made wrong | **2 of 48** (14 wrong made right) | answer by answer against the inline-only arm of the same round, same questions and seeds. Both are MATH L5 problems where **V7 closed the thinking too early** (8,433 and 6,582 tokens); without the closers the model finished right at 42,974 and 31,518 |
| A close on an answer that then ended wrong | A: 8 of 21; V7: 6 of 17; B: 6 of 6 | every act and how its answer ended. B acts only on an answer already looping, so it turns lost answers into one value |
| Cover cap acting on a right answer | **0 of 1,078** right answers in replay | every stored A100 trace replayed with cover cap; it stops 12 of the 13 loops past 150k tokens, and the 13th fires at 58k when fed directly |

## Not measured yet

- **The best config with every closer on a broad set, against the plain model** (#19): 120 questions x 2, both arms in one round;
  the scripts are ready and wait for A100 compute units.
- **V7's early closes** on long MATH derivations: whether a longer span keeps what V7 catches without the 2 lost answers.
- **After a close the model can still be wrong** (it can go back to dividing in the answer).
- **The thresholds on the target models** (Qwen3.8-Flash-Next, Qwen3.6-35B-A3B-NVFP4) and the put's time inside Strata's engine.

## Repo

| Path | What |
|---|---|
| `tools/igc/tir4b.py` | the harness: the model in vLLM, inline values and every closer |
| `tools/igc/sim_replay.py` | replays stored traces with a mechanism added: where it acts, on how many cut answers, whether on a right one, what vLLM would refuse |
| `tools/igc/mathbackend/`, `tools/igc/mathexpr.py` | the exact computation and the reader |
| `tools/igc/answer_check.py` | the API reader beside the rule (its key stays on the machine that runs it) |
| `tools/qwenputer/` | Native Qwenputer: computation compiled for the engine |
| `scripts/` | the audits and the replay gate behind the numbers here |

The engine side of MIP lives in Strata-xeno (`include/strata/core/igc_*.hpp`, `src/kernels/qwenputer.cu`);
`tools/test_igc_adapter_artifact.py` checks the schema header against a Strata-xeno checkout named by `STRATA_ROOT`.
`tir4b --execute/--box` needs Strata-xeno on `PYTHONPATH`. Issue numbers from before the split are mapped in #22.

```sh
python -m unittest discover -s tools -p "test_*.py" -t .     # 923 tests; 78 skip without a GPU, a model or STRATA_ROOT
```

---

## ภาษาไทย

**MIP (Model Inline Processing): ให้ software คำนวณ ให้ model ใช้เหตุผล** ระหว่างที่ model คิด MIP จะอ่านนิพจน์ที่ model เขียนก่อน `=`
คำนวณให้ตรงเป๊ะ แล้วใส่ค่าต่อท้าย `=` และยังปิด loop ที่ model เล็กออกเองไม่ได้ ให้ model ยังตอบออกมาได้ โดยไม่แตะน้ำหนักของ model เลย

repo นี้เก็บกลไก เครื่องมือทดลอง และผลวัด ส่วน [Strata-xeno](https://github.com/xenodeve/Strata-xeno) เป็นที่ที่เอา MIP ไปรันจริงใน engine กับ server
ให้ Qwen3.8-Flash-Next และ Qwen3.6-35B-A3B-NVFP4 เนื้อหาทั้งหมดในนี้เป็น **General Style** ตัวล่าสุด คือรูปแบบค่าที่ถูกไม่ว่าจะอ่านแบบไหน ไม่มี profile แยกตาม model
วัดบน **Qwen3.5-4B** (bf16, vLLM 0.33, Colab A100 80 GB, T 0.6, top-p 0.95, top-k 20, context เต็ม 262,144 token)

### สรุปสั้น ๆ

- **เลขคณิต: token ลดลง 95% และตอบถูกมากขึ้น:** พอให้ software คำนวณทั้งขั้น model 4B ตอบถูก **156 จาก 160** ข้อเลขคณิต DeepMind จากเดิม 118 ใช้ token **0.05 เท่า** (261,731 เหลือ 12,062) และตอบเร็วขึ้น 85%
- **ตอบถูกเพิ่ม 13 คำตอบ ในโจทย์ที่ model 4B วนติด:** 24 ข้อที่ model เปล่าวน ตอบถูกจาก 24 เป็น **37 จาก 48** และคำตอบที่ไม่มีคำตอบเลยลดจาก 21 เหลือ **1**
- **ไม่ต้อง fine-tune ไม่ต้องใช้ model ตัวที่สอง:** ใช้ checkpoint ปกติกับค่าการสุ่มของมันเอง MIP แค่อ่านข้อความแล้วต่อ token ลงไป
- **ไม่เคยใส่เลขผิด:** ใส่ค่า 39,046 ครั้ง **ผิด 0 ครั้ง** ตรวจซ้ำทุกค่าด้วยตัวคำนวณอิสระ เพราะคำนวณแบบตรงเป๊ะ (387 operation ไม่มี floating point) และถ้าไม่แน่ใจจะไม่ใส่ ไม่เดา
- **หยุดการคิดที่วนไม่จบ:** loop ที่เคยวิ่งไป 200k-262k token จนเต็ม context ถูกหยุดได้ที่ 40k-114k และคำตอบที่ถูกบน MATH ระดับ 5 **สั้นลง 40%**
- **ใช้ token น้อยลงในโจทย์ที่คำนวณตายตัว:** นับเฉพาะข้อที่ตอบถูกทั้งตอนมีและไม่มี MIP แทนค่าในพหุนามลด **33%** จัดรูปรากลด 29% แปลงหน่วยกับลำดับลด 11% แต่ถ้ารวมโจทย์คณิตทั่วไปทั้งหมด ลดได้แค่ 3%
- **ไม่ไปยุ่งกับงานอื่น:** ใส่ค่าเฉพาะหลัง `=` ที่ model เขียนเอง และเฉพาะเลขคณิตที่คำนวณได้ตรงเป๊ะ ส่วนเกณฑ์ของตัวปิด loop ตั้งไว้สูงกว่าคำตอบที่ถูกทุกตัว ตัวปิด loop ตัวล่าสุดแตะคำตอบที่ถูก **0 จาก 1,078** ใน replay
- **นับความผิดพลาดไว้ครบ ไม่ซ่อน:** มีคำตอบที่ถูกเสียไป 2 จาก 48 เพราะปิดการคิดเร็วเกินไป (ดูหัวข้อความผิดพลาด)

### โหมดเลขคณิต: token ลดลง 95%

ถ้าโจทย์ทั้งข้อเป็นการคำนวณแบบตายตัว MIP จะให้ software รับไปทั้งขั้น แทนที่จะใส่ค่าทีละ `=` (ดูคำสั่งในหัวข้อ Arithmetic mode ข้างบน)
1. **ทั้งขั้น:** ตอนเริ่มตอบ ให้ model เขียนการคำนวณทั้งหมดเป็นนิพจน์เดียว เรียงตามลำดับของโจทย์ แล้ว software คำนวณให้ตรงเป๊ะ ทั้งเครื่องหมายและการหารที่ไม่ลงตัว
2. **การคำนวณที่โจทย์เขียนมาเอง:** software อ่านการคำนวณจากตัวโจทย์ด้วย ไม่ว่าจะเขียนเป็นสัญลักษณ์หรือเป็นคำ (`less than`, `take away`, `product of` ...)
   ใน 160 ข้อนี้อ่านถูกครบทั้ง 160 ถ้า model ลอกตัวเลขจากโจทย์ผิดก็จับได้
3. **การส่งต่อ (hand-off):** บรรทัดที่แค่พูดค่าของ software ซ้ำ (`= v` หรือทศนิยมยาว ๆ) จะถูกตัด แล้วพา model ไปที่ `So the final answer is \boxed{`
   software ไม่ได้เขียนคำตอบให้ model เป็นคนเขียนเอง

ทดสอบบนเลขคณิต DeepMind 160 ข้อของ Qwenputer มี 4 ประเภท ประเภทละ 40 ข้อ คือ บวกลบเลขใหญ่ หารเลขใหญ่ คูณเลขใหญ่ และคูณหารต่อกันยาว ๆ
ใช้ Qwen3.5-4B ปิด thinking, greedy, llama-server บน RTX 5060 Ti ข้อละ 1 คำตอบ

| | model เปล่า | โหมดเลขคณิต |
|---|---|---|
| ตอบถูก | 118 / 160 | **156 / 160** |
| token รวม | 261,731 | **12,062 (0.046 เท่า)** |
| มัธยฐาน token ต่อคำตอบ | 1,191 | **38** |
| มัธยฐานวินาทีต่อคำตอบ | 61 | **8** |

| ประเภท | ตอบถูก เปล่า -> MIP | token มี MIP / ไม่มี |
|---|---|---|
| คูณเลขใหญ่ | 21 -> **40** | **0.018** |
| บวกลบเลขใหญ่ | 33 -> 39 | 0.044 |
| คูณหารต่อกันยาว ๆ | 35 -> **40** | 0.047 |
| หารเลขใหญ่ | 29 -> 37 | 0.072 |

- **117 คำตอบที่ถูกทั้งสองฝั่ง: token เหลือ 0.042 เท่า (ลด 96%) และเวลาเหลือ 0.15 เท่า (ลด 85%)** เป็นคำตอบเดียวกัน แค่ไม่ต้องไล่คิดตัวเลขเองทีละหลัก
  เวลาเป็นวินาทีต่อคำตอบ ตอนที่รันพร้อมกัน 16 คำตอบ ทั้งสองฝั่งรันบนเครื่องเดียวกัน แต่ไม่ได้ทำ ABBA
- **ตอบถูกเพิ่ม 39 ข้อ เสีย 1 ข้อ** (`add_or_sub_big-5409`)
- **ข้อควรรู้เรื่องเพดาน token:** model เปล่าโดนจำกัดที่ 4,096 token มี 20 จาก 160 คำตอบที่ชนเพดาน และถูกนับเป็นผิด ข้อนี้ทำให้ฝั่ง MIP ได้เปรียบเรื่องคะแนน
  แต่ตัวเลข token ที่ลดลงในข้อที่ถูกทั้งสองฝั่ง ไม่ได้ขึ้นกับเพดานนี้
- **โหมดนี้ใช้กับโจทย์คำนวณตอนปิด thinking:** ส่วน best config ข้างล่างใช้ตอนเปิด thinking ซึ่งการคำนวณแทรกอยู่ในเหตุผลยาว ๆ

### best config

ใช้ flag ตามคำสั่งข้างบน ประกอบด้วย 6 ส่วน
- **ตัวเขียนค่า (General Style):** วัดบน A100 แล้ว
- **A: loop close:** วัดบน A100 แล้ว
- **B: answer close:** วัดบน A100 แล้ว
- **V7: early cycle:** วัดบน A100 แล้ว
- **shape-cap:** วัดบน A100 แล้ว 3 ข้อ
- **cover-cap:** วัดจาก replay ของ trace ที่เก็บจาก A100

### ผลการทดสอบ

ใช้ 24 ข้อที่ model เปล่าวนติด ข้อละ 2 คำตอบ รวม 48 คำตอบ แบ่งเป็น
- **เลขคณิต DeepMind 17 ข้อ (34 คำตอบ):** ลำดับ ราก หารเลขใหญ่ นิพจน์ผสม แปลงหน่วย
- **MATH-500 ระดับ 5 อีก 7 ข้อ (14 คำตอบ)**

ตรวจคำตอบด้วยกฎ มี API reader ตรวจคู่กัน และข้อไหนที่สองฝั่งให้ผลไม่ตรงกันก็อ่านเองทุกข้อ

| แขน | เลขคณิต | MATH L5 | รวม | ไม่มีคำตอบ (ถูกตัด) |
|---|---|---|---|---|
| model เปล่า ไม่มี MIP | 17/34 | 7/14 | 24/48 | 21 |
| ใส่ค่าอย่างเดียว | 17/34 | 8/14 | 25/48 | 23 |
| **best config** (ใส่ค่า + A + B + V7) | **27/34** | **10/14** | **37/48** | **1** |

- **ตอบถูกเพิ่ม 13 คำตอบจาก model เปล่า (24 เป็น 37 จาก 48)** และเพิ่ม 12 คำตอบจากแบบใส่ค่าอย่างเดียว
- **คำตอบที่จบแบบไม่มีคำตอบ** ลดจาก 21 เหลือ 1
- **3 ข้อที่ยากที่สุดเมื่อเพิ่ม shape-cap (12 คำตอบ):** ถูก 8 และไม่มีคำตอบไหนวิ่งจนเต็ม context
- **เงื่อนไขการเทียบ:** best config กับแขนใส่ค่าอย่างเดียวรันในรอบเดียวกัน ข้อเดียวกัน seed เดียวกัน ส่วน model เปล่ามาจากอีกรอบ และ vLLM ให้คำตอบซ้ำเดิมข้ามรอบไม่ได้ (#21) ต่างกัน 1-2 คำตอบจึงถือเป็น noise

### token และเวลา

| แขน | token รวม | มัธยฐานต่อคำตอบ | มัธยฐานของคำตอบที่ถูก |
|---|---|---|---|
| model เปล่า | 631k | 10,096 | 7,144 |
| ใส่ค่าอย่างเดียว | 754k | 11,360 | 10,221 |
| **best config** | 936k | 10,961 | 8,829 |

**คำตอบที่ถูกของ best config แยกตามประเภท (มัธยฐาน)**
- **MATH:** 9,090 token เทียบกับ model เปล่า 15,260 สั้นลง 40%
- **เลขคณิต:** 8,371 token เทียบกับ 6,472 ยาวขึ้น เพราะมีคำตอบเลขคณิตเพิ่มมาอีก 10 คำตอบที่ตอนนี้คิดจนจบได้ ส่วน model เปล่าวนจนถูกตัด

- **token รวมเพิ่มขึ้นเพราะคำตอบจบมากขึ้น:** best config ทำให้คำตอบที่เคยถูกตัด 20 คำตอบคิดจนจบได้ และคำตอบที่จบต้องใช้ token เขียนคำตอบด้วย อีกเรื่องคือคำตอบที่วิ่งจนเต็ม context คำตอบเดียวกินไป 210-262k ประมาณหนึ่งในสี่ของทั้งแขน ยอดรวมเลยขยับตามคำตอบแบบนี้ไม่กี่คำตอบ
- **ที่ประหยัดได้จริงคือไม่ต้องวิ่งจนเต็ม context:** shape-cap ปิดการหารยาวได้ที่ 55k แทนที่จะเกิน 200k ส่วนใน replay cover-cap หยุด loop ที่เคยเกิน 150k ได้ทุกตัว ที่ 40k-114k
- **เวลา:** ในเครื่องมือทดสอบนี้ best config ได้ 25.7 token/s ต่อคำตอบ ส่วน model เปล่าได้ 37.9 (vLLM รันหลายคำตอบพร้อมกันใน batch)
  - สาเหตุมาจากเครื่องมือทดสอบ: `tir4b` ต้องจบ request ทุกครั้งที่เจอ `=` แล้วเริ่ม request ใหม่หลังใส่ค่า แขนหนึ่งมี request เพิ่มประมาณ 20,000 ครั้ง
  - ใน engine ของ Strata การใส่ค่าเกิดอยู่ใน decode loop เลย (`include/strata/core/igc_inline.hpp`) แต่ยังไม่ได้วัดเวลาตรงนั้น

### โจทย์คณิตทั่วไป: ข้อที่ตอบถูกทั้งสองฝั่ง

ใช้โจทย์คณิตทั่วไป 120 ข้อ (เลขคณิต DeepMind และ MATH-500 ระดับ 5) ข้อละ 2 คำตอบ เทียบแบบใส่ค่ากับ model เปล่าในรอบเดียวกัน seed เดียวกัน
นับเฉพาะ 170 คู่ที่ตอบถูกทั้งสองฝั่ง จึงเป็นการเทียบคำตอบที่ถูกคำตอบเดียวกัน ระหว่างตอนที่มีค่าจาก software กับตอนที่ไม่มี และตัวปิด loop ไม่ได้ทำงานกับคำตอบชุดนี้

| ประเภทโจทย์ | คู่ | token มี MIP / ไม่มี |
|---|---|---|
| แทนค่าในพหุนาม | 17 | **0.67** |
| จัดรูปราก | 3 | 0.71 |
| แปลงหน่วย | 18 | 0.89 |
| ลำดับ | 12 | 0.89 |
| หารเลขใหญ่ | 8 | 0.96 |
| หารมีเศษ | 18 | 0.97 |
| คูณหารต่อกัน | 18 | 1.01 |
| นิพจน์ผสมยาว | 17 | 1.13 |
| MATH-500 ระดับ 5 | 59 | 0.98 |
| **รวม** | **170** | **0.97** (มัธยฐานต่อคู่ 0.96 ใช้ token น้อยลง 93 จาก 170 คู่) |

- **ลดได้ชัดในโจทย์ที่คำนวณตายตัว:** ถ้าไม่มี MIP model ต้องคิดเลขพวกนี้ด้วยมือเอง พอมีค่าให้ คำตอบที่ถูกสั้นลงได้ถึงหนึ่งในสาม
- **ภาพรวมลดได้น้อย:** ถ้ารวมโจทย์คณิตทั่วไปทั้งหมด ลดลงไม่มาก
- **นิพจน์ผสมยาวกลับยาวขึ้น:** model ยังไปตรวจค่าที่ได้รับซ้ำอีกรอบ
- **เวลาในชุดนี้เทียบกันไม่ได้:** เครื่องมือทดสอบหยุดทุกครั้งที่เจอ `=`

### จุดที่ตรวจจับ

MIP อ่านข้อความที่ model เขียนเองอยู่ 5 จุด prompt, การสุ่ม และน้ำหนักของ model ยังเหมือนเดิมทั้งหมด

1. **token `=`:** ทุก request หยุดที่ `=` ทั้งตอนคิดและตอนตอบ MIP จึงได้อ่านก่อนที่ model จะเขียน token ถัดไป จุดนี้ใส่ค่าที่ตรงเป๊ะ
2. **token ถัดจากค่าที่ใส่ ถ้าค่านั้นลงท้ายด้วยตัวเลข** (รวม `.` ที่ตามมาทันที): กดตัวเลขไว้ 1 token ตัวเลขที่ model เขียนต่อจะได้ไม่ไปติดกับค่าที่ใส่
3. **ทุกครั้งที่จบบรรทัดตอนคิด:** V7 ปิดการคิดเมื่อเจอ cycle สั้นที่ซ้ำตรงตัว
4. **การคิดทั้งหมดตั้งแต่ครั้งล่าสุดที่ปิด หรือส่วนคำตอบทั้งหมด:** ตรวจทุก 1,024 token และทุกครั้งที่ request จบ ด้วยตัวตรวจการวน shape-cap และ cover-cap ถ้าเจอก็ปิดการคิด (A) หรือปิดคำตอบ (B)
5. **ส่วนคำตอบหลังประโยคบังคับตอบของ B:** ไม่ทำอะไรเพิ่ม อ่านแค่ค่าเดียวแล้วจบคำตอบ

### ใส่ค่ายังไง

1. ตัดข้อความก่อน `=` ให้เหลือนิพจน์เลขที่ยาวที่สุดที่จบตรงนั้น อ่านได้ทุกรูปแบบที่ model เขียน (ASCII, LaTeX ซ้อนกี่ชั้นก็ได้, Unicode และการเรียกฟังก์ชันอย่าง `\gcd(12, 18)`, `\binom{7}{3}`)
2. MathBackend คำนวณให้ตรงเป๊ะ
3. เขียนค่าในรูปแบบ General Style แล้วเช็กกับเงื่อนไขที่ไม่ให้ใส่
4. ต่อ ` value` ท้ายข้อความของ model เป็นข้อความธรรมดา แล้ว request ถัดไปก็เดินต่อจากตรงนั้น สำหรับ model ค่าที่ใส่ให้ก็คือคำถัดไปบนหน้ากระดาษ ถ้าขั้นไหนบอกว่าไม่ใส่ ก็ไม่ใส่อะไรเลย model จะเขียนค่าเอง

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

loop ในส่วนคิดจะถูกปิดด้วย A คือใส่ `</think>` แล้วให้ model ตอบจากสิ่งที่คิดมาแล้ว ใน best config A ปิดไป 21 ครั้ง และตอบถูก 13 ครั้ง

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
- **มีขอบเขตชัด:** ทุก operation ประกาศขอบเขตของตัวเอง ถ้าเกินขอบเขตหรืออยู่นอกโดเมน จะโยน `NotCovered` หรือ `BadInput` แล้วห้ามเดา ผลคือไม่ใส่ค่า
- **ตรวจกับ library อิสระ:** ทุก module มี `sim_*.py` ที่สุ่มค่าเข้าไป แล้วเทียบทุก operation กับ SymPy, mpmath หรือ NetworkX ซึ่งตัว backend เองไม่ได้ import

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
ใช้ค่าการสุ่มตาม model card ส่วนชุดตัวเลข token `=` และ token จบบรรทัด ได้มาจาก tokenizer กลไกเดียวกันจึงใช้กับ tokenizer ของ model อื่นได้ แค่ต้องเช็กเกณฑ์ใหม่ทีละ model

### ความผิดพลาด

นับจาก record บน A100 ของ best config ได้แก่ `loop-lcacv7` 48 คำตอบ และ `loop-shape` ที่เพิ่ม shape-cap 12 คำตอบ ด้วย `scripts/fail_audit.py` และ `scripts/put_reject.py`

- **ใส่ค่าที่คำนวณผิด: 0 จาก 39,046 ครั้งที่ตรวจได้**
  - คำนวณซ้ำทุกค่าด้วย `Fraction` ของ Python ไม่ได้ใช้ตัวอ่านของ MIP เอง
  - อีก 1,208 ครั้ง (ราก mod ฟังก์ชัน) ตรวจด้วยวิธีนี้ไม่ได้
- **model ปฏิเสธค่าที่ใส่ให้: 4.1% (832 จาก 20,490)**
  - นับจากคำอย่าง `wait`, `no,`, `wrong`, `actually` ใน 40 token ถัดไป
  - ตัวเทียบคือหลัง `=` ที่ model คิดเองโดยไม่มี MIP ซึ่งมี 15.2%
  - กรณีที่อ่านดูเป็น model สงสัยค่าที่ถูกอยู่แล้ว
- **คำตอบที่ถูกกลายเป็นผิด: 2 จาก 48** (ฝั่งกลับกัน ผิดกลายเป็นถูก 14)
  - เทียบทีละคำตอบกับแขนใส่ค่าอย่างเดียวในรอบเดียวกัน ข้อเดียวกัน seed เดียวกัน
  - ทั้ง 2 คำตอบเป็นโจทย์ MATH ระดับ 5 ที่ **V7 ปิดการคิดเร็วเกินไป** ที่ 8,433 และ 6,582 token
  - ถ้าไม่มีตัวปิด loop model คิดต่อจนตอบถูกที่ 42,974 และ 31,518 token
- **ปิด loop แล้วคำตอบยังผิด:**
  - A: 8 จาก 21
  - V7: 6 จาก 17
  - B: 6 จาก 6 เพราะ B ทำงานเฉพาะกับคำตอบที่วนอยู่แล้ว คือเปลี่ยนคำตอบที่เสียไปแล้วให้ออกมาเป็นค่าหนึ่งค่า
- **cover-cap แตะคำตอบที่ถูก: 0 จาก 1,078 ใน replay** หยุด loop ที่เกิน 150k ได้ 12 จาก 13 ส่วนตัวที่ 13 จับได้ที่ 58k เมื่อป้อนเข้าตัวจับตรง ๆ

### ยังไม่ได้วัด

- **best config ที่มีตัวปิด loop ครบ บนชุดกว้าง เทียบกับ model เปล่า (#19):** 120 ข้อ ข้อละ 2 คำตอบ สองแขนรันในรอบเดียวกัน สคริปต์พร้อมแล้ว รอ compute units ของ A100
- **V7 ที่ปิดเร็วเกินไปบนโจทย์ MATH ที่ต้องคิดยาว:** ถ้าขยายช่วงจะยังจับได้เท่าเดิมโดยไม่เสีย 2 คำตอบนั้นหรือเปล่า
- **ปิด loop แล้ว model ยังตอบผิดได้:** เช่น กลับไปหารต่อในส่วนคำตอบ
- **เกณฑ์บน model เป้าหมาย** (Qwen3.8-Flash-Next, Qwen3.6-35B-A3B-NVFP4) และเวลาของการใส่ค่าใน engine ของ Strata

### repo

ส่วน engine ของ MIP อยู่ที่ Strata-xeno (`include/strata/core/igc_*.hpp`, `src/kernels/qwenputer.cu`) test จะเทียบ header กับ Strata-xeno ให้ เมื่อตั้ง `STRATA_ROOT` ไว้
ส่วน `tir4b --execute/--box` ต้องมี Strata-xeno อยู่ใน `PYTHONPATH` และเลข issue ก่อนแยก repo ดูเทียบได้ที่ #22
