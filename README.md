# mip-research

**MIP (Model Inline Processing): software computes, the model reasons.** While a model thinks, MIP reads the expression it writes
before `=`, computes it exactly, and puts the value after the `=` so the model reads a right number instead of working it out by hand.
It also watches the thinking for loops a small model cannot leave and closes them, so the model still gives an answer.

This repo is the research side: the harness, the mechanisms, the simulations and every measured result.
[Strata-xeno](https://github.com/xenodeve/Strata-xeno) runs MIP in its engine and server, for Qwen3.8-Flash-Next and
Qwen3.6-35B-A3B-NVFP4. The work so far is on **Qwen3.5-4B** (bf16, vLLM 0.33, Colab A100 80 GB, T 0.6, the full 262,144-token
context), the **General Style** only: forms that are true under every reading, no per-model profile.

Contents: [Best config](#the-best-config-now) · [Results](#results) · [Tokens](#tokens) · [How each mechanism works](#how-each-mechanism-works) ·
[Detection to put](#from-detection-to-put-every-point-mip-acts-at) · [MathBackend](#mathbackend-how-the-exact-value-is-computed) ·
[Staying out of the way](#staying-out-of-the-way-why-the-other-work-is-left-alone) · [No fine-tuning](#no-fine-tuning) · [Failures](#failures-in-the-best-config-tests) ·
[What did not work](#what-did-not-work) · [Open](#open) · [The boundary with Strata-xeno](#the-boundary-with-strata-xeno) · [Running](#running) ·
[ภาษาไทย](#ภาษาไทย)

## The best config now

```sh
python tools/igc/tir4b.py run --model <Qwen3.5-4B> --rows rows.jsonl --samples 2 --seed 1 --ids --out out/arm \
  --inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress \
  --loop-close --answer-close --early-cycle --shape-cap 1024 --cover-cap 32768
```

| Layer | Flags | Status |
|---|---|---|
| Inline values, General Style | `--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress` | A100 |
| A: loop close | `--loop-close` | A100 (round 4) |
| B: answer close | `--answer-close` | A100 (round 5) |
| V7: early cycle | `--early-cycle` | A100 (round 7) |
| Shape cap | `--shape-cap 1024` | A100 (round 8, 3 rows) |
| Cover cap | `--cover-cap 32768` | **simulation only**; the A100 round is ready and waits for compute units |

The scores below were measured with the first four layers (A+B+V7 gave the same score as A alone; see the table). Shape cap and
cover cap close loops the others miss; their effect on the score is not measured on a full set yet.

## Results

All numbers are answers right out of answers given, two samples per question, scored by the rule, with an API reader beside it
(`answer_check.py`) and every row where they differ read by hand. **vLLM does not repeat an answer across runs (#21)**, so arms in different runs are not paired: a
difference of 1-2 answers is noise.

### The loop rows: what the loop closers buy

24 questions (48 answers) where earlier arms looped: DeepMind arithmetic (long mixed expressions, big divisions, surds, unit
conversion) and a few MATH problems. The arms below ran in the same rounds on the same rows and seeds; `loop-gen` is the General
Style with no closer, the baseline for the closers.

| Arm | Layers | Right | Cut (no answer) | Tokens, all answers | Median tokens of a right answer |
|---|---|---|---|---|---|
| loop-gen | inline values | 25/48 | 23 | 754k | 10,221 |
| loop-lc | + A | **37/48** | 3 | 724k | 10,531 |
| loop-lcac | + A + B | **37/48** | 1 | 861k | 10,723 |
| loop-lcacv7 | + A + B + V7 | **37/48** | 1 | 936k | 8,829 |

**+12 answers (25 -> 37 of 48)**, and the answers with no answer fell from 23 to 1. For reference, the same 24 questions in the
earlier no-MIP control runs: 24/48 (fresh-ctrl, 21 cut) and 19/48 (v2-ctrl, 26 cut). Those are other runs, and the rows were
chosen because they looped there, so read them as context, not as the gain.

Round 8 (A+B+V7+shape cap, 3 of the hardest rows x 4): 8/12. `div_big-2208`, a big division no answer had got right in 28 earlier
answers across 13 arms, got its first right answer (1/4); `simplify_surd-1931` 4/4 (0/2 in the control); no answer ran to the
context end, where these rows had run to 210k-262k tokens before.

### The broad set: inline values alone, against the control

120 fresh questions (240 answers), inline values with no loop closer, round 1:

| Arm | Right | Cut | Tokens |
|---|---|---|---|
| control (no MIP) | 192/240 | 31 | 2,348k |
| General Style, inline values only | 181/240 | 43 | 2,440k |
| Qwen profile, inline values only | 189/240 | 34 | 2,674k |

**Inline values alone lost 11 answers**, and the loss is in the cuts (31 -> 43): with a right value always in front of it, the
model looped more often, and nothing closed those loops. This is why the closers were built. The broad set has not been run with
the closers yet: that is the round waiting for compute units (`run_vm9.sh`: the best config and the control in one round, 120
questions x 2).

On 52 questions the model had got wrong (104 answers): control 71, General 73, Qwen profile 78.

## Tokens

- **A typical answer costs the same.** The median answer is 10-11k tokens on the loop rows in every arm, and 7.3-7.4k on the
  broad set with or without MIP.
- **Totals are set by the few answers that run to the context end.** One such answer is 210-262k tokens, a quarter of an arm's total,
  so the totals above move with how many of them an arm has, not with the mechanisms. The closers turn cut answers into finished ones,
  and a finished answer spends tokens on its answer.
- **What the closers save is the run to the end.** Round 8's division loop was closed at 55k model tokens where it had run past 200k;
  in simulation, cover cap stops every loop that ran past 150k at 40k-114k.
- **V7 closes loops sooner:** the median right answer fell from 10,723 to 8,829 tokens (not paired, so a hint, not a measurement).

## How each mechanism works

Each closer acts only on the thinking or the answer of a loop that the checks before it let run. A mechanism is reported as
caught / should have caught.

**Inline values (General Style).** At every `=` the model writes, the expression before it is parsed and computed exactly by
`tools/igc/mathbackend` (fractions, decimals, roots, powers, mod), and the value is put after the `=` in plain text
(`--inline-plain`), in a form true under every reading (`--inline-style general`). The model then reads the right number and
reasons on from it. `--no-digit-join` holds digits down for one token after a put that ends in a digit, so `-5` and the model's own
`75` cannot read `-575` (80 of 19,990 puts did that before). `--guard-progress` lets the repeat check pass a tail whose rounds are
nearly all new (trial division by prime after prime), which is work, not a loop. Coverage: **19,546 of 19,636** readable `=` on the
broad set got software's value (99.5 %); arithmetic in the head with no expression written is not counted.

**The repeat check** (every arm, MIP or not): an exact cycle of 20 rounds (8 past 512 tokens), a digit-masked counting cycle, or a
line that comes back. Without a closer it ends the answer `stuck`, with no answer. That is where the 23 cut answers of `loop-gen` went.

**A: loop close** (`--loop-close`). When the repeat check fires in the thinking, MIP puts `</think>` instead of ending the answer,
and the model answers from what it has worked out. Of 15 thinking loops closed in loop-lc, 12 then answered right. This is the +12.

**B: answer close** (`--answer-close`). The other 3 carried the same doubt into the answer and looped there (an endless decimal
`-5511170.918918...`, a recount of `19+1`), where A has nothing left to close. B cuts an exact cycle back to one round, puts
`The final answer is $\boxed{` and lets the model write one value (at most 64 tokens). Cut answers: 3 -> 1.

**V7: early cycle** (`--early-cycle`). The bench's check wants 20 rounds; V7 also reads the thinking at every line end for an exact
cycle of 6 rounds (3 past 512 tokens) spanning 500+ tokens, and closes it there. It fired on 17 of 48 answers and the score
stayed at 37/48: the same loops, closed sooner.

**Shape cap** (`--shape-cap 1024`). Two loops pass every check above: a counting loop where every round is new
(`check ... base 5815. No.`, so `--guard-progress` lets it go) and digit-by-digit division whose numbers change length. Shape cap reads
each run of digits as one token and closes an exact cycle of 1,024 rounds. Stored traces: right answers reach 805 rounds at most,
loops 1,874-4,267. Round 8: it caught 1 of the 2 long division loops.

**Cover cap** (`--cover-cap 32768`). The one shape cap missed: each zero digit made a shorter round, so the exact cycle broke every
few rounds (46 at most). Cover cap reads the last 32,768 tokens (digit runs as one) and asks what share of them repeat a 12-token
stretch seen in the 2,048 before; 97 % or more is a loop. One odd round moves the share by 1/32,768, so it cannot reset the count.
Stored traces of 16 arms: right answers reach 0.92 at most (two that looped long and then came right), the next 0.35; all 10 loops
past 150k tokens reach 0.999-1.0. Replay gate: **0 of 1,078 right answers acted on**, 12 of 13 long loops stopped at 40k-114k (the
13th parts from its record in the replay; fed directly, the check fires at 58k). Not yet run on the A100.

## From detection to put: every point MIP acts at

In the best config MIP watches the model's own text at five points. Nothing else in the run changes: the prompt, the sampling
(T 0.6, top-p 0.95, top-k 20) and the weights are the model's own.

| # | Where it looks | When | What it may do |
|---|---|---|---|
| 1 | The token `=` | Every request stops at an `=` token (in the thinking and in the answer), so MIP reads the text before it ahead of the model's next token | Put the exact value after the `=` |
| 2 | The token after a put that ends in a digit (and a `.` straight after it) | One token | Hold digit tokens down (`--no-digit-join`), so the model's next digit cannot glue onto the put |
| 3 | Every line end in the thinking | As the tokens arrive | V7: close the thinking on a short exact cycle |
| 4 | The whole thinking since the last close, or the whole answer | Every 1,024 tokens and at each request end | The repeat check, shape cap and cover cap: close the thinking (A) or the answer (B) |
| 5 | The answer after B's cue | Until a line end or 64 tokens | Nothing: only the value is read, then the answer ends |

**How a value is put.** The text before the `=` is cut to the longest arithmetic expression that ends there, read in every form the
model writes (`tools/igc/mathexpr.py`: ASCII, LaTeX at any depth, Unicode). It is computed exactly (MathBackend, below), spelled in the
General Style and checked against the refusals (below). If it passes, ` value` is appended to the model's text as plain text, and the
next request goes on from it: to the model the value is simply the next words on the page. If any step says no, nothing is put and the
model writes its own value, as it would without MIP.

**The General Style: a value true under every reading.** The same `=` can mean different things, so the form is chosen to be true
whichever way it is read:

| The model wrote | Put | Why this form |
|---|---|---|
| `12 \times 7 =` | `84` | an integer as itself |
| `9/2 =` | `4.5` | a terminating decimal, the model's own spelling (`9/2` broke a decimal sum into a loop once) |
| `6/26 =` | `3/13` | a quotient that reduces: its lowest terms are the step |
| `34 \div 4 =` in a long division | `8 \text{ rem } 2` | the model wants the digit; `q rem r` is true as a quotient and as a division |
| `1302 / 9 =` where the text works in decimals | `144.6666...` | cut and marked as cut, so it is not a false equality |
| `7^5 =` while working mod 11 | the residue, when the model's own last `=` there linked a value to its residue | what that `=` means on this line |

**The loops it guards against** (point 4, all with the counting rules in the mechanisms below):

| Loop | Example from the runs | Caught by |
|---|---|---|
| An exact cycle, short or long | the same re-check paragraph again and again | the repeat check (20 rounds, 8 past 512 tokens); V7 sooner (6 rounds, 3 past 512, 500+ tokens) |
| Counting with masked digits | `check 101. No. check 102. No.` | the counting check; `--guard-progress` lets it go when nearly every round is new (trial division) |
| A line that keeps coming back | `So the answer is 24?` between other lines | the line-back check |
| A cycle whose numbers change length | `$31090 / 10943 = 2.84...$ (Digit 5244: 2)` | shape cap (digit runs as one token, 1,024 rounds) |
| A cycle with odd rounds in it | long division where each zero digit makes a shorter round | cover cap (97 % of the last 32,768 tokens repeat) |
| The same doubt carried into the answer | `-5511170.918918...` without end, a recount of `19+1` | B: answer close |
| The same value put again and again | `= 15/13 = 15/13 = ...` | the third put in a row of one expression is left to the model (echo guard) |

## MathBackend: how the exact value is computed

`tools/igc/mathbackend/` is a registry of **387 operations in six modules**: numbers 100 (gcd, lcm, phi, modular inverse, digits,
divisors, primes ...), geometry 86, algebra 70, counting 70 (C, P, Catalan, derangements ...), search 43, symbolic 18. The arithmetic
reader (`mathexpr`) covers `+ - * / ^`, fractions, roots, absolute value, floor and ceiling, factorial and percent; `unified_detect`
adds the calls a model writes (`\gcd(12, 18)`, `\binom{7}{3}`, `\phi(36)`).

- **Exact only.** Python's standard library: `Fraction` for rationals, `Surd` for `p + q*sqrt(d)`, `Decimal` where a decimal is asked
  for. No floating point stands behind a put. A root or power that is not exact is refused, not rounded.
- **Bounded.** Every operation states its bound (`@op(name, kind, bound=...)`); past it, or outside its domain, it raises `NotCovered`
  or `BadInput`, and the caller must not guess: no put.
- **Checked against independent libraries.** Each module has a `sim_*.py` that draws inputs and compares every operation with SymPy,
  mpmath or NetworkX, which the backend itself never imports (`tools/test_mathbackend_*.py`; the 2026-11-05 review fixed or refused
  every wrong value and removed contest formulas as operations, #17).

## Staying out of the way: why the other work is left alone

MIP acts only where it is sure it is right and where the model is about to need the value. Everything else passes untouched.

- **Only after an `=` the model wrote, only on exact arithmetic.** A symbol (algebra), a word, a mixed number (`2\frac{1}{3}`), an
  inexact root, or a number too big to be a step: no put. On the broad set, puts went to 19,546 of the 19,636 readable `=`; the rest of
  the text was never touched.
- **Refused even when the arithmetic is right,** because the value is not what the model means there (each rule came from reading
  stored traces, #18 / #19):
  - a number alone: `m=4,512=8*64` is m=4 and 512, not 4512;
  - nothing new: the value would repeat the expression (`15/13 = 15/13`);
  - an operand of a longer expression still being written;
  - a column of a written subtraction that borrows (`0-9 = 1`), a count of coins (`2 \times 10 + 5 \times 1 = 7` coins);
  - an operator the question defined otherwise (`a * b = a - b`);
  - a congruence with no modulus named;
  - the third put in a row of the same expression (echo guard).
- **Closers act only on a loop, with thresholds set above every right answer.** Each threshold sits over the longest stretch any
  right answer reached in the stored traces (shape: 805 rounds against a cap of 1,024; cover: 0.92 against 0.97), and each closer
  went to the A100 only after the replay gate showed it acting on no right answer, or every act read and understood.
- **The model decides.** A put is a value on the page, not an instruction; a close hands the model `</think>` or the final-answer cue,
  and the model writes the answer.

## No fine-tuning

None of this changes a weight. MIP works on the text stream at inference time: it reads what the model writes and appends tokens to
it. The model is the plain checkpoint (Qwen3.5-4B, bf16, from its model card), with the model card's own sampling. The digit set and
the `=` and line-end tokens come from the tokenizer, so the same mechanisms run on another model's tokenizer; only the thresholds
must be checked again there.

## Failures in the best-config tests

From the A100 records of the four best-config arms (`loop-lc`, `loop-lcac`, `loop-lcacv7`, `loop-shape`; 156 answers), with
`scripts/fail_audit.py` and `scripts/put_reject.py`:

| Failure | Count | How it was measured |
|---|---|---|
| A wrong value put (the arithmetic of the put is wrong) | **0 of 51,015** puts checked | every put re-computed by Python's `Fraction` on a normalised copy, not by MIP's own reader; 2,850 more (roots, mod, functions) were not plain enough to re-check this way |
| A put the model pushed away | **4.1 %** of puts in loop-lcacv7 (832 of 20,490), 0.9 % in loop-shape | a rejection word (`wait`, `no,`, `wrong`, `actually`, `let me recheck` ...) within the next 40 tokens. Base rate: after the model's **own** `= value` in the no-MIP control, 15.2 %. The cases read were the model doubting a right value (`10856 / 5192 = 23/11? No, wait.`) |
| A right answer made wrong | **2** in loop-lcacv7 (1 in loop-lc, 3 in loop-lcac) | same questions and seeds as the no-closer arm of the same round, answer by answer. Both loop-lcacv7 cases are MATH level-5 problems where **V7 closed the thinking too early** (at 8,433 and 6,582 tokens); without the closers the model finished right at 42,974 and 31,518. Against these, 14 answers went from wrong to right |
| A closer acting on an answer that then ended wrong | loop-lcacv7: loop close 8 of 21, early cycle 6 of 17, answer close 6 of 6 | every act, and how its answer ended. B acts only when the answer is already looping, so its answers are mostly lost ones it turns into a value |

Not paired across runs (#21): the right-to-wrong count is per answer between two runs with the same seeds, so a part of it can be
sampling. The two V7 cases are read, and V7 caused them; whether a longer V7 span fixes them without losing what V7 catches is open.

## What did not work

| Tried | Result | Why it was dropped |
|---|---|---|
| D: loop value (`--loop-value`): put the question's own exact value before the close | 11/20 on the 5 hardest rows | the model ignored the value it was given |
| Suppress re-checks (`--suppress-recheck 60`) | 20/48, 28 cut (25/48 without) | holding the re-check words down made the loops worse |
| check-hand: correct a value the model computed by hand | 6 of 6 corrections in the replay were false (#18) | the parser misreads too much text to correct it |
| Logit bias over 1,024 tokens | vLLM refused the request; four arms died at start | the replay now counts every request vLLM would refuse (`vllm_refused`) |

## Open

- **The broad-set round with every closer** (#19): best config against the control on 120 questions x 2. Scripts are ready; it waits
  for compute units.
- After the loop is closed the model can still be wrong: on `div_big` it went back to dividing in the answer and lost the sign.
- Thresholds were measured on Qwen3.5-4B traces. Before a model profile, every threshold is checked on the target model
  (Qwen3.8-Flash-Next, Qwen3.6-35B-A3B-NVFP4).
- The MIP core (inline writer and closers) will come out of `tir4b.py` as a package with one token-level API, which Strata-xeno will
  take pinned.

## The boundary with Strata-xeno

Split out of Strata-xeno on 2026-11-08 with its git history; 21 issues moved with it (the old-to-new number map is pinned as #22,
since a bare `#N` written before the move means Strata-xeno #N). The engine side stays there: `include/strata/core/igc_*.hpp` and
`src/kernels/qwenputer.cu`. Generators here write those headers, and `tools/test_igc_adapter_artifact.py` checks the schema header
against a Strata-xeno checkout when `STRATA_ROOT` names one. `tir4b --execute/--box` and `external_bench --tool-loop` import
Strata's `serve.sandbox`, so they need Strata-xeno on `PYTHONPATH`.

| Path | What |
|---|---|
| `tools/igc/tir4b.py` | The harness: the model in vLLM, inline values and every closer |
| `tools/igc/sim_replay.py` | Replays stored traces with a mechanism added: where it acts, on how many cut answers, whether on a right one, what vLLM would refuse |
| `tools/igc/mathbackend/` | The exact computation behind a put |
| `tools/igc/answer_check.py` | The API reader beside the rule (its key stays on the machine that runs it, never on a VM) |
| `tools/igc/` (rest) | Readouts, adapters, corpora, benchmarks (LiveBench, LCB, standard sets) |
| `tools/qwenputer/` | Native Qwenputer: computation compiled for the engine |

## Running

```sh
python -m unittest discover -s tools -p "test_*.py" -t .     # 923 tests; 78 skip without a GPU, a model or STRATA_ROOT
```

The rules of the work: simulate first, the A100 last; the full 262,144-token context, never `--max-tokens`; sampled at T 0.6 with
several seeds; report caught / should have caught; never tune to the test sets.

---

## ภาษาไทย

**MIP (Model Inline Processing): ให้ software คำนวณ ให้ model ใช้เหตุผล** ระหว่างที่ model คิด MIP จะอ่านนิพจน์ที่ model เขียนไว้ก่อน `=`
คำนวณให้ตรงเป๊ะ แล้วใส่ค่าต่อท้าย `=` ให้ model ได้อ่านตัวเลขที่ถูกแทนที่จะคิดเลขเอง นอกจากนี้ MIP ยังคอยดูว่า model เล็กวนอยู่ใน loop
ที่ออกเองไม่ได้หรือเปล่า ถ้าใช่ก็ปิดให้ model ยังตอบออกมาได้

repo นี้เป็นฝั่งวิจัย มีเครื่องมือทดลอง ตัวกลไก sim และผลที่วัดได้ทั้งหมด ส่วน [Strata-xeno](https://github.com/xenodeve/Strata-xeno)
เป็นที่ที่เอา MIP ไปรันจริงใน engine กับ server ให้ Qwen3.8-Flash-Next และ Qwen3.6-35B-A3B-NVFP4 งานที่ผ่านมาทำบน **Qwen3.5-4B**
(bf16, vLLM 0.33, Colab A100 80 GB, T 0.6, context เต็ม 262,144 token) และใช้แค่ **General Style** คือรูปแบบที่ถูกไม่ว่าจะอ่านแบบไหน
ยังไม่เปิด Profile ของ model ตัวไหน

### best config ตอนนี้

ใช้ flag ตามคำสั่งข้างบน ประกอบด้วย 6 ชั้น
- **ตัวเขียนค่า (General Style):** ทดสอบบน A100 แล้ว
- **A: loop close:** A100 รอบ 4
- **B: answer close:** A100 รอบ 5
- **V7: early cycle:** A100 รอบ 7
- **shape-cap:** A100 รอบ 8 บน 3 ข้อ
- **cover-cap:** **ยังเป็นผลจาก sim อย่างเดียว** รอบ A100 พร้อมแล้ว รอ compute units

คะแนนที่วัดได้มาจาก 4 ชั้นแรก (A+B+V7 ได้คะแนนเท่ากับ A ตัวเดียว) ส่วน shape-cap กับ cover-cap ปิด loop ที่ตัวอื่นพลาด แต่ยังไม่ได้วัดผลต่อคะแนนบนชุดเต็ม

### ผลการทดสอบ

คะแนนทุกตัวคือจำนวนคำตอบที่ถูกจากทั้งหมด ข้อละ 2 คำตอบ ให้คะแนนด้วยกฎ มี API reader ตรวจคู่กัน และข้อไหนที่สองฝั่งให้ผลไม่ตรงกันก็อ่านเองทุกข้อ
**vLLM ไม่ได้ให้คำตอบซ้ำเดิมข้ามรอบ (#21)** เลยเทียบ arm ต่างรอบกันแบบจับคู่ไม่ได้ ต่างกัน 1-2 คำตอบถือเป็น noise

**ชุดข้อที่วน (24 ข้อ 48 คำตอบ ข้อที่ arm ก่อน ๆ วนติด):** ทุก arm รันในรอบเดียวกัน ข้อเดียวกัน seed เดียวกัน
- loop-gen (ตัวเขียนค่าอย่างเดียว ไม่มีตัวปิด loop) ได้ 25/48 ถูกตัดไม่มีคำตอบ 23
- เพิ่ม A ได้ **37/48** ถูกตัดเหลือ 3
- เพิ่ม A+B ได้ 37/48 ถูกตัดเหลือ 1
- เพิ่ม A+B+V7 ได้ 37/48 ถูกตัดเหลือ 1

**คะแนนเพิ่ม 12 คำตอบ (25 เป็น 37 จาก 48)** และคำตอบที่ไม่มีคำตอบลดจาก 23 เหลือ 1 ส่วน control ที่ไม่มี MIP บน 24 ข้อเดียวกัน
ได้ 24/48 กับ 19/48 แต่มาจากรอบอื่น และข้อพวกนี้ถูกเลือกมาเพราะวนในรอบนั้นพอดี เลยใช้ดูประกอบได้ แต่นับเป็นคะแนนที่เพิ่มไม่ได้

รอบ 8 (A+B+V7+shape-cap บน 3 ข้อที่ยากที่สุด ข้อละ 4 คำตอบ) ได้ 8/12
- `div_big-2208` ไม่เคยมีใครตอบถูกเลยใน 28 คำตอบของ 13 arm ก่อนหน้า รอบนี้ถูกเป็นครั้งแรก 1/4
- `simplify_surd-1931` ได้ 4/4 (control ได้ 0/2)
- ไม่มีคำตอบไหนวนจนเต็ม context ทั้งที่รอบก่อนข้อเหล่านี้วิ่งไปถึง 210k-262k

**ชุดกว้าง 120 ข้อ (240 คำตอบ) ใช้แค่ตัวเขียนค่า ยังไม่มีตัวปิด loop:**
- control ได้ 192
- General ได้ 181
- Qwen profile ได้ 189

**ตัวเขียนค่าอย่างเดียวเสียไป 11 คำตอบ** ทั้งหมดมาจากคำตอบที่ถูกตัด (31 เพิ่มเป็น 43) เพราะพอมีค่าถูกวางอยู่ตรงหน้าตลอด model กลับวนบ่อยขึ้น
แล้วก็ไม่มีอะไรปิด loop ให้ ตรงนี้เป็นเหตุผลที่สร้างตัวปิด loop ขึ้นมา ชุดกว้างนี้ยังไม่ได้รันพร้อมตัวปิด loop รอบนั้นคือรอบที่รอ compute units อยู่
ส่วนอีกชุดคือ 52 ข้อที่ model เคยตอบผิด control ได้ 71 General ได้ 73 และ Qwen profile ได้ 78

### token

- **คำตอบทั่วไปใช้ token เท่าเดิม:** ค่ามัธยฐานอยู่ที่ 10-11k ในชุดข้อที่วนทุก arm และ 7.3-7.4k ในชุดกว้าง จะมี MIP หรือไม่มีก็เท่ากัน
- **ยอดรวมขึ้นอยู่กับไม่กี่คำตอบที่วิ่งจนเต็ม context:** คำตอบแบบนี้คำตอบเดียวกินไป 210-262k คิดเป็นประมาณหนึ่งในสี่ของทั้ง arm
  ยอดรวมเลยขยับตามจำนวนคำตอบแบบนี้ ไม่ได้ขยับตามกลไก อีกเรื่องคือตัวปิด loop เปลี่ยนคำตอบที่เคยถูกตัดให้กลายเป็นคำตอบที่จบได้ ซึ่งก็ต้องใช้ token เขียนคำตอบด้วย
- **ที่ประหยัดได้จริงคือไม่ต้องวิ่งจนเต็ม context:** รอบ 8 ปิด loop หารยาวได้ที่ 55k แทนที่จะเกิน 200k
  และใน sim cover-cap หยุด loop ที่เคยเกิน 150k ได้ทุกตัวที่ 40k-114k
- **V7 ปิด loop ได้เร็วขึ้น:** มัธยฐานของคำตอบที่ถูกลดจาก 10,723 เหลือ 8,829 แต่ไม่ได้จับคู่ ถือเป็นแนวโน้ม ยังไม่ใช่ผลวัด

### กลไกแต่ละตัวทำงานยังไง

- **ตัวเขียนค่า (General Style):**
  - ทุก `=` ที่ model เขียน นิพจน์ข้างหน้าจะถูก parse แล้วคำนวณตรงเป๊ะด้วย `mathbackend` (เศษส่วน ทศนิยม ราก ยกกำลัง mod) จากนั้นใส่ค่าเป็นข้อความธรรมดา ในรูปที่ถูกทุกการอ่าน
  - `--no-digit-join` กันไม่ให้ตัวเลขที่ model เขียนต่อ ไปติดกับค่าที่ใส่ให้ เช่น `-5` ตามด้วย `75` กลายเป็น `-575` ก่อนแก้เคยเกิด 80 จาก 19,990 ครั้ง
  - `--guard-progress` ปล่อยงานที่แต่ละรอบเป็นของใหม่เกือบหมดให้วิ่งต่อ เช่น ลองหารด้วยจำนวนเฉพาะไปทีละตัว
  - ครอบคลุม **19,546 จาก 19,636** จุด `=` ที่อ่านได้ (99.5%) ไม่นับการคิดในหัวที่ไม่ได้เขียนนิพจน์ออกมา
- **ตัวตรวจการวน (มีในทุก arm):** จับ cycle ที่ซ้ำตรงตัว 20 รอบ (8 รอบเมื่อยาวเกิน 512 token) การนับที่ซ้ำเมื่อปิดตัวเลขทิ้ง หรือบรรทัดที่วนกลับมา
  ถ้าไม่มีตัวปิด loop คำตอบจะจบแบบ `stuck` ไม่มีคำตอบ 23 คำตอบที่ถูกตัดของ loop-gen หายไปตรงนี้
- **A: loop close:** แทนที่จะจบคำตอบ ให้ใส่ `</think>` ปิดการคิด แล้วให้ model ตอบจากสิ่งที่คิดมาแล้ว
  ปิดไป 15 ครั้ง ตอบถูก 12 ครั้ง ซึ่งก็คือ 12 คำตอบที่เพิ่มขึ้นมา
- **B: answer close:** อีก 3 คำตอบพาความสงสัยเดิมเข้าไปในส่วนคำตอบแล้วไปวนต่อที่นั่น เช่น ทศนิยมไม่รู้จบ หรือนับ `19+1` ซ้ำไปเรื่อย ๆ
  B ตัด cycle ให้เหลือรอบเดียว ใส่ `The final answer is $\boxed{` แล้วให้ model เขียนค่าเดียว ไม่เกิน 64 token
- **V7: early cycle:** ทุกครั้งที่จบบรรทัด จะดูว่ามี cycle ที่ซ้ำตรงตัว 6 รอบ (3 รอบเมื่อยาวเกิน 512) และกินพื้นที่ตั้งแต่ 500 token ขึ้นไปหรือเปล่า ถ้ามีก็ปิดตรงนั้นเลย
  ทำงานใน 17 จาก 48 คำตอบ คะแนนยังอยู่ที่ 37/48 เหมือนเดิม เป็น loop ชุดเดิม แค่ปิดได้เร็วขึ้น
- **shape-cap:** มี loop สองแบบที่ผ่านตัวตรวจข้างบนทั้งหมด แบบแรกคือการนับที่ทุกรอบเป็นของใหม่ แบบที่สองคือการหารทีละหลักที่ตัวเลขยาวไม่เท่ากัน
  shape-cap มองตัวเลขทั้งก้อนเป็น token เดียว แล้วปิดเมื่อเจอ cycle ที่ซ้ำตรงตัวครบ 1,024 รอบ
  คำตอบที่ถูกไปได้สูงสุด 805 รอบ ส่วน loop อยู่ที่ 1,874-4,267 รอบ ในรอบ 8 จับได้ 1 จาก 2
- **cover-cap:** ตัวที่ shape-cap พลาดคือการหารที่ทุกครั้งที่ผลหารได้หลัก 0 รอบนั้นจะสั้นลง cycle เลยขาดอยู่เรื่อย ๆ นับได้สูงสุดแค่ 46
  - cover-cap ดู 32,768 token ล่าสุดแทน แล้วถามว่ากี่เปอร์เซ็นต์ที่ซ้ำท่อน 12 token ซึ่งเคยเห็นใน 2,048 token ก่อนหน้า ถ้าได้ 97% ขึ้นไปถือว่าเป็น loop
    รอบแปลก ๆ หนึ่งรอบขยับสัดส่วนแค่ 1/32,768 เลยไม่ทำให้เริ่มนับใหม่
  - คำตอบที่ถูกได้สูงสุด 0.92 ส่วน loop ที่เกิน 150k ทุกตัวได้ 0.999-1.0
  - replay gate ไม่แตะคำตอบที่ถูกเลย 0 จาก 1,078 หยุด loop ยาวได้ 12 จาก 13 ส่วนตัวที่ 13 ถ้าป้อนเข้าตัวจับตรง ๆ จะจับได้ที่ 58k ยังไม่ได้รันบน A100

### ตั้งแต่ตรวจจับจนใส่ค่า: MIP ทำงานกี่จุด

ใน best config MIP ดูข้อความที่ model เขียนเองอยู่ 5 จุด นอกนั้นไม่ได้แตะอะไรเลย prompt, การสุ่ม (T 0.6, top-p 0.95, top-k 20) และน้ำหนักของ model ยังเหมือนเดิมทั้งหมด

1. **token `=`:** ทุก request หยุดตรง `=` ทั้งตอนคิดและตอนตอบ MIP จึงได้อ่านข้อความก่อน `=` ก่อนที่ model จะเขียน token ถัดไป จุดนี้ใส่ค่าที่ถูกต่อท้าย `=`
2. **token ถัดจากค่าที่ใส่ ถ้าค่านั้นลงท้ายด้วยตัวเลข** (รวม `.` ที่ตามมาทันที): กดตัวเลขไว้ 1 token (`--no-digit-join`) ตัวเลขที่ model เขียนต่อจะได้ไม่ไปติดกับค่าที่ใส่
3. **ทุกครั้งที่จบบรรทัดตอนคิด:** V7 ปิดการคิดเมื่อเจอ cycle สั้นที่ซ้ำตรงตัว
4. **การคิดทั้งหมดตั้งแต่ครั้งล่าสุดที่ปิด หรือส่วนคำตอบทั้งหมด:** ตรวจทุก 1,024 token และทุกครั้งที่ request จบ ด้วยตัวตรวจการวน shape-cap และ cover-cap ถ้าเจอก็ปิดการคิด (A) หรือปิดคำตอบ (B)
5. **ส่วนคำตอบหลังวางประโยคบังคับตอบของ B:** ไม่ทำอะไรเพิ่ม อ่านแค่ค่าเดียวจนจบบรรทัดหรือครบ 64 token แล้วจบคำตอบ

**ใส่ค่ายังไง:** ข้อความก่อน `=` จะถูกตัดเหลือนิพจน์เลขที่ยาวที่สุดที่จบตรงนั้น อ่านได้ทุกรูปแบบที่ model เขียน (`mathexpr`: ASCII, LaTeX ซ้อนกี่ชั้นก็ได้, Unicode)
จากนั้นคำนวณให้ตรงเป๊ะด้วย MathBackend เขียนในรูปแบบ General Style แล้วเช็กกับเงื่อนไขที่ไม่ให้ใส่ (ดูหัวข้อถัดไป) ถ้าผ่าน ` value` ก็จะถูกต่อท้ายข้อความของ model
เป็นข้อความธรรมดา แล้ว request ถัดไปก็เดินต่อจากตรงนั้น สำหรับ model ค่าที่ใส่ให้ก็คือคำถัดไปบนหน้ากระดาษเท่านั้น ถ้าขั้นไหนบอกว่าไม่ใส่ ก็ไม่ใส่อะไรเลย
model จะเขียนค่าเองเหมือนตอนไม่มี MIP

**General Style คือค่าที่ถูกไม่ว่าจะอ่านแบบไหน:** `=` ตัวเดียวกันมีความหมายได้หลายแบบ เลยเลือกรูปที่ถูกทุกแบบ
- `12 \times 7 =` ใส่ `84`
- `9/2 =` ใส่ `4.5` เพราะเป็นทศนิยมที่จบได้ เขียนแบบเดียวกับที่ model เขียนเอง
- `6/26 =` ใส่ `3/13` เพราะการย่อเศษส่วนก็คือขั้นนั้นพอดี
- `34 \div 4 =` ระหว่างหารยาว ใส่ `8 \text{ rem } 2` ถูกทั้งในแง่ผลหารและในแง่การหาร
- `1302 / 9 =` ตอนที่ model กำลังทำทศนิยม ใส่ `144.6666...` ตัดทศนิยมแล้วมีเครื่องหมายบอกไว้ จะได้ไม่เป็นสมการเท็จ
- ระหว่างทำ mod ใส่เศษเหลือ เฉพาะเมื่อ `=` ตัวล่าสุดที่ model เขียนเองบนบรรทัดนั้นหมายถึงเศษเหลือ

**loop ที่ป้องกันได้**
- **cycle ที่ซ้ำตรงตัว ทั้งสั้นและยาว:** ตัวตรวจการวน (20 รอบ หรือ 8 รอบเมื่อยาวเกิน 512) และ V7 ที่จับได้เร็วกว่า (6 รอบ หรือ 3 รอบ ยาว 500 token ขึ้นไป)
- **การนับเลขเมื่อปิดตัวเลขทิ้ง** เช่น `check 101. No. check 102. No.`: ตัวตรวจการนับ ส่วน `--guard-progress` ปล่อยให้วิ่งต่อถ้าแต่ละรอบเป็นของใหม่เกือบหมด
- **บรรทัดที่วนกลับมาเรื่อย ๆ:** ตัวตรวจ line-back
- **cycle ที่ตัวเลขยาวไม่เท่ากัน:** shape-cap
- **cycle ที่มีรอบแปลก ๆ แทรก:** cover-cap
- **ความสงสัยเดิมที่พาไปวนต่อในส่วนคำตอบ:** B
- **ค่าเดิมถูกใส่ซ้ำไปซ้ำมา** เช่น `= 15/13 = 15/13 = ...`: ครั้งที่ 3 ติดกันของนิพจน์เดียวกันจะปล่อยให้ model เขียนเอง

### MathBackend ทำงานยังไง

`tools/igc/mathbackend/` เป็นที่รวม **387 operation แบ่งเป็น 6 module**:
- numbers 100 (gcd, lcm, phi, inverse mod, หลักตัวเลข, ตัวหาร, จำนวนเฉพาะ ...)
- geometry 86
- algebra 70
- counting 70 (C, P, Catalan, derangement ...)
- search 43
- symbolic 18

ตัวอ่านนิพจน์รองรับ `+ - * / ^` เศษส่วน ราก ค่าสัมบูรณ์ floor/ceiling factorial และเปอร์เซ็นต์ ส่วน `unified_detect` เพิ่มการเรียกฟังก์ชันที่ model เขียน
เช่น `\gcd(12, 18)`, `\binom{7}{3}`

- **คำนวณแบบตรงเป๊ะเท่านั้น:** ใช้แค่ standard library ของ Python ได้แก่ `Fraction` สำหรับเศษส่วน `Surd` สำหรับ `p + q*sqrt(d)` และ `Decimal` เมื่อต้องการทศนิยม
  ไม่มี floating point อยู่เบื้องหลังค่าที่ใส่ ถ้าราก ยกกำลัง ได้ค่าไม่ลงตัว จะไม่ใส่ ไม่ใช่ปัดเศษ
- **มีขอบเขตชัด:** ทุก operation ประกาศขอบเขตของตัวเอง ถ้าเกินขอบเขตหรืออยู่นอกโดเมน จะโยน `NotCovered` หรือ `BadInput` แล้วห้ามเดา ผลคือไม่ใส่ค่า
- **ตรวจกับ library อิสระ:** ทุก module มี `sim_*.py` ที่สุ่มค่าเข้าไป แล้วเทียบทุก operation กับ SymPy, mpmath หรือ NetworkX ซึ่งตัว backend เองไม่ได้ import

### ไม่กระทบการทำงานอื่นได้ยังไง

MIP แทรกเฉพาะตอนที่มั่นใจว่าถูก และ model กำลังจะต้องใช้ค่านั้นจริง ๆ นอกนั้นปล่อยผ่านหมด

- **ใส่ค่าเฉพาะหลัง `=` ที่ model เขียนเอง และเฉพาะเลขคณิตที่คำนวณได้ตรงเป๊ะ:** ถ้ามีตัวแปร มีคำพูด มีจำนวนคละ มีรากที่ไม่ลงตัว หรือเลขใหญ่เกินกว่าจะเป็นขั้นหนึ่ง ก็ไม่ใส่
  ในชุดกว้างใส่ไป 19,546 จาก 19,636 จุด `=` ที่อ่านได้ ข้อความส่วนอื่นไม่ถูกแตะเลย
- **ถึงเลขจะถูก ก็ไม่ใส่ ถ้าค่านั้นไม่ใช่สิ่งที่ model หมายถึง:** ทุกกฎได้มาจากการอ่าน trace ที่เก็บไว้ ได้แก่
  - ตัวเลขเดี่ยว ๆ
  - ค่าที่ซ้ำกับนิพจน์ ไม่มีอะไรใหม่
  - ตัวเลขที่เป็นส่วนหนึ่งของนิพจน์ที่ยังเขียนไม่จบ
  - หลักที่ต้องยืมในการลบแบบตั้งหลัก
  - การนับเหรียญ
  - operator ที่โจทย์นิยามไว้เป็นอย่างอื่น
  - congruence ที่ไม่ได้บอก modulus
  - การใส่ค่าครั้งที่ 3 ติดกันของนิพจน์เดียวกัน
- **ตัวปิด loop ทำงานเฉพาะตอนที่วนจริง และเกณฑ์ตั้งไว้สูงกว่าคำตอบที่ถูกทุกตัว:** shape คำตอบที่ถูกได้สูงสุด 805 รอบ เกณฑ์อยู่ที่ 1,024 cover คำตอบที่ถูกได้สูงสุด 0.92 เกณฑ์อยู่ที่ 0.97
  ทุกตัวต้องผ่าน replay gate ก่อนขึ้น A100 คือต้องไม่แตะคำตอบที่ถูกเลย หรือถ้าแตะ ต้องอ่านแล้วเข้าใจทุกครั้ง
- **model เป็นคนตัดสินใจ:** ค่าที่ใส่เป็นแค่ตัวเลขบนหน้ากระดาษ ไม่ใช่คำสั่ง ส่วนการปิด loop ก็แค่ยื่น `</think>` หรือประโยคบังคับตอบให้ แล้ว model เขียนคำตอบเอง

### ไม่ต้อง fine-tune

ไม่มีอะไรไปแก้น้ำหนักของ model เลย MIP ทำงานกับข้อความระหว่าง inference อ่านสิ่งที่ model เขียน แล้วต่อ token ลงไป
model ที่ใช้คือ checkpoint ปกติ (Qwen3.5-4B, bf16) ใช้ค่าการสุ่มตาม model card ส่วนชุดตัวเลข token `=` และ token จบบรรทัด ได้มาจาก tokenizer
กลไกเดียวกันจึงใช้กับ tokenizer ของ model อื่นได้ แค่ต้องเช็กเกณฑ์ใหม่

### ความผิดพลาดในการทดสอบ best config

นับจาก record บน A100 ของ arm ที่เป็น best config ทั้ง 4 ชุด (156 คำตอบ) ด้วย `scripts/fail_audit.py` และ `scripts/put_reject.py`

- **ใส่ค่าที่คำนวณผิด: 0 จาก 51,015 ครั้งที่ตรวจได้**
  - คำนวณซ้ำทุกค่าด้วย `Fraction` ของ Python ไม่ได้ใช้ตัวอ่านของ MIP เอง
  - อีก 2,850 ครั้ง (ราก mod ฟังก์ชัน) ตรวจด้วยวิธีนี้ไม่ได้
- **model ปฏิเสธค่าที่ใส่ให้: 4.1%** ใน loop-lcacv7 (832 จาก 20,490) และ 0.9% ใน loop-shape
  - นับจากคำอย่าง `wait`, `no,`, `wrong`, `actually` ที่โผล่ใน 40 token ถัดไป
  - ตัวเทียบคือหลัง `=` ที่ model คิดเองใน control ซึ่งมี 15.2%
  - ตัวอย่างที่อ่านดูเป็น model สงสัยค่าที่ถูกอยู่แล้ว เช่น `10856 / 5192 = 23/11? No, wait.`
- **คำตอบที่ถูกกลายเป็นผิด: 2 คำตอบใน loop-lcacv7** (loop-lc 1, loop-lcac 3) เทียบทีละคำตอบกับ arm ที่ไม่มีตัวปิด loop ในรอบเดียวกัน ข้อเดียวกัน seed เดียวกัน
  - ทั้ง 2 คำตอบเป็นโจทย์ MATH ระดับ 5 ที่ **V7 ปิดการคิดเร็วเกินไป** ที่ 8,433 และ 6,582 token
  - ถ้าไม่มีตัวปิด model คิดต่อจนตอบถูกที่ 42,974 และ 31,518 token
  - ฝั่งกลับกัน มี 14 คำตอบที่จากผิดกลายเป็นถูก
- **ตัวปิด loop ทำงานแล้วคำตอบยังผิด (loop-lcacv7):**
  - loop close 8 จาก 21
  - early cycle 6 จาก 17
  - answer close 6 จาก 6 เพราะ B ทำงานเฉพาะตอนที่คำตอบวนอยู่แล้ว คำตอบส่วนใหญ่ที่ B แตะจึงเป็นคำตอบที่เสียไปแล้ว B แค่บังคับให้ออกมาเป็นค่าหนึ่งค่า

การเทียบข้ามรอบจับคู่ไม่ได้ (#21) ตัวเลข "ถูกกลายเป็นผิด" เลยอาจมีส่วนที่มาจากการสุ่ม แต่ 2 กรณีของ V7 ผมอ่านแล้ว เป็นเพราะ V7 จริง
ส่วนการขยายช่วงของ V7 จะแก้ได้โดยไม่เสียสิ่งที่ V7 จับได้หรือเปล่า ยังต้องทดสอบ

### สิ่งที่ไม่ได้ผล

- **D: ใส่ค่าที่ถูกของโจทย์ให้ก่อนปิด loop:** model ไม่ใช้ค่านั้นเลย
- **suppress-recheck:** แย่ลงเหลือ 20/48 ถูกตัด 28
- **check-hand:** แก้ค่าที่ model คิดเอง ผลคือแก้ผิดทั้ง 6 จาก 6 ครั้ง (#18)
- **logit bias เกิน 1,024 token:** vLLM ไม่รับ request ตอนนี้ replay นับ request ที่ vLLM จะไม่รับไว้ให้แล้ว

### งานที่ยังค้าง

- **รอบชุดกว้างที่เปิดตัวปิด loop ครบ (#19):** สคริปต์พร้อมแล้ว รอ compute units
- **ปิด loop แล้ว model ยังตอบผิดได้:** ข้อ div_big กลับไปหารต่อในส่วนคำตอบ แล้วเครื่องหมายลบหาย
- **เกณฑ์ทุกตัววัดจาก Qwen3.5-4B:** ก่อนทำ Profile ต้องเช็กกับ model เป้าหมายก่อน
- **แยกแกน MIP ออกจาก `tir4b.py`:** ทำเป็น package ที่มี API ระดับ token ตัวเดียว ให้ Strata-xeno ดึงไปใช้แบบตรึงเวอร์ชัน

### รอยต่อกับ Strata-xeno

repo นี้แยกออกมาจาก Strata-xeno เมื่อ 2026-11-08 พร้อมประวัติ git และย้าย issue มาด้วย 21 ใบ ตารางเทียบเลขเดิมกับเลขใหม่ปักหมุดไว้ที่ #22
ส่วน engine ยังอยู่ที่ Strata-xeno ได้แก่ `include/strata/core/igc_*.hpp` และ `src/kernels/qwenputer.cu`
test จะเทียบ header กับ Strata-xeno ให้ เมื่อตั้ง `STRATA_ROOT` ไว้

กติกาการทำงาน:
- simulate ก่อน ขึ้น A100 เป็นอย่างสุดท้าย
- ใช้ context เต็ม ไม่ใช้ `--max-tokens`
- สุ่มที่ T 0.6 หลาย seed
- รายงานกลไกเป็น "จับได้ / ควรจับ"
- ห้ามจูนให้เข้ากับชุดทดสอบ
