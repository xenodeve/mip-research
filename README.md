# mip-research

**MIP (Model Inline Processing): software computes, the model reasons.** While a model thinks, MIP reads the expression it writes
before `=`, computes it exactly, and puts the value after the `=` so the model reads a right number instead of working it out by hand.
It also watches the thinking for loops a small model cannot leave and closes them, so the model still gives an answer.

This repo is the research side: the harness, the mechanisms, the simulations and every measured result.
[Strata-xeno](https://github.com/xenodeve/Strata-xeno) runs MIP in its engine and server, for Qwen3.8-Flash-Next and
Qwen3.6-35B-A3B-NVFP4. The work so far is on **Qwen3.5-4B** (bf16, vLLM 0.33, Colab A100 80 GB, T 0.6, the full 262,144-token
context), the **General Style** only: forms that are true under every reading, no per-model profile.

Contents: [Best config](#the-best-config-now) · [Results](#results) · [Tokens](#tokens) · [How each mechanism works](#how-each-mechanism-works) ·
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
