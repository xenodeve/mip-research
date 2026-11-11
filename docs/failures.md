# Failure patterns: what made the model wrong, and what fixes it

Every way an answer went wrong that was read by hand, one row per pattern: what the model did, where it was seen (with the build),
what fixes it or is expected to, and **whether the fix has been tested**. A pattern stays here after it is fixed, so the next model
or round can be checked against it. The developer's rule (2026-11-10): a pattern that can occur gets fixed, however rare.

[ภาษาไทย](#ภาษาไทย)

**Status words:**
- **fixed, live** = the fix ran with the model and the pattern's answers came out right.
- **fixed, replay** = the fix acts correctly on stored text; no live run yet.
- **fix planned** = a mechanism is designed or exists elsewhere, not yet tried on this pattern.
- **no fix yet** = nothing in the plans reaches it.
- **ceiling** = the information is not in what the model reads.

**Source of the round 9 rows:** all 25 General answers that the rule scores wrong in round 9 (240 pairs, **4B NVFP4 W4A8 · llama.cpp ·
PC layer**, General Style + closers), each read by hand (plan:
[2026-11-10-round9-fixes](plans/2026-11-10-round9-fixes.md)): P1-P5 15, P7 2, P8-P10 6, P11 1, P12 1 = 25.

## Summary

| # | Pattern | Seen (answers) | Fix | Tested? | Status |
|---|---|---|---|---|---|
| P1 | Right value, wrong form: a rounded decimal or the integer part boxed instead of the exact fraction | round 9: 6 | `--exact-answer` | targeted live run: 6 of 6 right | fixed, live |
| P2 | Two values offered, or a hedge (`275 or -275`) | round 9: 3 | `--exact-answer` at the answer-close cue | live: 2 of 2 run right (`div_big-9991` x2); `5942#1` replay only | fixed, live (2) / replay (1) |
| P3 | A wrong value written during a loop, then boxed at the close | round 9: 3 (`9951#0`, `9991#1`, `2783#0` boxed `0`) | `--exact-answer` at the cue | live: all right; `2783#0` with an act, the other 2 with no act (another sample path) | fixed, live (1) / replay (2) |
| P4 | The question's precedence misread (`(-25)/X*1/(-4)` as `-25/(X/(-4))`) | round 9: 1 | `--exact-answer`, any box value | live: the act put `-198` over `-33` | fixed, live |
| P5 | `simplify_surd` loops counting parentheses | round 9: 2 (3480 x2; 9937#0 now right by the scorer fix) | `question_read.read_exact` + `--exact-answer` | live: 9937 x2 right with an act; 3480 x2 right with no act and no loop, at 2x the tokens | fixed, replay; live shows no harm |
| P6 | Minus lost after the answer-close cue: the cue ended in `{` and Qwen writes `{-` as one token | cue probe (2026-11-09): 22 of 28 gold-negative values | the cue ends before `{` | cue probe: minus on gold-negative values 41 -> 100 of 112 | fixed, live (probe) |
| P7 | Letters of the question's string miscounted or misquoted | round 9: 2 (`swr_p_sequence`) | Sandbox counting, with the string taken **from the question**, not from the model's copy | not tried | fix planned |
| P8 | A convention chosen wrong, extra roots kept (arccot) | round 9: 2 (`e3263989`) | Sandbox verify: substitute each candidate numerically | not tried on math | fix planned |
| P9 | A hard maximum guessed after a very long search (71k tokens) | round 9: 2 (`97b18f58`) | brute-force oracle from the statement (numeric search) | `--igc-sandbox-brute` never measured; not in tir4b | fix planned |
| P10 | The question refers to a figure the model cannot see | round 9: 2 (`370853d3`) | none | — | ceiling |
| P11 | The model's own value after `\approx` is wrong (`(1.01)^{30} \approx 1.3448888`) | round 9: 1; scan: 316 of 1,952 readable `\approx` values off | a put at `\approx` (needs its own design) | scan only | fix planned (design open) |
| P12 | The answer close boxes an unfinished step (`72 - 14 = 58`, a cross-product component) | round 9: 1 (`fbf10ed4#1`) | decide layer v1 gives a confirm then the close: likely still 58 | settle probe: 32 vs 31 of 49 (all loops) | **no fix yet** |
| P13 | The loop runs as long as before even when the closing value is fixed | targeted run: `2783#0` 70,772 tokens, `5070#1` 67,665 | decide layer: settle line with the close | settle probe: fewer tokens, no clear gain | fix planned |
| P14 | A put value read and not adopted: the model restarts the long division | A100 round 5 (`--loop-value`): 2 of 2 division rows | the box rule (`--exact-answer`) instead of a line before the close | P1 live run | replaced by P1's fix |
| P15 | Patterns MIP caused itself (put forms that started loops) | see the section below | each fixed in the put | replay gates | fixed |
| P16 | The rule scorer read a right answer as wrong (or a hedge as right) | round 9: 6 | the scorer rules | re-score of round 9 | fixed |

## Round 9, one by one

### P1. Right value, wrong form

- **What happened:** the model had the exact fraction on the page, then boxed a decimal (`-1.9886`, `0.004562376`,
  `10710884.384615...`) or the integer quotient (`-10710884`). The gold is the fraction in lowest terms.
- **Seen:** `dm_div_big` 1395 x2, 5070 x2, 4138 x2. Control lost the same rows to `stuck`.
- **Fix:** `--exact-answer` ([inject](inject.md#--exact-answer)): when `question_read` reads the whole question as one computation,
  the last box gets the exact value, at the answer-close cue or over the last box.
- **Tested:** replay on round 9: 24 acts, 16 wrong -> right, 0 right -> wrong. Targeted live run (2026-11-11, code `6c9331e`, 11 rows
  x 2 seeds): **22 of 22 right** against 7 of 22 for round 9's stored answers; all 9 `div_big` answers wrong in round 9 are right,
  each with an act. Not yet paired (next: the paired live test).

### P2. Two values, or a hedge

- **What happened:** `275 or -275` (`dm_mul_div_longer-5942#1`), both a fraction and a decimal offered (`div_big-9991` x2).
- **Fix and test:** as P1, at the cue. Live: `9991` x2 right. `5942#1` was scored right by the old rule, so it was not in the
  targeted run; replay right.

### P3. A wrong value from a loop, boxed at the close

- **What happened:** the answer looped, and the close boxed the value on the page: `4694/15` for 313 (`mixed_longer-9951#0`, a trial
  division that ran into zeros), `-136/3` for -136 (`mul_div_longer-9991#1`, a doubt loop on a true sub-step), `0` for
  `-171/2221234` (`div_big-2783#0`).
- **Fix:** `--exact-answer` puts the question's exact value after the cue.
- **Tested:** `2783#0` right live with an act. `9951#0` and `9991#1` came out right live **with no act**: the model took another path,
  so this run does not show the fix on them. The paired test has to.

### P4. Precedence misread

- **What happened:** `dm_mul_div_longer-1505#0`: `(-25)/X*1/(-4)` read as `-25/(X/(-4))`, boxed -12.375 (live run: -33); gold -198.
- **Fix:** `--exact-answer` widened: any last box whose value is not the exact value gets it, integers included (the developer:
  when the question is the computation, its value is the answer).
- **Tested:** live: the act wrote `-198` over `-33`; right.

### P5. Surd simplification loops

- **What happened:** `dm_simplify_surd` 3480 x2 looped counting parentheses in `(sqrt(...)...)**2`; the close boxed `9216` and
  `2633616 - 103680\sqrt{3}` (gold `884016 - 103680\sqrt{3}`).
- **Fix:** `question_read.read_exact` reads `Simplify <expression with square roots>` and computes `a + b\sqrt{n}` exactly (84 of 379
  stored rows read, 84 of 84 gold); `--exact-answer` then covers the module.
- **Tested:** live: 9937 x2 right with an act at the cue. 3480 x2 right with no act, no loop this time, 31k tokens against 13-17k in
  round 9.

### P7. Letters miscounted

- **What happened:** `dm_swr_p_sequence` 3480#0 (`1/66` for `1/78`) and 9951#1 (`7/55` for `56/495`): the model counted the letters of
  the question's string wrong from the start. On other rows of this module the model quoted the string wrongly
  (`vgcgp l cggp gvp` for `vgcgp lcgpgvp`, [decide layer](plans/2026-11-10-decide-layer-design.md)).
- **Expected fix:** a count by software. A Sandbox program helps only if the string it counts is the question's own: if the model
  copies its misquote into the code, the program counts the misquote. So the count reads the string from the question text. A reader
  for this one sentence shape would be tuned to DeepMind and is not used. The decide layer's B line (quote the question) missed both
  rows in simulation: no term of the loop is shared with the question.
- **Tested:** no.

### P8-P10. MATH level 5 reasoning

- `stdmath-e3263989` x2: the answer `3-2\sqrt{2}, 3+2\sqrt{2}, -3+2\sqrt{2}, ...` keeps roots that fail the arccot convention (gold
  `3\pm2\sqrt{2}`). **Expected:** a Sandbox check that substitutes each candidate into the equation numerically drops the false ones.
  The hypothesis-loop rules (`--igc-sandbox-verify`, `-expect`, `-why`) exist in the Strata engine only and were measured on
  LiveCodeBench, not on math ([sandbox](sandbox.md#hypothesis-loop-rules-draft-call-expect-why-gate-pass-close)).
- `stdmath-97b18f58` x2: a hard maximum, guessed (64 and 4 for 0), one answer 70,974 tokens. **Expected:** a brute-force numeric search
  from the statement as an oracle (`--igc-sandbox-brute`, engine only, never measured).
- `stdmath-370853d3` x2: the question's answer depends on a figure; boxed 7 for 14/3. **Ceiling:** no mechanism adds what the model
  cannot read.

### P11. The model's own `\approx` value

- **What happened:** `stdmath-45fa3ef2#0`: `(1.01)^{30} \approx 1.3448888` (true 1.3478489), so 32277 for 32348.
- **Measured** (`results/exact-answer-r9/approx_scan.py`): General wrote 3,391 `\approx` with a decimal; the computation before it is
  readable on 1,952, and the model's value is off on 316. The 316 mix real slips, scientific notation read as its mantissa, and
  deliberate estimates (`\approx 34000`). The `=` put's forms are wrong after `\approx`, and `(1.01)^{30}` is not read at all.
- **Expected fix:** a put at `\approx` with its own design: decimal places from what the model writes next, `\times 10^{k}` read, a
  rule for estimates. **Not built.**

### P12. The close boxed an unfinished step

- **What happened:** `stdmath-fbf10ed4#1`: the model looped doubting a true `72 - 14 = 58`, one component of a cross product; the
  answer close boxed 58 (gold 7). Control got it right.
- **Why the planned fix is not enough:** the decide layer v1 reads this as class A (a true equation) and puts a confirm, then closes
  as today, one beat later. Detecting unfinished work from the structure (a cross product with a component left) is deferred in the
  [decide layer plan](plans/2026-11-10-decide-layer-design.md#5-deferred-with-the-reason): no signal yet.
- **Needed:** a signal that the boxed value is a step, not the answer: the question asks for one quantity and the loop's value is a
  sub-result of it. **No fix yet.**

### P13. Long loops even with the right closing value

- `div_big-2783#0` 70,772 tokens, `div_big-5070#1` 67,665 in the targeted run. `--exact-answer` fixes the value at the close, not the
  loop's length. The settle probe (2026-11-10): settle line and close together 32 of 49, today's close 31, fewer tokens.

## Patterns MIP itself caused (fixed in the put)

| Pattern | Seen | Fix |
|---|---|---|
| A put spelled `9/2` for `0.9 \times 5` broke the model's decimal sum into a loop | DeepMind 6890 | `spelled()`: a terminating decimal is put as a decimal, the model's own spelling |
| A put on `= 15/13 (exact)` after `= 15/13` repeated: `= 15/13 = 15/13 = ...` until `stuck` | #306 | nothing new is put; the value is left to the model and the loop checks |
| The same expression put back to back many times (100 times `-1/2` on `livebench-652451e2`) | #13 | echo guard: the 3rd put in a row within 64 tokens is skipped |
| A nudge that cut 64 tokens of a line-back loop left the loop and ended `stuck` | #304 | nudge dropped; the close acts |
| The answer-close cue ending in `{` lost minus signs (P6) | cue probe | cue ends before `{` |

## Rule scorer errors (fixed)

| Answer | Read as | Fix |
|---|---|---|
| `dm_div_big-9977#0`, `dm_simplify_surd-9937#0` | a box asked about and rejected (`\boxed{70468}? No`) taken as the answer | a box followed by `?` on its line is not the answer (#24) |
| `dm_mul_div_longer-5942#1` | `275 or -275` counted right | two different values in the last box line: no answer |
| `stdmath-2aff3e30#0` | `0.2` for `1/5` in a vector counted wrong | vectors compared entry by entry |
| `stdmath-9f7f1563#0` | `3<\lambda\le4` for `(3,4]` counted wrong | an inequality read as an interval |
| `stdmath-bf7ece2b` x2 (a3) | `\{-2, 1-\sqrt{5}, 1+\sqrt{5}\}` for `\{1\pm\sqrt{5},-2\}` counted wrong | set braces stripped before the values are compared (`2fcfbb9`) |
| `dm_simplify_surd-3480#0` | the API reader counted `9216 if the sum is 0` right | an API reader error; the rule is right |

## ภาษาไทย

หน้านี้บันทึกทุกรูปแบบที่ทำให้โมเดลตอบผิดซึ่งอ่านด้วยมือแล้ว ว่าเกิดอะไร เจอที่ไหน แก้ยังไง หรือคาดว่าวิธีไหนจะช่วย และทดสอบแล้วหรือยัง
รูปแบบที่แก้แล้วยังอยู่ในหน้านี้ เพื่อใช้ตรวจโมเดลหรือรอบถัดไป ตามกฎของผู้พัฒนา: รูปแบบที่เกิดขึ้นได้ต้องแก้ทุกแบบ ไม่ว่าจะเจอน้อยแค่ไหน

| # | รูปแบบ | เจอ | วิธีแก้ | ทดสอบแล้ว? | สถานะ |
|---|---|---|---|---|---|
| P1 | ค่าถูกแต่ตอบเป็นทศนิยมหรือส่วนจำนวนเต็มแทนเศษส่วนเป๊ะ | รอบ 9: 6 | `--exact-answer` | รันเฉพาะข้อ ถูก 6/6 | แก้แล้ว รันจริงแล้ว |
| P2 | ตอบสองค่า หรือลังเล | รอบ 9: 3 | `--exact-answer` ตอนปิด | รันจริงถูก 2, replay 1 | แก้แล้ว |
| P3 | ค่าผิดจาก loop ถูกใส่กล่องตอนปิด | รอบ 9: 3 | `--exact-answer` ตอนปิด | ถูกทั้งหมด แต่ 2 ข้อถูกโดย flag ไม่ได้ทำงาน | แก้แล้ว (ต้องรอจับคู่ยืนยัน) |
| P4 | อ่านลำดับการคำนวณในโจทย์ผิด | รอบ 9: 1 | `--exact-answer` ทุกค่า | รันจริง: `-33` → `-198` | แก้แล้ว รันจริงแล้ว |
| P5 | `simplify_surd` วนกับการนับวงเล็บ | รอบ 9: 2 | `read_exact` + `--exact-answer` | replay แล้ว, รันจริงไม่ทำให้ผิด | แก้แล้ว |
| P6 | เครื่องหมายลบหายหลังประโยคปิด (`{-` เป็น token เดียว) | cue probe | ประโยคปิดจบก่อน `{` | 41 → 100 จาก 112 | แก้แล้ว |
| P7 | นับตัวอักษรในสตริงของโจทย์ผิด | รอบ 9: 2 | Sandbox นับจากสตริงในโจทย์เอง | ยัง | วางแผนไว้ |
| P8 | เลือกนิยามผิด เก็บรากเกินมา (arccot) | รอบ 9: 2 | Sandbox แทนค่าตรวจทีละค่า | ยัง | วางแผนไว้ |
| P9 | เดาค่าสูงสุดหลังค้นนานมาก | รอบ 9: 2 | brute-force oracle | ยัง (ยังไม่เคยวัด) | วางแผนไว้ |
| P10 | โจทย์อ้างถึงรูปที่โมเดลมองไม่เห็น | รอบ 9: 2 | ไม่มี | — | เพดานของโมเดล |
| P11 | ค่าที่โมเดลเขียนหลัง `\approx` ผิด | รอบ 9: 1, สแกนพบ 316 จาก 1,952 | ใส่ค่าหลัง `\approx` (ต้องออกแบบใหม่) | สแกนอย่างเดียว | วางแผนไว้ |
| P12 | ตัวปิดเอาค่ากลางทางของงานที่ยังไม่เสร็จไปตอบ | รอบ 9: 1 | decide layer v1 น่าจะยังไม่พอ | settle probe 32 ต่อ 31 | **ยังไม่มีวิธีแก้** |
| P13 | loop ยังยาวเท่าเดิมแม้ค่าตอนปิดถูก | รอบรันเฉพาะข้อ: 70k token | decide layer (settle พร้อมปิด) | probe: token ลดลง ผลยังไม่ชัด | วางแผนไว้ |
| P14 | โมเดลอ่านค่าที่ใส่ให้แต่ไม่ใช้ เริ่มหารยาวใหม่ | A100 รอบ 5 | ใช้กฎแก้กล่องแทน | ผลรันจริงของ P1 | ถูกแทนด้วย P1 |
| P15 | รูปแบบที่ MIP ทำให้เกิดเอง | ดูตาราง | แก้ที่วิธีใส่ค่า | replay | แก้แล้ว |
| P16 | ตัวตรวจคะแนนด้วยกฎตัดผิด | รอบ 9: 6 | แก้กฎ | ตรวจรอบ 9 ใหม่ | แก้แล้ว |
