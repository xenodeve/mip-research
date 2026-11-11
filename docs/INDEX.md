# MIP research catalog

Everything built and measured in this research, by category, with every solid benchmark in one place. The
[README](../README.md) tells the best config's story; this catalog lists **every** mechanism, including the many that are built and
measured but not part of the best config, and says where each lives and what it takes to use it. What MIP (Model Integrated
Protocol) is, how it differs from MCP, and why every result so far comes from the reference outside the engine:
[README](../README.md#what-mip-is-and-where-it-stands).

Claims in the issues that were later withdrawn, corrected or never confirmed: [ERRATA](ERRATA.md). Check it before quoting an issue.

Every pattern that made the model wrong, read by hand, with its fix and whether the fix was tested: [failure patterns](failures.md).

[ภาษาไทย](#ภาษาไทย)

## Categories

| Page | What it covers | Mechanisms | In best config | Engine only | Kept as option | Not measured yet / spec | Paused |
|---|---|---|---|---|---|---|---|
| [Detection](detect.md) | Where a calculation is or is about to be: the `=` stop and reader, the question reader, the engine's token parser and MTP-draft interception, natural-language levels, probes, router, readiness | 13 | 2 | 4 | 5 | 2 | 0 |
| [Inject](inject.md) | How a value or note gets into the model: the General Style put, refusals, digit-join hold, echo guard, arithmetic mode, check-hand, logit override, hints, the engine's result bridge and inline execution, notes | 17 | 5 | 4 | 6 | 2 | 0 |
| [Anti-loop](antiloop.md) | Loop detectors and closers: repeat check, counting guard, line-back, A, B, V7, shape cap, cover cap, the engine's own closes | 18 | 9 | 6 | 2 | 1 | 0 |
| [Compute](compute.md) | What computes the exact value: MathBackend, the inline bridge, the audit, the engine's CUDA executors ABI1-3, Qwenputer, oracles | 11 | 3 | 5 | 2 | 1 | 0 |
| [Sandbox and tools](sandbox.md) | Code run inside the reasoning: the C1 interpreter, forced block and close, hypothesis loop, tool route, real-Python box, workspace, MCP adapter, 4B port | 14 | 0 | 2 | 7 | 5 | 0 |
| [Learned and latent](learned.md) | Heads on the hidden state: observation, input readout, readiness, output mapper, controller, residual read/write, SEAL | 10 | 0 | 0 | 1 | 1 | 8 |
| [Benchmarks and method](benchmarks.md) | Every set and result with its setup (master table), and the runners, gates, readers and pairing rules | 16 | 5 | 3 | 5 | 3 | 0 |

Status words, used on every page: **in best config**; **engine only** (Strata-xeno branch `xeno/mip-sandbox`, not in `main`);
**built, measured, kept as option**; **built, not measured yet**; **spec only**; **built, measured, paused**.

## Model builds

A result holds only for the build it ran on; every result row names one of these labels, which give the weights, the engine (vLLM,
llama.cpp's llama-server, Strata-xeno, or HF transformers) and the machine. Two builds are never pooled in one
number, and a run that mixes them (one build's text continued by another) says so where its number is.

| Label | Weights and format | Runner, machine | Used for |
|---|---|---|---|
| **4B bf16 · vLLM · A100** | Qwen3.5-4B official weights, bf16 | `tir4b` on vLLM 0.33, Colab A100 80 GB (one round on an L4, labelled where it is) | every A100 round 1-8, the loop rows, the replay gate's stored arms |
| **4B bf16 GGUF · llama.cpp · 5060 Ti** | Qwen3.5-4B converted to GGUF, bf16 (8.67 GB; the file was in a scratch folder and is gone, the launch lines name it) | `tir4b --server`, llama-server, RTX 5060 Ti | DeepMind 160 (pc4-pc9), the PC smoke runs before 2026-11-08 |
| **4B NVFP4 W4A8 · llama.cpp · PC layer** | AxionML/Qwen3.5-4B-NVFP4 (ModelOpt), converted to GGUF (3.5 GB); activations q8_1 (W4A8) on both cards through a build for sm_89 only (llama.cpp `1deefcca3`) | `tir4b --server`, llama-server, RTX 5060 Ti and RTX 4070 SUPER, `-c 262144 -kvu --no-cache-idle-slots`, a slot for every answer in flight, KV q8_0 (round 9 ran `-np 4` without the flag) | the PC layer between the simulation and the A100, from round 9 (2026-11-08) |
| **4B Q8_0 GGUF · llama.cpp · 5060 Ti** | Qwen3.5-4B GGUF, Q8_0 | llama-server, RTX 5060 Ti | speed sweeps only (slots, context); no accuracy result |
| **4B Q8_0 · Strata-xeno · 5060 Ti** | Qwen3.5-4B as `qwen35dense` (Q8_0) | Strata engine, RTX 5060 Ti | the engine port's decode speed only; no MIP run |
| **4B bf16 · HF transformers** | Qwen3.5-4B official weights, bf16 | Hugging Face transformers (A100, or this PC) | the probes, SEAL, residual read/write |
| **Flash-Next IQ2_XS · Strata-xeno** | Qwen3.8-Flash-Next, Gyro-M Swift 1.5 IQ2_XS | Strata engine (D2x on this PC, or one Colab A100) | the engine pipeline's results |

NVFP4 loops more often than bf16, so the PC layer is the harder test of the anti-loop software and the result a user with one
consumer card gets; a PC-layer number is not an A100 number, and the other way round.

## Two pipelines

MIP runs in two places that share the rules but not the code:

| | Best config | Strata engine |
|---|---|---|
| Model | Qwen3.5-4B: **4B bf16 · vLLM · A100** (the best config's results), **4B bf16 GGUF · llama.cpp · 5060 Ti**, **4B NVFP4 W4A8 · llama.cpp · PC layer** ([builds](#model-builds)) | Qwen3.8-Flash-Next (Gyro-M Swift 1.5 IQ2_XS) |
| Runner | [`tir4b.py`](../tools/igc/tir4b.py) on vLLM (Colab A100) or llama-server (RTX 5060 Ti) | `strata.exe` built from Strata-xeno `xeno/mip-sandbox`, driven by [`external_bench.py`](../tools/igc/external_bench.py) |
| Detect | each request stops at an `=` token; Python reads the text ([detect](detect.md#the--stop-and-the-expression-reader)) | `parse_inline` on token ids inside the decode loop; the token after `=` is replaced before it enters the KV cache and the MTP draft after it is dropped ([detect](detect.md#engine-interception-before-the-kv-cache-and-the-mtp-draft)) |
| Put | General Style text, a new request goes on from it ([inject](inject.md#inline-put-general-style)) | the value's tokens go in as one forced draft window of up to the MTP window size ([inject](inject.md#engine-inline-execution)) |
| Compute | CPU Python: `mathexpr`, `exactexpr`, MathBackend ([compute](compute.md#mathbackend)) | three CUDA executors, integer / power / rational-decimal ([compute](compute.md#abi1-integer-----on-cuda)) |
| Loops | repeat, counting, line-back, A, B, V7, shape cap, cover cap ([anti-loop](antiloop.md)) | `in_cycle` to period 4,096, counting, line, value-count, third-repeat close, repeat note ([anti-loop](antiloop.md)) |
| Sandbox | port built (`--force-after --execute --box`), not in the best config | C1 interpreter, forced block, close after a print, tool route ([sandbox](sandbox.md)) |

The best config's [README numbers](../README.md#results) come from the first column only. The second column carries the
[Flash-Next results](#headline-results). Moving the best config onto the engine means porting the General Style, its refusals and
the closers A, B, V7, shape cap and cover cap into the decode loop, and setting their thresholds again on engine traces.[^port]

## Headline results

Only solid rows; each links to the page that holds its setup, and the [master table](benchmarks.md#master-results-table) has the rest.

| Set | Model, machine | Mechanism | Result | Page |
|---|---|---|---|---|
| DeepMind arithmetic, 160 rows, thinking off, greedy | 4B bf16 GGUF · llama.cpp · 5060 Ti | arithmetic mode | **156 vs 118** right; **12,062 vs 261,731** tokens (0.046x)[^dm160] | [inject](inject.md#arithmetic-mode) |
| Loop rows, 24 x 2, T 0.6, full context | 4B bf16 · vLLM · A100 | best-config closers (A + B + V7) | **37 vs 24** of 48 right; answers with no answer **1 vs 21**[^loop48] | [anti-loop](antiloop.md) |
| Same rows, median tokens of right answers | same | same | MATH L5 **15,260 → 9,090** | [benchmarks](benchmarks.md#master-results-table) |
| Round 9 frozen set (frozen9: 8 DeepMind modules x 10 + MATH-500 L5 x 40) x 2 seeds, T 0.6, full context | **4B NVFP4 W4A8 · llama.cpp · PC layer** | General Style + the closers (best config), against control | **218 vs 161** of 240 paired answers by the rule (API reader **216 vs 161**); General alone right 62, control alone 5, sign test p = 1e-13; 3 of General's are a correct decimal of a gold with no finite decimal, counted right with a form note (`-1.9886`, `-1.9887` for `-73388/36903`; `0.00456226` for `5647/1237763`); tokens 3.45 M vs 3.42 M[^r9] | [benchmarks](benchmarks.md#4b-nvfp4-w4a8--llamacpp--pc-layer-rtx-5060-ti--rtx-4070-super) |
| Rerun of the 52 rows any arm got wrong, 104 answers | same | Qwen profile + `--no-digit-join` + `--guard-progress` | **67 → 79** right (control 73 → 71)[^unpaired] | [inject](inject.md#model-profiles) |
| 5-digit multiplication, 40, ABBA | Flash-Next IQ2_XS · Strata-xeno · this PC (D2x) | `--igc-result-bridge --igc-inline-exec` | tokens **137, 142 vs 491, 522**; seconds **2.80, 2.82 vs 6.25, 6.61**; right 39, 39 vs 40, 39 | [inject](inject.md#engine-inline-execution) |
| GSM-Hard, first 200 | same | same | 177 vs 178 right; **571 vs 731** tokens; decode **1,942 vs 2,363 s** | [benchmarks](benchmarks.md#master-results-table) |
| Long-number word problems, 48 | same | same | 48 / 48 both; **249 vs 457** tokens; 4.21 vs 6.58 s | [detect](detect.md#engine-parser-parse_inline) |
| BIG-bench 3- and 5-digit + − ×, 240 | same | same | 240 / 240 both; 5-digit × **146 vs 485** tokens[^bigbench] | [compute](compute.md#engine-inline-execution---igc-inline-exec) |
| Two-digit + − × (EleutherAI), 120 | same | token-result bridge | 120 / 120 both; **14.0 / 13.1 / 16.4 vs 35.4 / 32.9 / 54.8** tokens | [inject](inject.md#engine-token-result-bridge) |
| 49 questions (29 GSM8K + 20 MATH) x 2 rounds | same | forced block at 512 + close after a print | **77 vs 73** of 98 right; **125,608 vs 247,162** tokens (0.508x)[^q49] | [sandbox](sandbox.md#forced-block--close-after-a-print) |
| MATH long 22 / GPQA 13 / SuperGPQA calc 29, T 0.6 | Flash-Next IQ2_XS · Strata-xeno · A100 | tool route + MIP | right 21/21, 11/12, 10/11; tokens **0.81x / 0.73x / 0.71x**[^tool] | [sandbox](sandbox.md#tool-route-the-models-own-run_python) |
| LiveCodeBench hard 20, paired, greedy | same | MIP values + loop checks | **18 vs 17** right at **0.40x** tokens[^lcb] | [benchmarks](benchmarks.md#master-results-table) |
| H3 frozen fresh set, 1,526 traces of 4 other models | offline replay | General Style put | `=` caught **94.4 %**; **0 false of 41,176** puts; next value = put 31,621 / 32,812 | [inject](inject.md#inline-put-general-style) |
| A100 best-config arms | 4B bf16 · vLLM · A100 | audit of every put | **0 wrong of 39,046** checked puts | [compute](compute.md#audit-of-every-value-put) |
| MathBackend against SymPy / mpmath / NetworkX | none | 387 operations | **31,064 cases, 0 disagreements** | [compute](compute.md#mathbackend) |
| CUDA executors ABI2 / ABI3 corpora | none | engine executors | **1,271 / 1,271** and **1,403 / 1,403** byte-exact, sanitizers clean | [compute](compute.md#abi2-floor-division-mod-gcd-power-on-cuda) |
| 46 traces, 3,176 computation starts, 2 tokens ahead | traces 4B bf16 · vLLM · A100; read by 4B bf16 · HF transformers / Qwen3.6-35B-A3B | residual probe / MoE router | **89 % / 87 %** probe, **83 %** router, text alone 53 % | [detect](detect.md#predictive-probes-and-the-moe-router) |
| Replay of stored A100 traces | 4B bf16 · vLLM · A100 | V7 / cover cap gates | **0 of 915** / **0 of 1,078** right answers touched | [anti-loop](antiloop.md) |
| HumanEval held-out half | C1 interpreter vs CPython 3.11.9 | sandbox | **78 of 82** match, **0 mismatch** | [sandbox](sandbox.md#c1-bounded-python-interpreter) |

## Built, measured, not in the best config

The mechanisms most worth carrying over, with what each needs. Every one has its section on its page.

| Mechanism | Page | Measured | What it takes |
|---|---|---|---|
| Engine detect + MTP draft drop + one-window put | [detect](detect.md#engine-interception-before-the-kv-cache-and-the-mtp-draft), [inject](inject.md#engine-inline-execution) | Flash-Next rows above | merge `xeno/mip-sandbox` into Strata-xeno `main`; CUDA build with the IGC option |
| Engine CUDA executors ABI1-3 | [compute](compute.md#abi1-integer-----on-cuda) | byte-exact corpora, 0.066-0.093 ms per ABI1 call | same build; only `+ − ×`, power, rationals and decimals are reached from text |
| Arithmetic mode | [inject](inject.md#arithmetic-mode) | 156 vs 118 | a calculation question with thinking off; `--whole-hint --whole-plain --question-value --hand-off` |
| Question reader | [detect](detect.md#question-reader-catch-point-1) | 160 / 160 read | part of arithmetic mode (`--question-value`) |
| Qwen model profile | [inject](inject.md#model-profiles) | 67 → 79 on the rerun rows | `--inline-style qwen`; a profile per family on top of the General Style |
| Forced block + close after a print | [sandbox](sandbox.md#forced-block--close-after-a-print) | 0.508x tokens on 49 questions | engine `--igc-sandbox --igc-sandbox-after 512 --igc-sandbox-close`; off for coding |
| Tool route (`run_python`) | [sandbox](sandbox.md#tool-route-the-models-own-run_python) | 0.71-0.81x tokens at equal right answers | engine or `external_bench --tool-loop` |
| Predictive probes and router | [detect](detect.md#predictive-probes-and-the-moe-router) | 83-89 % seen 2 tokens ahead | wire `UnifiedDetector` into a runner; signals may only arm, the exact parse acts |
| Engine closes (`in_cycle` 4,096, third-repeat close, repeat note) | [anti-loop](antiloop.md) | threshold scans of 7,000-12,000 stored runs | part of the engine build; not timed with the best-config closers |

## Notes

[^port]: หมายเหตุ: the closers' thresholds were set from vLLM traces of Qwen3.5-4B; no engine run with them exists yet. vLLM runs do
    not repeat token for token across runs (mip-research#21); the A100 Strata engine repeated 49/49 greedy (Strata-xeno#283),
    and whether it repeats a sampled run with the same seed is not measured yet.
[^dm160]: หมายเหตุ: thinking off, 4,096-token cap; the hand-off rules were set while reading these 160 rows, so 156 is a development
    result. Records: [`results/arithmetic-mode`](../results/arithmetic-mode/).
[^r9]: หมายเหตุ: 4B NVFP4 W4A8 only, not an A100 number. Each card ran both arms on its own half of the rows (half A on the 5060 Ti, half B on the 4070 SUPER); 4 of half A's MATH-500 rows (8 pairs, a3) moved to the 4070 and finished there: General 8 of 8, control 4 of 8 (4 `stuck`). Scored by the rule as fixed 2026-11-10/11 (`96cf412`: a `? No` box, a hedge, `0.2` for `1/5`, an inequality for an interval; `2fcfbb9`: a set in set braces); the first count, 204 vs 157 of 232, was the older rule. Of the 7 control-only answers, `stdmath-fbf10ed4#1` is the answer close boxing an unfinished step (58, a cross-product component) and `dm_div_big-9977#0` the cue dropping the minus (`{-` is one Qwen token); see [ERRATA](ERRATA.md).
[^loop48]: หมายเหตุ: the plain arm (24 / 48) is round 1's control, a different run from the closer arms; the inline-only arm of the
    same round had 25 / 48 and 23 with no answer. Two answers were lost to V7 closing too early ([anti-loop](antiloop.md)).
[^unpaired]: หมายเหตุ: a rerun of rows chosen because some arm got them wrong; not paired token for token across runs (mip-research#21).
[^bigbench]: หมายเหตุ: measured on a build before the parser fix that refuses a number written against a letter; the same comment
    records 2 wrong delivered values ([compute](compute.md#engine-inline-execution---igc-inline-exec)).
[^q49]: หมายเหตุ: +4 right is p 0.29, so the result is "no loss" at half the tokens; one plain run was stopped by the bench on a 2-token
    cycle before the engine's own close could act, so the plain arm's tokens are slightly low and one of its answers is unknown
    (mip-research#6); the close stays off by default because it cut
    coding short ([sandbox](sandbox.md#forced-block--close-after-a-print)).
[^tool]: หมายเหตุ: the saving comes mostly from the tool's prompt line; no call was made inside the thinking on these sets
    ([sandbox](sandbox.md#tool-route-the-models-own-run_python)).
[^lcb]: หมายเหตุ: greedy, one paired run; every sandbox close lost on the same 20 rows (14, 15, 13 right) ([benchmarks](benchmarks.md#master-results-table)).

## ภาษาไทย

แคตตาล็อกนี้รวมทุกอย่างที่สร้างและวัดผลในงานวิจัยนี้ แยกตามหมวด และรวม benchmark ที่ยืนยันได้ไว้ในที่เดียว
[README](../README.md) เล่าเรื่องของ best config ส่วนแคตตาล็อกนี้ลิสต์**ทุก**กลไก รวมถึงตัวที่สร้างและวัดแล้วแต่ไม่ได้อยู่ใน best config
พร้อมบอกว่าแต่ละตัวอยู่ที่ไหนและต้องทำอะไรถึงจะใช้ได้ หน้าหมวดย่อยเขียนเป็นภาษาอังกฤษ MIP (Model Integrated Protocol) คืออะไร
ต่างจาก MCP ตรงไหน และทำไมผลทุกรอบตอนนี้มาจากตัวต้นแบบที่อยู่นอก engine อ่านได้ที่ [README](../README.md#mip-คืออะไร-และตอนนี้อยู่ตรงไหน)

### หมวด

| หน้า | ครอบคลุม | กลไก | ใน best config | เฉพาะ engine | เก็บเป็นตัวเลือก | ยังไม่วัด / spec | พักไว้ |
|---|---|---|---|---|---|---|---|
| [Detect](detect.md) | หาว่าการคำนวณอยู่ตรงไหนหรือกำลังจะมา: หยุดที่ `=` ตัวอ่านโจทย์ parser ใน engine ที่ดักก่อน KV และทิ้ง MTP draft ระดับภาษาพูด probe router readiness | 13 | 2 | 4 | 5 | 2 | 0 |
| [ใส่ค่า](inject.md) | ค่าหรือหมายเหตุเข้าไปใน model ยังไง: put แบบ General Style กฎปฏิเสธ กันเลขต่อกัน echo guard arithmetic mode check-hand logit hint result bridge และ inline exec ของ engine | 17 | 5 | 4 | 6 | 2 | 0 |
| [ปิด loop](antiloop.md) | ตัวจับและตัวปิด loop: repeat counting line-back A B V7 shape cap cover cap และตัวปิดของ engine เอง | 18 | 9 | 6 | 2 | 1 | 0 |
| [ตัวคำนวณ](compute.md) | ใครคำนวณค่าที่แม่น: MathBackend สะพาน inline การตรวจทุก put ตัวคำนวณ CUDA ABI1-3 ของ engine Qwenputer oracle | 11 | 3 | 5 | 2 | 1 | 0 |
| [Sandbox และเครื่องมือ](sandbox.md) | รันโค้ดในส่วนคิด: ตัวแปลภาษา C1 บังคับเปิด block แล้วปิด hypothesis loop tool route กล่อง Python จริง workspace MCP adapter และพอร์ตสำหรับ 4B | 14 | 0 | 2 | 7 | 5 | 0 |
| [Learned และ latent](learned.md) | head ที่อ่าน hidden state: observation readout readiness output mapper controller การอ่านเขียน residual SEAL | 10 | 0 | 0 | 1 | 1 | 8 |
| [Benchmark และวิธีทดสอบ](benchmarks.md) | ทุกชุดทดสอบและผลพร้อม setup (ตารางรวม) ตัวรัน gate ตัวอ่านคำตอบ และกฎการจับคู่ | 16 | 5 | 3 | 5 | 3 | 0 |

### รุ่นของโมเดล

ผลแต่ละตัวใช้ได้กับรุ่นและ engine ที่รันจริงเท่านั้น (vLLM, llama.cpp หรือ Strata-xeno) ทุกแถวของผลบอกชื่อรุ่นตามตาราง [Model builds](#model-builds) ด้านบน ไม่รวมสองรุ่นเป็นตัวเลขเดียว
และถ้ารันแบบผสม (ข้อความของรุ่นหนึ่งให้อีกรุ่นเขียนต่อ) ต้องบอกไว้ตรงตัวเลขนั้น

- **4B bf16 · vLLM · A100:** น้ำหนักทางการ bf16 รันบน vLLM ใน Colab A100 ผลรอบ 1-8 ทั้งหมด
- **4B bf16 GGUF · llama.cpp · 5060 Ti:** GGUF แบบ bf16 บน llama-server ในเครื่องนี้ ผล DeepMind 160 ข้อ
- **4B NVFP4 W4A8 · llama.cpp · PC layer:** AxionML NVFP4 แปลงเป็น GGUF รันบน 5060 Ti + 4070 SUPER ด้วย build สำหรับ sm_89 เพื่อให้ทั้งสองการ์ดเป็น W4A8 เหมือนกัน เริ่มที่รอบ 9
- **4B Q8_0:** บน llama.cpp และบน Strata-xeno (port `qwen35dense`) ใช้วัดความเร็วอย่างเดียว ไม่มีผลความแม่น
- **Flash-Next IQ2_XS · Strata-xeno:** ผลทั้งหมดของฝั่ง engine

NVFP4 วนบ่อยกว่า bf16 จึงเป็นบททดสอบที่หนักกว่าสำหรับระบบกันวน และเป็นผลที่ผู้ใช้การ์ดทั่วไปได้จริง ตัวเลขของชั้น PC จึงเอาไปแทนตัวเลขของ A100 ไม่ได้ และกลับกันก็ไม่ได้

### สอง pipeline

MIP รันอยู่สองที่ ใช้กฎชุดเดียวกันแต่โค้ดคนละชุด

- **Best config:** Qwen3.5-4B รันผ่าน `tir4b.py` บน vLLM หรือ llama-server ทุก request หยุดที่ `=` ให้ Python อ่านแล้วต่อค่าเป็นข้อความ
  คำนวณบน CPU ด้วย MathBackend และปิด loop ด้วย A B V7 shape cap cover cap ตัวเลขใน README มาจากฝั่งนี้ทั้งหมด
- **Engine ของ Strata:** Flash-Next บน branch `xeno/mip-sandbox` ยังไม่เข้า `main` parser อ่าน token id ใน decode loop
  ดัก token หลัง `=` ก่อนเข้า KV แล้วทิ้ง MTP draft ที่ตามมา ใส่ค่าเป็น window เดียวยาวสุดเท่า window ของ MTP
  คำนวณด้วย CUDA 3 ตัว และมีตัวปิด loop กับ sandbox ของตัวเอง

ถ้าจะใช้ best config บน engine ต้องย้าย General Style กฎปฏิเสธ และตัวปิด A B V7 shape cap cover cap ลงไปใน decode loop
แล้วตั้งเกณฑ์ใหม่จาก trace ของ engine[^th-port]

### ผลเด่น

ตารางผลเด่นด้านบน (Headline results) ใช้ได้กับทั้งสองภาษา สรุปสั้นๆ:

- **Arithmetic mode (4B bf16 GGUF · llama.cpp · 5060 Ti):** ถูก 156 จาก 160 เทียบ 118 ใช้ token 12,062 เทียบ 261,731 (0.046 เท่า)
- **ตัวปิด loop ของ best config (4B bf16 · vLLM · A100):** ถูก 37 เทียบ 24 จาก 48 คำตอบที่ไม่มีคำตอบลดจาก 21 เหลือ 1
- **รอบ 9 ชุดแช่แข็ง (4B NVFP4 W4A8 · llama.cpp · PC layer):** ครบ 240 คู่ ตามกฎ General ถูก 218 เทียบ control 161 ตัวอ่านคำตอบ (API) ให้ 216 เทียบ 161 General ถูกฝ่ายเดียว 62 ข้อ control ถูกฝ่ายเดียว 5 ข้อ p = 1e-13 (3 ข้อเป็นทศนิยมที่ถูกแต่รูปไม่ตรงเฉลย นับถูกพร้อมหมายเหตุ)
- **Flash-Next บน engine:**
  - คูณ 5 หลัก token ลด 72% เวลาลด 56%
  - GSM-Hard token 571 เทียบ 731
  - บังคับเปิด block แล้วปิดหลังพิมพ์ผล ใช้ token 0.508 เท่า ถูก 77 เทียบ 73 จาก 98
  - tool route ใช้ token 0.71–0.81 เท่า ถูกเท่ากันภายใน 1 ข้อ
- **ความแม่นของค่า:**
  - ใส่ค่าผิด 0 จาก 41,176 ครั้งบนชุด H3 และ 0 จาก 39,046 ครั้งบน A100 (4B bf16)
  - MathBackend ไม่ขัดกับ SymPy เลยใน 31,064 เคส
- **เห็นการคำนวณล่วงหน้า 2 token:** probe 89% และ 87% router 83% ข้อความล้วน 53%

### สร้างและวัดแล้ว แต่ยังไม่อยู่ใน best config

ตาราง "Built, measured, not in the best config" ด้านบนบอกว่าแต่ละตัวต้องใช้อะไร ตัวที่ควรเอามาใช้ก่อนมีดังนี้

- **ตัวดักและใส่ค่าของ engine ที่ทำงานร่วมกับ MTP:** ต้อง merge `xeno/mip-sandbox` เข้า `main`
- **ตัวคำนวณ CUDA:** มากับ build เดียวกัน
- **Arithmetic mode และตัวอ่านโจทย์:** ใช้กับโจทย์คำนวณที่ปิดการคิด
- **Qwen profile:** ซ้อนบน General Style
- **Forced block และ tool route:** ใช้ใน engine
- **Probe และ router:** ต้องต่อ `UnifiedDetector` เข้าตัวรัน
- **ตัวปิด loop ของ engine**

[^th-port]: หมายเหตุ: เกณฑ์ของตัวปิดตั้งจาก trace ของ Qwen3.5-4B บน vLLM ยังไม่เคยรันบน engine และ vLLM ไม่ให้ผลซ้ำแบบ token ต่อ token
    ข้ามรอบ (mip-research#21) ส่วน engine ของ Strata บน A100 ให้ผลซ้ำ 49/49 ตอนรันแบบ greedy (Strata-xeno#283) แต่ยังไม่ได้วัดว่ารันแบบ sampled
    ด้วย seed เดิมแล้วซ้ำไหม
