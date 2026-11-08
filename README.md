# mip-research

Research on **MIP** (Model Inline Processing): software computes, the model reasons. While a model thinks, MIP reads the expression
it writes before `=`, puts the exact value after it, and closes the loops a small model falls into. This repo holds the research
harness and the mechanisms; [Strata-xeno](https://github.com/xenodeve/Strata-xeno) runs them in its engine and server for
Qwen3.8-Flash-Next and Qwen3.6-35B-A3B-NVFP4.

Split out of Strata-xeno on 2026-11-08 with its git history (`git filter-repo`); the issues moved with it. Paths are unchanged
(`tools/igc`, `tools/qwenputer`), so every command in the issues still runs.

## What is here

| Path | What |
|---|---|
| `tools/igc/tir4b.py` | The reference harness: Qwen3.5-4B in vLLM, inline values (General Style, `--inline-style general`) and every loop closer (`--loop-close`, `--answer-close`, `--early-cycle`, `--shape-cap`, `--cover-cap`) |
| `tools/igc/sim_replay.py` | Replays stored traces with a mechanism added, on a PC: where it acts, on how many cut answers, whether it ever acts on a right answer, and what vLLM would refuse |
| `tools/igc/mathbackend/` | The exact computation behind a put: numbers, algebra, counting, geometry, search |
| `tools/igc/answer_check.py` | The API reader beside the rule scorer (its key stays on the machine that runs it, never on a VM) |
| `tools/igc/` (rest) | Readouts, adapters, corpora, benchmarks (LiveBench, LCB, standard sets) |
| `tools/qwenputer/` | Native Qwenputer: computation compiled for the engine |
| `third_party/` | Qwenputer upstream (its own LICENSE) and the mathbackend notices |

## The boundary with Strata-xeno

The engine side of MIP stays in Strata-xeno: `include/strata/core/igc_inline.hpp`, `igc_sandbox_bridge.hpp`, `igc_adapter_schema.hpp`,
`igc_binding_tokens.hpp` and `src/kernels/qwenputer.cu`. Generators here write those headers; `tools/test_igc_adapter_artifact.py`
checks the schema header against a Strata-xeno checkout when `STRATA_ROOT` names one. `tir4b --execute/--box` and
`external_bench --tool-loop` import Strata's `serve.sandbox` and `serve.reasoning_tool`, so they need a Strata-xeno checkout on
`PYTHONPATH`. Strata-xeno will take the MIP core from here as a pinned package (planned: the inline writer and the loop closers
behind one token-level API).

## Running

```sh
python -m unittest discover -s tools -p "test_*.py" -t .     # every test (a few skip without a GPU, a model or STRATA_ROOT)
python tools/igc/tir4b.py run --model <Qwen3.5-4B> --rows rows.jsonl --samples 2 --seed 1 --ids --out out/arm \
    --inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress \
    --loop-close --answer-close --early-cycle --shape-cap 1024 --cover-cap 32768
```

Rules of the work (from Strata-xeno's AGENTS.md): simulate first and the A100 last; full 262,144-token context, no `--max-tokens`;
sampled at T 0.6 with several seeds; report a mechanism as caught / should have caught; never tune to the test sets.

---

## ภาษาไทย

repo นี้เก็บงานวิจัย **MIP** (Model Inline Processing) แนวคิดคือ software คำนวณ ส่วน model ใช้เหตุผล ระหว่างที่ model คิด
MIP จะอ่านนิพจน์ที่ model เขียนก่อน `=` แล้วใส่ค่าที่แม่นยำต่อท้ายให้ และปิด loop ที่ model เล็กชอบวนติด repo นี้มีเครื่องมือทดลอง
และตัวกลไกทั้งหมด ส่วน [Strata-xeno](https://github.com/xenodeve/Strata-xeno) เป็นที่ที่เอากลไกพวกนี้ไปรันจริงใน engine และ server
กับ Qwen3.8-Flash-Next และ Qwen3.6-35B-A3B-NVFP4

แยกออกมาจาก Strata-xeno เมื่อ 2026-11-08 พร้อมประวัติ git (`git filter-repo`) และย้าย issue มาด้วย path ทุกอย่างยังเหมือนเดิม
(`tools/igc`, `tools/qwenputer`) คำสั่งที่เขียนไว้ใน issue เลยยังรันได้

**รอยต่อกับ Strata-xeno:** ส่วน engine ของ MIP ยังอยู่ใน Strata-xeno ได้แก่ header ใน `include/strata/core/igc_*.hpp` และ
`src/kernels/qwenputer.cu` ตัว generator ใน repo นี้เป็นคนเขียน header พวกนั้น และ test จะเทียบ schema กับ Strata-xeno เมื่อตั้ง
`STRATA_ROOT` ไว้ ส่วน `tir4b --execute/--box` กับ `external_bench --tool-loop` ต้องมี Strata-xeno อยู่ใน `PYTHONPATH`
แผนต่อไปคือให้ Strata-xeno ดึงแกน MIP จากที่นี่ไปเป็น package ที่ตรึงเวอร์ชันไว้ (ตัวเขียนค่ากับตัวปิด loop รวมอยู่หลัง API เดียวที่ทำงานระดับ token)

**กติกาการทำงาน** (มาจาก AGENTS.md ของ Strata-xeno): simulate ก่อน ขึ้น A100 เป็นอย่างสุดท้าย ใช้ context เต็ม 262,144 token
ไม่ใช้ `--max-tokens` สุ่มที่ T 0.6 หลาย seed รายงานกลไกเป็น "จับได้ / ควรจับ" และห้ามจูนให้เข้ากับชุดทดสอบ
