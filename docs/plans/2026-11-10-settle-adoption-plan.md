# Plan: getting the model to take a settle line (2026-11-10)

Part of the [decide layer v1](2026-11-10-decide-layer-design.md). Its simulation (section 4a) showed every settle line is exact, and
left one question open: **does the model take it and leave the loop?** The first sample of the settle probe (20 of 147 jobs) said
"sometimes": after `Checked exactly: $198/90 = 2.2$ holds.` the model wrote `So $Y=2.2$ is correct.` and went on to the end
(`dm_mixed_longer-5053#0`); after `Checked exactly: $2928 / 28 = 732/7$ holds.` it wrote `Let me re-check $2928 / 28$.` and looped
again (`dm_mul_div_longer-5070#1`). The A100 round 5 had shown the same with `--loop-value`: the value was right, the model did not
take it.

The research calls this **code result distrust** (CoRT, arXiv 2506.09820; arXiv 2510.20342) and offers three ways round it. This plan
tests each on the same stored loops before anything goes into a round.

**Build:** 4B NVFP4 W4A8 · llama.cpp · PC layer (the round-9 prefixes, continued on the probe's own llama-servers, never a round's).

## The three ways, and what was built before

| Way | Source | What it changes | Built before here | Measured before |
|---|---|---|---|---|
| **1. The model's own voice** | Thinking Intervention, arXiv 2503.24370 | The settle line is written as the model's first-person thought (`Okay, $2928/28 = 732/7$ is exact; I have confirmed it. I'll stop re-checking it and move on.`), not an outside note (`Checked exactly: ... holds.`) | `--trust-line positive` (ATTC's first-person sentence), only after a hinted program; the loop nudge `Wait, let me rethink this derivation carefully.` (first person, but it asks to rethink) | `--trust-line cort`: the 4B re-checked a printed -530 for 12,000 tokens (lb1). The loop nudge touched 18 right answers in replay and was dropped. Neither was a first-person **confirm** at a loop. |
| **2. Re-check words held down, in a window** | NoWait, arXiv 2506.08343 | For 200 tokens after the settle line, the 45 token ids of the re-check words (`wait`, `hmm`, `check`, `verify`, `double`, `again`, ...) get logit -30 | `--suppress-recheck N` (same 45 ids, vLLM only) | #298, A100, 48 rows: the re-check word within 300 characters fell 47 % -> 16 %, but right 19 / 48 against 21, and cut answers 3 -> 9 with closer A; dropped. It was applied after **every** put (hundreds per answer); here it is once, after a settle line at a loop. On llama-server it is `logit_bias`. |
| **3. Close where the model is ready** | ThinkBrake, arXiv 2510.00546 | Nothing is put; in the thinking, the close comes at the first token where `</think>` is within 1.0 nat of the top token, instead of after 4 rounds of a loop | none; the nearest are `--candidate-close` (closes on an answer a program confirmed, needs `--execute`, not measured) and SEAL steering (`--seal`, built, no result) | none. It is the one way that reads the model, not the text, so it is the candidate for the close-on-unfinished-work problem (`stdmath-fbf10ed4#1`). |

## How each is measured

The same 49 closes that have a settle line (A 18, B 21, C 7, D 3), the model's own tokens up to the close, cut back to one round of the
loop, seed 1 for every arm. A continuation ends at the end of the turn, at a **relapse** (the loop's line back 3 times), or at 16,384
tokens (a probe cap, not a run). Scored by the rule and the API reader.

| Arm | Script | Continuation |
|---|---|---|
| plain | `settle_probe.py` | nothing put (the baseline: does the loop come back by itself?) |
| settle | `settle_probe.py` | the v1 settle line (outside voice) |
| close | `settle_probe.py` | today's close |
| voice | `settle_probe2.py` | way 1 |
| voice_nowait | `settle_probe2.py` | ways 1 + 2 |
| brake | `settle_probe2.py` | way 3 (the 47 closes in the thinking; an answer close has no thinking left) |
| settle_close | `settle_probe2.py` | the v1 settle line, then today's close at once (for C, D, and an A whose value is the answer) |
| settle_relapse_close | `settle_probe2.py` | the v1 settle line, then today's close only if the loop comes back: the decide-layer design itself |

Per arm: relapse / end / cap, right by rule and by the API reader, tokens to the end. Per way, against plain and against close on the
same keys: how many left the loop, how many of those ended right, and the touches that turned a key close got right into a wrong one.
For brake also: where it closed against where today's close came (earlier, so on unfinished work, or later).

## Combined, by loop type (developer, 2026-11-10)

Loops are of several kinds, so the ways are meant to work together, one pairing per kind, not one way for all. The first sample
of probe 1 (38 of 49 keys, rule only) said the same: the settle line alone ended 6 and was right 6, today's close ended 26 and was
right 24 (close-only right 18, settle-only right 0); the settle line cut relapses 25 -> 14 against plain but sent 18 answers to the
16k cap. So the settle line is a step before a close, not a replacement. The two combined arms test the pairing; the result is read
per class and gives each class its pairing (for example: A confirm then close, or brake when the value is a sub-step; B quote then
close on relapse; C and D the exact value then close; E close; F brake).

## What decides

- A way is worth a round only if, on these 49, it leaves the loop more often than `settle` does **and** ends right at least as often
  as `close` on the same keys. A way that ends more answers but less right is out (the aim: never lower the model's ability).
- Two runs from one seed are a sample, not a result. The way that passes gets more seeds on these prefixes first, then the
  development rows on the PC layer, then frozen10, as in the decide-layer plan's gate.
- Nothing here is tuned to these 49: each way is taken from its paper as written (the voice line's wording, the 45 words, the 200-token
  window and the 1.0-nat brake are set before the run and reported as set).

## Status

- `settle_probe.py` (plain, settle, close): running, 2026-11-10.
- `settle_probe2.py` (voice, voice_nowait, brake): built, runs after it. The brake reads `n_probs` from llama-server's stream; its field
  names are checked on one short request before the run.
