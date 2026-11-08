# Plan: carry the missed work into MIP (2026-11-08)

The [catalog](../INDEX.md) found 98 mechanisms; 26 are in the best config. This plan takes the ones that were measured or designed
and never carried over, and says for each: the fix, where it lives, the gate it must pass, what blocks it, and what it costs.
Status: decisions D1-D5 taken (below). Phase 0 starting.

## Facts the plan is built on

| # | Fact | Source |
|---|---|---|
| F1 | Strata-xeno `xeno/mip-sandbox` is 351 commits ahead of `main` and 0 behind; its `docs/BLUEPRINT.md` names all 29 `--igc-*` flags | `git rev-list --left-right --count`, blueprint grep, 2026-11-08 |
| F2 | `xeno/q36-nvfp4` (the Qwen3.6-35B-A3B target) does **not** contain `xeno/mip-sandbox` | `git merge-base --is-ancestor` |
| F3 | Qwen3.5-4B on Strata's engine drifts on long prompts (KL 0.10-0.26 at 2,048 tokens, greedy parts at position 3-26), so the best config's own model cannot run on the engine yet | Strata-xeno#297, HF gate table |
| F4 | `sim_replay` reads tir4b records only; `external_bench` keeps the text, not token ids, so no replay gate exists for engine runs | `tools/igc/sim_replay.py`, `external_bench.py`:210 |
| F5 | vLLM does not repeat across runs (mip-research#21); the Strata engine on the A100 repeats 49/49 (Strata-xeno#283) | catalog, benchmarks |
| F6 | The `(exact)` note draws re-checks on Qwen3.5-4B (+22-24 points). The Gyro-M arm where removing it cost +26.6 % tokens used the early engine delivery, one unpaired run per arm, median question unchanged: not evidence for a note under the General Style | [inject](../inject.md), Strata-xeno#276 `igc:note-and-effort-20261101` |
| F7 | Compute units: about 3.6 on all accounts; round 9 (best config vs control, 120 fresh rows x 2) needs 12-20 and is the active goal | session record |
| F8 | tir4b has no sum note or repeat note; `test_unified_detect.py` and `test_ood_metrics.py` exist only in Strata-xeno | grep |

## The missed items

| # | Item | Evidence | Fix | Gate | Blocker | Cost |
|---|---|---|---|---|---|---|
| M1 | Engine detect + MTP draft drop + one-window put not used by the best config | Flash-Next: 5-digit x tokens -72 %, time -56 %; vLLM harness 19,600 extra requests | Port the General Style forms and refusals into `parse_inline` / the put; keep the Python as the reference | Differential test: the same text gives the same put / refuse decision in C++ and Python, over H3 and the stored A100 traces, 0 differences | F1 merge; F4 reader | local build |
| M2 | Best-config closers (A, B, V7, shape cap, cover cap) not in the engine | [anti-loop](../antiloop.md) | Port them beside the engine's own `in_cycle` / counting / line closes | Same token ids give the same act at the same position as `sim_replay`; then thresholds re-set on Flash-Next traces with closers off | F3 (4B on engine), F4 | local + A100 |
| M3 | Natural-language levels 1 / 2 (71 % of computed numbers in GSM8K) not caught | [detect](../detect.md) | Extend the reader to word operands (`half of 180 is`, `5 pies * 8 each`) as exact-parse rules; latent signals may only arm | H3 replay: 0 false puts; next-value agreement not below level 0's; rules written from development traces only | mip-research#4 shows how a word operand misreads | offline |
| M4 | Probes / MoE router see 83-89 % of calculations 2 tokens ahead; `UnifiedDetector` not wired | [detect](../detect.md) | Wire `UnifiedDetector` behind a flag; measure arming coverage offline with `coverage.py` | Offline coverage and false-arm rate on the 46 traces; an arm never puts by itself | only useful where the runner does not stop at `=` (engine, Qwen3.6 router) | offline, then F2 |
| M5 | Whole-step compute (arithmetic mode: 156 vs 118, 0.046x) only with thinking off | [inject](../inject.md#arithmetic-mode) | Pilot `--whole-hint --question-value` with thinking on, full context | Local llama-server, sampled T 0.6, 2 seeds, small n first; it changes the prompt, so replay cannot test it | none | local |
| M6 | Sandbox close (0.508x) and tool route (0.71-0.81x) never combined with the General Style | [sandbox](../sandbox.md) | Arms on the engine after M1: General + forced block + close; General + tool route | Engine A100 paired A/B; off for coding | M1; `--execute` on this PC needs a new approval (D1 approval was for one round) | A100 |
| M7 | Engine notes: sum note (13 -> 0 delivered-sum events), repeat note, third-repeat close | [inject](../inject.md), [anti-loop](../antiloop.md) | Sum note: first count the delivered-sum pattern (mip-research#1) in the 4B records, add to tir4b only if it occurs. Repeat note / third-repeat close: Flash-Next only | Third-repeat close: the same-exe repeated test that never ran (`tl-otra` / `tl-otrb`). Repeat note: replay plus the mip-research#5 cases (`stop checking` ended thinking mid-work) | On the 4B the replay already rejected the repeat-close family (mip-research#13: 44 of 48 rows not looping) | local |
| M9 | CandidateAnswerTracker + candidate close never measured | [anti-loop](../antiloop.md) | Measure after M6, on runs that have program output | Replay on M6's traces, then live | M6 | with M6 |
| M10 | Paired A/B only on the engine | F5 | Move every A/B that can run on the engine there; tir4b on vLLM stays the reference for the 4B | — | F3 for the 4B | — |
| M11 | Missing pieces in the repo | F8 | Move the two test files; keep threshold scripts beside the numbers (done for shape / V7) | test suite green | none | none |

Kept as options and **not scheduled**, with the reason: `--suppress-recheck` (re-check 38.5 -> 16.1 %, tokens -10-15 %, no gain in
right answers) and D `--loop-value` (value right 12 of 20, adopted 0 times). On the 4B the aim is right answers, not tokens.
The learned / latent path stays paused (developer decision, Strata-xeno#276).

## Phases

**Phase 0, no GPU (now).**
1. README corrections (tool route model, MathBackend CPU vs CUDA row, BIG-bench note): waiting for the developer's yes.
2. M11: move `test_unified_detect.py`, `test_ood_metrics.py` into mip-research; suite green.
3. F4: an engine-trace reader for `sim_replay` (re-tokenise `external_bench` text with the pack tokenizer, or have `external_bench`
   keep token ids), with a test that a tir4b record and its converted form give the same acts.
4. M4 offline: `UnifiedDetector` behind a flag, arming coverage on the 46 traces.
5. M7 first half: count delivered-sum events in the 4B records.
6. M3 rules on development traces, gated on H3.

**Phase 1, this PC (5060 Ti; one engine at a time).**
1. Build from `xeno/mip-sandbox`; the merge to `main` per D1, with greedy raw-token parity of the default path (swaps off) in the
   same session.
2. M1: C++ forms and refusals, differential tests against the Python reference until 0 differences.
3. M2: closers in the engine, same-position test against `sim_replay` on converted traces.
4. M7: same-exe repeated test of the third-repeat close on Flash-Next.
5. M5 pilot on llama-server (4B, thinking on).

**Phase 2, A100 (needs compute units; check the balance, say it, stop the VM after the round).**
1. Round 9 first: it is the active goal, and anything else here would displace it.
2. Flash-Next on the engine: General Style + closers vs plain, paired (F5), thresholds from Flash-Next traces with closers off,
   tuned and tested on different rows.
3. M6 arms, then M9.

**Phase 3, Qwen3.6.** Merge `xeno/mip-sandbox` into `xeno/q36-nvfp4` (with a merge report), then M4 with the router as a free
arming signal, then the General Style + closers on Qwen3.6.

**Phase 4, the 4B on the engine,** once Strata-xeno#297's long-prompt drift is fixed: the best config on the engine against the
same config on vLLM, the cleanest check that the port is faithful.

## Decisions (developer, 2026-11-08)

| | Question | Decision |
|---|---|---|
| D1 | `xeno/mip-sandbox` into `main` | Merge now as one PR with the parity gate, so q36 and every later branch start from it |
| D2 | Where the shared MIP core lives | mip-research is the reference: Python rules, forms, refusals, thresholds and the replay gate, as data plus golden tests; the engine's C++ reimplements them and must pass those tests (Python in the decode loop would add latency) |
| D3 | The note on Flash-Next | The plain value for every model. The Gyro-M note result came from the early delivery, before the General Style, and was weak; the General Style's put is already the value the model trusts (doubt 4.1 % against 15.2 % for its own). Measure the doubt rate (`put_reject.py`) on the first Flash-Next run with the General Style; revisit only if it shows distrust |
| D4 | Order of models and compute units | **Qwen3.5-4B first, until the whole system is settled and no assumption changes**, then other models: a small model has the most headroom, so it shows whether the software works fastest. Round 9 first on the A100. Flash-Next and Qwen3.6 steps (Phase 2 step 2, Phase 3) wait for that |
| D5 | Running model-written code (`--execute`) | As the work needs, but by default on the A100: faster, and off this PC |

**What D4 changes in the order.** The 4B items run first: M3, M5, M7's sum-note count, M11 and the replay reader in Phase 0; M5's
pilot and round 9 next. M1/M2 (the engine port) need the 4B on the engine, so Strata-xeno#297's long-prompt drift moves up to Phase 1:
it is the blocker for testing the port on the same model as the best config. Flash-Next-only items (third-repeat close, repeat note) wait.
