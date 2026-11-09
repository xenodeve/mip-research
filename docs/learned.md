# Learned and latent path

This is the IGC learned path (PRD path A, "Latent IGC", in Strata-xeno#276). Small neural heads read the frozen model's hidden state, decide whether a calculation is asked for, and read its operands. An exact GPU executor computes the value, and a learned residual at the head input hands that value back to the model. The base weights are never trained. All of it runs inside the Strata engine on Qwen3.8-Flash-Next (Gyro-M Swift 1.5 IQ2_XS, D2x 262K / five-worker profile, RTX 5060 Ti) behind a diagnostic build option. None of it is in the best config. The developer paused early-layer latent injection as separate research (Strata-xeno#276 body, "Decisions", restated in comment `igc:sidecar-decisions-20261031`). For delivering values the developer chose the token-result bridge instead of the latent mapper; latent mapping stays a separate arm (`igc:bigbench-off-output-plateau`).

The page also lists the residual-stream read/write probes and SEAL steering on Qwen3.5-4B / Qwen3.6, which act on the hidden state from the reference runner side.

## Summary

| Mechanism | What it does (one line) | Runs on | In best config? | Best measured result | Status |
|---|---|---|---|---|---|
| [Activation observation](#activation-observation) | Captures the model's own F32 states (embedding, first attention, final mixed state, head input) with position and owner identity | Strata engine (xeno/mip-sandbox, not in main) | No | 970/970 live Verifier observations captured; the first four rows replay every logit bitwise through the Q8 head [^obs] | built, measured, paused |
| [Input readout heads](#input-readout-heads) | Read operator, signs, lengths and digits of a two-digit `a op b` from 80 pre-anchor rows | offline tool (training) + Strata engine (native forward) | No | Native CUDA forward gives the same field decisions as Torch on all 1,111 cached evaluation rows, max logit difference 3.4e-5 [^input] | built, measured, paused |
| [Readiness head](#readiness-head) | Decides call / no call from the final mixed state at the answer anchor | offline tool + Strata engine | No | Native forward gives the same decisions as Torch on all 320 real activation vectors, max logit difference 9.5e-6 [^ready] | built, measured, paused |
| [Output mapper](#output-mapper-late-head-input-residual) | Learned gated residual at the late head input that carries the executed value into the next tokens | offline tool + Strata engine | No | Native mapper matches Torch on all 421 captured positions (max difference 7.6e-6); all 421 head decisions agree [^output] | built, measured, paused |
| [Native head bridge and STE seam](#native-head-bridge-and-ste-seam) | Trains a head against the engine's real Q8_1/Q8_0 LM head with a straight-through backward | offline tool (CUDA DLL) | No | All 248,320 logits of a real capture replay bitwise | built, measured, paused |
| [Controller assembly](#controller-assembly) | Typed selection, one-shot device invocation, activation history, pipeline, trial/abort, lifecycle, bundle | Strata engine | No | Offline selection/request/result records match the oracle 64/64; all four ComputeSanitizer tools clean [^ctrl] | built, measured, paused |
| [Controller host timeline](#controller-host-timeline) | Eight `igc.controller.*` spans in the pipeline timeline | Strata engine | No | One component trace: bind 52,343.727 ms (model/tokenizer content hash), stage 3.486 ms over 12 calls | built, measured, kept as option |
| [Training and data tooling](#training-and-data-tooling) | Fail-closed training guards, identity-bound corpora and caches, warm start, mixed replay, artifacts | offline tool | No | Guards and loaders: 53/53 tests on CPU and CUDA Python | built, measured, paused |
| [Residual-stream read and write](#residual-stream-read-and-write-probes) | Reads answer digits from the 4B's hidden state at `=`; writes a value by class-mean steering | offline tool (HF / vLLM, A100 and local) | No | Layer 28 reads the first two answer digits at 0.99 / 0.89 held-out accuracy; the third is not there yet (0.15) [^rw] | built, measured, paused |
| [SEAL steering](#seal-steering) | Adds a steering vector at every `\n\n` to keep reasoning executing instead of reflecting | tir4b (vLLM) | No | One A100 arm ran; no SEAL-vs-control result posted [^seal] | built, not measured yet |

## Where it lives and what it takes to use

- **Engine build:** CMake `-DSTRATA_BUILD_IGC_HEAD_PROBE=ON` (CUDA only) defines `STRATA_IGC_NATIVE_DIAGNOSTIC` (Strata-xeno xeno/mip-sandbox: CMakeLists.txt:2023-2030). Default builds leave out every learned hook.
- **Engine flags** (Strata-xeno xeno/mip-sandbox: src/program/generate.cpp:959-961, parsed at 2815-2817 and 2842-2843): `--igc-learned-diagnostic-bundle PATH` (the learned first-anchor controller), `--igc-output-diagnostic-artifact PATH` + `--igc-output-diagnostic-fixture PATH` (output mapper alone, with checked results), `--igc-observation-plan` / `--igc-observation-root` (activation capture).
- **Model binding:** the bundle is bound by digest to Flash-Next's files, and its inputs are 4 x 2560 HC streams, so another model needs a new collection and new heads (mip-research#12, item 8).
- **Data:** activation caches and bundles are local and git-ignored (`build-igc-*`). A clean checkout cannot reproduce the scores without collecting again (Strata-xeno#276, `igc:branch-pushed-backfill`).
- **To use it at all:** a bundle whose readiness and input heads pass the frozen development gate and then the sealed acceptance64 set, repeated triggers inside reasoning, operands over two digits, and a hard-off ABBA. None of these exist yet.

## Activation observation

The engine copies the model's own states to a bounded archive at fixed seams, without changing any number on the decode path:

- **Pre-anchor rows:** up to 80 rows before the answer anchor, four HC streams x 2560 F32, each with absolute position, token and source identity. Maximum archive 3,278,124 bytes. Works on the verifier windows (`short-read=256`) and on the synchronized prefill callback (Strata-xeno#276, `igc:preanchor-observation:5727b99b`).
- **Early seams:** the embedding broadcast and the first native attention output (before layer 0's MoE combine). On two real requests all 135 embedding rows matched an independent Q8_0 decode, and the three stages were bitwise equal across 45 common-prefix rows (`igc:early-native-fidelity-37ab2164`; report docs/reports/2026-10-29-igc-early-observation.md).
- **Prefix probe (`igc_prefix_probe`):** runs only the verifier stage `[0,1)` with Gyro's 48-layer geometry and all 512 layer-0 experts, to collect first-attention states without the full model. Archived first-attention rows from full-model requests match it bitwise at window sizes 1 and 8 (`igc:native-prefix-qualification-7d1e01dc`). It filled the 10,624-case context cache (`igc:context-cache-and-output-training-27244c0f`) and the 1,536-row matched cache, whose 128 replayed old prefixes matched bitwise (`igc:i2-matched-context-collection-20261029`).
- **Coherent observation (v3 archives):** on the real batched/wave prefill and accepted-input path, it stages the pre-anchor window, takes a D2D snapshot of the head input and reads the final state for the same anchor. Before anchor five, the first four Verifier rows must reproduce every logit bitwise through the loaded Q8 head, or the run fails. Observer device storage 14,113,680 bytes (`igc:coherent-observation-v3-01`). A live run captured 970/970 observations (400 anchors, 570 output positions) and passed `coherent_audit` as `native_qualified: true` (`igc:coherent-live-02-collected`).
- **Prefill snapshot:** `PrefillSnapshot` copies bounded embedding/first-attention tails before source reuse in batched prefill (`igc:live-controller-cb7bbd66`).

Code: [../tools/igc/observation.py](../tools/igc/observation.py), [../tools/igc/context_cache.py](../tools/igc/context_cache.py), [../tools/igc/coherent_observation.py](../tools/igc/coherent_observation.py), [../tools/igc/coherent_collect.py](../tools/igc/coherent_collect.py), [../tools/igc/coherent_audit.py](../tools/igc/coherent_audit.py), [../tools/igc/coherent_evaluate.py](../tools/igc/coherent_evaluate.py), [../tools/igc/coherent_replay.py](../tools/igc/coherent_replay.py); engine Strata-xeno xeno/mip-sandbox: include/strata/program/igc_coherent_observation.hpp, include/strata/core/igc_prefill_snapshot.hpp, tests/igc/prefix_probe.cpp.

## Input readout heads

Each head reads the 80 pre-anchor rows (`[80,4,2560]` F32, valid-row mask, positions) and outputs nine categorical fields: operator (none/add/sub/mul), two signs, two two-digit flags and four digits. Token IDs are never an input feature. Operands are signed, magnitude at most 99. The kept head is the contextual learned-query attention head (96,964 parameters) on first-attention states. Its native CUDA forward (`igc_input.cu`, artifact kind 1) gives the same field decisions as the frozen Torch reference on all 1,111 cached evaluation rows (768 matched development + 343 legacy), max absolute logit difference 0.000034332 (`igc:retention-replay-native-qualification-01`).

The variants tried, all on the same nine-field target:

| Variant | What differs | Parameters | Tool | Outcome |
|---|---|---:|---|---|
| Final-vector MLP | One final-HC vector at the answer anchor | 334,516 | [../tools/igc/readout.py](../tools/igc/readout.py) | not selected [^in-final] |
| Activation attention | Attention over 80 final-HC pre-anchor rows | 227,396 | [../tools/igc/attention_readout.py](../tools/igc/attention_readout.py) | not selected [^in-final] |
| Embedding / first-attention seams | Same head on earlier seams | same | [../tools/igc/embedding_readout.py](../tools/igc/embedding_readout.py) | not selected [^in-early] |
| Query-free pointwise | Learned queries only, no final-query route | 63,492 | embedding_readout.py | not selected [^in-early] |
| Contextual block | One contextual block before readout, 4,096-row balanced corpus | 96,964 | [../tools/igc/balanced_corpus.py](../tools/igc/balanced_corpus.py) | basis of the kept head [^in-bal] |
| CUDA fit + cosine LR | 10,240-row corpus on GPU; constant LR replaced by cosine 0.001 to 0.00001 | 96,964 | embedding_readout.py | stable fit [^in-cos] |
| Ordered digit/PAD view | Left-aligned digits with explicit PAD class | 97,094 | [../tools/igc/ordered_input_fit.py](../tools/igc/ordered_input_fit.py) | rejected [^in-ord] |
| Digit-attention supervision | Adds positive-only digit/null attention loss | 96,964 | [../tools/igc/aligned_input_fit.py](../tools/igc/aligned_input_fit.py) | not selected [^in-align] |
| Native first-attention cache | Contextual head on first-attention states (10,624 cases) | 96,964 | context_cache.py | the retained head [^in-ctx] |
| Matched fresh fit | 768 paired compute / quote-copy rows, fresh init | 96,964 | [../tools/igc/matched_evaluate.py](../tools/igc/matched_evaluate.py) | not promoted [^in-fresh] |
| Warm start | Same, from a copy of the retained head | 96,964 | [../tools/igc/initial_state.py](../tools/igc/initial_state.py) | not promoted [^in-warm] |
| Mixed replay | Warm start with each batch 32 matched + 8 old rows per class | 96,964 | [../tools/igc/replay_run.py](../tools/igc/replay_run.py) | the replay candidate [^in-replay] |
| Worded refit | Adds worded arithmetic to coherent anchors | 96,964 | [../tools/igc/attention_refit.py](../tools/igc/attention_refit.py) | used in later bundles [^bundles] |

Data and identity: [../tools/igc/matched_corpus.py](../tools/igc/matched_corpus.py), [../tools/igc/input_manifest.py](../tools/igc/input_manifest.py), [../tools/igc/replay_protocol.py](../tools/igc/replay_protocol.py), [../tools/igc/development_audit.py](../tools/igc/development_audit.py). Engine: Strata-xeno xeno/mip-sandbox: include/strata/core/igc_input.hpp, src/kernels/igc_input.cu. Reports (Strata-xeno xeno/mip-sandbox): the `docs/reports/2026-10-29-igc-*.md` set (i2-readout, input-diagnostics, balanced-input, cuda-input, ordered-input-negative, input-alignment, native-context-input, matched-fixed-fit, matched-warm-start), plus docs/reports/2026-10-30-igc-replay-result.md and docs/reports/2026-10-30-igc-replay-native-qualification.md.

## Readiness head

A GELU MLP (2560 -> 128 -> 2, 328,066 parameters with fit-only mean/scale) reads the final mixed activation at the answer anchor and outputs call / no call. Its corpus holds operands constant and changes only the intent (calculate vs quote/copy), so the head cannot route on the numbers. Native CUDA forward (`igc_readiness.cu`, artifact kind 3) gives the same decisions as Torch on all 320 real activation vectors, max absolute logit difference 0.0000095367 (`igc:native-readiness-5b52d125`). A refit on old plus coherent anchors, with structural clusters held out by hash, is the readiness head of bundle v2 ([../tools/igc/readiness_refit.py](../tools/igc/readiness_refit.py)). [^ready]

Code: [../tools/igc/readiness_corpus.py](../tools/igc/readiness_corpus.py), [../tools/igc/readiness_fit.py](../tools/igc/readiness_fit.py), [../tools/igc/readiness_mapping.py](../tools/igc/readiness_mapping.py); engine Strata-xeno xeno/mip-sandbox: include/strata/core/igc_readiness.hpp, src/kernels/igc_readiness.cu.

## Output mapper (late head-input residual)

The mapper takes the typed exact result (ABI record, status, owner, anchor), turns it into ordered categorical features, and adds a per-position gated residual (with attention) to the final mixed state just before the native head quantizes it. No result text, token IDs or host arithmetic enter it. Hard-off skips the module, and a failed result leaves the activation bytes unchanged. The training target is the next token under teacher forcing, through the real Q8 head with the declared STE backward (708,481 parameters, [../tools/igc/output_fit.py](../tools/igc/output_fit.py)).

The Torch-free native forward (`igc_output.cu`, artifact kind 2) loads once, needs no per-call allocation or sync, and publishes only a wholly finite row. Against Torch on all 421 captured positions: max mixed difference 7.62939453125e-6, and all 421 native Q8-head decisions agree. Weights plus 64-row scratch: 10,305,284 bytes. All four sanitizer tools pass on 64-row valid/failure/malformed/NaN/overflow controls (`igc:native-output-forward-8c7ac9b4`).

In the decoder it sits after final HC mixing and before head quantization and sampling (`igc:controlled-output-injection-4fdad56a`). `LatentPipeline::end_output` stops injection once `</think>` is accepted, so the model copies the value from its own context (Strata-xeno xeno/mip-sandbox: src/core/igc_pipeline.cpp:96, called from src/program/generate.cpp:14742 through include/strata/program/igc_learned_diagnostic.hpp:662). [^output] [^bundles]

Code: [../tools/igc/output_corpus.py](../tools/igc/output_corpus.py), [../tools/igc/output_observe.py](../tools/igc/output_observe.py), [../tools/igc/output_refit.py](../tools/igc/output_refit.py), [../tools/igc/output_integration.py](../tools/igc/output_integration.py); engine Strata-xeno xeno/mip-sandbox: include/strata/core/igc_output.hpp, src/kernels/igc_output.cu, include/strata/program/igc_output_diagnostic.hpp.

## Native head bridge and STE seam

`igc_head_bridge` is an optional CUDA shared library ([../tools/igc/cuda_head.py](../tools/igc/cuda_head.py) wraps it). Its forward is the engine's own Q8_1 activation quantizer and Q8_0 LM head. Its backward is `W_dequant.T @ delta`, an explicit straight-through estimator at the head input only, not through Gyro's layers. It loads the head once per handle, uses caller-owned CUDA tensors and pins its own DLL hash. All 248,320 logits from a nonzero real Gyro capture match bitwise and on repeat. Memcheck, initcheck and synccheck report 0 errors and racecheck 0 hazards (`igc:native-head-training-bridge-fab8b10f`). [../tools/igc/head_surrogate.py](../tools/igc/head_surrogate.py) is the earlier process-per-call version of the same seam, kept for independent checks (`igc:checkpoint:I0:76d79b4c`). Engine: Strata-xeno xeno/mip-sandbox: tests/igc/head_bridge.cpp, tests/igc/head_vjp.cu (CMakeLists.txt:2041).

## Controller assembly

These parts connect the heads to the exact executor in the decode loop. Each one is test-first and all four ComputeSanitizer tools were run on it.

| Part | What it does | Engine file (Strata-xeno xeno/mip-sandbox) | Measured |
|---|---|---|---|
| Selective dispatch | Builds the typed request on the device from two 52-logit readouts + readiness. Requires valid finite results, readiness, all nine fields agreeing and canonical one/two-digit operands. A rejected call writes a valueless request, so an old answer cannot survive | src/kernels/igc_dispatch.cu | 64 fixtures pass in normal/graph A/B/A (`igc:selective-dispatch-938c31d9`) |
| Owned invocation | One-shot arm/execute/broadcast. A captured decode graph consumes one owned attempt and keeps its result across replays | include/strata/core/igc_invocation.hpp | counts 1/64, graph repeat after operand change (`igc:owned-invocation-76ed945b`) |
| ContextWindow | 80 embedding + 80 first-attention rows on the GPU, right-aligned with a mask. Prefix reuse survives; changed lineage or missing taps bypass IGC without a cold prefill | src/core/igc_context.cpp | 13,107,280 requested bytes; device taps match host observation bitwise (`igc:live-context-42b55543`) |
| LatentPipeline | Input heads -> readiness -> selection -> exact CUDA arithmetic -> output mapper, with retained provenance (160-byte request, selection reason) | src/core/igc_pipeline.cpp | 12,801,500 requested bytes; offline selection/request/result records 64/64 (`igc:latent-pipeline-624cda3c`) [^ctrl] |
| Pipeline trial | Checkpoint / abort / commit of module state on the owning stream | src/core/igc_pipeline.cpp | 7,816 requested bytes; full 64-row sanitizer run clean (`igc:pipeline-trial-ada2e825`) |
| Learned lifecycle | `IgcLearnedDiagnostic`: cancel revokes a retained result, audit statuses reset per invocation, release closes staging | include/strata/program/igc_learned_diagnostic.hpp | three reproduced lifecycle gaps fixed red/green (`igc:learned-lifecycle-2f017287`) |
| Controller bundle | 204-byte digest manifest of four networks (embedding input, first-attention input, readiness, output), bound to model and tokenizer files. Version 1 needs embedding/attention consensus; version 2 lets the attention head select alone and does not run the embedding head | src/core/igc_controller_bundle.cpp, include/strata/core/igc_controller_bundle.hpp:12 | four assets 4,947,912 bytes; 14 host tests (`igc:controller-bundle-160baf6a`, `igc:bundle-v2`) [^bundles] |
| Adapter artifact | Fixed bounded F32 weight format (no code, no pickle) binding model/tokenizer/state/protocol; refuses serving activation | src/core/igc_adapter.cpp | 8 host/native tests (`igc:native-context-outcome-and-host-artifact-851cffc9`) |

Code: [../tools/igc/adapter_artifact.py](../tools/igc/adapter_artifact.py), [../tools/igc/generate_adapter_schema.py](../tools/igc/generate_adapter_schema.py), [../tools/igc/controller_bundle.py](../tools/igc/controller_bundle.py), [../tools/igc/controller_diagnosis.py](../tools/igc/controller_diagnosis.py), [../tools/igc/native_qualification.py](../tools/igc/native_qualification.py), [../tools/igc/learned_integration.py](../tools/igc/learned_integration.py).

The exact executors under the pipeline (ABI1 add/sub/mul up to 62 digits, ABI2 floor div/mod/gcd/pow, ABI3 rational/decimal) belong to the compute category.

## Controller host timeline

Eight literal spans `igc.controller.{load_host,bind,prepare,begin,stage,cancel,audit,release}` in the actual wrapper, reported by the timeline analyzer as attempts, inclusive host ms and maxima. One component trace (old saved activations, no language weights loaded), from `igc:controller-host-timeline-20261029`:

| Host region | Attempts | Inclusive host ms | Max host ms |
|---|---:|---:|---:|
| load_host | 1 | 39.569 | 39.569 |
| bind | 1 | 52,343.727 | 52,343.727 |
| prepare | 3 | 32.763 | 16.457 |
| begin | 10 | 0.485 | 0.136 |
| stage | 12 | 3.486 | 2.195 |
| cancel | 4 | 0.118 | 0.046 |
| audit | 12 | 13.293 | 4.087 |
| release | 4 | 4.175 | 2.128 |

Bind is the startup content check of the model and tokenizer files (reading the 92 GB GGUF). The counts include deliberately refused calls; this is not a per-turn latency or a GPU cost.

## Training and data tooling

- **Training guards** ([../tools/igc/training_guards.py](../tools/igc/training_guards.py)): refuse a nonfinite loss, gradient, parameter or optimizer state. No clipping or repair. Admission needs 8 GiB physical, 12 GiB commit headroom and 4 GiB + 16 MiB disk. `--price-only` runs one backward batch without an optimizer, and a fit must present that pricing receipt (`igc:i2-fixed-training-guards-20261029`).
- **Warm start** ([../tools/igc/initial_state.py](../tools/igc/initial_state.py)): copies F32 tensors from an explicit file plus its SHA256, never optimizer state.
- **Mixed replay** ([../tools/igc/replay_protocol.py](../tools/igc/replay_protocol.py), [../tools/igc/replay_run.py](../tools/igc/replay_run.py)): schedules fixed batches from two caches, after filtering both training pools against every evaluation row by ordered pair, swapped pair, expression, canonical fact and prompt. `NpyRows.read_into` gathers rows into a caller buffer without preloading the cache. 53/53 tests on CPU and CUDA Python (`igc:mixed-replay-loader-gate-01`).
- **Low-memory scoring** (`coherent_audit`, `coherent_evaluate`): read one observation at a time. Lifetime peak about 694 MB private commit / 65 MB working set (`igc:coherent-scorecard-01`).

## Residual-stream read and write probes

These are offline probes on Qwen3.5-4B and Qwen3.6-35B-A3B (bf16, HF transformers, **4B bf16 · HF transformers**), outside the Strata learned path;
everything else on this page runs in **Strata-xeno** on Flash-Next.

- **Read** (`probe/num_rw.py`, 4B, 597 three-digit products): a linear read of the model's own answer digits from the hidden state at `=`, held out. At layer 28: digit 1 0.99, digit 2 0.89, digit 3 0.15. At layer 24: 0.93 / 0.24 / 0.08. Chance is about 0.1. At `=` the model holds only its first digits. It forms each later digit at the position that writes it, so no single position holds the whole value (Strata-xeno#298, comment "Residual-stream read/write of a number, first try").
- **Write** ([../tools/igc/probe/steer_lb.py](../tools/igc/probe/steer_lb.py), commit `f674ad38`): at the position that predicts digit j, add `alpha * (mean state of target digit - mean state of the digit the model held)` after block L. Class means come from thinking traces where the model wrote software's value. [^rw]

The residual read is the same kind of probe as the lookahead probe in the README's "Seeing a calculation coming" (detect category).

## SEAL steering

`--seal FILE --seal-layer L --seal-alpha A` in [../tools/igc/tir4b.py](../tools/igc/tir4b.py) (arguments at line 1977-1979, hook installed at line 1727) implements SEAL (arXiv 2504.07986) in vLLM. [../tools/igc/probe/seal_vector.py](../tools/igc/probe/seal_vector.py) builds the vector from the model's own stored thinking traces: S = mean over execution thoughts - mean over reflection and transition thoughts. Thoughts are pieces between `\n\n`, classified by the paper's keywords ([../tools/igc/seal.py](../tools/igc/seal.py)). At run time `alpha * S` is added to the output of block L at every `\n\n` token the model reads or writes. Defaults follow the paper: layer 20, alpha 1.0. The run logs how many positions were steered, so a run that steered none does not count as a SEAL test. It cannot run with `--server`. [^seal]

---

[^obs]: Collection is not a quality result. The live collector exited 1 after the archive was finalized (`coherent cleanup/telemetry incomplete`), so `report.json` does not exist. A `FAILURE.json` receipt was added afterwards. Commit headroom fell to 0.067 GiB during that run; Windows extended the limit from 67.39 to 71.54 GiB (`igc:coherent-live-02-collected`). An earlier attempt stopped at the 2 GiB commit guard before the first request (`igc:coherent-live-01-resource-stop`).

[^input]: The I2 gate (strict per-field development exactness and zero false calls, then the sealed acceptance64 set) never passed, and acceptance64 was never opened. All accuracies below are on inspected development data that shares templates with training; 19 of the 32 matched development pairs were in the original head's training corpus (`igc:i2-generalization-diagnosis-01`).

[^in-final]: Final-vector MLP and activation attention both fit 48/48 operator and 36/36 training tuples, and both got 0/12 exact operand tuples on the 16 template-held-out cases (operator 14/16 and 12/16). The predeclared stop rule fired (`igc:attention-outcome:75790e4a`).

[^in-early]: Known-diagnostic exact tuples /12: final-HC 0, embedding 0, first attention 0, query-free embedding 0 (`igc:input-diagnostics-25c63fd1`). Removing the final-query route is not a fix.

[^in-bal]: Development exact tuples /96 (seen values + new templates / new values + seen templates / both new): pointwise 20/6/5, contextual 47/60/22. Both made 16/32 false calls on each template-new axis (`igc:balanced-contextual-7e5d0a0e`).

[^in-cos]: The constant-rate CUDA run spiked at epoch 70 (loss 2.1e-5 to 1.03, final weights 130/7680 training tuples). Replaying the saved state showed an Adam overshoot on one batch. Cosine decay removed the spike; development was 50/96, 77/96, 33/96 with 28/32, 0/32, 32/32 false calls (`igc:cuda-overshoot-19b8f4b5`, `igc:cosine-final-quality-4bf31ff1`).

[^in-ord]: Ordered digit/PAD: 14/96, 64/96, 11/96 against 50/96, 77/96, 33/96 for the embedding control (`igc:ordered-input-negative-9cdc1e39`).

[^in-align]: Digit-attention supervision: 49/96, 94/96, 44/96 tuples and 16/32, 0/32, 16/32 false calls. Replacing digit readout by the literal value at the attention argmax lowered tuples (50 to 39, 77 to 29, 33 to 26), so the value is not only at the literal location (`igc:i2-alignment-a66567c7`, `igc:readiness-augmentation-cd5832d6`).

[^in-ctx]: Native first-attention cache: 96/96, 50/96, 67/96 tuples, 30/32, 0/32, 32/32 false calls; a mixed result, not a blanket gain over embedding (`igc:native-context-outcome-and-host-artifact-851cffc9`). On the 1,536-row matched cache the retained head scored 264/384 development tuples with 112/384 quote-copy false-call candidates; all 96 active-digit errors were the digit 3 (`igc:i2-retained-matched-evaluation-20261029`).

[^in-fresh]: Matched fresh fit: train 384/384, development 24/384 tuples with 0/384 false calls (`igc:i2-single-matched-fixed-fit-20261029`).

[^in-warm]: Warm start: matched development 384/384 and 0/384, but the frozen 343-row retention gate failed: 22 previously correct legacy tuples turned wrong (`igc:matched-warm-result-01`).

[^in-replay]: Mixed replay passed the frozen development gates on inspected data: matched 384/384 with 0/384 false calls, legacy 259/259 with no lost tuple and no new false call, 42/84 legacy negatives still predicting a nonzero operator. It is not fresh acceptance; there are only 32 independent signed pairs (`igc:retention-replay-result-01`). In the native controller on the old 64 it selected 33/48 like the original, limited by readiness and the embedding consensus (`igc:controller-gating-diagnosis-01`).

[^ready]: Readiness on inspected development: 111/112 (misses `(19) - (19)`); the linear control also got 111/112 but admitted an unsupported division. Adding 416 fit rows gave 110/112 (`igc:readiness-pairs-d8c9757a`, `igc:readiness-augmentation-cd5832d6`). On the old 64, 8 of the 11 readiness misses and all 4 disagreements were the `en-exact-result` question family (0/12 selected), which was absent from readiness training (`igc:controller-gating-diagnosis-01`).

[^output]: Output pilot on inspected development: 5/6 complete numeric sequences; development numeric 30/32 with the correct result against 26/32 with base, zero or swapped content (`igc:output-capture-fit-64f8d2a3`). Enabled logits are not bit-identical to the hook-free head (1,234,923 entries differ, max 0.0078). Later refits stopped improving: held-out whole sequences stayed near 181-183 of 283 after more data and 30 more epochs (`igc:bigbench-off-output-plateau`). Disabled raw-token parity was never shown; the same pre-hook executable already varied run to run on one Thai prompt (37/39/37/39 tokens, `igc:off-parity-control-66d7e27d`), so that variability predates the hook.

[^ctrl]: The 64 records are the old inspected pilot, not live generation. Full 64-row racecheck of the pipeline timed out at 240 s and was not counted; a bounded run was clean (`igc:latent-pipeline-624cda3c`).

[^bundles]: Not presented as results. The live and external results of controller bundles v1-v6 (live smokes, the 400-anchor scorecards, the EleutherAI two-digit and GSM8K subsets, the end-of-injection arms) were scored on inspected development data, and no bundle passed its frozen development gate or opened the sealed acceptance64 set. In the tracker itself, the claim that one-token windows changed nothing (`on-07`) was withdrawn because the edit never applied (`igc:correction-one-token`). After these runs the developer chose the token-result bridge (`--igc-result-bridge`, which uses bundle v6 for selection) in place of the latent output mapper (`igc:bigbench-off-output-plateau`). Bridge results belong to the inject/compute pages.

[^rw]: Write results are not adopted. Per-position steering on clean `a * b =` prompts: Qwen3.6 had only 2 wrong rows of 594 to fix (at most 1/2 caught per cell), control 0/178; 4B layer 31 caught 6/13 with control 0/167 (Strata-xeno#298, comment "Per-position residual steering, Qwen3.6-35B-A3B bf16"). On real 4B thinking traces (A100, 15 test traces, 9 fix points): 4/9 to 7/9 caught, but the same push rewrote correct values in 1/60 to 59/60 control points (59/60 at layer 31, alpha 10), so it is not selective. Re-check rate after the value: model alone 36.1 %, inserted text 36.1 %, steered 38.9 %, 36 samples each, too few to show a difference under about 16 points (Strata-xeno#298, comment "Residual-stream write on the computations of real thinking traces"). A whole-position class-mean write at `=` moved 0/13 (1/13 at layer 28).

[^seal]: The `a100r4/seal1` arm (Qwen3.5-4B, A100) ran, and its traces were reused as data with nothing inserted (mip-research#18). No SEAL-versus-control accuracy or token comparison was posted in the tracker, so no effect is claimed.
