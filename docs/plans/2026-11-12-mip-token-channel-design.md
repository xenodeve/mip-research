# Design: MIP on the token stream: the engine hands MIP every token, MIP steers the values it can compute (2026-11-12)

Status: **draft for the developer's approval.** Nothing in it is built. Phase 1 is written at builder detail; later phases are
directions. Every item marked *(my call)* is the writer's decision, open to the developer.

## Intent (the developer's words, 2026-11-12)

- "Every case that still has no mechanism, or has one that does not pass, gets a mechanism, without turning a right answer wrong.
  These cases can happen to any model, not only this model or this test set. Never refer to the test set: fix the root cause, the
  structure." This includes mechanisms that were tested and did not help, where software should be able to do it.
- "Software's value must be able to act on the probabilities of the tokens: make the model sure of the right path and drop the wrong
  one. That is using software to the full: intervening in the model's work with fixed data that is always right, as a guideline."
- "Only in the thinking. The final answer is not touched, except the anti-loop taking the answer already in the reasoning out when it
  finds a loop with an answer already written."
- "If the reasoning was intervened in and the answer is still wrong, it stays wrong. If software already knew the answer and let the
  LLM think on until it is right, that is no different from cheating. And if software knows the answer, why make the LLM think?"
- "A question written in symbols, like `2x240`, software reads and computes at once, and sends the value to the LLM: a fixed question,
  not against the rule. Whether to send only the value or the working too is the next stage: first the value; once the system is
  settled, test how well the model goes on from the value alone; if badly, send the working too."
- "The engine sends its data to MIP, and MIP stays a plugin as before." "No hints: a hint provokes; the intervention is at the step
  that generates the token." "If the right token is not among the candidates before the choice, the right token is added and its
  probability raised."
- "Software sees everything, but intervenes only in what MIP supports, such as arithmetic, never in the LLM's reasoning: that is the
  model's job."
- "With every token and the whole generation in MIP, sub-models can be added under MIP for other work than math, such as coding or the
  Sandbox. Every intervention is in MIP alone, so it is easier to manage. Even for coding or logic, MIP can raise the probability that
  the model uses the Sandbox to simulate and test a hypothesis in detail, without the risk of making the model re-check or loop."
- First engine: llama.cpp (the PC layer). Round 10 runs together with the new intervention system.

## Why, in numbers (measured 2026-11-12, `results/cuda-mathbackend/pause/`)

4B NVFP4 W4A8 · llama.cpp `build-w4a8` · RTX 4070 SUPER and · RTX 5060 Ti, reported apart; tir4b General flags, 12 answers in flight:

| Share of an answer's wall time | 4070 SUPER | 5060 Ti |
|---|---|---|
| generating | 86-89 % | 86 % |
| the server reading again after a stop (a put, a cut, a close) | 7-8 % | 8.5-9 % |
| HTTP, queueing, the rest | 4-5 % | 5 % |
| MIP's own host work | 0.4 % | 0.3 % |

So **the case for the channel is capability, not speed**: stops and resumes cost 12-14 % of wall time, a gain, but the reason is what
a stop cannot do: act mid-line, act where tir4b has no stop word, lower a wrong value's tokens, add a token the sampler filtered out,
and steer without inserting text.

## Goals

1. A channel by which llama-server hands MIP every token of every answer (the prompt at a request's start, then each sampled token)
   and MIP answers with a plan the sampler applies: nothing, bias token ids, force a token (adding it if the filters removed it), or
   end the request so the driver can do what only a request can (cut, rewind, run a Sandbox block). At the head of the chain the shim
  sees the whole vocabulary before any filter (`set_logits` fills `cur_p` with every token), so the developer's "add the right token if
  it is not among the candidates" is met by making its logit high enough to survive top-k, top-p and min-p *(builder check, taken)*.
2. MIP's math values in the thinking delivered through that channel instead of text puts, under the rules below.
3. A coverage table: every open failure pattern gets a mechanism or a measured reason why software cannot reach it.
4. The channel general enough that later modules (loops, Sandbox steering, coding, sub-models) plug into the same place.

## Rules (the developer's, applied to this design)

- **Local truths, never the answer.** What MIP steers toward is always a fact true where it stands: the value of an expression the
  model itself wrote, a count or a quote of the question's own text. Never the question's answer reached by analysis. The one
  exception, set by the developer: a question that is one fixed computation in symbols (read by the question reader) may have its
  value delivered: software computes it and steers it straight into the thinking, the model writing `2 x 240 = 480` as its first
  step under the channel; the thinking is not closed for it, and the model writes the final answer itself (the developer, 2026-11-12:
  "there is no reason for the model to reason it out itself when software already can"). Every arm counts the answers whose final value
  software computed and reports one-computation questions apart (most DeepMind rows are one computation).
- **The thinking only, in every arm from now on: the next step of the General Style** (the developer, 2026-11-12). Today's puts in
  the answer phase (`--inline-calc`, `--approx-put`, `phase in ('think', 'answer')`) end with it; a replay counts the answer-phase
  puts this removes before any live run. Nothing steers inside the final answer. The anti-loop may take the final answer out of the reasoning when it
  finds a loop with an answer already written: the model's own value, right or wrong. A wrong value is never corrected at extraction;
  it is fixed upstream or left wrong.
- **What "steering" means** (the developer, 2026-11-12): the tokens that can still spell the right value, in any tokenization the
  vocabulary allows, are made the most probable, and the model chooses among them; at the least, the right path's probability is the
  highest. Not a fixed token sequence from tir4b's own encoding (I4, joined digits). The plan therefore carries, per step, the set of
  tokens consistent with the value's remaining text (a prefix tree over its tokenizations), not one id.
- **Quotes and counts are not hints** (the developer, 2026-11-12): the question's own words (`--question-quote`) and the letter counts of
  its string (`--letter-count`) are always-true facts; they go through the channel as steered tokens, not inserted text.
- **Only what MIP supports.** MIP sees the whole stream but steers only values its computers make (arithmetic, MathBackend ops, letter
  counts, the question's quotes, later the Sandbox's output). It never steers the reasoning: no pressure on `Wait`, re-check words or
  the choice of method. (I14, re-check hold-down, made answers worse; this rule keeps it out.)
- **No hints.** No inserted notes or invitations. The model sees only tokens it wrote under the sampler's choice.
- **Never lower the model's ability; fix the case the structure allows; simulation, then PC layer, then A100; full context; sampled
  with two or more seeds; every touch checked right or wrong** (`AGENTS.md`).

## What the token channel changes, and what it does not (read before believing a gain)

A token forced through the logits and the same token put as plain text (`--inline-plain`, today) leave **the same context**: the model's
next step cannot tell them apart. Today's plain put already carries no sign of an outside source, and I10 still saw re-checks rise with
it. So a forced value is not by itself a new effect. What the channel adds, each a hypothesis to measure:

| Difference | Today (stops, text puts) | Channel |
|---|---|---|
| a pause | ~220-280 ms a put, a new request | none: the plan is applied at the next sample |
| strength | a put is a fact on the page | a bias (soft: the model may still write its own digit) or a force |
| where it can act | only at a stop word (`=`, `\approx`, a line end) | any position: mid-line, `\equiv`, the 621 points D2 leaves unput |
| a wrong value | the model's wrong digits stand until a reader sees them | the wrong digit's token lowered as it is chosen |
| tokenization | the put is encoded by tir4b (the model may join digits, I4) | the model's own token choice among the right ones |

`--logit-value` at +8 (I13) was a soft bias on the value's tokens through stops: 19 / 48 against 21 / 48, 2.8x the wall time. The first
experiment of phase 1 changes one variable against it: **force instead of +8** at the same positions, with no stop (so the
wall-time cost goes, and the effect of strength alone is seen). Group C (a value given and not used, I11, S12, I15) is a hypothesis
here, not a promise: if a value written by the model itself under force is re-checked as much as a put, the channel does not fix it.

## Architecture (phase 1)

**Transport, not a new brain** *(the advisor's shape, taken)*. MIP's logic stays tir4b's `Rollout` (the reference, its rules and golden
tests); a third driver joins vLLM's and `drive_http`. llama.cpp gets a thin sampler.

- **The shim:** a `llama_sampler` (`include/llama.h` `struct llama_sampler_i`, llama.cpp `1deefcca3`) added at the head of the chain
  in `common_sampler_init` (`common/sampling.cpp`), so it sees every logit of the vocabulary before penalties, top-k, top-p, min-p and
  temperature. `accept()` receives every token: the whole prompt when a request starts (`server-context.cpp` feeds
  `prompt.tokens` through `common_sampler_accept` at sampler init) and each sampled token after. `apply()` applies the current plan.
- **When the shim asks MIP:** only after a trigger token (the stop ids tir4b computes today: `=`, `\approx`, line ends, and the ones
  phase 2 adds). Between triggers it asks nothing. Sampling for every slot runs on llama-server's main thread (`update_slots`,
  `common_sampler_sample` per slot), so a blocking call stalls every answer. *(my call)* The shim sends the event at `accept` of the
  trigger and waits only at the next `apply`, so MIP works while the GPU decodes the step; a plan not ready in a set time (e.g. 20 ms)
  is skipped, counted and logged, never waited on.
- **The plan:** a short list the shim applies by itself token by token: for each of the next k positions, a token id to force (its logit
  set above all others, so it survives every filter after the shim) or a bias map; or "end the request here" with a reason. MIP
  keeps state per answer; the shim keeps none beyond the current plan.
- **The link:** *(my call)* a loopback socket between the shim (C++, Winsock, in llama-server) and a MIP service in Python (stdlib
  `socket`), one connection per server, messages tagged by slot and request id. No new library: linking Python into llama-server (an
  embed) is a dependency and needs a conversation first (`AGENTS.md`). The cost to check in phase 1: a round trip on loopback against
  MIP's 2.5-3.6 ms a consult.
- **The driver:** tir4b sends one request per answer segment with an id the shim reports back; the MIP service runs the answer's
  `Rollout` on the streamed tokens, and returns plans. Mechanisms that need a request (below) end the request; the driver then does
  what `drive_http` does today and starts the next.
- **Where the patch lives** *(my call)*: a branch of the local llama.cpp clone (`C:/AI/llama.cpp`, MIT) with the patch kept in this repo
  (`third_party/llama.cpp/mip-sampler.patch`, the notice and base commit recorded), built as `build-w4a8-mip` (sm_89, same flags). Build
  label: **4B NVFP4 W4A8 · llama.cpp+MIP · PC layer**.

## Every mechanism against what a sampler can do

| Mechanism (round 10 flags) | Channel can do it | Needs a request (stays on stops) |
|---|---|---|
| `--inline-calc --mathbackend --inline-plain` (a value after `=`) | force / bias the value's tokens | — |
| `--no-digit-join` | bias (today's logit bias, per token) | — |
| `--approx-put` (10 digits after `\approx`) | force the digits | — |
| `--question-quote` (the question's words when the copy differs) | steer the quote's tokens (not a hint: the developer, 2026-11-12) | — |
| `--letter-count` (counts at the start) | steer the counts' tokens (not a hint) | — |
| detectors: `--guard-progress`, `--early-cycle`, `--shape-cap`, `--cover-cap`, the repeat check | read the stream, no action of their own | — |
| `--loop-close` (`</think>` on a thinking loop) | force `</think>` | the cut of the loop's text (`cut_tokens`) |
| `--answer-close` (cut the answer loop, then the cue) | force the cue's tokens | the cut |
| `--answer-rewind 2` (cut and resample with another seed) | — | all of it |
| `--execute --box --loop-hint --verify-gate` (Sandbox) | force the fence's opening (a forced block, as today) | running the block (seconds) and resuming |

A cut cannot be done by a sampler: the text is in the KV cache. The anti-loop keeps its request path (end the request, cut, resume)
until a later phase gives the shim a server hook to roll a slot back.

## Phases

**Phase 0: two measurements before any patch.**
1. *Done:* the share of wall time stops cost (above).
2. *Proposed, a GPU run:* an `n_probs` probe on stored prefixes at put positions (round 9's traces, llama-server, the PC layer): the
   rank and probability of the right value's first token where tir4b put it. It sets soft against forced and how often "add the token"
   is needed. Reported with the nine items and run only on the developer's go-ahead.

Gates 1-3 below are accepted as written (the developer, 2026-11-12).

**Phase 1: the channel and math values through it.**
- Plan type "steer a line at the start of the thinking" for a fixed-question value, besides the steered value after `=`.
- The shim, the link, the MIP service, the driver; math values (`--inline-calc` and `--approx-put`) steered at the same positions as
  today's puts in the thinking (the answer-phase puts end, rule above); everything else on its request path as today.
- Gates, in order:
  1. **MIP off = stock:** the patched server with the shim attached and no plan reproduces the stock build's greedy tokens (3 questions x
     400 tokens, as the 2026-11-08 card check), and its sampled tokens for a fixed seed.
  2. **Replay parity:** on round 9's stored traces, the MIP service's plan at every stored put position equals the recorded put (value
     and position), 0 differences; a value it would steer that today's reader did not put is listed and read.
  3. **Every touch right:** each forced value checked again independently (`analyze.py` / `hand_values.py`).
  4. A 5-minute smoke through the launcher, then the live comparison: arms on the **same patched binary** (today's text puts with the
     channel off, against values through the channel), the same rows and seeds, read by the rules of round 10 (below).
- First experiment's variable: force against `--logit-value`'s +8 at the same positions.

**Phase 2: coverage, in the detection and on the stream.** The gaps where software does not see the case: D2 (the 621 unput points,
split by cause), D3 (`\equiv` and the rest of `\approx`), D4 (forms the reader cannot read: unbalanced tails, roots, `...` decimals,
fractions, spaced factors, fractional powers, pi in degrees), D14 (a name as an operand), D16 (`--check-hand` false flags: long
multiplication carries, chain splits), the question reader on (D1), and fixed-question values. Each built from the case's statement,
tested on wordings it was not written from, replay-gated (`docs/failures.md`, "How a pattern is fixed"). This phase needs no GPU and can
start beside phase 1.

**Phase 3: loops through the channel.** P13 (loops of 60-70k tokens even with the right closing value: measure where the tokens go
first), L7 / P12 (the close too early, an unfinished step boxed), LD (a factor search: the gcd of the numbers the model divided, a local
fact), I8 (the third-repeat close cutting work), I17 (closers' own early closes), and wrong-value suppression: a digit token that makes
the value wrong lowered where MIP knows the value.

**Phase 4: the Sandbox and other modules on the channel.** Raise the probability of opening a block at signals software sees for
certain (a loop found, a question asking for all solutions), never at a guess that the model is unsure; the hypothesis, the code and its
reading stay the model's. S4 (a forced `print(` gives a blind constant), S7 (blocks too long on the 4B) and S10 (calls on a made-up
premise) apply to a forced fence as they do today. Then the module interface for sub-models (coding, probes; hidden states later),
with a rule for conflicts between modules (a value being steered when the anti-loop wants to close). Then vLLM (a logits processor,
in-process on the A100) and Strata-xeno through the same plan format.

## Coverage of every open item

| Item | Where it is taken | Note |
|---|---|---|
| P13 long loops with the right closing value | phase 3 | measure first |
| L7 / P12 close too early | phase 3 | the extraction rule above: no correcting the extracted value |
| LA confirm line | phase 3 | a line of text is a hint: out under "no hints"; the close stays |
| LD factor search | phase 3 | the gcd only as a local fact at the model's own divisions |
| D1 question reader off | phase 2 | the fixed-question exception |
| D2, D3, D4, D14, D16 | phase 2 | |
| I3 the model writes ` (exact)` after its own wrong value | phase 3 (wrong-value suppression) | |
| I7, I8 engine closes | phase 3 | engine (Strata) items; on llama.cpp through the anti-loop |
| I11, S12, I15 a value given and not used | phase 1 hypothesis | not promised: a forced value leaves the same context as a put |
| I17 closers' early closes | phase 3 | |
| C2 ops with no work budget | phase 2 | a step bound in Python first (open question 3 of the CUDAMathBackend design) |
| C4 no value audit on the engine path | phase 1 gate 3 | every forced value audited |
| S1, S2 the model rarely opens a block | phase 4 | |
| S4, S7, S8, S10 | phase 4 | reasoning-side; the channel does not reach them by itself |
| S16 box isolation gaps | outside the channel | a Sandbox item, kept on its own track |
| LF, T1-T9, D17 | not taken | the model's reasoning (LF); the latent path is paused (T, D17) |

## Round 10

The developer: "run it together with the new intervention system". *(my reading, to confirm)* Round 10 waits and runs once phase 1
passes its gates, with its arms on the patched binary: A (round 9's General, text puts), B (A + the round 9 fixes), and B with values
through the channel; C (the Sandbox) as planned. Its acceptance rules stay those proposed in `2026-11-10-round9-fixes.md` ("What counts"),
for the developer to set before the run.

## Rules changed for this design (written into `AGENTS.md` and `CLAUDE.md`, 2026-11-12, the developer's go-ahead)

- `AGENTS.md`, "Now is research: no protocol spec and no module refactor until the methods stop changing". Added: "... until the
  methods stop changing; the engine-to-MIP token channel (2026-11-12) is a research instrument, built behind the same rules, with MIP's
  logic still tir4b's".
- `AGENTS.md` (2026-11-12), "Integrating and optimising MIP into the inference engine in use is the next stage". Added: "The
  token channel is not that step: it moves MIP's interventions to the token for what they can do; optimising MIP inside the engine for
  speed stays the next stage."
- `AGENTS.md` (2026-11-11), "Never report software's answer as the model's". Added beside it: "A fixed computation is software's, not
  the model's to reason out": computed by software, steered into the thinking, the thinking not closed, the final answer the model's,
  such answers reported apart. And the rule "Steer in the thinking, never in the final answer; no hints".

## Out of scope *(my calls)*

- Speculative decoding (`--spec-type draft-mtp`): not in the locked config; the shim is built for plain sampling. A config that adds
  drafting needs its own design (draft checking in `common_sampler_sample_and_accept_n`).
- frozen10: phase 1's experiments run on development rows only (round 9's and round 10's rows); frozen10 stays for round 10's
  measurement. `--logit-value` has never run on llama.cpp (tir4b refuses it with `--server`); a +8 arm is built through the channel.

## Left to the builder

The message format on the link; the plan's timeout; where the trigger ids come from (sent by the driver at request start, as tir4b computes them); the
MIP service's process model (one per server, slots as keys); the patch's tests in llama.cpp's own test style.

## Open for the developer

1. Round 10 as read above.
2. The phase 0 probe (a GPU run).
