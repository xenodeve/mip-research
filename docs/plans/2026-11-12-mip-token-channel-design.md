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
| `--question-quote` (the question's words when the copy differs) | steer the quote's tokens (not a hint: the developer, 2026-11-12). **Built 2026-11-13:** the driver finds the wrong copy line at a chunk's end and cuts after it, as the text arm; the quote is handed to the service and steered at the next request's first sample. At a thinking loop's close the close waits one line for the steered quote, then `</think>` (closed anyway if the model wrote something else) | the cut |
| `--letter-count` (counts at the start) | steer the counts' tokens (not a hint). **Built 2026-11-13:** in the start plan with the fixed-question line, nothing inserted | — |
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

## Phase 1 results (2026-11-12, 4B NVFP4 W4A8 · llama.cpp · RTX 5060 Ti)

- **Phase 0 probe** (`results/token-channel/rank_probe.txt`, stock server, 300 puts): the right value's first token is the model's rank 1
  at 92.3 % of puts, rank 2-20 at 7.3 %, outside the top 50 at 0.3 %; the misses are mostly another form (`\frac`, a bracket, no leading
  space), not another value.
- **Gate 1:** the patched server with no `mip_id` reproduces the stock build's tokens, 6 / 6 (`gate1.txt`).
- **Gate 2:** on round 9's traces, 29,645 of 29,645 puts planned with the same text, 0 different, none extra (`gate2.txt`).
- **Gate 3 and the live comparison** (`results/token-channel/dev2/`, launcher `scripts/pc/channel_compare.sh`, commit `85f5194`, shim
  `eae8e9a01`): 10 development rows x 2 seeds, arm A General text puts and arm B General through the channel, both `--think-only` and
  `MIP_EXPR_DEVICE=cuda` on the same patched binary. Rule and API reader agree on every answer: A 20 / 20, B 19 / 20. Every touch: 2,911
  planned values, on answers that finished right 2,909 right and 0 wrong, on the one that finished wrong 2 right and 0 wrong; 2,910 written
  as steered, 1 not (its ask was one of the shim's 4 misses in 7,944 asks at the 50 ms wait). Needed: under the mask nothing shows what
  the model would have written; the probe above is the estimate (the model's own first choice at 92.3 % of puts). Helped: 1 pair where
  only one arm is right, and it is A's; a sign test needs more such pairs than 20 answers give, so this run neither shows a gain nor a
  loss. Its pair was read (`dm_simplify_surd-9937`, sample 0): the two texts part at character 172, before the first steered value (token
  5,306), so the difference is path noise; B steered 2 values, both right, then spent the thinking re-reading the question's brackets
  until the loop close (token 13,350), and the slip (`-939940` for `-899940`) is in the final answer, which nothing steers. The
  fixed-question line was steered on all 12 answers whose question has a rational value (`div_big`, `mixed_longer`, `mul_div_longer`),
  all right in both arms; surd questions get none (the planner gives a question value only for a rational result). Tokens are noise at
  this size (footnote in `docs/inject.md`).
- **Found and fixed while running it:** the value DAG cost ~36 ms a line against the 50 ms wait (a character trie: under 5 ms,
  `a831a82`); an ask read after the driver took the reply double-counted the request and misplaced one event of 2,474 (the value itself
  steered in the right place; `055c05d`); gate 3 now checks surds, powers and `P(n, k)` by sympy (426 values had been left unread).
- **Found by the whole-branch review and fixed:** a value cut inside by the driver (a loop cut or a close) could be resumed at the
  next request, after `</think>` or after changed text; a resume now needs the text before the value unchanged and the thinking open
  (0 such cases in dev1); `--channel` with `--whole-hint` is refused; a plan replaced before it finished is counted dropped. In the
  shim (`eae8e9a01`, live test 4 / 4 on the RTX 5060 Ti): a reconnect that could hang the server thread, a ~2 s connect when the
  service is down (now a 100 ms non-blocking connect; a request then finishes in under 1 s), and a plan's node indexes checked.
- **Gate 3 itself:** it now reads `gcd(a, b)` and `\div`, and decodes 200 tokens at an event, not 40 (a long question line written a
  digit a token had been cut short and shown as differing); 0 values unchecked in dev1 and dev2.
- **On rows the model misses** (`results/token-channel/miss1/`, 2026-11-13, same binary, flags and launcher; development rows): 16 rows
  where round 9's PC-layer General arm was wrong on at least one seed (`frozen9a-stay`, `frozen9b`; 4 dropped where the API reader judged
  every miss right), x 2 seeds: DeepMind `div_big` 5, `mul_div_longer` 2, `swr_p_sequence` 2, `simplify_surd` 1, `mixed_longer` 1, and
  5 MATH-500 level 5. Rule: A 19 / 32, B 18 / 32. API reader: A 19 / 32, B 21 / 32; the two disagree on 7 answers, all read, and the reader
  is right on each (a boxed `-1.9887` the rule took; exact fractions and 40-digit repeating decimals the rule refused). Pairs where only one
  arm is right: reader B 7, A 5, one-sided sign test p = 0.39; rule B 7, A 8, p = 0.50. **No gain beyond noise.** Where the texts part:
  8 of the reader's 12 such pairs part before B's first steered value (path noise: all 5 of A's, 3 of B's). The other 4 part at
  character 0, all B's, all on the 16 answers whose question has a rational value, where B's first steer is the fixed-question line:
  there the reader gives A 9 / 16 and B 13 / 16, 4 to 0 (p = 0.06). Arm A has no question line (`--question-value` off), so these 4
  measure the question line and the channel together. After the question line the model closed its thinking itself on 4 of the 16
  (34-49 thinking tokens): 3 right, 1 boxed a rounded decimal (`div_big-5070` sample 1). Tokens: on the 16 question-line answers B
  249,601 against A 318,552 (median 11,557 against 14,407, B shorter on 11); on the other 16, B 368,750 against A 344,420.
- **Every touch in miss1:** 5,699 steered values. On answers that finished right, 3,517 checked right and 0 wrong; on answers that
  finished wrong, 2,180 right and 0 wrong; 2 not read by gate 3, checked by hand: `\sqrt{11.7^2 - 9^2} = \frac{9\sqrt{69}}{10}` is right,
  and `\sin(45) = \frac{\sqrt{2}}{2}` is true in degrees (the model's next line writes `\sin(45^\circ)`) but not if the bare 45 is read as
  radians. MIP reads an angle with no unit as degrees when the text works in degrees (`tools/igc/unified_detect.py:348`); whether that
  holds the General Style's "true under every reading" is open for the developer. 5,665 values were written as steered; the other 34 are
  `cut`: steered into looping text that the loop close then cut and replaced with `</think>` and the answer (2 answers); nothing was
  steered after the close. 7 asks missed the 50 ms wait (2 in the first attempt, 5 of 11,933 after the resume), none at a steered value.
- **Found while running miss1:** under `-kvu` the server's `-c` is one KV pool shared by every slot, not a window per answer. With 12
  answers in flight (one at 84k tokens) the 262,144 pool filled at minute 47, the server answered 12 requests with "Context size has been
  exceeded" (HTTP 500) and both arms stopped. The launcher now takes `CTX` (a 393,216 pool is 12.4 GB on the RTX 5060 Ti, ~21.5 KB a
  token); tir4b still caps each answer at 262,144 (`CONTEXT`), and the server reports `n_ctx_slot = 262144`. The run went on with
  `--resume` (A kept 12 finished answers, B 10) at 3 answers per arm in flight. Gate 3 now reads pi, trig, `\pm` and tuples by sympy,
  leaves a trig function of a bare number unread, and counts values at or past a loop close as `cut`; dev1 and dev2 read as before.
- **miss1's wrong answers, read in full** (B's 11; 2026-11-13): 3 copied the question's letter string wrong (`swr_p_sequence`), which
  `--letter-count` reads right but was off, and under the channel it must be a steered start line like the question line, not
  inserted text (not built); 1 wrote `1.01^{30} \approx 1.3448888` (`stdmath-45fa3ef2`), which `--approx-put` (off in miss1) and the
  new cut put at `=` ([failures](../failures.md#p11-the-models-own-approx-value)) cover; 1 looped on a factor search of 12301 past
  `</think>` (`div_big-1395#0`), anti-loop work; 2 boxed a rounded decimal or the integer part (final answer, out of reach by rule);
  2 gave four roots where the gold takes `\cot^{-1}(y) = \tan^{-1}(1/y)` (right under the `(0, \pi)` range the model used) and 2 are
  a MATH-500 question whose figure is in the solution, not the question. Scoring stays as it is: an equal value in another form is
  right with a note, a rounded value or another root set is not (developer, 2026-11-13).
- **Not yet run:** the first experiment's variable (forcing against `--logit-value` +8 at the same positions), an arm with the
  fixed-question line as text (to tell the question line from the channel), and a frozen set. The next channel arm turns on
  `--approx-put` and `--letter-count` as round 10's arm B planned, each counted apart.

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

**Nothing inserted under the channel (the developer, 2026-11-13):** "convert at once, so the results are not contaminated by the old
mechanism; measure separately how each mechanism is used". `--letter-count` and `--question-quote` are steered, not inserted (table
below); `_long_found`'s puts are their own kind (`long value`); gate 3 reports every count per mechanism (`what`: inline value, long
value, approx value, question value, letter count, question quote): plans, done / dropped / missed / cut, value right, written as
steered, on right and wrong answers. The quote and the letter line are MIP's wording around the question's own text
(`The question writes: $<expr>$.`, `The string <s> has <n> letters: <letter> <count>, ...`), the same wording the text arm puts.
The fixed-question line is one flag for both arms, `--question-line` (put as text, steered under the channel; kind `question line`,
apart from `--question-value`'s `question value`), so a text arm and a channel arm both carry it (miss1's channel arm had it and the
text arm did not: its 4-0 on question-line answers is that confound). Parity on round 9's 240 General traces
(`scripts/pc/channel_parity.py`, `results/token-channel/channel-parity.txt`, CPU): the channel's letter-count plan equals the text put
on 10 of 10 answers; its quote at a wrong copy (cut and plan) equals the text quote on 4 of 4; its quote at a loop close (the quote
steered, then the close) gives the same text as the text arm's on 41 of 41 of round 9's own close positions. A close or an end the model
writes on the quote's line is its own (no second `</think>`).

**Not limited to `=` (the developer, 2026-11-13: MIP reads the stream in real time, so values and other reasoning can be caught without
`=`).** Open, not built. Found so far: (1) the shim takes a plan only for an ask it sent (a trigger token or a request's first sample;
triggers go out once per server in `hello`), so a plan at a place with no trigger needs more triggers (each ask decodes the answer under
one lock) or a shim change (a plan the service pushes, taken at the next sample), which reopens gates 1 and 2; (2) a survey of round 9's
`<expression> is / -> / \to / gives <number>` (`scripts/pc/no_equals_survey.py`) shows those words carry other meanings in the model's
work (a borrow `6-8 -> 16`, a quotient digit `209 \div 723 -> 0`, a place `Digit at 10^3 is 8`, an integer part `16/5 is 3`), so each
form needs its own detector from its own statement, never a put at every link word; its remainder counts are not read by hand yet;
(3) loop kinds read from the stream need no plan (the service already sees every token): the cheapest under today's protocol.

**Confirmed (the developer, 2026-11-13):** round 9 and the fixes measured since are the base every arm builds on ("it has proven
many things work"); a channel arm is that base with its values through the channel, never a thinner flag set. miss1 broke this (dev2's
flags: no `--approx-put`, `--letter-count`, `--question-quote`, `--answer-rewind`), so its result is the channel on a thin base only.

The developer: "run it together with the new intervention system". Round 10 runs on the patched binary with three arms: A (round 9's
General exactly, the developer 2026-11-13: "ให้เทียบกับรอบ9ได้"), B (A + the round 9 fixes, put as text) and CH (B through the channel);
the Sandbox arm is a separate decision. Config, arms and the short run's result: [round 10](2026-11-13-round10.md). In short: on 19
rows picked for reach, CH 36 / 38 against A 27 / 38 by the API reader (pairs 9 / 0, p = 0.002) and B 35; the gain is on the questions
that are one fixed computation; every one of 5,789 steered values right.

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

1. ~~Round 10 as read above.~~ Confirmed 2026-11-13 (above).
2. The phase 0 probe (a GPU run).
