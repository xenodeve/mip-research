# Inject: how a computed value or a note gets into the model

Every way MIP puts something into the text the model reads next: an exact value after the model's `=`, the form that value is spelled in,
the guards that decide when not to put one, notes and hints written around values, the soft and forced alternatives to writing text, and
the engine path that delivers values inside Strata's decode loop. Detection (where an `=` is read), loop closers and the sandbox have their
own pages; this page covers what is written once a value exists.

There are two hosts. `tir4b` ([`../tools/igc/tir4b.py`](../tools/igc/tir4b.py)) ends a vLLM or llama-server request at each `=` and starts
the next with the put appended. The Strata engine (branch `xeno/mip-sandbox` of Strata-xeno, not in `main`) puts the value inside its own
decode loop, as token ids, between MTP draft windows.

## Summary

| Mechanism | What it does (one line) | Runs on | In best config? | Best measured result | Status |
|---|---|---|---|---|---|
| [Inline put, General Style](#inline-put-general-style) | After a model-written `=`, appends the exact value in a form true under every reading of `=` | tir4b (vLLM/llama-server) | yes | H3 (1,526 fresh traces): 94.4 % of plain-arithmetic `=` put, 0 false of 41,176 | in best config |
| [`--inline-plain` and the `(exact)` note](#--inline-plain-and-the-exact-note) | Puts the value alone, with no source note | tir4b | yes | Qwen3.5-4B: a ` (exact)` mark draws a re-check 22-24 points more often than a bare value | in best config |
| [`--no-digit-join`](#--no-digit-join) | Holds digit-led tokens down for one token after a put that ends in a digit | tir4b (vLLM) | yes | Digit joined onto a put: 92 of 21,398 -> 0 of 17,485 (General) | in best config |
| [Echo guard](#echo-guard) | The 3rd value in a row for one expression, each within 64 tokens, is left to the model | tir4b | yes | Echo occurs in 7 of 48 stored answers (replay) | in best config |
| [Refusals](#refusals) | No put where the literal value is not what `=` means there | tir4b; engine has its own | yes | Truth gate 0 false puts on dev and fresh sets | in best config |
| [Model profiles](#model-profiles) | A family layer that picks among the General Style's true forms | tir4b `--inline-style qwen` | no | Wrong-row rerun: 67 -> 79 of 104 (control 73 -> 71) | built, measured, kept as option |
| [Arithmetic mode](#arithmetic-mode) | Asks for the whole step as one expression, reads the question's own computation, hands off to the answer | tir4b (llama-server) | no | DeepMind 160, thinking off: 156 / 160 against 118, 0.05x tokens | built, measured, kept as option |
| [`--check-hand`](#--check-hand) | Checks each hand value at its line end and writes a correction | tir4b | no | DeepMind 160: 116 against base 121 | built, measured, kept as option |
| [Soft logit override](#soft-logit-override) | No text put; software's next value token gets +8 logit | tir4b (vLLM) | no | 19/48 against 21/48; re-check 37.6 % against 38.5 % | built, measured, kept as option |
| [Re-check hold-down](#re-check-hold-down) | Re-check words held at -30 logit for 60 steps after a put | tir4b (vLLM) | no | lb1 48 rows: median tokens 10-15 % lower, right 19/48 | built, measured, kept as option |
| [Program hints](#program-hints---hint-calc---trust-line) | CoRT's hint opens a python block at tedious arithmetic; a trust line after it | tir4b | no | not counted (the model's own blocks never ran) | built, not measured yet |
| [Prompt hints and few-shot](#prompt-hints-and-few-shot) | Text added to the system turn, after `<think>`, or as GSM8K exemplars | Strata engine bench | no | "marked values are from a calculator" hint: rejected | built, measured, kept as option |
| [Engine token-result bridge](#engine-token-result-bridge) | The executed value is emitted as the answer's own tokens at answer start | Strata engine (xeno/mip-sandbox, not in main) | no | 2-digit + - x: 120/120, 14.0 / 13.1 / 16.4 tokens against 35.4 / 32.9 / 54.8 | engine only |
| [Engine inline execution](#engine-inline-execution) | `A op B =` in generated tokens runs on the GPU executor; the value goes in as one forced draft window | Strata engine (xeno/mip-sandbox, not in main) | no | 5-digit multiplication: tokens -72 %, time -56 % (ABBA); GSM-Hard 200: tokens -22 %, decode time -18 % | engine only |
| [Engine notes: sum, repeat, close](#engine-notes---igc-sum-note---igc-repeat-note---igc-close-repeat) | ` (exact)`, ` (already confirmed, ...)`, a stop on sums, a close on the 3rd repeat | Strata engine | no | Sum note 1: delivered-sum events 13 -> 0 (24 runs) | engine only |
| [Latent output path](#latent-output-path-output-mapper) | A learned mapper adds the result to the head input instead of writing text | Strata engine, diagnostic build | no | functional checks only | engine only |

## Inline put, General Style

**Trigger.** Every request ends at a `=` token, in thinking and in the answer. The runner reads the text before the `=`
(`inline_found`, [`../tools/igc/tir4b.py`](../tools/igc/tir4b.py) line 302, then `_inline_general` line 322). With `--mathbackend` the
expression comes from `unified_detect.expression_before_with_mathbackend` (roots kept as roots, exact trig and logs, `a^b mod m`, gcd, `C`,
lowest terms); otherwise from `mathexpr.expression_before`. It refuses a symbol, a word, a mixed number, or a root or power that is not
exact.

**What is written.** ` value` as plain text, then decoding resumes. The form is chosen from the notation, never from how often a model
wrote one form in a set (`INLINE_STYLES['general']`, line 241):

| The model wrote | Put | Why it is true under every reading |
|---|---|---|
| an integer result | the integer | it is the value |
| `a/b` that reduces | the reduced fraction | lowest terms |
| an uneven quotient worked as a division (remainder, `÷`, bringing down) | `q rem r` | true as a quotient and as a step |
| an uneven quotient otherwise, `a/b` already in lowest terms | the cut decimal `1.1052...` | cut and marked, never rounded into a false equality |
| a power in a line that works mod m | `V ≡ r` | both the value and the residue |
| a decimal under a root | `\frac{23\sqrt{2}}{2} \approx 16.2635` | exact form with its decimal |
| a cube root or fractional power | the cut decimal | no exact short form exists |

**Flags.** `--inline-calc --mathbackend --inline-plain --inline-style general` (the style is the default).

**Measured** (mip-research#19, H3 section): H3 is 1,526 traces of models and sources no rule was written from (Qwen3-235B-A22B 303,
Gemini s1K-1.1 about 1,000, DeepSeek-R1 s1K-1.1 123, DeepSeek-R1 OpenThoughts science 100), measured once on frozen code `1d573ae3`:

| Measure | 4B dev arms | H1+H2 (dev) | H3 (fresh) |
|---|---|---|---|
| plain-arithmetic `=` put / should put | 96.4 % | 96.0 % | **94.4 %** (30,194 / 31,996) |
| false puts (truth gate) | 0 of 39,253 | 0 of 47,955 | **0 of 41,176** [^h3unchecked] |
| model's next value equals the put | 34,274 / 35,969 | 37,846 / 39,515 | **31,621 / 32,812** |

On the A100 (Qwen3.5-4B bf16, vLLM, T 0.6, 2 samples, full 262,144 context, 162 answers right and finished in all three round-1 arms),
the inline put cut the median tokens to the first right value from 1,068 to 985 (-8 %) and the tokens after it from 5,510 to 5,134
(-7 %); software computed 19,546 of 19,636 `=` (mip-research#19, "Loop round on the A100", part 4). [^generalacc] [^recheckwords]

## `--inline-plain` and the `(exact)` note

**What it does.** Without `--inline-plain`, tir4b writes ` value (exact)`, the engine's form; with it, the value alone.

**Measured** on Qwen3.5-4B (Strata-xeno#298, "Guideline arms at the same points": 42 points from 7 failing items, A100, 3 samples per arm and
point): against a value followed by a bare new line, the ` (exact)` mark raised the re-check word within 300 characters by +24 and +22 points
(41 -> 65 %, 35 -> 57 %), and the sentence `This value is exact and final.` by +27 and +35 points. The model's first word after the mark was
`wait` in 42 of 126 samples, against 12 of 126 without it. The best config therefore writes plain values. [^notemodel]

## `--no-digit-join`

**Trigger.** A plain put whose value ends in a digit. **What is done.** The next request is one token long with every digit-led token at
logit -100; after a trailing `.` only a digit is held (`join_guard`, tir4b line 829, set at line 1017). It never changes a value; it stops
the model running a number on (`-5` then the model's `75` read as `-575`; `8^{34} = ...` followed by zeros without end).

**Measured** (mip-research#19, "Rerun of the wrong rows", A100, Qwen3.5-4B, 52 rows x 2 samples): a digit written straight after a put,
against puts whose value ends in a digit: General 92 of 21,398 -> **0 of 17,485**; Qwen profile 234 of 35,319 -> **0 of 15,282**. [^joinsheld]

## Echo guard

**Trigger.** The 3rd value in a row for the same normalized expression, each within 64 model tokens of the last (`ECHO_RUN = 3`,
`ECHO_GAP = 64`, tir4b lines 89-91; `echoing()` line 1107). **What is done.** No put; the model writes the value itself. It never calls
the loop handler, so a repeated small expression (`2^2`, `180-45`) in a long derivation is not treated as a loop.

**Measured** (mip-research#13, fix `4dc5e546`): replayed on the 48 stored `inlineclose` traces, an echo occurs in 7 of 48 answers;
`livebench-652451e2` has 100 of them on `-1/2`. The record field `echo_skips` counts them. [^echo41]

The engine has a different rule for the same situation: the 3rd sight of an expression in thinking closes the thinking (see
[Engine notes](#engine-notes---igc-sum-note---igc-repeat-note---igc-close-repeat)).

## Refusals

A refusal never writes a wrong value; it only leaves the `=` to the model. In tir4b (`_meant_otherwise`, line 676, plus the reader's own
refusals) there is no put on: a decomposition (`601 = 560 + 41`), an operand of words (`cube root of`, `half of 68 + 10`, mip-research#4),
a count bracket (`18 girls (12+6)`), a borrow in a column (`0-9 = 1`), a lone number (`m=4,512=8*64`), nothing new, a congruence with no
modulus, a redefined operator, coins, the tail of a longer expression (`C(9, 3) + gcd(15, 25)`), bare `C(` / `P(` (point names; only
`\binom`, `comb(`, `perm(` count, mip-research#14), and a function the text defined itself (mip-research#18 audit). Question-shaped ops
(`disphenoid_inradius`, the n-gon rectangle patterns) were taken off the inline path (mip-research#14, `4dc5e546`).

The engine's parser refuses after any of 1,010 pack tokens that tie the next number to something earlier (`igc_binding_tokens.hpp`,
generated by [`../tools/igc/binding_tokens.py`](../tools/igc/binding_tokens.py)), a `(` against a word, a letter next to a number, and
unary minus (Strata-xeno#276, "Ladder rung 1"). Detection detail is on the detect page.

**Measured** (mip-research#19, bug gate on `1d573ae3`): sim_replay over every stored trace, 32 of 32 runs exit 0; truth gate 0 false puts
in the seven 4B arms (0 of 39,264), H1+H2 (0 of 48,158) and H3 (0 of 41,176).

## Model profiles

**What a profile is.** A layer on the General Style that only picks among the true forms the General Style already has at a decision point
(`true_forms`, tir4b line 268; `INLINE_STYLES['qwen']`). It never writes a value of its own, and the truth gate applies to it as to General.
Per family, not per model; a rule is written from a tune half of the family's traces (split by question id) and kept only if it also
gains on the check half and on another family member (mip-research#19, "Model profiles, the new approach").

**The Qwen profile** (`--inline-style qwen`, commits `340fa5ef`, `52fa92ac`). `pick_forms.py`'s fixed rule (at least 20 points per key; a
form replaces General only at more than 2 points higher agreement) picked one change: `a/b` already in lowest terms gets no cut decimal.

| Set | points | General agreement | profile agreement | net tokens General -> profile |
|---|---|---|---|---|
| tune (4B) | 5,936 | 77.0 % | 81.8 % | +4,379 -> +7,635 |
| check (4B, held out) | 6,905 | 82.9 % | **87.1 %** | +7,728 -> **+10,801** |
| Qwen3-235B (held out) | 3,797 | 70.4 % | **73.1 %** | -3,439 -> **-2,235** |

**Live** (A100, Qwen3.5-4B bf16, vLLM, T 0.6, 2 samples, seed 1, 262,144 context, 120 fresh rows: 80 DeepMind + 40 MATH-500 level 5):
the rows any arm got wrong (52 rows, 104 answers), rerun with `--no-digit-join --guard-progress`: Qwen profile **67 -> 79**, General 61 -> 73,
control 73 -> 71; loops (`stuck`) Qwen 32 -> 22, General 43 -> 29, control 31 -> 30 (mip-research#19, "Rerun of the wrong rows"). [^profilepair]

**Where it lives / what it takes.** `tir4b.py --inline-style qwen` with the General flags; nothing else. [^oldprofiles] [^cutdecimal]

## Arithmetic mode

For questions where the whole question is a calculation, with thinking off. Three steps:

| Step | Flag | What software does |
|---|---|---|
| Whole step | `--whole-hint --whole-plain` | Opens the reply with `WHOLE_HINT_PLAIN` (tir4b line 68): one plain expression in the question's order; the value is computed exactly (sign and uneven quotient included) and ends its line |
| Question's computation | `--question-value` | [`../tools/igc/question_read.py`](../tools/igc/question_read.py) reads the computation from the question (symbols or words: `less than`, `take away`, `product of`); the whole value given is the question's, catching an operand the model copied wrong |
| Hand-off | `--hand-off` | A line that only restates software's value (`= v`, or a 30+ place expansion) is cut and the model is led to `So the final answer is \boxed{`; when the model restates the value exactly, its answer is closed with that value; an exact fraction closes with the fraction |

**Measured** (Strata-xeno#298, pc4-pc9): DeepMind arithmetic 160 rows, Qwen3.5-4B base, llama-server on the RTX 5060 Ti, greedy, thinking off:

| Arm | right / 160 | tokens vs base |
|---|---|---|
| base | 118 | 1.00 |
| `--whole-plain` | 138 | 0.11 |
| + hand-off close (pc8) | 151 | 0.05 |
| **+ fraction close + question value (pc9)** | **156** | **0.05** |

pc9 against pc8: +5, lost 0. The question reader matched gold's exact value on 160 of 160 rows offline and fired on all 4 copy-error rows
in pc9. [^arithtuned] [^arithscope]

**What it takes.** `tir4b.py run ... --no-think --greedy --inline-calc --inline-plain --whole-hint --whole-plain --question-value --hand-off`.

## `--check-hand`

**Trigger.** Every ended line in thinking and in the answer (`check_line`, tir4b line 1338). `mathexpr.wrong_relations` reads every
relation on the line (`=` chains, `\approx`, arrows, congruences); a hand value that differs from the exact one gets a correction line.
Whole-number work written with `=` is exact; `\approx` keeps a slack for the places written.

**Measured.** DeepMind 160 (Qwen3.5-4B, L4, thinking off): 116 right against base 121 at 0.99x tokens; it corrected 22 hand values on 17
rows, mostly right, but the model redid the work by hand after each correction (mip-research#12). On reasoning misses it caught 4 of 14
wrong hand values (mip-research#12). On H3 its corrections fell from 432 to 303 after two rounds of misread fixes (mip-research#20). [^checkhand]

**What it takes.** `--check-hand` on any tir4b arm. Not in the best config.

## Soft logit override

**What it does** (`--logit-value 8`, tir4b `start_plan` line 863). At an `=` whose expression is read, nothing is put as text; the value's
tokens are asked one at a time with `logit_bias {token: +8}`, so the model writes the value itself.

**Measured** (Strata-xeno#298): Qwen3.5-4B bf16, thinking on, T 0.6, seed 2, 48 `lb1-rows` (LiveBench + AIME/SMC), vLLM on an A100 40 GB:
19/48 right against ctrl 21/48; re-check word within 300 characters after a computation 37.6 % against 38.5 %; the whole value was written
in 2,895 of 3,439 cases (84 %). Forcing a value's tokens through the sampler and inserting them as text gave the same 48-token greedy
continuation at 25 of 30 points, the same rate as two identical insertions (bf16 noise). [^logit]

## Re-check hold-down

**What it does** (`--suppress-recheck 60`, tir4b line 162). For 60 steps after a value software gave, 45 token ids of the re-check words
(wait, hmm, verify, check, double, again...) get logit -30.

**Measured** (Strata-xeno#298, same 48 `lb1-rows`, A100): with `--inline-calc --inline-plain`, the re-check word within 300 characters fell
from 47.2 % (inline) to 16.1 %; any re-check within 900 characters 78.3 % -> 55.4 %; median tokens 15,973 against 17,758-18,902 in the
two controls. Right 19/48 against 21 (inline) and 21 / 25 (controls). [^suppress]

## Program hints (`--hint-calc`, `--trust-line`)

**What they do.** `--hint-calc`: at an `=` that ends tedious hand arithmetic (a 3-digit number, 3 operators, or a value past 999; at most
once per 1,024 thinking tokens) the runner takes the line back and opens a python block with CoRT's hint `It looks tedious, and we can use
python code to simplify the reasoning.` (tir4b line 53). After a run that printed, `--trust-line` writes CoRT's `We don't need to doubt the
accuracy of python calculations.`, none, or ATTC's positive sentence (lines 54-59). `--loop-hint` cuts a thinking loop to one round and
opens a program. The sandbox page covers block execution. [^hintcalc]

## Prompt hints and few-shot

The engine bench ([`../tools/igc/external_bench.py`](../tools/igc/external_bench.py)) can add text to the prompt:

- `--hint` / `--hint-file`: text added to the system turn (the file form for rules too long for a command line).
- `--think-hint` / `--think-hint-file`: text written after the opening `<think>`.
- `--shots N`: the first N of Wei et al.'s four GSM8K chain-of-thought exemplars before each GSM8K question
  ([`../tools/igc/few_shot.py`](../tools/igc/few_shot.py)).
- [`../tools/igc/rules/`](../tools/igc/rules/): `tool-rules-system.txt` (four mandatory run_python rules) and `tool-rules-think.txt` (the
  same restated as the model's own sentence), used with the two hint files on the sandbox's tool-route arms.

**Measured.** A system line saying marked values come from an external calculator and only the setup needs checking (Strata-xeno#276,
"System line about marked values: rejected", engine `3e85ce835fd800a1`, Gyro-M, 60 selected cases, one run per arm): right after reading
52 without, 49 with; tokens 148,171 against 149,112; re-check sentences 290 against 333; and a new loop where the model copied the note
`(already confirmed)`. The hint stays out. [^prompthints]

## Engine token-result bridge

**What it does** (`--igc-result-bridge`). After the learned controller's first-anchor call executes exactly on the device, the engine
emits the result as fixed answer tokens (`result_tokens`, Strata-xeno xeno/mip-sandbox: include/strata/program/igc_learned_diagnostic.hpp:300:
`The result is N.` then `</think>` and the digits), but only when both operands are written in the prompt
(`written_in`, igc_learned_diagnostic.hpp:291-292). It replaces the learned output mapper.

**Measured** (Strata-xeno#276, "Token-result bridge", commit `72fc51f6`, engine `e5a86f03`, bundle v6, Gyro-M D2x, EleutherAI two-digit
arithmetic, first 40 per task):

| Task | Correct, off / bridge | Tokens per case, off -> bridge | Seconds per case, off -> bridge |
|---|---|---|---|
| Add | 40/40, 40/40 | 35.4 -> 14.0 | 0.98 -> 0.91 |
| Sub | 40/40, 40/40 | 32.9 -> 13.1 | 0.85 -> 0.79 |
| Mul | 40/40, 40/40 | 54.8 -> 16.4 | 1.12 -> 0.88 |

[^bridge]

**What it takes.** As for inline execution below; `--igc-inline-exec` turns the bridge on by itself (generate.cpp:2819).

## Engine inline execution

**Trigger.** After each accepted token, `parse_inline` (Strata-xeno xeno/mip-sandbox: include/strata/core/igc_inline.hpp:362) looks for
`A op B =` (chains, parentheses, percent, integer powers, decimals) at the end of the generated tokens
(igc_learned_diagnostic.hpp:563). Operands come from the model's own text, not from a learned head.

**Compute.** Integer programs run step by step on the GPU exact executor (`execute_exact`; powers on the ABI2 executor); decimals run on
the rational executor and are spelled by long division (`rational_tokens`, igc_inline.hpp:466): a value with no terminating decimal within
12 places is not put, nothing is rounded (igc_learned_diagnostic.hpp:568-631).

**Delivery state** (igc_learned_diagnostic.hpp). The value's token ids, plus the note, go into `bridge_tokens_` with a cursor `bridge_at_`.
`bridged_unread_` counts tokens a window took at once so they are not read back as the model's writing (line 419). `log_inject`
(line 371) writes every injected id with its reason (`strata igc inject: <why> at=<token> ids=...`), so the record says which mechanism
acted (mip-research#11). An end of turn inside thinking after a call is replaced by a newline and `</think>` (line 386). After 256 values in
one thinking the thinking is closed (`kMaxValues`, igc_inline.hpp:159; line 646).

**MTP-aware delivery** (Strata-xeno xeno/mip-sandbox: src/program/generate.cpp). In the serve loop, when a value is pending
(`delivering()`, line 14485) the window length is `min(bridge_left, S_mtp)` (line 14488): the known value goes in as **one forced draft
window of up to S_mtp tokens**, not one round per token. The window's draft slots are filled from the value (`bridge_ahead`, line 14528),
suffix and lookup drafts are skipped while delivering (lines 14495, 14505), and the sampled outputs are replaced by the value's tokens
(`bridge_sample`, lines 14688-14691). Acceptance stops at the token that triggered a call, because the rest of the window followed a value
the model never saw (lines 14697-14704). In `--batch` slots a delivery goes one token a window, and a slot's MTP draft after a forced
token is dropped (`slot_token`, igc_learned_diagnostic.hpp:341; generate.cpp:11960-11965).

**Measured.** All on Qwen3.8-Flash-Next (Gyro-M Swift 1.5 IQ2_XS, D2x profile), greedy, thinking on, this PC.

ABBA, order on1, off1, off2, on2 in one session, `--igc-result-bridge --igc-inline-exec` against the same build with both off (Strata-xeno#276,
"ABBA: inline execution on and off, Gyro-M D2x"):

| Suite | right on1, on2 / off1, off2 | tokens on / off | seconds on / off |
|---|---|---|---|
| 5-digit multiplication (40) | 39, 39 / 40, 39 | 137, 142 / 491, 522 | 2.80, 2.82 / 6.25, 6.61 |
| 3-digit multiplication (40) | 40, 40 / 40, 40 | 74, 72 / 118, 120 | 1.75, 1.68 / 1.87, 1.90 |
| MATH-500 integer answers (60) | 58, 58 / 58, 57 | 854, 874 / 1,035, 1,085 | 15.7, 15.3 / 16.6, 17.5 |
| GSM8K (100) | 94, 93 / 95, 94 | 290, 298 / 297, 298 | 6.14, 5.92 / 5.46, 5.51 |

The issue states: 5-digit multiplication tokens fell 72 %, time 56 %; MATH-500 tokens 18 %, time 8 %. 3- and 5-digit add and subtract were
40/40 in all four arms. [^abbalatency] [^abbabuild]

GSM-Hard, first 200 of 896 positive-integer cases of GSM-Hard v2, engine `3a6373d5119316aa`, full context (Strata-xeno#276, "GSM-Hard 200"):
right after reading disagreements 177 on, 178 off; tokens per case 571 against 731 (**-22 %**); decode time 1,942 against 2,363 s
(**-18 %**). [^gsmhard]

48 word problems with long numbers ([`../tools/igc/natural_problems.py`](../tools/igc/natural_problems.py), not a benchmark): 48 right on and
off; tokens 249 against 457; seconds 4.21 against 6.58 (Strata-xeno#276, "Sidecar in the engine", parser ownership fix arm). [^wordproblems]

BIG-bench 3- and 5-digit add, subtract, multiply, 40 each: 240/240 on and off; 5-digit multiplication 146 tokens per case on against 485
off, 3-digit 68 against 121 (Strata-xeno#276, "Ladder rung 1"). [^ladder]

**What it takes.** A `STRATA_IGC_NATIVE_DIAGNOSTIC` build of `xeno/mip-sandbox`; `--igc-learned-diagnostic-bundle <file>` (the controller
exists only when the bundle path is set, generate.cpp:3872-3882); `--igc-inline-exec`; `--batch-groups 1` when batching
(generate.cpp:12096-12099). The bench drives it with `external_bench.py --bundle <file> --inline`.

## Engine notes: `--igc-sum-note`, `--igc-repeat-note`, `--igc-close-repeat`

What follows a value delivered inside thinking (`after_value`, Strata-xeno xeno/mip-sandbox: include/strata/core/igc_inline.hpp:201):

| Seen before | Default text | Flag |
|---|---|---|
| never | ` (exact)`, with `).` on a top-level sum, `)` otherwise | `--igc-sum-note N` (0 open, 1 stop on sums, 2 no note on sums; default 1) |
| once or more | ` (already confirmed, stop checking)` | `--igc-repeat-note N` (0 stop checking, 1 `(already confirmed)`, 2 `(already confirmed, next step)`; default 0) |
| `close_at - 1` times | `.` newline `</think>` | `--igc-close-repeat N` (default 3, 0 never) |

`--igc-no-note` delivers the value alone; `--igc-no-reply` delivers no value after the thinking closes.

**The delivered-sum rule** (mip-research#1). With the note left open, the model read the value of a whole sum `a + b` as its first term and
added the last term again: `5*8387270 + 8 = 41936358 (exact) + 8 = 41936366`. A closing `.` only when the expression's top level is `+` or
`-` removes it, while products such as `2000*1.05 = 2100 (exact) per week` keep an open note.

| Measured on (Gyro-M, Strata engine) | Mode 0, open | Mode 1, stop on sums | Mode 2, no note on sums |
|---|---|---|---|
| 8 questions the fault reached, 3 rounds each (exe `4912153df3354050`) | 13 events in 10 of 24 runs | **0 events, 0 of 24** | 0 events, 0 of 24 |
| right on those 24 runs | 15 | 15 | 15 |
| 49 questions, 2 rounds (exe `b324679801c45fdc`) | 2 events, 73 of 98 right, 247,162 tokens | **0 events, 77 of 98 right, 245,142 tokens** | — |

10 of 24 against 0 of 24 is a Fisher two-sided p of about 0.0006. On the 49 questions tokens were 0.992x (sign-flip p 0.90) and right
answers net +4 (p 0.29): no measured cost or gain. One sentence was cut by the closed note in 270 closed notes, in a run that was right.
`--igc-sum-note` defaults to 1 since then; `--igc-inline-exec` itself stays opt-in.

**The third-repeat close.** Removing it (Gyro-M, 49 selected questions, one run each) added 43.7 % tokens (117,495 against 168,882; more on
36 questions, fewer on 13, sign-flip p 0.0012); right after reading 40 with the close against 37 without (Strata-xeno#276, "The
third-repeat close stays"). [^closerepeat] [^repeatnote] [^notegyro]

## Latent output path (output mapper)

The activation-side alternative to writing text: the learned output mapper adds the executed result to the final head input, before the
native head quantizes and samples (`--igc-output-diagnostic-artifact`, `--igc-output-diagnostic-fixture`; diagnostic builds only). The
learned-integration page covers the mapper and its training.

- **Controlled output injection** (Strata-xeno xeno/mip-sandbox: docs/reports/2026-10-29-igc-controlled-output-injection.md): a fixture of
  checked results bound to the request prefix; on 2 known prompts the enabled arm gave `92 - 18` -> 74 and `12 * 64 = 768`.
- **Native output forward** (docs/reports/2026-10-29-igc-native-output-forward.md): the Torch-free CUDA mapper matches the frozen Torch
  mapper at all 421 captured positions within 7.63e-6, and all 421 token decisions through the native Q8 head match; memcheck, initcheck,
  synccheck and racecheck report 0 errors after a shared-memory race in the reduction was fixed.
- **Runtime output connection** (docs/reports/2026-10-29-igc-runtime-output-connection.md): the fixture holds a typed `ExactRequest`
  (operands, no answer); the engine runs the exact CUDA execution on the verifier stream and the mapper consumes the device result, with
  no host arithmetic. 30 of 30 typed results match the independently checked bytes; 30 known questions gave consistent correct numbers.

[^latent]

**What it takes.** A diagnostic build of `xeno/mip-sandbox`, single-request serve, no batch, no layer split, and the two artifact/fixture
files; operands still come from a fixture or the learned input heads.

[^h3unchecked]: 289 puts on H3 were not checked: values past the exact field with no float, and puts the checker cannot read.
[^generalacc]: Note: on the same 120 fresh rows the General Style alone did not raise accuracy: 197 of 240 against 209 for the sampled control
    (General right and control wrong 12, the reverse 24, sign test p about 0.07). Every wrong answer in both arms was a loop; the arms part at
    the first put, so this is one unpaired round (mip-research#19, "A100 result"). div_big used 21,142 tokens on average against 12,537.
    The best config adds the loop closers for this reason.
[^recheckwords]: Note: on the same 162 answers the re-check words rose 4,618 -> 5,266 (+14 %), and about 83 % of every right answer comes after
    the right value is first written, a share the put does not shrink.
[^notemodel]: Note: the effect of the note is model-specific. On Gyro-M in the engine the opposite was seen: an unmarked value was distrusted
    (`= 2939885984? Let me compute carefully.` looped to the limit) and a source note fixed 5-digit multiplication (Strata-xeno#276,
    "Calculation inside reasoning"); removing the note on 49 questions added 26.6 % tokens in total (sign-flip p 0.0385, one of four
    comparisons) with the median question unchanged, one run per arm, not paired (Strata-xeno#276, "No-note and effort-medium arms").
    That arm used the engine's early inline delivery (`A op B =` only, before the General Style) and is too weak to carry over: one
    run per arm, 26 questions up and 23 down, three questions hold 30,089 of the net 44,982 tokens. With the General Style the 4B
    doubts a put less than its own values (4.1 % against 15.2 %, [compute](compute.md#audit-of-every-value-put)), so the plain value
    is the rule for every model until a run with the General Style shows otherwise.
[^joinsheld]: `joins_held` in the records counts every time the guard was set, not a digit held; it is not a coverage number.
[^echo41]: Withdrawn: the earlier rule counted repeats over the whole answer and closed or nudged the thinking at the 3rd; on today's code it
    would have acted in 41 of 48 stored rows, so the "inlineclose 41/48" figure and the "0 regressions on 96 traces" claim describe code that
    no longer exists (mip-research#13, #15). Neither is a result.
[^profilepair]: Note: one round each. vLLM on this A100 does not repeat an answer across runs with another row set (mip-research#21): control
    came back token-identical in 2 of 104, so nothing here is paired, and control's own change (-2) is the noise floor. On all 240 answers of
    round 1 the Qwen profile scored 203 against control 209 and General 197.
[^oldprofiles]: The older profiles `qwen3.5-4b`, `deepseek-r1` and `qwq-32b` (residue by habit, bare quotient, rounded decimal) were guesses
    fitted to mixed sets; the residue guess held on the 4B (5 of 6) and failed on R1/QwQ (22 of 61), and two of their forms write false
    equations. They stay in `INLINE_STYLES`, unused.
[^cutdecimal]: Note: dropping the cut decimal was measured as a General change across families and rejected: most models write a decimal of
    their own there, and Qwen3.5-4B's own decimal was wrong at 29 of 449 points (mip-research#19, "The cut decimal on a/b in lowest terms").
[^arithtuned]: Note: the hand-off rules were written while reading failures of earlier arms on these same 160 rows, so they are tuned to them.
    Two greedy base runs on this PC with identical settings gave 114 and 118; base on an L4 gave 121. One base-right row (5409) is still
    wrong in pc9. A rule version in between (pc7, 143/160) was reverted.
[^arithscope]: Note: the question reader finds a stated computation in 0 of 568 LiveBench questions, so this mode applies only where the
    question is the calculation. `--inline-calc` alone at every `=` on the same kind of set was cut: 81 of 105 against base 85 at 0.71x
    tokens (L4), because values arrived in the middle of the model's own method (mip-research#12).
[^checkhand]: Note: false corrections tell the model to replace a right value with a wrong one. A replay found 6 of 6 corrections false
    (tails of longer expressions, parity, `\to`, degrees; mip-research#18); H3 found thousands with spaces, `≈` with e-notation, questions,
    spaced slashes, inverse mod and word operands (mip-research#20). These were fixed with a test each; the remaining 303 H3 corrections were
    still being read. On top of `--whole-plain` it fired once and changed nothing (138 against 138).
[^logit]: Note: one round; the same flags gave 25 and 21 right across two A100 sessions (ctrl, ctrlb), so a difference of a few answers is
    inside the noise. The soft arm took 1,932 s against 687 s for ctrl (about 2.8x), because the runner asks one token per request.
[^suppress]: Note: dropped from the best config. On the 24 loop rows x 2 it was worse alone and with loop-close raised cut answers from 3 to 9;
    tokens per right answer +43 % (mip-research#19, "Loop round on the A100"). The fall in the 300-character re-check rate is by construction
    (the window is about 240 characters). In the 42-point probe the hold-down lowered the re-check word by 9 and 11 points, inside the
    ±15-point noise band (Strata-xeno#298).
[^hintcalc]: Note: on 10 paired rows `--hint-calc` was right on 5 against the control's 3, but the arm ran without `--execute`, so the model's
    own python blocks never ran and their outputs were made up; the numbers are not counted (mip-research#12).
[^prompthints]: Note: `--think-hint`, `--shots` and the tool-rules files were built for queued arms (the mandatory-rules arms of
    mip-research#8, a few-shot arm in the PRD's step 6); no result was found in the tracker. The local queue stopped when testing moved to the
    A100 (Strata-xeno#283).
[^bridge]: Note: one call at answer start, operands up to two digits, not a paired ABBA, the off arm on engine `34885aa6`, and no timing
    counters in that arm. The bridge text is fixed, so it replaces the model's wording for these requests.
[^abbalatency]: Note: added latency. GSM8K took 0.5-0.6 s longer per case (+10 %) with equal tokens, in both pairs; the parser, delivery and
    repeat guard run per token there. Not explained yet. Accuracy did not move: differences are one or two answers and pairs overlap.
[^abbabuild]: Note: this build lacked two later source fixes (`100 - 30%`, EOS in thinking) and the binding-token parser fix, and had no
    natural-language firing: it is the parser path only. This PC does not repeat itself token for token under adaptive swaps, so arms are
    compared as totals.
[^gsmhard]: Note: one run per arm. 20 cases have the same wrong value in both arms where the gold itself is wrong (five computed by hand), so
    scores understate both arms. One case stuck in the on arm (`gsm8k-34`, the model multiplied with a name `C`, which the parser refuses).
    Off's own large products had no error on this sample, so MIP moved tokens and time, not accuracy.
[^wordproblems]: Note: one run per arm on a generated set. The sidecar was loaded and fired 0 times; sidecar results are not claimed. The
    model also writes ` (exact)` after its own wrong value, so the note is not proof that a value came from the device.
[^ladder]: Note: this run also showed a real parser fault on MATH-500 (55 against 59 off): a sign merged into the token before a number, so
    `a+b=-1+5=` got 6. It was fixed in `af41da64` (binding tokens). The sidecar was loaded and fired 0 times.
[^closerepeat]: Note: two different exes and 48 of 49 text pairs diverged before the close, so only the totals carry; the close also cut
    thinking mid-work (`gsm8k-0`; 61 of 294 runs ended on it, 6 wrong runs on answerable questions read, mip-research#5). The same-exe
    3-against-0 A/B (`tl-otra`, `tl-otrb`) did not finish: `otrb1` stopped at 9 of 26 when the queue moved to the A100.
[^repeatnote]: Note: the repeat-note wordings 1 and 2 were built to test `stop checking`, which can end the thinking with work left (10 of 294
    runs ended on the note, 5 with a step to go; mip-research#5). That test was queued, not run.
[^notegyro]: Note: a system line about marked values made the model copy `(already confirmed)` into a new loop (see
    [Prompt hints](#prompt-hints-and-few-shot)).
[^latent]: Note: these are functional checks, not results. Raw-token parity of the build against the pre-hook build failed (3 of 4 prompts in
    the injection smoke; the runtime run also failed on an unmatched Thai control); enabled logits are not bit-identical (1,234,923 entries
    differ, max 0.00784); the frozen one-integer rubric scored the enabled arm 1/2 and 29/30; operands came from oracle fixtures, not learned
    reading. The PRD records the latent output side as plateaued near 65 % whole sequences. No quality, token or speed claim is made.
