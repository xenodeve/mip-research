# Detection

Detection is everything that decides **where a calculation is, or is about to be**: the stop at `=` and the reader that takes the
expression before it, the reader of the question itself, the engine's own token-level parser and the way it intercepts the decode
loop (including the MTP draft), the natural-language levels a parser cannot see, and the signals that see a calculation coming
before it is written (residual-stream probes, the MoE router, a readiness classifier). What is done with a detected value (the form
it is spelled in, the notes, the closers) is in the inject and antiloop pages.

**The detectors are meant to be used together; each covers another's blind spot** (developer, 2026-11-10). The text readers see
only what is written, and only once it is written: the `=` stop misses a value written with no `=`, the question reader sees only
what the question states, the loop checks see the shape of a repeat but not what is repeated. The drafter's draft tokens show the
next few tokens before they are verified; the MoE router and the residual probes show a calculation coming before its first token.
Full effect comes from combining them (the latent signals arm, an exact parse acts: [`UnifiedDetector`](#unifieddetector-fusion)),
not from picking one as best. **Not yet used live:** the drafter's draft as a reading (the engine only drops drafts after a put),
and the MoE router (offline: 83 % at k = 2 on Qwen3.6). Each detector is judged inside its own condition, and every one has a
status in the arm being run: [detection coverage plan](plans/2026-11-10-detection-coverage-plan.md). The router is the next step, with the move to **Qwen3.6-35B-A3B**; the
4B is dense and has none.

| Mechanism | What it does (one line) | Runs on | In best config? | Best measured result | Status |
|---|---|---|---|---|---|
| [`=` stop and expression reader](#the--stop-and-the-expression-reader) | Every request stops at a token that is `=`; `mathexpr` takes the longest exact expression that ends the line | tir4b (vLLM/llama-server) | yes | H3 fresh set: 94.4 % of plain-arithmetic `=` read (30,194 / 31,996) | in best config |
| [MathBackend-aware reader](#mathbackend-aware-reader) | Also reads general function calls (`\gcd`, `\binom`, `comb(`, `\det` ...) before `=` | tir4b (vLLM/llama-server) | yes (`--mathbackend`) | not separately measured | in best config |
| [Question reader](#question-reader-catch-point-1) | Reads the computation a question states, before the model starts | tir4b (llama-server) | no (arithmetic mode, `--question-value`) | 160 / 160 DeepMind questions read to gold's exact value; widened to any request wording and LaTeX (2026-11-11, [D19](failures.md)): MATH-500 6 reads, 6 gold, GSM8K 0 of 1,319 | built, measured, kept as option |
| [Engine parser `parse_inline`](#engine-parser-parse_inline) | Finds `A op B =` on token ids inside Strata's decode loop | Strata engine (xeno/mip-sandbox, not in main) | no | 48 / 48 long-number word problems, 249 against 457 tokens per answer | engine only |
| [Engine interception and MTP draft drop](#engine-interception-before-the-kv-cache-and-the-mtp-draft) | The token after `=` is replaced before it is fed back; drafts after it are dropped; the value goes in as one window | Strata engine (xeno/mip-sandbox, not in main) | no | 5-digit multiplication: 137, 142 against 491, 522 tokens per answer (ABBA) | engine only |
| [Natural-language levels 0 / 1 / 2](#natural-language-levels-0--1--2) | Labels each computed number by where its operands are | offline tool | no | GSM8K: level 0 29 %, 1 30 %, 2 41 % of 293 labelled numbers | built, measured, kept as option |
| [Sidecar (natural-language firing)](#sidecar-natural-language-firing) | A small model proposes operation + operand pointers at a number the parser cannot read | Strata engine (xeno/mip-sandbox, not in main) + offline training | no | none (see note) | engine only |
| [Predictive probes and the MoE router](#predictive-probes-and-the-moe-router) | Hidden-state signals that a calculation or a re-check starts within k tokens | offline tool (A100, HF weights) | no | k = 2: residual probe 89 % (4B), 87 % (Qwen3.6), router 83 %, text 53 % of 3,176 onsets | built, measured, kept as option |
| [`UnifiedDetector` fusion](#unifieddetector-fusion) | Latent signals may only arm; any act needs the exact parse | offline tool | no | — | built, not measured yet |
| [Readiness classifier](#readiness-classifier) | A head on the final 2,560-wide activation: is this a calculation-ready point? | Strata engine (xeno/mip-sandbox, not in main), diagnostic | no | 111 / 112 development prompts (Gyro-M, 320 real activations) | engine only |
| [Hand-value check reader](#hand-value-check-reader---check-hand) | Reads every relation on a finished line and flags a value that is wrong | tir4b (vLLM/llama-server) | no | no coverage number yet (see note) | built, measured, kept as option |
| [Catch-point map](#catch-point-map) | The 12 places in the pipeline where a calculation can be noticed, ranked | plan | — | — | spec only |
| [Coverage tool](#coverage-tool) | caught / should-have-caught per mechanism from a run's ids | offline tool | — | — | built, measured, kept as option |

## The `=` stop and the expression reader

- **Trigger.** `tir4b` builds `ids.equals`, every token whose text stripped is `=`, from the tokenizer itself
  ([`../tools/igc/tir4b.py`](../tools/igc/tir4b.py):128), so it runs unchanged on another model's tokenizer. With `--inline-calc`
  (or `--logit-value`) every request in thinking and in the answer stops on those tokens (`tir4b.py`:935); on llama-server they
  become the stop word `=` (`stop_words`, `tir4b.py`:1671).
- **What is read.** The text before the `=`, last line only. `mathexpr.expression_before`
  ([`../tools/igc/mathexpr.py`](../tools/igc/mathexpr.py):355) tries each start from the left; the first that reads whole and holds an
  operator wins. It reads ASCII, LaTeX at any depth and Unicode, keeps `12,510,204 \times 100` whole, and refuses a symbol, a word, a
  mixed number (`2\frac{1}{3}`), a root or power that is not exact and a value too large to be a step.
- **Rules that skip a start** (each from a case in the traces, `mathexpr.py`:363-370): a minus after a symbol is binary, `- ` opening a
  line is a bullet; a factor beside a number (`3^3 5^1`) or a function's argument (`\sin 84^\circ`) has no value of its own; the tail of
  a longer expression (`Z / (7/3)`, `\sum (1/4 \times 1/4)`) is not a value.
- **What is done.** The expression and its exact value go to the put (inject page). If nothing reads, nothing is put and the model
  writes its own value.
- **Measured** on H3, 1,526 traces of models and sources no rule was written from (Qwen3-235B-A22B OpenThoughts4 math 303, Gemini
  thinking s1K-1.1 about 1,000, DeepSeek-R1 s1K-1.1 non-math 123, DeepSeek-R1 OpenThoughts science 100), code `1d573ae3`, measured once
  before anyone read H3: plain-arithmetic `=` caught / should catch **94.4 % (30,194 / 31,996)**; the two development sets gave 96.4 %
  and 96.0 % (mip-research#19). What H3 still does not read, 1,052 of the 31,996: unbalanced tails (mostly named points and matrix rows)
  342, roots 95, `...` decimals 89, huge powers refused by design 87, fractions 70, spaced factors 52, fractional powers 52, pi in
  degrees 52 (mip-research#19).
- On the round-1 A100 arms (**4B bf16 · vLLM · A100**, 120 fresh questions), software computed **19,546 of 19,636** `=` (mip-research#19).
- Where it lives: [`../tools/igc/mathexpr.py`](../tools/igc/mathexpr.py), `inline_found` / `_inline_general` in
  [`../tools/igc/tir4b.py`](../tools/igc/tir4b.py):302-347.

## MathBackend-aware reader

- With `--mathbackend`, `_inline_general` reads through `unified_detect.expression_before_with_mathbackend`
  ([`../tools/igc/unified_detect.py`](../tools/igc/unified_detect.py):495), which takes `mathexpr`'s expression and, ahead of it, a
  longer one that holds a call (`\sqrt{2} \cdot 4 + 1` is not 5).
- **Calls it reads** (`_FUNC_CALL`, `unified_detect.py`:65): `\gcd`, `\lcm`, `\varphi`, `\det`, `\operatorname{...}`, and plain `gcd`,
  `lcm`, `totient`, `modinv`, `catalan`, `derangements`, `bell`, `partition`, `stirling1/2`, `lucas`, `fibonacci`, `comb`, `perm`;
  `\binom{n}{k}`; LaTeX matrices for a determinant only when `\det`, `determinant` or vertical bars say so.
- **The resulting rules** [^p23]:
  - only general primitives reach the inline path; ops named after one problem stay in the backend, out of the inline path;
  - bare `C(` and `P(` are not read: points are named P and C (`unified_detect.py`:83, :173), and `P(n, k)` with k > n is a point (:198);
  - Greek letters other than `\varphi` are not read (`\sigma(1)`, `\tau(2)`, `\mu` name other things, `unified_detect.py`:68-69);
  - arguments past a size bound are refused before computing (`unified_detect.py`:70-71);
  - the tail of a longer expression has no value of its own (`C(9, 3) + gcd(15, 25)` is not 5).

## Question reader (catch point 1)

- `question_read.read(question)` ([`../tools/igc/question_read.py`](../tools/igc/question_read.py)) returns the expression in plain
  notation and its exact value, or None. It reads an expression (`What is 20*(-8)/(-60)*329?`) or its English (`Subtract 3 from 7`,
  `What is 3 less than 7?`, `the product of 2 and 5`). None for a word problem, two candidate expressions, or symbols.
- Used by `--question-value` (`tir4b.py`:1247): when the model's copy of the question's computation gives another value, the question's
  value is put. Also used by `--loop-value` before a loop close (`tir4b.py`:1200).
- **Measured** offline on the 160 DeepMind rows: **160 / 160 equal gold's exact value, 0 wrong**, including the 4 rows the model copied
  wrong (Strata-xeno#298). In run pc9 it fired on 4 of 4 copy errors (Strata-xeno#298). [^qread]
- To use it: pass `--question-value` (arithmetic mode in the README); it acts only where the question states one computation.
- `question_read.read_exact(question)` (2026-11-10, for `--exact-answer`): as `read`, and also a question whose one computation has
  square roots (`Simplify (6*(-3*(3 + (sqrt(108) + ...)*-5 + 4) + 5))**2.`), computed exactly by `exactexpr` as `a + b\sqrt{n}`. Any
  letter beyond `sqrt` and the lead word, and it reads nothing. Round 9 read none of the 10 `simplify_surd` questions, and 3 General
  answers there were lost to loops on the parentheses ([plan](plans/2026-11-10-round9-fixes.md)). **Checked** offline against gold
  (`results/exact-answer-r9/reader_check.py`, frozen9, fresh, eval, lb1, pc9-dev rows): 84 of 379 rows read (every `div_big`,
  `mixed_longer`, `mul_div_longer`, `simplify_surd` row; no word problem, contest or MATH row), **84 of 84 equal gold, 0 wrong**.

## Engine parser `parse_inline`

The engine's reader works on token ids, not text, inside the decode loop.

- **Trigger.** The last written token closes an expression: `=`, ` =`, or one more `=` token (`closes_expression`,
  Strata-xeno xeno/mip-sandbox: include/strata/core/igc_inline.hpp:102).
- **What is read.** `parse_inline` (igc_inline.hpp:362) walks back over at most 128 tokens: digits, operators, spaces, `(`/`)` by depth,
  LaTeX `\times` / `\cdot` / `\div` as two tokens, `%`, `$`. The body is parsed by precedence climbing (`InlineParser`) into a postfix
  program; numbers are read right to left (`read_number`, igc_inline.hpp:234). Values come only from the device executors (integer,
  rational, integer-extended power).
- **Refusal rules** (each a misread found in a run) [^half] [^sentence]:
  - a `.` then a space ends a sentence, not a decimal: the walk stops there (igc_inline.hpp:374);
  - a word that scales the next number refuses the expression: half, third(s), quarter(s), fifth to tenth, twice, double, triple, or a
    number before ` of` (`1/4 of 20 + 5 =`) (`scales_next`, igc_inline.hpp:217). `sum of 68 + 10 =`, `total 68 + 10 =`, `25% of 40 + 3 =` still run;
  - a comma that does not split thousands is a list (`[6,9+4=` is not 69) (igc_inline.hpp:234-249);
  - a sign before the first number belongs to words the parser cannot see (igc_inline.hpp:391);
  - binding tokens: a start glued to a token that ties it to something before (`d12^2=`, `sin60^3=`, `a1 +1 =` are names) is refused
    (`binds_next`, `number_may_follow`, igc_inline.hpp:400-401; the table `igc_binding_tokens.hpp` is written by
    [`../tools/igc/binding_tokens.py`](../tools/igc/binding_tokens.py));
  - `**` after a value is a power, at a line start it is bold;
  - LaTeX with more than one product term shows its steps (`2 \times 6 + 4 \times 1.5 = 12 + 6`) and is refused (igc_inline.hpp:412);
  - 1 to 15 operations only (igc_inline.hpp:415);
  - percent only as a factor: `100 - 30%` is 70 to a reader and 99.7 to arithmetic (igc_inline.hpp:416).
- **Flags.** `--igc-result-bridge`; `--igc-inline-exec` (turns both on) (Strata-xeno xeno/mip-sandbox: src/program/generate.cpp:2818-2819);
  `--igc-no-reply` keeps values out of the visible answer (built, not run [^half]). The engine is built with
  `STRATA_IGC_NATIVE_DIAGNOSTIC`.
- **Measured** on 48 word problems with long numbers ([`../tools/igc/natural_problems.py`](../tools/igc/natural_problems.py), not a
  benchmark), engine build `a4b6e4521ee5c0ec`: MIP off 48 / 48 at 457 tokens and 6.58 s per answer; MIP on (parser ownership fix) 48 / 48 at 249 tokens
  and 4.21 s (Strata-xeno#276, comment `igc:engine-sidecar-20261031`). The full set of engine results is on the inject and bench pages.
- Where it lives: Strata-xeno xeno/mip-sandbox only. To use it: build that branch with the IGC diagnostic and pass `--igc-inline-exec`.

## Engine interception before the KV cache, and the MTP draft

The parse runs after every accepted token; the delivery is arranged so that neither the model's own token after `=` nor a draft
written after it reaches the cache.

- **One request (solo path).** After the verify window, each accepted token goes through `inline_token`
  (Strata-xeno xeno/mip-sandbox: include/strata/program/igc_learned_diagnostic.hpp:378; the parse is at :563). When it fires,
  the accepted prefix is cut at that token: `if (called) { a = i; break; }   // the rest of the window follows a value it never saw`
  (src/program/generate.cpp:14704). While a value is being delivered:
  - the window is the value itself: `T = min(bridge_left, S_mtp)` and `window[i] = bridge_ahead(i - 1)` (generate.cpp:14488, :14528),
    so a value goes in as one forced window rather than one token a round;
  - the suffix draft and the lookup chain are not proposed (`!igc_delivering`, generate.cpp:14495, :14505), the MTP speculative
    distribution is not set (:14566), and draft statistics skip these windows (:14710);
  - each sampled output is replaced by the next value token (`bridge_sample`, generate.cpp:14688-14691; igc_learned_diagnostic.hpp:656).
- **Batched slots (`--batch`).** `slot_token` (igc_learned_diagnostic.hpp:341) swaps in the slot's own reasoning state, replaces the
  sampled token `outb[i]` with the delivery and reports `forced` while a delivery runs. It is called before
  `commit_slot_prefixes` (generate.cpp:11961), and a forced token drops the MTP proposal that follows it:
  `if (igc_forced && batch_mtp) tok[i + 1] = -1;` (generate.cpp:11965), so `keep` is 1 for that slot.
- Where it lives: generate.cpp and igc_learned_diagnostic.hpp on xeno/mip-sandbox; it comes with `--igc-inline-exec`.
- **Measured.** ABBA on1 / off1 / off2 / on2, same session, Gyro-M D2x, greedy, `--igc-result-bridge --igc-inline-exec` against the
  same build with both off (Strata-xeno#276, comment `igc:abba-20261031`):

  | Suite | right, on | right, off | tokens per answer, on | tokens per answer, off | seconds, on | seconds, off |
  |---|---|---|---|---|---|---|
  | 5-digit multiplication (40) | 39, 39 | 40, 39 | 137, 142 | 491, 522 | 2.80, 2.82 | 6.25, 6.61 |
  | 3-digit multiplication (40) | 40, 40 | 40, 40 | 74, 72 | 118, 120 | 1.75, 1.68 | 1.87, 1.90 |
  | MATH-500 (60) | 58, 58 | 58, 57 | 854, 874 | 1035, 1085 | 15.7, 15.3 | 16.6, 17.5 |

  This table is shared with the inject and bench pages. [^abba] [^window]

## Natural-language levels 0 / 1 / 2

- **Levels** (spec Strata-xeno xeno/mip-sandbox: docs/plans/2026-10-31-mip-sidecar-spec.md): level 0, an expression before `=` that a
  parser reads (`68824 * 42716 =`); level 1, operands in the same sentence (`half of 180 is`, `5 pies * 8 pieces each`); level 2,
  operands earlier in the context or an earlier result (`the remaining apples cost`); level 3, a result with no working (out of scope).
- **Labelling.** [`../tools/igc/auto_label.py`](../tools/igc/auto_label.py) labels by value, not by wording: a new number is `computed`
  when exactly one short program over earlier numbers gives it; the level is where those numbers sit.
- **Measured** on MIP-off traces of Gyro-M, after filtering (spec, Strata-xeno xeno/mip-sandbox:
  docs/plans/2026-10-31-mip-sidecar-spec.md) [^levels]:

  | Off traces | Labelled numbers | Level 0 | Level 1 | Level 2 |
  |---|---:|---:|---:|---:|
  | GSM8K | 293 | 29 % | 30 % | 41 % |
  | MATH-500 | 184 | 28 % | 30 % | 42 % |

  The `=` parsers above see level 0 only.
- To use it: run `auto_label.py` on stored traces; it feeds the sidecar data and level counts, it does not act in a run.

## Sidecar (natural-language firing)

- **Mechanism.** At a *slot*, a sampled token that starts a number, a small model (`--igc-sidecar <file>`) proposes an operation and
  pointers into a quantity table of numbers written in the prompt and the text. It never predicts a value. In the engine
  (`NaturalFirer`, Strata-xeno xeno/mip-sandbox: include/strata/core/igc_natural.hpp, called from `inline_token`,
  igc_learned_diagnostic.hpp:388-395) the two leading digits stay the model's and must agree with the computed value; a later digit that
  differs is replaced and the rest delivered. A number closing an expression the parser reads is the parser's.
- **Planned fusion and veto** (spec only): parser, sidecar and an activation probe each propose; any disagreement does not fire; the
  model's per-digit probabilities can veto.
- Files: [`../tools/igc/sidecar_data.py`](../tools/igc/sidecar_data.py), `sidecar_traces.py`, `sidecar_fit.py`, `sidecar_eval.py`,
  `sidecar_export.py`, `rule_pointer.py` (the floor), and the engine headers above.
- Status: built, opt-in, off. To use it: train a sidecar, export it, pass `--igc-sidecar`. [^sidecar]

## Predictive probes and the MoE router

- **What is read.** For every token position of stored traces: the residual stream after layer L (a linear probe), the MoE router's
  expert picks (Qwen3.6 only), the next-token mass on digits and operators (a stand-in for MTP drafts), and a logistic regression on the
  last 12 tokens of text. Events are found in the text alone with `mathexpr`: a computation starts (the first token of an expression
  before `=`), a re-check starts (`wait`, `hmm` ...), a loop starts.
- **Script.** [`../tools/igc/probe/detect_cov2.py`](../tools/igc/probe/detect_cov2.py); modes: 3-fold by trace, `--train-traces` to fit
  on control traces and read an arm, `--check` (the state at t predicts token t+1: 93.5 % on the 4B, 91.1 % on Qwen3.6), and
  `--ood-split` (`livebench-` against `contest-` suites). See [`../tools/igc/probe/README.md`](../tools/igc/probe/README.md).
- **Measured** on an A100: 46 control traces written by **4B bf16 · vLLM · A100**, read with **4B bf16 · HF transformers** (no MIP in them, up to 30,000 tokens each, 1.16 M tokens), read by
  the 4B and by Qwen3.6-35B-A3B; 3 folds by trace; threshold set so 1 % of positions alarm; caught = an alarm in the k tokens before the
  onset (Strata-xeno#298):

  | Signal | Computation starts (3,176), k = 2 | k = 8 | k = 32 |
  |---|---:|---:|---:|
  | residual probe, Qwen3.5-4B (L24, L24, L31) | **2,813 (89 %)** | 2,293 (72 %) | 2,064 (65 %) |
  | residual probe, Qwen3.6-35B-A3B (L30, L38, L38) | **2,769 (87 %)** | 2,265 (71 %) | 2,089 (66 %) |
  | MoE router, Qwen3.6 | **2,639 (83 %)** | 2,184 (69 %) | 2,225 (70 %) |
  | text only | 1,690 (53 %) | 1,357 (43 %) | 1,350 (43 %) |
  | random at this alarm rate | 64 | 245 | 873 |

  Re-check starts (4,419 onsets), k = 2: next-token mass of re-check starters 3,443 (78 %) on the 4B and 3,102 (70 %) on Qwen3.6;
  best probe 3,256 (74 %) / 3,026 (68 %); router 2,443 (55 %); text 695 (16 %).
- What it is for: a 2-token lead cannot hold the value (software computes once the expression is written), but it can prepare.
- To use it: the probes need the hidden state; the Strata engine has it, llama-server and vLLM do not expose it. Nothing acts on these
  signals yet. [^probes] [^weak]

## `UnifiedDetector` fusion

- `UnifiedDetector` ([`../tools/igc/unified_detect.py`](../tools/igc/unified_detect.py):832) fuses `probe_L24`, `probe_L31`, `router`,
  `next` and `text_lr` with fixed weights into three gates (computation, re-check, wrong value). Thresholds are fitted on the training
  split only. Latent stages may only set `armed`; any put or replacement needs the exact parse (`can_intervene_exact`) of the text
  before `=`.
- Design: Strata-xeno xeno/mip-sandbox: docs/plans/2026-11-05-mip-unified-detection-pipeline.md (five stages: question and end of
  prompt, latent lookahead, onset, exact verifier, audit). Out-of-distribution metrics: `detect_cov2.py --ood-split`,
  [`../tools/igc/probe/ood_metrics.py`](../tools/igc/probe/ood_metrics.py).
- Status: unit-tested (tests on Strata-xeno xeno/mip-sandbox), not wired into `tir4b`, no fused coverage number. To use it: feed it the
  probe features of a run and wire its `armed` state into `tir4b` or the engine. [^ufa]

## Readiness classifier

- **Mechanism.** A readout on the final 2,560-wide activation at the answer start of Gyro-M decides whether the request is at a
  calculation it can execute. Head: 2560 → 128 GELU → 2 (328,066 parameters); a 2560 → 2 linear control. Native port: `ReadinessMapper`
  (Strata-xeno xeno/mip-sandbox: include/strata/core/igc_readiness.hpp, src/kernels/igc_readiness.cu, tests/igc/readiness_probe.cpp),
  artifact kind3. Fit: [`../tools/igc/readiness_corpus.py`](../tools/igc/readiness_corpus.py), `readiness_fit.py`, `readiness_refit.py`,
  `readiness_mapping.py`.
- **Measured** on 320 real activations of Gyro-M (144 matched calculation / copy-or-quotation pairs and 32 negatives; 208 fit, 112
  template- and pair-disjoint development), CPU fit, fixed rubric (Strata-xeno xeno/mip-sandbox:
  docs/reports/2026-10-29-igc-readiness-pairs.md): MLP 208 / 208 fit, development positives 47 / 48, false calls 0 / 64; linear 48 / 48
  and 1 / 64. Both 111 / 112. The CUDA port matches all 320 saved decisions (docs/reports/2026-10-29-igc-native-readiness.md).
- Status: diagnostic only; the report says it cannot be enabled in production by passing numerical tests. It belongs to the learned
  (latent) path; see that page. [^ready]

## Hand-value check reader (`--check-hand`)

- **Mechanism.** With `--check-hand`, every line in thinking and answer is a stop (`tir4b.py`:938); the finished line's relations are
  read by `mathexpr.wrong_relations` and each value that is not the exact value is flagged (`tir4b.py`:1339-1348). What is done with a
  flag (a correction line) is on the inject page.
- **Rules that came from misreads on H3** (mip-research#20) — a line is not flagged when:
  - thousands are grouped with a space or a LaTeX thin space (`90,090, 989`, `90,\!090,\!991`);
  - `≈` is followed by e-notation (`3^9≈2e4`) or `~` follows `=` (the written precision is the slack);
  - `6^{-1} = 11 mod13` is an inverse mod m;
  - a spaced `/` groups a product (`9*7 / 14*6 = 63 / 84`), or a tail sits inside `\frac`;
  - long-division notes put the divisor first; algebra goes on after the value (`9² =9² +8² -2*9*8 cosx`); an operand is words
    (`cube root of 1/2`); a factor is split by place value (`1152 * 34650 = 1152 * 30000 = …`);
  - a chain the model states to reject (`, which is false`, `impossible`), a side that goes on with a symbol, a factor after a space,
    and superscripts lost in a copy (`2.8×10−3`, the line is left unread).
  - A value followed by `?` is still checked: sparing it lost real slips (`= 3475360?`).
- Status: not in the best config. To use it: pass `--check-hand`. [^hand]

## Catch-point map

The plan (Strata-xeno xeno/mip-sandbox: docs/plans/2026-11-04-mip-catch-points.md) lists every point where MIP can notice a
calculation, each judged by lead time, caught / should-have-caught and false alarms, and, if it acts, by right answers lost.

| # | Point | Current state |
|---|---|---|
| 1 | The question itself | [Question reader](#question-reader-catch-point-1) |
| 2 | The hand-off (did the model use the value) | inject page (`--hand-off`) |
| 3 | The confirmed window (fix before the text goes out) | engine verify window; inject page |
| 4 | The shape of the thinking (loops, repeated `Wait`) | antiloop page |
| 5 | Next-token confidence on digits | measured [^weak] |
| 6 | The MoE router | [measured](#predictive-probes-and-the-moe-router), 83 % at k = 2 |
| 7 | The raw MTP draft trend | stand-in measured (next-token mass) [^weak]; the real MTP head is not measured |
| 8 | Hidden state during decode | [measured](#predictive-probes-and-the-moe-router), 89 % / 87 % at k = 2 |
| 9 | Hidden state at the end of the prompt | measured [^weak] |
| 10 | The final answer, substituted back | not built |
| 11 | The text layers (operators, `=`, fences, hand values) | [`=` reader](#the--stop-and-the-expression-reader), [check-hand](#hand-value-check-reader---check-hand) |
| 12 | Attention (copying a number against making one) | not built |

## Coverage tool

[`../tools/igc/coverage.py`](../tools/igc/coverage.py) `OUT [OUT ...] --model M` prints, per arm, what each mechanism caught out of what
it should have caught, from a run's `ids.jsonl`: computable `=` (put by software, taken back for a program, or left to the hand), hand
values the parser calls wrong (corrected or not), loops (and how many got a program), and programs by who opened them. A count of
firings is never reported alone; the denominator is always shown.

[^p23]: หมายเหตุ: these rules come from mip-research#14 (found 2026-11-05): `P(2, 3) =` and `C(4, 6) =` (points) got `0` put in, and a
    regex for rectangles in a regular n-gon put the gold answer 315 of `livebench-c3cc0058` after the question's own wording. Fixed in
    `4dc5e546`; each case is a test in Strata-xeno xeno/mip-sandbox: tools/test_unified_detect.py. Ops written from single target questions (`disphenoid_inradius`,
    `quartic_double_tangent`, `triangle_wave_intersections`, `polar_triangle_chain`) were taken out of the inline path.
[^qread]: หมายเหตุ: on LiveBench the reader finds a stated computation in 0 of 568 questions (math AMPS_Hard 0/150, math_comp 0/146,
    olympiad 0/72; reasoning zebra_puzzle 0/100, spatial 0/50 ...) (Strata-xeno#298). It applies where the question is a calculation.
    The DeepMind numbers are from **4B bf16 GGUF · llama.cpp · 5060 Ti**, not the Strata engine.
[^half]: หมายเหตุ: `scales_next` comes from mip-research#4: `half of 68 + 10 =` was read as `68 + 10` and 78 delivered in a reply
    (gold 54); with MIP on, 2 of 4 runs did not recover. After the fix (exe `50118d973e127e78`) six rounds of that question were all right
    and none wrote 78. `--igc-no-reply` is built and has not been run.
[^sentence]: หมายเหตุ: the sentence-end rule comes from `909. 59,237*900,000 =`, where the walk crossed the sentence end, the parse
    was refused and the model's wrong `53,333,300,000` stood (Strata-xeno#276). The README's Flash-Next numbers were measured on a build
    before the binding-token and letter-adjacency fixes.
[^abba]: หมายเหตุ: greedy runs on this PC do not repeat token for token, so arms are compared as totals. Accuracy did not move: on GSM8K
    (100) on 94, 93 against off 95, 94, at equal tokens, and GSM8K took 0.5-0.6 s longer per case (+10 %) with MIP on, not explained in
    that comment. Natural-language firing was not in this build, and the `100 - 30%` and EOS-in-thinking fixes were not either
    (Strata-xeno#276).
[^window]: หมายเหตุ: the one-window delivery (commit `3f2fd480`) was checked on 145 greedy cases (two-digit add, subtract, multiply
    40 / 40 each; GSM8K 24 / 25) with no paired run; the ceiling of the change was computed from timeline spans (15.24 against 16.10 ms
    per token), not measured (Strata-xeno#276). An earlier claim that one-token windows "did not change the total" was withdrawn because
    the edit had not been applied.
[^levels]: หมายเหตุ: approximate; ambiguous numbers are dropped. Reading 30 random labels showed 3 to 5 chance matches before angles,
    coordinates and approximations were filtered, so level 2 may be high. The earlier read before that filter (Strata-xeno#276, comment
    `igc:auto-label-rule-baseline`) differs by one to six points per level.
[^sidecar]: หมายเหตุ: sidecar results are withdrawn and none is presented here. Offline precision did not predict live firing: the live
    on15 arm (GSM8K 300 + MATH-500 150) fired three times at confidence 0.73 to 0.79, one followed by a wrong answer, while a replay of
    another arm's text found no firing at the engine's threshold; it stays out of the comparison arms (Strata-xeno#276). The rule-pointer
    floor (`rule_pointer.py`, corrected scoring) is a negative result: GSM8K 21 / 293 precision with recall 7 / 197, MATH-500 41 / 530
    with recall 2 / 105.
[^probes]: หมายเหตุ: alarm precision (alarms followed by an onset within k) for probe layer 24 on computations was 65 % / 83 % / 94 % at
    k = 2 / 8 / 32, measured on the A100 `inline` arm (48 LiveBench/AIME rows, Qwen3.5-4B thinking, detectors fitted on the 48 `ctrlb`
    traces). On that arm, with software's spans dropped and traces 2.4 times longer, computation onsets seen by the probe were 80 % /
    59 % / 50 % (Strata-xeno#298). The traces are the 4B's text read by both models; arithmetic `=` only, no code.
[^weak]: หมายเหตุ: what detection sees badly (Strata-xeno#298, 46-trace study): a wrong value, AUROC 0.705 (4B) and 0.668 (Qwen3.6) for
    the digit's confidence, 0.735 / 0.798 for a probe before the first digit (2,965 values, 115 wrong), and 0.666 on the `inline` arm; a
    loop, 24 onsets only (router 11 of 24 at k = 2); the end of the prompt, 0.52-0.69 against "this trace ends wrong" on 36 traces;
    the next-token mass of digits and operators (the MTP stand-in) sees 165 / 171 of 3,176 computation onsets at k = 2. A last-layer
    probe of "working by hand" on 277 windows of 19 LiveCodeBench questions ([`../tools/igc/hidpos_probe.py`](../tools/igc/hidpos_probe.py),
    `STRATA_HIDPOS`) gave AUROC 0.58-0.635 against 0.72 for text features (mip-research#8).
[^ufa]: หมายเหตุ: a claim of "0 destructive false alarms" for this gate (Strata-xeno#298) was contradicted by mip-research#14 (see the
    note on the MathBackend-aware reader); the exact parse was exact about numbers, not about what the text meant. The 43/48 and 0/96 figures reported with this
    pipeline are retracted.
[^ready]: หมายเหตุ: neither head met the predeclared all-development gate: the MLP misses `(19) - (19)` with a new template, the linear
    head admits the unsupported `Find the quotient 29 / 7.` More fit data (624 rows) made it worse, 110 / 112. Results of the latent
    bundles this classifier feeds are retracted.
[^hand]: หมายเหตุ: no caught / should-catch number holds yet. In a replay of stored control answers, 6 of 6 corrections were false
    (mip-research#18). On H3, flagged lines fell from 432 to 303 after the two rounds of fixes (mip-research#20); those are counts of
    flags, not a measure of accuracy, and the rest are still being read.
