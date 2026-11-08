# AGENTS.md: rules for agents working on mip-research

MIP (Model Inline Processing): software computes, the model reasons. This repo holds the research side: the reference rules
(`tools/igc/`), the harness (`tools/igc/tir4b.py`), the replay gate, the audits (`scripts/`), the catalog (`docs/INDEX.md`) and the
records behind published numbers (`results/`). The engine side lives in [Strata-xeno](https://github.com/xenodeve/Strata-xeno)
(`include/strata/core/igc_*.hpp`, `src/program/generate.cpp`).

These rules were set by the developer while the work lived in Strata-xeno (2026-11-01 to 2026-11-08) and moved here with it. A
bare `#N` written before 2026-11-08 means Strata-xeno #N; read it through the map in #22.

## The aim

- **Software computes, the model reasons** (developer, 2026-11-06). Every calculation software can do in the model's place is
  covered, or software is extended until it can: decimals, pi, roots, mod residues, column digits, quotient and remainder. The
  model analyses the question and forms hypotheses.
- **Never lower the model's ability; add to it** (2026-11-02). A change that loses right answers fails, whatever it saves. More
  tokens are accepted where they buy right answers; a rule that saves tokens by cutting reasoning short is out.
- **Order of models** (2026-11-08): Qwen3.5-4B first, until the whole system is settled and no assumption changes, then other
  models. A small model has the most headroom, so it shows fastest whether the software works. On a large model expect tokens and
  time, not accuracy (2026-11-01).
- **What the developer most wants to see:** the output steering what the model does afterwards, like a guideline that is always
  there. A guideline is only as good as what was detected, so detection coverage comes first.

## The General Style first, no tuning to the test

The developer, 2026-11-06 (#19): develop the General Style first, model profiles later, and make it as good as possible
**without tuning the software toward the exam**.

- **A put must be true under every reading of the `=`** (`16807 ≡ 7`, `6 rem 1`, `144.6666...`). The form is chosen by the
  semantics of the notation, never by which form a model wrote most often in a set.
- **A new rule is justified by mathematics or notation,** never by "this set's model wrote X 70 % of the time".
- **Development sets find bugs and gaps; they do not set thresholds.** Measure on a frozen fresh set (H3), read only after the
  run; never change rules by fitting H3.
- **Every new put form is judged by agreement with the value the model writes next,** on traces it was not written from (in and
  out of sample). A wrong value in the middle of a derivation misleads more than no value: in-sample rules looked clean (11
  differ) until out of sample showed 207. Expanding products and substituting a value set on the same line were measured and
  are not built.
- **The plain value for every model** (2026-11-08). A source note such as ` (exact)` drew re-checks on the 4B; revisit only if a
  General Style run on another model shows distrust (doubt rate after a put, `scripts/put_reject.py`).
- **Profiles are a layer on the General base,** after the General Style is measured alone. A profile only picks which of the true
  forms to write first, or which refusal to lift; it never writes a false equation. Tune per family from that family's traces,
  split by question into tune and check sets; keep an override only if it gains on the check set and on another member of the
  family; record its support count.

## Simulate until the system works at its best before using the NVIDIA A100 (developer, 2026-11-07)

"Before using the NVIDIA A100, run the simulation until the system works as well as it can; only then use the A100, because the
A100 can only be used for a limited time." The A100 runs on Colab compute units, so an hour lost to a bug is an hour no test gets.

- **The simulation comes first, every time.** Replay the stored traces with `tools/igc/sim_replay.py` (`--recorded` for an arm's
  own traces) with the mechanism added. It shows where the mechanism acts, on how many cut answers out of how many, and whether it
  ever acts on a right answer.
- **The gate, and then go.** The A100 is used when all of these hold: zero misreads in the replay; no act on a right answer, or
  every such act read and understood; every test passing; and `vllm_refused` 0 in the replay's summary. Once they hold, go: the
  questions left are the A100's, and polishing past the gate spends time the simulation cannot turn into an answer. A result that
  still needs a fix is fixed and simulated again here.
- **Read every intervention that is not the model's own value** before asking for the A100: each must be a slip of the model with
  software's value right, never a misread, crash or hang. Report the count (misreads / interventions) when asking (2026-11-06:
  the replay found 6 false corrections of 6 that the unit tests missed, #18).
- **The simulation must also catch what the A100's engine refuses.** A replay does not build a vLLM request, so it cannot see
  vLLM's own limits: on 2026-11-07 a logit bias of 1,106 tokens (vLLM's cap is 1,024) killed all four arms of a round within
  minutes (#19). Every request the replay asks is checked against vLLM's known limits (`tir4b.vllm_refusal`, counted as
  `vllm_refused`), and a newly met limit is added there first.
- **A short smoke run before the full run.** Start the queue on 1-2 rows for about 5 minutes and read the log: the engine is
  alive, the mechanism fired, nothing was refused. Then run the rest.
- **The A100 answers only what a simulation cannot:** what the model writes after a mechanism acts. Give it the smallest set of
  rows that answers that, and stop it the moment the question has an answer.
- **Compute units.** Check the balance and tell the developer before opening a VM; stop the VM the moment the round ends. An idle
  VM burns units no test gets.

## How a test is run

- **Full context, every round** (2026-11-06). Never `--max-tokens`; 262,144 tokens for the 4B. A cap turns "would have answered at
  90k" and "loops forever" into the same `length` finish. Report finishes by kind (stop, loop close, stuck, length). Keep the model
  from filling the window with loop closes, not with a cap.
- **Sampled, not greedy** (2026-11-03): T 0.6 with the model card's top-p / top-k, two or more seeds per comparison, and compare an
  arm with a sampled control on the same rows and seeds. At T 0 the model never leaves its first premise.
- **Pairing.** tir4b on vLLM does not repeat an answer across runs (#21): cross-run comparisons are totals, and a control's own
  change is the noise floor. Strata's engine on the A100 repeats itself (Strata-xeno#283), so a rule A/B there is paired.
- **Test with the smallest run that answers the question** (2026-11-02):
  - do not pick a suite the model nearly aces;
  - run a first pass, then repeat only the misses the feature can reach (`tools/igc/select_rows.py`), with a fresh baseline on the
    same rows;
  - drop misses the feature cannot reach, and say how many and by what rule;
  - when the misses satisfy, check the trade-off on a sample of right answers: a gain on misses alone is not a claim;
  - a whole suite runs only when a whole-suite number is what is asked for.
- **Every test with a written answer also goes through the API reader** (`tools/igc/answer_check.py`). Report the rule's score, the
  reader's and the corrected one, with a per-row verdict: really right, really wrong, or scorer wrong. Read every row where they
  differ. A suite scored by its own tests (LiveCodeBench) needs no reader. **The reader runs on this PC only, never on a VM that
  runs model code, so the key stays here.**
- **A wrong scorer does not cost the run.** Records keep the raw reply: fix the scorer, score the same records again, say which
  numbers changed, and rerun only what the text cannot answer.
- **Model-written code** runs on the A100 VM by default (2026-11-08). On the developer's PC only inside the AppContainer box
  (`--box`), and only with the developer's approval for that kind of use; bare CPython on the PC never.

## How a result is read and reported

- **Caught / should have caught,** with the denominator, never "it fired N times" (2026-11-04, 2026-11-05). Say where the
  denominator comes from, name what nothing can count, and never call a small denominator a weak mechanism. `tools/igc/coverage.py`
  counts it from a run's `ids.jsonl` (`--ids`).
- **Check every case before stating a cause** (2026-11-01). Open every case in a claim and say what was read and what was not. A
  tool's flag list is a lead, not a classification.
- **The deep read, unasked** (2026-11-02): every result with its per-row split (median, the few rows that drag a total), which
  mechanism acted on each row and when; then the research and the next test. No total alone.
- **Before reading a per-question difference between two arms, check where their texts first part.** Paths that part before the
  rule could fire are path noise.
- **Idle means scrutinize the tests and the mechanism** (2026-11-02): a number is not evidence until the code that produced it has
  been read. Ask what else could have produced it, prove the path ran from a log, try to break it against a reference, and land a
  fix with the check that would have caught it.

## How the work is presented

The README and the catalog are read by people outside the work (developer, 2026-11-08):

- **The latest state only.** No development history ("first we tried X, then Y"), no first-version numbers.
- **Negative or unconfirmed data is never presented as a result or as a "limitation"; it goes in a footnote note (หมายเหตุ).**
- **Every number is read from its source** (issue, report, records) with what it was measured on: model, set, n, machine,
  sampling. Records behind a published number go in `results/`, so it can be checked.
- **Withdrawn results never appear as results** (43/48, 0/96 regressions, inlineclose 41/48, sidecar results, latent bundle
  results); at most a footnote saying they were withdrawn.

## Working rules

- **Time spent is the cost** (2026-11-02): no machine sits idle unnoticed; check a job a minute after it starts; gate a scorer on
  real replies first; stop a run the moment its question has an answer; measure a machine's speed on one question before giving it
  a queue; script what repeats; write code with backslashes through the file tool, not a shell heredoc.
- **Run tracks in parallel and never stop to wait for one.** Build to the end of the plan alongside the tests.
- **Run everything in the background,** with no window on the developer's display.
- **One engine at a time on the developer's PC.** Before starting any engine (strata.exe, llama-server, vLLM, a training job), take
  the shared `engine.lock`, check free RAM, and check that no other engine is alive. Torch / HF / vLLM runs there set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (the other card is the display card).
- **The tracker holds the knowledge.** Post to the issue at major progress (a round's results, a decided mechanism, a found bug),
  bundled into one comment, and open an issue the moment a problem is found. Issue bodies and comments are bilingual (English plus a
  full Thai mirror). `gh` always with `--repo xenodeve/mip-research` (or `--repo xenodeve/Strata-xeno` for engine issues).
- **Code from other projects:** permissive licences (MIT, BSD, Apache-2.0, ISC, BSL-1.0, zlib, PSF) may be copied with their notice
  kept and recorded in `third_party/`; copyleft is never copied; linking a new dependency needs a conversation first.
- **The engine reimplements, this repo is the reference** (2026-11-08): rules, forms, refusals and thresholds live here as code,
  data and golden tests; Strata's C++ must pass the same tests.
- **`AGENTS.md` and `CLAUDE.md` mirror each other:** a rule added to one goes into the other in the same commit.

## Tests

```sh
python -m unittest discover -s tools -p "test_*.py" -t .     # some skip without a GPU, a model or STRATA_ROOT
```
