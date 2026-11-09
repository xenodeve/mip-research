# CLAUDE.md

The rules for every coding agent on this repository live in `AGENTS.md`, so Claude Code and Codex follow one file. A rule added to one
file goes into the other in the same commit.

The rules agents miss most, all in full in `AGENTS.md`:

1. **Simulation → PC layer → A100.** A mechanism leaves the simulation only when the replay (`tools/igc/sim_replay.py`) shows zero
   misreads, no act on a right answer (or every one read and understood), every test passing and `vllm_refused` 0. Then the **PC
   layer**: Qwen3.5-4B NVFP4 as W4A8 on the RTX 5060 Ti + RTX 4070 Super, one locked config (sm_89 build), both arms on the same
   card. It tests the concept, confirms before the A100, and records a normal user's environment. **More right answers than control
   beyond noise (sign test p < 0.05, rule and API reader both): go to the A100; otherwise review the method first.** NVFP4 loops
   more than bf16, so it is the hardest test of the anti-loop software and a normal user's real result; report it per environment,
   A frozen set measures once; once its round is read it becomes development rows and may be studied, and the next
   measurement (PC layer and A100) draws a new frozen set. On the A100: a
   5-minute smoke run, then the rest; check the compute-unit balance before opening a VM and stop it when the round ends.
2. **Software computes, the model reasons; never lower the model's ability.** A change that loses right answers fails, whatever it
   saves.
3. **The General Style first, no tuning to the test.** A put is true under every reading of the `=`; rules come from mathematics or
   notation, never from a set's frequencies; development sets find bugs, the frozen set measures. The plain value for every model.
4. **Qwen3.5-4B first,** until the system is settled, then other models.
5. **Full context, sampled.** Never `--max-tokens`; T 0.6 with two or more seeds; vLLM runs are not paired across runs (#21), and the engine's
   repeat (49/49) was measured greedy only.
6. **Test with the smallest run that answers the question:** first pass, then only the misses the feature can reach, then the
   trade-off on a sample of right answers.
7. **Every written-answer test also goes through the API reader** (`tools/igc/answer_check.py`), on this PC only; report the rule's,
   the reader's and the corrected score.
8. **Caught / should have caught,** with the denominator; check every case before stating a cause. **Every touch a mechanism made:
   was the value it put in right or wrong** (checked again independently), split by answers that finished right and wrong; whether
   the touch was needed comes after that, separately.
9. **Present the latest state only;** negative or unconfirmed data goes in a footnote, never as a result.
10. **Time spent is the cost:** parallel tracks, no idle machine, background only, one engine at a time on the developer's PC.
11. **The tracker holds the knowledge:** bilingual issue posts at major progress, `gh --repo` always explicit.
12. **The docs are the source of truth, updated as information arrives.** The catalog (`docs/INDEX.md` and its pages) is the
   current state, `docs/plans/` the open work, `docs/ERRATA.md` the withdrawn and corrected claims; issues and `docs/archive/` are
   history. A result, correction, decision or bug goes into the docs in the same turn as its issue comment. Never quote a number
   from an issue without checking the catalog and the errata; a contradiction found gets the doc fixed, an errata row and a banner
   on the issue.

@AGENTS.md
