# CLAUDE.md

The rules for every coding agent on this repository live in `AGENTS.md`, so Claude Code and Codex follow one file. A rule added to one
file goes into the other in the same commit.

The rules agents miss most, all in full in `AGENTS.md`:

1. **Simulate first; the A100 comes last.** A mechanism goes to the A100 only when the replay (`tools/igc/sim_replay.py`) shows zero
   misreads, no act on a right answer (or every one read and understood), every test passing and `vllm_refused` 0; then a 5-minute
   smoke run, then the rest. Check the compute-unit balance before opening a VM and stop it when the round ends.
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
8. **Caught / should have caught,** with the denominator; check every case before stating a cause.
9. **Present the latest state only;** negative or unconfirmed data goes in a footnote, never as a result.
10. **Time spent is the cost:** parallel tracks, no idle machine, background only, one engine at a time on the developer's PC.
11. **The tracker holds the knowledge:** bilingual issue posts at major progress, `gh --repo` always explicit.

@AGENTS.md
