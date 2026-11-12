# PRD: MIP token channel, phase 0-1 (2026-11-12)

Design: [`2026-11-12-mip-token-channel-design.md`](2026-11-12-mip-token-channel-design.md). Plan: [`2026-11-12-mip-token-channel-plan.md`](2026-11-12-mip-token-channel-plan.md).
Tracker: mip-research#25.

## Problem

MIP acts on the model only at stop words: the server stops at `=`, tir4b reads and puts the value as text, and a new request resumes
(218-277 ms a put, 12-14 % of an answer's wall time; 4B NVFP4 W4A8 · llama.cpp, 4070 SUPER and 5060 Ti apart). It cannot act mid-line,
where no stop word exists, on a wrong digit as the model chooses it, or without inserting text. The developer wants software's always-true
values to act on the token probabilities in the thinking, as a guideline, with every intervention in MIP alone.

## Who uses it

- The developer and the agents running MIP's tests on the PC layer (tir4b over llama-server, the RTX 5060 Ti).
- Later: an inference engine's developers plugging MIP in (vLLM, Strata-xeno) through the same plan format.

## What it must do

1. Every token of every answer reaches MIP while the model generates; MIP keeps the answer's state.
2. At a value MIP computes (after `=` and `\approx` in the thinking), the next tokens are restricted to those that can spell the right
   value, in any tokenization; the model chooses among them. Nothing is steered in the final answer.
3. A question that is one fixed computation in symbols starts its thinking with the steered line `<expression> = <value>`; the thinking
   is not closed; the model writes the answer.
4. The arithmetic runs on the CUDAMathBackend (the 5060 Ti); the Python MathBackend is the reference and fallback; every device value
   is checked against Python.
5. Everything else (loop closes with cuts, rewinds, the Sandbox) keeps working through tir4b's requests.
6. Off by default: a request without `mip_id` is generated exactly as by the stock server.

## Acceptance

- Gate 1: the patched server without `mip_id` is token-identical to the stock build (greedy and seeded, 3 questions x 400 tokens).
- Gate 2: replaying round 9's traces, the planner steers exactly the values today's reader put, at the same places.
- Gate 3: every steered value is right when evaluated again independently; device mismatches 0.
- The live shim test passes: steered text appears; with the service down the server still answers.
- The development-rows comparison runs on the 5060 Ti with 0 right -> wrong answers caused by a steered value; answers whose final value
  software computed are counted apart. (A gain beyond noise is for round 10 to show, on frozen10.)
- Docs, AGENTS.md and the tracker updated in the same turn as the results.

## Out of scope

Phases 2-4 of the design (detection gaps, loops on the channel, Sandbox steering, sub-modules, vLLM and Strata-xeno); speculative decoding;
steering quotes and letter counts (they stay text puts in phase 1); the RTX 4070 SUPER (in use by the developer); frozen10.
