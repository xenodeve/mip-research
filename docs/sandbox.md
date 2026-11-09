# Sandbox and tools

Code that runs inside the model's reasoning. The model (or the engine, on its behalf) writes a short Python program while it
thinks; software runs it and the printed output comes back into the thinking as forced tokens, so the model reads a computed
result instead of working it by hand. This page covers the bounded interpreter in the engine, the real-Python kernel and its
isolation box, the rules that open, test and close around a block, the model's own `run_python` tool route, the MCP bridge,
and the 4B harness port. None of it is in the [best config](../README.md#best-config); each section says where it lives and
what it would take to use it.

| Mechanism | What it does (one line) | Runs on | In best config? | Best measured result | Status |
|---|---|---|---|---|---|
| [C1 bounded interpreter](#c1-bounded-python-interpreter) | A Python subset run on the host inside the engine, no file/net/process API | Strata engine (xeno/mip-sandbox, not in main) | no | HumanEval held-out half: 78 of 82 match CPython, 0 mismatch | engine only |
| [Block in the reasoning](#a-block-in-the-reasoning---igc-sandbox) | A ```` ```python ```` block in the thinking runs at its closing fence; output delivered as an output block | Strata engine | no | carries every engine arm below | engine only |
| [Forced block + close after a print](#forced-block--close-after-a-print) | Engine opens a block after N thinking tokens; a run that printed closes the thinking | Strata engine | no | 49 math questions x 2: **0.508x tokens, 77 vs 73 of 98 right** (Flash-Next) | built, measured, kept as option |
| [Hypothesis-loop rules](#hypothesis-loop-rules-draft-call-expect-why-gate-pass-close) | Run each draft when written, test it, hold the end of thinking for a check, close on a pass | Strata engine | no | LCB hard 10: 8 of 10 vs 7 of 10 with no MIP, 207,797 vs 497,237 tokens | built, measured, kept as option |
| [Falsify / verify / brute-force oracle](#falsify-verify-and-the-brute-force-oracle) | Ask what input breaks a passing draft; turn `Let's verify` into a program; a brute force from the statement as oracle | Strata engine | no | — | built, not measured yet |
| [Tool route (model's own `run_python`)](#tool-route-the-models-own-run_python) | `run_python` declared in the model's own tool format; calls answered while thinking (engine) or in the reply (bench/server) | Strata engine + `external_bench.py` | no | Sampled pairs: MATH long 21/21 at 0.81x, GPQA 11 vs 12 at 0.73x, SuperGPQA 10 vs 11 at 0.71x | built, measured, kept as option |
| [Tool hints (START placement)](#tool-hints---hint-after---hint-at) | START's words + an open call after N thinking tokens, at a line end, after `Wait`, or in place of `</think>` | `external_bench.py` (Strata engine) | no | MATH long 22: 21/22 at 101,808 tokens vs 157,235 with no tool (line hint) | built, measured, kept as option |
| [Real-Python kernel + isolation box](#real-python-kernel-and-the-isolation-box) | What the interpreter refuses runs in real CPython (sympy, numpy) in an AppContainer + Job Object box | Strata server (`serve/sandbox/`), tir4b `--box` | no | box start 13.0 ms mean; 101 escape tests pass | built, measured (mechanics), kept as option |
| [Project workspace and sandbox tool route](#project-workspace-search-symbols-and-sandbox_-tools) | A writable copy of a project, search/symbols, `sandbox_run/test/apply`, conflict path | Strata server | no | 160 + 177 tests on real boxes | built, not measured yet (with a model) |
| [MIP Adapter (MCP bridge)](#mip-adapter-mcp-tools-from-a-block) | `mcp.<server>.<tool>(...)` from a block in the reasoning reaches MCP servers | Strata server + engine bridge | no | 80 tests | built, not measured yet |
| [Reasoning tool in a served request](#reasoning-tool-in-a-served-request) | One run-config key wires `run_python` into `/v1/messages` and `/v1/chat/completions` | Strata server | no | 32 tests | built, not measured yet |
| [tir4b sandbox port (Qwen3.5-4B)](#tir4b-sandbox-port-qwen35-4b) | The engine's block rules rebuilt in Python for vLLM / llama-server: `--force-after`, `--execute`, `--box`, `--hint-calc`, `--loop-hint`, `--native-tool` | tir4b (vLLM/llama-server) | no | — (see notes) | built, measured, kept as option |
| [4B fine-tune for block use](#qwen35-4b-fine-tune-with-the-forced-block) | Hint-RFT style LoRA so the 4B opens blocks itself | tir4b + `tir4b_train.py` (A100) | no | — | built, not measured yet |
| [Offline checkers](#offline-checkers) | Differential runner vs CPython; hypothesis reader; coverage counter | offline tool | no | 577 programs, 0 mismatches vs CPython | built, measured |

## C1 bounded Python interpreter

A bytecode interpreter for a Python subset, on the host, inside the engine. The engine never links or calls CPython. There is no
API for files, network, processes, clock or environment; limits on steps, memory, output, recursion, parser depth and integer
digits; what it does not support is refused with a short error, never approximated. It has classes in the form the model writes
(`class Solution` with methods and `__init__`, `lru_cache` on a method), `heapq`, `bisect`, `sys`/`input` with `feed(...)`, and a
block's last top-level expression shows its value as a notebook cell does. Allowed modules: math, fractions, itertools,
functools, collections, heapq, bisect and sys (`kOnlyModules`). Speed: about 450-550M steps/s at `/O2`, 0.09 ms for a trivial run.

- Files: Strata-xeno xeno/mip-sandbox: include/strata/core/igc_sandbox.hpp (10,599 lines); include/strata/core/igc_sandbox_bridge.hpp:32 (`kOnlyModules`).
- Measured (mip-research#3), against CPython 3.11.9, default limits, `/O2 /MD`: HumanEval canonical solutions with their tests, the held-out odd half run once: **78 of 82 match, 2 refused, 2 step limit, 0 mismatch** (even half, seen in development: 76 of 82, 0 mismatch). After classes were added: 577 programs against CPython, 0 mismatches; 15 of 16 finished LiveCodeBench programs run and pass their public cases (the 16th needs `array`). [^c1]
- To use: build the engine from xeno/mip-sandbox and pass `--igc-sandbox` (below).

## A block in the reasoning (`--igc-sandbox`)

The engine watches the thinking for a ```` ```python ```` fence. At the closing fence it runs the program in the turn's session and
delivers ```` ```output ```` with what it printed, through the same result bridge as inline values. Delivered output is cut at
1,024 bytes (one forced token per byte); a block still open after 2,048 tokens is stopped with `[too long, not run]`; a block that
printed nothing gets `[it ran and printed nothing: call your function and print the result]`; a function returning only `None` is
told it is unfinished. With `--igc-close-repeat N`, the same program run again closes the thinking (a loop of whole blocks is
far longer than `in_cycle` reaches). Names persist across blocks of one turn (the session replays finished blocks).

- Flags: `--igc-sandbox` (needs `--igc-inline-exec --igc-result-bridge` and a learned-diagnostic bundle). Strata-xeno xeno/mip-sandbox: src/program/generate.cpp:966 (help), :2822 (parse).
- Files: Strata-xeno xeno/mip-sandbox: include/strata/core/igc_sandbox_bridge.hpp:24-28 (`kShownOutput`, `kMaxBlockTokens`, `kNothingPrinted`), :462 (`Session`); include/strata/program/igc_learned_diagnostic.hpp:459-530 (run, deliver, repeated-program close, block too long).
- Measured: the mechanism alone is not an arm; see the arms below. [^block-alone]

## Forced block + close after a print

The model rarely opens a block by itself on math (2 of 450 runs with MIP on, mip-research#3). So the engine opens one: after N
tokens of thinking, at the next line end, once per reasoning, it forces `Need a short Python program here.`, a blank line and
an opening fence (`kOpenBlock`, 11 tokens). The model writes the program. With `--igc-sandbox-close`, a run that printed and
did not fail a check is followed by `</think>`, so the model writes its answer.

- Flags: `--igc-sandbox --igc-sandbox-after 512 --igc-sandbox-close`. Variants: `--igc-sandbox-close-line` (the model's own line before the close), `--igc-sandbox-read N` (read the result for N tokens, then close), `--igc-sandbox-call` (open `print(name(` after a block that printed nothing).
- Files: Strata-xeno xeno/mip-sandbox: include/strata/core/igc_sandbox_bridge.hpp:161 (`kOpenBlock`); include/strata/program/igc_learned_diagnostic.hpp:552-563 (forced open), :510-521 (close after a fitting result), :544-549 (read window), :481-486 (forced call).
- **Measured (mip-research#3):** Qwen3.8-Flash-Next (Gyro-M pack) in the Strata engine on the developer's PC, exe `b324679801c45fdc`, 49 questions (29 GSM, 20 MATH), 2 rounds of 3 arms in rotation in one session:

  | Arm | Tokens | Ratio | Largest run | Stuck | Right |
  |---|--:|--:|--:|--:|--:|
  | Plain | 247,162 | | 13,309 | 2 | 73 of 98 |
  | Block at 512 + close after a print | **125,608** | **0.508** | 9,189 | 0 | **77 of 98** |

  Tokens were lower on 41 of 49 questions (sign-flip p below 0.00001); median change -1,920 tokens per question over the 2 rounds.
  The close fired in 66 of 98 runs. [^forced-acc] [^forced-code] [^forced-4b]
- To use: off by default; it changes outputs and shortens replies. Build xeno/mip-sandbox, add the flags above to a generate or served profile.

## Hypothesis-loop rules: draft, call, expect, why, gate, pass-close

Aimed at the plan in Strata-xeno xeno/mip-sandbox: docs/plans/2026-11-02-hypothesis-loop-plan.md (mip-research#8): the model tests a
hypothesis while reasoning, sees it fail and changes course. Each rule is a flag:

| Flag | Trigger and action |
|---|---|
| `--igc-sandbox-draft` | A bare fence holding a whole draft (a `def` or `class`) runs too; each new draft gets its call |
| `--igc-sandbox-call` | After a draft that printed nothing, open `print(name(` with the last top-level def (or `Solution().method(`) |
| `--igc-sandbox-expect` | Open the call as `expect(result, expected)`, a test against the question; prints MATCH / MISMATCH |
| `--igc-sandbox-why` | After a mismatch, write `Wait, this does not match. Is my approach wrong, or did I expect the wrong value? Why?` |
| `--igc-sandbox-gate` | Hold the first `</think>` for `Let me run my code on the examples before I answer.` and an open block calling the draft; again while the last check failed, at most 3 times |
| `--igc-sandbox-pass-close` | End the thinking once every check of a run has passed, anywhere in the reasoning |

- Files: Strata-xeno xeno/mip-sandbox: include/strata/core/igc_sandbox_bridge.hpp:37-43, :127-131 (lines), :143-144 (`kMaxChecks`, `kMaxGates`); include/strata/program/igc_learned_diagnostic.hpp:398-410 (gate), :478-506 (draft, call, why, pass close).
- **Measured (mip-research#8):** Flash-Next in the Strata engine, Colab A100 80 GB, greedy, 10 chosen LiveCodeBench hard rows, scored by the benchmark's tests:

  | Arm on the same 10 rows | Right | Tokens |
  |---|--:|--:|
  | `lon` no MIP | 7 | 497,237 |
  | `ld1` draft + gate + expect + why + real Python | 9 | 247,755 |
  | `ld2` = `ld1` + `--igc-sandbox-pass-close` | 8 | 207,797 |

  The close on a pass lost no row: 3364 went from 31,265 tokens (`ld1`) to 9,490 (`ld2`). [^hyp]
- To use: build xeno/mip-sandbox; `--igc-sandbox --igc-sandbox-draft --igc-sandbox-call --igc-sandbox-expect --igc-sandbox-why --igc-sandbox-gate --igc-sandbox-pass-close`, with `--igc-sandbox-exec` for real Python.

## Falsify, verify and the brute-force oracle

- `--igc-sandbox-falsify`: after a passing check, the model's own line `The examples match. What input would break this?`; with an oracle present, a passing draft is set against it on random small inputs (`kCompareLine`, `import random`).
- `--igc-sandbox-verify`: the model's `Let's verify` (or `Check:` at a sentence start) is finished as ` with a short program.` and an open block calling the draft, at most 3 times a reasoning.
- `--igc-sandbox-brute`: the block forced by `--igc-sandbox-after` asks for `def brute(` straight from the statement, run on the examples, as an oracle for what follows.
- Files: Strata-xeno xeno/mip-sandbox: include/strata/core/igc_sandbox_bridge.hpp:131-146, :392-410; include/strata/program/igc_learned_diagnostic.hpp:494-508 (falsify), :532-543 (verify), :555-558 (brute).
- Measured: no completed arm. [^brute]
- To use: build xeno/mip-sandbox and add the flags; an arm on the LCB rows is what is missing.

## Tool route: the model's own `run_python`

`run_python` is declared in the model's own tool format (Qwen's `<tool_call>`), so the model calls it unforced. Two routes run
the same session: the engine answers a call written while thinking (`--igc-sandbox-tool`: at `</tool_call>` it runs the code and
writes `<tool_response>` into the thinking, which goes on; another tool is told to wait for the reply), and the bench answers a
call in the reply (`--tool-loop`: up to 24 runs, then a line to finish, 4 more turns; a program sent a third time running gets
`TOOL_SPENT`).

- Flags: engine `--igc-sandbox-tool` (Strata-xeno xeno/mip-sandbox: src/program/generate.cpp:974; include/strata/program/igc_learned_diagnostic.hpp:443-458; include/strata/core/igc_sandbox_bridge.hpp:45-49, :108); bench `--tool-loop --sandbox --sandbox-tool --sandbox-exec` in [`tools/igc/external_bench.py`](../tools/igc/external_bench.py).
- **Measured, greedy (mip-research#8),** Flash-Next, A100, 10 LCB hard rows: `lt2` (bench route) 8 of 10, no row lost against no MIP (7 of 10); on the 7 rows right in both, 148,587 -> **92,418 tokens (0.62x)**. `lt3` (engine route) 8 of 10 at 231,785 tokens, the same as `lt2`'s 217,235. Row 3478 shows the loop the plan wants, unforced: two algorithms run on the examples, set against each other on random inputs, a fine-grid oracle, a timing on 1,000 circles.
- **Measured, sampled ON vs OFF (mip-research#8),** Flash-Next (Gyro-M) in the Strata engine, two Colab A100 80 GB, T 0.6 seed 1. ON = MIP Native + tool route + sandbox; OFF = MIP off, no tool. Rule and API reader agree on every row:

  | Set | OFF right | ON right | Tokens ON/OFF |
  |---|--:|--:|--:|
  | MATH long 22 | 21 | 21 | 0.81 (167,743 -> 135,772) |
  | GPQA 13 | 12 | 11 | 0.73 (411,685 -> 300,110) |
  | SuperGPQA calc 29 | 11 | 10 | 0.707 (479,294 -> 339,071) |

  [^tool-route]
- To use: build xeno/mip-sandbox; served, set `"reasoning_tool": true` (below); on the bench, `--tool-loop --sandbox --sandbox-tool`.

## Tool hints (`--hint-after`, `--hint-at`)

Spontaneous calls land after the whole thinking (203 of 205 in five greedy arms). START's Hint-infer (arXiv 2503.04625) puts its
words into the reasoning and an open `run_python` call follows. `--hint-after N` places it at a line end after N thinking tokens
with no call; `--hint-at wait` only after the model's own `Wait`/`Alternatively` opening a sentence; `--hint-at stop` in place of
the model's own `</think>`, once a row. At most 2 hints a row, each recorded (`hints[]` with token, char, words).

- Files: [`tools/igc/external_bench.py`](../tools/igc/external_bench.py) (around lines 76-90 and 488).
- **Measured (mip-research#8),** Flash-Next, A100, greedy, long rows (no-tool arm used at least 2,000 tokens):

  | Placement | MATH long 22 | MMLU-Pro long 19 |
  |---|---|---|
  | none, no tool (`m5o` / `mpo`) | 21, 157,235 tokens | 16, 288,365 tokens |
  | tool route, no hint (`m5t`) | 21, 121,453 | — |
  | line, N = 512 | **21, 101,808 (0.65x of no tool)** | 11 |
  | wait | 20 | 14, per-row median 0.59x |
  | stop | 21, per-row median 0.74x | **15**, per-row median 0.81x |

  [^hint]
- To use: `external_bench.py ... --tool-loop --hint-after 512 --hint-at stop` (the accuracy-safe placement).

## Real-Python kernel and the isolation box

A block the interpreter refuses (an unknown module such as sympy, an unsupported construct, or `mcp.`) moves to a warm real-Python
kernel for the rest of the turn (`SBX`/`SBXR` lines between engine and server); the first such block replays the earlier blocks
silently so only its own output is shown (mip-research#9, fixed in 8569a1a2). A dead kernel (past 120 s) is replaced, never
written to (04cc1b26). On Windows the kernel runs in an AppContainer with a Job Object: no network, the real project path refused,
a disk poll and a wall clock (2 GB, 3,600 s for the server's box).

- Flags: engine `--igc-sandbox-exec` (not with `--batch`); tir4b `--execute --box`. Files: Strata-xeno xeno/mip-sandbox: serve/sandbox/exec_bridge.py, kernel_main.py, isolation.py; include/strata/core/igc_sandbox_bridge.hpp:544 (`run_outside`).
- Measured on the developer's PC, Windows 26300.9711, our own test programs only (mip-research#7): box start mean 13.0 ms (median 12.3); `python -c pass` in a warm box 31.7 ms vs 28.7 ms outside; `import sympy` 542.5 ms in a warm box vs 515.0 ms outside; `import numpy` 157.3 vs 150.8 ms; kill and cleanup 4.1 ms. `tests/sandbox`: 101 passed, 1 skipped, five rounds in a row (113 passed after later fixes). [^box]
- To use: D1 (real Python on the developer's PC) is the developer's call; it was approved once, for the 4B five-layer test (2026-11-04). On a throwaway VM no approval is needed.

## Project workspace, search, symbols and `sandbox_*` tools

The design (Strata-xeno xeno/mip-sandbox: docs/plans/2026-11-02-sandbox-workspace-design.md, 55 cases, 12 threats) gives the model a
writable copy of a project inside the box, where it can read, search, edit and test; the real project is read-only and a tested
diff reaches it only through the existing permission and checkpoint path (`sandbox_apply`, one question to the user with the diff
and the test line; `sandbox_conflict` / `sandbox_resolve` with `git merge-file --ours`). Off by default: `"sandbox": {"on": true}`,
and `sandbox_run`/`sandbox_test` also need `"real_python": true`.

- Files: Strata-xeno xeno/mip-sandbox: serve/sandbox/workspace.py, search.py, symbols.py, output.py, grep_worker.py, session.py, testrun.py, tools.py.
- Measured on the developer's PC, this repository's copy (mip-research#7): copy warm with 8 workers 2.0-2.5 s; `changed()` 64-74 ms; a box on the copy starts in about 1.0 s. Tests: `serve.test_sandbox_tools` 160 OK (real AppContainer boxes), `serve.test_sandbox_workspace` 177 OK (2 skipped); guards turned off one at a time: 62 of 62 went red. [^workspace]
- To use: set the run-config keys above; never run in a live chat.

## MIP Adapter: MCP tools from a block

A program in a block of the reasoning reaches existing MCP servers: `mcp.tools("words")` finds tools with signatures,
`mcp.<server>.<tool>(arg=value)` calls one and returns a dict, list or str; only what the program prints enters the context. Every
check runs server-side: the request offered the tool, the chat's deny rules, `annotations.readOnlyHint` must be true, `inputSchema`,
budgets of 8 calls per block, 32 per turn, 120 s of tool time per block. This is Roadmap M1 + M2 in text-feedback mode, the MCP
bridge of Roadmap Phase C (Strata-xeno#289).

- Files: Strata-xeno xeno/mip-sandbox: serve/mip/mcp_adapter.py, serve/mip/core.py, serve/sandbox/kernel_main.py, include/strata/core/igc_sandbox_bridge.hpp; design docs/plans/2026-11-02-mip-adapter-design.md.
- Measured: `serve/test_mip_adapter.py` 80 tests OK; 37 guards turned off one at a time, each caught; on the A100 the merged kernel imported numpy 2.1.3, sympy 1.14.0, mpmath, networkx (Strata-xeno#289). [^adapter]
- To use: a prompt line that tells the model `mcp` exists, an isolation box (or a VM), and an `agentStatus` step for the web app.

## Reasoning tool in a served request

One run-config key, `"reasoning_tool": true`, wires the tool route into the server: it adds `--igc-inline-exec --igc-result-bridge
--igc-sandbox --igc-sandbox-tool` (and `--igc-sandbox-exec` only with D1 and a box, or on a VM), declares `run_python` after the
client's tools on `/v1/messages` and `/v1/chat/completions`, answers a call in the reply itself (24 runs, then 4 to finish), and sends
`{"event": "reasoning_tool", ...}` as a call starts and ends, with status words in English and Thai. The server refuses to start
without a learned-diagnostic bundle or with replicas.

- Files: Strata-xeno xeno/mip-sandbox: serve/reasoning_tool.py, serve/test_reasoning_tool.py (mip-research#8, Strata-xeno#291).
- Measured: `serve.test_reasoning_tool` 32 tests OK; full serve suite 1,928 tests. [^rtool]
- To use: the engine-side fix of Strata-xeno#291 (names across the interpreter and the next round), D1, then a served Claude Code turn on D2x with a timeline.

## tir4b sandbox port (Qwen3.5-4B)

[`tools/igc/tir4b.py`](../tools/igc/tir4b.py) rebuilds the engine's block rules in Python with the engine's own words and limits, for
the 4B on vLLM (Colab, **4B bf16 · vLLM**) or llama.cpp's llama-server (the developer's PC, **4B bf16 GGUF · llama.cpp**). `--force-after N` opens `Need a short Python program here.` and a fence
at the next line end after N thinking tokens, once; `--execute` runs every block (forced or the model's own) in the kernel and returns
an output block cut at 1,024 bytes; `--close-fit` closes the thinking after a run that printed; `--box` runs the kernel in the
AppContainer on the PC; `--close-repeat N` closes on a repeated program; `--native-tool` offers `python` in Qwen3.5's own tool format;
`--hint-calc` opens a block with CoRT's hint where a tedious hand calculation ends (a 3-digit number, 3 operators or a value past 999,
at most once per 1,024 thinking tokens) and adds CoRT's trust line after a run (`--trust-line cort|none|positive`); `--loop-hint`
cuts a loop to one round and opens a program to test the idea (up to 3 times). A zero-argument function that is defined and never
called is called and its value printed (`auto_call`).

- Files: [`tools/igc/tir4b.py`](../tools/igc/tir4b.py) (flags around lines 1953-2000, kernel and box at 1749-1760, forced open at 961 and 1100); [`tools/test_igc_tir4b.py`](../tools/test_igc_tir4b.py). `--execute`/`--box` need Strata-xeno on `PYTHONPATH` (`serve.sandbox.exec_bridge.Kernel`).
- Measured (mip-research#12), Qwen3.5-4B bf16, vLLM, Colab A100, T 0.6, seed 2, one sample, paired rows of the control's set: `--native-tool --system` (the system turn asks for the tool) called on 2 of 9 rows, both right, at 0.68x tokens; on 066aabf2 it wrote the quartic, called sympy `nroots`, then `solve` for the exact root. [^tir4b]
- To use: add the flags to a tir4b run; any block that runs model code needs a VM or `--box` (D1).

## Qwen3.5-4B fine-tune with the forced block

`tir4b.py keep` picks one trajectory per question (right, ended, no inserted text but the open line and the close; a short tool-free
answer when the question was answered before any block). [`tools/igc/tir4b_train.py`](../tools/igc/tir4b_train.py) trains a bf16 LoRA
on every linear projection of the text decoder (Gated DeltaNet included), with the forced open line and close in the loss (START
Hint-RFT) and the kernel's output masked (ReTool), and merges it into the checkpoint's own class so vLLM serves it as the base.
Training rows: MATH train levels 3-5, with evaluated ids left out.

- Measured: none. [^ft]
- To use: an A100 for training and the held-out arms (AIME 2025 30, HMMT Feb 2025 30, MATH long 22, 2 seeds) as laid out in mip-research#12.

## Offline checkers

- [`tools/igc/sandbox_oracle.py`](../tools/igc/sandbox_oracle.py): runs a program under CPython and under the interpreter binary and requires the same stdout and the same exception class; verdicts match / expected / refused / limit / MISMATCH / CRASH / HANG. Behind the 577-program, 0-mismatch figure.
- [`tools/igc/hypothesis_read.py`](../tools/igc/hypothesis_read.py): per row of an arm, what the checks said and whether a real mismatch was followed by a run that matched; a stub's `got None` does not count.
- [`tools/igc/coverage.py`](../tools/igc/coverage.py): caught / should-have-caught per mechanism, including "loops given a program" and "programs that printed".
- [`tools/igc/arm_plan.py`](../tools/igc/arm_plan.py): reuses base rows no rule of an arm could change, runs shortest rows first, stops an arm at its first loss.

[^c1]: Refused on HumanEval: `random` 4, `hashlib` 1, `re` 1, `eval` 1. The engine links the static CRT, where a fuzz differs from CPython on 1 of 16,000 lines (a last-bit `pow`); with `/MD` 0 of 71,719. Not tested: GCC, Clang, Linux. Annotations are ignored. The speed figures are an agent's numbers from runs alongside other work (mip-research#3).
[^block-alone]: Note: with the sandbox on and no forced block, HumanEval tasks 0-39 passed 38 of 38 in both arms (63,392 vs 87,398 tokens, the difference is one task, `prime_fib`), so that set has no room. The model draws `def`s without calling them; most of 15 blocks returned "printed nothing" (mip-research#3). A first block moving to the kernel used to show the old prints and cut its own (mip-research#9), fixed in 8569a1a2.
[^forced-acc]: Note: right answers +4 (more on 6 questions, fewer on 2) is p 0.29, no measured accuracy gain; the run shows only that the close did not lower accuracy. The plain arm had 2 stuck runs. Seconds were 0.55x but there was no ABBA and no timeline, so it is not a speed claim. One wrong answer came from the close itself (a wrong program printed, the close fired). 13 close-arm runs opened a block and printed nothing (sympy refused 5, block too long 4), 41 % of the arm's tokens. 10 of 66 replies after a close were under 80 characters, 4 of them only the program. Machine: the developer's PC, which does not repeat itself token for token; the sampling setting is not stated in the source comment. The developer's later rule requires this result to clear the same bar on a paired machine; that has not been run. This result is not in the mip-research README or the Strata-xeno README as of 2026-11-08.
[^forced-code]: Note: on coding the close loses. LiveCodeBench hard, 20 rows, A100, paired, 6 s limit: plain 17, block at 512 + close 14 (0.58x tokens), + read 256 tokens then close 15, block at 2,048 + close 13. The developer ruled a close that cuts reasoning short out for coding (mip-research#3). On 14 long HumanEval tasks the forced block did nothing (12 tasks without `prime_fib`: 25,835 vs 25,136 tokens; 36 of 37 output blocks "printed nothing"). The forced `print(` call on 40 HumanEvalFix tasks: 40/40 both arms, but 12 of 22 calls printed a constant (`print(1)`), a blind close.
[^forced-4b]: Note: ported to Qwen3.5-4B (tir4b, A100, vLLM bf16, T 0.6, 82 held-out rows AIME 2025 + HMMT Feb 2025 + MATH long): base 58, forced block + close 56, tokens 0.931x; 40 of 82 forced blocks were too long (the model reasoned in comments) and the model opened a block itself 2 times in 82 rows (mip-research#12).
[^hyp]: Note: the main metric of the plan, "a real mismatch, then a pass", is 0 in every arm. `ld1`'s +2 is 3344 (a malformed hidden case `[0, 0]` passed by an early-return line no test led to) and 3460 (MIP's loop rules, not the sandbox). In `ld2` the only change against `lon` is 3460. On 3017 the model's `expect` used a value it made up and the why-line called a right program wrong. 10 rows, one greedy round. The forced block at 512 stood before any draft: 7 of 58 blocks gave a value in `lsb` (mip-research#8).
[^brute]: Note: `lb1` (brute force) stopped at its first row by design: an unfinished brute force returned `None` and was called a wrong approach (fixed in c1460ad2, `kStubNote`). `lb1b` was stopped mid-run when the tool route became the main path; `lfx` (every Native mechanism) was stopped 87 s in, before it ran; `lcb-full1` was held for compute. In an early forced arm the brute force gave a value 3 times in 181 blocks (mip-research#8).
[^tool-route]: Note: no wrong-to-right row on any set; the one changed row each on GPQA and SuperGPQA went right to wrong. On GPQA it was the only row where ON called on its own: 26 calls built on a sign convention the model made up, and they misled. Calls inside the thinking: 0 of 27 (GPQA), 0 of 16 (SuperGPQA), 0 on MATH long; every call came after `</think>`, so the saving is in the thinking under the tool prompt and MIP values, not in what calls returned. HLE 10: 4 vs 4 by the API reader at 1.054x tokens (the rule gave ON 5, a false positive on a hedged line). GPQA 13 and SuperGPQA 29 were picked as the greedy control's misses plus some right rows. One seed, two different A100 sessions (seconds not comparable). The source runs are Flash-Next in the Strata engine (mip-research#8, mip-research#12 "On Flash-Next ... GPQA: 0 of 27"). Greedy-era wins (GPQA gps 11/13 vs 6/13) were matched by sampled no-tool controls (6/7 and 3/7 on the 7 misses), so they are not the method's. `lm1` (30 new LCB hard rows, 25 of 29, later rescored 28 of 29) has no paired control: `lmo` was lost when Colab ended the session.
[^hint]: Note: no placement wins both suites, and no hint moved a question from wrong to right on Flash-Next. The stop hint's saving is the tool route's (about 1.0x of the tool route on both suites); line saves most on MATH and loses 5 rows on MMLU-Pro by cutting the weighing of a question's reading; wait loses a convention row. Placement tuning stopped by a rule registered beforehand. The first hint run (`m5h2`) lost 4 rows to a bench bug (fixed in d66a23fe); `m5w` lost one to the repeat rule e0fee1cc (fixed in eb3f79af) (mip-research#8).
[^box]: Note: a deny ACE does nothing against an AppContainer on 26300; the disk limit is a poll, not a quota (writers ended at 43-143 MiB with a 32 MiB limit); the job's CPU limit is a backstop; MXC cannot meet the isolation target alone. Not isolated: a box can tell whether any path exists, read `C:\Windows` and Program Files and HKLM. Linux / bubblewrap never ran. A box bug (`_Stays` had no `poll()`) failed 20 of 45 blocks in the 4B pc5 run, fixed in 3aed1e9b (mip-research#7, mip-research#12).
[^workspace]: Note: design figures that did not hold: first copy 29.8 s (1 sample) against 4.1 s; first change scan 16-20 s against 246 ms. tree-sitter never ran (no grammar installed). Not built: reattach after restart, read-only `node_modules`/`.venv` mapping, L3 (references, rename, diagnostics). Windows only (mip-research#7).
[^adapter]: Note: no real model has called `mcp`, no real MCP server has been used, no prompt tells the model about it, and the web app has no status step for it. `readOnlyHint` is the server's own word (Strata-xeno#289).
[^rtool]: Note: never run with the engine. Names do not reach a call in the reply or the next round unless the turn had moved to the kernel, and Claude Code's history lacks the reply's call, so the next request re-reads the prompt from there; neither is measured (Strata-xeno#291).
[^tir4b]: Note: on the 4B none of these beat its control on a counted run. `--native-tool` alone: 0 calls in 9 answers. `--hint-calc` (5 vs 4 on 15 rows) ran without `--execute`, so the model's own blocks never ran and their outputs were made up; not counted. `r1-full` (`--hint-calc --loop-hint --execute --loop-close --close-repeat 2`) reached 3 paired rows (2 vs 1, 0.37x) before compute ran out. The five-layer PC run `pc5-full` (`--box`) was stopped by the box bug and a 30K cap; its coverage counts (loops given a program 3 of 5; programs that printed 15 of 45, 20 failures from the box bug) are from unfinished arms (mip-research#12).
[^ft]: Note: the FT + MIP + Sandbox arm and the FT arm with no forcing were running when account a reached 0 CU; no result was posted (mip-research#12; Strata-xeno xeno/mip-sandbox: docs/reports/2026-11-04-mip-mechanism-evidence.md, row 10).
