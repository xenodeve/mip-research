# MIP token channel, phase 0-1: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** llama-server hands MIP every token of every answer; MIP steers the values it computes into the thinking by masking each step
to the tokens that can still spell the right value; arithmetic runs on the CUDAMathBackend.

**Architecture:** A thin `llama_sampler` at the head of llama.cpp's chain (`common/mip-channel.cpp`) streams accepted tokens over a
loopback socket to a Python service inside tir4b's process (`tools/igc/mip_channel.py`), which keeps one `Rollout` per answer and answers
trigger tokens with a plan: a token DAG (`tools/igc/token_trie.py`) the shim applies by itself. Mechanisms that need a request (cut,
rewind, Sandbox) stay on tir4b's request path (`drive_channel`, a copy of `drive_http`'s loop).

**Tech Stack:** llama.cpp `1deefcca3` (C++17, Winsock, vendored nlohmann/json), Python 3.11 stdlib (`socket`, `threading`, `json`),
tir4b, the CUDAMathBackend DLL (mode 2) through `mathexpr`'s `MIP_EXPR_DEVICE=cuda`.

**Spec:** `docs/plans/2026-11-12-mip-token-channel-design.md`

## Global Constraints

- Steer only in the thinking; nothing in the final answer; no inserted notes or invitations (AGENTS.md, 2026-11-12).
- Steering = the tokens that can still spell the right value, in any tokenization, are the only ones allowed at that step (mask the
  rest to -inf); the model chooses among them (the developer, 2026-11-12, option "ข").
- Only values MIP computes (arithmetic, MathBackend ops, `\approx` digits, the question's quotes and letter counts); never re-check
  words or the method.
- A fixed computation in symbols: steer the line `<expression> = <value>` at the start of the thinking; do not close the thinking.
- Arithmetic on the CUDAMathBackend (`MIP_EXPR_DEVICE=cuda`, `MIP_CUDA_RESIDENT=2`, the RTX 5060 Ti: `CUDA_DEVICE_ORDER=PCI_BUS_ID
  CUDA_VISIBLE_DEVICES=1`); Python MathBackend stays the reference and the fallback; every device value is compared with `Fraction`
  and mismatches are counted (the developer, 2026-11-12: "use the CUDAMathBackend as the main one, develop the Python one beside it").
- **The RTX 4070 SUPER is not touched** (the developer is using it). Every GPU run: `CUDA_VISIBLE_DEVICES=1` (PCI order), reported first
  with the nine items of AGENTS.md "How a test is run".
- llama.cpp base `1deefcca3`; build `build-w4a8-mip` (sm_89 only, the same cmake flags as `build-w4a8.cmd`); the patch kept at
  `third_party/llama.cpp/mip-channel.patch` with the MIT notice; build label **4B NVFP4 W4A8 · llama.cpp+MIP · PC layer**.
- Server config locked: `-c 262144 -kvu -fa on -ctk q8_0 -ctv q8_0 --no-cache-idle-slots -cram 0 --prio 2`, T 0.6, top-p 0.95,
  top-k 20, full context (never `--max-tokens`).
- No new library: Winsock and vendored nlohmann/json on the C++ side, stdlib on the Python side.
- The shim never blocks longer than `MIP_CHANNEL_MS` (default 50) waiting for a plan; a missed plan is skipped and counted.
- With no `mip_id` in a request, the patched server behaves exactly as the stock build (gate 1).

## Review Focus

1. Two answers of the same question in flight at once (same prompt, different `mip_id`): plans must reach the right slot. Test in task 4.
2. The service is down or slow: the server must keep generating, unsteered, and count misses. Test in task 5 (`MIP_CHANNEL_MS`).
3. A value whose text no token sequence can spell from the current position (a leading space the tokenizer merges with the `=`): the
   planner must return no plan, never a dead mask (all `-inf`). Test in task 2.
4. `</think>` written while a plan is active: the plan ends there (the answer is never steered). Test in task 3.
5. The request ends (stop word, n_predict) in the middle of a steered value: the next request's prompt carries the partial value, and
   the service must not steer it again from the start. Test in task 4.

---

### Task 1: Phase 0 probe: the rank of the right first token at put positions

**Files:**
- Create: `scripts/pc/rank_probe.py`
- Create: `results/token-channel/rank_probe.txt` (output)

**Interfaces:**
- Consumes: round 9's stored General traces (`results/round10/replay_gate.py` already loads them: rows, records, tokenizer).
- Produces: a table of the right value's first-token rank and probability at each stored put position, on the stock server.

- [ ] **Step 1:** Write `rank_probe.py`: for up to 300 stored inline-put events in the thinking, POST `/completion` with `prompt` = the
  answer's text up to and including the `=` token, `n_predict: 1`, `n_probs: 50`, `temperature: 0.6`, `top_k: 20`, `top_p: 0.95`,
  `cache_prompt: true`; read `completion_probabilities[0].top_logprobs`; record the rank of the put value's first token (encoded with
  a leading space as tir4b puts it, and without), and whether it is inside top-20.
- [ ] **Step 2:** Report the run first (nine items), then start a stock `build-w4a8` llama-server on the 5060 Ti (`-np 4`, locked flags)
  and run it; stop the server.
- [ ] **Step 3:** Write the table (rank 1 / 2-20 / outside / not in the top 50) to `results/token-channel/rank_probe.txt`; commit.

### Task 2: `token_trie.py`: the token DAG that spells a value

**Files:**
- Create: `tools/igc/token_trie.py`
- Test: `tools/test_token_trie.py`

**Interfaces:**
- Produces: `vocab_pieces(tokenizer) -> list[str]` (decoded text of every id, cached per tokenizer); `ValueDag.build(text: str,
  pieces: list[str]) -> ValueDag | None`; `ValueDag.allowed(node: int) -> list[int]`; `ValueDag.step(node: int, token: int) -> int |
  None` (the next node, or None); `ValueDag.end: int`; `ValueDag.to_json() -> dict` = `{"edges": [[[token, node], ...] per node],
  "end": int}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_dag_spells_every_tokenization():
    pieces = ['', ' ', '1', '2', '12', ' 1', '=', 'a']
    d = ValueDag.build(' 12', pieces)
    assert sorted(d.allowed(0)) == [1, 5]          # ' ' or ' 1'
    assert d.step(0, 5) is not None and sorted(d.allowed(d.step(0, 5))) == [3]
    n = d.step(d.step(0, 1), 4)                    # ' ' then '12'
    assert n == d.end

def test_no_dag_when_no_token_can_start():
    assert ValueDag.build(' 7', ['', '1', '2']) is None

def test_no_dead_node():
    d = ValueDag.build(' 12', ['', ' ', '1', ' 1'])  # ' ' '1' then nothing spells '2'
    assert d is None

def test_real_tokenizer_digits():                     # skipped when the HF tokenizer is not on disk
    tok = load_tokenizer()
    d = ValueDag.build(' 480', vocab_pieces(tok))
    assert d is not None and tok.encode(' 480', add_special_tokens=False)[0] in d.allowed(0)
```

- [ ] **Step 2:** Run `python -m unittest tools.test_token_trie -v`: fails (module missing).
- [ ] **Step 3:** Implement: nodes are text offsets 0..len(text); an edge from i to i+len(p) for every non-empty piece p with
  `text.startswith(p, i)` (index pieces by first character); keep only nodes that reach `end` (prune backwards); `build` returns None
  when node 0 does not reach the end. Pieces that are special tokens (`<|...|>`) are excluded.
- [ ] **Step 4:** Run the tests: pass. Commit.

### Task 3: the planner: what to steer, from the Rollout's rules

**Files:**
- Create: `tools/igc/mip_planner.py`
- Modify: `tools/igc/tir4b.py` (the `=` branch of `Rollout.take`, lines ~1129-1160: extract `Rollout.put_value(text: str) -> tuple[str,
  str] | None` returning `(expr, put_text)` with exactly today's rules: `inline_found`, echo guard, `question_value`, plain spelling;
  `take` calls it, behaviour unchanged)
- Test: `tools/test_mip_planner.py`

**Interfaces:**
- Consumes: `Rollout.put_value`, `approx_put` (the `\approx` value), `read_exact` (the question reader), `ValueDag`.
- Produces: `Planner(rollout, pieces).on_trigger(text: str, phase: str) -> Plan | None`; `Planner.at_start(question: str) -> Plan | None`;
  `Plan = (text: str, dag: ValueDag, kind: str, after_ban: list[int])` (`after_ban`: the digit ids banned for one step after the value,
  today's `--no-digit-join`).

- [ ] **Step 1: Write the failing tests:** `on_trigger('... $3 \\times 4 =', 'think').text == ' 12'`; same text with phase `'answer'` ->
  None; a text after `</think>` -> None; `at_start('Calculate 2 x 240.')` -> text `'2 x 240 = 480\n'` (the expression as the question
  writes it); `at_start('Find all x such that ...')` -> None; a value the DAG cannot spell -> None (Review Focus 3).
- [ ] **Step 2:** Run: fails.
- [ ] **Step 3:** Implement `put_value` extraction (a refactor: `tools/test_*` of tir4b still pass) and `Planner`.
- [ ] **Step 4:** Run `python -m unittest discover -s tools -p "test_*.py" -t .`: all pass. Commit.

### Task 4: `mip_channel.py`: the service

**Files:**
- Create: `tools/igc/mip_channel.py`
- Test: `tools/test_mip_channel.py`

**Interfaces:**
- Consumes: `Planner`, `Rollout`.
- Produces: `ChannelService(host='127.0.0.1', port=0, triggers: set[int], think_close: int).start() -> int` (the port);
  `register(mip_id: str, rollout: Rollout, planner: Planner)`; `stats() -> dict` (asks, plans, misses, steered tokens). The wire
  protocol, one JSON object per line:
  - service -> shim on connect: `{"t":"hello","triggers":[ids],"think_close":id}`
  - shim -> service: `{"t":"open","mip_id":s,"slot":k,"prompt":[ids]}`, `{"t":"tok","slot":k,"id":tok}`,
    `{"t":"ask","slot":k,"seq":n}`
  - service -> shim: `{"t":"plan","slot":k,"seq":n,"dag":ValueDag.to_json()|null,"after_ban":[ids]}`
  The service records each steered value on the Rollout as an event `dict(kind='steered value', expr=..., value=..., at=...)`.

- [ ] **Step 1: Write the failing tests** with a fake shim over a real socket: hello received; open + tokens of `'3 \\times 4 ='` + ask
  -> a plan whose DAG spells `' 12'`; two slots with the same prompt and different `mip_id` get plans for their own rollouts (Review
  Focus 1); an `open` whose prompt already ends in a partial steered value (`'... = 1'`) gets no plan for that `=` (Review Focus 5);
  an ask for an unknown slot -> `dag: null`; ask seq 0 right after the first `open` of an answer to a fixed computation -> the
  `at_start` plan, and seq 0 of a later request of the same answer -> `dag: null`.
- [ ] **Step 2:** Run: fails.
- [ ] **Step 3:** Implement: one thread per connection, a lock per slot; the slot's text = decoded prompt + accepted tokens; phase from
  `think_close`.
- [ ] **Step 4:** Run: pass. Commit.

### Task 5: the llama.cpp shim

**Files:**
- Create: `C:/AI/llama.cpp/common/mip-channel.h`, `common/mip-channel.cpp`
- Modify: `common/common.h` (`common_params_sampling`: `std::string mip_id;`), `common/sampling.cpp` (`common_sampler_init`: when
  `params.mip_id` is not empty, `llama_sampler_chain_add` the shim first), `tools/server/server-schema.cpp` (`field_str("mip_id")`),
  `common/CMakeLists.txt` (the new source, `ws2_32` on Windows)
- Create: `third_party/llama.cpp/mip-channel.patch`, `third_party/llama.cpp/README.md` (base commit, MIT notice), `scripts/pc/build-w4a8-mip.cmd`
- Test: `tools/test_mip_shim_live.py` (needs the built server and the 5060 Ti; skipped without `MIP_SHIM_TEST=1`)

**Interfaces:**
- Produces: `struct llama_sampler * mip_channel_init(const std::string & mip_id)` (nullptr when `MIP_CHANNEL` (host:port) is unset);
  env `MIP_CHANNEL_MS` (default 50). The shim: buffers prompt accepts until the first `apply`, then sends `open` followed
  by `ask` seq 0 (the service answers it with `Planner.at_start` when the answer has no thinking yet, else `dag: null`); sends each `tok`;
  after a trigger token sends `ask` and waits for that `seq` at the next `apply` up to `MIP_CHANNEL_MS`; with a plan, sets every token
  outside `allowed(node)` to `-INFINITY` in `cur_p`; advances the node on `accept`; at `end` applies `after_ban` for one step; drops the
  plan on `think_close` (Review Focus 4) or on a token outside the DAG.

- [ ] **Step 1: Write the failing live test:** a `ChannelService` with a planner that steers `' 999'` after `=`; the patched server
  (`MIP_CHANNEL` set) completes `Compute 3*4 =` with `mip_id`; the reply text starts with ` 999`; without `mip_id` the reply is the stock
  one; with the service stopped the request still completes (Review Focus 2) and the shim's miss counter (logged) is 1.
- [ ] **Step 2:** Build with `scripts/pc/build-w4a8-mip.cmd`; run the test: fails (no shim).
- [ ] **Step 3:** Implement the shim and the server field; write the patch (`git -C C:/AI/llama.cpp diff 1deefcca3 > third_party/llama.cpp/mip-channel.patch`).
- [ ] **Step 4:** Rebuild, run the live test on the 5060 Ti: pass. Commit (repo side: patch, notice, build script, test).

### Task 6: `drive_channel`: tir4b's driver on the channel

**Files:**
- Modify: `tools/igc/tir4b.py` (`--channel` flag; `Rollout._request` leaves `=` and `\approx` out of the stops when the arm has
  `channel`; `Rollout.take` puts no inline value when `channel`; `run()` starts the service and passes `mip_id` per request)
- Create: `tools/igc/mip_driver.py` (`drive_channel(post, tok, ids, rollouts, kernel_factory, service, live_most=16)`: `drive_http`'s loop
  plus `mip_id=f'{row_id}#{sample}#{n}'` on each request and `service.register(...)`)
- Test: `tools/test_mip_driver.py` (a fake `post` that replays tokens through a fake shim)

**Interfaces:**
- Consumes: `ChannelService`, `Planner`, `drive_http`'s request body.
- Produces: `--channel` in tir4b (refused without `--server`); records with `steered value` events; `report.json` gains `steered`,
  `channel_misses`, `device_mismatches`.

- [ ] **Step 1: Write the failing test:** with `--channel`, a rollout's requests carry `mip_id` and no `=` stop; a steered value appears
  in `records.json` events; the answer phase gets no plan.
- [ ] **Step 2:** Run: fails. **Step 3:** Implement. **Step 4:** All tools tests pass. Commit.

### Task 7: gates 1-3

**Files:**
- Create: `scripts/pc/gate1_parity.py`, `results/token-channel/gate1.txt`
- Create: `scripts/pc/gate2_replay.py`, `results/token-channel/gate2.txt`
- Modify: `results/round10/analyze.py` (check `steered value` events like puts: each value evaluated again exactly)

- [ ] **Step 1 (gate 1):** stock `build-w4a8` and `build-w4a8-mip` without `mip_id`, the 5060 Ti, the same 3 questions x 400 tokens greedy
  and 3 x 400 sampled at seed 1: token-identical, written to `gate1.txt`. Report the run first.
- [ ] **Step 2 (gate 2):** CPU only: round 9's General traces replayed through `Planner.on_trigger` at every stored `inline value` event in
  the thinking: planned text == recorded put, 0 differences; positions the planner would steer that today's reader did not put are
  listed (expected 0 with the same rules). `gate2.txt`.
- [ ] **Step 3 (gate 3):** `analyze.py` on a channel run's records: every steered value right.
- [ ] **Step 4:** Commit.

### Task 8: smoke and the development-rows comparison (the 5060 Ti)

**Files:**
- Create: `scripts/pc/channel_compare.sh` (one patched server, `MIP_CHANNEL` set, two arms at once: General text puts, General with
  `--channel`; rows `pause-rows.jsonl`-style development rows, 2 seeds)
- Create: `results/token-channel/compare-*/` (records, report, timing)

- [ ] **Step 1:** Report the run (nine items); 5-minute smoke; read the log (steered values, misses 0, device mismatches 0).
- [ ] **Step 2:** The comparison on development rows (round 10's `r10a` DeepMind rows, not frozen10): per arm the rule score and the API
  reader's, every touch right / needed / helped, right -> wrong pairs read by hand (must be 0 caused by a touch), answers whose final
  value software computed counted apart.
- [ ] **Step 3:** Docs in the same turn: the design (results), `docs/inject.md` and `docs/compute.md` (the channel; the CUDAMathBackend as
  MIP's main computer), `docs/INDEX.md`; AGENTS.md: "the CUDAMathBackend is MIP's main computer, the Python MathBackend the reference
  and fallback, developed beside it (the developer, 2026-11-12)". Commit; issue comment (bilingual) on mip-research#25.
