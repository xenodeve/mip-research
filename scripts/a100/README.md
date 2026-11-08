# Round 9 kit (A100), and the 5060 Ti first

Round 9 is the General Style test with no profile: the best config against a plain control in the same round, 120 frozen rows x 2
samples, seed 1, the whole 262,144-token context. Everything here runs from this PC; the A100 is opened only by `launch9.sh`.

## Data lives outside the repository

`WORK` (default `C:/Users/xenod/mip-work/a100`) holds the rows (they carry the answers, and this repository is public), the stored arms of
rounds 1-8 (`out/`), the replays (`sim/`), the PC runs (`pc/`) and the logs. The 4B for the 5060 Ti is in `C:/Users/xenod/mip-work`
(`q4b-hf/` the official weights, `gguf/qwen35-4b-{bf16,q8_0}.gguf`): not under Temp, which the system clears.

## The rows

`frozen9-rows.jsonl` is built by `frozen_rows.py` (seed 2): the make-up of round 1 (8 DeepMind modules x 10, MATH-500 level 5 x 40)
with every question id found in the stored work removed. Round 1's `fresh-rows.jsonl` is not used for round 9: it holds all 24 loop rows
the closers (A, B, V7, `--shape-cap`, `--cover-cap`) were tuned on. `smoke-rows.jsonl` is the one-row smoke.

## Order

1. `sh scripts/a100/gate9.sh` (background). Tests, the replay of every stored arm with the round-9 flags, and the shipped archive run
   alone: on the 5060 Ti's llama-server if it is up, else an import check. Writes `$WORK/sim/gate9.stamp`. Re-run after any commit
   to `tools/`, `serve/` or `third_party/`.
2. On the 5060 Ti (preliminary): llama-server with the Q8_0 GGUF, `-c 262144 -np 4 -kvu` (each answer may use the whole context; the 4
   slots share one 262,144-token cache), then `tir4b.py run --server http://127.0.0.1:8099` with `PYTHONPATH` at the repository.
   Q8_0 weights and llama.cpp numerics are not the A100's bf16 vLLM: compare arms within the PC run only.
3. A100, when the account has the units: `COLAB_ACCOUNT=a sh scripts/a100/launch9.sh` (refuses under `MIN_CU`=20, without a PASS stamp
   for this exact code, or with uncommitted code), then in the background `COLAB_ACCOUNT=a sh scripts/a100/watch9.sh` and
   `sh scripts/a100/readers9.sh`. The watcher stops the VM itself when both arms are copied.
4. Read: rule score, the API reader (`readers-<arm>.log`, `triage.json`) and the corrected score per arm; every act on a right answer read;
   the result into `docs/` and the issue in the same turn.

The VM's stack is pinned (`setup_vm.py`: vllm 0.33.0, transformers 5.20.2) and checked by `run_vm9.sh`; each arm's folder gets
`setup.done`, `setup.freeze` and `provenance.txt` (rows and code hashes, the commit).
