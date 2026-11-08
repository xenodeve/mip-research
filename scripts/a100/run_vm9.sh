#!/bin/bash
# Round 9 on the VM: the whole General Style (inline values + loop close + answer close + early cycle + shape cap + cover cap), no profile,
# against a plain control in the same round, ROWS x 2 samples, seed 1, the whole 262,144-token context (never --max-tokens).
# Usage: bash run_vm9.sh [ROWS]   (default frozen9-rows.jsonl: questions no rule or threshold was tuned on; see scripts/a100/README.md)
# Setup in /content/t4b unless setup.done is there; code in /content/t4b9. Tests, then a 1-row smoke with every flag. Writes run9.done.
ROWS=${1:-frozen9-rows.jsonl}
VLLM_PIN="vllm 0.33.0"
mkdir -p /content/t4b && cd /content/t4b || exit 1
[ -f setup.done ] || python setup_vm.py
# setup.done is opened (empty) before the versions are written into it: wait for its text, not for the file
while ! grep -q "vllm\|Error" setup.done 2>/dev/null; do sleep 30; done
grep -q "$VLLM_PIN" setup.done || { echo "ERROR setup: want $VLLM_PIN, got $(head -c 300 setup.done) $(tail -3 setup.log)" > /content/t4b9/run9.done; exit 2; }
mkdir -p /content/t4b9 && cd /content/t4b9 || exit 1
[ -f "$ROWS" ] || { echo "ERROR rows: $ROWS missing" > run9.done; exit 2; }
tar xzf code.tgz || { echo "ERROR untar" > run9.done; exit 2; }
export PYTHONPATH=/content/t4b9
python tools/test_igc_tir4b.py > tests.log 2>&1 || { echo "ERROR tests: $(tail -3 tests.log)" > run9.done; exit 2; }
T=/content/t4b9/tools/igc/tir4b.py
F="--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress --loop-close --answer-close --early-cycle --shape-cap 1024 --cover-cap 32768"
python $T run --model /content/q4b --rows smoke-rows.jsonl --samples 1 --seed 1 --ids --memory 0.45 --out out/smoke $F > smoke.log 2>&1
grep -q Traceback smoke.log && { echo "ERROR smoke: $(grep -a -m1 -E 'Error|refuse' smoke.log)" > run9.done; exit 2; }
echo smoke-ok > smoke.done
C="--model /content/q4b --rows $ROWS --samples 2 --seed 1 --ids"   # no --max-tokens (developer, 2026-11-06)
# both arms at once, 45 % of the A100 each: a round's tail is one long answer and the GPU otherwise serves it alone (round 8: the last
# answer ran ~1 h by itself)
python $T run $C --memory 0.45 --out out/final-gen $F > finalgen.log 2>&1 &
sleep 240   # let the first engine take its memory before the second starts
python $T run $C --memory 0.45 --out out/final-ctrl > finalctrl.log 2>&1 &
wait
# provenance beside the records: the stack, the code and the rows each arm ran on
for arm in final-gen final-ctrl; do
  cp /content/t4b/setup.done /content/t4b/setup.freeze out/$arm/ 2>/dev/null
  { echo "rows $ROWS $(sha256sum $ROWS | cut -c1-16)"; echo "code $(sha256sum code.tgz | cut -c1-16) $(cat code.commit 2>/dev/null)"; } > out/$arm/provenance.txt
done
grep -l Traceback final*.log > /dev/null 2>&1 && echo "ERROR Traceback in $(grep -l Traceback final*.log | tr '\n' ' ')" > run9.done || echo done > run9.done
