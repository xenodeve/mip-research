#!/bin/sh
# Round 9's gate on this exact shipped code, from this PC: the tests, every stored A100 arm replayed with the full round-9 flag set, and
# a smoke run of the archive the VM gets (git archive HEAD tools serve third_party) on the 5060 Ti's llama-server, if one answers on
# 127.0.0.1:8099. The verdict goes into $WORK/sim/gate9.stamp, which launch9.sh checks.
# PASS = tests green, no act on any answer that finished right, vllm_refused 0, the archive's smoke wrote its report (or no server:
# then the archive must at least import everything tir4b's run() imports). Re-run after every commit that touches tools/ or serve/.
#   sh scripts/a100/gate9.sh          (in the background; ~10-20 min, most of it the one 262k-token answer)
K=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$K/../.." && pwd)
WORK=${WORK:-C:/Users/xenod/mip-work/a100}
export WORK PYTHONIOENCODING=utf-8
cd "$R" || exit 1
git diff --quiet HEAD -- tools serve third_party || { echo "tools/ serve/ third_party/ have uncommitted changes: commit first"; exit 1; }
TREE=$(git rev-parse HEAD:tools)-$(git rev-parse HEAD:serve | cut -c1-12)
NAME=r9-$(echo "$TREE" | cut -c1-8)
MECHS=inline_calc,mathbackend,inline_plain,inline_style=general,no_digit_join,guard_progress,loop_close,answer_close,early_cycle,shape_cap=1024,cover_cap=32768
FLAGS="--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress --loop-close --answer-close --early-cycle --shape-cap 1024 --cover-cap 32768"
mkdir -p "$WORK/sim"
python tools/test_igc_tir4b.py > "$WORK/sim/$NAME-tests.log" 2>&1; t1=$?
python tools/test_sim_replay.py > "$WORK/sim/$NAME-tests-sim.log" 2>&1; t2=$?

# the archive the VM gets: extracted alone, so a module missing from the repository fails here and not on the A100
A="$WORK/arch-$NAME"; rm -rf "$A"; mkdir -p "$A"
git archive --format=tar HEAD tools serve third_party | tar x -C "$A"
if curl -s -m 5 http://127.0.0.1:8099/health | grep -q ok; then
  rm -rf "$WORK/pc/gate-smoke-$NAME"
  (cd "$A" && PYTHONPATH="$A" python tools/igc/tir4b.py run --model C:/Users/xenod/mip-work/q4b-hf --server http://127.0.0.1:8099 \
     --rows "$WORK/smoke-rows.jsonl" --samples 1 --seed 1 --ids --live 4 --out "$WORK/pc/gate-smoke-$NAME" $FLAGS) > "$WORK/sim/$NAME-smoke.log" 2>&1
  [ -s "$WORK/pc/gate-smoke-$NAME/report.json" ] && ! grep -q Traceback "$WORK/sim/$NAME-smoke.log"; t3=$?; SMOKE="pc-smoke=$t3"
else
  (cd "$A" && PYTHONPATH="$A" python -c "import tools.igc.tir4b, tools.igc.select_rows, serve.sandbox.exec_bridge, serve.reasoning_tool") \
     > "$WORK/sim/$NAME-smoke.log" 2>&1; t3=$?; SMOKE="imports-only=$t3"
fi

ARMS=$(ls "$WORK/out")
python scripts/sim_gate.py "$NAME" "$MECHS" $ARMS > "$WORK/sim/$NAME-run.log" 2>&1
python scripts/gate_read.py "$NAME" > "$WORK/sim/$NAME-read.txt" 2>&1
TOTAL=$(grep '^TOTAL' "$WORK/sim/$NAME-read.txt")
FAILED=$(grep -c FAILED "$WORK/sim/$NAME-run.log")
VERDICT=$(python -c "
import ast, sys
t = ast.literal_eval(sys.argv[1].split(' ', 1)[1]) if sys.argv[1] else {}
ok = $t1 == 0 and $t2 == 0 and $t3 == 0 and $FAILED == 0 and bool(t) and t.get('right acted', 0) == 0 and t.get('refused', 0) == 0
print('PASS' if ok else 'FAIL')" "$TOTAL")
{ echo "tree $TREE"; echo "commit $(git rev-parse HEAD)"; echo "verdict $VERDICT"; echo "tests tir4b=$t1 sim_replay=$t2 $SMOKE failed_arms=$FAILED";
  echo "$TOTAL"; echo "date $(date +%F_%T)"; } > "$WORK/sim/gate9.stamp"
cat "$WORK/sim/gate9.stamp"
