#!/bin/sh
# Round 9 launch from this PC: refuse unless the account has the compute units, the replay gate ran on this exact tools/ tree, and the
# rows are there; then open the A100, upload setup + code (git archive of HEAD) + rows + queue, and start run_vm9.sh.
#   COLAB_ACCOUNT=a|b|c sh scripts/a100/launch9.sh [ROWS]
# Env: WORK (default C:/Users/xenod/mip-work/a100: rows, out/, sim/, logs; outside the public repo), MIN_CU (default 20: ~2 h at 8.88/h
# plus margin). After it: sh scripts/a100/watch9.sh and sh scripts/a100/readers9.sh, both in the background.
K=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$K/../.." && pwd)
WORK=${WORK:-C:/Users/xenod/mip-work/a100}
ROWS=${1:-frozen9-rows.jsonl}
MIN_CU=${MIN_CU:-20}
export PYTHONIOENCODING=utf-8
[ -n "$COLAB_ACCOUNT" ] || { echo "set COLAB_ACCOUNT=a|b|c"; exit 1; }
[ -f "$WORK/$ROWS" ] && [ -f "$WORK/smoke-rows.jsonl" ] || { echo "rows missing in $WORK"; exit 1; }
cd "$R" || exit 1
git diff --quiet HEAD -- tools serve third_party || { echo "tools/ serve/ third_party/ have uncommitted changes: commit them and re-run the gate"; exit 1; }
TREE=$(git rev-parse HEAD:tools)-$(git rev-parse HEAD:serve | cut -c1-12)   # the same key gate9.sh stamps
STAMP="$WORK/sim/gate9.stamp"
grep -q "^tree $TREE" "$STAMP" 2>/dev/null || { echo "the replay gate did not run on tools/ $TREE: sh scripts/a100/gate9.sh"; exit 1; }
grep -q "^verdict PASS" "$STAMP" || { echo "the gate did not pass: $(cat "$STAMP")"; exit 1; }
CU=$(python "$K/vm.py" usage | sed -n 's/^CU //p' | tail -1)
echo "account $COLAB_ACCOUNT: $CU compute units (need $MIN_CU)"
python -c "import sys; sys.exit(0 if float('$CU' or 0) >= $MIN_CU else 1)" 2>/dev/null || { echo "not enough compute units"; exit 2; }
git archive --format=tar.gz -o "$WORK/code9.tgz" HEAD tools serve third_party || { echo "git archive failed"; exit 1; }
git rev-parse HEAD > "$WORK/code9.commit"
python "$K/vm.py" new || exit 3
python "$K/vm.py" sh 'mkdir -p /content/t4b /content/t4b9/out' 120
for f in "$K/setup_vm.py:t4b/setup_vm.py" "$WORK/code9.tgz:t4b9/code.tgz" "$WORK/code9.commit:t4b9/code.commit" "$WORK/$ROWS:t4b9/$ROWS" \
         "$WORK/smoke-rows.jsonl:t4b9/smoke-rows.jsonl" "$K/run_vm9.sh:t4b9/run_vm9.sh"; do
  python "$K/vm.py" up "${f%%:*}" "/content/${f##*:}" || { echo "upload failed $f"; exit 4; }
done
python "$K/vm.py" bg "bash /content/t4b9/run_vm9.sh $ROWS" /content/t4b9/run9.log
python "$K/vm.py" sh 'sleep 60; pgrep -f "run_vm[9][.]sh" && echo QUEUE RUNNING; tail -3 /content/t4b/setup.log' 180
echo "$COLAB_ACCOUNT $ROWS $(date +%F_%T)" > "$WORK/round9.launched"
