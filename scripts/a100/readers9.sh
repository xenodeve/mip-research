#!/bin/sh
# The API reader beside each round-9 arm, on this PC only (the key never goes to a VM): reads the records watch9.sh copies into
# $WORK/out/<arm>/ as they arrive and ends when that arm's report.json is there. Read each arm's triage.json and readers-<arm>.log.
#   sh scripts/a100/readers9.sh [ROWS]      (in the background)
K=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$K/../.." && pwd)
WORK=${WORK:-C:/Users/xenod/mip-work/a100}
ROWS=${1:-frozen9-rows.jsonl}
ENV=${ANSWER_CHECK_ENV:-D:/Github/Strata/.worktrees/native-qwenputer-p0/.env}
cd "$R" || exit 1
export PYTHONIOENCODING=utf-8
for arm in final-gen final-ctrl; do
  mkdir -p "$WORK/out/$arm"
  python -m tools.igc.answer_check --follow "$WORK/out/$arm" --rows "$WORK/$ROWS" --env "$ENV" > "$WORK/readers-$arm.log" 2>&1 &
done
wait
for arm in final-gen final-ctrl; do head -1 "$WORK/readers-$arm.log"; done
echo READERS DONE
