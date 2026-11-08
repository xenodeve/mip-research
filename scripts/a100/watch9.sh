#!/bin/sh
# Silent watcher of round 9 (run_vm9.sh): copies each arm's records every 10 minutes and exits only on
#   DONE      run9.done on the VM and every arm copied with a non-empty report.json: the VM is then STOPPED (compute units)
#   ERROR     run9.done says ERROR, or a Traceback in a log (the VM is left up so the error can be read; stop it by hand)
#   UNREACH   the session did not answer 4 times running (about 40 minutes)
#   STALL     the GPU sat idle (under 5 %) for 3 checks while the run was alive (one answer can run >30 min under the full context)
#   DEAD      neither the queue nor run9.done
#   COLAB_ACCOUNT=a|b|c sh scripts/a100/watch9.sh      (in the background; log in $WORK/watch9.log)
K=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-C:/Users/xenod/mip-work/a100}
export PYTHONIOENCODING=utf-8
[ -n "$COLAB_ACCOUNT" ] || { echo "set COLAB_ACCOUNT"; exit 1; }
V="python $K/vm.py"
ARMS="final-gen final-ctrl"
for arm in $ARMS; do mkdir -p "$WORK/out/$arm"; done
miss=0; same=0
fetch() {
  for arm in $ARMS; do
    for f in records.json ids.jsonl report.json arm.json setup.done setup.freeze provenance.txt; do
      timeout 900 $V get /content/t4b9/out/$arm/$f "$WORK/out/$arm/$f" > /dev/null 2>&1
    done
  done
}
while true; do
  st=$(timeout 600 $V sh 'cd /content/t4b9; echo ALIVE; test -f run9.done && echo "RUNDONE $(cat run9.done)"; for l in finalgen finalctrl; do echo "Q $l $(tr "\r" "\n" < $l.log 2>/dev/null | grep -a "done, " | tail -1)"; done; echo "GPU $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)"; pgrep -f "run_vm[9][.]sh" > /dev/null && echo QUEUED; test -f /content/t4b/setup.done || echo SETUP; grep -l Traceback finalgen.log finalctrl.log smoke.log 2>/dev/null | sed "s/^/TB /"' 280 2>&1)
  case "$st" in
    *ALIVE*) miss=0 ;;
    *) miss=$((miss + 1)); echo "$(date +%T) no answer ($miss)" >> "$WORK/watch9.log"
       if [ $miss -ge 4 ]; then echo "UNREACH: no answer 4 times running"; exit 3; fi
       sleep 600; continue ;;
  esac
  prog=$(echo "$st" | grep -E "^ ?Q " | tr '\n' ' ')
  echo "$(date +%T) $prog" >> "$WORK/watch9.log"
  fetch
  case "$st" in *"RUNDONE ERROR"*) echo "ERROR: $(echo "$st" | grep RUNDONE) (VM left up)"; exit 2 ;; esac
  case "$st" in *RUNDONE*)
    fetch
    ok=1; for arm in $ARMS; do [ -s "$WORK/out/$arm/report.json" ] && [ -s "$WORK/out/$arm/records.json" ] || ok=0; done
    if [ $ok = 1 ]; then $V stop > /dev/null 2>&1; echo "DONE, VM stopped: $prog"; exit 0; fi
    echo "DONE on the VM but a copy is missing: VM left up, copy by hand then stop it"; exit 6 ;;
  esac
  case "$st" in *"TB "*) echo "ERROR: Traceback in a log $prog (VM left up)"; exit 2 ;; esac
  case "$st" in *QUEUED*) ;; *) echo "DEAD: no queue, no run9.done $prog"; exit 5 ;; esac
  gpu=$(echo "$st" | grep -E "^ ?GPU " | awk '{print $2}')
  case "$st" in *SETUP*) gpu=100 ;; esac   # the install and the model download leave the GPU idle for ~15 min
  if [ -n "$gpu" ] && [ "$gpu" -lt 5 ] 2>/dev/null; then same=$((same + 1)); else same=0; fi
  if [ $same -ge 3 ]; then echo "STALL: GPU idle for 3 checks: $prog"; exit 4; fi
  sleep 600
done
