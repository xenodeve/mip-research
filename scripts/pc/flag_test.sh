#!/bin/sh
# Server-flag speed test on the PC layer's locked config (4B NVFP4 W4A8 · llama.cpp · PC layer): one llama-server per config, -np NP,
# NP answers in flight (General NP/2 + control NP/2) on 9 dev rows x 4 samples, 8 minutes each; tok/s from the server log (minutes 2-8).
# A config with --spec-type first passes scripts/pc/spec_probe.py (the reply's tokens stop where tir4b's stop words say); one that
# fails is skipped. Every config is one line of CONFIGS below: a label, then the extra server flags.
#   GPU=0 PORT=8100 NP=4 sh flag_test.sh
W=C:/Users/xenod/mip-work/a100
G=C:/Users/xenod/mip-work
GPU=${GPU:-0}; PORT=${PORT:-8100}; NP=${NP:-4}; MIN=${MIN:-8}
SRV=${SRV:-/c/AI/llama.cpp/build-w4a8/bin/llama-server.exe}
GGUF=${GGUF:-$G/gguf/qwen35-4b-nvfp4-axion.gguf}
cd /d/Github/mip-research || exit 1
export PYTHONPATH=/d/Github/mip-research PYTHONIOENCODING=utf-8
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin:$PATH"
F="--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress --loop-close --answer-close --early-cycle --shape-cap 1024 --cover-cap 32768"
CONFIGS=${CONFIGS:-"base|
ram-prio|-cram 0 --prio 2
backend-sampling|-cram 0 --prio 2 --backend-sampling
mtp2|-cram 0 --prio 2 --spec-type draft-mtp --spec-draft-n-max 2
mtp-ngram|-cram 0 --prio 2 --spec-type draft-mtp,ngram-mod --spec-draft-n-max 2"}
echo "$CONFIGS" | while IFS='|' read -r NAME EXTRA; do
  [ -z "$NAME" ] && continue
  TAG=flag-$NAME-gpu$GPU
  LOG=$G/llama-$TAG.log
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $SRV -m $GGUF -ngl 999 -c 262144 -np $NP -kvu -fa on --no-cache-idle-slots \
    -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port $PORT --no-webui $EXTRA > $LOG 2>&1 &
  SP=$!
  for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 2; done
  curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok || { echo "$NAME: server did not start: $(tail -3 $LOG)"; kill $SP; sleep 5; continue; }
  case "$EXTRA" in *spec-type*)
    if ! python scripts/pc/spec_probe.py --server http://127.0.0.1:$PORT --n 8 > $W/$TAG-probe.txt 2>&1; then
      echo "$NAME: spec_probe failed: $(tail -2 $W/$TAG-probe.txt)"; kill $SP; sleep 5; continue
    fi
    echo "$NAME: probe $(tail -1 $W/$TAG-probe.txt)";;
  esac
  L=$((NP / 2))
  C="--model $G/q4b-hf --server http://127.0.0.1:$PORT --rows $W/pc9-dev-rows.jsonl --samples 4 --seed 1 --ids --live $L"
  rm -rf $W/pc/$TAG-gen $W/pc/$TAG-ctrl
  T0=$(grep -c "eval time" $LOG)
  timeout $((MIN * 60)) python tools/igc/tir4b.py run $C --out $W/pc/$TAG-gen $F > $W/pc-$TAG-gen.log 2>&1 &
  P1=$!
  timeout $((MIN * 60)) python tools/igc/tir4b.py run $C --out $W/pc/$TAG-ctrl > $W/pc-$TAG-ctrl.log 2>&1 &
  P2=$!
  wait $P1 $P2
  python - "$LOG" "$NAME" "$MIN" "$T0" <<'EOF'
import re, sys
log, name, minutes, skip = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ev, lines = [], open(log, encoding='utf-8', errors='replace').read().splitlines()
n = 0
for l in lines:
    m = re.match(r'(\d+)\.(\d+)\.(\d+)\.\d+ I slot print_timing.*?\|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens', l)
    if m:
        n += 1
        if n > skip:   # the probe's requests are not the test
            ev.append((int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1000, int(m.group(5)), float(m.group(4))))
t0 = ev[0][0]
s = [e for e in ev if t0 + 120 <= e[0] <= t0 + 60 * minutes]
span = 60 * minutes - 120
per = sum(e[1] for e in s) / sum(e[2] for e in s) * 1000 if s else 0
acc = [l for l in lines if 'draft acceptance' in l]
print(f'{name}: {sum(e[1] for e in s) / span:.0f} tok/s total, per slot {per:.1f} tok/s, requests {len(s)}, '
      f'LCP {sum("by LCP" in l for l in lines)} LRU {sum("by LRU" in l for l in lines)}'
      f'{", " + acc[-1].split("|")[-1].strip() if acc else ""}', flush=True)
EOF
  kill $SP; sleep 5
done
echo FLAGS DONE
