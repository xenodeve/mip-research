#!/bin/sh
# 5060 Ti concurrency sweep: llama-server (4B Q8_0, -c 262144 -kvu) at -np N, both round-9 arms on 9 dev rows x 4 samples, 10 minutes each.
# tok/s is read from the server's own log (minutes 2-10), not from wall time, which the longest answer sets.
#   sh np_sweep.sh 8 16 32
W=C:/Users/xenod/mip-work/a100
G=C:/Users/xenod/mip-work
SRV=/c/AI/llama.cpp-blackwell/llama-server.exe
cd /d/Github/mip-research || exit 1
export PYTHONPATH=/d/Github/mip-research PYTHONIOENCODING=utf-8
F="--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress --loop-close --answer-close --early-cycle --shape-cap 1024 --cover-cap 32768"
for N in "$@"; do
  LOG=$G/llama-np$N.log
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 $SRV -m $G/gguf/qwen35-4b-q8_0.gguf -ngl 999 -c 262144 -np $N -kvu -fa on \
    --host 127.0.0.1 --port 8099 --no-webui > $LOG 2>&1 &
  SP=$!
  for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:8099/health | grep -q ok && break; sleep 2; done
  curl -s -m 2 http://127.0.0.1:8099/health | grep -q ok || { echo "np$N: server did not start: $(tail -3 $LOG)"; kill $SP; continue; }
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
  L=$((N / 2))
  C="--model $G/q4b-hf --server http://127.0.0.1:8099 --rows $W/pc9-dev-rows.jsonl --samples 4 --seed 1 --ids --live $L"
  rm -rf $W/pc/np$N-gen $W/pc/np$N-ctrl
  timeout 600 python tools/igc/tir4b.py run $C --out $W/pc/np$N-gen $F > $W/pc-np$N-gen.log 2>&1 &
  P1=$!
  timeout 600 python tools/igc/tir4b.py run $C --out $W/pc/np$N-ctrl > $W/pc-np$N-ctrl.log 2>&1 &
  P2=$!
  wait $P1 $P2
  python - "$LOG" "$N" "$MEM" <<'EOF'
import re, sys
log, n, mem = sys.argv[1:]
ev = []
for l in open(log, encoding='utf-8', errors='replace'):
    m = re.match(r'(\d+)\.(\d+)\.(\d+)\.\d+ I slot print_timing.*?\|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens', l)
    if m:
        ev.append((int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1000, int(m.group(5)), float(m.group(4))))
t0 = ev[0][0]
s = [e for e in ev if t0 + 120 <= e[0] <= t0 + 600]
per = sorted(1000 * e[1] / e[2] for e in s if e[1] >= 50)
full = sum(1 for l in open(log, encoding='utf-8', errors='replace') if re.search(r'failed to find|no space|kv cache full', l, re.I))
print(f'np{n}: {sum(e[1] for e in s) / 480:.0f} tok/s total, per answer median {per[len(per) // 2] if per else 0:.1f}, requests {len(s)}, '
      f'VRAM {mem} MiB, KV-full lines {full}')
EOF
  kill $SP; sleep 5
done
echo SWEEP DONE
