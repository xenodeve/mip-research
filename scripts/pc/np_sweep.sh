#!/bin/sh
# Concurrency sweep on the PC layer's locked config (4B NVFP4 W4A8 · llama.cpp · PC layer: build-w4a8, -c 262144 -kvu -fa on,
# KV q8_0, --no-cache-idle-slots): llama-server at -np N with N answers in flight (General N/2 + control N/2), 9 dev rows x 4 samples,
# 10 minutes each. tok/s is read from the server's own log (minutes 2-10), not from wall time, which the longest answer sets.
# The 2026-11-08 sweep (np 4 76 tok/s, np 8 64) ran on Q8_0 with idle slots cleared on every new task (AGENTS.md) and does not hold.
#   GPU=1 PORT=8099 sh np_sweep.sh 4 6 8     (GPU = PCI index: 1 the 5060 Ti, 0 the 4070 Super)
W=C:/Users/xenod/mip-work/a100
G=C:/Users/xenod/mip-work
GPU=${GPU:-1}; PORT=${PORT:-8099}
SRV=${SRV:-/c/AI/llama.cpp/build-w4a8/bin/llama-server.exe}
GGUF=${GGUF:-$G/gguf/qwen35-4b-nvfp4-axion.gguf}
cd /d/Github/mip-research || exit 1
export PYTHONPATH=/d/Github/mip-research PYTHONIOENCODING=utf-8
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin:$PATH"
F="--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress --loop-close --answer-close --early-cycle --shape-cap 1024 --cover-cap 32768"
for N in "$@"; do
  LOG=$G/llama-np$N-gpu$GPU.log
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $SRV -m $GGUF -ngl 999 -c 262144 -np $N -kvu -fa on --no-cache-idle-slots \
    -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port $PORT --no-webui > $LOG 2>&1 &
  SP=$!
  for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 2; done
  curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok || { echo "np$N: server did not start: $(tail -3 $LOG)"; kill $SP; continue; }
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
  L=$((N / 2))
  C="--model $G/q4b-hf --server http://127.0.0.1:$PORT --rows $W/pc9-dev-rows.jsonl --samples 4 --seed 1 --ids --live $L"
  rm -rf $W/pc/np$N-gpu$GPU-gen $W/pc/np$N-gpu$GPU-ctrl
  timeout 600 python tools/igc/tir4b.py run $C --out $W/pc/np$N-gpu$GPU-gen $F > $W/pc-np$N-gpu$GPU-gen.log 2>&1 &
  P1=$!
  timeout 600 python tools/igc/tir4b.py run $C --out $W/pc/np$N-gpu$GPU-ctrl > $W/pc-np$N-gpu$GPU-ctrl.log 2>&1 &
  P2=$!
  wait $P1 $P2
  python - "$LOG" "$N" "$MEM" "$GPU" <<'EOF'
import re, sys
log, n, mem, gpu = sys.argv[1:]
ev = []
lcp = lru = 0
for l in open(log, encoding='utf-8', errors='replace'):
    lcp += 'selected slot by LCP' in l
    lru += 'selected slot by LRU' in l
    m = re.match(r'(\d+)\.(\d+)\.(\d+)\.\d+ I slot print_timing.*?\|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens', l)
    if m:
        ev.append((int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1000, int(m.group(5)), float(m.group(4))))
t0 = ev[0][0]
s = [e for e in ev if t0 + 120 <= e[0] <= t0 + 600]
per = sum(e[1] for e in s) / sum(e[2] for e in s) * 1000 if s else 0
full = sum(1 for l in open(log, encoding='utf-8', errors='replace') if re.search(r'purging slot|failed to find|no space|kv cache full', l, re.I))
print(f'gpu{gpu} np{n}: {sum(e[1] for e in s) / 480:.0f} tok/s total, per slot {per:.1f} tok/s, requests {len(s)}, '
      f'LCP {lcp} LRU {lru}, VRAM {mem} MiB, KV-full lines {full}', flush=True)
EOF
  kill $SP; sleep 5
done
echo SWEEP DONE
