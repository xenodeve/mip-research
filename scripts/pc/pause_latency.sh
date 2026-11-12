#!/bin/sh
# The pause of an inline put, end to end, with the arithmetic on Python (Fraction) and on the CUDAMathBackend: one llama-server on the
# chosen card (4B NVFP4 W4A8 · llama.cpp build-w4a8 sm_89 · PC layer), two tir4b runs on it at once with the same General flags, rows and
# seeds; arm `py` computes in Python, arm `cuda` sends every + - * / of mathexpr to the CUDAMathBackend DLL (mode 2, the same card, another
# process from llama-server). Both write one line per request (TIR4B_TIMING); pause_latency.py splits each pause.
#   [GPU=1] sh scripts/pc/pause_latency.sh ROWS TAG [live per arm, default 6]   (GPU = PCI index: 1 the RTX 5060 Ti, 0 the RTX 4070 SUPER)
ROWS=${1:?rows}; TAG=${2:?tag}; LIVE=${3:-6}; GPU=${GPU:-1}
W=C:/Users/xenod/mip-work/a100
G=C:/Users/xenod/mip-work
cd ${REPO:-/d/Github/mip-research} || exit 1
export PYTHONPATH=${REPO:-/d/Github/mip-research} PYTHONIOENCODING=utf-8
git diff --quiet HEAD -- tools || { echo "uncommitted code"; exit 1; }
mkdir -p $W/pc
OUT=$W/pc/pause-$TAG
mkdir -p $OUT
git rev-parse HEAD > $OUT/commit
SRV=${SRV:-/c/AI/llama.cpp/build-w4a8/bin/llama-server.exe}
GGUF=${GGUF:-$G/gguf/qwen35-4b-nvfp4-axion.gguf}
PORT=${PORT:-8093}
NP=$((LIVE * 2 + 1))
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin:$PATH"
echo "server $SRV $(sha256sum $SRV | cut -c1-16) gguf $GGUF $(sha256sum $GGUF | cut -c1-16) gpu $GPU ($(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU)) np $NP" > $OUT/setup
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $SRV -m $GGUF -ngl 999 \
  -c 262144 -np $NP -kvu -fa on --no-cache-idle-slots -cram 0 --prio 2 --host 127.0.0.1 --port $PORT --no-webui > $OUT/llama.log 2>&1 &
SP=$!
for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 2; done
curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok || { echo "server did not start"; kill $SP; exit 2; }
F="--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress --loop-close --answer-close --early-cycle --shape-cap 1024 --cover-cap 32768"
C="--model $G/q4b-hf --server http://127.0.0.1:$PORT --rows $ROWS --samples 2 --seed 1 --ids --live $LIVE"
TIR4B_TIMING=$OUT/py.timing.jsonl python tools/igc/tir4b.py run $C --out $OUT/py $F > $OUT/py.log 2>&1 &
P1=$!
TIR4B_TIMING=$OUT/cuda.timing.jsonl MIP_EXPR_DEVICE=cuda MIP_CUDA_RESIDENT=2 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU MIP_CUDA_DEVICE=0 \
  python tools/igc/tir4b.py run $C --out $OUT/cuda $F > $OUT/cuda.log 2>&1 &
P2=$!
wait $P1 $P2
kill $SP
wait
python results/cuda-mathbackend/pause_latency.py $OUT > $OUT/pause.txt
cat $OUT/pause.txt
echo PAUSE DONE
