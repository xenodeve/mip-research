#!/bin/sh
# A targeted live test: one arm, round 9's General flags + extra flags, on the rows a fix is meant to move (4B NVFP4 W4A8 · llama.cpp ·
# PC layer, full 262,144-token context). Compared against round 9's stored answers of the same rows and seed; the paired test follows.
# Then the API reader (this PC).
#   sh pc_target.sh N GPU PORT GGUF ROWS TAG "extra tir4b flags" [server args...]   (GPU = PCI index: 1 the 5060 Ti, 0 the 4070 Super)
N=${1:?np}; GPU=${2:?gpu}; PORT=${3:?port}; GGUF=${4:?gguf}; ROWS=${5:?rows}; TAG=${6:?tag}; EXTRA=${7?extra}; shift 7
W=C:/Users/xenod/mip-work/a100
G=C:/Users/xenod/mip-work
cd ${REPO:-/d/Github/mip-research} || exit 1
export PYTHONPATH=${REPO:-/d/Github/mip-research} PYTHONIOENCODING=utf-8
git diff --quiet HEAD -- tools serve || { echo "uncommitted code"; exit 1; }
mkdir -p $W/pc
git rev-parse HEAD > $W/pc/$TAG.commit
SRV=${SRV:-/c/AI/llama.cpp/build-w4a8/bin/llama-server.exe}
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin:$PATH"
echo "server $SRV $(sha256sum $SRV | cut -c1-16) gguf $GGUF gpu $GPU np $N args $* extra $EXTRA" > $W/pc/$TAG.setup
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $SRV -m $GGUF -ngl 999 \
  -c 262144 -np $N -kvu -fa on --no-cache-idle-slots --host 127.0.0.1 --port $PORT --no-webui "$@" > $G/llama-$TAG.log 2>&1 &
SP=$!
for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 2; done
curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok || { echo "server did not start"; kill $SP; exit 2; }
F="--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress --loop-close --answer-close --early-cycle --shape-cap 1024 --cover-cap 32768"
C="--model $G/q4b-hf --server http://127.0.0.1:$PORT --rows $W/$ROWS --samples 2 --seed 1 --ids --live $N ${RESUME:+--resume}"
python tools/igc/tir4b.py run $C --out $W/pc/$TAG $F $EXTRA > $W/pc-$TAG.log 2>&1 &
P1=$!
python -m tools.igc.answer_check --follow $W/pc/$TAG --rows $W/$ROWS \
  --env D:/Github/Strata/.worktrees/native-qwenputer-p0/.env > $W/readers-pc-$TAG.log 2>&1 &
wait $P1
kill $SP
wait
echo "$TAG $(cat $W/pc/$TAG/report.json)"; head -1 $W/readers-pc-$TAG.log
echo TARGET DONE
