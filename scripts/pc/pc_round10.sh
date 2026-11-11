#!/bin/sh
# Round 10 on the PC layer (development rows): round 9's General flags (arm A) against them plus the round 9 fixes (arm B: --exact-answer,
# --approx-put, --answer-rewind 2, --letter-count), and on the MATH rows only arm C = B + the Sandbox in the AppContainer box (--execute
# --box --loop-hint --verify-gate; the developer, 2026-11-11: test the Sandbox on the PC before the A100). All arms on one llama-server per
# card, the card's half of the rows x 2 seeds, full 262,144 context. Then the API reader on every arm (this PC).
#   sh pc_round10.sh N GPU PORT GGUF ROWS ROWS_C TAG [server args...]   (GPU = PCI index: 1 the 5060 Ti, 0 the 4070 Super)
N=${1:?np}; GPU=${2:?gpu}; PORT=${3:?port}; GGUF=${4:?gguf}; ROWS=${5:?rows}; ROWS_C=${6:?rows_c}; TAG=${7:?tag}; shift 7
W=C:/Users/xenod/mip-work/a100
G=C:/Users/xenod/mip-work
cd ${REPO:-/d/Github/mip-research} || exit 1
export PYTHONPATH=${REPO:-/d/Github/mip-research} PYTHONIOENCODING=utf-8
git diff --quiet HEAD -- tools serve || { echo "uncommitted code"; exit 1; }
mkdir -p $W/pc
git rev-parse HEAD > $W/pc/r10$TAG.commit
SRV=${SRV:-/c/AI/llama.cpp/build-w4a8/bin/llama-server.exe}
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin:$PATH"
echo "server $SRV $(sha256sum $SRV | cut -c1-16) gguf $GGUF gpu $GPU np $N args $*" > $W/pc/r10$TAG.setup
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $SRV -m $GGUF -ngl 999 \
  -c 262144 -np $N -kvu -fa on --host 127.0.0.1 --port $PORT --no-webui "$@" > $G/llama-r10$TAG.log 2>&1 &
SP=$!
for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 2; done
curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok || { echo "server did not start"; kill $SP; exit 2; }
F="--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress --loop-close --answer-close --early-cycle --shape-cap 1024 --cover-cap 32768"
FIX="--exact-answer --approx-put --answer-rewind 2 --letter-count"
BOX="--execute --box --loop-hint --verify-gate"
C="--model $G/q4b-hf --server http://127.0.0.1:$PORT --rows $W/$ROWS --samples 2 --seed 1 --ids --live $((N / 2)) ${RESUME:+--resume}"
python tools/igc/tir4b.py run $C --out $W/pc/r10$TAG-a $F > $W/pc-r10$TAG-a.log 2>&1 &
P1=$!
python tools/igc/tir4b.py run $C --out $W/pc/r10$TAG-b $F $FIX > $W/pc-r10$TAG-b.log 2>&1 &
P2=$!
CC="--model $G/q4b-hf --server http://127.0.0.1:$PORT --rows $W/$ROWS_C --samples 2 --seed 1 --ids --live 1 ${RESUME:+--resume}"
python tools/igc/tir4b.py run $CC --out $W/pc/r10$TAG-c $F $FIX $BOX > $W/pc-r10$TAG-c.log 2>&1 &
P3=$!
for arm in r10$TAG-a r10$TAG-b; do
  python -m tools.igc.answer_check --follow $W/pc/$arm --rows $W/$ROWS \
    --env D:/Github/Strata/.worktrees/native-qwenputer-p0/.env > $W/readers-pc-$arm.log 2>&1 &
done
python -m tools.igc.answer_check --follow $W/pc/r10$TAG-c --rows $W/$ROWS_C \
  --env D:/Github/Strata/.worktrees/native-qwenputer-p0/.env > $W/readers-pc-r10$TAG-c.log 2>&1 &
wait $P1 $P2 $P3
kill $SP
wait
for a in r10$TAG-a r10$TAG-b r10$TAG-c; do echo "$a $(cat $W/pc/$a/report.json)"; head -1 $W/readers-pc-$a.log; done
echo R10 PC DONE
