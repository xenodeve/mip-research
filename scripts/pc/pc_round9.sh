#!/bin/sh
# Round 9 on the 5060 Ti (preliminary; the A100 bf16 run confirms): the full General Style + closers against control, frozen9 rows x 2,
# seed 1, full 262,144-token context, both arms at once on one llama-server (4B Q8_0, -c 262144 -kvu -np N). No rule or threshold is
# changed from this run's results. Then the API reader on both arms (this PC).
#   sh pc_round9.sh N GPU PORT GGUF ROWS TAG [server args...]   (GPU = PCI index: 1 the 5060 Ti, 0 the 4070 Super)
N=${1:?np}; GPU=${2:?gpu}; PORT=${3:?port}; GGUF=${4:?gguf}; ROWS=${5:?rows}; TAG=${6:?tag}; shift 6
W=C:/Users/xenod/mip-work/a100
G=C:/Users/xenod/mip-work
cd /d/Github/mip-research || exit 1
export PYTHONPATH=/d/Github/mip-research PYTHONIOENCODING=utf-8
git diff --quiet HEAD -- tools serve || { echo "uncommitted code"; exit 1; }
git rev-parse HEAD > $W/pc/r9$TAG.commit
# SRV: the sm_89-only build, so NVFP4 multiplies with q8_1 activations (W4A8) on both cards (the sm_120 build is W4A4 on the 5060 Ti)
SRV=${SRV:-/c/AI/llama.cpp/build-w4a8/bin/llama-server.exe}
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin:$PATH"   # cudart/cublas for the new build
echo "server $SRV $(sha256sum $SRV | cut -c1-16) gguf $GGUF gpu $GPU np $N args $*" > $W/pc/r9$TAG.setup
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU $SRV -m $GGUF -ngl 999 \
  -c 262144 -np $N -kvu -fa on --host 127.0.0.1 --port $PORT --no-webui "$@" > $G/llama-r9$TAG.log 2>&1 &
SP=$!
for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 2; done
curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok || { echo "server did not start"; exit 2; }
F="--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress --loop-close --answer-close --early-cycle --shape-cap 1024 --cover-cap 32768"
C="--model $G/q4b-hf --server http://127.0.0.1:$PORT --rows $W/$ROWS --samples 2 --seed 1 --ids --live $((N / 2)) ${RESUME:+--resume}"
python tools/igc/tir4b.py run $C --out $W/pc/r9$TAG-gen $F > $W/pc-r9$TAG-gen.log 2>&1 &
P1=$!
python tools/igc/tir4b.py run $C --out $W/pc/r9$TAG-ctrl > $W/pc-r9$TAG-ctrl.log 2>&1 &
P2=$!
for arm in r9$TAG-gen r9$TAG-ctrl; do
  python -m tools.igc.answer_check --follow $W/pc/$arm --rows $W/$ROWS \
    --env D:/Github/Strata/.worktrees/native-qwenputer-p0/.env > $W/readers-pc-$arm.log 2>&1 &
done
wait $P1 $P2
kill $SP
wait
for a in r9$TAG-gen r9$TAG-ctrl; do echo "$a $(cat $W/pc/$a/report.json)"; head -1 $W/readers-pc-$a.log; done
echo R9 PC DONE
