#!/bin/sh
# Round 10 on the PC layer (development rows, docs/plans/2026-11-10-round9-fixes.md), all on the RTX 5060 Ti (the 4070 SUPER is the
# developer's; it is not touched). One patched llama-server (build-w4a8-mip, 4B NVFP4 W4A8 · llama.cpp+MIP · PC layer), three tir4b arms
# on it at once with the same rows and seeds:
#   a  = round 9's General flags exactly (the developer, 2026-11-13: arm A compares with round 9 as it ran; no --think-only, no long value)
#   b  = a + --think-only + the round 9 fixes + --question-line + --long-value, values put as text
#   ch = b with the values steered through the token channel (--channel): nothing inserted
# Requests from a and b carry no mip_id, so the shim is off for them (gate 1: the patched server reproduces the stock tokens, 6 / 6).
# Every arm computes on the CUDAMathBackend (MIP_EXPR_DEVICE=cuda, resident on the same card). Then gate 3 on ch and the API reader on all.
#   sh scripts/pc/pc_round10.sh ROWS TAG [live per arm, default 5] [extra tir4b args, e.g. --limit 1 --samples 1 for a smoke]
ROWS=${1:?rows}; TAG=${2:?tag}; LIVE=${3:-5}; shift 3 2>/dev/null; MORE="$*"
W=C:/Users/xenod/mip-work/a100
G=C:/Users/xenod/mip-work
cd ${REPO:-/d/Github/mip-research} || exit 1
export PYTHONPATH=${REPO:-/d/Github/mip-research} PYTHONIOENCODING=utf-8
git diff --quiet HEAD -- tools serve scripts/pc || { echo "uncommitted code"; exit 1; }
OUT=$W/pc/r10-$TAG
mkdir -p $OUT
git rev-parse HEAD > $OUT/commit
SRV=C:/AI/llama.cpp/build-w4a8-mip/bin/llama-server.exe
GGUF=$G/gguf/qwen35-4b-nvfp4-axion.gguf
PORT=8093; MIP_PORT=8095
# every answer in flight has a slot of its own and idle slots are never cleared (--no-cache-idle-slots, AGENTS.md "The PC layer")
NP=$((LIVE * 3 + 1))
# under -kvu, -c is ONE KV pool shared by every slot, not a window per answer: every answer in flight must fit in it together. tir4b
# caps each answer at 262,144 itself (CONTEXT). miss1 overflowed a pool of 262,144 with 12 answers in flight; ~17 KB a token at q8_0.
CTX=${CTX:-524288}
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin:$PATH"
echo "server $SRV $(sha256sum $SRV | cut -c1-16) gguf $GGUF gpu 1 (RTX 5060 Ti) np $NP args -c $CTX -kvu -fa on -ctk q8_0 -ctv q8_0 --no-cache-idle-slots -cram 0 --prio 2; MIP_CHANNEL 127.0.0.1:$MIP_PORT; llama.cpp $(git -C C:/AI/llama.cpp rev-parse --short HEAD)" > $OUT/setup
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MIP_CHANNEL=127.0.0.1:$MIP_PORT $SRV -m $GGUF -ngl 999 -c $CTX -np $NP -kvu -fa on \
  -ctk q8_0 -ctv q8_0 --no-cache-idle-slots -cram 0 --prio 2 --host 127.0.0.1 --port $PORT --no-webui > $OUT/llama.log 2>&1 &
SP=$!
for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 2; done
curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok || { echo "server did not start"; kill $SP; exit 2; }
F="--inline-calc --mathbackend --inline-plain --inline-style general --no-digit-join --guard-progress --loop-close --answer-close --early-cycle --shape-cap 1024 --cover-cap 32768"
FIX="--think-only --question-quote --approx-put --answer-rewind 2 --letter-count --question-line --long-value"
C="--model $G/q4b-hf --server http://127.0.0.1:$PORT --rows $ROWS --samples 2 --seed 1 --ids --live $LIVE ${RESUME:+--resume} $MORE"
DEV="MIP_EXPR_DEVICE=cuda MIP_CUDA_RESIDENT=2 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 MIP_CUDA_DEVICE=0"
env $DEV TIR4B_TIMING=$OUT/a.timing.jsonl python tools/igc/tir4b.py run $C --out $OUT/a $F > $OUT/a.log 2>&1 &
P1=$!
env $DEV TIR4B_TIMING=$OUT/b.timing.jsonl python tools/igc/tir4b.py run $C --out $OUT/b $F $FIX > $OUT/b.log 2>&1 &
P2=$!
env $DEV TIR4B_TIMING=$OUT/ch.timing.jsonl python tools/igc/tir4b.py run $C --out $OUT/ch $F $FIX --channel --channel-port $MIP_PORT > $OUT/ch.log 2>&1 &
P3=$!
wait $P1 $P2 $P3
kill $SP
wait
python scripts/pc/gate3_values.py $OUT/ch > $OUT/gate3.log 2>&1
for arm in a b ch; do
  python -m tools.igc.answer_check $OUT/$arm --rows $ROWS --env D:/Github/Strata/.worktrees/native-qwenputer-p0/.env > $OUT/reader-$arm.log 2>&1
done
for arm in a b ch; do echo "$arm $(cat $OUT/$arm/report.json)"; tail -2 $OUT/reader-$arm.log; done
cat $OUT/gate3.log; grep -c "mip-channel: no plan" $OUT/llama.log
echo R10 PC DONE
