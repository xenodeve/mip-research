#!/bin/sh
# Settle probe on its own llama-servers, one per card (5060 Ti port 8097, 4070 SUPER port 8096; 3 slots each, one KV pool each),
# then the API reader on the 3 arms. Rerunnable: finished jobs are kept in probe-settle/done.jsonl.
W=C:/Users/xenod/mip-work/a100
G=C:/Users/xenod/mip-work
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/x64:/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin:$PATH"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 /c/AI/llama.cpp/build-w4a8/bin/llama-server.exe -m $G/gguf/qwen35-4b-nvfp4-axion.gguf -ngl 999 \
  -c 262144 -np 3 -kvu -fa on -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port 8097 --no-webui > $G/llama-probe-settle.log 2>&1 &
S1=$!
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 /c/AI/llama.cpp/build-w4a8/bin/llama-server.exe -m $G/gguf/qwen35-4b-nvfp4-axion.gguf -ngl 999 \
  -c 262144 -np 3 -kvu -fa on -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port 8096 --no-webui > $G/llama-probe-settle-b.log 2>&1 &
S2=$!
for p in 8097 8096; do for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:$p/health | grep -q ok && break; sleep 2; done; done
cd $W && CONC=3 PYTHONIOENCODING=utf-8 python -u settle_probe.py 8097,8096 1 >> probe-settle.log 2>&1
kill $S1 $S2
cd /d/Github/mip-research
for a in plain settle close; do
  PYTHONPATH=. python -m tools.igc.answer_check $W/probe-settle/$a --rows $W/probe-settle/rows.jsonl --env D:/Github/Strata/.worktrees/native-qwenputer-p0/.env > $W/probe-settle/reader-$a.log 2>&1
done
echo PROBE DONE
