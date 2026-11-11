#!/bin/sh
# Run a CUDAMathBackend check or speed test on the RTX 5060 Ti only (PCI order index 1), at Idle priority so the desktop and a game on
# the 4070 SUPER keep the CPU first. The command's output goes to the file named last.
#   sh cuda/mathbackend/run5060.sh <out file> python <args...>
OUT=$1; shift
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${MIP_PCI:-1} MIP_CUDA_DEVICE=0   # MIP_PCI=0: the 4070 SUPER (only with the developer's go-ahead)
ARGS=""
for a in "$@"; do ARGS="$ARGS,'$a'"; done
ARGS=${ARGS#,}
EXE=$1; shift
LIST=""
for a in "$@"; do LIST="$LIST,'$a'"; done
LIST=${LIST#,}
powershell -NoProfile -Command "\$p = Start-Process -FilePath '$EXE' -ArgumentList $LIST -NoNewWindow -PassThru -RedirectStandardOutput '$OUT' -RedirectStandardError '$OUT.err'; \$p.PriorityClass='Idle'; \$p.WaitForExit(); exit \$p.ExitCode"
