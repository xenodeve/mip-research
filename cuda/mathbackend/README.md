# CUDAMathBackend: how to use it

The MathBackend's exact operations run on the GPU behind a C ABI, byte-identical to the Python MathBackend
(`tools/igc/mathbackend`). Design, every measurement and the decisions behind this page:
[`docs/plans/2026-11-11-cuda-mathbackend-design.md`](../../docs/plans/2026-11-11-cuda-mathbackend-design.md).

**Ported so far:** `numbers.add`, `numbers.sub`, `numbers.mul`, `numbers.div` (exact rationals: ints, Fractions, and strings in Python's
`Fraction(str)` grammar with Unicode digits and whitespace). Any other operation returns `UNKNOWN_OP`: the caller keeps the Python
MathBackend for it.

**Not the decode bottleneck.** On round 9's traces tir4b reached `mathbackend.run()` once in 32 answers and `mathexpr.evaluate` 32,335
times: plain arithmetic is read and computed by the text readers, not by these operations. Moving the readers (design, phase 3) is what
would take MIP's CPU work out of decoding; this backend does not, yet.

## Build

```
cuda\mathbackend\build.cmd
```

Needs CUDA 13.3 and the Visual Studio 2022 Build Tools (`vcvars64.bat`). Builds `build-cuda\mipmath.dll` for sm_89 (RTX 4070 SUPER) and
sm_120 (RTX 5060 Ti) plus PTX, linked with the driver API (`-lcuda`). `build-cuda/` is not tracked.

## Call it

From Python, as a drop-in for one call (canonical bytes in and out, `tools/igc/mathbackend/canon.py`):

```python
from tools.igc.mathbackend import canon, cuda_backend

out = cuda_backend.call('numbers.add', canon.encode([1, '1/2']))   # b'["q","3","2"]'
value = canon.decode(out)                                           # Fraction(3, 2), or ('refusal', 'BadInput' | 'NotCovered')
```

`call` returns the canonical bytes of the value or of the refusal, exactly as `canon.call(name, args)` does for Python. Any other status
comes back as `device:<STATUS>` bytes (`DEVICE_FAILED`, `UNKNOWN_OP`, `HOST_ERROR`): **no value**, so the caller falls back to Python or
puts nothing. A device result is never a guess: past its limits the device refuses or fails, it does not approximate.

From C or C++, the ABI is `mipmath.h`: `mip_open(device, mode)`, `mip_call(op, args, len, out, cap, &out_len)` returning a `mip_status`,
`mip_close()`. Calls are serialised by a lock inside the DLL; one process holds one device.

## Pick the mode for where it runs

`MIP_CUDA_RESIDENT` (read once, at the first call):

| Value | Mode | Use it when |
|---|---|---|
| **`2`** (recommended) | one launch per call, zero copy: the kernel reads the request from mapped memory (small arguments ride in the launch parameters) and raises the answer's flag itself | **the engine is another process** (tir4b over llama-server, the PC layer): it costs decode nothing while idle (-0.1 to -0.2 %) and -8 % only under a flood of ~5,200 calls/s |
| `1` | a resident worker kernel polls for requests, relaunched every ~1 s (under Windows' 2 s TDR) | **MIP inside the engine's own process** (the module's end shape): sparse calls 7.5 us inside the DLL, the engine within 2 % (stand-in engine). **Never beside an engine in another process**: there it halves decode (-44 to -59 %), even idle |
| `0` | one launch per call with copies and a blocking sync | comparison only (240 us a call) |

`MIP_CUDA_DEVICE`: the CUDA device index after `CUDA_VISIBLE_DEVICES` (default 0). On this PC, select the card with
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (5060 Ti) or `=0` (4070 SUPER, the display card: ask first).

### Knobs (defaults are the measured best; change them only with a reason)

| Variable | Default | What it does | When to change |
|---|---|---|---|
| `MIP_LINGER_US` | 0 (off) | mode 2: after answering, the kernel waits this long for the next request, so a call that follows needs no launch (back to back 5.8-6.5 us) | only for **bursts** of calls with nothing else on the card; beside an engine a dense burst costs its decode up to -28 % |
| `MIP_LINGER_SLEEP` | 100 (ns) | the lingering kernel's sleep between polls | 0 gains ~0.1 us |
| `MIP_SPIN` | 2000 | host polls of the answer flag before it starts yielding the core | no measured effect |
| `MIP_NANOSLEEP` | 500 (ns) | mode 1: the resident worker's sleep between polls | no measured effect on decode impact |
| `MIP_PARAM` | 1 | mode 2: arguments of at most 256 bytes ride in the launch parameters | 0 only to compare |
| `MIP_PARAM_SMEM` | 0 | the parameter kernel also asks for the shared-memory arena | no gain measured |
| `MIP_GRAPH` | 0 | mode 2 as a captured CUDA graph | no gain measured |
| `MIP_STREAM_HIGH` | 0 | the work stream at the highest priority (default: the lowest, so an engine in the same process goes first) | no effect across processes; in one process it did not lower latency |
| `MIP_DIAG` | 1 | the register path writes its path bits and device cycles (two PCIe writes; the tests read them) | 0 saves ~0.1 us |
| `MIP_SYNC` | (mode 0: block) | mode 0's wait: `spin`, `yield` or `block` | comparison only |
| `MIP_POLL` | 1 | mode 1: the flag read past every cache | debugging only |
| `MIP_TEAM` | 256 (modes 0, 2), 1 (mode 1) | threads in each kernel's block: thread 0 does what the single thread did, warps 1 and up split the long loops of big numbers (multiply, the decimal conversions, Lehmer's steps); capped at what the kernels can launch with (256 here), at least 64 when above 1 | 1 to compare; above 1 in mode 1 only with no engine in the process (a resident team holds one SM's registers) |
| `MIP_TEAM_WORK` | 2048 | the least work (limb steps) worth a worker | 1 sends every case through the team (the tests) |
| `MIP_CALL_US` | (0.4 s) | a shorter per-call time guard | tests of the guard only |

Not knobs: **HAGS stays on** (the developer's requirement); a **locked GPU clock** (`nvidia-smi -lgc`) gained nothing. A 1 ms empty
kernel (`sparse.py --heartbeat-ms 1`) keeps an otherwise idle card awake (a call every 50 ms: 291 -> 102 us), but only matters with
nothing else on the card.

## What it costs (RTX 5060 Ti unless noted, HAGS on)

| Condition | Per call | For comparison |
|---|---|---|
| idle card, a single call, mode 2 | 10.3-10.4 us inside the DLL, ~12 us from Python | an empty kernel 8.7 us; Python MathBackend 7-12 us warm |
| idle card, back to back, `MIP_LINGER_US=50` | 6.4-6.5 us; **5.8-5.9 us on the 4070 SUPER** (PCIe x16) | |
| an engine in another process decoding on the card | ~175 us inside the DLL, ~200 us from Python | an empty kernel there: 175 us (the cross-process switch); Python: ~50 us |
| MIP in the engine's process, mode 1, a call every 5 ms | 7.5 us inside the DLL (35-38 us with the caller's own wake-up) | Python: 33-51 us on that rhythm |
| big numbers, mode 2, team 256 (2026-11-12) | 20,000 digits 14 ms, 4,000-character strings 6.3 ms, 3,000-digit mul 5.3 ms | team 1: 122, 8.9, 12.6 ms; Python 11-106 us (without reading or printing text) |

So beside another process's engine the device answers slower than Python but takes the arithmetic off the CPU; inside the engine's
process it is as fast as Python and takes nothing from the engine.

## Limits

- **Integers up to 8,192 limbs (262,144 bits) per part**; past that `NOT_COVERED`, where Python answers ("any size"): open question 3 of
  the design (a bound both sides keep).
- **A per-call time guard of 0.4 s**: past it `DEVICE_FAILED` (never a value). It keeps every kernel far under Windows' TDR watchdog,
  whose reset would take down every program using the GPU. The guard ends the team's jobs at a barrier, never between them (tested
  with `MIP_CALL_US` down to 30 us: no hang, no wrong value).
- Only the four rational operations; `UNKNOWN_OP` for the rest.

## Check it

Before using a build, and on both cards for a change to the kernels (`run5060.sh` runs a command at Idle priority on the 5060 Ti;
`MIP_PCI=0` moves it to the 4070 SUPER):

```
sh cuda/mathbackend/run5060.sh out.txt python -m tools.igc.mathbackend.harness check results/cuda-mathbackend/corpus/numbers-fuzz-s2.jsonl --backend cuda
sh cuda/mathbackend/run5060.sh out.txt python -m tools.igc.mathbackend.harness check results/cuda-mathbackend/corpus/numbers-paths-s3.jsonl --backend cuda
sh cuda/mathbackend/run5060.sh out.txt python -m tools.igc.mathbackend.harness check results/cuda-mathbackend/corpus/numbers-s1.jsonl --backend cuda
sh cuda/mathbackend/run5060.sh out.txt python -m tools.igc.mathbackend.harness check results/cuda-mathbackend/corpus/numbers-gcd-s4.jsonl --backend cuda
sh cuda/mathbackend/run5060.sh out.txt python -m tools.igc.mathbackend.harness check results/cuda-mathbackend/corpus/numbers-team-s5.jsonl --backend cuda
     (each also with MIP_TEAM_WORK=1, every case through the team; and MIP_CALL_US=30 with results/cuda-mathbackend/guard.py: no hang, 0 wrong)
sh cuda/mathbackend/run5060.sh out.txt python results/cuda-mathbackend/linger_stress.py --n 20000 --gap-us 120     (with MIP_LINGER_US=50)
python -m unittest tools.test_mathbackend_harness
```

Expected now: fuzz 4,000 / 4,000, paths 220 / 220, s1 1,816 / 1,816, gcd 101 / 101, team 214 / 214, 0 wrong in the stress run, in modes 1
and 2. `compute-sanitizer --tool memcheck` (and initcheck, racecheck, synccheck) reported 0 errors on the paths corpus on both cards
before the team; with the team, synccheck and memcheck 0 errors on the team corpus (racecheck did not finish: its process hung at exit).

Speed: `ping.py` (latency and its split), `floor.py` (the empty-kernel floor), `sparse.py`, `impact.py` (decode tok/s beside the calls,
ABBA), `under_load.py` (`--empty` for the floor there), `in_process.py` (a stand-in engine in the same process), `speed.py` (per op,
against Python). All in `results/cuda-mathbackend/`, all need the developer's go-ahead before they run (they use a GPU).

## Files

| File | What |
|---|---|
| `mipmath.h`, `mipmath.cu` | the C ABI and the device code |
| `unicode_tables.h`, `gen_tables.py` | Python's digit runs and whitespace, generated from the reference Python |
| `build.cmd`, `run5060.sh` | build; run a command on a chosen card at Idle priority |
| `../../tools/igc/mathbackend/canon.py`, `harness.py`, `cuda_backend.py` | the encoding, the gate (corpus, null and wrong backends), the Python binding |
