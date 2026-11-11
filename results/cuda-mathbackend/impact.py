"""impact.py: what the CUDAMathBackend's resident worker costs llama-server's decode on the same card (design: "the acceptance measure":
tokens per second with MIP on against off, ABBA in one server session; separate launches differ by about 15 %).

A running llama-server gets a fixed decode load: 4 threads, each asking again and again for 256 tokens of one fixed prompt (ignore_eos,
so every request decodes the same length). Meanwhile this process holds the MathBackend DLL and runs, window by window:
  none   no call (the worker ends ~1 s after the last call and stays gone)
  keep   one numbers.add call every 0.5 s (the worker is resident and polling; almost no work)
  busy   numbers.add calls back to back (the worker polls and works all the time)
in the order none keep busy busy keep none. The decode rate of a window is the tokens of the requests that finished in it over its
length (the first 5 s of each window are skipped).
python results/cuda-mathbackend/impact.py --server http://127.0.0.1:8099 [--window 60]   (MIP_NANOSLEEP etc. as the DLL reads them)
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.igc.mathbackend import canon, cuda_backend  # noqa: E402

DONE = []          # (finish time, tokens)
STOP = threading.Event()


def load(server, prompt, seed):
    body = dict(prompt=prompt, n_predict=256, ignore_eos=True, temperature=0.6, top_p=0.95, top_k=20, seed=seed, cache_prompt=True)
    while not STOP.is_set():
        req = urllib.request.Request(server + '/completion', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
        DONE.append((time.monotonic(), out['timings']['predicted_n']))
        body['seed'] += 4


def mip(mode, until):
    args = canon.encode([1, 1])
    n = 0
    while time.monotonic() < until:
        if mode == 'none':
            time.sleep(0.05)
            continue
        assert cuda_backend.call('numbers.add', args) == b'["i","2"]'
        n += 1
        if mode == 'keep':
            time.sleep(0.5)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='http://127.0.0.1:8099')
    ap.add_argument('--window', type=int, default=60)
    a = ap.parse_args()
    row = json.loads(open('C:/Users/xenod/mip-work/a100/pc9-dev-rows.jsonl', encoding='utf-8').readline())
    prompt = '<|im_start|>user\n' + row['question'] + '<|im_end|>\n<|im_start|>assistant\n<think>\n'
    threads = [threading.Thread(target=load, args=(a.server, prompt, 1 + k), daemon=True) for k in range(4)]
    for t in threads:
        t.start()
    time.sleep(20)   # warm: the prompt cached, all four slots decoding
    cuda_backend.call('numbers.add', canon.encode([1, 1]))
    time.sleep(2)    # the first worker ends; 'none' starts with no resident kernel
    rows = []
    for mode in ('none', 'keep', 'busy', 'busy', 'keep', 'none'):
        t0 = time.monotonic()
        calls = mip(mode, t0 + a.window)
        t1 = time.monotonic()
        toks = sum(n for t, n in DONE if t0 + 5 <= t < t1)
        rows.append((mode, toks / (t1 - t0 - 5), calls))
        print(f'{mode:5s} {toks / (t1 - t0 - 5):7.1f} tok/s  MIP calls {calls}', flush=True)
        if mode != 'none':
            time.sleep(1.5)   # let the worker end before a 'none' window
    STOP.set()
    by = {m: [r for mm, r, _ in rows if mm == m] for m in ('none', 'keep', 'busy')}
    base = statistics.mean(by['none'])
    print(f'MIP_NANOSLEEP {os.environ.get("MIP_NANOSLEEP", "500")}: none {base:.1f}, keep {statistics.mean(by["keep"]):.1f} '
          f'({100 * (statistics.mean(by["keep"]) / base - 1):+.1f} %), busy {statistics.mean(by["busy"]):.1f} '
          f'({100 * (statistics.mean(by["busy"]) / base - 1):+.1f} %); launches {cuda_backend.launches()}')
    cuda_backend.close()


if __name__ == '__main__':
    main()
