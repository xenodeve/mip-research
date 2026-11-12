"""rank_probe.py: phase 0 of the token channel (docs/plans/2026-11-12-mip-token-channel-plan.md, task 1). A GPU run on the RTX 5060 Ti.

At stored inline-put positions of round 9's General traces (4B NVFP4 W4A8 · llama.cpp · PC layer), where tir4b put a value after `=` in
the thinking, a stock llama-server is asked for one token with the answer's own prefix (prompt and text up to the `=`, as token ids) and
n_probs 50 (the distribution before sampling): the rank and probability of the put's first token. It tells how often the right value is
already the model's first choice, inside top-k 20, or outside, i.e. how much the channel's mask changes what the sampler would draw.
  python scripts/pc/rank_probe.py --server http://127.0.0.1:8099 [--n 300]   (writes results/token-channel/rank_probe.txt)
"""
import argparse
import json
import math
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = Path('C:/Users/xenod/mip-work/a100')
HALVES = ('r9a-gen', 'r9a3-gen', 'r9b-gen')


def post(server, body):
    req = urllib.request.Request(server.rstrip('/') + '/completion', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='http://127.0.0.1:8099')
    ap.add_argument('--n', type=int, default=300)
    a = ap.parse_args()
    sites = []   # (half, id, sample, prompt ids, prefix ids, put's first token)
    for half in HALVES:
        for line in (W / 'pc' / half / 'ids.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            for s, e, k in x['spans']:
                if k == 'inline value':
                    sites.append((half, x['id'], x.get('sample', 0), x['prompt_ids'], x['out_ids'][:s], x['out_ids'][s]))
    step = max(1, len(sites) // a.n)
    picked = sites[::step][:a.n]
    ranks, probs = Counter(), []
    lines = []
    for i, (half, rid, sample, prompt, prefix, first) in enumerate(picked):
        r = post(a.server, dict(prompt=prompt + prefix, n_predict=1, n_probs=50, temperature=0.6, top_k=20, top_p=0.95, min_p=0,
                                cache_prompt=True, seed=1))
        top = r['completion_probabilities'][0]['top_logprobs']
        order = [t['id'] for t in top]
        rank = order.index(first) + 1 if first in order else None
        p = math.exp(top[order.index(first)]['logprob']) if rank else 0.0
        probs.append(p)
        ranks['1' if rank == 1 else '2-20' if rank and rank <= 20 else '21-50' if rank else 'not in top 50'] += 1
        lines.append(f'{half} {rid}#{sample} at {len(prefix)}: rank {rank} p {p:.3f} (top {top[0]["token"]!r} {math.exp(top[0]["logprob"]):.3f})')
        if (i + 1) % 50 == 0:
            print(f'{i + 1}/{len(picked)}', flush=True)
    n = len(picked)
    head = [f'rank probe (token channel, phase 0): {n} of {len(sites)} stored inline puts in the thinking, round 9 General traces; '
            f'stock llama-server, 4B NVFP4 W4A8 · llama.cpp build-w4a8 · RTX 5060 Ti; n_probs 50, T 0.6, top-k 20, top-p 0.95',
            ' '.join(f'{k}: {ranks[k]} ({ranks[k] / n:.1%})' for k in ('1', '2-20', '21-50', 'not in top 50')),
            f'probability of the right first token: median {sorted(probs)[n // 2]:.3f}, mean {sum(probs) / n:.3f}']
    out = ROOT / 'results' / 'token-channel' / 'rank_probe.txt'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(head + lines) + '\n', encoding='utf-8')
    print('\n'.join(head))


if __name__ == '__main__':
    main()
