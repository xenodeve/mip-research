"""4 requests at once to a running llama-server, each generating N tokens, in four shapes, to split where the round-9 time goes:
short prompt / long prompt (a real round-9 answer so far, as token ids) x tir4b's sampling / greedy. Prints per-request decode speed
(the server's own timings) and the total. python serve_probe.py PORT IDS_JSONL"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

port, ids_path = sys.argv[1], sys.argv[2]
URL = f'http://127.0.0.1:{port}/completion'
answers = [json.loads(line) for line in open(ids_path, encoding='utf-8')]
longest = sorted(answers, key=lambda a: -len(a['out_ids']))[:4]
SAMPLE = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repeat_penalty=1.0)
GREEDY = dict(temperature=0.0, top_k=1)


def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def shape(name, prompts, sampling, n=256, warm=False):
    if warm:   # read each context first and keep it, as the real run does (cache_prompt): the measured request then only decodes
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(lambda p: post(dict(prompt=p, n_predict=1, cache_prompt=True, **GREEDY)), prompts))
    began = time.monotonic()
    with ThreadPoolExecutor(4) as ex:
        rs = list(ex.map(lambda p: post(dict(prompt=p, n_predict=n, cache_prompt=warm, seed=1, ignore_eos=True, **sampling)), prompts))
    wall = time.monotonic() - began
    per = [r['timings']['predicted_per_second'] for r in rs]
    ctx = [len(p) if isinstance(p, list) else 0 for p in prompts]
    print(f'{name:28} context {min(ctx)}-{max(ctx)}: per request {min(per):.1f}-{max(per):.1f} tok/s, total {4 * n / wall:.0f} tok/s incl. reading '
          f'(read {max(r["timings"]["prompt_ms"] for r in rs) / 1000:.1f} s)', flush=True)


short = [a['prompt_ids'] for a in longest]
long_ = [a['prompt_ids'] + a['out_ids'][:-1] for a in longest]
shape('short, sampled', short, SAMPLE, warm=True)
shape('long, sampled, cached', long_, SAMPLE, warm=True)
