"""Settle probe (decide layer v1, a sample): at each stored round-9 General close that has a settle line (results/decide-layer-r9/
sim-v1.json), the model's own tokens up to the close, cut back to one round of the loop, then one of three continuations, same seed:
  plain   nothing put, the model goes on
  settle  the settle line put on its own line
  close   today's close (`\\n</think>\\n\\n` in the thinking, the final-answer cue in the answer)
Streamed; a continuation stops when the loop's line comes back 3 times (relapse), at the end of the turn, or at 16,384 tokens (a probe
cap, not a run). Writes probe-settle/{plain,settle,close}/records.json and probe-settle/rows.jsonl.
python settle_probe.py PORT[,PORT] [SEEDS]; each finished job is appended to probe-settle/done.jsonl and skipped on a rerun"""
import glob
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, r'D:\Github\mip-research')
from transformers import AutoTokenizer  # noqa: E402
from tools.igc.select_rows import right  # noqa: E402
from tools.igc.tir4b import loop_start  # noqa: E402
from tools.igc.loop_class import cycle  # noqa: E402

PORTS = sys.argv[1].split(',')   # one llama-server per card
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
CAP = 16384
LOOP = '\n</think>\n\n'
CUE = '\n\nThe final answer is $\\boxed'   # the cue as fixed in 0e14ca1
tok = AutoTokenizer.from_pretrained(r'C:\Users\xenod\mip-work\q4b-hf')
sim = {x['key']: x for x in json.load(open(r'D:\Github\mip-research\results\decide-layer-r9\sim-v1.json', encoding='utf-8'))['labelled']}

questions = {}
for p in ('frozen9a-stay-rows.jsonl', 'frozen9b-rows.jsonl'):
    for line in open(p, encoding='utf-8'):
        d = json.loads(line)
        questions[d['id']] = d

cases = []
for h in 'ab':
    arm = f'pc/r9{h}-gen'
    ids = {(d['id'], d['sample']): d for d in (json.loads(l) for l in open(f'{arm}/ids.jsonl', encoding='utf-8'))}
    for r in json.load(open(f'{arm}/records.json', encoding='utf-8')):
        key = f"{r['id']}#{r['sample']}"
        s = sim.get(key)
        if not s or not s['settle']:
            continue
        ev = next(e for e in r['events'] if e['kind'] in ('loop close', 'answer close'))
        d = ids[(r['id'], r['sample'])]
        out = d['out_ids'][:ev['at']]
        start = loop_start(out)
        if start is not None:
            out = out[:start]
        before = tok.decode(out)
        unit = cycle(tok.decode(d['out_ids'][:ev['at']]))
        cases.append(dict(key=key, row=questions[r['id']], prompt=d['prompt_ids'], out=out, before=before, kind=ev['kind'],
                          settle=s['settle'], cls=s['cls'], unit=max(unit, key=len) if unit else None))
print(len(cases), 'closes with a settle line', flush=True)


def put(case, arm):
    if arm == 'plain':
        return ''
    if arm == 'settle':
        return ('' if case['before'].endswith('\n') else '\n') + case['settle'] + '\n'
    return LOOP if case['kind'] == 'loop close' else CUE


def gen(case, arm, seed, port):
    extra = put(case, arm)
    prompt = case['prompt'] + case['out'] + (tok.encode(extra, add_special_tokens=False) if extra else [])
    body = dict(prompt=prompt, n_predict=CAP, temperature=0.6, top_p=0.95, top_k=20, seed=seed, stream=True, cache_prompt=True)
    req = urllib.request.Request(f'http://127.0.0.1:{port}/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    text, why, t0 = '', 'cap', time.time()
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue
            msg = json.loads(line[5:])
            text += msg.get('content', '')
            if msg.get('stop'):
                why = 'end' if msg.get('stop_type') in ('eos', 'word') or msg.get('stopped_eos') else 'cap'
                break
            if case['unit'] and text.count(case['unit']) >= 3:
                why = 'relapse'
                break
    return extra, text, why, time.time() - t0


ARMS = ('plain', 'settle', 'close')
import threading
jobs = [(c, arm, s) for c in cases for s in range(1, SEEDS + 1) for arm in ARMS]
os.makedirs('probe-settle', exist_ok=True)
LOG = 'probe-settle/done.jsonl'
saved = {}
if os.path.exists(LOG):
    for line in open(LOG, encoding='utf-8'):
        d = json.loads(line)
        saved[(d['key'], d['arm'], d['seed'])] = (d['extra'], d['out'], d['why'], d['secs'])
lock = threading.Lock()
todo = [j for j in jobs if (j[0]['key'], j[1], j[2]) not in saved]
print(len(saved), 'saved,', len(todo), 'to run', flush=True)


def run(i_job):
    i, (c, arm, s) = i_job
    res = gen(c, arm, s, PORTS[i % len(PORTS)])
    with lock:
        saved[(c['key'], arm, s)] = res
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict(key=c['key'], arm=arm, seed=s, extra=res[0], out=res[1], why=res[2], secs=res[3]), ensure_ascii=False) + '\n')
        if len(saved) % 10 == 0:
            print(f'{len(saved)} / {len(jobs)}', flush=True)


with ThreadPoolExecutor(int(os.environ.get('CONC', '3')) * len(PORTS)) as ex:
    list(ex.map(run, enumerate(todo)))
outs = [saved[(c['key'], arm, s)] for c, arm, s in jobs]

recs = {a: [] for a in ARMS}
rows = {}
for (c, arm, s), (extra, out, why, secs) in zip(jobs, outs):
    row = dict(c['row'], id=c['key'])
    rows[c['key']] = row
    text = c['before'] + extra + out + ('<|im_end|>' if why == 'end' else '')
    ntok = len(tok.encode(out, add_special_tokens=False))
    recs[arm].append(dict(id=c['key'], suite=row.get('suite'), answer=row['answer'], sample=s, seed=s, text=text, cls=c['cls'],
                          kind=c['kind'], why=why, finish='stop' if why == 'end' else why, tokens=ntok, seconds=round(secs, 1),
                          right=bool(why == 'end' and right(row, text))))
for arm in ARMS:
    os.makedirs(f'probe-settle/{arm}', exist_ok=True)
    json.dump(recs[arm], open(f'probe-settle/{arm}/records.json', 'w', encoding='utf-8'), ensure_ascii=False)
with open('probe-settle/rows.jsonl', 'w', encoding='utf-8') as f:
    for row in rows.values():
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
for arm in ARMS:
    rs = recs[arm]
    by = {w: sum(r['why'] == w for r in rs) for w in ('end', 'relapse', 'cap')}
    print(arm, 'right', sum(r['right'] for r in rs), '/', len(rs), by, 'tokens', sum(r['tokens'] for r in rs), flush=True)
