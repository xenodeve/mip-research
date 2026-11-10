"""Settle probe 2 (three ways the model might leave a doubt loop): the settle line in the model's own voice ("voice"), and the same with the re-check words held down by -30 for 200
tokens after it ("voice_nowait"). Thinking Intervention (arXiv 2503.24370); NoWait (arXiv 2506.08343), here only in a window after the
settle line, not over the whole answer as --suppress-recheck did (#298); "brake": nothing put, the thinking closed where
</think> comes within BRAKE nats of the top token (ThinkBrake, arXiv 2510.00546). Same 49 closes, same seed as probe 1.
Writes probe-settle2/{voice,voice_nowait}/records.json. python settle_probe2.py PORT[,PORT] [SEEDS]"""
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


import re
RECHECK_WORDS = {'wait', 'hmm', 'hmmm', 'verify', 'verifying', 'check', 'checking', 'double', 'recheck', 'rechecking', 'recalculate',
                 'recompute', 'again'}   # as tir4b's --suppress-recheck
RECHECK = [i for i in range(len(tok)) if tok.decode([i]).strip().lower().rstrip('.,:!') in RECHECK_WORDS]
WINDOW = 200


def voice(line):
    m = re.fullmatch(r'Checked exactly: \$(.*)\$ holds\.', line)
    if m:
        return f"Okay, ${m.group(1)}$ is exact; I have confirmed it. I'll stop re-checking it and move on."
    m = re.fullmatch(r'The question writes, exactly: (.*)', line, re.S)
    if m:
        return f"Okay, the question says exactly: {m.group(1)}. That settles how it reads; I'll move on."
    m = re.match(r'Computed exactly: \$(.*?)\$', line)
    if m:
        return f"Okay, the exact value is ${m.group(1)}$. That settles it; I'll move on."
    raise ValueError(line)


def put(case, arm):
    return ('' if case['before'].endswith('\n') else '\n') + voice(case['settle']) + '\n'


def gen(case, arm, seed, port):
    extra = put(case, arm)
    prompt = case['prompt'] + case['out'] + tok.encode(extra, add_special_tokens=False)
    if arm == 'voice_nowait':
        first, why, secs = stream(case, prompt, WINDOW, seed, port, [[i, -30.0] for i in RECHECK], '')
        if why != 'cap':
            return extra, first, why, secs
        rest, why, secs2 = stream(case, prompt + tok.encode(first, add_special_tokens=False), CAP - WINDOW, seed, port, None, first)
        return extra, first + rest, why, secs + secs2
    if arm in ('settle_close', 'settle_relapse_close'):
        # the v1 settle line (outside voice), then today's close: at once, or only when the loop comes back (the plan's design)
        line = ('' if case['before'].endswith('\n') else '\n') + case['settle'] + '\n'
        close = LOOP if case['kind'] == 'loop close' else CUE
        if arm == 'settle_close':
            extra = line + close
            text, why, secs = stream(case, case['prompt'] + case['out'] + tok.encode(extra, add_special_tokens=False), CAP, seed, port, None, '')
            return extra, text, why, secs
        p0 = case['prompt'] + case['out'] + tok.encode(line, add_special_tokens=False)
        first, why, secs = stream(case, p0, CAP, seed, port, None, '')
        if why != 'relapse':
            return line, first, why, secs
        again = CUE if ('</think>' in first or case['kind'] != 'loop close') else LOOP
        rest, why, secs2 = stream(case, p0 + tok.encode(first + again, add_special_tokens=False), CAP, seed, port, None, '')
        return line, first + again + rest, why, secs + secs2
    if arm == 'brake':
        # ThinkBrake (arXiv 2510.00546): nothing put; in the thinking, close at the first token where </think> is within BRAKE nats of
        # the top token. Only for closes in the thinking; an answer close has no thinking left to brake.
        prompt = case['prompt'] + case['out']
        if case['kind'] != 'loop close':
            return '', '', 'n/a', 0.0
        first, why, secs = stream(case, prompt, CAP, seed, port, None, '', brake=True)
        if why != 'brake':
            return '', first, why, secs
        put_ = LOOP
        rest, why, secs2 = stream(case, prompt + tok.encode(first + put_, add_special_tokens=False), CAP, seed, port, None, '')
        return '', first + put_ + rest, why, secs + secs2   # the brake point is where the first </think> is
    text, why, secs = stream(case, prompt, CAP, seed, port, None, '')
    return extra, text, why, secs


THINK_CLOSE = 248069
BRAKE = float(os.environ.get('BRAKE', '1.0'))   # nats between the top token and </think>


def braked(msg):
    import math
    for p in msg.get('completion_probabilities') or []:
        tops = p.get('top_logprobs') or p.get('top_probs') or []
        if not tops:
            continue
        lp = lambda t: t['logprob'] if 'logprob' in t else math.log(max(t.get('prob', 0.0), 1e-30))
        best = max(lp(t) for t in tops)
        for t in tops:
            if t.get('id') == THINK_CLOSE or t.get('token') == '</think>':
                return best - lp(t) <= BRAKE
    return False


def stream(case, prompt, n, seed, port, bias, prior, brake=False):
    body = dict(prompt=prompt, n_predict=n, temperature=0.6, top_p=0.95, top_k=20, seed=seed, stream=True, cache_prompt=True)
    if brake:
        body['n_probs'] = 10
    if bias:
        body['logit_bias'] = bias
    req = urllib.request.Request(f'http://127.0.0.1:{port}/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    text, why, t0 = '', 'cap', time.time()
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue
            msg = json.loads(line[5:])
            if brake and braked(msg):
                why = 'brake'
                break
            text += msg.get('content', '')
            if msg.get('stop'):
                why = 'end' if msg.get('stop_type') in ('eos', 'word') or msg.get('stopped_eos') else 'cap'
                break
            if case['unit'] and (prior + text).count(case['unit']) >= 3:
                why = 'relapse'
                break
    return text, why, time.time() - t0


ARMS = ('voice', 'voice_nowait', 'brake', 'settle_close', 'settle_relapse_close')
import threading
jobs = [(c, arm, s) for c in cases for s in range(1, SEEDS + 1) for arm in ARMS]
os.makedirs('probe-settle2', exist_ok=True)
LOG = 'probe-settle2/done.jsonl'
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
    os.makedirs(f'probe-settle2/{arm}', exist_ok=True)
    json.dump(recs[arm], open(f'probe-settle2/{arm}/records.json', 'w', encoding='utf-8'), ensure_ascii=False)
with open('probe-settle2/rows.jsonl', 'w', encoding='utf-8') as f:
    for row in rows.values():
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
for arm in ARMS:
    rs = recs[arm]
    by = {w: sum(r['why'] == w for r in rs) for w in ('end', 'relapse', 'cap')}
    print(arm, 'right', sum(r['right'] for r in rs), '/', len(rs), by, 'tokens', sum(r['tokens'] for r in rs), flush=True)
