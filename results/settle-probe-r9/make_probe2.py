"""Builds settle_probe2.py from settle_probe.py: the settle line in the model's own voice, and the same with the re-check words held
down for 200 tokens after it. Each edit asserts it applied."""
p_in, p_out = 'settle_probe.py', 'settle_probe2.py'
s = open(p_in, encoding='utf-8').read()


def cut_between(s, start, end, new):
    i = s.index(start)
    j = s.index(end, i)
    return s[:i] + new + s[j:]


DOC = '''"""Settle probe 2: the settle line in the model's own voice ("voice"), and the same with the re-check words held down by -30 for 200
tokens after it ("voice_nowait"). Thinking Intervention (arXiv 2503.24370); NoWait (arXiv 2506.08343), here only in a window after the
settle line, not over the whole answer as --suppress-recheck did (#298). Same 49 closes, same seed as probe 1.
Writes probe-settle2/{voice,voice_nowait}/records.json. python settle_probe2.py PORT[,PORT] [SEEDS]"""
'''
s = cut_between(s, '"""Settle probe', 'import glob', DOC)

PUT = r'''import re
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


'''
s = cut_between(s, 'def put(case, arm):', "ARMS = (", PUT)
assert s.count("ARMS = ('plain', 'settle', 'close')") == 1
s = s.replace("ARMS = ('plain', 'settle', 'close')", "ARMS = ('voice', 'voice_nowait', 'brake', 'settle_close', 'settle_relapse_close')")
s = s.replace('"""Settle probe 2: the settle line', '"""Settle probe 2 (three ways the model might leave a doubt loop): the settle line', 1)
s = s.replace('settle line, not over the whole answer as --suppress-recheck did (#298). Same 49 closes',
              'settle line, not over the whole answer as --suppress-recheck did (#298); "brake": nothing put, the thinking closed where\n'
              '</think> comes within BRAKE nats of the top token (ThinkBrake, arXiv 2510.00546). Same 49 closes', 1)
n = s.count("probe-settle/") + s.count("'probe-settle'")
s = s.replace("probe-settle/", "probe-settle2/").replace("'probe-settle'", "'probe-settle2'")
assert n >= 4, n
open(p_out, 'w', encoding='utf-8').write(s)
print('ok', n)
