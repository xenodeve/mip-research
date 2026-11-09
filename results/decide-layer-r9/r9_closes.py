"""Round 9 (4B NVFP4 W4A8 · llama.cpp · PC layer), General arms: every loop close and answer close, with the control arm's
result on the same key. Writes r9_closes.json (rows) and r9_closes.txt (the text around each close, for reading)."""
import json, re, sys
from pathlib import Path

PC = Path(r'C:\Users\xenod\mip-work\a100\pc')
OUT = Path(sys.argv[1])
LOOP = '\n</think>\n\n'
CUE = '\n\nThe final answer is $\\boxed{'


def load(d):
    return {f"{x['id']}#{x['sample']}": x for x in json.load(open(PC / d / 'records.json', encoding='utf-8'))}


def ver(d):
    return json.load(open(PC / d / 'verdicts.json', encoding='utf-8'))


def boxed(s):
    out, i = [], 0
    while True:
        i = s.find('\\boxed{', i)
        if i < 0:
            return out
        j, depth = i + 7, 1
        while j < len(s) and depth:
            depth += {'{': 1, '}': -1}.get(s[j], 0)
            j += 1
        out.append(s[i + 7:j - 1])
        i = j


rows, dump = [], []
for h in 'ab':
    g, c, vg, vc = load(f'r9{h}-gen'), load(f'r9{h}-ctrl'), ver(f'r9{h}-gen'), ver(f'r9{h}-ctrl')
    for k, x in g.items():
        closes = [e for e in x['events'] if e['kind'] in ('loop close', 'answer close')]
        if not closes:
            continue
        t = x['text']
        cx = c.get(k)
        for e in closes:
            if e['kind'] == 'loop close':
                pos = t.find(LOOP)
                before, after = t[:pos], t[pos + len(LOOP):]
            else:
                pos = t.rfind(CUE)
                before, after = t[:pos], t[pos:]
            think = t.split('</think>', 1)[0]
            rows.append(dict(
                key=k, half=h, kind=e['kind'], model_tokens=e.get('model_tokens'), at=e.get('at'), cut=e.get('cut_tokens'),
                gold=x['answer'], gen_rule=x['right'], gen_api=vg.get(k, {}).get('verdict'), gen_value=vg.get(k, {}).get('value'),
                ctrl_rule=cx['right'] if cx else None, ctrl_finish=cx['finish'] if cx else None,
                ctrl_api=vc.get(k, {}).get('verdict') if cx else None, ctrl_tokens=cx['tokens'] if cx else None,
                boxed_in_think=boxed(think)[-5:], n_closes=len(closes), found=pos >= 0))
            dump.append(f"==== {k} [{h}] {e['kind']} at {e.get('model_tokens')} cut {e.get('cut_tokens')} | gold {x['answer']} | "
                        f"GEN rule {x['right']} api {rows[-1]['gen_api']} ({rows[-1]['gen_value']}) | CTRL {rows[-1]['ctrl_finish']} "
                        f"rule {rows[-1]['ctrl_rule']} api {rows[-1]['ctrl_api']} | boxed in think {rows[-1]['boxed_in_think']}\n"
                        f"--- before (last 700 chars)\n{before[-700:]}\n--- after (first 500 chars)\n{after[:500]}\n")
OUT.with_suffix('.json').write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding='utf-8')
OUT.with_suffix('.txt').write_text('\n'.join(dump), encoding='utf-8')
print(len(rows), 'closes;', sum(not r['found'] for r in rows), 'not located')
