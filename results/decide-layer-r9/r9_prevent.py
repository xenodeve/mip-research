"""Round 9: of the keys where control ended `stuck` and General ended right with no close, does a General put on the
equation control was repeating come first? Prints one block per key and the counts."""
import json, re, collections
from pathlib import Path
PC = Path(r'C:\Users\xenod\mip-work\a100\pc')
def load(d): return {f"{x['id']}#{x['sample']}": x for x in json.load(open(PC/d/'records.json', encoding='utf-8'))}
def ver(d): return json.load(open(PC/d/'verdicts.json', encoding='utf-8'))
NUM = re.compile(r'\d+(?:\.\d+)?')
tot = collections.Counter()
for h in 'ab':
    g, c, vg = load(f'r9{h}-gen'), load(f'r9{h}-ctrl'), ver(f'r9{h}-gen')
    for k in g:
        if k not in c or c[k]['finish'] != 'stuck': continue
        G = g[k]
        if any(e['kind'] != 'inline value' for e in G['events']): continue
        if not (G['right'] or vg.get(k, {}).get('verdict') == 'same'): continue
        lines = [l.strip() for l in c[k]['text'][-4000:].split('\n') if l.strip()]
        unit, n = collections.Counter(lines).most_common(1)[0]
        nums = set(NUM.findall(unit))
        hits = [e for e in G['events'] if e['kind'] == 'inline value' and len(nums) >= 2 and nums <= set(NUM.findall(e['expr'] + ' ' + e['value']))]
        kind = 'numbers in a put' if hits else ('no numbers' if len(nums) < 2 else 'no put on it')
        tot[kind] += 1
        print(f"## {k} [{h}] ctrl loop x{n}: {unit[:150]}\n   -> {kind}; first matching put: {(hits[0]['expr'][:60], hits[0]['value'][:40], hits[0]['model_tokens']) if hits else None}")
print(tot)
