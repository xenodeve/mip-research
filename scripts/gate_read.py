"""#307 read a replay gate (sim_gate.py NAME): per arm, acts on answers that finished right / wrong / were cut, and every act on an
answer that finished, set against the run's own events (a close the run made later is the same loop, sooner).
python gate_read.py NAME"""
import json
import sys
from pathlib import Path

HERE = Path(__import__('os').environ.get('WORK', r'C:\Users\xenod\mip-work\a100'))   # out/ and sim/, as sim_gate.py
sys.stdout.reconfigure(encoding='utf-8')
name = sys.argv[1]
tot = {}
for path in sorted((HERE / 'sim').glob(f'cmb-{name}-*.json')):
    arm = path.name[len(f'cmb-{name}-'):-5]
    sim = json.loads(path.read_text(encoding='utf-8'))
    recs = json.loads((HERE / 'out' / arm / 'records.json').read_text(encoding='utf-8'))
    c = {'refused': sim['summary'].get('vllm_refused', 0)}
    for rec, row in zip(recs, sim['rows']):
        g = 'cut' if rec['finish'] != 'stop' else 'right' if rec['right'] else 'wrong'
        c[g] = c.get(g, 0) + 1
        st = row['stop'] or {}
        kind = st.get('kind', '')
        if not kind or kind.startswith('ended'):
            continue
        c[g + ' acted'] = c.get(g + ' acted', 0) + 1
        if g != 'cut':
            own = [(e['kind'], e.get('model_tokens')) for e in rec['events'] if e['kind'] in ('loop close', 'loop nudge', 'answer close')]
            print(f'  {arm} {g} {rec["id"]}#{rec["sample"]}: {kind} at {st.get("at")} of {row["tokens"]} | run: {own}')
    print(arm, c)
    for k, v in c.items():
        tot[k] = tot.get(k, 0) + v
print('TOTAL', tot)
