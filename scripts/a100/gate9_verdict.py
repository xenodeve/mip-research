"""Round 9 gate verdict from the replays of sim_gate.py NAME (in $WORK/sim). An act on an answer that finished right is
- a parting: the replay put a value the record does not hold (`inline value`), so it cannot follow on; not an act of a closer;
- sooner: a close of the same loop the run itself closed later (`run closed later`), as gate A+B+V7 read them;
- read: listed in gate9_accepted.txt (arm id#sample kind at, then the reason), each one read in its text before it was listed;
- or unexplained, which fails the gate. Also fails on vllm_refused > 0.
python gate9_verdict.py NAME   -> prints the table and `VERDICT PASS|FAIL`"""
import json
import os
import sys
from pathlib import Path

WORK = Path(os.environ.get('WORK', r'C:\Users\xenod\mip-work\a100'))
ACCEPTED = Path(__file__).with_name('gate9_accepted.txt')
OWN_CLOSES = ('loop close', 'loop nudge', 'answer close')


def accepted():
    keys = set()
    for line in ACCEPTED.read_text(encoding='utf-8').splitlines() if ACCEPTED.exists() else []:
        if line.strip() and not line.startswith('#'):
            p = line.split('|')[0].split()   # arm id#sample kind (may be two words) at
            keys.add((p[0], p[1], ' '.join(p[2:-1]), p[-1]))
    return keys


def main(name):
    ok_keys = accepted()
    tot = dict(refused=0, right=0, parting=0, sooner=0, read=0, unexplained=0)
    for path in sorted((WORK / 'sim').glob(f'cmb-{name}-*.json')):
        arm = path.name[len(f'cmb-{name}-'):-5]
        sim = json.loads(path.read_text(encoding='utf-8'))
        recs = json.loads((WORK / 'out' / arm / 'records.json').read_text(encoding='utf-8'))
        tot['refused'] += sim['summary'].get('vllm_refused', 0)
        for rec, row in zip(recs, sim['rows']):
            if rec['finish'] != 'stop' or not rec['right']:
                continue
            tot['right'] += 1
            st = row['stop'] or {}
            kind = st.get('kind', '')
            if not kind or kind.startswith('ended'):
                continue
            at = st.get('at')
            own = [e.get('model_tokens') for e in rec['events'] if e['kind'] in OWN_CLOSES]
            if kind == 'inline value':
                how = 'parting'
            elif own and kind != 'answer close':
                how = 'sooner'
            elif (arm, f'{rec["id"]}#{rec["sample"]}', kind, str(at)) in ok_keys:
                how = 'read'
            else:
                how = 'unexplained'
            tot[how] += 1
            if how != 'parting':
                print(f'{how:11} {arm} {rec["id"]}#{rec["sample"]} {kind} at {at} of {row["tokens"]} | run closes: {own}')
    verdict = 'PASS' if tot['unexplained'] == 0 and tot['refused'] == 0 and tot['right'] else 'FAIL'
    print('TOTAL', tot)
    print('VERDICT', verdict)


if __name__ == '__main__':
    main(sys.argv[1])
