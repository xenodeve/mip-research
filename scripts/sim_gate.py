"""#307 replay gate: every stored A100 arm replayed (--recorded) with its own flags, read from its arm.json, plus MECHS; 4 at a time.
python sim_gate.py NAME MECHS [ARM...]   -> sim/cmb-NAME-ARM.json"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(os.environ.get('WORK', r'C:\Users\xenod\mip-work\a100'))   # stored arms in out/, replays in sim/ (outside the repo)
WT = str(Path(__file__).resolve().parents[1])   # this repository: the tools/ under test
FLAGS = ('inline_calc', 'mathbackend', 'inline_plain', 'no_digit_join', 'guard_progress', 'loop_close', 'answer_close', 'early_cycle',
         'loop_value', 'candidate_close')
VALUES = ('inline_style', 'suppress_recheck', 'shape_cap', 'cover_cap')


def spec_of(arm):
    a = json.loads((HERE / 'out' / arm / 'arm.json').read_text(encoding='utf-8'))
    parts = [f for f in FLAGS if a.get(f)]
    parts += [f'{k}={a[k]}' for k in VALUES if a.get(k)]
    return parts


def run(name, mechs, arm):
    out = HERE / 'sim' / f'cmb-{name}-{arm}'
    if out.with_suffix('.json').exists():
        return arm, 'kept'
    spec = spec_of(arm)
    for m in mechs.split(','):
        key = m.split('=')[0]
        spec = [s for s in spec if s.split('=')[0] != key] + [m]
    cmd = [sys.executable, '-m', 'tools.igc.sim_replay', '--records', str(HERE / 'out' / arm / 'records.json'), '--ids',
           str(HERE / 'out' / arm / 'ids.jsonl'), '--recorded', '--rows', str(HERE / 'fresh-rows.jsonl'), '--arm', ','.join(spec),
           '--out', str(out.with_suffix('.tmp'))]
    with open(out.with_suffix('.log'), 'w', encoding='utf-8') as log:
        p = subprocess.run(cmd, cwd=WT, stdout=log, stderr=subprocess.STDOUT, env=dict(__import__('os').environ, PYTHONIOENCODING='utf-8'),
                           creationflags=0x08000000)
    if p.returncode:
        return arm, 'FAILED'
    out.with_suffix('.tmp').replace(out.with_suffix('.json'))
    return arm, 'done'


if __name__ == '__main__':
    name, mechs, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
    with ThreadPoolExecutor(4) as ex:
        for arm, how in ex.map(lambda a: run(name, mechs, a), arms):
            print(arm, how, flush=True)
    print('GATE', name, 'DONE')
