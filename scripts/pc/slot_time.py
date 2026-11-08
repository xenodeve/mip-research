"""Where a llama-server slot's time goes, from its own log: per slot, the wall clock split into
  decode   generating tokens (eval time)
  read     reading the tokens added since the last request (prompt eval: an inline value, the cut text)
  gap      between a release and the next launch on the same slot (the client: Python, HTTP, the rules)
  empty    no request on the slot at all
plus request counts and the decode rate. python slot_time.py LOG [minutes from the end, default 15]"""
import re
import sys
from collections import defaultdict

log, minutes = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 15
T = re.compile(r'(\d+)\.(\d+)\.(\d+)\.\d+ I slot\s+(\S+): id\s+(\d+) \| task (-?\d+) \| (.*)')
ev = []
for line in open(log, encoding='utf-8', errors='replace'):
    m = T.match(line)
    if m:
        t = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1000
        ev.append((t, m.group(4), int(m.group(5)), int(m.group(6)), m.group(7)))
end = ev[-1][0]
start = end - minutes * 60
tasks = defaultdict(dict)
for t, kind, slot, task, rest in ev:
    if task < 0:
        continue
    d = tasks[task]
    d['slot'] = slot
    if kind == 'launch_slot_':
        d['launch'] = t
    elif kind == 'release':
        d['release'] = t
    elif kind == 'print_timing':
        m = re.search(r'prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens', rest)
        if m:
            d['read_ms'], d['read_tok'] = float(m.group(1)), int(m.group(2))
        m = re.search(r'^\s*eval time =\s+([\d.]+) ms /\s+(\d+) tokens', rest)
        if m:
            d['dec_ms'], d['dec_tok'] = float(m.group(1)), int(m.group(2))
done = [d for d in tasks.values() if 'launch' in d and d.get('release', end) > start]
for d in done:   # clip each request to the window; a request still running ends at the window's end
    full = d.get('release', end) - d['launch']
    d['release'] = min(d.get('release', end), end)
    d['launch'] = max(d['launch'], start)
    f = (d['release'] - d['launch']) / full if full > 0 else 1.0
    for k in ('dec_ms', 'read_ms', 'dec_tok', 'read_tok'):
        if k in d:
            d[k] *= f
    if 'dec_ms' not in d:   # still decoding: all of it decode, at the slot's rate (counted as busy, tokens unknown)
        d['dec_ms'] = 1000 * (d['release'] - d['launch'])
per_slot = defaultdict(list)
for d in done:
    per_slot[d['slot']].append(d)
win = end - start
tot = defaultdict(float)
for slot, ds in sorted(per_slot.items()):
    ds.sort(key=lambda d: d['launch'])
    dec = sum(d.get('dec_ms', 0) for d in ds) / 1000
    read = sum(d.get('read_ms', 0) for d in ds) / 1000
    busy = sum(d['release'] - d['launch'] for d in ds)
    gaps = [b['launch'] - a['release'] for a, b in zip(ds, ds[1:])]
    gap = sum(g for g in gaps if g < 2.0)        # the client turning a request around
    empty = win - busy - gap
    for k, v in (('decode', dec), ('read', read), ('other busy', busy - dec - read), ('gap', gap), ('empty', empty)):
        tot[k] += v
    tot['requests'] += len(ds)
    tot['tokens'] += sum(d.get('dec_tok', 0) for d in ds)
    tot['read tokens'] += sum(d.get('read_tok', 0) for d in ds)
    tot['short'] += sum(1 for d in ds if d.get('dec_tok', 0) <= 1)
    tot['gap_n'] += sum(1 for g in gaps if g < 2.0)
slots = len(per_slot)
wall = win * slots
print(f'{log}: last {minutes:.0f} min, {slots} slots, {int(tot["requests"])} requests ({int(tot["short"])} with <= 1 token decoded), '
      f'{int(tot["tokens"])} tokens decoded = {tot["tokens"] / win:.0f} tok/s')
for k in ('decode', 'read', 'other busy', 'gap', 'empty'):
    print(f'  {k:10} {tot[k]:7.0f} s  {100 * tot[k] / wall:5.1f} %')
print(f'  decode rate {tot["tokens"] / tot["decode"]:.1f} tok/s per slot while decoding; read {tot["read tokens"] / max(tot["read"], 1e-9):.1f} tok/s; '
      f'mean gap {1000 * tot["gap"] / max(tot["gap_n"], 1):.0f} ms; requests per minute {tot["requests"] / minutes:.0f}')
