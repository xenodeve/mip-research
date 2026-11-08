"""Which of the loop guard's three checks ended each `stuck` row (#307 A100), from the stored token ids, and whether the text was still
making progress there. python which_guard.py ARM_DIR

  cycle      the tail is one exact cycle, 20 rounds (8 past 512 tokens): the text repeats itself, a real loop
  line_back  one line of 12+ tokens came back 16 times
  counting   with every digit masked, the tail is one cycle of 64 rounds: the same line shape with other numbers, which may be
             progress (trial division, a list) or counting that cannot end
progress: the distinct lines in the last 64 lines (masked digits not), so a counting stop whose lines all differ was still moving."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc.external_score import counting, line_back  # noqa: E402


def cycle(tokens, longest=4096, rounds=20):
    for period in range(1, longest + 1):
        count = rounds if period <= 512 else 8
        if len(tokens) < period * count or tokens[-1] != tokens[-1 - period]:
            continue
        tail = tokens[-period * count:]
        if tail == tail[:period] * count:
            return period
    return 0


arm = Path(sys.argv[1])
records = {(r['id'], r['sample']): r for r in json.loads((arm / 'records.json').read_text(encoding='utf-8'))}
kinds = collections.Counter()
for line in (arm / 'ids.jsonl').read_text(encoding='utf-8').splitlines():
    d = json.loads(line)
    r = records.get((d['id'], d['sample']))
    if not r or r['finish'] != 'stuck':
        continue
    out = d['out_ids']
    p = cycle(out)
    kind = f'cycle/{p}' if p else 'counting' if counting(out) else 'line_back' if line_back(out) else 'none?'
    lines = [l.strip() for l in r['text'].splitlines() if l.strip()][-64:]
    distinct = len(set(lines))
    kinds[kind.split('/')[0] + (' (lines all differ)' if kind == 'counting' and distinct >= 56 else '')] += 1
    print(f"{d['id']:28} s{d['sample']} tok {r['tokens']:6} {kind:10} distinct lines {distinct:2}/64 | {lines[-1][:80] if lines else ''}")
print(dict(kinds))
