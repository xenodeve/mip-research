"""#307 does the model push a put away? For every inline put of an A100 arm (the exact injected spans in ids.jsonl), the 40 tokens the model
wrote next, decoded: a rejection word in them (wait / no, / not right / incorrect / mistake / actually / let me recompute|recheck) counts.
The same count after the model's OWN `= value` in the no-MIP control arm is the base rate: a rejection there is the model doubting itself.
python put_reject.py ARM [ARM...]"""
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent  # a folder with out/<arm>/ A100 records (records.json, ids.jsonl); not in this repo
TOK = AutoTokenizer.from_pretrained(r'D:\models\Qwen3.5-4B')
REJECT = re.compile(r"\bwait\b|\bno,|\bnot right\b|\bincorrect\b|\bmistake\b|\bwrong\b|\bactually\b|let me (?:re-?compute|re-?check|recalculate|double)", re.I)
EQ_VALUE = re.compile(r'=\s*-?\d[\d,]*(?:\.\d+)?')

for arm in sys.argv[1:]:
    puts = rej = 0
    examples = []
    for line in (HERE / 'out' / arm / 'ids.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        out, spans = row['out_ids'], row.get('spans') or []
        inline = [s for s in spans if (s[2] if len(s) > 2 else 'inline value') == 'inline value'] if spans and len(spans[0]) > 2 else spans
        if spans:
            for s in inline:
                end = s[1]
                nxt = TOK.decode(out[end:end + 40])
                puts += 1
                if REJECT.search(nxt):
                    rej += 1
                    if len(examples) < 6:
                        examples.append((row['id'], TOK.decode(out[max(0, s[0] - 25):s[1]]).replace('\n', ' ')[-90:], nxt.replace('\n', ' ')[:90]))
        else:   # the control: the model's own `= value`, base rate
            text = TOK.decode(out)
            for m in EQ_VALUE.finditer(text):
                puts += 1
                if REJECT.search(text[m.end():m.end() + 160]):
                    rej += 1
    print(f'{arm}: {rej} of {puts} ({100 * rej / max(puts, 1):.1f} %) followed by a rejection word within 40 tokens')
    for e in examples:
        print('   ', e)
