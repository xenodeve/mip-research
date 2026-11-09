"""Simulate the decide layer v1 (tools/igc/loop_class.py) on stored records, no model and no GPU.

1. The labelled closes: at each General answer's first close, classify the loop and compare with the hand labels
   (results/decide-layer-r9/labels.json), and say which settle line would be put and whether its content is exact.
2. Control's stuck loops: the same classifier at the point control stopped, on an arm with no closes (a different population).
3. Every settle line's value against the row's gold, where the line carries a value.

python -m tools.igc.decide_sim <pc dir> <labels.json> --rows a=<rows.jsonl> --rows b=<rows.jsonl> [--out <json>]"""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from tools.igc.loop_class import classify, settle
from tools.igc.mathexpr import evaluate

LOOP = '\n</think>\n\n'
CUE = '\n\nThe final answer is $\\boxed{'


def first_close(record):
    """The text before the first close of a General answer, and the close's kind."""
    kinds = [e['kind'] for e in record['events'] if e['kind'] in ('loop close', 'answer close')]
    if not kinds:
        return None, None
    text = record['text']
    cut = text.find(LOOP) if kinds[0] == 'loop close' else text.rfind(CUE)
    return text[:cut], kinds[0]


def gold_value(answer):
    v = evaluate(answer.replace('\\frac', '\\frac'))
    return v


def value_in(line):
    """The value a settle line states after its last `=`, or None."""
    if not line or '=' not in line:
        return None
    body = line.split('$')[1] if line.count('$') >= 2 else line
    return evaluate(body.rsplit('=', 1)[1].split(';')[0].strip())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('pc')
    ap.add_argument('labels')
    ap.add_argument('--rows', action='append', default=[])
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    labels = json.load(open(a.labels, encoding='utf-8'))
    questions = {}
    for spec_ in a.rows:
        _, path = spec_.split('=', 1)
        for line in open(path, encoding='utf-8'):
            row = json.loads(line)
            questions[row['id']] = (row['question'], row['answer'])
    pc = Path(a.pc)
    out = {'labelled': [], 'control': []}
    for h in 'ab':
        for arm in ('gen', 'ctrl'):
            for r in json.load(open(pc / f'r9{h}-{arm}' / 'records.json', encoding='utf-8')):
                key = f"{r['id']}#{r['sample']}"
                q, gold = questions[r['id']]
                if arm == 'gen':
                    before, kind = first_close(r)
                    if before is None:
                        continue
                    target = 'labelled'
                else:
                    if r['finish'] != 'stuck':
                        continue
                    before, kind, target = r['text'], 'stuck', 'control'
                cls, detail = classify(before, q)
                line = settle(cls, detail, q)
                v = value_in(line) if cls in ('C', 'D') else None
                g = gold_value(gold)
                out[target].append(dict(key=key, half=h, kind=kind, label=labels.get(key) if arm == 'gen' else None, cls=cls,
                                        detail=(detail or '')[:160], settle=line, value_is_gold=(v == g) if v is not None and g is not None else None))
    lab = out['labelled']
    agree = sum(x['cls'] == x['label'] for x in lab)
    print(f'labelled closes: {len(lab)}; classifier = hand label on {agree} of {len(lab)}')
    conf = Counter((x['label'], x['cls']) for x in lab)
    classes = 'ABCDEF'
    print('hand \\ classifier | ' + ' | '.join(classes))
    for c in classes:
        print(f'{c} | ' + ' | '.join(str(conf[(c, d)]) for d in classes) + f' | of {sum(conf[(c, d)] for d in classes)}')
    for name, rows in (('labelled', lab), ('control stuck', out['control'])):
        by = Counter(x['cls'] for x in rows)
        with_line = Counter(x['cls'] for x in rows if x['settle'])
        gold = Counter(x['cls'] for x in rows if x['value_is_gold'])
        valued = Counter(x['cls'] for x in rows if x['value_is_gold'] is not None)
        print(f'{name}: {len(rows)} loops; by class ' + ', '.join(f'{c} {by[c]}' for c in classes) +
              '; settle line ' + ', '.join(f'{c} {with_line[c]}/{by[c]}' for c in classes if by[c]) +
              '; value = gold ' + ', '.join(f'{c} {gold[c]}/{valued[c]}' for c in 'CD' if valued[c]))
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
