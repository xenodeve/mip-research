"""round10_deep.py: the reads of round 10's short run the first read left out (4B NVFP4 W4A8 · llama.cpp+MIP · PC layer, both cards).

1. tokens, B against CH, pair by pair; and after every value MIP put (A, B: the injected spans) or steered (CH: the steered events), the
   40 tokens the model wrote next: a rejection word in them (put_reject.py's words) counts as a doubt of the value;
2. the fixed-question line: per answer, the thinking tokens after the line, a doubt in the 40 tokens after it, and the result;
3. every answer where the rule and the API reader disagree, with both values;
4. arm B's values (text puts) checked again by gate 3's checker, as CH's are (A's values are round 9's General's, already gated).
  python scripts/pc/round10_deep.py > results/round10/short-deep.txt
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'pc'))
sys.stdout.reconfigure(encoding='utf-8')
from gate3_values import value_right   # noqa: E402

W = Path('C:/Users/xenod/mip-work/a100')
CARDS = ('r10-q-5060', 'r10-q-4070')
REJECT = re.compile(r"\bwait\b|\bno,|\bno\.|\bnot right\b|\bincorrect\b|\bmistake\b|\bwrong\b|\bactually\b|"
                    r"let me (?:re-?compute|re-?check|recalculate|double|verify)", re.I)


def load(arm):
    recs, ids, tri = {}, {}, {}
    for d in CARDS:
        base = W / 'pc' / d / arm
        for r in json.loads((base / 'records.json').read_text(encoding='utf-8')):
            recs[(r['id'], r.get('sample', 0))] = r
        for line in (base / 'ids.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                x = json.loads(line)
                ids[(x['id'], x.get('sample', 0))] = x
        for t in json.loads((base / 'triage.json').read_text(encoding='utf-8')):
            tri[(t['id'], t['sample'])] = t
    return recs, ids, tri


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('C:/Users/xenod/mip-work/q4b-hf')
    rows = {}
    for f in ('r10q-a-rows.jsonl', 'r10q-b-rows.jsonl'):
        for line in (W / f).read_text(encoding='utf-8').splitlines():
            if line.strip():
                x = json.loads(line)
                rows[x['id']] = x
    arms = {a: load(a) for a in ('a', 'b', 'ch')}

    print('== 1. tokens, B against CH ==')
    keys = sorted(arms['b'][0])
    more = sum(arms['ch'][0][k]['tokens'] > arms['b'][0][k]['tokens'] for k in keys)
    print(f"  CH longer on {more} of {len(keys)} pairs; B {sum(arms['b'][0][k]['tokens'] for k in keys):,} tokens, "
          f"CH {sum(arms['ch'][0][k]['tokens'] for k in keys):,}, A {sum(arms['a'][0][k]['tokens'] for k in keys):,}")
    print('\n== 1b. a doubt in the 40 tokens after a value (put in A and B, steered in CH) ==')
    for arm in ('a', 'b', 'ch'):
        recs, ids, _ = arms[arm]
        c = Counter()
        for k, x in ids.items():
            out = x['out_ids']
            if arm == 'ch':
                ends = []
                for e in recs[k]['events']:
                    if e.get('kind') == 'steered value' and e.get('done') and e.get('what') in ('inline value', 'long value', 'approx value'):
                        n = len(tok.encode(' ' + e['value'], add_special_tokens=False))
                        ends.append(e['at'] + n)
            else:
                ends = [s[1] for s in (x.get('spans') or []) if len(s) > 2 and s[2] in ('inline value', 'approx value')]
            for end in ends:
                nxt = tok.decode(out[end:end + 40])
                c['values'] += 1
                c['doubt'] += bool(REJECT.search(nxt))
        print(f"  {arm}: {c['doubt']} of {c['values']} values followed by a doubt word ({100 * c['doubt'] / max(1, c['values']):.1f} %)")

    print('\n== 2. the fixed-question line: thinking tokens after it, a doubt after it, the result ==')
    for arm in ('b', 'ch'):
        recs, ids, tri = arms[arm]
        c = Counter()
        for k in sorted(recs):
            r = recs[k]
            line = next((e for e in r['events'] if e.get('kind') == 'question line' or str(e.get('what', '')).startswith('question line')),
                        None)
            if line is None:
                continue
            out = ids[k]['out_ids']
            text_ids = tok.encode(r['text'][:0], add_special_tokens=False)
            # the line is at token 0; its end is its first newline
            first = tok.decode(out[:80])
            nl = len(tok.encode(first[:first.find('\n') + 1], add_special_tokens=False)) if '\n' in first else 0
            close = next((i for i, t in enumerate(out) if t == tok.convert_tokens_to_ids('</think>')), len(out))
            after = close - nl
            doubt = bool(REJECT.search(tok.decode(out[nl:nl + 40])))
            corrected = (tri.get(k, {}).get('verdict') or {}).get('verdict') == 'same' if tri.get(k, {}).get('verdict') else r['right']
            c['answers'] += 1
            c['closed within 1,000'] += after <= 1000
            c['doubt after the line'] += doubt
            c['right (corrected)'] += corrected
            print(f"  {arm} {k[0]}#{k[1]}: {after:,} thinking tokens after the line, doubt {'yes' if doubt else 'no'}, "
                  f"{'right' if corrected else 'WRONG'}; next: {tok.decode(out[nl:nl + 14])!r}")
        print(f'  {arm}: {dict(c)}')

    print('\n== 3. the rule and the API reader disagree ==')
    for arm in ('a', 'b', 'ch'):
        recs, _, tri = arms[arm]
        for k, t in sorted(tri.items()):
            if t.get('route') == 'disagree':
                v = t['verdict'] or {}
                print(f"  {arm} {k[0]}#{k[1]} gold {rows[k[0]]['answer']}: rule {'right' if t['rule'] else 'wrong'}, reader "
                      f"{v.get('verdict')} ({v.get('value')!s:.40})")

    print("\n== 4. arm B's values, checked again by gate 3's checker ==")
    recs, ids, _ = arms['b']
    c, bad = Counter(), []
    for k, r in recs.items():
        spans = [s for s in (ids[k].get('spans') or []) if len(s) > 2]
        out = ids[k]['out_ids']
        for e in r['events']:
            kind = e.get('kind')
            if kind in ('inline value', 'approx value', 'question line') and 'expr' in e:
                what = 'long value' if e.get('long') else kind
                ok = value_right(dict(what=what, expr=e['expr'], value=e['value']), rows[k[0]]['question'], rows[k[0]]['answer'])
            elif kind in ('letter count', 'question quote'):
                s = next((s for s in spans if s[2] == kind and s[0] >= e['at']), None)
                text = tok.decode(out[s[0]:s[1]]).strip() if s else ''
                if kind == 'letter count':
                    ok = value_right(dict(what=kind, expr=e.get('string', ''), value=text), rows[k[0]]['question'])
                else:
                    m = re.search(r'\$(.*)\$', text)
                    q = m.group(1) if m else ''
                    ok = value_right(dict(what=kind, expr=q, value=q), rows[k[0]]['question'], rows[k[0]]['answer'])
            else:
                continue
            c[(kind, 'right' if ok else 'unchecked' if ok is None else 'WRONG')] += 1
            if ok is not True:
                bad.append(f"  {k[0]}#{k[1]} {kind}: {e.get('expr', e.get('string'))!s:.60} = {e.get('value')!s:.40} -> {ok}")
    for x in sorted(c.items()):
        print(f'  {x[0][0]}: {x[0][1]} {x[1]}')
    print('\n'.join(bad[:40]))


if __name__ == '__main__':
    main()
