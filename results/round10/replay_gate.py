"""replay_gate.py: the simulation gate for round 10's arm B, per mechanism (AGENTS.md: zero misreads, no act on a right answer or every
one read, vllm_refused 0).

Round 9's General traces (4B NVFP4 W4A8 · llama.cpp · PC layer: r9a-gen, r9a3-gen, r9b-gen, 240 answers; r9a2's answers were folded
into r9a) are replayed with sim_replay.replay, `--recorded`, under arm B's flags = round 9's General flags + --question-quote
--approx-put --answer-rewind 2 --letter-count. The replay follows the arm's own puts and stops at the first new act, so each answer
gives at most one new act: where it is, on an answer that finished right or wrong, and whether the value put is right (checked again
here: approx by exact evaluation, letters by str.count on the question's string, the quote by the question's numbers in order).
The control traces are not replayed: round 10 has no control arm, and arm B is built on the General arm.
python -u results/round10/replay_gate.py > results/round10/replay_gate.txt   (-u: the first run lost its last sections to buffering)
"""
import json, re, sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding='utf-8')
from tools.igc import tir4b  # noqa: E402
from tools.igc.sim_replay import replay, arm_kwargs  # noqa: E402
from tools.igc.select_rows import right  # noqa: E402

W = Path('C:/Users/xenod/mip-work/a100')
HALVES = ('r9a-gen', 'r9a3-gen', 'r9b-gen')
ROWS = ('frozen9-rows.jsonl', 'frozen9a-rows.jsonl', 'frozen9b-rows.jsonl', 'frozen9a-move-rows.jsonl')
GENERAL = ('inline_calc,mathbackend,inline_plain,inline_style=general,no_digit_join,guard_progress,loop_close,answer_close,early_cycle,'
           'shape_cap=1024,cover_cap=32768')
ARM_B = GENERAL + ',question_quote,approx_put,answer_rewind=2,letter_count'
NEW = ('question quote', 'approx value', 'letter count', 'answer rewind')
# the replay stops at the first new act, so with all four on an early approx put hides a later rewind: each mechanism is also replayed
# alone on the General arm, which gives its own reach
ALONE = {'question quote': 'question_quote', 'approx value': 'approx_put', 'letter count': 'letter_count', 'answer rewind': 'answer_rewind=2'}


def approx_ok(ev):
    from tools.igc.mathexpr import evaluate, wide
    with wide():
        v = evaluate(ev.get('expr') or '')
    m = re.fullmatch(r'(-?[\d.]+)(?: \\times 10\^\{(-?\d+)\})?', str(ev.get('value') or ''))
    if v is None or m is None:
        return None
    scale = Fraction(10) ** int(m.group(2)) if m.group(2) else Fraction(1)
    places = len(m.group(1).split('.')[1]) if '.' in m.group(1) else 0
    return abs(Fraction(m.group(1)) * scale - v) <= Fraction(1, 2) * Fraction(1, 10 ** places) * scale


def letters_ok(ev, question):
    from tools.igc.letters import letter_case
    s = ev.get('string') or ''
    got = letter_case(question)
    # the string stands in the question as a word, and each count letter_case gives is the string's own (str.count)
    return bool(s) and re.search(r'(?<![A-Za-z])' + re.escape(s) + r'(?![A-Za-z])', question) is not None and got is not None \
        and got[0] == s and all(s.count(ch) == n for ch, n in got[1]) and sum(n for _, n in got[1]) == len(s)


def quote_ok(question):
    from tools.igc.question_read import read_exact
    from tools.igc.question_quote import numbers_of
    got = read_exact(question)
    q = question.split('\n')[0]
    return got is not None and numbers_of(got[0]) == [x for x in numbers_of(q)][:len(numbers_of(got[0]))]


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('D:/models/Qwen3.5-4B')
    IDS = tir4b.Ids(tok, [tok.convert_tokens_to_ids('<|endoftext|>')])
    rows = {}
    for f in ROWS:
        p = W / f
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[r['id']] = r
    for name, spec in [('arm B, all four', ARM_B)] + [(f'{k} alone', GENERAL + ',' + v) for k, v in ALONE.items()]:
        gate(name, spec, rows, tok, IDS)


def gate(name, spec, rows, tok, IDS):
    arm = arm_kwargs(spec)
    acts = defaultdict(Counter)       # kind -> (final, value verdict) -> n
    listed = defaultdict(list)
    refused = answers = no_row = 0
    for half in HALVES:
        d = W / 'pc' / half
        stored = {}
        for line in (d / 'ids.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                x = json.loads(line)
                stored[(x['id'], x.get('sample', 0))] = x['out_ids']
        for rec in json.loads((d / 'records.json').read_text(encoding='utf-8')):
            row = rows.get(rec['id'])
            if row is None:
                no_row += 1
                continue
            rec = dict(rec, question=row['question'], answer=row['answer'])
            ok = right(row, rec['text'])   # today's rule (the scorer fixes and the form rule of 2026-11-11), not the stored flag
            ids = stored.get((rec['id'], rec.get('sample', 0))) or tok.encode(rec['text'], add_special_tokens=False)
            out = replay(rec, ids, arm, tok, IDS, recorded=True)
            answers += 1
            refused += len(out.get('refused', []))
            for ev in out['events']:
                if ev['kind'] not in NEW:
                    continue
                if ev['kind'] == 'approx value':
                    v = approx_ok(ev)
                elif ev['kind'] == 'letter count':
                    v = letters_ok(ev, row['question'])
                elif ev['kind'] == 'question quote':
                    v = quote_ok(row['question'])
                else:
                    v = 'no value'
                final = 'right' if ok else 'wrong'
                acts[ev['kind']][(final, {True: 'value right', False: 'VALUE WRONG', None: 'unchecked'}.get(v, v))] += 1
                listed[ev['kind']].append(f"{half} {rec['id']}#{rec.get('sample', 0)} {final} at {ev['at']}/{len(ids)} "
                                          f"{json.dumps({k: ev[k] for k in ('expr', 'value', 'string', 'copy', 'at_loop', 'cut_tokens') if ev.get(k) is not None}, ensure_ascii=False)[:200]}")
    print(f'\n==== {name}: {spec}')
    print(f'{answers} answers replayed ({no_row} without a row), vllm_refused {refused}')
    for kind in NEW:
        c = acts.get(kind, Counter())
        print(f'\n{kind}: {sum(c.values())} first acts; ' + ', '.join(f'{f} answers, {v}: {n}' for (f, v), n in sorted(c.items())))
        for line in listed.get(kind, []):
            print('   ' + line)


if __name__ == '__main__':
    main()
