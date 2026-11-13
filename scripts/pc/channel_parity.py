"""channel_parity.py: the letter counts and the question quote steered through the channel against the text arm's puts, on round 9's
General traces (4B NVFP4 W4A8 · llama.cpp · PC layer: r9a-gen, r9a3-gen, r9b-gen). CPU only.

For every answer, the stored tokens are fed in CHUNK slices (a channel request's length) to two Rollouts with the same flags, one with
--channel. Letter counts: the text arm's first put against the channel planner's start plan. Question quote: the text arm's first quote
(where, what) against the channel arm's cut and the plan it hands the channel. Parity is the same text at the same token position.
The quote at a loop's close is not replayed here (it needs the driver's loop check; tools/test_igc_tir4b.py pins it).
python scripts/pc/channel_parity.py > results/token-channel/channel-parity.txt
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding='utf-8')
from tools.igc import tir4b  # noqa: E402
from tools.igc.mip_planner import Planner  # noqa: E402

W = Path('C:/Users/xenod/mip-work/a100')
HALVES = ('r9a-gen', 'r9a3-gen', 'r9b-gen')
ROWS = ('frozen9-rows.jsonl', 'frozen9a-rows.jsonl', 'frozen9b-rows.jsonl', 'frozen9a-move-rows.jsonl')


def arm(channel):
    return tir4b.Arm(question_quote=True, letter_count=True, think_only=True, channel=channel)


def first_quote(r, out_ids):
    """(position, text) of the first quote: the text arm's put, or the channel arm's cut and handed plan."""
    for i in range(0, len(out_ids), tir4b.CHUNK):
        if r.finish:
            break
        r.take(list(out_ids[i:i + tir4b.CHUNK]))
        for e in r.events:
            if e['kind'] == 'question quote':
                s = next(sp for sp in r.spans if sp[2] == 'question quote')
                return s[0], r.tok.decode(r.out[s[0]:s[1]])
        if r.steer_queue:
            return len(r.out), r.steer_queue[0][0]
    return None


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('C:/Users/xenod/mip-work/q4b-hf')
    ids = tir4b.Ids(tok, [tok.convert_tokens_to_ids('<|endoftext|>')])
    pieces = [tok.decode([i]) for i in range(len(tok))]
    rows = {}
    for f in ROWS:
        p = W / f
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    x = json.loads(line)
                    rows[x['id']] = x
    c, diff = Counter(), []
    for half in HALVES:
        d = W / 'pc' / half
        stored = {}
        for line in (d / 'ids.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                x = json.loads(line)
                stored[(x['id'], x.get('sample', 0))] = x['out_ids']
        for rec in json.loads((d / 'records.json').read_text(encoding='utf-8')):
            row = rows.get(rec['id'])
            out_ids = stored.get((rec['id'], rec.get('sample', 0)))
            if row is None or out_ids is None:
                c['no row or ids'] += 1
                continue
            c['answers'] += 1
            name = f"{half}:{rec['id']}#{rec.get('sample', 0)}"
            text_r = tir4b.Rollout(row, 0, [1], arm(False), ids, tok, seed=1)
            chan_r = tir4b.Rollout(row, 0, [1], arm(True), ids, tok, seed=1)
            put = tok.decode(text_r.out) if any(e['kind'] == 'letter count' for e in text_r.events) else None
            plan = Planner(chan_r, pieces).at_start(row['question'])
            steered = plan.text if plan is not None and 'letter count' in plan.kind else None
            if put or steered:
                c['letter count'] += 1
                c['letter count same' if put == steered else 'LETTER COUNT DIFFERS'] += 1
                if put != steered:
                    diff.append(f'LETTER {name}: put {put!r} steered {steered!r}')
            a = first_quote(text_r, out_ids)
            b = first_quote(chan_r, out_ids)
            if a or b:
                c['question quote'] += 1
                c['question quote same' if a == b else 'QUOTE DIFFERS'] += 1
                if a != b:
                    diff.append(f'QUOTE {name}: text {a!r} channel {b!r}')
            # the quote at a loop's close, at round 9's own close positions: the text arm puts quote and close; the channel arm hands
            # the quote over, the model is taken to write it as steered, and the close follows. Same text after the close
            for e in rec.get('events', []):
                if e.get('kind') != 'loop close':
                    continue
                texts = []
                for channel in (False, True):
                    r = tir4b.Rollout(row, 0, [1], tir4b.Arm(question_quote=True, think_only=True, loop_close=True, channel=channel),
                                      ids, tok, seed=1)
                    r.out = list(out_ids[:e['at']])
                    r.on_loop()
                    plan = r.take_pending()
                    if plan is not None:
                        r.take(tok.encode(plan[0], add_special_tokens=False))
                    texts.append(tok.decode(r.out[e['at']:]) if r.quoted else None)
                if texts[0] is None and texts[1] is None:
                    break   # no quote for this question
                c['quote at a loop close'] += 1
                c['quote at a loop close same' if texts[0] == texts[1] else 'LOOP QUOTE DIFFERS'] += 1
                if texts[0] != texts[1]:
                    diff.append(f'LOOP QUOTE {name}: text {texts[0]!r} channel {texts[1]!r}')
                break   # the first close only: the quote is put once an answer
    print(dict(c))
    for x in diff:
        print(x)


if __name__ == '__main__':
    main()
