"""gate2_replay.py: gate 2 of the token channel (docs/plans/2026-11-12-mip-token-channel-plan.md, task 7). CPU only.

Round 9's General traces (4B NVFP4 W4A8 · llama.cpp · PC layer: r9a-gen, r9a3-gen, r9b-gen) are walked token by token through the
thinking. At every `=` the model wrote (not inside a put), the channel's Planner is asked exactly as the service asks it, with the
Rollout's state as it stood there (the text, the spans put so far, the echo guard's history, since every earlier `=` was asked too).
The planned text must equal the recorded inline put that follows that `=`, and a `=` with no recorded put must get no plan.
Writes results/token-channel/gate2.txt.  python scripts/pc/gate2_replay.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.igc import tir4b  # noqa: E402
from tools.igc.mip_planner import Planner  # noqa: E402
from tools.igc.token_trie import vocab_pieces  # noqa: E402

W = Path('C:/Users/xenod/mip-work/a100')
HALVES = ('r9a-gen', 'r9a3-gen', 'r9b-gen')
ROWS = ('frozen9-rows.jsonl', 'frozen9a-rows.jsonl', 'frozen9b-rows.jsonl', 'frozen9a-move-rows.jsonl')
GENERAL = dict(inline_calc=True, mathbackend=True, inline_plain=True, inline_style='general', no_digit_join=True, guard_progress=True,
               loop_close=True, answer_close=True, early_cycle=True, shape_cap=1024, cover_cap=32768)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('D:/models/Qwen3.5-4B')
    ids = tir4b.Ids(tok, [tok.convert_tokens_to_ids('<|endoftext|>')])
    pieces = vocab_pieces(tok)
    rows = {}
    for f in ROWS:
        p = W / f
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows[r['id']] = r
    c = Counter()
    differ = []
    for half in HALVES:
        for line in (W / 'pc' / half / 'ids.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            row = rows.get(x['id'])
            if row is None:
                c['no row'] += 1
                continue
            out, spans = x['out_ids'], [s for s in x['spans']]
            puts = {s: (s, e) for s, e, k in spans if k == 'inline value'}
            inside = set()
            for s, e, _ in spans:
                inside.update(range(s, e))
            r = tir4b.Rollout(row, x.get('sample', 0), x['prompt_ids'], tir4b.Arm(**GENERAL), ids, tok, seed=1)
            planner = Planner(r, pieces)
            c['answers'] += 1
            for i, t in enumerate(out):
                if t == ids.think_close:
                    break
                if t not in ids.equals or i in inside:
                    continue
                r.out = out[:i + 1]
                r.spans = [[s, e, k] for s, e, k in spans if e <= i + 1]
                plan = planner.on_trigger(r.text(), 'think')
                rec = puts.get(i + 1)
                recorded = tok.decode(out[rec[0]:rec[1]]) if rec else None
                c['equals'] += 1
                c['recorded'] += recorded is not None
                c['planned'] += plan is not None
                if plan is None and recorded is None:
                    continue
                if plan is not None and recorded is not None and plan.text == recorded:
                    c['same'] += 1
                    continue
                kind = 'planned, not recorded' if recorded is None else 'recorded, not planned' if plan is None else 'different text'
                c[kind] += 1
                if len(differ) < 40:
                    differ.append(f"{half} {x['id']}#{x.get('sample', 0)} at {i}: {kind}: planned {plan.text if plan else None!r} "
                                  f"recorded {recorded!r} | ...{r.text()[-80:]!r}")
    lines = ['gate 2 (token channel): round 9 General traces, every `=` in the thinking asked through the Planner',
             f"answers {c['answers']} (no row {c['no row']}); `=` asked {c['equals']}; recorded puts {c['recorded']}; planned {c['planned']}",
             f"same text {c['same']}; different text {c['different text']}; recorded, not planned {c['recorded, not planned']}; "
             f"planned, not recorded {c['planned, not recorded']}",
             'PASS' if c['same'] == c['recorded'] == c['planned'] else 'DIFFERENCES (listed, read each)'] + differ
    out = ROOT / 'results' / 'token-channel' / 'gate2.txt'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:4]))


if __name__ == '__main__':
    main()
