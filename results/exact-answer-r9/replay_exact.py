"""replay_exact.py: --exact-answer replayed on round 9's stored answers (4B NVFP4 W4A8 · llama.cpp · PC layer).

For every answer of both arms: where the question states one computation (question_read), what --exact-answer would do.
- `cue`: the answer has an `answer close` event, so the value after the cue would be the question's exact value.
- `box`: no cue; the last box of the answer is a shortening of the exact value (tir4b._shortening), so it would be rewritten.
An answer that ended `stuck` (no box, no cue) is not touched: the flag acts only at an answer's end or at the cue.
Right is the rule (select_rows.right) on the rewritten text; the API reader's verdict of the stored text is shown beside it.

python results/exact-answer-r9/replay_exact.py <pc dir> <rows a> <rows b>  -> replay.json beside this file
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.igc import tir4b  # noqa: E402
from tools.igc.question_read import read  # noqa: E402
from tools.igc.select_rows import right  # noqa: E402


def rows_of(path):
    rows = [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
    return {r['id']: r for r in rows}


def last_box(answer):
    at = answer.rfind('\\boxed{')
    if at < 0:
        return None
    inner = tir4b._box_inner(answer, at + 7)
    return None if inner is None else (at, inner)


def main():
    pc, rows_a, rows_b = Path(sys.argv[1]), rows_of(sys.argv[2]), rows_of(sys.argv[3])
    out = []
    for half, rows in (('a', rows_a), ('b', rows_b)):
        for arm in ('gen', 'ctrl'):
            d = pc / f'r9{half}-{arm}'
            verdicts = json.loads((d / 'verdicts.json').read_text(encoding='utf-8'))
            for rec in json.loads((d / 'records.json').read_text(encoding='utf-8')):
                row = rows[rec['id']]
                got = read(row.get('question') or '')
                if got is None or not hasattr(got[1], 'denominator'):
                    continue
                key = f"{rec['id']}#{rec['sample']}"
                text = rec['text']
                kinds = [e['kind'] for e in rec.get('events') or []]
                start = text.rfind('</think>') + len('</think>') if '</think>' in text else 0
                value = tir4b.spelled(got[1])
                act, new, copied = None, text, None
                if 'answer close' in kinds:
                    cue = text.rfind(tir4b.ANSWER_CLOSE)
                    act, copied = 'cue', text[cue + len(tir4b.ANSWER_CLOSE):][:60]
                    new = text[:cue] + tir4b.ANSWER_CLOSE + '{' + value + '}$.'
                elif rec['finish'] == 'stop' and got[1].denominator != 1:
                    box = last_box(text[start:])
                    if box is not None:
                        at, inner = (start + box[0], start + box[1])
                        written = text[at + 7:inner]
                        if tir4b._shortening(written, got[1]):
                            act, copied = 'box', written
                            new = text[:at + 7] + value + text[inner:]
                if act is None:
                    continue
                was, now = rec['right'], right(row, new)
                out.append(dict(key=key, half=half, arm=arm, act=act, copied=copied, value=value, gold=row['answer'],
                                right_before=was, right_after=now, api_before=(verdicts.get(key) or {}).get('verdict')))
    (Path(__file__).parent / 'replay.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    for arm in ('gen', 'ctrl'):
        acts = [o for o in out if o['arm'] == arm]
        print(arm, 'acts', len(acts), '| on right answers', sum(o['right_before'] for o in acts),
              '| right after', sum(o['right_after'] for o in acts), '| wrong -> right', sum(o['right_after'] and not o['right_before'] for o in acts),
              '| right -> wrong', sum(o['right_before'] and not o['right_after'] for o in acts))
        for o in acts:
            print('   ', o['key'], o['act'], repr(o['copied'])[:40], '->', o['value'], 'gold', o['gold'], o['right_before'], '->', o['right_after'],
                  'api', o['api_before'])


if __name__ == '__main__':
    main()
