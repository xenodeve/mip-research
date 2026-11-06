"""sim_replay.py: drive tools/igc/tir4b.py's Rollout, the real state machine, with a stored answer as the model, to see what an arm's
software would do on that text: every value it puts in, every correction, close and nudge, and where.

The stored answers come from runs sampled at T > 0 with continuous batching: two arms with the same seed and prompt part within the first
few hundred characters (a100_262k ctrl and inlineclose: all 48), so no stored arm is what another arm would have written. The model here
writes the stored tokens, and the replay follows them while they stay what the model would write:

- a value put in that equals the model's next value: the model's own value is skipped and the text goes on;
- one that equals a later value of the same chain (`= 4 \\times 6 = 24`, 24 put in): the steps between are skipped too, the tokens
  software saved;
- anything else (another value, a correction, a close, a nudge, an echo of the model's own text cut): the replay stops there. What the
  model would write next is not in the log; the event, the tokens left, and whether the final answer was already written are recorded.

python -m tools.igc.sim_replay --records RUN/records.json [--ids RUN/ids.jsonl] --tokenizer D:/models/Qwen3.5-4B
       --arm inline_calc,mathbackend,inline_plain,loop_close --out sim.json
"""
import argparse
import collections
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.igc import tir4b  # noqa: E402
from tools.igc import exactexpr  # noqa: E402
from tools.igc.unified_detect import evaluate_exact  # noqa: E402

_STOP = re.compile(r'\$|\n|,\s|;|\.\s|\.$|\\quad|\\text|\\implies|\\Rightarrow|\\approx|\\\\|\\]|\\\)|\?| or | and | so | which'
                   r'|\\neq?\b|\\leq?\b|\\geq?\b|\\equiv|\\pmod|\\to\b|->|→|[<>≤≥≠]|\\pm|\s=|='
                   r'|≡|\s*\bmod\b|\s+(?:rem|remainder|R)\b|\s+[A-Za-z]{2,}')   # `16807≡7 mod25`, `6 remainder 1`, `1.4867 so` (#307)
_RELATION = re.compile(r'(?<![<>!=\\])=(?!=)')
_BOXED = re.compile(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}')


def _value(text):
    if not text:
        return None
    v = evaluate_exact(text)
    if v is None:
        return None
    return v if isinstance(v, exactexpr.Value) else exactexpr.Value.of(v)


def _as_values(said):
    """The values a put stands for, any of which the model may write next: the General Style's `16807 ≡ 7` is 16807 or 7, `6 rem 1` is 6,
    `1.4866...` is 1.4866 (cut) or 1.4867 (rounded) (#307). One value for every other put; [] when none can be read."""
    m = re.fullmatch(r'(-?\d+)\s*(?:≡|\\equiv)\s*(-?\d+)', said)
    if m:
        return [exactexpr.Value.of(int(m.group(1))), exactexpr.Value.of(int(m.group(2)))]
    m = re.fullmatch(r'(-?\d+)\s*(?:\\text\{\s*rem\s*\}|rem)\s*\d+', said)
    if m:
        return [exactexpr.Value.of(int(m.group(1)))]
    m = re.fullmatch(r'(-?)(\d+)\.(\d+)\.\.\.', said)
    if m:
        cut = Fraction(m.group(2) + '.' + m.group(3))
        up = cut + Fraction(1, 10 ** len(m.group(3)))
        sign = -1 if m.group(1) else 1
        return [exactexpr.Value.of(sign * cut), exactexpr.Value.of(sign * up)]
    v = _as_value(said)
    return [v] if v is not None else []


def _as_value(said):
    v = exactexpr.evaluate(said)
    if v is None:
        try:
            return exactexpr.Value.of(Fraction(said))
        except (ValueError, ZeroDivisionError):
            return None
    return v


def chain_skip(after, put):
    """The characters of `after` (the model's text after its `=`) the value `put` stands for: the model's next value when it is `put`,
    or a later value of the same chain on that line (`4 \\times 6 = 24`). None when the model never writes `put` there."""
    line = after.split('\n', 1)[0]
    at = 0
    for _ in range(8):   # at most 8 links of one chain
        lead = at + len(line[at:]) - len(line[at:].lstrip())
        m = _STOP.search(line, lead)
        end = m.start() if m else len(line)
        piece = line[at:end].strip()
        v = _value(piece) if piece else None
        if v is not None and v != put:
            return None   # the model wrote another value: a slip of its own, or software read the text otherwise
        if v == put and _plain(piece, put):
            return len(line[:end].rstrip())   # where the model states the value itself
        # the same value as a computation (`(1600 \times 7) + (74 \times 7)`), or a link nothing reads: the chain goes on
        if not m or m.group().strip() != '=':
            return None
        at = m.end()
    return None


def stop_kind(after, put):
    """Why the model's text after `=` is not `put`: `another value` (a link of its chain reads to something else: the model's slip, or a
    misread), `another form` (every link that reads is `put`, none written plainly: `2 \\times 16 = 2 \\times 2 \\times 8`, a Euclid step
    `\\gcd(38115, 29799)`), or `unread` (no link reads: words, a symbol, an equation)."""
    line = after.split('\n', 1)[0]
    at, seen = 0, False
    for _ in range(8):
        lead = at + len(line[at:]) - len(line[at:].lstrip())
        m = _STOP.search(line, lead)
        end = m.start() if m else len(line)
        v = _value(line[at:end].strip())
        if v is not None:
            if v != put:
                return 'another value'
            seen = True
        if not m or m.group().strip() != '=':
            break
        at = m.end()
    return 'another form' if seen else 'unread'


def _plain(piece, value):
    """The piece is the value written out, not a computation of it: `11718`, `-16`, `4\\sqrt{3}`, `\\frac{13}{14}`, `0.5`, `150^\\circ`."""
    piece = exactexpr._DEGREE.sub('', piece).replace('°', '')
    if re.fullmatch(r'-?\d+(?:\.\d+)?(?:/\d+)?', piece.replace(' ', '')):
        return True
    from tools.igc.unified_detect import _same_spelling
    return _same_spelling(piece, exactexpr.spell(value), value) or re.fullmatch(r'-?\\d?frac\{\d+\}\{\d+\}', piece.replace(' ', '')) is not None


class Script:
    """The stored tokens, read in order; skip() moves past characters the software wrote in the model's place."""
    def __init__(self, ids, tok):
        self.ids, self.pos, self.tok = ids, 0, tok

    def text_ahead(self, n=96):
        return self.tok.decode(self.ids[self.pos:self.pos + n])

    def skip_chars(self, chars):
        """Moves past the first `chars` characters ahead (after the leading space the value took). Returns the tokens skipped."""
        k = 0
        while self.pos + k < len(self.ids) and len(self.tok.decode(self.ids[self.pos:self.pos + k])) < chars:
            k += 1
        self.pos += k
        return k


def replay(rec, ids, arm_kw, tok, IDS, recorded=False):
    """recorded=True: `ids` is a trace the same arm wrote, so a put whose tokens are the next stored tokens is what happened there and
    is followed whatever the model wrote after it (`18 + 3 = 21`, `0.0714...`); used to see where a mechanism added to the arm acts."""
    row = dict(id=rec['id'], suite=rec.get('suite', ''), answer=rec.get('answer', ''))
    r = tir4b.Rollout(row, 0, [1, 2, 3], tir4b.Arm(**arm_kw), IDS, tok, seed=1)
    s = Script(ids, tok)
    out = dict(id=rec['id'], right=bool(rec.get('right')), finish=rec.get('finish'), tokens=len(ids), events=[], saved=0, stop=None)
    seen = 0
    while True:
        ask = r.request()
        if not ask:
            break
        most, stops = ask
        new = []
        while s.pos < len(s.ids) and len(new) < most:
            new.append(s.ids[s.pos])
            s.pos += 1
            if new[-1] in stops:
                break
        if not new:
            new = [next(iter(IDS.ends))]
        r.take(new)
        for e in r.events[seen:]:
            ev = dict(kind=e['kind'], at=s.pos, expr=e.get('expr'), value=e.get('value'))
            if e['kind'] == 'inline value' and recorded:
                start, end, _ = r.spans[-1]
                put_ids = r.out[start:end]
                if s.ids[s.pos:s.pos + len(put_ids)] == put_ids:
                    s.pos += len(put_ids)
                    ev.update(verdict='recorded', model=s.text_ahead(40).split('\n', 1)[0])
                    out['events'].append(ev)
                    continue
            if e['kind'] == 'inline value':
                puts = _as_values(str(e.get('value')))
                put = puts[0] if puts else None
                ahead = s.text_ahead(160)
                cut = next((c for c in (chain_skip(ahead, p) for p in puts) if c is not None), None)
                ev['model'] = ahead.split('\n', 1)[0][:60]
                if cut is None:
                    ev['verdict'] = stop_kind(ahead, put) if put is not None else 'unread'
                    out['events'].append(ev)
                    out['stop'] = ev
                    break
                skipped = s.skip_chars(cut)
                ev['verdict'] = 'same' if _STOP.search(ahead[:cut].strip()) is None else 'shortcut'
                ev['skipped'] = skipped
                # the value cost its own tokens: what software saved is what the model would have written past them
                ev['saved'] = max(0, skipped - len(tok.encode(' ' + str(e.get('value')), add_special_tokens=False)))
                out['saved'] += ev['saved']
                out['events'].append(ev)
                continue
            out['events'].append(ev)
            out['stop'] = ev
            break
        seen = len(r.events)
        if out['stop'] is not None:
            break
    if out['stop'] is None and s.pos < len(s.ids) - 1:   # the runner ended the answer before the model did: `stuck`, a length cap
        out['stop'] = dict(kind='ended: ' + str(r.finish), at=s.pos)
        out['events'].append(out['stop'])
    out['followed'] = s.pos
    out['to_end'] = out['stop'] is None
    done = tok.decode(ids[:s.pos])
    boxed = _BOXED.findall(tok.decode(ids))
    final = boxed[-1].strip() if boxed else None
    out['final'] = final
    out['final_before_stop'] = bool(final) and out['stop'] is not None and ('\\boxed{' + final + '}' in done or (final + '$') in done)
    out['gold_before_stop'] = bool(rec.get('answer')) and out['stop'] is not None and re.search(
        r'(?<![\d.])' + re.escape(str(rec['answer'])) + r'(?![\d.])', done[-4000:]) is not None
    out['left'] = len(ids) - s.pos if out['stop'] is not None else 0
    return out


def arm_kwargs(spec):
    """`inline_calc,mathbackend,inline_style=qwen,suppress_recheck=60`: a bare name is on; `name=value` passes the value, a whole number
    as an int (`suppress_recheck='60'` would meet `max(0, self.window - n)` as a string)."""
    kw = {}
    for part in filter(None, spec.split(',')):
        name, eq, value = part.partition('=')
        kw[name] = (int(value) if value.lstrip('-').isdigit() else value) if eq else True
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', required=True)
    ap.add_argument('--ids', default='')
    ap.add_argument('--tokenizer', default='D:/models/Qwen3.5-4B')
    ap.add_argument('--arm', default='inline_calc,mathbackend,inline_plain')
    ap.add_argument('--out', required=True)
    ap.add_argument('--recorded', action='store_true', help="the records are the arm's own run: follow its puts")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    IDS = tir4b.Ids(tok, [tok.convert_tokens_to_ids('<|endoftext|>')])
    arm_kw = arm_kwargs(a.arm)
    recs = json.load(open(a.records, encoding='utf-8'))
    stored = {}
    if a.ids:
        for line in open(a.ids, encoding='utf-8'):
            d = json.loads(line)
            stored[(d['id'], d.get('sample', 0))] = d['out_ids']
    rows = []
    for rec in recs:
        ids = stored.get((rec['id'], rec.get('sample', 0))) or tok.encode(rec['text'], add_special_tokens=False)
        rows.append(replay(rec, ids, arm_kw, tok, IDS, recorded=a.recorded))
    kinds = collections.Counter(e['verdict'] if e['kind'] == 'inline value' else e['kind'] for x in rows for e in x['events'])
    stops = collections.Counter((x['stop']['kind'] if x['stop']['kind'] != 'inline value' else 'value: ' + x['stop']['verdict'])
                                for x in rows if x['stop'])
    summary = dict(arm=a.arm, records=a.records, rows=len(rows), to_end=sum(x['to_end'] for x in rows), events=dict(kinds),
                   stops=dict(stops), saved=sum(x['saved'] for x in rows), tokens=sum(x['tokens'] for x in rows),
                   followed=sum(x['followed'] for x in rows))
    json.dump(dict(summary=summary, rows=rows), open(a.out, 'w', encoding='utf-8'), indent=1, default=str)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
