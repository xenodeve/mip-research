"""mip_planner.py: what MIP steers on the token channel, by the Rollout's own rules (token channel design, 2026-11-12).

Only in the thinking; only values MIP computes. After `=`: the inline value (Rollout.put_value, today's rules). After `\\approx` or `≈`:
the value to 10 significant digits (Rollout.approx_put_value). At the start of the thinking, when the question is one fixed computation in
symbols: the line `<expression> = <value>` (the developer, 2026-11-12: "there is no reason for the model to reason it out itself when
software already can"); the thinking is not closed. A plan carries a ValueDag: the sampler allows only the tokens that can still spell the
text, so the model writes it in its own tokenization. `after_ban`: the token ids banned for one step after the value (today's
--no-digit-join: no digit or slash glued onto software's value; no scale word after an approximation).

Planner(rollout, pieces).on_trigger(text, phase) -> Plan | None;  Planner.at_start(question) -> Plan | None
"""
from collections import namedtuple
from fractions import Fraction

from tools.igc.token_trie import ValueDag

Plan = namedtuple('Plan', 'text dag kind after_ban expr value')


class Planner:
    def __init__(self, rollout, pieces):
        self.rollout, self.pieces = rollout, pieces

    def _plan(self, text, kind, ban, expr, value):
        dag = ValueDag.build(text, self.pieces)
        return Plan(text, dag, kind, sorted(ban), expr, value) if dag is not None else None

    def on_trigger(self, text, phase):
        if phase != 'think' or '</think>' in text:
            return None
        r, ids = self.rollout, self.rollout.ids
        bare = text.rstrip()
        if bare.endswith('=') and r.arm.inline_calc:
            got = r.put_value(text)
            if got is None:
                return None
            expr, value, put, whole = got
            ban = set()
            if r.arm.no_digit_join and r.arm.inline_plain and not whole and value[-1:].isdigit():
                ban = set(ids.digit_start) | set(ids.slash_start)
            return self._plan(put, 'long value' if r.last_put_long else 'inline value', ban, expr, value)
        if (bare.endswith('\\approx') or bare.endswith('≈')) and r.arm.approx_put:
            got = r.approx_put_value(text)
            if got is None:
                return None
            expr, value, put = got
            ban = set()
            if r.arm.no_digit_join and r.arm.inline_plain:
                ban = set(ids.digit_start) | set(ids.slash_start) | set(ids.approx_hold)
            return self._plan(put, 'approx value', ban, expr, value)
        return None

    def at_start(self, question):
        """The lines steered at the start of the thinking: the fixed-question line (the question is one computation) and, with
        --letter-count, the letter counts of a string the question writes out. One plan; its kind names the lines in it."""
        if not question:
            return None
        lines, kinds, exprs, values = [], [], [], []
        from tools.igc.question_read import read_exact
        from tools.igc.mathexpr import spelled
        got = read_exact(question) if self.rollout.arm.question_line else None
        if got is not None and isinstance(got[1], Fraction):
            expr, value = got[0], spelled(got[1])
            lines.append(f'{expr} = {value}\n')
            kinds.append('question line')
            exprs.append(expr)
            values.append(value)
        if self.rollout.arm.letter_count:
            from tools.igc.letters import letter_case, letter_line
            got = letter_case(question)
            if got is not None:
                line = letter_line(*got)
                lines.append(line + '\n')
                kinds.append('letter count')
                exprs.append(got[0])
                values.append(line)
        if not lines:
            return None
        return self._plan(''.join(lines), ' + '.join(kinds), set(), ' | '.join(exprs), ' | '.join(values))

    def pending(self):
        """A plan the driver handed the channel (a question quote), steered at the next request's first sample, or None."""
        got = self.rollout.take_pending()
        if got is None:
            return None
        text, kind, expr, value = got
        return self._plan(text, kind, set(), expr, value)
