"""Tests for tools/igc/mip_planner.py: what MIP steers on the token channel (token channel design, 2026-11-12)."""
import unittest

from tools.igc import tir4b
from tools.igc.mip_planner import Planner
from tools.test_igc_tir4b import TOK, IDS

PIECES = [TOK.decode([i]) for i in range(len(TOK))]


def roll(text='', question='What is 6*7?', **arm):
    a = dict(inline_calc=True, inline_plain=True, mathbackend=True, no_digit_join=True, approx_put=True)
    a.update(arm)
    r = tir4b.Rollout(dict(id='q', suite='arith', question=question, answer='42'), 0, [1], tir4b.Arm(**a), IDS, TOK, seed=1)
    r.out = TOK.encode(text)
    return r


def spell(plan):
    node, out = 0, ''
    while node != plan.dag.end:
        t = min(plan.dag.allowed(node))
        out += PIECES[t]
        node = plan.dag.step(node, t)
    return out


class PlannerTests(unittest.TestCase):
    def test_value_after_equals_in_the_thinking(self):
        r = roll('Let me compute $3 \\times 4 =')
        p = Planner(r, PIECES).on_trigger(r.text(), 'think')
        self.assertEqual(p.text, ' 12')
        self.assertEqual(p.kind, 'inline value')
        self.assertEqual(spell(p), ' 12')
        self.assertTrue(set(p.after_ban) >= set(IDS.digit_start))   # today's --no-digit-join, for one step after the value

    def test_nothing_in_the_answer(self):
        r = roll('Let me compute $3 \\times 4 =')
        self.assertIsNone(Planner(r, PIECES).on_trigger(r.text(), 'answer'))

    def test_nothing_after_the_thinking_closed(self):
        r = roll('ok</think>So $3 \\times 4 =')
        self.assertIsNone(Planner(r, PIECES).on_trigger(r.text(), 'think'))

    def test_approx_digits(self):
        r = roll('so it is 1/3 \\approx')
        p = Planner(r, PIECES).on_trigger(r.text(), 'think')
        self.assertEqual((p.text, p.kind), (' 0.3333333333', 'approx value'))

    def test_no_plan_when_the_value_cannot_be_spelled(self):
        r = roll('Let me compute $3 \\times 4 =')
        no_digits = [p if not p[:1].isdigit() else '' for p in PIECES]
        self.assertIsNone(Planner(r, no_digits).on_trigger(r.text(), 'think'))

    def test_no_plan_for_a_text_that_does_not_end_at_a_trigger(self):
        r = roll('Let me compute $3 \\times 4')
        self.assertIsNone(Planner(r, PIECES).on_trigger(r.text(), 'think'))

    def test_fixed_question_line_at_the_start(self):
        r = roll('', question='What is 2*240?', question_line=True)
        p = Planner(r, PIECES).at_start(r.row['question'])
        self.assertEqual((p.text, p.kind), ('2*240 = 480\n', 'question line'))
        r = roll('', question='What is 2*240?')
        self.assertIsNone(Planner(r, PIECES).at_start(r.row['question']))   # --question-line off

    def test_letter_counts_at_the_start(self):
        q = 'Two letters picked without replacement from vgcgplcggpgvp. Give prob of sequence vv.'
        r = roll('', question=q, channel=True, letter_count=True)
        p = Planner(r, PIECES).at_start(q)
        self.assertEqual((p.text, p.kind), ('The string vgcgplcggpgvp has 13 letters: v 2, g 5, c 2, p 3, l 1.\n', 'letter count'))
        self.assertEqual(spell(p), p.text)
        self.assertIsNone(Planner(roll('', question=q, channel=True), PIECES).at_start(q))   # flag off

    def test_a_pending_plan_from_the_driver(self):
        r = roll('So x\n', channel=True)
        r.steer_queue.append(('The question writes: $2*240$.\n', 'question quote', '2*240', '2*240'))
        p = Planner(r, PIECES).pending()
        self.assertEqual((p.text, p.kind, p.expr), ('The question writes: $2*240$.\n', 'question quote', '2*240'))
        self.assertIsNone(Planner(r, PIECES).pending())

    def test_no_line_for_a_question_that_is_not_one_computation(self):
        r = roll('', question='Find all x such that x^2=4.')
        self.assertIsNone(Planner(r, PIECES).at_start(r.row['question']))

    def test_a_long_value_is_its_own_kind(self):
        r = roll('so $1.01^{30} =')
        p = Planner(r, PIECES).on_trigger(r.text(), 'think')
        self.assertEqual((p.text, p.kind), (' 1.347848915...', 'long value'))

    def test_put_value_is_take_s_rule(self):
        # the refactor keeps take's behaviour: a put through take() still writes ' 12'
        r = roll('Let me compute $3 \\times 4 =')
        self.assertEqual(r.put_value(r.text())[:3], ('3 \\times 4', '12', ' 12'))


if __name__ == '__main__':
    unittest.main()
