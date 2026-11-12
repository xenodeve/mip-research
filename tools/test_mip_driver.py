"""tir4b with --channel: the requests leave the value stops to the token channel, take() puts no text there, and drive_http's `extra` hook
gives each request its mip_id (token channel plan, task 6)."""
import unittest

from tools.igc import tir4b
from tools.test_igc_tir4b import TOK, IDS, SPECIAL

ROW = dict(id='q', suite='arith', question='Find x.', answer='12')


def roll(**arm):
    a = dict(inline_calc=True, inline_plain=True, mathbackend=True, no_digit_join=True, approx_put=True)
    a.update(arm)
    return tir4b.Rollout(ROW, 0, [1], tir4b.Arm(**a), IDS, TOK, seed=1)


class ChannelArmTests(unittest.TestCase):
    def test_no_value_stops_on_the_channel(self):
        q = roll(channel=True).request()
        self.assertFalse(q[1] & IDS.equals)
        self.assertFalse(q[1] & IDS.approx)

    def test_value_stops_without_the_channel(self):
        self.assertTrue(roll().request()[1] & IDS.equals)

    def test_take_puts_no_text_on_the_channel(self):
        r = roll(channel=True)
        r.take(TOK.encode('So $3 \\times 4 ='))
        self.assertFalse([e for e in r.events if e['kind'] == 'inline value'])

    def test_think_only_puts_nothing_in_the_answer(self):
        r = roll(think_only=True)
        r.take(TOK.encode('done</think>'))
        r.take(TOK.encode('So $3 \\times 4 ='))
        self.assertEqual(r.phase, 'answer')
        self.assertFalse([e for e in r.events if e['kind'] == 'inline value'])
        self.assertFalse(r.request()[1] & IDS.equals)

    def test_think_only_still_puts_in_the_thinking(self):
        r = roll(think_only=True)
        r.take(TOK.encode('So $3 \\times 4 ='))
        self.assertEqual([e['value'] for e in r.events if e['kind'] == 'inline value'], ['12'])


class ExtraHookTests(unittest.TestCase):
    def test_each_request_carries_the_hook_s_fields(self):
        r = roll(channel=True)
        bodies = []

        def post(body):
            bodies.append(body)
            return dict(tokens=TOK.encode('ok') + [SPECIAL['<|im_end|>']])
        seen = []

        def extra(rollout):
            seen.append(rollout)
            return dict(mip_id=f'{rollout.row["id"]}#{rollout.sample}#{len(seen) - 1}')
        tir4b.drive_http(post, TOK, IDS, [r], None, live_most=1, extra=extra)
        self.assertEqual(bodies[0]['mip_id'], 'q#0#0')
        self.assertIs(seen[0], r)


if __name__ == '__main__':
    unittest.main()
