"""tools/igc/sim_replay.py: the replay follows a stored answer while software's values are what the model wrote, and stops where they are
not. python tools/test_sim_replay.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc import exactexpr, sim_replay as sr
from tools.igc.tir4b import inline_found
from tools.test_igc_tir4b import TOK, IDS


def V(x):
    return exactexpr.evaluate(x)


class Chain(unittest.TestCase):
    def test_the_models_own_value(self):
        self.assertEqual(sr.chain_skip(' 468$. So', V('468')), len(' 468'))

    def test_a_later_value_of_the_chain_is_a_shortcut(self):
        after = ' (1600 \\times 7) + (74 \\times 7) = 11200 + 518 = 11718$. Correct.'
        self.assertEqual(after[:sr.chain_skip(after, V('11718'))], ' (1600 \\times 7) + (74 \\times 7) = 11200 + 518 = 11718')

    def test_a_relation_ends_the_value(self):
        self.assertEqual(sr.chain_skip(' 121 \\neq 89$.', V('121')), len(' 121'))
        self.assertEqual(sr.chain_skip(' 38 > 0$.', V('38')), len(' 38'))
        self.assertEqual(sr.chain_skip(' 150^{\\circ}$.', V('150')), len(' 150^{\\circ}'))

    def test_why_it_stops(self):
        self.assertIsNone(sr.chain_skip(' 200/210 = 20/21$.', V('13/14')))
        self.assertEqual(sr.stop_kind(' 200/210 = 20/21$.', V('13/14')), 'another value')
        self.assertEqual(sr.stop_kind(' 2 \\times 2 \\times 8 = 2^5$.', V('32')), 'another form')
        self.assertEqual(sr.stop_kind(' y + 4x$', V('448')), 'unread')


class GeneralForms(unittest.TestCase):
    """#307: the General Style's puts stand for more than one value; the replay follows the model when it writes any of them."""
    def test_each_reading_is_followed(self):
        def skip(said, after):
            return next((c for c in (sr.chain_skip(after, p) for p in sr._as_values(said)) if c is not None), None)
        self.assertEqual(skip('16807 ≡ 7', ' 16807≡7 mod25'), 6)
        self.assertEqual(skip('16807 ≡ 7', ' 7 mod25'), 2)
        self.assertEqual(skip('6 rem 1', ' 6 remainder 1'), 2)
        self.assertEqual(skip('1.4866...', ' 1.4867 so'), 7)    # rounded
        self.assertEqual(skip('1.4866...', ' 1.4866$'), 7)      # cut
        self.assertIsNone(skip('6 rem 1', ' 7 remainder 0'))


class ArmSpec(unittest.TestCase):
    def test_names_values_and_numbers(self):
        self.assertEqual(sr.arm_kwargs('inline_calc,inline_style=qwen,suppress_recheck=60'),
                         dict(inline_calc=True, inline_style='qwen', suppress_recheck=60))
        self.assertEqual(sr.arm_kwargs(''), {})


class Replay(unittest.TestCase):
    def test_follows_to_the_end_when_every_value_is_the_models(self):
        text = 'So $4 \\times 6 = 24$ and $24 + 1 = 25$.\n</think>\n\\boxed{25}<|im_end|>'
        out = sr.replay(dict(id='q', right=True, answer='25'), TOK.encode(text), dict(inline_calc=True, inline_plain=True), TOK, IDS)
        self.assertTrue(out['to_end'], out)
        self.assertEqual([e['verdict'] for e in out['events']], ['same', 'same'])

    def test_stops_at_a_value_the_model_did_not_write(self):
        text = 'So $4 \\times 6 = 25$ and then more text.\n</think>\n\\boxed{25}<|im_end|>'
        out = sr.replay(dict(id='q', right=False, answer='24'), TOK.encode(text), dict(inline_calc=True, inline_plain=True), TOK, IDS)
        self.assertFalse(out['to_end'])
        self.assertEqual(out['stop']['verdict'], 'another value')
        self.assertGreater(out['left'], 0)


    def test_an_arms_own_trace_is_followed_past_its_puts(self):
        """#307: an MIP arm's trace holds software's put; the model went on from it (`18 + 3 = 21`, a carry) or it cannot be read as a
        value (`0.0714...`). With recorded=True a put whose tokens are the stored ones is followed; without it, the replay stops there."""
        ids = (TOK.encode('So $3 \\times 6 =') + TOK.encode(' 18') + TOK.encode(' + 3 = 21$ and $1/14 =') + TOK.encode(' 0.0714...')
               + TOK.encode('$.\n</think>\n\\boxed{21}<|im_end|>'))
        arm = dict(inline_calc=True, inline_plain=True, mathbackend=True)
        old = sr.replay(dict(id='q', right=True, answer='21'), ids, arm, TOK, IDS)
        self.assertFalse(old['to_end'])
        new = sr.replay(dict(id='q', right=True, answer='21'), ids, arm, TOK, IDS, recorded=True)
        self.assertTrue(new['to_end'], new)
        self.assertEqual([e['verdict'] for e in new['events'] if e['kind'] == 'inline value'][:2], ['recorded', 'recorded'])

    def test_recorded_still_stops_at_a_put_the_trace_does_not_hold(self):
        ids = TOK.encode('So $3 \\times 6 =') + TOK.encode(' 19') + TOK.encode('$ more.\n</think>\n\\boxed{19}<|im_end|>')
        out = sr.replay(dict(id='q', right=False, answer='18'), ids, dict(inline_calc=True, inline_plain=True), TOK, IDS, recorded=True)
        self.assertFalse(out['to_end'])

    def test_a_request_vllm_would_refuse_is_counted(self):
        """#307: the replay never builds a vLLM request, so a 1,106-token bias went to the A100 unseen; each request is checked now."""
        ids = TOK.encode('So $3 \\times 6 =') + TOK.encode(' 18') + TOK.encode('$ more.\n</think>\n\\boxed{18}<|im_end|>')
        arm = dict(inline_calc=True, inline_plain=True, no_digit_join=True)
        out = sr.replay(dict(id='q', right=True, answer='18'), ids, arm, TOK, IDS, recorded=True)
        self.assertNotIn('refused', out)
        cap = sr.tir4b.BIAS_MAX
        sr.tir4b.BIAS_MAX = 3   # fewer than the digits held after the put
        try:
            out = sr.replay(dict(id='q', right=True, answer='18'), ids, arm, TOK, IDS, recorded=True)
        finally:
            sr.tir4b.BIAS_MAX = cap
        self.assertIn('logit bias tokens', out['refused'][0]['why'])

    def test_the_known_vllm_limits(self):
        f = sr.tir4b.vllm_refusal
        self.assertIsNone(f(100, 1024, {1: -100.0}))
        self.assertIn('at least 1', f(100, 0, None))
        self.assertIn('holds', f(sr.tir4b.CONTEXT, 10, None))
        self.assertIn('1025 logit bias', f(100, 1, {i: -1.0 for i in range(1025)}))


class BridgeGuards(unittest.TestCase):
    """What the replay of a100 ctrlb showed the bridge put in where the model meant something else (2026-11-06)."""
    def found(self, t):
        return inline_found(t, use_mathbackend=True)

    def test_tails_and_lines_that_work_mod_m(self):
        self.assertIsNone(self.found("Let's find where $f(x) = 1/4$.\n$|x| - 1/2"))
        self.assertEqual(self.found('$b, c = 4\\sqrt{30} \\pm 2\\sqrt{12}')[1], '4\\sqrt{30} \\pm 4\\sqrt{3}')   # the whole `±`, not its tail
        self.assertEqual(self.found('The parity is $1 + 1')[1], '2 \\equiv 0')   # the value and its residue mod 2 (#306)
        self.assertEqual(inline_found('The parity is $1 + 1', use_mathbackend=True, style='qwen3.5-4b')[1], '0')   # as the model wrote
        self.assertIsNone(self.found('$2^{10} \\equiv 1024 + 1'))
        self.assertIsNone(self.found('$60^\\circ'))

    def test_still_read(self):
        self.assertEqual(self.found('$|x| = 3 - 1/2')[1], '2.5')
        self.assertEqual(self.found('$4\\sqrt{30} + \\sqrt{12} \\cdot 2')[1], '4\\sqrt{3} + 4\\sqrt{30}')


if __name__ == '__main__':
    unittest.main()
