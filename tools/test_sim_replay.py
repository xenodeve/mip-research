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
