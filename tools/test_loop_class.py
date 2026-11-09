"""Decide layer v1 classifier: each class from a made-up loop written for its definition (not taken from a test set)."""
import unittest

from tools.igc.loop_class import classify, cycle, settle

Q = 'Calculate (12 + 30)/7 - 5.\n\nGive the final answer as a single number in \\boxed{} at the end.'


def loop(*lines, times=8):
    return 'Some work first.\n' + '\n'.join(list(lines) * times) + '\n'


class LoopClassTest(unittest.TestCase):
    def test_cycle_is_the_repeated_lines(self):
        self.assertEqual(cycle(loop('Wait, $6 + 1 = 7$.')), ['Wait, $6 + 1 = 7$.'])
        self.assertEqual(cycle('a\nb\nc\n'), [])

    def test_a_true_equation_rechecked(self):
        cls, rel = classify(loop('Wait, $42 / 7 = 6$.'), Q)
        self.assertEqual(cls, 'A')
        self.assertIn('42 / 7 = 6', rel)
        self.assertEqual(settle(cls, rel, Q), 'Checked exactly: $42 / 7 = 6$ holds.')

    def test_a_false_equation_is_not_confirmed(self):
        cls, _ = classify(loop('Wait, $42 / 7 = 5$.'), Q)
        self.assertNotEqual(cls, 'A')

    def test_b_doubt_about_the_input(self):
        cls, _ = classify(loop('Wait, what if the `(12 + 30)` is `(12 + 3)`?'), Q)
        self.assertEqual(cls, 'B')
        self.assertTrue(settle('B', None, Q).startswith('The question writes, exactly: Calculate (12 + 30)/7 - 5.'))

    def test_c_form_choice(self):
        cls, _ = classify(loop("I'll write 1/2.", "I'll write 0.5."), Q)
        self.assertEqual(cls, 'C')
        self.assertEqual(settle('C', None, Q), 'Computed exactly: $(12 + 30)/7 - 5 = 1$.')

    def test_d_factor_search(self):
        text = 'x\n' + '\n'.join(f'$1237763 / {p} = {1237763 // p} \\text{{ rem }} {1237763 % p}$.' for p in (6491, 6493, 6499, 6503, 6509, 6511))
        self.assertEqual(classify(text, Q)[0], 'D')
        q = 'Divide -11294 by -2475526.'
        self.assertEqual(settle('D', '1237763', q), 'Computed exactly: $(-11294)/(-2475526) = 5647/1237763$; $\\gcd(5647, 1237763) = 1$, so it is in lowest terms.')

    def test_e_a_run(self):
        self.assertEqual(classify('work\n7.' + '0' * 200, Q)[0], 'E')
        self.assertEqual(classify(loop('Wait.'), Q)[0], 'E')
        self.assertIsNone(settle('E', None, Q))

    def test_f_otherwise(self):
        cls, _ = classify(loop('If the circles touch the chord, they overlap.', 'So they cannot be distinct.'), Q)
        self.assertEqual(cls, 'F')
        self.assertIsNone(settle('F', None, Q))

    def test_no_settle_without_an_exact_value(self):
        self.assertIsNone(settle('C', None, 'Find the area of the shaded region.'))


if __name__ == '__main__':
    unittest.main()
