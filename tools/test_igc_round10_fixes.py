"""The round 9 cases still open, as cases (developer 2026-11-11: fix the case the structure allows, never the test set).
P11 a put after `\\approx`; P7 the letters of a string the question writes out. python -m unittest tools.test_igc_round10_fixes"""
import unittest
from fractions import Fraction


class ApproxPut(unittest.TestCase):
    """The case: the model writes `X \\approx` where X is arithmetic software can compute; its own decimal can be wrong (round 9 General:
    316 of 1,952 readable `\\approx` values off). A value to 10 significant digits is a true approximation under every reading."""

    def test_spelled_to_ten_significant_digits(self):
        from tools.igc.approx_put import approx_spelled
        self.assertEqual(approx_spelled(Fraction(101, 100) ** 30), '1.347848915')
        self.assertEqual(approx_spelled(Fraction(5647, 1237763)), '0.004562262727')
        self.assertEqual(approx_spelled(Fraction(1, 4)), '0.25')            # exact, no padding
        self.assertEqual(approx_spelled(Fraction(-73388, 36903)), '-1.988673008')
        self.assertEqual(approx_spelled(Fraction(2, 3)), '0.6666666667')
        self.assertEqual(approx_spelled(Fraction(3, 2372493604)), '1.264492345 \\times 10^{-9}')
        self.assertEqual(approx_spelled(Fraction(10) ** 12 / 7), '1.428571429 \\times 10^{11}')
        self.assertEqual(approx_spelled(Fraction(33869 * 143)), '4843267')

    def test_reads_the_computation_before_approx(self):
        from tools.igc.approx_put import approx_found
        self.assertEqual(approx_found('So $(1.01)^{30} '), ('(1.01)^{30}', '1.347848915'))
        self.assertEqual(approx_found('A = 1000(1.01)^{30} ')[1], '1347.848915')
        self.assertEqual(approx_found('Then -73388/36903 ')[1], '-1.988673008')
        self.assertEqual(approx_found('$\\frac{5647}{1237763} ')[1], '0.004562262727')

    def test_reads_nothing_that_is_not_arithmetic(self):
        from tools.igc.approx_put import approx_found
        for before in ('So $\\pi ', 'x ', 'The answer ', '$\\sqrt{2} + y ', 'about 12 ', '$f(3) '):
            self.assertIsNone(approx_found(before), before)


class LetterCount(unittest.TestCase):
    """The case: a question asks about the letters of a string it writes out. Tokenized models miscount letters (The Strawberry Problem,
    arXiv 2505.14172); round 9 `swr_p_sequence` 3480#0 and 9951#1. The count of a quoted string is exact."""

    def test_the_string_and_its_counts(self):
        from tools.igc.letters import letter_case, letter_line
        got = letter_case('Two letters picked without replacement from vgcgplcggpgvp. Give prob of sequence vv.')
        self.assertEqual(got[0], 'vgcgplcggpgvp')
        self.assertEqual(dict(got[1]), {'v': 2, 'g': 5, 'c': 2, 'p': 3, 'l': 1})
        self.assertEqual(letter_line(*got), 'The string vgcgplcggpgvp has 13 letters: v 2, g 5, c 2, p 3, l 1.')

    def test_other_wordings(self):
        from tools.igc.letters import letter_case
        self.assertEqual(letter_case('How many r\'s are in strawberry?')[0], 'strawberry')
        self.assertEqual(letter_case('Count the letters in the word "mississippi".')[0], 'mississippi')
        self.assertEqual(letter_case('What is prob of sequence ta when two letters picked without replacement from aaahto?')[0], 'aaahto')

    def test_no_string_no_count(self):
        from tools.igc.letters import letter_case
        for q in ('Three letters picked without replacement from {n: 1, l: 1, k: 2}. Give prob of sequence lno.',
                  'How many letters are in the English alphabet?', 'Write a letter to your friend.',
                  'What is 3 + 4?', 'Tom has 3 apples.'):
            self.assertIsNone(letter_case(q), q)


if __name__ == '__main__':
    unittest.main()
