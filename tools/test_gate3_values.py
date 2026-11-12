"""Tests for scripts/pc/gate3_values.py: the independent check of every value the token channel steered."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'pc'))
from gate3_values import value_right   # noqa: E402


def inline(expr, value):
    return dict(what='inline value', expr=expr, value=value)


class SurdTests(unittest.TestCase):
    # surds are checked by sympy, a computer MIP's own code does not use
    def test_a_right_surd(self):
        self.assertTrue(value_right(inline('\\sqrt{1573}', '11\\sqrt{13}'), ''))
        self.assertTrue(value_right(inline('-6 \\cdot (-2\\sqrt{204})', '24\\sqrt{51}'), ''))
        self.assertTrue(value_right(inline('\\frac{\\sqrt{96}}{\\sqrt{8} \\cdot 2}', '\\sqrt{3}'), ''))
        self.assertTrue(value_right(inline('\\sqrt{17}\\sqrt{3} - \\sqrt{17}', '\\sqrt{51} - \\sqrt{17}'), ''))
        self.assertTrue(value_right(inline('24\\sqrt{51}/\\sqrt{3}', '24\\sqrt{17}'), ''))

    def test_powers_and_permutations(self):
        self.assertTrue(value_right(inline('(5 + 24\\sqrt{17})^2', '9817 + 240\\sqrt{17}'), ''))
        self.assertTrue(value_right(inline('\\frac{\\sqrt{153}}{(\\sqrt{3})^2}', '\\sqrt{17}'), ''))
        self.assertTrue(value_right(inline('P(7, 3)', '210'), ''))
        self.assertIs(value_right(inline('P(15, 4)', '32761'), ''), False)

    def test_a_wrong_surd(self):
        self.assertIs(value_right(inline('\\sqrt{1573}', '11\\sqrt{11}'), ''), False)
        self.assertIs(value_right(inline('-6(-2\\sqrt{204})', '-24\\sqrt{51}'), ''), False)


if __name__ == '__main__':
    unittest.main()
