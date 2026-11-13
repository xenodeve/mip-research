"""Tests for scripts/pc/gate3_values.py: the independent check of every value the token channel steered."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'pc'))
from gate3_values import value_right   # noqa: E402


def inline(expr, value):
    return dict(what='inline value', expr=expr, value=value)


class MissedTests(unittest.TestCase):
    def test_missed_asks_from_the_server_log(self):
        from gate3_values import missed_asks
        log = ('9.30.049.948 W mip-channel: no plan for slot 52 seq 4 in 50 ms (21 missed)\n'
               'other line\n'
               '83.16.007.777 W mip-channel: the service closed the connection; requests go on unsteered\n'
               '69.11.619.787 W mip-channel: no plan for slot 264 seq 11 in 50 ms (73 missed)\n')
        self.assertEqual(missed_asks(log), {(52, 4), (264, 11)})


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

    def test_division_sign_and_gcd(self):
        # dev2 left these unchecked: a \div between surds, and the gcd MIP gives at a fraction's reduction
        self.assertTrue(value_right(inline('\\frac{\\sqrt{153}}{\\sqrt{3}} \\div \\sqrt{3}', '\\sqrt{17}'), ''))
        self.assertTrue(value_right(inline('gcd(171, 2221234)', '1'), ''))
        self.assertTrue(value_right(inline('\\gcd(84, 126)', '42'), ''))
        self.assertIs(value_right(inline('gcd(84, 126)', '21'), ''), False)

    def test_a_long_value_cut(self):
        # a value too long to be a step, put cut to 10 significant digits: checked against the exact value, not left unchecked
        self.assertIs(value_right(inline('1.01^{30}', '1.347848915...'), ''), True)
        self.assertIs(value_right(inline('0.99^{60}', '0.5471566423...'), ''), True)
        self.assertIs(value_right(inline('0.99^{60}', '0.5471566424...'), ''), False)   # rounded, not cut
        self.assertIs(value_right(inline('1.01^{30}', '1.3448888...'), ''), False)

    def test_question_words(self):
        q = 'Two letters picked without replacement from vgcgplcggpgvp. Give prob of sequence vv.'
        line = 'The string vgcgplcggpgvp has 13 letters: v 2, g 5, c 2, p 3, l 1.'
        self.assertIs(value_right(dict(what='letter count', expr='vgcgplcggpgvp', value=line), q), True)
        self.assertIs(value_right(dict(what='letter count', expr='vgcgplcggpgvp', value=line.replace('g 5', 'g 4')), q), False)
        e = '((-25)/((-3025)/(-44)))*1/(-4)'
        self.assertIs(value_right(dict(what='question quote', expr=e, value=e), f'Calculate {e}.'), True)
        self.assertIs(value_right(dict(what='question quote', expr=e, value='(-25)/((-3025)/(-44)*1/(-4))'), f'Calculate {e}.'), False)
        self.assertIs(value_right(dict(what='long value', expr='1.01^{30}', value='1.347848915...'), ''), True)

    def test_a_wrong_surd(self):
        self.assertIs(value_right(inline('\\sqrt{1573}', '11\\sqrt{11}'), ''), False)
        self.assertIs(value_right(inline('-6(-2\\sqrt{204})', '-24\\sqrt{51}'), ''), False)


class PiTrigTests(unittest.TestCase):
    # miss1 (MATH-500 level 5) left 146 values unchecked: pi, trig, plus-minus, tuples
    def test_pi_and_trig(self):
        self.assertTrue(value_right(inline('-\\pi/2 + \\pi', '\\frac{\\pi}{2}'), ''))
        self.assertTrue(value_right(inline('\\cos(\\pi/3)', '0.5'), ''))
        self.assertTrue(value_right(inline('|\\cos(2\\pi/3)|', '0.5'), ''))
        self.assertTrue(value_right(inline('2 \\cos \\frac{\\pi/3 - 2\\pi/3}{2}', '\\sqrt{3}'), ''))
        self.assertTrue(value_right(inline('\\sin(45^\\circ)', '\\frac{\\sqrt{2}}{2}'), ''))
        self.assertIs(value_right(inline('\\cos(\\pi/3)', '1'), ''), False)

    def test_a_cut_decimal_with_pi(self):
        self.assertTrue(value_right(inline('\\pi - 1.107', '2.0345...'), ''))
        self.assertIs(value_right(inline('\\pi - 1.107', '2.0346...'), ''), False)

    def test_plus_minus_and_tuples(self):
        self.assertTrue(value_right(inline('\\frac{6 \\pm \\sqrt{36 - 4}}{2}', '3 \\pm 2\\sqrt{2}'), ''))
        self.assertIs(value_right(inline('\\frac{6 \\pm \\sqrt{36 - 4}}{2}', '3 \\pm \\sqrt{2}'), ''), False)
        self.assertTrue(value_right(inline('(-5-3, -3+10, 6-1)', '(-8, 7, 5)'), ''))
        self.assertIs(value_right(inline('(-5-3, -3+10, 6-1)', '(-8, 7, 4)'), ''), False)

    def test_a_bare_angle_is_not_read(self):
        # no unit: true in degrees, false in radians; MIP reads it by the text's unit (unified_detect), this check does not decide
        self.assertIsNone(value_right(inline('\\sin(45)', '\\frac{\\sqrt{2}}{2}'), ''))


class StateTests(unittest.TestCase):
    def test_values_past_a_loop_close_are_cut(self):
        # miss1 dm_div_big-9991#0: the loop close cut the text at 5219; the 33 values steered into the looping text past it were
        # replaced with the close and the answer, never steered there
        from gate3_values import steer_state
        closes = [5219]
        self.assertEqual(steer_state(dict(at=5205, done=True), set(), closes), 'done')
        self.assertEqual(steer_state(dict(at=5235, done=True), set(), closes), 'cut')
        self.assertEqual(steer_state(dict(at=5219, done=True), set(), closes), 'cut')
        self.assertEqual(steer_state(dict(at=10, done=True, slot=3, seq=1), {(3, 1)}, closes), 'missed')
        self.assertEqual(steer_state(dict(at=10, dropped=True), set(), []), 'dropped')


if __name__ == '__main__':
    unittest.main()
