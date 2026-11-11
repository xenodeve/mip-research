"""question_read: the computation a question states. python -m unittest tools.test_igc_question_read"""
import unittest
from fractions import Fraction

from tools.igc.question_read import read

FORMAT = '\n\nGive the final answer as a single number in \\boxed{} at the end.'


def value(q):
    got = read(q + FORMAT)
    return None if got is None else got[1]


class Read(unittest.TestCase):
    def test_the_english_forms_and_their_order(self):
        self.assertEqual(value('What is 3 less than 7?'), 4)                 # 7 - 3, not 3 - 7 (pc4-whole add_or_sub 5180)
        self.assertEqual(value('Subtract 3 from 7'), 4)
        self.assertEqual(value('What is 7 take away -3?'), 10)
        self.assertEqual(value('What is the distance between 20412938679033416 and -0.4?'), Fraction('20412938679033416.4'))  # 6634
        self.assertEqual(value('What is the difference between 1508489756117 and -29.392?'), Fraction('1508489756146.392'))
        self.assertEqual(value('Product of -3388.02 and 32790'), Fraction('-111093175.8'))
        self.assertEqual(value('Divide 10 by -4'), Fraction(-5, 2))
        self.assertEqual(value('Sum 55839343876742 and -8.'), 55839343876734)
        self.assertEqual(value('Put together 2 and 5'), 7)

    def test_an_expression_as_written(self):
        self.assertEqual(value('What is 20*(-8)/(-60)*329*9/((-36)/2)*7*(-6)/49?'), 376)
        self.assertEqual(value('(-1*-638*(-7)/(-7))/(((-62)/(-279))/(((-170)/3060)/(1/(-2))))'), 319)

    def test_no_single_computation_is_none(self):
        self.assertIsNone(value('Tom has 3 apples and buys 4 more. How many does he have?'))
        self.assertIsNone(value('Solve 2x + 3 = 7 for x.'))
        self.assertIsNone(value('What is 12?'))


class ReadExact(unittest.TestCase):
    """read_exact: as read, and a computation with square roots computed exactly (round 9, dm_simplify_surd 3480 and 9937 looped on
    their parentheses; read gives a Fraction or nothing, so the surd questions were never read)."""

    def test_a_surd_expression_is_computed_exactly(self):
        from tools.igc.question_read import read_exact
        from tools.igc.exactexpr import spell
        q = 'Simplify (6*(-3*(3 + (sqrt(108) + sqrt(108) + -2*sqrt(108) + sqrt(108))*-5 + 4) + 5))**2.\n\nPut the final answer in \\boxed{} at the end.'
        expr, v = read_exact(q)
        self.assertEqual(spell(v, latex=False), '884016 - 103680*sqrt(3)')
        expr, v = read_exact('Simplify (-2*sqrt(77)*-3 + sqrt(77)*3*-1)/(1*sqrt(176)*-1*2).')
        self.assertEqual(spell(v, latex=False), '-3*sqrt(7)/8')

    def test_rational_questions_read_as_before(self):
        from tools.igc.question_read import read, read_exact
        for q in ('Divide -73388 by 36903.', 'What is 20*(-8)/(-60)*329*9/((-36)/2)*7*(-6)/49?', 'Simplify sqrt(16)*3.'):
            got = read_exact(q)
            self.assertIsInstance(got[1], Fraction, q)
        self.assertEqual(read_exact('Divide -73388 by 36903.'), read('Divide -73388 by 36903.'))

    def test_no_single_computation_is_still_none(self):
        from tools.igc.question_read import read_exact
        for q in ('Simplify sqrt(x)**2.', 'Tom has sqrt(16) apples. How many?', 'Simplify sqrt(108).  Round to 2 places.', 'What is 12?'):
            self.assertIsNone(read_exact(q), q)


class TheSituation(unittest.TestCase):
    """The situation, not the test's wording (developer 2026-11-11): a question that asks for one expression's value and holds nothing
    else that carries content is read in any request wording, LaTeX included; a word that could change what is asked (rounding, a base,
    a remainder, a percent, a unit, a variable) reads nothing. The 80 DeepMind questions re-asked in 4 other wordings were read 0 of 80."""

    def test_request_wordings_around_one_expression(self):
        from tools.igc.question_read import read_exact
        e = '(-25)/(-10)*1/(-4)'
        for q in (f'{e} = ?', f'Can you calculate {e} for me?', f'Please simplify {e}.', f'Could you tell me what {e} equals?',
                  f'I need the result of {e}', f'hey, how much is {e}?', f'{e}=?', f'Quick one: {e}'):
            got = read_exact(q)
            self.assertIsNotNone(got, q)
            self.assertEqual(got[1], Fraction(-5, 8), q)

    def test_latex_expressions(self):
        from tools.igc.question_read import read_exact
        cases = (('Compute $3 \\times 4 - 2$.', Fraction(10)), ('What is $\\frac{3}{4} + \\frac{1}{6}$?', Fraction(11, 12)),
                 ('Evaluate $\\left(3+4\\right) \\cdot 2 \\div 7$.', Fraction(2)), ('Find $\\dfrac{-12}{18}$.', Fraction(-2, 3)),
                 ('Calculate $2^{10} - 1$.', Fraction(1023)))
        for q, want in cases:
            got = read_exact(q)
            self.assertIsNotNone(got, q)
            self.assertEqual(got[1], want, q)
        from tools.igc.exactexpr import spell
        got = read_exact('Simplify $\\sqrt{8} + \\sqrt{2}$.')
        self.assertEqual(spell(got[1], latex=False), '3*sqrt(2)')

    def test_a_word_that_changes_the_question_reads_nothing(self):
        from tools.igc.question_read import read_exact
        e = '3/7 + 1/2'
        for q in (f'Round {e} to 2 decimal places.', f'What is {e} rounded to the nearest integer?', f'What is {e} in binary?',
                  f'What is the remainder when {e} is divided by 7?', f'Is {e} positive?', f'What is 10% of {e}?',
                  f'Approximately what is {e}?', f'What is {e} meters in feet?', f'What is {e} + x?', f'Solve {e} = y for y.',
                  'What is 3 + 4 and 5 + 6?', 'What is the sum of the digits of 123 + 456?', f'Write {e} as a decimal.'):
            self.assertIsNone(read_exact(q), q)


if __name__ == '__main__':
    unittest.main()
