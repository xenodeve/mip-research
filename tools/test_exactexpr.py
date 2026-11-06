"""tools/igc/exactexpr.py and the MIP bridge's use of it (tools/igc/unified_detect.py expression_before_exact). Each case is a form a
model wrote before `=` in the 48 control traces of a100_262k (2026-11-06), or a form that must stay unread. python tools/test_exactexpr.py"""
import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc import exactexpr as x
from tools.igc import unified_detect as ud
from tools.igc.tir4b import inline_found


def val(text, functions=None):
    v = x.evaluate(text, functions)
    return None if v is None else x.spell(v)


class Values(unittest.TestCase):
    def check(self, cases):
        for text, want in cases:
            with self.subTest(text=text):
                self.assertEqual(val(text), want)

    def test_products_with_no_sign_and_binomials(self):
        self.check([(r'4(117)', '468'), (r'9(256/81)', '256/9'), (r'4(13)(9)', '468'), (r'\binom{10}{4}', '210'),
                    (r'\binom{4}{3}\binom{6}{1}', '24'), (r'\frac{9!}{4! 3! 2!}', '1260'), (r'-3 \times -2 \times 2', '12')])

    def test_roots_kept_as_roots(self):
        self.check([(r'\sqrt{1674}', r'3\sqrt{186}'), (r'12 \sqrt{90}', r'36\sqrt{10}'), (r'\sqrt{9 \times 1302}', r'3\sqrt{1302}'),
                    (r'\frac{2\sqrt{2}}{\sqrt{7}}', r'\frac{2\sqrt{14}}{7}'), (r'\sqrt{1 - 12/13}', r'\frac{\sqrt{13}}{13}'),
                    (r'\frac{6+2\sqrt{5}}{4}', r'\frac{3 + \sqrt{5}}{2}'), (r'308 \frac{3+\sqrt{5}}{2}', r'462 + 154\sqrt{5}'),
                    (r'\frac{1}{1+\sqrt{2}}', r'\sqrt{2} - 1'), (r'\frac{1}{\sqrt{2}+\sqrt{3}}', r'\sqrt{3} - \sqrt{2}'),
                    (r'(\sqrt{2}+\sqrt{3})^2', r'5 + 2\sqrt{6}'), (r'\frac{2+\sqrt{5}-1 - 2\sqrt{5}-2}{2}', r'-\frac{1 + \sqrt{5}}{2}'),
                    (r'\sqrt[3]{27}', '3'), (r'8^{1/3}', '2'), (r'4^{-1/2}', '0.5'), (r'(-8)^{1/3}', '-2'), ('√2 \\cdot √8', '4'),
                    (r'sqrt(12)', r'2\sqrt{3}'), (r'\lfloor \sqrt{50} \rfloor', '7'), (r'\lceil \sqrt{50} \rceil', '8'),
                    (r'|1 - \sqrt{2}|', r'\sqrt{2} - 1')])

    def test_roots_that_are_no_step(self):
        # a decimal asks for a decimal; a root of a root, a cube root that does not come out
        self.check([(r'\sqrt{310.9666}', None), (r'\sqrt[3]{2}', None), (r'\sqrt{-4}', None)])

    def test_exact_trig(self):
        self.check([(r'\sin 30^\circ', '0.5'), (r'\cos 60^{\circ}', '0.5'), (r'\sin 90^{\circ}', '1'), (r'\sin 0', '0'),
                    (r'\sin \frac{\pi}{6}', '0.5'), (r'\cos(2\pi/3)', '-0.5'), (r'\tan 45^\circ', '1'),
                    (r'\sin 15^\circ', r'\frac{\sqrt{6} - \sqrt{2}}{4}'), (r'\sin^2 60^\circ', '0.75'), (r'\sin 30^\circ \cos 60^\circ', '0.25'),
                    (r'\cos 36^\circ', r'\frac{1 + \sqrt{5}}{4}'), (r'\sin 18^\circ', r'\frac{\sqrt{5} - 1}{4}'), (r'\cos 144^\circ', r'-\frac{1 + \sqrt{5}}{4}'),
                    (r'\cos(-60^\circ)', '0.5'), ('sin(60°)', r'\frac{\sqrt{3}}{2}'), (r'180^\circ - 30^\circ', '150')])

    def test_trig_with_no_exact_value(self):
        # `\sin 30` has no unit (30 radians or 30 degrees); 18 degrees' cosine is a root of a root
        self.check([(r'\sin 30', None), (r'\tan 90^\circ', None), (r'\cos 18^\circ', None), (r'\sin 10^\circ', None), (r'\sin x', None)])

    def test_a_root_of_a_root_that_denests(self):
        # #306 held-out: sqrt(a + b sqrt r) = sqrt x + sqrt y when a^2 - b^2 r is a square
        self.check([('sqrt(4 - 2√3)', r'\sqrt{3} - 1'), ('sqrt( (12 -6√3 +4 -2√3)/4 )', r'\sqrt{3} - 1'),
                    ('(sqrt(2 -√3)/2)^2', r'\frac{2 - \sqrt{3}}{4}'), ('sqrt(2 - √3)', r'\frac{\sqrt{6} - \sqrt{2}}{2}'),
                    (r'\sqrt{3 + 2\sqrt{2}}', r'1 + \sqrt{2}'), ('sqrt(7 - 4√3)', r'2 - \sqrt{3}'), ('sqrt(1 + √2)', None), ('sqrt(√2)', None)])

    def test_pi_is_a_value(self):
        # #306 held-out: 455 `=` with pi in them read nothing; pi is transcendental, so a value with it still spells one way
        self.check([(r'\pi', r'\pi'), (r'2\pi', r'2\pi'), ('π/2 - 5π/12', r'\frac{\pi}{12}'), ('(π - 3π/4) +5', r'5 + \frac{\pi}{4}'),
                    ('2π/12 + π/12 + 9π/12', r'\pi'), ('8*sqrt(5) * (9/(10*sqrt(5))) * π', r'\frac{36\pi}{5}'),
                    ('π*(6/5)*(6/sqrt(5))', r'\frac{36\sqrt{5}\pi}{25}'), ('128/π', r'\frac{128}{\pi}'), (r'\pi \cdot 3^2', r'9\pi'),
                    ('(1+√2)π', r'(1 + \sqrt{2})\pi'), ('π^2/π', r'\pi'), ('1/(1+π)', None), (r'\pi r^2', None)])
        self.assertEqual(x.spell(x.evaluate('π/2 - 5π/12'), latex=False), 'π/12')
        self.assertEqual(x.evaluate("3π/4").sign(), 1)
        self.assertEqual(x.evaluate("π - 3").floor(), x.Value.of(0))

    def test_logs_that_come_out(self):
        self.check([(r'\log_2 8', '3'), (r'\log_{10} 1000', '3'), (r'\log_4 2', '0.5'), (r'\log_8 \frac{1}{4}', '-2/3'), (r'\ln 1', '0'),
                    (r'\log_2 6', None), (r'\log 100', None), (r'\ln e', None)])

    def test_sums_and_products_over_a_range(self):
        self.check([(r'\sum_{k=1}^{10} k^2', '385'), (r'\sum_{i=0}^{4} 2^i', '31'), (r'\prod_{k=1}^{5} k', '120'),
                    (r'\sum_{k=1}^{4} \frac{1}{k(k+1)}', '0.8'), (r'\sum_{S=2}^6 \binom{S-1}{4}', '6'),
                    (r'\sum_{k=1}^{n} k', None), (r'\sum x_i', None)])

    def test_remainders(self):
        self.check([(r'2^{100} \bmod 7', '2'), (r'3^{2024} \pmod{1000}', str(pow(3, 2024, 1000))), (r'17 \bmod 5', '2'),
                    (r'2016 \pmod{1000}', '16'), ('491 mod 3', '2'), (r'2^{100}', None)])

    def test_double_factorial_bases_and_repeating_decimals(self):
        self.check([(r'(12-1)!!', '10395'), (r'8!!', '384'), (r'17_{49}', '56'), (r'97_{21} - 17_{21}', '168'), (r'19_{8}', None),
                    (r'0.\overline{49}', '49/99'), (r'0.\overline{4}', '4/9'), (r'0.4\overline{9}', '0.5'), (r'0.1\overline{09}', '6/55')])

    def test_degrees_by_context(self):
        self.assertEqual(x.spell(x.evaluate(r'\tan(45)', degrees=True)), '1')
        self.assertEqual(x.spell(x.evaluate(r'-\cot 135', degrees=True)), '1')
        self.assertIsNone(x.evaluate(r'\tan(45)'))
        self.assertEqual(ud.evaluate_exact(r'\sin 30', 'the angle is 60^\circ, so '), Fraction(1, 2))
        self.assertIsNone(ud.evaluate_exact(r'\sin 30', 'in radians, '))

    def test_what_reads_two_ways_is_refused(self):
        self.check([(r'2\frac{1}{3}', None), ('3 5', None), ('2 (2)', None), (r'1/2\sqrt{3}', None), ('x + 1', None),
                    (r'|a| |b|', None)])

    def test_one_value_one_spelling(self):
        a, b = x.evaluate(r'\sqrt{8}'), x.evaluate(r'2\sqrt{2}')
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(x.spell(x.evaluate(r'\frac{\sqrt{6}-\sqrt{2}}{4}'), latex=False), '(sqrt(6) - sqrt(2))/4')

    def test_past_the_limit_is_no_step(self):
        self.check([(r'\prod_{k=1}^{60} k', None), (r'\sqrt{10^{30}+1}', None)])


class NoDefinitions(unittest.TestCase):
    def test_a_function_the_text_defined_is_not_read(self):
        # #306: over the stored control answers the latest `f(x) = ...` put a wrong value in at 11 places (a redefined f, h defined
        # by pieces), with no rule that told them from the right ones; a call is no value
        self.assertIsNone(val('f(3)'))
        self.assertIsNone(val('2f(1)'))


class Bridge(unittest.TestCase):
    def found(self, text):
        return inline_found(text, use_mathbackend=True)

    def test_what_the_traces_wrote(self):
        for text, want in [(r'So $4(117)', '468'), (r'$\binom{10}{4}', '210'), (r'and $\sqrt{1674}', r'3\sqrt{186}'),
                           (r'$\sin 30^\circ', '0.5'), (r'$\sum_{k=1}^{10} k^2', '385'), (r'$\log_2 8', '3'), (r'$17 \bmod 5', '2'),
                           (r'$10044/42', '1674/7'), (r'$6/26', '3/13'), ('491 mod 3', '2'), (r'$-4 \binom{3}{2}', '-12')]:
            with self.subTest(text=text):
                got = self.found(text)
                self.assertEqual(got and got[1], want)

    def test_lowest_terms_but_not_in_a_long_division(self):
        self.assertEqual(self.found('so $10044/42')[1], '1674/7')
        self.assertIsNone(self.found("Let's check $70308 / 6$:\n$70 / 6"))   # livebench-88407ca2: `= 11$ rem 4`
        self.assertIsNone(self.found('Is 11718 divisible by anything else?\n$1302 / 9'))   # livebench-ce2f15a7: `= 144.66`

    def test_the_plain_arm_is_unchanged(self):
        # no --mathbackend: mathexpr alone, and a fraction that reduces is still left to the long division
        self.assertIsNone(inline_found(r'$\sqrt{1674}'))
        self.assertIsNone(inline_found(r'$6/26'))
        self.assertIsNone(inline_found(r'$4(117)'))

    def test_nothing_new_is_not_put_in(self):
        for text in (r'$3\sqrt{186}', r'$\sqrt{2}', r'$\frac{\sqrt{6} - \sqrt{2}}{4}', r'$5 + \sqrt{2}', r'$(3)', r'$21/19'):
            with self.subTest(text=text):
                self.assertIsNone(self.found(text))

    def test_the_tail_of_a_longer_expression(self):
        # the head is unread, so the tail is not the value: `x^2 + \sqrt{4}`, `\rho - (1+\sqrt{5})` (contest-d827a551)
        for text in (r'$x^2 + \sqrt{4}', r'$\rho - (1+\sqrt{5})', r'$\lambda \sqrt{8}', r'$y 4(117)', r'$(26/5) / \sqrt{48/5}x'):
            with self.subTest(text=text):
                self.assertIsNone(self.found(text))

    def test_a_longer_expression_than_mathexprs(self):
        # mathexpr reads `4 + 1` at the end of `\sqrt{2} \cdot 4 + 1`; the whole is the value
        self.assertEqual(self.found(r'$\sqrt{2} \cdot 4 + 1')[1], r'1 + 4\sqrt{2}')

    def test_two_values_in_two_math_spans(self):
        self.assertIsNone(self.found(r'$2$ and $\sqrt{8}$ $'))

    def test_out_of_sample_misreads(self):
        # seven other control arms (2026-11-06): each of these put in a value the model did not mean
        self.assertIsNone(self.found(r'$\cos \pi/4'))                   # cos(pi)/4 by the grammar, cos(pi/4) by the writer
        self.assertIsNone(self.found(r'$68 \div 58'))                   # dm div_big: a long division wants the digit 1
        self.assertIsNone(self.found('the mean is 8.33 and\n$100/12'))  # a text in decimals wants 8.33
        self.assertIsNone(self.found(r'$-192 + 6\sqrt{5}'))             # the same terms in another order
        self.assertIsNone(self.found('Let $f(x) = x + 2$.\n$f(1)^2'))     # a call is no value (#306)
        self.assertIsNone(self.found('$3+1 \\times 1+1'))               # (3+1)(1+1) by its spaces
        self.assertIsNone(self.found('$12(-66^2)'))                     # 12(-66)^2 as written for m = -66
        self.assertIsNone(self.found('$-1/4 / 1/2'))                    # (-1/4)/(1/2) by its spaces
        self.assertIsNone(self.found('$400/27 - 500/9 + 1080/27 = 400 - 1500 + 1080 / 27'))   # the numerators over 27

    def test_what_the_model_is_doing_there(self):
        # #306: the value is right for the text, not for the step: a long division's digit, a borrow, mod m, coins, a cut 5/3
        self.assertEqual(self.found('Bring down the 1. The remainder is 1.\n$1 \\div 2')[1], '0')   # the quotient's digit, not 0.5
        self.assertIsNone(self.found('    *   $38115 - 29799$.\n    *   $5-9'))
        self.assertEqual(self.found('Work $\\pmod{13}$.\n$(2^4)^2')[1], '256')   # no `=` for `≡` written yet: the full value, a true equation
        self.assertIsNone(self.found('In $\\mathbb{Z}_5$, $2^{-1} \\equiv 3$. So\n$3/2'))
        self.assertIsNone(self.found('Using 10 cent and 1 cent coins: $2 \\times 10 + 5 \\times 1'))
        self.assertIsNone(self.found('Tens: $3 + 8 + 4 + 6 + 2 + 3 + 2'))
        self.assertIsNone(self.found('Slope $= (0-1)/(3-1.66)'))
        self.assertEqual(self.found('$2 \\times 16')[1], '32')        # none of that around it: the value goes in

    def test_a_function_call_is_not_read(self):
        text = 'We have $f(y) = 9y^4 - 18y^3 - 143y^2 - 84y + 96$.\n$f(2)'
        self.assertIsNone(self.found(text))   # #306: a redefined or piecewise f was misread at 11 places


class GatedCalls(unittest.TestCase):
    def test_c_and_p_in_counting_text(self):
        ctx = 'The number of ways to choose 3 of the 10 people is '
        self.assertEqual(ud.evaluate_exact('C(10, 3)', ctx), 120)
        self.assertEqual(ud.evaluate_exact('C(10, 3) \\times P(5, 2)', ctx + ' and the arrangements '), 2400)

    def test_c_and_p_as_points(self):
        self.assertIsNone(ud.evaluate_exact('C(10, 3)', 'The point C(10, 3) lies on the circle; the number of ways to choose'))
        self.assertIsNone(ud.evaluate_exact('C(10, 3)', 'Let the vertices be A, B and '))
        self.assertIsNone(ud.evaluate_exact('P(2, 3)', 'the permutations of P(2, 3)'))   # k > n: a point, not 0

    def test_greek_letters_only_where_the_text_names_the_function(self):
        self.assertEqual(ud.evaluate_exact('\\sigma(12)', 'the sum of divisors of 12 is '), 28)
        self.assertEqual(ud.evaluate_exact('\\tau(12)', 'the number of divisors '), 6)
        self.assertEqual(ud.evaluate_exact('\\phi(12)', "Euler's totient "), 4)
        self.assertEqual(ud.evaluate_exact('\\mu(30)', 'the Mobius function '), -1)
        self.assertIsNone(ud.evaluate_exact('\\sigma(1)', 'the permutation sends 1 to 3, so '))
        self.assertIsNone(ud.evaluate_exact('\\tau(2)', 'the step function at 2 '))


if __name__ == '__main__':
    unittest.main()
