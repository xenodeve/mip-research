"""tools/igc/standard_score.py: the HLE rule. python tools/test_igc_standard_score.py"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc.standard_score import hle_right


def reply(line):
    return '<think>\nwork\n</think>\n\nExplanation: x\n' + line + '\nConfidence: 60%'


class Hle(unittest.TestCase):
    def test_one_value(self):
        self.assertTrue(hle_right(reply('Exact Answer: 5'), '5'))
        self.assertTrue(hle_right(reply('Exact Answer: n = 5'), '5'))
        self.assertTrue(hle_right(reply('Exact Answer: \\frac{3}{4}'), '3/4'))
        self.assertTrue(hle_right(reply('Exact Answer: 1,024 ways'), '1024'))
        self.assertTrue(hle_right(reply('Exact Answer: 3.14'), '3.15'))   # one unit in the gold's last place

    def test_hedged_line_is_no_answer(self):
        # hle-on1 6721b01e63e0 (#288): the bench rule took the first number of a hedged line and scored it right
        self.assertFalse(hle_right(reply('Exact Answer: cannot be determined; at most 5, possibly 0'), '5'))
        self.assertFalse(hle_right(reply('Exact Answer: 5 or 0'), '5'))
        self.assertFalse(hle_right(reply('Exact Answer: between 3 and 5'), '5'))

    def test_caveat_in_brackets_keeps_the_value(self):
        # hle-on1 670147f7dc19: the reader scored it right, the value before the bracket is the answer
        self.assertTrue(hle_right(reply('Exact Answer: 3 (for smooth functions; 4 if Morse/nondegenerate is required)'), '3'))

    def test_repeated_value_is_one_value(self):
        self.assertTrue(hle_right(reply('Exact Answer: 5 (that is, 5 rooms)'), '5'))

    def test_no_line_no_answer(self):
        self.assertFalse(hle_right('<think>a</think>\n\nThe answer is 5.', '5'))



class Math(unittest.TestCase):
    def test_final_answer_line_with_the_value_below(self):
        from tools.igc.standard_score import math_right
        text = '</think>\n\nThe final answer is:\n\\boxed{-8038578.5}<|im_end|>'
        self.assertTrue(math_right(text, '-16077157/2'))   # dmx_div_big-1649: was scored on `:`

    def test_final_answer_sentence_still_wins(self):
        from tools.igc.standard_score import math_right
        self.assertTrue(math_right('</think>\n\nThe final answer is 12. I hope it is correct.', '12'))

    def test_a_sentence_before_the_box_does_not_take_the_answer(self):
        """#307 rerun, dm_div_big-6241 s1: `the final answer is negative.` above `\\boxed{-3387650}` was read as `negative`."""
        from tools.igc.standard_score import math_right
        text = ('</think>\n\nSince the dividend is negative and the divisor is positive, the final answer is negative.\n\n'
                '$$ -4,027,915,850 \\div 1,189 = -3387650 $$\n\n\\boxed{-3387650}<|im_end|>')
        self.assertTrue(math_right(text, '-3387650'))
        self.assertFalse(math_right(text, '3387650'))
        # the later one wins either way: a sentence after the box is still the answer
        self.assertTrue(math_right('</think>\n\n\\boxed{11}\n\nThe final answer is 12. I hope it is correct.', '12'))

    def test_the_same_number_written_as_another_expression(self):
        from tools.igc.standard_score import math_right
        self.assertTrue(math_right('</think>\n\n\\boxed{26! 2^{25}}<|im_end|>', '2^{25}\\cdot26!'))   # r1-ctrl a6f7b142
        self.assertFalse(math_right('</think>\n\n\\boxed{26! 2^{24}}<|im_end|>', '2^{25}\\cdot26!'))
        self.assertFalse(math_right('</think>\n\n\\boxed{598.5377138413301}<|im_end|>', '1539439/2572'))   # a rounded quotient stays wrong

    def test_a_root_in_another_spelling(self):
        """#307 A100 General arm: every finished simplify_surd answer was right and scored wrong (13 of 13)."""
        from tools.igc.standard_score import math_right
        for reply, gold in (('\\boxed{7203 + 384\\sqrt{7}}', '384*sqrt(7) + 7203'),
                            ('\\boxed{-\\frac{11\\sqrt{2}}{3}}', '-11*sqrt(2)/3'),
                            ('\\boxed{45\\sqrt{13} - 5}', '-5 + 45*sqrt(13)'),
                            ('\\boxed{10\\sqrt{13}}', '10*sqrt(13)')):
            self.assertTrue(math_right('</think>\n\n' + reply, gold), reply)
        self.assertFalse(math_right('</think>\n\n\\boxed{45\\sqrt{13} + 5}', '-5 + 45*sqrt(13)'))
        self.assertFalse(math_right('</think>\n\n\\boxed{10\\sqrt{17}}', '10*sqrt(13)'))

    def test_a_set_of_answers_in_any_order(self):
        """#307 A100 General arm: `-2, 1` for gold `1,-2` and both roots for `1\\pm\\sqrt{19}` were right and scored wrong."""
        from tools.igc.standard_score import math_right
        self.assertTrue(math_right('</think>\n\n\\boxed{-2, 1}', '1,-2'))
        self.assertTrue(math_right('</think>\n\n\\boxed{1-\\sqrt{19}, 1+\\sqrt{19}}', '1\\pm\\sqrt{19}'))
        self.assertTrue(math_right('</think>\n\n\\boxed{1 \\pm \\sqrt{19}}', '1\\pm\\sqrt{19}'))
        self.assertFalse(math_right('</think>\n\n\\boxed{-2}', '1,-2'))            # half the set
        self.assertFalse(math_right('</think>\n\n\\boxed{-2, 1, 3}', '1,-2'))      # one too many
        self.assertFalse(math_right('</think>\n\n\\boxed{1+\\sqrt{19}}', '1\\pm\\sqrt{19}'))
        self.assertFalse(math_right('</think>\n\n\\boxed{2, -1}', '1,-2'))

    def test_a_set_in_set_braces(self):
        """Round 9 a3 (4B NVFP4 W4A8, llama.cpp, PC layer) stdmath-bf7ece2b x2: `\\{-2, 1-\\sqrt{5}, 1+\\sqrt{5}\\}` for gold
        `\\{1\\pm\\sqrt{5},-2\\}` was right and scored wrong: the set braces stayed on the first and last value."""
        from tools.igc.standard_score import math_right
        gold = '\\{1\\pm\\sqrt{5},-2\\}'
        self.assertTrue(math_right('</think>\n\n\\boxed{\\{-2, 1-\\sqrt{5}, 1+\\sqrt{5}\\}}', gold))
        self.assertTrue(math_right('</think>\n\n\\boxed{\\left\\{-2, 1+\\sqrt{5}, 1-\\sqrt{5}\\right\\}}', gold))
        self.assertTrue(math_right('</think>\n\n\\boxed{-2, 1\\pm\\sqrt{5}}', gold))
        self.assertFalse(math_right('</think>\n\n\\boxed{\\{-2, 1+\\sqrt{5}\\}}', gold))           # one value missing
        self.assertFalse(math_right('</think>\n\n\\boxed{\\{2, 1-\\sqrt{5}, 1+\\sqrt{5}\\}}', gold))  # a wrong value

    def test_a_decimal_of_a_value_with_no_finite_decimal_is_right_with_a_form_note(self):
        """The developer (2026-11-11): a correctly rounded or cut decimal of a gold that has no finite decimal counts as right, with a
        note that the form is not gold's (round 9 1395#0 boxed `-1.9886` for `-73388/36903`). The integer part, `0`, or a decimal whose
        digits are wrong is a wrong value, not a form."""
        from tools.igc.standard_score import form_note, math_right
        gold = '-73388/36903'   # -1.98867300...
        for box in ('-1.9886', '-1.9887', '-1.98867', '-1.988673008', '-1.9886730...'):
            text = '</think>\n\n\\boxed{' + box + '}'
            self.assertTrue(math_right(text, gold), box)
            self.assertEqual(form_note(text, gold), f'decimal for an exact value: answered {box}, gold {gold}', box)
        for box in ('-1', '-2', '0', '-2.0', '-1.9885', '1.9886', '-1.9868'):   # integer part, too few digits, wrong digits, wrong sign
            self.assertFalse(math_right('</think>\n\n\\boxed{' + box + '}', gold), box)
        self.assertTrue(math_right('</think>\n\n\\boxed{0.3333}', '1/3'))
        self.assertFalse(math_right('</think>\n\n\\boxed{1347.85}', '24000*(101/100)**30/24'))   # a finite decimal exists (2^a 5^b)
        self.assertTrue(math_right('</think>\n\n\\boxed{1.7320508}', '\\sqrt{3}'))
        self.assertFalse(math_right('</think>\n\n\\boxed{0.13}', '1/8'))     # 1/8 has the finite decimal 0.125: rounding it is another value
        self.assertIsNone(form_note('</think>\n\n\\boxed{-73388/36903}', gold))   # the exact form: no note
        self.assertFalse(math_right('</think>\n\n\\boxed{-10710884}', '-139241497/13'))   # the integer part

    def test_a_box_asked_about_and_rejected_is_not_the_answer(self):
        """#24, round 9 (4B NVFP4 W4A8, llama.cpp, PC layer) dm_div_big-9977#0 and dm_simplify_surd-9937#0: the last box was written
        as a question and rejected on its line."""
        from tools.igc.standard_score import math_right
        text = ('</think>\n\nThe final answer is \\boxed{-70468}.\n\n\\boxed{-70468}\n\n'
                'The final answer is $\\boxed{70468}$? No, dividend is negative.\n')
        self.assertTrue(math_right(text, '-70468'))
        self.assertFalse(math_right(text, '70468'))
        text = '</think>\n\nThe final answer is $\\boxed{899940 + 7216\\sqrt{13}}$? No, it\'s negative.\n'
        self.assertFalse(math_right(text, '-899940 - 7216*sqrt(13)'))   # nothing stated is left
        self.assertFalse(math_right(text, '899940 + 7216*sqrt(13)'))
        self.assertTrue(math_right('</think>\n\nIs it $\\boxed{5}$?\n\nYes: \\boxed{5}', '5'))

    def test_two_values_boxed_on_the_last_line_is_hedged(self):
        """Round 9 dm_mul_div_longer-5942#1: `The final answer is $\\boxed{275}$ or $\\boxed{-275}$.` was scored right."""
        from tools.igc.standard_score import math_right
        text = '</think>\n\nThe final answer is $\\boxed{275}$ or $\\boxed{-275}$.\n'
        self.assertFalse(math_right(text, '-275'))
        self.assertFalse(math_right(text, '275'))
        self.assertTrue(math_right('</think>\n\nSo $\\boxed{275}$, that is $\\boxed{275}$.', '275'))   # the same value twice

    def test_a_vector_with_decimals_for_fractions(self):
        """Round 9 stdmath-2aff3e30#0: `(0.2, -3.6)` for gold `(1/5, -18/5)` was scored wrong."""
        from tools.igc.standard_score import math_right
        gold = '\\begin{pmatrix}1/5\\\\-18/5\\end{pmatrix}'
        self.assertTrue(math_right('</think>\n\n\\boxed{\\begin{pmatrix} 0.2 \\\\ -3.6 \\end{pmatrix}}<|im_end|>', gold))
        self.assertFalse(math_right('</think>\n\n\\boxed{\\begin{pmatrix} 0.2 \\\\ 3.6 \\end{pmatrix}}<|im_end|>', gold))
        self.assertFalse(math_right('</think>\n\n\\boxed{\\begin{pmatrix} 0.2 \\end{pmatrix}}<|im_end|>', gold))
        gold = '\\begin{pmatrix}1/5\\-18/5\\end{pmatrix}'   # the stored gold: rows split by a lone backslash
        self.assertTrue(math_right('</think>\n\n\\boxed{\\begin{pmatrix} 0.2 \\\\ -3.6 \\end{pmatrix}}<|im_end|>', gold))
        self.assertFalse(math_right('</think>\n\n\\boxed{\\begin{pmatrix} 0.2 \\\\ 3.6 \\end{pmatrix}}<|im_end|>', gold))

    def test_an_inequality_for_an_interval(self):
        """Round 9 stdmath-9f7f1563#0: `3 < \\lambda \\le 4` for gold `(3,4]` was scored wrong."""
        from tools.igc.standard_score import math_right
        self.assertTrue(math_right('</think>\n\n\\boxed{3 < \\lambda \\le 4}<|im_end|>', '(3,4]'))
        self.assertTrue(math_right('</think>\n\n\\boxed{3 \\leq x < 4}', '[3,4)'))
        self.assertFalse(math_right('</think>\n\n\\boxed{3 \\le \\lambda \\le 4}', '(3,4]'))
        self.assertFalse(math_right('</think>\n\n\\boxed{3 < \\lambda < 5}', '(3,4]'))


if __name__ == '__main__':
    unittest.main()
