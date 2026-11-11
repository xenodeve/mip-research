"""question_quote.py: the question's own words, put back when the model's copy of them differs (round 9 P4, P5; replaces --exact-answer's
box rewrite, which wrote the answer for the model: docs/ERRATA.md).

The case, stated without a test set: the model copies the question's expression into its reasoning and the copy says something else
(a bracket moved, a sign lost), then works on the copy. Published work names this a number-copy / transcription error (GSM-Ranges,
arXiv 2502.08680; error classification, arXiv 2609.15145: 6.3 % of reasoning processes), and re-reading the question improves reasoning
(RE2, arXiv 2309.06275, EMNLP 2024). Round 9: `dm_mul_div_longer-1505#0` moved a bracket (`((-25)/((-3025)/(-44))*1/(-4))` copied as
`(-25)/((-3025)/(-44) \\cdot 1/(-4))`) and boxed -12.375 for -198.

What is put is the question's text, verbatim: `The question writes: <expression>.` It is true under every reading and decides nothing:
the model compares and chooses. The trigger: a finished line of the model's that holds every number of the question's expression, in
order and nothing else numeric, and reads to another value than the question's. A copy with the same value, a line that cannot be read,
or a line with other numbers (a step of the work) puts nothing.

copy_differs(line, numbers, value) -> bool
QUOTE.format(expr=...) -> the line put
"""
import re
from fractions import Fraction

QUOTE = 'The question writes: ${expr}$.\n'
_NUM = re.compile(r'\d+(?:\.\d+)?')


def numbers_of(expr):
    return _NUM.findall(expr)


def _body(line):
    """The expression a line states: the text after the last `:` (a label), without `$`, `\\[ \\]`, a trailing `=` or punctuation."""
    s = line.strip()
    s = s.split(':')[-1] if ':' in s else s
    s = s.replace('$$', '').replace('$', '').replace('\\[', '').replace('\\]', '').strip()
    s = re.sub(r'^\s*(?:[-*]\s+|\d+\.\s+)', '', s)
    s = re.sub(r'[\s=.,;]+$', '', s)
    return s


def copy_differs(line, numbers, value):
    if len(numbers) < 3 or numbers_of(line) != numbers:
        return False
    from tools.igc.mathexpr import evaluate
    from tools.igc.question_read import _plain
    body = _body(line)
    if not body:
        return False
    got = evaluate(_plain(body))
    if got is None:
        return False
    return Fraction(got) != Fraction(value) if not hasattr(value, 'has_pi') else True
