"""approx_put.py: the value after `\\approx` (round 9 P11, docs/failures.md).

The case, stated without a test set: the model writes `X \\approx` where X is arithmetic software can compute, and its own decimal can be
wrong (round 9 General: 316 of 1,952 readable `\\approx` values off; `(1.01)^{30} \\approx 1.3448888` for 1.3478489). Published work shows
the same on decimal powers and logs (MATH 401, arXiv 2304.02015; MedCalc-Bench audit, arXiv 2603.02222). A value to 10 significant
digits is a true approximation under every reading of `\\approx`, a deliberate estimate (`\\approx 34000`) included, so no rule for
estimates is needed. X is read by mathexpr in its wide mode: the value is spelled as a decimal, so its exact size does not matter.

approx_spelled(value) -> str          10 significant digits; `d.ddd \\times 10^{k}` below 1e-4 or from 1e10
approx_found(before) -> (expr, str)   the arithmetic that ends `before` and its value spelled, or None
"""
from decimal import ROUND_HALF_EVEN, Context, Decimal
from fractions import Fraction

DIGITS = 10


def approx_spelled(value):
    value = Fraction(value)
    if value == 0:
        return '0'
    ctx = Context(prec=DIGITS, rounding=ROUND_HALF_EVEN)
    d = ctx.divide(Decimal(value.numerator), Decimal(value.denominator))
    sign, digits, exp = d.as_tuple()
    digits = ''.join(map(str, digits)).rstrip('0') or '0'
    top = len(''.join(map(str, d.as_tuple().digits))) + exp - 1   # power of ten of the first digit
    if -4 <= top < 10:
        text = format(d.normalize(), 'f')
        return text
    mantissa = digits[0] + ('.' + digits[1:] if len(digits) > 1 else '')
    return ('-' if sign else '') + f'{mantissa} \\times 10^{{{top}}}'


def approx_found(before):
    from tools.igc.mathexpr import expression_before, wide
    with wide():
        got = expression_before(before.rstrip().rstrip('$').rstrip())
    if got is None:
        return None
    expr, value = got
    return expr, approx_spelled(value)
