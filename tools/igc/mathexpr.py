"""Exact arithmetic in every way a model writes it: ASCII, LaTeX at any depth, Unicode. The reader MIP's inline value and hand check run on.

The regex rewrite it replaces read one LaTeX layer and cut numbers at their commas: on Qwenputer's DeepMind rows the 4B rewrote the
question's nested expression as \\frac inside \\frac and it was never computed (14 rows lost), and `12,510,204 \\times 100` was read as
204 x 100. Other models write other forms, so the reader takes them all and refuses only what is not exact arithmetic: a symbol, a word, a
mixed number (`2\\frac{1}{3}` is ambiguous), a root or power that is not exact, a value too large to be a step.

evaluate(text) -> Fraction | None            the value of the whole text, or None
expression_before(text) -> (expr, value)     the longest arithmetic expression that ends the text (what precedes an `=`), or None
simple_quotient(expr) -> bool                an integer over an integer and nothing else (a long division's piece)"""
import os
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from fractions import Fraction
from math import factorial, isqrt

# MIP_EXPR_DEVICE=cuda (a measurement, 2026-11-12): the parser's + - * / go to the CUDAMathBackend (numbers.add/sub/mul/div, canonical bytes
# through the C ABI), one call per operation, the way the device would serve this reader today. The device's value is used when it is a
# value; a refusal or a device status falls back to Fraction's. Every call is compared with Fraction and counted per thread in DEVICE
# (calls, seconds, mismatches, fallbacks), so a run reports what the device cost and whether it ever differed. Unset: Fraction only.
_DEVICE = os.environ.get('MIP_EXPR_DEVICE', '') == 'cuda'
DEVICE = threading.local()
_DEVICE_OPS = {'+': 'numbers.add', '-': 'numbers.sub', '*': 'numbers.mul', '/': 'numbers.div'}


def device_stats():
    """this thread's device counters (calls, seconds, mismatches, fallbacks), created at zero"""
    if not hasattr(DEVICE, 'calls'):
        DEVICE.calls, DEVICE.seconds, DEVICE.mismatches, DEVICE.fallbacks = 0, 0.0, 0, 0
    return DEVICE


def _arith(op, a, b):
    py = a + b if op == '+' else a - b if op == '-' else a * b if op == '*' else a / b
    if not _DEVICE:
        return py
    from tools.igc.mathbackend import canon, cuda_backend
    st = device_stats()
    t = time.perf_counter()
    out = cuda_backend.call(_DEVICE_OPS[op], canon.encode([a, b]))
    st.seconds += time.perf_counter() - t
    st.calls += 1
    try:
        got = canon.decode(out)
    except Exception:
        got = None
    if isinstance(got, (int, Fraction)) and not isinstance(got, bool):
        got = Fraction(got)
        if got != py:
            st.mismatches += 1
        return got
    st.fallbacks += 1
    return py

# wide(): the size limits below keep a put at `=` to a value a step can hold; a put after `\approx` is spelled to 10 significant digits,
# so `(1.01)^{30}` (a 61-digit exact fraction) is read there (round 9 stdmath-45fa3ef2#0: the model wrote 1.3448888 for 1.3478489)
_WIDE = ContextVar('mathexpr_wide', default=False)


@contextmanager
def wide():
    token = _WIDE.set(True)
    try:
        yield
    finally:
        _WIDE.reset(token)

MUL, DIV, POW = '*', '/', '^'
_SPACE = re.compile(r'\\[,;:! ]|\\q?quad\b|~')
_SIZES = re.compile(r'\\(?:left|right|[bB]ig{1,2}[lr]?|middle)\s*')
_SUPERSCRIPT = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'}
_WORDS = {r'\times': MUL, r'\cdot': MUL, r'\ast': MUL, r'\div': DIV, r'\frac': 'frac', r'\dfrac': 'frac', r'\tfrac': 'frac', r'\sqrt': 'sqrt',
          r'\%': '%', r'\lparen': '(', r'\rparen': ')',   # not `\{ \}`: a set (`\{2\} \times \{2\}` got 4 put in, #306)
          r'\bmod': 'mod', r'\mod': 'mod', r'\pmod': 'pmod', r'\lvert': 'abs', r'\rvert': 'abs', r'\vert': 'abs',
          r'\lfloor': 'lfloor', r'\rfloor': 'rfloor', r'\lceil': 'lceil', r'\rceil': 'rceil', r'\binom': 'binom', r'\dbinom': 'binom',
          r'\tbinom': 'binom'}
_SINGLE = {'*': MUL, '×': MUL, '·': MUL, '⋅': MUL, '∙': MUL, '/': DIV, '÷': DIV, '^': POW, '+': '+', '-': '-', '−': '-', '–': '-',
           '(': '(', ')': ')', '[': '(', ']': ')', '{': '{', '}': '}', '!': '!', '%': '%', '|': 'abs', '⌊': 'lfloor', '⌋': 'rfloor',
           '⌈': 'lceil', '⌉': 'rceil', ',': ','}
_DEGREES = re.compile(r'\^\s*\{?\s*\\circ\s*\}?|°')   # 180^\circ - 30^\circ: a unit, not a power
_NUMBER = re.compile(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+')
LIMIT = 10 ** 40   # a numerator past this is no step of a calculation
DENOMINATOR = 10 ** 40   # 0.7412898 \times 10^{-9} is a step too: 10^15 refused it and the reader took 10^{-9} alone


class _NotExact(Exception):
    pass


def _tokens(text):
    s = re.sub(r'(?<=\d)\\,(?=\d{3}(?!\d))', '', text)   # 1\,000: a thin space between digit groups, before it becomes a space
    s = _SIZES.sub('', _SPACE.sub(' ', _DEGREES.sub('', s)))
    s = ''.join('^' + _SUPERSCRIPT[c] if c in _SUPERSCRIPT else c for c in s).replace('**', '^')
    out, i, spaced, gap = [], 0, set(), False
    while i < len(s):
        c = s[i]
        if c.isspace():
            i, gap = i + 1, True
            continue
        if gap:   # the next token comes after a space: `2 (2)` is no product
            spaced.add(len(out))
            gap = False
        if s.startswith('mod', i) and (i == 0 or not s[i - 1].isalpha()) and not s[i + 3:i + 4].isalpha():
            out.append(('op', 'mod'))   # 2016 mod 1000: the one word read as an operator
            i += 3
            continue
        m = _NUMBER.match(s, i)
        if m and (c.isdigit() or c == '.'):
            raw = m.group().replace(',', '')
            out.append(('num', Fraction(raw) if 'e' not in raw.lower() else Fraction(raw.lower())))
            i = m.end()
            if i < len(s) and (s[i].isalpha() or s[i] == '_' or (s[i] == '.' and i + 1 < len(s) and s[i + 1].isdigit())):
                raise _NotExact('a number glued to a symbol, or 1.2.3')
            continue
        if c == '\\':
            m = re.match(r'\\(?:[A-Za-z]+|.)', s[i:])
            if m is None:   # a backslash before a line end or at the end: `\=` crashed the reader with AttributeError (2026-11-06)
                raise _NotExact('a backslash at the end of a line')
            word = m.group()
            if word not in _WORDS:
                raise _NotExact(word)   # \pi, \text, \cdots: not arithmetic
            out.append(('op', _WORDS[word]))
            i += len(word)
            continue
        if c in _SINGLE:
            out.append(('op', _SINGLE[c]))
            i += 1
            continue
        raise _NotExact(c)   # a letter, a symbol, `$`, `=`
    return out, spaced


class _Parser:
    def __init__(self, tokens, spaced=frozenset()):
        self.t, self.i, self.spaced = tokens, 0, spaced
        self.loose = None   # the last term had a spaced product or quotient

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, value=None):
        kind, v = self.peek()
        if kind is None or (value is not None and v != value):
            raise _NotExact('expected ' + str(value))
        self.i += 1
        return kind, v

    def expr(self):
        value = self.term()
        loose = self.loose   # a spaced product or quotient in the first term
        while self.peek()[1] in ('+', '-'):
            if loose is not None and not (self.i in self.spaced or self.i + 1 in self.spaced):
                # `3+1 \times 1+1 = 8` meant (3+1)(1+1): a tight sum around a spaced product groups against the grammar (#306)
                raise _NotExact('the spacing groups a sum inside a product')
            op = self.take()[1]
            right = self.term()
            loose = loose if self.loose is None else self.loose
            value = _arith('+' if op == '+' else '-', value, right)
        if self.peek()[1] == 'pmod':
            # `9 + 54 \pmod{10}` is (9 + 54) mod 10 = 3, not 9 + (54 mod 10) = 13: \pmod takes all that comes before (#306, held-out)
            self.take()
            value = self.mod(value, self.argument())
        return value

    def term(self):
        value = self.unary()
        divided = None   # how the last division in this term was spaced
        loose = None
        while True:
            kind, v = self.peek()
            if v in (MUL, DIV) and (self.i in self.spaced or self.i + 1 in self.spaced):
                loose = True
            if v in (MUL, DIV, 'mod'):
                if v == DIV:
                    gap = self.i in self.spaced or self.i + 1 in self.spaced
                    if divided and not gap:
                        # `-1/4 / 1/2` meant (-1/4)/(1/2), `a / b/c` a/(b/c): a tight `/` after a spaced one is grouped by the spaces against
                        # the grammar (#306). `3005/6 / 5` reads one way either way: the spaces and left to right agree (#307)
                        raise _NotExact('a/b / c/d: the spacing groups it, the grammar does not')
                    divided = bool(divided) or gap
                self.take()
                right = self.unary()
                if v != MUL and right == 0:
                    raise _NotExact('division by zero')
                if v == 'mod':
                    value = self.mod(value, right)
                else:
                    value = _arith('*' if v == MUL else '/', value, right)
            elif kind == 'num' and self.i and self.t[self.i - 1][1] == '!':
                value = _arith('*', value, self.unary())   # \frac{9!}{4! 3! 2!}
            elif v in ('(', 'frac', 'sqrt', '{', 'binom') and self.i and self.t[self.i - 1][1] in (')', '}', '!', '%') or \
                    v in ('(', 'sqrt') and self.i and self.t[self.i - 1][0] == 'num' and self.i not in self.spaced or \
                    v == 'binom' and self.i and self.t[self.i - 1][0] == 'num':
                # 2(3 + 4), (2)(3), \binom{9}{6} \binom{3}{2}, 3\binom{4}{2}
                value = _arith('*', value, self.unary())   # 2(3 + 4), (2)(3), (a)\frac{..}{..}
            elif v == 'frac' and self.i and self.t[self.i - 1][0] == 'num':
                raise _NotExact('a mixed number: 2\\frac{1}{3} is 2 1/3 or 2/3')
            elif kind == 'num' and self.i and self.t[self.i - 1][1] == ')':
                value = _arith('*', value, self.unary())   # (2)3
            else:
                self.loose = loose   # read by expr(), whose terms these are; a bracket inside set its own and is done
                return value
            self.bound(value)

    def unary(self):
        if self.peek()[1] in ('-', '+'):
            opened = self.i and self.t[self.i - 1][1] == '('
            sign = -1 if self.take()[1] == '-' else 1
            if sign < 0 and opened and self.peek()[0] == 'num' and self.i + 1 < len(self.t) and self.t[self.i + 1][1] == POW:
                raise _NotExact('(-66^2): -(66^2) by the grammar, (-66)^2 as written for m = -66')   # #306
            return sign * self.unary()
        return self.power()

    def power(self):
        literal = self.peek()[0] == 'num'
        base = self.postfix()
        # a number written 1.8329827 \times 10^7 is one number: 37 / 1.8329827 \times 10^7 means 37 / (1.8329827 \times 10^7) (dm div_big 2289)
        if literal and self.peek()[1] == MUL and self.i + 2 < len(self.t) and self.t[self.i + 1] == ('num', 10) and self.t[self.i + 2][1] == POW:
            self.i += 3
            exponent = self.unary_exponent()
            if exponent.denominator != 1 or abs(exponent) > 64:
                raise _NotExact('a power of ten that is not a small integer one')
            return self.bound(base * Fraction(10) ** int(exponent))
        if self.peek()[1] == POW:
            self.take()
            exponent = self.unary_exponent()
            if exponent.denominator != 1 or abs(exponent) > 64:
                raise _NotExact('a power that is not a small integer one')
            if base == 0 and exponent == 0:
                raise _NotExact('0^0: 1 by one convention, undefined by another')   # `0^3 + 0^0 + 2014 \cdot 0^0 = 2014` (#306, held-out)
            if base == 0 and exponent < 0:
                raise _NotExact('0 to a negative power')
            if abs(base.numerator) > 1 and abs(exponent) * max(len(str(abs(base.numerator))), len(str(base.denominator))) > (2000 if _WIDE.get() else 60):
                raise _NotExact('too large')
            base = base ** int(exponent)
        return self.bound(base)

    def unary_exponent(self):
        if self.peek()[1] in ('-', '+'):
            sign = -1 if self.take()[1] == '-' else 1
            return sign * self.unary_exponent()
        return self.postfix()

    def postfix(self):
        value = self.primary()
        while self.peek()[1] in ('!', '%'):
            op = self.take()[1]
            if op == '!' and self.peek()[1] == '!' and self.i not in self.spaced:   # `3!! = 3`: the double factorial (#306)
                self.take()
                if value.denominator != 1 or not -1 <= value <= 60:
                    raise _NotExact('a double factorial out of reach')
                out = 1
                for j in range(int(value), 0, -2):
                    out *= j
                value = Fraction(out)
            elif op == '!':
                if value.denominator != 1 or not 0 <= value <= 30:
                    raise _NotExact('a factorial past 30!')
                value = Fraction(factorial(int(value)))
            else:
                value = value / 100
        return value

    @staticmethod
    def mod(a, m):
        if a.denominator != 1 or m.denominator != 1 or m == 0:
            raise _NotExact('mod of a non-integer')
        return Fraction(int(a) % int(m))

    def primary(self):
        kind, v = self.peek()
        if kind == 'num':
            self.take()
            return v
        if v in ('(', '{'):
            self.take()
            value = self.expr()
            self.take(')' if v == '(' else '}')
            return value
        if v == 'frac':
            self.take()
            top, bottom = self.argument(), self.argument()
            if bottom == 0:
                raise _NotExact('division by zero')
            return top / bottom
        if v == 'abs':
            self.take()
            value = self.expr()
            self.take('abs')
            return abs(value)
        if v in ('lfloor', 'lceil'):
            self.take()
            value = self.expr()
            self.take('rfloor' if v == 'lfloor' else 'rceil')
            return Fraction(value.numerator // value.denominator if v == 'lfloor' else -(-value.numerator // value.denominator))
        if v == 'binom':
            self.take()
            n = self.argument()
            self.take('{')
            ks = [self.expr()]
            while self.peek()[1] == ',':
                self.take()
                ks.append(self.expr())
            self.take('}')
            if any(x.denominator != 1 or x < 0 for x in [n] + ks) or n > 1000:
                raise _NotExact('a binomial of a non-integer')
            if len(ks) == 1:
                k = int(ks[0])
                return Fraction(factorial(int(n)) // (factorial(k) * factorial(int(n) - k)) if k <= n else 0)
            if sum(ks) != n:
                raise _NotExact('a multinomial whose parts do not add up')
            out = factorial(int(n))
            for k in ks:
                out //= factorial(int(k))
            return Fraction(out)
        if v == 'sqrt':
            self.take()
            inside = self.argument()
            if inside < 0:
                raise _NotExact('the root of a negative')
            a, b = isqrt(inside.numerator), isqrt(inside.denominator)
            if a * a != inside.numerator or b * b != inside.denominator:
                raise _NotExact('a root that is not exact')
            return Fraction(a, b)
        raise _NotExact('expected a number')

    def argument(self):
        """\\frac and \\sqrt take a {group} or one digit (\\frac12)."""
        if self.peek()[1] == '{':
            self.take()
            value = self.expr()
            self.take('}')
            return value
        kind, v = self.take()
        if kind != 'num' or v.denominator != 1 or not 0 <= v <= 9:
            raise _NotExact('an argument with no braces')
        return v

    @staticmethod
    def bound(value):
        if not _WIDE.get() and (abs(value.numerator) > LIMIT or value.denominator > DENOMINATOR):
            raise _NotExact('too large to be a step')
        return value


def evaluate(text):
    try:
        tokens, spaced = _tokens(text)
        if not tokens:
            return None
        p = _Parser(tokens, spaced)
        value = p.expr()
        if p.i != len(tokens):
            return None
        return _Parser.bound(value)
    except (_NotExact, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None


_OPERATOR = re.compile(r'[+\-*/^×÷·⋅∙−!%]|\\(?:times|cdot|ast|div|d?t?frac|sqrt)|[²³]')
_START = re.compile(r'[\d.(\[{\\\-−]')
# `~104 + 104 = 208`, `13→2+1+3 = 6`, `441≡441-11*37` (a relation sign ends what comes before, #306 / #307)
_BOUNDARY = set(' \t$([{=:;,~≈→⇒≡≤≥<>≠') | {'\\'}


# commands after which a new operand starts: operators, relations, spacing, words, brackets' sizes
_NOT_AN_OPERAND = {'times', 'cdot', 'div', 'ast', 'pm', 'mp', 'le', 'ge', 'leq', 'geq', 'lt', 'gt', 'neq', 'ne', 'approx', 'equiv', 'sim',
                   'simeq', 'cong', 'implies', 'impliedby', 'iff', 'Rightarrow', 'Leftarrow', 'Leftrightarrow', 'rightarrow', 'leftarrow',
                   'to', 'mapsto', 'quad', 'qquad', 'text', 'textbf', 'textit', 'mathrm', 'mathbf', 'left', 'right', 'big', 'Big', 'bigg',
                   'Bigg', 'colon', 'therefore', 'because', 'in', 'notin', 'subset', 'subseteq', 'cup', 'cap', 'land', 'lor', 'mid',
                   'displaystyle', 'textstyle', 'hline', 'frac', 'dfrac', 'tfrac', 'sqrt', 'boxed', 'and', 'or', 'wedge', 'vee', 'neg',
                   'lfloor', 'lceil', 'lvert', 'langle', 'lbrace', 'operatorname', 'begin', 'end', 'item', 'pmod', 'bmod', 'mod'}
_TAIL_OPERATOR = re.compile(r'(?:[+\-*/^×÷·⋅∙−±∓]|\\times|\\cdot|\\div|\\ast|\\pm|\\mp)\s*$')


def _tail(head):
    """True when what starts after `head` is the tail of a longer expression: head ends in a binary operator (`SS / (2-1)`, `x - 3 \\times
    4`), or in a symbol command (`\\sum (1/4 \\times 1/4)`, `\\rho (2)`). Its value is not the expression's: --check-hand corrected
    `s^2 = SS / (2-1) = 264.5` to 1, and an inline value would have put 1 in (#306)."""
    h = head.rstrip()
    if re.fullmatch(r'\s*(?:[-*+•−]|\d+[.)])?', h.lstrip()) and re.fullmatch(r'\s*[-*+•−]?', h):
        return False   # `- (-1)/(-15)`, `*   2 \times 3`: a list's bullet opens the line, no operator
    if _TAIL_OPERATOR.search(h):
        return True
    cmd = re.search(r'\\([A-Za-z]+)$', h)
    if cmd:
        return cmd.group(1) not in _NOT_AN_OPERAND and not _FUNCTION_BEFORE.search(h)
    if re.search(r'(?<![A-Za-z])(?:lg|ln|log|sin|cos|tan|cot|sec|csc|exp|sqrt|arcsin|arccos|arctan)$', h):
        return True   # `lg 5 + 1 ≈ 1.69897`: the argument of a function written as a word (#306, R1)
    # a symbol before it: `2 BX \cdot CX (3/5) = 576` read (3/5) as the left side (#306); a point's name, a single letter
    return re.search(r'(?<![A-Za-z\\])(?:[A-Z]{1,3}|[a-z])(?:_\{?\w+\}?|\^\{?\w+\}?|\')*$', h) is not None


def _symbol_before(head):
    """True when the text before a `-` ends in an operand (a number, a bracket, a one-letter symbol, `b_2`): the `-` is then binary."""
    h = head.rstrip()
    if not h:
        return False
    if h[-1].isdigit() or h[-1] in ')]}|':   # `|x| - 1/2`: an absolute value ends an operand too (#306)
        return True
    cmd = re.search(r'\\([A-Za-z]+)$', h)
    if cmd:   # `\rho - (1 + \sqrt{5})`, `\pi - 1`: a Greek letter or a constant; `\times`, `\le`, `\quad` end no operand
        return cmd.group(1) not in _NOT_AN_OPERAND
    word = re.search(r'([A-Za-z]+)(?:_\{?\w+\}?|\^\{?\w+\}?|\d*|\')*$', h)
    return bool(word) and (len(word.group(1)) == 1 or '_' in h[word.start():] or '^' in h[word.start():])


def expression_before(text):
    """(expression, value) for the longest exact arithmetic expression that ends `text` on its last line, or None. Each start is tried from
    the left: the first that reads whole and holds an operator wins, so `12,510,204 \\times 100` is not cut at a comma and `-(25 * 130)`
    keeps its sign, while `x - 3 \\times 4` gives 3 x 4: the `-` after a symbol is not a sign."""
    line = text.rsplit('\n', 1)[-1].rstrip()
    for i in range(len(line)):
        if not _START.match(line[i]) or (i and line[i - 1] not in _BOUNDARY):
            continue
        if line[i] in '-−' and (_symbol_before(line[:i]) or not line[:i].strip() and line[i + 1:i + 2] == ' '):
            continue   # a minus after a symbol is binary; `- ` opening a line is a list's bullet (Qwenputer evals.py says the same)
        if line[i] == '(' and re.search(r'[A-Za-z]\S*\s*$', line[:i]) and _symbol_before(line[:i]):
            continue   # `u^2 (1 + \frac{169 \times 3}{841})`: a factor of a product on a symbol, no value of its own
        if re.search(r'[\d)}\]]\s+$', line[:i]) or _FUNCTION_BEFORE.search(line[:i]):
            continue   # `3^3 5^1`, `19600 (81/196)`: a factor beside a number; `\log_5 2^6`, `\sin 84^\circ`: a function's argument
        if _tail(line[:i]):
            continue   # `Z / (7/3)`, `SS / (2-1)`, `\sum (1/4 \times 1/4)`: the tail of a longer expression, not a value of its own (#306)
        expr = line[i:].strip()
        if not _OPERATOR.search(expr.lstrip('-−')):
            continue
        value = evaluate(expr)
        if value is not None:
            return expr, value
    return None


def spelled(value):
    """An integer as itself, a value with a terminating decimal as that decimal, and only the rest as a reduced fraction."""
    if value.denominator == 1:
        return str(value.numerator)
    d, twos, fives = value.denominator, 0, 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d != 1:
        return f'{value.numerator}/{value.denominator}'
    return rounded(value, max(twos, fives))


def rounded(value, places):
    """`value` as a decimal with `places` places, rounded half away from zero."""
    scaled = (abs(value) * 10 ** places * 2 + 1) // 2
    digits = str(scaled).rjust(places + 1, '0')
    sign = '-' if value < 0 and scaled else ''
    return sign + (digits[:-places] + '.' + digits[-places:] if places else digits)


_RELATION = re.compile(r'(?<![<>!=])(=|\\approx|≈|\\to\b|\\rightarrow|→|->)(?!=)')
_PLAIN = re.compile(r'-?\d[\d,]*(?:\.(\d+))?')
_SCIENTIFIC = re.compile(r'-?\d+(?:\.(\d+))?\s*(?:\\times|×|\*|\\cdot|x)\s*(?:10\s*\^\s*\{?\s*(-?\d+)\s*\}?|1(0+)(?![\d.]))')
_ROUGH = re.compile(r'\s*(?:\.\.\.|\\dots|\\ldots|…)')
_FUNCTIONS = r'log|ln|lg|sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|exp|det|max|min|gcd|lcm|deg|operatorname|pi|mathrm|vec|mathbf|bar|hat'
_GOES_ON = re.compile(r'\s*(?:[+\-*/^(×÷·⋅−]|\\times|\\cdot|\\div|\\d?t?frac|\\sqrt|\\left|\\binom|\\(?:' + _FUNCTIONS + r')\b)')
_FUNCTION_BEFORE = re.compile(r'\\(?:' + _FUNCTIONS + r')\b(?:_\{?\w+\}?|\^\{?\w+\}?)*\s*\(?\s*$')


def _value_after(segment):
    """(text, value, rest) for the longest arithmetic that opens `segment` (after spaces and `$`), or None: a number glued to a unit (4.8k)
    or trailing off into `...` (an approximation) is no value to check."""
    s = segment.lstrip(' $')
    for end in range(len(s), 0, -1):
        head = s[:end]
        if not head[-1].isdigit() and head[-1] not in ')}|]°' and not head.endswith('\\rfloor') and not head.endswith('\\rceil') \
                and not head.endswith('\\circ') and not head.endswith('!') and not head.endswith('%'):
            continue
        value = evaluate(head)
        if value is None:
            continue
        rest = s[end:]
        if rest[:1].isalpha() or rest[:1] and rest[0] in '_\\' and not rest.startswith(('\\ne', '\\le', '\\ge', '\\quad', '\\text', '\\implies', '\\Rightarrow',
                                                                            '\\approx', '\\equiv', '\\pmod', '\\to', '\\rightarrow', '\\\\')) \
                or _ROUGH.match(rest) or _GOES_ON.match(rest) \
                or re.match(r'\s*(?::(?!=)|(?:million|billion|thousand|trillion|percent)\b|x\s*10\s*\^)', rest, re.I) \
                or re.match(r' [A-Za-z](?![A-Za-z])', rest):
            # `112 = 4 R x`: a number times a lone letter is a term of an equation in unknowns, not a value (2026-11-09)
            # 4.8k; 20.1852...; `6^2 + (\frac{s-r}{2})^2` or `2(r^2 + s^2)`: algebra goes on; `3/10 : 7/10` a ratio; `4.34 million`;
            # `2.4 x 10^3` is scientific, not 2.4. Not past a `$`: `47 + 316 + 4742 = 4105$ million` counts in millions (dm-base)
            return None
        return head.strip(), value, rest
    return None


def _power(m):
    """The power of ten of a _SCIENTIFIC match: 10^{-7} or 10000."""
    return int(m.group(2)) if m.group(2) is not None else len(m.group(3))


def _digits(text):
    """Significant digits written in all the numbers of a text: how precise a rewrite is."""
    return sum(len(n.replace(',', '').replace('.', '').strip('0')) for n in re.findall(r'\d[\d,]*(?:\.\d+)?', text))


def _unit(text):
    """The unit of a written value's last place: 0.01 for 6.83, 10^-11 for 2.01852 \\times 10^{-6}; None for an expression."""
    m = _SCIENTIFIC.fullmatch(text.strip())
    if m:
        return Fraction(10) ** (-len(m.group(1) or '') + _power(m))
    m = _PLAIN.fullmatch(text.strip())
    if not m:
        return None
    digits = text.strip().replace(',', '').lstrip('-')
    if m.group(1):   # a float's long zero run is no precision: 5.322000000000000 and 5.3220000000000005 are written to 5.322
        run = re.match(r'(\d*?)0{4,}\d{0,3}$', m.group(1))
        return Fraction(1, 10 ** len(run.group(1) if run else m.group(1)))
    zeros = len(digits) - len(digits.rstrip('0'))
    return Fraction(10 ** zeros) if zeros and digits.strip('0') else Fraction(1)   # 2190 is rounded to tens, 707889 is exact


_WHOLE_WORK = re.compile(r'(?:[\d\s+\-−*×()]|\\times|\\cdot)+')


def _tolerance(text, op):
    """How far a written value may be from the exact one and still be right: half a unit in its last significant place (2190 for 2189.66,
    6.84 for 6.8356), a whole unit when it says \\approx or ≈, nothing for an integer with no zeros at its end or for an expression."""
    unit = _unit(text)
    if unit is None:
        return Fraction(0)
    if op in ('\\approx', '≈'):
        return unit
    return unit / 2 if unit != 1 or _SCIENTIFIC.fullmatch(text.strip()) else Fraction(0)


def _estimate(before, after):
    """`after` is `before` on rounded numbers: as many numbers, each within 10 % of its own and written with no more significant digits,
    one of them changed. 4.2 \\times 53 = 4 \\times 53 is an estimate, 189 + 17 for 9 \\cdot 21 + 7 is not."""
    a = re.findall(r'\d[\d,]*(?:\.\d+)?', before)
    b = re.findall(r'\d[\d,]*(?:\.\d+)?', after)
    if len(a) != len(b) or not a or a == b:
        return False

    def digits(x):
        return len(x.replace(',', '').replace('.', '').strip('0'))
    def rounded(fx, fy, n):
        """fy is fx rounded or cut to n significant digits (3339 -> 3000, 2548 -> 2500 or 3000, 0.0457 -> 0.05)."""
        if fx == 0 or n < 1:
            return False
        import math
        place = Fraction(10) ** (math.floor(math.log10(abs(float(fx)))) - n + 1)
        down = (abs(fx) // place) * place
        return abs(fy) in (down, down + place) and (fx < 0) == (fy < 0)
    for x, y in zip(a, b):
        fx, fy = Fraction(x.replace(',', '')), Fraction(y.replace(',', ''))
        if fx == 0 or digits(y) > digits(x):
            return False
        if abs(fx - fy) > abs(fx) / 10 and not rounded(fx, fy, digits(y)):
            return False
    return True


_POWER_OF_TEN = re.compile(r'10\s*\^\s*\{?\s*-?\d+\s*\}?|10{2,}')
_LONE = re.compile(r'\(?\s*-?\d[\d,]*(?:\.\d+)?\s*(?:/\s*-?\d[\d,]*(?:\.\d+)?)?\s*\)?')
_MUL_SPLIT = re.compile(r'\\times|\\cdot|×|\*|·')


def _constants(left, right):
    """A fraction in lowest terms set equal to another fraction: `then 3/4 = -2/3` states a contradiction, it computes nothing. A fraction
    reduced (-3312 / 96 = -104 / 3) or turned into a decimal (115/18 \\approx 6.83) is a calculation, and checked."""
    m = re.fullmatch(r'\(?\s*(-?\d+)\s*/\s*(-?\d+)\s*\)?', left.strip())
    if not m or not re.fullmatch(r'\(?\s*-?\d+\s*/\s*-?\d+\s*\)?', right.strip()):
        return False
    from math import gcd
    return gcd(int(m.group(1)), int(m.group(2))) == 1


def _reads_otherwise(expr, wrote):
    """True when a looser reading of `expr` gives what was written: `3+1 \\times 1+1` meant as (3+1)(1+1), or `(-66^2)` meant as
    (-66)^2. Models write both; a correction by the strict reading would be wrong to them."""
    if re.search(r'[+\-]', expr) and _MUL_SPLIT.search(expr) and '(' not in expr:
        loose = evaluate('(' + _MUL_SPLIT.sub(')*(', expr) + ')')
        if loose is not None and loose == wrote:
            return True
    if re.search(r'\(\s*-\s*\d', expr):
        loose = evaluate(re.sub(r'\(\s*-\s*(\d+(?:\.\d+)?)\s*\^', r'((-\1)^', expr))
        if loose is not None and loose == wrote:
            return True
    return False


def _decimals_slack(text, rough):
    """An expression on the right with rounded decimals in it (`-56.44 + 152`, `1.13 - 1.14`): half a unit of each decimal's last place,
    a whole unit in an approximate chain. A plain number has its own tolerance; integers add none."""
    if _PLAIN.fullmatch(text.strip()) or _SCIENTIFIC.fullmatch(text.strip()):
        return Fraction(0)
    slack = Fraction(0)
    for places in re.findall(r'\d\.(\d+)', text):
        slack += Fraction(1, 10 ** len(places)) * (1 if rough else Fraction(1, 2))
    if rough:
        slack += sum(Fraction(1) for _ in re.findall(r'(?<![\d.])\d+(?![\d.])', text)) and Fraction(1)
    return slack


_MODULUS = re.compile(r'(?:\\pmod\s*\{?\s*|\\bmod\s*\{?\s*|\bmod(?:ulo)?\s*(?=\d)|\bmod(?:ulo)?\s+|\bdivided by\s+|\bremainder (?:when|of) .{0,60}?by\s+)\$?(\d[\d,]*)')


def moduli_in(text):
    """The moduli a text names (`\\pmod{13}`, `mod 27`, `divided by 1000`): arithmetic reduced by one of them, written with `=`, is right."""
    found = set()
    for m in _MODULUS.finditer(text):
        n = int(m.group(1).replace(',', ''))
        if n >= 2:
            found.add(n)
    return found


_CUSTOM_OP = re.compile(r'(?<![A-Za-z\\])[a-z]\s*(\*|\\ast|\\star|\\circ|\\diamond|\\triangle|\\oplus|\\otimes|\\odot|\\bullet|\\heartsuit|'
                        r'\\spadesuit|\\clubsuit|\\diamondsuit|⊕|⊗|∘|★|♦|#|@)\s*[a-z]\s*(?:=|:=|\bis defined)\s*[^=\n]*[a-z]')


_DEFINES = re.compile(r'defin|operation|operator|denote|\blet\b|\bwhere\b|\bfor all\b|\bfor any\b', re.I)


def custom_operators(text):
    """The operator symbols the text defines as a new operation (`a * b = a - b`, `x \\star y = 2x + y`): `5 * (3 * 1)` is no product there.
    A sign that is also the product (`*`, `\\ast`) is a new operation only where words define it (`Define a * b = a - b`, `the operation
    a*b = ...`) and its right side is no product of the two: `x*y = 3y` in a model's algebra once made every later product of the answer
    unread (#307: 753 refused `=` in the held-out traces, `13*20` among them)."""
    out = set()
    for m in _CUSTOM_OP.finditer(text):
        op = m.group(1)
        if op in ('*', '\\ast'):
            head = text[max(0, m.start() - 80):m.start()]
            a, b = re.match(r'([a-z])\s*\S+\s*([a-z])', text[m.start():m.end()]).groups()
            rest = re.split(r'=|:=|is defined(?: as| by)?', text[m.start():], 1)[-1]
            rhs = re.match(r'[\s$]*([^.,;:$\n]*)', rest).group(1).strip()   # the right side, to the end of its clause
            if not _DEFINES.search(head + text[m.start():m.end()]):
                continue
            if a == b:   # `x * x = 0` among identities that define the operation
                if not (re.match(r'-?\d+\b', rhs) and re.search(r'identit|for any|for all', head, re.I)):
                    continue
            else:
                product = re.fullmatch(rf'(?:{a}\s*\*?\s*{b}|{b}\s*\*?\s*{a})', rhs) is not None
                in_ab = re.fullmatch(rf'[\s{a}{b}\d+\-−*/^()·]+', rhs) is not None   # an expression in the two and numbers only
                if product or not in_ab or not (a in rhs and b in rhs):
                    continue
        out.add(op)
    return out


def _partial(base_text, text):
    """The link is a leading part of the base, the start of a running total: `17 + 78` of `17 + 78 + 85`, `4*49` of `4*49*328`."""
    a, b = re.sub(r'[\s()]', '', base_text), re.sub(r'[\s()]', '', text)
    # a binary operator in it: `-\frac{15}{2}` of `-\frac{15}{2} + \frac{15}{2}` is a wrong value, not a step of a total
    binary = re.search(r'[+\-*/×÷·−]|\\times|\\cdot|\\div', re.sub(r'\\d?t?frac', '', b.lstrip('-−')))
    return len(b) < len(a) and a.startswith(b) and bool(binary) and not a[len(b)].isdigit()


def _stated_later(line, at, right):
    """The right value is written as a number later on the line, past the end of this chain (`.`, `,`, `;`, `?`, `≠`): the model has
    it, a correction tells it nothing. Not the next link of the same chain: `9 \\cdot 21 + 7 = 189 + 17 = 196` still has a wrong rewrite."""
    tail = line[at:]
    sep = re.search(r'[.,;?≠]\s|\\neq?\b|\?|\bnot\b', tail)
    if sep is None:
        return False
    rest = tail[sep.start():]
    for m in re.finditer(r'(?<![\d.,])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])|(?<![\d.])-?\d+(?:\.(\d+))?(?!\d|\.\d)', rest):
        if re.search(r'(?:\\neq?|\\le|\\ge|\\leq|\\geq|[≠<>≤≥])\s*\$?\s*$', rest[:m.start()]):
            continue   # `= -5736 \neq 0`: the 0 is what it is compared with
        try:
            v = Fraction(m.group().replace(',', ''))
        except (ValueError, ZeroDivisionError):
            continue
        places = len(m.group().split('.')[1]) if '.' in m.group() else 0
        if abs(v - right) <= Fraction(1, 2 * 10 ** places) if places else v == right:
            return True
    return False


def _last_term(expr):
    """The value of the last term of a sum, or None: `3 + 2^{5}` -> 32."""
    terms = re.split(r'(?<=[\d)}\]])\s*[+\-−]\s*(?=[\d(\[{-])', expr.strip())
    return evaluate(terms[-1]) if len(terms) > 1 else None


def _stated_anywhere(tail, right, near=False):
    for m in re.finditer(r'(?<![\d.])-?\d[\d,]*(?:\.(\d+))?(?!\d|\.\d)', tail):
        try:
            v = Fraction(m.group().replace(',', '').rstrip('.'))
        except (ValueError, ZeroDivisionError):
            continue
        places = len(m.group(1) or '')
        if v == right or near and places and abs(v - right) <= Fraction(1, 10 ** places):
            return True
    return False


def _substep(prev, value):
    """A plain number that is the value of one term of the link before: `= 1001000 + 13*600 = 7800` is 13*600, worked out beside the
    sum (R1 writes `=` for the nearest piece)."""
    if prev is None:
        return False
    for term in re.split(r'\s[+\-−]\s|(?<=[\d)])\s*[+\-−]\s*(?=[\d(])', prev[0]):
        if _OPERATOR.search(term.lstrip('-−')):
            v = evaluate(term.strip())
            if v is not None and v == value:
                return True
    return False


def _arrow_sum(base, text, value):
    """`8+4+3 \\to 16`: an arrow that ends a sum of numbers into a number of its size is a computation to check."""
    b = base[0].strip(' $')
    if not re.fullmatch(r'-?\d+(?:\s*[+\-−]\s*\d+)+', b) or not _PLAIN.fullmatch(text.strip()) or not base[1] or not value:
        return False
    return Fraction(1, 2) <= value / base[1] <= 2


# a relation the model asks (`...$? No.`; not `Why did I think ... = 90360$?`, a wrong value asked about), marks false or wrong, or offers as an example (`like $...$`, `e.g.`, `or similar`):
# none of these states the relation as true, so a wrong side there is no slip to correct (developer, 2026-11-09, round 9's replay: 15 of
# the 40 corrections on right answers were relations the model had itself rejected)
_ASKED = re.compile(r'\?\s*\$?\s*(?:No|Not|Wrong|Incorrect)\b|\(\s*(?:False|false|wrong|not true)\s*\)|\bwhich is (?:false|wrong)\b'
                    r'|\bor similar\b|\blike\s+\$|\be\.g\.')
_CONDITION = re.compile(r'\b(?:false|not true|impossible|contradiction|need|needs|want|for this to|suppose|assume|if|whether|maybe|perhaps)\b',
                        re.I)
_COUNT_NOUN = re.compile(r'\s*\$?\s*(?:coins?|notes?|bills?|stamps?|pieces?|items?)\b', re.I)
_TERM = re.compile(r'\s*(\d+)\s*(?:\\times|\\cdot|\*|×)\s*(\d+)\s*')


def _counts_items(expr, wrote):
    """`3 \\times 10 + 5 \\times 1 = 8`: the count of the items (3 tens and 5 ones are 8 coins), not their value."""
    terms = re.split(r'\s*\+\s*', re.sub(r'[()]', ' ', expr).strip(' $'))
    pairs = [_TERM.fullmatch(t) or re.fullmatch(r'\s*(\d+)\s*', t) for t in terms]   # `1 \times 25 + 7 = 8`: seven ones (#306)
    if len(terms) < 2 or not all(pairs) or not any(p.re is _TERM for p in pairs):
        return False
    firsts = sum(int(p.group(1)) for p in pairs)
    seconds = sum(int(p.group(2)) if p.re is _TERM else 1 for p in pairs)
    return wrote in (firsts, seconds)


def _conditions(line, base_text):
    """`then $0 + 0 = 6$ and $0 + 0 = 0$`: the same left side twice with two values is two conditions tested, not two claims (#306)."""
    said = re.findall(re.escape(base_text.strip()) + r'\s*=\s*(-?[\d.]+)', line)
    return len(set(said)) > 1


# a decimal cut from a repeating one: thirds, sixths and ninths end in 33, 66 or 99 (`1.66`, `8.1666`), any other digit in a run of three
# (`0.7777`); `84.3488` and `0.843488` are written to their places (#306, held-out)
_REPEATING = re.compile(r'(?<![\d.])(\d+)\.(\d*?)(?=[369]{2}(?!\d)|(\d)\3\3)(\d)\4+(?![\d])')


def _nice(expr):
    """Each 2-place decimal near a fraction of denominator 3, 6, 7, 9, 11 or 12 replaced by it: `1.33`, `1.67` are 4/3 and 5/3, rounded."""
    def one(m):
        x = Fraction(m.group())
        for q in (3, 6, 7, 9, 11, 12):
            f = Fraction(round(x * q), q)
            if f != x and abs(f - x) <= Fraction(1, 200):
                return f'({f})'
        return m.group()
    return re.sub(r'(?<![\d.])\d+\.\d{2}(?![\d])', one, expr)


def _repeating_matches(expr, wrote, text, op):
    """`1563 - 8.1666 = 1554.8333`: 8.1666 is 8.1666..., cut; with it read as repeating the written value is right."""
    def exact(m):
        whole, head, digit = m.group(1), m.group(2), m.group(4)
        return f'({Fraction(whole + "." + head if head else whole) + Fraction(int(digit), 9 * 10 ** len(head))})'
    nice = _nice(expr)
    # a cut repeating decimal is a short one: `42.66`, `1.33`, `8.1666`; `0.843488` or `1.36664166` are written to their places
    cut_text = any(len(m.group(0).split('.', 1)[1]) <= 4 for m in _REPEATING.finditer(text))
    if nice != expr or cut_text:   # `(0-1)/(1.33-1.67) = -1/-0.33 = 3`, `128/3 - 16 = 42.66 - 16` (#306, held-out)
        a, b = evaluate(nice), evaluate(_REPEATING.sub(exact, text))
        # within the places of a plain number written, or 0.5 % of an expression: an absolute half unit per decimal was 50 % of 0.01
        tol = max(_tolerance(text, op), _unit(text) or 0) if _PLAIN.fullmatch(text.strip()) else abs(a or 0) / 200
        if a is not None and b is not None and abs(a - b) <= tol:
            return True
    if not _REPEATING.search(expr):
        return False

    def exact(m):
        whole, head, digit = m.group(1), m.group(2), m.group(4)
        return f'({Fraction(whole + "." + head if head else whole) + Fraction(int(digit), 9 * 10 ** len(head))})'
    value = evaluate(_REPEATING.sub(exact, expr))
    return value is not None and abs(value - wrote) <= max(_tolerance(text, op), _unit(text) or 0) + _decimals_slack(text, False)


def wrong_relations(line, moduli=(), custom_ops=()):
    """[(expression, what was written, the right value)] for each relation on the line whose sides differ: `=` chains (each part against the
    first: `= 189 + 17 = 196` is a wrong rewrite of 196), \\approx and ≈ (to the places written), arrows (`7+3+5 \\to 15`), and congruences
    (`2016 \\equiv 16 \\pmod{1000}`). Not wrong: a long division's quotient digit (`13 \\div 2 = 6` remainder 1), rounding or cutting to the
    places written, a unit after the number (4.8k), an approximation trailing off (20.1852...), a column with a borrow or carry, and two
    statements on one line (`$x = 3$, $y = 5$`). A relation is checked only when one side computes something."""
    if re.search(r'borrow|carry', line, re.I):
        return []
    if _ASKED.search(line):
        return []   # `= 5/12 \times 12/25$? No.`, `$3+4= 7 = 5$ (False)`, `like $1000 \equiv 1 \pmod 7$`: asked, rejected or an example, not stated
    division = re.search(r'(\d+)\s*(?:÷|/|\\div)\s*(\d+)\s*=\s*\d+\s*(?:\*|×|\\times|\\cdot)\s*(\d+)', line)
    if division and division.group(2) == division.group(3):
        return []   # `115542 ÷ 21609 = 5*21609 = 108045, remainder 7497`: a long division's notes, the product beside the quotient (#306)
    if '$' not in line and '\\' not in line and re.search(r'\d\.[A-Za-z]', line):
        return []   # `A=27:729.7*2*9=126.27-1=26.No.`: periods glued to the numbers, `729.` and `7` read as 729.7 (#306, R1)
    if any(op in line for op in custom_ops):
        return []   # `5 * (3 * 1) = 5 - (3 - 1)`: an operation the problem defined, written with `*` (#306, held-out)
    if re.search(r'\b(?:units|ones|tens|hundreds|thousands)\s*(?:digit|place|column)?\s*:|10\s*\^\s*\{?\s*-?\d+\s*\}?\s*:', line, re.I):
        return []   # `Tens: $3 + 8 + 4 + 6 + 2 + 3 + 2 = 30$`: one column of a sum, its carry added in the head (lb1-base livebench-88407ca2)
    moduli = set(moduli)
    if re.search(r'\bparity\b|\b(?:even|odd)\b', line, re.I):
        moduli.add(2)   # `The parity is $1 + 1 = 0$ (even)` (#306)
    if re.search(r'\^\s*\{?\s*\\circ|°|\bdegrees?\b', line):
        moduli.add(360)   # `-72 - 168 = -240 = 120^\circ`: one angle (#306)
    wrong = []
    m = re.search(r'\\equiv\s*(.+?)\\pmod\s*\{?\s*([\d,]+)\s*\}?', line)
    if m and (_CONDITION.search(line) or '?' in line[m.end():]):
        return []   # `we need $8 \equiv -1 \pmod{27}$, which is false`: a condition the model tested and rejected (#306)
    if m:
        left = expression_before(line[:m.start()]) or ((lambda v: (line[:m.start()].strip(' $'), v) if v is not None else None)(
            evaluate(line[:m.start()].strip(' $').rsplit('$', 1)[-1])))
        right, mod = evaluate(m.group(1).strip(' $')), evaluate(m.group(2))
        if left and right is not None and mod and left[1].denominator == right.denominator == 1 and (left[1] - right) % mod:
            wrong.append((left[0], f'{m.group(1).strip()} \\pmod{{{m.group(2)}}}', f'{int(left[1]) % int(mod)} \\pmod{{{m.group(2)}}}'))
        return wrong
    if re.search(r'\\equiv|≡', line):
        return wrong   # `11 \cdot 6^{-1} = 11 \cdot 1 = 11 \equiv 1`: a line that works mod m with no \pmod to name it (#306)
    marks = list(_RELATION.finditer(line))
    if not marks:
        return wrong
    base = expression_before(line[:marks[0].start()])
    origin = base   # the chain's first expression, kept when an approximate chain moves its base on
    prev = None   # the last link that read: a chain that rounds as it goes is checked against it too
    running = False   # this chain is a running total: its links are steps, not rewrites of the base
    # a line with no LaTeX: R1 and QwQ write `=` for the nearest piece (`13*77600=1001000 + 13*600=7800 = 1008800`, `(1 -1.99 +1)=0.01/(...)`),
    # so past its first link a plain chain says nothing checkable, and a value the line writes anyway needs no correction (#306)
    plain = '$' not in line and '\\' not in line
    links = 0   # links read since the base was set
    clean = True   # every link compared so far on this chain was right: a plain chain that stays right can still be checked
    rough = False   # the chain has said \approx: its later `=` parts are approximate too
    for n, mark in enumerate(marks):
        end = marks[n + 1].start() if n + 1 < len(marks) else len(line)
        rough = rough or mark.group(1) in ('\\approx', '≈')
        got = _value_after(line[mark.end():end])
        if got is None:
            base, rough, running, links, clean = expression_before(line[:end]), False, False, 0, True
            continue
        text, value, rest = got
        compared = base is not None   # a link that sets the base is no link of the chain yet
        first = _PLAIN.match(text.lstrip('-−'))
        if base is not None and (_partial(base[0], text) or prev is not None and _partial(prev[0], text)):
            running = True   # `17 + 78 + 85 = 17 + 78 = 95; 95 + 85 = 180`: a running total, each link a step of it, not the whole (#306, R1)
        # `= 1000 + 1)^4`: a bracket the model left open; `17*417=7, 089`: a thousands separator with a space after it
        broken = rest.startswith(')') and text.count('(') <= text.count(')') or bool(
            re.match(r',\s?\d{3}(?![\d^*=+\-/×.])', rest) and value.denominator == 1 and abs(value) < 1000 and not _OPERATOR.search(text))
        if base is None:
            base = (text, value)
            origin = origin or base
        elif running or broken or plain and links >= 1 and not clean:
            pass
        elif _LONE.fullmatch(base[0].strip()) and _OPERATOR.search(text.lstrip('-−')) and not _LONE.fullmatch(text.strip()) \
                and not _SCIENTIFIC.fullmatch(text.strip()):
            base = (text, value)   # `1/2 = \frac{(1/2)^2 - 3(1/2)}{...}`: f at x = 1/2, a substitution, not a claim that both are 1/2
        elif _POWER_OF_TEN.fullmatch(text.strip()) or _constants(base[0], text) or _reads_otherwise(base[0], value):
            pass   # `6^9 \approx 10^7`; `then 3/4 = -2/3`; `3+1 \times 1+1 = 8` meant as (3+1)(1+1), `(-66^2)` as (-66)^2
        elif mark.group(1) not in ('=', '\\approx', '≈') and not _arrow_sum(base, text, value):
            # an arrow maps unless it ends a sum (`8+4+3 \to 16`): `$1/90 \to 91$` (m + n), `$55^2 \to 2$`, `$6-9 \to 16-9=7$` (a
            # borrow), `(-300) \rightarrow 2 negatives`. What it maps to starts the next comparison (#306)
            base = (text, value)
        elif _counts_items(base[0], value) or _COUNT_NOUN.match(rest) or _repeating_matches(base[0], value, text, mark.group(1)) \
                or _conditions(line, base[0]):
            pass   # `3 \times 10 + 5 \times 1 = 8` coins; `1563 - 8.1666 = 1554.8333` with 8.1666... (#306)
        elif _OPERATOR.search(text.lstrip('-−')) and first and abs(Fraction(first.group().replace(',', '')) - abs(base[1])) <= (
                Fraction(1, 10 ** len(first.group(1))) if first.group(1) else 0):
            base = (text, value)   # `5 \times 2 = 10 + 1 = 11`: the value, then a carry added to it, a new step and not a wrong rewrite
        elif _estimate(base[0], text) or (_OPERATOR.search(text.lstrip('-−')) and not _SCIENTIFIC.fullmatch(text.strip())
                                           and _digits(text) < _digits(base[0]) and base[1] and abs(value - base[1]) <= abs(base[1]) / 500):
            # `0.00431 \times 58463 = 0.004 \times 58500`, or `\frac{27.44 \times 10^9}{22.11 \times 10^6}` for the same quotient: the step
            # on rounded numbers, an estimate. Within 0.2 %: -104 / 3 for -3312 / 96 is 0.5 % off and wrong
            base = (text, value)
        elif _OPERATOR.search(base[0].lstrip('-−')) or _OPERATOR.search(text.lstrip('-−')):
            right = base[1]
            off = abs(right - value)
            # a long division's quotient digit: `13 \div 2 = 6` remainder 1, or `80 / 189 = 4` for 800 / 189 with the zero brought down
            quotient = value.denominator == 1 and re.search(r'/|\\div|÷|\\frac', base[0]) and right > 0 and any(
                value == (right * 10 ** j).numerator // (right * 10 ** j).denominator for j in range(4) if j == 0 or 0 <= value <= 9)
            unit = _unit(text)   # 438 + 2/3 = 438.666: cut off in its last place, not rounded
            cut = unit is not None and unit < 1 and (right >= 0) == (value >= 0) and 0 <= abs(right) - abs(value) < unit
            tolerance = _tolerance(text, '\\approx' if rough else mark.group(1)) + _decimals_slack(text, rough)
            if unit == 1 and right.denominator != 1:
                tolerance = max(tolerance, Fraction(1))   # 130.98 - 22.88 = 108: rounded or cut to an integer
            if not rough and mark.group(1) == '=' and right.denominator == 1 and _WHOLE_WORK.fullmatch(base[0].strip(' $')):
                # whole numbers added, taken away and multiplied give a whole number, and `=` says it exactly: `14+7+...+1=40` for 41 is
                # a slip, and so is `505 \times 5148 = 2,600,000` for 2,599,740, a rounding written as an equality (`\approx` keeps its
                # slack). From the meaning of `=`, not from a threshold fitted to a set (#307)
                tolerance = Fraction(0)
            # `33[2/(1.2)^2 +1/(1.6)^2] = 33[1.3889 +0.3906]`: rounded decimals inside a product, the rounding times what multiplies them
            tolerance += abs(value) * sum(Fraction(1, 10 ** len(d)) / abs(Fraction(w + '.' + d))
                                          for w, d in re.findall(r'(?<![\d.])(\d+)\.(\d{2,})', text) if Fraction(w + '.' + d)) \
                if _OPERATOR.search(text.lstrip('-−')) else 0
            if rough:
                # `(0.707)^3 \approx 0.3535` for (\sqrt{2}/2)^3, `0.381 \times (-1.618) \approx -0.618`: the inputs are rounded, each by up
                # to a unit of its last place, and the error grows by their ratio to the value (#306, R1)
                tolerance += abs(right) * sum(Fraction(1, 10 ** len(d)) / abs(Fraction(w + '.' + d)) * (int(e) if e else 1)
                                              for w, d, e in re.findall(r'(?<![\d.])(\d+)\.(\d{2,})\)?(?:\^\{?(\d)\}?)?', base[0])
                                              if Fraction(w + '.' + d))
            if rough and _OPERATOR.search(text.lstrip('-−')) and not _SCIENTIFIC.fullmatch(text.strip()):
                tolerance = max(tolerance, abs(right) / 10)   # `(39/2)^2 / 8 \approx 19^2 / 8`: an estimate on rounded numbers (#306)
            modular = right.denominator == value.denominator == 1 and any((right - value) % m == 0 for m in moduli)
            # one column of a written sum or difference: `0-4=6` borrows, `7+5=2` carries; a digit that is the value mod 10 (#306)
            column = right.denominator == 1 and value.denominator == 1 and 0 <= value <= 9 and (right - value) % 10 == 0 \
                and re.fullmatch(r'\s*\d{1,2}(?:\s*[-+−]\s*\d{1,2}){1,2}\s*', base[0]) is not None
            # in work mod m a fraction is a product by an inverse: `$1/2 = 3$` is 2^{-1} = 3 mod 5 (stdmath-d6e306c4, found by sim_replay)
            inverse = bool(moduli) and right.denominator != 1 and simple_quotient(base[0]) and re.fullmatch(r'-?\d+', text.strip()) is not None
            # `2*3=6 mod5=1`: the link reduces the value mod m itself
            reduced = re.search(r'(?:\\bmod|\\pmod\s*\{?|\bmod)\s*(\d+)', text)
            reduces = bool(reduced) and right.denominator == value.denominator == 1 and int(reduced.group(1)) > 0 \
                and (right - value) % int(reduced.group(1)) == 0
            modular = modular or column or inverse or reduces
            # a chain that rounds as it goes: `= 91.26 + 9.56 - 101 = 100.82 - 101`, each link within its places of the last (#306)
            stepped = prev is not None and _decimals_slack(text, rough) and abs(prev[1] - value) <= tolerance
            # `360^{\circ} \approx 6.28`: degrees turned into radians (#306)
            radians = re.search(r'\^\s*\{?\s*\\circ|°', base[0]) is not None and right and abs(value - right * Fraction(355, 113) / 180) <= abs(right) / 1000 + (_unit(text) or 0)
            # `244000 - 245000 \approx 0`: nothing at the scale of the parts (#306)
            scale = max((Fraction(x.replace(',', '')) for x in _NUMBER.findall(base[0])), default=Fraction(0))
            nought = rough and value == 0 and abs(right) <= scale / 100
            # `14 \times 1.732 / 3 \approx 24.2 / 3 \approx 8.08`: an approximate chain goes on from its last approximation, but a
            # value nearer the first is no slip (#306)
            near_first = rough and origin is not None and abs(origin[1] - value) <= tolerance
            # the model wrote the right value later on the line: `8 + 4 = 4*4 = 16. 12≠16`, `3*2*4*7=3*2=6, 6*4=24, 24*7=168` (#306, R1)
            stated = _stated_later(line, mark.end() + len(text), right) or (
                _substep(prev or base, value) and _stated_anywhere(line[mark.end() + len(text):], right))
            # `(50 +25 +20)/3 ≈31.666%`: a percent of the same number
            pc = re.fullmatch(r'(-?\d+(?:\.(\d+))?)\s*\\?%', text.strip())
            percent = rest.lstrip().startswith(('%', '\\%')) and abs(value * 100 - right) <= tolerance * 100 or bool(pc) and abs(
                Fraction(pc.group(1)) - right) <= Fraction(1, 10 ** len(pc.group(2) or ''))
            if plain and _stated_anywhere(line[mark.end():], right, near=True):   # in the link too: `= 15,376,000,000 +496,004`
                stated = True
            # `- (-59)*167 = 59*167`: a minus opening the line, read as a list's bullet, is the sign the value differs by
            head = line[:line.find(base[0])] if base[0] in line else ''
            bullet_sign = value == -right and re.fullmatch(r'\s*[-−]\s*', head) is not None
            stated = stated or bullet_sign
            # `A=24:576.5*7*6=210.`: R1 glued a sentence's period to 576, and `5*7*6` read as `.5*7*6`
            glued = plain and re.search(r'(?<![\d.])\d{3,}\.\d(?!\d)', base[0]) is not None and value.denominator == 1 and value                 and abs(right / value) > 10
            stated = stated or glued
            if plain:
                # `18 girls (12+6)=32`: the bracket annotates a word and `=` takes the sum before it; `3 + 2^{5}=32, +1^4=1`: `=` takes
                # the nearest term (#306, R1)
                at = line.find(base[0])
                annotated = base[0].lstrip().startswith('(') and at > 0 and re.search(r'[A-Za-z]\s*$', line[:at]) is not None
                nearest = _last_term(base[0]) == value
                stated = stated or annotated or nearest
            clean = clean and not off > tolerance
            if off > tolerance and not quotient and not cut and not modular and not stepped and not radians and not nought and not near_first \
                    and not stated and not percent:   # `12+3=15=2` mod 13
                m, sci = _PLAIN.fullmatch(text), _SCIENTIFIC.fullmatch(text)
                places = len(m.group(1) or '') if m else 0
                said = spelled(right)
                if sci:   # in the written form: 20.1857 \times 10^{-7}, not 37/18329827
                    k = _power(sci)
                    tail = f'10^{{{k}}}' if sci.group(2) is not None else '1' + '0' * k
                    places = len(sci.group(1) or '')
                    q = right / Fraction(10) ** k
                    # `882 \times 10^6` for 88.2 \times 10^6: to the places written it read `88 \times 10^6`, not the value either. A value that
                    # ends within 8 places is given whole; one that does not is given to the places written (#306)
                    places = next((p for p in range(places, 9) if (q * 10 ** p).denominator == 1), places)
                    said = f'{rounded(right / Fraction(10) ** k, places)} \\times {tail}'
                elif places and (len(said.split('.')[-1]) > places if '.' in said else '/' in said):
                    said = rounded(right, places)
                wrong.append((base[0], text, said))
        # the last link that was right: one that follows a wrong link is not right by agreeing with it (`126 \times 70 / 19.6` after the
        # wrong `126 \times 700 / 196`)
        prev = (text, value) if not wrong or wrong[-1][1] != text else None
        links += compared
        if rough:
            base = (text, value)   # an approximate chain goes on from its last approximation: `\approx 6.89 - 4 = 2.89`
        if rest.strip(' $.,;:') and not _RELATION.match(rest.strip()):
            # a new statement: `$x = 3$, $y = 5$` is two, not a chain. Its own expression is checked: `12 * 13 = 156 and 7 * 8 = 54` left the
            # second unread when the base was dropped (found by tools/igc/coverage.py's first test)
            base, rough, running, links, clean = _statement_start(line[:end]), False, False, 0, True
    return wrong


def _statement_start(text):
    """The expression a new statement on a line checks, or None. Not one standing after a variable (`\\frac{8}{11} S \\frac{11}{18} = ...`
    is a product with S, r1-ctrl 06c279c7), nor one in a line that works mod something (`5/4 \\equiv 5 \\cdot 4^{-1} = 20`, d6e306c4)."""
    if re.search(r'\\equiv|≡', text):
        return None
    found = expression_before(text)
    if found is None:
        return None
    before = text[:text.rstrip().rfind(found[0])].rstrip().rstrip('$').rstrip()
    word = re.search(r'(\\?[A-Za-z]+)$', before)
    if word and (len(word.group(1)) == 1 or word.group(1).startswith('\\')):
        return None
    return found


def simple_quotient(expr):
    """An integer over an integer and nothing else: `21 \\div 19`, `21/19`, `\\frac{-2142}{-7037}`."""
    bare = re.sub(r'\s|\\left|\\right|[()]', '', expr)
    return bool(re.fullmatch(r'-?\d+(?:/|\\div|÷)-?\d+|\\d?t?frac\{-?\d+\}\{-?\d+\}', bare))
