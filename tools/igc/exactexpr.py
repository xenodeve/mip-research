"""Exact values a model writes past what tools.igc.mathexpr reads: square roots kept as roots, trig at exact angles, logs that come out,
sums and products over a range, `a^b mod m`, a function the text defined, a product written with no sign, `\\binom` alone.

The model works these out by hand: `\\sqrt{1674} = 3\\sqrt{186}`, `\\sin 30^\\circ = 1/2`, `\\log_2 8 = 3`, `\\sum_{k=1}^{10} k^2 = 385`,
`4(117) = 468`, `\\binom{10}{4} = 210`, `f(3) = ...`. Over the 48 control traces of a100_262k the bridge read none of them (2026-11-06).

A value is exact or there is none. A value is an element of Q(sqrt 2, sqrt 3, ...): a rational combination of square roots of squarefree
integers, which spell one way only, so two values are equal exactly when their spellings are. Refused: a symbol nothing binds, a word,
pi outside an angle, a root of a root, a cube root that does not come out, a decimal whose answer is irrational (the model wants a decimal
there, not `\\frac{\\sqrt{3109666}}{100}`), a value past mathexpr.LIMIT, and anything slower than its worth.

evaluate(text, functions=None) -> Value | None      the value of the whole text
Value.rational() -> Fraction | None                 the value when it is rational
spell(value, latex=True) -> str                     `3\\sqrt{186}`, `\\frac{\\sqrt{6} - \\sqrt{2}}{4}`; or `3*sqrt(186)` in plain text
definitions(text) -> {name: (param, tree)}          the one-variable functions the text defined (`f(x) = x^2 + 1`), the latest of each
"""
import functools
import math
import re
from fractions import Fraction

from tools.igc import mathexpr

LIMIT = mathexpr.LIMIT
MOST_TERMS = 16          # radicals in one value; a product of k sums has up to 2^k, and past this it is no hand step
MOST_RADICAND = 10 ** 12  # trial division to 10^6 factors it at once
MOST_RANGE = 20000       # terms of one \sum or \prod
MOST_POWER = 64


class NotExact(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------------------------
# Values: {squarefree radicand: rational coefficient}; radicand 1 is the rational part
# ---------------------------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=8192)
def _square_split(n):
    """n = s^2 * r with r squarefree: (s, r, primes of r). n <= MOST_RADICAND."""
    if n > MOST_RADICAND:
        raise NotExact('a radicand too large to factor at once')
    s, r, primes, p = 1, 1, [], 2
    while p * p <= n:
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        if e:
            s *= p ** (e // 2)
            if e % 2:
                r *= p
                primes.append(p)
        p += 1 if p == 2 else 2
    if n > 1:
        r *= n
        primes.append(n)
    return s, r, tuple(primes)


@functools.lru_cache(maxsize=8192)
def _factor(n):
    """{prime: exponent} of 1 <= n <= MOST_RADICAND."""
    if n > MOST_RADICAND:
        raise NotExact('too large to factor at once')
    out, p = {}, 2
    while p * p <= n:
        while n % p == 0:
            out[p] = out.get(p, 0) + 1
            n //= p
        p += 1 if p == 2 else 2
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


class Value:
    __slots__ = ('t',)

    def __init__(self, terms):
        t = {r: Fraction(c) for r, c in terms.items() if c}
        if len(t) > MOST_TERMS:
            raise NotExact('too many radicals')
        for c in t.values():
            if abs(c.numerator) > LIMIT or c.denominator > LIMIT:
                raise NotExact('too large to be a step')
        self.t = t

    @staticmethod
    def of(x):
        return Value({1: Fraction(x)})

    def rational(self):
        if not self.t:
            return Fraction(0)
        if set(self.t) == {1}:
            return self.t[1]
        return None

    def __eq__(self, other):
        return isinstance(other, Value) and self.t == other.t

    def __hash__(self):
        return hash(frozenset(self.t.items()))

    def __repr__(self):
        return f'Value({spell(self, latex=False)})'

    def __add__(self, o):
        t = dict(self.t)
        for r, c in o.t.items():
            t[r] = t.get(r, 0) + c
        return Value(t)

    def __neg__(self):
        return Value({r: -c for r, c in self.t.items()})

    def __sub__(self, o):
        return self + (-o)

    def __mul__(self, o):
        t = {}
        for r1, c1 in self.t.items():
            for r2, c2 in o.t.items():
                g = math.gcd(r1, r2)
                r = (r1 // g) * (r2 // g)   # both squarefree: sqrt(r1) sqrt(r2) = g sqrt(r1 r2 / g^2)
                t[r] = t.get(r, 0) + c1 * c2 * g
        return Value(t)

    def flip(self, p):
        """The conjugate that negates sqrt(p): self * self.flip(p) has no sqrt(p) left."""
        return Value({r: (-c if r % p == 0 else c) for r, c in self.t.items()})

    def inverse(self):
        q = self.rational()
        if q is not None:
            if q == 0:
                raise NotExact('division by zero')
            return Value.of(1 / q)
        num, den = Value.of(1), self
        for _ in range(8):
            q = den.rational()
            if q is not None:
                if q == 0:
                    raise NotExact('division by zero')
                return Value({r: c / q for r, c in num.t.items()})
            r = max(x for x in den.t if x != 1)
            p = _square_split(r)[2][0]
            conj = den.flip(p)
            num, den = num * conj, den * conj
        raise NotExact('a denominator with too many radicals')

    def __truediv__(self, o):
        return self * o.inverse()

    def interval(self, digits=30):
        """(lo, hi) Fractions around the value."""
        scale = 10 ** digits
        lo = hi = Fraction(0)
        for r, c in self.t.items():
            if r == 1:
                lo, hi = lo + c, hi + c
                continue
            a = Fraction(math.isqrt(r * scale * scale), scale)
            b = a + Fraction(1, scale)
            lo, hi = (lo + c * a, hi + c * b) if c > 0 else (lo + c * b, hi + c * a)
        return lo, hi

    def sign(self):
        if not self.t:
            return 0
        q = self.rational()
        if q is not None:
            return (q > 0) - (q < 0)
        lo, hi = self.interval()
        if lo > 0:
            return 1
        if hi < 0:
            return -1
        raise NotExact('a sign the digits do not settle')

    def floor(self):
        q = self.rational()
        if q is not None:
            return Value.of(q.numerator // q.denominator)
        lo, hi = self.interval()
        a, b = lo.numerator // lo.denominator, hi.numerator // hi.denominator
        if a != b:
            raise NotExact('a floor the digits do not settle')
        return Value.of(a)


ZERO, ONE = Value.of(0), Value.of(1)


def _sqrt(v):
    q = v.rational()
    if q is None:
        raise NotExact('a root of a root')
    if q < 0:
        raise NotExact('the root of a negative')
    n = q.numerator * q.denominator   # sqrt(a/b) = sqrt(ab)/b
    if n == 0:
        return ZERO
    s, r, _ = _square_split(n)
    return Value({r: Fraction(s, q.denominator)})


def _root(v, k):
    if k == 2:
        return _sqrt(v)
    q = v.rational()
    if q is None or k < 1 or k > 64:
        raise NotExact('a root of a root')
    if q < 0 and k % 2 == 0:
        raise NotExact('an even root of a negative')
    sign = -1 if q < 0 else 1
    out = []
    for n in (abs(q.numerator), q.denominator):
        x = round(n ** (1.0 / k)) if n < 10 ** 300 else None
        hit = next((y for y in (x - 1, x, x + 1) if y is not None and y >= 0 and y ** k == n), None) if x is not None else None
        if hit is None:
            raise NotExact('a root that does not come out')
        out.append(hit)
    return Value.of(sign * Fraction(out[0], out[1]))


def _power(base, e):
    q = e.rational()
    if q is None:
        raise NotExact('an irrational power')
    if q.denominator != 1:
        b = base.rational()
        if b is None:
            raise NotExact('a fractional power of a root')
        return _power(_root(base, q.denominator), Value.of(q.numerator))
    n = int(q)
    if abs(n) > MOST_POWER:
        raise NotExact('a power too large to be a step')
    if base == ZERO and n < 0:
        raise NotExact('0 to a negative power')
    b = base.rational()
    if b is not None and abs(b.numerator) > 1 and abs(n) * max(len(str(abs(b.numerator))), len(str(b.denominator))) > 60:
        raise NotExact('too large')
    out = ONE
    for _ in range(abs(n)):
        out = out * base
    return out.inverse() if n < 0 else out


def _integer(v, what='an integer'):
    q = v.rational()
    if q is None or q.denominator != 1:
        raise NotExact('not ' + what)
    return int(q)


# exact cosines: cos(k * 15 deg) for k = 0..6 and cos(36), cos(72); the rest by symmetry
_S2, _S3, _S5, _S6 = Value({2: 1}), Value({3: 1}), Value({5: 1}), Value({6: 1})
_COS15 = [ONE, Value({6: Fraction(1, 4), 2: Fraction(1, 4)}), Value({3: Fraction(1, 2)}), Value({2: Fraction(1, 2)}), Value.of(Fraction(1, 2)),
          Value({6: Fraction(1, 4), 2: Fraction(-1, 4)}), ZERO]
_COS36 = {36: Value({1: Fraction(1, 4), 5: Fraction(1, 4)}), 72: Value({1: Fraction(-1, 4), 5: Fraction(1, 4)})}


def _cos_degrees(d):
    """cos of d degrees, d a Fraction; exact only at multiples of 15 and of 36."""
    if d.denominator != 1:
        raise NotExact('an angle with no exact cosine')
    d = int(d) % 360
    sign = 1
    if d > 180:
        d = 360 - d
    if d > 90:
        d, sign = 180 - d, -1
    if d % 15 == 0:
        v = _COS15[d // 15]
    elif d in _COS36:
        v = _COS36[d]
    elif d in (18, 54):   # cos 18 = sqrt((5 + sqrt 5)/8): a root of a root
        raise NotExact('a nested root')
    else:
        raise NotExact('an angle with no exact cosine')
    return v if sign == 1 else -v


def _trig(name, degrees):
    if name == 'cos':
        return _cos_degrees(degrees)
    if name == 'sin':
        return _cos_degrees(90 - degrees)
    s, c = _cos_degrees(90 - degrees), _cos_degrees(degrees)
    if name == 'tan':
        return s / c
    if name == 'cot':
        return c / s
    if name == 'sec':
        return ONE / c
    if name == 'csc':
        return ONE / s
    raise NotExact(name)


def _log(base, x):
    """log_base x when it is rational: base and x powers of one rational."""
    b, a = base.rational(), x.rational()
    if b is None or a is None or b <= 0 or a <= 0 or b == 1:
        raise NotExact('a log with no exact value')
    if a == 1:
        return ZERO
    vb, va = {}, {}
    for vec, q in ((vb, b), (va, a)):
        for p, e in _factor(q.numerator).items():
            vec[p] = vec.get(p, 0) + e
        for p, e in _factor(q.denominator).items():
            vec[p] = vec.get(p, 0) - e
    if set(vb) != set(va):
        raise NotExact('a log that is not rational')
    ratios = {Fraction(va[p], vb[p]) for p in vb}
    if len(ratios) != 1:
        raise NotExact('a log that is not rational')
    return Value.of(ratios.pop())


# ---------------------------------------------------------------------------------------------------------------------------------
# Reading: tokens, then a tree, then its value
# ---------------------------------------------------------------------------------------------------------------------------------

_TRIG = ('sin', 'cos', 'tan', 'cot', 'sec', 'csc')
_COMMANDS = {r'\times': ('op', '*'), r'\cdot': ('op', '*'), r'\ast': ('op', '*'), r'\div': ('op', '/'), r'\frac': ('frac',),
             r'\dfrac': ('frac',), r'\tfrac': ('frac',), r'\sqrt': ('sqrt',), r'\binom': ('binom',), r'\dbinom': ('binom',),
             r'\tbinom': ('binom',), r'\pi': ('pi',), r'\bmod': ('op', 'mod'), r'\mod': ('op', 'mod'), r'\pmod': ('pmod',),
             r'\lfloor': ('(', 'floor'), r'\rfloor': (')', 'floor'), r'\lceil': ('(', 'ceil'), r'\rceil': (')', 'ceil'),
             r'\{': ('(', '{'), r'\}': (')', '{'), r'\lbrace': ('(', '{'), r'\rbrace': (')', '{'), r'\%': ('op', '%'),
             r'\sum': ('big', 'sum'), r'\prod': ('big', 'prod'), r'\log': ('log', None), r'\ln': ('log', 'e'), r'\lg': ('log', 'lg'),
             r'\lvert': ('bar',), r'\rvert': ('bar',), r'\vert': ('bar',), r'\mid': ('bar',), r'\overline': ('overline',)}
_COMMANDS.update({'\\' + t: ('trig', t) for t in _TRIG})
_WORD = {'sqrt': ('sqrt',), 'pi': ('pi',), 'mod': ('op', 'mod'), 'log': ('log', None), 'ln': ('log', 'e')}
_WORD.update({t: ('trig', t) for t in _TRIG})
_SINGLE = {'*': ('op', '*'), '×': ('op', '*'), '·': ('op', '*'), '⋅': ('op', '*'), '∙': ('op', '*'), '/': ('op', '/'), '÷': ('op', '/'),
           '+': ('op', '+'), '-': ('op', '-'), '−': ('op', '-'), '–': ('op', '-'), '^': ('^',), '_': ('_',), '(': ('(', '('),
           ')': (')', '('), '[': ('(', '['), ']': (')', '['), '{': ('(', 'g'), '}': (')', 'g'), '!': ('!',), '%': ('op', '%'),
           '|': ('bar',), '°': ('deg',), ',': (',',), '=': ('=',), '√': ('sqrt',), 'π': ('pi',), '⌊': ('(', 'floor'), '⌋': (')', 'floor'),
           '⌈': ('(', 'ceil'), '⌉': (')', 'ceil')}
_DEGREE = re.compile(r'\^\s*\{\s*\\circ\s*\}|\^\s*\\circ\b|\\degree\b')


def _tokens(text):
    s = re.sub(r'(?<=\d)\\,(?=\d{3}(?!\d))', '', text)
    s = mathexpr._SIZES.sub('', mathexpr._SPACE.sub(' ', s))
    s = re.sub(r'\\(?:displaystyle|textstyle)\b', ' ', s)
    s = _DEGREE.sub('°', s).replace('$', ' ').replace('**', '^')
    s = ''.join('^' + mathexpr._SUPERSCRIPT[c] if c in mathexpr._SUPERSCRIPT else c for c in s)
    out, i, gap = [], 0, False
    while i < len(s):
        c = s[i]
        if c.isspace():
            i, gap = i + 1, True
            continue
        spaced, gap = gap, False
        m = mathexpr._NUMBER.match(s, i)
        if m and (c.isdigit() or c == '.'):
            raw = m.group().replace(',', '')
            i = m.end()
            if s.startswith('.\\overline', i) and '.' not in raw:   # `0.\overline{49}`: the point belongs to the number
                raw += '.'
                i += 1
            out.append(('num', Fraction(raw.lower()) if 'e' in raw.lower() else Fraction(raw.rstrip('.')), '.' in raw or 'e' in raw.lower(),
                        raw, spaced))
            continue
        if c == '\\':
            m = re.match(r'\\(?:[A-Za-z]+|.)', s[i:])
            word = m.group()
            if word not in _COMMANDS:
                raise NotExact(word)
            out.append(_COMMANDS[word] + (spaced,) if len(_COMMANDS[word]) < 3 else _COMMANDS[word])
            i += len(word)
            continue
        if c.isalpha() and c.isascii():
            m = re.match(r'[A-Za-z]+', s[i:])
            word = m.group()
            if word in _WORD:
                out.append(_WORD[word] + (spaced,))
            elif len(word) == 1:
                out.append(('var', word, spaced))
            else:
                raise NotExact('a word: ' + word)
            i += len(word)
            continue
        if c in _SINGLE:
            out.append(_SINGLE[c] + (spaced,))
            i += 1
            continue
        raise NotExact(c)
    return out


class _Parser:
    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else ('end', None, False)

    def take(self, kind=None, value=None):
        tok = self.peek()
        if tok[0] == 'end' or (kind is not None and tok[0] != kind) or (value is not None and tok[1] != value):
            raise NotExact(f'expected {kind} {value}')
        self.i += 1
        return tok

    def at(self, kind, value=None):
        tok = self.peek()
        return tok[0] == kind and (value is None or tok[1] == value)

    def expr(self):
        node = self.term()
        while self.at('op', '+') or self.at('op', '-'):
            op = self.take()[1]
            node = ('+' if op == '+' else '-', node, self.term())
        if self.at('pmod'):   # `2016 + 5 \pmod{1000}`: the whole of what comes before
            self.take()
            node = ('mod', node, self.group())
        return node

    def _starts_factor(self):
        """The next token can begin a factor glued on by a product with no sign."""
        k = self.peek()[0]
        return k in ('num', 'var', 'pi', 'sqrt', 'frac', 'binom', 'trig', 'log', 'big', '(', 'bar')

    def term(self):
        node = self.unary()
        divided = False
        while True:
            tok = self.peek()
            if tok[0] == 'op' and tok[1] in ('*', '/', 'mod'):
                self.take()
                divided = divided or tok[1] == '/'
                node = (tok[1], node, self.unary())
                continue
            if self._starts_factor() and self.implicit_ok(node):
                if divided:
                    raise NotExact('a/bc: a product after a division is read two ways')
                node = ('*', node, self.unary())
                continue
            return node

    def implicit_ok(self, left):
        """A product written with no sign: `4(117)`, `(2)(3)`, `2\\sqrt{3}`, `\\binom{4}{3}\\binom{6}{1}`, `2k`, `3\\pi`. Not `3 5`, not
        `2 (2)`, not the mixed number `2\\frac{1}{3}`."""
        prev, tok = self.t[self.i - 1], self.peek()
        kind, spaced = tok[0], tok[-1] if isinstance(tok[-1], bool) else False
        prev_num = prev[0] == 'num'
        if kind == 'num':
            return prev[0] == '!' or (prev[0] == ')' and not spaced)   # \frac{9!}{4! 3! 2!}, (2)3
        if kind == '(':
            if tok[1] in ('floor', 'ceil', 'g'):
                return not prev_num or not spaced
            return not spaced and (prev_num or prev[0] in (')', '!', 'var', 'pi'))
        if kind == 'bar':
            return False   # |a| |b| reads two ways
        if kind == 'frac' and prev_num and self._proper_fraction():
            raise NotExact('a mixed number: 2\\frac{1}{3} is 2 1/3 or 2/3')
        return True

    def _proper_fraction(self):
        """The \\frac ahead is `\\frac{a}{b}` with integers 0 <= a < b: after a number it reads as a mixed number. `308 \\frac{3+\\sqrt{5}}{2}`
        is a product."""
        t = self.t[self.i:self.i + 7]
        if len(t) < 7 or [x[0] for x in t] != ['frac', '(', 'num', ')', '(', 'num', ')']:
            return False
        a, b = t[2][1], t[5][1]
        return a.denominator == 1 and b.denominator == 1 and 0 <= a < b

    def unary(self):
        if self.at('op', '-') or self.at('op', '+'):
            sign = self.take()[1]
            node = self.unary()
            return ('neg', node) if sign == '-' else node
        return self.power()

    def power(self):
        base = self.postfix()
        if self.at('^'):
            self.take()
            exponent = self.exponent()
            base = ('^', base, exponent)
            while self.at('!') or self.at('deg'):   # `2^{10}!` is rare; `30^\circ` was read before
                base = ('!', base) if self.take()[0] == '!' else ('deg', base)
        return base

    def exponent(self):
        if self.at('op', '-') or self.at('op', '+'):
            sign = self.take()[1]
            node = self.exponent()
            return ('neg', node) if sign == '-' else node
        if self.at('num'):
            tok = self.take()
            return ('num', tok[1])
        return self.postfix()

    def postfix(self):
        node = self.primary()
        while True:
            if self.at('!'):
                self.take()
                if self.at('!') and not self.peek()[-1]:   # (12 - 1)!! = 11 x 9 x ... x 1
                    self.take()
                    node = ('!!', node)
                else:
                    node = ('!', node)
            elif self.at('op', '%') and not self._percent_is_mod():
                self.take()
                node = ('%', node)
            elif self.at('deg'):
                self.take()
                node = ('deg', node)
            else:
                return node

    def _percent_is_mod(self):
        nxt = self.peek(1)
        return nxt[0] in ('num', '(')   # `a % m` is a remainder in code; `50%` ends its value

    def group(self):
        """A braced group, or one digit (`\\frac12`), or one symbol."""
        if self.at('(', 'g'):
            self.take()
            node = self.expr()
            self.take(')', 'g')
            return node
        if self.at('(', '('):
            self.take()
            node = self.expr()
            self.take(')', '(')
            return node
        tok = self.take()
        if tok[0] == 'num' and tok[1].denominator == 1 and 0 <= tok[1] <= 9:
            return ('num', tok[1])
        if tok[0] == 'var':
            return ('var', tok[1])
        if tok[0] == 'pi':
            return ('pi',)
        raise NotExact('an argument with no braces')

    def argument(self):
        """What a function name applies to with no brackets: `\\sin 30^\\circ`, `\\sin \\frac{\\pi}{6}`, `\\log_2 8`, `\\sin 2x`."""
        if self.at('(', '(') or self.at('(', 'g') or self.at('(', '['):
            return self.postfix_power()
        node = self.postfix_power()
        while self.peek()[0] in ('var', 'pi'):
            node = ('*', node, self.postfix_power())
        return node

    def postfix_power(self):
        node = self.power()
        return node

    def repeating(self, tok):
        """`0.\\overline{49}` = 49/99, `0.4\\overline{9}` = 1/2: the digits under the bar repeat for ever."""
        self.take('overline')
        self.take('(', 'g')
        digits = ''
        while self.at('num'):
            digits += self.take()[3]
        self.take(')', 'g')
        raw = tok[3]
        if '.' not in raw or not digits.isdigit():
            raise NotExact('a bar over no repeating digits')
        places = len(raw.split('.', 1)[1])
        value = Fraction(raw.rstrip('.')) + Fraction(int(digits), 10 ** places * (10 ** len(digits) - 1))
        return ('num', value, False)

    def based(self, tok):
        """`17_{49}` = 1 x 49 + 7: decimal digits written in base b, each under b (a digit past 9 has no decimal spelling)."""
        self.take('_')
        base = self.group()
        if base[0] != 'num' or base[1].denominator != 1 or not 2 <= base[1] <= 10 ** 6:
            raise NotExact('a subscript that is no base')
        b, digits = int(base[1]), tok[3]
        if not digits.isdigit() or any(int(c) >= b for c in digits):
            raise NotExact('a digit past its base')
        return ('num', Fraction(sum(int(c) * b ** j for j, c in enumerate(reversed(digits)))), False)

    def primary(self):
        tok = self.peek()
        kind = tok[0]
        if kind == 'num':
            self.take()
            if self.at('overline'):
                return self.repeating(tok)
            if self.at('_') and not tok[2]:
                return self.based(tok)
            return ('num', tok[1], tok[2])
        if kind == '(':
            self.take()
            node = self.expr()
            close = self.take(')')
            if close[1] != tok[1]:
                raise NotExact('brackets that do not match')
            if tok[1] == 'floor':
                return ('floor', node)
            if tok[1] == 'ceil':
                return ('ceil', node)
            return node
        if kind == 'bar':
            self.take()
            node = self.expr()
            self.take('bar')
            return ('abs', node)
        if kind == 'frac':
            self.take()
            return ('/', self.group(), self.group())
        if kind == 'binom':
            self.take()
            return ('binom', self.group(), self.group())
        if kind == 'sqrt':
            self.take()
            index = None
            if self.at('(', '['):
                self.take()
                index = self.expr()
                self.take(')', '[')
            if self.at('num') and index is None:   # √2, \sqrt2, \sqrt 12
                n = self.take()
                return ('sqrt', ('num', n[1], n[2]), None)
            return ('sqrt', self.group(), index)
        if kind == 'pi':
            self.take()
            return ('pi',)
        if kind == 'trig':
            self.take()
            power = None
            if self.at('^'):   # \sin^2 x
                self.take()
                power = self.exponent()
            arg = self.argument()
            return ('trig', tok[1], power, arg)
        if kind == 'log':
            self.take()
            base = None
            if self.at('_'):
                self.take()
                base = self.group()
            power = None
            if self.at('^'):
                self.take()
                power = self.exponent()
            arg = self.argument()
            node = ('log', tok[1], base, arg)
            return ('^', node, power) if power is not None else node
        if kind == 'big':
            self.take()
            self.take('_')
            self.take('(', 'g')
            var = self.take('var')[1]
            self.take('=')
            lo = self.expr()
            self.take(')', 'g')
            self.take('^')
            hi = self.exponent()
            body = self.term()
            return ('big', tok[1], var, lo, hi, body)
        if kind == 'var':
            self.take()
            if self.at('(', '(') and not self.peek()[-1]:   # f(3): a call when the text defined f, a product otherwise
                return ('call?', tok[1])
            return ('var', tok[1])
        raise NotExact('expected a value, got ' + str(kind))


def _parse(text):
    p = _Parser(_tokens(text))
    if not p.t:
        raise NotExact('empty')
    node = p.expr()
    if p.i != len(p.t):
        raise NotExact('text left over')
    return node


class _Eval:
    def __init__(self, functions, degrees=False):
        self.functions = functions or {}
        self.degrees = degrees
        self.decimal = False
        self.irrational = False
        self.steps = 0

    def tick(self):
        self.steps += 1
        if self.steps > 200000:
            raise NotExact('too much work for one step')

    def ev(self, node, env, pi=None):
        self.tick()
        k = node[0]
        if k == 'num':
            if len(node) > 2 and node[2]:
                self.decimal = True
            return Value.of(node[1])
        if k == 'var':
            if node[1] in env:
                return env[node[1]]
            raise NotExact('a symbol: ' + node[1])
        if k == 'call?':
            raise NotExact('a call with its argument unread')
        if k == 'call':
            param, tree = self.functions[node[1]]
            arg = self.ev(node[2], env, pi)
            return self.ev(tree, {param: arg}, None)
        if k == 'pi':
            if pi is None:
                raise NotExact('pi outside an angle')
            return pi
        if k == 'deg':
            v = self.ev(node[1], env, pi)
            return v if pi is None else v * pi / Value.of(180)
        if k == '+':
            return self.ev(node[1], env, pi) + self.ev(node[2], env, pi)
        if k == '-':
            return self.ev(node[1], env, pi) - self.ev(node[2], env, pi)
        if k == 'neg':
            return -self.ev(node[1], env, pi)
        if k == '*':
            return self.ev(node[1], env, pi) * self.ev(node[2], env, pi)
        if k == '/':
            return self.ev(node[1], env, pi) / self.ev(node[2], env, pi)
        if k == '%':
            return self.ev(node[1], env, pi) * Value.of(Fraction(1, 100))
        if k == 'mod':
            m = _integer(self.ev(node[2], env, pi), 'an integer modulus')
            if m == 0:
                raise NotExact('mod 0')
            a = node[1]
            if a[0] == '^':   # a^b mod m: by squaring, however large b is
                base = _integer(self.ev(a[1], env, pi))
                e = _integer(self.ev(a[2], env, pi))
                if e < 0 or e > 10 ** 18:
                    raise NotExact('a negative or huge exponent mod m')
                return Value.of(pow(base, e, abs(m)))
            return Value.of(_integer(self.ev(a, env, pi)) % abs(m))
        if k == '^':
            return _power(self.ev(node[1], env, pi), self.ev(node[2], env, pi))
        if k == '!!':
            n = _integer(self.ev(node[1], env, pi))
            if not -1 <= n <= 60:
                raise NotExact('a double factorial out of reach')
            out = 1
            for j in range(n, 0, -2):
                out *= j
            return Value.of(out)
        if k == '!':
            n = _integer(self.ev(node[1], env, pi))
            if not 0 <= n <= 34:
                raise NotExact('a factorial past 34!')
            return Value.of(math.factorial(n))
        if k == 'sqrt':
            v = self.ev(node[1], env, pi)
            idx = 2 if node[2] is None else _integer(self.ev(node[2], env, pi))
            out = _root(v, idx)
            if out.rational() is None:
                self.irrational = True
            return out
        if k == 'binom':
            n, r = _integer(self.ev(node[1], env, pi)), _integer(self.ev(node[2], env, pi))
            if n < 0 or n > 10000:
                raise NotExact('a binomial out of reach')
            return Value.of(math.comb(n, r) if 0 <= r <= n else 0)
        if k == 'abs':
            v = self.ev(node[1], env, pi)
            return -v if v.sign() < 0 else v
        if k == 'floor':
            return self.ev(node[1], env, pi).floor()
        if k == 'ceil':
            return -((-self.ev(node[1], env, pi)).floor())
        if k == 'trig':
            out = _trig(node[1], self.angle(node[3], env))
            if node[2] is not None:
                out = _power(out, self.ev(node[2], env, pi))
            if out.rational() is None:
                self.irrational = True
            return out
        if k == 'log':
            arg = self.ev(node[3], env, pi)
            if node[2] is not None:
                return _log(self.ev(node[2], env, pi), arg)
            if arg == ONE:
                return ZERO   # \ln 1, \log 1 in any base
            raise NotExact('a log whose base the text does not give')
        if k == 'big':
            _, op, var, lo, hi, body = node
            a, b = _integer(self.ev(lo, env, pi)), _integer(self.ev(hi, env, pi))
            if b - a + 1 > MOST_RANGE:
                raise NotExact('a range too long')
            out = ZERO if op == 'sum' else ONE
            for j in range(a, b + 1):
                v = self.ev(body, dict(env, **{var: Value.of(j)}), pi)
                out = out + v if op == 'sum' else out * v
            return out
        raise NotExact(k)

    def angle(self, node, env):
        """The angle in degrees: pi stands in once as 180 and once as 360; the angle must scale with it or carry a degree mark."""
        has_pi = _has(node, 'pi')
        has_deg = _has(node, 'deg')
        if has_pi and has_deg:
            raise NotExact('pi and degrees in one angle')
        if has_pi:
            a = self.ev(node, env, Value.of(180))
            b = self.ev(node, env, Value.of(360))
            if b != a * Value.of(2):
                raise NotExact('an angle with a part apart from pi')
        elif has_deg:
            a = self.ev(node, env, Value.of(180))
        else:
            a = self.ev(node, env, None)
            if a != ZERO and not self.degrees:
                raise NotExact('an angle with no unit: degrees or radians')
        q = a.rational()
        if q is None:
            raise NotExact('an irrational angle')
        return q


def _has(node, kind):
    if not isinstance(node, tuple):
        return False
    if node[0] == kind:
        return True
    return any(_has(x, kind) for x in node[1:] if isinstance(x, tuple))


def _bind_calls(node, functions):
    """('call?', f) followed by its bracket was read as a product f * (arg): a call when the text defined f."""
    if not isinstance(node, tuple):
        return node
    if node[0] == '*' and isinstance(node[1], tuple) and node[1][0] == 'call?':
        name = node[1][1]
        if name in functions:
            return ('call', name, _bind_calls(node[2], functions))
        return ('*', ('var', name), _bind_calls(node[2], functions))   # k(k + 1): a product, of value only where k is bound
    if node[0] == '*' and isinstance(node[1], tuple) and node[1][0] == '*':
        inner = _bind_calls(node[1], functions)
        return ('*', inner, _bind_calls(node[2], functions))
    return tuple(_bind_calls(x, functions) if isinstance(x, tuple) else x for x in node)


def evaluate(text, functions=None, degrees=False):
    """The exact value of the whole text, or None. `degrees`: an angle with no unit is in degrees (`\\tan(45)`, `\\cot 135` in a text that
    works in degrees); without it only 0 has a unit-free value."""
    try:
        tree = _bind_calls(_parse(text), functions or {})
        e = _Eval(functions, degrees)
        value = e.ev(tree, {})
        if e.decimal and value.rational() is None:
            return None   # `\sqrt{310.9666}`: the model wants a decimal
        return value
    except (NotExact, ValueError, ZeroDivisionError, OverflowError, RecursionError, KeyError, IndexError):
        return None


_DEFINITION = re.compile(r'(?<![A-Za-z\\])([A-Za-z])\s*\(\s*([a-z])\s*\)\s*(?::=|=)\s*([^=$\n;]+?)(?=\s*(?:\$|\n|;|,\s|\.\s|\.$|\\quad|\\text|\\implies|\\Rightarrow|=|$))')


WINDOW = 3000   # characters back: a definition further back is from another case of the work (livebench-ca697327 re-used f for another cubic)
_DOMAIN = re.compile(r'\s*\$?\s*(?:,\s*)?(?:\\text\{\s*)?(?:for|if|when|where|on|with)\b|\s*\$?\s*(?:\\in|\\ge|\\le|\\geq|\\leq|<|>|≤|≥|\\quad\s*\(?\s*[a-z]\s*(?:\\ge|\\le|<|>))', re.I)

_DOMAIN_BEFORE = re.compile(r'\\in\b|\\le|\\ge|[<>≤≥]|\b(?:if|for|when|case|on|interval|piece|region)\b', re.I)


def definitions(text):
    """{name: (param, tree)} for each `f(x) = <expression in x>` the text wrote, the latest of each. An expression in x only: `f(x) = 0` is an
    equation to solve, not a definition, and one with another symbol in it has a value only with that symbol known."""
    out = {}
    for m in _DEFINITION.finditer(text, max(0, len(text) - WINDOW)):
        name, param, body = m.group(1), m.group(2), m.group(3).strip()
        if name == param:
            continue
        head = text[text.rfind('\n', 0, m.start()) + 1:m.start()]
        if _DOMAIN.match(text, m.end()) or _DOMAIN_BEFORE.search(head):
            out.pop(name, None)   # `F(u) = 1 - 4u` for `u \in [0, 1/4]`: one piece of a function, not the function
            continue
        try:
            tree = _parse(body)
        except (NotExact, ValueError, ZeroDivisionError, RecursionError, IndexError):
            out.pop(name, None)   # a later definition we cannot read voids the earlier one: never call an old f
            continue
        names = _vars(tree)
        if names != {param} or _has(tree, 'call?'):
            out.pop(name, None)
            continue
        out[name] = (param, tree)
    return out


def _vars(node):
    if not isinstance(node, tuple):
        return set()
    if node[0] == 'var':
        return {node[1]}
    if node[0] == 'big':
        return (_vars(node[3]) | _vars(node[4]) | _vars(node[5])) - {node[2]}
    out = set()
    for x in node[1:]:
        if isinstance(x, tuple):
            out |= _vars(x)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# Spelling
# ---------------------------------------------------------------------------------------------------------------------------------

def spell(value, latex=True):
    """One spelling per value: rational part first, then the roots by radicand; a common denominator under one fraction bar."""
    q = value.rational()
    if q is not None:
        return mathexpr.spelled(q)
    if all(c < 0 for c in value.t.values()):   # `-\frac{1 + \sqrt{5}}{4}`; with no denominator `-2 - \sqrt{3}` below
        inner = spell(-value, latex)
        if latex and inner.startswith('\\frac'):
            return '-' + inner
        if not latex and '/' in inner:
            return f'-({inner})' if ' ' in inner else '-' + inner
    terms = sorted(value.t.items())
    den = 1
    for _, c in terms:
        den = den * c.denominator // math.gcd(den, c.denominator)
    parts = []
    for r, c in terms:
        n = c * den
        assert n.denominator == 1
        n = int(n)
        if r == 1:
            body = str(abs(n))
        else:
            root = f'\\sqrt{{{r}}}' if latex else f'sqrt({r})'
            body = root if abs(n) == 1 else (f'{abs(n)}{root}' if latex else f'{abs(n)}*{root}')
        parts.append(('-' if n < 0 else '+', body))
    lead = next((j for j, (sign, _) in enumerate(parts) if sign == '+'), 0)
    parts = [parts[lead]] + parts[:lead] + parts[lead + 1:]   # `\sqrt{2} - 1`, as a model writes it, not `-1 + \sqrt{2}`
    text = ('-' if parts[0][0] == '-' else '') + parts[0][1]
    for sign, body in parts[1:]:
        text += f' {sign} {body}'
    if den == 1:
        return text
    if len(parts) == 1:
        sign, body = parts[0]
        lead = '-' if sign == '-' else ''
        return f'{lead}\\frac{{{body}}}{{{den}}}' if latex else f'{lead}{body}/{den}'
    return f'\\frac{{{text}}}{{{den}}}' if latex else f'({text})/{den}'
