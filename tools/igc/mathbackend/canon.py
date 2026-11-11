"""canon.py: the canonical encoding of MathBackend values, the one the CUDAMathBackend's C ABI speaks
(docs/plans/2026-11-11-cuda-mathbackend-design.md). Two backends agree on a case when their encodings are the same bytes.

A value is encoded as compact JSON of a tagged tree; every number travels as decimal digits (the device parses and prints them, so no
host arithmetic is hidden in a conversion):

    int        ["i", "-123"]                    bool   ["b", true]          None   ["n"]
    Fraction   ["q", "-1", "3"]                 always reduced, the sign on the numerator, the denominator > 1 (else an int)
    Decimal    ["d", "-1.50E+3"]                str(Decimal): digits and exponent as Python keeps them (1.50 is not 1.5)
    float      ["f", "0.1"]                     repr; a backend output that is a float is reported, not hidden
    str        ["s", "abc"]
    Surd       ["r", <p>, <q>, "d"]             p, q encoded values (int or Fraction), d the radicand
    list       ["l", [...]]   tuple ["t", [...]]   dict ["m", [[key, value], ...]] in the order Python holds them
    set        ["e", [...]]                     sorted by the encoding of the members (a set has no order to keep)
    refusal    ["x", "NotCovered"] | ["x", "BadInput"]   the class only; the message is a reason, not a value

Anything else raises TypeError: a type the encoding does not know is a gap to close, never a silent `str()`.
"""
import json
import sys
from decimal import Decimal
from fractions import Fraction

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)   # the encoding writes integers of any size as digits


def _tree(x):
    from tools.igc.mathbackend import Surd
    if x is None:
        return ['n']
    if isinstance(x, bool):
        return ['b', x]
    if isinstance(x, int):
        return ['i', str(x)]
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return ['i', str(x.numerator)]
        return ['q', str(x.numerator), str(x.denominator)]
    if isinstance(x, Decimal):
        return ['d', str(x)]
    if isinstance(x, float):
        return ['f', repr(x)]
    if isinstance(x, str):
        return ['s', x]
    if isinstance(x, Surd):
        return ['r', _tree(x.p), _tree(x.q), str(x.d)]
    if isinstance(x, list):
        return ['l', [_tree(y) for y in x]]
    if isinstance(x, tuple):
        return ['t', [_tree(y) for y in x]]
    if isinstance(x, dict):
        return ['m', [[_tree(k), _tree(v)] for k, v in x.items()]]
    if isinstance(x, (set, frozenset)):
        return ['e', sorted((_tree(y) for y in x), key=lambda t: json.dumps(t, separators=(',', ':')))]
    raise TypeError(f'canon: no encoding for {type(x).__name__}')


def encode(x):
    """The canonical bytes of a value."""
    return json.dumps(_tree(x), separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def encode_refusal(exc):
    """The canonical bytes of a refusal: NotCovered or BadInput (the class only)."""
    from tools.igc.mathbackend import BadInput, NotCovered
    if isinstance(exc, NotCovered):
        return encode_tree(['x', 'NotCovered'])
    if isinstance(exc, BadInput):
        return encode_tree(['x', 'BadInput'])
    raise TypeError(f'canon: {type(exc).__name__} is not a refusal (a Python bug, design "Reference Python")')


def encode_tree(t):
    return json.dumps(t, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def _value(t):
    from tools.igc.mathbackend import Surd
    tag = t[0]
    if tag == 'n':
        return None
    if tag == 'b':
        return bool(t[1])
    if tag == 'i':
        return int(t[1])
    if tag == 'q':
        return Fraction(int(t[1]), int(t[2]))
    if tag == 'd':
        return Decimal(t[1])
    if tag == 'f':
        return float(t[1])
    if tag == 's':
        return t[1]
    if tag == 'r':
        return Surd(_value(t[1]), _value(t[2]), int(t[3]))
    if tag == 'l':
        return [_value(y) for y in t[1]]
    if tag == 't':
        return tuple(_value(y) for y in t[1])
    if tag == 'm':
        return {_value(k): _value(v) for k, v in t[1]}
    if tag == 'e':
        return {_value(y) for y in t[1]}
    raise ValueError(f'canon: unknown tag {tag!r}')


def decode(b):
    """The value of canonical bytes; a refusal decodes to ('refusal', class name)."""
    t = json.loads(b.decode('utf-8') if isinstance(b, (bytes, bytearray)) else b)
    if t[0] == 'x':
        return ('refusal', t[1])
    return _value(t)


def call(name, args):
    """Run a Python operation and return its canonical bytes (a refusal included). Any other exception propagates: it is a Python bug."""
    from tools.igc.mathbackend import BadInput, NotCovered, run
    try:
        return encode(run(name, *args))
    except (NotCovered, BadInput) as e:
        return encode_refusal(e)
