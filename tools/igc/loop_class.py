"""MIP's decide layer, v1 (docs/plans/2026-11-10-decide-layer-design.md): at a loop, read what the model is repeating and pick the
settle line software can give, before any close. Classes, each defined by the notation, not by a test set:

  E  a run: one token or one contentless line over and over (nothing to settle, close as today)
  D  a search for a common factor: one dividend over a run of divisors (settle: the gcd / lowest terms)
  C  a form choice or a settled answer re-confirmed: the value is found (settle: the exact value in its exact form)
  A  re-checking an equation that is true (settle: a confirm line)
  B  doubting how the question's input reads (settle: the question's text, verbatim)
  F  anything else: reasoning stuck (close as today)

A settle line is given only when its content is exact (a true equation, a verbatim quote, an exact value); otherwise None."""
from collections import Counter
from fractions import Fraction
from math import gcd
import re

from tools.igc.mathexpr import evaluate, wrong_relations

_BULLET = re.compile(r'^(?:[*\-•]\s+|\d+\.\s+)+')
_DOUBT = re.compile(r"\bwhat if\b|\bis it\b|\bmaybe\b|\blooking at\b|\bI see\b|\bmisread\b|\btypo\b|\bcould it be\b|\bis there\b|"
                    r"\bdoes it\b|\bdid I\b|\bthe string\b|\bthe prompt\b|\bthe question\b|\bmeant\b|\bmeans\b|\?", re.I)
_DECIDE = re.compile(r"\bI(?:'ll| will)\s+(?:write|put|provide|output|give|go with|stick with|use|assume)\b|\bthe (?:final )?answer is\b|"
                     r"\\boxed\{|\bfraction\b|\bdecimal\b|\binteger\b|\bexact (?:value|form)\b|\bround(?:ed|ing)?\b", re.I)
_DIVIDE = re.compile(r'(-?\d{2,})\s*(?:/|\\div|÷)\s*(\d+)')
_QUOTED = re.compile(r'`([^`]{2,})`|"([^"]{2,})"')
_RUN = re.compile(r'(.{1,4}?)\1{19,}')


def lines_of(text):
    """Non-empty lines, bullets and edge spaces stripped."""
    out = []
    for line in text.split('\n'):
        line = _BULLET.sub('', line.strip()).strip()
        if line:
            out.append(line)
    return out


def cycle(text, window=60, repeats=3):
    """The lines that repeat at least `repeats` times among the last `window` lines: the loop's unit. Empty when nothing repeats."""
    tail = lines_of(text)[-window:]
    counts = Counter(tail)
    return [line for line, n in counts.items() if n >= repeats]


def _math(line):
    return bool(re.search(r'\d|\$|\\[a-z]+|[=+*/^]', line))


def _true_relation(line):
    """The first `lhs = rhs` on the line that computes and is not wrong, or None."""
    body = line.replace('$', '')
    if '=' not in body or wrong_relations(line):
        return None
    parts = [p.strip(' .,;:?') for p in re.split(r'(?<![<>!=])=(?!=)', body)]
    for left, right in zip(parts, parts[1:]):
        left = re.sub(r'^.*?(?:Wait,?|So|Then|Thus|Check:?|Yes,?)\s+', '', left, flags=re.I).strip()
        lv, rv = evaluate(left), evaluate(right)
        if lv is not None and rv is not None and lv == rv and re.search(r'[-+*/^×÷]|\\(?:times|div|frac|cdot)', left.lstrip('-')):
            return f'{left} = {right}'
    return None


def _search(tail):
    """A dividend divided by at least 5 different divisors in the tail: the trial division of a factor search."""
    by = {}
    for line in tail:
        for a, b in _DIVIDE.findall(line):
            by.setdefault(a.lstrip('-'), set()).add(b)
    hits = [(a, ds) for a, ds in by.items() if len(ds) >= 5]
    return max(hits, key=lambda h: len(h[1]))[0] if hits else None


def _question_terms(question):
    q = question.split('\n')[0]
    words = set(re.findall(r'[A-Za-z]{3,}|\d{2,}|\d', q))
    return q, words


def _near_question(line, question):
    """The line refers to the question's input: a quoted or backticked piece that shares a run with the question, or 2+ of its terms."""
    q, terms = _question_terms(question)
    for a, b in _QUOTED.findall(line):
        piece = (a or b).strip()
        compact = re.sub(r'\s+', '', piece)
        if compact and (compact in re.sub(r'\s+', '', q) or _shared_run(compact, re.sub(r'\s+', '', q)) >= 4):
            return True
    found = set(re.findall(r'[A-Za-z]{3,}|\d{2,}|\d', line)) & terms
    return len(found) >= 2


def _shared_run(a, b):
    best = 0
    for i in range(len(a)):
        for j in range(i + best + 1, len(a) + 1):
            if a[i:j] in b:
                best = j - i
            else:
                break
    return best


def classify(before, question):
    """(class, detail) for the loop that ends `before` (the text up to the close point)."""
    tail = lines_of(before)[-60:]
    unit = cycle(before)
    last = tail[-1] if tail else ''
    if _RUN.search(last) or (unit and all(not _math(u) and len(u.split()) <= 6 and '?' not in u for u in unit)):
        return 'E', last[-80:]
    divided = _search(tail[-40:])
    if divided:
        return 'D', divided
    lines = unit or tail[-12:]
    if any(_DECIDE.search(u) for u in lines) and not any(_true_relation(u) for u in lines if not _DECIDE.search(u)):
        return 'C', next(u for u in lines if _DECIDE.search(u))
    for u in lines:
        rel = _true_relation(u)
        if rel:
            return 'A', rel
    if any(_DOUBT.search(u) and _near_question(u, question) for u in lines) or any(_QUOTED.search(u) and _near_question(u, question)
                                                                                 for u in lines):
        return 'B', next((u for u in lines if _near_question(u, question)), lines[0])
    return 'F', None


def settle(cls, detail, question):
    """The line software puts before any close, or None (close as today). Exact content only."""
    if cls == 'A':
        return f'Checked exactly: ${detail}$ holds.'
    if cls == 'B':
        q = question.split('\n')[0].strip()
        return f'The question writes, exactly: {q}' if len(q) <= 400 else None
    if cls in ('C', 'D'):
        from tools.igc.question_read import read
        got = read(question)
        if got is None:
            return None
        from tools.igc.question_read import as_answer
        value = Fraction(got[1])
        if cls == 'D' and value.denominator != 1:
            n, d = abs(value.numerator), value.denominator
            return f'Computed exactly: ${got[0]} = {as_answer(value)}$; $\\gcd({n}, {d}) = {gcd(n, d)}$, so it is in lowest terms.'
        return f'Computed exactly: ${got[0]} = {as_answer(value)}$.'
    return None
