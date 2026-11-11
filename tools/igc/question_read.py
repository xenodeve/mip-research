"""question_read.py: the computation a question states, read by software before the model starts (MIP catch point 1, #298).

pc4-whole lost 4 of 160 DeepMind rows because the model copied the question's numbers wrong, then the software computed exactly what
it wrote. Reading the question itself removes that step. A question gives either an expression (`What is 20*(-8)/(-60)*329?`) or the
English for one (`Subtract 3 from 7`, `What is 3 less than 7?`, `the product of 2 and 5`).

`read(question)` -> (expression in plain notation, exact value) or None. None when the question holds no single computation this
reader is sure of: a word problem, two candidate expressions, or symbols.
"""
import re
from fractions import Fraction

from tools.igc.mathexpr import evaluate

NUM = r'(-?\d[\d,]*(?:\.\d+)?)'
LEAD = r'(?:what is|what\'s|calculate|compute|evaluate|work out|find|determine|give)(?: the value of)?'
# (pattern, the expression it means); a and b are the first and second number as written
VERBAL = [
    (rf'(?:the )?product of {NUM} and {NUM}', '({a})*({b})'),
    (rf'multiply {NUM} (?:and|by) {NUM}', '({a})*({b})'),
    (rf'{NUM} (?:times|multiplied by) {NUM}', '({a})*({b})'),
    (rf'divide {NUM} by {NUM}', '({a})/({b})'),
    (rf'{NUM} divided by {NUM}', '({a})/({b})'),
    (rf'(?:the )?(?:sum|total) of {NUM} and {NUM}', '({a})+({b})'),
    (rf'(?:sum|add|put together|combine) {NUM} and {NUM}', '({a})+({b})'),
    (rf'add {NUM} to {NUM}', '({b})+({a})'),
    (rf'{NUM} plus {NUM}', '({a})+({b})'),
    (rf'{NUM} more than {NUM}', '({b})+({a})'),
    (rf'subtract {NUM} from {NUM}', '({b})-({a})'),
    (rf'take {NUM} (?:away )?from {NUM}', '({b})-({a})'),
    (rf'{NUM} take away {NUM}', '({a})-({b})'),
    (rf'(?:the )?distance between {NUM} and {NUM}', '|({a})-({b})|'),
    (rf'{NUM} less than {NUM}', '({b})-({a})'),
    (rf'{NUM} minus {NUM}', '({a})-({b})'),
    (rf'(?:the )?difference between {NUM} and {NUM}', '({a})-({b})'),
]


def _clean(text):
    first = text.strip().split('\n')[0].strip()   # the question, without the answer-format line after it
    return re.sub(r'\s+', ' ', first)


def read(question):
    q = _clean(question)
    body = re.sub(rf'^{LEAD}\s*', '', q, flags=re.I).rstrip(' ?.')
    for pattern, form in VERBAL:
        m = re.fullmatch(pattern, body, flags=re.I)
        if m:
            a, b = (x.replace(',', '') for x in m.groups())
            expr = form.format(a=a, b=b)
            value = evaluate(expr)
            return (expr, value) if value is not None else None
    if re.search(r'[A-Za-z]', body):   # words left: a word problem, or symbols
        return None
    value = evaluate(body)
    return (body, value) if value is not None and re.search(r'[-+*/×÷^]', body.lstrip('-')) else None


# The situation (developer 2026-11-11): the question asks for one expression's value and holds nothing else that carries content.
# What may stand around the expression is a request and nothing more: these words are the grammar of asking (courtesy, the asking
# verb, "the value/result of", "equals"). A word not here (round, decimal, binary, remainder, percent, a unit, a variable, `and`)
# could change what is asked, so it reads nothing. Written from English request grammar, not from any test's wording.
REQUEST_PHRASES = [r'the (?:value|result|answer) (?:of|to|for)', r'what do (?:you|i|we) get (?:when|if) (?:you|i|we) (?:compute|calculate|evaluate|work out|simplify)',
                   r'work out', r'figure out', r'how much is', r'how much does', r'thank you', r'let\'?s']
REQUEST_WORDS = {
    'please', 'pls', 'kindly', 'can', 'could', 'would', 'will', 'you', 'u', 'me', 'us', 'i', 'we', 'need', 'want', 'to', 'know', 'tell',
    'help', 'show', 'for', 'just', 'quick', 'quickly', 'one', 'now', 'hi', 'hey', 'hello', 'thanks', 'ok', 'okay', 'so', 'the', 'what', "what's",
    'whats', 'is', 'are', 'does', 'do', 'get', 'equal', 'equals', 'compute', 'calculate', 'evaluate', 'find', 'determine', 'give',
    'simplify', 'solve', 'result', 'value', 'answer', 'exact', 'exactly', 'expression', 'this', 'following', 'it',
}
_LATEX = [(r'\\(?:times|cdot|ast)\b', '*'), (r'\\div\b', '/'), (r'\\(?:left|right)\b', ''), (r'\\[,;!: ]', ''), (r'\$', ''),
          (r'\\[dt]?frac\b', r'\\frac')]


def _plain(expr):
    """LaTeX arithmetic to the plain notation the evaluators read: `\\times` `\\cdot` -> `*`, `\\div` -> `/`, `\\frac{a}{b}` -> `((a)/(b))`;
    `\\sqrt{n}` and `^{k}` stay (exactexpr reads them)."""
    for pattern, rep in _LATEX:
        expr = re.sub(pattern, rep, expr)
    frac = re.compile(r'\\frac\{([^{}]*)\}\{([^{}]*)\}')
    while frac.search(expr):
        expr = frac.sub(r'((\1)/(\2))', expr)
    return expr.strip()


def _one_expression(question):
    """The expression when the question is a request around one expression and nothing else, else None."""
    q = _clean(question)
    q = re.sub(r'\s*=\s*\?*\s*[.!]*\s*$', '', q)                      # `... = ?`
    q = re.sub(r'(?:\s*[?.!:,]+)+\s*$', '', q)                          # closing punctuation
    words = r'(?<![\\A-Za-z])(?:{})(?![A-Za-z])'
    for phrase in REQUEST_PHRASES:
        q = re.sub(words.format(phrase), ' ', q, flags=re.I)
    # request words only at the edges: in front of the expression and after it, never inside it
    edge = r"[A-Za-z']+"
    head = re.match(rf'^(?:\s*(?:{edge})\s*[,:!]?)*', q)
    lead, rest = q[:head.end()], q[head.end():]
    # the first token of the expression may be a LaTeX command or `sqrt`: give back a word glued to `(` or `{`
    m = re.search(r"([A-Za-z']+)\s*$", lead)
    if m and m.group(1).lower() == 'sqrt':
        lead, rest = lead[:m.start()], lead[m.start():] + rest
    tail = re.search(rf'(?:\s+(?:{edge}))*\s*$', rest)
    expr, after = rest[:tail.start()], rest[tail.start():]
    for w in re.findall(edge, lead + ' ' + after):
        if w.lower() not in REQUEST_WORDS:
            return None
    expr = _plain(expr.strip(' ,:'))
    if not expr or re.search(r'[A-Za-z]', re.sub(r'\\?sqrt|\\frac', '', expr)):
        return None
    if not re.search(r'[-+*/^×÷]|sqrt', expr.lstrip('-')):
        return None
    return expr


def read_exact(question):
    """As read, and also any question that asks for one expression's value in any request wording, LaTeX included (the situation,
    not DeepMind's `What is`/`Simplify`; round 9's reader saw 0 of 80 questions re-asked as `{e} = ?` or `Can you calculate {e} for
    me?`). Computed exactly by mathexpr, else exactexpr (square roots): (expression, Fraction or exactexpr.Value) or None. A Value only
    when the result is not rational; pi, a variable, or any word that is not a request reads nothing."""
    got = read(question)
    if got is not None:
        return got
    body = _one_expression(question)
    if body is None:
        return None
    value = evaluate(body)
    if value is not None:
        return (body, Fraction(value))
    from tools.igc.exactexpr import NotExact, evaluate as exact
    try:
        value = exact(body)
    except (NotExact, ArithmeticError, ValueError, OverflowError, RecursionError, KeyError, IndexError, TypeError):
        return None
    if value is None or value.has_pi():
        return None
    rational = value.rational()
    return (body, rational if rational is not None else value)


def as_answer(value):
    """The value as DeepMind spells it: an integer, a terminating decimal, or a reduced fraction."""
    from tools.igc.tir4b import spelled
    return spelled(Fraction(value))
