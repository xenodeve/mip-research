"""Qwen3.5-4B on vLLM with the sandbox rules of the Flash-Next engine, rebuilt here: the 4B fine-tune track (#288, PRD R13 layer 2).

Strata's engine runs only the Flash-Next family, so the 4B gets the same rules in Python, by the engine's own words and limits
(include/strata/program/igc_learned_diagnostic.hpp, include/strata/core/igc_sandbox_bridge.hpp):
- after `force_after` tokens of thinking a block opens at the next line end, once (`--igc-sandbox-after`, :552): forcing every N tokens looped;
- a block the model closes runs in the kernel and its output comes back inside the thinking (:460), whoever opened the block, so a fine-tuned
  model that opens its own blocks is measured by the same path;
- a run that finished and printed closes the thinking (`--igc-sandbox-close`, :512): with no close, sandbox alone cost 1.50x plain (#279);
- a thinking that loops is closed (`repeating`, the bench's copy of the engine's checks), and an end of turn inside the thinking after a run closes it.
Not here: MIP's inline arithmetic, which lives in the CUDA engine. Every text the runner put in is a span of the record, with its kind.

tir4b.py run --model M --rows R.jsonl --out DIR [--force-after 512] [--execute] [--close-fit] [--loop-close] [--samples K] [--seed S] [--ids]
tir4b.py math-train OUT.jsonl --n N [--levels 3,4,5] [--exclude ROWS.jsonl ...]     (on the VM: MATH train, never a test row)
tir4b.py keep DIR [DIR ...] --out TRAIN.jsonl [--max-len 32768]                      (the trajectory each question trains on)"""
import argparse, hashlib, json, math, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

OPEN_LINE = 'Need a short Python program here.\n\n```python\n'   # kOpenBlock, as text
NOTHING = '[it ran and printed nothing: call your function and print the result]'   # kNothingPrinted
SPENT = '[no more runs]'
TOO_LONG = '[too long, not run]'
SHOWN = 1024   # kShownOutput, bytes
MAX_BLOCK = 2048   # kMaxBlockTokens
MAX_RUNS = 128   # kMaxRuns
CLOSE = '</think>\n\n'   # {kThinkClose, 271}
LOOP_CLOSE = '\n</think>\n\n'   # {198, kThinkClose, 271}
# --answer-close: a loop in the answer, after the thinking is closed, ends with the model's one final value instead of `stuck`. loop-lc (#307)
# closed 15 thinking loops, 12 then answered right; 3 carried the same doubt into the answer and looped there (an endless decimal
# `-5511170.918918...`, a recount of `19+1`), where nothing was left to close
# The cue ends before `{`: Qwen writes `{-` as one token, and after a cue ending in `{` 22 of 28 gold-negative values lost the minus
# (A100 bf16 and PC NVFP4 alike; dm_div_big-9977 boxed -70468 five times, then wrote 70468 after the cue). Round 9 (2026-11-09)
ANSWER_CLOSE = '\n\nThe final answer is $\\boxed'
ANSWER_CLOSE_TOKENS = 64   # the value after the cue; a value still going past this is no value
LOOP_VALUE = 'Computed exactly: ${expr} = {value}$'   # --loop-value: the question's computation, put before a loop is closed
CHUNK = 1024   # thinking tokens between two loop checks; the engine checks every 16, so a close can come up to this much later
CONTEXT = 262144   # the model card's native context, all of it (developer, 2026-11-03)
# Qwen's precise set for thinking (model card). Its general set adds presence_penalty 1.5, which vLLM counts per request: every block the runner
# answers starts a new request and would reset it, so an arm with blocks would sample differently from one without.
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)
MATH_TAIL = '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'   # the model card's own math prompt
# --native-tool: Qwen3.5's own tool format (the checkpoint's chat_template.jinja), not a fence the runner watches. The tool goes in the system
# turn, the model calls it after its thinking, the answer comes back as a user turn of <tool_response>, and a new thinking opens.
PYTHON_TOOL = dict(type='function', function=dict(
    name='python', description='Runs Python 3 code in a sandbox and returns what it prints. Use it to compute, check or search instead of '
                               'working numbers out by hand. Print every value you need.',
    parameters=dict(type='object', properties=dict(code=dict(type='string', description='The program to run.')), required=['code'])))
TOOL_TURN = '<|im_end|>\n<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n'
NO_CODE = '[no code parameter: call python with <parameter=code>]'
_CODE = re.compile(r'<parameter=code>\n?(.*?)\n?</parameter>', re.S)
# --hint-calc: CoRT's hint-engineering (arXiv 2506.09820) at test time, START's Hint-infer (2503.04625). Where the model begins a tedious hand
# calculation, its line is taken back and CoRT's own hint opens a program; after a run that printed, CoRT's second hint answers the doubt that
# made the 4B re-check a sum ten times (r1 5456f419). The thinking goes on: no close, which FT1 learned as an early end.
HINT_OPEN = 'It looks tedious, and we can use python code to simplify the reasoning.\n```python\n'
TRUST = "We don't need to doubt the accuracy of python calculations.\n"
# --trust-line: lb1 (2026-11-04) put TRUST after every hinted program and the 4B still re-checked a printed -530 for 12,000 tokens (and a printed 320 ten times).
# Our own H-trust arms found that a line vouching for a value raises the re-check word (+22 to +35 points against a bare new line), and negated
# instructions can raise the thing they forbid (arXiv 2511.12381). 'none' leaves the line out; 'positive' is the sentence ATTC inserts when its confidence
# gate passes (arXiv 2604.08281, as the paper's Table 5 is reported to give it; ATTC gates it, this does not).
TRUST_LINES = {'cort': TRUST, 'positive': 'I have great confidence in the code above, I should believe the code execution result.\n'}
HINT_GAP = 1024   # thinking tokens between two hints
# --whole-hint: on Qwenputer's DeepMind rows the 4B cut each step into the pieces of its own method (a long division's `21 \div 19 =`), so a
# value put in at `=` came for a piece, in a form the method could not use, and looped. This opens the reply asking for the whole step as
# one expression; its value is computed whole, sign and uneven quotient included, as Qwenputer's model writes `48213 x 9917 =`.
WHOLE_HINT = 'Let me compute the whole expression exactly in one step:\n'
# --whole-plain: v1 above took base's 121 to 127 of 160 at 0.41x tokens, but lost 14 mul_div_multiple_longer rows: the model rewrote the
# question's plain nested expression as LaTeX \frac inside \frac in a $$ block with no `=`, so the whole was never computed. The question's
# own form is what the evaluator reads exactly (-303, 78 and 319 for three of those rows), so this asks for that form.
WHOLE_HINT_PLAIN = ('Let me write the whole calculation as one plain expression, with +, -, *, / and parentheses, in the order the question '
                    'gives, and compute it exactly:\n')
# --check-hand: what the model already worked out by hand is checked at the end of its line, and a wrong value is answered with the right
# one (developer, 2026-11-04). On 82 Base answers 462 of 8,924 hand values were wrong, 5.6 an answer; rewrites (`= 189 + 7 = 196`), rounding
# to the places written and a long division's uneven quotient digit are not wrong.
# The model met 'Correction: X = V, not W.' by redoing the work by hand, and got it wrong again as often as not (dm-check: 116 of 160
# against base's 121); the correction now says it was computed and is to be used, CoRT's answer to the same distrust.
CORRECTION = 'Correction (computed exactly): {expr} = {right}, not {wrote}. Use {right} and go on.\n'
# --hand-off: the model only restating a value software gave it (`= -18` on every line, or `= -0.0000020186299999...` for -37/18329827)
# lost 28 of 160 DeepMind rows whose value was right in the text (pc4-whole). The restating line is cut and the model is led on to its
# answer; software does not write the answer (developer, 2026-11-04: rules in MIP first).
HAND_OFF = 'So the answer is '
_RESTATE = re.compile(r'\s*=?\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*\$?\s*\.?\s*')
_EXPANSION = re.compile(r'-?\d*\.\d{30,}')
# --loop-hint: 14 of the control's 16 misses on the r1 rows were loops, 13 of them mid-work, several of them searches by hand. A loop in the
# thinking is cut back to one round and a program opens to test the idea, LOOP_HINTS times; the next loop ends the answer as before.
LOOP_HINT = 'I keep going around the same idea. Let me test it with python instead.\n```python\n'
LOOP_HINTS = 3
# --candidate-close: a loop in the thinking with no final answer yet is cut back to one round and nudged, at most LOOP_NUDGES times (#304)
LOOP_NUDGE = '\nWait, let me rethink this derivation carefully.\n'
LOOP_NUDGES = 3
# --inline-calc: the ECHO_RUN-th value in a row for one expression, each within ECHO_GAP model tokens of the last, is left to the model (#301)
ECHO_RUN = 3
ECHO_GAP = 64
# The words the model should learn to write itself, as START's Hint-RFT keeps its hints in the text it trains on.
TRAIN_KINDS = ('forced open', 'close after a fitting result', 'hint open', 'trust hint', 'loop hint')
MASK_KINDS = ('output', 'inline value', 'logit value', 'tool response', 'hand check')   # what the kernel printed and the values software computed: read, never learned


def output_block(body):
    return '\n```output\n' + body + '\n```\n\n'


def shown_text(ok, shown):
    """What the model reads back, cut as the engine cuts it."""
    body = shown or ''
    if ok and not body.strip():
        body = NOTHING
    data = body.encode('utf-8')
    return data[:SHOWN].decode('utf-8', 'ignore') if len(data) > SHOWN else body


JOIN_BIAS = -100.0   # --no-digit-join: a digit cannot follow a put that ends in one
BIAS_MAX = 1024      # vLLM 0.33's GPU sampler kills the engine on a logit_bias over this many tokens (#307: 1,106 ended all four A100 arms)
RECHECK_WORDS = {'wait', 'hmm', 'hmmm', 'verify', 'verifying', 'check', 'checking', 'double', 'recheck', 'rechecking', 'recalculate', 'recompute', 'again'}
SUPPRESS_BIAS = -30.0   # added to those words' logits: -inf in NoWait, a finite number that no sampled logit climbs back from


class Ids:
    """The special tokens of the 4B's tokenizer, looked up, never carried over from the Flash-Next pack."""
    def __init__(self, tokenizer, eos):
        self.think_close = tokenizer.convert_tokens_to_ids('</think>')
        self.ends = {tokenizer.convert_tokens_to_ids('<|im_end|>'), *eos}
        lines = [tokenizer.encode(s, add_special_tokens=False) for s in ('\n', '\n\n')]
        self.lines = {t[0] for t in lines if len(t) == 1}
        self.fence = {i for i in range(len(tokenizer)) if '```' in tokenizer.decode([i])}
        if not self.fence or len(self.lines) != 2 or not isinstance(self.think_close, int):
            raise RuntimeError('tokenizer: no fence, line or </think> token')
        # repeating() and line_back() read 198 and 271 as the line ends, the Flash-Next pack's ids: a tokenizer with others would blind them
        self.same_lines = self.lines == {198, 271}
        self.equals = {i for i in range(len(tokenizer)) if tokenizer.decode([i]).strip() == '='}   # the tokens an inline value follows
        # --suppress-recheck: the words a re-check starts with (NoWait / CGRS list, the same words the H-trust tests count)
        self.recheck = {i for i in range(len(tokenizer)) if tokenizer.decode([i]).strip().lower().rstrip('.,:!') in RECHECK_WORDS}
        # --no-digit-join: every token whose text starts with a digit, so none can run on a number software put
        self.digit_start = {i for i in range(len(tokenizer)) if tokenizer.decode([i])[:1].isdigit()}
        # and every token that starts a division with `/`: `-23/11 + 12/11 =` put `-1`, the model's planned `/11` read `-1/11` (#307, 121 of
        # 112,793 puts). Only `/` alone or before a digit, bracket, `-` or `\`: the 4B's vocabulary has 1,078 tokens starting with `/`, nearly
        # all code (`//`, `/**`), and holding them all broke vLLM's cap below (12 are left)
        self.slash_start = {i for i in range(len(tokenizer)) if self._division(tokenizer.decode([i]))}
        self.digits = frozenset(i for i in range(len(tokenizer)) if tokenizer.decode([i]) in tuple('0123456789'))   # --shape-cap
        self.point = {i for i in range(len(tokenizer)) if tokenizer.decode([i]) == '.'}   # `0` then `.75` read `0.75`: digits held after it
        self.tool_close = tokenizer.convert_tokens_to_ids('</tool_call>')
        held = len(self.recheck | self.digit_start | self.slash_start)   # the most bias() ever asks for at once
        if held > BIAS_MAX:
            raise RuntimeError(f'tokenizer: bias() could hold {held} tokens, vLLM refuses more than {BIAS_MAX}')

    @staticmethod
    def _division(text):
        return text[:1] == '/' and (text[1:2] in ('', '(', '[', '{', '-', '\\') or text[1:2].isdigit())


class Arm:
    def __init__(self, force_after=0, execute=False, close_fit=False, loop_close=False, close_repeat=0, inline_calc=False, native_tool=False,
                 hint_calc=False, loop_hint=False, inline_plain=False, whole_hint=False, check_hand=False, whole_plain=False, hand_off=False, question_value=False,
                 trust='cort', logit_value=0.0, suppress_recheck=0, mathbackend=False, candidate_close=False, inline_style='general',
                 no_digit_join=False, guard_progress=False, answer_close=False, loop_value=False, early_cycle=False, shape_cap=0, cover_cap=0,
                 exact_answer=False):
        self.exact_answer = exact_answer   # --exact-answer: the question's own computation, exact, is what the answer's box holds (round 9 form)
        self.cover_cap = cover_cap   # --cover-cap W: 97 % of the last W tokens repeat a 12-gram of the last 2,048: a loop (#307 round 8)
        self.shape_cap = shape_cap   # --shape-cap: a cycle of this many rounds with every number read as one token is a loop, progress or not (#307)
        self.early_cycle = early_cycle   # --early-cycle: the thinking is also read for a short exact cycle, every request (#307, V7)
        self.loop_value = loop_value   # --loop-value: before a loop is closed, the question's own computation, exact (#307)
        self.answer_close = answer_close   # --answer-close: a loop in the answer gets the final-answer cue and one value, not `stuck` (#307)
        self.trust = trust   # the line after a hinted program that printed: 'cort' (the original), 'none', 'positive'
        self.no_digit_join = no_digit_join   # --no-digit-join: the token after a put that ends in a digit starts with no digit (#307)
        self.guard_progress = guard_progress   # --guard-progress: the counting check lets a tail whose rounds are nearly all new go on (#307)
        self.suppress_recheck = suppress_recheck   # --suppress-recheck: this many steps after a value software gave, the re-check words' logits are lowered
        self.logit_value = logit_value   # --logit-value: the logit added to software's next value token while the model writes the value itself
        self.force_after, self.execute, self.close_fit, self.loop_close, self.close_repeat = force_after, execute, close_fit, loop_close, close_repeat
        self.inline_calc, self.native_tool, self.hint_calc, self.loop_hint = inline_calc, native_tool, hint_calc, loop_hint
        self.inline_plain, self.whole_hint, self.check_hand = inline_plain, whole_hint, check_hand
        self.whole_plain, self.hand_off, self.question_value = whole_plain, hand_off, question_value
        self.mathbackend, self.candidate_close = mathbackend, candidate_close
        if inline_style not in INLINE_STYLES:
            raise ValueError(f'--inline-style: one of {sorted(INLINE_STYLES)}')
        self.inline_style = inline_style


# --inline-calc: MIP Native's inline value (include/strata/core/igc_inline.hpp parse_inline / after_value) for a model the engine cannot run yet.
# When the reasoning writes an arithmetic expression and `=`, the value is computed exactly and put in, with the engine's ` (exact)`; the
# model never does the arithmetic itself. On 82 base answers the 4B wrote 7,283 such equations, 504 of them wrong, in 12 of its 24 misses.
_OPS = re.compile(r'[+\-*/^×÷·⋅−]|\\times|\\cdot|\\frac|\\div')


def with_call(code):
    """(code, name): a program that defines a function of no arguments and never calls it gets the call, and its value printed, as a notebook
    cell would show it. The 4B ended all 6 hinted programs of its first answer (r1 5456f419) with `def find_n(): ...` and no call: each printed
    nothing in 0.001 s, the model took the note to call it as a failure and went back to hand algebra, though the search would have printed
    1, 11, 37. A program that does not parse is left as written, so the run shows the model its error."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    top_calls = any(isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) for s in tree.body)
    free = [s.name for s in tree.body if isinstance(s, ast.FunctionDef) and not (s.args.args or s.args.posonlyargs or s.args.kwonlyargs
                                                                                 or s.args.vararg or s.args.kwarg) and s.name not in called]
    if not free or top_calls:
        return code, None
    name = free[-1]
    return code + ('' if code.endswith('\n') else '\n') + f'\n_r = {name}()\nif _r is not None:\n    print(_r)\n', name


def vllm_refusal(context, most, bias):
    """Why vLLM would refuse this request, or None. The known limits only, each one met on a run: a replay never builds the request, so it
    checks them here (#307: a 1,106-token logit bias, over the cap of 1,024, ended all four A100 arms of the loop round within minutes).
    context: the tokens the request reads (prompt and answer so far)."""
    if most < 1:
        return f'max_tokens {most}: vLLM needs at least 1'
    if context >= CONTEXT:
        return f'{context} tokens read: the model holds {CONTEXT}'
    if bias and len(bias) > BIAS_MAX:
        return f'{len(bias)} logit bias tokens: vLLM refuses more than {BIAS_MAX}'
    return None


def loop_start(tokens, longest=4096):
    """Where the second round of the cycle that ends `tokens` begins, so a cut there keeps one round; None when the tail is no cycle of at
    least four rounds (a counting or line-back loop)."""
    for period in range(1, min(longest, len(tokens) // 2) + 1):
        if tokens[-1] != tokens[-1 - period]:
            continue
        i = len(tokens) - period - 1
        while i >= 0 and tokens[i] == tokens[i + period]:
            i -= 1
        if len(tokens) - (i + 1) >= 4 * period:
            return i + 1 + period
    return None


def tedious(expr, value):
    """Arithmetic worth a program: a number of three or more digits, three operators, or a value past 999. `1 + 1` is not."""
    return any(len(n) >= 3 for n in re.findall(r'\d+', expr)) or len(_OPS.findall(expr)) >= 3 or abs(Fraction(value)) >= 1000


# How a value is written where the model's own `=` may mean something else (work mod m, a long division, a quotient in a text that works in
# decimals). `general`, the default, writes what is true under every reading and guesses nothing, so it holds for any model and any
# question: `16807 \equiv 7`, `6 rem 1`, `239.1428...` (cut, not rounded). A model's profile instead writes the one form that model was
# measured to want there (developer 2026-11-06: keep the guesses, as the software optimised for that model):
#   mod        'both': the value and its residue | 'habit': the residue only where the model's latest `=` near there meant `≡`, else the value
#   remainder  'both': `q rem r` | 'quotient': q alone
#   decimal    'cut': cut to 4 places and marked `...` | 'rounded': rounded to 4 places, no mark
# The profiles were measured on stored answers (#306): Qwen3.5-4B on its seven control arms (563 rows) and four held-out arms (H1, 484 rows);
# DeepSeek-R1 and QwQ-32B on the H2 traces (630 rows). Those sets were used to write the guesses, so they measure the profiles in sample.
INLINE_STYLES = {
    'general': {'mod': 'both', 'remainder': 'both', 'decimal': 'cut'},
    # a family profile: the General Style, then at each decision point one of its true forms, chosen by simulation on the family's traces
    # (tools/igc profile_sim: tune half of Qwen3.5-4B's stored traces, checked on the other half and on Qwen3-235B). `forms` maps
    # (decision, latex, decimals) or decision to a form name of `true_forms`; a point the table does not name keeps the General form.
    # picked by pick_forms' fixed rule on the tune half (2026-11-06): a/b already in lowest terms gets no cut decimal, since Qwen went on
    # from it at 3-11 % of such points (tune 20/371; check 33/368, Qwen3-235B 7/153); every other point keeps the General form
    'qwen': {'base': 'general', 'mod': 'both', 'remainder': 'both', 'decimal': 'cut', 'forms': {'quotient-lowest': 'none'}},
    'qwen3.5-4b': {'mod': 'habit', 'remainder': 'quotient', 'decimal': 'rounded'},
    'deepseek-r1': {'mod': 'habit', 'remainder': 'quotient', 'decimal': 'rounded'},
    'qwq-32b': {'mod': 'habit', 'remainder': 'quotient', 'decimal': 'rounded'},
}


def _cut4(v):
    q = abs(v.numerator) * 10 ** 4 // v.denominator
    w, f = divmod(q, 10 ** 4)
    return f"{'-' if v < 0 else ''}{w}.{f:04d}..."


def _frac(v, latex):
    if v.denominator == 1:
        return str(v.numerator)
    sign = '-' if v < 0 else ''
    return f'{sign}\\frac{{{abs(v.numerator)}}}{{{v.denominator}}}' if latex else f'{sign}{abs(v.numerator)}/{v.denominator}'


def true_forms(expr, value, general, before):
    """(decision, {form: spelling or None for no put}) of every form true for the expression at a point where the General Style has a
    choice, or None. The decisions and their forms are the ones tools/igc profile_sim scores; a profile picks among them."""
    from fractions import Fraction as _F
    from tools.igc.mathexpr import simple_quotient
    line = before.rsplit('\n', 1)[-1]
    latex = '\\' in line or '$' in line
    if '≡' in general or '\\equiv' in general:
        return 'mod', {'both': general, 'value': re.match(r'(-?\d+)', general).group(1)}
    if '\\approx' in general or '≈' in general:
        e, d = re.split(r'\s*(?:\\approx|≈)\s*', general, 1)
        return 'decimal-root', {'both': general, 'exact': e, 'decimal': d}
    if not isinstance(value, _F) or value.denominator == 1:
        return None
    terminates = '/' not in spelled(value)
    if simple_quotient(expr) and not re.search(r'\\d?frac', expr):
        nums = [int(x) for x in re.findall(r'\d+', expr)]
        if len(nums) != 2:
            return None
        a, b = nums
        dec = spelled(value) if terminates else _cut4(value)
        if 'rem' in general:
            return 'quotient-division', {'rem': general, 'fraction': _frac(value, latex), 'decimal': dec}
        if math.gcd(a, b) > 1:
            return 'quotient-reduces', {'fraction': _frac(value, latex), 'decimal': dec}
        if terminates:
            return 'quotient-terminates', {'decimal': dec, 'fraction': _frac(value, latex)}
        return 'quotient-lowest', {'decimal': dec, 'none': None}
    forms = {'fraction': _frac(value, latex)}
    if terminates:
        forms['decimal'] = spelled(value)
    return 'rational', forms


def inline_found(before, whole=False, use_mathbackend=False, custom_ops=(), style='general'):
    """The General Style's put (`_inline_general`), then a family profile's choice of form at a decision point."""
    got = _inline_general(before, whole, use_mathbackend, custom_ops, style)
    table = INLINE_STYLES[style].get('forms')
    if not got or not table or whole or not use_mathbackend:
        return got
    from tools.igc.unified_detect import expression_before_with_mathbackend
    raw = expression_before_with_mathbackend(before)
    alt = true_forms(got[0], raw[1] if raw else None, got[1], before)
    if alt is None:
        return got
    decision, forms = alt
    line = before.rsplit('\n', 1)[-1]
    key = (decision, '\\' in line or '$' in line, re.search(r'(?<![\d.])\d+\.\d+', before[-600:]) is not None)
    pick = table.get(key, table.get(decision))
    if pick is None or pick not in forms:
        return got
    return (got[0], forms[pick]) if forms[pick] is not None else None


def _inline_general(before, whole=False, use_mathbackend=False, custom_ops=(), style='general'):
    """(expression, spelled exact value) for the arithmetic that ends the text before an `=`, or None. The expression is read by
    tools.igc.mathexpr in every form a model writes (LaTeX at any depth, Unicode, ASCII); it refuses a symbol (algebra), a word, a mixed
    number, a root or power that is not exact. When `use_mathbackend` is True, also resolves exact calls to the 6-module mathbackend
    (gcd, lcm, phi, C, P, catalan, derangements, modinv, 2x2 det). `whole`: the expression the --whole-hint asked for, whose value is
    given even as an uneven quotient, since it is the step itself, not a piece of the model's long division. `style`: a key of
    INLINE_STYLES, `general` unless the model is known."""
    from tools.igc.mathexpr import expression_before, simple_quotient
    st = INLINE_STYLES[style]
    if use_mathbackend:   # it reads mathexpr's expression too, and a longer one ahead of it: `\sqrt{2} \cdot 4 + 1` is not 5
        from tools.igc.unified_detect import expression_before_with_mathbackend
        found = expression_before_with_mathbackend(before)
    else:
        found = expression_before(before)
    if found is None:
        return (_tuple_found(before) or _pm_found(before)) if use_mathbackend else None
    expr, value = found
    said = spelled(value)
    if whole:
        return expr, said
    from fractions import Fraction as _F
    general = style == 'general' or INLINE_STYLES[style].get('base') == 'general'
    quotient = isinstance(value, _F) and value.denominator != 1 and simple_quotient(expr) and not re.search(r'\\d?frac', expr)
    if general and use_mathbackend and quotient:
        # the General Style: every uneven integer quotient gets a value true however `=` is read (#307), never a refusal by a habit
        said = _general_quotient(expr, value, before)
        if said is None:
            return None
    elif isinstance(value, _F) and value.denominator != 1 and value > 0 and simple_quotient(expr) and _lowest(expr) \
            and _REMAINDER_WORK.search('\n'.join(before.split('\n')[-2:])):
        # a long division wants its quotient: `13 \div 2 = 6` remainder 1. Where a remainder is being worked out the model wrote the
        # whole quotient 156 times in 165 (held-out) and 281 in 304 (seven 4B arms, #306). On this line or the one before only, and in
        # lowest terms: a divisibility check far from it writes `174/4 = 43.5`, and `10044 / 42 = 5022 / 21` reduces
        q, r = divmod(value.numerator, value.denominator)
        if st['remainder'] == 'quotient':
            return expr, str(q)
        latex = '\\' in before.rsplit('\n', 1)[-1] or '$' in before.rsplit('\n', 1)[-1]
        return expr, f'{q} \\text{{ rem }} {r}' if latex else f'{q} rem {r}'   # true however the `=` is read
    elif '/' in said and simple_quotient(expr) and not (use_mathbackend and _reduces(expr, before)):
        # an integer quotient that does not come out: the model is doing long division and wants a digit, not 21/19 (dm div_big). One
        # that reduces is a step of its own: `10044/42 = 1674/7`, `6/26 = 3/13` (ctrl traces of a100_262k)
        decimal = _quotient_as_decimal(expr, value, before, st['decimal']) if use_mathbackend else None
        if decimal is None:
            return None
        said = decimal
    if _decomposes(expr, value, before):
        return None
    residue = _mod_residue(value, before, st['mod'])
    if residue == '':
        return None   # `$2^{10} \equiv 1024 + 1 =`: a congruence with no modulus named, its residue cannot be told
    if residue is not None:
        return expr, residue
    if _bare(said) == _bare(expr):
        return None   # nothing new, and `= 15/13 = 15/13 = ...` looped until stuck; `9√2 -3 =` is `9*sqrt(2) - 3` (#306, R1)
    if re.fullmatch(r'\s*\(?\s*-?\d[\d,]*(?:\.\d+)?\s*\)?\s*', expr):
        return None   # a number alone: `m=4,512=8*64` is m=4 and 512, not 4512 (#306, R1)
    if _operand_of_more(expr, before):
        return None
    if _meant_otherwise(expr, value, before, general=general) or any(op in expr for op in custom_ops):
        # `5 * (3 * 1)` where the question defined `a * b = a - b`: no product (#306, held-out)
        return None
    return expr, said


_PM = re.compile(r'±|\\pm(?![A-Za-z])')


def _pm_found(before):
    """(expression, spelled value) for one `±` worked out as its two values a, b and written back as P ± Q (P = (a+b)/2, Q = (a-b)/2), the
    form the model keeps: `[6 ± sqrt(32)] / 2 = 3 ± 2√2`, `\\frac{16 \\pm \\sqrt{256 - 240}}{2} = 8 \\pm 2` (#306 held-out: 417 `=` after
    a `±`; the model wrote the simplified `±` form next, then the two values)."""
    from fractions import Fraction as _F
    from tools.igc import exactexpr
    from tools.igc.unified_detect import evaluate_exact
    line = before.rsplit('\n', 1)[-1].rstrip()
    s = line.rstrip(' $')
    cut = max([m.end() for m in re.finditer(r'(?<![<>!\\])=|\$|\\\(|\\\[|:\s|⇒|\\implies|\\Rightarrow', s)] + [0])
    expr = s[cut:].strip()
    if len(_PM.findall(expr)) != 1 or not re.search(r'\d', expr):
        return None
    if re.fullmatch(r'(?:±|\\pm)\s*-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?', expr):
        return None   # `g=±1/2`: a value already
    vals = []
    for sign in '+-':
        v = evaluate_exact(_PM.sub(sign, expr), before)
        if v is None:
            return None
        vals.append(v if isinstance(v, exactexpr.Value) else exactexpr.Value.of(v) if isinstance(v, _F) else None)
    if None in vals:
        return None
    a, b = vals
    p, q = (a + b) * exactexpr.Value.of(_F(1, 2)), (a - b) * exactexpr.Value.of(_F(1, 2))
    try:
        if q.sign() < 0:
            q = -q
    except exactexpr.NotExact:
        return None
    if q == exactexpr.ZERO:
        return None
    latex = '\\' in line or '$' in line
    pm = ' \\pm ' if latex else ' ± '
    d = 1
    for c in list(p.t.values()) + list(q.t.values()):
        d = d * c.denominator // math.gcd(d, c.denominator)
    pn, qn = p * exactexpr.Value.of(d), q * exactexpr.Value.of(d)

    def sp(v):
        r = v.rational()
        t = spelled(r) if r is not None else exactexpr.spell(v, latex)
        return re.sub(r'\*?sqrt\((\d+)\)', r'√\1', t) if '√' in expr else t
    if p == exactexpr.ZERO:
        said = pm.strip() + ' ' + sp(q)   # `\pm \frac{5\sqrt{69}}{12}`
    else:
        body = sp(pn) + pm + sp(qn)
        said = body if d == 1 else (f'\\frac{{{body}}}{{{d}}}' if latex else f'({body})/{d}')
    return None if _bare(said) == _bare(expr) else (expr, said)


_DIVISION_WORK = re.compile(r'\brem\b|remainder|bring(?:s|ing)? down|long division|\bquotient\b|divisib|divides|divisor', re.I)


def _general_quotient(expr, value, before):
    """An uneven integer quotient in the General Style, the simplest value true however `=` is read (#307):
    - where a division is being worked as one (a remainder, a quotient, bringing down, a divisibility test, a `÷`), and the quotient is at
      least 1: `q rem r`, the quotient and what is left (`1302 / 9 = 144 rem 6`, `68 \\div 58 = 1 rem 10`);
    - one that reduces, or one whose signs simplify: that fraction (`10044/42 = 1674/7`, `-4 / -7 = 4/7`, `4 / (-3) = -4/3`);
    - one in lowest terms: its decimal, exact if it terminates (`174/4 = 43.5`), else cut to 4 places and marked (`21/19 = 1.1052...`).
    A proper fraction never gets `0 rem a`. Mod work (a quotient is an inverse there) is refused later, by `_meant_otherwise`."""
    nums = [int(x) for x in re.findall(r'\d+', expr)]
    if len(nums) != 2:
        return spelled(value)   # `99(1/18)` = 11/2: more than a quotient of two, its exact value
    a, b = nums
    line = before.rsplit('\n', 1)[-1]
    latex = '\\' in line or '$' in line
    near ='\n'.join(before.split('\n')[-2:])[-600:]   # this line and the one before: where the division is worked (#307)
    division = _DIVISION_WORK.search(near) or re.search(r'\\div|÷', expr) \
        or re.search(rf'(?<!\d){a}\d+\s*(?:/|\\div|÷)\s*{b}(?!\d)', before[-400:])   # the first piece of a long division of a{digits} by b
    worked = re.search(r'\\div|÷', expr) or re.search(r'bring(?:s|ing)? down|long division', near, re.I)
    # a long division's step may be under 1: `1 \div 2 = 0 rem 1` after bringing down. Never with a sign: the digits read here have none,
    # and `-1 \div 6 = 0 rem 1`, `20 \div (-8) = 2 rem 4` were false (#307 held-out audit)
    if division and value > 0 and not re.search(r'[-−]', expr) and (value > 1 or worked):
        q, r = divmod(a, b)   # the division as written: `10044 / 42 = 239 rem 6`
        return f'{q} \\text{{ rem }} {r}' if latex else f'{q} rem {r}'
    exact = spelled(value)
    if math.gcd(a, b) > 1 or re.search(r'/\s*\(?\s*[-−]', expr):
        return exact   # it reduces, or its signs do: the fraction is the computation
    if '/' not in exact:
        return exact
    q = abs(value.numerator) * 10 ** 4 // value.denominator
    whole, frac = divmod(q, 10 ** 4)
    return f"{'-' if value < 0 else ''}{whole}.{frac:04d}..."


def _lowest(expr):
    """The integer quotient `a / b` (or `a \\div b`) is written in lowest terms."""
    nums = [int(x) for x in re.findall(r'\d+', expr)]
    return len(nums) == 2 and math.gcd(*nums) == 1


def _tuple_found(before):
    """(expression, spelled value) for a point or vector worked out component by component, or None: `(5 -35/13, 12 -84/13) =` is
    `(30/13, 72/13)`, `M - A = (0, 1 - (-1.5), 0 - (-0.5)) =` is `(0, 2.5, 0.5)` (#306 held-out: 359 such `=` read nothing). Not after a
    letter: `g(0, 1/2)` is a function's value and `Q(2, 3)` names a point."""
    from fractions import Fraction as _F
    from tools.igc import exactexpr
    from tools.igc.unified_detect import evaluate_exact
    line = before.rsplit('\n', 1)[-1].rstrip()
    s = line.rstrip(' $')
    close = '\\right)' if s.endswith('\\right)') else ')' if s.endswith(')') else None
    if close is None:
        return None
    depth, j = 0, len(s) - len(close)
    open_at = None
    while j > 0:
        j -= 1
        ch = s[j]
        if ch in ')]}':
            depth += 1
        elif ch in '([{':
            if depth == 0:
                open_at = j if ch == '(' else None
                break
            depth -= 1
    if open_at is None:
        return None
    start = open_at - 5 if close == '\\right)' and s[:open_at].endswith('\\left') else open_at
    head = s[:start]
    if re.search(r'[A-Za-z_\d]$|\\[A-Za-z]+$', head) and not re.search(r'\\(?:cdot|times|quad|left)$', head):
        return None   # a call or a point's name, glued to its bracket: `g(0, 1/2)`, `Q(2, 3)`; `= (` or `$(` is a vector
    if re.search(r'(?<![A-Za-z])[A-Z]\s+$', head):
        return None   # `Q (2, 3)`: a point's name and a space
    inner, parts, depth, cur = s[open_at + 1:len(s) - len(close)], [], 0, ''
    for ch in inner:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    parts.append(cur)
    if len(parts) < 2 or len(parts) > 6 or any(not p.strip() for p in parts):
        return None
    if all(re.fullmatch(r'\s*-?\d+(?:\.\d+)?\s*', p) for p in parts):
        return None   # a point already: nothing to work out
    latex = '\\' in line or '$' in line
    said = []
    for p in parts:
        v = evaluate_exact(p, before)
        if v is None:
            return None
        if isinstance(v, exactexpr.Value):
            q = v.rational()
            said.append(spelled(q) if q is not None else exactexpr.spell(v, latex))
        elif isinstance(v, _F):
            said.append(spelled(v))
        else:
            return None
    expr = s[start:]
    out = '(' + ', '.join(said) + ')'
    return None if _bare(out) == _bare(expr) else (expr, out)


def _bare(s):
    """A spelling with its spacing, brackets, braces, `*` and `\\` taken out and `√` written `sqrt`: two spellings of one written form."""
    return re.sub(r'[\s(){}*\\]', '', s.replace('√', 'sqrt').replace('\\sqrt', 'sqrt'))


# words that make what follows the argument of a function or an exponent, whose own value is not the step: `cube root of 8^26 =
# 8^(26/3)`, `√2 raised to (3*0 -1) = √2^(-1)` (R1 traces, #306 held-out). Not `multiply by`, `divide by`, `times`, `over`: there the
# model names the step and then works out its operand, `Divide by $2-1 = 1$`, `multiply by $7 \times 2 = 14$` (60 such puts agreed)
_OPERAND_WORDS = re.compile(
    r'(?:\b(?:\w+th|square|cube)\s+root|\broot|\blog(?:arithm)?|\bln|\bsine|\bcosine|\btangent|\bfloor|\bceiling|\babsolute\s+value'
    r'|\bfactorial|\breciprocal|\binverse|\bhalf|\bthird|\bquarter|\bdouble|\bnegative)\s+of(?:\s+the)?'
    r'|\braised\s+to(?:\s+the(?:\s+power(?:\s+of)?)?)?|\bto\s+the\s+power(?:\s+of)?', re.I)


def _operand_of_more(expr, before):
    """The expression is a piece of a longer one the text began before it, so its own value is not what `=` gives:
    - a word that makes it an argument or an exponent: `cube root of 8^26 =`, `raised to (3*0 -1) =`;
    - a bracket that counts what a word names: `14 boys (10+4) +18 girls (12+6)=32` sums the counts, not 12+6;
    - a number, its sentence's period and the next sentence: `A=24:576.5*7*6=210` is 24^2 = 576 and its digits' product."""
    line = before.rsplit('\n', 1)[-1].rstrip()
    e = expr.strip()
    at = line.rfind(e)
    if at < 0:
        return False
    head = re.sub(r'[\s$]+$|\\\($|\\\[$', '', line[:at]).rstrip()
    if re.search(rf'(?:{_OPERAND_WORDS.pattern})$', head, re.I):
        return True
    count = re.search(r'(?<![\w.])\d+\s+([A-Za-z]{3,})$', head)
    if count and re.fullmatch(r'\([^()]*\)', e) and count.group(1).lower() not in _LINK_WORDS:
        return True   # not `z1 z2 approaches (-0.5)(-1)`, `61 or (26)/2`, `6^9 is (2*3)^9`
    return _glued_period(e)


# words between a number and a bracket that link a statement rather than name what the number counts
_LINK_WORDS = {'and', 'are', 'was', 'were', 'the', 'then', 'becomes', 'gives', 'equals', 'from', 'than', 'with', 'into', 'times',
               'plus', 'minus', 'over', 'divided', 'multiplied', 'approaches', 'yields', 'makes', 'leaves', 'means'}


def _glued_period(e):
    """`576.5*7*6`: a number, its sentence's period, and its digits' product."""
    glued = re.match(r'(\d{2,})\.(\d)((?:\s*[*×x+]\s*\d(?!\d))+)', e)
    return bool(glued) and list(glued.group(1)) == [glued.group(2)] + re.findall(r'\d', glued.group(3))


_REMAINDER_WORK = re.compile(r'\brem\b|remainder|bring(?:s|ing)? down|long division', re.I)
# the modulus a line works in: `≡ 7 mod 25`, `\pmod{29}`, `(mod 11)`, `mod25`
_LINE_MODULUS = re.compile(r'(?:\\pmod\s*\{?\s*|\\bmod\s*\{?\s*|\(\s*mod\s*|\bmod(?:ulo)?\s*)\$?(\d+)')


def _decomposes(expr, value, before):
    """The expression rewrites the number its chain opened with: `601 = 560 + 41 =`, `225 = 15^2 = (3 \\times 5)^2 =`. Its value is that
    number, already written; the model is taking it apart (for a remainder, a factorisation), not computing (#306: 45 such `=` in the
    mod-work refusals, the model wrote the next piece of the decomposition in every one)."""
    line = before.rsplit('\n', 1)[-1]
    at = line.rfind(expr.strip())
    if at < 0:
        return False
    head = line[:at]
    cut = max(head.rfind(c) for c in '$:;(') if not head.rstrip().endswith('(') else -1
    parts = head[cut + 1:].split('=')
    if len(parts) < 2:
        return False
    m = re.fullmatch(r'\s*(-?\d[\d,]*(?:\.\d+)?)\s*', parts[0])
    from fractions import Fraction as _F
    return bool(m) and isinstance(value, _F) and _F(m.group(1).replace(',', '')) == value


def _mod_residue(value, before, how='both'):
    """On a line that works mod m, an integer outside 0..m-1. `both` (the general style) writes the value and its residue, true however
    the model's `=` is read: `7^4*7 = 16807 \\equiv 7`, `4 \\times 2 = 8 \\equiv 3`. `habit` (a model's profile) writes the residue alone
    where the model's latest `=` near there already meant `≡` (`7^4=(7^2)^2=(-1)^2=1 mod25`, R1 and QwQ), else nothing here, so the
    value goes in (`6*24=144≡144-4*29`, 4B)."""
    from fractions import Fraction as _F
    if not isinstance(value, _F) or value.denominator != 1:
        return None
    lines = before.split('\n')
    line = lines[-1][-500:]
    near = '\n'.join(lines[-2:])[-700:]   # `Work $\pmod{13}$.` then `$(2^4)^2 = 3^2$` on the next line
    mods = [int(x) for x in _LINE_MODULUS.findall(near)]
    if re.search(r'\bparity\b', line, re.I):
        mods.append(2)   # `The parity is $1 + 1 = 0$` (contest-ab7dcc84)
    if not re.search(r'≡|\\equiv|\\pmod|\bmod\b|mod\s*\d|\bparity\b', line, re.I):
        return None   # the line itself works mod m: `Number of such u is 81 - 27 = 54` after a line mod 27 counts, it does not reduce
    if not mods:
        return '' if re.search(r'≡|\\equiv', line) else None
    m = mods[-1]
    if m < 2 or 0 <= value < m:
        return None
    # the residue only where the model has already written `=` for `≡` near here: `7^4=(7^2)^2=(-1)^2=1 mod25`. Elsewhere it writes the
    # full value and then reduces it, `$4 \times 2 = 8 \equiv 3 \pmod 5$`, `6*24=144≡144-4*29`, and a residue there is a false equation
    # (seven 4B arms: 4 \times 2 = 3 was put 9 times on such lines, #306)
    if how == 'both':
        eq = ' \\equiv ' if '\\' in line or '$' in line else ' ≡ '
        return f'{value}{eq}{value % m}'
    if re.search(r'\bparity\b', line, re.I):
        return str(value % 2)   # the line says what its `=` means
    return str(value % m) if _eq_means_congruence(near, m) else None


def _eq_means_congruence(text, m):
    """Whether the model's latest `=` in `text` that tells links an expression to its residue mod m rather than to its full value past m."""
    from tools.igc.mathexpr import evaluate, expression_before
    last = None
    for e in list(re.finditer(r'(?<![<>!=\\])=(?!=)', text))[-10:]:
        got = expression_before(text[:e.start()])
        # what the model set it equal to: a number or an expression, up to the next relation or the end of the statement
        right = re.split(r'(?<![<>!=\\])=|≡|\\equiv|\\pmod|\bmod\b|mod\s*\d|\$|,\s|;|\.\s|\.$|\n|\\quad|\\text|\?', text[e.end():], 1)[0]
        w = evaluate(right.strip()) if right.strip() else None
        if not got or w is None or not hasattr(got[1], 'denominator') or got[1].denominator != 1 \
                or not hasattr(w, 'denominator') or w.denominator != 1:
            continue
        v, w = got[1].numerator, w.numerator
        if w == v and not 0 <= v < m and re.fullmatch(r'\s*-?\d+\s*', right):
            last = 'full'      # `7^2=49`, then `≡ -1`: the full value written out (a rewrite `7^5=7^4*7` says nothing)
        elif w != v and (v - w) % m == 0:
            last = 'residue'   # `(7^2)^2=(-1)^2`, `23^4=4^2`
    # the latest use decides: R1 writes a small power out (`7^2=49≡-1`) and then works in residues
    return last == 'residue'


_LONG_DIVISION = re.compile(r'\brem\b|remainder|bring(?:s|ing)? down|long division|\bquotient\b|\bcarry\b|\bborrow|\bdigit|\bcolumn', re.I)
_MOD_WORK = re.compile(r'\\equiv|≡|\\pmod|\\bmod|\bmod(?:ulo)?\b', re.I)
_CUT_REPEATING = re.compile(r'(?<![\d.])\d+\.\d*?(?:(\d)\1{2,}|33|66)(?!\d)')
_COLUMN = re.compile(r'\d{3,}\s*[-+−]\s*\d{3,}\s*\$?\s*[:.]?\s*\n')   # a written sum or difference set out on a line of its own
_PLACE = re.compile(r'\b(?:units|ones|tens|hundreds|thousands)\s*(?:digit|place|column)?\s*:', re.I)
_COINS = re.compile(r'\b(?:coins?|notes?|bills?|stamps?|cents?)\b', re.I)
_AS_COUNT = re.compile(r'\s*\d+\s*(?:\\times|\\cdot|\*|×)\s*\d+(?:\s*\+\s*\d+(?:\s*(?:\\times|\\cdot|\*|×)\s*\d+)?)*\s*')


def _meant_otherwise(expr, value, before, general=False):
    """The value is right for the text but not for what the model is doing there (#306, the inline audit over seven stored control arms):
    - a long division's step wants its digit: `1 \\div 2 = 0` (rem 1), `5 \\div 2 = 2` with a remainder of 1, not 0.5 or 2.5;
    - a column of a subtraction borrows: `0-9 = 1` (borrow from 7), not -9;
    - in work mod m a quotient is a product by an inverse: `3/2 = 3 \\cdot 3 = 9 \\equiv 4`, not 1.5;
    - a cut repeating decimal means its fraction: `(0-1)/(3-1.66) = -1 / (4/3)`, not -50/67."""
    from fractions import Fraction as _F
    from tools.igc.mathexpr import simple_quotient, moduli_in
    q = value if isinstance(value, _F) else None
    recent = before[-600:]
    line = before.rsplit('\n', 1)[-1]
    if q is not None and q.denominator != 1 and simple_quotient(expr) and _LONG_DIVISION.search(recent) and not general:
        return True   # the General Style writes `q rem r` there instead (_general_quotient)
    # one column of a written sum or difference: `$106029 - 76230$.` on a line above, then `$5-9=6$ (borrow)`, `$12 - 2 = 0$`. A column
    # gives a digit; a value under 0 or past 9 is a borrow or a carry the model is about to write
    column = _LONG_DIVISION.search(recent) or _COLUMN.search(before[-400:])
    # `0-1-8` takes the borrow from the column before it (`10^-6: $0-1-8 = 9-8 = 1$`, held-out a100r4 seal1)
    # a carry is written after its sum (`7+9=16`, then `write 6 carry 1`: the model wrote the sum in 1,520 of 1,540 such `=`, held-out);
    # a borrow is not: `0-9 = 1`
    if q is not None and column and re.fullmatch(r'\s*\d{1,2}(?:\s*[-−+]\s*\d{1,2}){1,2}\s*', expr) and q < 0:
        return True
    if _PLACE.search(line) and not general:
        return True   # `Tens: $3 + 8 + 4 + 6 + 2 + 3 + 2 = 30$`: a column of a sum, its carry added in the head. The General Style
        # writes the sum of what is written (28), true as written; the model adds its carry to it
    if (q is None or q.denominator != 1) and _MOD_WORK.search('\n'.join(before.split('\n')[-2:])):
        return True   # `$3/2 = 3 \cdot 3$` mod 7: a quotient is a product by an inverse there
    # a mod word further back no longer refuses: past it the model wrote the plain value in 1,531 of 1,982 `=` (held-out) and 814 of
    # 1,152 (seven 4B arms); on the line itself `_mod_residue` gives the residue where the model's `=` means `≡`
    if _COINS.search(recent) and _AS_COUNT.fullmatch(expr.strip(' $')):
        return True   # `$f(25) = 2 \times 10 + 5 \times 1 = 7$` coins: a count, not the value 25
    if re.fullmatch(r'[^()]*[+\-−][^()]*\s/\s*\d+\s*', expr):
        return True   # `400 - 1500 + 1080 / 27 = -120/27`: the numerators over 27, the brackets left out
    return _CUT_REPEATING.search(expr) is not None


def _reduces(expr, before=''):
    """An integer over an integer with a common factor: its lowest terms are news. Not in a long division or a divisibility test, which
    want a digit or a decimal: `$70 / 6 = 11$ rem 4` after `Let's check $70308 / 6$`, `1302 / 9 = 144.66` after `divisible by` (ctrl traces
    of a100_262k: 50 reductions the model wrote, 8 places it wanted the quotient). Not with `\\div` (`68 \\div 58 = 1`, a long division in
    dm div_big), nor where the recent text works in decimals: over seven control arms the model wrote the fraction 497 times and a
    decimal 197, and with no decimal in the 600 characters before, 382 against 89."""
    if re.search(r'\\div|÷', expr):
        return False
    nums = [int(x) for x in re.findall(r'\d+', expr)]
    if len(nums) != 2 or math.gcd(*nums) == 1:
        return False
    a, b = nums
    recent = before[-400:]
    if re.search(r'\brem\b|remainder|long division|divisib|divides|\bdigit', recent, re.I):
        return False
    # decimals before no longer stop it: there the model still wrote the reduced fraction 159 times and a decimal 105 (#306, seven 4B
    # arms and the held-out traces; `11718/294 = 1674/7`)
    return not re.search(rf'(?<!\d){a}\d+\s*(?:/|\\div|÷)\s*{b}(?!\d)', recent)   # the first piece of a long division of a{digits} by b


def _quotient_as_decimal(expr, value, before, how='cut'):
    """An integer quotient that does not come out, given as a decimal where the model wants one, else None:
    - a divisibility test: `Is 4 a divisor of 174? $174/4 = 43.5$`, `1302 / 9 = 144.66` after `divisible by`;
    - a quotient in lowest terms where the text before works in decimals: the model wrote a decimal 277 times and the fraction 59
      (#306, both sets). Rounded to 4 places: the model writes 2 to 4 (2 most often), and rounds 133 times to 54 cut.
    Not in a long division (`\\div`, a remainder near: it wants the digit), not a `\\frac`, not the first piece of a long division."""
    if re.search(r'\\div|÷|\\frac|\\dfrac', expr) or value <= 0:
        return None
    nums = [int(x) for x in re.findall(r'\d+', expr)]
    if len(nums) != 2:
        return None
    a, b = nums
    recent = before[-400:]
    if re.search(r'\brem\b|remainder|long division|bring(?:s|ing)? down|\bdigit', recent, re.I):
        return None
    if re.search(rf'(?<!\d){a}\d+\s*(?:/|\\div|÷)\s*{b}(?!\d)', recent):
        return None
    test = re.search(r'divisib|divides|divisor|\bfactor', recent, re.I)
    if not test and (math.gcd(a, b) > 1 or not re.search(r'(?<![\d.])\d+\.\d+', before[-600:])):
        return None
    from fractions import Fraction as _F
    exact = spelled(value)
    if '/' not in exact:
        return exact   # `174/4 = 43.5`: it terminates, the decimal is the value
    if how == 'rounded':   # a model's profile: the form it writes, `1302 / 9 = 144.6667`
        q = round(value * 10 ** 4)
        whole, frac = divmod(abs(q), 10 ** 4)
        return f"{'-' if q < 0 else ''}{whole}.{frac:04d}".rstrip('0').rstrip('.')
    # the general style: cut, and marked as cut, so it is true as written: `1302 / 9 = 144.6666...`
    q = value.numerator * 10 ** 4 // value.denominator
    whole, frac = divmod(q, 10 ** 4)
    return f'{whole}.{frac:04d}...'


def inline_value(before, whole=False, use_mathbackend=False):
    """The spelled exact value of inline_found, or None."""
    found = inline_found(before, whole, use_mathbackend=use_mathbackend)
    return found[1] if found else None


def _box_inner(text, i):
    """The index of the `}` that closes the brace opened just before `i`, or None (an unclosed box)."""
    depth = 1
    while i < len(text):
        depth += {'{': 1, '}': -1}.get(text[i], 0)
        if depth == 0:
            return i
        i += 1
    return None


def _shortening(written, exact):
    """True when `written` (a box's content) is a shortening of the non-integer Fraction `exact`, not the value itself: rounded or cut at
    the places it shows, its integer part, or (with 4 or more significant digits) its first 4 significant digits. A lost sign, another value, or the
    exact value in any spelling is False."""
    from tools.igc.mathexpr import evaluate
    text = written.strip().strip('$').replace(',', '').replace('\\,', '')
    text = re.sub(r'(?:\\ldots|\\dots|\.\.\.|…)$', '', text).strip()
    m = re.fullmatch(r'-?\d+(?:\.(\d+))?', text)
    if not m:
        return False   # a fraction, a root or words: the exact value in another spelling, or another value
    got = Fraction(text)
    if got == exact or (got < 0) != (exact < 0) and got != 0:
        return False
    places = len(m.group(1) or '')
    scale = 10 ** places
    near = {Fraction(math.floor(exact * scale), scale), Fraction(math.ceil(exact * scale), scale)}
    if got in near:   # rounded either way, or cut toward zero: both are the floor or the ceiling at these places
        return True
    # the model's own digits drifting past the ones it got (5070#1: 0.00456226 for 0.0045623759...): its first 4 significant digits are
    # the value's, cut. A relative 0.1 % let -1.9875 for -1.98868 through, wrong in the third digit
    if len(text.lstrip('-').replace('.', '').lstrip('0')) < 4:
        return False
    top = math.floor(math.log10(abs(exact)))
    unit = Fraction(10) ** (top - 3)
    return math.floor(abs(got) / unit) == math.floor(abs(exact) / unit)


def spelled(value):
    """An integer as itself, a value with a terminating decimal as that decimal (the model's own spelling: 9/2 written for 0.9 x 5 broke its
    decimal sum into a loop, dm 6890), and only the rest as a reduced fraction, as Qwenputer's library prints."""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if not hasattr(value, 'denominator'):
        return str(value)
    if value.denominator == 1:
        return str(value.numerator)
    d = value.denominator
    twos = fives = 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d != 1:
        return f'{value.numerator}/{value.denominator}'
    places = max(twos, fives)
    digits = str(abs(value.numerator) * 10 ** places // value.denominator).rjust(places + 1, '0')
    return ('-' if value < 0 else '') + digits[:-places] + '.' + digits[-places:]


class Rollout:
    """One answer, a state machine the driver asks for its next request and hands each result to. Phases: think, seek (a line end for the
    forced open), peek (whether a bare fence opens python), block, answer."""
    def __init__(self, row, sample, prompt, arm, ids, tokenizer, seed, thinking=True):
        self.row, self.sample, self.prompt, self.arm, self.ids, self.tok, self.seed = row, sample, prompt, arm, ids, tokenizer, seed
        self.out, self.spans, self.events = [], [], []
        self.phase, self.thought, self.forced, self.block, self.runs, self.seen = 'think', 0, False, 0, 0, {}
        # --no-think: the prompt already ends in an empty thinking (enable_thinking=False), so the reply is the answer from its first token
        self.thinking = thinking
        if not thinking:
            self.phase = 'answer'
        self.whole_pending = False   # the next value is the whole step the --whole-hint asked for
        self.moduli = None   # --check-hand: the moduli the question and the answer so far name
        self._custom, self._custom_at = None, 0   # operators the question or the answer defines (`a * b = a - b`), read as the text grows
        self.given = []      # --hand-off: the whole values software put in (Fractions)
        if arm.whole_hint:
            self.inject(WHOLE_HINT_PLAIN if arm.whole_plain else WHOLE_HINT, 'whole hint')
            self.whole_pending = True
        self.code_at = self.fence_at = None
        self.by = None
        self.pending = None
        self.finish = None
        self.kernel = None
        self.started = time.monotonic()
        self.seconds = None
        self.cap = 0   # --max-tokens: training data only; an answer past it is never the shortest right one kept
        self.checked = 0   # len(out) at the last loop check
        self.last_hint = -HINT_GAP   # self.thought at the last hint
        self.loop_hints = 0
        self.plan = None     # --logit-value: the token ids of the value being written, one biased request each
        self.plan_at, self.plan_own, self.plan_event, self.last_lp = 0, [], None, None
        self.window = 0   # --suppress-recheck: steps left in which the re-check words are held down
        self.join_guard = False   # --no-digit-join: the next request is one token, digits held down
        self.joins_held = 0
        self.early_cycles = 0   # --early-cycle: loops its check found (the bench's check may have found some of them later)
        self.early_dropped = 0   # tokens taken and then dropped after the line end where it found one
        self.shape_loops = 0   # --shape-cap: loops found by the collapsed-number cycle and by no other check
        self.cover_loops = 0   # --cover-cap: loops found by the run of already-seen grams and by no other check
        self.cover = Cover(ids.digits, arm.cover_cap) if arm.cover_cap else None
        self.echo = (None, 0, 0)   # (expression, its run of back-to-back values, model_tokens at the last): see echoing()
        self.loop_floor = 0   # the loop checks of the thinking read from here: a nudge starts them afresh (#304)
        self.loop_nudges = 0
        self.answer_closed_at = None   # --answer-close: where the model's value after the cue begins
        self.loop_valued = False   # --loop-value: put once per answer
        self._question = ()   # question_read's (expression, value) of this row, read once; () = not read yet
        self.echo_skips = 0   # values echoing() left to the model
        if arm.candidate_close:
            from tools.igc.unified_detect import CandidateAnswerTracker
            self.tracker = CandidateAnswerTracker()
        else:
            self.tracker = None

    def bias(self):
        """{token id: logit added} for the request now asked, or None. --logit-value: software's next token, the model writes the value itself.
        --suppress-recheck: the re-check words, in the window after a value."""
        if self.plan is not None:
            return {self.plan[self.plan_at]: self.arm.logit_value}
        bias = {t: SUPPRESS_BIAS for t in self.ids.recheck} if self.window > 0 else {}
        if self.join_guard:
            bias.update({t: JOIN_BIAS for t in self.ids.digit_start})
            if self.join_guard != 'point':   # right after the put; after its `.` only a digit would make it another number
                bias.update({t: JOIN_BIAS for t in self.ids.slash_start})
        return bias or None

    def open_window(self):
        self.window = self.arm.suppress_recheck

    def start_plan(self, expr, value):
        self.plan = self.tok.encode(f' {value}', add_special_tokens=False)
        self.plan_at, self.plan_own = 0, []
        self.spans.append([len(self.out), len(self.out), 'logit value'])
        self.plan_event = dict(kind='logit value', at=len(self.out), model_tokens=self.model_tokens(), expr=expr[:200], value=value,
                               bias=self.arm.logit_value, tokens=len(self.plan))
        self.events.append(self.plan_event)

    def follow_plan(self, token):
        """True when the model took the biased token (the value goes on), False when its own prediction won against the bias."""
        want = self.plan[self.plan_at]
        lp = self.last_lp.get(want) if self.last_lp else None
        self.plan_own.append(None if lp is None else round(float(getattr(lp, 'logprob', lp)), 3))   # raw: the model's own log P of the wanted token, None past the top 20
        if token == want:
            self.plan_at += 1
            self.spans[-1][1] = len(self.out)
            if self.plan_at >= len(self.plan):
                self.end_plan()
            return True
        self.end_plan()
        return False

    def end_plan(self):
        self.plan_event.update(matched=self.plan_at, own_logprobs=self.plan_own, ok=self.plan_at >= len(self.plan))
        self.plan = None
        if self.plan_event['ok']:
            self.open_window()

    def left(self):
        room = CONTEXT - len(self.prompt) - len(self.out) - 64
        return min(room, self.cap - len(self.out)) if self.cap else room

    def model_tokens(self):
        return len(self.out) - sum(e - s for s, e, _ in self.spans)

    def text(self):
        return self.tok.decode(self.out)

    def inject(self, text, kind):
        t = self.tok.encode(text, add_special_tokens=False)
        self.spans.append([len(self.out), len(self.out) + len(t), kind])
        self.out += t
        self.events.append(dict(kind=kind, at=len(self.out) - len(t), model_tokens=self.model_tokens()))

    def done(self, finish):
        self.finish = finish
        self.seconds = round(time.monotonic() - self.started, 3)
        if self.kernel:
            self.kernel.close()
            self.kernel = None

    def request(self):
        """(max_tokens, stop token ids) of the next generation, or None once it is done. In the --suppress-recheck window the request is no
        longer than what is left of it, so one static logit bias covers it."""
        q = self._request()
        if q and self.join_guard:
            return 1, q[1]   # one token carries the digit bias, then the request goes on as before
        if q and self.window > 0 and self.plan is None:
            return min(q[0], self.window), q[1]
        return q

    def _request(self):
        if self.finish:
            return None
        left = self.left()
        if left <= 0:
            self.done('length')
            return None
        stops = set(self.ids.ends)
        if self.plan is not None:
            return 1, stops   # --logit-value: one token a request, so each carries its own bias
        if (self.arm.inline_calc or self.arm.logit_value) and self.phase in ('think', 'answer') or self.arm.hint_calc and self.phase == 'think':
            stops |= self.ids.equals
        if self.arm.native_tool and self.phase in ('think', 'answer'):
            stops.add(self.ids.tool_close)
        if self.arm.check_hand and self.phase in ('think', 'answer'):
            stops |= self.ids.lines   # every line is checked as it ends, so a correction follows the line it corrects
        if self.arm.candidate_close and self.phase == 'think':
            stops |= self.ids.lines   # candidate answers and reflection starters are checked at line ends
        if self.whole_pending:
            stops |= self.ids.lines   # the whole step may come as a line of its own, with no `=`
        if self.arm.hand_off and self.given:
            stops |= self.ids.lines   # each line after a given value is read for a restatement
        if self.phase == 'answer' and self.answer_closed_at is not None:
            return min(left, ANSWER_CLOSE_TOKENS - (len(self.out) - self.answer_closed_at)), stops | self.ids.lines
        if self.phase == 'answer':
            # a decimal expansion has no line end: after a given value, short requests so it is seen while it starts
            return min(left, 48 if self.arm.hand_off and self.given else CHUNK), stops
        stops.add(self.ids.think_close)
        if self.phase == 'peek':
            return min(2, left), stops
        if self.phase == 'seek':
            return min(64, left), stops | self.ids.lines
        if self.phase == 'block':
            return min(MAX_BLOCK - self.block, left), stops | self.ids.fence
        most = min(left, CHUNK)   # every arm is read for a loop between chunks: the stuck stop of the bench, or the close
        if self.arm.execute:
            stops |= self.ids.fence
        if self.arm.force_after and not self.forced and self.thought < self.arm.force_after:
            most = min(most, self.arm.force_after - self.thought)
        return most, stops

    def take(self, new):
        """The tokens of one generation. True when a block waits to run."""
        self.out += new
        self.window = max(0, self.window - len(new))
        # a `.` straight after a put ends a sentence (`= 42.`) or starts a decimal (`0` then `.75`): one more token with digits held
        self.join_guard = 'point' if self.join_guard is True and len(new) == 1 and new[0] in self.ids.point else False
        if self.phase != 'answer' and self.phase != 'block':
            self.thought += len(new)
        last = new[-1] if new else None
        if self.early_cut(new):
            if self.left() <= 0:
                self.done('length')
            return False
        if self.answer_closed_at is not None:
            # --answer-close: after the cue only the value is read, nothing else acts (a put, a check), and it ends at a line end or the cap
            wrote = len(self.out) - self.answer_closed_at
            if last in self.ids.ends or last in self.ids.lines and wrote > 1 or wrote >= ANSWER_CLOSE_TOKENS:
                self.done('stop')   # the value after the cue is the answer; the event says the runner asked for it
            elif self.left() <= 0:
                self.done('length')
            return False
        if self.plan is not None and new and self.follow_plan(last) and last not in self.ids.ends:
            if self.left() <= 0:
                self.done('length')
            return False
        if self.arm.check_hand and self.phase in ('think', 'answer') and last in self.ids.lines:
            self.check_line()
        if self.arm.candidate_close and self.phase == 'think' and last in self.ids.lines and self.check_candidate_line():
            return False
        if self.whole_pending and self.arm.inline_calc and last in self.ids.lines:
            self.whole_line()
        if self.arm.hand_off and self.given and self.restated(last in self.ids.lines):
            return False
        if self.arm.hint_calc and last in self.ids.equals and self.phase == 'think' and self.hint_due():
            return False
        if (self.arm.inline_calc or self.arm.hint_calc or self.arm.logit_value) and last in self.ids.equals and self.phase in ('think', 'answer'):
            text = self.text()
            whole = self.whole_pending
            found = inline_found(text[:text.rstrip().rfind('=')], whole=whole, use_mathbackend=self.arm.mathbackend,
                                 custom_ops=self.custom_ops(text), style=self.arm.inline_style) if self.arm.inline_calc or self.arm.logit_value else None
            if found is not None and self.arm.logit_value and not self.arm.inline_calc:
                self.start_plan(*found)   # the value is not put in: the model writes it, its prediction of each token raised
            elif found is not None:   # the value the model would have computed, computed: the reasoning goes on from it
                expr, value = found
                if not self.echoing(re.sub(r'[\s()]', '', expr)):
                    if whole:
                        value = self.question_value(value)
                    self.whole_pending = False
                    # the whole value ends its line: left open, an integer drew `.000...` from the model until the guard stopped it (pc4-whole dm div_big 7815)
                    self.inject((f' {value}' if self.arm.inline_plain else f' {value} (exact)') + ('\n' if whole else ''), 'inline value')
                    self.open_window()
                    if self.arm.no_digit_join and self.arm.inline_plain and not whole and value[-1:].isdigit():
                        self.join_guard = True   # `-5` then the model's own `75` read `-575` on the page (#307, 80 of 19,990 puts)
                        self.joins_held += 1
                    self.events[-1].update(expr=expr[:200], value=value, whole=whole)
                    if whole:
                        self.give(value)
            if self.left() <= 0:
                self.done('length')
            elif len(self.out) - self.checked >= CHUNK:   # a request that ends at `=` skips the checks below: a loop of equations would run on
                self.checked = len(self.out)
                if self.looping_now():
                    self.on_loop()
            return False
        self.checked = len(self.out)
        if self.arm.native_tool and last == self.ids.tool_close and self.phase in ('think', 'answer'):
            text = self.text()
            m = _CODE.search(text[text.rfind('<tool_call>'):])
            self.by = 'tool' if self.phase == 'answer' else 'tool in thinking'
            self.events.append(dict(kind='tool call', by=self.by, at=len(self.out), model_tokens=self.model_tokens()))
            self.pending = m.group(1) if m and m.group(1).strip() else None
            if self.pending is None:
                self.tool_reply(NO_CODE)
                return False
            return True
        if last in self.ids.ends:
            if self.phase == 'answer':
                self.exact_box()
                self.done('stop')
            elif self.runs:   # an end of turn inside thinking would leave no answer after a call (:384)
                self.out.pop()
                self.inject(LOOP_CLOSE, 'end of turn in thinking, closed it')
                self.phase = 'answer'
            else:
                self.done('stop')
            return False
        if self.phase == 'answer':
            if self.left() <= 0:
                self.done('length')
            elif self.looping_now():
                self.on_loop()
            return False
        if last == self.ids.think_close:
            if self.phase == 'block':
                self.events.append(dict(kind='thinking closed in a block', at=len(self.out), model_tokens=self.model_tokens()))
            self.phase = 'answer'
            return False
        if self.phase == 'seek':
            if last in self.ids.lines:
                self.forced = True
                self.inject(OPEN_LINE, 'forced open')
                self.enter('forced')
            return False
        if self.phase == 'peek':
            after = self.tok.decode(self.out[self.fence_at:])
            if after.startswith('python'):
                self.code_at = len(self.tok.decode(self.out[:self.fence_at])) + len('python')
                self.by, self.block, self.phase = 'model', 0, 'block'
                self.events.append(dict(kind='model opened a block', at=self.fence_at, model_tokens=self.model_tokens()))
            else:
                self.phase = 'think'
            return False
        if self.phase == 'block':
            self.block += len(new)
            if last in self.ids.fence:
                text = self.text()
                code = text[self.code_at:text.rfind('```')]
                code = code[1:] if code.startswith('\n') else code
                if not code.strip():
                    self.phase = 'think'
                    return False
                self.pending = code
                return True
            if self.block >= MAX_BLOCK:
                self.inject('\n```' + output_block(TOO_LONG), 'block too long')
                self.phase = 'think'
            return False
        # thinking
        if last in self.ids.fence and self.arm.execute:
            if '```python' in self.tok.decode([last]):
                self.code_at = len(self.text())
                self.by, self.block, self.phase = 'model', 0, 'block'
                self.events.append(dict(kind='model opened a block', at=len(self.out) - 1, model_tokens=self.model_tokens()))
            else:
                self.fence_at, self.phase = len(self.out), 'peek'
            return False
        if self.arm.force_after and not self.forced and self.thought >= self.arm.force_after:
            self.phase = 'seek'
            return False
        if self.looping_now():
            self.on_loop()
        return False

    def echoing(self, expr):
        """True when software's value for `expr` would be the ECHO_RUN-th back to back (no other value between, each within ECHO_GAP model
        tokens of the last): `= 15/13 (exact) = 15/13 (exact) = ...`. The value is then left to the model, and whether the thinking loops is
        for the loop checks to say. A count over the whole answer is no loop: `2^2` three times in a derivation is normal, and that count cut
        or nudged the thinking at the third use in 41 of 48 stored inlineclose answers (#301)."""
        last, run, at = self.echo
        now = self.model_tokens()
        run = run + 1 if expr == last and now - at <= ECHO_GAP else 1
        self.echo = (expr, run, now)
        self.echo_skips += run >= ECHO_RUN
        return run >= ECHO_RUN

    def looping_now(self):
        """The loop checks over the thinking since the last nudge (all of it before one): a nudge leaves the lines that looped in the text, and
        line_back counts lines over the whole history, so read from the start every check after a nudge fired again at once (#304).
        In the answer, over the answer alone: an answer restates lines of its thinking, and read with them line_back could end a right
        answer `stuck` (#305)."""
        window = self.out[self.loop_floor:] if self.phase == 'think' else self.out[self.answer_start():] if self.phase == 'answer' else self.out
        if looping(window, self.arm.guard_progress):
            return True
        if self.arm.shape_cap and self.phase in ('think', 'answer') and shape_rounds(window, self.ids.digits) >= self.arm.shape_cap:
            self.shape_loops += 1
            return True
        if self.arm.cover_cap and self.phase in ('think', 'answer'):
            start = self.loop_floor if self.phase == 'think' else self.answer_start()
            if self.cover.feed(self.out, start, self.early_dropped) >= COVER_SHARE:
                self.cover_loops += 1
                self.cover.__init__(self.ids.digits, self.arm.cover_cap)
                return True
        return False

    def early_cut(self, new):
        """--early-cycle, read at every line end of the tokens just taken (thinking only, since the last nudge), as the replay read it: a
        request runs up to CHUNK tokens with no line stop, so the check at its end alone would come up to 1,024 tokens late. On a hit the
        tokens after that line end are dropped (the loop going on: nothing has read them yet) and the loop is handled. True when it was."""
        if not self.arm.early_cycle or self.phase != 'think' or not new:
            return False
        start = len(self.out) - len(new)
        for i in range(max(start, self.loop_floor), len(self.out)):
            if self.out[i] in self.ids.lines and early_cycle(self.out[self.loop_floor:i + 1]):
                dropped = len(self.out) - (i + 1)
                del self.out[i + 1:]
                self.thought -= dropped
                self.early_dropped += dropped   # a replay hands these tokens back to its trace
                self.early_cycles += 1
                self.on_loop()
                return True
        return False

    def answer_start(self):
        """Where the answer begins: after the last `</think>` (the model's or the runner's close), or 0 when there is none."""
        close = self.ids.think_close
        for j in range(len(self.out) - 1, -1, -1):
            if self.out[j] == close:
                return j + 1
        return 0

    def on_loop(self):
        """A loop in the thinking: a program to test the idea (--loop-hint, at most LOOP_HINTS), else the close (--loop-close), else the
        bench's guard in every arm, MIP or not: a loop that cannot leave would run to the end of the context alone (external_bench).
        With --candidate-close and no final answer yet, a loop with an exact cycle is cut back to one round and nudged (at most LOOP_NUDGES);
        any other loop, or one past the nudges, is closed so the model still answers: a nudge that cut 64 tokens of a line_back loop left the
        loop in place and ended the answer `stuck`, with no answer (#304)."""
        if self.phase == 'think' and self.arm.loop_hint and self.loop_hints < LOOP_HINTS:
            floor = self.spans[-1][1] if self.spans else 0
            cut = max(loop_start(self.out) or len(self.out), floor)
            taken = len(self.out) - cut
            del self.out[cut:]
            self.loop_hints += 1
            self.inject(('' if not self.out or self.out[-1] in self.ids.lines else '\n') + LOOP_HINT, 'loop hint')
            self.events[-1].update(cut_tokens=taken)
            self.enter('loop')
        elif self.phase == 'think' and self.arm.loop_close:
            if self.tracker is not None and not self.tracker.has_live_candidate() and self.loop_nudges < LOOP_NUDGES:
                floor = max(self.spans[-1][1] if self.spans else 0, self.loop_floor)
                start = loop_start(self.out[floor:])
                if start is not None:
                    self.loop_nudges += 1
                    taken = len(self.out) - (floor + start)
                    del self.out[floor + start:]
                    self.inject(LOOP_NUDGE, 'loop nudge')
                    self.events[-1].update(cut_tokens=taken, nudge=self.loop_nudges)
                    self.loop_floor = len(self.out)
                    return
            self.put_loop_value()
            self.inject(LOOP_CLOSE, 'loop close')
            self.phase = 'answer'
        elif self.phase == 'answer' and self.arm.answer_close and self.answer_closed_at is None:
            self.close_answer()   # a loop of equations comes here from the `=` path, any other from the answer's own check
        else:
            self.done('stuck')

    def put_loop_value(self):
        """--loop-value: a loop is being closed. When the question states a computation (tools.igc.question_read: gold on 160 of 160
        DeepMind rows), its exact value is put on a line of its own first, once: software computes, the model decides what to answer.
        #307: 5 of 15 loop-lc closes were a doubt the close did not settle; 2 were an endless division of the question's own fraction."""
        if not self.arm.loop_value or self.loop_valued:
            return
        got = self.question_computation()
        if got is None:
            return
        self.loop_valued = True
        text = self.text()
        lead = '' if text.endswith('\n') else '\n'
        self.inject(f'{lead}{LOOP_VALUE.format(expr=got[0], value=spelled(got[1]))}\n', 'loop value')
        self.events[-1].update(expr=got[0][:200], value=spelled(got[1]))

    def close_answer(self):
        """--answer-close: the answer loops after its thinking was closed. An exact cycle is cut back to one round (so the cue does not land
        inside the 300th `918`), then the final-answer cue is put and the model writes one value, at most ANSWER_CLOSE_TOKENS, to a line end."""
        floor = max(self.answer_start(), self.spans[-1][1] if self.spans else 0)
        start = loop_start(self.out[floor:])
        taken = 0
        if start is not None:
            taken = len(self.out) - (floor + start)
            del self.out[floor + start:]
        self.join_guard = False
        self.put_loop_value()
        exact = self.exact_value()
        self.inject(ANSWER_CLOSE, 'answer close')
        self.events[-1].update(cut_tokens=taken)
        self.answer_closed_at = len(self.out)
        if exact is not None:
            # --exact-answer: the question states one computation, so the value after the cue is software's (2783#0: the cue cut `Let's
            # provide the fraction` and the model wrote `{0}`). Spelled with `{` here; the cue itself still ends before it (0e14ca1)
            value = spelled(exact[1])
            self.inject('{' + value + '}$.', 'exact answer')
            self.events[-1].update(via='cue', expr=exact[0][:200], value=value, copied=None)
            self.out.append(self.tok.convert_tokens_to_ids('<|im_end|>'))
            self.done('stop')

    def question_computation(self):
        """question_read's (expression, exact value) for this row's question, or None; read once per answer."""
        if self._question == ():
            self._question = None
            if self.row.get('question'):
                from tools.igc.question_read import read
                self._question = read(self.row['question'])
        return self._question

    def exact_value(self):
        """--exact-answer: the question's computation and its exact value when the question states one, else None."""
        if not self.arm.exact_answer:
            return None
        got = self.question_computation()
        return got if got is not None and isinstance(got[1], Fraction) else None

    def exact_box(self):
        """--exact-answer at the answer's end (round 9, 4B NVFP4 W4A8, llama.cpp): the model wrote `-73388/36903 \\approx -1.9886` and
        boxed `-1.9886` on purpose; told the exact value (loop value, the class-C settle line) it still chose a decimal. When the question
        states one computation whose value is not an integer and the answer's last box holds a shortening of that value (rounded, cut, its
        integer part, or its own digits within 0.1 %), the box gets the exact value. A different value or a lost sign is the model's answer
        and is left alone; an integer value is never rewritten."""
        exact = self.exact_value()
        if exact is None or exact[1].denominator == 1:
            return
        end = self.out.pop() if self.out and self.out[-1] in self.ids.ends else None
        start = max(self.answer_start(), self.spans[-1][1] if self.spans else 0)
        j = len(self.out)
        while j > start and '\\boxed' not in self.tok.decode(self.out[j:]):
            j -= 1
        tail = self.tok.decode(self.out[j:])
        at = tail.rfind('\\boxed{')
        inner = _box_inner(tail, at + len('\\boxed{')) if at >= 0 else None
        if inner is None or not _shortening(tail[at + 7:inner], exact[1]):
            if end is not None:
                self.out.append(end)
            return
        value = spelled(exact[1])
        copied = tail[at + 7:inner]
        del self.out[j:]
        self.inject(tail[:at + 7] + value + tail[inner:], 'exact answer')
        self.events[-1].update(via='box', expr=exact[0][:200], value=value, copied=copied[:120])
        if end is not None:
            self.out.append(end)

    def whole_line(self):
        """The --whole-hint's step written as a line of its own, with no `=` (pc-whole dm mul_div 3350: the whole expression, then pieces):
        its value is put after it, computed whole."""
        from tools.igc.mathexpr import _OPERATOR, evaluate
        line = self.text().rstrip('\n').rsplit('\n', 1)[-1]
        expr = re.sub(r'^\s*(?:[A-Za-z][A-Za-z ]*:\s*)?', '', line).strip().strip('$').replace('\\[', '').replace('\\]', '').strip()
        if not expr or not _OPERATOR.search(expr.lstrip('-−')):
            return
        value = evaluate(expr)
        if value is None:
            return
        self.whole_pending = False
        shown = self.question_value(spelled(value))
        self.inject(f'= {shown}\n', 'inline value')
        self.events[-1].update(expr=expr[:200], value=shown, whole=True)
        self.give(shown)

    def question_value(self, value):
        """--question-value (MIP catch point 1): the whole value is the computation the question states, read by software
        (tools.igc.question_read), when it states one and the model's copy of it gives another value. pc4/pc8 lost 4 DeepMind rows to a
        copied sign or term; question_read gives gold on 160 of 160."""
        if not self.arm.question_value:
            return value
        from tools.igc.mathexpr import evaluate
        got = self.question_computation()
        if got is None or evaluate(value) == got[1]:
            return value
        self.events.append(dict(kind='question value', at=len(self.out), model_tokens=self.model_tokens(), copied=value,
                                expr=got[0][:200], value=spelled(got[1])))
        return spelled(got[1])

    def give(self, value):
        from tools.igc.mathexpr import evaluate
        v = evaluate(value)
        if v is not None:
            self.given.append(v)

    def restated(self, line_ended):
        """--hand-off: the line now being written only restates a value software gave (an ended line that is just that number, or a
        decimal expansion of it 30 places long). It is cut, and HAND_OFF leads the model to its answer. True when it cut."""
        from fractions import Fraction
        floor = self.spans[-1][1] if self.spans else 0   # never take back what the runner put in
        start = len(self.out)
        while start > floor and (start == len(self.out) or self.out[start - 1] not in self.ids.lines):
            start -= 1
        if start <= floor and floor:
            start = floor
        line = self.tok.decode(self.out[start:]).rstrip('\n')
        hit = False
        m = _RESTATE.fullmatch(line) if line_ended else None
        if m:
            try:
                got = Fraction(m.group(1).replace(',', ''))
                hit = any(got == v for v in self.given)
            except (ValueError, ZeroDivisionError):
                hit = False
        if hit and self.phase == 'answer' and start >= floor:
            # the model wrote the value itself: its answer is closed with that value as it wrote it. A lead let the model answer from its
            # own belief instead: pc6/pc7 `= -18`, then 18 (P('-') 0.02-0.22 after the lead, HF 4B, probe/sign_drop.py). Developer, 2026-11-04
            self.inject(f'\\boxed{{{m.group(1)}}}', 'hand-off close')
            self.events[-1].update(cut=line[:120])
            self.out.append(self.tok.convert_tokens_to_ids('<|im_end|>'))
            self.done('stop')
            return True
        if self.phase == 'answer' and start >= floor:
            # a rounded or endless decimal of a fraction software gave (pc8 4104: 1539439/2572, then `= 598.5377138413301`, boxed and
            # scored wrong; 5081 ran an endless one to the guard): closed with software's exact value (developer, 2026-11-04)
            near = _EXPANSION.search(line)
            if not near and m:
                near = m
            if near:
                try:
                    got = Fraction((near.group(1) if near is m else near.group()).replace(',', ''))
                except (ValueError, ZeroDivisionError):
                    got = None
                exact = next((v for v in self.given if v.denominator != 1 and got is not None and got != v
                              and abs(got - v) <= abs(v) * Fraction(1, 1000)), None)
                if exact is not None:
                    del self.out[start:]
                    self.inject(f'\\boxed{{{spelled(exact)}}}', 'hand-off close exact')
                    self.events[-1].update(cut=line[:120])
                    self.out.append(self.tok.convert_tokens_to_ids('<|im_end|>'))
                    self.done('stop')
                    return True
        expansion = _EXPANSION.search(line)
        if not hit and expansion:
            got = Fraction(expansion.group())
            # the model's own digits drift (pc4-whole 2289: -0.00000201862999... for -37/18329827 = -0.0000020185...): within 0.1 %
            hit = any(abs(got - v) <= abs(v) * Fraction(1, 1000) for v in self.given)
        if not hit or start < floor:
            return False
        del self.out[start:]
        self.inject(HAND_OFF, 'hand-off')
        self.events[-1].update(cut=line[:120])
        self.given = []   # once: a second restatement is the model's own, the loop guard's
        return True

    def custom_ops(self, text):
        """The operator symbols the question or the answer so far defines as an operation of its own (`a * b = a - b`); read once over the
        question and then only over what the answer added, so a long answer is not scanned again at every `=`."""
        from tools.igc.mathexpr import custom_operators
        if self._custom is None:
            self._custom = custom_operators(prompt_text(self.row) if self.row.get('question') else '')
        self._custom |= custom_operators(text[max(0, self._custom_at - 200):])
        self._custom_at = len(text)
        return self._custom

    def check_line(self):
        """The line that just ended, every relation on it checked (tools.igc.mathexpr.wrong_relations), and a correction after it for each
        wrong one. A correction the runner put in is not checked again."""
        from tools.igc.mathexpr import moduli_in, wrong_relations
        if self.moduli is None:   # the question's moduli first: `the remainder when N is divided by 1000`
            self.moduli = moduli_in(prompt_text(self.row) if self.row.get('question') else '')
        line = self.text().rstrip('\n').rsplit('\n', 1)[-1]
        if line.startswith('Correction'):
            return
        self.moduli |= moduli_in(line)   # `12+3=15=2` is right in an answer working mod 13
        for expr, wrote, right in wrong_relations(line, self.moduli, self.custom_ops(self.text())):
            self.inject(CORRECTION.format(expr=expr, right=right, wrote=wrote), 'hand check')
            self.events[-1].update(expr=expr[:200], value=right, wrote=wrote)
            # No falsification here: a wrong value of one step says nothing of the final answer, and marking it falsified for the whole
            # answer kept an answer that happened to equal it from ever closing (#303)

    def check_candidate_line(self):
        """--candidate-close: records explicit final-answer conclusions (`\\boxed{v}`, `the final answer is v`) and checks whether the
        3-tier CandidateAnswerTracker (`CROSS_METHOD_CONFIRMED`, never `SAME_ASSUMPTION_REPEAT` or `FALSIFIED_BY_SEARCH`) closes the
        thinking when a reflection starter (`Wait`, `Hmm`, `Let me double-check`, `Let me verify`) begins a redundant re-check."""
        if self.tracker is None or self.phase != 'think':
            return False
        from tools.igc.unified_detect import detect_candidate_line
        text = self.text()
        line = text.rstrip('\n').rsplit('\n', 1)[-1].strip()
        dec = detect_candidate_line(self.tracker, line, text, at_token=self.model_tokens())
        if dec is not None and dec.should_close and dec.value is not None:
            self.inject(LOOP_CLOSE, 'candidate close')
            self.events[-1].update(value=spelled(dec.value), status=dec.status, methods=list(dec.methods_seen), signatures=dec.unique_signatures)
            self.phase = 'answer'
            return True
        return False

    def hint_due(self):
        """At an `=` that ends tedious arithmetic, spaced HINT_GAP apart: the line goes back and a hinted program opens. True when it opened."""
        if self.runs >= MAX_RUNS or self.thought - self.last_hint < HINT_GAP:
            return False
        text = self.text()
        before = text[:text.rstrip().rfind('=')]
        found = inline_found(before)
        if found is None:
            return False
        expr, value = found
        if not tedious(expr, value):
            return False
        floor = self.spans[-1][1] if self.spans else 0   # never take back text the runner put in
        cut = len(self.out)
        while cut > floor and self.out[cut - 1] not in self.ids.lines:
            cut -= 1
        taken = len(self.out) - cut
        del self.out[cut:]
        self.last_hint = self.thought
        self.inject(HINT_OPEN, 'hint open')
        self.events[-1].update(expr=expr[:200], value=value, cut_tokens=taken)
        self.enter('hint')
        return True

    def enter(self, by):
        self.code_at, self.by, self.block, self.phase = len(self.text()), by, 0, 'block'

    def tool_reply(self, body):
        """The tool's answer in the template's own shape: a user turn of <tool_response>, then a new thinking; or, for a call the model wrote
        inside its thinking (not the template's place), the response inline where it stands."""
        if self.by == 'tool':
            self.inject(TOOL_TURN.format(body), 'tool response')
        else:
            self.inject(f'\n<tool_response>\n{body}\n</tool_response>\n', 'tool response')
        self.phase = 'think'

    def execute(self, code, kernel_factory):
        """(finished, printed, seconds) of one run in this answer's kernel, which keeps its state across runs. The name of a function the run
        called for the model (with_call) is left in self.auto_call."""
        began = time.monotonic()
        self.auto_call = None
        if self.runs >= MAX_RUNS:
            ok, shown = None, SPENT
        else:
            if self.kernel and not self.kernel.alive():
                self.kernel.close()
                self.kernel = None
            self.kernel = self.kernel or kernel_factory()
            ran, self.auto_call = with_call(code)
            ok, shown = self.kernel.run(ran)
        self.runs += 1
        return ok, shown, round(time.monotonic() - began, 3)

    def run_tool(self, code, kernel_factory):
        ok, shown, took = self.execute(code, kernel_factory)
        at = len(self.out)
        self.tool_reply(shown_text(ok, shown))
        self.events[-1].update(kind='output', by=self.by, code=code[:20000], code_chars=len(code), shown=(shown or '')[:4096],
                               shown_chars=len(shown or ''), ok=ok, seconds=took, fits=bool(ok) and bool((shown or '').strip()), run_at=at, auto_call=self.auto_call)

    def run_pending(self, kernel_factory):
        """Runs the block the model closed, then puts its output and, after a fitting result, the close into the thinking."""
        code, self.pending = self.pending, None
        if self.by in ('tool', 'tool in thinking'):
            self.run_tool(code, kernel_factory)
            return
        # The repeat close is for a model that reruns its own program; a hinted block that repeats one was asked for by the runner (gen-mt5:
        # a second hint at the same sum gave the same program and closed most answers)
        if self.arm.close_repeat and self.by in ('model', 'forced') and self.seen.get(code, 0) + 1 >= self.arm.close_repeat:
            self.inject(LOOP_CLOSE, 'repeated program close')
            self.phase = 'answer'
            return
        self.seen[code] = self.seen.get(code, 0) + 1
        ok, shown, took = self.execute(code, kernel_factory)
        fits = bool(ok) and bool((shown or '').strip())   # printed(): it finished and said something
        at = len(self.out)
        self.inject(output_block(shown_text(ok, shown)), 'output')
        self.events[-1].update(kind='output', by=self.by, code=code[:20000], code_chars=len(code), shown=(shown or '')[:4096],
                               shown_chars=len(shown or ''), ok=ok, seconds=took, fits=fits, run_at=at, auto_call=self.auto_call)
        if fits and self.tracker is not None:
            # A program's last printed value confirms a final answer the model already wrote, and only that: any printed line went in
            # before, so a helper that printed an intermediate number became a "search" that falsified every other answer (#303)
            from tools.igc.unified_detect import METHOD_EXHAUSTIVE_SEARCH
            last_ln = (shown or '').strip().splitlines()[-1].strip()
            value = self.tracker.normalize_candidate(last_ln)
            if value is not None and any(o.value == value for o in self.tracker.observations):
                self.tracker.record_candidate(last_ln, code, method=METHOD_EXHAUSTIVE_SEARCH, at_token=self.model_tokens())
        if self.by == 'hint' and fits and self.arm.trust != 'none':
            self.inject(TRUST_LINES[self.arm.trust], 'trust hint')
        if self.arm.close_fit and fits:
            self.inject(CLOSE, 'close after a fitting result')
            self.phase = 'answer'
        else:
            self.phase = 'think'

    def record(self, right):
        text = self.text()
        runs = [e for e in self.events if e['kind'] == 'output']
        return dict(id=self.row['id'], suite=self.row['suite'], answer=self.row['answer'], sample=self.sample, seed=self.seed, text=text,
                    right=right(self.row, text if self.thinking else CLOSE + text), finish=self.finish, tokens=self.model_tokens(), injected=len(self.out) - self.model_tokens(),
                    prompt_tokens=len(self.prompt), thinking_tokens=self.thought, seconds=self.seconds, events=self.events,
                    runs=len(runs), runs_by_model=sum(e['by'] == 'model' for e in runs), runs_forced=sum(e['by'] == 'forced' for e in runs),
                    runs_by_tool=sum(e['by'] in ('tool', 'tool in thinking') for e in runs),
                    inline_values=sum(e['kind'] == 'inline value' for e in self.events), loop_nudges=self.loop_nudges, echo_skips=self.echo_skips, joins_held=self.joins_held, early_cycles=self.early_cycles, shape_loops=self.shape_loops, cover_loops=self.cover_loops,
                    first_run_at=runs[0]['model_tokens'] if runs else None, tool_seconds=round(sum(e['seconds'] for e in runs), 3))


def kv_tokens(llm):
    """The tokens vLLM's KV cache holds, or None when this vLLM does not say."""
    try:
        c = llm.llm_engine.vllm_config.cache_config
        return int(c.num_gpu_blocks) * int(c.block_size)
    except Exception:  # noqa: BLE001 - an older or newer engine object: the caller keeps the --live it was given
        return None


def looping(tokens, progress=False):
    from tools.igc.external_score import repeating
    return repeating(tokens, progress=progress)


SHAPE_PERIOD, SHAPE_LOOK = 128, 128 * 600   # --shape-cap: the longest round, and how far back a run is read (collapsed tokens)


def shape_rounds(tokens, digits=frozenset(range(15, 25))):
    """The rounds of the longest exact cycle (period <= SHAPE_PERIOD) ending at the tail, with each run of digit tokens read as one token:
    `$31090 / 10943 = 2.8410...$ (Digit 5244: 2)` and `$4496 / 10943 ...$ (Digit 5245: 4)` are one round. #307 round 7: that division ran
    210k tokens and `check ... base 5815. No.` 262k, neither seen by the counting check (the numbers change length, or each round is new
    so --guard-progress let it go). Stored traces of 15 A100 arms: the longest such run in an answer that finished right is 805 rounds
    (trial division by one prime after another), the four that ran to the context end reached 1,874-4,267."""
    c = []
    prev = False
    for t in tokens[-SHAPE_LOOK * 4:]:
        d = t in digits
        if not (d and prev):
            c.append(-1 if d else t)
        prev = d
    n, best = len(c), 0
    for p in range(1, min(SHAPE_PERIOD, n - 1) + 1):
        if c[-1] != c[-1 - p]:
            continue
        j = n - 1
        stop = max(p, n - SHAPE_LOOK)
        while j >= stop and c[j] == c[j - p]:
            j -= 1
        best = max(best, (n - 1 - j) // p + 1)
    return best


COVER_GRAM, COVER_BACK, COVER_SHARE = 12, 2048, 0.97


class Cover:
    """--cover-cap W: of the last W tokens (each digit run read as one token), the share whose COVER_GRAM-gram was already seen in the last
    COVER_BACK of them; COVER_SHARE or more over a full window is a loop. shape_rounds wants an exact cycle and one odd round breaks it:
    #307 round 8, a long division where every quotient digit 0 made a shorter round (`$320 / 10943 = 0.0292...$`, `$0$.`) ran 196k tokens
    to the context end with shape rounds never past 46. A run of seen grams broke there too, every ~20k tokens on a rarer round; a share
    moves 1/W for one new gram. Stored traces of 16 A100 arms (share_sim.py 12 2048 32768): the highest share in an answer that finished
    right is 0.92 (two answers that looped long and then came right), the next 0.35; all 10 answers that ran past 150k tokens in a loop
    reach 0.999-1.0. Fed as the text grows; a window that does not extend the one fed so far (a nudge, the answer's start, a dropped tail)
    starts afresh."""

    def __init__(self, digits, width=32768):
        self.digits, self.width, self.base, self.fed = digits, width, None, 0
        self.c, self.prev, self.seen, self.hits, self.share = [], False, {}, [], 0.0
        self.count = 0

    def feed(self, out, start, epoch=0):
        """Feeds out[start:] past what was fed; the share of the last full window, 0 until the window is full."""
        if (start, epoch) != self.base or len(out) < self.fed:
            self.__init__(self.digits, self.width)
            self.base, self.fed = (start, epoch), start
        for t in out[self.fed:]:
            d = t in self.digits
            if not (d and self.prev):
                self.c.append(-1 if d else t)
                i = len(self.c) - 1
                if i >= COVER_GRAM - 1:
                    g = tuple(self.c[i - COVER_GRAM + 1:])
                    j = self.seen.get(g)
                    h = 1 if j is not None and j >= i - COVER_BACK else 0
                    self.seen[g] = i
                    self.hits.append(h)
                    self.count += h
                    if len(self.hits) > self.width:
                        self.count -= self.hits[-self.width - 1]
            self.prev = d
        self.fed = len(out)
        return self.count / self.width if len(self.hits) >= self.width else 0.0


EARLY_ROUNDS, EARLY_ROUNDS_LONG, EARLY_SPAN = 6, 3, 500


def early_cycle(tokens, longest=4096):
    """--early-cycle: the tail is one exact cycle of EARLY_ROUNDS rounds (EARLY_ROUNDS_LONG past 512 tokens) that spans EARLY_SPAN tokens or
    more. The bench's check wants 20 rounds (8 past 512). #307 detector replay over every stored A100 arm with no closer (detect_sim V7):
    no fire on 915 answers that finished right nor on any that finished wrong, 212 of 243 loops caught sooner, 624k tokens sooner in all.
    The span keeps a short echo (`= 15/13 = 15/13`, a column of digits) out: those are for the bench's own rounds."""
    n = len(tokens)
    for period in range(1, min(longest, n // EARLY_ROUNDS_LONG) + 1):
        if tokens[-1] != tokens[-1 - period]:
            continue
        rounds = max(EARLY_ROUNDS if period <= 512 else EARLY_ROUNDS_LONG, -(-EARLY_SPAN // period))
        if period * rounds > n:
            continue
        tail = tokens[-period * rounds:]
        if tail[period:] == tail[:-period]:
            return True
    return False


def eos_ids(model, tok, generation_config):
    """The end tokens wherever the checkpoint states them: Qwen3.5-4B ships no generation_config.json, and its config keeps the id under text_config."""
    ids = {tok.eos_token_id} - {None}
    try:
        e = generation_config.from_pretrained(model).eos_token_id
        ids |= set(e if isinstance(e, list) else [e]) - {None}
    except OSError:
        pass
    config = json.loads((Path(model) / 'config.json').read_text(encoding='utf-8'))
    for c in (config, config.get('text_config', {})):
        e = c.get('eos_token_id')
        ids |= set(e if isinstance(e, list) else [e]) - {None}
    return sorted(ids)


def prompt_text(row):
    from tools.igc.standard_rows import plain_prompt
    text = row.get('question') or plain_prompt(row)
    return text + MATH_TAIL if row['suite'] == 'stdmath' else text


def seed_of(seed, row_id, sample):
    return int.from_bytes(hashlib.sha256(f'{seed}:{row_id}:{sample}'.encode()).digest()[:4], 'little')


def drive(llm, rollouts, kernel_factory, live_most=256, workers=32, on_round=None):
    """Continuous: an answer's next request goes into the engine the moment its last one returns, and a block it closed runs on a worker
    meanwhile. Lock-step rounds (one generate() for every live answer, then the next) held each answer to the longest request of its round:
    the inline arm stops at every `=` and ran 5x slower than its control (r1 on the A100, 2 rows to 11 in 20 minutes). At most `live_most`
    answers are live at once, each holding its kernel until it ends, so a few thousand answers do not hold a few thousand kernels."""
    from concurrent.futures import FIRST_COMPLETED, wait
    from vllm import SamplingParams, TokensPrompt
    engine = llm.llm_engine
    pool = ThreadPoolExecutor(workers)
    waiting_rows = list(rollouts)
    flying, running = {}, {}   # request id -> answer; a block's future -> answer
    serial = iter(range(1 << 62))

    def ask(r):
        """Sends r's next request; False once r has ended."""
        q = r.request()
        if not q:
            return False
        most, stops = q
        bias = r.bias()
        why = vllm_refusal(len(r.prompt) + len(r.out), most, bias)
        if why:
            raise RuntimeError(f'{r.row["id"]}#{r.sample}: {why}')
        params = SamplingParams(**SAMPLING, max_tokens=most, stop_token_ids=sorted(stops), seed=(r.seed + len(r.out)) % 2**31,
                                skip_special_tokens=False, detokenize=False,
                                **(dict(logit_bias=bias, **(dict(logprobs=20) if r.plan is not None else {})) if bias else {}))
        rid = str(next(serial))
        engine.add_request(rid, TokensPrompt(prompt_token_ids=r.prompt + r.out), params)
        flying[rid] = r
        return True

    while waiting_rows or flying or running:
        ended = False
        while waiting_rows and len(flying) + len(running) < live_most:
            r = waiting_rows.pop(0)
            r.started = time.monotonic()
            ended |= not ask(r)
        if flying:
            for o in engine.step():
                if not o.finished:
                    continue
                r = flying.pop(o.request_id)
                c = o.outputs[0]
                new = list(c.token_ids)
                if isinstance(c.stop_reason, int) and (not new or new[-1] != c.stop_reason):
                    new.append(c.stop_reason)
                lp = getattr(c, 'logprobs', None)
                r.last_lp = lp[0] if lp else None
                if r.take(new):
                    running[pool.submit(r.run_pending, kernel_factory)] = r
                else:
                    ended |= not ask(r)
        else:
            wait(list(running), return_when=FIRST_COMPLETED)
        for f in [f for f in running if f.done()]:
            r = running.pop(f)
            f.result()
            ended |= not ask(r)
        if ended and on_round:
            on_round()
    pool.shutdown()


def stop_words(stops, ids, tok):
    """The stop token ids of a request as the strings llama-server stops on: it takes words, not ids. The end of turn is its own eos."""
    words = set()
    if stops & ids.equals:
        words.add('=')
    if stops & ids.lines:
        words.add('\n')
    if stops & ids.fence:
        words.add('```')
    if ids.think_close in stops:
        words.add('</think>')
    if getattr(ids, 'tool_close', None) in stops:
        words.add('</tool_call>')
    return sorted(words)


def drive_http(post, tok, ids, rollouts, kernel_factory, live_most=16, on_round=None, drain=None):
    """drive() over llama-server's /completion, so the 5060 Ti runs the same rules on Windows, where vLLM does not (developer, 2026-11-04).
    Each answer runs on its own thread and the server batches its slots. The server stops on words and leaves the word out of the text, so
    it is put back and the text encoded again: the same Rollout takes it.
    drain: when it returns True, no answer starts; the ones in flight end, and the run is continued later with --resume."""
    import threading
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    lock = threading.Lock()
    sampling = dict(temperature=SAMPLING['temperature'], top_p=SAMPLING['top_p'], top_k=SAMPLING['top_k'], min_p=SAMPLING['min_p'],
                    presence_penalty=SAMPLING['presence_penalty'], repeat_penalty=SAMPLING['repetition_penalty'])

    def one(r):
        if drain and drain():
            return
        r.started = time.monotonic()
        while True:
            ask = r.request()
            if not ask:
                break
            most, stops = ask
            bias = r.bias()   # --no-digit-join's digit bias: llama-server takes it as [[id, bias], ...], like vLLM's dict
            reply = post(dict(prompt=r.prompt + r.out, n_predict=most, stop=stop_words(stops, ids, tok), cache_prompt=True,
                              seed=(r.seed + len(r.out)) % 2**31, return_tokens=True, **sampling,
                              **(dict(logit_bias=[[int(t), float(v)] for t, v in bias.items()]) if bias else {})))
            if reply.get('tokens') is not None:
                # the ids the model generated, its stop token among them: a stop on '\n' inside the token '\n\n' put back as '\n' fed the
                # model a text it never wrote, every line (pc-check: 21 of 48 answers ran to the cap against base's 7)
                new = list(reply['tokens'])
            else:
                text = reply.get('content', '') + (reply.get('stopping_word', '') if reply.get('stop_type') == 'word' else '')
                new = tok.encode(text, add_special_tokens=False) if text else []
                if reply.get('stop_type') == 'eos':
                    new.append(im_end)
            if not new:
                r.done('length')   # nothing came back at the limit: no answer can come
                break
            if r.take(new):
                r.run_pending(kernel_factory)
        if on_round:
            with lock:
                on_round()
    with ThreadPoolExecutor(live_most) as pool:
        list(pool.map(one, rollouts))


def install_seal(llm, args):
    """--seal: the SEAL vector (tools/igc/seal.py, built by probe/seal_vector.py) added at block --seal-layer to the state at every `\\n\\n` token the model reads or writes.
    Returns the counter {'steered', 'calls'} of the hook, read at the end of the run: a run that steered 0 positions did not test SEAL."""
    import numpy as np
    import torch
    from transformers import AutoConfig
    from tools.igc import seal
    z = np.load(args.seal)
    key = f'L{args.seal_layer}'
    if key not in z.files:
        raise SystemExit(f'--seal-layer {args.seal_layer}: {args.seal} holds {[k for k in z.files if k.startswith("L")]}')
    vector, delim = torch.tensor(z[key]), int(z['delim'])
    n_layers = AutoConfig.from_pretrained(args.model).get_text_config().num_hidden_layers
    results = llm.apply_model(lambda model: seal.install_steering(model, vector, args.seal_layer, args.seal_alpha, delim, n_layers))
    count = results[0][1]
    print(f'SEAL on: layer {args.seal_layer}, alpha {args.seal_alpha}, delimiter token {delim}, |S| {float(vector.norm()):.1f}', flush=True)
    return count


def run(args):
    import os
    os.environ.setdefault('STRATA_SANDBOX_THROWAWAY', '1')   # the kernel runs bare only on a VM that is thrown away (exec_bridge.throwaway)
    from serve.sandbox.exec_bridge import Kernel, throwaway
    from tools.igc.select_rows import right
    from transformers import AutoTokenizer, GenerationConfig

    kernel_factory = Kernel
    if args.box:   # D1: approved by the developer on 2026-11-04 for the five-layer test on this PC; a box per answer's kernel
        # the server's own box (2 GB, 3600 s): Policy()'s 60 s clock ended each kernel between two blocks of an answer (pc5-full)
        from serve.reasoning_tool import box_factory
        start = box_factory()
        kernel_factory = lambda: Kernel(box=start())
    elif (args.execute or args.native_tool or args.hint_calc or args.loop_hint) and not throwaway():
        raise SystemExit('a block runs only on a throwaway VM, or in the isolation box (--box)')
    rows = [json.loads(l) for l in Path(args.rows).read_text(encoding='utf-8').splitlines() if l.strip()]
    rows = rows[args.offset:args.offset + args.limit] if args.limit else rows[args.offset:]
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = Ids(tok, eos_ids(args.model, tok, GenerationConfig))
    arm = Arm(args.force_after, args.execute, args.close_fit, args.loop_close, args.close_repeat, args.inline_calc, args.native_tool, args.hint_calc, args.loop_hint, args.inline_plain,
              args.whole_hint, args.check_hand, args.whole_plain, args.hand_off, args.question_value, trust=args.trust_line, logit_value=args.logit_value,
              suppress_recheck=args.suppress_recheck, mathbackend=args.mathbackend, candidate_close=args.candidate_close,
              inline_style=args.inline_style, no_digit_join=args.no_digit_join, guard_progress=args.guard_progress, answer_close=args.answer_close,
              loop_value=args.loop_value, early_cycle=args.early_cycle, shape_cap=args.shape_cap, cover_cap=args.cover_cap,
              exact_answer=args.exact_answer)
    if args.no_digit_join and not (args.inline_calc and args.inline_plain):
        raise SystemExit('--no-digit-join holds digits down after a plain inline put: it needs --inline-calc --inline-plain')
    if args.suppress_recheck and (args.server or not (args.inline_calc or args.logit_value)):
        raise SystemExit('--suppress-recheck lowers the re-check words after a value software gave: it needs --inline-calc or --logit-value, and vLLM (not --server)')
    if args.logit_value and (args.inline_calc or args.server or args.whole_hint or args.hand_off):
        raise SystemExit('--logit-value is the value arriving by the prediction: not with --inline-calc (the value as text), --whole-hint, --hand-off, or --server (vLLM only)')
    if args.whole_hint and not args.inline_calc:
        raise SystemExit('--whole-hint asks for a value that only --inline-calc puts in')
    if args.greedy:   # drive() and arm.json read SAMPLING: temperature 0 is vLLM's greedy
        SAMPLING.update(temperature=0.0)
    if args.native_tool and tok.convert_tokens_to_ids('</tool_call>') == tok.unk_token_id:
        raise SystemExit('the tokenizer has no </tool_call> token')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # --resume: answers already written are kept and not run again. A restart to fix the live count lost every answer in flight but not these.
    kept = json.loads((out / 'records.json').read_text(encoding='utf-8')) if args.resume and (out / 'records.json').exists() else []
    have = {(r['id'], r['sample']) for r in kept}
    # the kept answers' token ids too: ids.jsonl was rewritten with this session's answers only, and the replay needs every one
    kept_ids = [line for line in (out / 'ids.jsonl').read_text(encoding='utf-8').splitlines()
                if line.strip() and (json.loads(line)['id'], json.loads(line)['sample']) in have] \
        if args.resume and (out / 'ids.jsonl').exists() else []
    rollouts = []
    for row in rows:
        # As text, then encoded: transformers 5 returns a dict from tokenize=True, and list() of it gave the keys as the prompt.
        system = [{'role': 'system', 'content': args.system}] if args.system else []
        text = tok.apply_chat_template(system + [{'role': 'user', 'content': prompt_text(row)}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=not args.no_think, tools=[PYTHON_TOOL] if args.native_tool else None)
        prompt = tok.encode(text, add_special_tokens=False)
        opened = text.rstrip().endswith('</think>') if args.no_think else text.rstrip().endswith('<think>')
        if not opened or not all(isinstance(t, int) for t in prompt):
            raise SystemExit('the chat template did not end as --no-think asks: ' + repr(text[-80:]))
        for k in range(args.samples):
            if (row['id'], k) in have:
                continue
            rollouts.append(Rollout(row, k, list(prompt), arm, ids, tok, seed_of(args.seed, row['id'], k), thinking=not args.no_think))
            rollouts[-1].cap = args.max_tokens
    if args.force_after and not args.execute:
        raise SystemExit('--force-after opens a block that only --execute runs')
    if not ids.same_lines:   # the stuck stop and the loop close both read them
        raise SystemExit(f'the line ends are {sorted(ids.lines)}, not the 198 and 271 the loop checks read')
    began = time.monotonic()
    kwargs, llm, kv, seal_count = dict(server=args.server), None, None, None
    if args.seal and args.server:
        raise SystemExit('--seal steers the model inside vLLM: not with --server')
    if not args.server:
        if args.seal:
            os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'   # the engine in this process, so a hook installed here reaches the model
        from vllm import LLM
        kwargs = dict(model=args.model, dtype='bfloat16', max_model_len=CONTEXT, gpu_memory_utilization=args.memory, enable_prefix_caching=True,
                      language_model_only=True, seed=args.seed)
        if args.logit_value:
            kwargs['logprobs_mode'] = 'raw_logprobs'   # the model's own log P of the wanted token, read before the bias is added
        if args.seal:
            kwargs['enforce_eager'] = True   # no CUDA graph or compiled graph: a forward hook is part of every step
        try:
            llm = LLM(**kwargs)
        except TypeError:   # a vLLM with no text-only switch loads the vision tower too, which costs memory and changes nothing here
            kwargs.pop('language_model_only')
            llm = LLM(**kwargs)
        if args.seal:
            seal_count = install_seal(llm, args)
        # Live answers past what the KV cache holds are run in turns and lose their prefix cache: every chunk reads its whole context again.
        # It stalled three runs (an L4 at 164 and 16 live, an A100 at 32 live on half the memory), so the count follows the cache here.
        kv = kv_tokens(llm)
    expect = args.max_tokens or args.expect_len
    if kv and args.live * expect > kv:
        fit = max(1, int(0.85 * kv / expect))
        print(f'--live {args.live} x {expect} tokens is past the KV cache ({kv} tokens): {fit} live', flush=True)
        args.live = fit
    (out / 'arm.json').write_text(json.dumps(dict(vars(args), kv_tokens=kv, sampling=SAMPLING, context=CONTEXT, load_seconds=round(time.monotonic() - began, 1),
                                                   llm={k: v for k, v in kwargs.items() if k != 'model'}, think_close=ids.think_close,
                                                   ends=sorted(ids.ends), lines=sorted(ids.lines), fences=len(ids.fence))), encoding='utf-8')
    began = time.monotonic()
    written = [0, 0.]

    def save(last=False):
        done = [r for r in rollouts if r.finish]
        if len(done) == written[0] or (not last and time.monotonic() - written[1] < 30):
            return
        written[0], written[1] = len(done), time.monotonic()
        records = kept + [r.record(right) for r in done]
        (out / 'records.json').write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
        if args.ids:
            with open(out / 'ids.jsonl', 'w', encoding='utf-8') as f:
                for line in kept_ids:
                    f.write(line + '\n')
                for r in done:
                    f.write(json.dumps(dict(id=r.row['id'], sample=r.sample, prompt_ids=r.prompt, out_ids=r.out, spans=r.spans)) + '\n')
        print(f'{len(done)}/{len(rollouts)} done, {sum(x["right"] for x in records)} right, {round(time.monotonic() - began)} s', flush=True)
    if args.server:   # llama-server on this PC's 5060 Ti: vLLM does not run on Windows
        import urllib.request

        def post(body):
            req = urllib.request.Request(args.server.rstrip('/') + '/completion', data=json.dumps(body).encode('utf-8'),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=3600) as reply:
                return json.loads(reply.read().decode('utf-8'))
        # `touch <out>/drain`: the answers in flight end, no new one starts, the run exits 3 without a report; --resume goes on
        flag = out / 'drain'
        flag.unlink(missing_ok=True)
        drive_http(post, tok, ids, rollouts, kernel_factory, live_most=args.live, on_round=save, drain=flag.exists)
        if flag.exists() and not all(r.finish for r in rollouts):
            save(last=True)
            print(f'DRAINED: {sum(bool(r.finish) for r in rollouts)}/{len(rollouts)} answered; go on with --resume', flush=True)
            raise SystemExit(3)
    else:
        drive(llm, rollouts, kernel_factory, live_most=args.live, on_round=save)
    save(last=True)
    records = kept + [r.record(right) for r in rollouts]
    (out / 'report.json').write_text(json.dumps(dict(rows=len(rows), samples=args.samples, answers=len(records), right=sum(r['right'] for r in records),
                                                      tokens=sum(r['tokens'] for r in records), seconds=round(time.monotonic() - began, 1),
                                                      runs=sum(r['runs'] for r in records), runs_by_model=sum(r['runs_by_model'] for r in records),
                                                      length=sum(r['finish'] == 'length' for r in records),
                                                      **(dict(seal_steered=seal_count['steered'], seal_calls=seal_count['calls']) if seal_count else {}))), encoding='utf-8')
    if seal_count:
        print(f'SEAL steered {seal_count["steered"]} positions in {seal_count["calls"]} forward passes', flush=True)
        if not seal_count['steered']:
            print('SEAL STEERED NOTHING: the hook did not see the ids; this run is a control, not a SEAL arm', flush=True)
    print('DONE', flush=True)


def math_train(args):
    """MATH's train split, levels picked, gold from the last \\boxed of the solution. A row whose id matches an evaluated row is left out:
    the ids are the hash of the problem (question_id), the same as standard_rows builds for the test rows."""
    from datasets import load_dataset
    from tools.igc.external_score import question_id
    from tools.igc.standard_score import last_boxed, normalize_math
    banned = set()
    for path in args.exclude:
        banned |= {json.loads(l)['id'] for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()}
    levels = {f'Level {n}' for n in args.levels.split(',')}
    rows = []
    for subject in ('algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'):
        for r in load_dataset('EleutherAI/hendrycks_math', subject, split='train'):
            gold = last_boxed(r['solution'])
            rid = question_id('stdmath', r['problem'])
            if r['level'] in levels and gold and rid not in banned:
                rows.append(dict(id=rid, suite='stdmath', question=r['problem'], answer=normalize_math(gold), group=subject, level=r['level']))
    random.Random(args.seed).shuffle(rows)
    Path(args.out).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows[:args.n]), encoding='utf-8')
    print('rows', min(args.n, len(rows)), 'of', len(rows), 'left out as evaluated', len(banned), flush=True)


def choose(records):
    """Per question, the trajectory it trains on: right, ended, no text of ours but the open and the close, and the shortest. A run whose result
    fitted is preferred, since that is the behaviour taught; a question answered before any block opened keeps its short tool-free answer, so the
    model also learns that an easy question needs no program (PRD R13)."""
    by = {}
    for r in records:
        kinds = {e['kind'] for e in r['events']} - set(TRAIN_KINDS) - set(MASK_KINDS) - {'model opened a block'}
        if r['right'] and r['finish'] == 'stop' and not kinds:
            by.setdefault(r['id'], []).append(r)
    chosen = {}
    for rid, rs in by.items():
        fitted = [r for r in rs if any(e.get('fits') for e in r['events'])]
        plain = [r for r in rs if not r['runs']]
        pick = fitted or plain
        if pick:
            chosen[rid] = min(pick, key=lambda r: r['tokens'] + r['injected'])
    return chosen


def keep(args):
    records, traj = [], {}
    for d in args.dirs:
        records += json.loads((Path(d) / 'records.json').read_text(encoding='utf-8'))
        for l in (Path(d) / 'ids.jsonl').read_text(encoding='utf-8').splitlines():
            t = json.loads(l)
            traj[(t['id'], t['sample'])] = t
    chosen = choose(records)
    n = long = 0
    with open(args.out, 'w', encoding='utf-8') as f:
        for rid, r in chosen.items():
            t = traj[(rid, r['sample'])]
            if len(t['prompt_ids']) + len(t['out_ids']) > args.max_len:
                long += 1
                continue
            f.write(json.dumps(dict(id=rid, sample=r['sample'], prompt_ids=t['prompt_ids'], out_ids=t['out_ids'],
                                    mask=[s[:2] for s in t['spans'] if s[2] in MASK_KINDS], runs=r['runs'])) + '\n')
            n += 1
    print(f'questions {len({r["id"] for r in records})}, kept {n} ({sum(c["runs"] > 0 for c in chosen.values())} with a fitting run), '
          f'over {args.max_len} tokens {long}', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--model', required=True)
    r.add_argument('--rows', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--samples', type=int, default=1)
    r.add_argument('--seed', type=int, default=1)
    r.add_argument('--limit', type=int, default=0)
    r.add_argument('--offset', type=int, default=0)
    r.add_argument('--max-tokens', type=int, default=0)   # 0: the whole context, as every evaluated arm runs
    r.add_argument('--force-after', type=int, default=0)
    r.add_argument('--execute', action='store_true')
    r.add_argument('--close-fit', action='store_true')
    r.add_argument('--loop-close', action='store_true')
    r.add_argument('--exact-answer', action='store_true', help="when the question states one computation, the answer's box holds its exact "
                   'value: the answer-close cue ends with it, and a rounded, cut or integer-part box of it is rewritten')
    r.add_argument('--loop-value', action='store_true', help="before a loop is closed, put the question's own computation, exact")
    r.add_argument('--shape-cap', type=int, default=0, help='a cycle of this many rounds, every number read as one token, is a loop even '
                                                            'when each round is new (#307: 1024; right answers reached 805 at most)')
    r.add_argument('--cover-cap', type=int, default=0, help='a window of this many tokens (numbers read as one) in which 97 %% repeat a 12-gram '
                                                            'of the last 2,048 is a loop, odd rounds or not (#307: 32768; right answers 0.92 at most)')
    r.add_argument('--early-cycle', action='store_true', help='the thinking is also read, every request, for an exact cycle of 6 rounds '
                                                              '(3 past 512 tokens) spanning 500+ tokens (#307 detector replay V7)')
    r.add_argument('--answer-close', action='store_true', help='a loop in the answer (thinking closed) gets the final-answer cue and one value, not `stuck`')
    r.add_argument('--close-repeat', type=int, default=0)
    r.add_argument('--inline-calc', action='store_true')
    r.add_argument('--whole-plain', action='store_true', help="--whole-hint asking for one plain expression in the question's order, not LaTeX")
    r.add_argument('--check-hand', action='store_true', help='check every hand value at the end of its line; a wrong one gets a correction')
    r.add_argument('--whole-hint', action='store_true', help='open the reply asking for the whole step as one expression, computed whole')
    r.add_argument('--inline-plain', action='store_true', help="the value alone, as Qwenputer prints it, without the engine's ' (exact)'")
    r.add_argument('--native-tool', action='store_true', help="offer a python tool in Qwen3.5's own tool format and answer its calls")
    r.add_argument('--hint-calc', action='store_true', help="CoRT's hint where a tedious hand calculation begins, and its trust line after a run")
    r.add_argument('--trust-line', choices=('cort', 'none', 'positive'), default='cort',
                   help="the line after a hinted program that printed: CoRT's \"We don't need to doubt...\" (default, as before), none, or ATTC's positive sentence")
    r.add_argument('--logit-value', type=float, default=0.0,
                   help="MIP's logit override, soft: at an `=` the value is not put in; the model writes it, and the logit of software's next token is raised by this much (vLLM)")
    r.add_argument('--seal', default='', help='SEAL steering vector (.npz from tools/igc/probe/seal_vector.py): added at every \\n\\n token, the reasoning is held to executing (arXiv 2504.07986)')
    r.add_argument('--seal-layer', type=int, default=20, help='the block whose output is steered (1-based), a layer saved in the .npz')
    r.add_argument('--seal-alpha', type=float, default=1.0, help='the strength: the stream gets alpha * S (the paper: 1.0)')
    r.add_argument('--suppress-recheck', type=int, default=0,
                   help="for this many steps after a value software gave (inline or logit), the logits of the re-check words (wait, hmm, verify, check, double, again...) are lowered by 30 (vLLM)")
    r.add_argument('--no-digit-join', action='store_true',
                   help="after a put that ends in a digit, the model's next token cannot start with one, so it cannot run the number on (#307: "
                        "80 of 19,990 puts were joined into another number, `-5` then `75`) (vLLM)")
    r.add_argument('--guard-progress', action='store_true',
                   help="the bench's counting check lets a tail whose rounds are nearly all new go on (trial division, a search), and stops only "
                        "rounds that come back (#307: 3 of 38 `stuck` rows were still moving)")
    r.add_argument('--loop-hint', action='store_true', help='a loop in the thinking is cut to one round and a program opens to test the idea')
    r.add_argument('--server', default='', help='a llama-server URL (127.0.0.1) to run on in place of vLLM: this PC, on Windows')
    r.add_argument('--question-value', action='store_true', help='the whole value is the computation the question states, read by software')
    r.add_argument('--hand-off', action='store_true', help='a line that only restates a value software gave is cut, and the model is led to its answer')
    r.add_argument('--mathbackend', action='store_true', help='extend --inline-calc with the exact bridge (unified_detect.expression_before_exact): mathbackend calls, roots kept as roots, exact trig and logs, sums and products over a range, a^b mod m, lowest terms')
    r.add_argument('--inline-style', default='general', choices=sorted(INLINE_STYLES),
                   help="how a value is written where the model's `=` may mean something else (mod m, a long division, a decimal text): "
                        "general (default, true under every reading) or a model's measured profile (#306)")
    r.add_argument('--candidate-close', action='store_true', help='close thinking on a reflection starter only when CandidateAnswerTracker confirms the answer across independent methods (never on same-assumption repeats)')
    r.add_argument('--box', action='store_true', help="blocks run in the OS isolation box (AppContainer) on this PC: D1, the developer's call")
    r.add_argument('--no-think', action='store_true', help='enable_thinking=False: the reply is the answer, as Qwenputer measures arithmetic')
    r.add_argument('--greedy', action='store_true', help='temperature 0, for a protocol that publishes greedy numbers (Qwenputer)')
    r.add_argument('--system', default='', help='a system turn before the question (with --native-tool it follows the tool list)')
    r.add_argument('--ids', action='store_true')
    r.add_argument('--memory', type=float, default=0.9)
    r.add_argument('--live', type=int, default=256)
    r.add_argument('--resume', action='store_true')
    r.add_argument('--expect-len', type=int, default=24000)   # tokens one live answer may reach, for the KV fit; a capped run uses its cap
    m = sub.add_parser('math-train')
    m.add_argument('out')
    m.add_argument('--n', type=int, required=True)
    m.add_argument('--levels', default='3,4,5')
    m.add_argument('--seed', type=int, default=1)
    m.add_argument('--exclude', nargs='*', default=[])
    k = sub.add_parser('keep')
    k.add_argument('dirs', nargs='+')
    k.add_argument('--out', required=True)
    k.add_argument('--max-len', type=int, default=32768)
    args = p.parse_args(argv)
    {'run': run, 'math-train': math_train, 'keep': keep}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
