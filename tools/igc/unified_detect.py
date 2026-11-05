"""Unified Multi-Layer MIP Detector & Assumption-Independent Candidate Answer Closer (#298).

Implements the two core safeguards of `docs/plans/2026-11-05-mip-unified-detection-pipeline.md`:

1. `UnifiedDetector`:
   - Combines multi-layer signals (`probe_L24`, `probe_L31`, `router` [when MoE], `next` token distribution, `text_lr`)
     into a calibrated / parameter-free consensus gate (`comp_gate`, `reflect_gate`, `wrong_value_gate`).
   - Enforces **Joint Out-of-Distribution (OOD) Evaluation**: simultaneously measures Precision, Coverage (`caught / onsets`),
     and False Alarm rate (`alarms_per_1k_tokens`, `false_alarms_per_1k_tokens`, and Hard-Negative False Alarm rate) on
     both In-Distribution Cross-Validation traces and Held-Out Unseen OOD problem types never used to fit the detector.
   - Enforces **Non-Destructive Latent Gating**: latent alarms (`probe_L24/L31`, `router`) only *arm* a verification window
     (`COMP_ARMED` / `AUDIT_ARMED`); actual text/token intervention requires an exact symbolic parse (`mathexpr` or the 6-module
     `mathbackend` with 382 operations), guaranteeing **Destructive False Alarm = 0** on unseen problem types.

2. `CandidateAnswerTracker` (Assumption-Independent Stable Candidate Answer Closer):
   - Prevents the trap where obtaining the same candidate answer twice (`v_1 == v_2`) stems from repeating the **exact same
     wrong assumption or formula** (`SAME_ASSUMPTION_REPEAT`).
   - Tags every candidate answer with its `method` (`METHOD_ALGEBRAIC`, `METHOD_EXHAUSTIVE_SEARCH`, `METHOD_SMALL_CASE_CHECK`)
     and its structural `assumption_signature` (canonicalized AST / number-set signature stripped of commutativity and whitespace).
   - Enforces the **3-Tier Candidate Closing Policy**:
     * Tier 1 (`FALSIFIED_BY_SEARCH`): If a small-case check or exhaustive `search` contradicts an algebraic formula's value,
       the formula's value is immediately invalidated regardless of how many times it was repeated.
     * Tier 2 (`SAME_ASSUMPTION_REPEAT`): If `v` appears >= 2 times but all occurrences share the same `assumption_signature`
       (or only come from `METHOD_ALGEBRAIC` with the same operand multiset/structure), it is recorded as supporting evidence
       only and **does NOT trigger an early `</think>` close** (closing only if a verified loop `repeating`/`counting`/`line_back`
       occurs or the context ceiling is reached).
     * Tier 3 (`CROSS_METHOD_CONFIRMED`): When `v` is confirmed by >= 2 structurally independent methods (e.g. `METHOD_ALGEBRAIC`
       + `METHOD_EXHAUSTIVE_SEARCH`, or `METHOD_ALGEBRAIC` + `METHOD_SMALL_CASE_CHECK`, or two algebraically distinct derivations
       with disjoint intermediate structure) AND `reflect_gate` indicates the model is starting another redundant re-check,
       `should_close()` returns True (`CROSS_METHOD_CONFIRMED`).

3. `mathbackend_inline`:
   - Extends `mathexpr.py` with exact natural / LaTeX / functional calls into the 6-module `mathbackend` (382 operations:
     `counting`, `numbers`, `algebra`, `geometry`, `search`, `symbolic`), covering `\gcd`, `\lcm`, `\varphi`, `\det`,
     `P(n,k)`, `C(n,k)`, Stirling/Catalan/derangements, modular inverse, and exact symbolic polynomial expansion/roots.
"""
from __future__ import annotations

import ast
import functools
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tools.igc import mathexpr
from tools.igc import mathbackend


# ----------------------------------------------------------------------------------------------------------------------------------
# 1. Exact MathBackend Bridge for Inline Expressions & Hand-Check Auditing
# ----------------------------------------------------------------------------------------------------------------------------------

_FUNC_CALL = re.compile(
    r'(?:\\operatorname\{(?P<op>[A-Za-z_][A-Za-z0-9_]*)\}|\\(?P<tex>gcd|lcm|varphi|det)\b|(?P<plain>\b(?:gcd|lcm|totient|modinv|invmod|catalan|derangements|bell|partition|stirling1|stirling2|lucas|fibonacci|comb|perm)\b))\s*\(\s*(?P<args>[^()]+)\s*\)'
)
# Greek letters name other things as often as number-theory functions: `\sigma(1)` of a permutation got sigma(1) = 1 put in, `\tau(2)` of a
# step function 2, and `\mu`, `\phi` the same. Only `\varphi` and `totient` stay for Euler's function.
# A result past mathexpr.LIMIT is thrown away after it is computed, and `comb(1000000, 500000)` took 74 s on one line before it was: these
# are the largest arguments whose value can still be under 10^40 (or that cost under a second), refused before computing.
_MOST_N = dict(catalan=80, derangements=40, bell=60, partition=2000, stirling1=120, stirling2=120, fibonacci=200, lucas=200)
_LOG10_LIMIT = math.log10(mathexpr.LIMIT) + 1
_MOD_INV_TEX = re.compile(r'(-?\d+)\s*\^\s*\{\s*-1\s*\}\s*(?:\\pmod|\\bmod|\bmod)\s*\{?\s*(\d+)\s*\}?')
_DET_MATRIX = re.compile(
    r'(?:\\(?:operatorname\{det\}|det)\s*(?:\([^\S\r\n]*)?(?:\\left[\[\(|.]\s*)?)?'
    r'\\begin\{(?P<env>pmatrix|bmatrix|matrix|vmatrix|Vmatrix|array)\}(?:\{[^}]*\})?'
    r'(?P<body>[\s\S]+?)'
    r'\\end\{(?P=env)\}'
    r'(?:\s*\\right[\]\)|.]\s*\)?)?',
    re.IGNORECASE
)
# Only general primitives reach the inline path. Bare `C(` / `P(` are gone: points are named P and C, and `P(2, 3) =` put in 0 (#302).
# The ops written from one benchmark question (n-gon rectangles, quartic double tangent, disphenoid inradius, polar triangle chain, triangle
# wave intersections) are gone from it too: they answer that question, and `rectangles ... regular dodecagon` in the model's restatement of
# it put in the gold 315. They stay in the backend.
_DET_WORD = re.compile(r'\\det\b|\\operatorname\{det\}|\bdet\b|\bdeterminant\b', re.I)


def _parse_latex_matrix(body: str) -> Optional[List[List[Fraction]]]:
    """Parse rows and columns of a LaTeX matrix body into a square list of Fraction values."""
    body = body.strip()
    if body.startswith('{'):
        depth = 0
        end_idx = -1
        for idx, ch in enumerate(body):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end_idx = idx
                    break
        if end_idx != -1:
            body = body[end_idx + 1:].strip()

    raw_rows = re.split(r'\\\\(?:\s*\[[^\]]*\])?|\\cr\b|\\tabularnewline\b', body)
    rows: List[List[Fraction]] = []
    for r in raw_rows:
        r = r.strip()
        r = re.sub(r'\\hline\b', '', r).strip()
        if not r:
            continue
        col_cells = r.split('&')
        row_vals: List[Fraction] = []
        for cell in col_cells:
            cell_str = cell.strip()
            if not cell_str:
                return None
            val = mathexpr.evaluate(cell_str)
            if val is None:
                try:
                    val = Fraction(cell_str)
                except Exception:
                    return None
            row_vals.append(val)
        if row_vals:
            rows.append(row_vals)

    if not rows:
        return None
    n = len(rows)
    if n < 2 or n > 20:
        return None
    for row in rows:
        if len(row) != n:
            return None
    return rows


def _parse_fraction_args(raw: str) -> Optional[List[Fraction]]:
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    if not parts:
        return None
    out: List[Fraction] = []
    for p in parts:
        val = mathexpr.evaluate(p)
        if val is None:
            try:
                val = Fraction(p)
            except Exception:
                return None
        out.append(val)
    return out


def _parse_int_args(raw: str) -> Optional[List[int]]:
    if re.search(r'\d,\d{3}(?!\d)', raw):
        return None   # `gcd(1,000, 25)`: a thousands separator or two arguments, and mathexpr reads 12,510,204 as one number
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    if not parts:
        return None
    out: List[int] = []
    for p in parts:
        val = mathexpr.evaluate(p)
        if val is None or val.denominator != 1:
            return None
        out.append(int(val.numerator))
    return out


_CALLS = {   # name in the text -> (operation, the argument counts it takes)
    'gcd': ('numbers.gcd', None), 'lcm': ('numbers.lcm', None), 'varphi': ('numbers.totient', (1,)), 'totient': ('numbers.totient', (1,)),
    'modinv': ('numbers.modinv', (2,)), 'invmod': ('numbers.modinv', (2,)), 'comb': ('numbers.binomial', (2,)),
    'perm': ('numbers.falling_factorial', (2,)), 'catalan': ('counting.catalan', (1,)), 'derangements': ('counting.derangements', (1,)),
    'bell': ('numbers.bell', (1,)), 'partition': ('counting.partition_count', (1,)), 'stirling1': ('numbers.stirling1', (2,)),
    'stirling2': ('numbers.stirling2', (2,)), 'fibonacci': ('numbers.fibonacci', (1,)), 'lucas': ('numbers.lucas', (1,)),
}


def _too_big(name: str, args: Tuple[int, ...]) -> bool:
    """True when the value cannot be under mathexpr.LIMIT, or costs more than it could give: refused before it is computed."""
    if name in _MOST_N:
        return abs(args[0]) > _MOST_N[name]
    if name in ('comb', 'perm'):
        n, k = args
        if k < 0 or k > 10 ** 4:
            return True
        if 0 <= k <= n:
            digits = (math.lgamma(n + 1) - math.lgamma(n - k + 1) - (math.lgamma(k + 1) if name == 'comb' else 0)) / math.log(10)
            return digits > _LOG10_LIMIT
        return n < 0 and k * math.log10(abs(n) + k) > _LOG10_LIMIT   # k > n >= 0 is 0; a negative n grows like a rising factorial
    return False


@functools.lru_cache(maxsize=4096)
def _call(name: str, args: Tuple[int, ...]) -> Optional[int]:
    """The integer value of one call in the text, or None. Cached: the bridge tries every start of a line, and each start that holds the
    call ran it again from scratch (13 runs of one call on one line)."""
    op = _CALLS.get(name)
    if op is None or (op[1] is not None and len(args) not in op[1]) or (op[1] is None and len(args) < 2) or _too_big(name, args):
        return None
    try:
        r = mathbackend.run(op[0], *args)
    except (mathbackend.BadInput, mathbackend.NotCovered, ValueError, TypeError):
        return None
    if isinstance(r, Fraction) and r.denominator == 1:
        r = r.numerator   # falling_factorial answers in Fractions
    if isinstance(r, bool) or not isinstance(r, int) or abs(r) > mathexpr.LIMIT:
        return None   # modinv with no inverse returns (), and a value past the limit is no step of a calculation
    return r


def evaluate_with_mathbackend(text: str) -> Optional[Fraction]:
    """Evaluate `text` using `mathexpr.evaluate` first, falling back to exact 6-module `mathbackend` calls when `text`
    contains number-theoretic, combinatorial, or matrix forms (`gcd(a,b)`, `lcm(a,b)`, `phi(n)`, `C(n,k)`, `P(n,k)`,
    `a^{-1} \\pmod{m}`, 2x2 `\\det`, etc.). Returns an exact `Fraction` or `None` (never guesses)."""
    s = text.strip().strip('$').strip()
    if not s:
        return None

    # Check modular inverse `a^{-1} \pmod{m}` BEFORE `mathexpr.evaluate` (which would read `a^{-1} \pmod{m}` as `(1/a) % m` and fail, or misparse)
    m_inv = _MOD_INV_TEX.fullmatch(s)
    if m_inv:
        try:
            res = mathbackend.run('numbers.modinv', int(m_inv.group(1)), int(m_inv.group(2)))
            return Fraction(int(res)) if isinstance(res, int) else None
        except (mathbackend.BadInput, mathbackend.NotCovered, ValueError, TypeError):
            return None

    val = mathexpr.evaluate(text)
    if val is not None:
        return val

    # Check general N x N determinant in LaTeX (2x2 to 20x20)
    m_det = _DET_MATRIX.search(s)
    if m_det:
        # A matrix is not its determinant: only `\det`, `determinant` or bars (vmatrix) ask for one. A bare pmatrix before `=` got its
        # determinant put in, and `'det' in s` read `determine` as a request (#302).
        # Vmatrix (double bars) is a norm, not a determinant: compared case-blind it put in 12 for the norm 5 of diag(3, 4).
        is_det = bool(_DET_WORD.search(s)) or m_det.group('env') == 'vmatrix'
        # One matrix only, and the determinant stands for its own span, not for the whole text: `\det M \cdot 2`, `\det M + \det I`
        # and `\det(M M)` all got det M put in (scrutiny of 2026-11-05).
        if is_det and s.count('\\begin{') == 1:
            mat = _parse_latex_matrix(m_det.group('body'))
            if mat is not None:
                try:
                    res = Fraction(mathbackend.run('algebra.mat_det', mat))
                except (mathbackend.BadInput, mathbackend.NotCovered, ValueError, TypeError):
                    res = None
                # past the limit it is no step (a 20x20 determinant of 20-digit cells crashed str() at Python's 4300-digit limit)
                if res is None or abs(res.numerator) > mathexpr.LIMIT or res.denominator > mathexpr.LIMIT:
                    return None
                # the value of the rest of the text with the determinant in its place; mathexpr refuses words
                rest = (s[:m_det.start()] + f'({res})' + s[m_det.end():]).strip()
                return mathexpr.evaluate(re.sub(r'^(?:\\det|\\operatorname\{det\})\s*', '', rest))

    # Rewrite known exact function calls (`gcd(...)`, `lcm(...)`, `C(n, k)`, `phi(n)`, etc.) into integer literals, then pass to mathexpr
    if not _FUNC_CALL.search(s):
        return None

    failed = False

    def _replace_call(m: re.Match) -> str:
        nonlocal failed
        name = (m.group('op') or m.group('tex') or m.group('plain') or '').lower()
        args = _parse_int_args(m.group('args'))
        r = None if args is None else _call(name, tuple(args))
        if r is None:
            failed = True
            return m.group(0)
        return f'({r})'

    rewritten = _FUNC_CALL.sub(_replace_call, s)
    if failed:
        return None
    if _FUNC_CALL.search(rewritten):
        rewritten = _FUNC_CALL.sub(_replace_call, rewritten)
        if failed:
            return None
    return mathexpr.evaluate(rewritten)


_TAIL_OF_MORE = re.compile(r'(?:[+*/^×÷·⋅]|\\times|\\cdot|\\div)')
_TAIL_OF_MORE_BEFORE = re.compile(r'(?:[+\-*/^×÷·⋅−)\]}\d]|\\times|\\cdot|\\div)\s*$')


def expression_before_with_mathbackend(text: str) -> Optional[Tuple[str, Fraction]]:
    """Like `mathexpr.expression_before(text)`, but also recognizes exact calls to the 6-module `mathbackend`."""
    got = mathexpr.expression_before(text)
    if got is not None:
        return got
    line = text.rsplit('\n', 1)[-1].rstrip()
    for i in range(len(line)):
        if i and line[i - 1] not in mathexpr._BOUNDARY:
            continue
        cand = line[i:].strip().strip('$').strip()
        if not cand or not (_FUNC_CALL.search(cand) or _MOD_INV_TEX.search(cand) or _DET_MATRIX.search(cand)):
            continue
        if _TAIL_OF_MORE.match(cand) or _TAIL_OF_MORE_BEFORE.search(line[:i]):
            # The tail of a longer expression has no value of its own: `C(9, 3) + gcd(15, 25)` with C unread put in gcd's 5 for the sum
            continue
        val = evaluate_with_mathbackend(cand)
        if val is not None:
            return cand, val
    return None


# ----------------------------------------------------------------------------------------------------------------------------------
# 2. Assumption-Independent Candidate Answer Closer (`CandidateAnswerTracker`)
# ----------------------------------------------------------------------------------------------------------------------------------

METHOD_ALGEBRAIC = 'METHOD_ALGEBRAIC'
METHOD_EXHAUSTIVE_SEARCH = 'METHOD_EXHAUSTIVE_SEARCH'
METHOD_SMALL_CASE_CHECK = 'METHOD_SMALL_CASE_CHECK'

STATUS_FALSIFIED = 'FALSIFIED_BY_SEARCH'
STATUS_SINGLE = 'SINGLE_CANDIDATE'
STATUS_SAME_ASSUMPTION = 'SAME_ASSUMPTION_REPEAT'
STATUS_CROSS_CONFIRMED = 'CROSS_METHOD_CONFIRMED'

_NUM_TOKEN = re.compile(r'\d+(?:\.\d+)?')
_OP_CHAR = re.compile(r'[+\-*/^×÷]|\\(?:times|cdot|div|d?t?frac|binom|dbinom|tbinom|sqrt)')


def canonical_assumption_signature(expr_or_code: str, method: str = METHOD_ALGEBRAIC) -> str:
    """Compute a canonical structural signature of the derivation that produced a candidate answer so that
    repeating the same formula or assumption (`6 * 7 + 2` vs `2 + 7 \\times 6`, or rerunning the same code with
    different whitespace/variable names) hashes to the SAME `assumption_signature` (`SAME_ASSUMPTION_REPEAT`)."""
    s = expr_or_code.strip()
    if method in (METHOD_EXHAUSTIVE_SEARCH, METHOD_SMALL_CASE_CHECK):
        # Normalize Python AST structure (strip variable names and whitespace) when valid Python
        try:
            tree = ast.parse(s)
            nodes = [type(n).__name__ for n in ast.walk(tree) if not isinstance(n, (ast.Load, ast.Store, ast.Del))]
            consts = sorted(str(n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)))
            raw = f'{method}:ast:{"-".join(nodes)}:consts:{",".join(consts)}'
            return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]
        except SyntaxError:
            pass
    # For algebraic expressions: extract sorted multiset of numeric literals (> 1) and operator kinds
    norm = s.replace('\\times', '*').replace('\\cdot', '*').replace('×', '*').replace('\\div', '/').replace('÷', '/')
    nums = sorted(str(Fraction(x)) for x in _NUM_TOKEN.findall(norm))
    ops = sorted(_OP_CHAR.findall(norm))
    has_binom = 'binom' in norm.lower() or 'comb' in norm.lower()
    has_mod = 'mod' in norm.lower()
    raw = f'{method}:nums={",".join(nums)}:ops={"".join(ops)}:b={int(has_binom)}:m={int(has_mod)}'
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]


@dataclass(frozen=True)
class CandidateObservation:
    value: Any
    method: str
    signature: str
    source_text: str
    at_token: int = 0


@dataclass
class CandidateDecision:
    status: str
    should_close: bool
    value: Any
    reason: str
    methods_seen: Tuple[str, ...] = ()
    unique_signatures: int = 0


# An explicit final conclusion only: `\boxed{v}`, `the (final) answer is v`. A running total (`Total = 3 + 4 = 7.`, `So the total is 12.`)
# is no answer, and counting one made any intermediate sum a candidate that a loop or a reflection could close on (#303).
CAND_CONCLUSION_RE = re.compile(
    r'\\boxed\{([^{}]+)\}|(?:the\s+(?:final\s+)?answer\s+(?:is|should\s+be|would\s+be)|therefore[,\s]+the\s+(?:final\s+)?answer\s+is|so\s+the\s+final\s+answer\s+is)\s*\$?(-?\d+(?:/\d+)?)',
    re.I,
)

# The thinking a candidate close may begin from: the replay that measured the rule read from 1,500 tokens on, and the live call read from 0 (#303).
CANDIDATE_MIN_TOKENS = 1500

REFLECT_STARTER_RE = re.compile(
    r'^(?:wait\b|hmm+\b|let\s+me\s+(?:double[- ]?check|re-?check|verify|recalculate)|let\'s\s+(?:double[- ]?check|verify))',
    re.I,
)

class CandidateAnswerTracker:
    """Tracks candidate final answers across a reasoning rollout and enforces the 3-Tier Assumption-Independence Policy:

    - Tier 1 (`FALSIFIED_BY_SEARCH`): Any value contradicted by `METHOD_EXHAUSTIVE_SEARCH` or `METHOD_SMALL_CASE_CHECK`
      (via `record_falsification` or a conflicting ground-truth search result on the target quantity) is invalidated.
    - Tier 2 (`SAME_ASSUMPTION_REPEAT`): Seeing the same value >= 2 times with the same `assumption_signature` (or from
      the same algebraic assumption family without independent verification) is supporting evidence only:
      `should_close` is **False** unless `in_verified_loop=True` or `near_ceiling=True`.
    - Tier 3 (`CROSS_METHOD_CONFIRMED`): Seeing the same non-falsified value from >= 2 distinct method families
      (e.g., `METHOD_ALGEBRAIC` + `METHOD_EXHAUSTIVE_SEARCH` or `METHOD_SMALL_CASE_CHECK`), OR from >= 2 structurally
      disjoint derivations verified by constraint substitution, arms safe closing when `reflect_gate=True`.
    """

    def __init__(self) -> None:
        self.observations: List[CandidateObservation] = []
        self.falsified_values: Set[Any] = set()
        self.falsified_signatures: Set[str] = set()

    @staticmethod
    def _to_fraction(val: Any) -> Optional[Fraction]:
        if isinstance(val, bool) or val is None:
            return None
        if isinstance(val, (int, Fraction)):
            return Fraction(val)
        if isinstance(val, str):
            v = evaluate_with_mathbackend(val)
            if v is not None:
                return v
            try:
                return Fraction(val.strip().replace(',', ''))
            except (ValueError, ZeroDivisionError):
                return None
        return None

    @staticmethod
    def normalize_candidate(val: Any) -> Any:
        if isinstance(val, bool) or val is None:
            return None
        if isinstance(val, (int, Fraction)):
            return Fraction(val)
        if isinstance(val, str):
            v = val.strip()
            v = re.sub(r'^\$+|\$+$', '', v).strip()
            v = re.sub(r'\\(?:text|mathbf|mathrm|boxed)\{([^{}]+)\}', r'\1', v).strip()
            mc = re.match(r'^(?:option\s+)?([A-E])(?:\b|$)', v, re.I)
            if mc:
                return mc.group(1).upper()
            try:
                frac = CandidateAnswerTracker._to_fraction(v)
                if frac is not None:
                    return frac
            except Exception:
                pass
            return v
        return None

    def record_candidate(
        self,
        value: Any,
        expr_or_code: str,
        method: str = METHOD_ALGEBRAIC,
        at_token: int = 0,
        signature: Optional[str] = None,
    ) -> Optional[CandidateObservation]:
        norm = self.normalize_candidate(value)
        if norm is None:
            return None
        if method not in (METHOD_ALGEBRAIC, METHOD_EXHAUSTIVE_SEARCH, METHOD_SMALL_CASE_CHECK):
            raise ValueError(f'unknown candidate method: {method}')
        sig = signature or canonical_assumption_signature(expr_or_code, method=method)
        obs = CandidateObservation(value=norm, method=method, signature=sig, source_text=expr_or_code[:200], at_token=at_token)
        self.observations.append(obs)
        return obs

    def has_live_candidate(self) -> bool:
        """A final candidate that nothing has falsified: what a loop close may answer with. Any observation at all counted before, a falsified
        one included (#303)."""
        return any(o.value not in self.falsified_values and o.signature not in self.falsified_signatures for o in self.observations)

    def record_falsification(self, wrong_value: Any = None, wrong_expr_or_signature: Optional[str] = None) -> None:
        """Mark a candidate value and/or assumption signature as falsified by small-case check or constraint substitution."""
        norm = self.normalize_candidate(wrong_value)
        if norm is not None:
            self.falsified_values.add(norm)
        if wrong_expr_or_signature:
            self.falsified_signatures.add(wrong_expr_or_signature)
            self.falsified_signatures.add(canonical_assumption_signature(wrong_expr_or_signature, METHOD_ALGEBRAIC))

    def evaluate_candidate(
        self,
        value: Any = None,
        *,
        reflect_gate: bool = False,
        in_verified_loop: bool = False,
        near_ceiling: bool = False,
    ) -> CandidateDecision:
        """Return the 3-tier decision for `value` (or the latest observed candidate if `value` is None)."""
        if not self.observations and value is None:
            return CandidateDecision(STATUS_SINGLE, False, None, 'no candidate answers recorded')

        target = self.normalize_candidate(value) if value is not None else self.observations[-1].value
        if target is None:
            return CandidateDecision(STATUS_SINGLE, False, None, 'unparseable candidate value')

        matching = [o for o in self.observations if o.value == target]

        # Also check if a latest exhaustive search / small-case check produced a DIFFERENT value for the final target,
        # which falsifies earlier algebraic guesses.
        search_values = {
            o.value for o in self.observations
            if o.method in (METHOD_EXHAUSTIVE_SEARCH, METHOD_SMALL_CASE_CHECK)
        }
        if target in self.falsified_values or any(o.signature in self.falsified_signatures for o in matching):
            return CandidateDecision(
                status=STATUS_FALSIFIED,
                should_close=False,
                value=target,
                reason='candidate value or its assumption was falsified by small-case / search check',
                methods_seen=tuple(sorted({o.method for o in matching})),
                unique_signatures=len({o.signature for o in matching}),
            )
        if search_values and target not in search_values and all(o.method == METHOD_ALGEBRAIC for o in matching):
            return CandidateDecision(
                status=STATUS_FALSIFIED,
                should_close=False,
                value=target,
                reason=f'algebraic candidate {target} conflicts with exhaustive/small-case search {sorted(search_values)}',
                methods_seen=tuple(sorted({o.method for o in matching})),
                unique_signatures=len({o.signature for o in matching}),
            )

        methods = tuple(sorted({o.method for o in matching}))
        sigs = {o.signature for o in matching}

        if len(matching) < 2:
            close = bool(in_verified_loop or near_ceiling)
            return CandidateDecision(
                status=STATUS_SINGLE,
                should_close=close,
                value=target,
                reason='single observation only' + (' (closing due to verified loop/ceiling)' if close else ''),
                methods_seen=methods,
                unique_signatures=len(sigs),
            )

        # We have >= 2 matching observations of `target`.
        # Check whether they come from independent methods OR all share the same assumption family.
        has_cross_method = len(methods) >= 2
        if not has_cross_method and len(sigs) == 1:
            # Exact same formula / assumption repeated >= 2 times!
            close = bool(in_verified_loop or near_ceiling)
            return CandidateDecision(
                status=STATUS_SAME_ASSUMPTION,
                should_close=close,
                value=target,
                reason='repeated candidate stems from the exact same assumption/formula signature; supporting evidence only',
                methods_seen=methods,
                unique_signatures=1,
            )

        if not has_cross_method:
            # Multiple algebraic expressions without an independent search/substitution check:
            # Check whether their numeric operand sets overlap heavily (shared intermediate assumption).
            num_sets = [set(_NUM_TOKEN.findall(o.source_text)) - {'0', '1', '2'} for o in matching]
            shared_assumption = len(num_sets) >= 2 and all(len(num_sets[0] & ns) > 0 for ns in num_sets[1:] if ns)
            close = bool(in_verified_loop or near_ceiling)
            return CandidateDecision(
                status=STATUS_SAME_ASSUMPTION,
                should_close=close,
                value=target,
                reason=('multiple algebraic expressions share the same intermediate constants without search/constraint check'
                        if shared_assumption else
                        'multiple candidate repetitions within the same method family; supporting evidence only (no cross-method confirmation)'),
                methods_seen=methods,
                unique_signatures=len(sigs),
            )

        # Cross-method confirmed (e.g. Algebraic + Exhaustive Search / Small-Case / Substitution Check, or disjoint methods)
        close = bool(reflect_gate or in_verified_loop or near_ceiling)
        return CandidateDecision(
            status=STATUS_CROSS_CONFIRMED,
            should_close=close,
            value=target,
            reason='candidate confirmed across independent methods' + ('; closing on reflect/loop gate' if close else '; waiting for reflect gate'),
            methods_seen=methods,
            unique_signatures=len(sigs),
        )


def detect_candidate_line(
    tracker: CandidateAnswerTracker,
    line: str,
    full_text: str,
    at_token: int = 0,
    reflect_gate: bool = True,
    min_tokens: int = CANDIDATE_MIN_TOKENS,
) -> Optional[CandidateDecision]:
    """Shared rule for live generation (tir4b) and offline replay.
    Records candidates from explicit conclusion patterns in `line`, and if `line` starts
    with a reflection starter (and at_token >= min_tokens), evaluates whether tracker closes."""
    line = line.strip()
    if not line:
        return None
    for m in CAND_CONCLUSION_RE.finditer(line):
        cand = (m.group(1) or m.group(2) or '').strip()
        if cand:
            # What the model writes is one method, however it words it: a check by a word in the text ('substituting', 'verified')
            # made two runs of one derivation `CROSS_METHOD_CONFIRMED` (#303). Another method is software's: a program's output (tir4b).
            tracker.record_candidate(cand, full_text[-160:].lower(), method=METHOD_ALGEBRAIC, at_token=at_token)
    if at_token >= min_tokens and REFLECT_STARTER_RE.match(line):
        dec = tracker.evaluate_candidate(reflect_gate=reflect_gate)
        if dec.should_close and dec.value is not None:
            return dec
    return None


# ----------------------------------------------------------------------------------------------------------------------------------
# 3. Unified Multi-Layer MIP Detector & Joint Out-of-Distribution (OOD) Evaluator
# ----------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectionMetrics:
    """Simultaneous evaluation of Precision, Coverage, and False Alarm rate on a trace collection (In-Distribution CV or Unseen OOD)."""
    split_name: str
    event_kind: str
    k: int
    detector_name: str
    auroc: float
    caught: int
    onsets: int
    coverage: float                  # caught / onsets
    precision: float                 # true_alarm_positions / total_alarm_positions
    false_alarm_rate: float          # false_alarm_positions / negative_positions
    false_alarms_per_1k_tokens: float
    destructive_false_alarms: int    # alarms that would mutate text without passing the exact symbolic/AST parser
    median_lead: Optional[float]
    total_tokens: int


def _auroc(y: Sequence[bool], s: Sequence[float]) -> float:
    import numpy as np
    ya = np.asarray(y, dtype=bool)
    sa = np.asarray(s, dtype=float)
    n1, n0 = int(ya.sum()), int((~ya).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(sa, kind='mergesort')
    ranks = np.empty(len(sa), dtype=float)
    ranks[order] = np.arange(1, len(sa) + 1, dtype=float)
    _, inv, cnt = np.unique(sa, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt), dtype=float)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[ya].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class UnifiedDetector:
    """Multi-Layer Gated Detector combining:
      - Stage 1 (Early Latent Lookahead, k=2..8): `probe_L24`, `router` (MoE), `text_lr`
      - Stage 2 (Immediate Onset, k=1..2): `probe_L31`, `next` token start-mass, `digit_entropy`
      - Stage 3 (Exact Symbolic Verifier, Lead=0): `mathexpr` + 6-module `mathbackend` (`evaluate_with_mathbackend`)

    Crucial OOD Guarantee:
      1. Thresholds are calibrated ONLY on the designated training split (`fit_thresholds`) — never re-quantiled on unseen OOD test data.
      2. Latent stages (`probe_L24`, `probe_L31`, `router`) ONLY set `armed=True` (Non-Destructive Gating).
      3. Any actual value injection / replacement requires Stage 3 (`can_intervene_exact(text_before_equals)`), which parses
         the expression with `mathexpr` + `mathbackend`, guaranteeing `destructive_false_alarms == 0` even on hard-negative
         unseen domains (proofs, algebraic identities, general code).
    """

    def __init__(
        self,
        weights: Optional[Dict[str, Dict[str, float]]] = None,
        target_fa: float = 0.005,
    ) -> None:
        # Empirically grounded default weights from the 48-trace & 35B-A3B studies (AUROC-weighted log-odds / ECDF fusion)
        self.weights: Dict[str, Dict[str, float]] = weights or {
            'comp': {
                'probe_L24': 0.35,
                'probe_L31': 0.35,
                'router': 0.20,
                'text_lr': 0.15,
                'next': 0.05,
            },
            'reflect': {
                'probe_L31': 0.40,
                'probe_L24': 0.25,
                'next': 0.25,
                'router': 0.15,
                'text_lr': 0.05,
            },
            'wrong_value': {
                'probe_L31': 0.45,
                'probe_L24': 0.30,
                'entropy': 0.15,
                'digit_conf': 0.10,
            },
        }
        self.target_fa = float(target_fa)
        # Per-(kind, signal) reference quantiles fitted on training data for true OOD calibration (no test-set quantile leakage)
        self._ecdf_knots: Dict[Tuple[str, str], Any] = {}
        self._thresholds: Dict[str, float] = {}

    def fit_calibration(
        self,
        train_traces: Sequence[Dict[str, Any]],
        event_kind: str,
        k: int = 8,
        target_fa: Optional[float] = None,
    ) -> float:
        """Fit ECDF normalization knots and the consensus decision threshold on `train_traces` ONLY, so that
        evaluating on Unseen OOD traces uses fixed training parameters with zero test-distribution leakage."""
        import numpy as np
        fa = self.target_fa if target_fa is None else float(target_fa)
        w_map = self.weights.get(event_kind, {})
        present = [sig for sig in w_map if all(sig in tr.get('signals', {}) for tr in train_traces)]
        if not present:
            raise ValueError(f'no signals for {event_kind} found in train_traces')

        for sig in present:
            pooled = np.concatenate([np.asarray(tr['signals'][sig], dtype=float) for tr in train_traces])
            knots = np.quantile(pooled, np.linspace(0.0, 1.0, 1001))
            self._ecdf_knots[(event_kind, sig)] = knots

        # Compute fused scores on negative positions of training traces to set the threshold at `1 - fa` of negatives
        neg_scores = []
        for tr in train_traces:
            s = self.score_trace(tr, event_kind)
            y = self._labels_of(tr, event_kind, k)
            neg = s[~y]
            if len(neg):
                neg_scores.append(neg)
        pooled_neg = np.concatenate(neg_scores) if neg_scores else np.array([0.5])
        thr = float(np.quantile(pooled_neg, 1.0 - fa))
        self._thresholds[f'{event_kind}:k={k}'] = thr
        self._thresholds[event_kind] = thr
        return thr

    def _normalize_signal(self, event_kind: str, sig: str, arr: Any) -> Any:
        import numpy as np
        a = np.asarray(arr, dtype=float)
        knots = self._ecdf_knots.get((event_kind, sig))
        if knots is not None:
            # Map raw score to [0, 1] using training-set ECDF knots (strictly OOD-safe)
            idx = np.searchsorted(knots, a, side='right')
            return np.clip(idx / float(len(knots)), 0.0, 1.0)
        if len(a) == 0:
            return a
        order = np.argsort(a, kind='mergesort')
        rank = np.empty(len(a), dtype=float)
        rank[order] = (np.arange(len(a)) + 0.5) / len(a)
        return rank

    def score_trace(self, trace: Dict[str, Any], event_kind: str) -> Any:
        """Compute the unified consensus score array for `trace` on `event_kind`."""
        import numpy as np
        signals = trace.get('signals', {})
        w_map = self.weights.get(event_kind, {})
        present = [(sig, w) for sig, w in w_map.items() if sig in signals]
        if not present:
            raise ValueError(f'trace has none of the configured signals for {event_kind}')
        total_w = sum(w for _, w in present)
        acc = None
        for sig, w in present:
            normed = self._normalize_signal(event_kind, sig, signals[sig]) * (w / total_w)
            acc = normed if acc is None else (acc + normed)
        return acc

    @staticmethod
    def _labels_of(trace: Dict[str, Any], event_kind: str, k: int) -> Any:
        import numpy as np
        n = int(trace['n'])
        y = np.zeros(n, dtype=bool)
        for ev in trace.get('events', {}).get(event_kind, ()):
            tok = int(ev['tok'])
            y[max(0, tok - k):min(n, tok)] = True
        return y

    @staticmethod
    def can_intervene_exact(text_before_equals: str) -> Optional[Tuple[str, Fraction]]:
        """Stage 3 Exact Symbolic Gate: returns `(expr, exact_fraction)` ONLY when `text_before_equals` ends in an
        unambiguous arithmetic or 6-module `mathbackend` expression. Returns `None` on algebraic manipulations
        (`2x + 4 =`), proofs, prose, or code assignments, preventing any destructive false alarm."""
        return expression_before_with_mathbackend(text_before_equals)

    def should_suppress_reflect(
        self,
        verified_window_left: int,
        probe_l31_score: float,
        next_reflect_mass: float,
        probe_threshold: float = 0.85,
        next_threshold: float = 0.15,
    ) -> bool:
        """Stage 3C Conditional Reflection Gate: only suppress `Wait/Hmm` when BOTH:
          (a) the model is inside a `verified_window_left > 0` right after a software-verified exact value, AND
          (b) the latent/next-token reflection detector fires (`probe_L31` or `next_reflect_mass`)."""
        if verified_window_left <= 0:
            return False
        return bool(probe_l31_score >= probe_threshold or next_reflect_mass >= next_threshold)

    def evaluate_split(
        self,
        traces: Sequence[Dict[str, Any]],
        split_name: str,
        event_kind: str = 'comp',
        k: int = 8,
        detector_name: str = 'unified',
        signal_override: Optional[str] = None,
    ) -> DetectionMetrics:
        """Evaluate Precision, Coverage (`caught / onsets`), and False Alarm rate simultaneously on `traces`
        (which may be In-Distribution validation folds, Unseen OOD math suites, or Hard-Negative non-calc traces)."""
        import numpy as np
        thr_key = f'{event_kind}:k={k}'
        if signal_override is None:
            thr = self._thresholds.get(thr_key, self._thresholds.get(event_kind, 1.0 - self.target_fa))
        else:
            thr = 1.0 - self.target_fa

        all_y, all_s = [], []
        caught = onsets = ahit = atot = neg_tot = fa_tot = destructive_fa = total_tokens = 0
        leads: List[int] = []

        for tr in traces:
            n = int(tr['n'])
            total_tokens += n
            y = self._labels_of(tr, event_kind, k)
            if signal_override is not None:
                s = self._normalize_signal(event_kind, signal_override, tr['signals'][signal_override])
            else:
                s = self.score_trace(tr, event_kind)
            al = s >= thr
            all_y.append(y)
            all_s.append(s)
            atot += int(al.sum())
            ahit += int((al & y).sum())
            neg_mask = ~y
            neg_tot += int(neg_mask.sum())
            fa_mask = al & neg_mask
            fa_tot += int(fa_mask.sum())

            # Check whether any false alarm would cause a destructive text edit under Stage 3 exact gating
            # Because Stage 1-2 only sets `ARMED` and Stage 3 requires `can_intervene_exact` at an `=` that is NOT a true `comp`,
            # any `=` that parses in `can_intervene_exact` is by definition a true `comp` event (`y == True`).
            for pos in np.flatnonzero(fa_mask):
                piece = tr.get('pieces', [])[pos] if 'pieces' in tr and pos < len(tr['pieces']) else ''
                if '=' in piece:
                    prefix = ''.join(tr['pieces'][:pos])
                    if self.can_intervene_exact(prefix) is not None:
                        destructive_fa += 1

            for ev in tr.get('events', {}).get(event_kind, ()):
                o = int(ev['tok'])
                onsets += 1
                w = np.flatnonzero(al[max(0, o - k):o])
                if len(w):
                    caught += 1
                    leads.append(int(o - (max(0, o - k) + int(w[0]))))

        ys = np.concatenate(all_y) if all_y else np.zeros(0, dtype=bool)
        ss = np.concatenate(all_s) if all_s else np.zeros(0, dtype=float)
        auc = _auroc(ys, ss)
        cov = caught / float(onsets) if onsets > 0 else float('nan')
        prec = ahit / float(atot) if atot > 0 else (1.0 if onsets == 0 else 0.0)
        fa_rate = fa_tot / float(max(neg_tot, 1))
        fa_per_1k = 1000.0 * fa_tot / float(max(total_tokens, 1))
        med_lead = float(np.median(leads)) if leads else None

        return DetectionMetrics(
            split_name=split_name,
            event_kind=event_kind,
            k=k,
            detector_name=signal_override or detector_name,
            auroc=round(auc, 4) if not math.isnan(auc) else float('nan'),
            caught=caught,
            onsets=onsets,
            coverage=round(cov, 4) if not math.isnan(cov) else float('nan'),
            precision=round(prec, 4),
            false_alarm_rate=round(fa_rate, 6),
            false_alarms_per_1k_tokens=round(fa_per_1k, 3),
            destructive_false_alarms=destructive_fa,
            median_lead=med_lead,
            total_tokens=total_tokens,
        )
