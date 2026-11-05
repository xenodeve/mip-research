"""ood_metrics.py: Out-of-Distribution (OOD) evaluation metrics and fixed-train calibration (#298).

Provides:
- SignalCalibration: fits fixed ECDF quantile knots and consensus threshold on TRAIN positions only.
- score_fusions_ood: normalizes test member scores through TRAIN knots before averaging (zero test-distribution leakage).
- compute_no_digit_mask: hard-negative prose mask (positions in lines with no digits).
- count_destructive_false_alarms: false comp alarms where software would wrongly edit text at next '=' within k tokens.
- evaluate_ood_detector: computes AUROC, caught/onsets, precision, FA/1k tokens, alarm rate & drift, median lead.
"""
import bisect
import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


def auroc(y: Sequence[bool], s: Sequence[float]) -> float:
    """Exact AUROC computation using rank sum."""
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


class SignalCalibration:
    """Stores training score quantiles / ECDF knots and fixed decision threshold for one signal."""

    def __init__(self, train_scores: np.ndarray, fa: float = 0.01, num_knots: int = 1001) -> None:
        self.fa = float(fa)
        arr = np.asarray(train_scores, dtype=float)
        if len(arr) == 0:
            self.train_threshold = 0.0
            self.knots_q = np.linspace(0.0, 1.0, num_knots)
            self.knots_v = np.zeros(num_knots)
        else:
            self.train_threshold = float(np.quantile(arr, 1.0 - self.fa))
            self.knots_q = np.linspace(0.0, 1.0, num_knots)
            self.knots_v = np.quantile(arr, self.knots_q)

    def map_to_percentile(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to [0, 1] percentiles using training ECDF knots."""
        s = np.asarray(scores, dtype=float)
        if len(s) == 0 or len(self.knots_v) == 0:
            return s
        return np.interp(s, self.knots_v, self.knots_q)


def fit_train_calibrations(
    train_scores_by_signal: Dict[str, Sequence[np.ndarray]],
    fusions_def: Dict[str, Optional[Sequence[str]]],
    fa: float = 0.01,
    num_knots: int = 1001,
) -> Dict[str, SignalCalibration]:
    """Fit SignalCalibration for every base signal and every fused signal on training scores."""
    calibs: Dict[str, SignalCalibration] = {}

    # 1. Base signals
    for name, trace_scores in train_scores_by_signal.items():
        pooled = np.concatenate(trace_scores) if trace_scores else np.zeros(0)
        calibs[name] = SignalCalibration(pooled, fa=fa, num_knots=num_knots)

    # 2. Fused signals on training data
    if train_scores_by_signal:
        first_sig = next(iter(train_scores_by_signal.values()))
        n_traces = len(first_sig)
    else:
        n_traces = 0

    for fname, members in fusions_def.items():
        names = [n for n in train_scores_by_signal if n != 'router'] if members is None else [m for m in members if m in train_scores_by_signal]
        if len(names) < 2:
            continue
        pct_by_member = {m: [calibs[m].map_to_percentile(s) for s in train_scores_by_signal[m]] for m in names}
        fused_traces = []
        for i in range(n_traces):
            trace_pcts = [pct_by_member[m][i] for m in names]
            fused_traces.append(np.mean(trace_pcts, axis=0))
        pooled_fused = np.concatenate(fused_traces) if fused_traces else np.zeros(0)
        calibs[fname] = SignalCalibration(pooled_fused, fa=fa, num_knots=num_knots)

    return calibs


def score_fusions_ood(
    test_scores_by_signal: Dict[str, Sequence[np.ndarray]],
    fusions_def: Dict[str, Optional[Sequence[str]]],
    calibs: Dict[str, SignalCalibration],
) -> Dict[str, Sequence[np.ndarray]]:
    """Compute fused test scores by mapping test member scores through training calibrations and averaging."""
    all_scores = dict(test_scores_by_signal)
    if not test_scores_by_signal:
        return all_scores
    first_sig = next(iter(test_scores_by_signal.values()))
    n_traces = len(first_sig)

    for fname, members in fusions_def.items():
        names = [n for n in test_scores_by_signal if n != 'router'] if members is None else [m for m in members if m in test_scores_by_signal]
        if len(names) < 2:
            continue
        if any(m not in calibs for m in names):
            continue
        pct_by_member = {m: [calibs[m].map_to_percentile(s) for s in test_scores_by_signal[m]] for m in names}
        fused_traces = []
        for i in range(n_traces):
            trace_pcts = [pct_by_member[m][i] for m in names]
            fused_traces.append(np.mean(trace_pcts, axis=0))
        all_scores[fname] = fused_traces

    return all_scores


def compute_no_digit_mask(r: Dict[str, Any]) -> np.ndarray:
    """Returns a boolean array of length r['n'] where mask[t] is True iff token t
    starts in a line that contains no digits (prose line)."""
    text = r.get('text', '')
    offs = r.get('offs', [])
    n = r.get('n', len(offs))
    if n == 0 or not text:
        return np.zeros(n, dtype=bool)

    lines = text.split('\n')
    line_has_digit = [bool(re.search(r'\d', ln)) for ln in lines]

    line_starts = []
    curr = 0
    for ln in lines:
        line_starts.append(curr)
        curr += len(ln) + 1  # accounts for '\n'

    mask = np.zeros(n, dtype=bool)
    for t in range(n):
        c = offs[t]
        line_idx = bisect.bisect_right(line_starts, c) - 1
        if 0 <= line_idx < len(line_has_digit):
            if not line_has_digit[line_idx]:
                mask[t] = True
    return mask


def count_destructive_false_alarms(
    r: Dict[str, Any],
    al_mask: np.ndarray,
    y_mask: np.ndarray,
    k: int,
    can_intervene_fn: Optional[Callable[[str], Any]] = None,
) -> int:
    """Count comp alarms that are false AND where can_intervene_fn(prefix) would return a value
    at the next '=' within k tokens (i.e. software would actually edit text wrongly)."""
    if can_intervene_fn is None:
        try:
            from tools.igc.unified_detect import UnifiedDetector
            can_intervene_fn = UnifiedDetector.can_intervene_exact
        except ImportError:
            try:
                from unified_detect import UnifiedDetector
                can_intervene_fn = UnifiedDetector.can_intervene_exact
            except ImportError:
                # Fallback to mathexpr expression_before if unified_detect is unavailable
                try:
                    from tools.igc.mathexpr import expression_before
                except ImportError:
                    try:
                        from mathexpr import expression_before
                    except ImportError:
                        return 0
                def can_intervene_fn(prefix: str) -> Optional[Any]:
                    try:
                        got = expression_before(prefix)
                        return got if got and got[1] is not None else None
                    except Exception:
                        return None

    fa_positions = np.flatnonzero(al_mask & ~y_mask)
    if len(fa_positions) == 0:
        return 0

    text = r.get('text', '')
    offs = r.get('offs', [])
    n = r.get('n', len(offs))
    destructive_count = 0

    # Legitimate comp event '=' positions
    comp_eq_positions = set()
    for e in r.get('events', {}).get('comp', []):
        if 'onset' in e:
            idx = text.find('=', e['onset'])
            if idx != -1:
                comp_eq_positions.add(idx)

    for t in fa_positions:
        c_start = offs[t]
        c_end = offs[min(t + k + 1, n - 1)] if t + k + 1 < n else len(text)
        eq_pos = text.find('=', c_start, c_end)
        if eq_pos != -1:
            # If the '=' belongs to a legitimate comp event, it is not editing text wrongly
            if eq_pos in comp_eq_positions:
                continue
            prefix = text[:eq_pos]
            val = can_intervene_fn(prefix)
            if val is not None:
                destructive_count += 1

    return destructive_count


def evaluate_ood_detector(
    test_rows: Sequence[Dict[str, Any]],
    kind: str,
    k: int,
    signal_name: str,
    test_scores: Sequence[np.ndarray],
    calib: SignalCalibration,
    can_intervene_fn: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    """Evaluate a single signal or fusion on test traces using fixed train threshold."""
    thr = calib.train_threshold
    all_y, all_s = [], []
    caught = 0
    onsets = 0
    leads = []
    atot = 0
    ahit = 0
    total_tokens = 0
    hard_neg_tokens = 0
    hard_neg_fa = 0
    destructive_fa_total = 0

    for r, s_ in zip(test_rows, test_scores):
        n = r['n']
        total_tokens += n
        y = np.zeros(n, dtype=bool)
        for e in r['events'][kind]:
            o = e['tok']
            y[max(0, o - k):o] = True

        s_ = np.asarray(s_, dtype=float)
        al = s_ >= thr
        all_y.append(y)
        all_s.append(s_)
        atot += int(al.sum())
        ahit += int((al & y).sum())

        for e in r['events'][kind]:
            o = e['tok']
            onsets += 1
            w = np.flatnonzero(al[max(0, o - k):o])
            if len(w):
                caught += 1
                leads.append(int(o - (max(0, o - k) + int(w[0]))))

        if kind == 'comp':
            # Hard negative readout: lines with no digits
            no_digit_mask = compute_no_digit_mask(r)
            hard_neg_tokens += int(no_digit_mask.sum())
            hard_neg_fa += int((al & ~y & no_digit_mask).sum())

            # Destructive false alarms
            destructive_fa_total += count_destructive_false_alarms(r, al, y, k, can_intervene_fn=can_intervene_fn)

    y_cat = np.concatenate(all_y) if all_y else np.zeros(0, dtype=bool)
    s_cat = np.concatenate(all_s) if all_s else np.zeros(0, dtype=float)
    auc = auroc(y_cat, s_cat)

    cov = caught / max(onsets, 1)
    prec = ahit / max(atot, 1)
    fa_tot = atot - ahit
    fa_per_1k = 1000.0 * fa_tot / max(total_tokens, 1)
    alarm_rate = atot / max(total_tokens, 1)
    alarm_drift = alarm_rate - calib.fa
    med_lead = float(np.median(leads)) if leads else None

    hard_neg_fa_per_1k = (1000.0 * hard_neg_fa / max(hard_neg_tokens, 1)) if kind == 'comp' else None

    return dict(
        kind=kind,
        k=k,
        signal=signal_name,
        train_thr=round(thr, 4),
        auroc=round(auc, 3) if not math.isnan(auc) else float('nan'),
        caught=caught,
        onsets=onsets,
        coverage=round(cov, 3),
        precision=round(prec, 3),
        fa_per_1k=round(fa_per_1k, 2),
        alarm_rate=round(alarm_rate, 4),
        alarm_drift=round(alarm_drift, 4),
        lead=round(med_lead, 1) if med_lead is not None else None,
        hard_neg_fa_per_1k=round(hard_neg_fa_per_1k, 2) if hard_neg_fa_per_1k is not None else None,
        destructive_fa=destructive_fa_total if kind == 'comp' else 0,
        total_tokens=total_tokens,
    )


def split_traces_by_suite(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group traces by suite prefix extracted from r['id'].split('-')[0]."""
    suites: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        suite = r['id'].split('-')[0]
        suites.setdefault(suite, []).append(r)
    return suites


def format_ood_row(res: Dict[str, Any], direction: str) -> str:
    """Format one OOD metric result into a readable report line."""
    lead_str = f"{res['lead']:.1f}" if res['lead'] is not None else "None"
    drift_str = f"({res['alarm_drift']:+.4f})"
    s = (
        f"{direction:20s} {res['kind']:7s} k={res['k']:2d} {res['signal']:22s} "
        f"AUROC {res['auroc']:.3f}  caught {res['caught']:3d}/{res['onsets']:3d} (cov {res['coverage']:.3f})  "
        f"lead {lead_str:4s}  prec {res['precision']:.3f}  FA/1k {res['fa_per_1k']:6.2f}  "
        f"alarm {res['alarm_rate']:.4f} {drift_str:9s}"
    )
    if res['kind'] == 'comp':
        s += f"  hard_neg {res['hard_neg_fa_per_1k']:6.2f}  dest_fa {res['destructive_fa']:d}"
    return s
