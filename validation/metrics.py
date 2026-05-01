"""
Evaluation metrics for VERDICT WEIGHT validation.

All metrics operate on flat numpy arrays of ``y_true`` (binary labels,
1 for adversarial, 0 for legitimate) and ``y_score`` (predicted
suppression scores, ``∈ [0, 1]``, where higher = more suppressed).

Convention: a "good" classifier produces high ``y_score`` for
adversarial samples and low ``y_score`` for legitimate ones.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def brier_score(y_true: Sequence[int], y_pred: Sequence[float]) -> float:
    """Mean squared error between predicted probabilities and labels.

    Lower is better; perfect calibration ⇒ Brier = 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))


def reliability(y_true: Sequence[int], y_pred: Sequence[float], n_bins: int = 10) -> float:
    """Reliability component of the Brier-score decomposition (REL).

    Lower is better. REL = 0 means the classifier is perfectly
    calibrated within bins.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    rel = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_pred >= lo) & (y_pred < hi) if i < n_bins - 1 else (y_pred >= lo) & (y_pred <= hi)
        n_k = int(np.sum(mask))
        if n_k == 0:
            continue
        avg_pred = float(np.mean(y_pred[mask]))
        avg_true = float(np.mean(y_true[mask]))
        rel += (n_k / n) * (avg_pred - avg_true) ** 2
    return float(rel)


def cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """Cohen's d effect size between two groups (pooled standard deviation).

    Returns negative if group_a mean is below group_b mean, positive
    otherwise. By convention we use group_a = adversarial,
    group_b = legitimate, so a competent suppression method should
    produce d < 0 (adversarial scored lower than legitimate, i.e.
    correctly suppressed) — except in this validation, ``y_score``
    represents the *certified CW*, so a lower value ON ADVERSARIAL is
    better, hence d < 0 = good.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    s2_a = float(np.var(a, ddof=1))
    s2_b = float(np.var(b, ddof=1))
    pooled = ((len(a) - 1) * s2_a + (len(b) - 1) * s2_b) / max(1, len(a) + len(b) - 2)
    pooled_sd = float(np.sqrt(pooled))
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Welch's two-sample t-test (unequal variances).

    Returns (t_statistic, two_sided_p_value).
    """
    from math import erf, sqrt

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    n_a, n_b = len(a), len(b)
    se = sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0
    t = (mean_a - mean_b) / se
    # Welch–Satterthwaite degrees of freedom
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = df_num / df_den if df_den > 0 else float(n_a + n_b - 2)
    # Two-sided p-value via Student's t survival function approximation.
    # We use the continuous CDF of the t distribution via an
    # incomplete beta approximation. For df > 30, the normal
    # approximation is sufficient for the precisions reported here.
    if df > 30:
        # Two-sided normal-approximation p
        z = abs(t)
        # Survival function 1 - Φ(z)
        sf = 0.5 * (1.0 - erf(z / sqrt(2)))
        return float(t), float(2 * sf)
    # Otherwise fall back to numerical integration of t pdf
    from math import gamma, pi
    def t_pdf(x: float, dof: float) -> float:
        coef = gamma((dof + 1) / 2) / (sqrt(dof * pi) * gamma(dof / 2))
        return coef * (1 + x * x / dof) ** (-(dof + 1) / 2)

    # Simpson's rule on |t|..50
    upper = max(abs(t) + 50, 60.0)
    n_pts = 4001
    xs = np.linspace(abs(t), upper, n_pts)
    ys = np.array([t_pdf(float(x), df) for x in xs])
    h = (upper - abs(t)) / (n_pts - 1)
    # Simpson weights
    weights = np.ones(n_pts)
    weights[1:-1:2] = 4
    weights[2:-1:2] = 2
    sf = float((h / 3.0) * np.sum(weights * ys))
    return float(t), float(2 * sf)


def confusion(y_true: Sequence[int], y_pred_binary: Sequence[int]) -> Dict[str, int]:
    """2x2 confusion matrix counts."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred_binary, dtype=int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def sensitivity(y_true: Sequence[int], y_pred_binary: Sequence[int]) -> float:
    c = confusion(y_true, y_pred_binary)
    denom = c["tp"] + c["fn"]
    return float(c["tp"] / denom) if denom > 0 else 0.0


def specificity(y_true: Sequence[int], y_pred_binary: Sequence[int]) -> float:
    c = confusion(y_true, y_pred_binary)
    denom = c["tn"] + c["fp"]
    return float(c["tn"] / denom) if denom > 0 else 0.0


def auc_roc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Area under ROC curve via the Mann-Whitney U statistic.

    Convention: y_score is the *adversarial* suppression score, i.e.
    higher means more suppressed. We compute AUC for the binary task
    "score correctly identifies adversarial samples" by treating
    1 - cw_certified as the adversarial-score (since lower CW for
    adversarial is the goal).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Wilcoxon-Mann-Whitney
    n_pos = len(pos)
    n_neg = len(neg)
    # Rank all scores
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, n_pos + n_neg + 1)
    sum_ranks_pos = float(np.sum(ranks[:n_pos]))
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def bootstrap_ci(
    values: Sequence[float], statistic, n_iter: int = 1000, alpha: float = 0.05, seed: int = 42
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval.

    Returns ``(point_estimate, ci_lower, ci_upper)``.
    """
    arr = np.asarray(values, dtype=float)
    point = float(statistic(arr))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots[i] = float(statistic(sample))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi


__all__ = [
    "brier_score",
    "reliability",
    "cohens_d",
    "welch_t_test",
    "confusion",
    "sensitivity",
    "specificity",
    "auc_roc",
    "bootstrap_ci",
]
