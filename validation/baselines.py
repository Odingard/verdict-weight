"""
Baseline fusion methods for IEEE head-to-head comparison.

Implements four reference multi-source fusion methods:

  * Dempster-Shafer Theory (DS) — classical evidence combination
  * Naive Bayes Fusion (NB) — independence-assumption maximum-likelihood
  * Simple Averaging (SA) — arithmetic mean
  * Max Voting (MV) — majority-vote binary fusion

Each baseline takes the same per-source signal vector that VERDICT
WEIGHT consumes (commercial-tier inputs ``{SR, CC, TD, HA}``) and
produces a single Consequence-Weight-equivalent score in ``[0, 1]``.

These baselines deliberately do NOT consume Streams 5–8 (CTC / SIS /
CPS / RIS) — that is the point of the head-to-head: VW's superiority
on adversarial inputs comes from the streams baselines lack.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def dempster_shafer(scores: Sequence[float]) -> float:
    """Dempster-Shafer combination of independent evidence masses.

    Treats each score as a mass on the proposition "true", with the
    remainder ``1 - score`` as mass on "uncertain" (no mass on
    "false", per Shafer's classical bba formulation).

    The combined belief is::

        bel = 1 - prod(1 - score_i)

    which is the standard DS rule for non-conflicting evidence
    against an uncertain frame of discernment.
    """
    if not scores:
        return 0.0
    bel_neg = 1.0
    for s in scores:
        bel_neg *= max(0.0, min(1.0, 1.0 - s))
    return float(np.clip(1.0 - bel_neg, 0.0, 1.0))


def naive_bayes(scores: Sequence[float], prior: float = 0.5) -> float:
    """Naive-Bayes posterior assuming source-conditional independence.

    Each ``score_i`` is interpreted as ``P(D | hypothesis)`` (likelihood
    that source i would produce this signal if the hypothesis is true)
    versus ``1 - score_i`` for the null. Combined posterior::

        P(H | D) = prior · ∏ score_i / (prior · ∏ score_i + (1-prior) · ∏ (1-score_i))
    """
    if not scores:
        return prior
    eps = 1e-12
    log_p_h = np.log(prior + eps)
    log_p_not_h = np.log(1.0 - prior + eps)
    for s in scores:
        s_ = float(np.clip(s, eps, 1.0 - eps))
        log_p_h += np.log(s_)
        log_p_not_h += np.log(1.0 - s_)
    # Numerically-stable softmax
    m = max(log_p_h, log_p_not_h)
    p = np.exp(log_p_h - m)
    q = np.exp(log_p_not_h - m)
    return float(p / (p + q))


def simple_averaging(scores: Sequence[float]) -> float:
    """Arithmetic mean of all scores. Floor zero, ceil one."""
    if not scores:
        return 0.0
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def max_voting(scores: Sequence[float], threshold: float = 0.5) -> float:
    """Majority-vote binary fusion: fraction of scores above threshold."""
    if not scores:
        return 0.0
    above = sum(1 for s in scores if s >= threshold)
    return float(above) / float(len(scores))


__all__ = [
    "dempster_shafer",
    "naive_bayes",
    "simple_averaging",
    "max_voting",
]
