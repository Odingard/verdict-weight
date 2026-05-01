"""
Stream 5 — Cross-Temporal Consistency (CTC)
==========================================

Detects Pattern C (spike-then-collapse) trajectories that are characteristic
of fabricated intelligence signals (attack class AC-3).

A signal trajectory is a time-ordered sequence of (timestamp, value) points
where ``value`` is the per-source confidence at that instant. Legitimate
signals exhibit one of three trajectory patterns:

  Pattern A  — stable high      (legitimate strong signal)
  Pattern B  — stable low       (legitimate noise)
  Pattern D  — gradual buildup  (legitimate corroboration accumulation)

The fourth pattern is the adversarial signature:

  Pattern C  — spike-then-collapse (AC-3, fabricated signal)

The Adversarial Trajectory Theorem (Section 4.2 of the unified paper)
proves that legitimate signals cannot produce Pattern C: the temporal
decay function is continuous, and corroborating sources cannot retract
simultaneously without coordination. Therefore an observed Pattern C is
necessarily adversarial.

Algorithm
---------
Given a trajectory of length ``n``:

1. Compute summary statistics: mean, std, min, max, peak index.
2. Classify by descending priority of structural evidence:
   * Pattern C detection — peak in the first 70% of the window, peak
     value above ``high_threshold``, drop from peak to terminal value
     above ``collapse_threshold``, and the post-peak segment trending
     downward (negative slope on the post-peak window).
   * Pattern A — mean above ``high_threshold`` and std below
     ``stability_threshold``.
   * Pattern B — mean below ``low_threshold`` and std below
     ``stability_threshold``.
   * Pattern D — monotonic positive slope with R² above
     ``buildup_r2_threshold``.
   * Otherwise: ambiguous, default to Pattern A with score 0.5.
3. Return a CTC score:
   * 0.00 for Pattern C (with magnitude scaled by collapse depth).
   * 1.00 for Patterns A, B, D (with magnitude reduced by per-pattern
     fit quality).
   * 0.50 when trajectory is too short to classify (n < 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional, Sequence

import numpy as np


# ─────────────────────────────────────────────────────────────
# Trajectory primitives
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrajectoryPoint:
    """A single observation in a signal trajectory.

    Attributes
    ----------
    timestamp:
        Unix epoch seconds (float). Must be strictly monotonically
        increasing across a trajectory.
    value:
        Per-source confidence score in ``[0, 1]``.
    """

    timestamp: float
    value: float


class TrajectoryPattern(str, Enum):
    """Trajectory pattern classification."""

    A = "A_stable_high"
    B = "B_stable_low"
    C = "C_spike_collapse"
    D = "D_gradual_buildup"
    INSUFFICIENT = "insufficient_data"


@dataclass(frozen=True)
class CTCResult:
    """Output of a CTC analysis.

    Attributes
    ----------
    score:
        ``S_CTC ∈ [0, 1]``. Values near 1.0 indicate Patterns A/B/D
        (legitimate); values near 0.0 indicate Pattern C (adversarial).
    pattern:
        The classified trajectory pattern.
    confidence:
        Classification confidence. 1.0 = unambiguous pattern fit;
        0.0 = pattern was assigned by default.
    metrics:
        Diagnostic metrics — mean, std, peak_index, peak_value,
        terminal_value, drop, slope, r2.
    """

    score: float
    pattern: TrajectoryPattern
    confidence: float
    metrics: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────────────────────


class CTCAnalyzer:
    """Cross-Temporal Consistency analyzer.

    Parameters
    ----------
    high_threshold:
        Minimum mean (or peak) value above which a trajectory is
        considered "high" for Pattern A and the spike of Pattern C.
        Default 0.70.
    low_threshold:
        Maximum mean below which a trajectory is considered "low" for
        Pattern B. Default 0.30.
    stability_threshold:
        Maximum std below which a high or low trajectory is considered
        "stable". Default 0.10.
    collapse_threshold:
        Minimum drop from peak value to terminal value to be classified
        as Pattern C. Default 0.40.
    early_peak_fraction:
        Maximum normalized peak index for Pattern C — the spike must
        occur within the first ``early_peak_fraction`` of the window.
        Default 0.70.
    buildup_r2_threshold:
        Minimum R² for the linear-trend fit to classify as Pattern D.
        Default 0.50.
    min_points:
        Minimum trajectory length below which classification is
        ``INSUFFICIENT``. Default 3.
    """

    def __init__(
        self,
        high_threshold: float = 0.70,
        low_threshold: float = 0.30,
        stability_threshold: float = 0.10,
        collapse_threshold: float = 0.40,
        early_peak_fraction: float = 0.70,
        buildup_r2_threshold: float = 0.50,
        min_points: int = 3,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.stability_threshold = stability_threshold
        self.collapse_threshold = collapse_threshold
        self.early_peak_fraction = early_peak_fraction
        self.buildup_r2_threshold = buildup_r2_threshold
        self.min_points = min_points

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def analyze(self, trajectory: Sequence[TrajectoryPoint]) -> CTCResult:
        """Classify a trajectory and return the CTC score."""
        n = len(trajectory)
        if n < self.min_points:
            return CTCResult(
                score=0.50,
                pattern=TrajectoryPattern.INSUFFICIENT,
                confidence=0.0,
                metrics={"n": n},
            )

        values = np.array([p.value for p in trajectory], dtype=float)
        timestamps = np.array([p.timestamp for p in trajectory], dtype=float)

        # Validate monotonic timestamps
        if not np.all(np.diff(timestamps) >= 0):
            raise ValueError(
                "Trajectory timestamps must be monotonically non-decreasing"
            )

        mean = float(values.mean())
        std = float(values.std())
        peak_idx = int(np.argmax(values))
        peak_value = float(values[peak_idx])
        terminal_value = float(values[-1])
        drop = peak_value - terminal_value

        # Linear trend fit (for Pattern D detection)
        slope, intercept, r2 = self._linear_fit(timestamps, values)

        # Post-peak slope (for Pattern C confirmation)
        if peak_idx < n - 1:
            post_peak_slope, _, _ = self._linear_fit(
                timestamps[peak_idx:], values[peak_idx:]
            )
        else:
            post_peak_slope = 0.0

        metrics = {
            "n": n,
            "mean": mean,
            "std": std,
            "peak_idx": peak_idx,
            "peak_value": peak_value,
            "terminal_value": terminal_value,
            "drop": drop,
            "slope": slope,
            "r2": r2,
            "post_peak_slope": post_peak_slope,
        }

        peak_norm = peak_idx / max(n - 1, 1)

        # Priority 1: Pattern C (spike-then-collapse)
        if (
            peak_value >= self.high_threshold
            and peak_norm <= self.early_peak_fraction
            and drop >= self.collapse_threshold
            and post_peak_slope < 0.0
        ):
            # Score scales inversely with collapse magnitude.
            # drop=0.4 → score ≈ 0.20; drop=1.0 → score ≈ 0.00.
            score = float(np.clip(1.0 - (drop * 1.5), 0.0, 0.30))
            confidence = float(
                np.clip((drop - self.collapse_threshold) / 0.5, 0.0, 1.0)
            )
            return CTCResult(
                score=score,
                pattern=TrajectoryPattern.C,
                confidence=confidence,
                metrics=metrics,
            )

        # Priority 2: Pattern A (stable high)
        if mean >= self.high_threshold and std <= self.stability_threshold:
            confidence = float(
                np.clip(
                    (mean - self.high_threshold) / 0.30
                    + (self.stability_threshold - std) / self.stability_threshold,
                    0.0,
                    1.0,
                )
                / 2.0
            )
            return CTCResult(
                score=float(np.clip(0.85 + 0.15 * confidence, 0.0, 1.0)),
                pattern=TrajectoryPattern.A,
                confidence=confidence,
                metrics=metrics,
            )

        # Priority 3: Pattern B (stable low)
        if mean <= self.low_threshold and std <= self.stability_threshold:
            confidence = float(
                np.clip(
                    (self.low_threshold - mean) / 0.30
                    + (self.stability_threshold - std) / self.stability_threshold,
                    0.0,
                    1.0,
                )
                / 2.0
            )
            return CTCResult(
                score=float(np.clip(0.85 + 0.15 * confidence, 0.0, 1.0)),
                pattern=TrajectoryPattern.B,
                confidence=confidence,
                metrics=metrics,
            )

        # Priority 4: Pattern D (gradual buildup)
        if slope > 0.0 and r2 >= self.buildup_r2_threshold:
            confidence = float(np.clip(r2, 0.0, 1.0))
            return CTCResult(
                score=float(np.clip(0.80 + 0.20 * confidence, 0.0, 1.0)),
                pattern=TrajectoryPattern.D,
                confidence=confidence,
                metrics=metrics,
            )

        # Default: ambiguous, treat as Pattern A with low confidence
        return CTCResult(
            score=0.50,
            pattern=TrajectoryPattern.A,
            confidence=0.0,
            metrics=metrics,
        )

    def score(self, trajectory: Sequence[TrajectoryPoint]) -> float:
        """Convenience wrapper that returns just the CTC score."""
        return self.analyze(trajectory).score

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    @staticmethod
    def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple:
        """Ordinary least-squares slope, intercept, and R² over (x, y)."""
        if len(x) < 2:
            return 0.0, float(y[0]) if len(y) else 0.0, 0.0
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        denom = float(((x - x_mean) ** 2).sum())
        if denom < 1e-12:
            return 0.0, y_mean, 0.0
        slope = float(((x - x_mean) * (y - y_mean)).sum() / denom)
        intercept = y_mean - slope * x_mean
        ss_res = float(((y - (slope * x + intercept)) ** 2).sum())
        ss_tot = float(((y - y_mean) ** 2).sum())
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        return slope, intercept, max(0.0, r2)


def analyze_trajectory(
    trajectory: Iterable[TrajectoryPoint],
    analyzer: Optional[CTCAnalyzer] = None,
) -> CTCResult:
    """Module-level convenience for a one-shot analysis."""
    analyzer = analyzer or CTCAnalyzer()
    return analyzer.analyze(list(trajectory))
