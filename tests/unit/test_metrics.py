"""Unit tests for validation.metrics — particularly tied-rank AUC handling.

Regression test guard for the mid-rank averaging fix in ``auc_roc``.
A naive implementation that uses ``np.argsort`` to assign consecutive
integer ranks (without averaging across ties) returns AUC = 0.0 for
fully-tied input — these tests exist to keep that bug from regressing.
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.metrics import (
    auc_roc,
    brier_score,
    cohens_d,
    reliability,
    sensitivity,
    specificity,
    welch_t_test,
)


class TestAUCTiedRanks:
    def test_fully_tied_is_05(self):
        """All-tied scores must give AUC exactly 0.5 regardless of label order."""
        assert auc_roc([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
        assert auc_roc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
        assert auc_roc([1, 0, 1, 0], [0.7, 0.7, 0.7, 0.7]) == pytest.approx(0.5)

    def test_perfect_separation(self):
        assert auc_roc([1, 1, 0, 0], [0.9, 0.9, 0.1, 0.1]) == pytest.approx(1.0)

    def test_inverse_perfect(self):
        assert auc_roc([1, 1, 0, 0], [0.1, 0.1, 0.9, 0.9]) == pytest.approx(0.0)

    def test_partial_tie_in_middle(self):
        # pos=0.9,0.5; neg=0.5,0.1 → 0.5 ties. expected: (1.0 + 0.5 + 1.0 + 0.5)/4 = 0.75? No.
        # Worked example with mid-rank: ranks [4, 2.5, 2.5, 1]; sum_pos=4+2.5=6.5; U=6.5-3=3.5; AUC=3.5/4=0.875.
        assert auc_roc([1, 1, 0, 0], [0.9, 0.5, 0.5, 0.1]) == pytest.approx(0.875)

    def test_partial_tie_in_high(self):
        # ranks [3.5, 3.5, 2, 1]; sum_pos = 3.5+3.5=7; U=7-3=4; AUC=1.0 (positives all ≥ negatives).
        assert auc_roc([1, 1, 0, 0], [0.9, 0.9, 0.5, 0.1]) == pytest.approx(1.0)

    def test_no_positives_returns_05(self):
        assert auc_roc([0, 0, 0], [0.1, 0.5, 0.9]) == pytest.approx(0.5)

    def test_no_negatives_returns_05(self):
        assert auc_roc([1, 1, 1], [0.1, 0.5, 0.9]) == pytest.approx(0.5)

    def test_against_known_imbalanced_case(self):
        """Hand-computed AUC for a small imbalanced example."""
        y_true = [1, 0, 1, 0, 1, 0]
        y_score = [0.9, 0.5, 0.8, 0.5, 0.7, 0.5]
        # All positives strictly greater than all negatives → AUC = 1.0
        assert auc_roc(y_true, y_score) == pytest.approx(1.0)


class TestBasicMetrics:
    def test_brier_perfect(self):
        assert brier_score([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0]) == pytest.approx(0.0)

    def test_brier_worst(self):
        assert brier_score([1, 0, 1, 0], [0.0, 1.0, 0.0, 1.0]) == pytest.approx(1.0)

    def test_reliability_zero_when_perfect(self):
        # A perfectly-calibrated classifier has REL = 0
        assert reliability([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0]) == pytest.approx(0.0)

    def test_cohens_d_signed(self):
        a = [0.0, 0.0, 0.0, 0.0]
        b = [1.0, 1.0, 1.0, 1.0]
        # mean(a) - mean(b) = -1, pooled sd = 0 → guard returns 0.0
        assert cohens_d(a, b) == 0.0

    def test_cohens_d_with_variance(self):
        rng = np.random.default_rng(42)
        a = rng.normal(loc=0.0, scale=1.0, size=200)
        b = rng.normal(loc=1.0, scale=1.0, size=200)
        d = cohens_d(a.tolist(), b.tolist())
        # Approx -1.0 ± 0.2 for these populations
        assert -1.3 < d < -0.7

    def test_welch_returns_finite(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0.0, 1.0, 200).tolist()
        b = rng.normal(1.0, 1.0, 200).tolist()
        t, p = welch_t_test(a, b)
        assert np.isfinite(t)
        assert 0.0 <= p <= 1.0

    def test_sensitivity_specificity_perfect(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        assert sensitivity(y_true, y_pred) == 1.0
        assert specificity(y_true, y_pred) == 1.0
