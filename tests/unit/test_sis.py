"""Unit tests for Stream 6 — Source Independence Score (SIS)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from verdict_weight import (
    SISAnalyzer,
    SISResult,
    Source,
)


def _independent_sources(k: int) -> list[Source]:
    """K sources from genuinely-distinct institutions / geographies / timings."""
    return [
        Source(
            source_id=f"src_{i}",
            institution=f"inst_{i}",
            geography=f"region_{i}",
            publish_time=1_000_000.0 + i * 100_000.0,
            primary_citations={f"cite_{i}_{j}" for j in range(3)},
        )
        for i in range(k)
    ]


def _curveball_sources(k: int) -> list[Source]:
    """K sources all sharing institution/geography/timing/citations (full Curveball)."""
    return [
        Source(
            source_id=f"src_{i}",
            institution="shared_inst",
            geography="shared_region",
            publish_time=1_000_000.0 + i * 60.0,  # within timing_threshold
            primary_citations={"shared_cite_a", "shared_cite_b", "shared_cite_c"},
        )
        for i in range(k)
    ]


class TestSISBasics:
    def test_empty_sources(self, sis):
        r = sis.analyze([])
        assert r.cc_raw == 0
        assert r.score == 0.0
        assert r.pattern == "EMPTY"

    def test_single_source(self, sis):
        r = sis.analyze(_independent_sources(1))
        assert r.cc_raw == 1
        assert r.score == 1.0
        assert r.pattern == "INDEPENDENT"

    def test_score_within_unit_interval(self, sis):
        for k in (2, 3, 5, 10):
            for sources in (_independent_sources(k), _curveball_sources(k)):
                r = sis.analyze(sources)
                assert 0.0 <= r.score <= 1.0


class TestIndependentSources:
    @pytest.mark.parametrize("k", [2, 3, 5, 10])
    def test_independent_sources_score_high(self, sis, k):
        r = sis.analyze(_independent_sources(k))
        assert r.score >= 0.95
        assert r.pattern == "INDEPENDENT"

    def test_cc_eff_equals_k_for_independent(self, sis):
        r = sis.analyze(_independent_sources(5))
        assert r.cc_eff == pytest.approx(5.0, abs=0.05)


class TestCurveballSources:
    @pytest.mark.parametrize("k", [4, 5, 8, 10])
    def test_curveball_collapses_independence(self, sis, k):
        r = sis.analyze(_curveball_sources(k))
        # Effective corroboration should collapse close to 1.
        assert r.cc_eff == pytest.approx(1.0, abs=0.05)
        # Score should be close to 1/K.
        assert r.score == pytest.approx(1.0 / k, abs=0.05)
        assert r.pattern == "CURVEBALL"

    def test_curveball_at_k_4_reduces_75_percent(self, sis):
        """Paper claim: K=4 Curveball reduces effective corroboration by 75%."""
        r = sis.analyze(_curveball_sources(4))
        # 1 - cc_eff/K should be ~0.75
        reduction = 1.0 - r.cc_eff / r.cc_raw
        assert reduction >= 0.74

    def test_curveball_detection_sensitivity(self, sis):
        """100 K=5 Curveballs should all produce SIS score < curveball_threshold."""
        detected = 0
        for _ in range(100):
            r = sis.analyze(_curveball_sources(5))
            if r.score < sis.curveball_threshold:
                detected += 1
        assert detected == 100


class TestPartialOverlap:
    def test_partial_overlap_pattern(self, sis):
        # Two sources share institution but not geography/timing/citations.
        sources = [
            Source(source_id="a", institution="X", geography="US",
                   publish_time=1_000_000.0,
                   primary_citations={"c1", "c2"}),
            Source(source_id="b", institution="X", geography="EU",
                   publish_time=2_000_000.0,
                   primary_citations={"c3", "c4"}),
        ]
        r = sis.analyze(sources)
        assert r.pattern == "PARTIAL_OVERLAP"
        assert 0.0 < r.score < 1.0


class TestIndependenceMatrix:
    def test_matrix_shape(self, sis):
        r = sis.analyze(_independent_sources(4))
        assert r.independence_matrix.shape == (4, 4)

    def test_matrix_symmetric(self, sis):
        r = sis.analyze(_curveball_sources(5))
        I = r.independence_matrix
        assert np.allclose(I, I.T)

    def test_diagonal_is_one(self, sis):
        r = sis.analyze(_independent_sources(3))
        assert np.all(np.diagonal(r.independence_matrix) == 1.0)

    def test_matrix_entries_within_unit_interval(self, sis):
        r = sis.analyze(_curveball_sources(6))
        I = r.independence_matrix
        assert (I >= 0.0).all()
        assert (I <= 1.0).all()


class TestEdgeCases:
    def test_missing_dimensions_treated_as_unknown(self, sis):
        sources = [
            Source(source_id="a"),
            Source(source_id="b"),
        ]
        r = sis.analyze(sources)
        # All dimensions missing → 0.5 contribution per dimension → I=0.5
        # → score lies in (1/K, 1).
        assert 0.5 <= r.score <= 1.0

    def test_weight_validation(self):
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            SISAnalyzer(w_institutional=0.5, w_geographic=0.5,
                        w_timing=0.5, w_citation=0.5)


class TestConvenienceAPI:
    def test_score_method(self, sis):
        sources = _independent_sources(3)
        assert sis.score(sources) == sis.analyze(sources).score


class TestNumericalStability:
    def test_no_nan_on_empty_sources(self, sis):
        r = sis.analyze([])
        assert not math.isnan(r.score)

    def test_no_nan_on_curveball(self, sis):
        r = sis.analyze(_curveball_sources(8))
        assert not math.isnan(r.score)
