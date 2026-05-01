"""Unit tests for the commercial-tier streams (1–4) in ``core.py``."""

from __future__ import annotations

import math

import pytest

from verdict_weight import (
    ContextType,
    StreamScorer,
    VerdictEngine,
    VerdictResult,
    VerdictWeight,
    WEIGHT_PROFILES,
)


# ─────────────────────────────────────────────────────────────
# Stream 1 — Source Reliability (SR)
# ─────────────────────────────────────────────────────────────


class TestSourceReliability:
    def test_clamps_to_unit_interval(self):
        assert StreamScorer.source_reliability(-0.5) == 0.01
        assert StreamScorer.source_reliability(1.5) == 0.99

    def test_passes_through_valid_range(self):
        assert StreamScorer.source_reliability(0.5) == 0.5
        assert StreamScorer.source_reliability(0.85) == 0.85

    def test_avoids_zero_and_one_for_log_safety(self):
        # The 0.01/0.99 floor/ceiling prevents log(0) downstream.
        assert 0.0 < StreamScorer.source_reliability(0.0)
        assert StreamScorer.source_reliability(1.0) < 1.0

    def test_returns_float(self):
        assert isinstance(StreamScorer.source_reliability(0.5), float)


# ─────────────────────────────────────────────────────────────
# Stream 2 — Cross-feed Corroboration (CC)
# ─────────────────────────────────────────────────────────────


class TestCrossfeedCorroboration:
    def test_zero_sources_returns_floor(self):
        assert StreamScorer.corroboration(0) == 0.08

    def test_negative_sources_clamped(self):
        # Function takes max(n, 0) so negative does not break anything.
        assert StreamScorer.corroboration(-3) == 0.08

    def test_monotonic_in_n_sources(self):
        a = StreamScorer.corroboration(1)
        b = StreamScorer.corroboration(3)
        c = StreamScorer.corroboration(10)
        assert a < b < c

    def test_saturates_below_one(self):
        for n in [10, 50, 100, 1000]:
            assert StreamScorer.corroboration(n) <= 0.99

    def test_saturation_rate_argument(self):
        slow = StreamScorer.corroboration(3, saturation_rate=0.2)
        fast = StreamScorer.corroboration(3, saturation_rate=1.0)
        assert fast > slow


# ─────────────────────────────────────────────────────────────
# Stream 3 — Temporal Decay (TD)
# ─────────────────────────────────────────────────────────────


class TestTemporalDecay:
    def test_zero_age_returns_max(self):
        assert StreamScorer.temporal(0, decay_lambda=0.05) == pytest.approx(0.99, abs=1e-9)

    def test_decays_monotonically(self):
        old = StreamScorer.temporal(100, decay_lambda=0.05)
        recent = StreamScorer.temporal(10, decay_lambda=0.05)
        assert old < recent

    def test_within_unit_interval(self):
        for age in [0, 1, 30, 365, 10_000]:
            score = StreamScorer.temporal(age, decay_lambda=0.05)
            assert 0.01 <= score <= 0.99

    def test_negative_age_clamped(self):
        # Future timestamps are treated as age=0.
        future = StreamScorer.temporal(-10, decay_lambda=0.05)
        zero = StreamScorer.temporal(0, decay_lambda=0.05)
        assert future == zero


# ─────────────────────────────────────────────────────────────
# Stream 4 — Historical Source Accuracy (HA)
# ─────────────────────────────────────────────────────────────


class TestHistoricalAccuracy:
    def test_no_history_returns_neutral_prior(self):
        assert StreamScorer.historical_accuracy(0, 0) == 0.5

    def test_perfect_history_approaches_one(self):
        score = StreamScorer.historical_accuracy(100, 100)
        assert score > 0.95

    def test_zero_history_approaches_zero(self):
        score = StreamScorer.historical_accuracy(0, 100)
        assert score < 0.10

    def test_within_unit_interval(self):
        for c, t in [(0, 0), (1, 10), (50, 100), (1000, 1000)]:
            score = StreamScorer.historical_accuracy(c, t)
            assert 0.01 <= score <= 0.99

    def test_smoothing_prevents_extremes(self):
        # Default smoothing=2 means even 0/2 doesn't go to zero.
        score = StreamScorer.historical_accuracy(0, 2)
        assert score > 0.0

    def test_monotonic_in_correct_predictions(self):
        a = StreamScorer.historical_accuracy(2, 10)
        b = StreamScorer.historical_accuracy(8, 10)
        assert a < b


# ─────────────────────────────────────────────────────────────
# VerdictEngine composition (Streams 1–4)
# ─────────────────────────────────────────────────────────────


class TestVerdictEngine:
    def test_score_returns_three_values_in_unit_interval(self):
        profile = WEIGHT_PROFILES[ContextType.CYBERSECURITY_GENERAL]
        SS, DI, CW = VerdictEngine.score(0.8, 0.7, 0.9, 0.85, profile)
        for v in (SS, DI, CW):
            assert 0.0 <= v <= 1.0

    def test_high_inputs_yield_high_cw(self):
        profile = WEIGHT_PROFILES[ContextType.CYBERSECURITY_GENERAL]
        _, _, cw_high = VerdictEngine.score(0.95, 0.95, 0.95, 0.95, profile)
        _, _, cw_low = VerdictEngine.score(0.20, 0.20, 0.20, 0.20, profile)
        assert cw_high > cw_low

    def test_low_corroboration_lowers_cw(self):
        profile = WEIGHT_PROFILES[ContextType.CYBERSECURITY_GENERAL]
        _, _, cw_low_cc = VerdictEngine.score(0.9, 0.10, 0.9, 0.9, profile)
        _, _, cw_high_cc = VerdictEngine.score(0.9, 0.95, 0.9, 0.9, profile)
        assert cw_low_cc < cw_high_cc

    def test_high_doubt_index_when_streams_disagree(self):
        profile = WEIGHT_PROFILES[ContextType.CYBERSECURITY_GENERAL]
        _, di_disagree, _ = VerdictEngine.score(0.95, 0.10, 0.95, 0.10, profile)
        _, di_agree, _ = VerdictEngine.score(0.85, 0.85, 0.85, 0.85, profile)
        assert di_disagree > di_agree

    @pytest.mark.parametrize("cw,expected_tier", [
        (0.90, "CRITICAL"),
        (0.70, "HIGH"),
        (0.55, "MEDIUM"),
        (0.30, "LOW"),
        (0.10, "NOISE"),
    ])
    def test_interpret_tier_mapping(self, cw, expected_tier):
        tier, _ = VerdictEngine.interpret(cw, 0.2)
        assert tier == expected_tier

    def test_interpret_warns_on_high_doubt(self):
        _, text_high = VerdictEngine.interpret(0.6, 0.85)
        _, text_low = VerdictEngine.interpret(0.6, 0.15)
        assert "WARNING" in text_high
        assert "WARNING" not in text_low


class TestVerdictWeightFacade:
    def test_score_streams_round_trip(self):
        vw = VerdictWeight()
        result = vw.score_streams(SR=0.85, CC=0.80, TD=0.90, HA=0.75)
        assert isinstance(result, VerdictResult)
        assert 0.0 <= result.consequence_weight <= 1.0
        assert result.action_tier in {"CRITICAL", "HIGH", "MEDIUM", "LOW", "NOISE"}

    def test_score_from_raw_inputs(self):
        vw = VerdictWeight()
        result = vw.score(
            source_reliability=0.85,
            n_corroborating_sources=4,
            age_value=10,
            correct_predictions=18,
            total_predictions=20,
        )
        assert 0.0 <= result.consequence_weight <= 1.0

    def test_to_dict_roundtrip(self):
        vw = VerdictWeight()
        result = vw.score_streams(SR=0.85, CC=0.80, TD=0.90, HA=0.75)
        d = result.to_dict()
        assert "consequence_weight" in d
        assert "doubt_index" in d
        assert "signal_strength" in d
        assert "streams" in d

    def test_to_json_is_valid_json(self):
        import json as _json

        vw = VerdictWeight()
        result = vw.score_streams(SR=0.85, CC=0.80, TD=0.90, HA=0.75)
        parsed = _json.loads(result.to_json())
        assert "consequence_weight" in parsed

    def test_context_switching_changes_weights(self):
        vw = VerdictWeight()
        r1 = vw.score_streams(0.9, 0.7, 0.9, 0.8, context=ContextType.CYBERSECURITY_GENERAL)
        r2 = vw.score_streams(0.9, 0.7, 0.9, 0.8, context=ContextType.CYBERSECURITY_APT)
        # Different profiles should produce different CW values
        assert r1.consequence_weight != r2.consequence_weight

    def test_list_contexts_includes_all_built_in_profiles(self):
        vw = VerdictWeight()
        ctxs = {c["context"] for c in vw.list_contexts()}
        assert ctxs.issuperset({c.value for c in ContextType if c in WEIGHT_PROFILES})


# ─────────────────────────────────────────────────────────────
# Numerical stability
# ─────────────────────────────────────────────────────────────


class TestNumericalStability:
    @pytest.mark.parametrize("x", [0.0, 1.0, 0.5, 1e-9, 1.0 - 1e-9])
    def test_no_nan_at_unit_endpoints(self, x):
        profile = WEIGHT_PROFILES[ContextType.CYBERSECURITY_GENERAL]
        SS, DI, CW = VerdictEngine.score(x, x, x, x, profile)
        for v in (SS, DI, CW):
            assert not math.isnan(v)
            assert 0.0 <= v <= 1.0

    def test_extreme_disagreement_remains_bounded(self):
        profile = WEIGHT_PROFILES[ContextType.CYBERSECURITY_GENERAL]
        for streams in [(0, 1, 0, 1), (1, 0, 1, 0), (0, 0, 0, 0), (1, 1, 1, 1)]:
            SS, DI, CW = VerdictEngine.score(*streams, profile)
            for v in (SS, DI, CW):
                assert 0.0 <= v <= 1.0
                assert not math.isnan(v)
