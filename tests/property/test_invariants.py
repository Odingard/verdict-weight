"""Property tests — invariants that must hold over random inputs.

These are not paper-claim replications; they are mathematical invariants
that should hold by construction. Failures here indicate code-level bugs.
"""

from __future__ import annotations

import math
import random as _random

import numpy as np
import pytest

from verdict_weight import (
    DeploymentTier,
    SISAnalyzer,
    Source,
    SourceRegistry,
    StreamScorer,
    UnifiedComposer,
    UnifiedInputs,
)


SEED = 42


# ─────────────────────────────────────────────────────────────
# Stream-level invariants
# ─────────────────────────────────────────────────────────────


class TestStreamRanges:
    """All stream functions return values in [0, 1]."""

    def test_source_reliability_range(self):
        rng = _random.Random(SEED)
        for _ in range(200):
            x = rng.uniform(-2.0, 2.0)
            r = StreamScorer.source_reliability(x)
            assert 0.0 <= r <= 1.0

    def test_corroboration_range(self):
        rng = _random.Random(SEED)
        for _ in range(200):
            n = rng.randint(0, 50)
            r = StreamScorer.corroboration(n)
            assert 0.0 <= r <= 1.0

    def test_temporal_range(self):
        rng = _random.Random(SEED)
        for _ in range(200):
            age = rng.uniform(0.0, 1000.0)
            lam = rng.uniform(0.001, 1.0)
            r = StreamScorer.temporal(age, lam)
            assert 0.0 <= r <= 1.0

    def test_historical_accuracy_range(self):
        rng = _random.Random(SEED)
        for _ in range(200):
            total = rng.randint(0, 100)
            correct = rng.randint(0, total) if total > 0 else 0
            r = StreamScorer.historical_accuracy(correct, total)
            assert 0.0 <= r <= 1.0


class TestStreamMonotonicity:
    """Stream functions monotonic in their natural argument."""

    def test_corroboration_monotonic_in_n(self):
        prev = -1.0
        for n in range(0, 30):
            r = StreamScorer.corroboration(n)
            assert r >= prev - 1e-9
            prev = r

    def test_temporal_monotonic_in_age(self):
        prev = 1.0 + 1e-6
        for age in range(0, 200, 5):
            r = StreamScorer.temporal(float(age), 0.05)
            assert r <= prev + 1e-9
            prev = r

    def test_historical_accuracy_monotonic_in_correct(self):
        prev = -1.0
        total = 100
        for correct in range(0, total + 1, 5):
            r = StreamScorer.historical_accuracy(correct, total)
            assert r >= prev - 1e-9
            prev = r


# ─────────────────────────────────────────────────────────────
# SIS invariants
# ─────────────────────────────────────────────────────────────


class TestSISInvariants:
    def test_sis_independence_matrix_symmetric(self):
        rng = _random.Random(SEED)
        analyzer = SISAnalyzer()
        for _ in range(20):
            k = rng.randint(2, 8)
            sources = [
                Source(
                    source_id=f"s{i}",
                    institution=rng.choice(["A", "B", "C"]),
                    geography=rng.choice(["X", "Y", "Z"]),
                    publish_time=rng.uniform(0.0, 1e6),
                    primary_citations={f"c{rng.randint(0, 5)}"},
                )
                for i in range(k)
            ]
            r = analyzer.analyze(sources)
            I = r.independence_matrix
            assert np.allclose(I, I.T)

    def test_sis_score_bounded_by_one_over_k_and_one(self):
        rng = _random.Random(SEED)
        analyzer = SISAnalyzer()
        for _ in range(20):
            k = rng.randint(2, 10)
            sources = [
                Source(source_id=f"s{i}",
                       institution=rng.choice(["A", "B"]),
                       geography=rng.choice(["X", "Y"]),
                       publish_time=rng.uniform(0.0, 1e6),
                       primary_citations={f"c{rng.randint(0, 3)}"})
                for i in range(k)
            ]
            r = analyzer.analyze(sources)
            assert 1.0 / k - 1e-6 <= r.score <= 1.0 + 1e-6


# ─────────────────────────────────────────────────────────────
# Composition invariants
# ─────────────────────────────────────────────────────────────


def _genesis():
    return SourceRegistry(
        entries={f"s_{i}": 0.5 for i in range(10)},
        version=1,
    )


class TestCompositionInvariants:
    def test_cw_certified_within_unit_interval(self):
        rng = _random.Random(SEED)
        registry = _genesis()
        c = UnifiedComposer(registry=registry)
        for _ in range(50):
            inputs = UnifiedInputs(
                SR=rng.uniform(0.0, 1.0),
                CC=rng.uniform(0.0, 1.0),
                TD=rng.uniform(0.0, 1.0),
                HA=rng.uniform(0.0, 1.0),
                registry=registry,
            )
            r = c.score(inputs)
            assert not r.halted
            assert 0.0 <= r.cw_certified <= 1.0

    def test_no_nan_under_random_inputs(self):
        rng = _random.Random(SEED)
        registry = _genesis()
        c = UnifiedComposer(registry=registry)
        for _ in range(50):
            inputs = UnifiedInputs(
                SR=rng.uniform(0.0, 1.0),
                CC=rng.uniform(0.0, 1.0),
                TD=rng.uniform(0.0, 1.0),
                HA=rng.uniform(0.0, 1.0),
                registry=registry,
            )
            r = c.score(inputs)
            assert not math.isnan(r.cw_certified)
            for k, v in r.streams.items():
                assert not math.isnan(v), f"{k} produced NaN"

    def test_higher_inputs_yield_higher_or_equal_cw(self):
        registry = _genesis()
        c = UnifiedComposer(registry=registry)
        low = c.score(UnifiedInputs(
            SR=0.3, CC=0.3, TD=0.3, HA=0.3, registry=registry,
        )).cw_certified
        high = c.score(UnifiedInputs(
            SR=0.9, CC=0.9, TD=0.9, HA=0.9, registry=registry,
        )).cw_certified
        assert high > low

    def test_government_tier_attenuates_at_least_as_much_as_commercial(self):
        """For attenuating inputs (S_CTC < 1 or S_SIS < 1), δ=γ=1 ≥ 0.5."""
        rng = _random.Random(SEED)
        registry = _genesis()
        c = UnifiedComposer(registry=registry)
        # Curveball-shaped sources
        sources = [
            Source(source_id=f"s{i}", institution="X", geography="Y",
                   publish_time=rng.uniform(0, 1e3),
                   primary_citations={"c1"})
            for i in range(5)
        ]
        kwargs = dict(SR=0.85, CC=0.85, TD=0.95, HA=0.90,
                      sources=sources, registry=registry)
        gov = c.score(UnifiedInputs(**kwargs, deployment_tier=DeploymentTier.GOVERNMENT))
        com = c.score(UnifiedInputs(**kwargs, deployment_tier=DeploymentTier.COMMERCIAL))
        assert gov.cw_certified <= com.cw_certified + 1e-9
