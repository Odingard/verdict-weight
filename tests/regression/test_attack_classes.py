"""Regression tests — known attack-class outcomes.

Each test fixes a canonical scenario and verifies the certified CW falls
in the expected range. Failures here mean the architecture has drifted
from its specified attack-class outcomes.

Numbers are NOT tuned to paper claims. They are conservative bounds
chosen so legitimate architectural improvements do not break the suite.
"""

from __future__ import annotations

import hashlib

import pytest

from verdict_weight import (
    DeploymentTier,
    ProvenanceStep,
    Source,
    SourceRegistry,
    TrajectoryPoint,
    UnifiedComposer,
    UnifiedInputs,
    build_provenance_chain,
)


def _genesis() -> SourceRegistry:
    return SourceRegistry(
        entries={f"s_{i:03d}": round(0.5 + i * 0.025, 3) for i in range(20)},
        version=1,
    )


def _stable_high(n=10):
    return [TrajectoryPoint(timestamp=float(i), value=0.85) for i in range(n)]


def _spike_collapse(n=10):
    out = [TrajectoryPoint(timestamp=0.0, value=0.95)]
    for i in range(1, n):
        out.append(TrajectoryPoint(timestamp=float(i),
                                   value=max(0.05, 0.95 * (0.3 ** i))))
    return out


def _independent_sources(k=5):
    return [
        Source(source_id=f"src_{i}",
               institution=f"inst_{i}",
               geography=f"region_{i}",
               publish_time=1_000_000.0 + i * 1e5,
               primary_citations={f"cite_{i}"})
        for i in range(k)
    ]


def _curveball_sources(k=5):
    return [
        Source(source_id=f"src_{i}",
               institution="X",
               geography="Y",
               publish_time=1_000_000.0 + i * 60,
               primary_citations={"shared"})
        for i in range(k)
    ]


def _valid_chain(n=4):
    return build_provenance_chain(
        [f"p_{i}".encode() for i in range(n)],
        [f"a_{i}" for i in range(n)],
        [1_000_000.0 + i * 60 for i in range(n)],
    )


class TestLegitimate:
    def test_legitimate_signal_high_cw(self):
        registry = _genesis()
        c = UnifiedComposer(registry=registry)
        r = c.score(UnifiedInputs(
            SR=0.85, CC=0.85, TD=0.95, HA=0.90,
            trajectory=_stable_high(),
            sources=_independent_sources(),
            provenance_chain=_valid_chain(),
            registry=registry,
            deployment_tier=DeploymentTier.GOVERNMENT,
        ))
        assert not r.halted
        # Legitimate signal must score above 0.5 in all tiers.
        assert r.cw_certified >= 0.5
        # And CW_base must be ≥ certified (composition is multiplicative ≤ 1).
        assert r.cw_base >= r.cw_certified


class TestAC1SourceSpoofing:
    def test_high_sr_low_cc_attenuates(self):
        registry = _genesis()
        c = UnifiedComposer(registry=registry)
        r = c.score(UnifiedInputs(
            SR=0.95, CC=0.10, TD=0.95, HA=0.95,
            trajectory=_stable_high(),
            sources=_independent_sources(1),
            provenance_chain=_valid_chain(),
            registry=registry,
        ))
        # Single un-corroborated source must not get HIGH+CRITICAL.
        assert r.cw_certified < 0.7


class TestAC2Curveball:
    def test_k_5_curveball_attenuates_strongly(self):
        registry = _genesis()
        c = UnifiedComposer(registry=registry)
        r = c.score(UnifiedInputs(
            SR=0.85, CC=0.85, TD=0.95, HA=0.90,
            trajectory=_stable_high(),
            sources=_curveball_sources(5),
            provenance_chain=_valid_chain(),
            registry=registry,
            deployment_tier=DeploymentTier.GOVERNMENT,
        ))
        # K=5 Curveball should drop S_SIS to ~1/5.
        assert r.streams["S_SIS"] < 0.3
        # And the certified CW should be lower than the legitimate baseline.
        assert r.cw_certified < 0.5


class TestAC3TrajectoryFabrication:
    def test_pattern_c_zeros_certified_cw(self):
        registry = _genesis()
        c = UnifiedComposer(registry=registry)
        r = c.score(UnifiedInputs(
            SR=0.95, CC=0.95, TD=0.95, HA=0.95,
            trajectory=_spike_collapse(),
            sources=_independent_sources(),
            provenance_chain=_valid_chain(),
            registry=registry,
            deployment_tier=DeploymentTier.GOVERNMENT,
        ))
        assert r.streams["S_CTC"] == pytest.approx(0.0, abs=1e-6)
        assert r.cw_certified == pytest.approx(0.0, abs=1e-3)


class TestAC4ProvenanceForgery:
    def test_tampered_chain_halts(self):
        registry = _genesis()
        c = UnifiedComposer(registry=registry)
        chain = _valid_chain(4)
        chain[2] = ProvenanceStep(
            data=b"FORGED", timestamp=chain[2].timestamp,
            actor=chain[2].actor, hash=chain[2].hash,
        )
        r = c.score(UnifiedInputs(
            SR=0.95, CC=0.95, TD=0.95, HA=0.95,
            trajectory=_stable_high(),
            sources=_independent_sources(),
            provenance_chain=chain,
            registry=registry,
        ))
        assert r.halted is True
        assert r.halted_at == "CPS"
        assert r.cw_certified is None


class TestAC5RegistryCompromise:
    def test_tampered_registry_halts(self):
        registry = _genesis()
        c = UnifiedComposer(registry=registry)
        bad = SourceRegistry(
            entries={**registry.entries, "evil_src": 0.99},
            version=registry.version,
        )
        r = c.score(UnifiedInputs(
            SR=0.95, CC=0.95, TD=0.95, HA=0.95,
            registry=bad,
        ))
        assert r.halted is True
        assert r.halted_at == "RIS"
