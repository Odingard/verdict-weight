"""Integration tests — full eight-stream composition pipeline."""

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


def _genesis_registry() -> SourceRegistry:
    return SourceRegistry(
        entries={f"src_{i:03d}": round(0.5 + i * 0.025, 3) for i in range(20)},
        version=1,
    )


def _stable_high_trajectory(n=10):
    return [TrajectoryPoint(timestamp=float(i), value=0.85) for i in range(n)]


def _spike_collapse_trajectory(n=10, peak=0.95, floor=0.05):
    out = [TrajectoryPoint(timestamp=0.0, value=peak)]
    for i in range(1, n):
        decay = floor + (peak - floor) * (0.3 ** i)
        out.append(TrajectoryPoint(timestamp=float(i), value=decay))
    return out


def _independent_sources(k=5):
    return [
        Source(source_id=f"src_{i}",
               institution=f"inst_{i}",
               geography=f"region_{i}",
               publish_time=1_000_000.0 + i * 100_000.0,
               primary_citations={f"cite_{i}_{j}" for j in range(3)})
        for i in range(k)
    ]


def _curveball_sources(k=5):
    return [
        Source(source_id=f"src_{i}",
               institution="shared_inst",
               geography="shared_region",
               publish_time=1_000_000.0 + i * 60.0,
               primary_citations={"shared_cite"})
        for i in range(k)
    ]


def _valid_chain(n=4):
    return build_provenance_chain(
        [f"payload_{i}".encode() for i in range(n)],
        [f"actor_{i}" for i in range(n)],
        [1_000_000.0 + i * 60.0 for i in range(n)],
    )


def _legitimate_inputs(registry):
    return UnifiedInputs(
        SR=0.85, CC=0.85, TD=0.95, HA=0.90,
        trajectory=_stable_high_trajectory(),
        sources=_independent_sources(),
        provenance_chain=_valid_chain(),
        registry=registry,
        deployment_tier=DeploymentTier.GOVERNMENT,
    )


# ─────────────────────────────────────────────────────────────
# Legitimate-signal end-to-end
# ─────────────────────────────────────────────────────────────


class TestLegitimatePipeline:
    def test_legitimate_inputs_yield_high_cw(self):
        registry = _genesis_registry()
        c = UnifiedComposer(registry=registry)
        r = c.score(_legitimate_inputs(registry))
        assert not r.halted
        assert r.cw_certified is not None
        assert r.cw_certified >= 0.5

    def test_streams_all_above_threshold_legitimate(self):
        registry = _genesis_registry()
        c = UnifiedComposer(registry=registry)
        r = c.score(_legitimate_inputs(registry))
        for k in ("S_CTC", "S_SIS", "S_CPS", "S_RIS"):
            assert r.streams[k] >= 0.5, f"{k} below 0.5: {r.streams[k]}"


# ─────────────────────────────────────────────────────────────
# HALT-state propagation (RIS → CPS → SIS → CTC)
# ─────────────────────────────────────────────────────────────


class TestRISHalt:
    def test_tampered_registry_halts_at_ris(self):
        registry = _genesis_registry()
        c = UnifiedComposer(registry=registry)
        bad = SourceRegistry(
            entries={**registry.entries, "evil": 0.99},
            version=registry.version,
        )
        inputs = UnifiedInputs(
            SR=0.85, CC=0.85, TD=0.95, HA=0.90,
            registry=bad,
        )
        r = c.score(inputs)
        assert r.halted is True
        assert r.halted_at == "RIS"
        assert r.cw_certified is None
        assert r.action_tier == "HALT"

    def test_no_genesis_registered_halts(self):
        c = UnifiedComposer()  # no registry
        registry = _genesis_registry()
        inputs = UnifiedInputs(
            SR=0.85, CC=0.85, TD=0.95, HA=0.90,
            registry=registry,
        )
        r = c.score(inputs)
        assert r.halted is True
        assert r.halted_at == "RIS"


class TestCPSHalt:
    def test_tampered_chain_halts_at_cps(self):
        registry = _genesis_registry()
        c = UnifiedComposer(registry=registry)
        chain = _valid_chain(4)
        chain[2] = ProvenanceStep(
            data=b"TAMPERED", timestamp=chain[2].timestamp,
            actor=chain[2].actor, hash=chain[2].hash,
        )
        inputs = UnifiedInputs(
            SR=0.85, CC=0.85, TD=0.95, HA=0.90,
            provenance_chain=chain,
            registry=registry,
        )
        r = c.score(inputs)
        assert r.halted is True
        assert r.halted_at == "CPS"
        assert r.cw_certified is None


# ─────────────────────────────────────────────────────────────
# Adversarial (non-HALT) attenuation
# ─────────────────────────────────────────────────────────────


class TestAC1SourceSpoofing:
    """High SR, low CC — single source with no corroboration."""

    def test_low_corroboration_lowers_cw(self):
        registry = _genesis_registry()
        c = UnifiedComposer(registry=registry)
        legit = c.score(_legitimate_inputs(registry))
        ac1 = c.score(UnifiedInputs(
            SR=0.95, CC=0.20, TD=0.95, HA=0.95,
            trajectory=_stable_high_trajectory(),
            sources=_independent_sources(1),
            provenance_chain=_valid_chain(),
            registry=registry,
        ))
        assert ac1.cw_certified < legit.cw_certified


class TestAC2Curveball:
    """K shared-origin sources — SIS collapses CC."""

    def test_curveball_attenuates_cw(self):
        registry = _genesis_registry()
        c = UnifiedComposer(registry=registry)
        legit_inputs = _legitimate_inputs(registry)
        legit = c.score(legit_inputs)
        ac2 = c.score(UnifiedInputs(
            SR=0.85, CC=0.85, TD=0.95, HA=0.90,
            trajectory=_stable_high_trajectory(),
            sources=_curveball_sources(5),
            provenance_chain=_valid_chain(),
            registry=registry,
        ))
        assert ac2.cw_certified < legit.cw_certified
        assert ac2.streams["S_SIS"] < 0.5


class TestAC3TrajectoryFabrication:
    """Pattern C spike-collapse — CTC scores 0."""

    def test_spike_collapse_attenuates_cw(self):
        registry = _genesis_registry()
        c = UnifiedComposer(registry=registry)
        ac3 = c.score(UnifiedInputs(
            SR=0.85, CC=0.85, TD=0.95, HA=0.90,
            trajectory=_spike_collapse_trajectory(),
            sources=_independent_sources(),
            provenance_chain=_valid_chain(),
            registry=registry,
        ))
        # S_CTC must be 0 → multiplicatively zeroes CW_certified at γ=1.
        assert ac3.streams["S_CTC"] == pytest.approx(0.0, abs=1e-6)
        assert ac3.cw_certified == pytest.approx(0.0, abs=1e-3)


# ─────────────────────────────────────────────────────────────
# Deployment tier behaviour
# ─────────────────────────────────────────────────────────────


class TestDeploymentTiers:
    def test_government_tier_amplifies_attenuation(self):
        registry = _genesis_registry()
        c = UnifiedComposer(registry=registry)
        # Curveball under each tier
        kwargs = dict(SR=0.85, CC=0.85, TD=0.95, HA=0.90,
                      trajectory=_stable_high_trajectory(),
                      sources=_curveball_sources(5),
                      provenance_chain=_valid_chain(),
                      registry=registry)
        gov = c.score(UnifiedInputs(**kwargs, deployment_tier=DeploymentTier.GOVERNMENT))
        com = c.score(UnifiedInputs(**kwargs, deployment_tier=DeploymentTier.COMMERCIAL))
        # Government tier (δ=1) penalizes more than commercial (δ=0.5).
        assert gov.cw_certified <= com.cw_certified


# ─────────────────────────────────────────────────────────────
# Optional-streams behaviour
# ─────────────────────────────────────────────────────────────


class TestOptionalStreams:
    def test_no_trajectory_skips_ctc(self):
        registry = _genesis_registry()
        c = UnifiedComposer(registry=registry)
        r = c.score(UnifiedInputs(
            SR=0.85, CC=0.85, TD=0.95, HA=0.90,
            registry=registry,
        ))
        assert r.streams["S_CTC"] == 1.0
        assert not r.halted

    def test_no_registry_no_halt(self):
        c = UnifiedComposer()  # no registry registered
        r = c.score(UnifiedInputs(SR=0.5, CC=0.5, TD=0.5, HA=0.5))
        assert not r.halted
        assert r.streams["S_RIS"] == 1.0


# ─────────────────────────────────────────────────────────────
# Halting order — RIS evaluated before CPS
# ─────────────────────────────────────────────────────────────


class TestHaltOrdering:
    def test_ris_evaluated_before_cps(self):
        registry = _genesis_registry()
        c = UnifiedComposer(registry=registry)
        # Both registry AND chain are tampered — halt should report RIS first.
        bad_registry = SourceRegistry(
            entries={"x": 0.5}, version=registry.version,
        )
        chain = _valid_chain(3)
        chain[1] = ProvenanceStep(
            data=b"tamper", timestamp=chain[1].timestamp,
            actor=chain[1].actor, hash=chain[1].hash,
        )
        r = c.score(UnifiedInputs(
            SR=0.85, CC=0.85, TD=0.95, HA=0.90,
            provenance_chain=chain,
            registry=bad_registry,
        ))
        assert r.halted_at == "RIS"
