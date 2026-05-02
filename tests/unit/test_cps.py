"""Unit tests for Stream 7 — Cryptographic Provenance Score (CPS)."""

from __future__ import annotations

import hashlib
import struct

import pytest

from verdict_weight import (
    CPSResult,
    CPSVerifier,
    ProvenanceStep,
    build_provenance_chain,
)


def _make_chain(n: int = 4, genesis: bytes = b"") -> list[ProvenanceStep]:
    payloads = [f"payload_{i}".encode("utf-8") for i in range(n)]
    actors = [f"actor_{i}" for i in range(n)]
    timestamps = [1_000_000.0 + i * 60.0 for i in range(n)]
    return build_provenance_chain(payloads, actors, timestamps, genesis_hash=genesis)


class TestBuildProvenanceChain:
    def test_chain_has_correct_length(self):
        chain = _make_chain(5)
        assert len(chain) == 5

    def test_each_step_has_32_byte_hash(self):
        chain = _make_chain(4)
        for step in chain:
            assert len(step.hash) == 32

    def test_genesis_chain_self_anchors(self):
        chain1 = _make_chain(3, genesis=b"")
        # Same inputs but different genesis → different first hash
        chain2 = build_provenance_chain(
            [s.data for s in chain1],
            [s.actor for s in chain1],
            [s.timestamp for s in chain1],
            genesis_hash=hashlib.sha256(b"different_genesis").digest(),
        )
        assert chain1[0].hash != chain2[0].hash

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_provenance_chain([b"a"], ["actor1", "actor2"], [1.0, 2.0])

    def test_non_monotonic_timestamps_raise(self):
        with pytest.raises(ValueError, match="monotonically"):
            build_provenance_chain([b"a", b"b"], ["x", "y"], [10.0, 5.0])

    def test_equal_timestamps_allowed(self):
        chain = build_provenance_chain([b"a", b"b"], ["x", "y"], [10.0, 10.0])
        assert len(chain) == 2


class TestValidChainVerification:
    def test_valid_chain_validates(self, cps):
        chain = _make_chain(4)
        r = cps.verify(chain)
        assert r.valid is True
        assert r.score == 1.0
        assert r.failed_step is None
        assert r.failure_reason is None

    def test_score_method_returns_one(self, cps):
        chain = _make_chain(4)
        assert cps.score(chain) == 1.0

    @pytest.mark.parametrize("n", [1, 2, 4, 10, 50])
    def test_chain_lengths(self, cps, n):
        chain = _make_chain(n)
        r = cps.verify(chain)
        assert r.valid is True
        assert r.chain_length == n


class TestEmptyChain:
    def test_empty_chain_fails(self, cps):
        r = cps.verify([])
        assert r.valid is False
        assert r.score == 0.0
        assert "empty" in r.failure_reason.lower()


class TestTamperedChainDetection:
    def test_modified_data_detected(self, cps):
        chain = _make_chain(4)
        # Tamper with step 2's data, leave hash intact
        chain[2] = ProvenanceStep(
            data=b"TAMPERED", timestamp=chain[2].timestamp,
            actor=chain[2].actor, hash=chain[2].hash,
        )
        r = cps.verify(chain)
        assert r.valid is False
        assert r.score == 0.0
        assert r.failed_step == 2
        assert "hash mismatch" in r.failure_reason.lower()

    def test_modified_actor_detected(self, cps):
        chain = _make_chain(4)
        chain[1] = ProvenanceStep(
            data=chain[1].data, timestamp=chain[1].timestamp,
            actor="impostor", hash=chain[1].hash,
        )
        r = cps.verify(chain)
        assert r.valid is False
        assert r.failed_step == 1

    def test_modified_timestamp_detected(self, cps):
        chain = _make_chain(4)
        # Adjust timestamp upward (still monotonic) but leave hash stale
        new_ts = chain[1].timestamp + 0.5
        chain[1] = ProvenanceStep(
            data=chain[1].data, timestamp=new_ts,
            actor=chain[1].actor, hash=chain[1].hash,
        )
        r = cps.verify(chain)
        assert r.valid is False
        assert r.failed_step == 1

    def test_modified_hash_detected(self, cps):
        chain = _make_chain(4)
        bad_hash = hashlib.sha256(b"forged").digest()
        chain[3] = ProvenanceStep(
            data=chain[3].data, timestamp=chain[3].timestamp,
            actor=chain[3].actor, hash=bad_hash,
        )
        r = cps.verify(chain)
        assert r.valid is False
        assert r.failed_step == 3

    def test_first_failure_short_circuits(self, cps):
        """If step 1 fails, we should report step 1, not later steps."""
        chain = _make_chain(5)
        chain[1] = ProvenanceStep(
            data=b"early_tamper", timestamp=chain[1].timestamp,
            actor=chain[1].actor, hash=chain[1].hash,
        )
        r = cps.verify(chain)
        assert r.failed_step == 1


class TestTamperDetectionAtScale:
    """Paper claim: 100% tamper detection on 1,000 modified chains."""

    def test_1000_random_tampers_all_detected(self, cps):
        import random as _random

        rng = _random.Random(42)
        detected = 0
        n_total = 1000
        for _ in range(n_total):
            chain = _make_chain(rng.randint(2, 6))
            target = rng.randint(0, len(chain) - 1)
            new_data = bytes(rng.randint(0, 255) for _ in range(8))
            chain[target] = ProvenanceStep(
                data=new_data, timestamp=chain[target].timestamp,
                actor=chain[target].actor, hash=chain[target].hash,
            )
            r = cps.verify(chain)
            if not r.valid:
                detected += 1
        assert detected == n_total, f"only {detected}/{n_total} tampers detected"


class TestFalsePositiveRate:
    """Paper claim: 0/1,000 false positives on valid chains."""

    def test_1000_valid_chains_all_pass(self, cps):
        import random as _random

        rng = _random.Random(7)
        fp = 0
        n_total = 1000
        for _ in range(n_total):
            chain = _make_chain(rng.randint(1, 8))
            r = cps.verify(chain)
            if not r.valid:
                fp += 1
        assert fp == 0, f"{fp}/{n_total} false positives"


class TestGenesisHashAnchoring:
    def test_genesis_mismatch_detected(self, cps):
        chain = _make_chain(3, genesis=hashlib.sha256(b"GENESIS_A").digest())
        # Verify with wrong genesis
        r = cps.verify(chain, genesis_hash=hashlib.sha256(b"GENESIS_B").digest())
        assert r.valid is False
        assert r.failed_step == 0

    def test_correct_genesis_validates(self, cps):
        g = hashlib.sha256(b"my_genesis").digest()
        chain = _make_chain(3, genesis=g)
        r = cps.verify(chain, genesis_hash=g)
        assert r.valid is True
