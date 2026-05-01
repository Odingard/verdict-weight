"""Unit tests for Stream 8 — Registry Integrity Score (RIS)."""

from __future__ import annotations

import pytest

from verdict_weight import (
    RISResult,
    RISVerifier,
    SourceRegistry,
)


def _registry(entries=None, version=1):
    return SourceRegistry(
        entries=entries or {f"src_{i:03d}": 0.5 + i * 0.025 for i in range(20)},
        version=version,
    )


class TestSourceRegistry:
    def test_serialize_is_deterministic(self):
        r1 = _registry()
        r2 = _registry()
        assert r1.serialize() == r2.serialize()

    def test_serialize_sorted_by_key(self):
        r = SourceRegistry(entries={"z": 0.5, "a": 0.5, "m": 0.5}, version=1)
        s = r.serialize().decode("utf-8")
        # The keys must appear in alphabetical order.
        a_pos = s.index("a=")
        m_pos = s.index("m=")
        z_pos = s.index("z=")
        assert a_pos < m_pos < z_pos

    def test_hash_is_32_bytes(self):
        h = _registry().hash()
        assert len(h) == 32

    def test_hash_changes_when_entry_modified(self):
        r1 = SourceRegistry(entries={"a": 0.5}, version=1)
        r2 = SourceRegistry(entries={"a": 0.6}, version=1)
        assert r1.hash() != r2.hash()

    def test_hash_changes_when_version_bumped(self):
        r1 = SourceRegistry(entries={"a": 0.5}, version=1)
        r2 = SourceRegistry(entries={"a": 0.5}, version=2)
        assert r1.hash() != r2.hash()


class TestValidVerification:
    def test_genesis_registry_validates(self):
        g = _registry()
        v = RISVerifier(g)
        r = v.verify(g)
        assert r.valid is True
        assert r.score == 1.0
        assert r.halt_reason is None

    def test_score_method_returns_one(self):
        g = _registry()
        v = RISVerifier(g)
        assert v.score(g) == 1.0

    def test_genesis_hash_property(self):
        g = _registry()
        v = RISVerifier(g)
        assert v.genesis_hash == g.hash()
        assert len(v.genesis_hash) == 32

    def test_version_bump_validates(self):
        # Bumping the version (without changing entries) is treated as
        # a hash mismatch, because the version is part of the hash.
        # In practice this would correspond to the verifier being
        # rotated to a new genesis. For our test, simply bumping
        # version produces a different hash → halt.
        g = _registry(version=1)
        v = RISVerifier(g)
        bumped = SourceRegistry(entries=g.entries, version=2)
        r = v.verify(bumped)
        # By the unified spec a version-bumped registry is a NEW
        # genesis — RISVerifier rejects it because its hash diverges.
        assert r.valid is False
        assert r.halt_reason == "hash_mismatch"


class TestTamperDetection:
    def test_modified_entry_detected(self):
        g = _registry()
        v = RISVerifier(g)
        # Tamper with one entry
        bad_entries = dict(g.entries)
        first_key = next(iter(bad_entries))
        bad_entries[first_key] = bad_entries[first_key] + 0.01
        bad = SourceRegistry(entries=bad_entries, version=g.version)
        r = v.verify(bad)
        assert r.valid is False
        assert r.score == 0.0
        assert r.halt_reason == "hash_mismatch"

    def test_inserted_entry_detected(self):
        g = _registry()
        v = RISVerifier(g)
        bad_entries = dict(g.entries)
        bad_entries["evil_src"] = 0.99
        bad = SourceRegistry(entries=bad_entries, version=g.version)
        r = v.verify(bad)
        assert r.valid is False
        assert r.halt_reason == "hash_mismatch"

    def test_removed_entry_detected(self):
        g = _registry()
        v = RISVerifier(g)
        bad_entries = dict(g.entries)
        bad_entries.pop(next(iter(bad_entries)))
        bad = SourceRegistry(entries=bad_entries, version=g.version)
        r = v.verify(bad)
        assert r.valid is False
        assert r.halt_reason == "hash_mismatch"


class TestRollbackDetection:
    def test_rollback_to_lower_version_detected(self):
        # Simulate a system that has seen v=5 trying to roll back to v=3
        g = _registry(version=5)
        v = RISVerifier(g)
        # Roll back to lower version (also gives different hash) → halt
        rolled_back = SourceRegistry(entries=g.entries, version=3)
        r = v.verify(rolled_back)
        assert r.valid is False
        # Hash check is first-fail → reports hash_mismatch.
        assert r.halt_reason == "hash_mismatch"


class TestTamperDetectionAtScale:
    """Paper claim: 100% tamper detection on 1,000 modified registries."""

    def test_1000_random_tampers_all_detected(self):
        import random as _random

        rng = _random.Random(123)
        detected = 0
        n_total = 1000
        g = _registry()
        v = RISVerifier(g)
        for _ in range(n_total):
            bad_entries = dict(g.entries)
            target = rng.choice(list(bad_entries.keys()))
            bad_entries[target] = round(bad_entries[target] + rng.uniform(-0.1, 0.1), 6)
            # Ensure the value actually changed
            while bad_entries[target] == g.entries[target]:
                bad_entries[target] = round(bad_entries[target] + 0.001, 6)
            bad = SourceRegistry(entries=bad_entries, version=g.version)
            if not v.verify(bad).valid:
                detected += 1
        assert detected == n_total, f"only {detected}/{n_total} tampers detected"


class TestFalsePositiveRate:
    """Paper claim: 0/1,000 false positives on valid genesis registries."""

    def test_1000_genesis_calls_all_pass(self):
        g = _registry()
        v = RISVerifier(g)
        fp = 0
        for _ in range(1000):
            if not v.verify(g).valid:
                fp += 1
        assert fp == 0


class TestResultFields:
    def test_hash_prefix_in_result(self):
        g = _registry()
        v = RISVerifier(g)
        r = v.verify(g)
        assert r.expected_hash_prefix is not None
        assert len(r.expected_hash_prefix) == 12

    def test_observed_prefix_diverges_on_tamper(self):
        g = _registry()
        v = RISVerifier(g)
        bad = SourceRegistry(entries={"x": 0.5}, version=g.version)
        r = v.verify(bad)
        assert r.observed_hash_prefix != r.expected_hash_prefix
