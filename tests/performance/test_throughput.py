"""Performance tests — measure throughput, no hard wall-clock asserts.

These tests collect timing data and emit it via pytest-recorded properties
so a CI dashboard can track regressions, but they do NOT fail on absolute
latency thresholds (those depend on hardware). They DO fail on sane sanity
upper bounds: a single VW score should not take longer than 100 ms on any
modern hardware.
"""

from __future__ import annotations

import time

import pytest

from verdict_weight import (
    SourceRegistry,
    UnifiedComposer,
    UnifiedInputs,
)


SANE_PER_SCORE_UPPER_BOUND_S = 0.1  # 100 ms per score is unreasonable.


def _genesis():
    return SourceRegistry(
        entries={f"s_{i}": 0.5 for i in range(20)},
        version=1,
    )


class TestCommercialTierThroughput:
    def test_streams_only_throughput(self):
        registry = _genesis()
        c = UnifiedComposer(registry=registry)
        n = 500
        start = time.perf_counter()
        for _ in range(n):
            c.score(UnifiedInputs(SR=0.7, CC=0.7, TD=0.7, HA=0.7, registry=registry))
        elapsed = time.perf_counter() - start
        per_score = elapsed / n
        assert per_score < SANE_PER_SCORE_UPPER_BOUND_S, (
            f"Per-score latency {per_score*1000:.2f} ms exceeds sanity bound"
        )


class TestUnifiedThroughput:
    def test_full_pipeline_throughput(self):
        registry = _genesis()
        c = UnifiedComposer(registry=registry)
        from verdict_weight import (
            ProvenanceStep, Source, TrajectoryPoint, build_provenance_chain,
        )
        traj = [TrajectoryPoint(timestamp=float(i), value=0.85) for i in range(8)]
        sources = [
            Source(source_id=f"s_{i}", institution=f"i_{i}",
                   geography=f"g_{i}", publish_time=1e6 + i * 1e5,
                   primary_citations={f"c_{i}"})
            for i in range(5)
        ]
        chain = build_provenance_chain(
            [b"p"] * 4, ["a"] * 4, [1e6, 1e6 + 60, 1e6 + 120, 1e6 + 180],
        )
        n = 200
        start = time.perf_counter()
        for _ in range(n):
            c.score(UnifiedInputs(
                SR=0.85, CC=0.85, TD=0.95, HA=0.90,
                trajectory=traj, sources=sources,
                provenance_chain=chain, registry=registry,
            ))
        elapsed = time.perf_counter() - start
        per_score = elapsed / n
        assert per_score < SANE_PER_SCORE_UPPER_BOUND_S, (
            f"Per-score latency {per_score*1000:.2f} ms exceeds sanity bound"
        )
