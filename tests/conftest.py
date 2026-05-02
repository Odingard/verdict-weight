"""Shared pytest fixtures for the VERDICT WEIGHT test suite."""

from __future__ import annotations

import random

import pytest

from verdict_weight import (
    CPSVerifier,
    CTCAnalyzer,
    RISVerifier,
    SISAnalyzer,
    SourceRegistry,
    UnifiedComposer,
)


@pytest.fixture
def seed() -> int:
    return 42


@pytest.fixture
def rng(seed: int) -> random.Random:
    return random.Random(seed)


@pytest.fixture
def genesis_registry() -> SourceRegistry:
    return SourceRegistry(
        entries={f"src_{i:03d}": round(0.5 + i * 0.025, 3) for i in range(20)},
        version=1,
    )


@pytest.fixture
def composer(genesis_registry: SourceRegistry) -> UnifiedComposer:
    return UnifiedComposer(registry=genesis_registry)


@pytest.fixture
def ctc() -> CTCAnalyzer:
    return CTCAnalyzer()


@pytest.fixture
def sis() -> SISAnalyzer:
    return SISAnalyzer()


@pytest.fixture
def cps() -> CPSVerifier:
    return CPSVerifier()


@pytest.fixture
def ris(genesis_registry: SourceRegistry) -> RISVerifier:
    return RISVerifier(genesis_registry)
