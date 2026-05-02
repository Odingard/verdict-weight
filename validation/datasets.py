"""
Synthetic dataset generators for VERDICT WEIGHT validation.

All generators are deterministic in the seed parameter. The defaults
in this module are the canonical seeds used to produce the numbers
cited in ``VALIDATION.md``.

The generators model five attack classes and matching legitimate
signals:

  AC-1  Source Spoofing               (single high-SR source w/ low corroboration)
  AC-2  Corroboration Manufacturing   ("Curveball" — K shared-origin sources)
  AC-3  Trajectory Fabrication        (Pattern C spike-then-collapse)
  AC-4  Provenance Forgery            (tampered hash chain)
  AC-5  Registry Compromise           (modified registry content)

Legitimate signals exhibit Pattern A / B / D trajectories, genuinely
independent sources, valid provenance chains, and untampered
registries.
"""

from __future__ import annotations

import hashlib
import random
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from verdict_weight import (
    ProvenanceStep,
    Source,
    SourceRegistry,
    TrajectoryPoint,
    build_provenance_chain,
)


# ─────────────────────────────────────────────────────────────
# Sample container
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Sample:
    """A single labelled validation sample.

    Attributes
    ----------
    label:
        ``"legitimate"`` or ``"adversarial"``.
    attack_class:
        ``None`` for legitimate signals, otherwise one of
        ``"AC-1"``..``"AC-5"``.
    SR / CC / TD / HA:
        Pre-computed commercial-tier stream values, ``∈ [0, 1]``.
    trajectory:
        Time-ordered list of trajectory points.
    sources:
        List of corroborating sources.
    provenance_chain:
        Provenance chain.
    registry:
        Registry snapshot (always the genesis registry — registry
        tamper is signalled via ``registry_tampered=True``).
    registry_tampered:
        ``True`` for AC-5 samples; the validator should detect this
        via the RIS verifier.
    """

    label: str
    attack_class: Optional[str]
    SR: float
    CC: float
    TD: float
    HA: float
    trajectory: Tuple[TrajectoryPoint, ...]
    sources: Tuple[Source, ...]
    provenance_chain: Tuple[ProvenanceStep, ...]
    registry: SourceRegistry
    registry_tampered: bool = False


# ─────────────────────────────────────────────────────────────
# Trajectory generators
# ─────────────────────────────────────────────────────────────


def _pattern_a(rng: random.Random, n: int = 10) -> List[TrajectoryPoint]:
    """Stable high — legitimate strong signal."""
    base = rng.uniform(0.78, 0.92)
    return [
        TrajectoryPoint(timestamp=float(t), value=max(0.0, min(1.0, base + rng.gauss(0, 0.025))))
        for t in range(n)
    ]


def _pattern_b(rng: random.Random, n: int = 10) -> List[TrajectoryPoint]:
    """Stable low — legitimate noise."""
    base = rng.uniform(0.10, 0.25)
    return [
        TrajectoryPoint(timestamp=float(t), value=max(0.0, min(1.0, base + rng.gauss(0, 0.025))))
        for t in range(n)
    ]


def _pattern_c(rng: random.Random, n: int = 10) -> List[TrajectoryPoint]:
    """Spike-then-collapse — adversarial AC-3."""
    peak_idx = rng.randint(1, max(1, int(n * 0.5)))
    spike_height = rng.uniform(0.85, 0.97)
    collapse_floor = rng.uniform(0.05, 0.20)
    points = []
    for t in range(n):
        if t < peak_idx:
            v = rng.uniform(0.20, 0.45)
        elif t == peak_idx:
            v = spike_height + rng.gauss(0, 0.02)
        else:
            # exponential decay to floor
            decay = (t - peak_idx) / max(1, n - peak_idx - 1)
            v = spike_height + (collapse_floor - spike_height) * decay
            v += rng.gauss(0, 0.02)
        points.append(TrajectoryPoint(timestamp=float(t), value=max(0.0, min(1.0, v))))
    return points


def _pattern_d(rng: random.Random, n: int = 10) -> List[TrajectoryPoint]:
    """Gradual buildup — legitimate corroboration accumulation."""
    start = rng.uniform(0.10, 0.30)
    end = rng.uniform(0.78, 0.95)
    points = []
    for t in range(n):
        frac = t / max(1, n - 1)
        v = start + frac * (end - start) + rng.gauss(0, 0.025)
        points.append(TrajectoryPoint(timestamp=float(t), value=max(0.0, min(1.0, v))))
    return points


# ─────────────────────────────────────────────────────────────
# Source generators
# ─────────────────────────────────────────────────────────────


_INSTITUTIONS = [
    "Mandiant", "CrowdStrike", "Microsoft", "Cisco_Talos", "Recorded_Future",
    "Palo_Alto_Unit42", "Kaspersky", "ESET", "Symantec", "Trend_Micro",
    "Group_IB", "FireEye", "Sophos_Labs", "F_Secure", "BitDefender",
]
_GEOGRAPHIES = ["US", "EU", "JP", "UK", "IL", "AU", "CA", "SG", "KR", "DE"]


def _independent_sources(rng: random.Random, k: int) -> List[Source]:
    """Generate ``k`` genuinely-independent sources."""
    insts = rng.sample(_INSTITUTIONS, min(k, len(_INSTITUTIONS)))
    while len(insts) < k:
        insts.append(f"Indep_Inst_{rng.randint(1000, 9999)}")
    geos = [rng.choice(_GEOGRAPHIES) for _ in range(k)]
    base_t = rng.uniform(0, 1_000_000)
    times = [base_t + rng.uniform(2 * 86400, 30 * 86400) * i for i in range(k)]
    return [
        Source(
            source_id=f"src_{i:03d}_{rng.randint(0, 999):03d}",
            institution=insts[i],
            geography=geos[i],
            publish_time=times[i],
            primary_citations={f"cite_{rng.randint(0, 9999):04d}" for _ in range(rng.randint(2, 5))},
        )
        for i in range(k)
    ]


def _curveball_sources(rng: random.Random, k: int) -> List[Source]:
    """K shared-origin sources — the AC-2 Curveball pattern."""
    inst = rng.choice(_INSTITUTIONS)
    geo = rng.choice(_GEOGRAPHIES)
    base_t = rng.uniform(0, 1_000_000)
    shared_citations = {f"cite_{rng.randint(0, 9999):04d}" for _ in range(3)}
    return [
        Source(
            source_id=f"channel_{i:02d}_{rng.randint(0, 999):03d}",
            institution=inst,
            geography=geo,
            publish_time=base_t + rng.uniform(0, 600),  # all within timing threshold
            primary_citations=set(shared_citations),
        )
        for i in range(k)
    ]


def _spoofed_single_source(rng: random.Random) -> List[Source]:
    """Single high-SR source with no corroboration — AC-1 surface."""
    inst = rng.choice(_INSTITUTIONS)
    return [
        Source(
            source_id=f"spoof_{rng.randint(0, 9999):04d}",
            institution=inst,
            geography=rng.choice(_GEOGRAPHIES),
            publish_time=rng.uniform(0, 1_000_000),
            primary_citations={f"cite_{rng.randint(0, 9999):04d}"},
        )
    ]


# ─────────────────────────────────────────────────────────────
# Provenance generators
# ─────────────────────────────────────────────────────────────


def _valid_chain(rng: random.Random, n_steps: int = 4) -> List[ProvenanceStep]:
    """Construct a syntactically-valid provenance chain."""
    payloads = [bytes(f"step_{i}_{rng.randint(0, 9999)}", "utf-8") for i in range(n_steps)]
    actors = [f"actor_{rng.randint(0, 99)}" for _ in range(n_steps)]
    base_t = rng.uniform(1_000_000, 10_000_000)
    timestamps: List[float] = []
    cur = base_t
    for _ in range(n_steps):
        timestamps.append(cur)
        cur += rng.uniform(1, 60)
    return build_provenance_chain(payloads, actors, timestamps)


def _tampered_chain(rng: random.Random, n_steps: int = 4) -> List[ProvenanceStep]:
    """Construct a chain whose payload at one step has been altered.

    The ``hash`` field is left as the original — verification will
    fail because the recomputed hash will differ.
    """
    chain = _valid_chain(rng, n_steps)
    if not chain:
        return chain
    target_idx = rng.randint(0, n_steps - 1)
    import dataclasses

    chain[target_idx] = dataclasses.replace(
        chain[target_idx],
        data=bytes(f"forged_{rng.randint(0, 9999)}", "utf-8"),
    )
    return chain


# ─────────────────────────────────────────────────────────────
# Registry generators
# ─────────────────────────────────────────────────────────────


def _genesis_registry(rng: random.Random) -> SourceRegistry:
    """Generate a deterministic genesis registry."""
    entries = {f"src_{i:03d}": round(rng.uniform(0.50, 0.99), 3) for i in range(20)}
    return SourceRegistry(entries=entries, version=1)


def _tampered_registry(genesis: SourceRegistry, rng: random.Random) -> SourceRegistry:
    """Modify one entry of an existing registry."""
    entries = dict(genesis.entries)
    keys = list(entries.keys())
    target = rng.choice(keys)
    # Add a small but detectable change
    entries[target] = round(min(0.999, max(0.001, entries[target] + rng.choice([-0.05, 0.05]))), 3)
    return SourceRegistry(entries=entries, version=genesis.version + 1)


# ─────────────────────────────────────────────────────────────
# Stream-1-4 input generators
# ─────────────────────────────────────────────────────────────


def _streams_legitimate(rng: random.Random) -> Tuple[float, float, float, float]:
    SR = rng.uniform(0.70, 0.95)
    CC = rng.uniform(0.65, 0.90)
    TD = rng.uniform(0.75, 0.95)
    HA = rng.uniform(0.65, 0.90)
    return SR, CC, TD, HA


def _streams_spoofed(rng: random.Random) -> Tuple[float, float, float, float]:
    """High SR, low CC — Stream 2 should detect AC-1."""
    SR = rng.uniform(0.85, 0.97)
    CC = rng.uniform(0.10, 0.30)
    TD = rng.uniform(0.65, 0.90)
    HA = rng.uniform(0.40, 0.70)
    return SR, CC, TD, HA


def _streams_curveball(rng: random.Random) -> Tuple[float, float, float, float]:
    """High SR + apparently-high CC. SIS will reduce effective CC."""
    SR = rng.uniform(0.75, 0.92)
    CC = rng.uniform(0.70, 0.90)
    TD = rng.uniform(0.70, 0.90)
    HA = rng.uniform(0.60, 0.85)
    return SR, CC, TD, HA


# ─────────────────────────────────────────────────────────────
# Top-level sample generators
# ─────────────────────────────────────────────────────────────


def generate_legitimate(rng: random.Random) -> Sample:
    """Generate one legitimate sample (Pattern A/B/D, independent sources)."""
    pattern_choice = rng.choice(["A", "B", "D"])
    if pattern_choice == "A":
        traj = _pattern_a(rng)
    elif pattern_choice == "B":
        traj = _pattern_b(rng)
    else:
        traj = _pattern_d(rng)

    k = rng.randint(2, 5)
    sources = _independent_sources(rng, k)
    chain = _valid_chain(rng)
    registry = _genesis_registry(rng)
    SR, CC, TD, HA = _streams_legitimate(rng)
    return Sample(
        label="legitimate",
        attack_class=None,
        SR=SR, CC=CC, TD=TD, HA=HA,
        trajectory=tuple(traj),
        sources=tuple(sources),
        provenance_chain=tuple(chain),
        registry=registry,
        registry_tampered=False,
    )


def generate_ac1(rng: random.Random) -> Sample:
    """AC-1 Source Spoofing: single high-SR source, no corroboration."""
    traj = _pattern_a(rng)  # signal looks "legitimate" trajectory-wise
    sources = _spoofed_single_source(rng)
    chain = _valid_chain(rng)
    registry = _genesis_registry(rng)
    SR, CC, TD, HA = _streams_spoofed(rng)
    return Sample(
        label="adversarial", attack_class="AC-1",
        SR=SR, CC=CC, TD=TD, HA=HA,
        trajectory=tuple(traj),
        sources=tuple(sources),
        provenance_chain=tuple(chain),
        registry=registry,
    )


def generate_ac2(rng: random.Random) -> Sample:
    """AC-2 Corroboration Manufacturing (Curveball): K shared-origin sources."""
    traj = _pattern_a(rng)
    k = rng.randint(3, 6)
    sources = _curveball_sources(rng, k)
    chain = _valid_chain(rng)
    registry = _genesis_registry(rng)
    SR, CC, TD, HA = _streams_curveball(rng)
    return Sample(
        label="adversarial", attack_class="AC-2",
        SR=SR, CC=CC, TD=TD, HA=HA,
        trajectory=tuple(traj),
        sources=tuple(sources),
        provenance_chain=tuple(chain),
        registry=registry,
    )


def generate_ac3(rng: random.Random) -> Sample:
    """AC-3 Trajectory Fabrication (Pattern C spike-then-collapse)."""
    traj = _pattern_c(rng)
    k = rng.randint(2, 5)
    sources = _independent_sources(rng, k)
    chain = _valid_chain(rng)
    registry = _genesis_registry(rng)
    SR, CC, TD, HA = _streams_legitimate(rng)  # commercial tier looks fine; trajectory betrays
    return Sample(
        label="adversarial", attack_class="AC-3",
        SR=SR, CC=CC, TD=TD, HA=HA,
        trajectory=tuple(traj),
        sources=tuple(sources),
        provenance_chain=tuple(chain),
        registry=registry,
    )


def generate_ac4(rng: random.Random) -> Sample:
    """AC-4 Provenance Forgery: tampered hash chain."""
    traj = _pattern_a(rng)
    sources = _independent_sources(rng, rng.randint(2, 5))
    chain = _tampered_chain(rng)
    registry = _genesis_registry(rng)
    SR, CC, TD, HA = _streams_legitimate(rng)
    return Sample(
        label="adversarial", attack_class="AC-4",
        SR=SR, CC=CC, TD=TD, HA=HA,
        trajectory=tuple(traj),
        sources=tuple(sources),
        provenance_chain=tuple(chain),
        registry=registry,
    )


def generate_ac5(rng: random.Random) -> Sample:
    """AC-5 Registry Compromise: registry content has been modified."""
    traj = _pattern_a(rng)
    sources = _independent_sources(rng, rng.randint(2, 5))
    chain = _valid_chain(rng)
    genesis = _genesis_registry(rng)
    tampered = _tampered_registry(genesis, rng)
    SR, CC, TD, HA = _streams_legitimate(rng)
    return Sample(
        label="adversarial", attack_class="AC-5",
        SR=SR, CC=CC, TD=TD, HA=HA,
        trajectory=tuple(traj),
        sources=tuple(sources),
        provenance_chain=tuple(chain),
        registry=tampered,           # NB: tampered, not genesis
        registry_tampered=True,
    )


# ─────────────────────────────────────────────────────────────
# Top-level factories
# ─────────────────────────────────────────────────────────────


def generate_dataset(
    n: int,
    *,
    seed: int = 42,
    legitimate_fraction: float = 0.5,
    attack_mix: Optional[Sequence[str]] = None,
) -> Tuple[List[Sample], SourceRegistry]:
    """Generate a labelled dataset of size ``n`` plus the genesis registry.

    Parameters
    ----------
    n:
        Total number of samples.
    seed:
        Deterministic seed.
    legitimate_fraction:
        Fraction of samples that are legitimate; the rest are adversarial.
    attack_mix:
        Optional sequence of ``"AC-1"``..``"AC-5"`` controlling the
        adversarial mix. If ``None``, all 5 classes appear with equal
        probability.

    Returns
    -------
    samples:
        List of ``n`` labelled samples.
    registry:
        The single genesis registry shared across all legitimate
        samples and AC-1..AC-4 adversarial samples. AC-5 samples
        ship a tampered copy of this same registry.
    """
    rng = random.Random(seed)
    attack_mix = list(attack_mix) if attack_mix else ["AC-1", "AC-2", "AC-3", "AC-4", "AC-5"]
    generators = {
        "AC-1": generate_ac1,
        "AC-2": generate_ac2,
        "AC-3": generate_ac3,
        "AC-4": generate_ac4,
        "AC-5": generate_ac5,
    }

    # Build a single shared genesis registry. AC-5 samples will
    # build their own tampered copies but with content derived from
    # the same RNG seed, so the AC-5 tampered registry is detectably
    # different from the genesis.
    shared_genesis = SourceRegistry(
        entries={f"src_{i:03d}": round(rng.uniform(0.5, 0.99), 3) for i in range(20)},
        version=1,
    )

    samples: List[Sample] = []
    n_legit = int(n * legitimate_fraction)
    n_adv = n - n_legit
    for _ in range(n_legit):
        s = generate_legitimate(rng)
        samples.append(_with_shared_registry(s, shared_genesis))

    for i in range(n_adv):
        ac = attack_mix[i % len(attack_mix)]
        s = generators[ac](rng)
        if ac == "AC-5":
            tampered = _tampered_registry(shared_genesis, rng)
            samples.append(_with_shared_registry(s, tampered, registry_tampered=True))
        else:
            samples.append(_with_shared_registry(s, shared_genesis))

    rng.shuffle(samples)
    return samples, shared_genesis


def _with_shared_registry(
    sample: Sample, registry: SourceRegistry, registry_tampered: bool = False
) -> Sample:
    """Return a copy of ``sample`` with its registry replaced."""
    import dataclasses

    return dataclasses.replace(
        sample, registry=registry, registry_tampered=registry_tampered
    )


# ─────────────────────────────────────────────────────────────
# Trajectory-only datasets (for Stream 5 isolated validation)
# ─────────────────────────────────────────────────────────────


def generate_trajectory_dataset(
    n: int, *, seed: int = 42, adversarial_fraction: float = 0.5
) -> List[Tuple[List[TrajectoryPoint], str]]:
    """Generate a dataset of trajectories labelled ``"adversarial"`` or ``"legitimate"``.

    Adversarial = Pattern C; legitimate = Pattern A / B / D mix.
    """
    rng = random.Random(seed)
    n_adv = int(n * adversarial_fraction)
    out: List[Tuple[List[TrajectoryPoint], str]] = []
    for _ in range(n_adv):
        out.append((_pattern_c(rng), "adversarial"))
    n_legit = n - n_adv
    legit_patterns = [_pattern_a, _pattern_b, _pattern_d]
    for i in range(n_legit):
        out.append((legit_patterns[i % 3](rng), "legitimate"))
    rng.shuffle(out)
    return out


__all__ = [
    "Sample",
    "generate_dataset",
    "generate_trajectory_dataset",
    "generate_legitimate",
    "generate_ac1",
    "generate_ac2",
    "generate_ac3",
    "generate_ac4",
    "generate_ac5",
]
