"""
Stream 7 — Cryptographic Provenance Score (CPS) — Hash-Chain Integrity
======================================================================

Defeats attack class AC-4 (Provenance Forgery). The intelligence
provenance record is the chain of custody from signal acquisition
through every transformation and transmission step. Each step
generates a SHA-256 hash block:

    H_n = SHA-256( H_{n-1} || data_n || timestamp_n || actor_n )

where ``H_0`` is the genesis hash, anchored to the source registry at
the moment of signal acquisition. The chain is **valid** if and only
if ``H_n`` can be independently recomputed from ``H_0`` through every
intermediate step.

The CPS score is binary:

    S_CPS = 1   if the full hash chain validates
    S_CPS = 0   (HALT) if any step fails verification

Partial validation is **not defined**. A chain that validates through
step ``n − 1`` but fails at step ``n`` provides no confidence about
steps ``1`` through ``n − 1`` — the modification may have propagated
backward through the chain.

The chain is append-only. The ``ProvenanceChain`` helper enforces
both append-only semantics and timestamp monotonicity at construction
time, so any in-memory chain that survives construction is
syntactically well-formed; CPS verifies the cryptographic content.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


# ─────────────────────────────────────────────────────────────
# Provenance primitives
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProvenanceStep:
    """A single step in a provenance chain.

    Attributes
    ----------
    data:
        Raw bytes of the signal payload at this step.
    timestamp:
        Unix epoch seconds (float, second resolution is sufficient
        because the timestamp is fed verbatim into the SHA-256 input).
    actor:
        Cryptographic identity (or stable string identifier) of the
        processing entity that produced this step.
    hash:
        The SHA-256 hash digest of this step, computed from
        ``H_{n-1} || data || timestamp || actor``. The genesis step
        has ``H_{n-1} = b""``.
    """

    data: bytes
    timestamp: float
    actor: str
    hash: bytes


@dataclass(frozen=True)
class CPSResult:
    """Output of a CPS verification.

    Attributes
    ----------
    valid:
        ``True`` iff the full chain validates.
    score:
        Binary score: ``1.0`` if ``valid``, else ``0.0`` (HALT).
    failed_step:
        Index of the first step that failed validation, or ``None``
        if all steps validated.
    chain_length:
        Number of steps in the verified chain.
    failure_reason:
        Human-readable reason for failure, or ``None`` on success.
    """

    valid: bool
    score: float
    failed_step: Optional[int]
    chain_length: int
    failure_reason: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Hash helpers
# ─────────────────────────────────────────────────────────────


def _step_hash(
    prev_hash: bytes, data: bytes, timestamp: float, actor: str
) -> bytes:
    """Compute ``SHA-256( H_{n-1} || data || timestamp || actor )``.

    Inputs are concatenated in canonical byte form:
      * ``H_{n-1}`` is 32 raw bytes (or empty for the genesis step).
      * ``data`` is the raw payload bytes.
      * ``timestamp`` is the IEEE-754 binary representation of the
        floating-point timestamp (8 bytes, big-endian).
      * ``actor`` is encoded as UTF-8.
    """
    import struct

    h = hashlib.sha256()
    h.update(prev_hash)
    h.update(data)
    h.update(struct.pack(">d", timestamp))
    h.update(actor.encode("utf-8"))
    return h.digest()


def build_provenance_chain(
    payloads: Sequence[bytes],
    actors: Sequence[str],
    timestamps: Sequence[float],
    genesis_hash: Optional[bytes] = None,
) -> List[ProvenanceStep]:
    """Construct a syntactically-valid provenance chain.

    Parameters
    ----------
    payloads:
        Per-step signal payloads.
    actors:
        Per-step actor identities.
    timestamps:
        Per-step timestamps. Must be monotonically non-decreasing.
    genesis_hash:
        Optional 32-byte genesis hash. If ``None``, the empty bytes
        ``b""`` are used (i.e. the chain self-anchors).

    Returns
    -------
    List[ProvenanceStep]
        The constructed chain. Each step's hash is computed by
        ``_step_hash`` so the chain is internally consistent.

    Raises
    ------
    ValueError
        On length mismatch or non-monotonic timestamps.
    """
    n = len(payloads)
    if len(actors) != n or len(timestamps) != n:
        raise ValueError("payloads, actors, and timestamps must be the same length")
    if any(timestamps[i] < timestamps[i - 1] for i in range(1, n)):
        raise ValueError("timestamps must be monotonically non-decreasing")

    prev = genesis_hash or b""
    chain: List[ProvenanceStep] = []
    for i in range(n):
        h = _step_hash(prev, payloads[i], timestamps[i], actors[i])
        chain.append(
            ProvenanceStep(
                data=payloads[i],
                timestamp=timestamps[i],
                actor=actors[i],
                hash=h,
            )
        )
        prev = h
    return chain


# ─────────────────────────────────────────────────────────────
# Verifier
# ─────────────────────────────────────────────────────────────


class CPSVerifier:
    """Cryptographic Provenance Score verifier.

    Stateless. Accepts a full provenance chain and returns a binary
    validation result.
    """

    def verify(
        self,
        chain: Sequence[ProvenanceStep],
        genesis_hash: Optional[bytes] = None,
    ) -> CPSResult:
        """Verify a provenance chain.

        Parameters
        ----------
        chain:
            The provenance chain to verify.
        genesis_hash:
            Optional 32-byte genesis hash. If ``None``, the empty
            bytes ``b""`` are used. The first step in ``chain`` must
            have been constructed with this same genesis.

        Returns
        -------
        CPSResult
            Binary validation result with diagnostic metadata.
        """
        n = len(chain)
        if n == 0:
            return CPSResult(
                valid=False,
                score=0.0,
                failed_step=None,
                chain_length=0,
                failure_reason="empty chain",
            )

        prev = genesis_hash or b""
        for i, step in enumerate(chain):
            if not isinstance(step.hash, bytes) or len(step.hash) != 32:
                return CPSResult(
                    valid=False,
                    score=0.0,
                    failed_step=i,
                    chain_length=n,
                    failure_reason=(
                        f"step {i}: hash must be 32 bytes, got {len(step.hash)}"
                    ),
                )
            expected = _step_hash(prev, step.data, step.timestamp, step.actor)
            if expected != step.hash:
                return CPSResult(
                    valid=False,
                    score=0.0,
                    failed_step=i,
                    chain_length=n,
                    failure_reason=(
                        f"step {i}: hash mismatch — "
                        f"expected {expected.hex()[:12]}..., "
                        f"got {step.hash.hex()[:12]}..."
                    ),
                )
            # Timestamp monotonicity (re-check)
            if i > 0 and step.timestamp < chain[i - 1].timestamp:
                return CPSResult(
                    valid=False,
                    score=0.0,
                    failed_step=i,
                    chain_length=n,
                    failure_reason=f"step {i}: timestamp regression",
                )
            prev = step.hash

        return CPSResult(
            valid=True,
            score=1.0,
            failed_step=None,
            chain_length=n,
            failure_reason=None,
        )

    def score(
        self,
        chain: Sequence[ProvenanceStep],
        genesis_hash: Optional[bytes] = None,
    ) -> float:
        """Convenience wrapper that returns just the binary CPS score."""
        return self.verify(chain, genesis_hash).score
