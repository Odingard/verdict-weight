"""
Unified Composition Rule — Eight-Stream VERDICT WEIGHT
=======================================================

Combines the commercial tier (Streams 1–4 from ``core.py``) with the
adversarial detection tier (Stream 5) and the government tier
(Streams 6–8) into a single certified Consequence Weight.

Per Section 6 of the unified paper:

    CW_certified = HALT  if  S_RIS = 0
    CW_certified = HALT  if  S_CPS = 0
    CW_certified = f(S1..5, S_SIS)  otherwise

where:

    f = CW_base · S_CTC^γ · S_SIS^δ

* ``CW_base`` is the commercial-tier output from ``VerdictEngine``
  (Streams 1–4: SR / DI / TD / HA fused into a doubt-penalized
  consequence weight).
* ``S_CTC`` is the trajectory score from Stream 5.
* ``S_SIS`` is the source-independence score from Stream 6.
* ``γ`` and ``δ`` are deployment-tier exponents:

    - Government tier:  γ = δ = 1.0  (full sensitivity to
      adversarial/independence signals).
    - Commercial tier:  γ = δ = 0.5  (reduced sensitivity for
      non-contested deployments).

The HALT states from Streams 7 (CPS) and 8 (RIS) are absorbing —
once triggered, no CW is produced regardless of other streams. This
implements the security principle that a compromised foundation
invalidates all scoring built upon it.

Tier ordering for short-circuit evaluation (cheapest gate first):

    RIS (O(1))  →  CPS (O(n))  →  SIS (O(K²))  →  CTC (O(w))
                →  Commercial tier (Streams 1–4, O(1))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from .core import (
    ContextType,
    StreamScorer,
    VerdictEngine,
    WeightProfile,
    WEIGHT_PROFILES,
)
from .streams import (
    CPSResult,
    CPSVerifier,
    CTCAnalyzer,
    CTCResult,
    ProvenanceStep,
    RISResult,
    RISVerifier,
    SISAnalyzer,
    SISResult,
    Source,
    SourceRegistry,
    TrajectoryPoint,
)


class DeploymentTier(str, Enum):
    """Deployment tier — sets exponents γ and δ in the composition rule."""

    COMMERCIAL = "commercial"   # γ = δ = 0.5
    ADVERSARIAL = "adversarial"  # γ = 1.0, δ = 0.5
    GOVERNMENT = "government"    # γ = δ = 1.0


_TIER_EXPONENTS = {
    DeploymentTier.COMMERCIAL:  (0.5, 0.5),
    DeploymentTier.ADVERSARIAL: (1.0, 0.5),
    DeploymentTier.GOVERNMENT:  (1.0, 1.0),
}


@dataclass(frozen=True)
class UnifiedResult:
    """Output of a unified eight-stream scoring event.

    Attributes
    ----------
    cw_certified:
        The certified Consequence Weight, ``∈ [0, 1]``. ``None`` if
        the pipeline halted before composition (RIS or CPS failed).
    halted:
        ``True`` iff a HALT-class stream (RIS or CPS) failed.
    halt_reason:
        Human-readable halt reason; ``None`` on success.
    halted_at:
        Stream identifier where halt occurred (``"RIS"`` or
        ``"CPS"``); ``None`` on success.
    streams:
        Dict of all per-stream scores (``S1..S8``) for diagnostics.
    cw_base:
        Commercial-tier ``CW_base`` from Streams 1–4. ``None`` on
        halt.
    action_tier:
        Action tier label (CRITICAL / HIGH / MEDIUM / LOW / NOISE / HALT).
    interpretation:
        Human-readable verdict text.
    deployment_tier:
        Deployment tier used for this scoring event.
    """

    cw_certified: Optional[float]
    halted: bool
    halt_reason: Optional[str]
    halted_at: Optional[str]
    streams: dict
    cw_base: Optional[float]
    action_tier: str
    interpretation: str
    deployment_tier: DeploymentTier
    metrics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class UnifiedInputs:
    """Inputs to a unified scoring event.

    The four government / adversarial streams are optional in
    commercial-tier deployments — when omitted, the corresponding
    score defaults to 1.0 (no penalty).

    Attributes
    ----------
    SR, CC, TD, HA:
        Pre-computed commercial-tier stream values, ``∈ [0, 1]``.
        See ``StreamScorer`` for raw-input → stream conversions.
    trajectory:
        Optional time-ordered list of ``TrajectoryPoint`` for Stream 5.
        If omitted, ``S_CTC = 1.0``.
    sources:
        Optional list of ``Source`` records for Stream 6. If omitted,
        ``S_SIS = 1.0``.
    provenance_chain:
        Optional provenance chain for Stream 7. If omitted, ``S_CPS = 1.0``
        (provenance verification skipped).
    registry:
        Optional ``SourceRegistry`` snapshot for Stream 8. If omitted,
        ``S_RIS = 1.0`` (registry verification skipped).
    context:
        Context for weight-profile selection.
    deployment_tier:
        Deployment tier; selects γ and δ exponents.
    """

    SR: float
    CC: float
    TD: float
    HA: float
    trajectory: Optional[Sequence[TrajectoryPoint]] = None
    sources: Optional[Sequence[Source]] = None
    provenance_chain: Optional[Sequence[ProvenanceStep]] = None
    registry: Optional[SourceRegistry] = None
    context: ContextType = ContextType.CYBERSECURITY_GENERAL
    deployment_tier: DeploymentTier = DeploymentTier.GOVERNMENT


class UnifiedComposer:
    """Eight-stream composition pipeline.

    Stateful only with respect to the RIS verifier (which holds the
    genesis registry hash). All other streams are stateless.

    Parameters
    ----------
    registry:
        Optional genesis registry. Required if any scoring event will
        provide a ``registry`` input — the genesis is captured at
        construction time.
    profiles:
        Optional weight-profile registry. Defaults to ``WEIGHT_PROFILES``.
    ctc_analyzer / sis_analyzer / cps_verifier:
        Optional pre-configured stream analyzers. Defaults are
        constructed with paper-default thresholds.
    """

    def __init__(
        self,
        registry: Optional[SourceRegistry] = None,
        profiles: Optional[dict] = None,
        ctc_analyzer: Optional[CTCAnalyzer] = None,
        sis_analyzer: Optional[SISAnalyzer] = None,
        cps_verifier: Optional[CPSVerifier] = None,
    ):
        self._profiles = profiles or dict(WEIGHT_PROFILES)
        self._ctc = ctc_analyzer or CTCAnalyzer()
        self._sis = sis_analyzer or SISAnalyzer()
        self._cps = cps_verifier or CPSVerifier()
        self._ris: Optional[RISVerifier] = (
            RISVerifier(registry) if registry is not None else None
        )

    @property
    def ris_verifier(self) -> Optional[RISVerifier]:
        """The internal RIS verifier (or ``None`` if no registry)."""
        return self._ris

    def score(self, inputs: UnifiedInputs) -> UnifiedResult:
        """Score a single intelligence event through all eight streams."""
        gamma, delta = _TIER_EXPONENTS[inputs.deployment_tier]
        streams: dict = {
            "SR": inputs.SR, "CC": inputs.CC, "TD": inputs.TD, "HA": inputs.HA,
            "S_CTC": 1.0, "S_SIS": 1.0, "S_CPS": 1.0, "S_RIS": 1.0,
        }
        metrics: dict = {}

        # Stage 1: RIS (O(1)) — fail fast on registry tamper
        if inputs.registry is not None:
            if self._ris is None:
                # No genesis registered. Fail closed.
                return UnifiedResult(
                    cw_certified=None,
                    halted=True,
                    halt_reason="No genesis registry on this UnifiedComposer",
                    halted_at="RIS",
                    streams=streams,
                    cw_base=None,
                    action_tier="HALT",
                    interpretation=(
                        "Composer was not initialized with a genesis registry. "
                        "Construct with `UnifiedComposer(registry=...)`."
                    ),
                    deployment_tier=inputs.deployment_tier,
                    metrics=metrics,
                )
            ris_result: RISResult = self._ris.verify(inputs.registry)
            streams["S_RIS"] = ris_result.score
            metrics["ris"] = {"halt_reason": ris_result.halt_reason}
            if not ris_result.valid:
                return UnifiedResult(
                    cw_certified=None,
                    halted=True,
                    halt_reason=f"RIS halt: {ris_result.halt_reason}",
                    halted_at="RIS",
                    streams=streams,
                    cw_base=None,
                    action_tier="HALT",
                    interpretation=(
                        "Source registry integrity failure. All scoring halted "
                        "until registry is restored or rotated."
                    ),
                    deployment_tier=inputs.deployment_tier,
                    metrics=metrics,
                )

        # Stage 2: CPS (O(n)) — provenance hash chain
        if inputs.provenance_chain is not None:
            cps_result: CPSResult = self._cps.verify(inputs.provenance_chain)
            streams["S_CPS"] = cps_result.score
            metrics["cps"] = {
                "valid": cps_result.valid,
                "failed_step": cps_result.failed_step,
                "failure_reason": cps_result.failure_reason,
            }
            if not cps_result.valid:
                return UnifiedResult(
                    cw_certified=None,
                    halted=True,
                    halt_reason=f"CPS halt: {cps_result.failure_reason}",
                    halted_at="CPS",
                    streams=streams,
                    cw_base=None,
                    action_tier="HALT",
                    interpretation=(
                        "Provenance chain integrity failure. Signal lineage "
                        "cannot be verified; scoring halted."
                    ),
                    deployment_tier=inputs.deployment_tier,
                    metrics=metrics,
                )

        # Stage 3: SIS (O(K²)) — independence matrix
        if inputs.sources is not None:
            sis_result: SISResult = self._sis.analyze(inputs.sources)
            streams["S_SIS"] = sis_result.score
            metrics["sis"] = {
                "cc_eff": sis_result.cc_eff,
                "cc_raw": sis_result.cc_raw,
                "pattern": sis_result.pattern,
            }

        # Stage 4: CTC (O(w)) — trajectory analysis
        if inputs.trajectory is not None:
            ctc_result: CTCResult = self._ctc.analyze(inputs.trajectory)
            streams["S_CTC"] = ctc_result.score
            metrics["ctc"] = {
                "pattern": ctc_result.pattern.value,
                "confidence": ctc_result.confidence,
            }

        # Stage 5: Commercial tier (Streams 1–4) — CW_base
        profile = self._profiles[inputs.context]
        SS, DI, CW_base = VerdictEngine.score(
            inputs.SR, inputs.CC, inputs.TD, inputs.HA, profile
        )
        metrics["commercial"] = {"SS": SS, "DI": DI, "CW_base": CW_base}

        # Stage 6: Composition  CW_certified = CW_base · S_CTC^γ · S_SIS^δ
        cw_certified = float(
            np.clip(
                CW_base
                * (streams["S_CTC"] ** gamma)
                * (streams["S_SIS"] ** delta),
                0.0,
                1.0,
            )
        )
        tier, text = VerdictEngine.interpret(cw_certified, DI)
        return UnifiedResult(
            cw_certified=cw_certified,
            halted=False,
            halt_reason=None,
            halted_at=None,
            streams=streams,
            cw_base=CW_base,
            action_tier=tier,
            interpretation=text,
            deployment_tier=inputs.deployment_tier,
            metrics=metrics,
        )


# Convenience: module-level scorer with a dedicated public API
def compose(inputs: UnifiedInputs, composer: Optional[UnifiedComposer] = None) -> UnifiedResult:
    """One-shot unified scoring with an internal composer.

    For repeated scoring against the same registry, construct a single
    ``UnifiedComposer`` and call ``.score()`` directly so the genesis
    hash and version high-water mark persist across calls.
    """
    composer = composer or UnifiedComposer(registry=inputs.registry)
    return composer.score(inputs)
