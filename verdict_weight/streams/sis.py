"""
Stream 6 — Source Independence Score (SIS) — "Curveball Detection"
==================================================================

Defeats attack class AC-2 (Corroboration Manufacturing). The Curveball
attack succeeds by routing a single fabricated signal through multiple
apparently-independent channels — inflating the Corroboration Count
while every channel ultimately originates from the same adversarial
actor. Stream 2 (DI/CC) cannot detect this on its own because the
manufactured channels look statistically independent at the surface.

SIS computes a pairwise independence matrix ``I ∈ [0,1]^{K × K}`` over
``K`` corroborating sources, where ``Iij = 1`` indicates full
independence and ``Iij = 0`` indicates certain shared origin. Four
weighted dimensions are evaluated:

================================  ======  ==========================================
Dimension                         Weight  Indicator of Shared Origin
================================  ======  ==========================================
Institutional lineage             0.35    Same parent organization, funding body,
                                          or editorial chain.
Geographic origin                 0.25    Same physical infrastructure or
                                          geopolitical jurisdiction.
Publication timing                0.20    Publication times within Δt < τ
                                          (domain-specific threshold).
Citation network                  0.20    Cite the same primary source within
                                          ``n_hops`` hops.
================================  ======  ==========================================

The effective independence is the average of the off-diagonal matrix
entries:

    effective_independence = Σᵢ Σⱼ≠ᵢ Iij / (K · (K − 1))   ∈ [0, 1]

The effective corroboration count is anchored to the trivially-present
source itself (which is always available and need not corroborate
itself):

    CCeff = 1 + (K − 1) · effective_independence            ∈ [1, K]

The SIS score normalizes against the raw corroboration count:

    SSIS = CCeff / CCraw                                    ∈ [1/K, 1]

For the canonical Curveball case (``K`` channels, all sharing origin),
``Iij ≈ 0`` for every pair, so ``CCeff ≈ 1`` and ``SSIS ≈ 1/K``. At
``K = 4`` this reduces the effective corroboration by 75%, which is
sufficient to collapse the certified CW to UNVERIFIED in the unified
composition rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────
# Source primitives
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Source:
    """A single corroborating source.

    All fields are optional except ``source_id``. Missing dimensions
    are treated as "unknown" — the dimension's contribution to ``Iij``
    is ``0.5`` for that pair (neither full independence nor shared
    origin can be asserted).

    Attributes
    ----------
    source_id:
        Stable identifier for this source.
    institution:
        Parent organization, funding body, or editorial chain.
        Two sources with the same string are considered to share
        institutional lineage.
    geography:
        Country, region, or jurisdiction. Two sources with the same
        string share geographic origin.
    publish_time:
        Unix epoch seconds (float). Sources within ``timing_threshold``
        seconds of each other are considered to share timing.
    primary_citations:
        Set of primary-source identifiers cited by this source. Sources
        whose primary citations intersect within ``citation_overlap``
        threshold are considered to share citation network.
    """

    source_id: str
    institution: Optional[str] = None
    geography: Optional[str] = None
    publish_time: Optional[float] = None
    primary_citations: Optional[Set[str]] = None


@dataclass(frozen=True)
class SISResult:
    """Output of an SIS analysis.

    Attributes
    ----------
    score:
        ``S_SIS ∈ [1/K, 1]``. 1.0 indicates fully-independent sources;
        ``1/K`` indicates total Curveball collapse (all sources share
        origin).
    cc_eff:
        Effective corroboration count, in ``[1, K]``.
    cc_raw:
        Raw corroboration count (``K``).
    independence_matrix:
        Full ``K × K`` matrix of pairwise independence scores.
    pattern:
        Either ``"INDEPENDENT"`` (sources are genuinely independent),
        ``"PARTIAL_OVERLAP"`` (some shared dimensions, partial reduction),
        or ``"CURVEBALL"`` (sources share origin to a degree that
        collapses effective corroboration below ``curveball_threshold``).
    """

    score: float
    cc_eff: float
    cc_raw: int
    independence_matrix: np.ndarray
    pattern: str
    metrics: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────────────────────


class SISAnalyzer:
    """Source Independence Score analyzer.

    Parameters
    ----------
    w_institutional:
        Weight on institutional lineage. Default 0.35.
    w_geographic:
        Weight on geographic origin. Default 0.25.
    w_timing:
        Weight on publication timing. Default 0.20.
    w_citation:
        Weight on citation network. Default 0.20.
    timing_threshold:
        Publication-time delta in seconds below which two sources are
        considered to share timing. Default 3600 (1 hour).
    citation_overlap:
        Jaccard similarity above which two sources are considered to
        share citation network. Default 0.30.
    curveball_threshold:
        SSIS value below which the trajectory is labelled
        ``"CURVEBALL"``. Default 0.50 (i.e. effective corroboration
        is reduced by ≥50%).
    """

    def __init__(
        self,
        w_institutional: float = 0.35,
        w_geographic: float = 0.25,
        w_timing: float = 0.20,
        w_citation: float = 0.20,
        timing_threshold: float = 3600.0,
        citation_overlap: float = 0.30,
        curveball_threshold: float = 0.50,
    ):
        weight_sum = w_institutional + w_geographic + w_timing + w_citation
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Independence dimension weights must sum to 1.0, got {weight_sum}"
            )
        self.w_institutional = w_institutional
        self.w_geographic = w_geographic
        self.w_timing = w_timing
        self.w_citation = w_citation
        self.timing_threshold = timing_threshold
        self.citation_overlap = citation_overlap
        self.curveball_threshold = curveball_threshold

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def analyze(self, sources: Sequence[Source]) -> SISResult:
        """Compute the independence matrix and the SIS score."""
        K = len(sources)
        if K == 0:
            return SISResult(
                score=0.0,
                cc_eff=0.0,
                cc_raw=0,
                independence_matrix=np.zeros((0, 0)),
                pattern="EMPTY",
                metrics={"K": 0},
            )
        if K == 1:
            # Trivially independent, but no corroboration.
            return SISResult(
                score=1.0,
                cc_eff=1.0,
                cc_raw=1,
                independence_matrix=np.array([[1.0]]),
                pattern="INDEPENDENT",
                metrics={"K": 1},
            )

        I = self._independence_matrix(sources)

        # Average off-diagonal entries
        off_diag_sum = float(I.sum() - np.trace(I))
        denom = K * (K - 1)
        effective_independence = off_diag_sum / denom

        cc_eff = 1.0 + (K - 1) * effective_independence
        ssis = cc_eff / K

        if ssis >= 1.0 - 1e-6:
            pattern = "INDEPENDENT"
        elif ssis < self.curveball_threshold:
            pattern = "CURVEBALL"
        else:
            pattern = "PARTIAL_OVERLAP"

        return SISResult(
            score=float(ssis),
            cc_eff=float(cc_eff),
            cc_raw=K,
            independence_matrix=I,
            pattern=pattern,
            metrics={
                "K": K,
                "effective_independence": effective_independence,
                "off_diag_mean": effective_independence,
            },
        )

    def score(self, sources: Sequence[Source]) -> float:
        """Convenience wrapper that returns just the SIS score."""
        return self.analyze(sources).score

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _independence_matrix(self, sources: Sequence[Source]) -> np.ndarray:
        K = len(sources)
        I = np.ones((K, K), dtype=float)
        for i in range(K):
            for j in range(K):
                if i == j:
                    I[i, j] = 1.0
                    continue
                I[i, j] = self._pairwise_independence(sources[i], sources[j])
        return I

    def _pairwise_independence(self, a: Source, b: Source) -> float:
        """Compute Iij for a pair of sources."""
        contributions: List[Tuple[float, float]] = []  # (weight, value)

        # Institutional lineage
        if a.institution is not None and b.institution is not None:
            value = 0.0 if a.institution == b.institution else 1.0
        else:
            value = 0.5
        contributions.append((self.w_institutional, value))

        # Geographic origin
        if a.geography is not None and b.geography is not None:
            value = 0.0 if a.geography == b.geography else 1.0
        else:
            value = 0.5
        contributions.append((self.w_geographic, value))

        # Publication timing
        if a.publish_time is not None and b.publish_time is not None:
            dt = abs(a.publish_time - b.publish_time)
            value = 0.0 if dt < self.timing_threshold else 1.0
        else:
            value = 0.5
        contributions.append((self.w_timing, value))

        # Citation network
        if a.primary_citations is not None and b.primary_citations is not None:
            value = self._citation_independence(
                a.primary_citations, b.primary_citations
            )
        else:
            value = 0.5
        contributions.append((self.w_citation, value))

        return sum(w * v for w, v in contributions)

    def _citation_independence(self, a: Set[str], b: Set[str]) -> float:
        """Jaccard-based citation independence."""
        if not a and not b:
            return 0.5
        if not a or not b:
            return 1.0
        intersection = len(a & b)
        union = len(a | b)
        jaccard = intersection / union if union else 0.0
        return 0.0 if jaccard >= self.citation_overlap else 1.0
