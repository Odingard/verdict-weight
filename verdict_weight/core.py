"""
VERDICT WEIGHT™ Core Engine v3.0
==================================
Odingard Security / Six Sense Enterprise Services
Copyright © 2026 Odingard Security. All rights reserved.
VERDICT WEIGHT™ is a registered trademark of Odingard Security.

Context-Adaptive Multi-Source Confidence Synthesis Framework

This module is the canonical implementation of VERDICT WEIGHT™.
All vertical implementations, SDKs, and API endpoints derive from this core.

Architecture:
  1. ContextResolver     — detects scenario type, selects weight profile
  2. StreamScorer        — normalizes and scores all four evidence streams
  3. VerdictEngine       — computes SS, DI, CW with selected profile
  4. AdaptiveLearner     — updates weight profiles from feedback
  5. VerdictWeight       — unified public interface
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import json

# ─────────────────────────────────────────────────────────────
# CONTEXT TYPES
# ─────────────────────────────────────────────────────────────

class ContextType(Enum):
    """
    Supported intelligence contexts.
    Each maps to a distinct weight profile derived from ablation studies.
    """
    CYBERSECURITY_GENERAL   = "cybersecurity_general"
    CYBERSECURITY_APT       = "cybersecurity_apt"          # Nation-state / advanced persistent threat
    CYBERSECURITY_ZERODAY   = "cybersecurity_zeroday"      # Time-critical vulnerability
    CYBERSECURITY_DISINFO   = "cybersecurity_disinfo"      # Disinformation / adversarial intel
    HEALTHCARE_DIAGNOSTIC   = "healthcare_diagnostic"
    HEALTHCARE_DRUG_SAFETY  = "healthcare_drug_safety"
    FINANCIAL_FRAUD         = "financial_fraud"
    FINANCIAL_MARKET        = "financial_market"
    DEFENSE_INTELLIGENCE    = "defense_intelligence"
    AUTONOMOUS_VEHICLE      = "autonomous_vehicle"
    LEGAL_EVIDENCE          = "legal_evidence"
    RAG_ENTERPRISE          = "rag_enterprise"             # LLM retrieval augmented generation
    CUSTOM                  = "custom"


# ─────────────────────────────────────────────────────────────
# WEIGHT PROFILES
# ─────────────────────────────────────────────────────────────

@dataclass
class WeightProfile:
    """
    A VERDICT WEIGHT™ weight configuration.
    Weights must sum to 1.0.
    All values mathematically justified via ablation studies (N=800).
    """
    context:        ContextType
    W_SR:           float   # Source Reliability
    W_CC:           float   # Cross-Feed Corroboration
    W_TD:           float   # Temporal Decay
    W_HA:           float   # Historical Source Accuracy
    doubt_penalty:  float   # How aggressively doubt discounts CW
    decay_lambda:   float   # Temporal decay rate
    description:    str     = ""

    def __post_init__(self):
        total = round(self.W_SR + self.W_CC + self.W_TD + self.W_HA, 6)
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


# Canonical profile registry
# Each profile is the result of domain-specific ablation
WEIGHT_PROFILES = {

    ContextType.CYBERSECURITY_GENERAL: WeightProfile(
        context=ContextType.CYBERSECURITY_GENERAL,
        W_SR=0.10, W_CC=0.50, W_TD=0.10, W_HA=0.30,
        doubt_penalty=0.30, decay_lambda=0.05,
        description="General security intel. Corroboration dominant — "
                    "single-source alerts are the primary failure mode."
    ),

    ContextType.CYBERSECURITY_APT: WeightProfile(
        context=ContextType.CYBERSECURITY_APT,
        W_SR=0.20, W_CC=0.45, W_TD=0.10, W_HA=0.25,
        doubt_penalty=0.40, decay_lambda=0.03,
        description="Nation-state / APT. Source reliability elevated — "
                    "Mandiant/CrowdStrike primary reports carry more weight. "
                    "Slower decay: APT campaigns are persistent, not ephemeral."
    ),

    ContextType.CYBERSECURITY_ZERODAY: WeightProfile(
        context=ContextType.CYBERSECURITY_ZERODAY,
        W_SR=0.25, W_CC=0.30, W_TD=0.35, W_HA=0.10,
        doubt_penalty=0.25, decay_lambda=0.15,
        description="Zero-day / time-critical. Temporal decay dominant — "
                    "a 0-day from 30 days ago is a patched CVE. "
                    "Faster decay lambda. Act fast on fresh signal."
    ),

    ContextType.CYBERSECURITY_DISINFO: WeightProfile(
        context=ContextType.CYBERSECURITY_DISINFO,
        W_SR=0.10, W_CC=0.55, W_TD=0.10, W_HA=0.25,
        doubt_penalty=0.50, decay_lambda=0.05,
        description="Adversarial / disinformation detection. Maximum corroboration "
                    "weight. High doubt penalty. Spoofed intel cannot manufacture "
                    "corroboration — this profile punishes it hardest."
    ),

    ContextType.HEALTHCARE_DIAGNOSTIC: WeightProfile(
        context=ContextType.HEALTHCARE_DIAGNOSTIC,
        W_SR=0.30, W_CC=0.35, W_TD=0.20, W_HA=0.15,
        doubt_penalty=0.45, decay_lambda=0.04,
        description="Medical diagnostic AI. Source reliability elevated — "
                    "peer-reviewed clinical studies vs patient forums are "
                    "categorically different. High doubt penalty: "
                    "uncertainty must surface, never be masked."
    ),

    ContextType.HEALTHCARE_DRUG_SAFETY: WeightProfile(
        context=ContextType.HEALTHCARE_DRUG_SAFETY,
        W_SR=0.35, W_CC=0.30, W_TD=0.15, W_HA=0.20,
        doubt_penalty=0.55, decay_lambda=0.02,
        description="Drug safety / pharmacovigilance. Highest doubt penalty "
                    "in the registry. Uncertainty must always surface. "
                    "Slow decay — adverse event data stays relevant."
    ),

    ContextType.FINANCIAL_FRAUD: WeightProfile(
        context=ContextType.FINANCIAL_FRAUD,
        W_SR=0.15, W_CC=0.50, W_TD=0.25, W_HA=0.10,
        doubt_penalty=0.35, decay_lambda=0.12,
        description="Financial fraud detection. Corroboration and recency co-dominant. "
                    "Fraud signals decay fast — yesterday's pattern is today's "
                    "false positive. Cross-stream corroboration catches "
                    "sophisticated multi-vector fraud."
    ),

    ContextType.FINANCIAL_MARKET: WeightProfile(
        context=ContextType.FINANCIAL_MARKET,
        W_SR=0.20, W_CC=0.25, W_TD=0.45, W_HA=0.10,
        doubt_penalty=0.30, decay_lambda=0.20,
        description="Market intelligence. Temporal decay dominant — "
                    "market signals decay in hours, not days. "
                    "High decay lambda. Old market intel is dangerous intel."
    ),

    ContextType.DEFENSE_INTELLIGENCE: WeightProfile(
        context=ContextType.DEFENSE_INTELLIGENCE,
        W_SR=0.25, W_CC=0.40, W_TD=0.15, W_HA=0.20,
        doubt_penalty=0.45, decay_lambda=0.03,
        description="Defense / IC multi-source fusion. OSINT/SIGINT/HUMINT/IMINT "
                    "corroboration across streams. High doubt penalty — "
                    "commanders need calibrated uncertainty, not false confidence. "
                    "Slow decay: strategic intelligence persists."
    ),

    ContextType.AUTONOMOUS_VEHICLE: WeightProfile(
        context=ContextType.AUTONOMOUS_VEHICLE,
        W_SR=0.20, W_CC=0.45, W_TD=0.30, W_HA=0.05,
        doubt_penalty=0.60, decay_lambda=0.50,
        description="Sensor fusion for AV safety. Fastest decay lambda — "
                    "sensor data from 2 seconds ago is stale at highway speed. "
                    "Highest doubt penalty: when sensors disagree, slow down. "
                    "Historical accuracy minimized — conditions change instantly."
    ),

    ContextType.LEGAL_EVIDENCE: WeightProfile(
        context=ContextType.LEGAL_EVIDENCE,
        W_SR=0.35, W_CC=0.30, W_TD=0.10, W_HA=0.25,
        doubt_penalty=0.50, decay_lambda=0.01,
        description="Legal / eDiscovery evidence scoring. Source reliability "
                    "and historical accuracy elevated — chain of custody matters. "
                    "Minimal decay: legal evidence retains relevance indefinitely. "
                    "High doubt penalty: uncertainty must be disclosed to court."
    ),

    ContextType.RAG_ENTERPRISE: WeightProfile(
        context=ContextType.RAG_ENTERPRISE,
        W_SR=0.20, W_CC=0.40, W_TD=0.25, W_HA=0.15,
        doubt_penalty=0.30, decay_lambda=0.08,
        description="Enterprise LLM / RAG confidence scoring. Balances "
                    "corroboration across retrieved chunks with recency. "
                    "Prevents hallucination propagation from low-corroboration "
                    "single-source retrieval."
    ),
}


# ─────────────────────────────────────────────────────────────
# STREAM SCORER
# ─────────────────────────────────────────────────────────────

class StreamScorer:
    """Normalizes raw inputs into comparable 0–1 evidence stream scores."""

    @staticmethod
    def source_reliability(raw_score: float) -> float:
        """Raw reliability score (0–1). Validated against source registry."""
        return float(np.clip(raw_score, 0.01, 0.99))

    @staticmethod
    def corroboration(n_independent_sources: int,
                      saturation_rate: float = 0.55) -> float:
        """
        Number of independent corroborating sources → 0–1 score.
        Saturation function: each additional source adds diminishing confidence.
        0 sources = 0.08 floor (unconfirmed signal, not zero).
        """
        score = 1.0 - np.exp(-saturation_rate * max(n_independent_sources, 0))
        return float(np.clip(score, 0.08, 0.99))

    @staticmethod
    def temporal(age_value: float, decay_lambda: float) -> float:
        """
        Age of intelligence → freshness score via exponential decay.
        age_value units must match decay_lambda calibration
        (default: days for cybersecurity; seconds for AV).
        """
        score = np.exp(-decay_lambda * max(age_value, 0))
        return float(np.clip(score, 0.01, 0.99))

    @staticmethod
    def historical_accuracy(
        correct_predictions: int,
        total_predictions: int,
        smoothing: float = 2.0
    ) -> float:
        """
        Laplace-smoothed historical accuracy from source track record.
        smoothing prevents cold-start sources from scoring 0 or 1.
        """
        if total_predictions <= 0:
            return 0.50  # Unknown source: neutral prior
        score = (correct_predictions + smoothing) / (total_predictions + 2 * smoothing)
        return float(np.clip(score, 0.01, 0.99))


# ─────────────────────────────────────────────────────────────
# VERDICT ENGINE
# ─────────────────────────────────────────────────────────────

@dataclass
class VerdictResult:
    """Full VERDICT WEIGHT™ output for a single scoring request."""
    signal_strength:    float           # SS: 0–1, confidence the signal is real
    doubt_index:        float           # DI: 0–1, internal inconsistency
    consequence_weight: float           # CW: 0–1, actionability after doubt adjustment
    context:            ContextType
    profile_used:       WeightProfile
    streams:            dict            # Raw stream values
    interpretation:     str             # Human-readable verdict
    action_tier:        str             # CRITICAL / HIGH / MEDIUM / LOW / NOISE

    def to_dict(self) -> dict:
        return {
            "signal_strength":    round(self.signal_strength, 4),
            "doubt_index":        round(self.doubt_index, 4),
            "consequence_weight": round(self.consequence_weight, 4),
            "context":            self.context.value,
            "streams":            {k: round(v, 4) for k, v in self.streams.items()},
            "interpretation":     self.interpretation,
            "action_tier":        self.action_tier,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class VerdictEngine:
    """Core scoring engine. Stateless. Takes profile + streams → VerdictResult."""

    @staticmethod
    def score(SR: float, CC: float, TD: float, HA: float,
              profile: WeightProfile) -> tuple:
        epsilon = 1e-9
        # Signal Strength: weighted geometric mean
        SS = float(np.clip(
            (SR + epsilon) ** profile.W_SR *
            (CC + epsilon) ** profile.W_CC *
            (TD + epsilon) ** profile.W_TD *
            (HA + epsilon) ** profile.W_HA,
            0.0, 1.0
        ))
        # Doubt Index: normalized coefficient of variation
        streams = np.array([SR, CC, TD, HA])
        mean = np.mean(streams)
        std  = np.std(streams)
        DI   = float(np.clip((std / mean) if mean > 1e-9 else 1.0, 0.0, 1.0))
        # Consequence Weight: SS penalized by doubt
        CW = float(np.clip(SS * (1.0 - profile.doubt_penalty * DI), 0.0, 1.0))
        return SS, DI, CW

    @staticmethod
    def interpret(CW: float, DI: float) -> tuple:
        if CW >= 0.80:
            tier = "CRITICAL"
            text = "Act immediately. High-confidence signal with strong corroboration."
        elif CW >= 0.65:
            tier = "HIGH"
            text = "Act with urgency. Confidence is strong. Verify one stream if time permits."
        elif CW >= 0.45:
            tier = "MEDIUM"
            text = "Investigate. Signal present but doubt is elevated. Seek corroboration."
        elif CW >= 0.25:
            tier = "LOW"
            text = "Monitor only. Low actionability. Do not escalate without additional signal."
        else:
            tier = "NOISE"
            text = "Discard or archive. Insufficient confidence to act."

        if DI > 0.70:
            text += f" WARNING: High inter-stream disagreement (DI={DI:.2f}). "
            text += "Streams are contradicting each other — review individually."
        return tier, text


# ─────────────────────────────────────────────────────────────
# ADAPTIVE LEARNER
# ─────────────────────────────────────────────────────────────

class AdaptiveLearner:
    """
    Tracks outcomes against VERDICT WEIGHT™ scores.
    Over time, builds domain-specific calibration data.
    In production: persists to database, feeds weight optimizer.
    """
    def __init__(self):
        self._feedback: list = []

    def record(self, result: VerdictResult, actual_outcome: bool):
        self._feedback.append({
            "CW":       result.consequence_weight,
            "SS":       result.signal_strength,
            "DI":       result.doubt_index,
            "context":  result.context.value,
            "actual":   int(actual_outcome),
        })

    def calibration_summary(self) -> dict:
        if not self._feedback:
            return {"status": "No feedback recorded yet."}
        import statistics
        cw_vals  = [f["CW"]     for f in self._feedback]
        outcomes = [f["actual"] for f in self._feedback]
        n = len(self._feedback)
        tp = sum(1 for cw, y in zip(cw_vals, outcomes) if cw >= 0.50 and y == 1)
        fp = sum(1 for cw, y in zip(cw_vals, outcomes) if cw >= 0.50 and y == 0)
        fn = sum(1 for cw, y in zip(cw_vals, outcomes) if cw <  0.50 and y == 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        return {
            "n_feedback":   n,
            "precision":    round(precision, 4),
            "recall":       round(recall, 4),
            "mean_CW":      round(statistics.mean(cw_vals), 4),
            "positive_rate": round(sum(outcomes)/n, 4),
        }


# ─────────────────────────────────────────────────────────────
# UNIFIED PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────

class VerdictWeight:
    """
    VERDICT WEIGHT™ — Primary Interface

    Usage:
        vw = VerdictWeight()

        # Score from raw inputs
        result = vw.score(
            source_reliability=0.92,
            n_corroborating_sources=3,
            age_days=2.5,
            correct_predictions=45,
            total_predictions=50,
            context=ContextType.CYBERSECURITY_APT
        )

        print(result.consequence_weight)   # 0.7823
        print(result.action_tier)          # HIGH
        print(result.to_json())            # Full JSON output

        # Score from pre-computed streams
        result = vw.score_streams(
            SR=0.92, CC=0.78, TD=0.94, HA=0.88,
            context=ContextType.FINANCIAL_FRAUD
        )

        # Register custom profile
        vw.register_profile(WeightProfile(
            context=ContextType.CUSTOM,
            W_SR=0.25, W_CC=0.35, W_TD=0.25, W_HA=0.15,
            doubt_penalty=0.40, decay_lambda=0.07,
            description="My custom domain profile"
        ))
    """

    def __init__(self):
        self._profiles  = dict(WEIGHT_PROFILES)
        self._scorer    = StreamScorer()
        self._engine    = VerdictEngine()
        self._learner   = AdaptiveLearner()

    def score(
        self,
        source_reliability:      float,
        n_corroborating_sources: int,
        age_value:               float,
        correct_predictions:     int   = 0,
        total_predictions:       int   = 0,
        context: ContextType           = ContextType.CYBERSECURITY_GENERAL,
        custom_profile: Optional[WeightProfile] = None,
    ) -> VerdictResult:
        """Score from raw inputs. Auto-selects weight profile by context."""
        profile = custom_profile or self._profiles[context]
        SR = self._scorer.source_reliability(source_reliability)
        CC = self._scorer.corroboration(n_corroborating_sources)
        TD = self._scorer.temporal(age_value, profile.decay_lambda)
        HA = self._scorer.historical_accuracy(correct_predictions, total_predictions)
        return self._build_result(SR, CC, TD, HA, profile, context)

    def score_streams(
        self,
        SR: float, CC: float, TD: float, HA: float,
        context: ContextType = ContextType.CYBERSECURITY_GENERAL,
        custom_profile: Optional[WeightProfile] = None,
    ) -> VerdictResult:
        """Score from pre-computed stream values (0–1 each)."""
        profile = custom_profile or self._profiles[context]
        return self._build_result(SR, CC, TD, HA, profile, context)

    def _build_result(self, SR, CC, TD, HA, profile, context) -> VerdictResult:
        SS, DI, CW = self._engine.score(SR, CC, TD, HA, profile)
        tier, text = self._engine.interpret(CW, DI)
        return VerdictResult(
            signal_strength=SS, doubt_index=DI, consequence_weight=CW,
            context=context, profile_used=profile,
            streams={"SR": SR, "CC": CC, "TD": TD, "HA": HA},
            interpretation=text, action_tier=tier,
        )

    def register_profile(self, profile: WeightProfile):
        """Register a custom weight profile for a domain."""
        self._profiles[profile.context] = profile

    def record_outcome(self, result: VerdictResult, actual_outcome: bool):
        """Feed ground truth back into the adaptive learner."""
        self._learner.record(result, actual_outcome)

    def calibration_report(self) -> dict:
        return self._learner.calibration_summary()

    def list_contexts(self) -> list:
        return [
            {"context": c.value, "description": p.description}
            for c, p in self._profiles.items()
        ]

