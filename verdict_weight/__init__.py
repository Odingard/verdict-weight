"""
VERDICT WEIGHT™
===============
Context-Adaptive Multi-Source Confidence Synthesis Framework
for Autonomous AI Intelligence Systems.

© 2026 Six Sense Enterprise Services LLC (Odingard Security)
VERDICT WEIGHT™ is a trademark of Six Sense Enterprise Services LLC.
USPTO Serial Number: 99747827.

Quick start:
    from verdict_weight import VerdictWeight, ContextType

    vw = VerdictWeight()
    result = vw.score(
        source_reliability=0.92,
        n_corroborating_sources=3,
        age_value=2.5,
        correct_predictions=45,
        total_predictions=50,
        context=ContextType.CYBERSECURITY_APT
    )
    print(result.action_tier)           # CRITICAL
    print(result.consequence_weight)    # 0.8547
    print(result.to_json())             # Full JSON output

Paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6532658
Code:  https://github.com/Odingard/verdict-weight
DOI:   https://doi.org/10.5281/zenodo.19447547
"""

from .core import (
    VerdictWeight,
    VerdictResult,
    ContextType,
    WeightProfile,
    StreamScorer,
    VerdictEngine,
    AdaptiveLearner,
    WEIGHT_PROFILES,
)

__version__ = "1.0.0"
__author__ = "Andre Byrd"
__email__ = "andre.byrd@odingard.com"
__url__ = "https://github.com/Odingard/verdict-weight"
__doi__ = "10.5281/zenodo.19447547"
__ssrn__ = "6532658"
__trademark__ = (
    "VERDICT WEIGHT is a trademark of Six Sense Enterprise Services LLC. "
    "USPTO Serial Number 99747827."
)

__all__ = [
    "VerdictWeight",
    "VerdictResult",
    "ContextType",
    "WeightProfile",
    "StreamScorer",
    "VerdictEngine",
    "AdaptiveLearner",
    "WEIGHT_PROFILES",
]
