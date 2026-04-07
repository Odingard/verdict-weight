"""
VERDICT WEIGHT™
===============
Context-Adaptive Multi-Source Confidence Synthesis Framework
for Autonomous AI Intelligence Systems.

© 2026 Odingard Security / Six Sense Enterprise Services LLC
VERDICT WEIGHT™ is a trademark of Six Sense Enterprise Services LLC.
USPTO Serial Number: 99747827.

SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6532658
"""

from .core import VerdictWeight, VerdictResult
from .types import ContextType, WeightProfile

__version__ = "1.0.0"
__author__ = "Andre Byrd"
__email__ = "andre.byrd@odingard.com"
__trademark__ = "VERDICT WEIGHT is a trademark of Six Sense Enterprise Services LLC. USPTO 99747827."

__all__ = ["VerdictWeight", "VerdictResult", "ContextType", "WeightProfile"]
