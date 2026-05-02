"""
VERDICT WEIGHT — Stream Modules

Streams 1-4 (commercial tier) live in core.py for backward compatibility:
  - Stream 1: Source Reliability (SR)
  - Stream 2: Corroboration / Doubt Index (CC / DI)
  - Stream 3: Temporal Decay (TD)
  - Stream 4: Historical Accuracy (HA)

Streams 5-8 (adversarial + government tiers) are defined here:
  - Stream 5: Cross-Temporal Consistency (CTC)       — adversarial trajectory detection
  - Stream 6: Source Independence Score (SIS)         — Curveball / corroboration manufacturing detection
  - Stream 7: Cryptographic Provenance Score (CPS)    — SHA-256 hash chain integrity
  - Stream 8: Registry Integrity Score (RIS)          — binary kill switch on registry tamper

All four government / adversarial streams are pure-Python with no third-party
runtime dependencies (numpy is used only in CTC for vector ops).
"""

from .ctc import (
    CTCAnalyzer,
    TrajectoryPoint,
    TrajectoryPattern,
    CTCResult,
)
from .sis import (
    SISAnalyzer,
    Source,
    SISResult,
)
from .cps import (
    CPSVerifier,
    ProvenanceStep,
    CPSResult,
    build_provenance_chain,
)
from .ris import (
    RISVerifier,
    SourceRegistry,
    RISResult,
)

__all__ = [
    # Stream 5 (CTC)
    "CTCAnalyzer",
    "TrajectoryPoint",
    "TrajectoryPattern",
    "CTCResult",
    # Stream 6 (SIS)
    "SISAnalyzer",
    "Source",
    "SISResult",
    # Stream 7 (CPS)
    "CPSVerifier",
    "ProvenanceStep",
    "CPSResult",
    "build_provenance_chain",
    # Stream 8 (RIS)
    "RISVerifier",
    "SourceRegistry",
    "RISResult",
]
