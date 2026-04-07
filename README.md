# VERDICT WEIGHT™

**A Context-Adaptive Multi-Source Confidence Synthesis Framework for Autonomous AI Intelligence Systems**

[![SSRN](https://img.shields.io/badge/SSRN-6532658-blue)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6532658)
[![USPTO](https://img.shields.io/badge/USPTO-99747827-green)](https://tmsearch.uspto.gov)
[![PyPI](https://img.shields.io/pypi/v/verdict-weight)](https://pypi.org/project/verdict-weight/)
[![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)

> *"Calibrated multi-source confidence scoring is not an optional feature of autonomous AI systems — it is a foundational architectural requirement."*
> — VERDICT WEIGHT™ Paper, SSRN 6532658

---

## The Problem

Autonomous AI systems ingest intelligence from multiple heterogeneous sources and treat them as roughly equal. A rumor on a threat forum and a Mandiant primary incident report receive the same weight. A 30-day-old signal and a real-time alert are scored identically. A spoofed high-credibility source sails straight through.

That is not a UI problem. That is a systematic architectural vulnerability.

## The Solution

VERDICT WEIGHT™ computes three complementary confidence outputs from four evidence streams:

### Four Evidence Streams
| Stream | Symbol | Description |
|--------|--------|-------------|
| Source Reliability | SR | Credibility of the originating source |
| Cross-Feed Corroboration | CC | Independent confirmation across feeds |
| Temporal Decay | TD | Recency of the intelligence signal |
| Historical Source Accuracy | HA | Empirical track record of the source |

### Three Output Components
| Output | Symbol | Description |
|--------|--------|-------------|
| Signal Strength | SS | Confidence the signal is real (0.0–1.0) |
| Doubt Index | DI | Inter-stream disagreement (0.0–1.0) |
| Consequence Weight | CW | Actionability score after doubt adjustment (0.0–1.0) |

### Twelve Context Profiles
VERDICT WEIGHT™ ships with optimized weight profiles for twelve operational domains:

| Domain | Profile |
|--------|---------|
| Cybersecurity (General) | Corroboration-dominant |
| Cybersecurity (APT) | Source reliability elevated |
| Cybersecurity (Zero-Day) | Temporal decay dominant |
| Cybersecurity (Disinformation) | Maximum corroboration, maximum doubt penalty |
| Healthcare (Diagnostic) | High doubt penalty, surfaces uncertainty |
| Healthcare (Drug Safety) | Highest doubt penalty in registry |
| Financial (Fraud) | Corroboration + recency co-dominant |
| Financial (Market) | Temporal decay dominant, fast decay |
| Defense Intelligence | Multi-source fusion, slow decay |
| Autonomous Vehicle | Sub-second decay, highest doubt penalty |
| Legal Evidence | Minimal decay, chain of custody weighted |
| Enterprise RAG | LLM retrieval confidence scoring |

---

## Validated Results

Synthetic validation across N=1,000 controlled scenarios:

| Metric | VERDICT WEIGHT™ | Equal Weight | Single Source |
|--------|----------------|--------------|---------------|
| Brier Score ↓ | **0.2077** | 0.2126 | 0.2561 |
| AUC-ROC | 0.7391 | 0.7398 | 0.6370 |
| AUC-PR | **0.7377** | 0.7374 | 0.6203 |
| Adv. Suppression ↓ | **0.409** | 0.560 | 0.803 |

**+18.9% Brier Score improvement** and **49.1% adversarial intelligence suppression** vs single-source baselines.

---

## Quick Start

```python
from verdict_weight import VerdictWeight, ContextType

vw = VerdictWeight()

# Score a cybersecurity threat intel signal
result = vw.score(
    source_reliability=0.92,
    n_corroborating_sources=3,
    age_value=2.5,
    correct_predictions=45,
    total_predictions=50,
    context=ContextType.CYBERSECURITY_APT
)

print(result.action_tier)          # CRITICAL
print(result.consequence_weight)   # 0.8547
print(result.doubt_index)          # 0.0691
print(result.interpretation)       # Act immediately...
print(result.to_json())            # Full JSON output
```

---

## Installation

```bash
pip install verdict-weight
```

Or from source:

```bash
git clone https://github.com/Odingard/verdict-weight.git
cd verdict-weight
pip install -e .
```

---

## Repository Structure

```
verdict-weight/
├── verdict_weight/
│   ├── __init__.py          # Public API
│   ├── core.py              # VerdictWeight engine
│   ├── profiles.py          # 12 context weight profiles
│   ├── streams.py           # Stream scoring functions
│   └── types.py             # Data types and enums
├── validation/
│   ├── synthetic_validation.py   # N=1,000 validation engine
│   ├── ablation_study.py         # 324-config weight ablation
│   └── results/                  # Validation outputs
├── examples/
│   ├── cybersecurity.py
│   ├── healthcare.py
│   ├── financial.py
│   └── defense.py
├── docs/
│   └── VERDICT_WEIGHT_Paper.pdf  # SSRN 6532658
├── setup.py
├── requirements.txt
└── README.md
```

---

## Citation

If you use VERDICT WEIGHT™ in your research, please cite:

```bibtex
@misc{byrd2026verdictweight,
  title={VERDICT WEIGHT: A Context-Adaptive Multi-Source Confidence Synthesis 
         Framework for Autonomous AI Intelligence Systems},
  author={Byrd, Andre},
  year={2026},
  howpublished={SSRN},
  note={Abstract ID: 6532658},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6532658}
}
```

---

## Legal

VERDICT WEIGHT™ is a trademark of Six Sense Enterprise Services LLC (Odingard Security).
USPTO Serial Number: 99747827.

© 2026 Odingard Security / Six Sense Enterprise Services LLC. All rights reserved.

For licensing inquiries: andre.byrd@odingard.com
