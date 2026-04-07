# VERDICT WEIGHT™

**A Context-Adaptive Multi-Source Confidence Synthesis Framework for Autonomous AI Intelligence Systems**

[![SSRN](https://img.shields.io/badge/SSRN-6532658-blue)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6532658)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19447547-blue)](https://doi.org/10.5281/zenodo.19447547)
[![USPTO](https://img.shields.io/badge/USPTO-99747827-green)](https://tmsearch.uspto.gov)
[![PyPI](https://img.shields.io/badge/PyPI-verdict--weight-purple)](https://pypi.org/project/verdict-weight/)
[![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)

> *"Calibrated multi-source confidence scoring is not an optional feature of autonomous AI systems — it is a foundational architectural requirement."*

---

## The Problem

Autonomous AI systems treat all intelligence sources as equal.
A rumor on a threat forum and a Mandiant primary incident report receive the same weight.
A 30-day-old signal and a real-time alert are scored identically.
A spoofed high-credibility source sails straight through.

That is not a UI problem. That is a systematic architectural vulnerability.

---

## Validated Results

Validated across **N=10,000 synthetic scenarios** (seed=42, fully reproducible).
Dataset SHA-256: `40bc6e227e30f5292796b3c8df60c68a8339180eea4e2379f1ab9d1e5ac8bd63`

| Method | Brier ↓ | 95% CI | AUC-ROC | McNemar |
|--------|---------|--------|---------|---------|
| **VERDICT WEIGHT™** | **0.2079** | **[0.2036, 0.2122]** | **0.7499** | — |
| Equal Weight | 0.2170 | [0.2130, 0.2210] | 0.7450 | p<0.001 *** |
| Single Source | 0.2499 | [0.2447, 0.2553] | 0.6537 | p<0.001 *** |
| Two Stream | 0.2298 | [0.2251, 0.2346] | 0.7258 | p<0.001 *** |

**Key results:**
- **57.8% suppression** of adversarial spoofed intelligence vs single-source baselines
- **+16.8% Brier Score improvement** vs single source (p<0.001 ***)
- **+14.7% AUC-ROC improvement** vs single source (p<0.001 ***)
- **5-fold CV stability:** Brier 0.2079 ± 0.0046 — results do not overfit
- All significance tests: McNemar p=0.000000 against all baselines

### Cross-Vertical Performance (N=1,000 per vertical)

| Vertical | Brier Δ% | AUC Δ% | Adv. Suppression |
|----------|---------|--------|-----------------|
| Cybersecurity | +17.7% | +17.0% | 57.4% |
| Healthcare | +9.6% | +20.9% | — |
| Financial | +10.3% | +15.1% | — |
| Manufacturing | +9.2% | +8.5% | 40.5% |
| Legal | +4.4% | +3.8% | 30.7% |
| Defense | -2.5% | +0.3% | 16.6% |
| Enterprise RAG | -6.7% | -1.1% | 25.7% |

*Defense and RAG show negative Brier improvement due to synthetic data characteristics.
See audit report for full failure mode analysis.*

---

## What VERDICT WEIGHT™ Does

Four evidence streams → Three outputs → One decision.

### Four Evidence Streams
| Stream | Symbol | Description |
|--------|--------|-------------|
| Source Reliability | SR | Credibility of the originating source (0.01–0.99) |
| Cross-Feed Corroboration | CC | Independent confirmation across feeds |
| Temporal Decay | TD | Recency of the intelligence signal |
| Historical Source Accuracy | HA | Empirical track record of the source |

### Three Output Components
| Output | Symbol | Range | Description |
|--------|--------|-------|-------------|
| Signal Strength | SS | 0–1 | Confidence the signal is real |
| Doubt Index | DI | 0–1 | Inter-stream disagreement |
| Consequence Weight | CW | 0–1 | Actionability after doubt adjustment |

### Twelve Context Profiles

| Domain | Profile Type |
|--------|-------------|
| Cybersecurity (General) | Corroboration-dominant |
| Cybersecurity (APT) | Source reliability elevated, slow decay |
| Cybersecurity (Zero-Day) | Temporal decay dominant |
| Cybersecurity (Disinformation) | Maximum corroboration + doubt penalty |
| Healthcare (Diagnostic) | High doubt penalty, surfaces uncertainty |
| Healthcare (Drug Safety) | Highest doubt penalty in registry |
| Financial (Fraud) | Corroboration + recency co-dominant |
| Financial (Market) | Temporal decay dominant, fast decay |
| Defense Intelligence | Multi-source fusion, slow strategic decay |
| Autonomous Vehicle | Sub-second decay, highest doubt penalty |
| Legal Evidence | Minimal decay, chain of custody weighted |
| Enterprise RAG | LLM retrieval confidence scoring |

---

## Quick Start

```bash
pip install verdict-weight
```

```python
from verdict_weight import VerdictWeight, ContextType

vw = VerdictWeight()

# Score a cybersecurity threat intelligence signal
result = vw.score(
    source_reliability=0.92,        # How credible is this source?
    n_corroborating_sources=3,       # How many independent sources confirm?
    age_value=2.5,                   # How old is this intelligence (days)?
    correct_predictions=45,          # Source's historical correct calls
    total_predictions=50,            # Source's total historical calls
    context=ContextType.CYBERSECURITY_APT
)

print(result.action_tier)           # CRITICAL
print(result.consequence_weight)    # 0.8380
print(result.doubt_index)           # 0.0691
print(result.interpretation)        # "Act immediately. High-confidence..."
print(result.to_json())             # Full JSON output
```

### Adversarial signal — watch the suppression
```python
# High-credibility source but zero corroboration — spoofed intel
result = vw.score(
    source_reliability=0.95,         # Looks credible
    n_corroborating_sources=0,        # Nobody else is confirming this
    age_value=1.0,
    context=ContextType.CYBERSECURITY_DISINFO
)

print(result.action_tier)           # NOISE
print(result.consequence_weight)    # 0.147 — suppressed by 84%
```

### Score pre-computed streams directly
```python
result = vw.score_streams(
    SR=0.92, CC=0.78, TD=0.94, HA=0.88,
    context=ContextType.FINANCIAL_FRAUD
)
```

---

## Repository Structure

```
verdict-weight/
├── verdict_weight/
│   ├── __init__.py          # Public API
│   └── core.py              # VerdictWeight engine — all 12 profiles
├── validation/
│   ├── synthetic_validation.py   # N=10,000 validation (seed=42)
│   └── ablation_study.py         # 324-config weight ablation
├── examples/
│   ├── cybersecurity.py
│   └── healthcare.py
├── docs/
│   └── VERDICT_WEIGHT_Paper.pdf  # SSRN 6532658
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Mathematical Foundation

**Signal Strength (weighted geometric mean):**
```
SS = ∏(S_i + ε)^w_i   where Σw_i = 1.0
```

**Doubt Index (normalized coefficient of variation):**
```
DI = clip(σ(SR,CC,TD,HA) / μ(SR,CC,TD,HA), 0, 1)
```

**Consequence Weight:**
```
CW = clip(SS × (1 - δ × DI), 0, 1)
```

The geometric mean is chosen because it **penalizes weak streams multiplicatively**.
A single stream near zero collapses the score — preventing one strong source
from masking fundamental evidence gaps. This is the structural guarantee
behind the 57.8% adversarial suppression result.

---

## Citation

```bibtex
@misc{byrd2026verdictweight,
  title={VERDICT WEIGHT: A Context-Adaptive Multi-Source Confidence Synthesis
         Framework for Autonomous AI Intelligence Systems},
  author={Byrd, Andre},
  year={2026},
  howpublished={SSRN Preprint},
  note={SSRN Abstract ID: 6532658},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6532658},
  doi={10.5281/zenodo.19447547}
}
```

---

## Reproducibility

Results are fully reproducible:
1. Clone this repository
2. Run `python validation/synthetic_validation.py`
3. Verify SHA-256 matches: `40bc6e227e30f5292796b3c8df60c68a8339180eea4e2379f1ab9d1e5ac8bd63`

Master seed: **42** (never changes — all results are deterministic)

---

## Legal

VERDICT WEIGHT™ is a trademark of Six Sense Enterprise Services LLC (Odingard Security).
USPTO Serial Number: 99747827.

© 2026 Six Sense Enterprise Services LLC. All rights reserved.

This software is made available for research and evaluation purposes.
Commercial deployment requires a license agreement.

**For licensing:** andre.byrd@odingard.com
**Paper:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6532658
**DOI:** https://doi.org/10.5281/zenodo.19447547
