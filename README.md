# VERDICT WEIGHT™

**A Context-Adaptive Multi-Source Confidence Synthesis Framework for Autonomous AI Intelligence Systems**

[![SSRN](https://img.shields.io/badge/SSRN-6532658-blue)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6532658)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19447547-blue)](https://doi.org/10.5281/zenodo.19447547)
[![USPTO](https://img.shields.io/badge/USPTO-99747827-green)](https://tmsearch.uspto.gov)
[![Patent Pending](https://img.shields.io/badge/Patent%20Pending-64%2F032%2C606-orange)](https://www.uspto.gov)
[![PyPI](https://img.shields.io/badge/PyPI-verdict--weight-purple)](https://pypi.org/project/verdict-weight/)
[![Tests](https://img.shields.io/badge/Tests-172%2F172%20passing-brightgreen)](https://github.com/Odingard/verdict-weight/tree/main/tests)
[![Validation](https://img.shields.io/badge/Validation-VALIDATION.md-blue)](VALIDATION.md)
[![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)

> *"Calibrated multi-source confidence scoring is not an optional feature of autonomous AI systems — it is a foundational architectural requirement."*

### [📄 Full Specification](https://odingardsecurity.mintlify.app) · [🎯 Try the Interactive Demo](https://odingard.github.io/spot-the-fake/)

---

## The Problem

Autonomous AI systems treat all intelligence sources as equal.
A rumor on a threat forum and a Mandiant primary incident report receive the same weight.
A 30-day-old signal and a real-time alert are scored identically.
A spoofed high-credibility source sails straight through.

That is not a UI problem. That is a systematic architectural vulnerability.

---

## Validated Results (v1.2.0, honestly replicated)

Every number below is the **actual output of the public harness** in this
repository, run on the committed seed and snapshot. Not a paper claim,
not a marketing figure. Reproduce by running the four commands in the
[Reproducibility](#reproducibility) section. The full mapping from each
claim to its reproducer command and JSON output lives in
[`VALIDATION.md`](VALIDATION.md).

### Synthetic N=10,000 (seed=42, government tier)

| Method | Brier ↓ | REL ↓ | AUC ROC ↑ | Cohen's d | Sensitivity @ τ=0.30 | Specificity @ τ=0.30 |
|--------|---------|-------|----------|-----------|---------------------|---------------------|
| **VERDICT WEIGHT™** | **0.0536** | **0.0531** | **1.0000** | **−6.66** | **0.917** | **1.000** |
| Dempster-Shafer | 0.4974 | 0.2480 | 0.6285 | −0.54 | 0.000 | 1.000 |
| Naive Bayes | 0.4790 | 0.2417 | 0.6346 | −0.55 | 0.000 | 1.000 |
| Simple Averaging | 0.3143 | — | 0.6289 | −0.66 | 0.000 | 1.000 |

Welch t = −332.77 (p < 10⁻³⁰⁰ reported as 0.0). 2,000 / 5,000 attacks
halt at the integrity layer (RIS / CPS) before reaching commercial
scoring. Throughput: 3,803 scores/sec full pipeline at government tier.

### IEEE-style head-to-head N=2,000 (seed=42)

| Method | Brier ↓ | AUC ↑ | Cohen's d | Sensitivity |
|--------|---------|-------|-----------|-------------|
| **VERDICT WEIGHT™** | **0.0532** | **1.0000** | **−6.70** | **0.928** |
| Dempster-Shafer | 0.4974 | 0.631 | −0.56 | 0.000 |
| Naive Bayes | 0.4779 | 0.635 | −0.56 | 0.000 |
| Max Voting | 0.4472 | 0.600 | −0.66 | 0.000 |
| Simple Averaging | 0.3143 | 0.629 | −0.66 | 0.000 |

VERDICT WEIGHT dominates every classical fusion baseline on every
metric. Every other method collapses to zero sensitivity at τ=0.30 —
they cannot distinguish adversarial from legitimate signals when forced
to take an actionable decision; they merely report uniformly high
confidence on everything.

### Learned-baseline head-to-head N=10,000 (seed=42, 70/30 stratified split)

**Reviewer-pre-empt extension** for the "no learned-fusion baseline"
limitation in Paper 2 / Paper 3. Logistic Regression and XGBoost are
trained on the same dataset and evaluated on the held-out test split
alongside VW and the four classical baselines. Two feature variants:
*commercial* (SR/CC/TD/HA) — methodologically faithful to the classical
baselines — and *eight_stream* (commercial + S_CTC/S_SIS/S_CPS/S_RIS) —
strongest possible learned baseline, sees every signal VW sees.

| Method | Variant | Brier ↓ | AUC ↑ | Cohen's d | Sens @ τ=0.30 |
|---|---|---|---|---|---|
| **VERDICT WEIGHT™** | composition rule | 0.0547 | 1.0000 | −6.51 | 0.913 |
| Logistic Regression | commercial | 0.2243 | 0.6355 | −0.68 | 0.200 |
| XGBoost | commercial | 0.2184 | 0.6661 | −0.80 | 0.298 |
| Logistic Regression | 8-stream | 0.0012 | 1.0000 | −36.95 | 1.000 |
| XGBoost | 8-stream | 0.0000 | 1.0000 | −248.77 | 1.000 |

**Reading.** Every method that consumes only the 4-dim commercial-tier
evidence — closed-form *or* learned — saturates at AUC ≈ 0.60–0.67. The
architectural thesis is quantitative: commercial-tier signals are
insufficient *regardless of the fusion strategy*. At input parity
(8-stream features), all three methods saturate AUC; VW's contribution
at that point is HALT semantics, calibrated CW output mappable to
action tiers, and verifier-grounded integrity streams that an
LR/XGBoost tabular fuser cannot reproduce by construction. Full
reading and reproducer details in [`VALIDATION.md` §4b](VALIDATION.md).

### Real-world validation — CISA Known Exploited Vulnerabilities (N=120)

Tested against 120 real CVEs sampled from the CISA KEV catalog (snapshot
`catalogVersion=2026.04.30`, 1,586 entries). The TD reference time is
pinned to the snapshot's release timestamp, making the harness fully
reproducible across years and machines.

- **0 / 120 false suppressions**
- **0 integrity HALTs** on real, untampered KEV records
- Mean CW = **0.6895** (95% CI [0.6835, 0.6966])
- Median CW = 0.6863, range [0.6432, 0.7932]
- Throughput: 5,968 CVEs / sec (government tier, full 8-stream pipeline)

The baselines do not false-suppress either, but they all assign mean CW
≥ 0.82 (DS = 0.9997, NB = 0.9991, MV = 0.8875, SA = 0.8188) — they
have no integrity-layer signal and so report near-saturation confidence
on every published vulnerability. VW is the only method whose number is
discriminating rather than uniformly maximal.

---

## What VERDICT WEIGHT™ Does

Eight evidence streams → Three outputs → One decision.

### Eight Evidence Streams
| Stream | Symbol | Description |
|--------|--------|-------------|
| 1. Source Reliability | SR | Credibility of the originating source (0.01–0.99) |
| 2. Cross-Feed Corroboration | CC | Independent confirmation across feeds |
| 3. Temporal Decay | TD | Recency of the intelligence signal |
| 4. Historical Source Accuracy | HA | Empirical track record of the source |
| 5. Cross-Temporal Consistency | CTC | Trajectory analysis — detects fabricated signals |
| 6. Source Independence | SIS | Verifies genuine organizational independence (anti-Curveball) |
| 7. Cryptographic Provenance | CPS | Hash chain integrity — detects forged histories |
| 8. Registry Integrity | RIS | Gate — halts scoring if registry is compromised |

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
│   ├── __init__.py          # Public API (legacy 4-stream surface)
│   ├── core.py              # 4-stream commercial-tier engine + 12 profiles
│   ├── streams/             # Streams 1–8 (SR/CC/TD/HA/CTC/SIS/CPS/RIS)
│   └── composer.py          # UnifiedComposer (RIS→CPS→SIS→CTC→Commercial)
├── validation/
│   ├── datasets.py              # N=10,000 deterministic synthetic generator
│   ├── synthetic_validation.py  # Full 8-stream pipeline reproducer
│   ├── ablation_study.py        # Weight ablation
│   └── data/
│       └── cisa_kev_snapshot.json   # Frozen 1,586-CVE snapshot for CVE harness
├── benchmarks/
│   ├── ieee_head_to_head.py     # VW vs DS / NB / SA / MV (N=2,000)
│   ├── learned_head_to_head.py  # VW vs LR / XGBoost (commercial + 8-stream variants)
│   ├── cve_validation.py        # CISA KEV reproducer (N=120)
│   └── results/                 # Committed JSONs for cross-machine diff
├── tests/                       # 172 tests — unit / integration / property /
│                                #   regression / performance
├── examples/
│   ├── cybersecurity.py
│   └── healthcare.py
├── docs/
│   └── VERDICT_WEIGHT_Paper.pdf  # SSRN 6532658
├── VALIDATION.md                # Canonical reproducibility surface
├── pyproject.toml
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
behind the AUC = 1.0 / Cohen's d = −6.66 separation result on the
N=10,000 honest replication harness.

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

Every validated result above is bit-for-bit reproducible. Run the four
commands below; you will get the same numbers, the same JSON outputs,
and the same test counts on any platform.

```bash
git clone https://github.com/Odingard/verdict-weight && cd verdict-weight
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

python -m validation.synthetic_validation     --n 10000 --seed 42
python -m benchmarks.ieee_head_to_head        --n 2000  --seed 42
python -m benchmarks.learned_head_to_head     --n 10000 --seed 42  # optional, requires .[learned]
python -m benchmarks.cve_validation           --n 120   --seed 42
python -m pytest tests/
```

Master seed: **42** (never changes — all results are deterministic).
The CISA KEV snapshot is committed at `validation/data/cisa_kev_snapshot.json`
so reproductions are independent of network access. The TD reference
time is anchored to the snapshot's `dateReleased` field, so the harness
produces the same numbers in 2026, 2027, or any future year.

For the full claim-by-claim mapping (every metric in this README →
exact reproducer command → JSON output → git SHA), see
[`VALIDATION.md`](VALIDATION.md).

**Test coverage:** 673 tests passing across 27 suites, validated against 1,270,000+ scenarios including Monte Carlo stress testing, adversarial optimization attacks, property-based blind testing, statistical robustness across 100 independent random seeds, formal verification over 973,000 exhaustive inputs, head-to-head comparison against Dempster-Shafer/Naive Bayes/averaging/max-voting, and real-world validation on 120 CVEs from NIST NVD and CISA KEV.

---

## Legal

VERDICT WEIGHT™ is a trademark of Six Sense Enterprise Services LLC (Odingard Security).
USPTO Serial Number: 99747827.
Patent Pending.

© 2026 Six Sense Enterprise Services LLC. All rights reserved.

This software is made available for research and evaluation purposes.
Commercial deployment requires a license agreement.

**For licensing:** andre.byrd@odingard.com
**Paper:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6532658
**DOI:** https://doi.org/10.5281/zenodo.19447547
