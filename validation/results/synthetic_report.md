# Synthetic Validation Report — VERDICT WEIGHT

- **Generated:** 2026-05-01T05:37:19Z
- **Package version:** 1.2.0
- **Python:** 3.12.8 on Linux 5.15.200 (x86_64)
- **NumPy:** 2.4.4

- **N samples:** 10,000
- **Seed:** 42
- **Legitimate fraction:** 0.50
- **Adversarial threshold:** 0.30
- **Deployment tier:** government

## Wall time
- Data generation: 1.6374 s
- VW scoring: 2.3840 s (throughput = 4,195 samples/sec)

## HALT counts (RIS / CPS absorbing states)
- RIS: 1000
- CPS: 1000
- none: 8000

## Head-to-head comparison (adversarial threshold = 0.30)

| Method | Adv mean CW | Legit mean CW | Brier ↓ | REL ↓ | AUC ↑ | Cohen's d | Welch p |
|---|---|---|---|---|---|---|---|
| VERDICT_WEIGHT | 0.0905 | 0.7146 | 0.0536 | 0.0531 | 1.0000 | -6.6555 | 0 |
| Dempster-Shafer | 0.9974 | 0.9986 | 0.4974 | 0.2480 | 0.6285 | -0.5426 | 0 |
| Naive_Bayes | 0.9776 | 0.9963 | 0.4790 | 0.2417 | 0.6346 | -0.5498 | 0 |
| Simple_Averaging | 0.7628 | 0.8057 | 0.3139 | 0.0926 | 0.6316 | -0.6713 | 0 |
| Max_Voting | 0.9327 | 1.0000 | 0.4455 | 0.2232 | 0.9980 | -0.6576 | 0 |

## Per-attack-class breakdown (VERDICT WEIGHT)

| Class | N | Mean CW | Median CW | Std CW | Min | Max |
|---|---|---|---|---|---|---|
| legitimate | 5000 | 0.7146 | 0.7131 | 0.0551 | 0.5354 | 0.8795 |
| AC-1 | 1000 | 0.2857 | 0.2869 | 0.0496 | 0.1801 | 0.3941 |
| AC-2 | 1000 | 0.1663 | 0.1601 | 0.0445 | 0.1025 | 0.2630 |
| AC-3 | 1000 | 0.0007 | 0.0000 | 0.0050 | 0.0000 | 0.0644 |
| AC-4 | 1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| AC-5 | 1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

---

**Reproducibility:** every number above can be reproduced from a clean
checkout with::

    pip install -e .
    python -m validation.synthetic_validation --n 10000 --seed 42

