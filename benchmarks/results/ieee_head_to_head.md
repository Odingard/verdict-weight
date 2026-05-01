# IEEE Head-to-Head Benchmark — VW vs Classical Fusion

**Harness:** `ieee_head_to_head` (Paper 3 §8.2 Table 4 reproducer)

**N = 2,000**, **seed = 42**, NumPy 2.4.4, Python 3.12.8

**VW throughput:** 3,812.9 samples/sec (524 ms total)


## Comparative metrics

| Method | Mean Adv CW | Mean Legit CW | Brier | REL | Cohen's d | Welch p | AUC |
|---|---|---|---|---|---|---|---|
| VERDICT_WEIGHT | 0.0905 | 0.7158 | 0.0532 | 0.0527 | -6.6998 | 0.00e+00 | 1.0000 |
| DEMPSTER_SHAFER | 0.9974 | 0.9987 | 0.4974 | 0.2480 | -0.5576 | 0.00e+00 | 0.6311 |
| NAIVE_BAYES | 0.9763 | 0.9963 | 0.4779 | 0.2420 | -0.5551 | 0.00e+00 | 0.6352 |
| SIMPLE_AVERAGING | 0.7634 | 0.8065 | 0.3143 | 0.0928 | -0.6647 | 0.00e+00 | 0.6289 |
| MAX_VOTING | 0.9355 | 1.0000 | 0.4472 | 0.2250 | -0.6577 | 0.00e+00 | 0.6000 |

## Notes

- Adversarial samples cover AC-1 through AC-5; legitimate samples are stable-high trajectories with independent sources and valid provenance / registry.

- VW evaluated under government tier (γ=δ=1).

- Baselines see only commercial-tier evidence (SR/CC/TD/HA); they do not have access to trajectory / source-independence / provenance / registry signals by construction (these dimensions do not exist in classical fusion).

