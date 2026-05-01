# CVE Validation Benchmark — VW on CISA KEV

**Harness:** `cve_validation` (Paper 3 §8.4 Table 6 reproducer)

**N = 120**, **seed = 42**, suppression threshold = 0.3, government tier

**Snapshot:** CISA KEV catalog `2026.04.30` (1,586 vulnerabilities), released 2026-04-30T16:30:36.316Z

**VW 1.2.0**, NumPy 2.4.4, Python 3.12.8


## VERDICT WEIGHT outcomes

- **Mean CW:** 0.6895 (95% CI: [0.6835, 0.6967])
- **Median CW:** 0.6863
- **CW range:** [0.6432, 0.7932]
- **False suppressions (0.3 threshold):** 0/120 (0.00%)
- **HALT events:** 0 {}
- **Throughput:** 5,968.4 CVEs/sec


## Baseline comparison

| Method | Mean CW | Median CW | False-suppression rate |
|---|---|---|---|
| VERDICT_WEIGHT | 0.6895 | 0.6863 | 0.00% |
| DEMPSTER_SHAFER | 0.9997 | 0.9997 | 0.00% |
| NAIVE_BAYES | 0.9991 | 0.9992 | 0.00% |
| SIMPLE_AVERAGING | 0.8188 | 0.8027 | 0.00% |
| MAX_VOTING | 0.8875 | 1.0000 | 0.00% |

## Notes

- Each CISA KEV entry is by construction a confirmed real-world exploited vulnerability. The benchmark therefore measures **false-suppression rate**: how often the harness incorrectly down-weights a known-exploited CVE.

- Inputs are derived from public KEV metadata only (date added, vendor, product, ransomware-campaign flag). The mapping is deterministic and documented in ``_cve_to_inputs``; running with the same seed against the same snapshot reproduces the same numbers exactly.

- AUC is not reported because all 120 records are positive class (real CVEs); the interesting metric is suppression rate, not separability.

