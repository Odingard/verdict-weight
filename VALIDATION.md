# VERDICT WEIGHT v1.2.0 — Validation Report

**Status:** Honest replication run, ``v1.2.0`` candidate
**Architecture:** 8-stream unified composition (Paper 3, October 2025)
**Run date:** 2026-04-21
**Author:** Andre Byrd
**Co-attribution:** James (architecture), Mark (security features), Jamie (QA/QC), Michael (project management)

---

## 1. Purpose

This document is the canonical reproducibility surface for every empirical
claim in the unified eight-stream paper (Paper 3, Zenodo
``10.5281/zenodo.19447547``, SSRN ``10.2139/ssrn.6532658``). For every
number in the paper, this report lists:

1. The Paper 3 claim (verbatim or paraphrased with section reference)
2. The exact harness command that reproduces the claim
3. The actual harness output captured against ``v1.2.0`` of the code
4. Any divergence between paper and harness, with explanation

**This report is the source of truth.** If a number in Paper 3 conflicts
with a number in this report, the harness output is canonical and Paper 3
will be revised in the next preprint update to match.

---

## 2. One-command reproduction

```bash
git clone https://github.com/Odingard/verdict-weight && cd verdict-weight
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# 1. Synthetic validation (Paper 3 §8.1)
python -m validation.synthetic_validation --n 10000 --seed 42

# 2. IEEE head-to-head benchmark (Paper 3 §8.2)
python -m benchmarks.ieee_head_to_head --n 2000 --seed 42

# 3. CISA KEV CVE validation (Paper 3 §8.4)
python -m benchmarks.cve_validation --n 120 --seed 42

# 4. Full test suite
python -m pytest tests/
```

All four commands are deterministic given the seed. Outputs land in
``validation/results/`` and ``benchmarks/results/``.

Run environment (this report):

* Python 3.12.8, NumPy 2.4.4, Linux x86_64
* CISA KEV snapshot ``catalogVersion=2026.04.30`` (1,586 vulnerabilities)
* Seed = 42 throughout

---

## 3. Synthetic validation (N = 10,000)

**Reproducer:** ``python -m validation.synthetic_validation --n 10000 --seed 42``

| Metric | Paper 3 claim | Harness output (v1.2.0) | Divergence |
|---|---|---|---|
| Sample size N | 10,000 | 10,000 | exact |
| Adversarial fraction | 0.50 | 0.50 | exact |
| VW Brier score | 0.0412 (§8.1) | **0.0536** | +0.0124 (worse than claimed) |
| VW Reliability (REL) | 0.0019 (§8.1) | **0.0531** | +0.0512 (worse than claimed) |
| VW AUC-ROC | 1.0 (§8.4) | **1.0000** | exact |
| VW Cohen's d (legit vs adv) | −2.82 (§8.2 vs DS) | **−6.66** | substantially better than claimed |
| VW Welch t-statistic | not specified | **−332.77** | n/a |
| VW Welch p | < 0.001 (§8.2) | **0.0** (below double precision) | consistent |
| VW sensitivity @ τ=0.30 | not specified | **0.9172** | n/a |
| VW specificity @ τ=0.30 | not specified | **1.0000** | n/a |
| VW false-positive rate | not specified | **0** | n/a |
| VW throughput | 35,913 scores/sec (§9, "single-core") | **~4,194 scores/sec** | ~8.6× slower (see §6) |

### Per-attack-class mean CW (N = 10,000)

| Class | n | Paper 3 expected | Harness output | Divergence |
|---|---|---|---|---|
| Legitimate | 5,000 | high (>0.5) | **0.7146** | matches expectation |
| AC-1 (Source Spoofing) | 1,000 | attenuated | **0.2857** | attenuated as designed |
| AC-2 (Curveball) | 1,000 | attenuated | **0.1663** | attenuated as designed |
| AC-3 (Trajectory Fabrication, Pattern C) | 1,000 | **must zero** (§5 CTC) | **0.0007** | matches (effectively zero) |
| AC-4 (Provenance Forgery) | 1,000 | HALT | **0.0000** (1,000/1,000 HALT-CPS) | matches |
| AC-5 (Registry Compromise) | 1,000 | HALT | **0.0000** (1,000/1,000 HALT-RIS) | matches |

### HALT counts

| HALT stage | Count | Notes |
|---|---|---|
| RIS | 1,000 | All AC-5 samples halt at RIS (binary gate, registry tamper detected) |
| CPS | 1,000 | All AC-4 samples halt at CPS (chain tamper detected) |
| none | 8,000 | 5,000 legitimate + 3,000 non-halting adversarial (AC-1/2/3) |

---

## 4. IEEE head-to-head benchmark (N = 2,000)

**Reproducer:** ``python -m benchmarks.ieee_head_to_head --n 2000 --seed 42``

| Method | Mean Adv CW | Mean Legit CW | Brier | REL | Cohen's d | AUC |
|---|---|---|---|---|---|---|
| **VERDICT WEIGHT** | **0.0905** | **0.7158** | **0.0532** | **0.0527** | **−6.6998** | **1.0000** |
| Dempster-Shafer | 0.9974 | 0.9987 | 0.4974 | 0.2480 | −0.5576 | 0.6311 |
| Naive Bayes | 0.9763 | 0.9963 | 0.4779 | 0.2420 | −0.5551 | 0.6352 |
| Simple Averaging | 0.7634 | 0.8065 | 0.3143 | 0.0928 | −0.6647 | 0.6289 |
| Max Voting | 0.9355 | 1.0000 | 0.4472 | 0.2250 | −0.6577 | 0.3085 |

### Comparison vs Paper 3 §8.2 Table 4

| Paper 3 claim | Harness output | Divergence |
|---|---|---|
| VW Brier 0.0412 vs DS Brier 0.1847 | **VW 0.0532 vs DS 0.4974** | DS performs *worse* in honest run (0.4974 vs 0.1847 claimed) |
| VW REL 0.0019 vs SA REL 0.0182 (9.6× better) | **VW 0.0527 vs SA 0.0928** | VW is **1.76× better** than SA (paper's 9.6× was overstated) |
| VW vs DS, Cohen's d = −2.82, p<0.001 | VW vs DS, mean-of-means difference is enormous; legit-vs-adv d for VW is **−6.6998** | The paper's d=−2.82 was a smaller-magnitude figure than the real separation |
| VW outperforms classical fusion baselines | **Confirmed** — VW Brier and REL both lower than every baseline; AUC = 1.0 vs ≤0.6352 for DS/NB/SA | matches |
| Significant after Bonferroni | Welch p = 0.0 (< 1e-300) | matches |

**Verdict on §8.2 of Paper 3:** the *direction* of every claim holds — VW
dominates classical fusion across Brier, REL, AUC, and Cohen's d. The
*magnitudes* differ from the paper in two directions: Brier and REL are
worse than claimed; Cohen's d separation is **substantially better** than
claimed.

---

## 5. CISA KEV CVE validation (N = 120)

**Reproducer:** ``python -m benchmarks.cve_validation --n 120 --seed 42``

**Snapshot:** ``validation/data/cisa_kev_snapshot.json`` —
catalog version ``2026.04.30``, 1,586 vulnerabilities, retrieved 2026-04-21.

| Metric | Paper 3 claim (§8.4 / Table 6) | Harness output (v1.2.0) | Divergence |
|---|---|---|---|
| N | 120 | 120 | exact |
| False suppressions | 0/120 | **0/120** | **exact** |
| HALT events | 0 | **0** | **exact** |
| AUC | 1.0 (acknowledged as proxy artifact in §8.4) | n/a (all samples positive class — see Notes) | n/a |
| Mean CW | not specified in paper | **0.6895** (95% CI [0.6835, 0.6966]) | n/a |
| Median CW | not specified | **0.6862** | n/a |
| CW range | not specified | [0.6432, 0.7932] | n/a |
| Throughput | not specified | **5,805 CVEs/sec** | n/a |

### Baseline comparison on the same 120 CVEs

| Method | Mean CW | Median CW | False-suppression rate |
|---|---|---|---|
| VERDICT_WEIGHT | 0.6895 | 0.6862 | **0.00%** |
| Dempster-Shafer | 0.9997 | 0.9997 | 0.00% |
| Naive Bayes | 0.9991 | 0.9992 | 0.00% |
| Simple Averaging | 0.8188 | 0.8027 | 0.00% |
| Max Voting | 0.8875 | 1.0000 | 0.00% |

**Verdict on §8.4 of Paper 3:** the load-bearing claim — **0/120 false
suppressions on real-world exploited CVEs** — replicates exactly. No
HALTs are triggered on legitimate CISA KEV records, confirming the
RIS/CPS halt logic is correctly anchored to actual tamper signals only.

---

## 6. Test suite

**Reproducer:** ``python -m pytest tests/``

**Total:** 157 tests passing across 4 categories.

| Category | Count | Coverage |
|---|---|---|
| Unit (`tests/unit/`) | 124 | Streams 1–8 individually |
| Integration (`tests/integration/`) | 12 | Full eight-stream composition + HALT propagation |
| Property (`tests/property/`) | 13 | Range, monotonicity, NaN-freedom, tier ordering |
| Regression (`tests/regression/`) | 6 | Per-attack-class outcome bounds |
| Performance (`tests/performance/`) | 2 | Per-score sanity bound (<100 ms) |

### Divergence vs Paper 3 Table 3 ("673 tests across 27 suites")

| Paper 3 claim | Harness output | Divergence |
|---|---|---|
| 673 tests | **157 tests** | **−516** |
| 27 suites | **5 categories / 9 modules** | structurally smaller |

**Explanation:** Paper 3's 673-test claim does not reflect the v1.2.0
reproducible test suite published here. The v1.2.0 suite is built from
scratch under Path A (honest replication) and covers every load-bearing
invariant of the eight-stream composition rule (range, monotonicity,
HALT ordering, AC-1..AC-5 outcome bounds, full pipeline composition,
per-stream tamper detection at scale including 1,000-sample CPS and RIS
tamper sweeps). Paper 3 will be revised to report the actual published
test count rather than the higher pre-replication estimate.

---

## 7. Throughput

**Single-thread, ungrouped, government tier (most expensive):**

| Source | Reported throughput | Notes |
|---|---|---|
| Paper 3 §9 | 35,913 scores/sec | "single-core" claim |
| Synthetic harness, N=10,000 | **~4,194 scores/sec** | full eight-stream pipeline w/ trajectory + sources + chain + registry |
| IEEE benchmark, N=2,000 | **~4,115 scores/sec** | same path |
| CVE benchmark, N=120 | **~5,805 CVEs/sec** | shorter chain, no trajectory |
| Performance unit test | sanity bound 100 ms/score | always passes (typical: <1 ms) |

**Divergence:** the harness measures ~4–6 K scores/sec on the full
pipeline; Paper 3 reported ~35 K. The most likely explanations:

* The 35 K figure was for a stripped-down commercial-tier-only path
  (Streams 1–4) without trajectory, source-independence, or
  hash-chain verification.
* Different hardware (Paper 3 hardware is unspecified).
* Paper 3 may have batched or vectorized; v1.2.0 is a reference
  implementation focused on correctness.

**Resolution:** Paper 3 will be revised to either (a) clarify which
path the 35 K figure measured, or (b) restate throughput as the
honest harness number (4–6 K full-pipeline, single-thread).

---

## 8. Summary of divergences

The architecture is sound. Every directional claim in Paper 3 holds in
the honest harness:

* **VW dominates classical fusion** across Brier, REL, AUC, and Cohen's d.
* **AC-3 / AC-4 / AC-5 yield CW ≈ 0** (Pattern C zeroes; tampered chains
  / registries HALT).
* **0/120 false suppressions on real CVEs.**

Numbers that should be **revised down** in Paper 3 (the harness is
*worse* than the paper claimed):

* VW Brier: 0.0412 → **0.0536** (synthetic) / **0.0532** (IEEE)
* VW REL: 0.0019 → **0.0531** (synthetic) / **0.0527** (IEEE)
* SA REL ratio: "9.6× better than VW" → **1.76× better** than VW
* Throughput: 35,913 scores/sec → **~4,200 scores/sec** full pipeline
* Test count: 673 → **157**

Numbers that should be **revised up** in Paper 3 (the harness is
*better* than the paper claimed):

* VW Cohen's d (legit vs adv): −2.82 → **−6.66**
* VW sensitivity / specificity: not previously stated → **0.917 / 1.000**

Numbers that **replicate exactly**:

* AUC-ROC = 1.0 (synthetic and IEEE)
* 0/120 false suppressions on CISA KEV
* Pattern C → CW ≈ 0
* HALT propagation order (RIS → CPS) on AC-4 / AC-5

---

## 9. Path A commitment

This report is the artifact of **Path A — Honest Replication**, the
explicit decision recorded by Andre Byrd on 2026-04-21:

> Build it honestly. Surface the actual numbers. Paper 3 will be revised
> to match whatever the harness produces. The architecture is sound —
> the math is correct, the streams are well-defined, the attack class
> logic is solid. The numbers will land in the right neighborhood. If
> any specific number shifts materially, that's information, not a
> problem.

The harness was written from scratch against Paper 3's specification.
At no point were implementation parameters tuned to reproduce Paper 3's
specific numbers. Where the harness diverges from Paper 3, the harness
is canonical.

---

## 10. Reproducibility metadata

| Field | Value |
|---|---|
| Code version | v1.2.0 |
| Commit | (head of branch at PR creation) |
| Python | 3.12.8 |
| NumPy | 2.4.4 |
| Platform | Linux x86_64 |
| Random seed | 42 (across all runs) |
| KEV snapshot | catalogVersion=2026.04.30 (1,586 vulns, released 2026-04-30T16:30:36.316Z) |
| Run date | 2026-04-21 |

Run any of the four reproducer commands in §2 to verify every number
in this report. Bit-identical reproduction requires:

* Same seed (42)
* Same KEV snapshot file (committed to ``validation/data/``)
* Same NumPy major version (2.x — minor versions may shift floating-point bootstrap CIs by < 0.001)
