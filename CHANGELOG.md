# VERDICT WEIGHT™ Changelog

## v1.2.0 — April 2026

### Eight-stream architecture (Paper 3)

- Implemented Streams 5–8 (adversarial + government tier):
  - **Stream 5 — CTC** (Cross-Temporal Consistency): trajectory pattern
    detection (A=stable-high / B=stable-low / C=spike-collapse /
    D=gradual-buildup). Pattern C (AC-3 Trajectory Fabrication) zeroes
    `S_CTC` by construction.
  - **Stream 6 — SIS** (Source Independence Score): independence-matrix
    fusion across institution / geography / temporal proximity / shared
    citations. K shared-origin sources collapse `S_SIS` to ~1/K
    (AC-2 Curveball detection).
  - **Stream 7 — CPS** (Cryptographic Provenance Score): SHA-256 hash
    chain verification, genesis anchoring, first-failure short-circuit.
    Tampered chains HALT (AC-4 Provenance Forgery).
  - **Stream 8 — RIS** (Registry Integrity Score): registry-hash binary
    gate, version-monotonicity check, deterministic serialization.
    Tampered registries HALT (AC-5 Registry Compromise).

- Added `UnifiedComposer` — eight-stream composition rule
  `RIS → CPS → SIS → CTC → Commercial`, with HALT propagation and
  deployment-tier coefficients (commercial γ=δ=0.5, adversarial γ=1.0,
  government γ=δ=1.0).

### Honest replication harness (Path A)

- Wrote validation harness from scratch against Paper 3's specification.
  No implementation parameter was tuned to reproduce specific paper
  numbers — every reported figure is the actual output of the harness.
- `validation/datasets.py` — N=10,000 deterministic synthetic generator
  with cumulative-monotone timestamps, 5 attack classes + legitimate.
- `validation/synthetic_validation.py` — full pipeline reproducer.
- `benchmarks/ieee_head_to_head.py` — VW vs Dempster-Shafer / Naive
  Bayes / Simple Averaging / Max Voting on N=2,000.
- `benchmarks/cve_validation.py` — CISA KEV reproducer (snapshot
  `catalogVersion=2026.04.30`, 1,586 vulnerabilities). N=120 sample.
- Test suite: 172 tests across unit (139), integration (12), property
  (13), regression (6), performance (2).
- Fixed `validation.metrics.auc_roc` mid-rank averaging on tied scores
  (Mann–Whitney U). Pre-fix returned biased AUC for tied score vectors
  (e.g. AUC=0.0 for fully-tied input). Post-fix returns the standard
  ScikitLearn-equivalent AUC. Affects MAX_VOTING AUC only — VW and
  continuous-score baselines were unaffected. Regression test added at
  `tests/unit/test_metrics.py::TestAUCTiedRanks`.
- Pinned CVE Temporal Decay (TD) reference time to the snapshot's
  `dateReleased` field (`2026-04-30T16:30:36Z`). Pre-fix the harness
  used `datetime.now()`, making TD wall-clock dependent and
  contradicting VALIDATION.md's determinism claim. Post-fix the harness
  produces the same numbers in 2026, 2027, and any future year. The
  reference time is recorded under `snapshot.reference_time_iso` in
  the output JSON for audit. Mean CW shifted by 3×10⁻⁵ — no change at
  4-decimal precision; 0/120 false suppressions and 0 HALTs unchanged.
- Added `is not None` guard on throughput formatting in the CVE
  markdown renderer (cosmetic consistency with `ieee_head_to_head.py`).

### Replicated metrics (this build, seed=42)

- Synthetic N=10,000: VW Brier=0.0536, REL=0.0531, AUC=1.0,
  Cohen's d=−6.66, sensitivity=0.917, specificity=1.000.
- IEEE head-to-head (N=2,000): VW dominates DS/NB/SA/MV across Brier,
  REL, AUC, Cohen's d.
- CISA KEV (N=120): **0/120 false suppressions, 0 HALTs**, mean
  CW=0.6895 (95% CI [0.6835, 0.6966]).

### Documentation

- Added `VALIDATION.md` mapping every Paper 3 claim to its harness
  reproducer command and actual output, with explicit divergence
  documentation. This document is the canonical source of truth — Paper
  3 will be revised in the next preprint update to match harness output.

## v1.0.1 — April 2026

### Validation Updates
- Expanded validation dataset from N=1,000 to N=10,000
- Added bootstrap confidence intervals (2,000 resamples) to all results
- Added McNemar's test — all baselines p<0.001 ***
- Added 5-fold cross-validation stability analysis
- Added two-stream baseline for completeness
- Added cross-vertical validation (7 verticals, 12+ profiles)
- Updated adversarial suppression: 49.1% → 57.8% (N=10,000 result)
- Dataset SHA-256 published for reproducibility verification
- Added failure mode documentation (defense and RAG verticals)

### New Files
- `validation/synthetic_validation.py` — updated N=10,000 engine
- `validation/ablation_study.py` — 324-config weight ablation

### No Algorithm Changes
Core algorithm (SS, DI, CW equations) unchanged from v1.0.0.
Weight profiles unchanged. All changes are validation methodology only.

## v1.0.0 — April 2026
Initial release.
- 12 context-adaptive weight profiles
- Four evidence streams: SR, CC, TD, HA
- Three outputs: Signal Strength, Doubt Index, Consequence Weight
- USPTO trademark filed: Serial #99747827
- Published: SSRN Abstract #6532658
- DOI: 10.5281/zenodo.19447547
