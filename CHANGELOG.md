# VERDICT WEIGHT™ Changelog

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
