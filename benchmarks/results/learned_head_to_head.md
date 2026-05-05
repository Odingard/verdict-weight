# Learned-Baseline Head-to-Head — VW vs LR / XGBoost vs Classical Fusion

**Harness:** `learned_head_to_head` (reviewer-pre-empt extension of Paper 3 §8.2 Table 4)

**N = 10,000** (train = 7,000 / test = 3,000, `test_fraction = 0.30`)

**seed = 42**, **threshold = 0.3**, NumPy 2.4.4, Python 3.12.8


## Comparative metrics (all methods, test split)

| Method | Variant | Mean Adv CW | Mean Legit CW | Brier | REL | Cohen's d | Welch p | AUC | Sens | Spec |
|---|---|---|---|---|---|---|---|---|---|---|
| VERDICT_WEIGHT | 8-stream | 0.0923 | 0.7122 | 0.0547 | 0.0543 | -6.5144 | 0.00e+00 | 1.0000 | 0.9127 | 1.0000 |
| DEMPSTER_SHAFER | commercial | 0.9974 | 0.9986 | 0.4974 | 0.2480 | -0.5360 | 0.00e+00 | 0.6260 | 0.0000 | 1.0000 |
| NAIVE_BAYES | commercial | 0.9785 | 0.9962 | 0.4798 | 0.2420 | -0.5519 | 0.00e+00 | 0.6317 | 0.0000 | 1.0000 |
| SIMPLE_AVERAGING | commercial | 0.7634 | 0.8050 | 0.3145 | 0.0928 | -0.6545 | 0.00e+00 | 0.6252 | 0.0000 | 1.0000 |
| MAX_VOTING | commercial | 0.9347 | 1.0000 | 0.4467 | 0.2244 | -0.6575 | 0.00e+00 | 0.6000 | 0.0000 | 1.0000 |
| LR_COMMERCIAL | commercial | 0.4459 | 0.5522 | 0.2243 | 0.0035 | -0.6776 | 0.00e+00 | 0.6355 | 0.2000 | 1.0000 |
| XGB_COMMERCIAL | commercial | 0.4162 | 0.5938 | 0.2184 | 0.0093 | -0.7998 | 0.00e+00 | 0.6661 | 0.2980 | 0.9807 |
| LR_EIGHT_STREAM | 8-stream | 0.0237 | 0.9761 | 0.0012 | 0.0009 | -36.9504 | 0.00e+00 | 1.0000 | 1.0000 | 1.0000 |
| XGB_EIGHT_STREAM | 8-stream | 0.0007 | 0.9991 | 0.0000 | 0.0000 | -248.7712 | 0.00e+00 | 1.0000 | 1.0000 | 1.0000 |

## Notes

- Closed-form baselines (DS / NB / SA / MV) consume only commercial-tier evidence (SR / CC / TD / HA) by construction. They have no notion of trajectory consistency, source independence, or cryptographic provenance.

- `LR_COMMERCIAL` and `XGB_COMMERCIAL` see the same 4-dim commercial-tier feature vector as the closed-form baselines. **Methodologically faithful comparison** to the request 'what if you fit a learned model on the same data the classical baselines see?'.

- `LR_EIGHT_STREAM` and `XGB_EIGHT_STREAM` see the full 8-dim feature vector including the integrity-tier stream values (`S_CTC`, `S_SIS`, `S_CPS`, `S_RIS`). **Strongest possible learned baseline** — they have access to every signal VW sees. Any residual delta vs VW on this comparison is attributable to the **composition rule** (HALT propagation, tier-aware γ/δ, RIS/CPS HALT-class semantics), not to feature availability.

- All learned baselines use `random_state=42` and `n_jobs=1` for reproducibility. Calling `fit()` twice on the same data with the same seed produces bit-identical model parameters (verified by tests).

- VW evaluated under government tier (γ=δ=1). Threshold = 0.30 (CW).

- Train / test split is stratified by `(label, attack_class)` so every stratum is represented in both halves.

