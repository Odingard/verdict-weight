"""Learned-baseline head-to-head benchmark — VW vs LR / XGBoost vs DS / NB / SA / MV.

Pre-empts the most-likely peer-review request for both companion
papers (TIFS adversarial-trajectory and TDSC eight-stream): a head-to-
head comparison against learned tabular fusion baselines on the same
N=10,000 synthetic dataset used elsewhere in the harness.

Methodology
-----------

The benchmark uses a stratified train/test split (default 70/30,
``seed=42``) over the ``(label, attack_class)`` strata. Learned
baselines are fit on the train portion. **Every** method (closed-form
and learned) is then evaluated on the held-out test portion so that
comparisons are at parity. VW does not require fitting and is
evaluated directly on the test split.

For learned baselines we report two feature-set variants:

  * ``commercial`` (4 features: SR, CC, TD, HA)  — direct comparison
    to DS / NB / SA / MV which only see commercial-tier evidence.
  * ``eight_stream`` (8 features: commercial + S_CTC + S_SIS + S_CPS +
    S_RIS) — the strongest possible learned baseline, which sees every
    signal VW sees. If VW still dominates, the delta is attributable
    to the **composition rule** (HALT propagation, tier-aware
    exponents γ/δ, and the structural relationship between RIS / CPS
    as HALT-class streams), not to feature availability.

Run with::

    python -m benchmarks.learned_head_to_head --n 10000 --seed 42

Outputs (under ``--out``, default ``benchmarks/results``):

  * ``learned_head_to_head.json`` — machine-readable
  * ``learned_head_to_head.md``   — human-readable
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import verdict_weight as vw
from validation import baselines, learned_baselines
from validation.datasets import Sample, generate_dataset
from validation.synthetic_validation import evaluate_method, vw_score_one
from verdict_weight import (
    SourceRegistry,
    UnifiedComposer,
)


# ---------------------------------------------------------------------
# Per-method scoring helpers
# ---------------------------------------------------------------------


def _build_composer(registry: SourceRegistry) -> UnifiedComposer:
    return UnifiedComposer(registry=registry)


def _score_vw(
    composer: UnifiedComposer, samples: List[Sample]
) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
    """Score every sample through the unified eight-stream pipeline.

    Returns
    -------
    (cw_array, vw_results, elapsed_seconds)
    """
    t0 = time.perf_counter()
    cw_list: List[float] = []
    vw_results: List[Dict[str, Any]] = []
    for s in samples:
        r = vw_score_one(composer, s)
        vw_results.append(r)
        cw_list.append(0.0 if r["halted"] else float(r["cw_certified"] or 0.0))
    return np.asarray(cw_list, dtype=float), vw_results, time.perf_counter() - t0


def _score_classical(samples: List[Sample]) -> Dict[str, np.ndarray]:
    """Closed-form fusion baselines on commercial-tier evidence."""
    n = len(samples)
    out = {
        "DEMPSTER_SHAFER":   np.zeros(n, dtype=float),
        "NAIVE_BAYES":       np.zeros(n, dtype=float),
        "SIMPLE_AVERAGING":  np.zeros(n, dtype=float),
        "MAX_VOTING":        np.zeros(n, dtype=float),
    }
    for i, s in enumerate(samples):
        ev = [s.SR, s.CC, s.TD, s.HA]
        out["DEMPSTER_SHAFER"][i]  = baselines.dempster_shafer(ev)
        out["NAIVE_BAYES"][i]      = baselines.naive_bayes(ev)
        out["SIMPLE_AVERAGING"][i] = baselines.simple_averaging(ev)
        out["MAX_VOTING"][i]       = baselines.max_voting(ev)
    return out


# ---------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------


def run(
    n: int = 10_000,
    seed: int = 42,
    test_fraction: float = 0.30,
    out_dir: Path = Path("benchmarks/results"),
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Generating N={n:,} synthetic samples (seed={seed})...", flush=True)
    t0 = time.time()
    samples, registry = generate_dataset(n, seed=seed)
    composer = _build_composer(registry)
    t_gen = time.time() - t0

    labels = np.array(
        [1 if s.label == "adversarial" else 0 for s in samples], dtype=int
    )
    attack_classes = [s.attack_class for s in samples]

    print(f"[2/5] Stratified train/test split "
          f"(test_fraction={test_fraction}, seed={seed})...", flush=True)
    train_idx, test_idx = learned_baselines.stratified_split(
        n=n,
        labels=labels,
        attack_classes=attack_classes,
        test_fraction=test_fraction,
        seed=seed,
    )

    print(f"[3/5] Scoring all N={n:,} samples through eight-stream pipeline...",
          flush=True)
    vw_cw_all, vw_results_all, vw_elapsed = _score_vw(composer, samples)

    print(f"[4/5] Building features + fitting learned baselines...", flush=True)
    X_commercial = learned_baselines.commercial_features(samples)
    X_eight = learned_baselines.eight_stream_features(samples, vw_results_all)

    learned_models: Dict[str, learned_baselines._LearnedBaselineBase] = {}
    fit_timings: Dict[str, float] = {}

    def _fit_one(name: str, model_cls, variant: str, X: np.ndarray) -> None:
        t = time.perf_counter()
        model = model_cls(feature_variant=variant, random_state=seed)
        model.fit(X[train_idx], labels[train_idx])
        learned_models[name] = model
        fit_timings[name] = time.perf_counter() - t

    _fit_one("LR_COMMERCIAL",   learned_baselines.LogisticRegressionBaseline,
             "commercial",   X_commercial)
    _fit_one("LR_EIGHT_STREAM", learned_baselines.LogisticRegressionBaseline,
             "eight_stream", X_eight)
    _fit_one("XGB_COMMERCIAL",  learned_baselines.XGBoostBaseline,
             "commercial",   X_commercial)
    _fit_one("XGB_EIGHT_STREAM", learned_baselines.XGBoostBaseline,
             "eight_stream", X_eight)

    print(f"[5/5] Evaluating ALL methods on the test split (n_test={len(test_idx):,})...",
          flush=True)

    # Evaluate VW on test split
    vw_cw_test = vw_cw_all[test_idx]
    labels_test = labels[test_idx]
    methods: Dict[str, Dict[str, Any]] = {}
    methods["VERDICT_WEIGHT"] = evaluate_method("VERDICT_WEIGHT", vw_cw_test, labels_test)

    # Closed-form baselines on test split
    classical_full = _score_classical(samples)
    for name, cw in classical_full.items():
        methods[name] = evaluate_method(name, cw[test_idx], labels_test)

    # Learned baselines on test split
    learned_predictions: Dict[str, np.ndarray] = {}
    for name, model in learned_models.items():
        X = X_commercial if "COMMERCIAL" in name else X_eight
        cw_test = model.predict_cw(X[test_idx])
        learned_predictions[name] = cw_test
        methods[name] = evaluate_method(name, cw_test, labels_test)

    # Build report
    summary = {
        "schema_version": "1.0.0",
        "harness": "learned_head_to_head",
        "version": "v1.2.0",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package": {"name": "verdict_weight", "version": vw.__version__},
        "platform": {
            "python":  sys.version.split()[0],
            "system":  platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "numpy":   np.__version__,
            "platform": platform.platform(),
        },
        "configuration": {
            "n_samples":         int(n),
            "seed":              int(seed),
            "test_fraction":     float(test_fraction),
            "n_train":           int(len(train_idx)),
            "n_test":            int(len(test_idx)),
            "adversarial_threshold": 0.30,
            "deployment_tier":   "government",
        },
        "wall_seconds": {
            "data_generation":  round(t_gen, 4),
            "vw_scoring_full":  round(vw_elapsed, 4),
            "learned_fit": {k: round(v, 4) for k, v in fit_timings.items()},
        },
        "methods": methods,
        "feature_variants": {
            "commercial":   list(learned_baselines.feature_names("commercial")),
            "eight_stream": list(learned_baselines.feature_names("eight_stream")),
        },
        "learned_baseline_metadata": _learned_metadata(learned_models),
    }

    json_path = out_dir / "learned_head_to_head.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=str)
    print(f"     wrote {json_path}", flush=True)

    md_path = out_dir / "learned_head_to_head.md"
    with open(md_path, "w") as f:
        f.write(_render_md(summary))
    print(f"     wrote {md_path}", flush=True)

    return summary


def _learned_metadata(models: Dict[str, Any]) -> Dict[str, Any]:
    """Capture per-model fitted metadata for the JSON output."""
    out: Dict[str, Any] = {}
    for name, model in models.items():
        entry: Dict[str, Any] = {
            "model_class":     model.__class__.__name__,
            "feature_variant": model.feature_variant,
            "feature_names":   list(model.feature_names_),
            "random_state":    int(model.random_state),
        }
        report = model.training_report
        if report is not None:
            entry["training_report"] = {
                "n_train":        report.n_train,
                "n_features":     report.n_features,
                "train_seconds":  round(report.train_seconds, 4),
            }
        if isinstance(model, learned_baselines.LogisticRegressionBaseline):
            if model.coef_ is not None:
                entry["coef_"]      = model.coef_.tolist()
            if model.intercept_ is not None:
                entry["intercept_"] = model.intercept_.tolist()
            entry["hyperparameters"] = {
                "C": model.C, "max_iter": model.max_iter, "solver": "lbfgs"
            }
        elif isinstance(model, learned_baselines.XGBoostBaseline):
            if model.feature_importances_ is not None:
                entry["feature_importances_"] = model.feature_importances_.tolist()
            entry["hyperparameters"] = {
                "n_estimators":  model.n_estimators,
                "max_depth":     model.max_depth,
                "learning_rate": model.learning_rate,
                "tree_method":   "hist",
            }
        out[name] = entry
    return out


def _render_md(summary: Dict[str, Any]) -> str:
    cfg = summary["configuration"]
    lines: List[str] = []
    lines.append("# Learned-Baseline Head-to-Head — VW vs LR / XGBoost vs Classical Fusion\n")
    lines.append(
        f"**Harness:** `learned_head_to_head` "
        f"(reviewer-pre-empt extension of Paper 3 §8.2 Table 4)\n"
    )
    lines.append(
        f"**N = {cfg['n_samples']:,}** "
        f"(train = {cfg['n_train']:,} / test = {cfg['n_test']:,}, "
        f"`test_fraction = {cfg['test_fraction']:.2f}`)\n"
    )
    lines.append(
        f"**seed = {cfg['seed']}**, **threshold = {cfg['adversarial_threshold']}**, "
        f"NumPy {summary['platform']['numpy']}, "
        f"Python {summary['platform']['python']}\n"
    )
    lines.append("")

    lines.append("## Comparative metrics (all methods, test split)\n")
    lines.append(
        "| Method | Variant | Mean Adv CW | Mean Legit CW | "
        "Brier | REL | Cohen's d | Welch p | AUC | Sens | Spec |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    order = [
        ("VERDICT_WEIGHT",   "8-stream"),
        ("DEMPSTER_SHAFER",  "commercial"),
        ("NAIVE_BAYES",      "commercial"),
        ("SIMPLE_AVERAGING", "commercial"),
        ("MAX_VOTING",       "commercial"),
        ("LR_COMMERCIAL",    "commercial"),
        ("XGB_COMMERCIAL",   "commercial"),
        ("LR_EIGHT_STREAM",  "8-stream"),
        ("XGB_EIGHT_STREAM", "8-stream"),
    ]
    for name, variant in order:
        m = summary["methods"].get(name)
        if not m:
            continue
        lines.append(
            f"| {name} | {variant} | "
            f"{m['mean_cw_adversarial']:.4f} | "
            f"{m['mean_cw_legitimate']:.4f} | "
            f"{m['brier_score']:.4f} | "
            f"{m['reliability_rel']:.4f} | "
            f"{m['cohens_d']:.4f} | "
            f"{m['welch_p']:.2e} | "
            f"{m['auc_roc']:.4f} | "
            f"{m['sensitivity']:.4f} | "
            f"{m['specificity']:.4f} |"
        )
    lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "- Closed-form baselines (DS / NB / SA / MV) consume only commercial-tier "
        "evidence (SR / CC / TD / HA) by construction. They have no notion of "
        "trajectory consistency, source independence, or cryptographic "
        "provenance.\n"
    )
    lines.append(
        "- `LR_COMMERCIAL` and `XGB_COMMERCIAL` see the same 4-dim commercial-tier "
        "feature vector as the closed-form baselines. **Methodologically faithful "
        "comparison** to the request 'what if you fit a learned model on the "
        "same data the classical baselines see?'.\n"
    )
    lines.append(
        "- `LR_EIGHT_STREAM` and `XGB_EIGHT_STREAM` see the full 8-dim feature "
        "vector including the integrity-tier stream values "
        "(`S_CTC`, `S_SIS`, `S_CPS`, `S_RIS`). **Strongest possible learned "
        "baseline** — they have access to every signal VW sees. Any residual "
        "delta vs VW on this comparison is attributable to the **composition "
        "rule** (HALT propagation, tier-aware γ/δ, RIS/CPS HALT-class semantics), "
        "not to feature availability.\n"
    )
    lines.append(
        "- All learned baselines use `random_state=42` and `n_jobs=1` for "
        "reproducibility. Calling `fit()` twice on the same data with the same "
        "seed produces bit-identical model parameters (verified by tests).\n"
    )
    lines.append(
        "- VW evaluated under government tier (γ=δ=1). Threshold = 0.30 (CW).\n"
    )
    lines.append(
        "- Train / test split is stratified by `(label, attack_class)` so every "
        "stratum is represented in both halves.\n"
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-fraction", type=float, default=0.30)
    p.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args(argv)
    summary = run(
        n=args.n,
        seed=args.seed,
        test_fraction=args.test_fraction,
        out_dir=args.out,
    )
    print(json.dumps(summary["methods"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
