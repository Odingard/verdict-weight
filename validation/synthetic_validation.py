"""
Reproducer: synthetic validation of the eight-stream architecture.

This is the canonical reproducer for the synthetic empirical surface
cited in ``VALIDATION.md``. Run from the repo root::

    python -m validation.synthetic_validation

or with a custom output directory and sample count::

    python -m validation.synthetic_validation \\
        --n 10000 --seed 42 --out validation/results

Outputs (under ``--out``):

  * ``synthetic_report.json`` — machine-readable report
  * ``synthetic_report.md``   — human-readable report

Both files include the platform fingerprint, package version, sample
count, seed, and the actual measured numbers from this run. Numbers
are NOT tuned to match any external claim — this script is the
authoritative empirical surface.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

import verdict_weight as vw
from verdict_weight import (
    ContextType,
    DeploymentTier,
    UnifiedComposer,
    UnifiedInputs,
)

from .baselines import (
    dempster_shafer,
    naive_bayes,
    simple_averaging,
    max_voting,
)
from .datasets import Sample, generate_dataset
from .metrics import (
    auc_roc,
    bootstrap_ci,
    brier_score,
    cohens_d,
    confusion,
    reliability,
    sensitivity,
    specificity,
    welch_t_test,
)


# ─────────────────────────────────────────────────────────────
# Per-sample VW score
# ─────────────────────────────────────────────────────────────


def vw_score_one(composer: UnifiedComposer, sample: Sample) -> Dict:
    """Run one sample through the unified composer."""
    inputs = UnifiedInputs(
        SR=sample.SR,
        CC=sample.CC,
        TD=sample.TD,
        HA=sample.HA,
        trajectory=list(sample.trajectory),
        sources=list(sample.sources),
        provenance_chain=list(sample.provenance_chain),
        registry=sample.registry,
        context=ContextType.CYBERSECURITY_GENERAL,
        deployment_tier=DeploymentTier.GOVERNMENT,
    )
    result = composer.score(inputs)
    return {
        "label": sample.label,
        "attack_class": sample.attack_class,
        "halted": result.halted,
        "halted_at": result.halted_at,
        "halt_reason": result.halt_reason,
        "cw_certified": result.cw_certified,
        "cw_base": result.cw_base,
        "S_RIS": result.streams["S_RIS"],
        "S_CPS": result.streams["S_CPS"],
        "S_SIS": result.streams["S_SIS"],
        "S_CTC": result.streams["S_CTC"],
        "action_tier": result.action_tier,
    }


# ─────────────────────────────────────────────────────────────
# Baseline runners
# ─────────────────────────────────────────────────────────────


def baseline_score_one(sample: Sample, method: str) -> float:
    """Run one sample through a baseline fusion method."""
    scores = [sample.SR, sample.CC, sample.TD, sample.HA]
    if method == "DS":
        return dempster_shafer(scores)
    if method == "NB":
        return naive_bayes(scores)
    if method == "SA":
        return simple_averaging(scores)
    if method == "MV":
        return max_voting(scores)
    raise ValueError(f"Unknown baseline method: {method}")


# ─────────────────────────────────────────────────────────────
# Per-method evaluation
# ─────────────────────────────────────────────────────────────


def evaluate_method(
    name: str,
    cw_values: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.30,
) -> Dict:
    """Compute the standard metric battery for a single fusion method.

    Parameters
    ----------
    name:
        Method label.
    cw_values:
        Predicted Consequence Weights (or fusion probabilities), one
        per sample.
    labels:
        Binary labels: 1 = adversarial, 0 = legitimate.
    threshold:
        Adversarial threshold: ``cw < threshold`` is classified
        "adversarial-suppressed". Lower CW = more suppression.
    """
    # Convert to "adversarial probability" — high values mean "this is adversarial"
    # In our convention, lower CW means more suppressed (treated as adversarial).
    # So adversarial_prob = 1 - cw.
    adv_prob = 1.0 - cw_values

    # Binary classification by threshold
    pred_adv = (cw_values < threshold).astype(int)
    cm = confusion(labels, pred_adv)
    sens = sensitivity(labels, pred_adv)
    spec = specificity(labels, pred_adv)

    # Calibration
    brier = brier_score(labels, adv_prob)
    rel = reliability(labels, adv_prob, n_bins=10)

    # AUC (using adv_prob: higher = more adversarial)
    auc = auc_roc(labels, adv_prob)

    # Group means + Cohen's d
    adv_cw = cw_values[labels == 1]
    legit_cw = cw_values[labels == 0]
    d = cohens_d(adv_cw, legit_cw)
    t_stat, p_val = welch_t_test(adv_cw, legit_cw)

    # Bootstrap CIs on the means
    point_adv, lo_adv, hi_adv = bootstrap_ci(
        adv_cw.tolist(), np.mean, n_iter=1000, seed=42
    )
    point_legit, lo_legit, hi_legit = bootstrap_ci(
        legit_cw.tolist(), np.mean, n_iter=1000, seed=42
    )

    return {
        "name": name,
        "n_samples": int(len(labels)),
        "n_adversarial": int(np.sum(labels == 1)),
        "n_legitimate": int(np.sum(labels == 0)),
        "threshold": threshold,
        "confusion": cm,
        "sensitivity": sens,
        "specificity": spec,
        "brier_score": brier,
        "reliability_rel": rel,
        "auc_roc": auc,
        "mean_cw_adversarial": point_adv,
        "ci95_cw_adversarial": [lo_adv, hi_adv],
        "mean_cw_legitimate": point_legit,
        "ci95_cw_legitimate": [lo_legit, hi_legit],
        "cohens_d": d,
        "welch_t": t_stat,
        "welch_p": p_val,
    }


# ─────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────


def run(n: int = 10_000, seed: int = 42, out_dir: Path = Path("validation/results")) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Generating N={n} synthetic samples (seed={seed})...", flush=True)
    t0 = time.time()
    samples, genesis = generate_dataset(n, seed=seed)
    t_gen = time.time() - t0

    # Construct VW composer with the genesis registry
    composer = UnifiedComposer(registry=genesis)

    print(f"[2/4] Scoring N={n} samples through eight-stream pipeline...", flush=True)
    t0 = time.time()
    vw_results = []
    cw_values: List[float] = []
    labels: List[int] = []
    halt_counts: Dict[str, int] = {"RIS": 0, "CPS": 0, "none": 0}
    per_class_cw: Dict[str, List[float]] = {
        "legitimate": [], "AC-1": [], "AC-2": [], "AC-3": [], "AC-4": [], "AC-5": [],
    }
    for s in samples:
        r = vw_score_one(composer, s)
        vw_results.append(r)
        # If halted, treat CW as 0 (maximum suppression)
        cw = 0.0 if r["halted"] else float(r["cw_certified"] or 0.0)
        cw_values.append(cw)
        labels.append(1 if s.label == "adversarial" else 0)
        if r["halted"]:
            halt_counts[r["halted_at"]] = halt_counts.get(r["halted_at"], 0) + 1
        else:
            halt_counts["none"] += 1
        key = s.attack_class or "legitimate"
        per_class_cw[key].append(cw)
    t_score = time.time() - t0

    cw_values_np = np.asarray(cw_values, dtype=float)
    labels_np = np.asarray(labels, dtype=int)
    vw_eval = evaluate_method("VERDICT_WEIGHT", cw_values_np, labels_np, threshold=0.30)

    print(f"[3/4] Scoring N={n} samples through baseline methods...", flush=True)
    baseline_evals: Dict[str, Dict] = {}
    for method in ("DS", "NB", "SA", "MV"):
        t0 = time.time()
        bvals = np.asarray([baseline_score_one(s, method) for s in samples], dtype=float)
        baseline_evals[method] = evaluate_method(
            {"DS": "Dempster-Shafer", "NB": "Naive_Bayes", "SA": "Simple_Averaging", "MV": "Max_Voting"}[method],
            bvals, labels_np, threshold=0.30,
        )
        baseline_evals[method]["wall_seconds"] = round(time.time() - t0, 4)

    print(f"[4/4] Building report...", flush=True)
    # Per-attack-class breakdown
    class_breakdown = {}
    for cls, vals in per_class_cw.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        class_breakdown[cls] = {
            "n": int(len(arr)),
            "mean_cw": float(np.mean(arr)),
            "median_cw": float(np.median(arr)),
            "std_cw": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min_cw": float(np.min(arr)),
            "max_cw": float(np.max(arr)),
        }

    report = {
        "schema_version": "1.0.0",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {
            "python": sys.version.split()[0],
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "numpy": np.__version__,
        },
        "package": {
            "name": "verdict_weight",
            "version": vw.__version__,
        },
        "configuration": {
            "n_samples": int(n),
            "seed": int(seed),
            "legitimate_fraction": 0.5,
            "adversarial_threshold": 0.30,
            "deployment_tier": "government",
        },
        "wall_seconds": {
            "data_generation": round(t_gen, 4),
            "vw_scoring": round(t_score, 4),
        },
        "halt_counts": halt_counts,
        "verdict_weight": vw_eval,
        "baselines": baseline_evals,
        "per_attack_class_cw": class_breakdown,
    }

    # Write JSON
    json_path = out_dir / "synthetic_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"     wrote {json_path}", flush=True)

    # Write Markdown
    md_path = out_dir / "synthetic_report.md"
    with open(md_path, "w") as f:
        f.write(_render_markdown(report))
    print(f"     wrote {md_path}", flush=True)

    return report


def _render_markdown(report: Dict) -> str:
    lines = []
    lines.append("# Synthetic Validation Report — VERDICT WEIGHT")
    lines.append("")
    lines.append(f"- **Generated:** {report['generated_at_utc']}")
    lines.append(f"- **Package version:** {report['package']['version']}")
    lines.append(f"- **Python:** {report['platform']['python']} on "
                 f"{report['platform']['system']} {report['platform']['release']} "
                 f"({report['platform']['machine']})")
    lines.append(f"- **NumPy:** {report['platform']['numpy']}")
    lines.append("")
    cfg = report["configuration"]
    lines.append(f"- **N samples:** {cfg['n_samples']:,}")
    lines.append(f"- **Seed:** {cfg['seed']}")
    lines.append(f"- **Legitimate fraction:** {cfg['legitimate_fraction']:.2f}")
    lines.append(f"- **Adversarial threshold:** {cfg['adversarial_threshold']:.2f}")
    lines.append(f"- **Deployment tier:** {cfg['deployment_tier']}")
    lines.append("")
    lines.append("## Wall time")
    ws = report["wall_seconds"]
    lines.append(f"- Data generation: {ws['data_generation']:.4f} s")
    lines.append(f"- VW scoring: {ws['vw_scoring']:.4f} s "
                 f"(throughput = {cfg['n_samples'] / max(ws['vw_scoring'], 1e-9):,.0f} samples/sec)")
    lines.append("")
    lines.append("## HALT counts (RIS / CPS absorbing states)")
    for k, v in report["halt_counts"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Head-to-head comparison (adversarial threshold = 0.30)")
    lines.append("")
    lines.append("| Method | Adv mean CW | Legit mean CW | Brier ↓ | REL ↓ | AUC ↑ | Cohen's d | Welch p |")
    lines.append("|---|---|---|---|---|---|---|---|")
    methods = [("VERDICT_WEIGHT", report["verdict_weight"])] + [
        (k, v) for k, v in report["baselines"].items()
    ]
    for label, ev in methods:
        lines.append(
            f"| {ev['name']} | {ev['mean_cw_adversarial']:.4f} | "
            f"{ev['mean_cw_legitimate']:.4f} | {ev['brier_score']:.4f} | "
            f"{ev['reliability_rel']:.4f} | {ev['auc_roc']:.4f} | "
            f"{ev['cohens_d']:.4f} | {ev['welch_p']:.4g} |"
        )
    lines.append("")
    lines.append("## Per-attack-class breakdown (VERDICT WEIGHT)")
    lines.append("")
    lines.append("| Class | N | Mean CW | Median CW | Std CW | Min | Max |")
    lines.append("|---|---|---|---|---|---|---|")
    for cls, stats in report["per_attack_class_cw"].items():
        lines.append(
            f"| {cls} | {stats['n']} | {stats['mean_cw']:.4f} | "
            f"{stats['median_cw']:.4f} | {stats['std_cw']:.4f} | "
            f"{stats['min_cw']:.4f} | {stats['max_cw']:.4f} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**Reproducibility:** every number above can be reproduced from a clean")
    lines.append("checkout with::")
    lines.append("")
    lines.append("    pip install -e .")
    lines.append(f"    python -m validation.synthetic_validation --n {cfg['n_samples']} --seed {cfg['seed']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="synthetic_validation")
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("validation/results"))
    args = p.parse_args(argv)
    run(n=args.n, seed=args.seed, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
