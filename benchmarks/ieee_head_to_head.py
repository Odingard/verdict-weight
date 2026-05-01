"""IEEE head-to-head benchmark — VW vs Dempster-Shafer / NB / SA / MV.

Reproducer for Paper 3 §8.2 (Table 4).

This script is the canonical replication of the head-to-head fusion
comparison cited in the unified eight-stream paper. It generates a
deterministic synthetic dataset (``N=2,000`` by default) across the
five attack classes plus legitimate signals, then scores every sample
through:

  * VERDICT WEIGHT  (full eight-stream composition, government tier)
  * Dempster-Shafer (DS) combination over commercial-tier evidence
  * Naive-Bayes fusion (NB)
  * Simple averaging (SA)
  * Max voting (MV)

For each method the harness reports:

  * Mean adversarial CW
  * Mean legitimate CW
  * Brier score
  * Reliability (REL)
  * Cohen's d (legitimate vs adversarial)
  * Welch t-test p-value
  * AUC-ROC

Run with::

    python -m benchmarks.ieee_head_to_head --n 2000 --seed 42

Outputs ``ieee_head_to_head.json`` (machine-readable) and
``ieee_head_to_head.md`` (human-readable) under ``--out``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from validation import baselines, metrics
from validation.datasets import Sample, generate_dataset
from validation.synthetic_validation import (
    evaluate_method,
    vw_score_one,
)
from verdict_weight import (
    SourceRegistry,
    UnifiedComposer,
)


def _build_composer(registry: SourceRegistry) -> UnifiedComposer:
    return UnifiedComposer(registry=registry)


def _commercial_evidence(sample: Sample) -> List[float]:
    """Per-stream evidence vector used by all classical fusion baselines."""
    return [sample.SR, sample.CC, sample.TD, sample.HA]


def _score_baselines(samples: List[Sample]) -> Dict[str, np.ndarray]:
    n = len(samples)
    ds = np.zeros(n, dtype=float)
    nb = np.zeros(n, dtype=float)
    sa = np.zeros(n, dtype=float)
    mv = np.zeros(n, dtype=float)
    for i, s in enumerate(samples):
        ev = _commercial_evidence(s)
        ds[i] = baselines.dempster_shafer(ev)
        nb[i] = baselines.naive_bayes(ev)
        sa[i] = baselines.simple_averaging(ev)
        mv[i] = baselines.max_voting(ev)
    return {
        "DEMPSTER_SHAFER": ds,
        "NAIVE_BAYES": nb,
        "SIMPLE_AVERAGING": sa,
        "MAX_VOTING": mv,
    }


def run(n: int = 2000, seed: int = 42, out_dir: Path = Path("benchmarks/results")) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, registry = generate_dataset(n, seed=seed)
    composer = _build_composer(registry)
    labels = np.array([1 if s.label == "adversarial" else 0 for s in samples], dtype=int)

    # VERDICT WEIGHT — HALT coerces to 0 (full suppression)
    t0 = time.perf_counter()
    vw_cw_list: List[float] = []
    for s in samples:
        r = vw_score_one(composer, s)
        vw_cw_list.append(0.0 if r["halted"] else float(r["cw_certified"] or 0.0))
    vw_cw = np.array(vw_cw_list, dtype=float)
    vw_elapsed = time.perf_counter() - t0

    # Baselines
    t0 = time.perf_counter()
    baseline_cw = _score_baselines(samples)
    baseline_elapsed = time.perf_counter() - t0

    methods: Dict[str, Dict[str, Any]] = {}
    methods["VERDICT_WEIGHT"] = evaluate_method("VERDICT_WEIGHT", vw_cw, labels)
    for name, cw in baseline_cw.items():
        methods[name] = evaluate_method(name, cw, labels)

    summary = {
        "harness": "ieee_head_to_head",
        "version": "v1.2.0",
        "config": {
            "n": n,
            "seed": seed,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "timings": {
            "vw_score_seconds": round(vw_elapsed, 4),
            "baseline_score_seconds": round(baseline_elapsed, 4),
            "vw_throughput_samples_per_sec": round(n / vw_elapsed, 1) if vw_elapsed > 0 else None,
        },
        "methods": methods,
    }

    (out_dir / "ieee_head_to_head.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "ieee_head_to_head.md").write_text(_render_md(summary))
    return summary


def _render_md(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# IEEE Head-to-Head Benchmark — VW vs Classical Fusion\n")
    cfg = summary["config"]
    lines.append(f"**Harness:** `ieee_head_to_head` (Paper 3 §8.2 Table 4 reproducer)\n")
    lines.append(f"**N = {cfg['n']:,}**, **seed = {cfg['seed']}**, "
                 f"NumPy {cfg['numpy']}, Python {cfg['python']}\n")
    t = summary["timings"]
    if t["vw_throughput_samples_per_sec"] is not None:
        lines.append(f"**VW throughput:** {t['vw_throughput_samples_per_sec']:,.1f} samples/sec "
                     f"({t['vw_score_seconds']*1000:.0f} ms total)\n")
    lines.append("")
    lines.append("## Comparative metrics\n")
    lines.append("| Method | Mean Adv CW | Mean Legit CW | Brier | REL | Cohen's d | Welch p | AUC |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, m in summary["methods"].items():
        lines.append(
            f"| {name} | {m['mean_cw_adversarial']:.4f} | {m['mean_cw_legitimate']:.4f} | "
            f"{m['brier_score']:.4f} | {m['reliability_rel']:.4f} | "
            f"{m['cohens_d']:.4f} | {m['welch_p']:.2e} | {m['auc_roc']:.4f} |"
        )
    lines.append("")
    lines.append("## Notes\n")
    lines.append("- Adversarial samples cover AC-1 through AC-5; legitimate samples are stable-high "
                 "trajectories with independent sources and valid provenance / registry.\n")
    lines.append("- VW evaluated under government tier (γ=δ=1).\n")
    lines.append("- Baselines see only commercial-tier evidence (SR/CC/TD/HA); they do not have "
                 "access to trajectory / source-independence / provenance / registry signals "
                 "by construction (these dimensions do not exist in classical fusion).\n")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args(argv)
    summary = run(n=args.n, seed=args.seed, out_dir=args.out)
    print(json.dumps(summary["methods"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
