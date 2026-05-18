"""
Per-vertical government-tier benchmark for VERDICT WEIGHT™ v1.2.0.

Runs the full 8-stream pipeline with vertical-specific parameters:
  - SR ranges drawn from the vertical's registry profile
  - TD computed from the vertical's lambda_decay
  - CC drawn from the vertical's typical_cc range
  - Source count drawn from the vertical's n_sources

Usage from repo root:

    python -m benchmarks.vertical_validation --vertical cybersecurity --seed 42
    python -m benchmarks.vertical_validation --vertical healthcare --seed 42
    python -m benchmarks.vertical_validation --all --seed 42

Outputs per-vertical JSON reports to benchmarks/results/vertical/.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import verdict_weight as vw
from verdict_weight import (
    ContextType,
    DeploymentTier,
    UnifiedComposer,
    UnifiedInputs,
    Source,
    SourceRegistry,
    TrajectoryPoint,
    ProvenanceStep,
    build_provenance_chain,
)

from validation.baselines import (
    dempster_shafer,
    naive_bayes,
    simple_averaging,
    max_voting,
)
from validation.metrics import (
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
from validation.datasets import Sample


# ─────────────────────────────────────────────────────────────
# Vertical profiles (from VW_VERT_ALL_PROFILES_v1_2_0.json)
# ─────────────────────────────────────────────────────────────

VERTICAL_PROFILES = {
    "cybersecurity": {
        "display_name": "Cybersecurity / Threat Intelligence",
        "sr_range": (0.10, 0.98),
        "sr_tier1_floor": 0.94,
        "lambda_decay": 0.693,
        "cc_range": (2, 5),
        "n_sources": 22,
        "context": ContextType.CYBERSECURITY_GENERAL,
    },
    "healthcare": {
        "display_name": "Healthcare / Clinical Decision Support",
        "sr_range": (0.20, 0.97),
        "sr_tier1_floor": 0.94,
        "lambda_decay": 0.0019,
        "cc_range": (1, 3),
        "n_sources": 12,
        "context": ContextType.HEALTHCARE_DIAGNOSTIC,
    },
    "financial": {
        "display_name": "Financial Services / Fraud Detection",
        "sr_range": (0.30, 0.98),
        "sr_tier1_floor": 0.92,
        "lambda_decay": 0.231,
        "cc_range": (2, 4),
        "n_sources": 10,
        "context": ContextType.FINANCIAL_FRAUD,
    },
    "defense": {
        "display_name": "Defense / Intelligence Community",
        "sr_range": (0.10, 0.97),
        "sr_tier1_floor": 0.93,
        "lambda_decay": 0.0347,
        "cc_range": (1, 4),
        "n_sources": 10,
        "context": ContextType.DEFENSE_INTELLIGENCE,
    },
    "legal": {
        "display_name": "Legal / E-Discovery / Digital Forensics",
        "sr_range": (0.58, 0.95),
        "sr_tier1_floor": 0.92,
        "lambda_decay": 0.00274,
        "cc_range": (2, 6),
        "n_sources": 7,
        "context": ContextType.LEGAL_EVIDENCE,
    },
    "rag": {
        "display_name": "Enterprise RAG / Knowledge Retrieval",
        "sr_range": (0.30, 0.95),
        "sr_tier1_floor": 0.90,
        "lambda_decay": 0.0087,
        "cc_range": (1, 3),
        "n_sources": 7,
        "context": ContextType.RAG_ENTERPRISE,
    },
    "manufacturing": {
        "display_name": "Manufacturing / Industrial OT",
        "sr_range": (0.35, 0.96),
        "sr_tier1_floor": 0.93,
        "lambda_decay": 0.0231,
        "cc_range": (2, 5),
        "n_sources": 10,
        "context": ContextType.CYBERSECURITY_GENERAL,  # No MANUFACTURING profile; OT security is closest to cybersecurity
    },
}

# Map vertical names to ContextType, falling back to general
CONTEXT_FALLBACK = ContextType.CYBERSECURITY_GENERAL


# ─────────────────────────────────────────────────────────────
# Vertical-parameterized data generation
# ─────────────────────────────────────────────────────────────

_INSTITUTIONS = [
    "inst_alpha", "inst_beta", "inst_gamma", "inst_delta", "inst_epsilon",
    "inst_zeta", "inst_eta", "inst_theta", "inst_iota", "inst_kappa",
    "inst_lambda", "inst_mu", "inst_nu", "inst_xi", "inst_omicron",
]

_GEOS = [
    "us_west", "us_east", "europe_central", "asia_east", "asia_south",
    "middle_east", "oceania", "europe_north", "us_southeast", "international",
]


def _make_sources_independent(rng: random.Random, k: int) -> List[Source]:
    insts = rng.sample(_INSTITUTIONS, min(k, len(_INSTITUTIONS)))
    geos = rng.sample(_GEOS, min(k, len(_GEOS)))
    return [
        Source(
            source_id=f"src_{i}_{rng.randint(1000,9999)}",
            institution=insts[i % len(insts)],
            geography=geos[i % len(geos)],
            publish_time=float(rng.randint(1_600_000_000, 1_700_000_000)),
            primary_citations={f"cite_{rng.randint(100,999)}"},
        )
        for i in range(k)
    ]


def _make_sources_curveball(rng: random.Random, k: int) -> List[Source]:
    shared_inst = rng.choice(_INSTITUTIONS)
    shared_geo = rng.choice(_GEOS)
    shared_cite = f"cite_{rng.randint(100,999)}"
    shared_time = float(rng.randint(1_600_000_000, 1_700_000_000))
    return [
        Source(
            source_id=f"src_cb_{i}_{rng.randint(1000,9999)}",
            institution=shared_inst,
            geography=shared_geo,
            publish_time=shared_time,
            primary_citations={shared_cite},
        )
        for i in range(k)
    ]


def _valid_chain(rng: random.Random) -> List[ProvenanceStep]:
    n_steps = rng.randint(3, 6)
    payloads = [f"step_{i}_payload".encode() for i in range(n_steps)]
    actors = [f"actor_{i}" for i in range(n_steps)]
    timestamps = [float(1_700_000_000 + i * 3600) for i in range(n_steps)]
    return list(build_provenance_chain(payloads, actors, timestamps))


def _tampered_chain(rng: random.Random) -> List[ProvenanceStep]:
    chain = _valid_chain(rng)
    if len(chain) > 1:
        original = chain[-1]
        # Create a new step with tampered data but keep the old hash
        # This makes the chain invalid because hash won't match
        tampered = ProvenanceStep(
            data=b"TAMPERED_DATA" + rng.randbytes(16),
            timestamp=original.timestamp,
            actor=original.actor,
            hash=original.hash,  # hash is now wrong for the new data
        )
        chain[-1] = tampered
    return chain


def _pattern_stable(rng: random.Random, n: int = 10) -> List[TrajectoryPoint]:
    base = rng.uniform(0.78, 0.92)
    return [
        TrajectoryPoint(timestamp=float(t), value=max(0.0, min(1.0, base + rng.gauss(0, 0.025))))
        for t in range(n)
    ]


def _pattern_spike_collapse(rng: random.Random, n: int = 10) -> List[TrajectoryPoint]:
    peak_idx = rng.randint(1, max(1, int(n * 0.5)))
    spike = rng.uniform(0.85, 0.97)
    floor = rng.uniform(0.05, 0.20)
    points = []
    for t in range(n):
        if t < peak_idx:
            v = rng.uniform(0.20, 0.45)
        elif t == peak_idx:
            v = spike
        else:
            decay = (t - peak_idx) / max(1, n - peak_idx - 1)
            v = spike + (floor - spike) * decay
        points.append(TrajectoryPoint(timestamp=float(t), value=max(0.0, min(1.0, v + rng.gauss(0, 0.02)))))
    return points


def _td_from_lambda(rng: random.Random, lam: float) -> float:
    """Compute TD score from vertical-specific lambda_decay."""
    age_hours = rng.uniform(0.5, 168.0)  # 30 min to 1 week
    return math.exp(-lam * age_hours)


def _td_from_lambda_adversarial(rng: random.Random, lam: float) -> float:
    age_hours = rng.uniform(0.5, 168.0)
    return math.exp(-lam * age_hours)


def generate_vertical_dataset(
    vertical: str,
    n: int,
    seed: int = 42,
    legitimate_fraction: float = 0.5,
) -> Tuple[List[Sample], SourceRegistry]:
    """Generate synthetic dataset parameterized by vertical profile."""
    profile = VERTICAL_PROFILES[vertical]
    sr_lo, sr_hi = profile["sr_range"]
    lam = profile["lambda_decay"]
    cc_lo, cc_hi = profile["cc_range"]
    n_reg = profile["n_sources"]

    rng = random.Random(seed)

    # Build registry with vertical-appropriate SR distribution
    entries = {}
    for i in range(n_reg):
        sr = rng.uniform(sr_lo, sr_hi)
        entries[f"{vertical}_src_{i:03d}"] = round(sr, 4)
    genesis = SourceRegistry(entries=entries, version=1)

    samples: List[Sample] = []
    n_legit = int(n * legitimate_fraction)
    n_adv = n - n_legit

    attack_classes = ["AC-1", "AC-2", "AC-3", "AC-4", "AC-5"]

    # Legitimate samples
    for _ in range(n_legit):
        k = rng.randint(cc_lo, cc_hi)
        SR = rng.uniform(max(sr_lo, 0.60), sr_hi)
        CC = rng.uniform(0.60, 0.90)
        TD = _td_from_lambda(rng, lam)
        HA = rng.uniform(0.65, 0.90)
        sources = _make_sources_independent(rng, k)
        traj = _pattern_stable(rng)
        chain = _valid_chain(rng)
        samples.append(Sample(
            label="legitimate", attack_class=None,
            SR=SR, CC=CC, TD=TD, HA=HA,
            trajectory=tuple(traj), sources=tuple(sources),
            provenance_chain=tuple(chain), registry=genesis,
        ))

    # Adversarial samples
    for i in range(n_adv):
        ac = attack_classes[i % len(attack_classes)]
        k = rng.randint(cc_lo, cc_hi)

        if ac == "AC-1":  # Source spoofing
            SR = rng.uniform(0.85, 0.97)
            CC = rng.uniform(0.10, 0.30)
            TD = _td_from_lambda(rng, lam)
            HA = rng.uniform(0.40, 0.70)
            sources = _make_sources_independent(rng, 1)
            traj = _pattern_stable(rng)
            chain = _valid_chain(rng)
            reg = genesis

        elif ac == "AC-2":  # Curveball
            SR = rng.uniform(max(sr_lo, 0.65), sr_hi)
            CC = rng.uniform(0.70, 0.90)
            TD = _td_from_lambda(rng, lam)
            HA = rng.uniform(0.60, 0.85)
            sources = _make_sources_curveball(rng, max(k, 3))
            traj = _pattern_stable(rng)
            chain = _valid_chain(rng)
            reg = genesis

        elif ac == "AC-3":  # Trajectory fabrication
            SR = rng.uniform(max(sr_lo, 0.60), sr_hi)
            CC = rng.uniform(0.60, 0.90)
            TD = _td_from_lambda(rng, lam)
            HA = rng.uniform(0.65, 0.90)
            sources = _make_sources_independent(rng, k)
            traj = _pattern_spike_collapse(rng)
            chain = _valid_chain(rng)
            reg = genesis

        elif ac == "AC-4":  # Provenance forgery
            SR = rng.uniform(max(sr_lo, 0.60), sr_hi)
            CC = rng.uniform(0.60, 0.90)
            TD = _td_from_lambda(rng, lam)
            HA = rng.uniform(0.65, 0.90)
            sources = _make_sources_independent(rng, k)
            traj = _pattern_stable(rng)
            chain = _tampered_chain(rng)
            reg = genesis

        else:  # AC-5 Registry tamper
            SR = rng.uniform(max(sr_lo, 0.60), sr_hi)
            CC = rng.uniform(0.60, 0.90)
            TD = _td_from_lambda(rng, lam)
            HA = rng.uniform(0.65, 0.90)
            sources = _make_sources_independent(rng, k)
            traj = _pattern_stable(rng)
            chain = _valid_chain(rng)
            # Tamper the registry
            tampered_entries = dict(genesis.entries)
            tampered_key = rng.choice(list(tampered_entries.keys()))
            tampered_entries[tampered_key] = round(rng.uniform(0.01, 0.99), 4)
            reg = SourceRegistry(entries=tampered_entries, version=1)

        samples.append(Sample(
            label="adversarial", attack_class=ac,
            SR=SR, CC=CC, TD=TD, HA=HA,
            trajectory=tuple(traj), sources=tuple(sources),
            provenance_chain=tuple(chain), registry=reg,
            registry_tampered=(ac == "AC-5"),
        ))

    rng.shuffle(samples)
    return samples, genesis


# ─────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────


def vw_score_one(composer: UnifiedComposer, sample: Sample, context: ContextType) -> Dict:
    inputs = UnifiedInputs(
        SR=sample.SR,
        CC=sample.CC,
        TD=sample.TD,
        HA=sample.HA,
        trajectory=list(sample.trajectory),
        sources=list(sample.sources),
        provenance_chain=list(sample.provenance_chain),
        registry=sample.registry,
        context=context,
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
        "S_RIS": result.streams.get("S_RIS"),
        "S_CPS": result.streams.get("S_CPS"),
        "S_SIS": result.streams.get("S_SIS"),
        "S_CTC": result.streams.get("S_CTC"),
        "action_tier": result.action_tier,
    }


def evaluate_method(
    name: str,
    cw_values: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.30,
) -> Dict:
    adv_prob = 1.0 - cw_values
    pred_adv = (cw_values < threshold).astype(int)
    cm = confusion(labels, pred_adv)
    sens = sensitivity(labels, pred_adv)
    spec = specificity(labels, pred_adv)
    brier = brier_score(labels, adv_prob)
    rel = reliability(labels, adv_prob, n_bins=10)
    auc = auc_roc(labels, adv_prob)
    adv_cw = cw_values[labels == 1]
    legit_cw = cw_values[labels == 0]
    d = cohens_d(adv_cw, legit_cw)
    t_stat, p_val = welch_t_test(adv_cw, legit_cw)
    point_adv, lo_adv, hi_adv = bootstrap_ci(adv_cw.tolist(), np.mean, n_iter=500, seed=42)
    point_legit, lo_legit, hi_legit = bootstrap_ci(legit_cw.tolist(), np.mean, n_iter=500, seed=42)

    return {
        "name": name,
        "n_samples": int(len(labels)),
        "threshold": threshold,
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


def baseline_score_one(sample: Sample, method: str) -> float:
    scores = [sample.SR, sample.CC, sample.TD, sample.HA]
    if method == "DS": return dempster_shafer(scores)
    if method == "NB": return naive_bayes(scores)
    if method == "SA": return simple_averaging(scores)
    if method == "MV": return max_voting(scores)
    raise ValueError(method)


# ─────────────────────────────────────────────────────────────
# Per-vertical runner
# ─────────────────────────────────────────────────────────────


def run_vertical(vertical: str, n: int = 10_000, seed: int = 42, out_dir: Path = None) -> Dict:
    if out_dir is None:
        out_dir = Path("benchmarks/results/vertical")
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = VERTICAL_PROFILES[vertical]
    ctx = profile.get("context", CONTEXT_FALLBACK)
    print(f"\n{'='*60}")
    print(f"  VERTICAL: {profile['display_name']}")
    print(f"  N={n}, seed={seed}, tier=GOVERNMENT")
    print(f"  SR range: {profile['sr_range']}")
    print(f"  λ decay:  {profile['lambda_decay']}")
    print(f"  CC range: {profile['cc_range']}")
    print(f"{'='*60}\n")

    t0 = time.time()
    samples, genesis = generate_vertical_dataset(vertical, n, seed=seed)
    t_gen = time.time() - t0
    print(f"  [1/3] Generated {n} samples in {t_gen:.2f}s")

    composer = UnifiedComposer(registry=genesis)

    t0 = time.time()
    cw_values = []
    labels = []
    per_class_cw: Dict[str, List[float]] = {
        "legitimate": [], "AC-1": [], "AC-2": [], "AC-3": [], "AC-4": [], "AC-5": [],
    }
    halt_counts = {"RIS": 0, "CPS": 0, "none": 0}

    for s in samples:
        r = vw_score_one(composer, s, ctx)
        cw = 0.0 if r["halted"] else float(r["cw_certified"] or 0.0)
        cw_values.append(cw)
        labels.append(1 if s.label == "adversarial" else 0)
        key = s.attack_class or "legitimate"
        per_class_cw[key].append(cw)
        if r["halted"]:
            halt_counts[r["halted_at"]] = halt_counts.get(r["halted_at"], 0) + 1
        else:
            halt_counts["none"] += 1

    t_score = time.time() - t0
    print(f"  [2/3] Scored {n} samples in {t_score:.2f}s ({n/t_score:.0f}/sec)")

    cw_np = np.asarray(cw_values, dtype=float)
    labels_np = np.asarray(labels, dtype=int)
    vw_eval = evaluate_method("VERDICT_WEIGHT_GOV", cw_np, labels_np)

    # Baselines
    baseline_evals = {}
    for method in ("DS", "NB", "SA", "MV"):
        bvals = np.asarray([baseline_score_one(s, method) for s in samples], dtype=float)
        baseline_evals[method] = evaluate_method(method, bvals, labels_np)
    print(f"  [3/3] Baselines computed")

    # Per-class stats
    per_class_stats = {}
    for cls, vals in per_class_cw.items():
        if vals:
            arr = np.array(vals)
            per_class_stats[cls] = {
                "n": len(vals),
                "mean_cw": float(np.mean(arr)),
                "std_cw": float(np.std(arr)),
                "min_cw": float(np.min(arr)),
                "max_cw": float(np.max(arr)),
            }

    report = {
        "vertical": vertical,
        "display_name": profile["display_name"],
        "harness_version": "1.2.0",
        "stream_tier": "government",
        "gamma": 1.0,
        "delta": 1.0,
        "n": n,
        "seed": seed,
        "vertical_params": {
            "sr_range": list(profile["sr_range"]),
            "lambda_decay": profile["lambda_decay"],
            "cc_range": list(profile["cc_range"]),
            "n_sources": profile["n_sources"],
        },
        "platform": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "verdict_weight": vw.__version__,
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "timing": {
            "generation_sec": round(t_gen, 3),
            "scoring_sec": round(t_score, 3),
            "throughput_per_sec": round(n / t_score, 1),
        },
        "halt_counts": halt_counts,
        "vw_gov": vw_eval,
        "baselines": baseline_evals,
        "per_attack_class": per_class_stats,
    }

    out_file = out_dir / f"vertical_{vertical}_gov_n{n}_seed{seed}.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  → Saved: {out_file}")

    # Print summary
    print(f"\n  RESULTS — {profile['display_name']}:")
    print(f"    Brier (VW GOV):     {vw_eval['brier_score']:.4f}")
    print(f"    AUC (VW GOV):       {vw_eval['auc_roc']:.4f}")
    print(f"    Cohen's d:          {vw_eval['cohens_d']:.4f}")
    print(f"    Sensitivity:        {vw_eval['sensitivity']:.4f}")
    print(f"    Specificity:        {vw_eval['specificity']:.4f}")
    print(f"    Adv CW mean:        {vw_eval['mean_cw_adversarial']:.4f}")
    print(f"    Legit CW mean:      {vw_eval['mean_cw_legitimate']:.4f}")
    for cls in ["AC-1", "AC-2", "AC-3", "AC-4", "AC-5"]:
        if cls in per_class_stats:
            s = per_class_stats[cls]
            print(f"    {cls} mean CW:       {s['mean_cw']:.4f} (n={s['n']})")
    print(f"    HALTs — RIS: {halt_counts.get('RIS',0)}, CPS: {halt_counts.get('CPS',0)}")
    print(f"    Best baseline:      {min((e['brier_score'], k) for k, e in baseline_evals.items())[1]} "
          f"(Brier={min(e['brier_score'] for e in baseline_evals.values()):.4f})")

    return report


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Per-vertical government-tier benchmark")
    parser.add_argument("--vertical", type=str, help="Vertical name (e.g. cybersecurity)")
    parser.add_argument("--all", action="store_true", help="Run all 7 verticals")
    parser.add_argument("--n", type=int, default=10_000, help="Samples per vertical")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="benchmarks/results/vertical")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.all:
        all_reports = {}
        for v in VERTICAL_PROFILES:
            report = run_vertical(v, n=args.n, seed=args.seed, out_dir=out_dir)
            all_reports[v] = report
        # Write consolidated report
        consolidated = out_dir / f"vertical_all_gov_n{args.n}_seed{args.seed}.json"
        with open(consolidated, "w") as f:
            json.dump(all_reports, f, indent=2, default=str)
        print(f"\n{'='*60}")
        print(f"  CONSOLIDATED REPORT: {consolidated}")
        print(f"{'='*60}")
    elif args.vertical:
        if args.vertical not in VERTICAL_PROFILES:
            print(f"Unknown vertical: {args.vertical}")
            print(f"Available: {', '.join(VERTICAL_PROFILES.keys())}")
            sys.exit(1)
        run_vertical(args.vertical, n=args.n, seed=args.seed, out_dir=out_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
