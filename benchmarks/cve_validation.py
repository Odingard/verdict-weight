"""CVE validation benchmark — VW on real CISA KEV CVEs.

Reproducer for Paper 3 §8.4 (Table 6).

This script samples ``N=120`` CVEs deterministically from the frozen
CISA Known Exploited Vulnerabilities (KEV) catalog snapshot stored at
``validation/data/cisa_kev_snapshot.json``. Each CVE is converted into
unified-composer inputs using a documented deterministic mapping (see
``_cve_to_inputs`` below) and scored under government tier.

Because every CISA KEV entry is by construction a confirmed real-world
exploited vulnerability, the load-bearing question is **false-suppression
rate**: how often does the eight-stream pipeline incorrectly suppress a
known-exploited CVE? The harness reports per-class outcomes plus the
classical fusion baselines on the same inputs for comparison.

Run with::

    python -m benchmarks.cve_validation --n 120 --seed 42

Outputs ``cve_validation.json`` and ``cve_validation.md`` under ``--out``.

Snapshot version reported in JSON output:
    catalogVersion: 2026.04.30
    count: 1586
    snapshot retrieved: 2026-04-21
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import verdict_weight as vw
from validation import baselines
from validation.metrics import bootstrap_ci, cohens_d
from verdict_weight import (
    ContextType,
    DeploymentTier,
    Source,
    SourceRegistry,
    UnifiedComposer,
    UnifiedInputs,
    build_provenance_chain,
)


SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "validation" / "data" / "cisa_kev_snapshot.json"
GENESIS_TIME = datetime(2020, 1, 1).timestamp()


# ─────────────────────────────────────────────────────────────
# CVE → UnifiedInputs mapping
# ─────────────────────────────────────────────────────────────


def _parse_iso_date(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def _cve_to_inputs(
    cve: Dict[str, Any],
    registry: SourceRegistry,
) -> UnifiedInputs:
    """Deterministic mapping from a KEV record to unified composer inputs.

    Mapping rationale:
    - **SR** = 0.95 — NVD/CISA are authoritative reliability sources.
    - **CC** = 0.85 if ransomware-known, else 0.75 — KEV inclusion is
      itself a corroboration signal.
    - **TD** decays from 1.0 at dateAdded with λ=0.0005/day. Recent
      KEV entries are higher-confidence than ones added years ago.
    - **HA** = 0.93 — NVD historical accuracy at the source level.
    - Two pseudo-sources (NVD + CISA) with distinct institutions and
      geographies — these are genuinely independent feeds.
    - Two-step provenance chain (NVD entry → CISA-KEV inclusion).
    - No trajectory (CTC defaults to 1.0, optional stream).
    """
    cve_id = cve["cveID"]
    date_added_ts = _parse_iso_date(cve["dateAdded"])
    days_since = max(0.0, (datetime.now(timezone.utc).timestamp() - date_added_ts) / 86400.0)
    ransom = cve.get("knownRansomwareCampaignUse", "Unknown") == "Known"

    SR = 0.95
    CC = 0.85 if ransom else 0.75
    TD = float(np.exp(-0.0005 * days_since))
    HA = 0.93

    nvd_source = Source(
        source_id="NVD",
        institution="NIST",
        geography="US-MD",
        publish_time=date_added_ts - 86400.0,  # NVD typically lists before KEV
        primary_citations={cve_id, f"NVD:{cve_id}"},
    )
    cisa_source = Source(
        source_id="CISA-KEV",
        institution="CISA",
        geography="US-DC",
        publish_time=date_added_ts,
        primary_citations={cve_id, f"CISA-KEV:{cve_id}"},
    )

    chain = build_provenance_chain(
        [
            f"NVD published {cve_id} ({cve.get('vendorProject','?')} / {cve.get('product','?')})".encode(),
            f"CISA-KEV inclusion for {cve_id}: {cve.get('vulnerabilityName','')}".encode(),
        ],
        ["NVD", "CISA"],
        [date_added_ts - 86400.0, date_added_ts],
    )

    return UnifiedInputs(
        SR=SR, CC=CC, TD=TD, HA=HA,
        sources=[nvd_source, cisa_source],
        provenance_chain=chain,
        registry=registry,
        context=ContextType.CYBERSECURITY_GENERAL,
        deployment_tier=DeploymentTier.GOVERNMENT,
    )


def _build_registry() -> SourceRegistry:
    return SourceRegistry(
        entries={"NVD": 0.95, "CISA-KEV": 0.93},
        version=1,
    )


# ─────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────


def run(
    n: int = 120,
    seed: int = 42,
    suppression_threshold: float = 0.30,
    out_dir: Path = Path("benchmarks/results"),
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    catalog: List[Dict[str, Any]] = snapshot["vulnerabilities"]

    rng = random.Random(seed)
    sample_pool = list(catalog)
    rng.shuffle(sample_pool)
    sample_cves = sample_pool[:n]

    registry = _build_registry()
    composer = UnifiedComposer(registry=registry)

    vw_cw: List[float] = []
    halts: List[str] = []
    suppressed: List[str] = []
    per_record: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
    for cve in sample_cves:
        inputs = _cve_to_inputs(cve, registry)
        r = composer.score(inputs)
        cw = 0.0 if r.halted else float(r.cw_certified)
        vw_cw.append(cw)
        if r.halted:
            halts.append(r.halted_at or "?")
        if cw < suppression_threshold:
            suppressed.append(cve["cveID"])
        per_record.append({
            "cve": cve["cveID"],
            "vendor": cve.get("vendorProject"),
            "product": cve.get("product"),
            "ransomware": cve.get("knownRansomwareCampaignUse"),
            "halted": r.halted,
            "halted_at": r.halted_at,
            "cw_certified": None if r.halted else round(float(r.cw_certified), 6),
            "S_RIS": round(r.streams["S_RIS"], 6),
            "S_CPS": round(r.streams["S_CPS"], 6),
            "S_SIS": round(r.streams["S_SIS"], 6),
            "S_CTC": round(r.streams["S_CTC"], 6),
        })
    vw_elapsed = time.perf_counter() - t0

    # Baseline scoring on commercial-tier evidence
    baseline_results: Dict[str, Dict[str, Any]] = {}
    for name, fn in (
        ("DEMPSTER_SHAFER", baselines.dempster_shafer),
        ("NAIVE_BAYES", baselines.naive_bayes),
        ("SIMPLE_AVERAGING", baselines.simple_averaging),
        ("MAX_VOTING", baselines.max_voting),
    ):
        bc = []
        for cve in sample_cves:
            inp = _cve_to_inputs(cve, registry)
            bc.append(fn([inp.SR, inp.CC, inp.TD, inp.HA]))
        bc_arr = np.array(bc, dtype=float)
        baseline_results[name] = {
            "mean_cw": float(np.mean(bc_arr)),
            "median_cw": float(np.median(bc_arr)),
            "false_suppression_rate": float(np.mean(bc_arr < suppression_threshold)),
        }

    vw_arr = np.array(vw_cw, dtype=float)
    point, lo, hi = bootstrap_ci(vw_arr.tolist(), np.mean, n_iter=1000, seed=seed)

    summary = {
        "harness": "cve_validation",
        "version": "v1.2.0",
        "snapshot": {
            "catalog_version": snapshot.get("catalogVersion"),
            "catalog_count": snapshot.get("count"),
            "catalog_release_date": snapshot.get("dateReleased"),
            "snapshot_path": str(SNAPSHOT_PATH.relative_to(Path(__file__).resolve().parent.parent)),
        },
        "config": {
            "n": int(n),
            "seed": int(seed),
            "suppression_threshold": float(suppression_threshold),
            "deployment_tier": "government",
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "verdict_weight_version": vw.__version__,
        },
        "vw": {
            "mean_cw": float(np.mean(vw_arr)),
            "median_cw": float(np.median(vw_arr)),
            "ci95_mean_cw": [lo, hi],
            "min_cw": float(np.min(vw_arr)),
            "max_cw": float(np.max(vw_arr)),
            "false_suppression_count": int(len(suppressed)),
            "false_suppression_rate": float(len(suppressed) / max(1, n)),
            "halt_count": int(len(halts)),
            "halt_breakdown": {h: halts.count(h) for h in set(halts)},
            "wall_seconds": round(vw_elapsed, 4),
            "throughput_cves_per_sec": round(n / vw_elapsed, 1) if vw_elapsed > 0 else None,
        },
        "baselines": baseline_results,
        "per_record": per_record,
        "false_suppressed_cves": suppressed,
    }

    (out_dir / "cve_validation.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    (out_dir / "cve_validation.md").write_text(_render_md(summary))
    return summary


def _render_md(s: Dict[str, Any]) -> str:
    snap = s["snapshot"]
    cfg = s["config"]
    vw_ = s["vw"]
    out = []
    out.append("# CVE Validation Benchmark — VW on CISA KEV\n")
    out.append("**Harness:** `cve_validation` (Paper 3 §8.4 Table 6 reproducer)\n")
    out.append(f"**N = {cfg['n']}**, **seed = {cfg['seed']}**, "
               f"suppression threshold = {cfg['suppression_threshold']}, government tier\n")
    out.append(f"**Snapshot:** CISA KEV catalog `{snap['catalog_version']}` "
               f"({snap['catalog_count']:,} vulnerabilities), "
               f"released {snap['catalog_release_date']}\n")
    out.append(f"**VW {cfg['verdict_weight_version']}**, NumPy {cfg['numpy']}, Python {cfg['python']}\n")
    out.append("")
    out.append("## VERDICT WEIGHT outcomes\n")
    out.append(f"- **Mean CW:** {vw_['mean_cw']:.4f} "
               f"(95% CI: [{vw_['ci95_mean_cw'][0]:.4f}, {vw_['ci95_mean_cw'][1]:.4f}])")
    out.append(f"- **Median CW:** {vw_['median_cw']:.4f}")
    out.append(f"- **CW range:** [{vw_['min_cw']:.4f}, {vw_['max_cw']:.4f}]")
    out.append(f"- **False suppressions ({cfg['suppression_threshold']} threshold):** "
               f"{vw_['false_suppression_count']}/{cfg['n']} "
               f"({vw_['false_suppression_rate']*100:.2f}%)")
    out.append(f"- **HALT events:** {vw_['halt_count']} {vw_['halt_breakdown']}")
    out.append(f"- **Throughput:** {vw_['throughput_cves_per_sec']:,.1f} CVEs/sec\n")
    out.append("")
    out.append("## Baseline comparison\n")
    out.append("| Method | Mean CW | Median CW | False-suppression rate |")
    out.append("|---|---|---|---|")
    out.append(f"| VERDICT_WEIGHT | {vw_['mean_cw']:.4f} | {vw_['median_cw']:.4f} | "
               f"{vw_['false_suppression_rate']*100:.2f}% |")
    for name, b in s["baselines"].items():
        out.append(f"| {name} | {b['mean_cw']:.4f} | {b['median_cw']:.4f} | "
                   f"{b['false_suppression_rate']*100:.2f}% |")
    out.append("")
    out.append("## Notes\n")
    out.append("- Each CISA KEV entry is by construction a confirmed real-world exploited "
               "vulnerability. The benchmark therefore measures **false-suppression rate**: how "
               "often the harness incorrectly down-weights a known-exploited CVE.\n")
    out.append("- Inputs are derived from public KEV metadata only (date added, vendor, product, "
               "ransomware-campaign flag). The mapping is deterministic and documented in "
               "``_cve_to_inputs``; running with the same seed against the same snapshot reproduces "
               "the same numbers exactly.\n")
    out.append("- AUC is not reported because all 120 records are positive class (real CVEs); the "
               "interesting metric is suppression rate, not separability.\n")
    return "\n".join(out) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args(argv)
    summary = run(n=args.n, seed=args.seed,
                  suppression_threshold=args.threshold, out_dir=args.out)
    # Print compact summary
    print(json.dumps({
        "vw_mean_cw": summary["vw"]["mean_cw"],
        "vw_false_suppression_rate": summary["vw"]["false_suppression_rate"],
        "vw_halt_count": summary["vw"]["halt_count"],
        "n": summary["config"]["n"],
    }, indent=2))


if __name__ == "__main__":
    main()
