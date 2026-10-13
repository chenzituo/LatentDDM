"""Paired bootstrap audit for a frozen Darcy latent-Schur validation run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LOWER_IS_BETTER = (
    "relative_l2",
    "energy_relative_square",
    "residual_relative_square",
    "interface_value_square",
    "interface_flux_square",
    "complete_analysis_cycle_relative_square",
    "real_defect_cycle_relative_square",
    "q_phys_weak_conservation",
)


def _paired_ci(delta: np.ndarray, seed: int, draws: int) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    sample = delta[generator.integers(0, len(delta), size=(draws, len(delta)))]
    means = sample.mean(1)
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "fraction_cases_improved": float(np.mean(delta < 0.0)),
        "probability_mean_improves": float(np.mean(means < 0.0)),
    }


def summarize(args: argparse.Namespace) -> dict[str, object]:
    rows = list(csv.DictReader(args.cases.open(encoding="utf-8")))
    paired: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        paired.setdefault(int(row["case_index"]), {})[row["method"]] = row
    if not paired or any(set(value) != {"local", "latent_schur"} for value in paired.values()):
        raise ValueError("every case must contain exactly local and latent_schur rows")
    result = {}
    for ordinal, metric in enumerate(LOWER_IS_BETTER):
        local = np.asarray([
            float(paired[index]["local"][metric]) for index in sorted(paired)
        ])
        corrected = np.asarray([
            float(paired[index]["latent_schur"][metric]) for index in sorted(paired)
        ])
        audit = _paired_ci(corrected - local, args.seed + 104729 * ordinal, args.draws)
        audit.update({
            "local_mean": float(local.mean()),
            "latent_schur_mean": float(corrected.mean()),
            "relative_mean_improvement": float(
                (local.mean() - corrected.mean()) / max(abs(local.mean()), 1.0e-15)
            ),
            "ci_strictly_improves": audit["ci95_high"] < 0.0,
        })
        result[metric] = audit
    payload = {
        "contract": "darcy_latent_schur_paired_bootstrap_v1",
        "source": str(args.cases.resolve()),
        "cases": len(paired),
        "bootstrap_draws": args.draws,
        "seed": args.seed,
        "metrics": result,
        "all_mean_nonregression": all(
            value["mean_delta"] <= 0.0 for value in result.values()
        ),
        "all_ci_strictly_improve": all(
            value["ci_strictly_improves"] for value in result.values()
        ),
        "test_subsets_opened": False,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(summarize(parse_args()), indent=2))
