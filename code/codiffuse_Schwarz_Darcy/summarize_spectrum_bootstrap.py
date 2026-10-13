"""Paired bootstrap summary for the target-free Darcy port-spectrum audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def interval(ratio: np.ndarray, rng: np.random.Generator,
             draws: int) -> dict[str, float | int]:
    if np.any(ratio <= 0.0) or ratio.ndim != 1:
        raise ValueError("bootstrap ratios must be positive one-dimensional data")
    log_ratio = np.log(ratio)
    indices = rng.integers(0, ratio.size, size=(draws, ratio.size))
    sampled = log_ratio[indices].mean(1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return {
        "paired_cases": int(ratio.size),
        "geometric_mean_ratio": float(np.exp(log_ratio.mean())),
        "geometric_mean_ratio_ci95_low": float(np.exp(low)),
        "geometric_mean_ratio_ci95_high": float(np.exp(high)),
        "median_ratio": float(np.median(ratio)),
        "fraction_ratio_at_most_one": float(np.mean(ratio <= 1.0)),
        "bootstrap_draws": draws,
    }


def summarize(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.draws <= 0:
        raise ValueError("bootstrap draw count must be positive")
    paired: dict[int, dict[str, dict[str, float]]] = {}
    with args.cases.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            paired.setdefault(int(row["case_index"]), {})[row["analysis"]] = {
                key: float(value) for key, value in row.items()
                if key not in ("case_index", "analysis")
            }
    ordered = [paired[key] for key in sorted(paired)]
    if any(set(row) != {"value", "value_flux"} for row in ordered):
        raise ValueError("every case must contain paired value/value_flux rows")
    condition = np.asarray([
        row["value_flux"]["supported_condition_number"]
        / row["value"]["supported_condition_number"] for row in ordered
    ])
    minimum = np.asarray([
        row["value_flux"]["minimum_supported_singular_value"]
        / row["value"]["minimum_supported_singular_value"] for row in ordered
    ])
    rng = np.random.default_rng(args.seed)
    result = {
        "contract": "darcy_target_free_cauchy_port_spectrum_paired_bootstrap_v1",
        "selection_split": "validation",
        "protected_test_opened": False,
        "target_values_used": False,
        "condition_ratio_value_flux_over_value": interval(
            condition, rng, args.draws,
        ),
        "minimum_singular_ratio_value_flux_over_value": interval(
            minimum, rng, args.draws,
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(parse_args()), indent=2))
