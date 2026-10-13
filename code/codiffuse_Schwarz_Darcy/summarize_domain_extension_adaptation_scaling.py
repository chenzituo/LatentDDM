"""Aggregate the three-seed Darcy domain-extension adaptation study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from .summarize_domain_extension_adaptation import (
    _composer_error,
    _global_error,
    _read,
)


BUDGETS = (4, 16, 64)
SEEDS = (20260810, 20260811, 20260812)
METHODS = (
    ("LatentDDM composer-only", "composer", "validation_ladder", _composer_error),
    ("Transolver full fine-tune", "transolver", "validation", _global_error),
    ("Patch Transformer full fine-tune", "patch", "validation", _global_error),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    domain_root = args.root.parent
    source_errors = {
        "LatentDDM composer-only": _composer_error(
            domain_root / "latentddm_value_zero_shot" / "summary.json"
        ),
        "Transolver full fine-tune": _global_error(
            domain_root / "transolver_zero_shot" / "summary.json"
        ),
        "Patch Transformer full fine-tune": _global_error(
            domain_root / "global_vit_zero_shot" / "summary.json"
        ),
    }
    rows = []
    per_run = []
    for budget in BUDGETS:
        for label, stem, validation_dir, error_reader in METHODS:
            method_runs = []
            for seed in SEEDS:
                run = args.root / f"budget{budget}" / f"{stem}_seed{seed}"
                training = _read(run / "summary.json")
                if int(training["train_cases"]) != budget:
                    raise ValueError(f"nonmatching train-case count in {run}")
                adapted_error = error_reader(run / validation_dir / "summary.json")
                trainable = int(
                    training["trainable_parameter_count"]
                    if "trainable_parameter_count" in training
                    else training["parameter_count"]
                )
                record = {
                    "method": label,
                    "budget": budget,
                    "seed": seed,
                    "adapted_relative_l2": adapted_error,
                    "trainable_parameters": trainable,
                    "training_wall_seconds": float(training["wall_seconds"]),
                    "peak_training_gpu_bytes": int(training["peak_gpu_bytes"]),
                    "optimizer_state_bytes": int(training["optimizer_state_bytes"]),
                }
                method_runs.append(record)
                per_run.append(record)
            errors = [item["adapted_relative_l2"] for item in method_runs]
            wall = [item["training_wall_seconds"] for item in method_runs]
            peak = [item["peak_training_gpu_bytes"] for item in method_runs]
            source = source_errors[label]
            rows.append({
                "method": label,
                "budget": budget,
                "seeds": list(SEEDS),
                "source_zero_shot_relative_l2": source,
                "adapted_relative_l2_mean": _mean(errors),
                "adapted_relative_l2_sample_std": float(statistics.stdev(errors)),
                "relative_error_change_from_zero_shot": (_mean(errors) - source) / source,
                "trainable_parameters": method_runs[0]["trainable_parameters"],
                "training_wall_seconds_mean": _mean(wall),
                "peak_training_gpu_bytes_mean": _mean(peak),
                "optimizer_state_bytes": method_runs[0]["optimizer_state_bytes"],
            })
    for budget in BUDGETS:
        budget_rows = [row for row in rows if row["budget"] == budget]
        best = min(row["adapted_relative_l2_mean"] for row in budget_rows)
        for row in budget_rows:
            row["error_over_budget_best"] = row["adapted_relative_l2_mean"] / best
    result = {
        "contract": "darcy_domain_extension_adaptation_scaling_v1",
        "status": "three_seed_validation_study",
        "data": str((domain_root.parent.parent / "data" / "darcy_domain_extension_128_v1.h5").resolve()),
        "source_domain": "96x96",
        "target_domain": "128x128",
        "nested_target_label_budgets": list(BUDGETS),
        "validation_cases": 128,
        "protected_test_accessed": False,
        "rows": rows,
        "per_run": per_run,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
