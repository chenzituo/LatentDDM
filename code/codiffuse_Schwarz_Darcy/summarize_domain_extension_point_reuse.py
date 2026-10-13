"""Append matched Point full-model reuse to the Darcy adaptation study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from .summarize_domain_extension_adaptation import _global_error, _read


BUDGETS = (4, 16, 64)
SEEDS = (20260810, 20260811, 20260812)
LABEL = "Point Transformer full fine-tune"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-aggregate", type=Path, required=True)
    parser.add_argument("--point-root", type=Path, required=True)
    parser.add_argument("--point-zero-shot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = _read(args.base_aggregate)
    if base.get("contract") != "darcy_domain_extension_adaptation_scaling_v1":
        raise ValueError("unexpected base adaptation contract")
    source_error = _global_error(args.point_zero_shot)
    rows = list(base["rows"])
    per_run = list(base["per_run"])
    for budget in BUDGETS:
        run_rows = []
        for seed in SEEDS:
            run = args.point_root / f"budget{budget}" / f"point_seed{seed}"
            training = _read(run / "summary.json")
            if training.get("contract") != "darcy_global_masked_anchor_operator_v1":
                raise ValueError(f"wrong Point checkpoint contract in {run}")
            if int(training["train_cases"]) != budget:
                raise ValueError(f"nonmatching train-case count in {run}")
            if training.get("adaptation_mode") != "full":
                raise ValueError(f"Point was not fully fine-tuned in {run}")
            record = {
                "method": LABEL,
                "budget": budget,
                "seed": seed,
                "adapted_relative_l2": _global_error(
                    run / "validation" / "summary.json"
                ),
                "trainable_parameters": int(training["trainable_parameter_count"]),
                "training_wall_seconds": float(training["wall_seconds"]),
                "peak_training_gpu_bytes": int(training["peak_gpu_bytes"]),
                "optimizer_state_bytes": int(training["optimizer_state_bytes"]),
                "source_checkpoint": str(training["initial_checkpoint"]),
            }
            run_rows.append(record)
            per_run.append(record)
        errors = [row["adapted_relative_l2"] for row in run_rows]
        mean_error = statistics.fmean(errors)
        rows.append({
            "method": LABEL,
            "budget": budget,
            "seeds": list(SEEDS),
            "source_zero_shot_relative_l2": source_error,
            "adapted_relative_l2_mean": mean_error,
            "adapted_relative_l2_sample_std": statistics.stdev(errors),
            "relative_error_change_from_zero_shot": (
                mean_error - source_error
            ) / source_error,
            "trainable_parameters": run_rows[0]["trainable_parameters"],
            "training_wall_seconds_mean": statistics.fmean(
                row["training_wall_seconds"] for row in run_rows
            ),
            "peak_training_gpu_bytes_mean": statistics.fmean(
                row["peak_training_gpu_bytes"] for row in run_rows
            ),
            "optimizer_state_bytes": run_rows[0]["optimizer_state_bytes"],
        })
    rows.sort(key=lambda row: (int(row["budget"]), str(row["method"])))
    for budget in BUDGETS:
        selected = [row for row in rows if int(row["budget"]) == budget]
        best = min(float(row["adapted_relative_l2_mean"]) for row in selected)
        for row in selected:
            row["error_over_budget_best"] = (
                float(row["adapted_relative_l2_mean"]) / best
            )
    result = {
        **base,
        "contract": "darcy_domain_extension_adaptation_scaling_with_point_v1",
        "status": "three_seed_validation_study_with_point_full_reuse",
        "rows": rows,
        "per_run": per_run,
        "point_zero_shot_source": str(args.point_zero_shot.resolve()),
        "base_aggregate": str(args.base_aggregate.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([row for row in rows if row["method"] == LABEL], indent=2))


if __name__ == "__main__":
    main()

