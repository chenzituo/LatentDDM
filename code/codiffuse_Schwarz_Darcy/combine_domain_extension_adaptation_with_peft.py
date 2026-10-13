"""Combine promoted low-rank controls with the Darcy adaptation study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from .summarize_domain_extension_adaptation import _global_error, _read


BUDGETS = (4, 16, 64)
SEEDS = (20260810, 20260811, 20260812)
METHODS = (
    ("Transolver low-rank adapter", "transolver_lora", 4),
    ("Patch Transformer low-rank adapter", "patch_lora", 4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--peft-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = _read(args.base)
    if base.get("contract") != "darcy_domain_extension_adaptation_scaling_v1":
        raise ValueError("base artifact uses another contract")
    source_by_family = {
        "transolver_lora": next(
            row["source_zero_shot_relative_l2"]
            for row in base["rows"]
            if row["method"] == "Transolver full fine-tune"
        ),
        "patch_lora": next(
            row["source_zero_shot_relative_l2"]
            for row in base["rows"]
            if row["method"] == "Patch Transformer full fine-tune"
        ),
    }

    rows = list(base["rows"])
    per_run = list(base["per_run"])
    for budget in BUDGETS:
        for label, stem, rank in METHODS:
            runs = []
            for seed in SEEDS:
                run = args.peft_root / f"budget{budget}" / f"{stem}_seed{seed}"
                training = _read(run / "summary.json")
                if training.get("adaptation_mode") != "low_rank":
                    raise ValueError(f"{run} is not a low-rank adaptation")
                if int(training.get("low_rank", {}).get("rank", -1)) != rank:
                    raise ValueError(f"rank mismatch in {run}")
                if float(training.get("low_rank_zero_delta_max_abs", -1.0)) != 0.0:
                    raise ValueError(f"non-centered initialization in {run}")
                if int(training["train_cases"]) != budget:
                    raise ValueError(f"label-budget mismatch in {run}")
                record = {
                    "method": label,
                    "budget": budget,
                    "seed": seed,
                    "adapted_relative_l2": _global_error(
                        run / "validation" / "summary.json"
                    ),
                    "trainable_parameters": int(training["trainable_parameter_count"]),
                    "training_wall_seconds": float(training["wall_seconds"]),
                    "peak_training_gpu_bytes": int(training["peak_gpu_bytes"]),
                    "optimizer_state_bytes": int(training["optimizer_state_bytes"]),
                    "rank": rank,
                }
                runs.append(record)
                per_run.append(record)
            errors = [row["adapted_relative_l2"] for row in runs]
            walls = [row["training_wall_seconds"] for row in runs]
            peaks = [row["peak_training_gpu_bytes"] for row in runs]
            source = float(source_by_family[stem])
            mean_error = float(statistics.mean(errors))
            rows.append({
                "method": label,
                "budget": budget,
                "seeds": list(SEEDS),
                "source_zero_shot_relative_l2": source,
                "adapted_relative_l2_mean": mean_error,
                "adapted_relative_l2_sample_std": float(statistics.stdev(errors)),
                "relative_error_change_from_zero_shot": (mean_error - source) / source,
                "trainable_parameters": runs[0]["trainable_parameters"],
                "training_wall_seconds_mean": float(statistics.mean(walls)),
                "peak_training_gpu_bytes_mean": float(statistics.mean(peaks)),
                "optimizer_state_bytes": runs[0]["optimizer_state_bytes"],
                "rank": rank,
            })

    for budget in BUDGETS:
        budget_rows = [row for row in rows if int(row["budget"]) == budget]
        best = min(float(row["adapted_relative_l2_mean"]) for row in budget_rows)
        for row in budget_rows:
            row["error_over_budget_best"] = float(
                row["adapted_relative_l2_mean"]
            ) / best
    result = {
        **{key: value for key, value in base.items() if key not in {"contract", "rows", "per_run"}},
        "contract": "darcy_domain_extension_adaptation_with_peft_v1",
        "status": "three_seed_validation_study_with_parameter_matched_controls",
        "protected_test_accessed": False,
        "rows": rows,
        "per_run": per_run,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
