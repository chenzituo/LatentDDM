"""Aggregate validation-only field-token adaptation and matched global controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


FIELD_CONTRACT = "darcy_field_token_latentddm_core_v1"
GLOBAL_CONTRACTS = frozenset({
    "darcy_domain_extension_adaptation_scaling_v1",
    "darcy_domain_extension_adaptation_scaling_with_point_v1",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-root", type=Path, required=True)
    parser.add_argument("--global-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aggregate(args: argparse.Namespace) -> dict:
    zero_path = args.field_root / "zero_shot_validation.json"
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    if zero.get("contract") != "darcy_field_token_validation_v1":
        raise ValueError("unexpected field-token zero-shot contract")
    global_payload = json.loads(
        args.global_aggregate.read_text(encoding="utf-8")
    )
    if global_payload.get("contract") not in GLOBAL_CONTRACTS:
        raise ValueError("unexpected global adaptation contract")

    rows: list[dict] = []
    for budget_dir in sorted(
        args.field_root.glob("budget*"), key=lambda path: int(path.name[6:]),
    ):
        budget = int(budget_dir.name[6:])
        seed_rows = []
        for seed_dir in sorted(budget_dir.glob("seed*")):
            summary_path = seed_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("contract") != FIELD_CONTRACT:
                raise ValueError(f"unexpected contract in {summary_path}")
            seed_rows.append({
                "seed": int(seed_dir.name[4:]),
                "relative_l2": float(summary["best_validation_relative_l2"]),
                "trainable_parameters": int(summary["trainable_parameter_count"]),
                "wall_seconds": float(summary["wall_seconds"]),
                "source": str(summary_path.resolve()),
            })
        if len(seed_rows) < 2:
            raise ValueError(f"budget {budget} needs at least two seeds")
        errors = [row["relative_l2"] for row in seed_rows]
        parameters = {row["trainable_parameters"] for row in seed_rows}
        if len(parameters) != 1:
            raise ValueError(f"budget {budget} changed trainable capacity")
        rows.append({
            "method": "LatentDDM field-token composer",
            "budget": budget,
            "seeds": [row["seed"] for row in seed_rows],
            "source_zero_shot_relative_l2": float(
                zero["metrics"]["relative_l2"]
            ),
            "adapted_relative_l2_mean": statistics.fmean(errors),
            "adapted_relative_l2_sample_std": statistics.stdev(errors),
            "trainable_parameters": parameters.pop(),
            "training_wall_seconds_mean": statistics.fmean(
                row["wall_seconds"] for row in seed_rows
            ),
            "seed_rows": seed_rows,
        })

    budgets = {row["budget"] for row in rows}
    global_rows = [
        row for row in global_payload["rows"]
        if row["method"] in {
            "Transolver full fine-tune",
            "Patch Transformer full fine-tune",
            "Point Transformer full fine-tune",
        } and int(row["budget"]) in budgets
    ]
    for row in global_rows:
        row["budget"] = int(row["budget"])
    return {
        "contract": "darcy_field_token_domain_extension_comparison_v1",
        "split": "validation",
        "protected_test_cases_opened": 0,
        "selection": "best checkpoint selected on the declared validation split",
        "field_zero_shot": zero,
        "rows": sorted(
            rows + global_rows,
            key=lambda row: (int(row["budget"]), str(row["method"])),
        ),
        "sources": {
            "field_zero_shot": str(zero_path.resolve()),
            "global_aggregate": str(args.global_aggregate.resolve()),
        },
    }


def main() -> None:
    args = parse_args()
    payload = aggregate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["rows"], indent=2))


if __name__ == "__main__":
    main()
