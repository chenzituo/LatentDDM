"""Select validation learning rates for matched low-rank global controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .summarize_domain_extension_adaptation import _global_error, _read


METHODS = {
    "Transolver low-rank": (
        "transolver_lora_r4",
        4,
        ("2e-5", "6e-5", "1e-4", "2e-4", "6e-4", "2e-3"),
    ),
    "Patch Transformer low-rank": (
        "patch_lora_r4",
        4,
        ("2e-4", "6e-4", "2e-3", "4e-3", "6e-3", "1e-2"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    domain_root = args.root.parent
    source_errors = {
        "Transolver low-rank": _global_error(
            domain_root / "transolver_zero_shot" / "summary.json"
        ),
        "Patch Transformer low-rank": _global_error(
            domain_root / "global_vit_zero_shot" / "summary.json"
        ),
    }
    rows = []
    selected = {}
    for label, (stem, rank, learning_rates) in METHODS.items():
        candidates = []
        for learning_rate in learning_rates:
            run = args.root / f"{stem}_lr{learning_rate}"
            training = _read(run / "summary.json")
            if training.get("adaptation_mode") != "low_rank":
                raise ValueError(f"{run} is not a low-rank adaptation run")
            if int(training.get("low_rank", {}).get("rank", -1)) != rank:
                raise ValueError(f"rank mismatch in {run}")
            if float(training.get("low_rank_zero_delta_max_abs", -1.0)) != 0.0:
                raise ValueError(f"non-centered initialization in {run}")
            error = _global_error(run / "validation" / "summary.json")
            source_error = source_errors[label]
            row = {
                "method": label,
                "rank": rank,
                "learning_rate": float(learning_rate),
                "validation_relative_l2": error,
                "source_zero_shot_relative_l2": source_error,
                "relative_error_change_from_zero_shot": (
                    error - source_error
                ) / source_error,
                "trainable_parameters": int(training["trainable_parameter_count"]),
                "training_wall_seconds": float(training["wall_seconds"]),
                "peak_training_gpu_bytes": int(training["peak_gpu_bytes"]),
                "optimizer_state_bytes": int(training["optimizer_state_bytes"]),
            }
            rows.append(row)
            candidates.append(row)
        selected[label] = min(candidates, key=lambda item: item["validation_relative_l2"])
    result = {
        "contract": "darcy_domain_extension_peft_pilot16_v1",
        "status": "validation_only_hyperparameter_selection",
        "train_cases": 16,
        "validation_cases": 128,
        "protected_test_accessed": False,
        "source_zero_shot_relative_l2": source_errors,
        "rows": rows,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
