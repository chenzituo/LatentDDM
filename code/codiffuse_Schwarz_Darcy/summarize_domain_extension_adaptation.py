"""Summarize the protected-test-closed Darcy adaptation decision pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _global_error(path: Path) -> float:
    payload = _read(path)
    if payload.get("contract") != "darcy_global_full_domain_evaluation_v3":
        raise ValueError(f"unexpected global evaluation contract in {path}")
    return float(payload["summary"]["validation/sweep_001"]["relative_l2"])


def _composer_error(path: Path) -> float:
    payload = _read(path)
    if payload.get("contract") != "darcy_residual_port_validation_ladder_v1":
        raise ValueError(f"unexpected composer evaluation contract in {path}")
    if payload.get("test_subsets_opened") is not False:
        raise ValueError(f"protected test status is not closed in {path}")
    return float(payload["summary"]["predicted/radius_global"]["relative_l2"])


def _row(
    name: str,
    source_error: float,
    adapted_error: float,
    training: dict,
) -> dict:
    return {
        "method": name,
        "source_zero_shot_relative_l2": source_error,
        "adapted_relative_l2": adapted_error,
        "relative_error_change": (adapted_error - source_error) / source_error,
        "trainable_parameters": int(
            training["trainable_parameter_count"]
            if "trainable_parameter_count" in training
            else training["parameter_count"]
        ),
        "training_wall_seconds": float(training["wall_seconds"]),
        "peak_training_gpu_bytes": int(training["peak_gpu_bytes"]),
        "optimizer_state_bytes": int(training["optimizer_state_bytes"]),
        "target_train_cases": int(training["train_cases"]),
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
    root = args.root
    source_root = root.parent
    composer_train = _read(root / "composer_seed20260810" / "summary.json")
    transolver_train = _read(root / "transolver_seed20260810" / "summary.json")
    patch_train = _read(root / "patch_seed20260810" / "summary.json")
    rows = [
        _row(
            "LatentDDM composer-only",
            _composer_error(source_root / "latentddm_value_zero_shot" / "summary.json"),
            _composer_error(
                root / "composer_seed20260810" / "validation_ladder" / "summary.json"
            ),
            composer_train,
        ),
        _row(
            "Transolver full fine-tune",
            _global_error(source_root / "transolver_zero_shot" / "summary.json"),
            _global_error(
                root / "transolver_seed20260810" / "validation" / "summary.json"
            ),
            transolver_train,
        ),
        _row(
            "Patch Transformer full fine-tune",
            _global_error(source_root / "global_vit_zero_shot" / "summary.json"),
            _global_error(
                root / "patch_seed20260810" / "validation" / "summary.json"
            ),
            patch_train,
        ),
    ]
    composer = rows[0]
    for row in rows:
        row["error_over_composer"] = (
            row["adapted_relative_l2"] / composer["adapted_relative_l2"]
        )
        row["trainable_parameters_over_composer"] = (
            row["trainable_parameters"] / composer["trainable_parameters"]
        )
        row["training_wall_over_composer"] = (
            row["training_wall_seconds"] / composer["training_wall_seconds"]
        )
        row["peak_gpu_memory_over_composer"] = (
            row["peak_training_gpu_bytes"] / composer["peak_training_gpu_bytes"]
        )
    result = {
        "contract": "darcy_domain_extension_adaptation_decision_pilot_v1",
        "status": "one_seed_resource_decision_only",
        "promotion_requirement": "nested label budgets and three seeds",
        "data": str((source_root.parent.parent / "data" / "darcy_domain_extension_128_v1.h5").resolve()),
        "source_domain": "96x96",
        "target_domain": "128x128",
        "target_train_cases": 16,
        "validation_cases": 128,
        "protected_test_accessed": False,
        "seed": 20260810,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
