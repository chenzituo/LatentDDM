"""Aggregate the three-seed boundary-input LatentDDM extension study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


BUDGETS = (4, 16, 64)
SEEDS = (20260810, 20260811, 20260812)


def tensor_count(value: object) -> int:
    """Count parameters in both flat and nested composer state dictionaries."""
    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, dict):
        return sum(tensor_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_count(item) for item in value)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result: dict[str, object] = {
        "contract": "darcy_boundary_input_domain_extension_summary_v1",
        "method": "LatentDDM boundary-input composer-only adaptation",
        "frozen_local_bvp": True,
        "budgets": {},
    }
    source_hashes: set[str] = set()
    local_hashes: set[str] = set()
    local_means: list[float] = []
    for budget in BUDGETS:
        rows = []
        for seed in SEEDS:
            run = args.root / f"budget{budget}" / f"seed{seed}"
            checkpoint_path = run / "checkpoint_best.pt"
            evaluation_path = run / "validation_128.json"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False,
            )
            evaluation = json.loads(evaluation_path.read_text())
            if checkpoint.get("contract") != "darcy_input_correction_latentddm_v1":
                raise ValueError(f"wrong checkpoint contract: {checkpoint_path}")
            if int(checkpoint["args"]["max_train_cases"]) != budget:
                raise ValueError(f"wrong label budget: {checkpoint_path}")
            if int(checkpoint["args"]["seed"]) != seed:
                raise ValueError(f"wrong seed: {checkpoint_path}")
            if evaluation.get("protected_test_cases_opened") != 0:
                raise ValueError(f"protected cases entered {evaluation_path}")
            source_hashes.add(checkpoint["initial_composer_checkpoint_sha256"])
            local_hashes.add(checkpoint["local_checkpoint_sha256"])
            latent = evaluation["summary"]["latentddm"]
            local = evaluation["summary"]["local_core"]
            local_means.append(float(local["relative_l2_mean"]))
            parameters = tensor_count(checkpoint["composer"])
            rows.append({
                "seed": seed,
                "relative_l2": float(latent["relative_l2_mean"]),
                "energy_relative_error": float(
                    latent["energy_relative_error_mean"],
                ),
                "residual_rms": float(latent["residual_rms_mean"]),
                "training_wall_seconds": float(sum(
                    record["seconds"] for record in checkpoint["history"]
                )),
                "trainable_parameters": int(parameters),
                "checkpoint": str(checkpoint_path.resolve()),
                "evaluation": str(evaluation_path.resolve()),
            })
        values = np.asarray([row["relative_l2"] for row in rows])
        result["budgets"][str(budget)] = {
            "relative_l2_mean": float(values.mean()),
            "relative_l2_sample_std": float(values.std(ddof=1)),
            "training_wall_seconds_mean": float(np.mean([
                row["training_wall_seconds"] for row in rows
            ])),
            "trainable_parameters": rows[0]["trainable_parameters"],
            "seeds": rows,
        }
    if len(source_hashes) != 1 or len(local_hashes) != 1:
        raise ValueError("adaptation runs do not share source composer and local BVP")
    if float(np.ptp(np.asarray(local_means))) > 1.0e-7:
        raise ValueError("Local/Core changed across composer-only adaptation runs")
    result["source_composer_sha256"] = next(iter(source_hashes))
    result["local_bvp_sha256"] = next(iter(local_hashes))
    result["local_core_relative_l2"] = float(np.mean(local_means))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
