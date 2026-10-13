"""Aggregate matched Transolver seeds using validation-only artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = (20260805, 20260806, 20260807)
    records = []
    expected_spec = None
    for seed in seeds:
        run = args.root / f"global_transolver_matched_pretraining_seed{seed}" / "full_domain_finetune"
        checkpoint_path = run / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        evaluation = json.loads(
            (run / "validation_only_face_v2_60" / "summary.json").read_text()
        )
        if evaluation["groups"] != ["validation"]:
            raise ValueError("validation-only audit encountered another split")
        spec = checkpoint["model_spec"]
        if expected_spec is None: expected_spec = spec
        if spec != expected_spec: raise ValueError("global model specifications differ")
        count = sum(value.numel() for value in checkpoint["model"].values())
        if count != 3_528_193: raise ValueError("global parameter count differs")
        per_sweep = {
            str(sweep): evaluation["summary"][f"validation/sweep_{sweep:03d}"]
            for sweep in range(1, 5)
        }
        records.append({
            "seed": seed, "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_sha256": sha256(checkpoint_path),
            "parameter_count": count,
            "training_data_schema": checkpoint["training_data_schema"],
            "training_data_role": checkpoint["training_data_role"],
            "statistics_checkpoint": checkpoint["statistics_checkpoint"],
            "selection_metric": "validation final_relative_l2 + 0.1 residual_rms",
            "selection_used_test": False, "evaluation_groups": ["validation"],
            "per_sweep": per_sweep,
        })
    aggregate = {}
    for sweep in range(1, 5):
        aggregate[str(sweep)] = {}
        for metric in (
            "relative_l2", "energy_relative_error", "residual_rms",
            "wall_seconds", "global_nfe", "processed_points",
        ):
            values = np.asarray([
                row["per_sweep"][str(sweep)][metric] for row in records
            ])
            aggregate[str(sweep)][metric + "_mean"] = float(values.mean())
            aggregate[str(sweep)][metric + "_std_population"] = float(values.std())
    result = {
        "contract": "darcy_global_transolver_validation_only_audit_v1",
        "selection_split": "validation",
        "evaluation_split": "validation_first_60",
        "checkpoint_selection_data_reused_for_evaluation": True,
        "interpretation": "checkpoint-selection diagnostic, not an unbiased test estimate",
        "protected_A_D_opened_by_this_audit": False,
        "seed_aggregation_rule": "mean and population std over all three predeclared seeds",
        "model_spec": expected_spec,
        "parameter_count": 3_528_193,
        "face_local_parameter_count": 3_957_666,
        "capacity_ratio_global_over_face_local": 3_528_193 / 3_957_666,
        "proposed_total_parameter_count_reference": 3_957_666 + 58_447 + 97_994,
        "capacity_note": "matched to the local endpoint, not the full local+chart+communicator stack",
        "validation_manifest_note": (
            "v1 and face-v2 validation physics arrays are byte-identical; family string storage differs"
        ),
        "records": records, "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
