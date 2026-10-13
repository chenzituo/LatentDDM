"""Aggregate all three predeclared face-v2 global-control seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-dir", default="validation_only_60")
    parser.add_argument("--expected-cases", type=int, default=60)
    args = parser.parse_args()
    if args.expected_cases <= 0:
        raise ValueError("--expected-cases must be positive")
    rows = []
    data_paths: set[Path] = set()
    model_spec = None
    for seed in (20260805, 20260806, 20260807):
        run = args.root / f"global_transolver_face_v2_full_only_seed{seed}"
        checkpoint = torch.load(
            run / "checkpoint.pt", map_location="cpu", weights_only=False,
        )
        evaluation = json.loads(
            (run / args.evaluation_dir / "summary.json").read_text()
        )
        if checkpoint["training_data_schema"] != \
                "darcy_full_domain_cartesian_v2_face_dirichlet":
            raise ValueError("non-face-v2 checkpoint in aggregate")
        if checkpoint["initial_checkpoint"] is not None or \
                checkpoint["statistics_provenance"] != \
                "computed_from_current_train_split":
            raise ValueError("face-v2 control provenance differs")
        if evaluation["groups"] != ["validation"]:
            raise ValueError("protected/non-validation group encountered")
        if evaluation.get("configuration", {}).get("max_cases") \
                != args.expected_cases:
            raise ValueError(
                "face-v2 control case count differs from --expected-cases"
            )
        count = sum(value.numel() for value in checkpoint["model"].values())
        if count != 3_528_193:
            raise ValueError("face-v2 global capacity differs")
        current_spec = checkpoint["model_spec"]
        if model_spec is None:
            model_spec = current_spec
        elif current_spec != model_spec:
            raise ValueError("face-v2 global model specification differs by seed")
        data_path = Path(checkpoint["args"]["data"]).resolve()
        if Path(evaluation["data"]).resolve() != data_path:
            raise ValueError("training/evaluation face-v2 corpus identity differs")
        data_paths.add(data_path)
        rows.append({
            "seed": seed, "selected_epoch": int(checkpoint["epoch"]),
            "checkpoint": str((run / "checkpoint.pt").resolve()),
            "checkpoint_sha256": _sha256(run / "checkpoint.pt"),
            "training_data_schema": checkpoint["training_data_schema"],
            "training_data_role": checkpoint["training_data_role"],
            "statistics_provenance": checkpoint["statistics_provenance"],
            "selection_validation": checkpoint["validation"],
            "statistics": checkpoint["statistics"],
            "per_sweep": {
                str(sweep): evaluation["summary"][f"validation/sweep_{sweep:03d}"]
                for sweep in range(1, 5)
            },
        })
    if len(data_paths) != 1:
        raise ValueError("face-v2 global seeds used different corpora")
    data_path = next(iter(data_paths))
    aggregate = {}
    for sweep in range(1, 5):
        aggregate[str(sweep)] = {}
        for metric in (
            "relative_l2", "energy_relative_error", "residual_rms",
            "wall_seconds", "global_nfe", "processed_points",
        ):
            values = np.asarray([row["per_sweep"][str(sweep)][metric] for row in rows])
            aggregate[str(sweep)][metric + "_mean"] = float(values.mean())
            aggregate[str(sweep)][metric + "_std_population"] = float(values.std())
    result = {
        "contract": "darcy_global_transolver_face_v2_full_only_v2",
        "training_split": "face-v2 train", "selection_split": "face-v2 validation",
        "evaluation_split": f"face-v2 validation all {args.expected_cases}",
        "evaluation_cases": args.expected_cases,
        "checkpoint_selection_data_reused_for_evaluation": True,
        "interpretation": (
            "checkpoint-selection diagnostic, not an unbiased held-out estimate"
        ),
        "protected_A_D_or_test_opened": False,
        "full_domain_only_random_initialization": True,
        "seed_selection_performed": False,
        "seeds": [20260805, 20260806, 20260807],
        "model_spec": model_spec,
        "parameter_count": 3_528_193,
        "face_local_parameter_count": 3_957_666,
        "capacity_ratio_global_over_face_local": 3_528_193 / 3_957_666,
        "proposed_total_parameter_count_reference": 4_114_107,
        "capacity_note": (
            "matched to the face-local endpoint, not the complete local+chart+"
            "communicator stack"
        ),
        "data": str(data_path),
        "data_sha256": _sha256(data_path),
        "training_cases": 4000, "validation_cases": 250,
        "samples_per_epoch": 4096, "epochs": 40,
        "records": rows, "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
