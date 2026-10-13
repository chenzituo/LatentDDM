"""Evaluate one fixed-cadence global Airfoil model on L5 H20 rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .compare_head_to_head_v3 import (
    abupt_step,
    global_method_name,
    load_abupt,
    load_global,
    model_input_batch,
    transolver_step,
)
from .dataset import FieldStatistics
from .evaluate_airfoil_sni import metrics
from .evaluate_global_validation_protocol import rollout_force_summary, rollout_starts
from .rollout_absolute_robin_sni import (
    physical_relative_l2,
    vorticity_relative_l2,
)
from .trajectory_corpus import TrajectoryCorpus


POINT_CONTRACT = "global_two_state_abupt_fixed_cadence_v4"
SLICE_CONTRACT = "global_two_state_transolver_l5_fixed_cadence_v2"
PATCH_CONTRACT = "global_two_state_patch_transformer_l5_fixed_cadence_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def method_from_checkpoint(payload: dict[str, object]) -> str:
    contract = payload.get("training_contract")
    if contract == POINT_CONTRACT:
        return "Point"
    if contract == SLICE_CONTRACT and global_method_name(payload) == "transolver":
        return "Slice"
    if contract == PATCH_CONTRACT and global_method_name(payload) == "patch_transformer":
        patch = tuple(payload["model_args"].get("patch_size", ()))
        if patch != (32, 32):
            raise ValueError(f"Patch baseline uses {patch}, not 32x32")
        return "Patch"
    raise ValueError(f"unsupported fixed-cadence checkpoint contract: {contract}")


def evaluation_names(
    checkpoint_split: dict[str, list[str]],
    split_manifest: Path | None,
    split_part: str,
) -> tuple[list[str], str, str | None]:
    if split_part not in ("validation", "test"):
        raise ValueError("split part must be validation or test")
    if split_manifest is None:
        source = checkpoint_split
        label = f"checkpoint_{split_part}"
        digest = None
    else:
        source = json.loads(split_manifest.read_text(encoding="utf-8"))
        label = f"manifest_{split_part}"
        digest = sha256(split_manifest)
    names = [str(value) for value in source.get(split_part, ())]
    if not names:
        raise ValueError(f"evaluation {split_part} is empty")
    return names, label, digest


def numeric_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    excluded = {"trajectory", "item", "start", "step"}
    keys = sorted({
        key for row in rows for key, value in row.items()
        if key not in excluded and isinstance(value, (int, float))
    })
    result: dict[str, object] = {"examples": len(rows)}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_q10"] = float(np.quantile(values, 0.10))
        result[f"{key}_q90"] = float(np.quantile(values, 0.90))
    return result


def fluid_field_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sdf: torch.Tensor,
    dx: float,
) -> dict[str, float]:
    """Use the exact fluid-mask field convention of the local evaluators."""
    fluid = (sdf.to(prediction) > 0.0).to(prediction)
    return {
        "field_relative_l2": physical_relative_l2(
            prediction, target, fluid,
        ),
        "vorticity_relative_l2": vorticity_relative_l2(
            prediction, target, fluid, float(dx),
        ),
    }


def comparable_metrics(prediction, target, batch, dataset) -> dict[str, float]:
    """Retain common load diagnostics while replacing legacy all-grid errors."""
    result = metrics(prediction, target, batch, dataset)
    result.update(fluid_field_metrics(
        prediction, target, batch["sdf_np1"], float(dataset.dx),
    ))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split-part", choices=("validation", "test"), default="validation")
    parser.add_argument("--one-step-samples-per-trajectory", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--starts-per-trajectory", type=int, default=3)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(
        args.one_step_samples_per_trajectory,
        args.horizon,
        args.starts_per_trajectory,
    ) < 1:
        raise ValueError("evaluation counts must be positive")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    method = method_from_checkpoint(payload)
    names, split_label, manifest_hash = evaluation_names(
        payload["split"], args.split_manifest, args.split_part,
    )
    device = torch.device("cuda")
    statistics = FieldStatistics.from_json(payload["statistics"])
    if method == "Point":
        model, context_stride = load_abupt(payload, device)
    else:
        model = load_global(payload, device)
        context_stride = None
    model.requires_grad_(False).eval()
    corpus = TrajectoryCorpus(args.data_root, names, history_steps=2)

    def predict(sample, dataset, previous, current):
        batch = {key: value[None] for key, value in sample.items()}
        batch["q_nm1"] = previous
        batch["q_n"] = current
        inference = model_input_batch(batch)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        if method == "Point":
            prediction = abupt_step(
                model, inference, dataset, statistics,
                int(context_stride), device,
            )
        else:
            prediction = transolver_step(
                model, inference, dataset, statistics, device,
            )
        torch.cuda.synchronize(device)
        return prediction, time.perf_counter() - started, batch

    one_step_rows: list[dict[str, object]] = []
    rollout_rows: list[dict[str, object]] = []
    try:
        warm = corpus.datasets[0][0]
        warm_previous = warm["q_nm1"][None].to(device)
        warm_current = warm["q_n"][None].to(device)
        predict(warm, corpus.datasets[0], warm_previous, warm_current)
        for trajectory, dataset in enumerate(corpus.datasets):
            trajectory_name = Path(names[trajectory]).stem
            one_step_indices = np.linspace(
                0, len(dataset) - 1,
                min(args.one_step_samples_per_trajectory, len(dataset)),
                dtype=int,
            )
            for item in one_step_indices:
                sample = dataset[int(item)]
                previous = sample["q_nm1"][None].to(device)
                current = sample["q_n"][None].to(device)
                target = sample["q_np1"][None].to(device)
                prediction, seconds, batch = predict(
                    sample, dataset, previous, current,
                )
                one_step_rows.append({
                    "trajectory": trajectory_name,
                    "item": int(item),
                    **comparable_metrics(prediction, target, batch, dataset),
                    "seconds": float(seconds),
                })
            for start in rollout_starts(
                len(dataset), args.horizon, args.starts_per_trajectory,
            ):
                first = dataset[start]
                previous = first["q_nm1"][None].to(device)
                current = first["q_n"][None].to(device)
                for offset in range(args.horizon):
                    sample = dataset[start + offset]
                    target = sample["q_np1"][None].to(device)
                    prediction, seconds, batch = predict(
                        sample, dataset, previous, current,
                    )
                    rollout_rows.append({
                        "trajectory": trajectory_name,
                        "start": int(start),
                        "step": offset + 1,
                        **comparable_metrics(prediction, target, batch, dataset),
                        "seconds": float(seconds),
                    })
                    previous, current = current, prediction
    finally:
        corpus.close()
    final = [row for row in rollout_rows if int(row["step"]) == args.horizon]
    by_step = {
        f"step_{step:02d}": numeric_summary([
            row for row in rollout_rows if int(row["step"]) == step
        ])
        for step in range(1, args.horizon + 1)
    }
    all_rollout = numeric_summary(rollout_rows)
    all_rollout.update(rollout_force_summary(rollout_rows, args.horizon))
    result = {
        "contract": "airfoil_fixed_cadence_global_rollout_v1",
        "method": method,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "training_contract": payload["training_contract"],
        "initialization": payload.get("initialization"),
        "initialization_sha256": payload.get("initialization_sha256"),
        "full_model_adaptation": bool(payload.get("full_model_adaptation", False)),
        "split": split_label,
        "split_part": args.split_part,
        "split_manifest": (
            str(args.split_manifest.resolve())
            if args.split_manifest is not None else None
        ),
        "split_manifest_sha256": manifest_hash,
        "evaluation_trajectories": names,
        "protected_test_trajectories_opened": (
            len(names) if args.split_part == "test" else 0
        ),
        "spatial_level": "L5",
        "cadence": 0.025,
        "model_owned_rollout": True,
        "reference_used_by_inference": False,
        "history_update": "previous <- current; current <- prediction",
        "horizon": args.horizon,
        "starts_per_trajectory": args.starts_per_trajectory,
        "timing_warmup_steps_per_method": 1,
        "timing_scope": (
            "CUDA-synchronized model inference; data loading, condition "
            "construction, diagnostics, and force reconstruction excluded"
        ),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "one_step": numeric_summary(one_step_rows),
        "rollout_all_steps": all_rollout,
        "rollout_final_step": numeric_summary(final),
        "rollout_by_step": by_step,
        "rows": {"one_step": one_step_rows, "rollout": rollout_rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = evaluate(parse_args())
    print(json.dumps({
        "method": result["method"],
        "one_step": result["one_step"],
        "rollout_final_step": result["rollout_final_step"],
    }), flush=True)


if __name__ == "__main__":
    main()
