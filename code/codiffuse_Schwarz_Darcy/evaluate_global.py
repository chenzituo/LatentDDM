"""Evaluate a one-shot or recurrent full-domain Darcy baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .adapter import DarcyAdapter, DarcyStatistics, H5DarcyDataset
from .evaluate import (
    energy_relative_error,
    initial_state,
    move_batch,
    relative_l2,
    residual_rms,
)
from .global_operator import MaskedTransolverOperator
from .masked_anchor_operator import MaskedAnchorOperator
from .train_global import (
    ANCHOR_CHECKPOINT_CONTRACT,
    CHECKPOINT_CONTRACT,
    VIT_CHECKPOINT_CONTRACT,
    global_step,
)
from codiffuse_Schwarz.global_vit import GlobalPatchTransformer


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path("codiffuse_Schwarz_Darcy/data/darcy_complex_v1.h5"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--groups", nargs="+", default=("A", "B", "C", "D"),
        choices=("validation", "val", "test", "A", "B", "C", "D"),
    )
    parser.add_argument("--sweeps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--initialization", choices=("zero", "mean"), default="zero")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--allow-protected-test", action="store_true",
        help="required acknowledgement before reading a protected test group",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def build_from_checkpoint(
    checkpoint: dict[str, Any],
) -> MaskedTransolverOperator | GlobalPatchTransformer | MaskedAnchorOperator:
    spec = checkpoint["model_spec"]
    if spec.get("architecture") == "global_patch_transformer":
        if spec.get("position_encoding") not in (
            "normalized_linear_fourier_v1", "fixed_tile_fourier_v1",
            "fixed_tile_fourier_locality_v1",
        ):
            raise ValueError("global ViT uses an obsolete position encoding")
        return GlobalPatchTransformer(
            input_channels=int(spec["input_channels"]),
            output_channels=int(spec["output_channels"]),
            dimensions=int(spec["dimensions"]),
            patch_size=tuple(spec["patch_size"]),
            width=int(spec["width"]),
            depth=int(spec["depth"]),
            heads=int(spec["heads"]),
            mlp_ratio=int(spec["mlp_ratio"]),
            scalar_dim=int(spec["scalar_dim"]),
            dropout=float(spec.get("dropout", 0.0)),
            position_encoding=str(spec["position_encoding"]),
        )
    if spec.get("architecture") == "darcy_masked_anchor_operator_v1":
        return MaskedAnchorOperator(
            input_channels=int(spec["input_channels"]),
            scalar_dim=int(spec["scalar_dim"]),
            width=int(spec["width"]),
            depth=int(spec["depth"]),
            heads=int(spec["heads"]),
            anchors=int(spec["anchors"]),
            mlp_ratio=int(spec["mlp_ratio"]),
            dropout=float(spec.get("dropout", 0.0)),
        )
    if spec.get("architecture") != "masked_point_transolver_v1":
        raise ValueError("checkpoint is not a recognized global Darcy baseline")
    return MaskedTransolverOperator(
        input_channels=int(spec["input_channels"]),
        scalar_dim=int(spec["scalar_dim"]),
        width=int(spec["width"]),
        depth=int(spec["depth"]),
        heads=int(spec["heads"]),
        slices=int(spec["slices"]),
        mlp_ratio=int(spec["mlp_ratio"]),
    )


def _metadata(batch: dict[str, Any], item: int) -> dict[str, Any]:
    result = {"case_index": int(batch["case_index"][item])}
    if "seed" in batch:
        result["case_seed"] = int(batch["seed"][item])
    if "family" in batch:
        result["family"] = str(batch["family"][item])
    return result


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    metrics = (
        "relative_l2", "energy_relative_error", "residual_rms",
        "wall_seconds", "global_nfe", "processed_points",
    )
    keys = sorted({(row["group"], row["sweep"]) for row in records})
    for group, sweep in keys:
        selected = [
            row for row in records
            if row["group"] == group and row["sweep"] == sweep
        ]
        result[f"{group}/sweep_{sweep:03d}"] = {
            metric: float(np.mean([float(row[metric]) for row in selected]))
            for metric in metrics
        }
        result[f"{group}/sweep_{sweep:03d}"]["cases"] = len(selected)
    return result


@torch.no_grad()
def evaluate_global(args: argparse.Namespace) -> dict[str, Any]:
    if args.sweeps <= 0 or not 0.0 < args.relaxation <= 1.5:
        raise ValueError("sweeps must be positive and relaxation in (0,1.5]")
    opens_test = "test" in args.groups
    if opens_test and not bool(getattr(args, "allow_protected_test", False)):
        raise ValueError("test group requires --allow-protected-test")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") not in (
        CHECKPOINT_CONTRACT, VIT_CHECKPOINT_CONTRACT,
        ANCHOR_CHECKPOINT_CONTRACT,
    ):
        raise ValueError("checkpoint contract does not match global baseline")
    current_policy = str(checkpoint.get(
        "current_policy", checkpoint.get("args", {}).get(
            "current_policy", "arbitrary",
        ),
    ))
    use_discrete_residual = bool(checkpoint.get(
        "use_discrete_residual", checkpoint.get("args", {}).get(
            "use_discrete_residual", True,
        ),
    ))
    if current_policy == "zero" and args.sweeps != 1:
        raise ValueError(
            "direct-zero global checkpoints have a one-pass inference contract"
        )
    statistics = DarcyStatistics.from_json(checkpoint["statistics"])
    adapter = DarcyAdapter(
        statistics, use_discrete_residual=use_discrete_residual,
    )
    model = build_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    records: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for group_name in args.groups:
        dataset = H5DarcyDataset(args.data, group_name)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        cases_seen = 0
        for host_batch in loader:
            if args.max_cases is not None and cases_seen >= args.max_cases:
                break
            batch = move_batch(host_batch, device)
            if args.max_cases is not None:
                keep = min(batch["solution"].shape[0], args.max_cases - cases_seen)
                for key, value in list(batch.items()):
                    if isinstance(value, Tensor) and value.shape[0] == batch["solution"].shape[0]:
                        batch[key] = value[:keep]
                    elif isinstance(value, list) and len(value) == batch["solution"].shape[0]:
                        batch[key] = value[:keep]
            batch_size = batch["solution"].shape[0]
            current = initial_state(batch, statistics, args.initialization)
            previous = current.clone()
            cumulative_time = 0.0
            points = int(batch["mask"].shape[-2] * batch["mask"].shape[-1])
            for sweep in range(1, args.sweeps + 1):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                next_state = global_step(
                    model, adapter, batch, current, previous, args.relaxation,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                cumulative_time += (time.perf_counter() - started) / batch_size
                l2 = relative_l2(next_state, batch["solution"], batch["mask"])
                energy = energy_relative_error(
                    next_state, batch["solution"], batch["logk"], batch["mask"],
                )
                residual = residual_rms(next_state, batch)
                active_points = batch["mask"].flatten(1).sum(1)
                for item in range(batch_size):
                    records.append({
                        **_metadata(batch, item),
                        "group": group_name,
                        "regime": (
                            "distribution_matched_geometry"
                            if group_name in ("validation", "val", "test")
                            else "geometry_topology_shift"
                        ),
                        "sweep": sweep,
                        "relative_l2": float(l2[item]),
                        "energy_relative_error": float(energy[item]),
                        "residual_rms": float(residual[item]),
                        "wall_seconds": cumulative_time,
                        "global_nfe": sweep,
                        "active_points": float(active_points[item]),
                        "processed_points": float(points * sweep),
                        "value_jump": float("nan"),
                        "flux_jump": float("nan"),
                    })
                previous, current = current, next_state
            cases_seen += batch_size

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if records:
        fields = sorted({key for row in records for key in row})
        with (args.output_dir / "iteration_metrics.csv").open(
            "w", newline="", encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    result = {
        "contract": "darcy_global_full_domain_evaluation_v3",
        "data": str(args.data.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "groups": list(args.groups),
        "protected_test_cases_opened": (
            len({int(row["case_index"]) for row in records if row["group"] == "test"})
            if opens_test else 0
        ),
        "configuration": {
            key: value for key, value in vars(args).items()
            if key not in ("data", "checkpoint", "output_dir")
        },
        "model_spec": checkpoint["model_spec"],
        "current_policy": current_policy,
        "use_discrete_residual": use_discrete_residual,
        "inference_contract": checkpoint.get("inference_contract"),
        "statistics": checkpoint["statistics"],
        "evaluated_records": len(records),
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "summary": _aggregate(records),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8",
    )
    return result


def main() -> None:
    result = evaluate_global(parse_args())
    print(json.dumps({
        "evaluated_records": result["evaluated_records"],
        "peak_gpu_bytes": result["peak_gpu_bytes"],
        "groups": result["groups"],
    }, indent=2))


if __name__ == "__main__":
    main()
