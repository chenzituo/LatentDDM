"""Matched validation of Darcy Local/Core, input LatentDDM, and Init-SNI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import H5DarcyDataset
from .evaluate import energy_relative_error, move_batch, relative_l2, residual_rms
from .input_correction_latentddm import (
    DarcyInputCorrectionLatentDDM,
    solve_corrected_boundary_input,
)
from .train_input_correction_latentddm import prepared_context
from .train_residual_port_chart import load_frozen_local, sha256
from .validate_face_sni import face_sni_step


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--composer-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=128)
    parser.add_argument(
        "--split", choices=("validation", "val", "test"),
        default="validation",
    )
    parser.add_argument(
        "--allow-protected-test", action="store_true",
        help="required acknowledgement before reading the protected test split",
    )
    parser.add_argument("--sni-sweeps", type=int, nargs="+", default=(0, 1, 2, 4, 8, 16))
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def case_metrics(
    prediction: Tensor,
    batch: dict[str, Tensor],
) -> dict[str, Tensor]:
    return {
        "relative_l2": relative_l2(
            prediction, batch["solution"], batch["mask"],
        ),
        "energy_relative_error": energy_relative_error(
            prediction, batch["solution"], batch["logk"], batch["mask"],
        ),
        "residual_rms": residual_rms(prediction, batch),
    }


def append_rows(
    rows: list[dict[str, Any]],
    method: str,
    prediction: Tensor,
    batch: dict[str, Tensor],
    seconds_per_case: float,
    sequential_local_rounds: int,
) -> None:
    metrics = case_metrics(prediction, batch)
    for item in range(prediction.shape[0]):
        rows.append({
            "method": method,
            "case_index": int(batch["case_index"][item]),
            "relative_l2": float(metrics["relative_l2"][item]),
            "energy_relative_error": float(metrics["energy_relative_error"][item]),
            "residual_rms": float(metrics["residual_rms"][item]),
            "seconds": seconds_per_case,
            "sequential_local_rounds": sequential_local_rounds,
        })


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    sweeps = sorted(set(int(value) for value in args.sni_sweeps))
    if (
        min(args.batch_size, args.max_cases) <= 0
        or not sweeps or sweeps[0] != 0 or min(sweeps) < 0
        or not 0.0 < args.tau <= 1.5
    ):
        raise ValueError("invalid Darcy matched evaluation controls")
    split = str(getattr(args, "split", "validation"))
    if split == "test" and not bool(getattr(args, "allow_protected_test", False)):
        raise ValueError("test split requires --allow-protected-test")
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda",
    )
    local_operator, adapter, _ = load_frozen_local(
        args.local_checkpoint, device,
    )
    saved = torch.load(
        args.composer_checkpoint, map_location="cpu", weights_only=False,
    )
    if saved.get("contract") != DarcyInputCorrectionLatentDDM.contract:
        raise ValueError("composer checkpoint has the wrong contract")
    if saved.get("local_checkpoint_sha256") != sha256(args.local_checkpoint):
        raise ValueError("composer and local BVP checkpoints differ")
    config = saved["configuration"]
    if config.get("assembly") != "normalized_overlap":
        raise ValueError("matched evaluation requires normalized overlap")
    model = DarcyInputCorrectionLatentDDM(
        29, 4,
        token_width=int(config["token_width"]),
        heads=int(config["heads"]),
        communication_depth=int(config["communication_depth"]),
        trace_modes=int(config.get("trace_modes", 2)),
        spatial_token_bins=int(config.get("spatial_token_bins", 0)),
    ).to(device)
    model.load_state_dict(saved["composer"], strict=True)
    model.eval().requires_grad_(False)
    dataset = H5DarcyDataset(args.data, split)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    rows: list[dict[str, Any]] = []
    identity_max = 0.0
    processed = 0
    tile_count = None
    try:
        for host in loader:
            if processed >= args.max_cases:
                break
            batch = move_batch(host, device)
            original = batch["solution"].shape[0]
            keep = min(original, args.max_cases - processed)
            for name, value in list(batch.items()):
                if isinstance(value, Tensor) and value.shape[0] == original:
                    batch[name] = value[:keep]
            cover = CartesianCover(
                batch["solution"].shape[-2], batch["solution"].shape[-1],
                int(config["core"]), int(config["halo"]),
            )
            if tile_count is None:
                tile_count = cover.tile_count
            elif tile_count != cover.tile_count:
                raise RuntimeError("evaluation batches use inconsistent covers")
            synchronize(device)
            started = time.perf_counter()
            local, context = prepared_context(
                local_operator, adapter, batch, cover,
                include_target=False,
                trace_modes=int(config.get("trace_modes", 2)),
            )
            local_field = (
                cover.assemble_weighted(local, keep)
                * float(adapter.statistics.solution_std) * batch["mask"]
            )
            synchronize(device)
            local_seconds = time.perf_counter() - started
            append_rows(
                rows, "local_core", local_field, batch,
                local_seconds / keep, 1,
            )

            synchronize(device)
            started = time.perf_counter()
            latent = model(
                local, context.physical, context.scalar,
                context.domain_mask * context.valid, cover.grid_shape,
            )
            final_local, _ = solve_corrected_boundary_input(
                local_operator, adapter, context, latent.correction_code,
            )
            latent_field = (
                cover.assemble_weighted(final_local, keep)
                * float(adapter.statistics.solution_std) * batch["mask"]
            )
            synchronize(device)
            latent_seconds = local_seconds + time.perf_counter() - started
            append_rows(
                rows, "latentddm", latent_field, batch,
                latent_seconds / keep, 2,
            )
            replay, _ = solve_corrected_boundary_input(
                local_operator, adapter, context,
                local.new_zeros(
                    keep, cover.tile_count,
                    int(config["trace_code_width"]),
                ),
            )
            identity_max = max(identity_max, float((replay - local).abs().max()))

            current = local_field
            cumulative = local_seconds
            if 0 in sweeps:
                append_rows(
                    rows, "init_sni_0", current, batch,
                    cumulative / keep, 1,
                )
            for sweep in range(1, max(sweeps) + 1):
                synchronize(device)
                started = time.perf_counter()
                current = face_sni_step(
                    local_operator, adapter, batch, current, cover,
                    "weighted_pou", args.tau,
                )
                synchronize(device)
                cumulative += time.perf_counter() - started
                if sweep in sweeps:
                    append_rows(
                        rows, f"init_sni_{sweep}", current, batch,
                        cumulative / keep, 1 + sweep,
                    )
            processed += keep
    finally:
        dataset.close()
    if identity_max > 2.0e-5:
        raise RuntimeError(
            f"zero communication does not replay Local/Core: {identity_max:.3e}",
        )
    summary = {}
    for method in sorted({str(row["method"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "cases": len(selected),
            **{
                f"{name}_mean": float(np.mean([row[name] for row in selected]))
                for name in (
                    "relative_l2", "energy_relative_error", "residual_rms",
                    "seconds", "sequential_local_rounds",
                )
            },
        }
    result = {
        "contract": (
            "darcy_input_correction_matched_test_v1"
            if split == "test" else
            "darcy_input_correction_matched_validation_v1"
        ),
        "split": split,
        "protected_test_cases_opened": processed if split == "test" else 0,
        "target_used_by_inference": False,
        "pde_residual_used_by_inference": False,
        "assembly": "normalized_overlap",
        "tau": args.tau,
        "cases": processed,
        "tile_count": tile_count,
        "zero_message_local_core_max_abs": identity_max,
        "local_checkpoint": str(args.local_checkpoint.resolve()),
        "local_checkpoint_sha256": sha256(args.local_checkpoint),
        "composer_checkpoint": str(args.composer_checkpoint.resolve()),
        "composer_checkpoint_sha256": sha256(args.composer_checkpoint),
        "communication_depth": int(config["communication_depth"]),
        "within_round_execution": "all local subdomains are GPU-batched",
        "sni_sweeps_are_sequential": True,
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args())["summary"], indent=2), flush=True)
