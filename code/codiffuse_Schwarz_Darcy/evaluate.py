"""Evaluate repeated deterministic POU/RAS composition on held-out Darcy domains."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover
from codiffuse_Schwarz_Airfoil.code_transfer.schwarz import PreparedLevel

from .adapter import (
    DarcyAdapter,
    DarcyStatistics,
    FaceDarcyAdapter,
    H5DarcyDataset,
    MODEL_SPEC,
    batch_to_transition,
    build_model,
    checkpoint_communication_stage,
    enforce_domain_boundary,
    raw_darcy_residual,
    raw_face_darcy_residual,
    tiled_darcy_conditions, tiled_sni_conditions,
)


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(
        "codiffuse_Schwarz_Darcy/data/darcy_complex_v1.h5"
    ))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("validation", "val", "test", "A", "B", "C", "D"),
        default="test",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=16)
    parser.add_argument(
        "--methods", nargs="+",
        choices=(
            "global", "pou", "ras", "sni_official_mean", "sni_patched", "sni_additive",
            "pou_aitken", "pou_energy",
        ),
        default=(
            "global", "pou", "ras", "sni_official_mean", "pou_aitken", "pou_energy",
        ),
    )
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument(
        "--sni-relaxation", type=float, default=0.1,
        help="relaxation for the released SNI overlap-averaged patch map",
    )
    parser.add_argument(
        "--asm-relaxation", type=float, default=0.04,
        help="relaxation for the unnormalized additive-Schwarz equation",
    )
    parser.add_argument(
        "--operator-free", action="store_true",
        help="zero the legacy residual input and never call a discrete operator in the model",
    )
    parser.add_argument("--decomposition-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--offsets", nargs="+",
        help="explicit y,x cover offsets, e.g. 0,0 8,8 16,16; overrides seeds",
    )
    parser.add_argument(
        "--cover-policy", choices=("fixed", "random_each_sweep"), default="fixed",
    )
    parser.add_argument("--initialization", choices=("zero", "mean"), default="zero")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def parse_offsets(values: list[str] | None) -> list[tuple[int, int]] | None:
    if values is None:
        return None
    result: list[tuple[int, int]] = []
    for value in values:
        pieces = value.split(",")
        if len(pieces) != 2:
            raise ValueError(f"offset {value!r} must have y,x form")
        offset = int(pieces[0]), int(pieces[1])
        if min(offset) < 0 or max(offset) >= MODEL_SPEC["core"]:
            raise ValueError(f"offset {offset} lies outside [0,{MODEL_SPEC['core']})")
        result.append(offset)
    return result


def relative_l2(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    numerator = ((prediction - target).square() * mask).flatten(1).sum(1)
    denominator = (target.square() * mask).flatten(1).sum(1).clamp_min(1.0e-12)
    return (numerator / denominator).sqrt()


def energy_relative_error(
    prediction: Tensor, target: Tensor, logk: Tensor, mask: Tensor,
) -> Tensor:
    error = prediction - target
    coefficient = torch.pow(10.0, logk.clamp(-6.0, 6.0))
    zero = torch.zeros_like(target)

    def energy(field: Tensor) -> Tensor:
        operator = raw_darcy_residual(field, coefficient, zero, mask, zero)
        return (field * operator).flatten(1).sum(1).clamp_min(0.0)

    return (energy(error) / energy(target).clamp_min(1.0e-12)).sqrt()


def residual_rms(state: Tensor, batch: dict[str, Any]) -> Tensor:
    coefficient = torch.pow(10.0, batch["logk"].clamp(-6.0, 6.0))
    spacing = batch.get("grid_spacing")
    if all(name in batch for name in (
        "dirichlet_face_mask", "dirichlet_face_value",
    )):
        residual_function = lambda value: raw_face_darcy_residual(
            value, coefficient, batch["source"], batch["mask"],
            batch["dirichlet_face_mask"], batch["dirichlet_face_value"],
            dx=spacing,
        )
    else:
        residual_function = lambda value: raw_darcy_residual(
            value, coefficient, batch["source"], batch["mask"], batch["bc_value"],
            dx=spacing,
        )
    residual = residual_function(state)
    rhs = -residual_function(torch.zeros_like(state))
    active = batch["mask"] > 0.5
    numerator = (residual.square() * active).flatten(1).sum(1)
    denominator = (rhs.square() * active).flatten(1).sum(1).clamp_min(1.0e-12)
    return (numerator / denominator).sqrt()


def _interface_diagnostics(
    local_endpoint: Tensor,
    assembled_endpoint: Tensor,
    current: Tensor,
    logk: Tensor,
    cover: CartesianCover,
    solution_std: float,
) -> tuple[Tensor, Tensor]:
    """RMS duplicate value and flux disagreement before assembly."""
    batch = current.shape[0]
    consensus = cover.tile(assembled_endpoint)
    valid = cover.valid_tiles(batch, current.device, current.dtype)
    halo, core = cover.halo, cover.core
    owned = torch.zeros_like(valid)
    owned[..., halo:halo + core, halo:halo + core] = 1.0
    trace = valid * (1.0 - owned)
    value_difference = solution_std * (local_endpoint - consensus)
    value = (
        (value_difference.square() * trace).sum((-1, -2, -3))
        / trace.sum((-1, -2, -3)).clamp_min(1.0)
    ).sqrt().reshape(batch, cover.tile_count).mean(1)

    current_tiles = cover.tile(current)
    local_state = current_tiles + solution_std * local_endpoint
    consensus_state = current_tiles + solution_std * consensus
    coefficient = torch.pow(10.0, cover.tile(logk).clamp(-6.0, 6.0))
    dx = 2.0 / max(cover.height, cover.width, 1)
    local_dx = (local_state[..., 2:] - local_state[..., :-2]) / (2.0 * dx)
    global_dx = (consensus_state[..., 2:] - consensus_state[..., :-2]) / (2.0 * dx)
    local_dy = (local_state[..., 2:, :] - local_state[..., :-2, :]) / (2.0 * dx)
    global_dy = (consensus_state[..., 2:, :] - consensus_state[..., :-2, :]) / (2.0 * dx)
    trace_x = trace[..., 1:-1]
    trace_y = trace[..., 1:-1, :]
    kx = coefficient[..., 1:-1]
    ky = coefficient[..., 1:-1, :]
    flux_square = (
        ((kx * (local_dx - global_dx)).square() * trace_x).sum((-1, -2, -3))
        + ((ky * (local_dy - global_dy)).square() * trace_y).sum((-1, -2, -3))
    )
    flux_count = trace_x.sum((-1, -2, -3)) + trace_y.sum((-1, -2, -3))
    flux = (flux_square / flux_count.clamp_min(1.0)).sqrt()
    return value, flux.reshape(batch, cover.tile_count).mean(1)


@torch.no_grad()
def composition_step(
    model: torch.nn.Module,
    adapter: DarcyAdapter,
    batch: dict[str, Any],
    current: Tensor,
    previous: Tensor,
    cover: CartesianCover,
    method: str,
    relaxation: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    if method == "sni_official_mean":
        physical, scalar = tiled_sni_conditions(adapter, batch, current, cover)
        local_normalized_solution = model(
            physical, scalar, cover.tile_count,
            tile_grid_shape=cover.grid_shape,
        )
        local_solution = adapter.statistics.solution_std * local_normalized_solution
        active = cover.tile(batch["mask"].to(local_solution.dtype))
        patched_solution = cover.assemble_uniform(
            local_solution, current.shape[0], active,
        )
        next_state = enforce_domain_boundary(
            batch, current + relaxation * (patched_solution - current),
        )
        local_correction = (
            local_solution - cover.tile(current)
        ) / adapter.statistics.solution_std
        assembled_correction = (
            patched_solution - current
        ) / adapter.statistics.solution_std
        value_jump, flux_jump = _interface_diagnostics(
            local_correction, assembled_correction, current, batch["logk"], cover,
            adapter.statistics.solution_std,
        )
        return next_state, {"value_jump": value_jump, "flux_jump": flux_jump}

    transition = batch_to_transition(
        batch, current, previous=previous, target=batch.get("solution"),
    )
    level = PreparedLevel(transition, adapter, cover)
    physical, scalar = tiled_darcy_conditions(
        adapter, transition, cover, level_scale=level.level_scale,
    )
    local = model(
        physical, scalar, cover.tile_count,
        tile_grid_shape=cover.grid_shape,
    )
    if method == "pou":
        assembled = cover.assemble_weighted(local, current.shape[0])
    elif method == "ras":
        assembled = cover.assemble_ras(local, current.shape[0])
    elif method == "sni_additive":
        assembled = cover.assemble_additive(local, current.shape[0])
    elif method == "sni_patched":
        active = cover.tile(batch["mask"].to(local.dtype))
        assembled = cover.assemble_uniform(local, current.shape[0], active)
    else:
        raise ValueError(method)
    proposal = adapter.apply_endpoint(transition, assembled)
    next_state = enforce_domain_boundary(
        batch, current + relaxation * (proposal - current),
    )
    value_jump, flux_jump = _interface_diagnostics(
        local, assembled, current, batch["logk"], cover,
        adapter.statistics.solution_std,
    )
    return next_state, {"value_jump": value_jump, "flux_jump": flux_jump}


def initial_state(
    batch: dict[str, Any], statistics: DarcyStatistics, mode: str,
) -> Tensor:
    reference = batch["solution"]
    state = torch.zeros_like(reference)
    if mode == "mean":
        state.fill_(statistics.solution_mean)
    return enforce_domain_boundary(batch, state)


def aitken_factor(
    previous_update: Tensor, update: Tensor, previous_factor: Tensor,
) -> Tensor:
    difference = (update - previous_update).flatten(1)
    numerator = (previous_update.flatten(1) * difference).sum(1)
    denominator = difference.square().sum(1).clamp_min(1.0e-20)
    return (-previous_factor * numerator / denominator).clamp(0.05, 1.5)


def energy_factor(current: Tensor, update: Tensor, batch: dict[str, Any]) -> Tensor:
    """Exact FV energy line search along an assembled neural update."""
    coefficient = torch.pow(10.0, batch["logk"].clamp(-6.0, 6.0))
    residual = raw_darcy_residual(
        current, coefficient, batch["source"], batch["mask"], batch["bc_value"],
    )
    operator_update = raw_darcy_residual(
        update, coefficient, torch.zeros_like(update), batch["mask"],
        torch.zeros_like(update),
    )
    numerator = -(residual * update).flatten(1).sum(1)
    denominator = (operator_update * update).flatten(1).sum(1).clamp_min(1.0e-20)
    return (numerator / denominator).clamp(0.0, 1.5)


def case_metadata(batch: dict[str, Any], item: int) -> dict[str, Any]:
    result: dict[str, Any] = {"case_index": int(batch["case_index"][item])}
    if "seed" in batch:
        result["case_seed"] = int(batch["seed"][item])
    if "family" in batch:
        result["family"] = str(batch["family"][item])
    return result


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["method"], record["sweep"])].append(record)
    summary: dict[str, Any] = {}
    metric_names = (
        "relative_l2", "energy_relative_error", "residual_rms",
        "value_jump", "flux_jump", "relaxation", "wall_seconds", "local_nfe",
    )
    for (method, sweep), values in grouped.items():
        key = f"{method}/sweep_{sweep:03d}"
        summary[key] = {
            name: float(np.mean([float(value[name]) for value in values]))
            for name in metric_names
        }
        summary[key]["cases"] = len(values)
    return summary


def checkpoint_local_model_state(
    checkpoint: dict[str, Any],
) -> tuple[str, dict[str, Tensor], str]:
    """Resolve the exact local endpoint embedded in a frozen checkpoint."""
    if checkpoint.get("contract") == "darcy_operator_only_one_message_pou_v1":
        state = {
            name.removeprefix("local."): value
            for name, value in checkpoint["model"].items()
            if name.startswith("local.")
        }
        if not state:
            raise ValueError("operator-only composer checkpoint has no local endpoint tensors")
        return "none", state, "embedded_finetuned_local_endpoint"
    return (
        checkpoint_communication_stage(checkpoint), checkpoint["model"],
        "top_level_checkpoint_model",
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.sweeps <= 0 or any(
        not 0.0 < value <= 1.5
        for value in (args.relaxation, args.sni_relaxation, args.asm_relaxation)
    ):
        raise ValueError("sweeps must be positive and relaxations in (0,1.5]")
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "sni_official_mean" in args.methods:
        direct_boundary_solver = (
            checkpoint.get("contract") == "darcy_sni_face_boundary_to_solution_v1"
        )
        frozen_boundary_backbone = (
            checkpoint.get("source_contract") == "darcy_sni_face_boundary_to_solution_v1"
            and bool(checkpoint.get("local_frozen", False))
        )
        if not (direct_boundary_solver or frozen_boundary_backbone):
            raise ValueError(
                "sni_official_mean requires a dedicated boundary-to-solution "
                "checkpoint or a composer with that local backbone frozen"
            )
    statistics = DarcyStatistics.from_json(checkpoint["statistics"])
    face_checkpoint = (
        checkpoint.get("contract") == "darcy_sni_face_boundary_to_solution_v1"
        or checkpoint.get("source_contract")
        == "darcy_sni_face_boundary_to_solution_v1"
    )
    adapter = (
        FaceDarcyAdapter(statistics)
        if face_checkpoint else DarcyAdapter(
            statistics, use_discrete_residual=not args.operator_free,
        )
    )
    communication_stage, model_state, model_state_source = checkpoint_local_model_state(
        checkpoint,
    )
    model = build_model(
        adapter, communication_stage=communication_stage,
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()
    dataset = H5DarcyDataset(args.data, args.split)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    records: list[dict[str, Any]] = []
    explicit_offsets = parse_offsets(args.offsets)
    if explicit_offsets is not None and args.cover_policy != "fixed":
        raise ValueError("explicit offsets require --cover-policy fixed")
    layouts: list[tuple[int, tuple[int, int] | None]] = (
        [(-1, offset) for offset in explicit_offsets]
        if explicit_offsets is not None
        else [(seed, None) for seed in args.decomposition_seeds]
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for batch_index, host_batch in enumerate(loader):
        if args.max_cases is not None and batch_index * args.batch_size >= args.max_cases:
            break
        batch = move_batch(host_batch, device)
        if args.max_cases is not None:
            keep = min(batch["solution"].shape[0], args.max_cases - batch_index * args.batch_size)
            for key, value in list(batch.items()):
                if isinstance(value, Tensor) and value.shape[0] == batch["solution"].shape[0]:
                    batch[key] = value[:keep]
                elif isinstance(value, list) and len(value) == batch["solution"].shape[0]:
                    batch[key] = value[:keep]
        target = batch["solution"]
        for method in args.methods:
            method_layouts = [(-1, (0, 0))] if method == "global" else layouts
            for decomposition_seed, explicit_offset in method_layouts:
                current = initial_state(batch, statistics, args.initialization)
                previous = current.clone()
                previous_update: Tensor | None = None
                previous_factor = torch.ones(
                    target.shape[0], device=target.device, dtype=target.dtype,
                )
                cumulative_wall_seconds = 0.0
                local_record = args.split in ("validation", "val", "test")
                if local_record or method == "global":
                    # These groups already contain individual 40x40 local BVPs.
                    # The global baseline likewise applies the fully
                    # convolutional local checkpoint once to the entire square.
                    fixed_cover = CartesianCover(
                        target.shape[-2], target.shape[-1], target.shape[-1], 0,
                    )
                else:
                    if explicit_offset is None:
                        fixed_cover = CartesianCover(
                            target.shape[-2], target.shape[-1],
                            MODEL_SPEC["core"], MODEL_SPEC["halo"],
                        ).random_shift(decomposition_seed)
                    else:
                        fixed_cover = CartesianCover(
                            target.shape[-2], target.shape[-1],
                            MODEL_SPEC["core"], MODEL_SPEC["halo"],
                            explicit_offset[0], explicit_offset[1],
                        )
                for sweep in range(1, args.sweeps + 1):
                    cover = fixed_cover
                    if args.cover_policy == "random_each_sweep" and not local_record:
                        cover = CartesianCover(
                            target.shape[-2], target.shape[-1],
                            MODEL_SPEC["core"], MODEL_SPEC["halo"],
                        ).random_shift(decomposition_seed + 1_000_003 * sweep)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    if method == "ras":
                        base_method = "ras"
                    elif method in ("sni_official_mean", "sni_patched", "sni_additive"):
                        base_method = method
                    else:
                        base_method = "pou"
                    candidate, interface = composition_step(
                        model, adapter, batch, current, previous, cover,
                        base_method, 1.0,
                    )
                    update = candidate - current
                    if method == "pou_aitken" and previous_update is not None:
                        factor = aitken_factor(previous_update, update, previous_factor)
                    elif method == "pou_energy":
                        factor = energy_factor(current, update, batch)
                    elif method in ("sni_official_mean", "sni_patched"):
                        factor = torch.full_like(previous_factor, args.sni_relaxation)
                    elif method == "sni_additive":
                        factor = torch.full_like(previous_factor, args.asm_relaxation)
                    else:
                        factor = torch.full_like(previous_factor, args.relaxation)
                    next_state = enforce_domain_boundary(
                        batch, current + factor[:, None, None, None] * update,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) / target.shape[0]
                    cumulative_wall_seconds += elapsed
                    metrics = {
                        "relative_l2": relative_l2(next_state, target, batch["mask"]),
                        "energy_relative_error": energy_relative_error(
                            next_state, target, batch["logk"], batch["mask"],
                        ),
                        "residual_rms": residual_rms(next_state, batch),
                    }
                    for item in range(target.shape[0]):
                        records.append({
                            **case_metadata(batch, item),
                            "method": method,
                            "decomposition_seed": decomposition_seed,
                            "offset_y": cover.offset_y,
                            "offset_x": cover.offset_x,
                            "sweep": sweep,
                            "relative_l2": float(metrics["relative_l2"][item]),
                            "energy_relative_error": float(metrics["energy_relative_error"][item]),
                            "residual_rms": float(metrics["residual_rms"][item]),
                            "value_jump": float(interface["value_jump"][item]),
                            "flux_jump": float(interface["flux_jump"][item]),
                            "relaxation": float(factor[item]),
                            "wall_seconds": cumulative_wall_seconds,
                            "local_nfe": sweep * cover.tile_count,
                        })
                    previous_update = update
                    previous_factor = factor
                    previous, current = current, next_state

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "iteration_metrics.csv"
    if records:
        fieldnames = sorted({key for record in records for key in record})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    result = {
        "contract": "darcy_repeated_deterministic_composition_v1",
        "data": str(args.data.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "model_spec": {
            **MODEL_SPEC, "communication_stage": communication_stage,
        },
        "checkpoint_contract": checkpoint.get("contract"),
        "model_state_source": model_state_source,
        "operator_free_inference": bool(args.operator_free),
        "communication_stage": communication_stage,
        "statistics": statistics.to_json(),
        "configuration": {
            key: value for key, value in vars(args).items()
            if key not in ("data", "checkpoint", "output_dir")
        },
        "evaluated_records": len(records),
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "summary": aggregate(records),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8",
    )
    return result


def main() -> None:
    result = evaluate(parse_args())
    print(json.dumps({
        "evaluated_records": result["evaluated_records"],
        "peak_gpu_bytes": result["peak_gpu_bytes"],
        "output": result["configuration"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
