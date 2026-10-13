"""Train the frozen face-resolved Dirichlet local solver for neural SNI.

Only ``train`` and ``validation`` are opened. The learned map consumes a full
local boundary-value problem and returns its complete solution in one call;
the global interior iterate is never an input.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover
from codiffuse_Schwarz_Airfoil.code_transfer.schwarz import PreparedLevel

from .adapter import (
    DarcyStatistics,
    FaceDarcyAdapter,
    H5DarcyDataset,
    MODEL_SPEC,
    batch_to_transition,
    build_model,
    compute_statistics,
)
from .evaluate import move_batch, relative_l2
from .train import make_loader, seed_everything


Tensor = torch.Tensor
CONTRACT = "darcy_sni_face_boundary_to_solution_v1"
DATA_SCHEMA = "darcy_complex_cartesian_v2_face_dirichlet"


@dataclass(frozen=True)
class FaceBVPLoss:
    total: Tensor
    endpoint: Tensor
    local_endpoint: Tensor
    overlap: Tensor
    face_trace: Tensor
    face_flux: Tensor
    prediction: Tensor


def _masked_mean(value: Tensor, weight: Tensor) -> Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def face_supervision_terms(
    normalized_prediction: Tensor,
    normalized_target: Tensor,
    batch: dict[str, Tensor],
    cover: CartesianCover,
) -> tuple[Tensor, Tensor]:
    """Directional near-face and normal-flux surrogate losses.

    The half-cell Dirichlet flux error is
    ``2 k (u_pred-u_true)/dx``. Dividing permeability by its per-sample mean
    and expressing the field error in normalized units keeps the term
    dimensionless while preserving coefficient and face sensitivity.
    """
    face_mask = cover.tile(batch["dirichlet_face_mask"].to(
        normalized_prediction.dtype,
    ))
    if face_mask.shape[1] != 4:
        raise ValueError("face supervision requires N/S/W/E masks")
    target_tiles = cover.tile(normalized_target)
    error = normalized_prediction - target_tiles
    valid = cover.valid_tiles(
        normalized_target.shape[0], error.device, error.dtype,
    )
    weight = face_mask * valid
    trace = _masked_mean(error.expand(-1, 4, -1, -1).square(), weight)

    permeability = torch.pow(
        10.0, cover.tile(batch["logk"].to(error.dtype)).clamp(-6.0, 6.0),
    )
    active = cover.tile(batch["mask"].to(error.dtype)) * valid
    count = active.flatten(1).sum(1).clamp_min(1.0)
    mean_k = (permeability * active).flatten(1).sum(1) / count
    relative_k = permeability / mean_k[:, None, None, None].clamp_min(1.0e-8)
    flux_error = 2.0 * relative_k * error
    flux = _masked_mean(
        flux_error.expand(-1, 4, -1, -1).square(), weight,
    )
    return trace, flux


def face_bvp_loss(
    model: torch.nn.Module,
    adapter: FaceDarcyAdapter,
    level: PreparedLevel,
    *,
    local_weight: float,
    overlap_weight: float,
    trace_weight: float,
    flux_weight: float,
) -> FaceBVPLoss:
    if min(local_weight, overlap_weight, trace_weight, flux_weight) < 0.0:
        raise ValueError("loss weights must be nonnegative")
    if torch.count_nonzero(level.transition.current).item() != 0:
        raise ValueError("face local-BVP training requires a zero interior state")
    phase = torch.ones(
        level.batch_size,
        device=level.transition.current.device,
        dtype=level.transition.current.dtype,
    )
    physical, scalar = level.conditions(phase)
    local = model(
        physical, scalar, level.cover.tile_count,
        tile_grid_shape=level.cover.grid_shape,
    )
    target = adapter.normalized_target(level.transition)
    assembled = level.cover.assemble_weighted(local, level.batch_size)
    target_tiles = level.cover.tile(target)
    valid = level.cover.valid_tiles(level.batch_size, local.device, local.dtype)
    local_endpoint = _masked_mean((local - target_tiles).square(), valid)
    domain = level.transition.payload["mask"].to(local.dtype)
    endpoint = _masked_mean((assembled - target).square(), domain)
    consensus = level.cover.tile(assembled)
    overlap = _masked_mean((local - consensus).square(), valid)
    trace, flux = face_supervision_terms(
        local, target, level.transition.payload, level.cover,
    )
    total = (
        endpoint + local_weight * local_endpoint + overlap_weight * overlap
        + trace_weight * trace + flux_weight * flux
    )
    prediction = adapter.apply_endpoint(level.transition, assembled)
    return FaceBVPLoss(
        total, endpoint, local_endpoint, overlap, trace, flux, prediction,
    )


def _level(
    adapter: FaceDarcyAdapter, batch: dict[str, Tensor],
) -> PreparedLevel:
    current = torch.zeros_like(batch["solution"])
    transition = batch_to_transition(
        batch, current, previous=current, target=batch["solution"],
    )
    height, width = current.shape[-2:]
    return PreparedLevel(
        transition, adapter, CartesianCover(height, width, width, 0),
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    adapter: FaceDarcyAdapter,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for host_batch in loader:
        if count >= max_batches:
            break
        batch = move_batch(host_batch, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            report = face_bvp_loss(
                model, adapter, _level(adapter, batch),
                local_weight=args.local_weight,
                overlap_weight=args.overlap_weight,
                trace_weight=args.trace_weight,
                flux_weight=args.flux_weight,
            )
        values = {
            "relative_l2": relative_l2(
                report.prediction, batch["solution"], batch["mask"],
            ).mean(),
            "total": report.total,
            "endpoint": report.endpoint,
            "face_trace": report.face_trace,
            "face_flux": report.face_flux,
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        count += 1
    if count == 0:
        raise RuntimeError("face local validation produced no batches")
    return {name: value / count for name, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stats-cases", type=int, default=512)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--max-train-cases", type=int)
    parser.add_argument("--max-val-cases", type=int)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--min-lr", type=float, default=2.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--overlap-weight", type=float, default=0.25)
    parser.add_argument("--trace-weight", type=float, default=0.5)
    parser.add_argument("--flux-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _validate_dataset(dataset: H5DarcyDataset, name: str) -> None:
    schema = str(dataset.attrs.get("schema_version", ""))
    if schema != DATA_SCHEMA or not dataset.has_face_dirichlet:
        raise ValueError(
            f"{name} must use {DATA_SCHEMA!r} with all directional face fields"
        )


def train(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.epochs, args.samples_per_epoch, args.batch_size) <= 0:
        raise ValueError("training sizes must be positive")
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    train_set = H5DarcyDataset(args.data, "train")
    val_set = H5DarcyDataset(args.data, "validation")
    _validate_dataset(train_set, "train")
    _validate_dataset(val_set, "validation")
    if args.max_train_cases is not None:
        train_set.length = min(train_set.length, args.max_train_cases)
    if args.max_val_cases is not None:
        val_set.length = min(val_set.length, args.max_val_cases)
    statistics = compute_statistics(train_set, args.stats_cases, args.seed)
    adapter = FaceDarcyAdapter(statistics)
    model = build_model(adapter, communication_stage="none").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.min_lr,
    )
    start_epoch, best, history = 0, math.inf, []
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("contract") != CONTRACT:
            raise ValueError("resume checkpoint is not a face local-BVP solver")
        if saved.get("statistics") != statistics.to_json():
            raise ValueError("resume statistics differ from the v2 training corpus")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"])
        history = list(saved.get("history", []))
        best = min((float(row["selection_score"]) for row in history), default=best)

    train_loader = make_loader(
        train_set, args.batch_size, True, args.workers, device,
        args.samples_per_epoch,
    )
    val_loader = make_loader(
        val_set, args.batch_size, False, min(args.workers, 2), device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        count = 0
        for host_batch in train_loader:
            batch = move_batch(host_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                report = face_bvp_loss(
                    model, adapter, _level(adapter, batch),
                    local_weight=args.local_weight,
                    overlap_weight=args.overlap_weight,
                    trace_weight=args.trace_weight,
                    flux_weight=args.flux_weight,
                )
            report.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name in (
                "total", "endpoint", "local_endpoint", "overlap",
                "face_trace", "face_flux",
            ):
                value = getattr(report, name)
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            count += 1
        scheduler.step()
        validation = validate(
            model, adapter, val_loader, device, args.validation_batches, args,
        )
        score = (
            validation["relative_l2"]
            + 0.25 * math.sqrt(max(validation["face_trace"], 0.0))
            + 0.10 * math.sqrt(max(validation["face_flux"], 0.0))
        )
        row = {
            "epoch": epoch,
            "selection_score": score,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{name}": value / max(count, 1) for name, value in totals.items()},
            **{f"validation_{name}": value for name, value in validation.items()},
        }
        history.append(row)
        payload = {
            "contract": CONTRACT,
            "operator_free": True,
            "boundary_to_solution": True,
            "face_resolved": True,
            "local_frozen_ready": True,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "statistics": statistics.to_json(),
            "model_spec": {
                **MODEL_SPEC,
                "state_channels": adapter.state_channels,
                "spatial_condition_channels": adapter.spatial_condition_channels,
                "scalar_condition_dim": adapter.scalar_condition_dim,
                "face_order": ["north", "south", "west", "east"],
                "face_feature_blocks": [
                    "mask", "normalized_value", "physical_role", "artificial_role",
                ],
                "source": "codiffuse_Schwarz_Airfoil.code_transfer.model",
            },
            "training_data_schema": DATA_SCHEMA,
            "training_split_names": ["train", "validation"],
            "test_subsets_opened": False,
            "local_calls_per_bvp": 1,
            "args": vars(args),
            "validation": validation,
            "history": history,
        }
        torch.save(payload, args.output_dir / "last_checkpoint.pt")
        if score < best:
            best = score
            torch.save(payload, args.output_dir / "checkpoint.pt")
        print(json.dumps(row, sort_keys=True), flush=True)

    if not history:
        raise RuntimeError("face local training produced no epochs")
    with (args.output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    result = {
        "contract": CONTRACT,
        "data": str(args.data.resolve()),
        "best_selection_score": best,
        "train_cases": len(train_set),
        "validation_cases": len(val_set),
        "test_subsets_opened": False,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "wall_seconds": time.perf_counter() - started,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(train(parse_args()), indent=2, default=str))


if __name__ == "__main__":
    main()
