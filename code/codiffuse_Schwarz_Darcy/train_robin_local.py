"""Train the distinct mixed Dirichlet/Robin local Darcy BVP operator.

Only train and validation are opened.  The map receives a complete local BVP
at zero current and predicts its solution in one endpoint call.  This v3
contract is intentionally shape-incompatible with the v2 Dirichlet model.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import MODEL_SPEC, build_model, compute_statistics
from .evaluate import move_batch, relative_l2
from .robin_v3 import (
    H5RobinDarcyDataset, ROBIN_LOCAL_CONTRACT, ROBIN_SCHEMA,
    RobinDarcyAdapter, raw_mixed_robin_residual, robin_batch_to_transition,
)
from .train import make_loader, seed_everything


Tensor = torch.Tensor


@dataclass(frozen=True)
class RobinBVPLoss:
    total: Tensor
    field: Tensor
    trace: Tensor
    residual: Tensor
    prediction: Tensor


def _masked_mean(value: Tensor, weight: Tensor) -> Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def robin_bvp_loss(
    model: torch.nn.Module,
    adapter: RobinDarcyAdapter,
    batch: dict[str, Tensor],
    *,
    trace_weight: float,
    residual_weight: float,
) -> RobinBVPLoss:
    if min(trace_weight, residual_weight) < 0.0:
        raise ValueError("loss weights must be nonnegative")
    current = torch.zeros_like(batch["solution"])
    transition = robin_batch_to_transition(batch, current, batch["solution"])
    height, width = current.shape[-2:]
    cover = CartesianCover(height, width, width, 0)
    phase = torch.ones(
        current.shape[0], device=current.device, dtype=current.dtype,
    )
    conditions = adapter.conditions(transition, phase, 1.0)
    physical, scalar = conditions.tiled(cover)
    normalized = model(
        physical, scalar, cover.tile_count,
        tile_grid_shape=cover.grid_shape,
    )
    target = adapter.normalized_target(transition)
    domain = batch["mask"].to(normalized)
    field = _masked_mean((normalized - target).square(), domain)
    prediction = adapter.apply_endpoint(transition, normalized)

    dx = 2.0 / max(height, width)
    kappa = torch.pow(10.0, batch["logk"].to(prediction).clamp(-6.0, 6.0))
    face_kappa = kappa.expand(-1, 4, -1, -1)
    face_prediction = prediction.expand(-1, 4, -1, -1)
    p = batch["robin_face_impedance"].to(prediction)
    g = batch["robin_face_value"].to(prediction)
    implied_trace = (
        g + 2.0 * face_kappa * face_prediction / dx
    ) / (p + 2.0 * face_kappa / dx).clamp_min(1.0e-12)
    trace_error = (
        implied_trace - batch["robin_face_trace"].to(prediction)
    ) / adapter.statistics.solution_std
    robin_mask = batch["robin_face_mask"].to(prediction)
    trace = _masked_mean(trace_error.square(), robin_mask)

    raw = raw_mixed_robin_residual(
        prediction, kappa, batch["source"].to(prediction), domain,
        batch["dirichlet_face_mask"].to(prediction),
        batch["dirichlet_face_value"].to(prediction), robin_mask, g, p,
    )
    active_count = domain.flatten(1).sum(1).clamp_min(1.0)
    mean_kappa = (kappa * domain).flatten(1).sum(1) / active_count
    dimensionless = raw * dx**2 / (
        mean_kappa[:, None, None, None] * adapter.statistics.solution_std
    ).clamp_min(1.0e-12)
    residual = _masked_mean(dimensionless.square(), domain)
    return RobinBVPLoss(
        field + trace_weight * trace + residual_weight * residual,
        field, trace, residual, prediction,
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    adapter: RobinDarcyAdapter,
    loader: DataLoader,
    device: torch.device,
    maximum_batches: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for host in loader:
        if batches >= maximum_batches:
            break
        batch = move_batch(host, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            report = robin_bvp_loss(
                model, adapter, batch, trace_weight=args.trace_weight,
                residual_weight=args.residual_weight,
            )
        values = {
            "relative_l2": relative_l2(
                report.prediction, batch["solution"], batch["mask"],
            ).mean(),
            "total": report.total,
            "field": report.field,
            "trace": report.trace,
            "residual": report.residual,
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        batches += 1
    if batches == 0:
        raise RuntimeError("Robin validation produced no batches")
    return {name: value / batches for name, value in totals.items()}


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
    parser.add_argument("--trace-weight", type=float, default=0.5)
    parser.add_argument("--residual-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.epochs, args.samples_per_epoch, args.batch_size) <= 0:
        raise ValueError("training sizes must be positive")
    seed_everything(args.seed)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda",
    )
    train_set = H5RobinDarcyDataset(args.data, "train")
    val_set = H5RobinDarcyDataset(args.data, "validation")
    if args.max_train_cases is not None:
        train_set.length = min(train_set.length, args.max_train_cases)
    if args.max_val_cases is not None:
        val_set.length = min(val_set.length, args.max_val_cases)
    corpus_sha256 = _file_sha256(args.data.resolve())
    statistics = compute_statistics(train_set, args.stats_cases, args.seed)
    adapter = RobinDarcyAdapter(statistics)
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
        if saved.get("contract") != ROBIN_LOCAL_CONTRACT:
            raise ValueError("resume checkpoint is not a Robin-v3 local solver")
        if saved.get("training_data_sha256") != corpus_sha256:
            raise ValueError("resume corpus hash differs from this Robin-v3 corpus")
        if saved.get("statistics") != statistics.to_json():
            raise ValueError("resume statistics differ from this Robin-v3 corpus")
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
        batches = 0
        for host in train_loader:
            batch = move_batch(host, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                report = robin_bvp_loss(
                    model, adapter, batch, trace_weight=args.trace_weight,
                    residual_weight=args.residual_weight,
                )
            if not torch.isfinite(report.total):
                raise FloatingPointError("non-finite Robin local loss")
            report.total.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError("non-finite Robin local gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name in ("total", "field", "trace", "residual"):
                totals[name] = totals.get(name, 0.0) + float(
                    getattr(report, name).detach(),
                )
            batches += 1
        scheduler.step()
        validation = validate(
            model, adapter, val_loader, device,
            args.validation_batches, args,
        )
        score = (
            validation["relative_l2"]
            + 0.25 * math.sqrt(max(validation["trace"], 0.0))
            + 0.10 * math.sqrt(max(validation["residual"], 0.0))
        )
        row = {
            "epoch": epoch,
            "selection_score": score,
            "lr": optimizer.param_groups[0]["lr"],
            **{
                f"train_{name}": value / max(batches, 1)
                for name, value in totals.items()
            },
            **{f"validation_{name}": value for name, value in validation.items()},
        }
        history.append(row)
        payload = {
            "contract": ROBIN_LOCAL_CONTRACT,
            "operator_free": True,
            "boundary_to_solution": True,
            "mixed_robin": True,
            "local_frozen_ready": True,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "statistics": statistics.to_json(),
            "training_data_sha256": corpus_sha256,
            "model_spec": {
                **MODEL_SPEC,
                "state_channels": adapter.state_channels,
                "spatial_condition_channels": adapter.spatial_condition_channels,
                "scalar_condition_dim": adapter.scalar_condition_dim,
                "face_order": ["north", "south", "west", "east"],
                "robin_convention": "outward kappa*d_n u + p*u_face = g",
                "source": "codiffuse_Schwarz_Airfoil.code_transfer.model",
            },
            "training_data_schema": ROBIN_SCHEMA,
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
        raise RuntimeError("Robin local training produced no epochs")
    with (args.output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    result = {
        "contract": ROBIN_LOCAL_CONTRACT,
        "data": str(args.data.resolve()),
        "training_data_sha256": corpus_sha256,
        "best_selection_score": best,
        "train_cases": len(train_set),
        "validation_cases": len(val_set),
        "test_subsets_opened": False,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "wall_seconds": time.perf_counter() - started,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, default=str))
