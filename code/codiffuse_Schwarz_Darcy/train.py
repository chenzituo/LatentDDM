"""Train the fixed Airfoil-family local endpoint model on Darcy corrections."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover
from codiffuse_Schwarz_Airfoil.code_transfer.direct import direct_endpoint_loss
from codiffuse_Schwarz_Airfoil.code_transfer.schwarz import PreparedLevel

from .adapter import (
    DarcyAdapter,
    DarcyStatistics,
    H5DarcyDataset,
    MODEL_SPEC,
    batch_to_transition,
    build_model,
    compute_statistics,
    enforce_domain_boundary,
)
from .evaluate import (
    composition_step,
    initial_state,
    move_batch,
    relative_l2,
    residual_rms,
)


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(
        "codiffuse_Schwarz_Darcy/data/darcy_complex_v1.h5"
    ))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--min-lr", type=float, default=2.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--overlap-weight", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stats-cases", type=int, default=512)
    parser.add_argument("--max-train-cases", type=int)
    parser.add_argument("--max-val-cases", type=int)
    parser.add_argument(
        "--samples-per-epoch", type=int, default=4096,
        help="random train records per epoch; the full 40k corpus remains eligible",
    )
    parser.add_argument("--validation-sweeps", type=int, default=4)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--operator-free", action="store_true",
        help="zero the legacy residual channel and never construct a discrete operator",
    )
    parser.add_argument(
        "--boundary-to-solution", action="store_true",
        help=(
            "EXPERIMENTAL cellwise Schwarz local BVP map; not a valid SNI "
            "baseline until face-resolved boundary channels are implemented"
        ),
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: H5DarcyDataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    device: torch.device,
    samples: int | None = None,
) -> DataLoader:
    sampler = None
    if shuffle and samples is not None:
        if samples <= 0:
            raise ValueError("samples per epoch must be positive")
        sampler = RandomSampler(
            dataset, replacement=samples > len(dataset), num_samples=samples,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=shuffle and len(dataset) >= batch_size,
    )


def arbitrary_current(
    batch: dict[str, Any], statistics: DarcyStatistics, generator: torch.Generator,
) -> Tensor:
    """Sample causal off-solution iterates, including exact zero-start cases."""
    solution, mask = batch["solution"], batch["mask"]
    noise = torch.randn(
        solution.shape, device=solution.device, dtype=solution.dtype, generator=generator,
    )
    noise = F.avg_pool2d(noise, kernel_size=9, stride=1, padding=4)
    count = mask.flatten(1).sum(1).clamp_min(1.0)
    noise_mean = (noise * mask).flatten(1).sum(1) / count
    noise = (noise - noise_mean[:, None, None, None]) * mask
    noise_rms = (
        (noise.square() * mask).flatten(1).sum(1) / count
    ).sqrt().clamp_min(1.0e-6)
    noise = noise / noise_rms[:, None, None, None]
    batch_size = solution.shape[0]
    alpha = 0.95 * torch.rand(
        batch_size, device=solution.device, generator=generator,
    )
    amplitude = statistics.solution_std * (
        0.25 + 1.25 * torch.rand(batch_size, device=solution.device, generator=generator)
    )
    baseline = statistics.solution_mean + amplitude[:, None, None, None] * noise
    current = (
        alpha[:, None, None, None] * solution
        + (1.0 - alpha[:, None, None, None]) * baseline
    )
    curriculum_draw = torch.rand(
        batch_size, device=solution.device, generator=generator,
    )
    zero_start = curriculum_draw < 0.25
    fixed_point = curriculum_draw > 0.90
    current[zero_start] = 0.0
    # Exact clean states teach the repeated map that zero correction is its
    # desired fixed point, rather than merely a limit approached in training.
    current[fixed_point] = solution[fixed_point]
    return enforce_domain_boundary(batch, current)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    adapter: DarcyAdapter,
    loader: DataLoader,
    device: torch.device,
    statistics: DarcyStatistics,
    sweeps: int,
    max_batches: int,
    relaxation: float,
    seed: int,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, list[Tensor]] = {
        "sweep1_relative_l2": [],
        "final_relative_l2": [],
        "final_residual_rms": [],
        "final_value_jump": [],
        "final_flux_jump": [],
    }
    for batch_index, host_batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = move_batch(host_batch, device)
        target = batch["solution"]
        current = initial_state(batch, statistics, "zero")
        previous = current.clone()
        # Train/validation records are already one halo-extended 40x40 local
        # BVP.  They must be evaluated directly, not decomposed a second time.
        cover = CartesianCover(
            target.shape[-2], target.shape[-1], target.shape[-1], 0,
        )
        interface: dict[str, Tensor] | None = None
        for sweep in range(sweeps):
            next_state, interface = composition_step(
                model, adapter, batch, current, previous, cover, "pou", relaxation,
            )
            if sweep == 0:
                metrics["sweep1_relative_l2"].append(
                    relative_l2(next_state, target, batch["mask"]).cpu()
                )
            previous, current = current, next_state
        assert interface is not None
        metrics["final_relative_l2"].append(
            relative_l2(current, target, batch["mask"]).cpu()
        )
        metrics["final_residual_rms"].append(residual_rms(current, batch).cpu())
        metrics["final_value_jump"].append(interface["value_jump"].cpu())
        metrics["final_flux_jump"].append(interface["flux_jump"].cpu())
    if not metrics["final_relative_l2"]:
        raise RuntimeError("validation produced no batches")
    return {
        name: float(torch.cat(values).mean()) for name, values in metrics.items()
    }


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    statistics: DarcyStatistics,
    args: argparse.Namespace,
    epoch: int,
    validation: dict[str, float],
    history: list[dict[str, float]],
) -> dict[str, Any]:
    return {
        "contract": (
            "darcy_sni_cellwise_boundary_to_solution_experimental_v1"
            if args.boundary_to_solution
            else "darcy_operator_free_local_endpoint_v1"
            if args.operator_free else "darcy_arbitrary_current_direct_endpoint_v1"
        ),
        "operator_free": bool(args.operator_free),
        "boundary_to_solution": bool(args.boundary_to_solution),
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "statistics": statistics.to_json(),
        "model_spec": {
            **MODEL_SPEC,
            "state_channels": DarcyAdapter.state_channels,
            "spatial_condition_channels": DarcyAdapter.spatial_condition_channels,
            "scalar_condition_dim": DarcyAdapter.scalar_condition_dim,
            "source": "codiffuse_Schwarz_Airfoil.code_transfer.model",
        },
        "args": vars(args),
        "validation": validation,
        "history": history,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0 or args.batch_size <= 0 or args.validation_sweeps <= 0:
        raise ValueError("epochs, batch size, and validation sweeps must be positive")
    seed_everything(args.seed)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    train_dataset = H5DarcyDataset(args.data, "train")
    val_dataset = H5DarcyDataset(args.data, "validation")
    if args.max_train_cases is not None:
        train_dataset.length = min(train_dataset.length, args.max_train_cases)
    if args.max_val_cases is not None:
        val_dataset.length = min(val_dataset.length, args.max_val_cases)
    statistics = compute_statistics(train_dataset, args.stats_cases, args.seed)
    adapter = DarcyAdapter(
        statistics, use_discrete_residual=not args.operator_free,
    )
    model = build_model(adapter).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr,
    )
    start_epoch = 0
    history: list[dict[str, float]] = []
    best_score = math.inf
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("model_spec", {}).get("source") \
                != "codiffuse_Schwarz_Airfoil.code_transfer.model":
            raise ValueError("resume checkpoint violates the shared model contract")
        if bool(checkpoint.get("operator_free", False)) != bool(args.operator_free):
            raise ValueError("resume checkpoint operator-free setting differs")
        if bool(checkpoint.get("boundary_to_solution", False)) != bool(args.boundary_to_solution):
            raise ValueError("resume checkpoint boundary-to-solution setting differs")
        if checkpoint["statistics"] != statistics.to_json():
            raise ValueError("resume train statistics differ from current data")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint.get("history", []))
        best_score = min(
            (record.get("selection_score", math.inf) for record in history),
            default=math.inf,
        )

    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.workers, device,
        args.samples_per_epoch,
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.workers, device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    training_start = time.perf_counter()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        totals: list[float] = []
        endpoints: list[float] = []
        locals_: list[float] = []
        overlaps: list[float] = []
        generator = torch.Generator(device=device).manual_seed(args.seed + 10_007 * epoch)
        for batch_index, host_batch in enumerate(train_loader):
            batch = move_batch(host_batch, device)
            current = (
                torch.zeros_like(batch["solution"])
                if args.boundary_to_solution
                else arbitrary_current(batch, statistics, generator)
            )
            transition = batch_to_transition(
                batch, current, previous=current, target=batch["solution"],
            )
            # Each stored training item is exactly one local 40x40 BVP
            # (32 owned cells plus a four-cell halo), so it is one network
            # example rather than a global field to re-tile.
            cover = CartesianCover(
                current.shape[-2], current.shape[-1], current.shape[-1], 0,
            )
            level = PreparedLevel(transition, adapter, cover)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                report = direct_endpoint_loss(
                    model, level,
                    local_weight=args.local_weight,
                    overlap_weight=args.overlap_weight,
                )
            report.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals.append(float(report.total.detach()))
            endpoints.append(float(report.endpoint.detach()))
            locals_.append(float(report.local_endpoint.detach()))
            overlaps.append(float(report.overlap.detach()))
        scheduler.step()
        validation = validate(
            model, adapter, val_loader, device, statistics,
            1 if args.boundary_to_solution else args.validation_sweeps,
            args.validation_batches,
            args.relaxation, args.seed + epoch,
        )
        selection_score = validation["final_relative_l2"] \
            + 0.1 * validation["final_residual_rms"]
        record = {
            "epoch": float(epoch),
            "train_total": float(np.mean(totals)),
            "train_endpoint": float(np.mean(endpoints)),
            "train_local": float(np.mean(locals_)),
            "train_overlap": float(np.mean(overlaps)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "selection_score": selection_score,
            **validation,
        }
        history.append(record)
        payload = checkpoint_payload(
            model, optimizer, scheduler, statistics, args, epoch,
            validation, history,
        )
        torch.save(payload, args.output_dir / "last_checkpoint.pt")
        if selection_score < best_score:
            best_score = selection_score
            torch.save(payload, args.output_dir / "checkpoint.pt")
        print(json.dumps(record, sort_keys=True), flush=True)

    history_path = args.output_dir / "history.csv"
    if history:
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    result = {
        "contract": (
            "darcy_sni_cellwise_boundary_to_solution_experimental_v1"
            if args.boundary_to_solution
            else "darcy_operator_free_local_endpoint_v1"
            if args.operator_free else "darcy_arbitrary_current_direct_endpoint_v1"
        ),
        "operator_free": bool(args.operator_free),
        "boundary_to_solution": bool(args.boundary_to_solution),
        "data": str(args.data.resolve()),
        "model_spec": MODEL_SPEC,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_cases": len(train_dataset),
        "validation_cases": len(val_dataset),
        "statistics": statistics.to_json(),
        "best_selection_score": best_score,
        "final_epoch": args.epochs,
        "wall_seconds": time.perf_counter() - training_start,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8",
    )
    return result


def main() -> None:
    result = train(parse_args())
    print(json.dumps({
        "best_selection_score": result["best_selection_score"],
        "parameter_count": result["parameter_count"],
        "peak_gpu_bytes": result["peak_gpu_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
