"""Train the operator-only one-message Darcy composition model.

The retained local endpoint network is evaluated once per overlapping unity.
One pooled token per unity is then globally communicated and decoded into
constant/linear correction modes.  No discrete Darcy residual or Galerkin
operator enters training or inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from codiffuse_Schwarz.learned_composer import LearnedModeComposer
from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import (
    DarcyAdapter, DarcyStatistics, FaceDarcyAdapter, H5DarcyDataset, MODEL_SPEC,
    batch_to_transition, build_model, tiled_darcy_conditions,
    tiled_sni_conditions,
)
from .evaluate import move_batch, relative_l2
from .train import make_loader, seed_everything


Tensor = torch.Tensor
CONTRACT = "darcy_operator_only_one_message_pou_v1"


def masked_mse(prediction: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    return ((prediction - target).square() * weight).sum() / weight.sum().clamp_min(1.0)


class OperatorOnlyDarcyComposer(torch.nn.Module):
    """Local neural endpoints plus one learned global mode composition."""

    def __init__(self, local: torch.nn.Module, adapter: DarcyAdapter) -> None:
        super().__init__()
        self.local = local
        self.composer = LearnedModeComposer(
            field_channels=adapter.state_channels,
            physical_channels=adapter.spatial_condition_channels + 1,
            scalar_dim=adapter.scalar_condition_dim,
            dimension=2,
            width=128,
            heads=4,
        )

    def forward(
        self, physical: Tensor, scalar: Tensor, valid: Tensor,
        grid_shape: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        local, _ = self.local.predict_fields(
            physical, scalar, math.prod(grid_shape), None,
        )
        composed = self.composer(local, physical, scalar, valid, grid_shape)
        return composed.corrected, local, composed.coefficients


def one_step(
    model: OperatorOnlyDarcyComposer,
    adapter: DarcyAdapter,
    batch: dict[str, Any],
    core: int,
    halo: int,
    cover: CartesianCover | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    current = torch.zeros_like(batch["solution"])
    transition = batch_to_transition(batch, current, target=batch["solution"])
    if cover is None:
        cover = CartesianCover(current.shape[-2], current.shape[-1], core, halo)
    if isinstance(adapter, FaceDarcyAdapter):
        physical, scalar = tiled_sni_conditions(adapter, batch, current, cover)
    else:
        physical, scalar = tiled_darcy_conditions(adapter, transition, cover)
    valid = cover.valid_tiles(current.shape[0], current.device, current.dtype)
    corrected, local, coefficient = model(physical, scalar, valid, cover.grid_shape)
    assembled = cover.assemble_weighted(corrected, current.shape[0])
    local_assembled = cover.assemble_weighted(local, current.shape[0])
    prediction = adapter.apply_endpoint(transition, assembled)
    target = adapter.normalized_target(transition)
    mask = batch["mask"].to(target.dtype)
    local_target = cover.tile(target)
    local_mask = cover.tile(mask) * valid
    boundary = torch.exp(-8.0 * batch["sdf"].abs()) * mask
    terms = {
        "endpoint": masked_mse(assembled, target, mask),
        "local": masked_mse(corrected, local_target, local_mask),
        "consensus": masked_mse(corrected, cover.tile(assembled), local_mask),
        "near_boundary": masked_mse(assembled, target, boundary),
        "mode_regularization": coefficient.square().mean(),
        "local_endpoint": masked_mse(local_assembled, target, mask),
    }
    return prediction, terms


def objective(terms: dict[str, Tensor]) -> Tensor:
    return (
        terms["endpoint"]
        + 0.25 * terms["local"]
        + 0.20 * terms["consensus"]
        + 0.25 * terms["near_boundary"]
        + 1.0e-4 * terms["mode_regularization"]
        + 0.10 * terms["local_endpoint"]
    )


@torch.no_grad()
def validate(
    model: OperatorOnlyDarcyComposer,
    adapter: DarcyAdapter,
    loader,
    device: torch.device,
    core: int,
    halo: int,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for host_batch in loader:
        batch = move_batch(host_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            prediction, terms = one_step(model, adapter, batch, core, halo)
            values = {
                "loss": objective(terms),
                "relative_l2": relative_l2(
                    prediction, batch["solution"], batch["mask"],
                ).mean(),
                **terms,
            }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        count += 1
        if count >= max_batches:
            break
    return {name: value / max(count, 1) for name, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--local-lr-scale", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--freeze-local", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    source = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    if not bool(source.get("operator_free", False)):
        raise ValueError(
            "composer requires a from-scratch operator-free local checkpoint"
        )
    statistics = DarcyStatistics.from_json(source["statistics"])
    adapter = (
        FaceDarcyAdapter(statistics)
        if source.get("contract") == "darcy_sni_face_boundary_to_solution_v1"
        else DarcyAdapter(statistics, use_discrete_residual=False)
    )
    local = build_model(adapter, communication_stage="none")
    incompatible = local.load_state_dict(source["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"incompatible local checkpoint: {incompatible}")
    model = OperatorOnlyDarcyComposer(local, adapter).to(device)
    if args.freeze_local:
        for parameter in model.local.parameters():
            parameter.requires_grad_(False)
    parameter_groups = [{"params": model.composer.parameters(), "lr": args.lr}]
    local_parameters = [p for p in model.local.parameters() if p.requires_grad]
    if local_parameters:
        parameter_groups.append({
            "params": local_parameters, "lr": args.lr * args.local_lr_scale,
        })
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1.0e-6,
    )
    start_epoch, best, history = 0, math.inf, []
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resumed.get("contract") != CONTRACT:
            raise ValueError("resume checkpoint is not an operator-only composer")
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        start_epoch = int(resumed["epoch"])
        history = list(resumed.get("history", []))
        best = min((float(row["selection_score"]) for row in history), default=best)

    train_set = H5DarcyDataset(args.data, "train")
    val_set = H5DarcyDataset(args.data, "validation")
    train_loader = make_loader(
        train_set, args.batch_size, True, args.workers, device, args.samples_per_epoch,
    )
    val_loader = make_loader(
        val_set, args.batch_size, False, min(args.workers, 2), device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        count = 0
        started = time.perf_counter()
        for host_batch in train_loader:
            batch = move_batch(host_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                _, terms = one_step(model, adapter, batch, args.core, args.halo)
                loss = objective(terms)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name, value in {"loss": loss, **terms}.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            count += 1
        scheduler.step()
        validation = validate(
            model, adapter, val_loader, device, args.core, args.halo,
            args.validation_batches,
        )
        score = validation["relative_l2"]
        row = {
            "epoch": epoch, "seconds": time.perf_counter() - started,
            "selection_score": score,
            "composer_lr": optimizer.param_groups[0]["lr"],
            **{f"train_{name}": value / max(count, 1) for name, value in totals.items()},
            **{f"validation_{name}": value for name, value in validation.items()},
        }
        history.append(row)
        payload = {
            "contract": CONTRACT,
            "operator_free": True,
            "communication_rounds": 1,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "statistics": statistics.to_json(),
            "model_spec": {**MODEL_SPEC, "composer_width": 128, "composer_modes": 3},
            "source_checkpoint": str(args.pretrained.resolve()),
            "source_contract": source.get("contract"),
            "local_frozen": bool(args.freeze_local),
            "args": vars(args),
            "history": history,
        }
        torch.save(payload, args.output_dir / "last_checkpoint.pt")
        if score < best:
            best = score
            torch.save(payload, args.output_dir / "checkpoint.pt")
        print(json.dumps(row, sort_keys=True), flush=True)

    if not history:
        raise ValueError("training produced no epochs")
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    result = {
        "contract": CONTRACT,
        "best_selection_score": best,
        "operator_free": True,
        "communication_rounds": 1,
        "source_checkpoint": str(args.pretrained.resolve()),
        "source_contract": source.get("contract"),
        "local_frozen": bool(args.freeze_local),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, default=str))
