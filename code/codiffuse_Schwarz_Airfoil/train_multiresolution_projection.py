"""Fine-tune one shared Cartesian operator on paired level-5/level-6 fields.

This script is intentionally separate from recurrent rollout training. It first
establishes a scale-consistent one-step operator and conservative adaptive
hierarchy; only a passing checkpoint should enter recurrent optimization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from codiffuse_Schwarz_Airfoil.coupled_trajectory import CyclicWindowDataset
from codiffuse_Schwarz_Airfoil.dataset import FieldStatistics
from codiffuse_Schwarz_Airfoil.multiresolution import (
    PairedCartesianLevelDataset,
    constant_prolongation,
    level_batch,
)
from codiffuse_Schwarz_Airfoil.multiresolution_inference import (
    MultiresolutionPrediction,
    composite_vorticity_relative_l2,
    predict_multiresolution_step,
)
from codiffuse_Schwarz_Airfoil.train_one_step_multilevel import (
    interior_vorticity,
    load_pretrained,
)


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=128)
    parser.add_argument("--validation-stride", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--min-lr", type=float, default=2e-7)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--fine-fraction", type=float, default=.35)
    parser.add_argument("--fine-trust", type=float, default=.5)
    parser.add_argument("--coarse-state-weight", type=float, default=1.0)
    parser.add_argument("--fine-state-weight", type=float, default=1.0)
    parser.add_argument("--composite-vorticity-weight", type=float, default=4.0)
    parser.add_argument("--near-body-weight", type=float, default=1.0)
    parser.add_argument("--far-field-weight", type=float, default=2.0)
    parser.add_argument("--unfreeze-endpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def batched(sample: dict[str, Tensor], level: str) -> dict[str, Tensor]:
    return {key: value[None] for key, value in level_batch(sample, level).items()}


def masked_relative_square(
    prediction: Tensor, target: Tensor, mask: Tensor,
) -> Tensor:
    weight = mask.to(device=prediction.device, dtype=prediction.dtype)
    if weight.shape[1] == 1 and prediction.shape[1] != 1:
        weight = weight.expand(-1, prediction.shape[1], -1, -1)
    numerator = ((prediction - target).square() * weight).sum()
    denominator = (target.square() * weight).sum().clamp_min(1e-20)
    return numerator / denominator


def multiresolution_losses(
    prediction: MultiresolutionPrediction,
    coarse_target: Tensor,
    fine_target: Tensor,
    coarse_sdf: Tensor,
    fine_sdf: Tensor,
    coarse_dx: float,
    fine_dx: float,
    composite_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    coarse_state = masked_relative_square(
        prediction.coarse, coarse_target, torch.ones_like(coarse_sdf)
    )
    fine_mask = constant_prolongation(
        prediction.refinement_mask.to(fine_target.dtype)
    )
    fine_state = masked_relative_square(prediction.fine, fine_target, fine_mask)
    composite_vorticity = composite_vorticity_relative_l2(
        prediction.coarse, prediction.fine, coarse_target, fine_target,
        prediction.refinement_mask if composite_mask is None else composite_mask,
        coarse_dx=coarse_dx, fine_dx=fine_dx,
    ).square()
    near_body = (fine_sdf.abs() <= 8.0 * fine_dx).to(fine_target.dtype) * fine_mask
    near_body_velocity = masked_relative_square(
        prediction.fine[:, :2], fine_target[:, :2], near_body,
    )
    predicted_vorticity = interior_vorticity(prediction.coarse, coarse_dx)
    target_vorticity = interior_vorticity(coarse_target, coarse_dx)
    amplitude = target_vorticity.abs().flatten(1).amax(1)[:, None, None]
    quiet = (
        (coarse_sdf[:, 0, 1:-1, 1:-1].abs() > 8.0 * coarse_dx)
        & (target_vorticity.abs() < .02 * amplitude)
    )
    far_field = (
        (predicted_vorticity - target_vorticity).square()
        * quiet.to(predicted_vorticity.dtype)
    ).sum() / target_vorticity.square().sum().clamp_min(1e-20)
    boundary_band = max(1, min(target_vorticity.shape[-2:]) // 24)
    domain_boundary = torch.zeros_like(target_vorticity, dtype=torch.bool)
    domain_boundary[:, :boundary_band] = True
    domain_boundary[:, -boundary_band:] = True
    domain_boundary[:, :, :boundary_band] = True
    domain_boundary[:, :, -boundary_band:] = True
    domain_boundary_vorticity = (
        (predicted_vorticity - target_vorticity).square()
        * domain_boundary.to(predicted_vorticity.dtype)
    ).sum() / target_vorticity.square().sum().clamp_min(1e-20)
    return {
        "coarse_state": coarse_state,
        "fine_state": fine_state,
        "composite_vorticity": composite_vorticity,
        "near_body_velocity": near_body_velocity,
        "far_field_vorticity": far_field,
        "domain_boundary_vorticity": domain_boundary_vorticity,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    paired: PairedCartesianLevelDataset,
    coarse_dataset: CyclicWindowDataset,
    fine_dataset: CyclicWindowDataset,
    statistics: FieldStatistics,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    values = []
    for item in range(0, len(paired), args.validation_stride):
        sample = paired[item]
        coarse_batch = batched(sample, "coarse")
        fine_batch = batched(sample, "fine")
        prediction = predict_multiresolution_step(
            model, coarse_batch, fine_batch, coarse_dataset, fine_dataset,
            statistics, fine_fraction=args.fine_fraction, halo=args.halo,
            seed=args.seed + item * 1009, device=device,
            fine_trust=args.fine_trust,
        )
        losses = multiresolution_losses(
            prediction, coarse_batch["q_np1"].to(device),
            fine_batch["q_np1"].to(device), coarse_batch["sdf_np1"].to(device),
            fine_batch["sdf_np1"].to(device), coarse_dataset.dx, fine_dataset.dx,
        )
        values.append({
            key: float(torch.sqrt(value.clamp_min(0.0)))
            for key, value in losses.items()
        } | {"executed_fine_tile_fraction": float(
            prediction.executed_fine_tiles.float().mean()
        )})
    return {
        f"{key}_mean": float(np.mean([row[key] for row in values]))
        for key in values[0]
    } | {
        "composite_vorticity_p90": float(np.quantile(
            [row["composite_vorticity"] for row in values], .9
        )),
        "samples": len(values),
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.samples_per_epoch <= 0 or args.validation_stride <= 0:
        raise ValueError("epochs, samples per epoch, and validation stride must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    model, spec = load_pretrained(checkpoint, device, history_steps=2)
    statistics_payload = json.loads(args.statistics.read_text(encoding="utf-8"))
    statistics = FieldStatistics.from_json(statistics_payload["statistics"])
    model.set_endpoint_statistics(
        statistics.state_mean, statistics.state_std, statistics.delta_std,
        statistics.delta_mean,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable = []
    for name, parameter in model.named_parameters():
        selected = name.startswith("projection_")
        if args.unfreeze_endpoint:
            selected = selected or name.startswith("endpoint_")
        if selected:
            parameter.requires_grad = True
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("checkpoint has no trainable endpoint/projection parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.min_lr,
    )

    train = PairedCartesianLevelDataset(args.coarse, args.fine, "cycle1")
    validation = PairedCartesianLevelDataset(args.coarse, args.fine, "cycle2")
    coarse_train = CyclicWindowDataset(args.coarse, "cycle1", history_steps=2)
    fine_train = CyclicWindowDataset(args.fine, "cycle1", history_steps=2)
    coarse_validation = CyclicWindowDataset(args.coarse, "cycle2", history_steps=2)
    fine_validation = CyclicWindowDataset(args.fine, "cycle2", history_steps=2)
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        generator = random.Random(args.seed + epoch * 1_000_003)
        items = [generator.randrange(len(train)) for _ in range(args.samples_per_epoch)]
        records = []
        for local, item in enumerate(items):
            sample = train[item]
            coarse_batch = batched(sample, "coarse")
            fine_batch = batched(sample, "fine")
            optimizer.zero_grad(set_to_none=True)
            prediction = predict_multiresolution_step(
                model, coarse_batch, fine_batch, coarse_train, fine_train,
                statistics, fine_fraction=args.fine_fraction, halo=args.halo,
                seed=args.seed + epoch * 10_000_019 + local, device=device,
                fine_trust=args.fine_trust,
            )
            losses = multiresolution_losses(
                prediction, coarse_batch["q_np1"].to(device),
                fine_batch["q_np1"].to(device), coarse_batch["sdf_np1"].to(device),
                fine_batch["sdf_np1"].to(device), coarse_train.dx, fine_train.dx,
            )
            loss = (
                args.coarse_state_weight * losses["coarse_state"]
                + args.fine_state_weight * losses["fine_state"]
                + args.composite_vorticity_weight * losses["composite_vorticity"]
                + args.near_body_weight * losses["near_body_velocity"]
                + args.far_field_weight * losses["far_field_vorticity"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            records.append({"loss": float(loss.detach())} | {
                key: float(torch.sqrt(value.detach().clamp_min(0.0)))
                for key, value in losses.items()
            })
        scheduler.step()
        metrics = evaluate(
            model, validation, coarse_validation, fine_validation,
            statistics, args, device,
        )
        record = {
            "epoch": epoch, "train_loss": float(np.mean([x["loss"] for x in records])),
            "train_composite_vorticity_l2": float(np.mean([
                x["composite_vorticity"] for x in records
            ])), "lr": optimizer.param_groups[0]["lr"], **metrics,
        }
        selection = metrics["composite_vorticity_mean"]
        if selection < best:
            best = selection
            torch.save({
                "model": model.state_dict(), "model_spec": spec,
                "statistics": statistics.to_json(), "args": vars(args),
                "epoch": epoch, "validation": metrics,
                "training_contract": "paired_l5_l6_sparse_composite_one_step",
            }, args.output_dir / "checkpoint.pt")
        history.append(record)
        print(json.dumps(record), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "training_contract": "paired_l5_l6_sparse_composite_one_step",
        "best_composite_vorticity_l2": best, "history": history,
    }, indent=2) + "\n", encoding="utf-8")
    for dataset in (
        train, validation, coarse_train, fine_train,
        coarse_validation, fine_validation,
    ):
        dataset.close()


if __name__ == "__main__":
    main()
