"""Train PDE-agnostic Darcy LatentDDM by correcting local BVP inputs."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz.boundary_trace_basis import trace_relative_square
from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import H5DarcyDataset, MODEL_SPEC, tiled_sni_face_conditions
from .evaluate import move_batch, relative_l2
from .input_correction_latentddm import (
    DarcyInputCorrectionLatentDDM,
    boundary_input_context,
    solve_corrected_boundary_input,
)
from .train import make_loader, seed_everything
from .budget_protocol import apply_training_manifest, validate_resume_protocol
from .train_residual_port_chart import (
    load_frozen_local,
    local_model_call,
    masked_mse,
    sha256,
)


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--max-train-cases", type=int)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--selection-metric", choices=("legacy", "relative_l2"), default="legacy")
    parser.add_argument("--max-val-cases", type=int)
    parser.add_argument("--core", type=int, default=MODEL_SPEC["core"])
    parser.add_argument("--halo", type=int, default=MODEL_SPEC["halo"])
    parser.add_argument("--token-width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--communication-depth", type=int, choices=(2, 4, 6), default=4,
    )
    parser.add_argument("--trace-modes", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument(
        "--spatial-token-bins", type=int, choices=(0, 2, 3, 4), default=0,
        help="relative within-tile moment grid retained in each latent token",
    )
    parser.add_argument(
        "--assembly", choices=("normalized_overlap",),
        default="normalized_overlap",
    )
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--trace-weight", type=float, default=0.05)
    parser.add_argument("--gradient-weight", type=float, default=0.20)
    parser.add_argument("--near-boundary-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initial-field-token-checkpoint", type=Path,
        help="optional encoder/communication initialization from the old chart decoder",
    )
    parser.add_argument(
        "--initial-composer-checkpoint", type=Path,
        help=(
            "initialize the complete boundary-input composer from a source "
            "domain while resetting the optimizer"
        ),
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def gradient_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    numerator = prediction.new_zeros(())
    denominator = prediction.new_zeros(())
    for dimension in (-2, -1):
        predicted_difference = torch.diff(prediction, dim=dimension)
        target_difference = torch.diff(target, dim=dimension)
        left = [slice(None)] * mask.ndim
        right = [slice(None)] * mask.ndim
        left[dimension] = slice(None, -1)
        right[dimension] = slice(1, None)
        pair = mask[tuple(left)] * mask[tuple(right)]
        numerator = numerator + (
            (predicted_difference - target_difference).square() * pair
        ).sum()
        denominator = denominator + pair.sum()
    return numerator / denominator.clamp_min(1.0)


def prepared_context(
    local_operator,
    adapter,
    batch: dict[str, Tensor],
    cover: CartesianCover,
    *,
    include_target: bool = True,
    trace_modes: int = 2,
):
    zero = torch.zeros_like(batch["solution"])
    conditions = tiled_sni_face_conditions(
        adapter, batch, zero, cover,
    )
    with torch.no_grad():
        local = local_model_call(
            local_operator, conditions.physical, conditions.scalar, cover,
        )
        local = local * conditions.mask * conditions.valid
    context = boundary_input_context(
        adapter, batch, conditions, cover,
        include_target=include_target,
        trace_modes=trace_modes,
    )
    return local, context


def terms(
    model,
    local_operator,
    adapter,
    batch: dict[str, Tensor],
    args: argparse.Namespace,
) -> tuple[Tensor, dict[str, Tensor]]:
    cover = CartesianCover(
        batch["solution"].shape[-2], batch["solution"].shape[-1],
        args.core, args.halo,
    )
    local, context = prepared_context(
        local_operator, adapter, batch, cover,
        trace_modes=args.trace_modes,
    )
    latent = model(
        local, context.physical, context.scalar,
        context.domain_mask * context.valid, cover.grid_shape,
    )
    final_local, delta = solve_corrected_boundary_input(
        local_operator, adapter, context, latent.correction_code,
    )
    batch_size = batch["solution"].shape[0]
    prediction_normalized = cover.assemble_weighted(
        final_local, batch_size,
    ) * batch["mask"]
    local_normalized = cover.assemble_weighted(
        local, batch_size,
    ) * batch["mask"]
    target_normalized = (
        batch["solution"] / float(adapter.statistics.solution_std)
    ) * batch["mask"]
    boundary = torch.exp(-8.0 * batch["sdf"].abs()) * batch["mask"]
    field = masked_mse(
        prediction_normalized, target_normalized, batch["mask"],
    )
    local_field = masked_mse(
        local_normalized, target_normalized, batch["mask"],
    ).detach()
    gradient = gradient_mse(
        prediction_normalized, target_normalized, batch["mask"],
    )
    near = masked_mse(
        prediction_normalized, target_normalized, boundary,
    )
    trace = trace_relative_square(
        delta, context.target_delta, context.roles, 1,
    )
    loss = (
        field
        + args.gradient_weight * gradient
        + args.near_boundary_weight * near
        + args.trace_weight * trace
        + 1.0e-5 * latent.correction_code.square().mean()
    )
    physical_prediction = (
        prediction_normalized * float(adapter.statistics.solution_std)
    )
    physical_local = local_normalized * float(adapter.statistics.solution_std)
    return loss, {
        "field": field,
        "local_field": local_field,
        "gradient": gradient,
        "near_boundary": near,
        "trace": trace,
        "relative_l2": relative_l2(
            physical_prediction, batch["solution"], batch["mask"],
        ).mean(),
        "local_relative_l2": relative_l2(
            physical_local, batch["solution"], batch["mask"],
        ).mean().detach(),
        "code_rms": latent.correction_code.square().mean().sqrt(),
    }


@torch.no_grad()
def validate(model, local_operator, adapter, loader, device, args):
    model.eval(); local_operator.eval()
    totals: dict[str, float] = {}
    rows = []
    identity_max = 0.0
    batches = cases = 0
    for host in loader:
        if batches >= args.validation_batches:
            break
        batch = move_batch(host, device)
        loss, values = terms(model, local_operator, adapter, batch, args)
        cover = CartesianCover(
            batch["solution"].shape[-2], batch["solution"].shape[-1],
            args.core, args.halo,
        )
        local, context = prepared_context(
            local_operator, adapter, batch, cover,
            trace_modes=args.trace_modes,
        )
        replay, _ = solve_corrected_boundary_input(
            local_operator, adapter, context,
            local.new_zeros(
                batch["solution"].shape[0], cover.tile_count,
                4 * args.trace_modes,
            ),
        )
        identity_max = max(identity_max, float((replay - local).abs().max()))
        count = int(batch["solution"].shape[0])
        row = {
            "batch": batches,
            "cases": count,
            "loss": float(loss),
            **{name: float(value) for name, value in values.items()},
        }
        rows.append(row)
        for name, value in row.items():
            if name not in ("batch", "cases"):
                totals[name] = totals.get(name, 0.0) + count * float(value)
        cases += count; batches += 1
    if not cases:
        raise RuntimeError("Darcy input-correction validation produced no cases")
    if identity_max > 2.0e-5:
        raise RuntimeError(
            f"zero communication does not replay Local/Core: {identity_max:.3e}",
        )
    return {
        name: value / cases for name, value in totals.items()
    } | {
        "examples": cases,
        "zero_message_local_core_max_abs": identity_max,
    }, rows


def initialize_from_field_token(model, checkpoint: Path) -> None:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("contract") != "darcy_field_token_latentddm_core_v1":
        raise ValueError("initial checkpoint is not a Darcy field-token composer")
    spec = saved.get("composer_spec", {})
    if int(spec.get("token_width", -1)) != model.encoder.width \
            or int(spec.get("communication_depth", -1)) != model.communication_depth:
        raise ValueError("initial field-token dimensions differ")
    state = saved["composer"]
    encoder = {
        name.removeprefix("encoder."): value
        for name, value in state.items() if name.startswith("encoder.")
    }
    communication = {
        name.removeprefix("communication."): value
        for name, value in state.items() if name.startswith("communication.")
    }
    model.encoder.load_state_dict(encoder, strict=True)
    target = model.communication.state_dict()
    compatible = {
        name: value for name, value in communication.items()
        if name in target and target[name].shape == value.shape
    }
    model.communication.load_state_dict(compatible, strict=False)


def payload(
    model, optimizer, scheduler, local_checkpoint, adapter, args,
    epoch, history, validation_rows,
) -> dict[str, Any]:
    return {
        "contract": model.contract,
        "epoch": epoch,
        "composer": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "history": history,
        "validation_rows": validation_rows,
        "statistics": adapter.statistics.to_json(),
        "local_checkpoint": str(args.local_checkpoint.resolve()),
        "local_checkpoint_sha256": sha256(args.local_checkpoint),
        "initial_field_token_checkpoint": (
            str(args.initial_field_token_checkpoint.resolve())
            if args.initial_field_token_checkpoint is not None else None
        ),
        "initial_field_token_checkpoint_sha256": (
            sha256(args.initial_field_token_checkpoint)
            if args.initial_field_token_checkpoint is not None else None
        ),
        "initial_composer_checkpoint": (
            str(args.initial_composer_checkpoint.resolve())
            if args.initial_composer_checkpoint is not None else None
        ),
        "initial_composer_checkpoint_sha256": (
            sha256(args.initial_composer_checkpoint)
            if args.initial_composer_checkpoint is not None else None
        ),
        "configuration": {
            "core": args.core,
            "halo": args.halo,
            "assembly": args.assembly,
            "token_width": args.token_width,
            "heads": args.heads,
            "communication_depth": args.communication_depth,
            "trace_basis": "constant_linear_tangential",
            "trace_modes": args.trace_modes,
            "trace_code_width": 4 * args.trace_modes,
            "spatial_token_bins": args.spatial_token_bins,
            "boundary_features": "state",
        },
        "method_contract": {
            "local_core": "F(x,b0)",
            "latentddm": "F(x,b0+Delta_b)",
            "pde_residual_used": False,
            "flux_law_used": False,
            "local_bvp_frozen": True,
            "normalized_overlap": True,
            "target_used_at_inference": False,
            "protected_test_cases_opened": 0,
        },
        "args": vars(args),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and args.resume is None:
        raise FileExistsError(args.output_dir)
    if min(
        args.epochs, args.samples_per_epoch, args.batch_size,
        args.validation_batches, args.core,
    ) <= 0 or args.halo < 0 or min(
        args.trace_weight, args.gradient_weight, args.near_boundary_weight,
    ) < 0:
        raise ValueError("invalid Darcy input-correction schedule")
    initializers = (
        args.resume, args.initial_field_token_checkpoint,
        args.initial_composer_checkpoint,
    )
    if sum(value is not None for value in initializers) > 1:
        raise ValueError(
            "resume, field-token initialization, and composer initialization "
            "are mutually exclusive",
        )
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    local_operator, adapter, local_checkpoint = load_frozen_local(
        args.local_checkpoint, device,
    )
    model = DarcyInputCorrectionLatentDDM(
        adapter.spatial_condition_channels + 1,
        adapter.scalar_condition_dim,
        token_width=args.token_width, heads=args.heads,
        communication_depth=args.communication_depth,
        trace_modes=args.trace_modes,
        spatial_token_bins=args.spatial_token_bins,
    ).to(device)
    if args.initial_field_token_checkpoint is not None:
        initialize_from_field_token(model, args.initial_field_token_checkpoint)
    if args.initial_composer_checkpoint is not None:
        source = torch.load(
            args.initial_composer_checkpoint, map_location="cpu",
            weights_only=False,
        )
        if source.get("contract") != model.contract:
            raise ValueError("source composer contract differs")
        if source.get("local_checkpoint_sha256") != sha256(args.local_checkpoint):
            raise ValueError("source composer uses a different frozen local BVP")
        model.load_state_dict(source["composer"], strict=True)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr * .01,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_set = H5DarcyDataset(args.data, "train")
    val_set = H5DarcyDataset(args.data, "validation")
    args.training_subset = apply_training_manifest(
        train_set, args.train_manifest, args.max_train_cases,
    )
    if args.max_train_cases is not None and args.training_subset is None:
        train_set.length = min(train_set.length, args.max_train_cases)
    if args.max_val_cases is not None:
        val_set.length = min(val_set.length, args.max_val_cases)
    if not train_set.has_face_dirichlet or not val_set.has_face_dirichlet:
        raise ValueError("Darcy input correction requires face-resolved data")
    train_loader = make_loader(
        train_set, args.batch_size, True, args.workers, device,
        args.samples_per_epoch,
    )
    val_loader = make_loader(
        val_set, args.batch_size, False, min(args.workers, 2), device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best, history = 0, math.inf, []
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.training_subset is not None:
            validate_resume_protocol(saved.get("args", {}), args)
        if saved.get("contract") != model.contract:
            raise ValueError("resume contract differs")
        if saved.get("local_checkpoint_sha256") != sha256(args.local_checkpoint):
            raise ValueError("resume local BVP checkpoint differs")
        model.load_state_dict(saved["composer"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"])
        history = list(saved["history"])
        best = min(float(row["selection_score"]) for row in history)
    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            model.train(); local_operator.eval()
            totals: dict[str, float] = {}
            batches = 0
            started = time.perf_counter()
            for host in train_loader:
                batch = move_batch(host, device)
                optimizer.zero_grad(set_to_none=True)
                autocast = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else nullcontext()
                )
                with autocast:
                    loss, values = terms(
                        model, local_operator, adapter, batch, args,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite Darcy input-correction objective")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters, 1.0)
                scaler.step(optimizer); scaler.update()
                for name, value in {"loss": loss, **values}.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach())
                batches += 1
            scheduler.step()
            validation, validation_rows = validate(
                model, local_operator, adapter, val_loader, device, args,
            )
            selection = (
                validation["relative_l2"]
                + .10 * math.sqrt(max(validation["gradient"], 0.0))
                + .05 * math.sqrt(max(validation["near_boundary"], 0.0))
            )
            if args.selection_metric == "relative_l2":
                selection = validation["relative_l2"]
            row = {
                "epoch": epoch,
                "seconds": time.perf_counter() - started,
                "lr": optimizer.param_groups[0]["lr"],
                **{
                    f"train_{name}": value / max(batches, 1)
                    for name, value in totals.items()
                },
                **{f"validation_{name}": value for name, value in validation.items()},
                "selection_score": selection,
            }
            history.append(row)
            saved = payload(
                model, optimizer, scheduler, local_checkpoint, adapter, args,
                epoch, history, validation_rows,
            )
            torch.save(saved, args.output_dir / "checkpoint_last.pt")
            if selection < best:
                best = selection
                torch.save(saved, args.output_dir / "checkpoint_best.pt")
            with (args.output_dir / "metrics.csv").open(
                "w", newline="", encoding="utf-8",
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                writer.writeheader(); writer.writerows(history)
            print(json.dumps(row), flush=True)
        if not history:
            raise RuntimeError("Darcy input-correction training produced no epochs")
        selected = min(history, key=lambda row: float(row["selection_score"]))
        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        frozen_local_parameters = sum(
            parameter.numel() for parameter in local_operator.parameters()
        )
        completed_epochs = int(history[-1]["epoch"])
        summary = {
            "contract": "darcy_input_correction_training_summary_v1",
            "model_contract": model.contract,
            "data": str(args.data.resolve()),
            "split": "train/validation",
            "protected_test_cases_opened": 0,
            "train_cases": int(len(train_set)),
            "validation_cases": int(len(val_set)),
            "epochs_completed": completed_epochs,
            "samples_per_epoch": int(args.samples_per_epoch),
            "sample_exposures": completed_epochs * int(args.samples_per_epoch),
            "wall_seconds": float(sum(
                float(row["seconds"]) for row in history
            )),
            "trainable_parameter_count": int(trainable_parameters),
            "frozen_local_parameter_count": int(frozen_local_parameters),
            "best_epoch": int(selected["epoch"]),
            "best_validation_relative_l2": float(
                selected["validation_relative_l2"]
            ),
            "best_selection_score": float(selected["selection_score"]),
            "local_checkpoint": str(args.local_checkpoint.resolve()),
            "local_checkpoint_sha256": sha256(args.local_checkpoint),
            "initial_composer_checkpoint": (
                str(args.initial_composer_checkpoint.resolve())
                if args.initial_composer_checkpoint is not None else None
            ),
            "initial_composer_checkpoint_sha256": (
                sha256(args.initial_composer_checkpoint)
                if args.initial_composer_checkpoint is not None else None
            ),
            "configuration": {
                "core": int(args.core),
                "halo": int(args.halo),
                "assembly": args.assembly,
                "token_width": int(args.token_width),
                "heads": int(args.heads),
                "communication_depth": int(args.communication_depth),
                "trace_modes": int(args.trace_modes),
                "spatial_token_bins": int(args.spatial_token_bins),
            },
            "adaptation_contract": (
                "frozen_local_composer_only"
                if args.initial_composer_checkpoint is not None else
                "source_composer_training"
            ),
            "history": history,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8",
        )
    finally:
        train_set.close(); val_set.close()


if __name__ == "__main__":
    main()
