"""Train stacked Darcy field-token communication with disjoint core writeback.

The frozen local neural operator and frozen local response chart are evaluated
without any discrete Darcy operator in the training objective.  The composer
receives only local inputs and completed endpoint fields; target solutions are
used as supervision, never as an inference input.
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
import torch.nn.functional as F

from codiffuse_Schwarz.residual_port import scale_port_code

from .adapter import H5DarcyDataset
from .evaluate import move_batch, relative_l2
from .field_token_composer import DarcyFieldTokenComposer
from .train import make_loader, seed_everything
from .train_residual_port_chart import (
    FrozenContext,
    frozen_context,
    load_frozen_local,
    masked_mse,
    sha256,
)
from .train_residual_port_communicator import load_frozen_chart


Tensor = torch.Tensor
CONTRACT = DarcyFieldTokenComposer.contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--max-train-cases", type=int)
    parser.add_argument("--max-val-cases", type=int)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--token-width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--communication-depth", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--spatial-port-bins", type=int, choices=(0, 2, 3, 4), default=0)
    parser.add_argument(
        "--training-phase", choices=("all", "port_extension"), default="all",
    )
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--code-weight", type=float, default=0.1)
    parser.add_argument("--tile-weight", type=float, default=0.25)
    parser.add_argument("--trace-weight", type=float, default=0.05)
    parser.add_argument(
        "--seam-weight", type=float, default=0.0,
        help=(
            "weight on target-relative error jumps across owned-core "
            "interfaces; this regularizes the decoded chart without overlap "
            "averaging or a discrete PDE operator"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initial-composer-checkpoint", type=Path,
        help=(
            "initialize only the field-token composer weights from a source "
            "domain checkpoint; optimizer, scheduler, and epoch are reset"
        ),
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _owned_mask(context: FrozenContext) -> Tensor:
    owned = torch.zeros_like(context.valid)
    halo, core = context.cover.halo, context.cover.core
    owned[..., halo:halo + core, halo:halo + core] = 1.0
    return owned * context.valid


def _trace_mask(context: FrozenContext) -> Tensor:
    return context.valid - _owned_mask(context)


def seam_error_mse(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    core: int,
) -> Tensor:
    """Squared jump of the prediction error across owned-core interfaces.

    Neighboring grid samples need not have equal physical values, so forcing
    the prediction itself to be constant across a seam would be biased.  The
    target-relative error should, however, remain smooth.  This diagnostic and
    training term therefore compares adjacent *errors* on the two sides of
    every disjoint-core boundary.  It uses supervision only during training
    and introduces neither overlap blending nor a PDE solve at inference.
    """

    if core <= 0:
        raise ValueError("core must be positive")
    error = prediction - target
    numerator = prediction.new_zeros(())
    denominator = prediction.new_zeros(())
    height, width = prediction.shape[-2:]
    for boundary in range(core, width, core):
        valid = mask[..., boundary] * mask[..., boundary - 1]
        jump = error[..., boundary] - error[..., boundary - 1]
        numerator = numerator + (jump.square() * valid).sum()
        denominator = denominator + valid.sum()
    for boundary in range(core, height, core):
        valid = mask[..., boundary, :] * mask[..., boundary - 1, :]
        jump = error[..., boundary, :] - error[..., boundary - 1, :]
        numerator = numerator + (jump.square() * valid).sum()
        denominator = denominator + valid.sum()
    return numerator / denominator.clamp_min(1.0)


def terms(
    composer: DarcyFieldTokenComposer,
    chart: torch.nn.Module,
    context: FrozenContext,
    batch: dict[str, Tensor],
    port_scale: Tensor,
) -> tuple[dict[str, Tensor], Tensor]:
    output = composer(chart, context, batch["mask"])
    target_code = scale_port_code(context.oracle_code, port_scale)
    consensus = context.cover.tile(output.assembled)
    values = {
        "field": masked_mse(
            output.assembled, context.target_normalized, batch["mask"],
        ),
        "owned_tile": masked_mse(
            output.corrected_tiles, context.target_tiles, _owned_mask(context),
        ),
        "trace_consistency": masked_mse(
            output.corrected_tiles, consensus, _trace_mask(context),
        ),
        "seam_error": seam_error_mse(
            output.assembled, context.target_normalized, batch["mask"],
            context.cover.core,
        ),
        "code": F.smooth_l1_loss(
            output.latent.correction_code, target_code,
        ),
        "code_square": output.latent.correction_code.square().mean(),
    }
    return values, output.assembled


def objective(values: dict[str, Tensor], args: argparse.Namespace) -> Tensor:
    return (
        values["field"]
        + args.tile_weight * values["owned_tile"]
        + args.trace_weight * values["trace_consistency"]
        + getattr(args, "seam_weight", 0.0) * values["seam_error"]
        + args.code_weight * values["code"]
        + 1.0e-5 * values["code_square"]
    )


@torch.no_grad()
def validate(
    composer: DarcyFieldTokenComposer,
    chart: torch.nn.Module,
    local: torch.nn.Module,
    adapter: Any,
    loader,
    device: torch.device,
    spec: Any,
    port_scale: Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    composer.eval(); chart.eval(); local.eval()
    totals: dict[str, float] = {}
    cases = 0
    batches = 0
    for host in loader:
        if batches >= args.validation_batches:
            break
        batch = move_batch(host, device)
        context = frozen_context(
            local, adapter, batch, spec, args.core, args.halo,
        )
        values, prediction = terms(
            composer, chart, context, batch, port_scale,
        )
        baseline = context.cover.assemble_ras(
            context.local, batch["solution"].shape[0],
        ) * batch["mask"]
        physical_prediction = prediction * adapter.statistics.solution_std
        physical_baseline = baseline * adapter.statistics.solution_std
        field_l2 = relative_l2(
            physical_prediction, batch["solution"], batch["mask"],
        )
        baseline_l2 = relative_l2(
            physical_baseline, batch["solution"], batch["mask"],
        )
        count = int(batch["solution"].shape[0])
        row = {
            "objective": float(objective(values, args)),
            **{name: float(value) for name, value in values.items()},
            "relative_l2": float(field_l2.mean()),
            "local_core_relative_l2": float(baseline_l2.mean()),
        }
        for name, value in row.items():
            totals[name] = totals.get(name, 0.0) + count * value
        cases += count
        batches += 1
    if not cases:
        raise RuntimeError("field-token validation produced no cases")
    return {name: value / cases for name, value in totals.items()}


def _payload(
    composer: DarcyFieldTokenComposer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    chart_checkpoint: dict[str, Any],
    spec: Any,
    scale: Tensor,
    adapter: Any,
    args: argparse.Namespace,
    epoch: int,
    history: list[dict[str, float]],
) -> dict[str, Any]:
    return {
        "contract": CONTRACT,
        "epoch": epoch,
        "composer": composer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "statistics": adapter.statistics.to_json(),
        "port_spec": spec.to_json(),
        "port_scale": scale.detach().cpu(),
        "source_local_checkpoint": str(args.local_checkpoint.resolve()),
        "source_local_sha256": sha256(args.local_checkpoint),
        "source_chart_checkpoint": str(args.chart_checkpoint.resolve()),
        "source_chart_sha256": sha256(args.chart_checkpoint),
        "chart_spec": chart_checkpoint["chart_spec"],
        "composer_spec": {
            "physical_channels": adapter.spatial_condition_channels + 1,
            "scalar_dim": adapter.scalar_condition_dim,
            "code_width": spec.code_width,
            "token_width": args.token_width,
            "heads": args.heads,
            "communication_depth": args.communication_depth,
            "spatial_port_bins": args.spatial_port_bins,
            "training_phase": args.training_phase,
            "token_definition": "z0=C(local_input, executable_local_field)",
            "communication_definition": "z_next=T(z; relative_geometry_graph)",
            "centering": "shared_full_graph_path_minus_diagonal_graph_path",
            "absolute_tile_prompt": False,
            "assembly": "disjoint_owned_core",
        },
        "frozen_groups": ["local_neural_operator", "local_response_chart"],
        "trainable_groups": (
            ["spatial_port_encoder", "spatial_port_latent_stack"]
            if args.training_phase == "port_extension" else
            ["field_token_encoder", "latent_stack", "code_head", "spatial_ports"]
        ),
        "training_objective_uses_discrete_operator": False,
        "model_input_uses_target_solution": False,
        "local_calls_per_case": "one batched call; all subdomains independent",
        "initial_composer_checkpoint": (
            str(args.initial_composer_checkpoint.resolve())
            if args.initial_composer_checkpoint is not None else None
        ),
        "initial_composer_sha256": (
            sha256(args.initial_composer_checkpoint)
            if args.initial_composer_checkpoint is not None else None
        ),
        "args": vars(args),
        "history": history,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    if min(
        args.epochs, args.samples_per_epoch, args.batch_size,
        args.validation_batches, args.core,
    ) <= 0 or args.halo < 0 or args.seam_weight < 0.0:
        raise ValueError("invalid training extent")
    if args.resume is not None and args.initial_composer_checkpoint is not None:
        raise ValueError("resume and initial composer checkpoint are mutually exclusive")
    if not args.resume and any(
        (args.output_dir / name).exists()
        for name in ("checkpoint.pt", "last_checkpoint.pt", "history.csv")
    ):
        raise FileExistsError(f"output directory already contains a run: {args.output_dir}")
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, chart_checkpoint = load_frozen_chart(
        args.chart_checkpoint, device,
    )
    if chart_checkpoint["source_local_sha256"] != sha256(args.local_checkpoint):
        raise ValueError("frozen response chart and endpoint differ")
    composer = DarcyFieldTokenComposer(
        adapter.spatial_condition_channels + 1,
        adapter.scalar_condition_dim,
        spec.code_width,
        token_width=args.token_width,
        heads=args.heads,
        communication_depth=args.communication_depth,
        spatial_port_bins=args.spatial_port_bins,
    ).to(device)
    if args.initial_composer_checkpoint is not None:
        initial = torch.load(
            args.initial_composer_checkpoint, map_location="cpu",
            weights_only=False,
        )
        if initial.get("contract") != CONTRACT:
            raise ValueError("initial composer checkpoint has the wrong contract")
        initial_spec = initial.get("composer_spec", {})
        if int(initial_spec.get("communication_depth", -1)) != args.communication_depth:
            raise ValueError("initial composer communication depth differs")
        if initial.get("source_local_sha256") != sha256(args.local_checkpoint):
            raise ValueError("initial composer endpoint differs")
        if initial.get("source_chart_sha256") != sha256(args.chart_checkpoint):
            raise ValueError("initial composer response chart differs")
        initial_bins = int(initial_spec.get("spatial_port_bins", 0))
        if initial_bins not in (0, args.spatial_port_bins):
            raise ValueError("initial composer spatial ports differ")
        if initial_bins == args.spatial_port_bins:
            composer.load_state_dict(initial["composer"], strict=True)
        else:
            incompatible = composer.load_state_dict(
                initial["composer"], strict=False,
            )
            allowed = (
                "spatial_port_projection.", "spatial_port_scalar.",
                "spatial_port_embedding.", "spatial_port_communication.",
            )
            if incompatible.unexpected_keys or any(
                not name.startswith(allowed) for name in incompatible.missing_keys
            ):
                raise ValueError("unsafe spatial-port parent initialization")
    local.eval().requires_grad_(False)
    chart.eval().requires_grad_(False)
    if args.training_phase == "port_extension":
        if args.spatial_port_bins == 0:
            raise ValueError("port_extension requires spatial ports")
        composer.requires_grad_(False)
        for name, parameter in composer.named_parameters():
            if name.startswith((
                "spatial_port_projection.", "spatial_port_scalar.",
                "spatial_port_embedding.", "spatial_port_communication.",
            )):
                parameter.requires_grad_(True)

    train_set = H5DarcyDataset(args.data, "train")
    val_set = H5DarcyDataset(args.data, "validation")
    if args.max_train_cases is not None:
        train_set.length = min(train_set.length, args.max_train_cases)
    if args.max_val_cases is not None:
        val_set.length = min(val_set.length, args.max_val_cases)
    if not train_set.has_face_dirichlet or not val_set.has_face_dirichlet:
        raise ValueError("field-token training requires face-resolved data")
    train_loader = make_loader(
        train_set, args.batch_size, True, args.workers, device,
        args.samples_per_epoch,
    )
    val_loader = make_loader(
        val_set, args.batch_size, False, min(args.workers, 2), device,
    )
    trainable = [
        parameter for parameter in composer.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1.0e-6,
    )
    start_epoch, best, history = 0, math.inf, []
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("contract") != CONTRACT:
            raise ValueError("resume checkpoint has the wrong contract")
        saved_spec = saved.get("composer_spec", {})
        if int(saved_spec.get("communication_depth", -1)) != args.communication_depth:
            raise ValueError("resume communication depth differs")
        if saved.get("source_local_sha256") != sha256(args.local_checkpoint):
            raise ValueError("resume endpoint differs")
        if saved.get("source_chart_sha256") != sha256(args.chart_checkpoint):
            raise ValueError("resume response chart differs")
        composer.load_state_dict(saved["composer"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"])
        history = list(saved.get("history", []))
        best = min((float(row["selection_score"]) for row in history), default=best)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            epoch_started = time.perf_counter()
            composer.train(); chart.eval(); local.eval()
            totals: dict[str, float] = {}
            batches = 0
            for host in train_loader:
                batch = move_batch(host, device)
                context = frozen_context(
                    local, adapter, batch, spec, args.core, args.halo,
                )
                optimizer.zero_grad(set_to_none=True)
                values, _ = terms(composer, chart, context, batch, scale)
                loss = objective(values, args)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite field-token objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(composer.parameters(), 1.0)
                optimizer.step()
                for name, value in {"objective": loss, **values}.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach())
                batches += 1
            scheduler.step()
            validation = validate(
                composer, chart, local, adapter, val_loader, device,
                spec, scale, args,
            )
            score = validation["relative_l2"]
            row = {
                "epoch": epoch,
                "selection_score": score,
                "seconds": time.perf_counter() - epoch_started,
                "lr": optimizer.param_groups[0]["lr"],
                **{
                    f"train_{name}": value / max(batches, 1)
                    for name, value in totals.items()
                },
                **{f"validation_{name}": value for name, value in validation.items()},
            }
            history.append(row)
            payload = _payload(
                composer, optimizer, scheduler, chart_checkpoint, spec, scale,
                adapter, args, epoch, history,
            )
            torch.save(payload, args.output_dir / "last_checkpoint.pt")
            if score < best:
                best = score
                torch.save(payload, args.output_dir / "checkpoint.pt")
            print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        train_set.close(); val_set.close()

    with (args.output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    result = {
        "contract": CONTRACT,
        "communication_depth": args.communication_depth,
        "best_validation_relative_l2": best,
        "local_core_validation_relative_l2": history[-1][
            "validation_local_core_relative_l2"
        ],
        "training_objective_uses_discrete_operator": False,
        "assembly": "disjoint_owned_core",
        "test_subsets_opened": False,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in composer.parameters()
            if parameter.requires_grad
        ),
        "frozen_local_parameter_count": sum(
            parameter.numel() for parameter in local.parameters()
        ),
        "train_cases": len(train_set),
        "validation_cases": len(val_set),
        "epochs_completed": len(history),
        "samples_per_epoch": args.samples_per_epoch,
        "sample_exposures": len(history) * args.samples_per_epoch,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0
        ),
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, default=str))
