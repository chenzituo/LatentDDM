"""Phase 1: train only the oracle residual-port response chart."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from codiffuse_Schwarz.residual_port import (
    ResidualPortSpec, fit_zero_preserving_scale, project_residual_ports,
    scale_port_code,
)
from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import (
    DarcyStatistics, FaceDarcyAdapter, H5DarcyDataset, MODEL_SPEC,
    build_model, raw_face_darcy_residual, tiled_sni_conditions,
)
from .evaluate import move_batch
from .residual_port_composer import (
    GeometryLinearResidualResponseChart, LinearResidualResponseChart,
    ResidualResponseChart, build_response_chart,
)
from .train import make_loader, seed_everything
from .train_face_local import CONTRACT as LOCAL_CONTRACT


Tensor = torch.Tensor
CONTRACT = "darcy_residual_port_oracle_chart_v1"


@dataclass(frozen=True)
class FrozenContext:
    local: Tensor
    physical: Tensor
    scalar: Tensor
    valid: Tensor
    target_normalized: Tensor
    target_tiles: Tensor
    oracle_code: Tensor
    cover: CartesianCover


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_local(
    path: Path, device: torch.device,
) -> tuple[torch.nn.Module, FaceDarcyAdapter, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != LOCAL_CONTRACT:
        raise ValueError("residual-port stages require the frozen face-local contract")
    adapter = FaceDarcyAdapter(DarcyStatistics.from_json(checkpoint["statistics"]))
    local = build_model(adapter, communication_stage="none").to(device)
    local.load_state_dict(checkpoint["model"], strict=True)
    local.eval()
    for parameter in local.parameters():
        parameter.requires_grad_(False)
    return local, adapter, checkpoint


@torch.no_grad()
def frozen_context(
    local: torch.nn.Module,
    adapter: FaceDarcyAdapter,
    batch: dict[str, Tensor],
    spec: ResidualPortSpec,
    core: int,
    halo: int,
    cover: CartesianCover | None = None,
) -> FrozenContext:
    if cover is None:
        cover = CartesianCover(
            batch["solution"].shape[-2], batch["solution"].shape[-1], core, halo,
        )
    zero = torch.zeros_like(batch["solution"])
    physical, scalar = tiled_sni_conditions(adapter, batch, zero, cover)
    local = local_model_call(local, physical, scalar, cover)
    cover_valid = cover.valid_tiles(
        batch["solution"].shape[0], local.device, local.dtype,
    )
    valid = cover.tile(batch["mask"].to(local.dtype)) * cover_valid
    local = local * valid
    target_normalized = (
        batch["solution"] / adapter.statistics.solution_std
    ) * batch["mask"]
    target_tiles = cover.tile(target_normalized) * valid
    residual = target_tiles - local
    oracle = project_residual_ports(residual, valid, spec).reshape(
        batch["solution"].shape[0], cover.tile_count, spec.code_width,
    )
    return FrozenContext(
        local, physical, scalar, valid, target_normalized, target_tiles,
        oracle, cover,
    )


def local_model_call(
    local: torch.nn.Module, physical: Tensor, scalar: Tensor,
    cover: CartesianCover,
) -> Tensor:
    """One batched frozen-local call, isolated for auditable call counting."""
    return local(
        physical, scalar, cover.tile_count, tile_grid_shape=cover.grid_shape,
    )


def masked_mse(prediction: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    return ((prediction - target).square() * weight).sum() \
        / weight.sum().clamp_min(1.0)


def interface_losses(
    corrected: Tensor,
    assembled: Tensor,
    context: FrozenContext,
    batch: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    """Differentiable overlap value and permeability-weighted flux defects."""
    consensus = context.cover.tile(assembled)
    halo, core = context.cover.halo, context.cover.core
    owned = torch.zeros_like(context.valid)
    owned[..., halo:halo + core, halo:halo + core] = 1.0
    trace = context.valid * (1.0 - owned)
    value = masked_mse(corrected, consensus, trace)
    dx = 2.0 / max(context.cover.height, context.cover.width)
    local_dx = (corrected[..., 2:] - corrected[..., :-2]) / (2.0 * dx)
    global_dx = (consensus[..., 2:] - consensus[..., :-2]) / (2.0 * dx)
    local_dy = (corrected[..., 2:, :] - corrected[..., :-2, :]) / (2.0 * dx)
    global_dy = (consensus[..., 2:, :] - consensus[..., :-2, :]) / (2.0 * dx)
    permeability = torch.pow(
        10.0, context.cover.tile(batch["logk"].to(corrected.dtype)).clamp(-6, 6),
    )
    flux_x = permeability[..., 1:-1] * (local_dx - global_dx)
    flux_y = permeability[..., 1:-1, :] * (local_dy - global_dy)
    weight_x = trace[..., 1:-1]
    weight_y = trace[..., 1:-1, :]
    flux = (
        (flux_x.square() * weight_x).sum()
        + (flux_y.square() * weight_y).sum()
    ) / (weight_x.sum() + weight_y.sum()).clamp_min(1.0)
    return value, flux


def physics_losses(
    assembled_normalized: Tensor,
    batch: dict[str, Tensor],
    solution_std: float,
) -> tuple[Tensor, Tensor]:
    """Exact face-FV energy-error and residual diagnostics/losses."""
    mask = batch["mask"].to(assembled_normalized.dtype)
    prediction = assembled_normalized * solution_std * mask
    target = batch["solution"].to(prediction.dtype)
    coefficient = torch.pow(10.0, batch["logk"].to(prediction.dtype).clamp(-6, 6))
    residual = raw_face_darcy_residual(
        prediction, coefficient, batch["source"].to(prediction.dtype), mask,
        batch["dirichlet_face_mask"], batch["dirichlet_face_value"],
    )
    rhs = -raw_face_darcy_residual(
        torch.zeros_like(prediction), coefficient,
        batch["source"].to(prediction.dtype), mask,
        batch["dirichlet_face_mask"], batch["dirichlet_face_value"],
    )
    residual_loss = (residual.square() * mask).sum() \
        / (rhs.square() * mask).sum().clamp_min(1.0e-12)
    error = (prediction - target) * mask
    homogeneous = raw_face_darcy_residual(
        error, coefficient, torch.zeros_like(error), mask,
        batch["dirichlet_face_mask"],
        torch.zeros_like(batch["dirichlet_face_value"]),
    )
    target_homogeneous = raw_face_darcy_residual(
        target * mask, coefficient, torch.zeros_like(target), mask,
        batch["dirichlet_face_mask"],
        torch.zeros_like(batch["dirichlet_face_value"]),
    )
    energy = (error * homogeneous).sum() \
        / (target * target_homogeneous).sum().clamp_min(1.0e-12)
    return energy.clamp_min(0.0), residual_loss


def chart_terms(
    chart: ResidualResponseChart | LinearResidualResponseChart | GeometryLinearResidualResponseChart,
    context: FrozenContext,
    port_scale: Tensor,
    batch_data: dict[str, Tensor],
    solution_std: float,
) -> tuple[dict[str, Tensor], Tensor]:
    batch = context.target_normalized.shape[0]
    scaled = scale_port_code(context.oracle_code, port_scale).reshape(
        -1, port_scale.numel(),
    )
    output = chart(
        context.local, context.physical, context.scalar, context.valid, scaled,
    )
    assembled = context.cover.assemble_weighted(output.corrected, batch)
    mask = (context.target_normalized != 0).to(assembled.dtype)
    # The true domain mask is exactly represented by tiled validity; reconstruct
    # it without relying on nonzero target values.
    mask = context.cover.assemble_uniform(context.valid, batch) > 0.5
    assembled = assembled * mask
    consensus = context.cover.tile(assembled)
    value, flux = interface_losses(
        output.corrected, assembled, context, batch_data,
    )
    energy, residual = physics_losses(
        assembled, batch_data, solution_std,
    )
    baseline_assembled = context.cover.assemble_weighted(
        context.local, batch,
    ) * batch_data["mask"]
    baseline_value, baseline_flux = interface_losses(
        context.local, baseline_assembled, context, batch_data,
    )
    baseline_energy, baseline_residual = physics_losses(
        baseline_assembled, batch_data, solution_std,
    )
    terms = {
        "local": masked_mse(
            output.corrected, context.target_tiles, context.valid,
        ),
        "assembled": masked_mse(
            assembled, context.target_normalized, mask,
        ),
        "consensus": masked_mse(
            output.corrected, consensus, context.valid,
        ),
        "coefficient": output.coefficients.square().mean(),
        "energy": energy,
        "residual": residual,
        "value_interface": value,
        "flux_interface": flux,
        "baseline_energy": baseline_energy.detach(),
        "baseline_residual": baseline_residual.detach(),
        "baseline_value_interface": baseline_value.detach(),
        "baseline_flux_interface": baseline_flux.detach(),
    }
    return terms, assembled


def chart_objective(
    terms: dict[str, Tensor],
    *,
    feasibility_weight: float = 1.0,
    physics_ratio_tolerance: float = 1.01,
) -> Tensor:
    if feasibility_weight < 0.0 or physics_ratio_tolerance < 1.0:
        raise ValueError("invalid feasibility objective controls")
    ratios = {
        name: terms[name] / terms[f"baseline_{name}"].clamp_min(1.0e-8)
        for name in (
            "energy", "residual", "value_interface", "flux_interface",
        )
    }
    hinge = sum(
        torch.relu(ratio - physics_ratio_tolerance).square()
        for ratio in ratios.values()
    )
    return (
        terms["assembled"] + 0.5 * terms["local"]
        + 0.1 * terms["consensus"] + 1.0e-4 * terms["coefficient"]
        + 0.05 * terms["energy"] + 0.05 * terms["residual"]
        + 0.05 * terms["value_interface"] + 0.01 * terms["flux_interface"]
        + feasibility_weight * hinge
    )


def assert_finite_chart_step(
    chart: torch.nn.Module,
    context: FrozenContext,
    port_scale: Tensor,
    terms: dict[str, Tensor],
    loss: Tensor,
    *,
    include_gradients: bool = False,
) -> None:
    """Fail before optimizer mutation at the first non-finite chart quantity."""
    tensors = {
        "frozen_local": context.local,
        "physical": context.physical,
        "scalar": context.scalar,
        "valid": context.valid,
        "target": context.target_normalized,
        "oracle_code": context.oracle_code,
        "port_scale": port_scale,
        "objective": loss,
        **{f"term/{name}": value for name, value in terms.items()},
        **{f"parameter/{name}": value for name, value in chart.named_parameters()},
    }
    if include_gradients:
        tensors.update({
            f"gradient/{name}": value.grad
            for name, value in chart.named_parameters() if value.grad is not None
        })
    invalid = [
        name for name, value in tensors.items()
        if value is not None and not torch.isfinite(value).all()
    ]
    if invalid:
        raise FloatingPointError(
            "non-finite response-chart state before optimizer step: "
            + ", ".join(invalid)
        )


@torch.no_grad()
def fit_scale(
    local: torch.nn.Module,
    adapter: FaceDarcyAdapter,
    loader,
    device: torch.device,
    spec: ResidualPortSpec,
    core: int,
    halo: int,
    maximum_cases: int,
) -> Tensor:
    codes = []
    seen = 0
    for host_batch in loader:
        if seen >= maximum_cases:
            break
        batch = move_batch(host_batch, device)
        context = frozen_context(local, adapter, batch, spec, core, halo)
        codes.append(context.oracle_code.reshape(-1, spec.code_width).cpu())
        seen += batch["solution"].shape[0]
    if not codes:
        raise RuntimeError("port-scale fit saw no training cases")
    matrix = torch.cat(codes, 0)
    preliminary = fit_zero_preserving_scale(matrix, floor=1.0e-12)
    supported = preliminary[preliminary > 1.0e-8]
    adaptive_floor = max(
        1.0e-3,
        0.05 * float(supported.median()) if supported.numel() else 1.0e-3,
    )
    return fit_zero_preserving_scale(matrix, floor=adaptive_floor).to(device)


@torch.no_grad()
def validate(
    chart: ResidualResponseChart | LinearResidualResponseChart | GeometryLinearResidualResponseChart,
    local: torch.nn.Module,
    adapter: FaceDarcyAdapter,
    loader,
    device: torch.device,
    spec: ResidualPortSpec,
    port_scale: Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    chart.eval()
    totals: dict[str, float] = {}
    count = 0
    for host_batch in loader:
        if count >= args.validation_batches:
            break
        batch = move_batch(host_batch, device)
        context = frozen_context(
            local, adapter, batch, spec, args.core, args.halo,
        )
        terms, _ = chart_terms(
            chart, context, port_scale, batch, adapter.statistics.solution_std,
        )
        values = {
            "objective": chart_objective(
                terms, feasibility_weight=args.feasibility_weight,
                physics_ratio_tolerance=args.physics_ratio_tolerance,
            ),
            **terms,
        }
        for name in (
            "energy", "residual", "value_interface", "flux_interface",
        ):
            values[f"ratio_{name}"] = (
                terms[name] / terms[f"baseline_{name}"].clamp_min(1.0e-8)
            )
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        count += 1
    if count == 0:
        raise RuntimeError("chart validation produced no batches")
    return {name: value / count for name, value in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--scale-cases", type=int, default=512)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument(
        "--chart-type", choices=(
            "nonlinear_12mode", "tied_linear_green",
            "geometry_tied_linear_green",
        ),
        default="nonlinear_12mode",
    )
    parser.add_argument(
        "--response-width", type=int,
        help="defaults to width 64 for both matched chart types",
    )
    parser.add_argument("--learned-modes", type=int, default=12)
    parser.add_argument("--token-width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--feasibility-weight", type=float, default=1.0)
    parser.add_argument("--physics-ratio-tolerance", type=float, default=1.01)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    local, adapter, local_checkpoint = load_frozen_local(
        args.local_checkpoint, device,
    )
    train_set = H5DarcyDataset(args.data, "train")
    val_set = H5DarcyDataset(args.data, "validation")
    if not train_set.has_face_dirichlet or not val_set.has_face_dirichlet:
        raise ValueError("residual-port chart requires face-resolved train/validation")
    train_loader = make_loader(
        train_set, args.batch_size, True, args.workers, device,
        args.samples_per_epoch,
    )
    scale_loader = make_loader(
        train_set, args.batch_size, False, min(args.workers, 2), device,
    )
    val_loader = make_loader(
        val_set, args.batch_size, False, min(args.workers, 2), device,
    )
    spec = ResidualPortSpec(2, 1, polynomial_degree=args.degree)
    if args.degree == 3 and spec.code_width != 16:
        raise RuntimeError("2D scalar degree-3 trace contract must have width 16")
    port_scale = fit_scale(
        local, adapter, scale_loader, device, spec, args.core, args.halo,
        args.scale_cases,
    )
    if not torch.isfinite(port_scale).all() or torch.any(port_scale <= 0):
        raise FloatingPointError("train-only residual-port scale is not finite/positive")
    chart = build_response_chart(
        args.chart_type,
        1, adapter.spatial_condition_channels + 1, adapter.scalar_condition_dim,
        spec.code_width, response_width=args.response_width,
        learned_modes=args.learned_modes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        chart.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1.0e-6,
    )
    start_epoch, best, history = 0, math.inf, []
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("contract") != CONTRACT:
            raise ValueError("resume checkpoint is not an oracle response chart")
        saved_type = saved.get("chart_spec", {}).get(
            "chart_type", "nonlinear_12mode",
        )
        if saved_type != args.chart_type:
            raise ValueError("resume response-chart type differs")
        if saved["source_local_sha256"] != sha256(args.local_checkpoint):
            raise ValueError("resume local checkpoint identity differs")
        torch.testing.assert_close(saved["port_scale"], port_scale.cpu())
        chart.load_state_dict(saved["chart"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"])
        history = list(saved.get("history", []))
        best = min((float(row["selection_score"]) for row in history), default=best)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        chart.train()
        totals: dict[str, float] = {}
        count = 0
        for host_batch in train_loader:
            batch = move_batch(host_batch, device)
            context = frozen_context(
                local, adapter, batch, spec, args.core, args.halo,
            )
            optimizer.zero_grad(set_to_none=True)
            terms, _ = chart_terms(
                chart, context, port_scale, batch,
                adapter.statistics.solution_std,
            )
            loss = chart_objective(
                terms, feasibility_weight=args.feasibility_weight,
                physics_ratio_tolerance=args.physics_ratio_tolerance,
            )
            assert_finite_chart_step(chart, context, port_scale, terms, loss)
            loss.backward()
            assert_finite_chart_step(
                chart, context, port_scale, terms, loss,
                include_gradients=True,
            )
            torch.nn.utils.clip_grad_norm_(chart.parameters(), 1.0)
            optimizer.step()
            for name, value in {"objective": loss, **terms}.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            count += 1
        scheduler.step()
        validation = validate(
            chart, local, adapter, val_loader, device, spec, port_scale, args,
        )
        feasibility_violation = sum(
            max(
                validation[f"ratio_{name}"] - args.physics_ratio_tolerance,
                0.0,
            )
            for name in (
                "energy", "residual", "value_interface", "flux_interface",
            )
        )
        feasible = feasibility_violation <= 1.0e-8
        score = (
            validation["assembled"] + 0.25 * validation["local"]
            + 1000.0 * feasibility_violation
        )
        row = {
            "epoch": epoch, "selection_score": score,
            "validation_feasible": feasible,
            "validation_feasibility_violation": feasibility_violation,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{name}": value / max(count, 1) for name, value in totals.items()},
            **{f"validation_{name}": value for name, value in validation.items()},
        }
        history.append(row)
        payload = {
            "contract": CONTRACT,
            "phase": "oracle_response_chart_only",
            "epoch": epoch,
            "chart": chart.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "statistics": adapter.statistics.to_json(),
            "port_spec": spec.to_json(),
            "port_scale": port_scale.detach().cpu(),
            "scale_fit_split": "train",
            "source_local_checkpoint": str(args.local_checkpoint.resolve()),
            "source_local_contract": local_checkpoint.get("contract"),
            "source_local_sha256": sha256(args.local_checkpoint),
            "chart_spec": {
                "chart_type": chart.chart_type,
                "physical_channels": adapter.spatial_condition_channels + 1,
                "scalar_dim": adapter.scalar_condition_dim,
                "learned_modes": args.learned_modes,
                "response_width": chart.response_width,
                "parameter_count": sum(
                    parameter.numel() for parameter in chart.parameters()
                ),
                "linear_in_port_code": (
                    chart.chart_type in (
                        "tied_linear_green", "geometry_tied_linear_green",
                    )
                ),
                "one_mode_per_port_coordinate": (
                    chart.chart_type in (
                        "tied_linear_green", "geometry_tied_linear_green",
                    )
                ),
                "geometry_coefficient_only_modes": (
                    chart.chart_type == "geometry_tied_linear_green"
                ),
                "coordinate_roles": ["compatibility"] * spec.code_width,
                "qoi_coordinate_count": 0,
                "zero_code_unchanged_endpoint_bitwise": True,
            },
            "feasibility_contract": {
                "metrics": [
                    "energy", "residual", "value_interface", "flux_interface",
                ],
                "ratio_tolerance": args.physics_ratio_tolerance,
                "hinge_weight": args.feasibility_weight,
                "selection": "field among feasible; 1000x violation otherwise",
            },
            "trainable_groups": ["chart"],
            "frozen_groups": ["local", "communicator_absent"],
            "training_splits": ["train", "validation"],
            "test_subsets_opened": False,
            "args": vars(args),
            "history": history,
        }
        torch.save(payload, args.output_dir / "last_checkpoint.pt")
        if score < best:
            best = score
            torch.save(payload, args.output_dir / "checkpoint.pt")
        print(json.dumps(row, sort_keys=True), flush=True)
    with (args.output_dir / "history.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    result = {
        "contract": CONTRACT,
        "best_selection_score": best,
        "port_width": spec.code_width,
        "test_subsets_opened": False,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, default=str))
