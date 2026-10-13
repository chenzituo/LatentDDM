"""Phase 2: freeze local/chart and train one residual-port communicator."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from codiffuse_Schwarz.residual_port import (
    ResidualPortSpec, project_residual_ports, scale_port_code,
)
from codiffuse_Schwarz.tile_communication import (
    RelativeTileAttention, ResidualKernelAttention,
)

from .adapter import H5DarcyDataset
from .evaluate import move_batch
from .residual_port_composer import (
    GeometryLinearResidualResponseChart, LinearResidualResponseChart,
    ResidualPortCommunicator,
    ResidualResponseChart, SourceLinearResidualPortCommunicator,
    build_response_chart,
)
from .train import make_loader, seed_everything
from .train_residual_port_chart import (
    CONTRACT as CHART_CONTRACT, FrozenContext, frozen_context,
    interface_losses, load_frozen_local, masked_mse, physics_losses, sha256,
)


Tensor = torch.Tensor
CONTRACT = "darcy_residual_port_one_attention_v1"
DEFECT_SUMMARY_WIDTH = 8
CANONICAL_ANALYZER_CONTRACT = "homogeneous_compatibility_residual_kernel"


def source_linear_signal_width(spec: ResidualPortSpec) -> int:
    """Port defect plus eight homogeneous linear defect moments."""
    return spec.code_width + DEFECT_SUMMARY_WIDTH


def defect_aligned_features(
    context: FrozenContext,
    spec: ResidualPortSpec,
    port_scale: Tensor,
) -> Tensor:
    """Label-free provisional overlap defect plus source/physical-BC summaries."""
    batch = context.target_normalized.shape[0]
    consensus = context.cover.tile(
        context.cover.assemble_weighted(context.local, batch),
    )
    defect = (consensus - context.local) * context.valid
    defect_code = project_residual_ports(
        defect, context.valid, spec,
    ).reshape(batch, context.cover.tile_count, spec.code_width)
    defect_code = scale_port_code(defect_code, port_scale)

    physical = context.physical
    valid = context.valid
    denominator = valid.flatten(2).sum(2).clamp_min(1.0)
    source = physical[:, 2:3]
    source_mean = (source * valid).flatten(2).sum(2) / denominator
    source_rms = (
        (source.square() * valid).flatten(2).sum(2) / denominator
    ).sqrt()
    # Face-aware adapter layout: normalized N/S/W/E values are channels 16:20
    # and physical-role indicators are channels 20:24.
    face_value = physical[:, 16:20]
    physical_role = physical[:, 20:24]
    face_denominator = physical_role.flatten(2).sum(2).clamp_min(1.0)
    face_mean = (
        (face_value * physical_role).flatten(2).sum(2) / face_denominator
    )
    all_denominator = physical_role.flatten(1).sum(1).clamp_min(1.0)
    overall_mean = (
        (face_value * physical_role).flatten(1).sum(1) / all_denominator
    )[:, None]
    overall_rms = (
        (face_value.square() * physical_role).flatten(1).sum(1)
        / all_denominator
    ).sqrt()[:, None]
    summary = torch.cat((
        source_mean, source_rms, face_mean, overall_mean, overall_rms,
    ), dim=1).reshape(batch, context.cover.tile_count, DEFECT_SUMMARY_WIDTH)
    return torch.cat((defect_code, summary), dim=-1)


def source_linear_features(
    context: FrozenContext,
    spec: ResidualPortSpec,
    port_scale: Tensor,
) -> Tensor:
    """Operator-free homogeneous compatibility RHS.

    Every coordinate is a fixed linear functional of the POU consensus defect
    ``R_i A_chi(v)-v_i``.  Therefore a compatible restriction has bitwise-zero
    signal irrespective of nonzero source or physical boundary data.  Those
    known problem terms already affect the frozen local endpoint; they are not
    appended as non-homogeneous pseudo-residual coordinates.
    """
    batch_size = context.local.shape[0] // context.cover.tile_count
    consensus = context.cover.tile(
        context.cover.assemble_weighted(context.local, batch_size),
    )
    defect = (consensus - context.local) * context.valid
    defect_code = scale_port_code(
        project_residual_ports(defect, context.valid, spec).reshape(
            batch_size, context.cover.tile_count, spec.code_width,
        ),
        port_scale,
    )

    coordinate = torch.linspace(
        -1.0, 1.0, context.cover.tile_size,
        device=defect.device, dtype=defect.dtype,
    )
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    basis = torch.stack((
        torch.ones_like(xx), xx, yy, xx * yy,
    ))[None]
    denominator = context.valid.flatten(2).sum(2).clamp_min(1.0)
    volume_moments = (
        defect[:, None] * basis[:, :, None] * context.valid[:, None]
    ).flatten(3).sum(3)[:, :, 0] / denominator
    face_moments = []
    for axis, index in ((-2, 0), (-2, -1), (-1, 0), (-1, -1)):
        face = defect.select(axis, index)
        weight = context.valid.select(axis, index)
        face_moments.append(
            (face * weight).flatten(1).sum(1)
            / weight.flatten(1).sum(1).clamp_min(1.0)
        )
    face_moments = torch.stack(face_moments, 1)
    signal = torch.cat((
        defect_code,
        volume_moments.reshape(batch_size, context.cover.tile_count, 4),
        face_moments.reshape(batch_size, context.cover.tile_count, 4),
    ), -1)
    expected_width = source_linear_signal_width(spec)
    if signal.shape[-1] != expected_width:
        raise RuntimeError(
            f"source-linear analysis width {signal.shape[-1]} != "
            f"checkpoint-derived width {expected_width}"
        )
    return signal


def communicator_features(
    communicator,
    context: FrozenContext,
    batch: dict[str, Tensor],
    spec: ResidualPortSpec,
    scale: Tensor,
    adapter,
) -> Tensor | None:
    mode = getattr(communicator, "analyzer_mode", "generic")
    if mode == "generic":
        return None
    if mode == "defect_aligned":
        return defect_aligned_features(context, spec, scale)
    if mode == "source_linear":
        return source_linear_features(
            context, spec, scale,
        )
    raise ValueError(f"unknown communicator analyzer mode: {mode}")


def load_frozen_chart(
    path: Path, device: torch.device,
) -> tuple[
    ResidualResponseChart | LinearResidualResponseChart | GeometryLinearResidualResponseChart,
    ResidualPortSpec, Tensor, dict[str, Any],
]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CHART_CONTRACT:
        raise ValueError("communicator requires a selected oracle response chart")
    spec = ResidualPortSpec.from_json(checkpoint["port_spec"])
    chart_spec = checkpoint["chart_spec"]
    chart = build_response_chart(
        chart_spec.get("chart_type", "nonlinear_12mode"),
        1, int(chart_spec["physical_channels"]), int(chart_spec["scalar_dim"]),
        spec.code_width,
        response_width=chart_spec.get("response_width"),
        learned_modes=int(chart_spec.get("learned_modes", 12)),
    ).to(device)
    chart.load_state_dict(checkpoint["chart"], strict=True)
    chart.eval()
    for parameter in chart.parameters():
        parameter.requires_grad_(False)
    return chart, spec, checkpoint["port_scale"].to(device), checkpoint


def forward_composer(
    communicator: ResidualPortCommunicator,
    chart: ResidualResponseChart | LinearResidualResponseChart | GeometryLinearResidualResponseChart,
    context: FrozenContext,
    port_scale: Tensor,
    batch: dict[str, Tensor],
    *,
    code_override: Tensor | None = None,
    max_graph_radius: int | None = None,
    aligned_features: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    if code_override is None:
        predicted_code, token = communicator(
            context.local, context.physical, context.scalar, context.valid,
            context.cover.grid_shape, max_graph_radius=max_graph_radius,
            aligned_features=aligned_features,
        )
        used_code = predicted_code
    else:
        used_code = code_override
        predicted_code = torch.zeros_like(code_override)
        token = context.local.new_empty(
            context.target_normalized.shape[0], context.cover.tile_count, 0,
        )
    chart_output = chart(
        context.local, context.physical, context.scalar, context.valid,
        used_code.reshape(-1, port_scale.numel()),
    )
    assembled = context.cover.assemble_weighted(
        chart_output.corrected, context.target_normalized.shape[0],
    ) * batch["mask"]
    value, flux = interface_losses(
        chart_output.corrected, assembled, context, batch,
    )
    detail = {
        "value_interface": value,
        "flux_interface": flux,
        "corrected": chart_output.corrected,
        "token": token,
    }
    return assembled, predicted_code, used_code, detail


def compatibility_cycle_metrics(
    corrected: Tensor,
    context: FrozenContext,
    spec: ResidualPortSpec,
    port_scale: Tensor,
    baseline_q: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Observable real-defect cycle ||q(v+L(C(q)))||/||q(v)||."""
    if baseline_q is None:
        baseline_q = source_linear_features(context, spec, port_scale)
    corrected_q = source_linear_features(
        replace(context, local=corrected), spec, port_scale,
    )
    axes = tuple(range(1, baseline_q.ndim))
    numerator = corrected_q.square().sum(axes)
    denominator = baseline_q.square().sum(axes).clamp_min(1.0e-12)
    ratio_square = numerator / denominator
    return ratio_square.sqrt().mean(), ratio_square.mean()


def training_terms(
    communicator: ResidualPortCommunicator,
    chart: ResidualResponseChart | LinearResidualResponseChart | GeometryLinearResidualResponseChart,
    context: FrozenContext,
    spec: ResidualPortSpec,
    port_scale: Tensor,
    batch: dict[str, Tensor],
    solution_std: float,
    aligned_features: Tensor | None = None,
) -> tuple[dict[str, Tensor], Tensor]:
    oracle_scaled = scale_port_code(context.oracle_code, port_scale)
    assembled, predicted_code, _, detail = forward_composer(
        communicator, chart, context, port_scale, batch,
        aligned_features=aligned_features,
    )
    corrected = detail["corrected"]
    consensus = context.cover.tile(assembled)
    energy, residual = physics_losses(
        assembled, batch, solution_std,
    )
    terms = {
        "code": F.smooth_l1_loss(predicted_code, oracle_scaled),
        "assembled": masked_mse(
            assembled, context.target_normalized, batch["mask"],
        ),
        "local": masked_mse(corrected, context.target_tiles, context.valid),
        "consensus": masked_mse(corrected, consensus, context.valid),
        "energy": energy,
        "residual": residual,
        "value_interface": detail["value_interface"],
        "flux_interface": detail["flux_interface"],
    }
    if getattr(communicator, "analyzer_mode", "generic") == "source_linear":
        ratio, ratio_square = compatibility_cycle_metrics(
            corrected, context, spec, port_scale,
            baseline_q=aligned_features,
        )
        terms["compatibility_cycle_ratio"] = ratio
        terms["compatibility_cycle_relative_square"] = ratio_square
    return terms, assembled


def objective(terms: dict[str, Tensor]) -> Tensor:
    result = (
        terms["assembled"] + 0.5 * terms["code"] + 0.2 * terms["local"]
        + 0.1 * terms["consensus"] + 0.05 * terms["energy"]
        + 0.05 * terms["residual"] + 0.05 * terms["value_interface"]
        + 0.01 * terms["flux_interface"]
    )
    if "compatibility_cycle_relative_square" in terms:
        result = result + 0.1 * torch.relu(
            terms["compatibility_cycle_relative_square"] - 1.0,
        ).square()
    return result


@torch.no_grad()
def validate(
    communicator, chart, local, adapter, loader, device, spec, scale, args,
) -> dict[str, float]:
    communicator.eval()
    totals: dict[str, float] = {}
    count = 0
    for host_batch in loader:
        if count >= args.validation_batches:
            break
        batch = move_batch(host_batch, device)
        context = frozen_context(
            local, adapter, batch, spec, args.core, args.halo,
        )
        aligned = communicator_features(
            communicator, context, batch, spec, scale, adapter,
        )
        terms, _ = training_terms(
            communicator, chart, context, spec, scale, batch,
            adapter.statistics.solution_std,
            aligned,
        )
        values = {"objective": objective(terms), **terms}
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        count += 1
    if count == 0:
        raise RuntimeError("communicator validation produced no batches")
    return {name: value / count for name, value in totals.items()}


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
    parser.add_argument(
        "--analyzer-mode", choices=(
            "generic", "defect_aligned", "source_linear",
        ),
        default="generic",
    )
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initial-checkpoint", type=Path,
        help=(
            "load a pretrained residual-port communicator while resetting "
            "optimizer and schedule for composer-only adaptation"
        ),
    )
    parser.add_argument(
        "--initial-communicator", type=Path,
        help="optional frozen-chart probe-invert pretraining checkpoint",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict[str, Any]:
    starts = (
        args.resume, args.initial_checkpoint, args.initial_communicator,
    )
    if sum(item is not None for item in starts) > 1:
        raise ValueError(
            "resume, initial-checkpoint, and initial-communicator are mutually exclusive"
        )
    if not args.resume and any(
        (args.output_dir / name).exists()
        for name in ("checkpoint.pt", "last_checkpoint.pt", "history.csv")
    ):
        raise FileExistsError(
            f"output directory {args.output_dir} already contains a training run"
        )
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, chart_checkpoint = load_frozen_chart(
        args.chart_checkpoint, device,
    )
    if chart_checkpoint["source_local_sha256"] != sha256(args.local_checkpoint):
        raise ValueError("chart and communicator local checkpoint identities differ")
    if args.analyzer_mode == "source_linear":
        signal_width = source_linear_signal_width(spec)
        communicator = SourceLinearResidualPortCommunicator(
            adapter.spatial_condition_channels + 1,
            adapter.scalar_condition_dim, spec.code_width,
            signal_width,
            token_width=args.token_width, heads=args.heads,
        ).to(device)
    else:
        communicator = ResidualPortCommunicator(
            1, adapter.spatial_condition_channels + 1,
            adapter.scalar_condition_dim, spec.code_width,
            token_width=args.token_width, heads=args.heads,
            aligned_width=(
                spec.code_width + DEFECT_SUMMARY_WIDTH
                if args.analyzer_mode == "defect_aligned" else 0
            ),
        ).to(device)
        communicator.analyzer_mode = args.analyzer_mode
    attention_count = sum(
        isinstance(module, (
            RelativeTileAttention, ResidualKernelAttention,
        )) for module in communicator.modules()
    )
    if attention_count != 1:
        raise RuntimeError("production communicator must contain exactly one attention")
    if args.initial_communicator is not None:
        initial = torch.load(
            args.initial_communicator, map_location="cpu", weights_only=False,
        )
        if initial.get("contract") != "darcy_compatibility_inverse_probe_v1":
            raise ValueError("initial communicator is not probe-invert pretraining")
        if initial.get("source_local_sha256") != sha256(args.local_checkpoint):
            raise ValueError("probe pretraining local checkpoint identity differs")
        if initial.get("source_chart_sha256") != sha256(args.chart_checkpoint):
            raise ValueError("probe pretraining chart checkpoint identity differs")
        initial_spec = initial.get("communicator_spec", {})
        if int(initial_spec.get("code_width", -1)) != spec.code_width:
            raise ValueError("probe pretraining port degree/width differs")
        communicator.load_state_dict(initial["communicator"], strict=True)
    if args.initial_checkpoint is not None:
        initial = torch.load(
            args.initial_checkpoint, map_location="cpu", weights_only=False,
        )
        if initial.get("contract") != CONTRACT:
            raise ValueError("initial checkpoint is not a residual-port communicator")
        if initial.get("source_local_sha256") != sha256(args.local_checkpoint):
            raise ValueError("adaptation checkpoint local endpoint differs")
        if initial.get("source_chart_sha256") != sha256(args.chart_checkpoint):
            raise ValueError("adaptation checkpoint response chart differs")
        initial_spec = initial.get("communicator_spec", {})
        if initial_spec.get("analyzer_mode", "generic") != args.analyzer_mode:
            raise ValueError("adaptation checkpoint analyzer mode differs")
        communicator.load_state_dict(initial["communicator"], strict=True)
    train_set = H5DarcyDataset(args.data, "train")
    val_set = H5DarcyDataset(args.data, "validation")
    if args.max_train_cases is not None:
        if args.max_train_cases <= 0:
            raise ValueError("max-train-cases must be positive")
        train_set.length = min(train_set.length, args.max_train_cases)
    if args.max_val_cases is not None:
        if args.max_val_cases <= 0:
            raise ValueError("max-val-cases must be positive")
        val_set.length = min(val_set.length, args.max_val_cases)
    if not train_set.has_face_dirichlet or not val_set.has_face_dirichlet:
        raise ValueError("communicator requires face-resolved train/validation")
    train_loader = make_loader(
        train_set, args.batch_size, True, args.workers, device,
        args.samples_per_epoch,
    )
    val_loader = make_loader(
        val_set, args.batch_size, False, min(args.workers, 2), device,
    )
    optimizer = torch.optim.AdamW(
        communicator.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=1.0e-6,
    )
    start_epoch, best, history = 0, math.inf, []
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("contract") != CONTRACT:
            raise ValueError("resume checkpoint is not a residual-port communicator")
        saved_spec = saved.get("communicator_spec", {})
        if saved_spec.get("analyzer_mode", "generic") != args.analyzer_mode:
            raise ValueError("resume analyzer mode differs")
        if saved.get("source_local_sha256") != sha256(args.local_checkpoint):
            raise ValueError("resume local checkpoint identity differs")
        if saved.get("source_chart_sha256") != sha256(args.chart_checkpoint):
            raise ValueError("resume response chart identity differs")
        communicator.load_state_dict(saved["communicator"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"])
        history = list(saved.get("history", []))
        best = min((float(row["selection_score"]) for row in history), default=best)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_started = time.perf_counter()
        communicator.train(); chart.eval(); local.eval()
        totals: dict[str, float] = {}
        count = 0
        for host_batch in train_loader:
            batch = move_batch(host_batch, device)
            context = frozen_context(
                local, adapter, batch, spec, args.core, args.halo,
            )
            aligned = communicator_features(
                communicator, context, batch, spec, scale, adapter,
            )
            optimizer.zero_grad(set_to_none=True)
            terms, _ = training_terms(
                communicator, chart, context, spec, scale, batch,
                adapter.statistics.solution_std,
                aligned,
            )
            loss = objective(terms)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(communicator.parameters(), 1.0)
            optimizer.step()
            for name, value in {"objective": loss, **terms}.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach())
            count += 1
        scheduler.step()
        validation = validate(
            communicator, chart, local, adapter, val_loader, device,
            spec, scale, args,
        )
        cycle_violation = max(
            validation.get("compatibility_cycle_relative_square", 0.0) - 1.0,
            0.0,
        )
        score = (
            validation["assembled"] + 0.1 * validation["residual"]
            + 0.05 * validation["energy"] + 1000.0 * cycle_violation
        )
        row = {
            "epoch": epoch, "selection_score": score,
            "seconds": time.perf_counter() - epoch_started,
            "validation_compatibility_cycle_nonregression": cycle_violation == 0.0,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{name}": value / max(count, 1) for name, value in totals.items()},
            **{f"validation_{name}": value for name, value in validation.items()},
        }
        history.append(row)
        payload = {
            "contract": CONTRACT,
            "phase": "communicator_only_after_frozen_chart",
            "epoch": epoch,
            "communicator": communicator.state_dict(),
            "chart": chart.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "statistics": adapter.statistics.to_json(),
            "port_spec": spec.to_json(),
            "port_scale": scale.detach().cpu(),
            "source_local_checkpoint": str(args.local_checkpoint.resolve()),
            "source_local_sha256": sha256(args.local_checkpoint),
            "source_chart_checkpoint": str(args.chart_checkpoint.resolve()),
            "source_chart_sha256": sha256(args.chart_checkpoint),
            "initial_probe_checkpoint": (
                str(args.initial_communicator.resolve())
                if args.initial_communicator is not None else None
            ),
            "initial_adaptation_checkpoint": (
                str(args.initial_checkpoint.resolve())
                if args.initial_checkpoint is not None else None
            ),
            "chart_spec": chart_checkpoint["chart_spec"],
            "communicator_spec": {
                "physical_channels": adapter.spatial_condition_channels + 1,
                "scalar_dim": adapter.scalar_condition_dim,
                "token_width": args.token_width,
                "heads": args.heads,
                "relative_attention_count": 1,
                "absolute_tile_identifier": False,
                "analyzer_mode": args.analyzer_mode,
                "canonical_analyzer_contract": (
                    CANONICAL_ANALYZER_CONTRACT
                    if args.analyzer_mode == "source_linear" else args.analyzer_mode
                ),
                "legacy_internal_alias": (
                    "source_linear" if args.analyzer_mode == "source_linear" else None
                ),
                "compatibility_residual_sign": (
                    "q=d=P_chi(v)-v" if args.analyzer_mode == "source_linear" else None
                ),
                "q_d_coordinates": [],
                "end_to_end_source_linearity_claimed": False,
                "aligned_width": communicator.aligned_width,
                "source_linear_signal_width": (
                    source_linear_signal_width(spec)
                    if args.analyzer_mode == "source_linear" else 0
                ),
                "qk_geometry_only": args.analyzer_mode == "source_linear",
                "bias_free_linear_value_and_output": (
                    args.analyzer_mode == "source_linear"
                ),
                "shared_residual_kernel_code_map": (
                    args.analyzer_mode == "source_linear"
                ),
                "zero_signal_zero_code_bitwise": (
                    args.analyzer_mode == "source_linear"
                ),
                "signal_definition": (
                    "homogeneous POU consensus defect: port projection plus "
                    "eight fixed linear defect moments"
                    if args.analyzer_mode == "source_linear" else None
                ),
                "raw_source_or_boundary_coordinates": False,
                "compatible_endpoint_signal_zero_up_to_roundoff": (
                    args.analyzer_mode == "source_linear"
                ),
                "coordinate_roles": ["compatibility"] * spec.code_width,
                "qoi_coordinate_count": 0,
            },
            "trainable_groups": ["analyzer", "relative_attention", "code_head"],
            "frozen_groups": ["local", "response_chart"],
            "training_splits": ["train", "validation"],
            "test_subsets_opened": False,
            "local_calls_per_case": "one batched call; local_nfe=tile_count",
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
        "contract": CONTRACT, "best_selection_score": best,
        "relative_attention_count": 1, "test_subsets_opened": False,
        "adaptation_contract": (
            "composer_only; local endpoint and response chart frozen"
        ),
        "initial_checkpoint": (
            str(args.initial_checkpoint.resolve())
            if args.initial_checkpoint is not None else None
        ),
        "train_cases": len(train_set),
        "validation_cases": len(val_set),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in communicator.parameters()
            if parameter.requires_grad
        ),
        "wall_seconds": time.perf_counter() - training_started,
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0
        ),
        "optimizer_state_bytes": sum(
            value.numel() * value.element_size()
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value)
        ),
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, default=str))
