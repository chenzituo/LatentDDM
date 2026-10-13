"""Rollout-aware fine-tuning of the adaptive Cartesian learned projection.

The model architecture is unchanged from the v3 one-step checkpoint. During
training, model-owned L5/L6 states are fed back, the adaptive hierarchy is
redrawn from those states, and overlap plus coarse/fine consensus is applied at
every physical step. This isolates exposure-bias correction from architecture
changes and from physics guidance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

from codiffuse_Schwarz_Airfoil.airfoil_two_level import (
    fluid_divergence_rms,
    project_multiresolution_prediction,
)
from codiffuse_Schwarz_Airfoil.cell_role_metrics import role_vorticity_metrics
from codiffuse_Schwarz_Airfoil.dataset import FieldStatistics
from codiffuse_Schwarz_Airfoil.expanded_protocol import write_or_validate_snapshot
from codiffuse_Schwarz_Airfoil.force_diagnostics import (
    balanced_force_coefficient_relative_square,
    force_coefficient_relative_square,
    penalty_force_coefficients,
)
from codiffuse_Schwarz_Airfoil.interface_trace import (
    interface_trace_relative_square,
)
from codiffuse_Schwarz_Airfoil.interface_operator import (
    clone_interface_trace_checkpoint,
    load_interface_trace_checkpoint,
    upgrade_interface_transport_checkpoint,
    upgrade_interface_strip_checkpoint,
    upgrade_interface_force_checkpoint,
    upgrade_interface_vorticity_checkpoint,
)
from codiffuse_Schwarz_Airfoil.level_specific import (
    clone_geometry_operator_checkpoint,
    clone_geometry_residual_checkpoint,
    load_geometry_specific_checkpoint,
    load_geometry_residual_checkpoint,
)
from codiffuse_Schwarz_Airfoil.multiresolution import (
    AdaptiveGridController,
    decode_composite,
    synchronize_levels,
)
from codiffuse_Schwarz_Airfoil.multiresolution_inference import (
    MultiresolutionPrediction,
    composite_vorticity_relative_l2,
    predict_multiresolution_step,
)
from codiffuse_Schwarz_Airfoil.train_multiresolution_projection import (
    batched,
    multiresolution_losses,
)
from codiffuse_Schwarz_Airfoil.train_multitrajectory_multiresolution import (
    PairedTrajectoryCorpus,
)
from codiffuse_Schwarz_Airfoil.train_one_step_multilevel import (
    interior_vorticity,
    load_pretrained,
)
from codiffuse_Schwarz_Airfoil.trajectory_corpus import write_or_validate_split


Tensor = torch.Tensor
History = dict[str, tuple[Tensor, Tensor]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-in-place", action="store_true")
    parser.add_argument("--trajectory-count", type=int, default=32)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--sequences-per-epoch", type=int, default=96)
    parser.add_argument("--curriculum-horizons", type=int, nargs="+", default=(2, 4, 8))
    parser.add_argument("--bptt-steps", type=int, default=2)
    parser.add_argument("--one-step-fraction", type=float, default=.25)
    parser.add_argument("--lr", type=float, default=1.0e-6)
    parser.add_argument("--min-lr", type=float, default=1.0e-7)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--fine-fraction", type=float, default=.35)
    parser.add_argument("--fine-trust", type=float, default=.5)
    parser.add_argument("--placement-ensemble", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--endpoint-relaxation", type=float, default=1.0)
    parser.add_argument("--history-momentum", type=float, default=0.0)
    parser.add_argument("--enable-boundary-memory", action="store_true")
    parser.add_argument("--geometry-specific", action="store_true")
    parser.add_argument("--geometry-body-frame", action="store_true")
    parser.add_argument("--geometry-residual-adapter", action="store_true")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--geometry-band-cells", type=float, default=8.0)
    parser.add_argument("--geometry-width", type=int, default=64)
    parser.add_argument("--geometry-depth", type=int, default=4)
    parser.add_argument("--geometry-two-sided", action="store_true")
    parser.add_argument("--enable-geometry-latent", action="store_true")
    parser.add_argument("--geometry-latent-width", type=int, default=32)
    parser.add_argument("--geometry-latent-depth", type=int, default=2)
    parser.add_argument("--geometry-latent-only", action="store_true")
    parser.add_argument("--boundary-memory-width", type=int, default=32)
    parser.add_argument("--boundary-memory-depth", type=int, default=3)
    parser.add_argument("--boundary-memory-only", action="store_true")
    parser.add_argument("--enable-domain-boundary-adapter", action="store_true")
    parser.add_argument("--enable-motion-transport-adapter", action="store_true")
    parser.add_argument("--motion-transport-width", type=int, default=32)
    parser.add_argument("--motion-transport-depth", type=int, default=2)
    parser.add_argument("--motion-transport-lr", type=float, default=3.0e-6)
    parser.add_argument("--enable-prescribed-motion-warp", action="store_true")
    parser.add_argument("--prescribed-motion-band-cells", type=float, default=8.0)
    parser.add_argument("--motion-domain-adapters-only", action="store_true")
    parser.add_argument("--domain-boundary-width", type=int, default=32)
    parser.add_argument("--domain-boundary-depth", type=int, default=2)
    parser.add_argument("--domain-boundary-lr", type=float, default=3.0e-5)
    parser.add_argument("--enable-crop-ghost-feature", action="store_true")
    parser.add_argument("--domain-boundary-only", action="store_true")
    parser.add_argument("--enable-rollout-closure", action="store_true")
    parser.add_argument("--rollout-closure-width", type=int, default=32)
    parser.add_argument("--rollout-closure-only", action="store_true")
    parser.add_argument("--enable-curl-memory", action="store_true")
    parser.add_argument("--curl-memory-width", type=int, default=32)
    parser.add_argument("--curl-memory-depth", type=int, default=2)
    parser.add_argument("--curl-memory-only", action="store_true")
    parser.add_argument("--enable-feature-schwarz", action="store_true")
    parser.add_argument("--feature-schwarz-only", action="store_true")
    parser.add_argument("--enable-tile-communication", action="store_true")
    parser.add_argument("--tile-communication-only", action="store_true")
    parser.add_argument("--vorticity-amplitude-fraction", type=float, default=.15)
    parser.add_argument("--coarse-state-weight", type=float, default=1.0)
    parser.add_argument("--fine-state-weight", type=float, default=1.0)
    parser.add_argument("--composite-vorticity-weight", type=float, default=4.0)
    parser.add_argument("--near-body-weight", type=float, default=2.0)
    parser.add_argument("--surface-vorticity-weight", type=float, default=0.0)
    parser.add_argument("--interface-trace-weight", type=float, default=0.0)
    parser.add_argument("--interface-trace-stations", type=int, default=64)
    parser.add_argument("--interface-operator", action="store_true")
    parser.add_argument("--interface-operator-width", type=int, default=64)
    parser.add_argument("--interface-operator-depth", type=int, default=4)
    parser.add_argument("--interface-solid-velocity-scale", type=float, default=0.0)
    parser.add_argument("--interface-vorticity-latent", action="store_true")
    parser.add_argument("--interface-strip-operator", action="store_true")
    parser.add_argument(
        "--interface-strip-transport-operator", action="store_true",
    )
    parser.add_argument("--interface-strip-normal-samples", type=int, default=24)
    parser.add_argument("--interface-operator-only", action="store_true")
    parser.add_argument(
        "--interface-shared-only", action="store_true",
        help=(
            "With an interface checkpoint, train only the wrapped endpoint, "
            "vorticity, and learned-projection dynamics."
        ),
    )
    parser.add_argument(
        "--interface-shared-and-strip", action="store_true",
        help=(
            "Jointly train wrapped endpoint/vorticity/projection dynamics and "
            "the geometry-strip closure of an interface checkpoint."
        ),
    )
    parser.add_argument("--interface-strip-only", action="store_true")
    parser.add_argument("--interface-trace-transport-operator", action="store_true")
    parser.add_argument("--interface-trace-transport-only", action="store_true")
    parser.add_argument("--interface-interior-force-operator", action="store_true")
    parser.add_argument("--interface-interior-force-only", action="store_true")
    parser.add_argument(
        "--interface-force-consistent-only", action="store_true",
        help=(
            "jointly train surface trace/strip communication and the integrated "
            "force head while keeping the shared endpoint backbone frozen"
        ),
    )
    parser.add_argument("--far-field-weight", type=float, default=2.0)
    parser.add_argument(
        "--domain-boundary-vorticity-weight", type=float, default=0.0,
    )
    parser.add_argument("--force-weight", type=float, default=.05)
    parser.add_argument("--balanced-force-loss", action="store_true")
    parser.add_argument("--validation-horizon", type=int, default=20)
    parser.add_argument("--validation-starts-per-trajectory", type=int, default=2)
    parser.add_argument(
        "--selection-domain-boundary-weight", type=float, default=0.0,
    )
    parser.add_argument("--selection-surface-weight", type=float, default=0.0)
    parser.add_argument("--selection-force-weight", type=float, default=.025)
    parser.add_argument(
        "--force-gate-relative", type=float,
        help=(
            "save a candidate only when both drag and lift errors are no more "
            "than this fraction above their pretrained validation values"
        ),
    )
    parser.add_argument("--selection-interface-weight", type=float, default=0.0)
    parser.add_argument("--selection-geometry-role-weight", type=float, default=0.0)
    parser.add_argument("--selection-right-boundary-weight", type=float, default=0.0)
    parser.add_argument(
        "--parameter-anchor-weight", type=float, default=0.0,
        help="L2-SP weight that anchors trainable parameters to the input checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def curriculum_horizon(epoch: int, epochs: int, horizons: tuple[int, ...]) -> int:
    if epoch <= 0 or epochs <= 0 or not horizons or min(horizons) <= 0:
        raise ValueError("epochs and curriculum horizons must be positive")
    stage = min(len(horizons) - 1, (epoch - 1) * len(horizons) // epochs)
    return horizons[stage]


def is_interface_shared_dynamics(name: str) -> bool:
    """Select wrapped endpoint dynamics without unfreezing the MPP backbone."""
    prefix = "shared_operator."
    if not name.startswith(prefix):
        return False
    return name.removeprefix(prefix).startswith((
        "projection_", "endpoint_", "vorticity_",
    ))


def is_interface_shared_and_strip(name: str) -> bool:
    """Select generic shared dynamics plus the moving-geometry strip closure."""
    return is_interface_shared_dynamics(name) or name.startswith("strip_")


def relative_parameter_drift(
    anchored: list[tuple[Tensor, Tensor]],
) -> Tensor:
    """Squared L2-SP distance normalized by the checkpoint parameter energy."""
    if not anchored:
        raise ValueError("parameter drift requires at least one anchored parameter")
    numerator = sum(
        (parameter - reference).square().sum()
        for parameter, reference in anchored
    )
    denominator = sum(
        reference.square().sum() for _, reference in anchored
    )
    return numerator / denominator.clamp_min(1.0e-20)


def initial_history(sample: dict[str, Tensor], device: torch.device) -> History:
    return {
        level: (
            sample[f"{level}_q_nm1"][None].to(device),
            sample[f"{level}_q_n"][None].to(device),
        )
        for level in ("coarse", "fine")
    }


def inject_history(
    sample: dict[str, Tensor], history: History,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    batches = {level: batched(sample, level) for level in ("coarse", "fine")}
    for level, (previous, current) in history.items():
        batches[level]["q_nm1"] = previous
        batches[level]["q_n"] = current
    return batches["coarse"], batches["fine"]


def detach_history(history: History) -> History:
    return {
        level: (previous.detach(), current.detach())
        for level, (previous, current) in history.items()
    }


def native_parent_mask(sample: dict[str, Tensor], device: torch.device) -> Tensor:
    fine = sample["native_refinement_mask_np1"][None].to(device).float()
    return F.max_pool2d(fine, kernel_size=2, stride=2) > 0.0


def surface_vorticity_relative_square(
    prediction: Tensor, target: Tensor, sdf: Tensor, dx: float,
) -> Tensor:
    predicted = interior_vorticity(prediction, dx)
    reference = interior_vorticity(target, dx)
    surface = sdf[:, 0, 1:-1, 1:-1].abs() <= 4.0 * dx
    weight = surface.to(predicted.dtype)
    numerator = ((predicted - reference).square() * weight).sum()
    denominator = (reference.square() * weight).sum().clamp_min(1.0e-20)
    return numerator / denominator


def relax_prediction(
    prediction: MultiresolutionPrediction,
    history: History,
    endpoint_relaxation: float,
    history_momentum: float,
) -> MultiresolutionPrediction:
    """Apply a scalar Schwarz-style temporal correction and resynchronize levels."""
    if endpoint_relaxation == 1.0 and history_momentum == 0.0:
        return prediction
    coarse_previous, coarse_current = history["coarse"]
    fine_previous, fine_current = history["fine"]
    coarse = (
        coarse_current
        + endpoint_relaxation * (prediction.coarse - coarse_current)
        + history_momentum * (coarse_current - coarse_previous)
    )
    fine = (
        fine_current
        + endpoint_relaxation * (prediction.fine - fine_current)
        + history_momentum * (fine_current - fine_previous)
    )
    coarse, fine = synchronize_levels(
        coarse, fine, prediction.refinement_mask, fine_trust=1.0,
    )
    return MultiresolutionPrediction(
        coarse=coarse,
        fine=fine,
        refinement_mask=prediction.refinement_mask,
        composite_fine_view=decode_composite(
            coarse, fine, prediction.refinement_mask,
        ),
        executed_fine_tiles=prediction.executed_fine_tiles,
    )


def force_values(
    prediction: Tensor, target: Tensor, batch: dict[str, Tensor], dataset,
) -> tuple[float, float, float, float]:
    predicted = penalty_force_coefficients(
        prediction, batch["sdf_np1"].to(prediction.device),
        batch["motion_np1"].to(prediction.device), dx=dataset.dx,
        window=dataset.window, penalty_time=dataset.penalty_time,
        rotation_sign=dataset.rotation_sign,
    )
    reference = penalty_force_coefficients(
        target, batch["sdf_np1"].to(target.device),
        batch["motion_np1"].to(target.device), dx=dataset.dx,
        window=dataset.window, penalty_time=dataset.penalty_time,
        rotation_sign=dataset.rotation_sign,
    )
    return tuple(float(value) for value in (
        predicted.drag, reference.drag, predicted.lift, reference.lift,
    ))


def step_prediction_and_losses(
    model: torch.nn.Module,
    sample: dict[str, Tensor],
    history: History,
    coarse,
    fine,
    statistics: FieldStatistics,
    args: argparse.Namespace,
    controller: AdaptiveGridController,
    seed: int,
    device: torch.device,
):
    coarse_batch, fine_batch = inject_history(sample, history)
    grid_mode = getattr(args, "grid_mode", "adaptive")
    refinement_mask_override = None
    if grid_mode == "native":
        refinement_mask_override = native_parent_mask(sample, device)
    elif grid_mode == "full":
        refinement_mask_override = torch.ones(
            coarse_batch["q_n"].shape[0], 1,
            *coarse_batch["q_n"].shape[-2:], device=device, dtype=torch.bool,
        )
    elif grid_mode != "adaptive":
        raise ValueError(f"unknown grid mode: {grid_mode}")
    prediction = predict_multiresolution_step(
        model, coarse_batch, fine_batch, coarse, fine, statistics,
        fine_fraction=args.fine_fraction, halo=args.halo, seed=seed,
        device=device, fine_trust=args.fine_trust,
        vorticity_amplitude_fraction=args.vorticity_amplitude_fraction,
        grid_controller=controller,
        placement_ensemble=args.placement_ensemble,
        refinement_mask_override=refinement_mask_override,
    )
    prediction = relax_prediction(
        prediction, history, args.endpoint_relaxation, args.history_momentum,
    )
    if getattr(args, "structured_pressure_projection", False):
        prediction, _ = project_multiresolution_prediction(
            prediction, coarse_batch["sdf_np1"].to(device),
            coarse_dx=coarse.dx,
            strength=float(getattr(args, "pressure_projection_strength", 1.0)),
            core=int(getattr(args, "pressure_projection_core", 32)),
            halo=int(getattr(args, "pressure_projection_halo", 4)),
            body_zero_cells=float(getattr(args, "pressure_body_zero_cells", 2.0)),
            body_ramp_cells=float(getattr(args, "pressure_body_ramp_cells", 4.0)),
            boundary_zero_cells=float(getattr(args, "pressure_boundary_zero_cells", 0.0)),
            boundary_ramp_cells=float(getattr(args, "pressure_boundary_ramp_cells", 4.0)),
        )
    coarse_target = coarse_batch["q_np1"].to(device)
    fine_target = fine_batch["q_np1"].to(device)
    parent_mask = native_parent_mask(sample, device)
    losses = multiresolution_losses(
        prediction, coarse_target, fine_target,
        coarse_batch["sdf_np1"].to(device), fine_batch["sdf_np1"].to(device),
        coarse.dx, fine.dx, composite_mask=parent_mask,
    )
    force_loss = (
        balanced_force_coefficient_relative_square
        if getattr(args, "balanced_force_loss", False)
        else force_coefficient_relative_square
    )
    if args.force_weight > 0.0:
        losses["force"] = force_loss(
            prediction.composite_fine_view, fine_target,
            fine_batch["sdf_np1"].to(device),
            fine_batch["motion_np1"].to(device),
            dx=fine.dx, window=fine.window, penalty_time=fine.penalty_time,
            rotation_sign=fine.rotation_sign,
        )
    else:
        with torch.no_grad():
            losses["force"] = force_loss(
                prediction.composite_fine_view, fine_target,
                fine_batch["sdf_np1"].to(device),
                fine_batch["motion_np1"].to(device),
                dx=fine.dx, window=fine.window, penalty_time=fine.penalty_time,
                rotation_sign=fine.rotation_sign,
            )
    losses["surface_vorticity"] = surface_vorticity_relative_square(
        prediction.composite_fine_view, fine_target,
        fine_batch["sdf_np1"].to(device), fine.dx,
    )
    trace_arguments = (
        prediction.composite_fine_view, fine_target,
        fine_batch["motion_np1"].to(device),
    )
    trace_keywords = {
        "dx": fine.dx, "window": fine.window,
        "thickness": float(fine.physical_parameters["thickness"]),
        "pivot_fraction": float(fine.physical_parameters["pivot_fraction"]),
        "stations": getattr(args, "interface_trace_stations", 64),
    }
    if args.interface_trace_weight > 0.0:
        losses["interface_trace"] = interface_trace_relative_square(
            *trace_arguments, **trace_keywords,
        )
    else:
        with torch.no_grad():
            losses["interface_trace"] = interface_trace_relative_square(
                *trace_arguments, **trace_keywords,
            )
    next_history = {
        "coarse": (history["coarse"][1], prediction.coarse),
        "fine": (history["fine"][1], prediction.fine),
    }
    return prediction, losses, next_history, coarse_target, fine_target, fine_batch


def weighted_objective(losses: dict[str, Tensor], args: argparse.Namespace) -> Tensor:
    return (
        args.coarse_state_weight * losses["coarse_state"]
        + args.fine_state_weight * losses["fine_state"]
        + args.composite_vorticity_weight * losses["composite_vorticity"]
        + args.near_body_weight * losses["near_body_velocity"]
        + getattr(args, "surface_vorticity_weight", 0.0)
        * losses["surface_vorticity"]
        + getattr(args, "interface_trace_weight", 0.0)
        * losses["interface_trace"]
        + args.far_field_weight * losses["far_field_vorticity"]
        + args.domain_boundary_vorticity_weight
        * losses["domain_boundary_vorticity"]
        + args.force_weight * losses["force"]
    )


def train_sequence(
    model: torch.nn.Module,
    paired,
    coarse,
    fine,
    start: int,
    horizon: int,
    statistics: FieldStatistics,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    anchored: list[tuple[Tensor, Tensor]],
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    history = initial_history(paired[start], device)
    controller = AdaptiveGridController(
        dx=coarse.dx, patch_cells=16,
        refine_amplitude_fraction=args.vorticity_amplitude_fraction,
    )
    optimizer.zero_grad(set_to_none=True)
    records: list[dict[str, float]] = []
    segment_loss: Tensor | None = None
    step_weights = torch.arange(1, horizon + 1, device=device, dtype=torch.float32)
    step_weights /= step_weights.sum()
    for step in range(horizon):
        prediction, losses, history, _, _, _ = step_prediction_and_losses(
            model, paired[start + step], history, coarse, fine, statistics, args,
            controller, seed + step, device,
        )
        objective = step_weights[step] * weighted_objective(losses, args)
        segment_loss = objective if segment_loss is None else segment_loss + objective
        records.append({
            key: float(torch.sqrt(value.detach().clamp_min(0.0)))
            for key, value in losses.items()
        } | {
            "executed_fine_tile_fraction": float(
                prediction.executed_fine_tiles.float().mean().detach()
            )
        })
        segment_end = (step + 1) % args.bptt_steps == 0 or step + 1 == horizon
        if segment_end:
            if step + 1 == horizon and args.parameter_anchor_weight > 0.0:
                segment_loss = segment_loss + (
                    args.parameter_anchor_weight * relative_parameter_drift(anchored)
                )
            segment_loss.backward()
            segment_loss = None
            if step + 1 < horizon:
                history = detach_history(history)
    torch.nn.utils.clip_grad_norm_(
        [parameter for group in optimizer.param_groups for parameter in group["params"]],
        1.0,
    )
    optimizer.step()
    result = {
        f"{key}_mean": float(np.mean([row[key] for row in records]))
        for key in records[0]
    }
    return result


def relative_l2(values: list[float], references: list[float]) -> float:
    value = np.asarray(values)
    reference = np.asarray(references)
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-20))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    corpus: PairedTrajectoryCorpus,
    statistics: FieldStatistics,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rows = []
    force = {key: [] for key in ("drag_prediction", "drag_reference", "lift_prediction", "lift_reference")}
    for trajectory, paired in enumerate(corpus.paired):
        maximum = len(paired) - args.validation_horizon
        starts = np.linspace(
            0, maximum, args.validation_starts_per_trajectory, dtype=int,
        ).tolist()
        for start_index, start in enumerate(starts):
            history = initial_history(paired[start], device)
            controller = AdaptiveGridController(
                dx=corpus.coarse[trajectory].dx, patch_cells=16,
                refine_amplitude_fraction=args.vorticity_amplitude_fraction,
            )
            for step in range(args.validation_horizon):
                sample = paired[start + step]
                prediction, losses, history, coarse_target, fine_target, fine_batch = (
                    step_prediction_and_losses(
                        model, sample, history, corpus.coarse[trajectory],
                        corpus.fine[trajectory], statistics, args, controller,
                        args.seed + trajectory * 10_000_019
                        + start_index * 100_003 + step, device,
                    )
                )
                parent_mask = native_parent_mask(sample, device)
                vorticity = float(composite_vorticity_relative_l2(
                    prediction.coarse, prediction.fine, coarse_target, fine_target,
                    parent_mask, coarse_dx=corpus.coarse[trajectory].dx,
                    fine_dx=corpus.fine[trajectory].dx,
                ))
                inner_target = decode_composite(
                    coarse_target, fine_target, parent_mask,
                )
                role_metrics = role_vorticity_metrics(
                    prediction.composite_fine_view, inner_target,
                    fine_batch["sdf_n"].to(device),
                    fine_batch["sdf_np1"].to(device),
                    dx=corpus.fine[trajectory].dx,
                )
                rows.append({
                    "step": step + 1,
                    "vorticity": vorticity,
                    "near_body_velocity": float(torch.sqrt(
                        losses["near_body_velocity"].clamp_min(0.0)
                    )),
                    "surface_vorticity": float(torch.sqrt(
                        losses["surface_vorticity"].clamp_min(0.0)
                    )),
                    "interface_trace": float(torch.sqrt(
                        losses["interface_trace"].clamp_min(0.0)
                    )),
                    "domain_boundary_vorticity": float(torch.sqrt(
                        losses["domain_boundary_vorticity"].clamp_min(0.0)
                    )),
                    **{
                        key: float(value)
                        for key, value in role_metrics.items()
                    },
                    "coarse_divergence_rms": float(fluid_divergence_rms(
                        prediction.coarse, F.avg_pool2d(
                            fine_batch["sdf_np1"].to(device), 2, 2,
                        ),
                        corpus.coarse[trajectory].dx,
                    ).mean()),
                    "fine_divergence_rms": float(fluid_divergence_rms(
                        prediction.fine, fine_batch["sdf_np1"].to(device),
                        corpus.fine[trajectory].dx,
                    ).mean()),
                })
                drag_p, drag_r, lift_p, lift_r = force_values(
                    prediction.composite_fine_view, fine_target, fine_batch,
                    corpus.fine[trajectory],
                )
                for key, value in zip(force, (drag_p, drag_r, lift_p, lift_r)):
                    force[key].append(value)
    checkpoints = sorted(set((1, min(4, args.validation_horizon), args.validation_horizon)))
    result = {
        f"step{step}_amr_vorticity_l2_mean": float(np.mean([
            row["vorticity"] for row in rows if row["step"] == step
        ]))
        for step in checkpoints
    }
    result["all_step_amr_vorticity_l2_mean"] = float(np.mean([
        row["vorticity"] for row in rows
    ]))
    for metric in (
        "near_body_velocity", "surface_vorticity",
        "interface_trace", "domain_boundary_vorticity",
        "geometry_vorticity", "bulk_vorticity",
        "outer_boundary_vorticity", "right_boundary_vorticity",
        "right_boundary_energy_ratio", "right_exit_flux",
        "coarse_divergence_rms", "fine_divergence_rms",
    ):
        result[f"step{args.validation_horizon}_{metric}_l2_mean"] = float(
            np.mean([
                row[metric] for row in rows
                if row["step"] == args.validation_horizon
            ])
        )
        result[f"all_step_{metric}_l2_mean"] = float(np.mean([
            row[metric] for row in rows
        ]))
    result["c_drag_relative_l2"] = relative_l2(
        force["drag_prediction"], force["drag_reference"],
    )
    result["c_lift_relative_l2"] = relative_l2(
        force["lift_prediction"], force["lift_reference"],
    )
    return result


def selection_score(
    metrics: dict[str, float], horizon: int,
    domain_boundary_weight: float = 0.0,
    surface_weight: float = 0.0,
    force_weight: float = .025,
    interface_weight: float = 0.0,
    geometry_role_weight: float = 0.0,
    right_boundary_weight: float = 0.0,
) -> float:
    if min(
        domain_boundary_weight, surface_weight, force_weight, interface_weight,
        geometry_role_weight, right_boundary_weight,
    ) < 0.0:
        raise ValueError("selection weights must be nonnegative")
    return (
        metrics[f"step{horizon}_amr_vorticity_l2_mean"]
        + .25 * metrics[f"step{min(4, horizon)}_amr_vorticity_l2_mean"]
        + .10 * metrics["step1_amr_vorticity_l2_mean"]
        + force_weight * math.log1p(metrics["c_drag_relative_l2"])
        + force_weight * math.log1p(metrics["c_lift_relative_l2"])
        + domain_boundary_weight * metrics.get(
            f"step{horizon}_domain_boundary_vorticity_l2_mean", 0.0,
        )
        + surface_weight * metrics.get(
            f"step{horizon}_surface_vorticity_l2_mean", 0.0,
        )
        + interface_weight * metrics.get(
            f"step{horizon}_interface_trace_l2_mean", 0.0,
        )
        + geometry_role_weight * metrics.get(
            f"step{horizon}_geometry_vorticity_l2_mean", 0.0,
        )
        + right_boundary_weight * metrics.get(
            f"step{horizon}_right_boundary_vorticity_l2_mean", 0.0,
        )
    )


def main() -> None:
    args = parse_args()
    horizons = tuple(args.curriculum_horizons)
    if (
        args.epochs <= 0 or args.sequences_per_epoch <= 0
        or args.bptt_steps <= 0 or args.validation_horizon <= 0
        or args.validation_starts_per_trajectory <= 0
        or args.placement_ensemble <= 0
        or not 0.0 <= args.one_step_fraction <= 1.0
        or args.parameter_anchor_weight < 0.0
        or (args.force_gate_relative is not None and args.force_gate_relative < 0.0)
    ):
        raise ValueError("invalid positive count or one-step fraction")
    exclusive_interface_modes = (
        args.interface_shared_only + args.interface_shared_and_strip
        + args.interface_strip_only + args.interface_operator_only
        + args.interface_interior_force_only
        + args.interface_trace_transport_only
        + args.interface_force_consistent_only
    )
    if exclusive_interface_modes > 1:
        raise ValueError("interface training ownership modes are mutually exclusive")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume_in_place)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    split = (
        write_or_validate_snapshot(
            args.split_manifest, args.output_dir / "trajectory_split.json",
        )
        if args.split_manifest is not None
        else write_or_validate_split(
            args.data_root, args.output_dir / "trajectory_split.json",
            trajectory_count=args.trajectory_count,
        )
    )
    train = PairedTrajectoryCorpus(args.data_root, split["train"], "all")
    validation = PairedTrajectoryCorpus(args.data_root, split["validation"], "all")
    checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    start_epoch = int(checkpoint.get("epoch", 0)) if args.resume_in_place else 0
    if args.resume_in_place:
        if checkpoint.get("training_contract") != (
            "codiffuse_schwarz_model_owned_rollout_v4"
        ):
            raise ValueError("resume checkpoint does not use the rollout-v4 contract")
        if args.epochs <= start_epoch:
            raise ValueError("epochs must exceed the resume checkpoint epoch")
        if checkpoint.get("split") != split:
            raise ValueError("resume checkpoint split differs from the output split")
    if args.interface_operator:
        interface_checkpoint = bool(checkpoint.get("model_spec", {}).get(
            "interface_trace_operator", False,
        ))
        has_vorticity_latent = bool(checkpoint.get("model_spec", {}).get(
            "interface_vorticity_latent", False,
        ))
        has_strip_operator = bool(checkpoint.get("model_spec", {}).get(
            "interface_strip_operator", False,
        ))
        has_strip_transport = bool(checkpoint.get("model_spec", {}).get(
            "interface_strip_transport_operator", False,
        ))
        has_force_operator = bool(checkpoint.get("model_spec", {}).get(
            "interface_interior_force_operator", False,
        ))
        has_trace_transport = bool(checkpoint.get("model_spec", {}).get(
            "interface_trace_transport_operator", False,
        ))
        if interface_checkpoint and args.interface_interior_force_operator \
                and not has_force_operator:
            model, spec = upgrade_interface_force_checkpoint(checkpoint, device)
        elif interface_checkpoint and args.interface_trace_transport_operator \
                and not has_trace_transport:
            model, spec = upgrade_interface_transport_checkpoint(
                checkpoint, device,
            )
        elif interface_checkpoint and args.interface_strip_transport_operator \
                and not has_strip_transport:
            model, spec = upgrade_interface_strip_checkpoint(
                checkpoint, device,
                normal_samples=args.interface_strip_normal_samples,
                transport_operator=True,
            )
        elif interface_checkpoint and args.interface_strip_operator \
                and not has_strip_operator:
            model, spec = upgrade_interface_strip_checkpoint(
                checkpoint, device,
                normal_samples=args.interface_strip_normal_samples,
            )
        elif interface_checkpoint and args.interface_vorticity_latent \
                and not has_vorticity_latent:
            model, spec = upgrade_interface_vorticity_checkpoint(
                checkpoint, device,
            )
        elif interface_checkpoint:
            model, spec = load_interface_trace_checkpoint(
                checkpoint, device,
                endpoint_curl_memory=args.enable_curl_memory,
                endpoint_curl_memory_width=args.curl_memory_width,
                endpoint_curl_memory_depth=args.curl_memory_depth,
                endpoint_tile_communication=args.enable_tile_communication,
            )
        else:
            model, spec = clone_interface_trace_checkpoint(
                checkpoint, device, width=args.interface_operator_width,
                depth=args.interface_operator_depth,
                stations=args.interface_trace_stations,
                solid_velocity_scale=args.interface_solid_velocity_scale,
                vorticity_latent=args.interface_vorticity_latent,
                strip_operator=(
                    args.interface_strip_operator
                    or args.interface_strip_transport_operator
                ),
                strip_transport_operator=args.interface_strip_transport_operator,
                strip_normal_samples=args.interface_strip_normal_samples,
                trace_transport_operator=args.interface_trace_transport_operator,
                interior_force_operator=args.interface_interior_force_operator,
            )
    elif args.geometry_residual_adapter:
        if bool(checkpoint.get("model_spec", {}).get(
            "geometry_residual_operator", False,
        )):
            model, spec = load_geometry_residual_checkpoint(checkpoint, device)
        else:
            model, spec = clone_geometry_residual_checkpoint(
                checkpoint, device, width=args.geometry_width,
                depth=args.geometry_depth,
                geometry_band_cells=args.geometry_band_cells,
                preserve_destination_solid=not args.geometry_two_sided,
            )
    elif args.geometry_specific:
        if any((
            args.enable_boundary_memory, args.enable_motion_transport_adapter,
            args.enable_domain_boundary_adapter, args.enable_rollout_closure,
            args.enable_curl_memory, args.enable_feature_schwarz,
            args.enable_tile_communication,
            args.enable_geometry_latent,
        )):
            raise ValueError(
                "geometry-specific routing must be isolated from other new adapters"
            )
        if bool(checkpoint.get("model_spec", {}).get(
            "geometry_specific_operator", False,
        )):
            model, spec = load_geometry_specific_checkpoint(checkpoint, device)
        else:
            model, spec = clone_geometry_operator_checkpoint(
                checkpoint, device, args.geometry_band_cells,
                body_frame=args.geometry_body_frame,
            )
    else:
        if args.geometry_only:
            raise ValueError(
                "geometry-only requires a geometry-specific or residual operator"
            )
        model, spec = load_pretrained(
            checkpoint, device, history_steps=2,
            endpoint_boundary_memory=args.enable_boundary_memory,
            endpoint_boundary_memory_width=args.boundary_memory_width,
            endpoint_boundary_memory_depth=args.boundary_memory_depth,
            endpoint_motion_transport_adapter=args.enable_motion_transport_adapter,
            endpoint_motion_transport_width=args.motion_transport_width,
            endpoint_motion_transport_depth=args.motion_transport_depth,
            endpoint_geometry_latent=args.enable_geometry_latent,
            endpoint_geometry_latent_width=args.geometry_latent_width,
            endpoint_geometry_latent_depth=args.geometry_latent_depth,
            endpoint_prescribed_motion_warp=args.enable_prescribed_motion_warp,
            endpoint_prescribed_motion_band_cells=args.prescribed_motion_band_cells,
            endpoint_domain_boundary_adapter=args.enable_domain_boundary_adapter,
            endpoint_domain_boundary_width=args.domain_boundary_width,
            endpoint_domain_boundary_depth=args.domain_boundary_depth,
            endpoint_crop_ghost_feature=args.enable_crop_ghost_feature,
            endpoint_rollout_closure=args.enable_rollout_closure,
            endpoint_rollout_closure_width=args.rollout_closure_width,
            endpoint_curl_memory=args.enable_curl_memory,
            endpoint_curl_memory_width=args.curl_memory_width,
            endpoint_curl_memory_depth=args.curl_memory_depth,
            endpoint_feature_schwarz=args.enable_feature_schwarz,
            endpoint_tile_communication=args.enable_tile_communication,
        )
    statistics = FieldStatistics.from_json(checkpoint["statistics"])
    model.set_endpoint_statistics(
        statistics.state_mean, statistics.state_std, statistics.delta_std,
        statistics.delta_mean,
    )
    model.gradient_checkpointing = args.gradient_checkpointing
    trainable = []
    trainable_named: list[tuple[str, Tensor]] = []
    for name, parameter in model.named_parameters():
        if args.geometry_only:
            local_name = name.removeprefix("geometry_operator.")
            selected = (
                name.startswith("geometry_operator.") and local_name.startswith(
                    ("projection_", "endpoint_", "vorticity_"),
                )
            ) or name.startswith((
                "geometry_stem", "geometry_condition", "geometry_blocks",
                "geometry_output",
            ))
        elif args.feature_schwarz_only:
            selected = name == "feature_schwarz_gate"
        elif args.tile_communication_only:
            selected = name.startswith((
                "endpoint_tile_communicator.",
                "shared_operator.endpoint_tile_communicator.",
            ))
        elif args.geometry_latent_only:
            selected = name.startswith("geometry_latent_")
        elif args.interface_shared_and_strip:
            selected = is_interface_shared_and_strip(name)
        elif args.interface_shared_only:
            selected = is_interface_shared_dynamics(name)
        elif args.interface_interior_force_only:
            selected = name.startswith("force_")
        elif args.interface_force_consistent_only:
            selected = name.startswith(("trace_", "strip_", "force_"))
        elif args.interface_trace_transport_only:
            selected = name.startswith("trace_transport_output.")
        elif args.interface_strip_only:
            selected = name.startswith("strip_")
        elif args.interface_operator_only:
            selected = name.startswith(("trace_", "strip_"))
        elif args.rollout_closure_only:
            selected = name.startswith("rollout_closure_")
        elif args.curl_memory_only:
            selected = "curl_memory_" in name
        elif args.boundary_memory_only:
            selected = name.startswith("boundary_memory_")
        elif args.domain_boundary_only:
            selected = name.startswith("domain_boundary_")
        elif args.motion_domain_adapters_only:
            selected = name.startswith((
                "motion_transport_", "domain_boundary_",
            ))
        else:
            selected = name.startswith((
                "projection_", "endpoint_", "vorticity_", "boundary_memory_",
                "motion_transport_", "domain_boundary_",
                "rollout_closure_", "geometry_latent_",
                "trace_",
            ))
        parameter.requires_grad = selected
        if selected:
            trainable.append(parameter)
            trainable_named.append((name, parameter))
    if not trainable:
        raise RuntimeError("checkpoint has no learned-projection parameters")
    anchor_state = checkpoint.get("anchor_state", {}) if args.resume_in_place else {}
    anchored = []
    if args.parameter_anchor_weight > 0.0:
        for name, parameter in trainable_named:
            reference = anchor_state.get(name)
            if reference is None:
                if args.resume_in_place:
                    raise ValueError(f"resume checkpoint is missing anchor {name}")
                reference = parameter.detach().cpu().clone()
            anchored.append((parameter, reference.to(parameter)))
        anchor_state = {
            name: reference.detach().cpu()
            for (name, _), (_, reference) in zip(trainable_named, anchored)
        }
    boundary = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("domain_boundary_") and parameter.requires_grad
    ]
    motion = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("motion_transport_") and parameter.requires_grad
    ]
    special_ids = {
        id(parameter) for parameter in (*boundary, *motion)
    }
    ordinary = [
        parameter for parameter in trainable if id(parameter) not in special_ids
    ]
    groups = []
    if ordinary:
        groups.append({"params": ordinary, "lr": args.lr})
    if boundary:
        groups.append({"params": boundary, "lr": args.domain_boundary_lr})
    if motion:
        groups.append({"params": motion, "lr": args.motion_transport_lr})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.min_lr,
    )
    if args.resume_in_place:
        if "optimizer" not in checkpoint or "scheduler" not in checkpoint:
            raise ValueError("resume checkpoint lacks optimizer or scheduler state")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable)
    summary_path = args.output_dir / "summary.json"

    def checkpoint_payload(
        epoch: int, metrics: dict[str, float], *, training_state: bool,
    ) -> dict:
        payload = {
            "model": model.state_dict(), "model_spec": spec,
            "statistics": statistics.to_json(), "args": vars(args),
            "split": split, "epoch": epoch, "validation": metrics,
            "training_contract": "codiffuse_schwarz_model_owned_rollout_v4",
        }
        if training_state:
            payload |= {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "anchor_state": anchor_state,
            }
        return payload

    def write_summary(
        history_records: list[dict], best: float, initial_pretrained: str,
    ) -> None:
        summary_path.write_text(json.dumps({
            "training_contract": "codiffuse_schwarz_model_owned_rollout_v4",
            "pretrained": initial_pretrained,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "best_selection_score": best,
            "final_epoch": int(history_records[-1]["epoch"]),
            "history": history_records,
        }, indent=2) + "\n", encoding="utf-8")

    if args.resume_in_place:
        if not summary_path.exists():
            raise ValueError("resume requires the existing training summary")
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        history_records = [
            record for record in prior["history"]
            if int(record["epoch"]) <= start_epoch
        ]
        if not history_records or int(history_records[-1]["epoch"]) != start_epoch:
            raise ValueError("resume summary does not end at the checkpoint epoch")
        best = float(prior["best_selection_score"])
        initial_pretrained = str(prior["pretrained"])
        baseline_force = {
            "drag": float(history_records[0]["c_drag_relative_l2"]),
            "lift": float(history_records[0]["c_lift_relative_l2"]),
        }
    else:
        baseline = evaluate(model, validation, statistics, args, device)
        best = selection_score(
            baseline, args.validation_horizon,
            args.selection_domain_boundary_weight,
            args.selection_surface_weight,
            args.selection_force_weight,
            args.selection_interface_weight,
            args.selection_geometry_role_weight,
            args.selection_right_boundary_weight,
        )
        history_records = [{"epoch": 0, "selection_score": best, **baseline}]
        baseline_force = {
            "drag": float(baseline["c_drag_relative_l2"]),
            "lift": float(baseline["c_lift_relative_l2"]),
        }
        initial_pretrained = str(args.pretrained.resolve())
        torch.save(
            checkpoint_payload(0, baseline, training_state=False),
            args.output_dir / "checkpoint.pt",
        )
        torch.save(
            checkpoint_payload(0, baseline, training_state=True),
            args.output_dir / "last_checkpoint.pt",
        )
        write_summary(history_records, best, initial_pretrained)
        print(json.dumps(history_records[-1]), flush=True)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        generator = random.Random(args.seed + epoch * 1_000_003)
        stage_horizon = curriculum_horizon(epoch, args.epochs, horizons)
        records = []
        for sequence in range(args.sequences_per_epoch):
            trajectory = generator.randrange(len(train.paired))
            paired = train.paired[trajectory]
            horizon = 1 if generator.random() < args.one_step_fraction else stage_horizon
            start = generator.randrange(len(paired) - horizon + 1)
            records.append(train_sequence(
                model, paired, train.coarse[trajectory], train.fine[trajectory],
                start, horizon, statistics, args, optimizer,
                anchored,
                args.seed + epoch * 100_000_007 + sequence * 10_007,
                device,
            ))
        scheduler.step()
        metrics = evaluate(model, validation, statistics, args, device)
        score = selection_score(
            metrics, args.validation_horizon,
            args.selection_domain_boundary_weight,
            args.selection_surface_weight,
            args.selection_force_weight,
            args.selection_interface_weight,
            args.selection_geometry_role_weight,
            args.selection_right_boundary_weight,
        )
        force_gate_passed = (
            args.force_gate_relative is None
            or (
                metrics["c_drag_relative_l2"]
                <= baseline_force["drag"] * (1.0 + args.force_gate_relative)
                and metrics["c_lift_relative_l2"]
                <= baseline_force["lift"] * (1.0 + args.force_gate_relative)
            )
        )
        record = {
            "epoch": epoch, "curriculum_horizon": stage_horizon,
            "train_composite_vorticity_l2": float(np.mean([
                row["composite_vorticity_mean"] for row in records
            ])),
            "train_force_l2": float(np.mean([row["force_mean"] for row in records])),
            "train_interface_trace_l2": float(np.mean([
                row["interface_trace_mean"] for row in records
            ])),
            "train_executed_fine_tile_fraction": float(np.mean([
                row["executed_fine_tile_fraction_mean"] for row in records
            ])),
            "train_parameter_drift": (
                float(relative_parameter_drift(anchored).detach())
                if anchored else 0.0
            ),
            "selection_score": score, "lr": optimizer.param_groups[0]["lr"],
            "force_gate_passed": force_gate_passed,
            "drag_improvement_fraction": (
                1.0 - metrics["c_drag_relative_l2"] / max(baseline_force["drag"], 1.0e-12)
            ),
            "lift_improvement_fraction": (
                1.0 - metrics["c_lift_relative_l2"] / max(baseline_force["lift"], 1.0e-12)
            ),
            **metrics,
        }
        history_records.append(record)
        print(json.dumps(record), flush=True)
        if force_gate_passed and score < best:
            best = score
            torch.save(
                checkpoint_payload(epoch, metrics, training_state=False),
                args.output_dir / "checkpoint.pt",
            )
        # Keep the final optimization state even when the scalar selector
        # rejects it. Multi-objective surface/force experiments often expose
        # useful Pareto points that require protected test evaluation.
        torch.save(
            checkpoint_payload(epoch, metrics, training_state=True),
            args.output_dir / "last_checkpoint.pt",
        )
        write_summary(history_records, best, initial_pretrained)
    train.close()
    validation.close()


if __name__ == "__main__":
    main()
