"""Train the strictly local Airfoil time-step operator on absolute Robin ports."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from .absolute_robin_sni import (
    CONTRACT as ABSOLUTE_ROBIN_CONTRACT,
    AbsoluteRobinPointTileSolver2D,
    absolute_robin_port_2d,
    absolute_robin_solver_from_payload,
    absolute_robin_sni_sweep_2d,
    build_absolute_robin_tile_solver,
    transfer_point_zero_port_function,
)
from .airfoil_flux_conditioned_sni import flux_scales_2d
from .airfoil_locality import (
    STRICT_LOCALITY_CONTRACT,
    STRICT_LOCAL_SCALAR_NAMES,
    strict_local_endpoint_conditions,
)
from .airfoil_trace_sni import assemble_subdomains_2d, neighboring_face_mask_2d
from .airfoil_two_level import fluid_divergence_rms
from .airfoil_physical_losses import (
    near_body_relative_square_loss,
    physical_relative_square,
    solid_velocity_relative_square_loss,
)
from .dataset import FieldStatistics
from .force_diagnostics import (
    balanced_force_coefficient_relative_square,
    force_component_relative_square,
    penalty_force_coefficients,
)
from .train_multiresolution_projection import batched
from .train_multitrajectory_multiresolution import PairedTrajectoryCorpus
from .train_one_step_multilevel import random_uniform_transition
from .optimized_stokes_robin import (
    normalized_stokes_robin_residual_port_2d,
    retract_normalized_stokes_robin_residual_2d,
    stokes_robin_increment_scales_2d,
)
from .stokes_robin_sni import (
    CONTRACT as STOKES_ROBIN_CONTRACT,
    build_stokes_robin_tile_solver,
    stokes_robin_solver_from_payload,
    stokes_robin_sni_sweep_2d,
)
from .native_dirichlet_sni import (
    CONTRACT as NATIVE_DIRICHLET_CONTRACT,
    VELOCITY_CONTRACT as NATIVE_VELOCITY_DIRICHLET_CONTRACT,
    SeparableNativeDirichletTileSolver2D,
    impose_normalized_dirichlet_trace_2d,
    impose_normalized_velocity_dirichlet_trace_2d,
    native_dirichlet_sni_sweep_2d,
    native_dirichlet_solver_from_payload,
    normalized_dirichlet_trace_2d,
    normalized_velocity_dirichlet_trace_2d,
)


Tensor = torch.Tensor


def checkpoint_contract(args: argparse.Namespace) -> str:
    if args.robin_kind == "dirichlet":
        return NATIVE_DIRICHLET_CONTRACT
    if args.robin_kind == "velocity_dirichlet":
        return NATIVE_VELOCITY_DIRICHLET_CONTRACT
    return (
        STOKES_ROBIN_CONTRACT
        if args.robin_kind == "optimized_stokes"
        else ABSOLUTE_ROBIN_CONTRACT
    )


def local_solver_from_payload(
    payload: dict[str, object], device: torch.device | str | None = None,
) -> torch.nn.Module:
    """Rebuild any strict-local Airfoil BVP behind the common interface."""
    contract = payload.get("contract")
    target = torch.device(device) if device is not None else torch.device("cpu")
    if contract == ABSOLUTE_ROBIN_CONTRACT:
        return absolute_robin_solver_from_payload(payload, device=target)
    if contract == STOKES_ROBIN_CONTRACT:
        return stokes_robin_solver_from_payload(payload, device=target)
    if contract in (
        NATIVE_DIRICHLET_CONTRACT,
        NATIVE_VELOCITY_DIRICHLET_CONTRACT,
    ):
        return native_dirichlet_solver_from_payload(payload, device=target)
    raise ValueError(f"unknown strict-local Airfoil contract: {contract}")


def robin_channels(args: argparse.Namespace) -> int:
    return 8 if args.robin_kind in (
        "optimized_stokes", "velocity_dirichlet",
    ) else 12


def configure_point_port_column_training(
    solver: AbsoluteRobinPointTileSolver2D,
) -> tuple[dict[str, Tensor], dict[str, Tensor], int]:
    """Freeze a Point endpoint except for its two port-input column blocks.

    The surface and volume input embeddings are the only tensors that directly
    see the artificial-interface port.  Restricting optimization to those
    columns preserves the complete zero-port Local/Core function exactly while
    learning a boundary-conditioned response through the frozen AB-UPT
    processor and decoder.  The returned masks also allow the training loop to
    restore every non-port scalar after each optimizer step as a fail-closed
    guard against weight decay or optimizer-state drift.
    """
    embedding_names = {
        "operator.surface_embedding.0.weight",
        "operator.volume_embedding.0.weight",
    }
    masks: dict[str, Tensor] = {}
    frozen: dict[str, Tensor] = {}
    effective = 0
    for name, parameter in solver.named_parameters():
        parameter.requires_grad_(name in embedding_names)
        if name not in embedding_names:
            continue
        if parameter.ndim != 2 or parameter.shape[1] < (
            solver.physical_channels + solver.robin_channels
        ):
            raise ValueError(f"unexpected Point input embedding shape for {name}")
        mask = torch.zeros_like(parameter)
        start = solver.physical_channels
        stop = start + solver.robin_channels
        mask[:, start:stop] = 1.0
        masks[name] = mask
        frozen[name] = parameter.detach().clone() * (1.0 - mask)
        effective += int(mask.sum().item())
        parameter.register_hook(
            lambda gradient, active=mask: gradient * active
        )
    if set(masks) != embedding_names or effective <= 0:
        raise RuntimeError("Point endpoint omitted a trainable port embedding")
    return masks, frozen, effective


@torch.no_grad()
def restore_point_non_port_columns(
    solver: AbsoluteRobinPointTileSolver2D,
    masks: dict[str, Tensor],
    frozen: dict[str, Tensor],
) -> None:
    parameters = dict(solver.named_parameters())
    if set(parameters).isdisjoint(masks):
        raise RuntimeError("Point port-column restoration found no parameters")
    for name, mask in masks.items():
        parameter = parameters[name]
        parameter.copy_(parameter * mask + frozen[name])


def active_robin_mask(roles: Tensor, args: argparse.Namespace) -> Tensor:
    return roles.repeat_interleave(
        2 if args.robin_kind in (
            "optimized_stokes", "velocity_dirichlet",
        ) else 3, dim=1,
    )


def projected_affine_recurrence_target(
    current_port: Tensor,
    target_port: Tensor,
    active: Tensor,
    base_field: Tensor,
    target_field: Tensor,
    *,
    origin_port: Tensor | None = None,
    alpha_min: float = -0.25,
    alpha_max: float = 1.25,
) -> tuple[Tensor, Tensor]:
    """Label an on-policy port by the local affine BVP response chart.

    A time-discrete Stokes/Oseen subproblem with all volume data fixed is
    affine in its artificial-boundary data.  The training corpus supplies one
    exact response pair: zero-port ``base_field`` and target-port
    ``target_field``.  For a model-owned port, project its active entries onto
    that observed target direction and interpolate the corresponding local
    field.  Components orthogonal to the observed direction are left to the
    exact boundary retraction and are not given a fabricated interior label.

    The projection is detached by construction.  This helper therefore
    supplies a local BVP label; it cannot turn recurrence training into an
    end-to-end unrolled global network.
    """
    if current_port.shape != target_port.shape or active.shape != target_port.shape:
        raise ValueError("recurrence ports and active mask must have equal shape")
    if base_field.shape != target_field.shape or base_field.ndim != 4:
        raise ValueError("recurrence base and target fields must have equal NCHW shape")
    if base_field.shape[0] != current_port.shape[0]:
        raise ValueError("recurrence ports and fields must share the tile batch")
    if not math.isfinite(alpha_min) or not math.isfinite(alpha_max) \
            or alpha_min >= alpha_max:
        raise ValueError("invalid affine recurrence projection interval")
    with torch.no_grad(), torch.autocast(
        device_type=current_port.device.type, enabled=False,
    ):
        origin = (
            torch.zeros_like(current_port)
            if origin_port is None else origin_port
        )
        if origin.shape != current_port.shape:
            raise ValueError("recurrence origin port must match current port")
        current = (current_port.detach() - origin.detach()).float()
        target = (target_port.detach() - origin.detach()).float()
        mask = active.detach().float()
        numerator = (current * target * mask).sum(dim=(1, 2, 3))
        denominator = (target.square() * mask).sum(dim=(1, 2, 3))
        alpha = torch.where(
            denominator > 1.0e-12,
            numerator / denominator.clamp_min(1.0e-12),
            torch.zeros_like(denominator),
        ).clamp(float(alpha_min), float(alpha_max))
        alpha_field = alpha.to(base_field).reshape(-1, 1, 1, 1)
        field = base_field.detach() + alpha_field * (
            target_field.detach() - base_field.detach()
        )
    return field, alpha.to(base_field)


def relaxed_fixed_point_port_target(
    current_port: Tensor,
    fixed_point_port: Tensor,
    active: Tensor,
    relaxation: float,
) -> Tensor:
    """Return the training target in the same relaxed coordinate as SNI.

    Supervising ``next_port`` directly against ``fixed_point_port`` would ask
    the unrelaxed local proposal to overshoot by ``1 / relaxation``.  Applying
    the identical relaxation on both sides makes equality equivalent to an
    unrelaxed proposal at the fixed point.
    """
    if current_port.shape != fixed_point_port.shape \
            or active.shape != current_port.shape:
        raise ValueError("relaxed port target tensors must have equal shape")
    if not math.isfinite(relaxation) or not 0.0 < relaxation <= 1.5:
        raise ValueError("invalid relaxed port target coefficient")
    return (
        current_port.detach()
        + float(relaxation)
        * (fixed_point_port.detach() - current_port.detach())
    ) * active.detach()


def uses_defect_coordinate(args: argparse.Namespace) -> bool:
    return args.port_coordinate in ("base_defect", "lagged_defect")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_affine(statistics: FieldStatistics, reference: Tensor) -> tuple[Tensor, Tensor]:
    mean = reference.new_tensor(statistics.state_mean).reshape(1, 3, 1, 1)
    std = reference.new_tensor(statistics.state_std).reshape(1, 3, 1, 1)
    return mean, std


def projection_affine(
    statistics: dict[str, object], reference: Tensor,
) -> tuple[Tensor, Tensor]:
    """Affine coordinates for incflo's cell-centered pressure gradient."""
    mean_values = tuple(float(value) for value in statistics["state_mean"])
    std_values = tuple(float(value) for value in statistics["state_std"])
    if len(mean_values) != 2 or len(std_values) != 2 or min(std_values) <= 0.0:
        raise ValueError("projection statistics require two positive channel scales")
    mean = reference.new_tensor(mean_values).reshape(1, 2, 1, 1)
    std = reference.new_tensor(std_values).reshape(1, 2, 1, 1)
    return mean, std


def effective_stokes_impedances(
    example: dict[str, object], args: argparse.Namespace,
) -> tuple[float, float]:
    """Return the optimized-Robin coefficients for this native time step.

    The leading terms of the time-discrete Stokes Dirichlet-to-Neumann map
    scale as ``a0 ~ 1 / dt`` in the normal component and
    ``b0 ~ 1 / sqrt(dt)`` in the tangential component when the resolved
    interface band is fixed.  A zero reference step preserves the historical
    fixed-coefficient behavior; otherwise the command-line coefficients are
    interpreted at ``impedance_reference_dt`` and adapted to the recorded
    adaptive-CFL step.  Every tile in one example shares that known time step.
    """
    normal = float(args.normal_impedance)
    tangential = float(args.tangential_impedance)
    reference_dt = float(getattr(args, "impedance_reference_dt", 0.0))
    if reference_dt <= 0.0:
        return normal, tangential
    dt = float(example["dt"])
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("native-step impedance scaling requires a positive dt")
    ratio = reference_dt / dt
    return normal * ratio, tangential * math.sqrt(ratio)


def masked_increment_mse(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    residual_scale: Tensor,
) -> Tensor:
    """MSE in train-set one-step increment coordinates."""
    difference = (prediction - target) / residual_scale.to(prediction)
    return (difference.square() * valid).sum() / (
        valid.sum() * prediction.shape[1]
    ).clamp_min(1.0)


def masked_channel_mse(
    prediction: Tensor, target: Tensor, valid: Tensor,
) -> Tensor:
    if prediction.shape != target.shape or valid.shape != prediction[:, :1].shape:
        raise ValueError("masked channel MSE received incompatible tensors")
    return ((prediction - target).square() * valid).sum() / (
        valid.sum() * prediction.shape[1]
    ).clamp_min(1.0)


def masked_relative_l2(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    return (
        ((prediction - target).square() * valid).sum()
        / (target.square() * valid).sum().clamp_min(1.0e-8)
    ).sqrt()


def masked_per_channel_relative_l2(
    prediction: Tensor, target: Tensor, valid: Tensor,
) -> Tensor:
    """Relative L2 for each field channel over the complete covered domain."""
    if prediction.shape != target.shape or valid.shape != prediction[:, :1].shape:
        raise ValueError("per-channel relative L2 received incompatible tensors")
    dimensions = (0, 2, 3)
    return (
        ((prediction - target).square() * valid).sum(dim=dimensions)
        / (target.square() * valid).sum(dim=dimensions).clamp_min(1.0e-8)
    ).sqrt()


def masked_pressure_constant_decomposition(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
) -> dict[str, Tensor]:
    """Split pressure error into one constant per batch item and its shape.

    Each batch item is treated as one pressure domain.  Calling this on an
    assembled field measures the single global gauge mode; calling it on local
    tiles measures the oracle one-constant-per-subdomain floor.  Both
    components use the ordinary target-pressure energy as denominator, so
    their squared relative values add to the squared ordinary pressure error.
    The target is used only for this diagnostic decomposition.
    """
    if prediction.shape != target.shape or prediction.ndim != 4 \
            or prediction.shape[1] < 3 \
            or valid.shape != prediction[:, :1].shape:
        raise ValueError("pressure decomposition received incompatible tensors")
    weight = valid.to(prediction)
    error = (prediction[:, 2:3] - target[:, 2:3]) * weight
    count = weight.sum(dim=(-1, -2, -3)).clamp_min(1.0)
    offset = error.sum(dim=(-1, -2, -3)) / count
    constant = offset[:, None, None, None] * weight
    shape = error - constant
    total_square = error.square().sum()
    constant_square = constant.square().sum()
    shape_square = shape.square().sum()
    pressure_energy = (
        target[:, 2:3].square() * weight
    ).sum().clamp_min(1.0e-12)
    # Orthogonality of a weighted mean and its centered residual is an
    # executable guard against accidentally conflating a gauge shift with a
    # spatial pressure-shape error.
    closure = (total_square - constant_square - shape_square).abs()
    tolerance = 1.0e-5 * total_square.abs().clamp_min(1.0)
    if bool(closure > tolerance):
        raise RuntimeError("pressure constant/shape decomposition lost closure")
    return {
        "total_relative_l2": (total_square / pressure_energy).sqrt(),
        "constant_relative_l2": (constant_square / pressure_energy).sqrt(),
        "shape_relative_l2": (shape_square / pressure_energy).sqrt(),
        "constant_error_fraction": (
            constant_square / total_square.clamp_min(1.0e-12)
        ),
        "offset_rms": offset.square().mean().sqrt(),
    }


def masked_increment_relative_l2(
    prediction: Tensor,
    target: Tensor,
    previous: Tensor,
    valid: Tensor,
) -> Tensor:
    """Error relative to the monodomain update, not the nearly unchanged state."""
    return (
        ((prediction - target).square() * valid).sum()
        / ((target - previous).square() * valid).sum().clamp_min(1.0e-12)
    ).sqrt()


def masked_per_channel_increment_relative_l2(
    prediction: Tensor,
    target: Tensor,
    previous: Tensor,
    valid: Tensor,
) -> Tensor:
    """Per-channel error relative to the corresponding monodomain update."""
    if prediction.shape != target.shape or previous.shape != target.shape \
            or valid.shape != prediction[:, :1].shape:
        raise ValueError("per-channel increment L2 received incompatible tensors")
    dimensions = (0, 2, 3)
    return (
        ((prediction - target).square() * valid).sum(dim=dimensions)
        / ((target - previous).square() * valid).sum(
            dim=dimensions,
        ).clamp_min(1.0e-12)
    ).sqrt()


def vorticity_relative_square(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    mean: Tensor,
    std: Tensor,
    dx: float,
) -> Tensor:
    predicted = prediction * std + mean
    reference = target * std + mean
    pred_vort = torch.gradient(predicted[:, 1], spacing=dx, dim=-1)[0] \
        - torch.gradient(predicted[:, 0], spacing=dx, dim=-2)[0]
    true_vort = torch.gradient(reference[:, 1], spacing=dx, dim=-1)[0] \
        - torch.gradient(reference[:, 0], spacing=dx, dim=-2)[0]
    return ((pred_vort - true_vort).square() * valid[:, 0]).sum() / (
        true_vort.square() * valid[:, 0]
    ).sum().clamp_min(1.0e-8)


def make_example(
    sample: dict[str, Tensor],
    dataset,
    statistics: FieldStatistics,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    projection_statistics: dict[str, object] | None = None,
) -> dict[str, object]:
    spatial_level = str(getattr(args, "spatial_level", "fine"))
    if spatial_level not in ("coarse", "fine"):
        raise ValueError("spatial_level must be coarse or fine")
    selected = batched(sample, spatial_level)
    transition = random_uniform_transition(
        selected, statistics, dataset, device, seed, args.halo, "random",
    )
    physical, scalar = strict_local_endpoint_conditions(
        transition, radius_cells=args.local_sdf_radius_cells,
    )
    mean, std = state_affine(statistics, transition.q_np1)
    target_global = (transition.q_np1.float() - mean) / std
    previous_global = (transition.q_n.float() - mean) / std
    previous_previous_global = (transition.q_nm1.float() - mean) / std
    target_tiles = transition.cover.tile(
        target_global, padding_mode="replicate",
    )
    target_projection = None
    target_projection_tiles = None
    projection_mean = None
    projection_std = None
    if projection_statistics is not None:
        if transition.gradp_np1 is None:
            raise RuntimeError("projection-state training requires gradp_np1")
        projection_mean, projection_std = projection_affine(
            projection_statistics, transition.gradp_np1,
        )
        target_projection = (
            transition.gradp_np1.float() - projection_mean
        ) / projection_std
        target_projection_tiles = transition.cover.tile(
            target_projection, padding_mode="replicate",
        )
    neighbor_faces = neighboring_face_mask_2d(
        transition.cover.grid_shape,
        batch=transition.batch_size,
        device=device,
        dtype=target_tiles.dtype,
    )
    viscosity = 1.0 / float(dataset.physical_parameters["reynolds"])
    flux_scale = flux_scales_2d(
        mean.flatten(), std.flatten(),
    ).to(device=device, dtype=target_tiles.dtype)
    return {
        "physical": physical,
        "scalar": scalar,
        "target_tiles": target_tiles,
        "target_projection": target_projection,
        "target_projection_tiles": target_projection_tiles,
        "projection_mean": projection_mean,
        "projection_std": projection_std,
        "target": target_global,
        "previous": previous_global,
        "previous_previous": previous_previous_global,
        "valid_tiles": physical[:, -1:],
        "cover": transition.cover,
        "neighbor_faces": neighbor_faces,
        "mean": mean,
        "std": std,
        "delta_std": target_global.new_tensor(
            statistics.delta_std,
        ).reshape(1, 3, 1, 1),
        "flux_scale": flux_scale,
        "viscosity": viscosity,
        "dx": float(dataset.dx),
        "dt": float((transition.time_np1 - transition.time_n).mean()),
        "previous_dt": float(
            (selected["time_n"] - selected["time_nm1"]).mean()
        ),
    }


def subsample_training_tiles(
    example: dict[str, object], count: int, generator: random.Random,
) -> dict[str, object]:
    """Select a geometry-stratified local minibatch from one global cover.

    Endpoint pretraining does not require all subdomains in one optimizer
    update.  Sampling local BVPs increases the number of parameter updates per
    unit compute while preserving strict locality.  At least half of the
    requested budget is offered to tiles that contain a geometry interface;
    the remainder is sampled uniformly from the cover.
    """
    if count <= 0:
        return example
    physical = example["physical"]
    if not isinstance(physical, Tensor) or physical.ndim != 4:
        raise ValueError("training example has no tile-batched physical tensor")
    tile_count = int(physical.shape[0])
    if count >= tile_count:
        return example
    # Channels 3 and 4 are the source and endpoint cube-local SDFs.  A
    # geometry-free tile has a constant clipped SDF; an interface tile does
    # not.  This test uses only local input data.
    sdf = physical[:, 3:5]
    variation = sdf.amax(dim=(-1, -2, -3)) - sdf.amin(dim=(-1, -2, -3))
    geometry = torch.nonzero(variation > 1.0e-6, as_tuple=False).flatten().tolist()
    geometry_quota = min(len(geometry), max(1, count // 2))
    selected = (
        generator.sample(geometry, geometry_quota)
        if len(geometry) > geometry_quota else list(geometry)
    )
    selected_set = set(selected)
    remaining = [index for index in range(tile_count) if index not in selected_set]
    selected.extend(generator.sample(remaining, count - len(selected)))
    selected = sorted(selected)
    index = torch.as_tensor(selected, device=physical.device, dtype=torch.long)
    result = dict(example)
    for key in (
        "physical", "scalar", "target_tiles", "target_projection_tiles",
        "valid_tiles", "neighbor_faces",
    ):
        value = result.get(key)
        if isinstance(value, Tensor):
            if value.shape[0] != tile_count:
                raise ValueError(f"{key} is not aligned with the tile batch")
            result[key] = value.index_select(0, index)
    result["training_tile_indices"] = selected
    result["full_training_tile_count"] = tile_count
    return result


def target_port(
    example: dict[str, object], args: argparse.Namespace,
) -> tuple[Tensor, Tensor]:
    if args.robin_kind in ("dirichlet", "velocity_dirichlet"):
        trace_operator = (
            normalized_velocity_dirichlet_trace_2d
            if args.robin_kind == "velocity_dirichlet"
            else normalized_dirichlet_trace_2d
        )
        return trace_operator(
            example["target_tiles"],
            face_width=args.face_width,
            neighbor_faces=example["neighbor_faces"],
            valid=example["valid_tiles"],
        )
    if args.robin_kind == "optimized_stokes":
        normal_impedance, tangential_impedance = effective_stokes_impedances(
            example, args,
        )
        scale = stokes_robin_increment_scales_2d(
            example["delta_std"].flatten(),
            viscosity=float(example["viscosity"]), dx=float(example["dx"]),
            normal_impedance=normal_impedance,
            tangential_impedance=tangential_impedance,
        ).to(example["target_tiles"])
        return normalized_stokes_robin_residual_port_2d(
            example["target_tiles"], example["physical"][:, :3],
            state_mean=example["mean"].flatten(),
            state_std=example["std"].flatten(),
            robin_increment_scale=scale,
            dx=example["dx"],
            viscosity=example["viscosity"],
            normal_impedance=normal_impedance,
            tangential_impedance=tangential_impedance,
            face_width=args.face_width,
            neighbor_faces=example["neighbor_faces"],
            valid=example["valid_tiles"],
            symmetric_stress=args.symmetric_stress,
        )
    return absolute_robin_port_2d(
        example["target_tiles"],
        state_mean=example["mean"].flatten(),
        state_std=example["std"].flatten(),
        flux_scale=example["flux_scale"],
        dx=example["dx"],
        viscosity=example["viscosity"],
        impedance=(args.mass_impedance, args.momentum_impedance),
        face_width=args.face_width,
        neighbor_faces=example["neighbor_faces"],
        valid=example["valid_tiles"],
    )


def oracle_prediction(
    solver: torch.nn.Module,
    example: dict[str, object],
    args: argparse.Namespace,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    custom_oracle = getattr(solver, "oracle_prediction", None)
    if callable(custom_oracle):
        return custom_oracle(example, args)
    absolute_target, roles = target_port(example, args)
    if uses_defect_coordinate(args):
        point_joint = getattr(args, "backbone", "cnn") == "point"
        with torch.no_grad() if point_joint else nullcontext():
            base = solver(
                example["physical"], example["scalar"],
                torch.zeros_like(absolute_target), roles,
            )
        reference_port = realized_port(
            (
                base[:, :3]
                if args.port_coordinate == "base_defect"
                else example["physical"][:, :3]
            ),
            example, args,
        )
        if args.training_stage == "response" or point_joint:
            # The particular solution is frozen in the staged response fit.
            # Although the separable response is exactly zero for a zero
            # port, evaluating it still creates a formal autograd path through
            # the response weights.  Detach the frozen coordinate chart so
            # the primary and sequential counterfactual objectives own
            # independent graphs and cannot redefine Local/Core.
            base = base.detach()
            reference_port = reference_port.detach()
        port = (absolute_target - reference_port) * active_robin_mask(
            roles, args,
        )
        if args.training_stage == "response" or point_joint:
            port = port.detach()
        if args.training_stage == "base":
            # The staged particular-solution fit is, by definition, the
            # zero-port branch.  Its frozen response is not part of the base
            # objective, so a second identical base evaluation is redundant.
            return base, port, roles, base, reference_port
        prediction = solver(
            example["physical"], example["scalar"], port, roles,
        )
        prediction = maybe_retract_prediction(
            prediction, port, example, args, reference_port,
        )
        return prediction, port, roles, base, reference_port
    prediction = solver(
        example["physical"], example["scalar"], absolute_target, roles,
    )
    prediction = maybe_retract_prediction(
        prediction, absolute_target, example, args,
    )
    return prediction, absolute_target, roles, prediction, None


def realized_port_and_roles(
    prediction: Tensor,
    example: dict[str, object],
    args: argparse.Namespace,
) -> tuple[Tensor, Tensor]:
    """Evaluate a field's configured face port without consulting its target."""
    if prediction.ndim != 4 or prediction.shape[1] < 3:
        raise ValueError("a Robin trace requires at least (u,v,p)")
    prediction = prediction[:, :3]
    if args.robin_kind in ("dirichlet", "velocity_dirichlet"):
        trace_operator = (
            normalized_velocity_dirichlet_trace_2d
            if args.robin_kind == "velocity_dirichlet"
            else normalized_dirichlet_trace_2d
        )
        trace, roles = trace_operator(
            prediction,
            face_width=args.face_width,
            neighbor_faces=example["neighbor_faces"],
            valid=example["valid_tiles"],
        )
        if bool(getattr(args, "flux_compatible_velocity_trace", False)):
            if args.robin_kind != "velocity_dirichlet":
                raise ValueError(
                    "flux-compatible traces require velocity transmission"
                )
            from .incflo_helmholtz_projection_response import (
                project_port_to_zero_net_flux,
            )

            trace, _ = project_port_to_zero_net_flux(
                trace, roles, state_std=example["std"].flatten(),
                dx=float(example["dx"]),
            )
        return trace, roles
    if args.robin_kind == "optimized_stokes":
        normal_impedance, tangential_impedance = effective_stokes_impedances(
            example, args,
        )
        scale = stokes_robin_increment_scales_2d(
            example["delta_std"].flatten(),
            viscosity=float(example["viscosity"]), dx=float(example["dx"]),
            normal_impedance=normal_impedance,
            tangential_impedance=tangential_impedance,
        ).to(prediction)
        return normalized_stokes_robin_residual_port_2d(
            prediction, example["physical"][:, :3],
            state_mean=example["mean"].flatten(),
            state_std=example["std"].flatten(),
            robin_increment_scale=scale,
            dx=example["dx"],
            viscosity=example["viscosity"],
            normal_impedance=normal_impedance,
            tangential_impedance=tangential_impedance,
            face_width=args.face_width,
            neighbor_faces=example["neighbor_faces"],
            valid=example["valid_tiles"],
            symmetric_stress=args.symmetric_stress,
        )
    return absolute_robin_port_2d(
        prediction,
        state_mean=example["mean"].flatten(),
        state_std=example["std"].flatten(),
        flux_scale=example["flux_scale"],
        dx=example["dx"],
        viscosity=example["viscosity"],
        impedance=(args.mass_impedance, args.momentum_impedance),
        face_width=args.face_width,
        neighbor_faces=example["neighbor_faces"],
        valid=example["valid_tiles"],
    )


def realized_port(
    prediction: Tensor,
    example: dict[str, object],
    args: argparse.Namespace,
) -> Tensor:
    """Evaluate only the configured normalized face values of a field."""
    return realized_port_and_roles(prediction, example, args)[0]


def declared_reference_port_and_roles(
    solver: torch.nn.Module,
    example: dict[str, object],
    args: argparse.Namespace,
    cached_particular: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the local BVP's declared target-free trace origin.

    Historical endpoints center a defect at the lagged state.  A response
    wrapper may instead declare a history-based origin while keeping the same
    defect/SNI contract.  Centralizing this dispatch prevents validation,
    rollout, and recurrence training from silently reverting to ``q_n``.
    """
    custom = getattr(solver, "reference_trace", None)
    if callable(custom):
        reference, roles = custom(
            example, face_width=int(args.face_width),
            cached_particular=cached_particular,
        )
        return reference.detach(), roles
    return realized_port_and_roles(
        example["physical"][:, :3], example, args,
    )


def declared_reference_port(
    solver: torch.nn.Module,
    example: dict[str, object],
    args: argparse.Namespace,
    cached_particular: Tensor | None = None,
) -> Tensor:
    """Return only the checkpoint-declared target-free trace origin."""
    return declared_reference_port_and_roles(
        solver, example, args, cached_particular=cached_particular,
    )[0]


def temporal_extrapolated_defect_port(
    example: dict[str, object],
    args: argparse.Namespace,
    reference_port: Tensor,
) -> Tensor:
    """Predict the endpoint defect from two accepted physical time levels.

    ``realized_port`` is already centered at ``q_n``.  Hence its value on
    ``q_nm1`` is ``R(q_nm1)-R(q_n)`` and the first-order endpoint prediction
    is its negative, scaled by the ratio of consecutive native time steps.
    Subtracting the chosen particular-solution trace expresses that prediction
    in the same defect coordinate consumed by the frozen local BVP.
    """
    if not uses_defect_coordinate(args):
        raise ValueError("temporal extrapolation requires a defect coordinate")
    previous_tiles = example["cover"].tile(
        example["previous_previous"], padding_mode="replicate",
    )
    previous_residual = realized_port(previous_tiles, example, args)
    previous_dt = float(example["previous_dt"])
    ratio = (
        float(example["dt"]) / previous_dt
        if math.isfinite(previous_dt) and previous_dt > 1.0e-12 else 1.0
    )
    return (-ratio * previous_residual - reference_port).detach()


def temporal_extrapolated_absolute_port(
    example: dict[str, object],
    args: argparse.Namespace,
) -> Tensor:
    """Predict the next absolute interface port from ``q_nm1`` and ``q_n``.

    This is the absolute-coordinate counterpart of the defect extrapolation
    above and of non-iterative time-dependent DDM interface prediction.  It
    uses accepted physical states only; no endpoint target enters inference.
    """
    if uses_defect_coordinate(args):
        raise ValueError("absolute extrapolation requires an absolute port")
    current_tiles = example["cover"].tile(
        example["previous"], padding_mode="replicate",
    )
    previous_tiles = example["cover"].tile(
        example["previous_previous"], padding_mode="replicate",
    )
    current_port = realized_port(current_tiles, example, args)
    previous_port = realized_port(previous_tiles, example, args)
    previous_dt = float(example["previous_dt"])
    ratio = (
        float(example["dt"]) / previous_dt
        if math.isfinite(previous_dt) and previous_dt > 1.0e-12 else 1.0
    )
    return (current_port + ratio * (current_port - previous_port)).detach()


def maybe_retract_prediction(
    prediction: Tensor,
    imposed_port: Tensor,
    example: dict[str, object],
    args: argparse.Namespace,
    reference_port: Tensor | None = None,
) -> Tensor:
    """Apply the optional strong artificial-face Robin condition."""
    if args.robin_kind in ("dirichlet", "velocity_dirichlet"):
        if not bool(getattr(args, "postsolve_strong_trace", True)):
            return prediction
        if reference_port is None:
            return prediction
        trace_operator = (
            normalized_velocity_dirichlet_trace_2d
            if args.robin_kind == "velocity_dirichlet"
            else normalized_dirichlet_trace_2d
        )
        impose_operator = (
            impose_normalized_velocity_dirichlet_trace_2d
            if args.robin_kind == "velocity_dirichlet"
            else impose_normalized_dirichlet_trace_2d
        )
        _, roles = trace_operator(
            prediction[:, :3], face_width=args.face_width,
            neighbor_faces=example["neighbor_faces"],
            valid=example["valid_tiles"],
        )
        state = impose_operator(
            prediction[:, :3], reference_port + imposed_port,
            roles,
        )
        return (
            state if prediction.shape[1] == 3
            else torch.cat((state, prediction[:, 3:]), dim=1)
        )
    if not getattr(args, "analytic_robin_retraction", False):
        return prediction
    if args.robin_kind != "optimized_stokes" or args.face_width != 1:
        raise ValueError(
            "analytic Robin retraction requires one-cell optimized-Stokes faces"
        )
    if uses_defect_coordinate(args) and reference_port is None:
        # This is the zero-port particular solution from which the defect
        # chart itself is defined.  Its realized trace is exact by definition.
        return prediction
    desired_residual = (
        imposed_port + reference_port
        if uses_defect_coordinate(args) else imposed_port
    )
    normal_impedance, tangential_impedance = effective_stokes_impedances(
        example, args,
    )
    scale = stokes_robin_increment_scales_2d(
        example["delta_std"].flatten(),
        viscosity=float(example["viscosity"]), dx=float(example["dx"]),
        normal_impedance=normal_impedance,
        tangential_impedance=tangential_impedance,
    ).to(prediction)
    state = retract_normalized_stokes_robin_residual_2d(
        prediction[:, :3], example["physical"][:, :3], desired_residual,
        state_mean=example["mean"].flatten(),
        state_std=example["std"].flatten(),
        robin_increment_scale=scale,
        dx=example["dx"], viscosity=example["viscosity"],
        normal_impedance=normal_impedance,
        tangential_impedance=tangential_impedance,
        neighbor_faces=example["neighbor_faces"],
        valid=example["valid_tiles"],
        symmetric_stress=args.symmetric_stress,
    )
    return (
        state if prediction.shape[1] == 3
        else torch.cat((state, prediction[:, 3:]), dim=1)
    )


def port_mse(
    prediction: Tensor,
    imposed: Tensor,
    roles: Tensor,
    example: dict[str, object],
    args: argparse.Namespace,
    reference_port: Tensor | None = None,
) -> Tensor:
    realized = realized_port(prediction, example, args)
    if reference_port is not None:
        realized = realized - reference_port
    active = active_robin_mask(roles, args)
    return ((realized - imposed).square() * active).sum() / active.sum().clamp_min(1.0)


def masked_direction_diagnostics(
    proposal: Tensor,
    target: Tensor,
    mask: Tensor,
) -> tuple[float, float]:
    """Return cosine and norm ratio for one proposed physical correction."""
    if proposal.shape != target.shape:
        raise ValueError("proposal and target directions must have equal shape")
    if mask.ndim != proposal.ndim or mask.shape[0] != proposal.shape[0] or \
            mask.shape[-2:] != proposal.shape[-2:]:
        raise ValueError("direction mask is incompatible with the fields")
    if mask.shape[1] == 1 and proposal.shape[1] != 1:
        mask = mask.expand(-1, proposal.shape[1], -1, -1)
    if mask.shape != proposal.shape:
        raise ValueError("direction mask must be scalar or channel-matched")
    proposed_flat = (proposal * mask).flatten()
    target_flat = (target * mask).flatten()
    proposed_norm = proposed_flat.norm()
    target_norm = target_flat.norm()
    cosine = torch.dot(proposed_flat, target_flat) / (
        proposed_norm * target_norm
    ).clamp_min(1.0e-12)
    ratio = proposed_norm / target_norm.clamp_min(1.0e-12)
    return float(cosine), float(ratio)


def defect_sni_sweep(
    solver: torch.nn.Module,
    example: dict[str, object],
    args: argparse.Namespace,
    current: Tensor,
    imposed_port: Tensor | None = None,
    reference_port: Tensor | None = None,
    pressure_reference_mean: Tensor | None = None,
    pressure_coarse_correction: bool | None = None,
    cached_particular: Tensor | None = None,
):
    """Execute one model-owned sweep for the configured transmission law."""
    use_pressure_coarse = (
        getattr(args, "pressure_coarse_correction", False)
        if pressure_coarse_correction is None
        else bool(pressure_coarse_correction)
    )
    custom_sweep = getattr(solver, "native_sni_sweep", None)
    if callable(custom_sweep):
        return custom_sweep(
            example, args, current, imposed_port=imposed_port,
            reference_port=reference_port,
            pressure_reference_mean=pressure_reference_mean,
            pressure_coarse_correction=use_pressure_coarse,
            cached_particular=cached_particular,
        )
    if args.robin_kind == "conservative_flux":
        if reference_port is not None:
            raise ValueError(
                "absolute conservative-flux SNI has no defect reference port"
            )
        return absolute_robin_sni_sweep_2d(
            solver,
            example["physical"], example["scalar"], current,
            example["cover"],
            state_mean=example["mean"].flatten(),
            state_std=example["std"].flatten(),
            flux_scale=example["flux_scale"],
            dx=example["dx"], viscosity=example["viscosity"],
            impedance=(args.mass_impedance, args.momentum_impedance),
            relaxation=args.relaxation,
            face_width=args.face_width,
            neighbor_faces=example["neighbor_faces"],
            imposed_port=imposed_port,
            assembly=args.assembly,
            pressure_coarse_correction=use_pressure_coarse,
            pressure_reference_mean=pressure_reference_mean,
        )
    if args.robin_kind in ("dirichlet", "velocity_dirichlet"):
        transmission_valid = example["valid_tiles"]
        correction_valid = example["valid_tiles"]
        if bool(getattr(args, "fluid_only_transmission", False)) and bool(
            getattr(args, "fluid_only_correction", False)
        ):
            raise ValueError(
                "fluid-only transmission and correction are distinct protocols"
            )
        if bool(getattr(args, "fluid_only_transmission", False)):
            if example["physical"].shape[1] <= 4:
                raise ValueError(
                    "fluid-only transmission requires endpoint local SDF"
                )
            transmission_valid = transmission_valid * (
                example["physical"][:, 4:5] > 0.0
            ).to(transmission_valid)
            correction_valid = transmission_valid
        elif bool(getattr(args, "fluid_only_correction", False)):
            if example["physical"].shape[1] <= 4:
                raise ValueError(
                    "fluid-only correction requires endpoint local SDF"
                )
            correction_valid = correction_valid * (
                example["physical"][:, 4:5] > 0.0
            ).to(correction_valid)
        return native_dirichlet_sni_sweep_2d(
            solver,
            example["physical"], example["scalar"],
            current, example["cover"],
            relaxation=args.relaxation,
            face_width=args.face_width,
            neighbor_faces=example["neighbor_faces"],
            imposed_port=imposed_port,
            reference_port=reference_port,
            assembly=args.assembly,
            strong_trace=True,
            velocity_only=args.robin_kind == "velocity_dirichlet",
            pressure_coarse_correction=use_pressure_coarse,
            pressure_reference_mean=pressure_reference_mean,
            cached_particular=cached_particular,
            transmission_valid=transmission_valid,
            correction_valid=correction_valid,
        )
    if args.robin_kind != "optimized_stokes":
        raise ValueError("defect SNI sweep requires a native transmission law")
    normal_impedance, tangential_impedance = effective_stokes_impedances(
        example, args,
    )
    robin_scale = stokes_robin_increment_scales_2d(
        example["delta_std"].flatten(),
        viscosity=float(example["viscosity"]),
        dx=float(example["dx"]),
        normal_impedance=normal_impedance,
        tangential_impedance=tangential_impedance,
    ).to(current)
    return stokes_robin_sni_sweep_2d(
        solver,
        example["physical"], example["scalar"],
        current, example["cover"],
        state_mean=example["mean"].flatten(),
        state_std=example["std"].flatten(),
        robin_scale=robin_scale,
        dx=example["dx"], viscosity=example["viscosity"],
        normal_impedance=normal_impedance,
        tangential_impedance=tangential_impedance,
        relaxation=args.relaxation,
        face_width=args.face_width,
        neighbor_faces=example["neighbor_faces"],
        imposed_port=imposed_port,
        reference_port=reference_port,
        defect_coordinate=uses_defect_coordinate(args),
        assembly=args.assembly,
        symmetric_stress=args.symmetric_stress,
        analytic_retraction=getattr(
            args, "analytic_robin_retraction", False,
        ),
        pressure_coarse_correction=use_pressure_coarse,
        pressure_reference_mean=pressure_reference_mean,
    )


@torch.inference_mode()
def validate(
    solver: torch.nn.Module,
    corpus: PairedTrajectoryCorpus,
    statistics: FieldStatistics,
    args: argparse.Namespace,
    device: torch.device,
    projection_statistics: dict[str, object] | None = None,
) -> dict[str, object]:
    solver.eval()
    rows = []
    for trajectory, paired in enumerate(corpus.paired):
        indices = np.linspace(
            0, len(paired) - 1,
            min(args.validation_samples_per_trajectory, len(paired)),
            dtype=int,
        )
        for index in indices:
            reference_sample = paired[int(index)]
            dataset = getattr(corpus, args.spatial_level)[trajectory]
            example = make_example(
                reference_sample, dataset, statistics, args,
                getattr(args, "validation_cover_seed", 20260813)
                + trajectory * 10_000_019 + int(index), device,
                projection_statistics=projection_statistics,
            )
            oracle, port, roles, base_prediction, reference_port = (
                oracle_prediction(solver, example, args)
            )
            oracle_field = assemble_subdomains_2d(
                example["cover"], oracle[:, :3], 1, args.assembly,
            )
            valid = assemble_subdomains_2d(
                example["cover"], example["valid_tiles"], 1, args.assembly,
            )
            shuffled_port = torch.roll(port, 1, 0)
            shuffled = solver(
                example["physical"],
                example["scalar"],
                shuffled_port,
                roles,
            )
            if bool(getattr(solver, "postsolve_strong_trace", True)):
                shuffled = maybe_retract_prediction(
                    shuffled, shuffled_port, example, args, reference_port,
                )
            target_fixed_point_field = None
            target_fixed_point_port_update = None
            custom_sweep = getattr(solver, "native_sni_sweep", None)
            if callable(custom_sweep):
                fixed_args = SimpleNamespace(**vars(args))
                fixed_args.relaxation = 1.0
                fixed_point = custom_sweep(
                    example, fixed_args, example["target"],
                    imposed_port=port, reference_port=reference_port,
                    cached_particular=base_prediction,
                )
                target_fixed_point_field = float(masked_increment_relative_l2(
                    fixed_point.field, example["target"], example["previous"],
                    valid,
                ))
                target_fixed_point_port_update = float(
                    fixed_point.port_defect_rms
                )
            elif args.robin_kind == "optimized_stokes":
                normal_impedance, tangential_impedance = (
                    effective_stokes_impedances(example, args)
                )
                robin_scale = stokes_robin_increment_scales_2d(
                    example["delta_std"].flatten(),
                    viscosity=float(example["viscosity"]),
                    dx=float(example["dx"]),
                    normal_impedance=normal_impedance,
                    tangential_impedance=tangential_impedance,
                ).to(example["target"])
                fixed_point = stokes_robin_sni_sweep_2d(
                    solver,
                    example["physical"],
                    example["scalar"],
                    example["target"],
                    example["cover"],
                    state_mean=example["mean"].flatten(),
                    state_std=example["std"].flatten(),
                    robin_scale=robin_scale,
                    dx=example["dx"],
                    viscosity=example["viscosity"],
                    normal_impedance=normal_impedance,
                    tangential_impedance=tangential_impedance,
                    relaxation=1.0,
                    face_width=args.face_width,
                    neighbor_faces=example["neighbor_faces"],
                    imposed_port=port,
                    reference_port=reference_port,
                    defect_coordinate=uses_defect_coordinate(args),
                    assembly=args.assembly,
                    symmetric_stress=args.symmetric_stress,
                    analytic_retraction=getattr(
                        args, "analytic_robin_retraction", False,
                    ),
                    pressure_coarse_correction=getattr(
                        args, "pressure_coarse_correction", False,
                    ),
                )
                target_fixed_point_field = float(masked_increment_relative_l2(
                    fixed_point.field, example["target"], example["previous"],
                    valid,
                ))
                target_fixed_point_port_update = float(
                    fixed_point.port_defect_rms
                )
            elif args.robin_kind in ("dirichlet", "velocity_dirichlet"):
                fixed_point = native_dirichlet_sni_sweep_2d(
                    solver,
                    example["physical"], example["scalar"],
                    example["target"], example["cover"],
                    relaxation=1.0,
                    face_width=args.face_width,
                    neighbor_faces=example["neighbor_faces"],
                    imposed_port=port,
                    reference_port=reference_port,
                    assembly=args.assembly,
                    strong_trace=True,
                    velocity_only=args.robin_kind == "velocity_dirichlet",
                    pressure_coarse_correction=getattr(
                        args, "pressure_coarse_correction", False,
                    ),
                )
                target_fixed_point_field = float(masked_increment_relative_l2(
                    fixed_point.field, example["target"], example["previous"],
                    valid,
                ))
                target_fixed_point_port_update = float(
                    fixed_point.port_defect_rms
                )
            oracle_physical = oracle_field * example["std"] + example["mean"]
            target_physical = example["target"] * example["std"] + example["mean"]
            force_common = {
                "dx": float(dataset.dx),
                "window": tuple(dataset.window),
                "penalty_time": float(dataset.penalty_time),
                "rotation_sign": float(dataset.rotation_sign),
            }
            endpoint_sdf = reference_sample[
                f"{args.spatial_level}_sdf_np1"
            ][None].to(device)
            motion = reference_sample["motion_np1"][None].to(device)
            predicted_force = penalty_force_coefficients(
                oracle_physical, endpoint_sdf, motion, **force_common,
            )
            reference_force = penalty_force_coefficients(
                target_physical, endpoint_sdf, motion, **force_common,
            )
            errors: list[float] = []
            increment_errors: list[float] = []
            channel_errors: list[list[float]] = []
            channel_increment_errors: list[list[float]] = []
            pressure_constant_errors: list[float] = []
            pressure_shape_errors: list[float] = []
            pressure_constant_fractions: list[float] = []
            tile_pressure_constant_errors: list[float] = []
            tile_pressure_shape_errors: list[float] = []
            tile_pressure_constant_fractions: list[float] = []
            projection_errors: list[float] = []
            projection_channel_errors: list[list[float]] = []
            divergence_values: list[float] = []
            interfaces = []
            port_defects = []
            realization_defects = []
            port_state = None
            recurrence_reference_port = None
            pressure_reference_mean = None
            first_port_target_cosine = None
            first_port_target_norm_ratio = None
            first_port_target_group_cosine = None
            first_port_target_group_norm_ratio = None
            oracle_boundary_response_cosine = None
            oracle_boundary_response_norm_ratio = None
            first_sni_field_update_cosine = None
            first_sni_field_update_norm_ratio = None
            sweep0_field = None
            if args.port_coordinate == "lagged_defect":
                recurrence_reference_port = reference_port
            current = example["previous"]
            # Sweep 0 is the Local/Core prediction initialized from q^n.  Init
            # SNI is exactly the same state and only diverges after its first
            # Robin refresh, eliminating the historical plotting mismatch.
            for sweep_index in range(args.validation_sweeps + 1):
                if callable(custom_sweep):
                    sweep = custom_sweep(
                        example, args, current,
                        imposed_port=port_state,
                        reference_port=recurrence_reference_port,
                        pressure_reference_mean=pressure_reference_mean,
                        pressure_coarse_correction=False,
                        cached_particular=base_prediction,
                    )
                elif args.robin_kind == "optimized_stokes":
                    normal_impedance, tangential_impedance = (
                        effective_stokes_impedances(example, args)
                    )
                    robin_scale = stokes_robin_increment_scales_2d(
                        example["delta_std"].flatten(),
                        viscosity=float(example["viscosity"]),
                        dx=float(example["dx"]),
                        normal_impedance=normal_impedance,
                        tangential_impedance=tangential_impedance,
                    ).to(current)
                    sweep = stokes_robin_sni_sweep_2d(
                        solver,
                        example["physical"],
                        example["scalar"],
                        current,
                        example["cover"],
                        state_mean=example["mean"].flatten(),
                        state_std=example["std"].flatten(),
                        robin_scale=robin_scale,
                        dx=example["dx"],
                        viscosity=example["viscosity"],
                        normal_impedance=normal_impedance,
                        tangential_impedance=tangential_impedance,
                        relaxation=args.relaxation,
                        face_width=args.face_width,
                        neighbor_faces=example["neighbor_faces"],
                        imposed_port=port_state,
                        reference_port=recurrence_reference_port,
                        defect_coordinate=uses_defect_coordinate(args),
                        assembly=args.assembly,
                        symmetric_stress=args.symmetric_stress,
                        analytic_retraction=getattr(
                            args, "analytic_robin_retraction", False,
                        ),
                        pressure_coarse_correction=getattr(
                            args, "pressure_coarse_correction", False,
                        ) and sweep_index > 0,
                        pressure_reference_mean=pressure_reference_mean,
                    )
                elif args.robin_kind in ("dirichlet", "velocity_dirichlet"):
                    sweep = native_dirichlet_sni_sweep_2d(
                        solver,
                        example["physical"], example["scalar"],
                        current, example["cover"],
                        relaxation=args.relaxation,
                        face_width=args.face_width,
                        neighbor_faces=example["neighbor_faces"],
                        imposed_port=port_state,
                        reference_port=recurrence_reference_port,
                        assembly=args.assembly,
                        strong_trace=True,
                        velocity_only=args.robin_kind == "velocity_dirichlet",
                        pressure_coarse_correction=getattr(
                            args, "pressure_coarse_correction", False,
                        ) and sweep_index > 0,
                        pressure_reference_mean=pressure_reference_mean,
                    )
                else:
                    sweep = absolute_robin_sni_sweep_2d(
                        solver,
                        example["physical"],
                        example["scalar"],
                        current,
                        example["cover"],
                        state_mean=example["mean"].flatten(),
                        state_std=example["std"].flatten(),
                        flux_scale=example["flux_scale"],
                        dx=example["dx"],
                        viscosity=example["viscosity"],
                        impedance=(args.mass_impedance, args.momentum_impedance),
                        relaxation=args.relaxation,
                        face_width=args.face_width,
                        neighbor_faces=example["neighbor_faces"],
                        imposed_port=port_state,
                        assembly=args.assembly,
                        pressure_coarse_correction=getattr(
                            args, "pressure_coarse_correction", False,
                        ) and sweep_index > 0,
                        pressure_reference_mean=pressure_reference_mean,
                    )
                current = sweep.field
                port_state = sweep.next_port
                recurrence_reference_port = getattr(
                    sweep, "reference_port", None,
                )
                returned_pressure_reference = getattr(
                    sweep, "pressure_reference_mean", None,
                )
                if returned_pressure_reference is not None:
                    pressure_reference_mean = returned_pressure_reference
                elif (
                    sweep_index == 0
                    and getattr(args, "pressure_coarse_correction", False)
                ):
                    pressure_reference_mean = (
                        (current[:, 2:3] * valid).sum(dim=(-1, -2, -3))
                        / valid.sum(dim=(-1, -2, -3)).clamp_min(1.0)
                    )
                if (
                    sweep_index == 0
                    and getattr(
                        args, "sni_initialization", "local_refresh",
                    ) == "temporal_extrapolation"
                ):
                    if args.robin_kind == "conservative_flux":
                        port_state = temporal_extrapolated_absolute_port(
                            example, args,
                        )
                    else:
                        if recurrence_reference_port is None:
                            raise RuntimeError(
                                "temporal SNI initialization requires a "
                                "reference port"
                            )
                        port_state = temporal_extrapolated_defect_port(
                            example, args, recurrence_reference_port,
                        )
                if sweep_index == 0:
                    active_port = active_robin_mask(roles, args)
                    if uses_defect_coordinate(args):
                        proposed_direction = port_state
                        target_direction = port
                    else:
                        proposed_direction = port_state - sweep.imposed_port
                        target_direction = port - sweep.imposed_port
                    proposed_flat = (
                        proposed_direction * active_port
                    ).flatten()
                    target_flat = (target_direction * active_port).flatten()
                    first_port_target_cosine = float(
                        torch.dot(proposed_flat, target_flat)
                        / (
                            proposed_flat.norm() * target_flat.norm()
                        ).clamp_min(1.0e-12)
                    )
                    first_port_target_norm_ratio = float(
                        proposed_flat.norm()
                        / target_flat.norm().clamp_min(1.0e-12)
                    )
                    groups = (
                        ((0, 2, 4, 6), (1, 3, 5, 7))
                        if args.robin_kind in (
                            "optimized_stokes", "velocity_dirichlet",
                        )
                        else ((0, 3, 6, 9), (1, 4, 7, 10), (2, 5, 8, 11))
                    )
                    group_cosines = []
                    group_ratios = []
                    for indices in groups:
                        proposed_group = (
                            proposed_direction[:, indices]
                            * active_port[:, indices]
                        ).flatten()
                        target_group = (
                            target_direction[:, indices]
                            * active_port[:, indices]
                        ).flatten()
                        group_cosines.append(float(
                            torch.dot(proposed_group, target_group)
                            / (
                                proposed_group.norm() * target_group.norm()
                            ).clamp_min(1.0e-12)
                        ))
                        group_ratios.append(float(
                            proposed_group.norm()
                            / target_group.norm().clamp_min(1.0e-12)
                        ))
                    first_port_target_group_cosine = group_cosines
                    first_port_target_group_norm_ratio = group_ratios
                    local_candidates = sweep.local_candidates[:, :3]
                    oracle_boundary_response_cosine, \
                        oracle_boundary_response_norm_ratio = (
                            masked_direction_diagnostics(
                                oracle[:, :3] - local_candidates,
                                example["target_tiles"] - local_candidates,
                                example["valid_tiles"],
                            )
                        )
                    sweep0_field = current.detach().clone()
                elif sweep_index == 1 and sweep0_field is not None:
                    first_sni_field_update_cosine, \
                        first_sni_field_update_norm_ratio = (
                            masked_direction_diagnostics(
                                current - sweep0_field,
                                example["target"] - sweep0_field,
                                valid,
                            )
                        )
                errors.append(float(masked_relative_l2(
                    current, example["target"], valid,
                )))
                increment_errors.append(float(masked_increment_relative_l2(
                    current, example["target"], example["previous"], valid,
                )))
                channel_errors.append(masked_per_channel_relative_l2(
                    current, example["target"], valid,
                ).tolist())
                channel_increment_errors.append(
                    masked_per_channel_increment_relative_l2(
                        current, example["target"], example["previous"], valid,
                    ).tolist()
                )
                pressure = masked_pressure_constant_decomposition(
                    current, example["target"], valid,
                )
                tile_pressure = masked_pressure_constant_decomposition(
                    sweep.local_candidates[:, :3],
                    example["target_tiles"], example["valid_tiles"],
                )
                pressure_constant_errors.append(float(
                    pressure["constant_relative_l2"]
                ))
                pressure_shape_errors.append(float(
                    pressure["shape_relative_l2"]
                ))
                pressure_constant_fractions.append(float(
                    pressure["constant_error_fraction"]
                ))
                tile_pressure_constant_errors.append(float(
                    tile_pressure["constant_relative_l2"]
                ))
                tile_pressure_shape_errors.append(float(
                    tile_pressure["shape_relative_l2"]
                ))
                tile_pressure_constant_fractions.append(float(
                    tile_pressure["constant_error_fraction"]
                ))
                current_physical = current * example["std"] + example["mean"]
                divergence_values.append(float(fluid_divergence_rms(
                    current_physical, endpoint_sdf, float(dataset.dx),
                ).mean()))
                interfaces.append(float(sweep.interface_rms))
                port_defects.append(float(sweep.port_defect_rms))
                realization_defects.append(float(sweep.realization_rms))
                if example["target_projection"] is not None:
                    auxiliary = getattr(sweep, "auxiliary_field", None)
                    if auxiliary is None:
                        raise RuntimeError(
                            "projection-state checkpoint omitted its auxiliary output"
                        )
                    projection_errors.append(float(masked_relative_l2(
                        auxiliary, example["target_projection"], valid,
                    )))
                    projection_channel_errors.append(
                        masked_per_channel_relative_l2(
                            auxiliary, example["target_projection"], valid,
                        ).tolist()
                    )
            rows.append({
                "trajectory": trajectory,
                "index": int(index),
                "oracle_relative_l2": float(masked_relative_l2(
                    oracle_field, example["target"], valid,
                )),
                "oracle_tile_relative_l2": float(masked_relative_l2(
                    oracle[:, :3], example["target_tiles"], example["valid_tiles"],
                )),
                "oracle_increment_relative_l2": float(masked_increment_relative_l2(
                    oracle_field, example["target"], example["previous"], valid,
                )),
                "oracle_channel_increment_relative_l2": (
                    masked_per_channel_increment_relative_l2(
                        oracle_field, example["target"],
                        example["previous"], valid,
                    ).tolist()
                ),
                "oracle_pressure_decomposition": {
                    key: float(value) for key, value in
                    masked_pressure_constant_decomposition(
                        oracle_field, example["target"], valid,
                    ).items()
                },
                "oracle_tile_pressure_decomposition": {
                    key: float(value) for key, value in
                    masked_pressure_constant_decomposition(
                        oracle[:, :3], example["target_tiles"],
                        example["valid_tiles"],
                    ).items()
                },
                "oracle_port_rms": float(port_mse(
                    oracle, port, roles, example, args, reference_port,
                ).sqrt()),
                "first_port_target_cosine": first_port_target_cosine,
                "first_port_target_norm_ratio": first_port_target_norm_ratio,
                "first_port_target_group_cosine": (
                    first_port_target_group_cosine
                ),
                "first_port_target_group_norm_ratio": (
                    first_port_target_group_norm_ratio
                ),
                "oracle_boundary_response_cosine": (
                    oracle_boundary_response_cosine
                ),
                "oracle_boundary_response_norm_ratio": (
                    oracle_boundary_response_norm_ratio
                ),
                "first_sni_field_update_cosine": first_sni_field_update_cosine,
                "first_sni_field_update_norm_ratio": (
                    first_sni_field_update_norm_ratio
                ),
                "shuffled_port_tile_relative_l2": float(masked_relative_l2(
                    shuffled[:, :3], example["target_tiles"], example["valid_tiles"],
                )),
                "target_fixed_point_increment_relative_l2": (
                    target_fixed_point_field
                ),
                "target_fixed_point_port_update_rms": (
                    target_fixed_point_port_update
                ),
                "oracle_projection_relative_l2": (
                    float(masked_relative_l2(
                        oracle[:, 3:], example["target_projection_tiles"],
                        example["valid_tiles"],
                    ))
                    if example["target_projection_tiles"] is not None else None
                ),
                "c_drag_prediction": float(predicted_force.drag[0]),
                "c_drag_reference": float(reference_force.drag[0]),
                "c_lift_prediction": float(predicted_force.lift[0]),
                "c_lift_reference": float(reference_force.lift[0]),
                "field_relative_l2_by_sweep": errors,
                "increment_relative_l2_by_sweep": increment_errors,
                "channel_relative_l2_by_sweep": channel_errors,
                "channel_increment_relative_l2_by_sweep": (
                    channel_increment_errors
                ),
                "pressure_constant_relative_l2_by_sweep": (
                    pressure_constant_errors
                ),
                "pressure_shape_relative_l2_by_sweep": pressure_shape_errors,
                "pressure_constant_error_fraction_by_sweep": (
                    pressure_constant_fractions
                ),
                "tile_pressure_constant_relative_l2_by_sweep": (
                    tile_pressure_constant_errors
                ),
                "tile_pressure_shape_relative_l2_by_sweep": (
                    tile_pressure_shape_errors
                ),
                "tile_pressure_constant_error_fraction_by_sweep": (
                    tile_pressure_constant_fractions
                ),
                "projection_relative_l2_by_sweep": projection_errors,
                "projection_channel_relative_l2_by_sweep": (
                    projection_channel_errors
                ),
                "divergence_rms_by_sweep": divergence_values,
                "target_divergence_rms": float(fluid_divergence_rms(
                    target_physical, endpoint_sdf, float(dataset.dx),
                ).mean()),
                "interface_rms_by_sweep": interfaces,
                "port_defect_rms_by_sweep": port_defects,
                "realization_rms_by_sweep": realization_defects,
            })
    result = {
        "examples": len(rows),
        "oracle_relative_l2": float(np.mean([
            row["oracle_relative_l2"] for row in rows
        ])),
        "oracle_tile_relative_l2": float(np.mean([
            row["oracle_tile_relative_l2"] for row in rows
        ])),
        "oracle_increment_relative_l2": float(np.mean([
            row["oracle_increment_relative_l2"] for row in rows
        ])),
        "oracle_channel_increment_relative_l2": np.mean([
            row["oracle_channel_increment_relative_l2"] for row in rows
        ], axis=0).tolist(),
        "oracle_pressure_decomposition": {
            key: float(np.mean([
                row["oracle_pressure_decomposition"][key] for row in rows
            ]))
            for key in rows[0]["oracle_pressure_decomposition"]
        },
        "oracle_tile_pressure_decomposition": {
            key: float(np.mean([
                row["oracle_tile_pressure_decomposition"][key]
                for row in rows
            ]))
            for key in rows[0]["oracle_tile_pressure_decomposition"]
        },
        "oracle_port_rms": float(np.mean([
            row["oracle_port_rms"] for row in rows
        ])),
        "first_port_target_cosine": (
            float(np.mean([
                row["first_port_target_cosine"] for row in rows
            ]))
            if rows[0]["first_port_target_cosine"] is not None else None
        ),
        "first_port_target_norm_ratio": (
            float(np.mean([
                row["first_port_target_norm_ratio"] for row in rows
            ]))
            if rows[0]["first_port_target_norm_ratio"] is not None else None
        ),
        "first_port_target_group_cosine": (
            np.mean([
                row["first_port_target_group_cosine"] for row in rows
            ], axis=0).tolist()
            if rows[0]["first_port_target_group_cosine"] is not None else None
        ),
        "first_port_target_group_norm_ratio": (
            np.mean([
                row["first_port_target_group_norm_ratio"] for row in rows
            ], axis=0).tolist()
            if rows[0]["first_port_target_group_norm_ratio"] is not None else None
        ),
        "oracle_boundary_response_cosine": float(np.mean([
            row["oracle_boundary_response_cosine"] for row in rows
        ])),
        "oracle_boundary_response_norm_ratio": float(np.mean([
            row["oracle_boundary_response_norm_ratio"] for row in rows
        ])),
        "first_sni_field_update_cosine": float(np.mean([
            row["first_sni_field_update_cosine"] for row in rows
        ])),
        "first_sni_field_update_norm_ratio": float(np.mean([
            row["first_sni_field_update_norm_ratio"] for row in rows
        ])),
        "shuffled_port_tile_relative_l2": float(np.mean([
            row["shuffled_port_tile_relative_l2"] for row in rows
        ])),
        "field_relative_l2_by_sweep": np.mean([
            row["field_relative_l2_by_sweep"] for row in rows
        ], axis=0).tolist(),
        "increment_relative_l2_by_sweep": np.mean([
            row["increment_relative_l2_by_sweep"] for row in rows
        ], axis=0).tolist(),
        "channel_relative_l2_by_sweep": np.mean([
            row["channel_relative_l2_by_sweep"] for row in rows
        ], axis=0).tolist(),
        "channel_increment_relative_l2_by_sweep": np.mean([
            row["channel_increment_relative_l2_by_sweep"] for row in rows
        ], axis=0).tolist(),
        "pressure_constant_relative_l2_by_sweep": np.mean([
            row["pressure_constant_relative_l2_by_sweep"] for row in rows
        ], axis=0).tolist(),
        "pressure_shape_relative_l2_by_sweep": np.mean([
            row["pressure_shape_relative_l2_by_sweep"] for row in rows
        ], axis=0).tolist(),
        "pressure_constant_error_fraction_by_sweep": np.mean([
            row["pressure_constant_error_fraction_by_sweep"] for row in rows
        ], axis=0).tolist(),
        "tile_pressure_constant_relative_l2_by_sweep": np.mean([
            row["tile_pressure_constant_relative_l2_by_sweep"] for row in rows
        ], axis=0).tolist(),
        "tile_pressure_shape_relative_l2_by_sweep": np.mean([
            row["tile_pressure_shape_relative_l2_by_sweep"] for row in rows
        ], axis=0).tolist(),
        "tile_pressure_constant_error_fraction_by_sweep": np.mean([
            row["tile_pressure_constant_error_fraction_by_sweep"]
            for row in rows
        ], axis=0).tolist(),
        "divergence_rms_by_sweep": np.mean([
            row["divergence_rms_by_sweep"] for row in rows
        ], axis=0).tolist(),
        "target_divergence_rms": float(np.mean([
            row["target_divergence_rms"] for row in rows
        ])),
        "interface_rms_by_sweep": np.mean([
            row["interface_rms_by_sweep"] for row in rows
        ], axis=0).tolist(),
        "port_defect_rms_by_sweep": np.mean([
            row["port_defect_rms_by_sweep"] for row in rows
        ], axis=0).tolist(),
        "realization_rms_by_sweep": np.mean([
            row["realization_rms_by_sweep"] for row in rows
        ], axis=0).tolist(),
        "paired_final_minus_sweep0_mean": float(np.mean([
            row["field_relative_l2_by_sweep"][-1]
            - row["field_relative_l2_by_sweep"][0]
            for row in rows
        ])),
        "paired_improvement_fraction": float(np.mean([
            row["field_relative_l2_by_sweep"][-1]
            < row["field_relative_l2_by_sweep"][0]
            for row in rows
        ])),
        "paired_increment_final_minus_local_mean": float(np.mean([
            row["increment_relative_l2_by_sweep"][-1]
            - row["increment_relative_l2_by_sweep"][0]
            for row in rows
        ])),
        "paired_increment_improvement_fraction": float(np.mean([
            row["increment_relative_l2_by_sweep"][-1]
            < row["increment_relative_l2_by_sweep"][0]
            for row in rows
        ])),
        "rows": rows,
    }
    if args.robin_kind in (
        "optimized_stokes", "dirichlet", "velocity_dirichlet",
    ):
        result.update({
            "target_fixed_point_increment_relative_l2": float(np.mean([
                row["target_fixed_point_increment_relative_l2"] for row in rows
            ])),
            "target_fixed_point_port_update_rms": float(np.mean([
                row["target_fixed_point_port_update_rms"] for row in rows
            ])),
        })
    if projection_statistics is not None:
        result["oracle_projection_relative_l2"] = float(np.mean([
            row["oracle_projection_relative_l2"] for row in rows
        ]))
        result["projection_relative_l2_by_sweep"] = np.mean([
            row["projection_relative_l2_by_sweep"] for row in rows
        ], axis=0).tolist()
        result["projection_channel_relative_l2_by_sweep"] = np.mean([
            row["projection_channel_relative_l2_by_sweep"] for row in rows
        ], axis=0).tolist()
    for component in ("c_drag", "c_lift"):
        prediction = np.asarray([
            row[f"{component}_prediction"] for row in rows
        ])
        reference = np.asarray([
            row[f"{component}_reference"] for row in rows
        ])
        result[f"oracle_{component}_relative_l2"] = float(
            np.linalg.norm(prediction - reference)
            / max(np.linalg.norm(reference), 1.0e-12)
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--reference-checkpoint", type=Path,
        help=(
            "optional provenance checkpoint supplying the legacy split and "
            "statistics; a from-scratch corpus may instead provide both "
            "--split-manifest and --statistics-json"
        ),
    )
    parser.add_argument(
        "--statistics-json", type=Path,
        help="optional normalization payload, required for a new temporal cadence",
    )
    parser.add_argument(
        "--split-manifest", type=Path,
        help=(
            "optional train/validation/test JSON; used for solver-step-aligned "
            "pilot corpora instead of the frozen publication split"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-epoch", type=int, default=512)
    parser.add_argument(
        "--training-tiles-per-sample", type=int, default=0,
        help=(
            "geometry-stratified local BVPs per optimizer update; zero uses "
            "the complete cover and is required for global/recurrence losses"
        ),
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=0,
        help=(
            "write a model-only candidate every N epochs for an independent "
            "model-owned rollout screen; zero writes only last/best"
        ),
    )
    parser.add_argument("--validation-samples-per-trajectory", type=int, default=8)
    parser.add_argument(
        "--validation-cover-seed", type=int, default=20260813,
        help=(
            "run-independent seed for validation cover offsets; keeping it "
            "separate from the optimizer seed makes checkpoint comparisons "
            "use identical subdomain placements"
        ),
    )
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument(
        "--spatial-level", choices=("coarse", "fine"), default="fine",
        help=(
            "paired HDF level used by the local BVP; coarse selects the L5 "
            "fixed-cadence field used by the OOD protocol"
        ),
    )
    parser.add_argument(
        "--history-steps", type=int, choices=(1, 2), default=2,
        help=(
            "one uses the exact native incflo Markov state (q_n and lagged "
            "pressure gradient); two additionally supplies the local q_n-q_nm1 history"
        ),
    )
    parser.add_argument(
        "--predict-projection-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "predict incflo's two cell-centered pressure-gradient channels so "
            "the accepted local solve is a closed native-step Markov state"
        ),
    )
    parser.add_argument("--face-width", type=int, default=2)
    parser.add_argument(
        "--robin-kind",
        choices=(
            "conservative_flux", "optimized_stokes", "dirichlet",
            "velocity_dirichlet",
        ),
        default="conservative_flux",
        help="artificial-interface condition used by the local BVP and SNI",
    )
    parser.add_argument("--local-sdf-radius-cells", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument(
        "--backbone", choices=("cnn", "point"), default="cnn",
        help="shared strict-local BVP backbone",
    )
    parser.add_argument(
        "--response-architecture", choices=("entangled", "separable"),
        default="entangled",
        help=(
            "optimized-Stokes port parameterization; separable preserves the "
            "zero-port Local/Core map exactly"
        ),
    )
    parser.add_argument(
        "--port-coordinate",
        choices=("centered_absolute", "base_defect", "lagged_defect"),
        default="centered_absolute",
        help=(
            "absolute centered Robin data or the homogeneous defect relative "
            "to the frozen Local/Core particular solution"
        ),
    )
    parser.add_argument(
        "--training-stage", choices=("joint", "base", "response"),
        default="joint",
        help=(
            "joint optimization, zero-port local pretraining, or frozen-base "
            "Robin-response training"
        ),
    )
    parser.add_argument(
        "--base-reference-dt", type=float, default=0.0,
        help=(
            "positive reference step for explicit dt scaling of the separable "
            "zero-port base update; zero disables the scaling"
        ),
    )
    parser.add_argument(
        "--response-reference-dt", type=float, default=0.0,
        help=(
            "positive reference step for explicit dt scaling of the "
            "separable communicated state correction; zero disables it"
        ),
    )
    parser.add_argument(
        "--response-global-attention-layers", type=int, default=0,
        help=(
            "zero-preserving context-attention layers in the local Robin "
            "response; attention is global only within one subdomain"
        ),
    )
    parser.add_argument("--response-attention-grid", type=int, default=8)
    parser.add_argument("--response-attention-heads", type=int, default=8)
    parser.add_argument(
        "--base-face-role-dilation", type=int, default=0,
        help=(
            "inward dilation applied only to the frozen base branch's face "
            "roles, allowing a one-cell strong trace to reuse a base trained "
            "with a wider face collar"
        ),
    )
    parser.add_argument("--point-d-model", type=int, default=320)
    parser.add_argument("--point-surface-anchors", type=int, default=256)
    parser.add_argument("--point-volume-anchors", type=int, default=512)
    parser.add_argument("--point-geometry-anchors", type=int, default=128)
    parser.add_argument("--point-depth", type=int, default=5)
    parser.add_argument("--point-decoder-depth", type=int, default=3)
    parser.add_argument("--point-heads", type=int, default=8)
    parser.add_argument(
        "--point-port-input-scale", type=float, default=1.0,
        help=(
            "fixed normalization applied to artificial-boundary channels "
            "before the Point input embeddings"
        ),
    )
    parser.add_argument(
        "--point-interface-anchor-fraction", type=float, default=0.0,
        help=(
            "fraction of the unchanged local Point surface-anchor budget "
            "reserved for artificial Robin faces"
        ),
    )
    parser.add_argument(
        "--point-architecture-reference", type=Path,
        help=(
            "global DynamicABUPT checkpoint whose width, anchor budgets, "
            "depths, heads, and total capacity must match --backbone point"
        ),
    )
    parser.add_argument(
        "--point-capacity-tolerance", type=float, default=0.01,
        help=(
            "maximum relative parameter-count difference from the global "
            "Point reference; local BVP input embeddings may differ"
        ),
    )
    parser.add_argument("--validation-sweeps", type=int, default=8)
    parser.add_argument(
        "--sni-initialization",
        choices=("local_refresh", "temporal_extrapolation"),
        default="local_refresh",
        help=(
            "initialize the first corrected sweep from the Local/Core trace "
            "refresh or a first-order trace prediction from q_nm1 and q_n"
        ),
    )
    parser.add_argument("--relaxation", type=float, default=0.5)
    parser.add_argument("--mass-impedance", type=float, default=1.0)
    parser.add_argument("--momentum-impedance", type=float, default=1.0)
    parser.add_argument(
        "--normal-impedance", type=float, default=20.0,
        help="degree-zero S^{-1,0} normal-velocity coefficient",
    )
    parser.add_argument(
        "--tangential-impedance", type=float, default=0.25,
        help="degree-zero S^{-1,0} tangential-velocity coefficient",
    )
    parser.add_argument(
        "--impedance-reference-dt", type=float, default=0.0,
        help=(
            "if positive, interpret a0/b0 at this step and scale them as "
            "dt^{-1}/dt^{-1/2} on each recorded adaptive-CFL transition"
        ),
    )
    parser.add_argument(
        "--symmetric-stress", action="store_true",
        help="use symmetric rather than the paper's default nonsymmetric stress",
    )
    parser.add_argument(
        "--analytic-robin-retraction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "strongly impose the one-cell nonsymmetric Stokes--Robin trace "
            "after each learned local response"
        ),
    )
    parser.add_argument(
        "--pressure-coarse-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "solve one overlap-graph coarse problem for the otherwise free "
            "pressure constant of every incompressible subdomain"
        ),
    )
    parser.add_argument(
        "--assembly", choices=("normalized_overlap", "disjoint_core"),
        default="normalized_overlap",
    )
    parser.add_argument("--vorticity-weight", type=float, default=0.05)
    parser.add_argument(
        "--increment-field-weight", type=float, default=1.0,
        help="weight of the train-set increment-coordinate field loss",
    )
    parser.add_argument(
        "--velocity-field-weight", type=float, default=0.0,
        help=(
            "additional increment-coordinate weight on (u,v), separated from "
            "the pressure gauge mode"
        ),
    )
    parser.add_argument(
        "--local-field-weight", type=float, default=0.0,
        help=(
            "additional increment-coordinate supervision at the centered "
            "zero Robin port"
        ),
    )
    parser.add_argument(
        "--homotopy-field-weight", type=float, default=0.0,
        help=(
            "supervise intermediate Robin responses using the affine "
            "modified-Stokes boundary-response homotopy"
        ),
    )
    parser.add_argument(
        "--physical-field-weight", type=float, default=0.0,
        help="weight of squared relative error in physical state coordinates",
    )
    parser.add_argument(
        "--projection-state-weight", type=float, default=1.0,
        help="weight of the standardized next pressure-gradient state loss",
    )
    parser.add_argument("--solid-velocity-weight", type=float, default=0.0)
    parser.add_argument("--near-body-field-weight", type=float, default=0.0)
    parser.add_argument("--force-weight", type=float, default=0.0)
    parser.add_argument("--lift-force-weight", type=float, default=0.0)
    parser.add_argument("--selection-force-weight", type=float, default=0.0)
    parser.add_argument(
        "--selection-projection-weight", type=float, default=0.1,
        help="checkpoint-selection weight for next projection-state error",
    )
    parser.add_argument(
        "--selection-metric", choices=("state", "increment"), default="state",
        help="checkpoint metric; solver-step pilots should use increment",
    )
    parser.add_argument(
        "--selection-monotonicity-weight", type=float, default=1.0,
        help=(
            "penalty on increases between successive validation sweeps when "
            "selecting checkpoint_best_convergence.pt"
        ),
    )
    parser.add_argument("--port-weight", type=float, default=0.1)
    parser.add_argument("--counterfactual-port-weight", type=float, default=0.05)
    parser.add_argument(
        "--recurrence-field-weight", type=float, default=0.0,
        help=(
            "supervise detached model-owned SNI iterates in addition to the "
            "oracle Robin response; zero disables recurrence-aware training"
        ),
    )
    parser.add_argument(
        "--recurrence-velocity-field-weight", type=float, default=0.0,
        help=(
            "additional on-policy recurrence weight on velocity only; useful "
            "after the pressure nullspace is removed by the coarse solve"
        ),
    )
    parser.add_argument(
        "--recurrence-realization-weight", type=float, default=0.0,
        help=(
            "weight on realizing each imposed model-owned Robin port during "
            "recurrence-aware response training"
        ),
    )
    parser.add_argument(
        "--recurrence-target-port-weight", type=float, default=0.0,
        help=(
            "supervise each model-owned refreshed port toward the recorded "
            "monodomain fixed-point port"
        ),
    )
    parser.add_argument(
        "--recurrence-sweeps", type=int, default=2,
        help="number of corrected, detached-on-policy SNI sweeps to supervise",
    )
    parser.add_argument(
        "--recurrence-affine-alpha-min", type=float, default=-0.25,
        help="lower safeguard for the on-policy affine BVP coordinate",
    )
    parser.add_argument(
        "--recurrence-affine-alpha-max", type=float, default=1.25,
        help="upper safeguard for the on-policy affine BVP coordinate",
    )
    parser.add_argument(
        "--sequential-counterfactual-backward",
        action="store_true",
        help=(
            "backpropagate the oracle and counterfactual Robin graphs "
            "sequentially to reduce peak memory"
        ),
    )
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True,
        help="use bfloat16 autocast and gradient scaling on CUDA",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize", type=Path,
        help="load local weights only and restart the optimizer/schedule",
    )
    parser.add_argument(
        "--initialize-base-only", action="store_true",
        help="load only separable base_* weights from --initialize",
    )
    parser.add_argument(
        "--initialize-point-zero-port-transfer", action="store_true",
        help=(
            "transfer a capacity-identical Point local endpoint across Robin "
            "contracts while preserving its zero-port Local/Core function"
        ),
    )
    parser.add_argument(
        "--initialize-point-port-policy",
        choices=("zero", "reuse"), default="zero",
        help=(
            "during zero-port-preserving Point transfer, either initialize "
            "the artificial-boundary embedding columns to zero or reuse "
            "same-width source columns; both policies preserve Local/Core"
        ),
    )
    parser.add_argument(
        "--point-port-columns-only", action="store_true",
        help=(
            "freeze the capacity-matched Point endpoint and optimize only the "
            "surface/volume embedding columns multiplying interface-port "
            "inputs; this preserves Local/Core exactly"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.initialize_base_only and args.initialize is None:
        raise ValueError("--initialize-base-only requires --initialize")
    if args.initialize_point_zero_port_transfer and args.initialize is None:
        raise ValueError(
            "--initialize-point-zero-port-transfer requires --initialize"
        )
    if args.initialize_base_only and args.initialize_point_zero_port_transfer:
        raise ValueError("Point transfer and base-only initialization conflict")
    if args.initialize_point_zero_port_transfer and args.backbone != "point":
        raise ValueError("zero-port Point transfer requires --backbone point")
    if args.initialize_point_port_policy != "zero" \
            and not args.initialize_point_zero_port_transfer:
        raise ValueError(
            "a nonzero Point port policy requires zero-port-preserving transfer"
        )
    if args.point_port_columns_only and (
        args.backbone != "point"
        or args.initialize is None
        or not args.initialize_point_zero_port_transfer
        or args.training_stage != "joint"
        or not uses_defect_coordinate(args)
    ):
        raise ValueError(
            "Point port-column training requires a zero-port-transferred "
            "joint Point defect endpoint initialization"
        )
    if args.resume is not None and args.initialize is not None:
        raise ValueError("resume and initialize are mutually exclusive")
    if args.backbone == "point" and (
        args.response_architecture != "entangled"
        or args.training_stage != "joint"
    ):
        raise ValueError(
            "the Point local backbone currently requires joint entangled training"
        )
    if args.backbone == "point" and args.robin_kind in (
        "optimized_stokes", "dirichlet", "velocity_dirichlet",
    ) and not uses_defect_coordinate(args):
        raise ValueError(
            "native Point endpoints require a base- or lagged-defect port"
        )
    if args.point_architecture_reference is not None and args.backbone != "point":
        raise ValueError(
            "--point-architecture-reference requires --backbone point"
        )
    if not 0.0 <= args.point_capacity_tolerance < 1.0:
        raise ValueError("--point-capacity-tolerance must lie in [0,1)")
    if not 0.0 <= args.point_interface_anchor_fraction <= 1.0:
        raise ValueError("--point-interface-anchor-fraction must lie in [0,1]")
    if not math.isfinite(args.point_port_input_scale) \
            or args.point_port_input_scale <= 0.0:
        raise ValueError("--point-port-input-scale must be finite and positive")
    if args.output_dir.exists() and args.resume is None:
        raise FileExistsError(args.output_dir)
    if args.checkpoint_interval < 0 or args.training_tiles_per_sample < 0 or min(
        args.epochs, args.samples_per_epoch,
        args.validation_samples_per_trajectory, args.face_width,
        args.local_sdf_radius_cells, args.width, args.depth,
        args.response_attention_grid, args.response_attention_heads,
        args.point_d_model, args.point_surface_anchors,
        args.point_volume_anchors, args.point_geometry_anchors,
        args.point_depth, args.point_decoder_depth, args.point_heads,
        args.mass_impedance, args.momentum_impedance,
        args.normal_impedance, args.tangential_impedance,
    ) <= 0.0 or args.halo < 0 or min(
        args.base_reference_dt, args.response_reference_dt,
        args.impedance_reference_dt,
    ) < 0.0 or min(
        args.increment_field_weight, args.physical_field_weight,
        args.velocity_field_weight,
        args.projection_state_weight,
        args.solid_velocity_weight, args.near_body_field_weight,
        args.force_weight, args.lift_force_weight,
        args.selection_force_weight,
        args.selection_projection_weight,
        args.selection_monotonicity_weight,
        args.vorticity_weight, args.port_weight, args.local_field_weight,
        args.homotopy_field_weight,
        args.counterfactual_port_weight,
        args.recurrence_field_weight,
        args.recurrence_velocity_field_weight,
        args.recurrence_realization_weight,
        args.recurrence_target_port_weight,
    ) < 0.0:
        raise ValueError("invalid Airfoil Robin training configuration")
    recurrence_training = (
        args.recurrence_field_weight > 0.0
        or args.recurrence_velocity_field_weight > 0.0
        or args.recurrence_realization_weight > 0.0
        or args.recurrence_target_port_weight > 0.0
    )
    global_training_losses = any(weight > 0.0 for weight in (
        args.solid_velocity_weight, args.near_body_field_weight,
        args.force_weight, args.lift_force_weight,
    ))
    if args.training_tiles_per_sample > 0 and (
        recurrence_training or global_training_losses
    ):
        raise ValueError(
            "tile-minibatch endpoint pretraining is incompatible with "
            "whole-cover force/near-body or recurrence losses"
        )
    if args.recurrence_sweeps <= 0:
        raise ValueError("--recurrence-sweeps must be positive")
    if (
        not math.isfinite(args.recurrence_affine_alpha_min)
        or not math.isfinite(args.recurrence_affine_alpha_max)
        or args.recurrence_affine_alpha_min
        >= args.recurrence_affine_alpha_max
    ):
        raise ValueError("invalid recurrence affine-coordinate safeguards")
    if args.response_global_attention_layers < 0 or (
        args.response_global_attention_layers > 0
        and (
            args.response_architecture != "separable"
            or args.width % args.response_attention_heads != 0
        )
    ):
        raise ValueError("invalid zero-preserving response attention configuration")
    if args.base_face_role_dilation < 0:
        raise ValueError("base face-role dilation must be nonnegative")
    if args.analytic_robin_retraction and (
        args.robin_kind != "optimized_stokes"
        or args.face_width != 1
        or args.symmetric_stress
    ):
        raise ValueError(
            "analytic Robin retraction requires one-cell nonsymmetric "
            "optimized-Stokes faces"
        )
    native_point_response = (
        args.backbone == "point"
        and args.response_architecture == "entangled"
        and args.training_stage == "joint"
        and uses_defect_coordinate(args)
    )
    native_recurrence = (
        args.robin_kind in (
            "optimized_stokes", "dirichlet", "velocity_dirichlet",
        )
        and (
            (
                args.response_architecture == "separable"
                and args.training_stage == "response"
                and uses_defect_coordinate(args)
            )
            or native_point_response
        )
    )
    absolute_recurrence = (
        args.robin_kind == "conservative_flux"
        and args.response_architecture == "entangled"
        and args.training_stage == "joint"
        and not uses_defect_coordinate(args)
    )
    if recurrence_training and not (native_recurrence or absolute_recurrence):
        raise ValueError(
            "recurrence-aware training requires either a staged separable "
            "native response in a defect coordinate or a joint absolute "
            "conservative-flux response"
        )
    native_homotopy = (
        args.robin_kind in (
            "optimized_stokes", "dirichlet", "velocity_dirichlet",
        )
        and (
            (
                args.response_architecture == "separable"
                and args.training_stage != "base"
            )
            or native_point_response
        )
    )
    absolute_homotopy = (
        args.robin_kind == "conservative_flux"
        and args.response_architecture == "entangled"
        and args.training_stage == "joint"
    )
    if args.homotopy_field_weight > 0.0 and not (
        native_homotopy or absolute_homotopy
    ):
        raise ValueError(
            "Robin homotopy supervision requires either a non-base native "
            "separable response or a joint absolute conservative-flux response"
        )
    if args.backbone == "point" and args.homotopy_field_weight > 0.0 \
            and not args.sequential_counterfactual_backward:
        raise ValueError(
            "full-capacity Point homotopy requires sequential backward"
        )
    if args.predict_projection_state and (
        args.robin_kind not in (
            "optimized_stokes", "dirichlet", "velocity_dirichlet",
        )
        or not (
            args.response_architecture == "separable"
            or args.backbone == "point"
        )
    ):
        raise ValueError(
            "projection-state closure requires a separable native operator "
            "or a native Point endpoint"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.reference_checkpoint is None and (
        args.split_manifest is None or args.statistics_json is None
    ):
        raise ValueError(
            "without --reference-checkpoint, both --split-manifest and "
            "--statistics-json are required"
        )
    reference = (
        torch.load(
            args.reference_checkpoint, map_location="cpu", weights_only=False,
        )
        if args.reference_checkpoint is not None else None
    )
    split = (
        json.loads(args.split_manifest.read_text(encoding="utf-8"))
        if args.split_manifest is not None else reference["split"]
    )
    if not split.get("train") or not split.get("validation"):
        raise ValueError("Airfoil BVP split needs nonempty train and validation sets")
    if args.split_manifest is None and tuple(
        len(split[name]) for name in ("train", "validation", "test")
    ) != (41, 5, 6):
        raise RuntimeError("strict Airfoil BVP requires the frozen 41/5/6 split")
    if args.statistics_json is None:
        statistics_document = reference
    elif args.statistics_json.suffix in (".pt", ".pth"):
        statistics_document = torch.load(
            args.statistics_json, map_location="cpu", weights_only=False,
        )
    else:
        statistics_document = json.loads(
            args.statistics_json.read_text(encoding="utf-8")
        )
    statistics_payload = statistics_document["statistics"]
    statistics = FieldStatistics.from_json(statistics_payload)
    normalization_levels = statistics_document.get("levels")
    projection_statistics = (
        statistics_document.get("projection_statistics")
        if args.predict_projection_state else None
    )
    if args.predict_projection_state and projection_statistics is None:
        raise ValueError(
            "--predict-projection-state requires projection_statistics in "
            "the normalization payload"
        )
    residual_scale = tuple(
        float(delta) / float(state)
        for delta, state in zip(statistics.delta_std, statistics.state_std)
    )
    training = PairedTrajectoryCorpus(
        args.data_root, split["train"], "all", args.history_steps,
    )
    validation = PairedTrajectoryCorpus(
        args.data_root, split["validation"], "all", args.history_steps,
    )
    training_levels = getattr(training, args.spatial_level)
    validation_levels = getattr(validation, args.spatial_level)
    if normalization_levels is not None and tuple(
        int(level) for level in normalization_levels
    ) != (int(training_levels[0].level),):
        raise ValueError(
            "strict Airfoil BVP training uses the fine level only; its "
            "normalization levels must match that level exactly"
        )
    # q_n(3), two local SDFs, optional source/endpoint native AMR scale, two
    # wall velocities at both endpoints, optional lagged -dt*grad(p)(2),
    # optional q_n-q_nm1(3), and validity.  Mesh hierarchy and pressure-gradient
    # state are part of the actual incflo discretization; two-step history is an
    # explicit ablation rather than a hidden requirement.
    physical_channels = (
        10
        + (2 if training_levels[0].has_native_level else 0)
        + (2 if training_levels[0].has_gradp else 0)
        + (3 if args.history_steps == 2 else 0)
    )
    separable_options = {
        "projection_channels": (2 if args.predict_projection_state else 0),
        "response_global_attention_layers": (
            args.response_global_attention_layers
        ),
        "response_attention_grid": args.response_attention_grid,
        "response_attention_heads": args.response_attention_heads,
        "base_face_role_dilation": args.base_face_role_dilation,
        **(
            {"base_reference_dt": args.base_reference_dt}
            if args.base_reference_dt > 0.0 else {}
        ),
        **(
            {"response_reference_dt": args.response_reference_dt}
            if args.response_reference_dt > 0.0 else {}
        ),
    }
    point_options = {
        "physical_channels": physical_channels,
        "d_model": args.point_d_model,
        "surface_anchors": args.point_surface_anchors,
        "volume_anchors": args.point_volume_anchors,
        "geometry_anchors": args.point_geometry_anchors,
        "depth": args.point_depth,
        "decoder_depth": args.point_decoder_depth,
        "heads": args.point_heads,
        "robin_channels": robin_channels(args),
        "projection_channels": (2 if args.predict_projection_state else 0),
        "interface_anchor_fraction": args.point_interface_anchor_fraction,
        "port_input_scale": args.point_port_input_scale,
        "residual_scale": residual_scale,
    }
    solver = (
        AbsoluteRobinPointTileSolver2D(**point_options)
        if args.backbone == "point"
        else build_stokes_robin_tile_solver(
            response_architecture=args.response_architecture,
            physical_channels=physical_channels,
            width=args.width,
            depth=args.depth,
            residual_scale=residual_scale,
            **(
                separable_options
                if args.response_architecture == "separable" else {}
            ),
        )
        if args.robin_kind == "optimized_stokes"
        else SeparableNativeDirichletTileSolver2D(
            physical_channels=physical_channels,
            width=args.width,
            depth=args.depth,
            residual_scale=residual_scale,
            dirichlet_channels=(
                2 if args.robin_kind == "velocity_dirichlet" else 3
            ),
            **separable_options,
        )
        if args.robin_kind in ("dirichlet", "velocity_dirichlet")
        else build_absolute_robin_tile_solver(
            backbone="cnn",
            physical_channels=physical_channels,
            width=args.width,
            depth=args.depth,
            point_d_model=args.point_d_model,
            point_surface_anchors=args.point_surface_anchors,
            point_volume_anchors=args.point_volume_anchors,
            point_geometry_anchors=args.point_geometry_anchors,
            point_depth=args.point_depth,
            point_decoder_depth=args.point_decoder_depth,
            point_heads=args.point_heads,
            point_interface_anchor_fraction=(
                args.point_interface_anchor_fraction
            ),
            point_port_input_scale=args.point_port_input_scale,
            residual_scale=residual_scale,
        )
    ).to(device)
    total_parameter_count = sum(
        parameter.numel() for parameter in solver.parameters()
    )
    point_architecture_match = None
    if args.backbone == "point":
        if not isinstance(solver, AbsoluteRobinPointTileSolver2D):
            raise TypeError("point backbone did not build the strict-local AB-UPT")
        local_signature = solver.architecture_signature()
        point_architecture_match = {
            "local_signature": local_signature,
            "local_parameter_count": total_parameter_count,
            "reference_checkpoint": None,
            "reference_checkpoint_sha256": None,
            "reference_signature": None,
            "reference_parameter_count": None,
            "relative_parameter_gap": None,
            "verified": False,
        }
        if args.point_architecture_reference is not None:
            point_reference = torch.load(
                args.point_architecture_reference,
                map_location="cpu",
                weights_only=False,
            )
            reference_args = point_reference.get("model_args")
            if not isinstance(reference_args, dict):
                raise ValueError(
                    "Point architecture reference has no model_args dictionary"
                )
            reference_signature = {
                key: int(reference_args[key]) for key in local_signature
            }
            if reference_signature != local_signature:
                raise ValueError(
                    "local Point architecture does not match the global "
                    f"reference: {local_signature} != {reference_signature}"
                )
            reference_state = point_reference.get("model")
            if not isinstance(reference_state, dict):
                raise ValueError("Point architecture reference has no model state")
            reference_parameter_count = sum(
                int(value.numel()) for value in reference_state.values()
            )
            relative_gap = abs(
                total_parameter_count - reference_parameter_count
            ) / reference_parameter_count
            if relative_gap > args.point_capacity_tolerance:
                raise ValueError(
                    "local Point capacity differs from the global reference "
                    f"by {relative_gap:.3%}, above "
                    f"{args.point_capacity_tolerance:.3%}"
                )
            point_architecture_match.update({
                "reference_checkpoint": str(
                    args.point_architecture_reference.resolve()
                ),
                "reference_checkpoint_sha256": sha256(
                    args.point_architecture_reference
                ),
                "reference_signature": reference_signature,
                "reference_parameter_count": reference_parameter_count,
                "relative_parameter_gap": relative_gap,
                "verified": True,
            })
            del point_reference
    if args.training_stage != "joint" and args.response_architecture != "separable":
        raise ValueError("staged training requires the separable response architecture")
    if uses_defect_coordinate(args) and (
        args.robin_kind not in (
            "optimized_stokes", "dirichlet", "velocity_dirichlet",
        )
        or not (
            args.response_architecture == "separable"
            or args.backbone == "point"
        )
    ):
        raise ValueError(
            "defect ports require a separable native response or Point endpoint"
        )
    if args.backbone == "point" and uses_defect_coordinate(args) \
            and args.local_field_weight <= 0.0:
        raise ValueError(
            "a native Point defect endpoint requires a positive Local/Core loss"
        )
    initialization_payload = None
    point_zero_port_transfer = None
    if args.initialize is not None:
        initialization_payload = torch.load(
            args.initialize, map_location="cpu", weights_only=False,
        )
        if initialization_payload.get("contract") != checkpoint_contract(args) \
                and not (
                    args.initialize_base_only
                    or args.initialize_point_zero_port_transfer
                ):
            raise ValueError("initialization has a different Airfoil contract")
        initialized_backbone = initialization_payload.get(
            "backbone", initialization_payload.get("args", {}).get(
                "backbone", "cnn",
            ),
        )
        if initialized_backbone != args.backbone:
            raise ValueError("initialization uses a different local backbone")
        if initialization_payload.get(
            "response_architecture", "entangled",
        ) != args.response_architecture:
            raise ValueError("initialization uses a different response architecture")
        if int(initialization_payload.get("projection_channels", 0)) != (
            2 if args.predict_projection_state else 0
        ):
            raise ValueError("initialization uses a different projection-state contract")
        if tuple(initialization_payload["residual_scale_state_coordinates"]) != residual_scale:
            raise ValueError("initialization uses different Airfoil statistics")
        if args.initialize_point_zero_port_transfer:
            source_solver = local_solver_from_payload(
                initialization_payload, device=device,
            )
            source_solver = getattr(source_solver, "particular", source_solver)
            if not isinstance(source_solver, AbsoluteRobinPointTileSolver2D) \
                    or not isinstance(solver, AbsoluteRobinPointTileSolver2D):
                raise TypeError("zero-port transfer requires Point endpoints")
            port_mapping = None
            if args.initialize_point_port_policy == "reuse":
                if source_solver.robin_channels != solver.robin_channels:
                    raise ValueError(
                        "Point port reuse requires identical source and target "
                        "interface-channel counts"
                    )
                port_mapping = tuple(range(solver.robin_channels))
            point_zero_port_transfer = transfer_point_zero_port_function(
                source_solver, solver, port_column_mapping=port_mapping,
            )
            del source_solver
        elif args.initialize_base_only:
            current = solver.state_dict()
            source = initialization_payload["solver"]
            base_keys = [key for key in current if key.startswith("base_")]
            missing = [
                key for key in base_keys
                if key not in source or source[key].shape != current[key].shape
            ]
            if missing:
                raise ValueError(
                    f"base-only initialization is missing compatible keys: {missing[:4]}"
                )
            current.update({key: source[key] for key in base_keys})
            solver.load_state_dict(current, strict=True)
        else:
            solver.load_state_dict(initialization_payload["solver"], strict=True)
    if args.training_stage == "base":
        for name, parameter in solver.named_parameters():
            parameter.requires_grad_(name.startswith("base_"))
    elif args.training_stage == "response":
        if args.initialize is None and args.resume is None:
            raise ValueError("response-stage training requires a base initialization")
        for name, parameter in solver.named_parameters():
            parameter.requires_grad_(name.startswith("response_"))
    point_port_masks: dict[str, Tensor] = {}
    point_port_frozen: dict[str, Tensor] = {}
    point_port_effective_parameters = 0
    if args.point_port_columns_only:
        if not isinstance(solver, AbsoluteRobinPointTileSolver2D):
            raise TypeError("Point port-column training built a non-Point solver")
        (
            point_port_masks,
            point_port_frozen,
            point_port_effective_parameters,
        ) = configure_point_port_column_training(solver)
    trainable_parameters = [
        parameter for parameter in solver.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("training stage selected no parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr,
        weight_decay=(0.0 if args.point_port_columns_only else 1.0e-5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr * 0.05,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.amp,
    )
    start_epoch = 0
    best = float("inf")
    best_convergence = float("inf")
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("contract") != checkpoint_contract(args):
            raise ValueError("resume checkpoint has a different contract")
        saved_backbone = saved.get(
            "backbone", saved.get("args", {}).get("backbone", "cnn"),
        )
        if saved_backbone != args.backbone:
            raise ValueError("resume checkpoint uses a different local backbone")
        if saved.get(
            "response_architecture", "entangled",
        ) != args.response_architecture:
            raise ValueError("resume checkpoint uses a different response architecture")
        if int(saved.get("projection_channels", 0)) != (
            2 if args.predict_projection_state else 0
        ):
            raise ValueError("resume checkpoint uses a different projection-state contract")
        if saved.get("port_coordinate", "centered_absolute") != args.port_coordinate:
            raise ValueError("resume checkpoint uses a different port coordinate")
        if saved.get("training_stage", "joint") != args.training_stage:
            raise ValueError("resume checkpoint uses a different training stage")
        solver.load_state_dict(saved["solver"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = int(saved["epoch"]) + 1
        best = float(saved.get(
            "best_selection_score", saved["selection_score"],
        ))
        saved_curve = saved.get("validation", {}).get(
            "field_relative_l2_by_sweep", [],
        )
        best_convergence = float(saved.get(
            "best_convergence_score",
            saved.get(
                "convergence_score",
                min(saved_curve[1:] or saved_curve or [float("inf")]),
            ),
        ))
        convergence_path = args.output_dir / "checkpoint_best_convergence.pt"
        if not convergence_path.exists():
            torch.save(saved, convergence_path)
    generator = random.Random(args.seed)
    log = args.output_dir / "metrics.jsonl"
    try:
        for epoch in range(start_epoch, args.epochs):
            solver.train()
            started = time.perf_counter()
            totals = {
                "loss": 0.0, "field": 0.0, "velocity_field": 0.0,
                "local_field": 0.0,
                "projection_state": 0.0,
                "homotopy_field": 0.0,
                "physical_field": 0.0,
                "vorticity": 0.0,
                "solid_velocity": 0.0, "near_body_field": 0.0,
                "force": 0.0, "lift_force": 0.0,
                "port": 0.0, "counterfactual_port": 0.0,
                "recurrence_field": 0.0,
                "recurrence_velocity_field": 0.0,
                "recurrence_realization": 0.0,
                "recurrence_target_port": 0.0,
                "recurrence_affine_alpha": 0.0,
            }
            for sample_index in range(args.samples_per_epoch):
                trajectory = generator.randrange(len(training.paired))
                paired = training.paired[trajectory]
                index = generator.randrange(len(paired))
                reference_sample = paired[index]
                dataset = training_levels[trajectory]
                example = make_example(
                    reference_sample, dataset, statistics, args,
                    args.seed + epoch * 10_000_019 + sample_index, device,
                    projection_statistics=projection_statistics,
                )
                example = subsample_training_tiles(
                    example, args.training_tiles_per_sample, generator,
                )
                optimizer.zero_grad(set_to_none=True)
                context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" and args.amp else nullcontext()
                )
                with context:
                    need_local_prediction = (
                        args.local_field_weight > 0.0
                        or args.homotopy_field_weight > 0.0
                    )
                    local_prediction_precomputed = False
                    precompute_point_defect = (
                        args.backbone == "point"
                        and uses_defect_coordinate(args)
                    )
                    if need_local_prediction and (
                        args.robin_kind == "conservative_flux"
                        or precompute_point_defect
                    ):
                        # A capacity-matched Point endpoint nearly fills an
                        # H200 on a complete shifted cover.  Evaluate and, if
                        # required, backpropagate the communication-free
                        # particular solution before constructing the oracle
                        # graph.  Gradients from this objective accumulate in
                        # the same optimizer step, so this changes peak memory
                        # rather than the training objective.
                        absolute_target, local_roles = target_port(example, args)
                        local_port = (
                            torch.zeros_like(absolute_target)
                            if precompute_point_defect else
                            realized_port(
                                example["physical"][:, :3], example, args,
                            ).detach()
                        )
                        with (
                            nullcontext()
                            if args.local_field_weight > 0.0
                            else torch.no_grad()
                        ):
                            local_prediction_full = solver(
                                example["physical"], example["scalar"],
                                local_port, local_roles,
                            )
                        local_prediction = local_prediction_full[:, :3]
                        local_field = masked_increment_mse(
                            local_prediction, example["target_tiles"],
                            example["valid_tiles"], solver.residual_scale,
                        )
                        if args.local_field_weight > 0.0:
                            scaler.scale(
                                args.local_field_weight * local_field
                            ).backward()
                            local_prediction = local_prediction.detach()
                            local_field = local_field.detach()
                            del local_prediction_full
                        local_prediction_precomputed = True

                    prediction_full, port, roles, base_prediction, reference_port = (
                        oracle_prediction(solver, example, args)
                    )
                    prediction = prediction_full[:, :3]
                    field = masked_increment_mse(
                        prediction, example["target_tiles"], example["valid_tiles"],
                        solver.residual_scale,
                    )
                    velocity_field = masked_increment_mse(
                        prediction[:, :2], example["target_tiles"][:, :2],
                        example["valid_tiles"], solver.residual_scale[:, :2],
                    )
                    if need_local_prediction:
                        # In a lagged trace chart the zero response still has
                        # a prescribed q^n artificial-face value.  Use that
                        # actual particular solution for both the local loss
                        # and the affine boundary-response homotopy; the raw
                        # unconstrained base tensor is not the executed BVP.
                        if not local_prediction_precomputed:
                            if args.robin_kind == "conservative_flux":
                                local_port = realized_port(
                                    example["physical"][:, :3], example, args,
                                ).detach()
                                with (
                                    nullcontext()
                                    if args.local_field_weight > 0.0
                                    else torch.no_grad()
                                ):
                                    local_prediction_full = solver(
                                        example["physical"], example["scalar"],
                                        local_port, roles,
                                    )
                            else:
                                local_port = torch.zeros_like(port)
                                local_prediction_full = maybe_retract_prediction(
                                    base_prediction, local_port,
                                    example, args, reference_port,
                                )
                            local_prediction = local_prediction_full[:, :3]
                            local_field = masked_increment_mse(
                                local_prediction, example["target_tiles"],
                                example["valid_tiles"], solver.residual_scale,
                            )
                    else:
                        local_field = field.new_zeros(())
                        local_port = torch.zeros_like(port)
                    if args.homotopy_field_weight > 0.0:
                        # With the explicit transport/RHS state fixed, one
                        # native incflo step is a modified-Stokes solve.  Its
                        # solution is affine in Robin data.  This supplies
                        # labeled intermediate responses without consulting a
                        # monodomain target at any new boundary condition.
                        alpha = 0.2 + 0.6 * torch.rand(
                            (), device=port.device, dtype=port.dtype,
                        )
                        homotopy_port = (
                            local_port + alpha * (port - local_port)
                            if args.robin_kind == "conservative_flux"
                            else alpha * port
                        )
                        homotopy_target = local_prediction.detach() + alpha * (
                            example["target_tiles"] - local_prediction.detach()
                        )
                        # Evaluated together with the counterfactual branch
                        # below so the full Point graph can be backpropagated
                        # after the primary graph has been released.
                        homotopy_field = field.new_zeros(())
                    else:
                        homotopy_port = torch.zeros_like(port)
                        homotopy_target = example["target_tiles"]
                        homotopy_field = field.new_zeros(())
                    if example["target_projection_tiles"] is not None:
                        projection_state = masked_channel_mse(
                            prediction_full[:, 3:],
                            example["target_projection_tiles"],
                            example["valid_tiles"],
                        )
                    else:
                        projection_state = field.new_zeros(())
                    physical_field = physical_relative_square(
                        prediction, example["target_tiles"],
                        example["valid_tiles"], example["mean"], example["std"],
                    )
                    if global_training_losses:
                        prediction_global = assemble_subdomains_2d(
                            example["cover"], prediction, 1, args.assembly,
                        )
                        prediction_physical = (
                            prediction_global * example["std"] + example["mean"]
                        )
                        target_physical = (
                            example["target"] * example["std"] + example["mean"]
                        )
                        endpoint_sdf = reference_sample[
                            f"{args.spatial_level}_sdf_np1"
                        ][None].to(device)
                        solid_velocity = solid_velocity_relative_square_loss(
                            prediction_physical, target_physical, endpoint_sdf,
                        )
                        near_body_field = near_body_relative_square_loss(
                            prediction_physical, target_physical, endpoint_sdf,
                            band_width=4.0 * float(dataset.dx),
                        )
                        force_common = {
                            "dx": float(dataset.dx),
                            "window": tuple(dataset.window),
                            "penalty_time": float(dataset.penalty_time),
                            "rotation_sign": float(dataset.rotation_sign),
                        }
                        force = balanced_force_coefficient_relative_square(
                            prediction_physical, target_physical, endpoint_sdf,
                            reference_sample["motion_np1"][None].to(device),
                            **force_common,
                        )
                        lift_force = force_component_relative_square(
                            prediction_physical, target_physical, endpoint_sdf,
                            reference_sample["motion_np1"][None].to(device),
                            component="lift", **force_common,
                        )
                    else:
                        solid_velocity = field.new_zeros(())
                        near_body_field = field.new_zeros(())
                        force = field.new_zeros(())
                        lift_force = field.new_zeros(())
                    vort = vorticity_relative_square(
                        prediction, example["target_tiles"], example["valid_tiles"],
                        example["mean"], example["std"], example["dx"],
                    )
                    port_loss = port_mse(
                        prediction, port, roles, example, args, reference_port,
                    )
                    primary_loss = args.increment_field_weight * field \
                        + args.velocity_field_weight * velocity_field \
                        + args.local_field_weight * local_field \
                        + args.homotopy_field_weight * homotopy_field \
                        + args.physical_field_weight * physical_field \
                        + args.projection_state_weight * projection_state \
                        + args.vorticity_weight * vort \
                        + args.solid_velocity_weight * solid_velocity \
                        + args.near_body_field_weight * near_body_field \
                        + args.force_weight * force \
                        + args.lift_force_weight * lift_force \
                        + args.port_weight * port_loss
                    active = active_robin_mask(roles, args)
                    counterfactual_port = (
                        homotopy_port
                        if args.homotopy_field_weight > 0.0 else
                        0.5 * port + 0.5 * torch.roll(port, 1, 0)
                    ) * active
                    use_counterfactual = (
                        args.training_stage != "base"
                        and (
                            args.counterfactual_port_weight > 0.0
                            or args.homotopy_field_weight > 0.0
                        )
                    )
                    if use_counterfactual and not args.sequential_counterfactual_backward:
                        counterfactual_prediction = solver(
                            example["physical"], example["scalar"],
                            counterfactual_port, roles,
                        )
                        counterfactual_prediction = maybe_retract_prediction(
                            counterfactual_prediction, counterfactual_port,
                            example, args, reference_port,
                        )
                        counterfactual_loss = port_mse(
                            counterfactual_prediction, counterfactual_port,
                            roles, example, args, reference_port,
                        )
                        if args.homotopy_field_weight > 0.0:
                            homotopy_field = masked_increment_mse(
                                counterfactual_prediction[:, :3],
                                homotopy_target, example["valid_tiles"],
                                solver.residual_scale,
                            )
                        loss = primary_loss \
                            + args.counterfactual_port_weight * counterfactual_loss \
                            + args.homotopy_field_weight * homotopy_field
                    elif not use_counterfactual:
                        counterfactual_loss = primary_loss.new_zeros(())
                        loss = primary_loss
                if use_counterfactual and args.sequential_counterfactual_backward:
                    scaler.scale(primary_loss).backward()
                    with (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" and args.amp else nullcontext()
                    ):
                        counterfactual_prediction = solver(
                            example["physical"], example["scalar"],
                            counterfactual_port, roles,
                        )
                        counterfactual_prediction = maybe_retract_prediction(
                            counterfactual_prediction, counterfactual_port,
                            example, args, reference_port,
                        )
                        counterfactual_loss = port_mse(
                            counterfactual_prediction, counterfactual_port,
                            roles, example, args, reference_port,
                        )
                        if args.homotopy_field_weight > 0.0:
                            homotopy_field = masked_increment_mse(
                                counterfactual_prediction[:, :3],
                                homotopy_target, example["valid_tiles"],
                                solver.residual_scale,
                            )
                        counterfactual_objective = (
                            args.counterfactual_port_weight * counterfactual_loss
                            + args.homotopy_field_weight * homotopy_field
                        )
                    scaler.scale(counterfactual_objective).backward()
                    loss = primary_loss.detach() + counterfactual_objective.detach()
                else:
                    scaler.scale(loss).backward()

                # Oracle-port supervision alone does not constrain the ports
                # actually visited by the learned Schwarz map.  Generate that
                # trajectory with the current model, detach between sweeps,
                # and train every local response on its own on-policy port.
                # Detachment makes this a local BVP calibration rather than a
                # hidden end-to-end global network, while still requiring the
                # model-owned SNI fixed point to approach the native target.
                recurrence_field_total = field.new_zeros(())
                recurrence_velocity_field_total = field.new_zeros(())
                recurrence_realization_total = field.new_zeros(())
                recurrence_target_port_total = field.new_zeros(())
                recurrence_alpha_total = field.new_zeros(())
                if recurrence_training:
                    recurrence_reference = None
                    recurrence_pressure_reference = None
                    if args.port_coordinate == "lagged_defect":
                        recurrence_reference = declared_reference_port(
                            solver, example, args,
                        )
                    with torch.no_grad(), (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" and args.amp else nullcontext()
                    ):
                        initial_sweep = defect_sni_sweep(
                            solver, example, args, example["previous"],
                            reference_port=recurrence_reference,
                            pressure_reference_mean=recurrence_pressure_reference,
                            pressure_coarse_correction=False,
                        )
                    recurrence_current = initial_sweep.field.detach()
                    recurrence_port = initial_sweep.next_port.detach()
                    recurrence_origin_port = (
                        initial_sweep.imposed_port.detach()
                    )
                    recurrence_base_local = (
                        initial_sweep.local_candidates.detach()
                    )
                    recurrence_reference = (
                        getattr(initial_sweep, "reference_port", None)
                    )
                    if recurrence_reference is not None:
                        recurrence_reference = recurrence_reference.detach()
                    recurrence_pressure_reference = (
                        initial_sweep.pressure_reference_mean.detach()
                        if getattr(
                            initial_sweep, "pressure_reference_mean", None,
                        ) is not None else None
                    )
                    if args.sni_initialization == "temporal_extrapolation":
                        if args.robin_kind == "conservative_flux":
                            recurrence_port = temporal_extrapolated_absolute_port(
                                example, args,
                            )
                        else:
                            if recurrence_reference is None:
                                raise RuntimeError(
                                    "temporal SNI initialization requires a "
                                    "reference port"
                                )
                            recurrence_port = temporal_extrapolated_defect_port(
                                example, args, recurrence_reference,
                            )
                    global_valid = assemble_subdomains_2d(
                        example["cover"], example["valid_tiles"],
                        int(example["target"].shape[0]), args.assembly,
                    )
                    if (
                        getattr(args, "pressure_coarse_correction", False)
                        and recurrence_pressure_reference is None
                    ):
                        recurrence_pressure_reference = (
                            (recurrence_current[:, 2:3] * global_valid).sum(
                                dim=(-1, -2, -3),
                            )
                            / global_valid.sum(
                                dim=(-1, -2, -3),
                            ).clamp_min(1.0)
                        ).detach()
                    for _ in range(args.recurrence_sweeps):
                        with torch.no_grad():
                            recurrence_local_target, recurrence_alpha = (
                                projected_affine_recurrence_target(
                                    recurrence_port, port, active,
                                    recurrence_base_local,
                                    example["target_tiles"],
                                    origin_port=recurrence_origin_port,
                                    alpha_min=(
                                        args.recurrence_affine_alpha_min
                                    ),
                                    alpha_max=(
                                        args.recurrence_affine_alpha_max
                                    ),
                                )
                            )
                            # Strong native traces are part of the BVP label,
                            # not a loss shortcut.  The projected interior is
                            # paired with the actual on-policy boundary data.
                            recurrence_local_target = maybe_retract_prediction(
                                recurrence_local_target, recurrence_port,
                                example, args, recurrence_reference,
                            )[:, :3]
                            recurrence_global_target = assemble_subdomains_2d(
                                example["cover"], recurrence_local_target,
                                int(example["target"].shape[0]),
                                args.assembly,
                            )
                            # Compare like with like: a relaxed predicted port
                            # is supervised against the equally relaxed move
                            # toward the fixed point.  Comparing directly to
                            # ``port`` would demand 1/tau overshoot.
                            recurrence_target_next_port = (
                                relaxed_fixed_point_port_target(
                                    recurrence_port, port, active,
                                    args.relaxation,
                                )
                            )
                        with (
                            torch.autocast("cuda", dtype=torch.bfloat16)
                            if device.type == "cuda" and args.amp else nullcontext()
                        ):
                            recurrence_sweep = defect_sni_sweep(
                                solver, example, args, recurrence_current,
                                imposed_port=recurrence_port,
                                reference_port=recurrence_reference,
                                pressure_reference_mean=(
                                    recurrence_pressure_reference
                                ),
                            )
                            recurrence_global_field = masked_increment_mse(
                                recurrence_sweep.field,
                                recurrence_global_target,
                                global_valid, solver.residual_scale,
                            )
                            recurrence_local_field = masked_increment_mse(
                                recurrence_sweep.local_candidates,
                                recurrence_local_target,
                                example["valid_tiles"], solver.residual_scale,
                            )
                            recurrence_field_loss = 0.5 * (
                                recurrence_global_field
                                + recurrence_local_field
                            )
                            recurrence_global_velocity = masked_increment_mse(
                                recurrence_sweep.field[:, :2],
                                recurrence_global_target[:, :2], global_valid,
                                solver.residual_scale[:, :2],
                            )
                            recurrence_local_velocity = masked_increment_mse(
                                recurrence_sweep.local_candidates[:, :2],
                                recurrence_local_target[:, :2],
                                example["valid_tiles"],
                                solver.residual_scale[:, :2],
                            )
                            recurrence_velocity_field_loss = 0.5 * (
                                recurrence_global_velocity
                                + recurrence_local_velocity
                            )
                            recurrence_realization_loss = (
                                recurrence_sweep.realization_rms.square()
                            )
                            recurrence_target_port_loss = (
                                (
                                    (
                                        recurrence_sweep.next_port
                                        - recurrence_target_next_port
                                    ) / max(float(args.relaxation), 1.0e-6)
                                ).square() * active
                            ).sum() / active.sum().clamp_min(1.0)
                            # Do not attach an exactly-zero RMS through a
                            # zero scalar weight: d(sqrt(x)^2)/dx at x=0 can
                            # still produce 0*inf=NaN in autograd.  Dirichlet
                            # retraction realizes its trace exactly, so this
                            # branch is normally absent there.
                            recurrence_objective = (
                                args.recurrence_field_weight
                                * recurrence_field_loss
                                + args.recurrence_velocity_field_weight
                                * recurrence_velocity_field_loss
                                + args.recurrence_target_port_weight
                                * recurrence_target_port_loss
                            )
                            if args.recurrence_realization_weight > 0.0:
                                recurrence_objective = recurrence_objective \
                                    + args.recurrence_realization_weight \
                                    * recurrence_realization_loss
                            recurrence_objective = (
                                recurrence_objective / args.recurrence_sweeps
                            )
                        scaler.scale(recurrence_objective).backward()
                        recurrence_field_total = (
                            recurrence_field_total
                            + recurrence_field_loss.detach()
                        )
                        recurrence_velocity_field_total = (
                            recurrence_velocity_field_total
                            + recurrence_velocity_field_loss.detach()
                        )
                        recurrence_realization_total = (
                            recurrence_realization_total
                            + recurrence_realization_loss.detach()
                        )
                        recurrence_target_port_total = (
                            recurrence_target_port_total
                            + recurrence_target_port_loss.detach()
                        )
                        recurrence_alpha_total = (
                            recurrence_alpha_total
                            + recurrence_alpha.detach().mean()
                        )
                        recurrence_current = (
                            recurrence_sweep.field.detach()
                        )
                        recurrence_port = (
                            recurrence_sweep.next_port.detach()
                        )
                        recurrence_reference = (
                            getattr(recurrence_sweep, "reference_port", None)
                        )
                        if recurrence_reference is not None:
                            recurrence_reference = recurrence_reference.detach()
                        recurrence_pressure_reference = (
                            recurrence_sweep.pressure_reference_mean.detach()
                            if getattr(
                                recurrence_sweep,
                                "pressure_reference_mean", None,
                            ) is not None else None
                        )
                    recurrence_field_total /= args.recurrence_sweeps
                    recurrence_velocity_field_total /= args.recurrence_sweeps
                    recurrence_realization_total /= args.recurrence_sweeps
                    recurrence_target_port_total /= args.recurrence_sweeps
                    recurrence_alpha_total /= args.recurrence_sweeps
                    loss = loss.detach() \
                        + args.recurrence_field_weight * recurrence_field_total \
                        + args.recurrence_velocity_field_weight \
                        * recurrence_velocity_field_total \
                        + args.recurrence_realization_weight \
                        * recurrence_realization_total \
                        + args.recurrence_target_port_weight \
                        * recurrence_target_port_total
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                if args.point_port_columns_only:
                    restore_point_non_port_columns(
                        solver, point_port_masks, point_port_frozen,
                    )
                totals["loss"] += float(loss.detach())
                totals["field"] += float(field.detach())
                totals["velocity_field"] += float(velocity_field.detach())
                totals["local_field"] += float(local_field.detach())
                totals["projection_state"] += float(projection_state.detach())
                totals["homotopy_field"] += float(homotopy_field.detach())
                totals["physical_field"] += float(physical_field.detach())
                totals["vorticity"] += float(vort.detach())
                totals["solid_velocity"] += float(solid_velocity.detach())
                totals["near_body_field"] += float(near_body_field.detach())
                totals["force"] += float(force.detach())
                totals["lift_force"] += float(lift_force.detach())
                totals["port"] += float(port_loss.detach())
                totals["counterfactual_port"] += float(
                    counterfactual_loss.detach()
                )
                totals["recurrence_field"] += float(
                    recurrence_field_total
                )
                totals["recurrence_velocity_field"] += float(
                    recurrence_velocity_field_total
                )
                totals["recurrence_realization"] += float(
                    recurrence_realization_total
                )
                totals["recurrence_target_port"] += float(
                    recurrence_target_port_total
                )
                totals["recurrence_affine_alpha"] += float(
                    recurrence_alpha_total
                )
            metrics = validate(
                solver, validation, statistics, args, device,
                projection_statistics=projection_statistics,
            )
            scheduler.step()
            score = float(metrics[
                "oracle_increment_relative_l2"
                if args.selection_metric == "increment"
                else "oracle_relative_l2"
            ]) \
                + args.selection_force_weight * 0.5 * (
                    float(metrics["oracle_c_drag_relative_l2"])
                    + float(metrics["oracle_c_lift_relative_l2"])
                )
            if projection_statistics is not None:
                score += args.selection_projection_weight * float(
                    metrics["oracle_projection_relative_l2"]
                )
            curve = metrics[
                "increment_relative_l2_by_sweep"
                if args.selection_metric == "increment"
                else "field_relative_l2_by_sweep"
            ]
            convergence_curve = list(curve)
            if projection_statistics is not None:
                convergence_curve = [
                    float(field_error)
                    + args.selection_projection_weight * float(projection_error)
                    for field_error, projection_error in zip(
                        curve, metrics["projection_relative_l2_by_sweep"],
                    )
                ]
            # A minimum-over-sweeps score can select a response that improves
            # once and subsequently diverges.  The convergence checkpoint is
            # instead judged at the declared final sweep and penalized for
            # every increase along the mean held-validation trajectory.
            monotonicity_penalty = float(sum(
                max(0.0, right - left)
                for left, right in zip(
                    convergence_curve, convergence_curve[1:],
                )
            ))
            convergence_score = float(convergence_curve[-1]) \
                + args.selection_monotonicity_weight * monotonicity_penalty
            record = {
                "epoch": epoch,
                "seconds": time.perf_counter() - started,
                "lr": optimizer.param_groups[0]["lr"],
                **{
                    f"train_{key}": value / args.samples_per_epoch
                    for key, value in totals.items()
                },
                **{
                    f"validation_{key}": value
                    for key, value in metrics.items() if key != "rows"
                },
            }
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            payload = {
                "contract": checkpoint_contract(args),
                "backbone": args.backbone,
                "response_architecture": args.response_architecture,
                "port_coordinate": args.port_coordinate,
                "training_stage": args.training_stage,
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in trainable_parameters
                ),
                "effective_trainable_parameter_count": (
                    point_port_effective_parameters
                    if args.point_port_columns_only else
                    sum(parameter.numel() for parameter in trainable_parameters)
                ),
                "zero_port_function_frozen": bool(
                    args.point_port_columns_only
                ),
                "initial_point_port_policy": (
                    args.initialize_point_port_policy
                    if args.initialize_point_zero_port_transfer else None
                ),
                "total_parameter_count": total_parameter_count,
                "point_architecture_match": point_architecture_match,
                "point_architecture_variant": (
                    solver.architecture_variant
                    if isinstance(solver, AbsoluteRobinPointTileSolver2D)
                    else None
                ),
                "point_anchor_policy": (
                    (
                        "geometry_bank_physical_surface_bank_artificial_v1"
                        if args.point_interface_anchor_fraction == 1.0 else
                        "physical_and_artificial_surface_split_v1"
                        if args.point_interface_anchor_fraction > 0.0 else
                        "near_surface_or_even_if_geometry_free_v1"
                    ) if args.backbone == "point" else None
                ),
                "point_anchor_configuration": (
                    {
                        "surface_anchor_count": args.point_surface_anchors,
                        "geometry_anchor_count": args.point_geometry_anchors,
                        "artificial_boundary_surface_fraction": (
                            args.point_interface_anchor_fraction
                        ),
                        "parameter_neutral": True,
                    }
                    if args.backbone == "point" else None
                ),
                "locality_contract": STRICT_LOCALITY_CONTRACT,
                "solver": solver.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "selection_score": score,
                "convergence_score": convergence_score,
                # Preserve historical minima in checkpoint_last so a
                # wall-time resume cannot overwrite an earlier better model.
                "best_selection_score": min(best, score),
                "best_convergence_score": min(
                    best_convergence, convergence_score,
                ),
                "field_only_best_sweep_error": float(min(curve[1:])),
                "field_only_final_sweep_error": float(curve[-1]),
                "convergence_final_minus_sweep0": float(
                    convergence_curve[-1] - convergence_curve[0]
                ),
                "convergence_monotonicity_penalty": monotonicity_penalty,
                "args": vars(args),
                "reference_checkpoint": (
                    str(args.reference_checkpoint.resolve())
                    if args.reference_checkpoint is not None else None
                ),
                "reference_checkpoint_sha256": (
                    sha256(args.reference_checkpoint)
                    if args.reference_checkpoint is not None else None
                ),
                "statistics": statistics.to_json(),
                "normalization_levels": normalization_levels,
                "projection_statistics": projection_statistics,
                "projection_channels": (
                    2 if args.predict_projection_state else 0
                ),
                "native_markov_state_closed": bool(
                    args.predict_projection_state
                    and training_levels[0].has_gradp
                ),
                "native_amr_discretization_conditioned": bool(
                    training_levels[0].has_native_level
                ),
                "statistics_json": (
                    str(args.statistics_json.resolve())
                    if args.statistics_json is not None else None
                ),
                "statistics_json_sha256": (
                    sha256(args.statistics_json)
                    if args.statistics_json is not None else None
                ),
                "split": split,
                "split_manifest": (
                    str(args.split_manifest.resolve())
                    if args.split_manifest is not None else None
                ),
                "split_manifest_sha256": (
                    sha256(args.split_manifest)
                    if args.split_manifest is not None else None
                ),
                "validation": metrics,
                "physical_channels": tuple(filter(None, (
                    "q_n", "local_sdf_n", "local_sdf_np1",
                    (
                        "local_mesh_scale_n"
                        if training_levels[0].has_native_level else None
                    ),
                    (
                        "local_mesh_scale_np1"
                        if training_levels[0].has_native_level else None
                    ),
                    "local_wall_velocity_n", "local_wall_velocity_np1",
                    (
                        "lagged_minus_dt_gradp"
                        if training_levels[0].has_gradp else None
                    ),
                    (
                        "q_n_minus_q_nm1"
                        if args.history_steps == 2 else None
                    ),
                    "valid",
                ))),
                "physical_channel_count": physical_channels,
                "scalar_channels": STRICT_LOCAL_SCALAR_NAMES,
                "residual_scale_state_coordinates": residual_scale,
                "absolute_coordinate_prompt": False,
                "relative_subdomain_coordinate_prompt": args.backbone == "point",
                "global_sdf_crop": False,
                "boundary_semantics": (
                    "strong standardized velocity Dirichlet defect; pressure "
                    "left to the local incompressible solve"
                    if args.robin_kind == "velocity_dirichlet" else
                    "strong standardized state Dirichlet defect"
                    if args.robin_kind == "dirichlet" else
                    (
                        "normalized -sigma*n-S*v optimized Stokes Robin "
                        "port with analytic one-cell retraction"
                        if args.analytic_robin_retraction else
                        "normalized -sigma*n-S*v optimized Stokes Robin port"
                    )
                    if args.robin_kind == "optimized_stokes"
                    else "absolute normalized conservative-flux Robin port"
                ),
                "optimized_robin_time_scaling": (
                    "a0*(dt_ref/dt), b0*sqrt(dt_ref/dt)"
                    if args.robin_kind == "optimized_stokes"
                    and args.impedance_reference_dt > 0.0
                    else "fixed coefficients"
                ),
                "particular_solution_time_scaling": (
                    "first-order dt/dt_ref"
                    if args.base_reference_dt > 0.0 else "learned only"
                ),
                "communicated_correction_time_scaling": (
                    "first-order dt/dt_ref"
                    if args.response_reference_dt > 0.0 else "learned only"
                ),
                "robin_channels": robin_channels(args),
                "frozen_field_center": None,
                "sni_recurrence_used_for_training": recurrence_training,
                "training_recurrence_sweeps": (
                    args.recurrence_sweeps if recurrence_training else 0
                ),
                "training_recurrence_gradient_policy": (
                    "detached_between_sweeps"
                    if recurrence_training else None
                ),
                "training_recurrence_target_policy": (
                    "projected_affine_local_bvp_with_relaxed_fixed_point_port"
                    if recurrence_training else None
                ),
                "training_recurrence_affine_alpha_safeguards": (
                    [
                        args.recurrence_affine_alpha_min,
                        args.recurrence_affine_alpha_max,
                    ] if recurrence_training else None
                ),
                "checkpoint_selection_uses_recurrence": True,
                "protected_test_trajectories_opened": 0,
                "initialization": (
                    str(args.initialize.resolve())
                    if args.initialize is not None else None
                ),
                "initialization_sha256": (
                    sha256(args.initialize)
                    if args.initialize is not None else None
                ),
                "initialization_base_only": args.initialize_base_only,
                "initialization_point_zero_port_transfer": (
                    point_zero_port_transfer
                ),
            }
            torch.save(payload, args.output_dir / "checkpoint_last.pt")
            if args.checkpoint_interval > 0 and (
                (epoch + 1) % args.checkpoint_interval == 0
                or epoch + 1 == args.epochs
            ):
                candidate = dict(payload)
                candidate.pop("optimizer", None)
                candidate.pop("scheduler", None)
                candidate["checkpoint_selection_role"] = (
                    "candidate_only_pending_model_owned_rollout_screen"
                )
                torch.save(
                    candidate,
                    args.output_dir / f"checkpoint_epoch{epoch:03d}.pt",
                )
            if score < best:
                best = score
                torch.save(payload, args.output_dir / "checkpoint_best.pt")
            if convergence_score < best_convergence:
                best_convergence = convergence_score
                convergence_payload = dict(payload)
                # This file is selected by the held-validation recurrence.
                # The payload separately records whether training also saw
                # detached model-owned recurrence states.
                convergence_payload["checkpoint_selection_uses_recurrence"] = True
                convergence_payload["checkpoint_selection_role"] = (
                    "held_validation_final_sweep_with_monotonicity_penalty"
                )
                torch.save(
                    convergence_payload,
                    args.output_dir / "checkpoint_best_convergence.pt",
                )
            print(json.dumps(record), flush=True)
    finally:
        training.close()
        validation.close()


if __name__ == "__main__":
    main()
