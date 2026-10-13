"""Supervised objective shared by feature-first and residual-token controls."""

from __future__ import annotations

import torch

from .force_diagnostics import balanced_force_coefficient_relative_square
from .interface_trace import interface_trace_relative_square
from .train_codiffuse_schwarz_rollout import (
    surface_vorticity_relative_square,
)


Tensor = torch.Tensor


def relative_square(
    prediction: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
) -> Tensor:
    if weight is None:
        weight = torch.ones_like(target[:, :1])
    numerator = ((prediction - target).square() * weight).sum()
    denominator = (target.square() * weight).sum().clamp_min(1.0e-20)
    return numerator / denominator


def matched_patch_response_objective(
    predicted_tiles: Tensor,
    predicted_state: Tensor,
    example,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply the Patch Transformer target loss to a LatentDDM response.

    The local endpoint, cover, normalized target, near-body weighting, surface
    diagnostic, interface quadrature, and load metric are identical to the
    feature-first control.  Only the communicated representation and response
    ordering differ.
    """
    _, target_tiles, _, _, _, _, transition, dataset = example
    normalized = transition.cover.assemble_weighted(
        predicted_tiles, transition.batch_size,
    )
    target = transition.target
    field = relative_square(normalized, target)

    valid_tiles = transition.cover.valid_tiles(
        transition.batch_size, predicted_tiles.device, predicted_tiles.dtype,
    )
    local = relative_square(predicted_tiles, target_tiles, valid_tiles)
    consensus = relative_square(
        predicted_tiles, transition.cover.tile(normalized), valid_tiles,
    )
    sdf = transition.batch["sdf_np1"].to(predicted_state)
    near = torch.exp(-sdf.abs() / (4.0 * dataset.dx))
    near_body = relative_square(
        predicted_state, transition.q_np1, near,
    )
    surface = surface_vorticity_relative_square(
        predicted_state, transition.q_np1, sdf, dataset.dx,
    )
    motion = transition.batch["motion_np1"].to(predicted_state)
    interface = interface_trace_relative_square(
        predicted_state, transition.q_np1, motion,
        dx=dataset.dx, window=dataset.window,
        thickness=float(dataset.physical_parameters["thickness"]),
        pivot_fraction=float(dataset.physical_parameters["pivot_fraction"]),
        stations=64,
    )
    force = balanced_force_coefficient_relative_square(
        predicted_state, transition.q_np1, sdf, motion,
        dx=dataset.dx, window=dataset.window,
        penalty_time=dataset.penalty_time,
        rotation_sign=dataset.rotation_sign,
    )
    loss = (
        field + 0.20 * local + 0.10 * consensus + 0.20 * near_body
        + 0.03 * torch.log1p(surface)
        + 0.03 * torch.log1p(interface)
        + 0.03 * torch.log1p(force)
    )
    return loss, {
        "field": field,
        "local": local,
        "consensus": consensus,
        "near_body": near_body,
        "surface": surface,
        "interface": interface,
        "force": force,
    }
