"""Training-only value+flux latent-Schur teacher utilities."""

from __future__ import annotations

from dataclasses import replace

import torch

from .validate_latent_schur import (
    chart_corrected, measured_q, q_from_corrected,
)


Tensor = torch.Tensor


def inference_only_context(context):
    """Erase supervision-only fields before any teacher computation.

    The frozen endpoint and all interface observations have already been
    formed from geometry, coefficients, forcing, physical boundary data, and
    a fixed zero initial state.  The remaining target/oracle tensors are kept
    only as shape-compatible zeros because the shared chart context type also
    serves supervised training code.
    """
    return replace(
        context,
        target_normalized=torch.zeros_like(context.target_normalized),
        target_tiles=torch.zeros_like(context.target_tiles),
        oracle_code=torch.zeros_like(context.oracle_code),
    )


@torch.no_grad()
def _linearized_value_flux(chart, context, batch, spec, scale):
    """Exact chart Jacobian while evaluating geometry modes only once."""
    if getattr(chart, "chart_type", None) != "geometry_tied_linear_green":
        raise ValueError("teacher export requires the selected tied-linear chart")
    q0, weight = measured_q(context, batch, spec, scale, "value_flux")
    modes = chart._response_modes(
        context.local, context.physical, context.scalar, context.valid,
    )
    columns = []
    for tile in range(context.cover.tile_count):
        for coordinate in range(spec.code_width):
            corrected = context.local.clone()
            corrected[tile] = corrected[tile] + modes[tile, :, coordinate]
            q1 = q_from_corrected(
                corrected, context, batch, spec, scale, "value_flux",
            )
            columns.append((q1 - q0).reshape(-1))
    return q0.reshape(-1), torch.stack(columns, 1), weight.reshape(-1)


@torch.no_grad()
def solve_value_flux_teacher(
    chart, context, batch, spec, scale, *, ridge: float = 0.01,
    damping: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
) -> tuple[Tensor, float, float]:
    """Return target-free code, selected damping, and complete-q cycle square.

    This routine is permitted only for offline teacher export/diagnostics.  A
    production communicator must predict the returned code in one message and
    may not call this dense reduced solve.
    """
    case_count = context.local.shape[0] // context.cover.tile_count
    if case_count != 1:
        raise ValueError("teacher solve currently requires one case")
    if ridge <= 0.0 or not damping or any(not 0.0 <= x <= 1.0 for x in damping):
        raise ValueError("invalid teacher ridge/damping")
    q0, jacobian, weight = _linearized_value_flux(
        chart, context, batch, spec, scale,
    )
    active = weight > 0.5
    if not active.any():
        raise RuntimeError("value+flux teacher has no supported coordinates")
    j = jacobian[active].double(); q = q0[active].double()
    identity = torch.eye(j.shape[1], device=j.device, dtype=j.dtype)
    direction = torch.linalg.solve(
        j.T @ j + ridge * identity, -(j.T @ q),
    ).to(context.local.dtype)
    denominator = q.square().sum().clamp_min(1.0e-12)
    candidates = []
    for alpha in damping:
        code = alpha * direction
        corrected = chart_corrected(
            chart, context,
            code.reshape(1, context.cover.tile_count, spec.code_width),
        )
        measured = q_from_corrected(
            corrected, context, batch, spec, scale, "value_flux",
        ).reshape(-1)[active]
        ratio_square = float(measured.double().square().sum() / denominator)
        candidates.append((ratio_square, float(alpha), code))
    ratio_square, alpha, code = min(candidates, key=lambda item: item[0])
    if ratio_square > 1.0:
        return torch.zeros_like(direction).reshape(
            1, context.cover.tile_count, spec.code_width,
        ), 0.0, 1.0
    return code.reshape(1, context.cover.tile_count, spec.code_width), alpha, ratio_square
