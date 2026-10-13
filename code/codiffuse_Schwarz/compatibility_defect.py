"""Label-free compatibility defects for one-pass local composition.

The provisional tile fields are already available after the single frozen
local-operator call.  Their normalized-POU realization can therefore be
assembled and restricted back to the cover without another neural evaluation.
The difference

    d_i = R_i A_chi({v_j}) - v_i

is the correction-direction form of the broken-space consensus residual.  It
is zero for restrictions of a common global field and is an observable right
hand side for a reduced interface/Schur correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .residual_port import ResidualPortSpec, project_residual_ports


Tensor = torch.Tensor


@dataclass(frozen=True)
class ConsensusDefect:
    """POU consensus and its correction-direction broken-space defect."""

    assembled: Tensor
    restricted_consensus: Tensor
    correction_direction: Tensor


@dataclass(frozen=True)
class CompatibilityDamping:
    """One-shot residual-minimizing scale and its auditable residual norms."""

    scale: Tensor
    unconstrained_scale: Tensor
    baseline_squared_norm: Tensor
    damped_squared_norm: Tensor


def residual_minimizing_damping(
    residual_zero: Tensor,
    residual_direction: Tensor,
    *,
    weight: Tensor | None = None,
    lower: float = 0.0,
    upper: float = 1.0,
    relative_ridge: float = 1.0e-8,
) -> CompatibilityDamping:
    """Minimize a linear compatibility residual along one proposed direction.

    Inputs have one leading case dimension; every remaining dimension is part
    of the compatibility norm.  For ``r(alpha)=r0+alpha*dr``, the unregularized
    minimizer is ``-<r0,dr>/<dr,dr>``.  A nonnegative relative ridge handles a
    null direction, and clipping to an interval containing zero guarantees
    that the selected weighted residual norm cannot exceed the baseline norm
    (up to floating-point roundoff).

    This operation consumes no target, PDE solve, or neural call.  It only
    safeguards a correction direction already predicted by the one-message
    model; it does not guarantee field or quantity-of-interest accuracy.
    """
    if residual_zero.shape != residual_direction.shape or residual_zero.ndim < 2:
        raise ValueError("compatibility residuals must have matching B,... shapes")
    if not residual_zero.is_floating_point() or not residual_direction.is_floating_point():
        raise ValueError("compatibility residuals must be floating-point tensors")
    if not lower <= 0.0 <= upper or relative_ridge < 0.0:
        raise ValueError("damping interval must contain zero and ridge be nonnegative")
    if not torch.isfinite(residual_zero).all() or not torch.isfinite(residual_direction).all():
        raise ValueError("compatibility residuals must be finite")
    if weight is None:
        weighted = torch.ones_like(residual_zero)
    else:
        try:
            weighted = torch.broadcast_to(weight.to(residual_zero), residual_zero.shape)
        except RuntimeError as error:
            raise ValueError("compatibility weight is not broadcastable") from error
        if not torch.isfinite(weighted).all() or torch.any(weighted < 0):
            raise ValueError("compatibility weights must be finite and nonnegative")

    axes = tuple(range(1, residual_zero.ndim))
    baseline = (weighted * residual_zero.square()).sum(axes)
    cross = (weighted * residual_zero * residual_direction).sum(axes)
    direction = (weighted * residual_direction.square()).sum(axes)
    ridge = relative_ridge * baseline.clamp_min(torch.finfo(residual_zero.dtype).tiny)
    unconstrained = -cross / (direction + ridge).clamp_min(
        torch.finfo(residual_zero.dtype).tiny,
    )
    scale = unconstrained.clamp(min=lower, max=upper)
    view = (scale.shape[0],) + (1,) * (residual_zero.ndim - 1)
    damped = residual_zero + scale.reshape(view) * residual_direction
    damped_norm = (weighted * damped.square()).sum(axes)
    return CompatibilityDamping(scale, unconstrained, baseline, damped_norm)


def pou_consensus_defect(
    local_tiles: Tensor,
    cover: Any,
    batch: int,
    *,
    valid: Tensor | None = None,
    retile_padding_mode: str | None = None,
) -> ConsensusDefect:
    """Assemble, retile, and subtract a 2D or 3D Cartesian tile field.

    ``cover`` must expose ``tile_count``, ``assemble_weighted`` and ``tile``.
    The optional padding mode is useful for the 3D cover; the 2D cover has no
    padding-mode argument.  Invalid/padded cells are removed after subtraction.
    The operation is deterministic and differentiable, but consumes neither a
    target field nor a discrete PDE operator.
    """
    if batch <= 0 or local_tiles.ndim not in (4, 5):
        raise ValueError("local tiles must be a nonempty 2D/3D tile tensor")
    tile_count = int(cover.tile_count)
    if local_tiles.shape[0] != batch * tile_count:
        raise ValueError("local tile batch is incompatible with the cover")
    if valid is not None and valid.shape != (
        local_tiles.shape[0], 1, *local_tiles.shape[2:]
    ):
        raise ValueError("validity must be a matching one-channel tile field")

    assembled = cover.assemble_weighted(local_tiles, batch)
    if retile_padding_mode is None:
        consensus = cover.tile(assembled)
    else:
        consensus = cover.tile(assembled, padding_mode=retile_padding_mode)
    if consensus.shape != local_tiles.shape:
        raise RuntimeError("retiled POU consensus changed the local tile shape")
    defect = consensus - local_tiles
    if valid is not None:
        defect = defect * valid
        consensus = consensus * valid
    return ConsensusDefect(assembled, consensus, defect)


def project_consensus_defect_ports(
    local_tiles: Tensor,
    cover: Any,
    batch: int,
    valid: Tensor,
    spec: ResidualPortSpec,
    *,
    functionals: Tensor | None = None,
    retile_padding_mode: str | None = None,
) -> tuple[Tensor, ConsensusDefect]:
    """Project the observable consensus defect in the target port basis."""
    result = pou_consensus_defect(
        local_tiles,
        cover,
        batch,
        valid=valid,
        retile_padding_mode=retile_padding_mode,
    )
    ports = project_residual_ports(
        result.correction_direction,
        valid,
        spec,
        functionals=functionals,
    )
    return ports, result
