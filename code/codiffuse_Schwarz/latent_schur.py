"""Target-free one-step projection in a frozen response chart.

The case adapters provide a differentiable map ``q(z)`` from a supported
response code to inference-observable compatibility/conservation residuals,

    q(z) = Q F_X(v + L_X(z)).

This module never sees a reference solution or a discrete PDE solve.  It
linearizes that map at the unchanged local endpoint and solves the reduced
least-squares problem

    min_z  1/2 ||q(0) + J z||_W^2 + ridge/2 ||z||^2,

where ``J = d q / d z |_{z=0}``.  The result is an exact latent-Schur control
for a linear chart/constraint and a single Gauss--Newton correction otherwise.
It is useful both as a production map when the reduced solve is affordable and
as a target-free teacher for an amortized one-message residual kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


Tensor = torch.Tensor
ResidualFunction = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class LatentSchurResult:
    """Result of one reduced Gauss--Newton/latent-Schur projection."""

    code: Tensor
    raw_code: Tensor
    jacobian: Tensor
    residual_before: Tensor
    residual_after: Tensor
    damping: float
    cycle_ratio: Tensor
    linearized_cycle_ratio: Tensor
    ridge: float


def _flat_finite(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have a floating dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value.reshape(-1)


def dense_response_jacobian(
    residual_function: ResidualFunction,
    zero_code: Tensor,
    *,
    vectorize: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return ``q(0)`` and the dense Jacobian ``d q / d z``.

    ``zero_code`` may retain any case-specific tile/code layout.  The returned
    residual and code axes are flattened only in the Jacobian.  Case adapters
    should mask unsupported coordinates in ``residual_function`` or pass a
    supported-code mask to :func:`one_step_latent_schur`.
    """
    zero = _flat_finite(zero_code, "zero_code").reshape_as(zero_code)

    def flattened(code: Tensor) -> Tensor:
        return _flat_finite(residual_function(code), "residual_function output")

    with torch.enable_grad():
        base = zero.detach().requires_grad_(True)
        residual = flattened(base)
        jacobian = torch.autograd.functional.jacobian(
            flattened,
            base,
            create_graph=False,
            strict=False,
            vectorize=vectorize,
        ).reshape(residual.numel(), base.numel())
    if not torch.isfinite(jacobian).all():
        raise ValueError("response Jacobian must be finite")
    return residual.detach(), jacobian.detach()


def solve_reduced_least_squares(
    residual: Tensor,
    jacobian: Tensor,
    *,
    residual_weights: Tensor | None = None,
    supported_code: Tensor | None = None,
    ridge: float = 1.0e-6,
) -> Tensor:
    """Solve the damped linearized compatibility equation ``J z = -q``.

    A square-root residual weight is applied to both ``q`` and rows of ``J``.
    Unsupported code coordinates are removed from the solve and returned as
    exact zeros.  ``torch.linalg.lstsq`` is used when ``ridge == 0``; positive
    ridge uses an augmented least-squares system and is therefore stable for a
    rank-deficient response chart without forming normal equations.
    """
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    q = _flat_finite(residual, "residual")
    J = _flat_finite(jacobian, "jacobian").reshape(jacobian.shape)
    if J.ndim != 2 or J.shape[0] != q.numel():
        raise ValueError("jacobian must be [residual_dim, code_dim]")

    if residual_weights is None:
        weight = torch.ones_like(q)
    else:
        weight = _flat_finite(residual_weights.to(q), "residual_weights")
        if weight.shape != q.shape or torch.any(weight < 0.0):
            raise ValueError("residual weights must be nonnegative and match q")
    sqrt_weight = weight.sqrt()
    weighted_J = J * sqrt_weight[:, None]
    weighted_rhs = -q * sqrt_weight

    if supported_code is None:
        support = torch.ones(J.shape[1], dtype=torch.bool, device=J.device)
    else:
        supplied = supported_code.to(device=J.device).reshape(-1)
        if supplied.numel() != J.shape[1]:
            raise ValueError("supported-code mask must match the code dimension")
        support = supplied > 0.5
    code = J.new_zeros(J.shape[1])
    if not bool(support.any()):
        return code

    active_J = weighted_J[:, support]
    if ridge > 0.0:
        identity = torch.eye(
            active_J.shape[1], dtype=active_J.dtype, device=active_J.device,
        )
        active_J = torch.cat((active_J, ridge**0.5 * identity), dim=0)
        weighted_rhs = torch.cat((
            weighted_rhs,
            weighted_rhs.new_zeros(identity.shape[0]),
        ))
    active_code = torch.linalg.lstsq(active_J, weighted_rhs).solution
    code[support] = active_code
    if not torch.isfinite(code).all():
        raise ValueError("reduced least-squares solution must be finite")
    return code


def _weighted_norm(residual: Tensor, weights: Tensor | None, floor: float) -> Tensor:
    flat = residual.reshape(-1)
    if weights is None:
        square = flat.square().sum()
    else:
        weight = weights.to(flat).reshape(-1)
        if weight.shape != flat.shape:
            raise ValueError("residual weights must match the residual")
        square = (weight * flat.square()).sum()
    return square.clamp_min(floor).sqrt()


def one_step_latent_schur(
    residual_function: ResidualFunction,
    zero_code: Tensor,
    *,
    residual_weights: Tensor | None = None,
    supported_code: Tensor | None = None,
    ridge: float = 1.0e-6,
    damping_candidates: Sequence[float] = (1.0, 0.5, 0.25, 0.0),
    vectorize_jacobian: bool = True,
    norm_floor: float = 1.0e-12,
) -> LatentSchurResult:
    """Compute one target-free latent-Schur correction with a code-space search.

    Damping selection observes only the supplied residual function and
    evaluates that function at every ``alpha * raw_code``. Including zero in
    ``damping_candidates`` makes its measured residual non-regression exact by
    construction. The returned code has the same layout as ``zero_code``.

    For a nonlinear response chart, this code-space search generally decodes
    the chart once per candidate and is not equivalent to scaling one decoded
    field direction. Production adapters that require one chart decode should
    solve for the raw code here, decode once, and perform an actual-residual
    field-space search along that decoded direction instead.
    """
    if norm_floor <= 0.0 or not damping_candidates:
        raise ValueError("norm_floor must be positive and damping candidates nonempty")
    candidates = tuple(float(alpha) for alpha in damping_candidates)
    if any(not 0.0 <= alpha <= 1.0 for alpha in candidates):
        raise ValueError("damping candidates must lie in [0,1]")

    q0, J = dense_response_jacobian(
        residual_function, zero_code, vectorize=vectorize_jacobian,
    )
    raw_flat = solve_reduced_least_squares(
        q0, J, residual_weights=residual_weights,
        supported_code=supported_code, ridge=ridge,
    )
    raw = raw_flat.reshape_as(zero_code)
    before_norm = _weighted_norm(q0, residual_weights, norm_floor)

    best_alpha = candidates[0]
    best_residual = _flat_finite(
        residual_function(raw * best_alpha), "residual after correction",
    ).detach()
    best_norm = _weighted_norm(best_residual, residual_weights, norm_floor)
    for alpha in candidates[1:]:
        candidate_residual = _flat_finite(
            residual_function(raw * alpha), "residual after correction",
        ).detach()
        candidate_norm = _weighted_norm(
            candidate_residual, residual_weights, norm_floor,
        )
        if bool(candidate_norm < best_norm):
            best_alpha = alpha
            best_residual = candidate_residual
            best_norm = candidate_norm

    code = raw * best_alpha
    linearized = q0 + J @ code.reshape(-1)
    return LatentSchurResult(
        code=code.detach(),
        raw_code=raw.detach(),
        jacobian=J,
        residual_before=q0,
        residual_after=best_residual,
        damping=best_alpha,
        cycle_ratio=(best_norm / before_norm).detach(),
        linearized_cycle_ratio=(
            _weighted_norm(linearized, residual_weights, norm_floor) / before_norm
        ).detach(),
        ridge=ridge,
    )
