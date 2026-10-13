"""Aerodynamic load diagnostics for penalized-IB airfoil state fields."""

from __future__ import annotations

from dataclasses import dataclass

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class ForceCoefficients:
    """Penalty-reaction coefficients and fictitious-fluid momentum."""

    drag: Tensor
    lift: Tensor
    momentum_x: Tensor
    momentum_y: Tensor


def penalty_force_coefficients(
    state: Tensor,
    sdf: Tensor,
    motion: Tensor,
    *,
    dx: float,
    window: tuple[float, float, float, float],
    penalty_time: float,
    rotation_sign: float,
) -> ForceCoefficients:
    """Reproduce the benchmark's Brinkman penalty-reaction load diagnostic.

    The nondimensional benchmark uses ``rho=U_inf=chord=1``, so division by
    dynamic pressure gives the factor two used below. Positive paper lift is
    mapped from the solver y force using the trajectory's rotation convention.
    """
    if state.ndim != 4 or state.shape[1] < 2:
        raise ValueError("state must have shape (batch, channels>=2, height, width)")
    if sdf.shape != (state.shape[0], 1, state.shape[2], state.shape[3]):
        raise ValueError("sdf shape must match the batched state grid")
    if motion.ndim != 2 or motion.shape[0] != state.shape[0] or motion.shape[1] < 2:
        raise ValueError("motion must have shape (batch, channels>=2)")
    if penalty_time <= 0.0:
        raise ValueError("penalty_time must be positive")

    _, _, height, width = state.shape
    x0, _, y0, _ = window
    x = x0 + (torch.arange(width, device=state.device, dtype=state.dtype) + 0.5) * dx
    y = y0 + (torch.arange(height, device=state.device, dtype=state.dtype) + 0.5) * dx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    omega = motion[:, 1].to(device=state.device, dtype=state.dtype)[:, None, None]
    wall_u = -omega * yy
    wall_v = omega * xx
    solid = sdf[:, 0] <= 0.0
    area = dx * dx

    penalty_x = ((wall_u - state[:, 0]) * solid).sum(dim=(-1, -2)) * area / penalty_time
    penalty_y = ((wall_v - state[:, 1]) * solid).sum(dim=(-1, -2)) * area / penalty_time
    reaction_x = -penalty_x
    reaction_y = -penalty_y
    momentum_x = (state[:, 0] * solid).sum(dim=(-1, -2)) * area
    momentum_y = (state[:, 1] * solid).sum(dim=(-1, -2)) * area
    return ForceCoefficients(
        drag=2.0 * reaction_x,
        lift=-float(rotation_sign) * 2.0 * reaction_y,
        momentum_x=momentum_x,
        momentum_y=momentum_y,
    )


def corrected_force_coefficients(
    endpoint: ForceCoefficients,
    previous: ForceCoefficients,
    dt: Tensor,
    *,
    rotation_sign: float,
) -> tuple[Tensor, Tensor]:
    """Add a backward-difference fictitious-fluid momentum correction."""
    dt = dt.to(endpoint.drag.device).flatten().clamp_min(1.0e-12)
    correction_x = (endpoint.momentum_x - previous.momentum_x) / dt
    correction_y = (endpoint.momentum_y - previous.momentum_y) / dt
    return (
        endpoint.drag + 2.0 * correction_x,
        endpoint.lift - float(rotation_sign) * 2.0 * correction_y,
    )


def force_coefficient_relative_square(
    prediction: Tensor,
    target: Tensor,
    sdf: Tensor,
    motion: Tensor,
    *,
    dx: float,
    window: tuple[float, float, float, float],
    penalty_time: float,
    rotation_sign: float,
    scale_floor: float = 1.0,
) -> Tensor:
    """Normalized supervised load error with a stable near-zero force scale."""
    predicted = penalty_force_coefficients(
        prediction, sdf, motion, dx=dx, window=window,
        penalty_time=penalty_time, rotation_sign=rotation_sign,
    )
    reference = penalty_force_coefficients(
        target, sdf, motion, dx=dx, window=window,
        penalty_time=penalty_time, rotation_sign=rotation_sign,
    )
    error = (predicted.drag - reference.drag).square() \
        + (predicted.lift - reference.lift).square()
    scale = reference.drag.square() + reference.lift.square() + scale_floor
    return (error / scale).mean()


def force_coefficient_action_relative_square(
    prediction: Tensor,
    teacher: Tensor,
    uncorrected: Tensor,
    sdf: Tensor,
    motion: Tensor,
    *,
    dx: float,
    window: tuple[float, float, float, float],
    penalty_time: float,
    rotation_sign: float,
    action_floor: float = 1.0e-8,
) -> Tensor:
    """Fraction of a teacher's load correction left unrecovered.

    This is the load-space counterpart of comparing corrected fields relative
    to the teacher's change from the frozen endpoint. It avoids normalizing a
    small but important load action by an arbitrary unit-force scale. The
    floor only stabilizes examples where the teacher intentionally makes no
    load correction.
    """
    if action_floor <= 0.0:
        raise ValueError("force-action floor must be positive")
    predicted = penalty_force_coefficients(
        prediction, sdf, motion, dx=dx, window=window,
        penalty_time=penalty_time, rotation_sign=rotation_sign,
    )
    expected = penalty_force_coefficients(
        teacher, sdf, motion, dx=dx, window=window,
        penalty_time=penalty_time, rotation_sign=rotation_sign,
    )
    endpoint = penalty_force_coefficients(
        uncorrected, sdf, motion, dx=dx, window=window,
        penalty_time=penalty_time, rotation_sign=rotation_sign,
    )
    error = (predicted.drag - expected.drag).square() \
        + (predicted.lift - expected.lift).square()
    teacher_action = (expected.drag - endpoint.drag).square() \
        + (expected.lift - endpoint.lift).square()
    return (error / teacher_action.clamp_min(action_floor)).mean()


def balanced_force_coefficient_relative_square(
    prediction: Tensor,
    target: Tensor,
    sdf: Tensor,
    motion: Tensor,
    *,
    dx: float,
    window: tuple[float, float, float, float],
    penalty_time: float,
    rotation_sign: float,
    component_floor: float = 0.1,
) -> Tensor:
    """Balance supervised drag and lift irrespective of their amplitudes."""
    if component_floor <= 0.0:
        raise ValueError("force component floor must be positive")
    predicted = penalty_force_coefficients(
        prediction, sdf, motion, dx=dx, window=window,
        penalty_time=penalty_time, rotation_sign=rotation_sign,
    )
    reference = penalty_force_coefficients(
        target, sdf, motion, dx=dx, window=window,
        penalty_time=penalty_time, rotation_sign=rotation_sign,
    )
    terms = []
    for value, expected in (
        (predicted.drag, reference.drag),
        (predicted.lift, reference.lift),
    ):
        scale = expected.square() + component_floor**2
        terms.append(((value - expected).square() / scale).mean())
    return torch.stack(terms).mean()


def force_component_relative_square(
    prediction: Tensor,
    target: Tensor,
    sdf: Tensor,
    motion: Tensor,
    *,
    component: str,
    dx: float,
    window: tuple[float, float, float, float],
    penalty_time: float,
    rotation_sign: float,
    component_floor: float = 0.1,
) -> Tensor:
    """Supervise one penalty-reaction component without cross-component masking."""
    if component not in ("drag", "lift") or component_floor <= 0.0:
        raise ValueError("force component must be drag/lift with a positive floor")
    predicted = penalty_force_coefficients(
        prediction, sdf, motion, dx=dx, window=window,
        penalty_time=penalty_time, rotation_sign=rotation_sign,
    )
    reference = penalty_force_coefficients(
        target, sdf, motion, dx=dx, window=window,
        penalty_time=penalty_time, rotation_sign=rotation_sign,
    )
    value = getattr(predicted, component)
    expected = getattr(reference, component)
    return (
        (value - expected).square()
        / (expected.square() + component_floor**2)
    ).mean()
