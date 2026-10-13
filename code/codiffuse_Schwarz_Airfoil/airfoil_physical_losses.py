"""Metric-aligned field losses shared by Airfoil local and rollout training."""

from __future__ import annotations

import torch


Tensor = torch.Tensor


def physical_relative_square(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    """Squared relative error after undoing state standardization."""
    physical_target = target * std + mean
    weight = valid.expand_as(prediction)
    return (
        (prediction - target).square() * std.square() * weight
    ).sum() / (physical_target.square() * weight).sum().clamp_min(1.0e-8)


def solid_velocity_relative_square_loss(
    prediction_physical: Tensor,
    target_physical: Tensor,
    sdf: Tensor,
    *,
    velocity_floor: float = 0.05,
) -> Tensor:
    """Resolve the solid velocity that determines Brinkman reaction loads."""
    if velocity_floor <= 0.0:
        raise ValueError("solid-velocity floor must be positive")
    solid = (sdf <= 0.0).to(prediction_physical)
    error = (
        (prediction_physical[:, :2] - target_physical[:, :2]).square()
        * solid
    ).sum()
    denominator = (
        target_physical[:, :2].square() * solid
    ).sum() + velocity_floor**2 * solid.sum() * 2
    return error / denominator.clamp_min(1.0e-8)


def near_body_relative_square_loss(
    prediction_physical: Tensor,
    target_physical: Tensor,
    sdf: Tensor,
    *,
    band_width: float,
) -> Tensor:
    """Metric-aligned field loss in a smooth band around the moving body."""
    if band_width <= 0.0:
        raise ValueError("near-body band width must be positive")
    weight = torch.exp(-sdf.abs() / band_width).expand_as(prediction_physical)
    return (
        (prediction_physical - target_physical).square() * weight
    ).sum() / (target_physical.square() * weight).sum().clamp_min(1.0e-8)
