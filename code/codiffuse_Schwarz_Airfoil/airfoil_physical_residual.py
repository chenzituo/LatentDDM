"""Target-free mass, momentum, and Brinkman/control-volume consistency probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .airfoil_two_level import centered_gradient, fluid_divergence_rms
from .force_diagnostics import penalty_force_coefficients


Tensor = torch.Tensor


@dataclass(frozen=True)
class PhysicalConsistencyResidual:
    mass_divergence_rms: Tensor
    mass_boundary_flux: Tensor
    momentum_x: Tensor
    momentum_y: Tensor
    normalized_momentum_force: Tensor
    viscous_outer_x: Tensor
    viscous_outer_y: Tensor


def _outer_momentum_flux(
    state: Tensor, dx: float, viscosity: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Integrate full Newtonian momentum and mass flux on the outer boundary."""
    u, v, pressure = state[:, 0], state[:, 1], state[:, 2]
    du_dx, du_dy = centered_gradient(u, dx)
    dv_dx, dv_dy = centered_gradient(v, dx)
    nu = viscosity[:, None, None]
    mass = dx * (
        -u[:, :, 0].sum(1) + u[:, :, -1].sum(1)
        -v[:, 0, :].sum(1) + v[:, -1, :].sum(1)
    )
    flux_x = dx * (
        -(u[:, :, 0].square() + pressure[:, :, 0]).sum(1)
        +(u[:, :, -1].square() + pressure[:, :, -1]).sum(1)
        -(u[:, 0, :] * v[:, 0, :]).sum(1)
        +(u[:, -1, :] * v[:, -1, :]).sum(1)
    )
    flux_y = dx * (
        -(u[:, :, 0] * v[:, :, 0]).sum(1)
        +(u[:, :, -1] * v[:, :, -1]).sum(1)
        -(v[:, 0, :].square() + pressure[:, 0, :]).sum(1)
        +(v[:, -1, :].square() + pressure[:, -1, :]).sum(1)
    )
    # Conservative momentum flux is uu + pI - nu(grad u + grad u^T).
    normal_x = 2.0 * nu * du_dx
    normal_y = 2.0 * nu * dv_dy
    shear = nu * (du_dy + dv_dx)
    viscous_x = dx * (
        +normal_x[:, :, 0].sum(1)
        -normal_x[:, :, -1].sum(1)
        +shear[:, 0, :].sum(1)
        -shear[:, -1, :].sum(1)
    )
    viscous_y = dx * (
        +shear[:, :, 0].sum(1)
        -shear[:, :, -1].sum(1)
        +normal_y[:, 0, :].sum(1)
        -normal_y[:, -1, :].sum(1)
    )
    return mass, flux_x + viscous_x, flux_y + viscous_y, viscous_x, viscous_y


def physical_consistency_residual(
    previous_state: Tensor,
    state: Tensor,
    sdf: Tensor,
    motion: Tensor,
    *,
    dt: Tensor | float,
    dx: float,
    window: tuple[float, float, float, float],
    penalty_time: float,
    rotation_sign: float,
    viscosity: Tensor | float = 0.0,
    previous_sdf: Tensor | None = None,
    momentum_identity: Literal["all_box", "fluid_corrected"] = "all_box",
) -> PhysicalConsistencyResidual:
    """Compare Brinkman reaction with target-free control-volume momentum.

    ``all_box`` is the direct Brinkman identity: full-box momentum rate plus
    outer conservative stress flux plus raw penalty reaction.  The equivalent
    ``fluid_corrected`` form integrates only fluid momentum and adds the
    moving fictitious-solid momentum correction to the reaction.  Both remain
    prototype diagnostics rather than replacements for the CFD discretization
    or supervised drag/lift losses.
    """
    if previous_state.shape != state.shape or state.ndim != 4 or state.shape[1] < 3:
        raise ValueError("previous/state must be matching B,C>=3,H,W tensors")
    if sdf.shape != (state.shape[0], 1, *state.shape[-2:]):
        raise ValueError("sdf must match the state grid")
    delta_t = torch.as_tensor(dt, device=state.device, dtype=state.dtype).flatten()
    if delta_t.numel() == 1:
        delta_t = delta_t.expand(state.shape[0])
    if delta_t.shape != (state.shape[0],) or torch.any(delta_t <= 0.0):
        raise ValueError("dt must contain one positive value per batch item")

    if previous_sdf is None:
        previous_sdf = sdf
    if previous_sdf.shape != sdf.shape:
        raise ValueError("previous_sdf must match endpoint sdf")
    fluid = (sdf[:, 0] > 0.0).to(state)
    previous_fluid = (previous_sdf[:, 0] > 0.0).to(state)
    area = dx * dx
    nu = torch.as_tensor(
        viscosity, device=state.device, dtype=state.dtype,
    ).flatten()
    if nu.numel() == 1:
        nu = nu.expand(state.shape[0])
    if nu.shape != (state.shape[0],) or torch.any(nu < 0.0):
        raise ValueError("viscosity must contain one nonnegative value per batch item")
    mass_flux, flux_x, flux_y, viscous_x, viscous_y = _outer_momentum_flux(
        state, dx, nu,
    )
    reaction = penalty_force_coefficients(
        state, sdf, motion, dx=dx, window=window,
        penalty_time=penalty_time, rotation_sign=rotation_sign,
    )
    reaction_x = .5 * reaction.drag
    reaction_y = -.5 * reaction.lift / float(rotation_sign)
    if momentum_identity == "all_box":
        momentum_rate_x = (
            (state[:, 0] - previous_state[:, 0]).sum((-2, -1))
            * area / delta_t
        )
        momentum_rate_y = (
            (state[:, 1] - previous_state[:, 1]).sum((-2, -1))
            * area / delta_t
        )
    elif momentum_identity == "fluid_corrected":
        momentum_rate_x = (
            (
                state[:, 0] * fluid
                - previous_state[:, 0] * previous_fluid
            ).sum((-2, -1)) * area / delta_t
        )
        momentum_rate_y = (
            (
                state[:, 1] * fluid
                - previous_state[:, 1] * previous_fluid
            ).sum((-2, -1)) * area / delta_t
        )
        solid = 1.0 - fluid
        previous_solid = 1.0 - previous_fluid
        reaction_x = reaction_x + (
            (
                state[:, 0] * solid
                - previous_state[:, 0] * previous_solid
            ).sum((-2, -1)) * area / delta_t
        )
        reaction_y = reaction_y + (
            (
                state[:, 1] * solid
                - previous_state[:, 1] * previous_solid
            ).sum((-2, -1)) * area / delta_t
        )
    else:
        raise ValueError(f"unknown momentum identity {momentum_identity!r}")
    residual_x = momentum_rate_x + flux_x + reaction_x
    residual_y = momentum_rate_y + flux_y + reaction_y
    scale = (
        momentum_rate_x.square() + momentum_rate_y.square()
        + flux_x.square() + flux_y.square()
        + reaction_x.square() + reaction_y.square()
    ).clamp_min(1.0e-8)
    return PhysicalConsistencyResidual(
        mass_divergence_rms=fluid_divergence_rms(state, sdf, dx),
        mass_boundary_flux=mass_flux,
        momentum_x=residual_x,
        momentum_y=residual_y,
        normalized_momentum_force=(
            (residual_x.square() + residual_y.square()) / scale
        ),
        viscous_outer_x=viscous_x,
        viscous_outer_y=viscous_y,
    )
