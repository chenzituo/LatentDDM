"""Optimized stress--velocity Robin traces for a time-discrete Airfoil BVP.

This is the degree-zero :math:`S^{-1,0}` interface condition of Blayo et al.
specialized to axis-aligned Cartesian subdomains.  It intentionally excludes
the advective momentum flux: after semi-implicit time discretization the local
iteration solves a modified Stokes/Oseen BVP and exchanges its stress trace.
"""

from __future__ import annotations

import torch

from .airfoil_flux_conditioned_sni import (
    FACE_NORMALS_2D,
    _batch_viscosity,
    directional_face_roles_2d,
)


Tensor = torch.Tensor
CONTRACT = "airfoil_time_discrete_stokes_robin_s_minus_1_0_v1"


def stokes_robin_port_2d(
    state: Tensor,
    *,
    state_mean: Tensor,
    state_std: Tensor,
    dx: float,
    viscosity: Tensor | float,
    normal_impedance: Tensor | float,
    tangential_impedance: Tensor | float,
    face_width: int = 1,
    neighbor_faces: Tensor | None = None,
    valid: Tensor | None = None,
    symmetric_stress: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return four physical ``-sigma n - S v`` face blocks.

    Each face block is represented in its receiver's local normal/tangential
    frame.  The outward normals follow north, south, west, east ordering.  The
    same receiver normal must be used when evaluating a neighboring field;
    therefore a global RAS field can be tiled and passed directly here.
    """
    if state.ndim != 4 or state.shape[1] != 3 or dx <= 0.0:
        raise ValueError("state must be normalized B,3,H,W with positive dx")
    if face_width <= 0 or 2 * face_width > min(state.shape[-2:]):
        raise ValueError("invalid face width")
    mean = torch.as_tensor(
        state_mean, device=state.device, dtype=state.dtype,
    ).reshape(1, 3, 1, 1)
    std = torch.as_tensor(
        state_std, device=state.device, dtype=state.dtype,
    ).reshape(1, 3, 1, 1)
    if bool(torch.any(std <= 0.0)):
        raise ValueError("state standard deviations must be positive")
    physical = state * std + mean
    u, v, pressure = physical.unbind(1)
    du_dy, du_dx = torch.gradient(
        u, spacing=(dx, dx), dim=(-2, -1), edge_order=1,
    )
    dv_dy, dv_dx = torch.gradient(
        v, spacing=(dx, dx), dim=(-2, -1), edge_order=1,
    )
    nu = _batch_viscosity(viscosity, state)
    if symmetric_stress:
        sigma_xx = nu * du_dx - pressure
        sigma_xy = 0.5 * nu * (du_dy + dv_dx)
        sigma_yx = sigma_xy
        sigma_yy = nu * dv_dy - pressure
    else:
        sigma_xx = nu * du_dx - pressure
        sigma_xy = nu * du_dy
        sigma_yx = nu * dv_dx
        sigma_yy = nu * dv_dy - pressure

    roles = directional_face_roles_2d(
        state.shape[-2], state.shape[-1], face_width=face_width,
        device=state.device, dtype=state.dtype,
    ).expand(state.shape[0], -1, -1, -1)
    if neighbor_faces is not None:
        if neighbor_faces.shape != (state.shape[0], 4, 1, 1):
            raise ValueError("neighbor face mask must have shape B,4,1,1")
        roles = roles * neighbor_faces
    if valid is not None:
        if valid.shape != state[:, :1].shape:
            raise ValueError("valid mask must have shape B,1,H,W")
        roles = roles * valid

    normal_z = torch.as_tensor(
        normal_impedance, device=state.device, dtype=state.dtype,
    ).flatten()
    tangent_z = torch.as_tensor(
        tangential_impedance, device=state.device, dtype=state.dtype,
    ).flatten()
    for name, value in (("normal", normal_z), ("tangential", tangent_z)):
        if value.numel() not in (1, state.shape[0]) or bool(torch.any(value < 0.0)):
            raise ValueError(f"{name} impedance must be nonnegative scalar or batch")
    if normal_z.numel() == 1:
        normal_z = normal_z.expand(state.shape[0])
    if tangent_z.numel() == 1:
        tangent_z = tangent_z.expand(state.shape[0])
    normal_z = normal_z[:, None, None]
    tangent_z = tangent_z[:, None, None]

    blocks = []
    for face, (nx, ny) in enumerate(FACE_NORMALS_2D):
        # Deterministic right-handed local frame t=(-ny,nx).
        tx, ty = -ny, nx
        velocity_n = nx * u + ny * v
        velocity_t = tx * u + ty * v
        sigma_n_x = sigma_xx * nx + sigma_xy * ny
        sigma_n_y = sigma_yx * nx + sigma_yy * ny
        minus_traction_n = -(nx * sigma_n_x + ny * sigma_n_y)
        minus_traction_t = -(tx * sigma_n_x + ty * sigma_n_y)
        robin = torch.stack((
            minus_traction_n - normal_z * velocity_n,
            minus_traction_t - tangent_z * velocity_t,
        ), dim=1)
        blocks.append(robin * roles[:, face : face + 1])
    return torch.cat(blocks, dim=1), roles


def stokes_robin_scales_2d(
    state_mean: Tensor,
    state_std: Tensor,
    *,
    viscosity: float,
    dx: float,
    normal_impedance: float,
    tangential_impedance: float,
) -> Tensor:
    """Fixed positive scales for four normal/tangential Robin blocks."""
    mean = torch.as_tensor(state_mean).flatten()
    std = torch.as_tensor(state_std).flatten()
    if mean.numel() != 3 or std.numel() != 3 or min(std) <= 0.0:
        raise ValueError("state statistics must contain positive u,v,p scales")
    if min(viscosity, dx, normal_impedance, tangential_impedance) < 0.0 or dx == 0.0:
        raise ValueError("invalid physical scale")
    velocity = torch.maximum(mean[:2].abs().max(), std[:2].max()).clamp_min(1.0e-4)
    pressure = torch.maximum(mean[2].abs(), std[2]).clamp_min(1.0e-4)
    derivative = viscosity * velocity / dx
    normal = torch.maximum(pressure, derivative + normal_impedance * velocity)
    tangent = torch.maximum(pressure, derivative + tangential_impedance * velocity)
    return torch.stack((normal, tangent)).repeat(4)


def normalized_stokes_robin_port_2d(
    state: Tensor,
    *,
    robin_scale: Tensor,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    """Return the physical Robin port divided by a fixed channel scale."""
    port, roles = stokes_robin_port_2d(state, **kwargs)
    scale = torch.as_tensor(
        robin_scale, device=state.device, dtype=state.dtype,
    ).flatten()
    if scale.numel() != 8 or bool(torch.any(scale <= 0.0)):
        raise ValueError("Robin scale must contain eight positive values")
    return port / scale.reshape(1, 8, 1, 1), roles


def stokes_robin_increment_scales_2d(
    delta_std: Tensor,
    *,
    viscosity: float,
    dx: float,
    normal_impedance: float,
    tangential_impedance: float,
) -> Tensor:
    """Scale a Robin *change* using the training one-step increments."""
    delta = torch.as_tensor(delta_std).flatten().abs()
    if delta.numel() != 3 or min(delta) <= 0.0:
        raise ValueError("delta statistics must contain positive u,v,p scales")
    if min(viscosity, dx, normal_impedance, tangential_impedance) < 0.0 or dx == 0.0:
        raise ValueError("invalid physical scale")
    velocity = delta[:2].max().clamp_min(1.0e-8)
    pressure = delta[2].clamp_min(1.0e-8)
    derivative = viscosity * velocity / dx
    normal = (pressure + derivative + normal_impedance * velocity).clamp_min(1.0e-8)
    tangent = (pressure + derivative + tangential_impedance * velocity).clamp_min(1.0e-8)
    return torch.stack((normal, tangent)).repeat(4)


def normalized_stokes_robin_residual_port_2d(
    state: Tensor,
    reference_state: Tensor,
    *,
    robin_increment_scale: Tensor,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    """Represent an absolute Robin condition as ``R(state)-R(reference)``.

    The representation is invertible because ``reference_state`` is part of
    every local BVP input.  Centering removes the dominant known trace and
    exposes the solver-step interface update at order one.
    """
    if reference_state.shape != state.shape:
        raise ValueError("state and Robin reference must share B,3,H,W")
    port, roles = stokes_robin_port_2d(state, **kwargs)
    reference, reference_roles = stokes_robin_port_2d(reference_state, **kwargs)
    if not torch.equal(roles, reference_roles):
        raise RuntimeError("state-independent Robin roles disagree")
    scale = torch.as_tensor(
        robin_increment_scale, device=state.device, dtype=state.dtype,
    ).flatten()
    if scale.numel() != 8 or bool(torch.any(scale <= 0.0)):
        raise ValueError("Robin increment scale must contain eight positives")
    return (port - reference) / scale.reshape(1, 8, 1, 1), roles


def retract_normalized_stokes_robin_residual_2d(
    state: Tensor,
    reference_state: Tensor,
    desired_residual: Tensor,
    *,
    state_mean: Tensor,
    state_std: Tensor,
    robin_increment_scale: Tensor,
    dx: float,
    viscosity: Tensor | float,
    normal_impedance: Tensor | float,
    tangential_impedance: Tensor | float,
    neighbor_faces: Tensor | None = None,
    valid: Tensor | None = None,
    symmetric_stress: bool = False,
) -> Tensor:
    """Strongly impose a normalized nonsymmetric Stokes--Robin trace.

    The neural response supplies the interior field.  On each one-cell
    artificial face, the tangential velocity is recovered from the one-sided
    discrete Robin equation and pressure is then recovered from its normal
    equation.  This is a differentiable boundary retraction, not an extra
    PDE iteration.  It makes the learned local map obey the same interface
    contract assumed by the Schwarz recurrence.
    """
    if symmetric_stress:
        raise ValueError(
            "analytic Robin retraction currently supports nonsymmetric stress"
        )
    if state.shape != reference_state.shape or state.ndim != 4 \
            or state.shape[1] != 3:
        raise ValueError("Robin retraction requires matching B,3,H,W states")
    if desired_residual.shape != (state.shape[0], 8, *state.shape[-2:]):
        raise ValueError("desired Robin residual must contain four two-channel blocks")
    if dx <= 0.0:
        raise ValueError("Robin retraction requires positive dx")
    mean = torch.as_tensor(
        state_mean, device=state.device, dtype=state.dtype,
    ).reshape(1, 3, 1, 1)
    std = torch.as_tensor(
        state_std, device=state.device, dtype=state.dtype,
    ).reshape(1, 3, 1, 1)
    if bool(torch.any(std <= 0.0)):
        raise ValueError("state standard deviations must be positive")
    scale = torch.as_tensor(
        robin_increment_scale, device=state.device, dtype=state.dtype,
    ).flatten()
    if scale.numel() != 8 or bool(torch.any(scale <= 0.0)):
        raise ValueError("Robin increment scale must contain eight positives")
    reference_port, roles = stokes_robin_port_2d(
        reference_state,
        state_mean=state_mean,
        state_std=state_std,
        dx=dx,
        viscosity=viscosity,
        normal_impedance=normal_impedance,
        tangential_impedance=tangential_impedance,
        face_width=1,
        neighbor_faces=neighbor_faces,
        valid=valid,
        symmetric_stress=False,
    )
    desired = reference_port + desired_residual * scale.reshape(1, 8, 1, 1)
    physical = state * std + mean
    u, v, pressure = physical.unbind(1)
    nu = _batch_viscosity(viscosity, state)

    def batch_coefficient(value: Tensor | float, name: str) -> Tensor:
        coefficient = torch.as_tensor(
            value, device=state.device, dtype=state.dtype,
        ).flatten()
        if coefficient.numel() == 1:
            coefficient = coefficient.expand(state.shape[0])
        if coefficient.shape != (state.shape[0],) or bool(torch.any(
            coefficient < 0.0
        )):
            raise ValueError(f"{name} impedance must be nonnegative per sample")
        return coefficient[:, None, None]

    normal_z = batch_coefficient(normal_impedance, "normal")
    tangent_z = batch_coefficient(tangential_impedance, "tangential")
    tangent_denominator = (nu / dx + tangent_z).clamp_min(1.0e-12)
    u_sum = torch.zeros_like(u)
    u_count = torch.zeros_like(u)
    v_sum = torch.zeros_like(v)
    v_count = torch.zeros_like(v)
    for face, (nx, ny) in enumerate(FACE_NORMALS_2D):
        mask = roles[:, face]
        tangential = desired[:, 2 * face + 1]
        if nx != 0.0:
            interior = torch.roll(v, shifts=int(nx), dims=-1)
            proposal = (
                nu / dx * interior - nx * tangential
            ) / tangent_denominator
            v_sum = v_sum + proposal * mask
            v_count = v_count + mask
        else:
            interior = torch.roll(u, shifts=int(ny), dims=-2)
            proposal = (
                nu / dx * interior + ny * tangential
            ) / tangent_denominator
            u_sum = u_sum + proposal * mask
            u_count = u_count + mask
    u = torch.where(u_count > 0.0, u_sum / u_count.clamp_min(1.0), u)
    v = torch.where(v_count > 0.0, v_sum / v_count.clamp_min(1.0), v)

    _, du_dx = torch.gradient(u, spacing=(dx, dx), dim=(-2, -1))
    dv_dy, _ = torch.gradient(v, spacing=(dx, dx), dim=(-2, -1))
    pressure_sum = torch.zeros_like(pressure)
    pressure_count = torch.zeros_like(pressure)
    for face, (nx, ny) in enumerate(FACE_NORMALS_2D):
        mask = roles[:, face]
        normal = desired[:, 2 * face]
        if nx != 0.0:
            proposal = normal + nu * du_dx + normal_z * (nx * u)
        else:
            proposal = normal + nu * dv_dy + normal_z * (ny * v)
        pressure_sum = pressure_sum + proposal * mask
        pressure_count = pressure_count + mask
    pressure = torch.where(
        pressure_count > 0.0,
        pressure_sum / pressure_count.clamp_min(1.0), pressure,
    )
    normalized = (torch.stack((u, v, pressure), dim=1) - mean) / std
    if valid is not None:
        normalized = normalized * valid
    return normalized
