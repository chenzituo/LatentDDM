"""Differentiable body-coordinate observables on a moving NACA interface."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from codiffuse_Schwarz_Airfoil.vorticity_projection import nonperiodic_vorticity


Tensor = torch.Tensor


def naca_half_thickness(xi: Tensor, thickness: float) -> tuple[Tensor, Tensor]:
    """Return NACA 00xx half thickness and its chordwise derivative."""
    root = xi.clamp_min(1.0e-6).sqrt()
    half = 5.0 * thickness * (
        .2969 * root - .1260 * xi - .3516 * xi.square()
        + .2843 * xi.pow(3) - .1036 * xi.pow(4)
    )
    derivative = 5.0 * thickness * (
        .2969 / (2.0 * root) - .1260 - 2.0 * .3516 * xi
        + 3.0 * .2843 * xi.square() - 4.0 * .1036 * xi.pow(3)
    )
    return half, derivative


def _surface_chart(
    motion: Tensor,
    *,
    thickness: float,
    pivot_fraction: float,
    dx: float,
    stations: int,
    normal_offsets: tuple[float, ...],
) -> tuple[Tensor, Tensor, Tensor]:
    if stations < 4 or not normal_offsets:
        raise ValueError("interface chart requires stations and normal offsets")
    xi = torch.linspace(
        .01, .99, stations, device=motion.device, dtype=motion.dtype,
    )
    half, slope = naca_half_thickness(xi, thickness)
    x_body = xi - pivot_fraction
    side = torch.tensor((1.0, -1.0), device=motion.device, dtype=motion.dtype)
    y_body = side[:, None] * half[None]
    tangent = torch.stack((
        torch.ones_like(y_body), side[:, None] * slope[None],
    ), dim=-1)
    tangent = F.normalize(tangent, dim=-1)
    normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
    normal[1] = -normal[1]
    offsets = torch.tensor(
        normal_offsets, device=motion.device, dtype=motion.dtype,
    ) * dx
    body = torch.stack((
        x_body[None].expand(2, -1), y_body,
    ), dim=-1)[:, None] + offsets[None, :, None, None] * normal[:, None]

    angle = motion[:, 0, None, None, None]
    cosine, sine = angle.cos(), angle.sin()
    x = cosine * body[None, ..., 0] - sine * body[None, ..., 1]
    y = sine * body[None, ..., 0] + cosine * body[None, ..., 1]
    points = torch.stack((x, y), dim=-1)
    return points, tangent, normal


def interface_trace(
    state: Tensor,
    motion: Tensor,
    *,
    dx: float,
    window: tuple[float, float, float, float],
    thickness: float,
    pivot_fraction: float,
    stations: int = 64,
    normal_offsets: tuple[float, ...] = (-1.5, -.5, .5, 1.5, 3.0),
) -> Tensor:
    """Sample `(u_t-wall, u_n-wall, p-gauge, omega)` on an ordered chart.

    The result has shape `B,4,2,normal_offsets,stations`; its two surface
    branches and signed-normal locations remain explicit rather than being
    pooled into an Eulerian SDF band.
    """
    if state.ndim != 4 or state.shape[1] < 3:
        raise ValueError("state must have shape B,C>=3,H,W")
    if motion.ndim != 2 or motion.shape[0] != state.shape[0] or motion.shape[1] < 2:
        raise ValueError("motion must have shape B,C>=2")
    if dx <= 0.0 or thickness <= 0.0 or not 0.0 < pivot_fraction < 1.0:
        raise ValueError("invalid geometry or spacing")
    points, tangent, normal = _surface_chart(
        motion.to(state), thickness=thickness, pivot_fraction=pivot_fraction,
        dx=dx, stations=stations, normal_offsets=normal_offsets,
    )
    batch, _, height, width = state.shape
    x0, _, y0, _ = window
    grid = points.clone()
    grid[..., 0] = 2.0 * (grid[..., 0] - x0) / (width * dx) - 1.0
    grid[..., 1] = 2.0 * (grid[..., 1] - y0) / (height * dx) - 1.0
    grid = grid.reshape(batch, 2 * len(normal_offsets), stations, 2)
    omega = nonperiodic_vorticity(state, dx)[:, None]
    sampled = F.grid_sample(
        torch.cat((state[:, :3], omega), dim=1), grid,
        mode="bilinear", padding_mode="border", align_corners=False,
    ).reshape(batch, 4, 2, len(normal_offsets), stations)

    angular_velocity = motion[:, 1, None, None, None]
    wall_u = -angular_velocity * points[..., 1]
    wall_v = angular_velocity * points[..., 0]
    relative_u = sampled[:, 0] - wall_u
    relative_v = sampled[:, 1] - wall_v
    angle = motion[:, 0, None, None, None]
    body_u = angle.cos() * relative_u + angle.sin() * relative_v
    body_v = -angle.sin() * relative_u + angle.cos() * relative_v
    tangent_full = tangent.to(state)
    normal_full = normal.to(state)
    tangential = (
        body_u * tangent_full[None, :, None, :, 0]
        + body_v * tangent_full[None, :, None, :, 1]
    )
    normal_velocity = (
        body_u * normal_full[None, :, None, :, 0]
        + body_v * normal_full[None, :, None, :, 1]
    )
    pressure = sampled[:, 2] - sampled[:, 2].mean(dim=-1, keepdim=True)
    return torch.stack((tangential, normal_velocity, pressure, sampled[:, 3]), dim=1)


def interface_trace_relative_square(
    prediction: Tensor,
    target: Tensor,
    motion: Tensor,
    *,
    dx: float,
    window: tuple[float, float, float, float],
    thickness: float,
    pivot_fraction: float,
    stations: int = 64,
) -> Tensor:
    """Channel-balanced supervised error on the moving interface chart."""
    predicted = interface_trace(
        prediction, motion, dx=dx, window=window, thickness=thickness,
        pivot_fraction=pivot_fraction, stations=stations,
    )
    reference = interface_trace(
        target, motion, dx=dx, window=window, thickness=thickness,
        pivot_fraction=pivot_fraction, stations=stations,
    )
    floors = prediction.new_tensor((.05, .05, .10, 1.0))[None, :, None, None, None]
    error = (predicted - reference).square().mean(dim=(2, 3, 4))
    scale = reference.square().mean(dim=(2, 3, 4)) + floors[:, :, 0, 0, 0].square()
    return (error / scale).mean()
