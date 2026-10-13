"""Differentiable discrete curl projection for incompressible 2D velocity."""

from __future__ import annotations

import torch


Tensor = torch.Tensor


def periodic_vorticity(velocity: Tensor, dx: float) -> Tensor:
    """Centered periodic curl with the same interior stencil used for evaluation."""
    if velocity.ndim != 4 or velocity.shape[1] < 2:
        raise ValueError("velocity must have shape (batch, channels>=2, height, width)")
    dv_dx = (
        torch.roll(velocity[:, 1], shifts=-1, dims=-1)
        - torch.roll(velocity[:, 1], shifts=1, dims=-1)
    ) / (2.0 * dx)
    du_dy = (
        torch.roll(velocity[:, 0], shifts=-1, dims=-2)
        - torch.roll(velocity[:, 0], shifts=1, dims=-2)
    ) / (2.0 * dx)
    return dv_dx - du_dy


def nonperiodic_vorticity(velocity: Tensor, dx: float) -> Tensor:
    """Centered interior curl with one-sided physical-domain boundaries."""
    if velocity.ndim != 4 or velocity.shape[1] < 2:
        raise ValueError("velocity must have shape (batch, channels>=2, height, width)")
    if velocity.shape[-2] < 2 or velocity.shape[-1] < 2 or dx <= 0.0:
        raise ValueError("spatial dimensions must be at least two and dx positive")
    u, v = velocity[:, 0], velocity[:, 1]
    dv_dx = torch.empty_like(v)
    du_dy = torch.empty_like(u)
    if v.shape[-1] > 2:
        dv_dx[..., 1:-1] = (v[..., 2:] - v[..., :-2]) / (2.0 * dx)
    dv_dx[..., 0] = (v[..., 1] - v[..., 0]) / dx
    dv_dx[..., -1] = (v[..., -1] - v[..., -2]) / dx
    if u.shape[-2] > 2:
        du_dy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
    du_dy[:, 0] = (u[:, 1] - u[:, 0]) / dx
    du_dy[:, -1] = (u[:, -1] - u[:, -2]) / dx
    return dv_dx - du_dy


def project_velocity_curl(
    field: Tensor,
    target_vorticity: Tensor,
    dx: float,
) -> Tensor:
    """Minimum-norm periodic velocity correction matching a target discrete curl.

    Pressure and any channels after `(u,v)` are copied unchanged.  The centered
    derivative's constant and Nyquist null modes are left untouched; a target
    produced by the same curl stencil has no component in those modes.
    """
    if field.ndim != 4 or field.shape[1] < 2:
        raise ValueError("field must have shape (batch, channels>=2, height, width)")
    if target_vorticity.shape != (field.shape[0], field.shape[2], field.shape[3]):
        raise ValueError("target vorticity shape does not match the field")

    working = field.float()
    target = target_vorticity.float()
    residual_hat = torch.fft.fft2(target - periodic_vorticity(working, dx))
    height, width = field.shape[-2:]
    kx = 2.0 * torch.pi * torch.fft.fftfreq(width, device=field.device)
    ky = 2.0 * torch.pi * torch.fft.fftfreq(height, device=field.device)
    symbol_y, symbol_x = torch.meshgrid(
        torch.sin(ky) / dx, torch.sin(kx) / dx, indexing="ij"
    )
    denominator = symbol_x.square() + symbol_y.square()
    representable = denominator[None] > 1.0e-12
    safe_denominator = torch.where(
        representable, denominator[None], torch.ones_like(denominator[None])
    )
    stream_hat = -residual_hat / safe_denominator
    stream_hat = torch.where(
        representable, stream_hat, torch.zeros_like(stream_hat)
    )
    correction_u = torch.fft.ifft2(
        -1j * symbol_y[None] * stream_hat
    ).real
    correction_v = torch.fft.ifft2(
        1j * symbol_x[None] * stream_hat
    ).real
    result = working.clone()
    result[:, 0] += correction_u
    result[:, 1] += correction_v
    return result.to(field.dtype)
