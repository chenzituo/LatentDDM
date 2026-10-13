"""Reduced pressure-like projection for one-pass multiresolution Airfoil fields."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover
from codiffuse_Schwarz_Airfoil.multiresolution import decode_composite, synchronize_levels
from codiffuse_Schwarz_Airfoil.multiresolution_inference import MultiresolutionPrediction


Tensor = torch.Tensor


@dataclass(frozen=True)
class PressureProjectionDiagnostics:
    divergence_before: Tensor
    divergence_after: Tensor
    rank: Tensor
    condition: Tensor


def centered_gradient(field: Tensor, dx: float) -> tuple[Tensor, Tensor]:
    """Centered interior gradient with one-sided outer boundaries."""
    if field.ndim < 3 or dx <= 0.0:
        raise ValueError("field must end in H,W and dx must be positive")
    grad_x = torch.empty_like(field)
    grad_y = torch.empty_like(field)
    grad_x[..., 1:-1] = (field[..., 2:] - field[..., :-2]) / (2.0 * dx)
    grad_x[..., 0] = (field[..., 1] - field[..., 0]) / dx
    grad_x[..., -1] = (field[..., -1] - field[..., -2]) / dx
    grad_y[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) / (2.0 * dx)
    grad_y[..., 0, :] = (field[..., 1, :] - field[..., 0, :]) / dx
    grad_y[..., -1, :] = (field[..., -1, :] - field[..., -2, :]) / dx
    return grad_x, grad_y


def velocity_divergence(field: Tensor, dx: float) -> Tensor:
    if field.ndim != 4 or field.shape[1] < 2:
        raise ValueError("field must have shape B,C>=2,H,W")
    du_dx, _ = centered_gradient(field[:, 0], dx)
    _, dv_dy = centered_gradient(field[:, 1], dx)
    return du_dx + dv_dy


def fluid_divergence_rms(field: Tensor, sdf: Tensor, dx: float) -> Tensor:
    """RMS divergence in the smooth fluid-interior diagnostic region."""
    divergence = velocity_divergence(field, dx)
    weight = _physical_taper(sdf, dx).square()
    return (
        (divergence.square() * weight).flatten(1).sum(1)
        / weight.flatten(1).sum(1).clamp_min(1.0)
    ).sqrt()


def _tile_pressure_basis(
    height: int,
    width: int,
    core: int,
    halo: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    cover = CartesianCover(height, width, core, halo)
    size = cover.tile_size
    patches = torch.zeros(
        cover.tile_count, cover.tile_count, size, size,
        device=device, dtype=dtype,
    )
    for tile in range(cover.tile_count):
        patches[tile, tile] = 1.0
    return cover.assemble_weighted(patches, 1)[0]


def _physical_taper(
    sdf: Tensor,
    dx: float,
    boundary_zero_cells: float = 0.0,
    boundary_ramp_cells: float = 4.0,
    body_zero_cells: float = 2.0,
    body_ramp_cells: float = 4.0,
) -> Tensor:
    """Smoothly suppress coarse corrections at the body and outer boundary."""
    if sdf.ndim != 4 or sdf.shape[1] != 1:
        raise ValueError("sdf must have shape B,1,H,W")
    if body_zero_cells < 0.0 or body_ramp_cells <= 0.0:
        raise ValueError("body taper cells must be nonnegative/positive")
    body = (
        (sdf[:, 0] - body_zero_cells * dx) / (body_ramp_cells * dx)
    ).clamp(0.0, 1.0)
    body = body.square() * (3.0 - 2.0 * body)
    height, width = sdf.shape[-2:]
    y = torch.arange(height, device=sdf.device, dtype=sdf.dtype)
    x = torch.arange(width, device=sdf.device, dtype=sdf.dtype)
    edge_y = torch.minimum(y + 1.0, torch.tensor(float(height), device=sdf.device) - y)
    edge_x = torch.minimum(x + 1.0, torch.tensor(float(width), device=sdf.device) - x)
    if boundary_zero_cells < 0.0 or boundary_ramp_cells <= 0.0:
        raise ValueError("boundary taper cells must be nonnegative/positive")
    edge = torch.minimum(edge_y[:, None], edge_x[None, :]) - 1.0
    outer = (
        (edge - boundary_zero_cells) / boundary_ramp_cells
    ).clamp(0.0, 1.0)
    outer = outer.square() * (3.0 - 2.0 * outer)
    return body * outer


def reduced_pressure_projection(
    field: Tensor,
    sdf: Tensor,
    *,
    dx: float,
    strength: float = 1.0,
    core: int = 32,
    halo: int = 4,
    gradient_regularization: float = 1.0e-4,
    relative_cutoff: float = 1.0e-8,
    body_zero_cells: float = 2.0,
    body_ramp_cells: float = 4.0,
    boundary_zero_cells: float = 0.0,
    boundary_ramp_cells: float = 4.0,
) -> tuple[Tensor, Tensor, PressureProjectionDiagnostics]:
    """Project coarse divergence onto gradients of scalar tile-POU modes.

    Returns the corrected field, its two-channel velocity correction, and
    diagnostics.  The pressure channel is intentionally left unchanged in this
    first no-training experiment because the dataset's pressure scaling does
    not directly expose the fractional-step time normalization.
    """
    if not 0.0 <= strength <= 1.5:
        raise ValueError("strength must lie in [0,1.5]")
    batch, _, height, width = field.shape
    base = _tile_pressure_basis(
        height, width, core, halo, field.device, field.dtype,
    )
    taper = _physical_taper(
        sdf, dx, body_zero_cells=body_zero_cells,
        body_ramp_cells=body_ramp_cells,
        boundary_zero_cells=boundary_zero_cells,
        boundary_ramp_cells=boundary_ramp_cells,
    )
    pressure_modes = base[None] * taper[:, None]
    grad_x, grad_y = centered_gradient(pressure_modes, dx)
    velocity_modes = torch.stack((grad_x, grad_y), dim=2)
    mode_fields = torch.zeros(
        batch * base.shape[0], 2, height, width,
        device=field.device, dtype=field.dtype,
    )
    mode_fields[:, 0] = grad_x.reshape(-1, height, width)
    mode_fields[:, 1] = grad_y.reshape(-1, height, width)
    response = velocity_divergence(mode_fields, dx).reshape(
        batch, base.shape[0], height, width,
    )
    divergence = velocity_divergence(field, dx)
    weight = taper.square()

    work_response = response.to(torch.float64)
    work_velocity = velocity_modes.to(torch.float64)
    work_weight = weight.to(torch.float64)
    gram = torch.einsum(
        "bmhw,bnhw,bhw->bmn", work_response, work_response, work_weight,
    )
    if gradient_regularization > 0.0:
        gram = gram + gradient_regularization * torch.einsum(
            "bmchw,bnchw,bhw->bmn", work_velocity, work_velocity, work_weight,
        )
    gram = 0.5 * (gram + gram.transpose(-1, -2))
    rhs = torch.einsum(
        "bmhw,bhw,bhw->bm", work_response, divergence.to(torch.float64), work_weight,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = eigenvalues.amax(-1, keepdim=True).clamp_min(1.0e-30)
    retained = eigenvalues > relative_cutoff * maximum
    inverse = torch.where(retained, eigenvalues.reciprocal(), torch.zeros_like(eigenvalues))
    spectral_rhs = torch.einsum("bmi,bm->bi", eigenvectors, rhs)
    coefficients = torch.einsum(
        "bmi,bi->bm", eigenvectors, inverse * spectral_rhs,
    )
    correction = -strength * torch.einsum(
        "bm,bmchw->bchw", coefficients, work_velocity,
    ).to(field.dtype)
    corrected = field.clone()
    corrected[:, :2] += correction
    divergence_after = velocity_divergence(corrected, dx)
    normalizer = work_weight.flatten(1).sum(1).clamp_min(1.0)
    before_rms = (
        (divergence.square() * weight).flatten(1).sum(1) / normalizer.to(field.dtype)
    ).sqrt()
    after_rms = (
        (divergence_after.square() * weight).flatten(1).sum(1)
        / normalizer.to(field.dtype)
    ).sqrt()
    smallest = torch.where(
        retained, eigenvalues, torch.full_like(eigenvalues, torch.inf),
    ).amin(-1)
    return corrected, correction, PressureProjectionDiagnostics(
        divergence_before=before_rms,
        divergence_after=after_rms,
        rank=retained.sum(-1),
        condition=maximum[:, 0] / smallest.clamp_min(1.0e-30),
    )


def project_multiresolution_prediction(
    prediction: MultiresolutionPrediction,
    coarse_sdf: Tensor,
    *,
    coarse_dx: float,
    strength: float,
    core: int = 32,
    halo: int = 4,
    body_zero_cells: float = 2.0,
    body_ramp_cells: float = 4.0,
    boundary_zero_cells: float = 0.0,
    boundary_ramp_cells: float = 4.0,
) -> tuple[MultiresolutionPrediction, PressureProjectionDiagnostics]:
    coarse, correction, diagnostics = reduced_pressure_projection(
        prediction.coarse, coarse_sdf, dx=coarse_dx, strength=strength,
        core=core, halo=halo, body_zero_cells=body_zero_cells,
        body_ramp_cells=body_ramp_cells,
        boundary_zero_cells=boundary_zero_cells,
        boundary_ramp_cells=boundary_ramp_cells,
    )
    fine_correction = F.interpolate(
        correction, size=prediction.fine.shape[-2:], mode="bilinear", align_corners=False,
    )
    fine = prediction.fine.clone()
    fine[:, :2] += fine_correction
    coarse, fine = synchronize_levels(
        coarse, fine, prediction.refinement_mask, fine_trust=1.0,
    )
    return MultiresolutionPrediction(
        coarse=coarse,
        fine=fine,
        refinement_mask=prediction.refinement_mask,
        composite_fine_view=decode_composite(
            coarse, fine, prediction.refinement_mask,
        ),
        executed_fine_tiles=prediction.executed_fine_tiles,
    ), diagnostics
