"""Diagnostics for distinct Cartesian-cell roles in airfoil rollouts."""

from __future__ import annotations

import torch

from codiffuse_Schwarz_Airfoil.train_one_step_multilevel import interior_vorticity


Tensor = torch.Tensor


def cartesian_role_masks(
    sdf_n: Tensor,
    sdf_np1: Tensor,
    *,
    dx: float,
    boundary_cells: int = 32,
    geometry_band_cells: float = 8.0,
) -> dict[str, Tensor]:
    """Partition interior vorticity samples by geometry and domain role.

    The geometry role uses the swept source/destination SDF support. Boundary
    roles are one local-core width, matching the 32-cell Cartesian operator.
    Geometry takes precedence if a future geometry approaches an outer edge.
    """
    if sdf_n.shape != sdf_np1.shape or sdf_n.ndim != 4 or sdf_n.shape[1] != 1:
        raise ValueError("SDF fields must have matching B1HW shapes")
    if dx <= 0.0 or boundary_cells <= 0 or geometry_band_cells <= 0.0:
        raise ValueError("role widths and grid spacing must be positive")
    height, width = sdf_n.shape[-2:]
    if min(height, width) <= 2 or 2 * boundary_cells >= min(height - 2, width - 2):
        raise ValueError("boundary role consumes the complete interior grid")

    source = sdf_n[:, 0, 1:-1, 1:-1]
    destination = sdf_np1[:, 0, 1:-1, 1:-1]
    geometry = (
        torch.minimum(source.abs(), destination.abs())
        <= geometry_band_cells * dx
    ) | (source < 0.0) | (destination < 0.0)

    shape = (sdf_n.shape[0], height - 2, width - 2)
    left = torch.zeros(shape, dtype=torch.bool, device=sdf_n.device)
    right = torch.zeros_like(left)
    bottom = torch.zeros_like(left)
    top = torch.zeros_like(left)
    left[..., :boundary_cells] = True
    right[..., -boundary_cells:] = True
    bottom[:, :boundary_cells] = True
    top[:, -boundary_cells:] = True
    for mask in (left, right, bottom, top):
        mask &= ~geometry
    outer = left | right | bottom | top
    bulk = ~(geometry | outer)
    return {
        "geometry": geometry,
        "left_boundary": left,
        "right_boundary": right,
        "bottom_boundary": bottom,
        "top_boundary": top,
        "outer_boundary": outer,
        "bulk": bulk,
    }


def _masked_relative_l2(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    weight = mask.to(prediction.dtype)
    numerator = ((prediction - target).square() * weight).sum()
    denominator = (target.square() * weight).sum().clamp_min(1.0e-20)
    return torch.sqrt(numerator / denominator)


def role_vorticity_metrics(
    prediction: Tensor,
    target: Tensor,
    sdf_n: Tensor,
    sdf_np1: Tensor,
    *,
    dx: float,
    boundary_cells: int = 32,
    geometry_band_cells: float = 8.0,
) -> dict[str, Tensor]:
    """Measure moving-geometry, bulk, and wake-exit behavior separately."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have matching BCHW shapes")
    masks = cartesian_role_masks(
        sdf_n, sdf_np1, dx=dx, boundary_cells=boundary_cells,
        geometry_band_cells=geometry_band_cells,
    )
    predicted_vorticity = interior_vorticity(prediction, dx)
    target_vorticity = interior_vorticity(target, dx)
    result = {
        f"{role}_vorticity": _masked_relative_l2(
            predicted_vorticity, target_vorticity, masks[role],
        )
        for role in ("geometry", "bulk", "outer_boundary", "right_boundary")
    }

    right = masks["right_boundary"].to(predicted_vorticity.dtype)
    predicted_energy = (predicted_vorticity.square() * right).sum()
    target_energy = (target_vorticity.square() * right).sum().clamp_min(1.0e-20)
    result["right_boundary_energy_ratio"] = predicted_energy / target_energy

    # Flux distribution at the final represented interior face. Comparing the
    # profile avoids cancellation between positive and negative wake vortices.
    predicted_flux = (
        prediction[:, 0, 1:-1, -2] * predicted_vorticity[..., -1]
    )
    target_flux = target[:, 0, 1:-1, -2] * target_vorticity[..., -1]
    result["right_exit_flux"] = torch.linalg.vector_norm(
        predicted_flux - target_flux
    ) / torch.linalg.vector_norm(target_flux).clamp_min(1.0e-20)
    return result
