"""Physics-defined one-pass coarse corrections for masked Darcy fields.

The local neural operator supplies a preliminary POU endpoint.  This module
constructs tile-supported coarse modes and computes the exact Galerkin energy
projection in their span.  It deliberately contains no learned global mixing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import enforce_domain_boundary, raw_darcy_residual


Tensor = torch.Tensor


@dataclass(frozen=True)
class GalerkinDiagnostics:
    rank: Tensor
    condition: Tensor
    correction_energy: Tensor


def tile_pou_basis(
    cover: CartesianCover,
    *,
    degree: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return tile-supported constant/linear POU modes as ``[M,H,W]``.

    ``degree=0`` gives one mode per tile.  ``degree=1`` additionally multiplies
    each tile POU function by its canonical local x and y coordinates.  The
    modes depend on the cover, not on absolute world-coordinate prompts.
    """
    if degree not in (0, 1):
        raise ValueError("degree must be 0 or 1")
    tiles = cover.tile_count
    size = cover.tile_size
    per_tile = 1 if degree == 0 else 3
    modes = tiles * per_tile
    coordinate = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    local_modes = [torch.ones_like(xx)]
    if degree == 1:
        local_modes.extend((xx, yy))
    patches = torch.zeros(
        tiles, modes, size, size, device=device, dtype=dtype,
    )
    for tile in range(tiles):
        for local_index, mode in enumerate(local_modes):
            patches[tile, tile * per_tile + local_index] = mode
    return cover.assemble_weighted(patches, 1)[0]


def apply_homogeneous_operator(
    fields: Tensor,
    coefficient: Tensor,
    mask: Tensor,
) -> Tensor:
    """Apply the exact homogeneous Darcy operator to ``[B,M,H,W]`` fields."""
    if fields.ndim != 4:
        raise ValueError("fields must have shape B,M,H,W")
    batch, modes, height, width = fields.shape
    repeated_coefficient = coefficient[:, None].expand(
        batch, modes, 1, height, width,
    ).reshape(batch * modes, 1, height, width)
    repeated_mask = mask[:, None].expand(
        batch, modes, 1, height, width,
    ).reshape(batch * modes, 1, height, width)
    flattened = fields.reshape(batch * modes, 1, height, width)
    zero = torch.zeros_like(flattened)
    applied = raw_darcy_residual(
        flattened, repeated_coefficient, zero, repeated_mask, zero,
    )
    return applied.reshape(batch, modes, height, width)


def galerkin_correct(
    preliminary: Tensor,
    batch: dict[str, Tensor],
    basis: Tensor,
    *,
    relative_cutoff: float = 1.0e-8,
) -> tuple[Tensor, GalerkinDiagnostics]:
    """Energy-project the exact residual onto the supplied coarse basis."""
    if preliminary.ndim != 4 or preliminary.shape[1] != 1:
        raise ValueError("preliminary must be a scalar BCHW field")
    if basis.ndim != 3 or tuple(basis.shape[-2:]) != tuple(preliminary.shape[-2:]):
        raise ValueError("basis must have shape M,H,W on the preliminary grid")
    coefficient = torch.pow(10.0, batch["logk"].clamp(-6.0, 6.0))
    mask = batch["mask"].to(preliminary.dtype)
    active_basis = basis[None].expand(preliminary.shape[0], -1, -1, -1) * mask
    residual = -raw_darcy_residual(
        preliminary, coefficient, batch["source"], mask, batch["bc_value"],
    )[:, 0]

    work_basis = active_basis.to(torch.float64)
    applied = apply_homogeneous_operator(
        active_basis, coefficient, mask,
    ).to(torch.float64)
    gram = torch.einsum("bmhw,bnhw->bmn", work_basis, applied)
    gram = 0.5 * (gram + gram.transpose(-1, -2))
    reduced_rhs = torch.einsum("bmhw,bhw->bm", work_basis, residual.to(torch.float64))

    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = eigenvalues.amax(dim=-1, keepdim=True).clamp_min(1.0e-30)
    retained = eigenvalues > relative_cutoff * maximum
    inverse = torch.where(retained, eigenvalues.reciprocal(), torch.zeros_like(eigenvalues))
    projected_rhs = torch.einsum("bmi,bm->bi", eigenvectors, reduced_rhs)
    coefficients = torch.einsum(
        "bmi,bi->bm", eigenvectors, inverse * projected_rhs,
    )
    correction = torch.einsum("bm,bmhw->bhw", coefficients, work_basis)
    corrected = enforce_domain_boundary(
        batch, preliminary + correction[:, None].to(preliminary.dtype),
    )

    smallest = torch.where(
        retained, eigenvalues, torch.full_like(eigenvalues, torch.inf),
    ).amin(dim=-1)
    condition = maximum[:, 0] / smallest.clamp_min(1.0e-30)
    correction_energy = torch.einsum("bm,bm->b", coefficients, reduced_rhs)
    return corrected, GalerkinDiagnostics(
        rank=retained.sum(dim=-1),
        condition=condition,
        correction_energy=correction_energy,
    )

