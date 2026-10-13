"""Overlap-based pressure-level alignment for Cartesian neural DDMs.

In incompressible subdomain solves, pressure is determined only up to one
constant per subdomain.  Local velocity convergence therefore does not imply
that independently predicted pressure levels agree.  This module removes that
small coarse space without changing the spatial pressure variation produced by
any local neural operator.

For every neighboring pair, the overlap supplies a least-squares equation for
the difference of its two pressure constants.  The resulting weighted graph
Laplacian has one global null mode.  We fix that mode by preserving a supplied
global pressure mean (or, by default, the mean before correction).

This is deliberately named *alignment*, not the pressure-recovery theorem of
Bui--Japhet--Omnes (2024).  Their coarse right-hand side is constructed from
the Robin data and local pressure means after a convergent velocity iteration.
The routine below instead fits pressure differences directly on overlapping
predicted cells.  It is a useful diagnostic/assembly heuristic, but it does
not inherit the convergence guarantee of their equation (38) and must not be
used to claim that a learned Schwarz velocity map converges.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class PressureCoarseCorrection2D:
    patches: Tensor
    offsets: Tensor
    reference_mean: Tensor
    overlap_rms_before: Tensor
    overlap_rms_after: Tensor


def _global_pressure_mean(
    patches: Tensor, valid_tiles: Tensor, cover, batch: int,
) -> Tensor:
    field = cover.assemble_weighted(patches, batch)
    valid = cover.assemble_average(valid_tiles, batch).clamp(0.0, 1.0)
    numerator = (field[:, 2:3] * valid).sum(dim=(-1, -2, -3))
    denominator = valid.sum(dim=(-1, -2, -3)).clamp_min(1.0)
    return numerator / denominator


def overlap_pressure_constant_correction_2d(
    patches: Tensor,
    cover,
    batch: int,
    *,
    valid_tiles: Tensor | None = None,
    reference_mean: Tensor | None = None,
) -> PressureCoarseCorrection2D:
    """Align one pressure constant per overlapping tile.

    Parameters
    ----------
    patches:
        Local states with shape ``[B*M,C,S,S]`` and pressure in channel two.
    cover:
        The :class:`CartesianCover` that produced the patches.
    batch:
        Number of global examples ``B``.
    valid_tiles:
        Optional local validity weights with shape ``[B*M,1,S,S]``.
    reference_mean:
        Desired global pressure mean, one value per global example.  When it
        is omitted, the pre-correction assembled mean is preserved.

    Notes
    -----
    This overlap estimator changes only tile-wise constants, so pressure
    gradients and both velocity channels are invariant.  It is not the
    Robin-data pressure recovery of Bui, Japhet, and Omnes (2024).
    """
    ny, nx = cover.grid_shape
    tiles = ny * nx
    size = cover.core + 2 * cover.halo
    if patches.ndim != 4 or patches.shape[0] != batch * tiles \
            or patches.shape[1] < 3 or patches.shape[-2:] != (size, size):
        raise ValueError("patches are incompatible with the Cartesian cover")
    if valid_tiles is None:
        valid_tiles = patches.new_ones(batch * tiles, 1, size, size)
    if valid_tiles.shape != (batch * tiles, 1, size, size):
        raise ValueError("valid tiles must have shape B*M,1,S,S")

    raw_reference = _global_pressure_mean(
        patches, valid_tiles, cover, batch,
    )
    if reference_mean is None:
        reference = raw_reference
    else:
        reference = torch.as_tensor(
            reference_mean, device=patches.device, dtype=patches.dtype,
        ).flatten()
        if reference.numel() == 1:
            reference = reference.expand(batch)
        if reference.shape != (batch,):
            raise ValueError("reference pressure mean must be scalar or length B")

    # With no overlap there is no relative coarse mode to infer.  Retain only
    # the globally anchored constant correction.
    if cover.halo == 0 or tiles == 1:
        common = reference - raw_reference
        additive = patches.new_zeros(batch, tiles, patches.shape[1], 1, 1)
        additive[:, :, 2] = common[:, None, None, None]
        corrected = patches + additive.reshape(
            batch * tiles, patches.shape[1], 1, 1,
        )
        zero = patches.new_zeros(batch)
        return PressureCoarseCorrection2D(
            corrected, common[:, None].expand(-1, tiles), reference,
            zero, zero,
        )

    pressure = patches[:, 2].reshape(batch, ny, nx, size, size)
    validity = valid_tiles[:, 0].reshape(batch, ny, nx, size, size)
    overlap = 2 * cover.halo
    incidence_rows: list[Tensor] = []
    differences: list[Tensor] = []
    weights: list[Tensor] = []
    squared_before: list[Tensor] = []
    counts: list[Tensor] = []

    def add_edge(
        first_index: int, second_index: int,
        first: Tensor, second: Tensor,
        first_valid: Tensor, second_valid: Tensor,
    ) -> None:
        mask = first_valid * second_valid
        count = mask.sum(dim=(-1, -2)).clamp_min(1.0)
        difference = ((second - first) * mask).sum(
            dim=(-1, -2),
        ) / count
        row = patches.new_zeros(tiles)
        row[first_index] = 1.0
        row[second_index] = -1.0
        incidence_rows.append(row)
        differences.append(difference)
        counts.append(count)
        weights.append(count / float(first.shape[-1] * first.shape[-2]))
        squared_before.append(
            ((first - second).square() * mask).sum(dim=(-1, -2))
        )

    for iy in range(ny):
        for ix in range(nx - 1):
            first_index = iy * nx + ix
            second_index = first_index + 1
            add_edge(
                first_index, second_index,
                pressure[:, iy, ix, :, cover.core:size],
                pressure[:, iy, ix + 1, :, :overlap],
                validity[:, iy, ix, :, cover.core:size],
                validity[:, iy, ix + 1, :, :overlap],
            )
    for iy in range(ny - 1):
        for ix in range(nx):
            first_index = iy * nx + ix
            second_index = first_index + nx
            add_edge(
                first_index, second_index,
                pressure[:, iy, ix, cover.core:size, :],
                pressure[:, iy + 1, ix, :overlap, :],
                validity[:, iy, ix, cover.core:size, :],
                validity[:, iy + 1, ix, :overlap, :],
            )

    incidence = torch.stack(incidence_rows).to(torch.float32)
    edge_difference = torch.stack(differences, dim=1).to(torch.float32)
    edge_weight = torch.stack(weights, dim=1).to(torch.float32)
    # A c = d encodes p_i+c_i = p_j+c_j on every overlap.  The rank-one
    # term fixes sum(c)=0 while retaining a symmetric positive definite solve.
    laplacian = torch.einsum(
        "em,be,en->bmn", incidence, edge_weight, incidence,
    )
    rhs = torch.einsum(
        "em,be,be->bm", incidence, edge_weight, edge_difference,
    )
    gauge = torch.ones(
        tiles, tiles, device=patches.device, dtype=torch.float32,
    ) / float(tiles)
    # CUDA autocast may downcast the einsum RHS even though its operands were
    # promoted explicitly.  Dense solves require matching float32 operands.
    with torch.autocast(device_type=patches.device.type, enabled=False):
        offsets = torch.linalg.solve(
            (laplacian + gauge).float(), rhs.float()[..., None],
        )[..., 0]
    offsets = offsets.to(patches.dtype)

    additive = patches.new_zeros(batch, tiles, patches.shape[1], 1, 1)
    additive[:, :, 2] = offsets[:, :, None, None]
    corrected = patches + additive.reshape(
        batch * tiles, patches.shape[1], 1, 1,
    )
    corrected_mean = _global_pressure_mean(
        corrected, valid_tiles, cover, batch,
    )
    common = reference - corrected_mean
    additive[:, :, 2] = common[:, None, None, None]
    corrected = corrected + additive.reshape(
        batch * tiles, patches.shape[1], 1, 1,
    )
    offsets = offsets + common[:, None]

    before_sum = torch.stack(squared_before, dim=1).sum(1)
    count_sum = torch.stack(counts, dim=1).sum(1).clamp_min(1.0)
    corrected_pressure = corrected[:, 2].reshape(batch, ny, nx, size, size)
    after_sum: list[Tensor] = []
    for iy in range(ny):
        for ix in range(nx - 1):
            mask = (
                validity[:, iy, ix, :, cover.core:size]
                * validity[:, iy, ix + 1, :, :overlap]
            )
            residual = (
                corrected_pressure[:, iy, ix, :, cover.core:size]
                - corrected_pressure[:, iy, ix + 1, :, :overlap]
            )
            after_sum.append((residual.square() * mask).sum(dim=(-1, -2)))
    for iy in range(ny - 1):
        for ix in range(nx):
            mask = (
                validity[:, iy, ix, cover.core:size, :]
                * validity[:, iy + 1, ix, :overlap, :]
            )
            residual = (
                corrected_pressure[:, iy, ix, cover.core:size, :]
                - corrected_pressure[:, iy + 1, ix, :overlap, :]
            )
            after_sum.append((residual.square() * mask).sum(dim=(-1, -2)))
    overlap_before = (before_sum / count_sum).clamp_min(0.0).sqrt()
    overlap_after = (
        torch.stack(after_sum, dim=1).sum(1) / count_sum
    ).clamp_min(0.0).sqrt()
    return PressureCoarseCorrection2D(
        corrected, offsets, reference, overlap_before, overlap_after,
    )
