"""Operator-only one-step composition of overlapping local predictions.

The composer is intentionally independent of a PDE discretization.  It sees
only local endpoint predictions, local conditioning fields, case controls, and
the relative layout of the cover.  One pooled token per unity participates in
a single global message.  The decoded correction is restricted to smooth
constant/linear modes inside each unity and is assembled by a deterministic
partition of unity by the caller.

This is the shared communication contract used by the Darcy, airfoil, and car
experiments.  In particular it never forms a residual, stiffness matrix, or
coarse Galerkin operator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tile_communication import RelativeTileAttention


Tensor = torch.Tensor


def _masked_moments(field: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Return per-channel mean and RMS for local 2D/3D fields."""
    axes = tuple(range(2, field.ndim))
    denominator = valid.sum(axes).clamp_min(1.0)
    mean = (field * valid).sum(axes) / denominator
    rms = (field.square() * valid).sum(axes).div(denominator).sqrt()
    return mean, rms


def canonical_modes(
    spatial_shape: Sequence[int], device: torch.device, dtype: torch.dtype,
) -> Tensor:
    """Constant and axis-linear modes on a canonical local unity."""
    axes = [torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
            for size in spatial_shape]
    coordinate = torch.meshgrid(*axes, indexing="ij")
    return torch.stack((torch.ones_like(coordinate[0]), *coordinate), dim=0)


@dataclass
class ComposerOutput:
    corrected: Tensor
    correction: Tensor
    coefficients: Tensor


class LearnedModeComposer(nn.Module):
    """One global token message followed by low-order local mode decoding.

    ``local`` and ``physical`` use a flattened batch/tile dimension.  The
    module preserves that convention so it can be inserted between any local
    neural endpoint and a 2D or 3D partition-of-unity assembler.
    """

    def __init__(
        self,
        field_channels: int,
        physical_channels: int,
        scalar_dim: int,
        dimension: int,
        *,
        width: int = 128,
        heads: int = 4,
        maximum_correction: float = 2.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if field_channels <= 0 or physical_channels < 0 or scalar_dim < 0:
            raise ValueError("invalid composer channel dimensions")
        if dimension not in (2, 3) or width <= 0 or width % heads:
            raise ValueError("dimension must be 2/3 and width divisible by heads")
        self.field_channels = field_channels
        self.physical_channels = physical_channels
        self.scalar_dim = scalar_dim
        self.dimension = dimension
        self.width = width
        self.maximum_correction = float(maximum_correction)

        token_channels = 2 * (field_channels + physical_channels) + scalar_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(token_channels),
            nn.Linear(token_channels, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        convolution = nn.Conv2d if dimension == 2 else nn.Conv3d
        self.neighbour_mixer = convolution(width, width, 3, padding=1)
        self.attention = RelativeTileAttention(width, heads, dimension)
        self.decoder = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, field_channels * (dimension + 1)),
        )
        if zero_init:
            nn.init.zeros_(self.decoder[-1].weight)
            nn.init.zeros_(self.decoder[-1].bias)

    def forward(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, ...],
    ) -> ComposerOutput:
        expected_ndim = self.dimension + 2
        if local.ndim != expected_ndim or physical.ndim != expected_ndim:
            raise ValueError("local and physical fields have incompatible rank")
        if local.shape[0] != physical.shape[0] or local.shape[2:] != physical.shape[2:]:
            raise ValueError("local and physical tile shapes must match")
        if local.shape[1] != self.field_channels:
            raise ValueError("local field channel count does not match composer")
        if physical.shape[1] != self.physical_channels:
            raise ValueError("physical channel count does not match composer")
        if valid.shape != (local.shape[0], 1, *local.shape[2:]):
            raise ValueError("validity must be a one-channel local field")
        if scalar.shape != (local.shape[0], self.scalar_dim):
            raise ValueError("scalar controls do not match composer")
        if len(grid_shape) != self.dimension:
            raise ValueError("grid shape dimension does not match composer")
        tile_count = math.prod(grid_shape)
        if local.shape[0] % tile_count:
            raise ValueError("flattened local batch is incompatible with grid shape")
        batch = local.shape[0] // tile_count

        local_mean, local_rms = _masked_moments(local, valid)
        physical_mean, physical_rms = _masked_moments(physical, valid)
        raw_token = torch.cat(
            (local_mean, local_rms, physical_mean, physical_rms, scalar), dim=1,
        )
        token = self.encoder(raw_token).reshape(batch, tile_count, self.width)
        global_token = self.attention(token, grid_shape)
        grid = token.transpose(1, 2).reshape(batch, self.width, *grid_shape)
        neighbour = self.neighbour_mixer(F.silu(grid))
        neighbour = neighbour.reshape(batch, self.width, tile_count).transpose(1, 2)
        case = token.mean(1, keepdim=True).expand(-1, tile_count, -1)
        coefficient = self.maximum_correction * torch.tanh(
            self.decoder(torch.cat((global_token, neighbour, case), dim=-1))
        )
        coefficient = coefficient.reshape(
            batch * tile_count, self.field_channels, self.dimension + 1,
        )
        modes = canonical_modes(local.shape[2:], local.device, local.dtype)
        correction = torch.einsum("bcm,m...->bc...", coefficient, modes) * valid
        return ComposerOutput(
            corrected=(local + correction) * valid,
            correction=correction,
            coefficients=coefficient,
        )

