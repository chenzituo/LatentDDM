"""Shared one-message compatibility composer for 2D and 3D local operators.

The module implements the paper's exact causal contract after a frozen local
endpoint.  It pools volume and face moments from every halo field, evaluates
one globally receptive relative-layout message, predicts one compatibility
code per tile, and decodes that code through locally generated response modes.

The response coefficients are centered at the zero code.  Consequently the
zero-code ablation is exactly the original local endpoint, independent of
decoder biases.  This makes local/oracle/predicted/shuffled comparisons causal
and checkpoint-consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .learned_composer import canonical_modes
from .tile_communication import RelativeTileAttention


Tensor = torch.Tensor


def _group_count(width: int) -> int:
    groups = min(16, width)
    while width % groups:
        groups -= 1
    return groups


def _masked_moments(field: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    axes = tuple(range(2, field.ndim))
    denominator = valid.sum(axes).clamp_min(1.0)
    mean = (field * valid).sum(axes) / denominator
    rms = (field.square() * valid).sum(axes).div(denominator).sqrt()
    return mean, rms


def _face_moments(field: Tensor, valid: Tensor) -> Tensor:
    """Return mean/RMS channels on both faces of every spatial axis."""
    outputs: list[Tensor] = []
    dimension = field.ndim - 2
    for spatial_axis in range(dimension):
        axis = spatial_axis + 2
        for index in (0, -1):
            face = field.select(axis, index)
            face_valid = valid.select(axis, index)
            reduction = tuple(range(2, face.ndim))
            denominator = face_valid.sum(reduction).clamp_min(1.0)
            mean = (face * face_valid).sum(reduction) / denominator
            rms = (face.square() * face_valid).sum(reduction).div(denominator).sqrt()
            outputs.extend((mean, rms))
    return torch.cat(outputs, dim=1)


def _normalize_modes(modes: Tensor, valid: Tensor) -> Tensor:
    axes = tuple(range(3, modes.ndim))
    denominator = valid[:, :, None].sum(axes).clamp_min(1.0)
    # Clamp before sqrt.  Clamping only after sqrt leaves an infinite
    # derivative at an exactly zero, zero-initialized response mode and can
    # produce 0*inf=NaN in its first backward pass.
    mean_square = (modes.square() * valid[:, :, None]).sum(axes).div(denominator)
    rms = mean_square.clamp_min(1.0e-12).sqrt()
    view = (*rms.shape, *((1,) * (modes.ndim - 3)))
    return modes / rms.clamp_min(1.0e-4).reshape(view)


@dataclass
class CompatibilityOutput:
    corrected: Tensor
    correction: Tensor
    predicted_code: Tensor
    used_code: Tensor
    coefficients: Tensor
    local_token: Tensor


class CompatibilityResponseComposer(nn.Module):
    """One relative global message followed by local response-mode decoding.

    ``local``, ``physical``, ``scalar``, and ``valid`` use a flattened
    batch/tile dimension.  The caller remains responsible for normalized POU
    assembly, which keeps geometry realization independent of this learned
    compatibility map.
    """

    def __init__(
        self,
        field_channels: int,
        physical_channels: int,
        scalar_dim: int,
        dimension: int,
        *,
        code_width: int = 16,
        token_width: int = 128,
        response_width: int = 64,
        learned_modes: int = 12,
        heads: int = 4,
        include_fixed_modes: bool = True,
        maximum_coefficient: float = 2.0,
    ) -> None:
        super().__init__()
        if field_channels <= 0 or physical_channels < 0 or scalar_dim < 0:
            raise ValueError("invalid channel dimensions")
        if dimension not in (2, 3):
            raise ValueError("dimension must be two or three")
        if min(code_width, token_width, response_width, learned_modes) <= 0:
            raise ValueError("composer widths and learned_modes must be positive")
        if token_width % heads:
            raise ValueError("token_width must be divisible by heads")
        self.field_channels = field_channels
        self.physical_channels = physical_channels
        self.scalar_dim = scalar_dim
        self.dimension = dimension
        self.code_width = code_width
        self.learned_modes = learned_modes
        self.include_fixed_modes = include_fixed_modes
        self.maximum_coefficient = float(maximum_coefficient)

        summarized_channels = field_channels + physical_channels
        # Two volume moments and two moments on each of 2*d faces.
        token_channels = (2 + 4 * dimension) * summarized_channels + scalar_dim
        self.token_encoder = nn.Sequential(
            nn.LayerNorm(token_channels),
            nn.Linear(token_channels, token_width),
            nn.SiLU(),
            nn.Linear(token_width, token_width),
        )
        self.attention = RelativeTileAttention(token_width, heads, dimension)
        self.code_decoder = nn.Sequential(
            nn.LayerNorm(3 * token_width),
            nn.Linear(3 * token_width, 2 * token_width),
            nn.SiLU(),
            nn.Linear(2 * token_width, code_width),
        )

        convolution = nn.Conv2d if dimension == 2 else nn.Conv3d
        dense_channels = field_channels + physical_channels + scalar_dim + 1
        self.response_stem = nn.Sequential(
            convolution(dense_channels, response_width, 3, padding=1),
            nn.GroupNorm(_group_count(response_width), response_width),
            nn.SiLU(),
            convolution(response_width, response_width, 3, padding=1),
            nn.GroupNorm(_group_count(response_width), response_width),
            nn.SiLU(),
        )
        self.response_head = convolution(
            response_width, field_channels * learned_modes, 1,
        )
        fixed_count = dimension + 1 if include_fixed_modes else 0
        self.mode_count = fixed_count + learned_modes
        self.coefficient_decoder = nn.Sequential(
            nn.LayerNorm(code_width),
            nn.Linear(code_width, 2 * code_width),
            nn.SiLU(),
            nn.Linear(2 * code_width, field_channels * self.mode_count),
        )
        nn.init.zeros_(self.coefficient_decoder[-1].weight)
        nn.init.zeros_(self.coefficient_decoder[-1].bias)

    def _tokens(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        batch: int,
        tile_count: int,
    ) -> Tensor:
        summarized = torch.cat((local, physical), dim=1)
        mean, rms = _masked_moments(summarized, valid)
        raw = torch.cat(
            (mean, rms, _face_moments(summarized, valid), scalar), dim=1,
        )
        return self.token_encoder(raw).reshape(batch, tile_count, -1)

    def _response_modes(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
    ) -> Tensor:
        spatial = local.shape[2:]
        broadcast_shape = (local.shape[0], self.scalar_dim, *((1,) * self.dimension))
        scalar_field = scalar.reshape(broadcast_shape).expand(
            local.shape[0], self.scalar_dim, *spatial,
        )
        dense = torch.cat((local, physical, scalar_field, valid), dim=1)
        learned = self.response_head(self.response_stem(dense)).reshape(
            local.shape[0], self.field_channels, self.learned_modes, *spatial,
        )
        learned = _normalize_modes(learned * valid[:, :, None], valid)
        if not self.include_fixed_modes:
            return learned
        fixed = canonical_modes(spatial, local.device, local.dtype)
        fixed = fixed[None, None].expand(
            local.shape[0], self.field_channels, -1, *spatial,
        )
        fixed = _normalize_modes(fixed * valid[:, :, None], valid)
        return torch.cat((fixed, learned), dim=2)

    def forward(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, ...],
        *,
        code_override: Tensor | None = None,
        max_graph_radius: int | None = None,
    ) -> CompatibilityOutput:
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

        local_token = self._tokens(
            local, physical, scalar, valid, batch, tile_count,
        )
        global_token = self.attention(
            local_token, grid_shape, max_graph_radius=max_graph_radius,
        )
        case_token = local_token.mean(1, keepdim=True).expand(-1, tile_count, -1)
        predicted_code = self.code_decoder(
            torch.cat((local_token, global_token, case_token), dim=-1),
        )
        if code_override is None:
            used_code = predicted_code
        else:
            if code_override.shape != predicted_code.shape:
                raise ValueError(
                    "code_override must have shape batch,tile_count,code_width",
                )
            used_code = code_override

        flat_code = used_code.reshape(batch * tile_count, self.code_width)
        zero_code = torch.zeros_like(flat_code)
        raw_coefficient = (
            self.coefficient_decoder(flat_code)
            - self.coefficient_decoder(zero_code)
        )
        coefficient = self.maximum_coefficient * torch.tanh(raw_coefficient)
        coefficient = coefficient.reshape(
            batch * tile_count, self.field_channels, self.mode_count,
        )
        modes = self._response_modes(local, physical, scalar, valid)
        correction = torch.einsum("bcm,bcm...->bc...", coefficient, modes) * valid
        return CompatibilityOutput(
            corrected=(local + correction) * valid,
            correction=correction,
            predicted_code=predicted_code,
            used_code=used_code,
            coefficients=coefficient,
            local_token=local_token,
        )
