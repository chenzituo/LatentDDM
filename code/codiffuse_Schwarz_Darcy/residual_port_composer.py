"""Phase-separated residual-port chart and one-message communicator."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from codiffuse_Schwarz.learned_composer import canonical_modes
from codiffuse_Schwarz.tile_communication import (
    RelativeTileAttention, ResidualKernelCodeMap,
    StateGatedResidualKernelCodeMap,
)
from codiffuse_Schwarz.signed_tile_communication import SignedResidualKernelCodeMap


Tensor = torch.Tensor


def _groups(width: int) -> int:
    count = min(16, width)
    while width % count:
        count -= 1
    return count


def _moments(field: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    axes = tuple(range(2, field.ndim))
    denominator = valid.sum(axes).clamp_min(1.0)
    mean = (field * valid).sum(axes) / denominator
    rms = ((field.square() * valid).sum(axes) / denominator).sqrt()
    return mean, rms


def _face_moments(field: Tensor, valid: Tensor) -> Tensor:
    values = []
    for spatial_axis in range(field.ndim - 2):
        axis = spatial_axis + 2
        for index in (0, -1):
            face = field.select(axis, index)
            weight = valid.select(axis, index)
            reduction = tuple(range(2, face.ndim))
            denominator = weight.sum(reduction).clamp_min(1.0)
            values.extend((
                (face * weight).sum(reduction) / denominator,
                ((face.square() * weight).sum(reduction) / denominator).sqrt(),
            ))
    return torch.cat(values, dim=1)


def _normalize_modes(modes: Tensor, valid: Tensor) -> Tensor:
    axes = tuple(range(3, modes.ndim))
    denominator = valid[:, :, None].sum(axes).clamp_min(1.0)
    # Clamp the mean square *before* sqrt.  Clamping only after sqrt leaves an
    # infinite sqrt derivative at an exactly-zero learned mode; a zero chart
    # coefficient then produces the indeterminate backward product 0*inf=NaN.
    mean_square = (modes.square() * valid[:, :, None]).sum(axes) / denominator
    rms = mean_square.clamp_min(1.0e-8).sqrt()
    shape = (*rms.shape, *((1,) * (modes.ndim - 3)))
    return modes / rms.reshape(shape)


@dataclass(frozen=True)
class ChartOutput:
    corrected: Tensor
    correction: Tensor
    coefficients: Tensor


class ResidualResponseChart(nn.Module):
    """Local response chart trained only against fixed oracle port codes."""

    def __init__(
        self,
        field_channels: int,
        physical_channels: int,
        scalar_dim: int,
        code_width: int,
        *,
        response_width: int = 64,
        learned_modes: int = 12,
        maximum_coefficient: float = 1.5,
    ) -> None:
        super().__init__()
        self.field_channels = field_channels
        self.physical_channels = physical_channels
        self.scalar_dim = scalar_dim
        self.code_width = code_width
        self.chart_type = "nonlinear_12mode"
        self.response_width = response_width
        self.learned_modes = learned_modes
        self.mode_count = 3 + learned_modes
        self.maximum_coefficient = float(maximum_coefficient)
        dense_channels = field_channels + physical_channels + scalar_dim + 1
        self.response_stem = nn.Sequential(
            nn.Conv2d(dense_channels, response_width, 3, padding=1),
            nn.GroupNorm(_groups(response_width), response_width), nn.SiLU(),
            nn.Conv2d(response_width, response_width, 3, padding=1),
            nn.GroupNorm(_groups(response_width), response_width), nn.SiLU(),
        )
        self.response_head = nn.Conv2d(
            response_width, field_channels * learned_modes, 1,
        )
        self.coefficient_decoder = nn.Sequential(
            nn.LayerNorm(code_width),
            nn.Linear(code_width, 2 * code_width), nn.SiLU(),
            nn.Linear(2 * code_width, field_channels * self.mode_count),
        )
        nn.init.zeros_(self.coefficient_decoder[-1].weight)
        nn.init.zeros_(self.coefficient_decoder[-1].bias)

    def _modes(
        self, local: Tensor, physical: Tensor, scalar: Tensor, valid: Tensor,
    ) -> Tensor:
        scalar_field = scalar[:, :, None, None].expand(
            local.shape[0], self.scalar_dim, *local.shape[-2:],
        )
        dense = torch.cat((local, physical, scalar_field, valid), dim=1)
        learned = self.response_head(self.response_stem(dense)).reshape(
            local.shape[0], self.field_channels, self.learned_modes,
            *local.shape[-2:],
        )
        learned = _normalize_modes(learned * valid[:, :, None], valid)
        fixed = canonical_modes(local.shape[-2:], local.device, local.dtype)
        fixed = fixed[None, None].expand(
            local.shape[0], self.field_channels, -1, *local.shape[-2:],
        )
        fixed = _normalize_modes(fixed * valid[:, :, None], valid)
        return torch.cat((fixed, learned), dim=2)

    def forward(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        scaled_code: Tensor,
    ) -> ChartOutput:
        if scaled_code.shape != (local.shape[0], self.code_width):
            raise ValueError("chart code must have shape flattened_tiles,code_width")
        zero = torch.zeros_like(scaled_code)
        raw = self.coefficient_decoder(scaled_code) \
            - self.coefficient_decoder(zero)
        coefficients = self.maximum_coefficient * torch.tanh(raw)
        coefficients = coefficients.reshape(
            local.shape[0], self.field_channels, self.mode_count,
        )
        modes = self._modes(local, physical, scalar, valid)
        correction = torch.einsum(
            "bcm,bcmhw->bchw", coefficients, modes,
        ) * valid
        return ChartOutput(
            corrected=(local + correction) * valid,
            correction=correction,
            coefficients=coefficients,
        )


class LinearResidualResponseChart(nn.Module):
    r"""Tied Green-response chart ``L_X(z)=sum_k z_k psi_k(X)``.

    There is exactly one geometry-conditioned response field ``psi_k`` for
    every residual-port coordinate.  In particular there is no learned map
    from code to coefficients, no saturation, and no mixing between port
    coordinates.  Thus the chart is linear in ``z`` for each fixed local
    coefficient/geometry state ``X`` and the zero port is exactly the frozen
    local endpoint.
    """

    def __init__(
        self,
        field_channels: int,
        physical_channels: int,
        scalar_dim: int,
        code_width: int,
        *,
        response_width: int = 64,
    ) -> None:
        super().__init__()
        self.field_channels = field_channels
        self.physical_channels = physical_channels
        self.scalar_dim = scalar_dim
        self.code_width = code_width
        self.chart_type = "tied_linear_green"
        self.response_width = response_width
        dense_channels = field_channels + physical_channels + scalar_dim + 1
        self.response_stem = nn.Sequential(
            nn.Conv2d(dense_channels, response_width, 3, padding=1),
            nn.GroupNorm(_groups(response_width), response_width), nn.SiLU(),
            nn.Conv2d(response_width, response_width, 3, padding=1),
            nn.GroupNorm(_groups(response_width), response_width), nn.SiLU(),
        )
        self.response_head = nn.Conv2d(
            response_width, field_channels * code_width, 1,
        )
        # Start at the frozen endpoint while retaining a nonzero gradient for
        # each response field under nonzero oracle port excitation.
        nn.init.zeros_(self.response_head.weight)
        nn.init.zeros_(self.response_head.bias)

    def _response_modes(
        self, local: Tensor, physical: Tensor, scalar: Tensor, valid: Tensor,
    ) -> Tensor:
        scalar_field = scalar[:, :, None, None].expand(
            local.shape[0], self.scalar_dim, *local.shape[-2:],
        )
        dense = torch.cat((local, physical, scalar_field, valid), dim=1)
        return self.response_head(self.response_stem(dense)).reshape(
            local.shape[0], self.field_channels, self.code_width,
            *local.shape[-2:],
        ) * valid[:, :, None]

    def forward(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        scaled_code: Tensor,
    ) -> ChartOutput:
        if scaled_code.shape != (local.shape[0], self.code_width):
            raise ValueError("chart code must have shape flattened_tiles,code_width")
        modes = self._response_modes(local, physical, scalar, valid)
        correction = torch.einsum(
            "bk,bckhw->bchw", scaled_code, modes,
        ) * valid
        coefficients = scaled_code[:, None, :].expand(
            -1, self.field_channels, -1,
        )
        return ChartOutput(
            corrected=(local + correction) * valid,
            correction=correction,
            coefficients=coefficients,
        )


class GeometryLinearResidualResponseChart(nn.Module):
    r"""Tied linear chart with modes conditioned only on geometry and kappa."""

    GEOMETRY_CHANNELS = SourceLinearResidualPortCommunicator.GEOMETRY_CHANNELS \
        if "SourceLinearResidualPortCommunicator" in globals() else (
            1, 3, 4, 6, 7, 8, 9, 10,
            12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        )
    GEOMETRY_SCALARS = (0, 2, 3)

    def __init__(
        self,
        field_channels: int,
        physical_channels: int,
        scalar_dim: int,
        code_width: int,
        *,
        response_width: int = 69,
    ) -> None:
        super().__init__()
        if physical_channels != 29 or scalar_dim != 4:
            raise ValueError("geometry-only chart requires FaceDarcyAdapter v2")
        self.field_channels = field_channels
        self.physical_channels = physical_channels
        self.scalar_dim = scalar_dim
        self.code_width = code_width
        self.chart_type = "geometry_tied_linear_green"
        self.response_width = response_width
        dense_channels = len(self.GEOMETRY_CHANNELS) + len(self.GEOMETRY_SCALARS)
        self.response_stem = nn.Sequential(
            nn.Conv2d(dense_channels, response_width, 3, padding=1),
            nn.GroupNorm(_groups(response_width), response_width), nn.SiLU(),
            nn.Conv2d(response_width, response_width, 3, padding=1),
            nn.GroupNorm(_groups(response_width), response_width), nn.SiLU(),
        )
        self.response_head = nn.Conv2d(
            response_width, field_channels * code_width, 1,
        )
        nn.init.zeros_(self.response_head.weight)
        nn.init.zeros_(self.response_head.bias)

    def _response_modes(
        self, local: Tensor, physical: Tensor, scalar: Tensor, valid: Tensor,
    ) -> Tensor:
        del local
        geometry_scalar = scalar[:, self.GEOMETRY_SCALARS, None, None].expand(
            scalar.shape[0], len(self.GEOMETRY_SCALARS), *physical.shape[-2:],
        )
        dense = torch.cat((
            physical[:, self.GEOMETRY_CHANNELS], geometry_scalar,
        ), 1)
        return self.response_head(self.response_stem(dense)).reshape(
            physical.shape[0], self.field_channels, self.code_width,
            *physical.shape[-2:],
        ) * valid[:, :, None]

    def forward(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        scaled_code: Tensor,
    ) -> ChartOutput:
        if scaled_code.shape != (local.shape[0], self.code_width):
            raise ValueError("chart code must have shape flattened_tiles,code_width")
        modes = self._response_modes(local, physical, scalar, valid)
        correction = torch.einsum(
            "bk,bckhw->bchw", scaled_code, modes,
        ) * valid
        return ChartOutput(
            corrected=(local + correction) * valid,
            correction=correction,
            coefficients=scaled_code[:, None, :].expand(
                -1, self.field_channels, -1,
            ),
        )


def build_response_chart(
    chart_type: str,
    field_channels: int,
    physical_channels: int,
    scalar_dim: int,
    code_width: int,
    *,
    response_width: int | None = None,
    learned_modes: int = 12,
) -> ResidualResponseChart | LinearResidualResponseChart | GeometryLinearResidualResponseChart:
    """Build a response chart from checkpoint-stable metadata."""
    if chart_type == "nonlinear_12mode":
        return ResidualResponseChart(
            field_channels, physical_channels, scalar_dim, code_width,
            response_width=64 if response_width is None else response_width,
            learned_modes=learned_modes,
        )
    if chart_type == "tied_linear_green":
        return LinearResidualResponseChart(
            field_channels, physical_channels, scalar_dim, code_width,
            response_width=64 if response_width is None else response_width,
        )
    if chart_type == "geometry_tied_linear_green":
        return GeometryLinearResidualResponseChart(
            field_channels, physical_channels, scalar_dim, code_width,
            response_width=69 if response_width is None else response_width,
        )
    raise ValueError(f"unknown response chart type: {chart_type}")


class ResidualPortCommunicator(nn.Module):
    """Volume/face analyzer, exactly one relative attention, and code head."""

    def __init__(
        self,
        field_channels: int,
        physical_channels: int,
        scalar_dim: int,
        code_width: int,
        *,
        token_width: int = 128,
        heads: int = 4,
        aligned_width: int = 0,
    ) -> None:
        super().__init__()
        summarized = field_channels + physical_channels
        token_channels = (2 + 8) * summarized + scalar_dim
        self.scalar_dim = scalar_dim
        self.token_width = token_width
        if not 0 <= aligned_width <= token_channels:
            raise ValueError("aligned feature width exceeds analyzer input width")
        self.aligned_width = aligned_width
        self.analyzer = nn.Sequential(
            nn.LayerNorm(token_channels),
            nn.Linear(token_channels, token_width), nn.SiLU(),
            nn.Linear(token_width, token_width),
        )
        self.attention = RelativeTileAttention(token_width, heads, 2)
        self.code_head = nn.Sequential(
            nn.LayerNorm(2 * token_width),
            nn.Linear(2 * token_width, 2 * token_width), nn.SiLU(),
            nn.Linear(2 * token_width, code_width),
        )
        nn.init.zeros_(self.code_head[-1].weight)
        nn.init.zeros_(self.code_head[-1].bias)

    def tokens(
        self, local: Tensor, physical: Tensor, scalar: Tensor, valid: Tensor,
        grid_shape: tuple[int, int],
        aligned_features: Tensor | None = None,
    ) -> Tensor:
        tile_count = math.prod(grid_shape)
        if local.shape[0] % tile_count:
            raise ValueError("local batch is incompatible with tile grid")
        summarized = torch.cat((local, physical), dim=1)
        mean, rms = _moments(summarized, valid)
        raw = torch.cat((mean, rms, _face_moments(summarized, valid), scalar), 1)
        if self.aligned_width:
            expected = (
                local.shape[0] // tile_count, tile_count, self.aligned_width,
            )
            if aligned_features is None or aligned_features.shape != expected:
                raise ValueError(
                    f"defect-aligned analyzer requires features shaped {expected}"
                )
            raw = raw.clone()
            raw[:, -self.aligned_width:] = aligned_features.reshape(
                local.shape[0], self.aligned_width,
            ).to(raw)
        elif aligned_features is not None:
            raise ValueError("generic analyzer does not accept aligned features")
        return self.analyzer(raw).reshape(local.shape[0] // tile_count, tile_count, -1)

    def forward(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
        *,
        max_graph_radius: int | None = None,
        aligned_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        local_token = self.tokens(
            local, physical, scalar, valid, grid_shape, aligned_features,
        )
        message = self.attention(
            local_token, grid_shape, max_graph_radius=max_graph_radius,
        )
        code = self.code_head(torch.cat((local_token, message), -1))
        return code, local_token


class SourceLinearResidualPortCommunicator(nn.Module):
    """Geometry-gated, exactly linear map of a supplied reduced RHS.

    The supplied vector contains only homogeneous linear functionals of the
    broken-space POU consensus defect. Linearity and the zero fixed point are
    claims about that vector; Q/K never receives provisional-solution or
    forcing amplitude.
    """

    # FaceDarcyAdapter physical channels excluding local solution/source/BC
    # values. These contain log-k, masks, SDF, normals, canonical coordinates,
    # face roles, and tile validity only.
    GEOMETRY_CHANNELS = (
        1, 3, 4, 6, 7, 8, 9, 10,
        12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    )
    GEOMETRY_SCALARS = (0, 2, 3)

    def __init__(
        self,
        physical_channels: int,
        scalar_dim: int,
        code_width: int,
        signal_width: int,
        *,
        token_width: int = 128,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if physical_channels != 29 or scalar_dim != 4:
            raise ValueError("source-linear analyzer requires FaceDarcyAdapter v2")
        self.scalar_dim = scalar_dim
        self.token_width = token_width
        self.aligned_width = signal_width
        self.analyzer_mode = "source_linear"
        geometry_input = (
            10 * len(self.GEOMETRY_CHANNELS) + len(self.GEOMETRY_SCALARS)
        )
        self.geometry_analyzer = nn.Sequential(
            nn.LayerNorm(geometry_input),
            nn.Linear(geometry_input, token_width), nn.SiLU(),
            nn.Linear(token_width, token_width),
        )
        self.code_map = ResidualKernelCodeMap(
            token_width, signal_width, token_width, code_width,
            heads, 2, zero_init=True,
        )

    @property
    def attention(self) -> nn.Module:
        """Compatibility accessor for analysis code; parameters live in code_map."""
        return self.code_map.global_kernel

    @property
    def local_value(self) -> nn.Module:
        return self.code_map.local_value

    @property
    def code_head(self) -> nn.Module:
        return self.code_map.code_head

    def geometry_tokens(
        self,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
    ) -> Tensor:
        tile_count = math.prod(grid_shape)
        selected = physical[:, self.GEOMETRY_CHANNELS]
        mean, rms = _moments(selected, valid)
        raw = torch.cat((
            mean, rms, _face_moments(selected, valid),
            scalar[:, self.GEOMETRY_SCALARS],
        ), 1)
        return self.geometry_analyzer(raw).reshape(
            physical.shape[0] // tile_count, tile_count, self.token_width,
        )

    def forward(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
        *,
        max_graph_radius: int | None = None,
        aligned_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        del local
        expected = (
            physical.shape[0] // math.prod(grid_shape),
            math.prod(grid_shape), self.aligned_width,
        )
        if aligned_features is None or aligned_features.shape != expected:
            raise ValueError(f"source-linear analysis must have shape {expected}")
        geometry = self.geometry_tokens(physical, scalar, valid, grid_shape)
        code = self.code_map(
            geometry, aligned_features, grid_shape,
            max_graph_radius=max_graph_radius,
        )
        return code, geometry

    def forward_decomposed(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
        split_index: int,
        *,
        max_graph_radius: int | None = None,
        aligned_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return total and prefix-signal codes using one shared kernel.

        This exposes an exact warm-start decomposition without adding another
        attention map: query/key weights are computed once inside
        :meth:`ResidualKernelCodeMap.forward_decomposed`.  When every suffix
        weight is zero, both returned codes are bitwise identical.
        """
        del local
        expected = (
            physical.shape[0] // math.prod(grid_shape),
            math.prod(grid_shape), self.aligned_width,
        )
        if aligned_features is None or aligned_features.shape != expected:
            raise ValueError(f"source-linear analysis must have shape {expected}")
        geometry = self.geometry_tokens(physical, scalar, valid, grid_shape)
        total, prefix = self.code_map.forward_decomposed(
            geometry, aligned_features, grid_shape, split_index,
            max_graph_radius=max_graph_radius,
        )
        return total, prefix, geometry


class ExecutableStateSourceLinearResidualPortCommunicator(
    SourceLinearResidualPortCommunicator,
):
    """Identity-lifted residual kernel whose context may use the local field.

    The accepted Darcy analyzer routes a homogeneous compatibility residual
    with geometry-only tokens.  This controlled extension adds two paths that
    are exactly zero at initialization: a gated provisional-field context and
    a low-rank context-conditioned residual-value chart.  Context is never a
    value by itself, so fixed-context q-linearity and the exact zero fixed point
    are retained.
    """

    def __init__(
        self,
        physical_channels: int,
        scalar_dim: int,
        code_width: int,
        signal_width: int,
        *,
        token_width: int = 128,
        heads: int = 4,
        value_rank: int = 4,
    ) -> None:
        super().__init__(
            physical_channels, scalar_dim, code_width, signal_width,
            token_width=token_width, heads=heads,
        )
        if value_rank <= 0:
            raise ValueError("executable-state value rank must be positive")
        self.value_rank = value_rank
        self.code_map = StateGatedResidualKernelCodeMap(
            token_width, signal_width, token_width, code_width,
            heads, 2, value_rank=value_rank, zero_init=True,
        )
        self.state_analyzer = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, token_width), nn.SiLU(),
            nn.Linear(token_width, token_width),
        )
        self.state_context_gate = nn.Parameter(torch.zeros(token_width))
        self.analyzer_mode = "executable_state_source_linear"

    def context_tokens(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
    ) -> Tensor:
        geometry = self.geometry_tokens(physical, scalar, valid, grid_shape)
        mean, rms = _moments(local, valid)
        state_raw = torch.cat((mean, rms, _face_moments(local, valid)), dim=1)
        if state_raw.shape[1] != 10:
            raise RuntimeError("Darcy executable-state summary must have width 10")
        tile_count = math.prod(grid_shape)
        state = self.state_analyzer(state_raw).reshape(
            local.shape[0] // tile_count, tile_count, self.token_width,
        )
        return geometry + torch.tanh(self.state_context_gate)[None, None] * state

    def forward(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
        *,
        max_graph_radius: int | None = None,
        aligned_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        expected = (
            physical.shape[0] // math.prod(grid_shape),
            math.prod(grid_shape), self.aligned_width,
        )
        if aligned_features is None or aligned_features.shape != expected:
            raise ValueError(f"state-aware analysis must have shape {expected}")
        context = self.context_tokens(
            local, physical, scalar, valid, grid_shape,
        )
        return self.code_map(
            context, aligned_features, grid_shape,
            max_graph_radius=max_graph_radius,
        ), context

    def forward_decomposed(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
        split_index: int,
        *,
        max_graph_radius: int | None = None,
        aligned_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        expected = (
            physical.shape[0] // math.prod(grid_shape),
            math.prod(grid_shape), self.aligned_width,
        )
        if aligned_features is None or aligned_features.shape != expected:
            raise ValueError(f"state-aware analysis must have shape {expected}")
        context = self.context_tokens(
            local, physical, scalar, valid, grid_shape,
        )
        total, prefix = self.code_map.forward_decomposed(
            context, aligned_features, grid_shape, split_index,
            max_graph_radius=max_graph_radius,
        )
        return total, prefix, context


class RelativeGreenFluxCodeMap(nn.Module):
    """Signed translation-relative Green map from flux defects to port codes.

    For a 3x3 tile cover every receiver/source displacement lies in the 5x5
    set ``[-2,2]^2``.  Each displacement owns a bias-free ``code_width x 8``
    matrix.  The map is therefore exactly linear and zero-preserving in the
    supplied flux compatibility coordinates, but unlike softmax attention it
    can learn oriented, sign-changing actions for every relative displacement.
    """

    def __init__(
        self, code_width: int, flux_width: int = 8, max_offset: int = 2,
    ) -> None:
        super().__init__()
        if min(code_width, flux_width) <= 0 or max_offset < 0:
            raise ValueError("invalid relative Green dimensions")
        self.code_width = code_width
        self.flux_width = flux_width
        self.max_offset = max_offset
        side = 2 * max_offset + 1
        self.kernel = nn.Parameter(torch.zeros(
            side, side, code_width, flux_width,
        ))

    def forward(
        self,
        flux: Tensor,
        grid_shape: tuple[int, int],
        max_graph_radius: int | None = None,
    ) -> Tensor:
        if flux.ndim != 3 or flux.shape[-1] != self.flux_width:
            raise ValueError("flux signal must have shape B,N,flux_width")
        if len(grid_shape) != 2 or math.prod(grid_shape) != flux.shape[1]:
            raise ValueError("grid shape is incompatible with flux tokens")
        if max(grid_shape) - 1 > self.max_offset:
            raise ValueError("tile displacement exceeds relative Green table")
        if max_graph_radius is not None and max_graph_radius < 0:
            raise ValueError("max graph radius must be nonnegative or None")
        axes = [
            torch.arange(size, device=flux.device) for size in grid_shape
        ]
        coordinates = torch.stack(
            torch.meshgrid(*axes, indexing="ij"), -1,
        ).reshape(-1, 2)
        # receiver minus source: K_delta maps q_source into z_receiver.
        delta = coordinates[:, None] - coordinates[None, :]
        selected = self.kernel[
            delta[..., 0] + self.max_offset,
            delta[..., 1] + self.max_offset,
        ]
        if max_graph_radius is not None:
            keep = delta.abs().amax(-1) <= max_graph_radius
            selected = selected * keep[..., None, None]
        return torch.einsum("ijcf,bjf->bic", selected, flux)


class RelativeGreenFluxResidualPortCommunicator(nn.Module):
    """Accepted value map plus one signed relative Green flux component.

    The base softmax component and signed relative component consume the same
    globally available compatibility message in one communication stage.  The
    decomposition is disclosed explicitly: the accepted value-only code is
    unchanged, while only the eight flux coordinates enter the new branch.
    """

    def __init__(
        self,
        physical_channels: int,
        scalar_dim: int,
        code_width: int,
        prefix_width: int,
        *,
        flux_width: int = 8,
        token_width: int = 128,
        heads: int = 4,
        max_offset: int = 2,
    ) -> None:
        super().__init__()
        self.base = SourceLinearResidualPortCommunicator(
            physical_channels, scalar_dim, code_width, prefix_width,
            token_width=token_width, heads=heads,
        )
        self.flux_green = RelativeGreenFluxCodeMap(
            code_width, flux_width, max_offset,
        )
        self.scalar_dim = scalar_dim
        self.token_width = token_width
        self.prefix_width = prefix_width
        self.flux_width = flux_width
        self.aligned_width = prefix_width + flux_width
        self.analyzer_mode = "value_flux_relative_green"

    @property
    def attention(self) -> nn.Module:
        return self.base.attention

    def forward_decomposed(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
        split_index: int,
        *,
        max_graph_radius: int | None = None,
        aligned_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if split_index != self.prefix_width:
            raise ValueError("relative Green split does not match value prefix")
        expected = (
            physical.shape[0] // math.prod(grid_shape), math.prod(grid_shape),
            self.aligned_width,
        )
        if aligned_features is None or aligned_features.shape != expected:
            raise ValueError(f"relative Green analysis must have shape {expected}")
        base_code, geometry = self.base(
            local, physical, scalar, valid, grid_shape,
            max_graph_radius=max_graph_radius,
            aligned_features=aligned_features[..., :self.prefix_width],
        )
        increment = self.flux_green(
            aligned_features[..., self.prefix_width:], grid_shape,
            max_graph_radius=max_graph_radius,
        )
        return base_code + increment, base_code, geometry

    def forward(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
        *,
        max_graph_radius: int | None = None,
        aligned_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        total, _, geometry = self.forward_decomposed(
            local, physical, scalar, valid, grid_shape, self.prefix_width,
            max_graph_radius=max_graph_radius,
            aligned_features=aligned_features,
        )
        return total, geometry


class ContextualRelativeGreenFluxCodeMap(RelativeGreenFluxCodeMap):
    """Geometry-gated signed relative Green map, still exactly q-linear.

    The scalar pair gate is ``exp(tanh(a(c_i)+b(c_j)+d_delta))``.  Both
    bias-free maps and the 5x5 relative scalar table are initialized to zero,
    so lifting a fitted relative kernel preserves its behavior exactly.
    Geometry is fixed with respect to the compatibility RHS;
    hence zero preservation, homogeneity, and additivity remain exact.
    """

    def __init__(
        self, code_width: int, context_width: int,
        flux_width: int = 8, max_offset: int = 2,
    ) -> None:
        super().__init__(code_width, flux_width, max_offset)
        if context_width <= 0:
            raise ValueError("context width must be positive")
        if max_offset < 2:
            raise ValueError("a contextual 3x3 cover requires max_offset >= 2")
        self.context_width = context_width
        self.receiver_gate = nn.Linear(context_width, 1, bias=False)
        self.source_gate = nn.Linear(context_width, 1, bias=False)
        self.relative_gate = nn.Parameter(torch.zeros(
            2 * max_offset + 1, 2 * max_offset + 1,
        ))
        nn.init.zeros_(self.receiver_gate.weight)
        nn.init.zeros_(self.source_gate.weight)

    def forward(
        self,
        flux: Tensor,
        context: Tensor,
        grid_shape: tuple[int, int],
        max_graph_radius: int | None = None,
    ) -> Tensor:
        if context.ndim != 3 or context.shape[:2] != flux.shape[:2] or \
                context.shape[-1] != self.context_width:
            raise ValueError("geometry context is incompatible with flux tokens")
        if flux.ndim != 3 or flux.shape[-1] != self.flux_width:
            raise ValueError("flux signal must have shape B,N,flux_width")
        if grid_shape != (3, 3) or flux.shape[1] != 9:
            raise ValueError("contextual relative Green expects a 3x3 cover")
        axes = [torch.arange(size, device=flux.device) for size in grid_shape]
        coordinates = torch.stack(
            torch.meshgrid(*axes, indexing="ij"), -1,
        ).reshape(-1, 2)
        delta = coordinates[:, None] - coordinates[None, :]
        selected = self.kernel[
            delta[..., 0] + self.max_offset,
            delta[..., 1] + self.max_offset,
        ]
        if max_graph_radius is not None:
            if max_graph_radius < 0:
                raise ValueError("max graph radius must be nonnegative or None")
            selected = selected * (
                delta.abs().amax(-1) <= max_graph_radius
            )[..., None, None]
        receiver = self.receiver_gate(context)[:, :, None, 0]
        source = self.source_gate(context)[:, None, :, 0]
        relative = self.relative_gate[
            delta[..., 0] + self.max_offset,
            delta[..., 1] + self.max_offset,
        ]
        gate = torch.exp(torch.tanh(receiver + source + relative[None]))
        return torch.einsum("bij,ijcf,bjf->bic", gate, selected, flux)


class ContextualRelativeGreenFluxResidualPortCommunicator(
    RelativeGreenFluxResidualPortCommunicator,
):
    """Accepted value map plus geometry-gated signed relative Green flux."""

    def __init__(
        self,
        physical_channels: int,
        scalar_dim: int,
        code_width: int,
        prefix_width: int,
        *,
        flux_width: int = 8,
        token_width: int = 128,
        heads: int = 4,
        max_offset: int = 2,
    ) -> None:
        super().__init__(
            physical_channels, scalar_dim, code_width, prefix_width,
            flux_width=flux_width, token_width=token_width, heads=heads,
            max_offset=max_offset,
        )
        self.flux_green = ContextualRelativeGreenFluxCodeMap(
            code_width, token_width, flux_width, max_offset,
        )
        self.analyzer_mode = "value_flux_contextual_relative_green"

    def forward_decomposed(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
        split_index: int,
        *,
        max_graph_radius: int | None = None,
        aligned_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if split_index != self.prefix_width:
            raise ValueError("contextual Green split does not match value prefix")
        expected = (
            physical.shape[0] // math.prod(grid_shape), math.prod(grid_shape),
            self.aligned_width,
        )
        if aligned_features is None or aligned_features.shape != expected:
            raise ValueError(f"contextual Green analysis must have shape {expected}")
        base_code, geometry = self.base(
            local, physical, scalar, valid, grid_shape,
            max_graph_radius=max_graph_radius,
            aligned_features=aligned_features[..., :self.prefix_width],
        )
        increment = self.flux_green(
            aligned_features[..., self.prefix_width:], geometry, grid_shape,
            max_graph_radius=max_graph_radius,
        )
        return base_code + increment, base_code, geometry


class DynamicRelativeGreenFluxCodeMap(RelativeGreenFluxCodeMap):
    r"""Low-rank geometry-conditioned residual around a relative Green map.

    For fixed geometry tokens, the communicated code is

    ``z_i = sum_j [K_delta + sum_r a_ijr(h_i,h_j,delta) B_r] q_j``.

    The coefficient head producing ``a`` is initialized to exactly zero, so a
    fitted :class:`RelativeGreenFluxCodeMap` can be lifted without changing a
    single output bit.  Its input features and the residual bases are nonzero,
    however, so the zero final layer receives a live gradient on the first
    optimization step.  Geometry may change the linear map but never enters
    ``q``; consequently the map remains exactly additive, homogeneous, and
    zero-preserving in the compatibility signal for every fixed context.

    The factorization avoids predicting a dense ``code_width x flux_width``
    matrix for all 81 tile pairs.  With the production defaults it adds fewer
    than 11k parameters to the fitted 5x5 relative kernel.
    """

    def __init__(
        self,
        code_width: int,
        context_width: int,
        flux_width: int = 8,
        max_offset: int = 2,
        *,
        hidden_width: int = 32,
        residual_rank: int = 4,
    ) -> None:
        super().__init__(code_width, flux_width, max_offset)
        if min(context_width, hidden_width, residual_rank) <= 0:
            raise ValueError("dynamic Green dimensions must be positive")
        if max_offset < 2:
            raise ValueError("a dynamic 3x3 cover requires max_offset >= 2")
        self.context_width = context_width
        self.hidden_width = hidden_width
        self.residual_rank = residual_rank
        self.receiver_features = nn.Linear(context_width, hidden_width)
        self.source_features = nn.Linear(context_width, hidden_width)
        side = 2 * max_offset + 1
        self.relative_features = nn.Parameter(torch.empty(
            side, side, hidden_width,
        ))
        self.coefficient_head = nn.Linear(
            hidden_width, residual_rank, bias=False,
        )
        self.residual_basis = nn.Parameter(torch.empty(
            residual_rank, code_width, flux_width,
        ))
        nn.init.normal_(self.relative_features, std=0.02)
        nn.init.normal_(
            self.residual_basis,
            std=1.0 / math.sqrt(float(residual_rank * flux_width)),
        )
        # This is the identity-lift invariant.  Do not also zero the basis:
        # doing so would make the first backward pass dead in both factors.
        nn.init.zeros_(self.coefficient_head.weight)

    def trust_path_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        """Parameters whose zero endpoint removes the entire flux action.

        Hidden feature maps and residual bases deliberately stay off this
        path.  Scaling the fitted ``K`` and the coefficient output layer to
        zero exactly recovers the value-only communicator, while preserving
        learned pair features for every nonzero trust candidate.
        """
        return self.kernel, self.coefficient_head.weight

    def feature_parameters(self) -> tuple[nn.Parameter, ...]:
        """Nonzero geometry features/bases that must not be trust-scaled."""
        return (
            self.receiver_features.weight, self.receiver_features.bias,
            self.source_features.weight, self.source_features.bias,
            self.relative_features, self.residual_basis,
        )

    def forward(
        self,
        flux: Tensor,
        context: Tensor,
        grid_shape: tuple[int, int],
        max_graph_radius: int | None = None,
    ) -> Tensor:
        if flux.ndim != 3 or flux.shape[-1] != self.flux_width:
            raise ValueError("flux signal must have shape B,N,flux_width")
        if context.ndim != 3 or context.shape[:2] != flux.shape[:2] or \
                context.shape[-1] != self.context_width:
            raise ValueError("geometry context is incompatible with flux tokens")
        if grid_shape != (3, 3) or flux.shape[1] != 9:
            raise ValueError("dynamic relative Green expects a 3x3 cover")
        if max_graph_radius is not None and max_graph_radius < 0:
            raise ValueError("max graph radius must be nonnegative or None")

        axes = [torch.arange(size, device=flux.device) for size in grid_shape]
        coordinates = torch.stack(
            torch.meshgrid(*axes, indexing="ij"), -1,
        ).reshape(-1, 2)
        delta = coordinates[:, None] - coordinates[None, :]
        delta_index = (
            delta[..., 0] + self.max_offset,
            delta[..., 1] + self.max_offset,
        )
        fixed = self.kernel[delta_index]
        pair_features = (
            self.receiver_features(context)[:, :, None]
            + self.source_features(context)[:, None, :]
            + self.relative_features[delta_index][None]
        )
        coefficients = self.coefficient_head(torch.nn.functional.silu(
            pair_features,
        ))
        if max_graph_radius is not None:
            keep = delta.abs().amax(-1) <= max_graph_radius
            fixed = fixed * keep[..., None, None]
            coefficients = coefficients * keep[None, ..., None]
        fixed_code = torch.einsum("ijcf,bjf->bic", fixed, flux)
        dynamic_code = torch.einsum(
            "bijr,rcf,bjf->bic", coefficients, self.residual_basis, flux,
        )
        return fixed_code + dynamic_code


class DynamicRelativeGreenFluxResidualPortCommunicator(
    RelativeGreenFluxResidualPortCommunicator,
):
    """Accepted value map plus one geometry-conditioned Green communication."""

    def __init__(
        self,
        physical_channels: int,
        scalar_dim: int,
        code_width: int,
        prefix_width: int,
        *,
        flux_width: int = 8,
        token_width: int = 128,
        heads: int = 4,
        max_offset: int = 2,
        hidden_width: int = 32,
        residual_rank: int = 4,
    ) -> None:
        super().__init__(
            physical_channels, scalar_dim, code_width, prefix_width,
            flux_width=flux_width, token_width=token_width, heads=heads,
            max_offset=max_offset,
        )
        self.flux_green = DynamicRelativeGreenFluxCodeMap(
            code_width, token_width, flux_width, max_offset,
            hidden_width=hidden_width, residual_rank=residual_rank,
        )
        self.analyzer_mode = "value_flux_dynamic_relative_green"

    def forward_decomposed(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
        split_index: int,
        *,
        max_graph_radius: int | None = None,
        aligned_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if split_index != self.prefix_width:
            raise ValueError("dynamic Green split does not match value prefix")
        expected = (
            physical.shape[0] // math.prod(grid_shape), math.prod(grid_shape),
            self.aligned_width,
        )
        if aligned_features is None or aligned_features.shape != expected:
            raise ValueError(f"dynamic Green analysis must have shape {expected}")
        base_code, geometry = self.base(
            local, physical, scalar, valid, grid_shape,
            max_graph_radius=max_graph_radius,
            aligned_features=aligned_features[..., :self.prefix_width],
        )
        increment = self.flux_green(
            aligned_features[..., self.prefix_width:], geometry, grid_shape,
            max_graph_radius=max_graph_radius,
        )
        return base_code + increment, base_code, geometry


class StateGatedDynamicRelativeGreenFluxResidualPortCommunicator(
    DynamicRelativeGreenFluxResidualPortCommunicator,
):
    """Dynamic Green communicator with zero-preserving executable-state gates."""

    def __init__(
        self,
        physical_channels: int,
        scalar_dim: int,
        code_width: int,
        prefix_width: int,
        *,
        flux_width: int = 8,
        token_width: int = 128,
        heads: int = 4,
        max_offset: int = 2,
        hidden_width: int = 32,
        residual_rank: int = 4,
        value_rank: int = 4,
    ) -> None:
        super().__init__(
            physical_channels, scalar_dim, code_width, prefix_width,
            flux_width=flux_width, token_width=token_width, heads=heads,
            max_offset=max_offset, hidden_width=hidden_width,
            residual_rank=residual_rank,
        )
        self.base = ExecutableStateSourceLinearResidualPortCommunicator(
            physical_channels, scalar_dim, code_width, prefix_width,
            token_width=token_width, heads=heads, value_rank=value_rank,
        )
        self.value_rank = value_rank
        self.analyzer_mode = "value_flux_state_gated_dynamic_relative_green"

    def trust_path_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters whose zero setting exactly recovers the accepted map."""
        return (
            self.base.state_context_gate,
            self.base.code_map.global_kernel.state_value_coefficients.weight,
        )

    def trainable_state_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.base.state_analyzer,
            self.base.code_map.global_kernel,
        )


class SignedSourceLinearResidualPortCommunicator(
    SourceLinearResidualPortCommunicator,
):
    """Homogeneous-q communicator with a signed L1-normalized global kernel.

    This is a distinct fallback architecture for oriented remote cancellation.
    It preserves the exact zero signal fixed point and fixed-context linearity
    of :class:`SourceLinearResidualPortCommunicator` but does not overwrite or
    silently reinterpret a softmax-kernel checkpoint.
    """

    def __init__(
        self,
        physical_channels: int,
        scalar_dim: int,
        code_width: int,
        signal_width: int,
        *,
        token_width: int = 128,
        heads: int = 4,
    ) -> None:
        super().__init__(
            physical_channels, scalar_dim, code_width, signal_width,
            token_width=token_width, heads=heads,
        )
        self.code_map = SignedResidualKernelCodeMap(
            token_width, signal_width, token_width, code_width,
            heads, 2, zero_init=True,
        )
        self.kernel_type = "signed_l1_tanh"
