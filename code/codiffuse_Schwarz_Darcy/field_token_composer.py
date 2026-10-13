"""Darcy adapter for field-first, no-averaging LatentDDM composition."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from codiffuse_Schwarz.field_token_composer import (
    CenteredPortFieldTokenExtension,
    CenteredFieldTokenTransformer,
    FieldTokenComposition,
)

from .residual_port_composer import (
    ChartOutput,
    GeometryLinearResidualResponseChart,
    LinearResidualResponseChart,
    ResidualResponseChart,
    _face_moments,
    _moments,
)
from .train_residual_port_chart import FrozenContext


Tensor = torch.Tensor
DarcyChart = (
    ResidualResponseChart
    | LinearResidualResponseChart
    | GeometryLinearResidualResponseChart
)


def coarse_spatial_moments(
    field: Tensor, valid: Tensor, bins: int,
) -> Tensor:
    """Return validity-weighted mean/RMS values on a local spatial grid."""
    if field.ndim != 4 or valid.shape != (
        field.shape[0], 1, *field.shape[-2:]
    ):
        raise ValueError("field and validity mask are incompatible")
    if bins <= 1:
        raise ValueError("spatial ports require at least two bins")
    weight = valid.to(field).clamp(0.0, 1.0)
    mass = F.adaptive_avg_pool2d(weight, (bins, bins))
    support = (mass > 0.0).to(field)
    denominator = mass.clamp_min(1.0e-8)
    mean = F.adaptive_avg_pool2d(field * weight, (bins, bins)) / denominator
    square = F.adaptive_avg_pool2d(field.square() * weight, (bins, bins))
    rms = torch.sqrt(square / denominator + 1.0e-8)
    return torch.cat((mean * support, rms * support), dim=1)


class DarcyFieldTokenEncoder(nn.Module):
    """Encode local inputs and their executable endpoint field into ``z_i^0``."""

    def __init__(
        self,
        field_channels: int,
        physical_channels: int,
        scalar_dim: int,
        width: int,
    ) -> None:
        super().__init__()
        summarized = field_channels + physical_channels
        # Mean/rms plus mean/rms on four oriented faces.
        feature_width = 10 * summarized + scalar_dim
        self.width = width
        self.encoder = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
    ) -> Tensor:
        tile_count = math.prod(grid_shape)
        if local.shape[0] % tile_count:
            raise ValueError("local batch is incompatible with the cover")
        summarized = torch.cat((physical, local), dim=1)
        mean, rms = _moments(summarized, valid)
        features = torch.cat((
            mean, rms, _face_moments(summarized, valid), scalar,
        ), dim=1)
        return self.encoder(features).reshape(
            local.shape[0] // tile_count, tile_count, self.width,
        )


@dataclass(frozen=True)
class DarcyFieldTokenOutput:
    assembled: Tensor
    corrected_tiles: Tensor
    chart: ChartOutput
    latent: FieldTokenComposition


class DarcyFieldTokenComposer(nn.Module):
    """Compose a frozen Darcy endpoint through stacked field-token exchange."""

    contract = "darcy_field_token_latentddm_core_v1"

    def __init__(
        self,
        physical_channels: int,
        scalar_dim: int,
        code_width: int,
        *,
        token_width: int = 128,
        heads: int = 4,
        communication_depth: int = 2,
        spatial_port_bins: int = 0,
    ) -> None:
        super().__init__()
        if spatial_port_bins not in (0, 2, 3, 4):
            raise ValueError("spatial_port_bins must be 0, 2, 3, or 4")
        self.physical_channels = physical_channels
        self.scalar_dim = scalar_dim
        self.code_width = code_width
        self.token_width = token_width
        self.communication_depth = communication_depth
        self.spatial_port_bins = int(spatial_port_bins)
        self.encoder = DarcyFieldTokenEncoder(
            1, physical_channels, scalar_dim, token_width,
        )
        self.communication = CenteredFieldTokenTransformer(
            token_width, code_width, heads, dimension=2,
            depth=communication_depth,
        )
        port_count = self.spatial_port_bins**2
        summarized = 1 + physical_channels
        self.spatial_port_projection = (
            nn.Sequential(
                nn.LayerNorm(2 * summarized),
                nn.Linear(2 * summarized, token_width),
                nn.SiLU(),
                nn.Linear(token_width, token_width, bias=False),
            ) if port_count else None
        )
        self.spatial_port_scalar = (
            nn.Linear(scalar_dim, token_width, bias=False)
            if port_count else None
        )
        self.spatial_port_embedding = (
            nn.Embedding(port_count, token_width) if port_count else None
        )
        if self.spatial_port_embedding is not None:
            nn.init.normal_(self.spatial_port_embedding.weight, std=0.02)
        self.spatial_port_communication = (
            CenteredPortFieldTokenExtension(
                token_width, code_width, heads, dimension=2,
                ports=port_count, depth=communication_depth,
            ) if port_count else None
        )

    def encode_spatial_port_tokens(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
    ) -> Tensor:
        """Keep coarse local position while forming communicated summaries."""
        if (
            self.spatial_port_projection is None
            or self.spatial_port_scalar is None
            or self.spatial_port_embedding is None
        ):
            raise ValueError("spatial ports are not configured")
        summarized = torch.cat((physical, local), dim=1)
        moments = coarse_spatial_moments(
            summarized, valid, self.spatial_port_bins,
        )
        port_count = self.spatial_port_bins**2
        moments = moments.flatten(2).permute(0, 2, 1)
        ports = self.spatial_port_projection(moments)
        tile_count = math.prod(grid_shape)
        if local.shape[0] % tile_count:
            raise ValueError("local batch is incompatible with the cover")
        batch = local.shape[0] // tile_count
        scalar_token = self.spatial_port_scalar(scalar).reshape(
            batch, tile_count, self.token_width,
        )
        port_index = torch.arange(port_count, device=local.device)
        role = self.spatial_port_embedding(port_index)[None, None]
        return ports.reshape(
            batch, tile_count, port_count, self.token_width,
        ) + scalar_token[:, :, None] + role

    def forward(
        self,
        chart: DarcyChart,
        context: FrozenContext,
        mask: Tensor,
        *,
        graph_mask: Tensor | None = None,
    ) -> DarcyFieldTokenOutput:
        batch = context.target_normalized.shape[0]
        token = self.encoder(
            context.local, context.physical, context.scalar, context.valid,
            context.cover.grid_shape,
        )
        latent = self.communication(
            token, grid_shape=context.cover.grid_shape,
            graph_mask=graph_mask,
        )
        if self.spatial_port_communication is not None:
            ports = self.encode_spatial_port_tokens(
                context.local, context.physical, context.scalar,
                context.valid, context.cover.grid_shape,
            )
            port_result = self.spatial_port_communication(
                ports, grid_shape=context.cover.grid_shape,
                graph_mask=graph_mask,
            )
            latent = replace(
                latent,
                global_code=latent.global_code + port_result.global_code,
                diagonal_code=latent.diagonal_code + port_result.diagonal_code,
                correction_code=(
                    latent.correction_code + port_result.correction_code
                ),
            )
        chart_output = chart(
            context.local, context.physical, context.scalar, context.valid,
            latent.correction_code.reshape(-1, self.code_width),
        )
        assembled = context.cover.assemble_ras(
            chart_output.corrected, batch,
        ) * mask
        return DarcyFieldTokenOutput(
            assembled=assembled,
            corrected_tiles=chart_output.corrected,
            chart=chart_output,
            latent=latent,
        )
