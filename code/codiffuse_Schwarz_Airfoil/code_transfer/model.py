"""Generic local Flow operator and learned neural endpoint projection."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from codiffuse_Schwarz.tile_communication import TileGridCommunicator


Tensor = torch.Tensor


def timestep_embedding(time: Tensor, width: int) -> Tensor:
    half = width // 2
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=time.device) / max(half - 1, 1)
    )
    angles = time[:, None] * frequencies[None]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    return F.pad(embedding, (0, width - embedding.shape[1]))


class AxialAttentionBlock(nn.Module):
    """Attention along rows and columns avoids quadratic full-patch attention."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.row_norm = nn.LayerNorm(width)
        self.row_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.column_norm = nn.LayerNorm(width)
        self.column_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, field: Tensor) -> Tensor:
        batch, channels, height, width = field.shape
        rows = field.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        normalized = self.row_norm(rows)
        rows = rows + self.row_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        field = rows.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        columns = field.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        normalized = self.column_norm(columns)
        columns = columns + self.column_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        tokens = columns.reshape(batch, width, height, channels).permute(0, 3, 2, 1)
        flat = tokens.permute(0, 2, 3, 1).reshape(-1, channels)
        flat = flat + self.mlp(self.mlp_norm(flat))
        return flat.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


class FiLMResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int = 1) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, width), width)
        self.conv1 = nn.Conv2d(
            width, width, 3, padding=dilation, dilation=dilation
        )
        self.conv2 = nn.Conv2d(
            width, width, 3, padding=dilation, dilation=dilation
        )
        self.affine = nn.Linear(width, 2 * width)

    def forward(self, field: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.affine(condition).chunk(2, dim=1)
        hidden = self.norm(field)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(self.conv1(F.silu(hidden))))
        return field + hidden


class LocalFlowOperator(nn.Module):
    """Shared conditional Flow-Matching velocity model for every Cartesian cell."""

    def __init__(
        self, state_channels: int, physical_channels: int, scalar_dim: int,
        width: int = 128, depth: int = 6, heads: int = 8,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.state_channels = state_channels
        self.physical_channels = physical_channels
        self.scalar_dim = scalar_dim
        self.width = width
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(state_channels + physical_channels, width, 5, padding=2), nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),
        )
        self.processor = nn.ModuleList(
            AxialAttentionBlock(width, heads) for _ in range(depth)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(width, width // 2, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width // 2, state_channels, 3, padding=1),
        )

    def forward(
        self, flow_state: Tensor, physical: Tensor, scalar: Tensor, flow_time: Tensor,
    ) -> Tensor:
        if physical.shape[1] != self.physical_channels:
            raise ValueError("physical condition channel count does not match model")
        condition = self.time_mlp(timestep_embedding(flow_time, self.width))
        condition = condition + self.condition_mlp(scalar)
        hidden = self.encoder(torch.cat((flow_state, physical), dim=1))
        hidden = hidden + condition[:, :, None, None]
        for block in self.processor:
            hidden = block(hidden)
        velocity = self.decoder(hidden)
        if velocity.shape[-2:] != flow_state.shape[-2:]:
            velocity = F.interpolate(
                velocity, size=flow_state.shape[-2:], mode="bilinear", align_corners=False
            )
        return velocity * physical[:, -1:]


class NeuralEndpointProjector(nn.Module):
    """Learned boundary/interface-compatible correction at a clean endpoint.

    This is the transferable neural EoS role. The adapter places every
    system-specific feature needed by the projection in `physical`; no
    divergence, curl, SDF, or boundary type is hard-coded here.
    """

    def __init__(
        self, state_channels: int, physical_channels: int, scalar_dim: int,
        width: int = 96, depth: int = 6, cell_attention: bool = True,
        grid_communication: bool = False,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(state_channels + physical_channels, width, 5, padding=2)
        self.condition = nn.Sequential(
            nn.Linear(scalar_dim, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        self.blocks = nn.ModuleList(
            FiLMResidualBlock(width, (1, 2, 4)[index % 3]) for index in range(depth)
        )
        self.output = nn.Conv2d(width, state_channels, 3, padding=1)
        self.attention_norm = nn.LayerNorm(width) if cell_attention else None
        self.attention = (
            nn.MultiheadAttention(width, 8 if width % 8 == 0 else 4, batch_first=True)
            if cell_attention else None
        )
        self.attention_affine = nn.Linear(width, 2 * width) if cell_attention else None
        self.grid_communicator = (
            TileGridCommunicator(
                width, scalar_dim, dimension=2,
                heads=8 if width % 8 == 0 else 4,
            )
            if grid_communication else None
        )
        if self.attention_affine is not None:
            # Cell attention is an optional communication refinement. Starting
            # from zero FiLM modulation preserves the local pretrained model.
            nn.init.zeros_(self.attention_affine.weight)
            nn.init.zeros_(self.attention_affine.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, endpoint: Tensor, physical: Tensor, scalar: Tensor,
        tile_count: int | None = None,
        tile_grid_shape: tuple[int, int] | None = None,
    ) -> Tensor:
        condition = self.condition(scalar)
        hidden = self.stem(torch.cat((endpoint, physical), dim=1))
        for block in self.blocks:
            hidden = block(hidden, condition)
        if self.attention is not None and tile_count is not None:
            if hidden.shape[0] % tile_count:
                raise ValueError("local batch is not divisible by tile_count")
            batch = hidden.shape[0] // tile_count
            tokens = hidden.mean((-1, -2)).reshape(batch, tile_count, -1)
            normalized = self.attention_norm(tokens)  # type: ignore[operator]
            attended = self.attention(
                normalized, normalized, normalized, need_weights=False
            )[0]
            scale, shift = self.attention_affine(attended).chunk(2, -1)  # type: ignore[operator]
            hidden = hidden * (1.0 + scale.reshape(-1, scale.shape[-1], 1, 1)) \
                + shift.reshape(-1, shift.shape[-1], 1, 1)
        if self.grid_communicator is not None:
            hidden = self.grid_communicator(hidden, scalar, tile_grid_shape)
        correction = self.output(F.silu(hidden))
        return (endpoint + correction) * physical[:, -1:]


class DirectEndpointRegressor(nn.Module):
    """Predict a clean normalized physical increment without a noise path.

    State history, geometry, controls, coefficients, and scale information are
    supplied through ``physical`` and ``scalar`` by the adapter.  The final
    physical channel is the patch-validity mask.  Optional cell attention is
    disabled by default because the retained airfoil model communicates when
    endpoints are assembled, not through a dense hidden global operation.
    """

    def __init__(
        self, state_channels: int, physical_channels: int, scalar_dim: int,
        width: int = 128, depth: int = 8, cell_attention: bool = False,
        grid_communication: bool = False,
    ) -> None:
        super().__init__()
        self.state_channels = state_channels
        self.physical_channels = physical_channels
        self.scalar_dim = scalar_dim
        self.stem = nn.Conv2d(physical_channels, width, 5, padding=2)
        self.condition = nn.Sequential(
            nn.Linear(scalar_dim, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        self.blocks = nn.ModuleList(
            FiLMResidualBlock(width, (1, 2, 4)[index % 3]) for index in range(depth)
        )
        self.attention_norm = nn.LayerNorm(width) if cell_attention else None
        if width % 8 == 0:
            heads = 8
        elif width % 4 == 0:
            heads = 4
        elif width % 2 == 0:
            heads = 2
        else:
            heads = 1
        self.attention = (
            nn.MultiheadAttention(width, heads, batch_first=True)
            if cell_attention else None
        )
        self.attention_affine = (
            nn.Linear(width, 2 * width) if cell_attention else None
        )
        self.grid_communicator = (
            TileGridCommunicator(
                width, scalar_dim, dimension=2,
                heads=8 if width % 8 == 0 else 4,
            )
            if grid_communication else None
        )
        self.output = nn.Conv2d(width, state_channels, 3, padding=1)
        # Zero is the normalized no-change endpoint.  This gives residual
        # learning a stable identity physical prediction at initialization.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if self.attention_affine is not None:
            nn.init.zeros_(self.attention_affine.weight)
            nn.init.zeros_(self.attention_affine.bias)

    def forward(
        self, physical: Tensor, scalar: Tensor, tile_count: int | None = None,
        tile_grid_shape: tuple[int, int] | None = None,
    ) -> Tensor:
        if physical.shape[1] != self.physical_channels:
            raise ValueError("physical condition channel count does not match model")
        condition = self.condition(scalar)
        hidden = self.stem(physical)
        for block in self.blocks:
            hidden = block(hidden, condition)
        if self.attention is not None and tile_count is not None:
            if hidden.shape[0] % tile_count:
                raise ValueError("local batch is not divisible by tile_count")
            batch = hidden.shape[0] // tile_count
            tokens = hidden.mean((-1, -2)).reshape(batch, tile_count, -1)
            normalized = self.attention_norm(tokens)  # type: ignore[operator]
            attended = self.attention(
                normalized, normalized, normalized, need_weights=False
            )[0]
            scale, shift = self.attention_affine(attended).chunk(2, -1)  # type: ignore[operator]
            hidden = hidden * (1.0 + scale.reshape(-1, scale.shape[-1], 1, 1)) \
                + shift.reshape(-1, shift.shape[-1], 1, 1)
        if self.grid_communicator is not None:
            hidden = self.grid_communicator(hidden, scalar, tile_grid_shape)
        return self.output(F.silu(hidden)) * physical[:, -1:]


class DirectEndpointOperator(nn.Module):
    """Retained-model abstraction: direct proposal plus learned endpoint EoS."""

    def __init__(
        self, proposal: DirectEndpointRegressor,
        projector: NeuralEndpointProjector | None = None,
    ) -> None:
        super().__init__()
        self.proposal = proposal
        self.projector = projector

    def predict_fields(
        self, physical: Tensor, scalar: Tensor, tile_count: int | None = None,
        tile_grid_shape: tuple[int, int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        proposal = self.proposal(physical, scalar, tile_count, tile_grid_shape)
        corrected = proposal if self.projector is None else self.projector(
            proposal, physical, scalar, tile_count, tile_grid_shape
        )
        return corrected, proposal

    def forward(
        self, physical: Tensor, scalar: Tensor, tile_count: int | None = None,
        tile_grid_shape: tuple[int, int] | None = None,
    ) -> Tensor:
        return self.predict_fields(
            physical, scalar, tile_count, tile_grid_shape,
        )[0]


class CoDiffuseOperator(nn.Module):
    """Experimental Flow/PnP container retained for future development."""

    def __init__(
        self, flow: LocalFlowOperator,
        projector: NeuralEndpointProjector | None = None,
    ) -> None:
        super().__init__()
        self.flow = flow
        self.projector = projector

    def forward(self, state: Tensor, physical: Tensor, scalar: Tensor, time: Tensor) -> Tensor:
        return self.flow(state, physical, scalar, time)

    def project_endpoint(
        self, endpoint: Tensor, physical: Tensor, scalar: Tensor,
        tile_count: int | None = None,
    ) -> Tensor:
        if self.projector is None:
            return endpoint * physical[:, -1:]
        return self.projector(endpoint, physical, scalar, tile_count)


# Compatibility name for callers that want to make the non-default status
# explicit without breaking earlier imports of CoDiffuseOperator.
ExperimentalFlowOperator = CoDiffuseOperator
