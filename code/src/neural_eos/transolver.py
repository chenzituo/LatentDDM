"""Full-field Transolver components for structured Cartesian PDE states.

Physics-state slicing follows the Transolver formulation (Wu et al., 2024).
The spatial projections are depthwise-separable so a 192x384 field remains
tractable while each global slice assignment still sees local grid context.
"""

from __future__ import annotations

import torch
import torch.nn as nn


Tensor = torch.Tensor


class ChannelLayerNorm(nn.Module):
    """Layer normalization over channels of a BCHW tensor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, field: Tensor) -> Tensor:
        return self.norm(field.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SpatialPointProjection(nn.Module):
    """Efficient structured-mesh replacement for Transolver's dense 3x3 map."""

    def __init__(self, channels: int, output_channels: int) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels,
        )
        self.pointwise = nn.Conv2d(channels, output_channels, kernel_size=1)

    def forward(self, field: Tensor) -> Tensor:
        return self.pointwise(self.spatial(field))


class PhysicsAttention2D(nn.Module):
    """Slice, attend, and deslice a complete structured 2D physical field."""

    def __init__(
        self, channels: int, heads: int = 8, slices: int = 32,
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        if slices <= 0:
            raise ValueError("slices must be positive")
        self.heads = heads
        self.head_channels = channels // heads
        self.slices = slices
        self.scale = self.head_channels**-0.5
        self.temperature = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        self.feature_projection = SpatialPointProjection(channels, channels)
        self.assignment_projection = SpatialPointProjection(channels, channels)
        self.slice_projection = nn.Linear(self.head_channels, slices)
        nn.init.orthogonal_(self.slice_projection.weight)
        self.to_q = nn.Linear(self.head_channels, self.head_channels, bias=False)
        self.to_k = nn.Linear(self.head_channels, self.head_channels, bias=False)
        self.to_v = nn.Linear(self.head_channels, self.head_channels, bias=False)
        self.output = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, field: Tensor) -> Tensor:
        batch, channels, height, width = field.shape
        points = height * width
        features = self.feature_projection(field).reshape(
            batch, self.heads, self.head_channels, points,
        ).permute(0, 1, 3, 2)
        assignments = self.assignment_projection(field).reshape(
            batch, self.heads, self.head_channels, points,
        ).permute(0, 1, 3, 2)
        temperature = self.temperature.clamp(0.1, 5.0)
        weights = torch.softmax(
            self.slice_projection(assignments) / temperature, dim=-1,
        )
        mass = weights.sum(dim=2).clamp_min(1.0e-5)
        tokens = torch.einsum("bhnd,bhns->bhsd", features, weights)
        tokens = tokens / mass[..., None]

        queries = self.to_q(tokens)
        keys = self.to_k(tokens)
        values = self.to_v(tokens)
        attention = torch.softmax(
            torch.matmul(queries, keys.transpose(-1, -2)) * self.scale, dim=-1,
        )
        attended = torch.matmul(attention, values)
        output = torch.einsum("bhsd,bhns->bhnd", attended, weights)
        output = output.permute(0, 1, 3, 2).reshape(
            batch, channels, height, width,
        )
        return self.output(output)


class TransolverBlock2D(nn.Module):
    def __init__(
        self, channels: int, heads: int, slices: int, mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        expanded = channels * mlp_ratio
        self.attention_norm = ChannelLayerNorm(channels)
        self.attention = PhysicsAttention2D(channels, heads, slices)
        self.mlp_norm = ChannelLayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, expanded, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(expanded, channels, kernel_size=1),
        )

    def forward(self, field: Tensor) -> Tensor:
        field = field + self.attention(self.attention_norm(field))
        return field + self.mlp(self.mlp_norm(field))


class LocalResidualHead(nn.Module):
    """Full-field local path for derivatives that physics slices may smooth."""

    def __init__(self, input_channels: int, width: int = 96, depth: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(input_channels, width, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, width),
                nn.GELU(),
                nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width),
                nn.Conv2d(width, width, kernel_size=1),
            )
            for _ in range(depth)
        ])
        self.output = nn.Conv2d(width, 3, kernel_size=3, padding=1)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.stem(inputs)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(torch.nn.functional.gelu(hidden))


class StructuredTransolverOperator(nn.Module):
    """A non-stitched global operator with Transolver physics attention."""

    def __init__(
        self,
        input_channels: int,
        width: int = 368,
        depth: int = 8,
        heads: int = 8,
        slices: int = 32,
        mlp_ratio: int = 4,
        local_width: int = 96,
        local_depth: int = 4,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.width = width
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=1), nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
        )
        self.blocks = nn.ModuleList([
            TransolverBlock2D(width, heads, slices, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = ChannelLayerNorm(width)
        self.global_output = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, 3, kernel_size=1),
        )
        self.local_output = LocalResidualHead(
            input_channels, local_width, local_depth,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"expected BCHW input with {self.input_channels} channels"
            )
        hidden = self.stem(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return self.global_output(self.norm(hidden)) + self.local_output(inputs)

