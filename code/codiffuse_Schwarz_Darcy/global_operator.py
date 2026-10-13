"""Mask-aware global Transolver for variable-domain Darcy problems.

This is a point operator rather than a tiled convolutional model.  Every
active Cartesian cell is projected to a point token, softly assigned to a
small set of physics slices, globally mixed in slice space, and desliced back
to the original points.  The mask participates in every slice reduction, so
padding/outside-domain values cannot affect active predictions.  The same
weights accept the shared 96x96 full domains and differently sized A--D
evaluation domains.
"""

from __future__ import annotations

import torch
import torch.nn as nn


Tensor = torch.Tensor


class MaskedPhysicsAttention(nn.Module):
    """Transolver-style slice attention over a padded set of active points."""

    def __init__(self, width: int, heads: int, slices: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if min(width, heads, slices) <= 0:
            raise ValueError("attention dimensions must be positive")
        self.width = width
        self.heads = heads
        self.slices = slices
        self.head_width = width // heads
        self.scale = self.head_width**-0.5
        self.feature = nn.Linear(width, width)
        self.assignment = nn.Linear(width, heads * slices)
        self.temperature = nn.Parameter(torch.full((1, 1, heads, 1), 0.5))
        self.to_q = nn.Linear(self.head_width, self.head_width, bias=False)
        self.to_k = nn.Linear(self.head_width, self.head_width, bias=False)
        self.to_v = nn.Linear(self.head_width, self.head_width, bias=False)
        self.output = nn.Linear(width, width)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
            raise ValueError("tokens must be B,N,C and mask must be B,N")
        batch, points, _ = tokens.shape
        valid = mask[:, :, None, None].to(tokens.dtype)
        features = self.feature(tokens).reshape(
            batch, points, self.heads, self.head_width,
        )
        logits = self.assignment(tokens).reshape(
            batch, points, self.heads, self.slices,
        )
        weights = torch.softmax(
            logits / self.temperature.clamp(0.1, 5.0), dim=-1,
        ) * valid
        mass = weights.sum(dim=1).clamp_min(1.0e-6)
        physics_tokens = torch.einsum("bnhd,bnhs->bhsd", features, weights)
        physics_tokens = physics_tokens / mass[..., None]

        query = self.to_q(physics_tokens)
        key = self.to_k(physics_tokens)
        value = self.to_v(physics_tokens)
        attention = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) * self.scale, dim=-1,
        )
        mixed = torch.matmul(attention, value)
        output = torch.einsum("bhsd,bnhs->bnhd", mixed, weights)
        output = output.reshape(batch, points, self.width)
        return self.output(output) * mask[:, :, None].to(tokens.dtype)


class MaskedTransolverBlock(nn.Module):
    def __init__(
        self, width: int, heads: int, slices: int, mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        self.attention_norm = nn.LayerNorm(width)
        self.attention = MaskedPhysicsAttention(width, heads, slices)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, mlp_ratio * width),
            nn.GELU(),
            nn.Linear(mlp_ratio * width, width),
        )

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        valid = mask[:, :, None].to(tokens.dtype)
        tokens = tokens + self.attention(self.attention_norm(tokens), mask)
        tokens = tokens + self.mlp(self.mlp_norm(tokens)) * valid
        return tokens * valid


class MaskedTransolverOperator(nn.Module):
    """Global scalar correction operator for arbitrarily sized masked grids."""

    def __init__(
        self,
        input_channels: int,
        scalar_dim: int,
        width: int = 192,
        depth: int = 8,
        heads: int = 8,
        slices: int = 32,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        if min(input_channels, scalar_dim, width, depth) <= 0:
            raise ValueError("model dimensions must be positive")
        self.input_channels = input_channels
        self.scalar_dim = scalar_dim
        self.width = width
        self.depth = depth
        self.heads = heads
        self.slices = slices
        self.mlp_ratio = mlp_ratio
        self.input_projection = nn.Sequential(
            nn.Linear(input_channels, width), nn.GELU(), nn.Linear(width, width),
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(scalar_dim, 2 * width), nn.GELU(), nn.Linear(2 * width, width),
        )
        self.blocks = nn.ModuleList([
            MaskedTransolverBlock(width, heads, slices, mlp_ratio)
            for _ in range(depth)
        ])
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def model_spec(self) -> dict[str, int | str]:
        return {
            "architecture": "masked_point_transolver_v1",
            "input_channels": self.input_channels,
            "scalar_dim": self.scalar_dim,
            "width": self.width,
            "depth": self.depth,
            "heads": self.heads,
            "slices": self.slices,
            "mlp_ratio": self.mlp_ratio,
        }

    def forward(self, spatial: Tensor, scalars: Tensor, mask: Tensor) -> Tensor:
        if spatial.ndim != 4 or spatial.shape[1] != self.input_channels:
            raise ValueError(
                f"expected B,{self.input_channels},H,W spatial conditions"
            )
        if scalars.shape != (spatial.shape[0], self.scalar_dim):
            raise ValueError("scalar condition shape does not match model")
        if mask.shape != (spatial.shape[0], 1, *spatial.shape[-2:]):
            raise ValueError("mask must have shape B,1,H,W")
        batch, _, height, width = spatial.shape
        point_mask = mask[:, 0].reshape(batch, height * width) > 0.5
        inputs = spatial.permute(0, 2, 3, 1).reshape(
            batch, height * width, self.input_channels,
        )
        tokens = self.input_projection(inputs)
        tokens = tokens + self.scalar_projection(scalars)[:, None]
        tokens = tokens * point_mask[:, :, None].to(tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, point_mask)
        output = self.output(self.output_norm(tokens))
        output = output * point_mask[:, :, None].to(output.dtype)
        return output.reshape(batch, height, width, 1).permute(0, 3, 1, 2)
