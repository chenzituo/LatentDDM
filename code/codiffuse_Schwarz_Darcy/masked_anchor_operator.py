"""Masked anchor/point Transformer baseline for variable-domain Darcy flow.

This is an end-to-end global operator.  Active grid cells are point tokens,
learned anchors pool the point cloud, a Transformer processes the anchors, and
the active points query the processed anchors before decoding the pressure
field.  It does not reuse the LatentDDM local endpoint or its decomposition.
"""

from __future__ import annotations

import torch
import torch.nn as nn


Tensor = torch.Tensor


class MaskedAnchorOperator(nn.Module):
    """Global point/anchor operator with explicit padding-mask semantics."""

    def __init__(
        self,
        input_channels: int,
        scalar_dim: int,
        *,
        width: int = 192,
        depth: int = 8,
        heads: int = 8,
        anchors: int = 64,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(input_channels, scalar_dim, width, depth, heads, anchors) <= 0:
            raise ValueError("anchor-operator dimensions must be positive")
        if width % heads:
            raise ValueError("anchor width must be divisible by attention heads")
        if mlp_ratio <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("invalid anchor-operator MLP or dropout")
        self.input_channels = int(input_channels)
        self.scalar_dim = int(scalar_dim)
        self.width = int(width)
        self.depth = int(depth)
        self.heads = int(heads)
        self.anchors_count = int(anchors)
        self.mlp_ratio = int(mlp_ratio)
        self.dropout = float(dropout)

        self.point_embedding = nn.Sequential(
            nn.Linear(input_channels, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.scalar_embedding = nn.Sequential(
            nn.Linear(scalar_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.anchors = nn.Parameter(torch.empty(anchors, width))
        nn.init.trunc_normal_(self.anchors, std=0.02)
        self.pool_norm = nn.LayerNorm(width)
        self.pool = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=mlp_ratio * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.processor = nn.TransformerEncoder(layer, num_layers=depth)
        self.query_norm = nn.LayerNorm(width)
        self.anchor_norm = nn.LayerNorm(width)
        self.decode = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def model_spec(self) -> dict[str, int | float | str]:
        return {
            "architecture": "darcy_masked_anchor_operator_v1",
            "input_channels": self.input_channels,
            "scalar_dim": self.scalar_dim,
            "width": self.width,
            "depth": self.depth,
            "heads": self.heads,
            "anchors": self.anchors_count,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
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
        active = mask[:, 0].reshape(batch, height * width) > 0.5
        if not bool(active.any(dim=1).all()):
            raise ValueError("each Darcy sample must contain an active point")
        points = spatial.permute(0, 2, 3, 1).reshape(
            batch, height * width, self.input_channels,
        )
        tokens = self.point_embedding(points)
        tokens = tokens + self.scalar_embedding(scalars)[:, None]
        tokens = tokens * active[:, :, None].to(tokens.dtype)

        anchors = self.anchors[None].expand(batch, -1, -1)
        pooled, _ = self.pool(
            anchors,
            self.pool_norm(tokens),
            self.pool_norm(tokens),
            key_padding_mask=~active,
            need_weights=False,
        )
        anchors = self.processor(anchors + pooled)
        decoded, _ = self.decode(
            self.query_norm(tokens),
            self.anchor_norm(anchors),
            self.anchor_norm(anchors),
            need_weights=False,
        )
        prediction = self.output(tokens + decoded)
        prediction = prediction * active[:, :, None].to(prediction.dtype)
        return prediction.reshape(batch, height, width, 1).permute(0, 3, 1, 2)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
