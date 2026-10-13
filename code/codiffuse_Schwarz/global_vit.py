"""Plain full-domain Vision Transformer baselines for physical fields.

The model in this module is deliberately *not* a domain-decomposition model.
It partitions a complete Cartesian field into disjoint patches, embeds every
patch, performs global self-attention over all patch tokens, and decodes the
tokens directly back to a complete field.  It has no local neural-operator
checkpoint, overlap, partition of unity, or communication bypass.
"""

from __future__ import annotations

from math import prod
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _tuple(value: int | Sequence[int], dimensions: int) -> tuple[int, ...]:
    if isinstance(value, int):
        result = (value,) * dimensions
    else:
        result = tuple(int(item) for item in value)
    if len(result) != dimensions or any(item <= 0 for item in result):
        raise ValueError(
            f"expected {dimensions} positive patch dimensions, got {result!r}"
        )
    return result


def patch_position_encoding(
    grid_shape: Sequence[int], width: int, device: torch.device, dtype: torch.dtype,
) -> Tensor:
    """Return a deterministic Fourier encoding of patch-center positions."""

    shape = tuple(int(item) for item in grid_shape)
    if not shape or any(item <= 0 for item in shape):
        raise ValueError(f"invalid patch grid {shape!r}")
    if width <= 0:
        raise ValueError("width must be positive")
    coordinates = torch.meshgrid(
        *[
            torch.linspace(-1.0, 1.0, count, device=device, dtype=torch.float32)
            for count in shape
        ],
        indexing="ij",
    )
    dimensions = len(shape)
    pairs_per_axis = max(1, width // (2 * dimensions))
    # Bounded integer harmonics avoid backend-dependent argument reduction.
    # Exponentially growing frequencies become numerically meaningless in
    # float32 long before the embedding width is filled.
    frequencies = torch.pi * torch.arange(
        1, pairs_per_axis + 1, device=device, dtype=torch.float32,
    )
    features = []
    for coordinate in coordinates:
        angles = coordinate.reshape(-1, 1) * frequencies.reshape(1, -1)
        features.extend((angles.sin(), angles.cos()))
    encoding = torch.cat(features, dim=1)
    if encoding.shape[1] < width:
        encoding = F.pad(encoding, (0, width - encoding.shape[1]))
    return encoding[:, :width].to(dtype=dtype)


class TransformerBlock(nn.Module):
    """Pre-normalized global self-attention block."""

    def __init__(
        self, width: int, heads: int, mlp_ratio: int = 4, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"width {width} must be divisible by heads {heads}")
        self.norm_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(width)
        hidden = width * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, padding_mask: Tensor | None = None,
                attention_bias: Tensor | None = None) -> Tensor:
        normalized = self.norm_attention(tokens)
        if attention_bias is not None and padding_mask is not None:
            padding_mask = torch.zeros_like(padding_mask, dtype=tokens.dtype).masked_fill(
                padding_mask, float('-inf'),
            )
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            attn_mask=attention_bias,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.mlp(self.norm_mlp(tokens))


class GlobalPatchTransformer(nn.Module):
    """End-to-end full-domain ViT with direct patch reconstruction.

    Parameters are always initialized normally by PyTorch and are intended to
    be optimized jointly from scratch.  ``mask`` is only a physical-domain
    validity mask; it never creates overlap or POU weights.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dimensions: int,
        patch_size: int | Sequence[int],
        width: int,
        depth: int,
        heads: int,
        mlp_ratio: int = 4,
        scalar_dim: int = 0,
        dropout: float = 0.0,
        position_encoding: str = "normalized_linear_fourier_v1",
    ) -> None:
        super().__init__()
        if dimensions not in (2, 3):
            raise ValueError("GlobalPatchTransformer supports 2-D or 3-D fields")
        if min(input_channels, output_channels, width, depth, heads) <= 0:
            raise ValueError("channel, width, depth, and head counts must be positive")
        if scalar_dim < 0:
            raise ValueError("scalar_dim must be nonnegative")
        if position_encoding not in (
            "normalized_linear_fourier_v1", "fixed_tile_fourier_v1",
            "fixed_tile_fourier_locality_v1",
        ):
            raise ValueError(f"unsupported position encoding {position_encoding!r}")
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.dimensions = int(dimensions)
        self.patch_size = _tuple(patch_size, dimensions)
        self.width = int(width)
        self.depth = int(depth)
        self.heads = int(heads)
        self.mlp_ratio = int(mlp_ratio)
        self.scalar_dim = int(scalar_dim)
        self.dropout = float(dropout)
        self.position_encoding = position_encoding
        convolution = nn.Conv2d if dimensions == 2 else nn.Conv3d
        self.patch_embedding = convolution(
            input_channels,
            width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.scalar_embedding = (
            nn.Linear(scalar_dim, width) if scalar_dim else None
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(width, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width)
        self.patch_decoder = nn.Linear(
            width, output_channels * prod(self.patch_size),
        )

    @property
    def model_args(self) -> dict[str, int | float | str | tuple[int, ...]]:
        return {
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "dimensions": self.dimensions,
            "patch_size": self.patch_size,
            "width": self.width,
            "depth": self.depth,
            "heads": self.heads,
            "mlp_ratio": self.mlp_ratio,
            "scalar_dim": self.scalar_dim,
            "dropout": self.dropout,
            "position_encoding": self.position_encoding,
        }

    def model_spec(self) -> dict[str, int | float | tuple[int, ...] | str]:
        """Serializable architecture contract used by experiment checkpoints."""

        return {"architecture": "global_patch_transformer", **self.model_args}

    def _padding(self, spatial_shape: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        amounts = tuple(
            (-int(size)) % patch
            for size, patch in zip(spatial_shape, self.patch_size, strict=True)
        )
        # torch.nn.functional.pad lists the last dimension first.
        torch_padding = tuple(value for amount in reversed(amounts) for value in (0, amount))
        return amounts, torch_padding

    def _prepare_mask(
        self,
        mask: Tensor | None,
        batch: int,
        spatial_shape: tuple[int, ...],
        torch_padding: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Tensor | None, Tensor | None]:
        if mask is None:
            return None, None
        if mask.ndim == self.dimensions + 1:
            mask = mask.unsqueeze(1)
        expected = (batch, 1, *spatial_shape)
        if tuple(mask.shape) != expected:
            raise ValueError(f"mask has shape {tuple(mask.shape)}, expected {expected}")
        cell_mask = mask.to(device=device, dtype=dtype)
        padded = F.pad(cell_mask, torch_padding)
        pooling = F.max_pool2d if self.dimensions == 2 else F.max_pool3d
        patch_valid = pooling(
            padded, kernel_size=self.patch_size, stride=self.patch_size,
        ) > 0
        padding_mask = ~patch_valid.flatten(2).squeeze(1)
        if padding_mask.all(dim=1).any():
            raise ValueError("each sample must contain at least one valid patch")
        return cell_mask, padding_mask

    def _unpatchify(self, decoded: Tensor, grid_shape: tuple[int, ...]) -> Tensor:
        batch = decoded.shape[0]
        if self.dimensions == 2:
            grid_h, grid_w = grid_shape
            patch_h, patch_w = self.patch_size
            field = decoded.reshape(
                batch, grid_h, grid_w, self.output_channels, patch_h, patch_w,
            )
            return field.permute(0, 3, 1, 4, 2, 5).reshape(
                batch,
                self.output_channels,
                grid_h * patch_h,
                grid_w * patch_w,
            )
        grid_d, grid_h, grid_w = grid_shape
        patch_d, patch_h, patch_w = self.patch_size
        field = decoded.reshape(
            batch,
            grid_d,
            grid_h,
            grid_w,
            self.output_channels,
            patch_d,
            patch_h,
            patch_w,
        )
        return field.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            batch,
            self.output_channels,
            grid_d * patch_d,
            grid_h * patch_h,
            grid_w * patch_w,
        )

    def forward(
        self,
        spatial: Tensor,
        scalars: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        if spatial.ndim != self.dimensions + 2:
            raise ValueError(
                f"expected {self.dimensions + 2}-D input, got {spatial.ndim}-D"
            )
        if spatial.shape[1] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} channels, got {spatial.shape[1]}"
            )
        batch = spatial.shape[0]
        spatial_shape = tuple(int(item) for item in spatial.shape[2:])
        _, torch_padding = self._padding(spatial_shape)
        cell_mask, padding_mask = self._prepare_mask(
            mask,
            batch,
            spatial_shape,
            torch_padding,
            spatial.dtype,
            spatial.device,
        )
        if cell_mask is not None:
            spatial = spatial * cell_mask
        embedded = self.patch_embedding(F.pad(spatial, torch_padding))
        grid_shape = tuple(int(item) for item in embedded.shape[2:])
        tokens = embedded.flatten(2).transpose(1, 2)
        attention_bias = None
        if self.position_encoding == "normalized_linear_fourier_v1":
            encoding = patch_position_encoding(
                grid_shape, self.width, tokens.device, tokens.dtype,
            )
        else:
            encoding, attention_bias = fixed_tile_geometry(
                grid_shape, self.width, self.heads, batch, tokens.device,
                tokens.dtype,
                locality=self.position_encoding == "fixed_tile_fourier_locality_v1",
            )
        tokens = tokens + encoding.unsqueeze(0)
        if self.scalar_embedding is not None:
            if scalars is None or tuple(scalars.shape) != (batch, self.scalar_dim):
                raise ValueError(
                    f"scalars must have shape {(batch, self.scalar_dim)}, got "
                    f"{None if scalars is None else tuple(scalars.shape)}"
                )
            tokens = tokens + self.scalar_embedding(scalars).unsqueeze(1)
        elif scalars is not None and scalars.numel():
            raise ValueError("this model was constructed without scalar inputs")
        for block in self.blocks:
            tokens = block(tokens, padding_mask, attention_bias)
        decoded = self.patch_decoder(self.norm(tokens))
        output = self._unpatchify(decoded, grid_shape)
        slices = (slice(None), slice(None), *[slice(0, size) for size in spatial_shape])
        output = output[slices]
        if cell_mask is not None:
            output = output * cell_mask
        return output


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def fixed_tile_geometry(grid_shape, width, heads, batch, device, dtype, *, locality):
    """Positions and Gaussian attention bias in fixed patch units.

    Unlike per-domain normalization, adjacent patches stay one unit apart
    at every cover size. Half the heads retain unrestricted global attention;
    the other half have fixed support scales from one to four patch widths.
    No learned parameters or target-dependent coordinate rescaling are added.
    """
    coordinates = torch.stack(torch.meshgrid(*[
        torch.arange(n, device=device, dtype=torch.float32) + .5
        for n in grid_shape
    ], indexing="ij"), dim=-1).reshape(-1, len(grid_shape))
    pairs = max(1, width // (2 * len(grid_shape)))
    frequencies = torch.exp(torch.linspace(0., -9.210340372, pairs, device=device))
    angles = coordinates[..., None] * frequencies
    encoding = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(1)
    encoding = F.pad(encoding, (0, max(0, width - encoding.shape[1])))[:, :width]
    bias = None
    if locality:
        distance_squared = (coordinates[:, None] - coordinates[None, :]).square().sum(-1)
        local_heads = max(1, heads // 2)
        scales = torch.logspace(0., .6020599913, local_heads, device=device)
        bias = torch.zeros(heads, len(coordinates), len(coordinates), device=device)
        bias[:local_heads] = -distance_squared[None] / (2 * scales[:, None, None].square())
        bias = bias.repeat(batch, 1, 1).to(dtype=dtype)
    return encoding.to(dtype=dtype), bias
