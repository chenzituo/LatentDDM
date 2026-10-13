"""Resolution-agnostic full-field baselines for Cartesian PDE states."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


def _group_count(channels: int, maximum: int = 8) -> int:
    """Return the largest conventional GroupNorm count dividing channels."""
    return math.gcd(channels, maximum)


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = _group_count(output_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    def forward(self, field: Tensor) -> Tensor:
        return self.layers(field)


class FullFieldUNet(nn.Module):
    """Non-stitched U-Net mapping a complete Cartesian state to its increment."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 3,
        width: int = 42,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        channels = [width * 2**level for level in range(depth)]
        self.input_channels = input_channels
        self.encoders = nn.ModuleList()
        previous = input_channels
        for channels_at_level in channels:
            self.encoders.append(DoubleConv(previous, channels_at_level))
            previous = channels_at_level
        self.bottleneck = DoubleConv(channels[-1], 2 * channels[-1])
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        previous = 2 * channels[-1]
        for channels_at_level in reversed(channels):
            self.upsamples.append(nn.ConvTranspose2d(
                previous, channels_at_level, kernel_size=2, stride=2,
            ))
            self.decoders.append(DoubleConv(
                2 * channels_at_level, channels_at_level,
            ))
            previous = channels_at_level
        self.output = nn.Conv2d(width, output_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"expected BCHW input with {self.input_channels} channels"
            )
        skips = []
        hidden = inputs
        for encoder in self.encoders:
            hidden = encoder(hidden)
            skips.append(hidden)
            hidden = F.avg_pool2d(hidden, kernel_size=2)
        hidden = self.bottleneck(hidden)
        for upsample, decoder, skip in zip(
            self.upsamples, self.decoders, reversed(skips), strict=True,
        ):
            hidden = upsample(hidden)
            if hidden.shape[-2:] != skip.shape[-2:]:
                hidden = F.interpolate(
                    hidden, size=skip.shape[-2:], mode="bilinear",
                    align_corners=False,
                )
            hidden = decoder(torch.cat((hidden, skip), dim=1))
        return self.output(hidden)


class SpectralConv2d(nn.Module):
    """Truncated 2D Fourier convolution with real-counted complex weights."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        modes_y: int,
        modes_x: int,
    ) -> None:
        super().__init__()
        if min(modes_y, modes_x) <= 0:
            raise ValueError("Fourier mode counts must be positive")
        self.output_channels = output_channels
        self.modes_y = modes_y
        self.modes_x = modes_x
        scale = 1.0 / math.sqrt(input_channels * output_channels)
        shape = (input_channels, output_channels, modes_y, modes_x, 2)
        self.positive_weights = nn.Parameter(scale * torch.randn(*shape))
        self.negative_weights = nn.Parameter(scale * torch.randn(*shape))

    @staticmethod
    def _multiply(coefficients: Tensor, weights: Tensor) -> Tensor:
        complex_weights = torch.view_as_complex(weights.contiguous())
        return torch.einsum("bixy,ioxy->boxy", coefficients, complex_weights)

    def forward(self, field: Tensor) -> Tensor:
        batch, _, height, width = field.shape
        coefficients = torch.fft.rfft2(field.float(), norm="ortho")
        modes_y = min(self.modes_y, height // 2)
        modes_x = min(self.modes_x, width // 2 + 1)
        output = coefficients.new_zeros(
            batch, self.output_channels, height, width // 2 + 1,
        )
        output[:, :, :modes_y, :modes_x] = self._multiply(
            coefficients[:, :, :modes_y, :modes_x],
            self.positive_weights[:, :, :modes_y, :modes_x],
        )
        output[:, :, -modes_y:, :modes_x] = self._multiply(
            coefficients[:, :, -modes_y:, :modes_x],
            self.negative_weights[:, :, :modes_y, :modes_x],
        )
        return torch.fft.irfft2(output, s=(height, width), norm="ortho")


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes_y: int, modes_x: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(width), width)
        self.spectral = SpectralConv2d(width, width, modes_y, modes_x)
        self.local = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, field: Tensor) -> Tensor:
        normalized = self.norm(field)
        update = self.spectral(normalized) + self.local(normalized)
        return field + F.gelu(update)


class FullFieldFNO(nn.Module):
    """Non-stitched Fourier neural operator for complete Cartesian fields."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 3,
        width: int = 48,
        depth: int = 6,
        modes_y: int = 15,
        modes_x: int = 16,
        padding: int = 8,
    ) -> None:
        super().__init__()
        if depth < 1 or padding < 0:
            raise ValueError("depth must be positive and padding nonnegative")
        self.input_channels = input_channels
        self.padding = padding
        self.lift = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
        )
        self.blocks = nn.ModuleList([
            FNOBlock2d(width, modes_y, modes_x) for _ in range(depth)
        ])
        self.norm = nn.GroupNorm(_group_count(width), width)
        self.project = nn.Sequential(
            nn.Conv2d(width, 2 * width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * width, output_channels, kernel_size=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"expected BCHW input with {self.input_channels} channels"
            )
        hidden = self.lift(inputs)
        if self.padding:
            hidden = F.pad(hidden, (0, self.padding, 0, self.padding))
        for block in self.blocks:
            hidden = block(hidden)
        if self.padding:
            hidden = hidden[..., :-self.padding, :-self.padding]
        return self.project(self.norm(hidden))
