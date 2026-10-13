"""Shifted Cartesian covers with overlap-aware assembly.

The cover owns only index geometry. It deliberately knows nothing about the
PDE, boundary condition, field semantics, or neural network.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def confidence_window(
    core: int, halo: int, device: torch.device, dtype: torch.dtype,
) -> Tensor:
    """Cosine taper used as a partition of unity in duplicate overlap cells."""
    if halo == 0:
        return torch.ones(1, 1, core, core, device=device, dtype=dtype)
    ramp = 0.5 - 0.5 * torch.cos(
        torch.pi * torch.arange(1, halo + 1, device=device, dtype=dtype)
        / (halo + 1)
    )
    one_d = torch.cat((ramp, torch.ones(core, device=device, dtype=dtype), ramp.flip(0)))
    return (one_d[:, None] * one_d[None, :])[None, None]


@dataclass(frozen=True)
class CartesianCover:
    """A complete shifted cover of a rectangular cell-centered field."""

    height: int
    width: int
    core: int
    halo: int
    offset_y: int = 0
    offset_x: int = 0

    def __post_init__(self) -> None:
        if min(self.height, self.width, self.core) <= 0 or self.halo < 0:
            raise ValueError("field/core sizes must be positive and halo nonnegative")
        if not 0 <= self.offset_y < self.core or not 0 <= self.offset_x < self.core:
            raise ValueError("offsets must lie in [0, core)")

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (
            (self.height + self.offset_y + self.core - 1) // self.core,
            (self.width + self.offset_x + self.core - 1) // self.core,
        )

    @property
    def tile_count(self) -> int:
        ny, nx = self.grid_shape
        return ny * nx

    @property
    def tile_size(self) -> int:
        return self.core + 2 * self.halo

    def _padding(self) -> tuple[int, int, int, int]:
        ny, nx = self.grid_shape
        return (
            self.halo + self.offset_x,
            self.halo + nx * self.core - self.width - self.offset_x,
            self.halo + self.offset_y,
            self.halo + ny * self.core - self.height - self.offset_y,
        )

    def tile(self, field: Tensor) -> Tensor:
        """Extract all halo-extended patches in batch-major, row-major order."""
        if field.ndim != 4 or tuple(field.shape[-2:]) != (self.height, self.width):
            raise ValueError("field must have shape [B,C,height,width]")
        padded = F.pad(field, self._padding())
        patches = padded.unfold(2, self.tile_size, self.core).unfold(
            3, self.tile_size, self.core
        )
        batch, channels = field.shape[:2]
        ny, nx = self.grid_shape
        return patches.permute(0, 2, 3, 1, 4, 5).reshape(
            batch * ny * nx, channels, self.tile_size, self.tile_size
        )

    def valid_tiles(
        self, batch: int, device: torch.device, dtype: torch.dtype,
    ) -> Tensor:
        valid = torch.ones(batch, 1, self.height, self.width, device=device, dtype=dtype)
        return self.tile(valid)

    def assemble_ras(self, patches: Tensor, batch: int) -> Tensor:
        """Restricted additive Schwarz writeback: only owned cores are retained."""
        ny, nx = self.grid_shape
        h, c, channels = self.halo, self.core, patches.shape[1]
        cores = patches[:, :, h:h + c, h:h + c].reshape(
            batch, ny, nx, channels, c, c
        )
        field = cores.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, channels, ny * c, nx * c
        )
        return field[
            :, :, self.offset_y:self.offset_y + self.height,
            self.offset_x:self.offset_x + self.width,
        ]

    def assemble_weighted(self, patches: Tensor, batch: int) -> Tensor:
        """Least-squares consensus of every duplicate prediction in the overlap."""
        ny, nx = self.grid_shape
        size, channels = self.tile_size, patches.shape[1]
        confidence = confidence_window(
            self.core, self.halo, patches.device, patches.dtype
        )
        columns = (patches * confidence).reshape(
            batch, ny * nx, channels * size * size
        ).transpose(1, 2)
        output_shape = (ny * self.core + 2 * self.halo, nx * self.core + 2 * self.halo)
        accumulated = F.fold(
            columns, output_size=output_shape, kernel_size=size, stride=self.core
        )
        weight_columns = confidence.expand(batch * ny * nx, 1, size, size).reshape(
            batch, ny * nx, size * size
        ).transpose(1, 2)
        weights = F.fold(
            weight_columns, output_size=output_shape,
            kernel_size=size, stride=self.core,
        )
        averaged = accumulated / weights.clamp_min(torch.finfo(patches.dtype).eps)
        y0, x0 = self.halo + self.offset_y, self.halo + self.offset_x
        return averaged[:, :, y0:y0 + self.height, x0:x0 + self.width]

    def assemble_uniform(
        self, patches: Tensor, batch: int, weights: Tensor | None = None,
    ) -> Tensor:
        """Patch by overlap counting, matching the released SNI inference map.

        ``weights`` is normally the tiled physical-domain mask.  It prevents
        an inactive part of a cut-cell subdomain from contributing a spurious
        zero to the overlap average.
        """
        ny, nx = self.grid_shape
        size, channels = self.tile_size, patches.shape[1]
        if weights is None:
            weights = torch.ones_like(patches[:, :1])
        if weights.shape != (patches.shape[0], 1, size, size):
            raise ValueError(
                "uniform assembly weights must have shape "
                f"{(patches.shape[0], 1, size, size)}, got {tuple(weights.shape)}"
            )
        output_shape = (
            ny * self.core + 2 * self.halo,
            nx * self.core + 2 * self.halo,
        )
        columns = (patches * weights).reshape(
            batch, ny * nx, channels * size * size,
        ).transpose(1, 2)
        accumulated = F.fold(
            columns, output_size=output_shape, kernel_size=size, stride=self.core,
        )
        weight_columns = weights.reshape(
            batch, ny * nx, size * size,
        ).transpose(1, 2)
        counts = F.fold(
            weight_columns, output_size=output_shape,
            kernel_size=size, stride=self.core,
        )
        averaged = accumulated / counts.clamp_min(torch.finfo(patches.dtype).eps)
        y0, x0 = self.halo + self.offset_y, self.halo + self.offset_x
        return averaged[:, :, y0:y0 + self.height, x0:x0 + self.width]

    def assemble_extension_sum(
        self, patches: Tensor, batch: int, weights: Tensor | None = None,
    ) -> Tensor:
        """Raw zero-extension sum for the paper additive-Schwarz equation.

        No confidence taper and no overlap normalization are applied.  With
        ``weights`` equal to tiled active-domain indicators this is exactly
        ``sum_i R_i^T v_i`` on the represented Cartesian cover.
        """
        ny, nx = self.grid_shape
        size, channels = self.tile_size, patches.shape[1]
        if weights is None:
            weights = torch.ones_like(patches[:, :1])
        if weights.shape != (patches.shape[0], 1, size, size):
            raise ValueError(
                "extension-sum weights must have shape "
                f"{(patches.shape[0], 1, size, size)}, got {tuple(weights.shape)}"
            )
        columns = (patches * weights).reshape(
            batch, ny * nx, channels * size * size,
        ).transpose(1, 2)
        output_shape = (
            ny * self.core + 2 * self.halo,
            nx * self.core + 2 * self.halo,
        )
        accumulated = F.fold(
            columns, output_size=output_shape, kernel_size=size, stride=self.core,
        )
        y0, x0 = self.halo + self.offset_y, self.halo + self.offset_x
        return accumulated[:, :, y0:y0 + self.height, x0:x0 + self.width]

    def assemble_additive(self, patches: Tensor, batch: int) -> Tensor:
        """Unnormalized overlap sum used by additive Schwarz/SNI baselines.

        Unlike :meth:`assemble_weighted`, this deliberately does not divide by
        overlap multiplicity.  A relaxation parameter must therefore control
        the summed local correction.  Keeping the two assemblers separate
        makes the learned one-step POU method and iterative additive baseline
        impossible to conflate in timing or accuracy tables.
        """
        ny, nx = self.grid_shape
        size, channels = self.tile_size, patches.shape[1]
        confidence = confidence_window(
            self.core, self.halo, patches.device, patches.dtype,
        )
        columns = (patches * confidence).reshape(
            batch, ny * nx, channels * size * size,
        ).transpose(1, 2)
        output_shape = (
            ny * self.core + 2 * self.halo,
            nx * self.core + 2 * self.halo,
        )
        accumulated = F.fold(
            columns, output_size=output_shape, kernel_size=size, stride=self.core,
        )
        y0, x0 = self.halo + self.offset_y, self.halo + self.offset_x
        return accumulated[:, :, y0:y0 + self.height, x0:x0 + self.width]

    def random_shift(self, seed: int) -> "CartesianCover":
        generator = torch.Generator().manual_seed(seed)
        return CartesianCover(
            self.height, self.width, self.core, self.halo,
            int(torch.randint(self.core, (), generator=generator)),
            int(torch.randint(self.core, (), generator=generator)),
        )
