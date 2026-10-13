"""Shifted Cartesian covers and field-aware placement policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


Tensor = torch.Tensor
PlacementMode = Literal["fixed", "random", "swept_geometry", "vorticity", "hybrid"]


@dataclass(frozen=True)
class CartesianCover:
    height: int
    width: int
    core: int
    halo: int
    offset_y: int = 0
    offset_x: int = 0

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0 or self.core <= 0 or self.halo < 0:
            raise ValueError("cover dimensions/core must be positive and halo nonnegative")
        if not 0 <= self.offset_y < self.core or not 0 <= self.offset_x < self.core:
            raise ValueError("cover offsets must lie in [0, core)")

    @property
    def grid_shape(self) -> tuple[int, int]:
        ny = (self.height + self.offset_y + self.core - 1) // self.core
        nx = (self.width + self.offset_x + self.core - 1) // self.core
        return ny, nx

    @property
    def tile_count(self) -> int:
        ny, nx = self.grid_shape
        return ny * nx

    @property
    def padded_core_shape(self) -> tuple[int, int]:
        ny, nx = self.grid_shape
        return ny * self.core, nx * self.core

    def padding(self) -> tuple[int, int, int, int]:
        padded_h, padded_w = self.padded_core_shape
        left = self.halo + self.offset_x
        right = self.halo + padded_w - self.width - self.offset_x
        top = self.halo + self.offset_y
        bottom = self.halo + padded_h - self.height - self.offset_y
        return left, right, top, bottom

    def tile(
        self, field: Tensor, padding_mode: Literal["constant", "replicate"] = "constant",
    ) -> Tensor:
        batch, channels, height, width = field.shape
        if (height, width) != (self.height, self.width):
            raise ValueError(f"field shape {(height, width)} != cover shape {(self.height, self.width)}")
        if padding_mode not in ("constant", "replicate"):
            raise ValueError(f"unsupported boundary padding mode: {padding_mode}")
        padded = F.pad(field, self.padding(), mode=padding_mode)
        size = self.core + 2 * self.halo
        patches = padded.unfold(2, size, self.core).unfold(3, size, self.core)
        ny, nx = self.grid_shape
        if patches.shape[2:4] != (ny, nx):
            raise RuntimeError("internal shifted-cover extraction mismatch")
        return patches.permute(0, 2, 3, 1, 4, 5).reshape(
            batch * ny * nx, channels, size, size
        )

    def tile_with_right_ghost(
        self,
        field: Tensor,
        right_ghost: Tensor,
        padding_mode: Literal["constant", "replicate"] = "constant",
    ) -> Tensor:
        """Tile a field after replacing its right padding with exterior state."""
        batch, channels, height, width = field.shape
        if (height, width) != (self.height, self.width):
            raise ValueError("field shape does not match the Cartesian cover")
        if (
            right_ghost.ndim != 4 or right_ghost.shape[:3] != (batch, channels, height)
        ):
            raise ValueError("right ghost must have shape B,C,H,G")
        if padding_mode not in ("constant", "replicate"):
            raise ValueError(f"unsupported boundary padding mode: {padding_mode}")
        left, right, top, bottom = self.padding()
        if right_ghost.shape[-1] < right:
            raise ValueError("right ghost is narrower than the shifted-cover padding")
        padded = F.pad(field, (left, right, top, bottom), mode=padding_mode)
        ghost = F.pad(
            right_ghost[..., :right], (0, 0, top, bottom), mode="replicate",
        )
        padded[..., left + width :] = ghost
        size = self.core + 2 * self.halo
        patches = padded.unfold(2, size, self.core).unfold(3, size, self.core)
        ny, nx = self.grid_shape
        return patches.permute(0, 2, 3, 1, 4, 5).reshape(
            batch * ny * nx, channels, size, size,
        )

    def assemble_ras(self, patches: Tensor, batch: int) -> Tensor:
        ny, nx = self.grid_shape
        channels = patches.shape[1]
        h = self.halo
        cores = patches[:, :, h : h + self.core, h : h + self.core]
        cores = cores.reshape(batch, ny, nx, channels, self.core, self.core)
        full = cores.permute(0, 3, 1, 4, 2, 5).reshape(
            batch, channels, ny * self.core, nx * self.core
        )
        return full[
            :, :, self.offset_y : self.offset_y + self.height,
            self.offset_x : self.offset_x + self.width,
        ]

    def _assemble_overlap(self, patches: Tensor, batch: int, confidence: Tensor) -> Tensor:
        ny, nx = self.grid_shape
        size = self.core + 2 * self.halo
        channels = patches.shape[1]
        weighted = patches * confidence
        columns = weighted.reshape(batch, ny * nx, channels * size * size).transpose(1, 2)
        output_shape = (ny * self.core + 2 * self.halo, nx * self.core + 2 * self.halo)
        accumulated = F.fold(
            columns, output_size=output_shape, kernel_size=size, stride=self.core
        )
        weight_columns = confidence.expand(batch * ny * nx, 1, size, size).reshape(
            batch, ny * nx, size * size
        ).transpose(1, 2)
        weights = F.fold(
            weight_columns, output_size=output_shape, kernel_size=size, stride=self.core
        )
        averaged = accumulated / weights.clamp_min(torch.finfo(patches.dtype).eps)
        y0 = self.halo + self.offset_y
        x0 = self.halo + self.offset_x
        return averaged[:, :, y0 : y0 + self.height, x0 : x0 + self.width]

    def assemble_average(self, patches: Tensor, batch: int) -> Tensor:
        size = self.core + 2 * self.halo
        confidence = torch.ones(1, 1, size, size, device=patches.device, dtype=patches.dtype)
        return self._assemble_overlap(patches, batch, confidence)

    def assemble_weighted(self, patches: Tensor, batch: int) -> Tensor:
        confidence = confidence_window(self.core, self.halo, patches.device, patches.dtype)
        return self._assemble_overlap(patches, batch, confidence)

    def extend_sum(self, patches: Tensor, batch: int) -> Tensor:
        """Apply the literal unweighted sum of patch extensions."""
        ny, nx = self.grid_shape
        size = self.core + 2 * self.halo
        if patches.ndim != 4 or patches.shape[0] != batch * ny * nx \
                or patches.shape[-2:] != (size, size):
            raise ValueError("patch tensor is incompatible with this cover")
        channels = patches.shape[1]
        columns = patches.reshape(
            batch, ny * nx, channels * size * size,
        ).transpose(1, 2)
        output_shape = (
            ny * self.core + 2 * self.halo,
            nx * self.core + 2 * self.halo,
        )
        accumulated = F.fold(
            columns, output_size=output_shape, kernel_size=size,
            stride=self.core,
        )
        y0 = self.halo + self.offset_y
        x0 = self.halo + self.offset_x
        return accumulated[
            :, :, y0:y0 + self.height, x0:x0 + self.width,
        ]

    def exchange_selected_weighted(
        self, patches: Tensor, selected: Tensor, batch: int,
    ) -> Tensor:
        """Exchange overlapping features among an active tile subset."""
        if selected.shape != (batch, self.tile_count):
            raise ValueError("selected tile mask has incompatible shape")
        active = selected.reshape(-1).bool()
        if patches.shape[0] != int(active.sum()):
            raise ValueError("active patch count does not match selected mask")
        size = self.core + 2 * self.halo
        full = patches.new_zeros(batch * self.tile_count, patches.shape[1], size, size)
        full[active] = patches
        confidence = confidence_window(
            self.core, self.halo, patches.device, patches.dtype,
        ).expand(batch * self.tile_count, -1, -1, -1)
        confidence = confidence * active[:, None, None, None]
        global_features = self._assemble_overlap(full, batch, confidence)
        return self.tile(global_features)[active]

    def valid_tiles(self, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        valid = torch.ones(batch, 1, self.height, self.width, device=device, dtype=dtype)
        return self.tile(valid)

    def centers(self, x0: float, y0: float, dx: float, device: torch.device) -> Tensor:
        ny, nx = self.grid_shape
        x = x0 + (torch.arange(nx, device=device) * self.core - self.offset_x + 0.5 * self.core) * dx
        y = y0 + (torch.arange(ny, device=device) * self.core - self.offset_y + 0.5 * self.core) * dx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx.flatten(), yy.flatten()), dim=1)


def confidence_window(
    core: int, halo: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    if halo == 0:
        return torch.ones(1, 1, core, core, device=device, dtype=dtype)
    ramp = 0.5 - 0.5 * torch.cos(
        torch.pi * torch.arange(1, halo + 1, device=device, dtype=dtype) / (halo + 1)
    )
    one_d = torch.cat((ramp, torch.ones(core, device=device, dtype=dtype), ramp.flip(0)))
    return (one_d[:, None] * one_d[None, :])[None, None]


def interface_mask(
    shape: tuple[int, int], core: int, offset_y: int, offset_x: int,
    width: int, device: torch.device,
) -> Tensor:
    mask = torch.zeros(shape, device=device, dtype=torch.bool)
    for boundary in range(core - offset_y, shape[0], core):
        mask[max(0, boundary - width) : min(shape[0], boundary + width)] = True
    for boundary in range(core - offset_x, shape[1], core):
        mask[:, max(0, boundary - width) : min(shape[1], boundary + width)] = True
    return mask


def vorticity_magnitude(q: Tensor, dx: float) -> Tensor:
    dv_dx = torch.gradient(q[:, 1], spacing=(dx,), dim=(-1,), edge_order=2)[0]
    du_dy = torch.gradient(q[:, 0], spacing=(dx,), dim=(-2,), edge_order=2)[0]
    return (dv_dx - du_dy).abs()


def choose_offset(
    mode: PlacementMode,
    q_n: Tensor,
    sdf_n: Tensor,
    sdf_np1: Tensor,
    core: int,
    halo: int,
    dx: float,
    seed: int,
    candidate_stride: int = 4,
) -> tuple[int, int, dict[str, float]]:
    if mode == "fixed":
        return 0, 0, {"cost": 0.0}
    generator = torch.Generator(device=q_n.device).manual_seed(seed)
    if mode == "random":
        oy = int(torch.randint(core, (1,), generator=generator, device=q_n.device))
        ox = int(torch.randint(core, (1,), generator=generator, device=q_n.device))
        return oy, ox, {"cost": float("nan")}
    if candidate_stride <= 0 or core % candidate_stride:
        raise ValueError("candidate stride must be a positive divisor of core")
    body = torch.maximum(
        torch.exp(-sdf_n[:, 0].abs() / (4 * dx)),
        torch.exp(-sdf_np1[:, 0].abs() / (4 * dx)),
    ).mean(dim=0)
    vort = vorticity_magnitude(q_n, dx).mean(dim=0)
    vort = vort / vort.quantile(0.99).clamp_min(1.0e-8)
    if mode == "swept_geometry":
        indicator = body
    elif mode == "vorticity":
        indicator = vort.clamp_max(2.0)
    elif mode == "hybrid":
        indicator = body + 0.5 * vort.clamp_max(2.0)
    else:
        raise ValueError(mode)
    best: tuple[float, int, int] | None = None
    for oy in range(0, core, candidate_stride):
        for ox in range(0, core, candidate_stride):
            mask = interface_mask(indicator.shape, core, oy, ox, halo, indicator.device)
            cost = float(indicator[mask].mean()) if mask.any() else 0.0
            candidate = (cost, oy, ox)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[1], best[2], {"cost": best[0]}
