"""Interface-conditioned next-state local solver for Airfoil SNI."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cartesian_cover import CartesianCover


Tensor = torch.Tensor


def tile_face_mask(
    size: int,
    width: int = 1,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if size <= 0 or width <= 0 or 2 * width > size:
        raise ValueError("face width must be positive and no larger than half the tile")
    mask = torch.zeros(1, 1, size, size, device=device, dtype=dtype)
    mask[..., :width, :] = 1
    mask[..., -width:, :] = 1
    mask[..., :, :width] = 1
    mask[..., :, -width:] = 1
    return mask


def _groups(width: int) -> int:
    groups = min(8, width)
    while width % groups:
        groups -= 1
    return groups


class _FiLMBlock2D(nn.Module):
    def __init__(self, width: int, scalar_dim: int, dilation: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(width), width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(_groups(width), width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.condition = nn.Linear(scalar_dim, 2 * width) if scalar_dim else None

    def forward(self, state: Tensor, scalar: Tensor) -> Tensor:
        hidden = self.norm1(state)
        if self.condition is not None:
            scale, shift = self.condition(scalar).chunk(2, 1)
            hidden = hidden * (1 + scale[..., None, None]) + shift[..., None, None]
        hidden = self.conv1(F.silu(hidden))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return state + hidden


class BoundaryConditionedTileSolver2D(nn.Module):
    """Complete local next-state map with exact artificial Dirichlet traces."""

    def __init__(
        self,
        *,
        state_channels: int = 3,
        scalar_dim: int = 4,
        width: int = 48,
        depth: int = 5,
    ) -> None:
        super().__init__()
        if min(state_channels, width, depth) <= 0 or scalar_dim < 0:
            raise ValueError("model dimensions are invalid")
        self.state_channels = state_channels
        self.scalar_dim = scalar_dim
        # base next state, q_n, sdf_np1, artificial trace, mask
        input_channels = 3 * state_channels + 2
        self.stem = nn.Conv2d(input_channels, width, 5, padding=2)
        cycle = (1, 2, 4, 1, 2, 4)
        self.blocks = nn.ModuleList(
            _FiLMBlock2D(width, scalar_dim, cycle[index % len(cycle)])
            for index in range(depth)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(width), width), nn.SiLU(),
            nn.Conv2d(width, state_channels, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        base: Tensor,
        current_state: Tensor,
        sdf_np1: Tensor,
        scalar: Tensor,
        boundary_state: Tensor,
        boundary_mask: Tensor,
    ) -> Tensor:
        if base.ndim != 4 or current_state.shape != base.shape:
            raise ValueError("base/current state must share BCHW shape")
        if boundary_state.shape != base.shape:
            raise ValueError("boundary state must match base")
        if sdf_np1.shape != base[:, :1].shape or boundary_mask.shape != sdf_np1.shape:
            raise ValueError("SDF and boundary mask must have shape B1HW")
        if scalar.shape != (base.shape[0], self.scalar_dim):
            raise ValueError("scalar shape does not match solver")
        mask = boundary_mask.to(dtype=base.dtype)
        trace = boundary_state * mask
        hidden = self.stem(torch.cat(
            (base, current_state, sdf_np1, trace, mask), dim=1,
        ))
        for block in self.blocks:
            hidden = block(hidden, scalar)
        candidate = base + self.output(hidden)
        return candidate * (1.0 - mask) + boundary_state * mask


@dataclass(frozen=True)
class AirfoilSNISweep:
    field: Tensor
    local_candidates: Tensor
    correction_norm: float
    interface_norm: float


@torch.no_grad()
def airfoil_sni_sweep(
    solver: BoundaryConditionedTileSolver2D,
    base_tiles: Tensor,
    state_tiles: Tensor,
    sdf_tiles: Tensor,
    scalars: Tensor,
    current: Tensor,
    cover: CartesianCover,
    *,
    tau: float,
    assembly: str = "huang_raw_additive",
    chunk_tiles: int = 32,
    face_width: int = 1,
) -> AirfoilSNISweep:
    if not 0 < tau <= 1.5:
        raise ValueError("relaxation must lie in (0,1.5]")
    if assembly not in ("huang_raw_additive", "normalized_pou"):
        raise ValueError("unknown SNI assembly")
    if current.ndim != 4 or current.shape[0] != 1:
        raise ValueError("current must have shape 1,C,H,W")
    current_tiles = cover.tile(current, padding_mode="replicate")
    if base_tiles.shape != current_tiles.shape or state_tiles.shape != current_tiles.shape:
        raise ValueError("base/state tiles are incompatible with the cover")
    if sdf_tiles.shape != current_tiles[:, :1].shape:
        raise ValueError("SDF tiles are incompatible with the cover")
    device = next(solver.parameters()).device
    face = tile_face_mask(
        base_tiles.shape[-1], face_width, device=current.device,
        dtype=current.dtype,
    ).expand(base_tiles.shape[0], -1, -1, -1)
    candidates = []
    for start in range(0, base_tiles.shape[0], chunk_tiles):
        stop = min(start + chunk_tiles, base_tiles.shape[0])
        candidates.append(solver(
            base_tiles[start:stop].to(device),
            state_tiles[start:stop].to(device),
            sdf_tiles[start:stop].to(device),
            scalars[start:stop].to(device),
            current_tiles[start:stop].to(device),
            face[start:stop].to(device),
        ).float().to(current.device))
    local = torch.cat(candidates)
    correction = local - current_tiles
    if assembly == "huang_raw_additive":
        next_field = current + tau * cover.extend_sum(correction, batch=1)
    else:
        proposal = cover.assemble_weighted(local, batch=1)
        next_field = current + tau * (proposal - current)
    next_tiles = cover.tile(next_field, padding_mode="replicate")
    denominator = (torch.tensor(next_field.numel(), device=current.device,
                                dtype=current.dtype)).sqrt()
    return AirfoilSNISweep(
        field=next_field,
        local_candidates=local,
        correction_norm=float((next_field - current).square().sum().sqrt() / denominator),
        interface_norm=float((local - next_tiles).square().sum().sqrt()
                             / torch.tensor(local.numel(), device=current.device,
                                            dtype=current.dtype).sqrt()),
    )
