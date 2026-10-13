"""Boundary-state Schwarz iteration for the Airfoil benchmark.

Each subdomain keeps its frozen local prediction.  A sweep restricts the
current global state to every halo, measures its directional face difference
from that local prediction, solves all local boundary-value problems in one
batch, and writes each subdomain core once.  The correction is centered so a
zero boundary difference reproduces the frozen local prediction exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .airfoil_flux_conditioned_sni import (
    directional_conservative_flux_2d,
    directional_face_roles_2d,
    flux_scales_2d,
)
from .airfoil_sni import _FiLMBlock2D, _groups
from .cartesian_cover import CartesianCover


Tensor = torch.Tensor


def assemble_subdomains_2d(cover, patches: Tensor, batch: int, assembly: str) -> Tensor:
    if assembly == "normalized_overlap":
        return cover.assemble_weighted(patches, batch)
    if assembly == "disjoint_core":
        return cover.assemble_ras(patches, batch)
    raise ValueError(f"unsupported Airfoil assembly: {assembly!r}")
FACE_NORMALS_2D = ((0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0))


def neighboring_face_mask_2d(
    grid_shape: tuple[int, int],
    *,
    batch: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Mark tile faces with a neighbor inside the current cover."""
    ny, nx = grid_shape
    if min(ny, nx, batch) <= 0:
        raise ValueError("grid dimensions and batch must be positive")
    mask = torch.zeros(
        batch, ny, nx, 4, 1, 1, device=device, dtype=dtype,
    )
    mask[:, 1:, :, 0] = 1.0
    mask[:, :-1, :, 1] = 1.0
    mask[:, :, 1:, 2] = 1.0
    mask[:, :, :-1, 3] = 1.0
    return mask.reshape(batch * ny * nx, 4, 1, 1)


def directional_state_trace_2d(
    current_tiles: Tensor,
    base_tiles: Tensor,
    *,
    state_mean: Tensor,
    state_std: Tensor,
    face_width: int = 1,
    incoming_temperature: float = 0.1,
    neighbor_faces: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return directional state differences and flow-aware face weights.

    States are in the endpoint's normalized coordinates.  The returned trace
    has four ``(u, v, p)`` blocks.  ``incoming`` is a smooth indicator based on
    the sign of the physical normal velocity; it is supplied as context rather
    than used to discard the outflow pressure trace.
    """
    if (
        current_tiles.ndim != 4
        or current_tiles.shape != base_tiles.shape
        or current_tiles.shape[1] != 3
        or incoming_temperature <= 0.0
    ):
        raise ValueError("Airfoil traces require matching B,3,H,W states")
    mean = torch.as_tensor(
        state_mean, device=current_tiles.device, dtype=current_tiles.dtype,
    ).reshape(1, 3, 1, 1)
    std = torch.as_tensor(
        state_std, device=current_tiles.device, dtype=current_tiles.dtype,
    ).reshape(1, 3, 1, 1)
    if bool(torch.any(std <= 0)):
        raise ValueError("state standard deviations must be positive")
    roles = directional_face_roles_2d(
        current_tiles.shape[-2], current_tiles.shape[-1],
        face_width=face_width, device=current_tiles.device,
        dtype=current_tiles.dtype,
    ).expand(current_tiles.shape[0], -1, -1, -1)
    if neighbor_faces is not None:
        if neighbor_faces.shape != (current_tiles.shape[0], 4, 1, 1):
            raise ValueError("neighbor face mask must have shape B,4,1,1")
        roles = roles * neighbor_faces
    physical = current_tiles * std + mean
    velocity_scale = std[:, :2].amax().clamp_min(1.0e-6)
    blocks = []
    incoming = []
    difference = current_tiles - base_tiles
    for face, (normal_x, normal_y) in enumerate(FACE_NORMALS_2D):
        role = roles[:, face : face + 1]
        normal_velocity = (
            normal_x * physical[:, 0:1] + normal_y * physical[:, 1:2]
        )
        incoming.append(
            torch.sigmoid(
                -normal_velocity / (incoming_temperature * velocity_scale)
            ) * role
        )
        blocks.append(difference * role)
    return torch.cat(blocks, dim=1), roles, torch.cat(incoming, dim=1)


def transport_characteristic_trace_2d(
    trace: Tensor,
    roles: Tensor,
    incoming: Tensor,
) -> Tensor:
    """Retain incoming velocity traces and two-sided pressure traces."""
    if trace.ndim != 4 or trace.shape[1] != 12:
        raise ValueError("Airfoil state trace must have four three-channel faces")
    if roles.shape != incoming.shape or roles.shape != (
        trace.shape[0], 4, *trace.shape[-2:],
    ):
        raise ValueError("Airfoil face weights do not align with the trace")
    faces = trace.reshape(trace.shape[0], 4, 3, *trace.shape[-2:])
    velocity = faces[:, :, :2] * incoming[:, :, None]
    pressure = faces[:, :, 2:] * roles[:, :, None]
    return torch.cat((velocity, pressure), dim=2).flatten(1, 2)


class TraceConditionedTileSolver2D(nn.Module):
    """Local correction conditioned on directional neighboring state traces."""

    def __init__(
        self,
        *,
        state_channels: int = 3,
        scalar_dim: int = 4,
        width: int = 64,
        depth: int = 6,
    ) -> None:
        super().__init__()
        if state_channels != 3 or min(width, depth) <= 0 or scalar_dim < 0:
            raise ValueError("invalid Airfoil trace solver dimensions")
        self.state_channels = state_channels
        self.scalar_dim = scalar_dim
        # local prediction, previous-time state, SDF, four state traces,
        # directional face roles, and four incoming-flow indicators.
        channels = 2 * state_channels + 1 + 12 + 4 + 4
        self.stem = nn.Conv2d(channels, width, 5, padding=2)
        cycle = (1, 2, 4, 1, 2, 4)
        self.blocks = nn.ModuleList(
            _FiLMBlock2D(width, scalar_dim, cycle[index % len(cycle)])
            for index in range(depth)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(),
            nn.Conv2d(width, state_channels, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        base: Tensor,
        previous_state: Tensor,
        sdf_np1: Tensor,
        scalar: Tensor,
        boundary_trace: Tensor,
        face_roles: Tensor,
        incoming: Tensor,
    ) -> Tensor:
        if base.ndim != 4 or previous_state.shape != base.shape:
            raise ValueError("base and previous state must share B,C,H,W")
        if sdf_np1.shape != base[:, :1].shape:
            raise ValueError("SDF must have shape B,1,H,W")
        if boundary_trace.shape != (base.shape[0], 12, *base.shape[-2:]):
            raise ValueError("boundary trace must have shape B,12,H,W")
        if face_roles.shape != incoming.shape \
                or face_roles.shape != (base.shape[0], 4, *base.shape[-2:]):
            raise ValueError("face roles and incoming weights must be B,4,H,W")
        if scalar.shape != (base.shape[0], self.scalar_dim):
            raise ValueError("scalar conditioning shape does not match")
        common = (base, previous_state, sdf_np1)
        conditioned = torch.cat(
            (*common, boundary_trace, face_roles, incoming), dim=1,
        )
        zero = torch.cat(
            (
                *common,
                torch.zeros_like(boundary_trace),
                face_roles,
                incoming,
            ),
            dim=1,
        )
        hidden = self.stem(torch.cat((conditioned, zero), dim=0))
        doubled_scalar = torch.cat((scalar, scalar), dim=0)
        for block in self.blocks:
            hidden = block(hidden, doubled_scalar)
        response, zero_response = self.output(hidden).chunk(2, dim=0)
        return base + response - zero_response


def directional_state_flux_trace_2d(
    current_tiles: Tensor,
    base_tiles: Tensor,
    *,
    state_mean: Tensor,
    state_std: Tensor,
    dx: float,
    viscosity: Tensor | float,
    face_width: int = 1,
    incoming_temperature: float = 0.1,
    neighbor_faces: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return state and conservative-flux differences on each active face."""
    state_trace, roles, incoming = directional_state_trace_2d(
        current_tiles,
        base_tiles,
        state_mean=state_mean,
        state_std=state_std,
        face_width=face_width,
        incoming_temperature=incoming_temperature,
        neighbor_faces=neighbor_faces,
    )
    mean = torch.as_tensor(
        state_mean, device=current_tiles.device, dtype=current_tiles.dtype,
    ).reshape(1, 3, 1, 1)
    std = torch.as_tensor(
        state_std, device=current_tiles.device, dtype=current_tiles.dtype,
    ).reshape(1, 3, 1, 1)
    current_flux, _ = directional_conservative_flux_2d(
        current_tiles * std + mean,
        dx=dx,
        viscosity=viscosity,
        face_width=face_width,
    )
    base_flux, _ = directional_conservative_flux_2d(
        base_tiles * std + mean,
        dx=dx,
        viscosity=viscosity,
        face_width=face_width,
    )
    active = roles.repeat_interleave(3, dim=1)
    scale = flux_scales_2d(mean.flatten(), std.flatten()).to(
        current_tiles,
    ).reshape(1, 12, 1, 1)
    flux_trace = (current_flux - base_flux) / scale * active
    return state_trace, flux_trace, roles, incoming


class RobinConditionedTileSolver2D(nn.Module):
    """Local correction using neighboring state and conservative flux."""

    def __init__(
        self,
        *,
        state_channels: int = 3,
        scalar_dim: int = 4,
        width: int = 64,
        depth: int = 6,
    ) -> None:
        super().__init__()
        if state_channels != 3 or min(width, depth) <= 0 or scalar_dim < 0:
            raise ValueError("invalid Airfoil Robin solver dimensions")
        self.state_channels = state_channels
        self.scalar_dim = scalar_dim
        channels = 2 * state_channels + 1 + 12 + 12 + 4 + 4
        self.stem = nn.Conv2d(channels, width, 5, padding=2)
        cycle = (1, 2, 4, 1, 2, 4)
        self.blocks = nn.ModuleList(
            _FiLMBlock2D(width, scalar_dim, cycle[index % len(cycle)])
            for index in range(depth)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(),
            nn.Conv2d(width, state_channels, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        base: Tensor,
        previous_state: Tensor,
        sdf_np1: Tensor,
        scalar: Tensor,
        boundary_trace: Tensor,
        boundary_flux: Tensor,
        face_roles: Tensor,
        incoming: Tensor,
    ) -> Tensor:
        if base.ndim != 4 or previous_state.shape != base.shape:
            raise ValueError("base and previous state must share B,C,H,W")
        if sdf_np1.shape != base[:, :1].shape:
            raise ValueError("SDF must have shape B,1,H,W")
        if boundary_trace.shape != (base.shape[0], 12, *base.shape[-2:]):
            raise ValueError("boundary trace must have shape B,12,H,W")
        if boundary_flux.shape != boundary_trace.shape:
            raise ValueError("boundary flux must have shape B,12,H,W")
        expected_faces = (base.shape[0], 4, *base.shape[-2:])
        if face_roles.shape != expected_faces or incoming.shape != expected_faces:
            raise ValueError("face roles and incoming weights must be B,4,H,W")
        if scalar.shape != (base.shape[0], self.scalar_dim):
            raise ValueError("scalar conditioning shape does not match")
        common = (base, previous_state, sdf_np1)
        conditioned = torch.cat(
            (*common, boundary_trace, boundary_flux, face_roles, incoming),
            dim=1,
        )
        zero = torch.cat(
            (
                *common,
                torch.zeros_like(boundary_trace),
                torch.zeros_like(boundary_flux),
                face_roles,
                incoming,
            ),
            dim=1,
        )
        hidden = self.stem(torch.cat((conditioned, zero), dim=0))
        doubled_scalar = torch.cat((scalar, scalar), dim=0)
        for block in self.blocks:
            hidden = block(hidden, doubled_scalar)
        response, zero_response = self.output(hidden).chunk(2, dim=0)
        return base + response - zero_response


@dataclass(frozen=True)
class TraceSNISweep2D:
    field: Tensor
    local_candidates: Tensor
    trace: Tensor
    incoming: Tensor


@dataclass(frozen=True)
class RobinSNISweep2D:
    field: Tensor
    local_candidates: Tensor
    trace: Tensor
    flux: Tensor
    incoming: Tensor


def trace_sni_sweep_2d(
    solver: TraceConditionedTileSolver2D,
    base_tiles: Tensor,
    previous_tiles: Tensor,
    sdf_tiles: Tensor,
    scalars: Tensor,
    current: Tensor,
    cover: CartesianCover,
    *,
    state_mean: Tensor,
    state_std: Tensor,
    tau: float = 0.5,
    face_width: int = 1,
    incoming_temperature: float = 0.1,
    chunk_tiles: int | None = None,
    neighbor_faces: Tensor | None = None,
    assembly: str = "disjoint_core",
    transmission: str = "all_state",
) -> TraceSNISweep2D:
    """Perform one differentiable Schwarz sweep."""
    if not 0.0 < tau <= 1.5:
        raise ValueError("relaxation must lie in (0, 1.5]")
    current_tiles = cover.tile(current, padding_mode="replicate")
    if (
        base_tiles.shape != current_tiles.shape
        or previous_tiles.shape != current_tiles.shape
        or sdf_tiles.shape != current_tiles[:, :1].shape
        or scalars.shape[0] != current_tiles.shape[0]
    ):
        raise ValueError("local tensors are incompatible with the cover")
    if neighbor_faces is None:
        neighbor_faces = neighboring_face_mask_2d(
            cover.grid_shape,
            batch=current.shape[0],
            device=current.device,
            dtype=current.dtype,
        )
    trace, roles, incoming = directional_state_trace_2d(
        current_tiles,
        base_tiles,
        state_mean=state_mean,
        state_std=state_std,
        face_width=face_width,
        incoming_temperature=incoming_temperature,
        neighbor_faces=neighbor_faces,
    )
    if transmission == "transport_characteristic":
        trace = transport_characteristic_trace_2d(trace, roles, incoming)
    elif transmission != "all_state":
        raise ValueError("unknown Airfoil state transmission")
    count = base_tiles.shape[0]
    chunk = count if chunk_tiles is None else int(chunk_tiles)
    if chunk <= 0:
        raise ValueError("chunk size must be positive")
    candidates = []
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        candidates.append(
            solver(
                base_tiles[start:stop],
                previous_tiles[start:stop],
                sdf_tiles[start:stop],
                scalars[start:stop],
                trace[start:stop],
                roles[start:stop],
                incoming[start:stop],
            )
        )
    local = torch.cat(candidates, dim=0)
    proposal = assemble_subdomains_2d(
        cover, local, current.shape[0], assembly,
    )
    field = current + tau * (proposal - current)
    return TraceSNISweep2D(
        field=field,
        local_candidates=local,
        trace=trace,
        incoming=incoming,
    )


def robin_sni_sweep_2d(
    solver: RobinConditionedTileSolver2D,
    base_tiles: Tensor,
    previous_tiles: Tensor,
    sdf_tiles: Tensor,
    scalars: Tensor,
    current: Tensor,
    cover: CartesianCover,
    *,
    state_mean: Tensor,
    state_std: Tensor,
    dx: float,
    viscosity: Tensor | float,
    tau: float = 0.5,
    face_width: int = 1,
    incoming_temperature: float = 0.1,
    chunk_tiles: int | None = None,
    neighbor_faces: Tensor | None = None,
    assembly: str = "disjoint_core",
) -> RobinSNISweep2D:
    """Perform one differentiable state--flux SNI sweep."""
    if not 0.0 < tau <= 1.5:
        raise ValueError("relaxation must lie in (0, 1.5]")
    current_tiles = cover.tile(current, padding_mode="replicate")
    if (
        base_tiles.shape != current_tiles.shape
        or previous_tiles.shape != current_tiles.shape
        or sdf_tiles.shape != current_tiles[:, :1].shape
        or scalars.shape[0] != current_tiles.shape[0]
    ):
        raise ValueError("local tensors are incompatible with the cover")
    if neighbor_faces is None:
        neighbor_faces = neighboring_face_mask_2d(
            cover.grid_shape,
            batch=current.shape[0],
            device=current.device,
            dtype=current.dtype,
        )
    trace, flux, roles, incoming = directional_state_flux_trace_2d(
        current_tiles,
        base_tiles,
        state_mean=state_mean,
        state_std=state_std,
        dx=dx,
        viscosity=viscosity,
        face_width=face_width,
        incoming_temperature=incoming_temperature,
        neighbor_faces=neighbor_faces,
    )
    count = base_tiles.shape[0]
    chunk = count if chunk_tiles is None else int(chunk_tiles)
    if chunk <= 0:
        raise ValueError("chunk size must be positive")
    candidates = []
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        candidates.append(
            solver(
                base_tiles[start:stop],
                previous_tiles[start:stop],
                sdf_tiles[start:stop],
                scalars[start:stop],
                trace[start:stop],
                flux[start:stop],
                roles[start:stop],
                incoming[start:stop],
            )
        )
    local = torch.cat(candidates, dim=0)
    proposal = assemble_subdomains_2d(
        cover, local, current.shape[0], assembly,
    )
    field = current + tau * (proposal - current)
    return RobinSNISweep2D(
        field=field,
        local_candidates=local,
        trace=trace,
        flux=flux,
        incoming=incoming,
    )
