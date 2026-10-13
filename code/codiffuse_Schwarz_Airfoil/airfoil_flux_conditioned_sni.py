"""Flux-conditioned local BVP solver for transport SNI on Airfoil.

Unlike :mod:`airfoil_flux_sni`, this module does not apply a flux correction
to a solver trained with Dirichlet traces.  The local neural operator is
trained on four *directional flux residuals* relative to its frozen local
endpoint.  At
inference each Schwarz sweep restricts the current global field, evaluates
the incompressible conservative flux, calls every flux-conditioned local
operator in parallel, and performs one normalized POU update.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .airfoil_sni import _FiLMBlock2D, _groups
from .cartesian_cover import CartesianCover


Tensor = torch.Tensor


def physical_zero_normalized_2d(
    reference: Tensor, state_mean: Tensor, state_std: Tensor,
) -> Tensor:
    """Represent the physical zero flow in the endpoint's affine coordinates."""
    if reference.ndim != 4:
        raise ValueError("Airfoil state reference must have shape B,C,H,W")
    mean = torch.as_tensor(
        state_mean, device=reference.device, dtype=reference.dtype,
    ).flatten()
    std = torch.as_tensor(
        state_std, device=reference.device, dtype=reference.dtype,
    ).flatten()
    if mean.numel() != reference.shape[1] or std.numel() != reference.shape[1]:
        raise ValueError("Airfoil affine statistics do not match state channels")
    if bool(torch.any(std <= 0)):
        raise ValueError("Airfoil state standard deviations must be positive")
    value = (-mean / std).reshape(1, -1, 1, 1)
    return value.expand_as(reference).clone()
FACE_NORMALS_2D = ((0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0))


def directional_face_roles_2d(
    height: int,
    width: int,
    *,
    face_width: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return separate north/south/west/east role masks, including corners."""
    if min(height, width, face_width) <= 0 or 2 * face_width > min(height, width):
        raise ValueError("invalid face dimensions")
    roles = torch.zeros(1, 4, height, width, device=device, dtype=dtype)
    roles[:, 0, :face_width, :] = 1.0
    roles[:, 1, -face_width:, :] = 1.0
    roles[:, 2, :, :face_width] = 1.0
    roles[:, 3, :, -face_width:] = 1.0
    return roles


def _batch_viscosity(viscosity: Tensor | float, state: Tensor) -> Tensor:
    result = torch.as_tensor(
        viscosity, device=state.device, dtype=state.dtype,
    ).flatten()
    if result.numel() == 1:
        result = result.expand(state.shape[0])
    if result.shape != (state.shape[0],) or torch.any(result < 0):
        raise ValueError("viscosity must be one nonnegative value per sample")
    return result[:, None, None]


def directional_conservative_flux_2d(
    state: Tensor,
    *,
    dx: float,
    viscosity: Tensor | float,
    face_width: int = 1,
) -> tuple[Tensor, Tensor]:
    """Return four face-role blocks of ``(mass, momentum-x, momentum-y)``.

    ``state`` is physical ``(u,v,p)``.  Momentum uses the full Newtonian
    conservative flux

    ``(u.n) u + p n - nu (grad u + grad u^T) n``.

    The result has 12 channels: three quantities for each of north, south,
    west, and east.  Keeping corner roles separate avoids averaging two
    distinct normals into a nonphysical diagonal trace.
    """
    if state.ndim != 4 or state.shape[1] != 3 or dx <= 0:
        raise ValueError("state must have shape B,3,H,W and dx must be positive")
    roles = directional_face_roles_2d(
        state.shape[-2], state.shape[-1], face_width=face_width,
        device=state.device, dtype=state.dtype,
    ).expand(state.shape[0], -1, -1, -1)
    u, v, pressure = state.unbind(1)
    du_dy, du_dx = torch.gradient(
        u, spacing=(dx, dx), dim=(-2, -1), edge_order=1,
    )
    dv_dy, dv_dx = torch.gradient(
        v, spacing=(dx, dx), dim=(-2, -1), edge_order=1,
    )
    nu = _batch_viscosity(viscosity, state)
    normal_x_stress = 2.0 * nu * du_dx
    normal_y_stress = 2.0 * nu * dv_dy
    shear_stress = nu * (du_dy + dv_dx)
    blocks = []
    for face, (nx, ny) in enumerate(FACE_NORMALS_2D):
        normal_velocity = nx * u + ny * v
        viscous_x = nx * normal_x_stress + ny * shear_stress
        viscous_y = nx * shear_stress + ny * normal_y_stress
        flux = torch.stack((
            normal_velocity,
            normal_velocity * u + nx * pressure - viscous_x,
            normal_velocity * v + ny * pressure - viscous_y,
        ), dim=1)
        blocks.append(flux * roles[:, face:face + 1])
    return torch.cat(blocks, dim=1), roles


def flux_scales_2d(state_mean: Tensor, state_std: Tensor) -> Tensor:
    """Fixed physical scales for the 12 flux channels."""
    mean = torch.as_tensor(state_mean).flatten()
    std = torch.as_tensor(state_std).flatten()
    if mean.numel() != 3 or std.numel() != 3:
        raise ValueError("state statistics must contain u, v, p")
    velocity = torch.maximum(mean[:2].abs().max(), std[:2].max()).clamp_min(.1)
    momentum = torch.maximum(velocity.square(), torch.maximum(
        mean[2].abs(), std[2],
    )).clamp_min(.1)
    block = torch.stack((velocity, momentum, momentum))
    return block.repeat(4)


class FluxConditionedTileSolver2D(nn.Module):
    """Predict a complete local next state from directional physical fluxes."""

    def __init__(
        self,
        *,
        state_channels: int = 3,
        scalar_dim: int = 4,
        width: int = 48,
        depth: int = 5,
    ) -> None:
        super().__init__()
        if state_channels != 3:
            raise ValueError("the incompressible flux contract requires (u,v,p)")
        self.state_channels = state_channels
        self.scalar_dim = scalar_dim
        # base, previous state, SDF, 4x(mass,mom-x,mom-y), and 4 face roles
        input_channels = 2 * state_channels + 1 + 12 + 4
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
        previous_state: Tensor,
        sdf_np1: Tensor,
        scalar: Tensor,
        boundary_flux: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        if base.ndim != 4 or previous_state.shape != base.shape:
            raise ValueError("base/previous state must share BCHW shape")
        if sdf_np1.shape != base[:, :1].shape:
            raise ValueError("SDF must have shape B,1,H,W")
        if boundary_flux.shape != (base.shape[0], 12, *base.shape[-2:]):
            raise ValueError("boundary flux must have shape B,12,H,W")
        if face_roles.shape != (base.shape[0], 4, *base.shape[-2:]):
            raise ValueError("face roles must have shape B,4,H,W")
        if scalar.shape != (base.shape[0], self.scalar_dim):
            raise ValueError("scalar conditioning shape does not match")
        common = (base, previous_state, sdf_np1)
        conditioned = torch.cat((*common, boundary_flux, face_roles), dim=1)
        zero_trace = torch.cat((
            *common, torch.zeros_like(boundary_flux), face_roles,
        ), dim=1)
        hidden = self.stem(torch.cat((conditioned, zero_trace), dim=0))
        doubled_scalar = torch.cat((scalar, scalar), dim=0)
        for block in self.blocks:
            hidden = block(hidden, doubled_scalar)
        conditioned_response, zero_response = self.output(hidden).chunk(2, dim=0)
        # Exact centered-chart invariant: zero flux residual cannot alter the
        # frozen local endpoint, independent of learned biases or geometry.
        return base + (conditioned_response - zero_response)


@dataclass(frozen=True)
class FluxConditionedSNISweep:
    field: Tensor
    local_candidates: Tensor
    correction_norm: float
    interface_norm: float
    imposed_flux_defect_norm: float


@torch.no_grad()
def flux_conditioned_sni_sweep(
    solver: FluxConditionedTileSolver2D,
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
    tau: float = 1.0,
    chunk_tiles: int = 64,
    face_width: int = 1,
    assembly: str = "weighted",
) -> FluxConditionedSNISweep:
    """Apply one physical-flux trace refresh and one assembled update."""
    if not 0 < tau <= 1.5 or chunk_tiles <= 0:
        raise ValueError("invalid relaxation or chunk size")
    if assembly not in {"weighted", "core_ras"}:
        raise ValueError("assembly must be weighted or core_ras")
    current_tiles = cover.tile(current, padding_mode="replicate")
    if base_tiles.shape != current_tiles.shape or previous_tiles.shape != current_tiles.shape:
        raise ValueError("base/previous tiles are incompatible with the cover")
    mean = torch.as_tensor(state_mean, device=current.device, dtype=current.dtype).reshape(
        1, 3, 1, 1,
    )
    std = torch.as_tensor(state_std, device=current.device, dtype=current.dtype).reshape(
        1, 3, 1, 1,
    )
    scale = flux_scales_2d(mean.flatten(), std.flatten()).to(current).reshape(
        1, 12, 1, 1,
    )
    imposed_physical, roles = directional_conservative_flux_2d(
        current_tiles * std + mean, dx=dx, viscosity=viscosity,
        face_width=face_width,
    )
    base_physical, _ = directional_conservative_flux_2d(
        base_tiles * std + mean, dx=dx, viscosity=viscosity,
        face_width=face_width,
    )
    imposed = (imposed_physical - base_physical) / scale
    device = next(solver.parameters()).device
    candidates = []
    for start in range(0, cover.tile_count, chunk_tiles):
        stop = min(start + chunk_tiles, cover.tile_count)
        candidates.append(solver(
            base_tiles[start:stop].to(device),
            previous_tiles[start:stop].to(device),
            sdf_tiles[start:stop].to(device),
            scalars[start:stop].to(device),
            imposed[start:stop].to(device),
            roles[start:stop].to(device),
        ).float().to(current.device))
    local = torch.cat(candidates)
    proposal = (
        cover.assemble_ras(local, batch=1)
        if assembly == "core_ras"
        else cover.assemble_weighted(local, batch=1)
    )
    next_field = current + tau * (proposal - current)
    consensus = cover.tile(next_field, padding_mode="replicate")
    local_flux, _ = directional_conservative_flux_2d(
        local * std + mean, dx=dx, viscosity=viscosity,
        face_width=face_width,
    )
    active = roles.repeat_interleave(3, dim=1)
    flux_defect = (local_flux - imposed_physical) / scale * active
    denominator = (active.sum().clamp_min(1.0)).sqrt()
    field_denominator = torch.tensor(
        next_field.numel(), device=current.device, dtype=current.dtype,
    ).sqrt()
    return FluxConditionedSNISweep(
        field=next_field,
        local_candidates=local,
        correction_norm=float((next_field - current).square().sum().sqrt()
                              / field_denominator),
        interface_norm=float((local - consensus).square().mean().sqrt()),
        imposed_flux_defect_norm=float(flux_defect.square().sum().sqrt()
                                      / denominator),
    )
