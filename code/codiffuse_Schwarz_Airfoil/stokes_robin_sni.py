"""Neural Schwarz sweep using optimized time-discrete Stokes Robin data."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .airfoil_sni import _FiLMBlock2D, _groups
from .airfoil_trace_sni import assemble_subdomains_2d, neighboring_face_mask_2d
from .optimized_stokes_robin import (
    normalized_stokes_robin_residual_port_2d,
    retract_normalized_stokes_robin_residual_2d,
)
from .pressure_coarse_correction import (
    overlap_pressure_constant_correction_2d,
)


Tensor = torch.Tensor
CONTRACT = "airfoil_stokes_robin_local_step_v1"


class StokesRobinTileSolver2D(nn.Module):
    """Predict one local solver step from local fields and stress Robin data."""

    def __init__(
        self,
        *,
        physical_channels: int = 13,
        scalar_dim: int = 4,
        width: int = 96,
        depth: int = 8,
        residual_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if min(physical_channels, width, depth) <= 0 or scalar_dim < 0:
            raise ValueError("invalid Stokes Robin solver dimensions")
        if len(residual_scale) != 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in residual_scale
        ):
            raise ValueError("residual scale must contain three positive values")
        self.physical_channels = int(physical_channels)
        self.scalar_dim = int(scalar_dim)
        self.register_buffer(
            "residual_scale",
            torch.tensor(residual_scale, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.stem = nn.Conv2d(physical_channels + 8 + 4, width, 5, padding=2)
        cycle = (1, 2, 4, 1, 2, 4)
        self.blocks = nn.ModuleList(
            _FiLMBlock2D(width, scalar_dim, cycle[index % len(cycle)])
            for index in range(depth)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(),
            nn.Conv2d(width, 3, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        physical: Tensor,
        scalar: Tensor,
        robin_port: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        shape = physical.shape[-2:]
        if physical.ndim != 4 or physical.shape[1] != self.physical_channels:
            raise ValueError("physical condition has the wrong shape")
        if robin_port.shape != (physical.shape[0], 8, *shape):
            raise ValueError("Robin port must have four two-channel blocks")
        if face_roles.shape != (physical.shape[0], 4, *shape):
            raise ValueError("face roles have the wrong shape")
        if scalar.shape != (physical.shape[0], self.scalar_dim):
            raise ValueError("scalar condition has the wrong shape")
        hidden = self.stem(torch.cat((physical, robin_port, face_roles), dim=1))
        for block in self.blocks:
            hidden = block(hidden, scalar)
        return (
            physical[:, :3] + self.output(hidden) * self.residual_scale.to(hidden)
        ) * physical[:, -1:]


class _ZeroPreservingResponseBlock2D(nn.Module):
    """Context-modulated block that remains exactly linear in its port state."""

    def __init__(self, width: int, scalar_dim: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            width, width, 3, padding=dilation, dilation=dilation, bias=False,
        )
        self.conv2 = nn.Conv2d(
            width, width, 3, padding=dilation, dilation=dilation, bias=False,
        )
        self.context_gate = nn.Conv2d(width, 2 * width, 1)
        self.scalar_gate = (
            nn.Linear(scalar_dim, 2 * width) if scalar_dim else None
        )

    def forward(
        self, state: Tensor, context: Tensor, scalar: Tensor,
    ) -> Tensor:
        gate = self.context_gate(context)
        if self.scalar_gate is not None:
            gate = gate + self.scalar_gate(scalar)[..., None, None]
        gate1, gate2 = (
            1.0 + 0.5 * torch.tanh(gate)
        ).chunk(2, dim=1)
        hidden = self.conv1(state * gate1)
        hidden = self.conv2(hidden * gate2)
        return state + hidden


class _ZeroPreservingGlobalResponseAttention2D(nn.Module):
    """Transport a Robin response across one tile without changing its zero.

    Queries and keys depend only on the known local BVP context and relative
    tile position; values depend only on the current Robin response.  Thus the
    block is globally coupled within a tile but remains exactly linear in the
    boundary message for a fixed local condition.
    """

    def __init__(
        self, width: int, scalar_dim: int, heads: int, grid: int,
    ) -> None:
        super().__init__()
        if width % heads != 0 or min(width, heads, grid) <= 0:
            raise ValueError("invalid zero-preserving attention dimensions")
        self.grid = int(grid)
        self.attention = nn.MultiheadAttention(
            width, heads, batch_first=True, bias=False,
        )
        self.position = nn.Parameter(torch.empty(1, grid * grid, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.scalar_context = (
            nn.Linear(scalar_dim, width) if scalar_dim else None
        )
        self.output = nn.Conv2d(width, width, 1, bias=False)

    def forward(
        self, response: Tensor, context: Tensor, scalar: Tensor,
    ) -> Tensor:
        size = (self.grid, self.grid)
        response_coarse = F.adaptive_avg_pool2d(response, size)
        context_coarse = F.adaptive_avg_pool2d(context, size)
        query = context_coarse.flatten(2).transpose(1, 2) + self.position
        if self.scalar_context is not None:
            query = query + self.scalar_context(scalar)[:, None]
        values = response_coarse.flatten(2).transpose(1, 2)
        mixed, _ = self.attention(
            query, query, values, need_weights=False,
        )
        mixed = mixed.transpose(1, 2).reshape(
            response.shape[0], response.shape[1], self.grid, self.grid,
        )
        mixed = F.interpolate(
            mixed, size=response.shape[-2:], mode="bilinear",
            align_corners=False,
        )
        return response + self.output(mixed)


class SeparableStokesRobinTileSolver2D(nn.Module):
    """Local predictor plus an exactly zero-at-zero-port boundary response.

    The no-message Local/Core map is represented by the base branch.  Robin
    data enter only through a context-conditioned linear response branch whose
    convolutions have no additive bias.  Consequently, setting the centered
    Robin port to zero removes the response identically, and superposition in
    the port holds exactly for a fixed local condition.  This is the affine
    boundary-response structure of the time-discrete modified-Stokes solve.
    It prevents response training from silently redefining Local/Core while
    retaining nonlinear dependence of the response operator on geometry and
    the known explicit-RHS state.
    """

    def __init__(
        self,
        *,
        physical_channels: int = 13,
        scalar_dim: int = 4,
        width: int = 96,
        depth: int = 8,
        residual_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        base_reference_dt: float | None = None,
        response_reference_dt: float | None = None,
        projection_channels: int = 0,
        response_global_attention_layers: int = 0,
        response_attention_grid: int = 8,
        response_attention_heads: int = 8,
        base_face_role_dilation: int = 0,
    ) -> None:
        super().__init__()
        if min(physical_channels, width, depth) <= 0 or scalar_dim < 0:
            raise ValueError("invalid separable Stokes Robin solver dimensions")
        if len(residual_scale) != 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in residual_scale
        ):
            raise ValueError("residual scale must contain three positive values")
        self.physical_channels = int(physical_channels)
        self.scalar_dim = int(scalar_dim)
        if projection_channels not in (0, 2):
            raise ValueError("projection channels must be zero or two")
        self.projection_channels = int(projection_channels)
        for name, reference_dt in (
            ("base", base_reference_dt),
            ("response", response_reference_dt),
        ):
            if reference_dt is not None and (
                not math.isfinite(float(reference_dt))
                or float(reference_dt) <= 0.0
            ):
                raise ValueError(
                    f"{name} reference dt must be positive and finite"
                )
            if reference_dt is not None and scalar_dim < 2:
                raise ValueError(
                    f"dt-scaled {name} response requires the dt scalar"
                )
        self.base_reference_dt = (
            None if base_reference_dt is None else float(base_reference_dt)
        )
        self.response_reference_dt = (
            None
            if response_reference_dt is None
            else float(response_reference_dt)
        )
        if base_face_role_dilation < 0:
            raise ValueError("base face-role dilation must be nonnegative")
        self.base_face_role_dilation = int(base_face_role_dilation)
        self.register_buffer(
            "residual_scale",
            torch.tensor(residual_scale, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )
        cycle = (1, 2, 4, 1, 2, 4)
        self.base_stem = nn.Conv2d(physical_channels + 4, width, 5, padding=2)
        self.base_blocks = nn.ModuleList(
            _FiLMBlock2D(width, scalar_dim, cycle[index % len(cycle)])
            for index in range(depth)
        )
        self.base_output = nn.Sequential(
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(),
            nn.Conv2d(width, 3 + self.projection_channels, 3, padding=1),
        )
        self.response_context = nn.Conv2d(
            physical_channels + 4, width, 3, padding=1,
        )
        self.response_stem = nn.Conv2d(8, width, 5, padding=2, bias=False)
        self.response_blocks = nn.ModuleList(
            _ZeroPreservingResponseBlock2D(
                width, scalar_dim, cycle[index % len(cycle)],
            )
            for index in range(depth)
        )
        if response_global_attention_layers < 0:
            raise ValueError("response attention layer count must be nonnegative")
        self.response_global_attention = nn.ModuleList(
            _ZeroPreservingGlobalResponseAttention2D(
                width, scalar_dim, response_attention_heads,
                response_attention_grid,
            )
            for _ in range(response_global_attention_layers)
        )
        self.response_output = nn.Conv2d(
            width, 3 + self.projection_channels, 3, padding=1, bias=False,
        )
        for output in (self.base_output[-1], self.response_output):
            nn.init.zeros_(output.weight)
        nn.init.zeros_(self.base_output[-1].bias)

    def forward(
        self,
        physical: Tensor,
        scalar: Tensor,
        robin_port: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        shape = physical.shape[-2:]
        if physical.ndim != 4 or physical.shape[1] != self.physical_channels:
            raise ValueError("physical condition has the wrong shape")
        if robin_port.shape != (physical.shape[0], 8, *shape):
            raise ValueError("Robin port must have four two-channel blocks")
        if face_roles.shape != (physical.shape[0], 4, *shape):
            raise ValueError("face roles have the wrong shape")
        if scalar.shape != (physical.shape[0], self.scalar_dim):
            raise ValueError("scalar condition has the wrong shape")
        base_roles = face_roles
        for _ in range(self.base_face_role_dilation):
            base_roles = F.max_pool2d(
                base_roles, kernel_size=3, stride=1, padding=1,
            )
        base_condition = torch.cat((physical, base_roles), dim=1)
        response_condition = torch.cat((physical, face_roles), dim=1)
        base = self.base_stem(base_condition)
        for block in self.base_blocks:
            base = block(base, scalar)
        context = self.response_context(response_condition)
        response = self.response_stem(robin_port)
        for block in self.response_blocks:
            response = block(response, context, scalar)
        for block in self.response_global_attention:
            response = block(response, context, scalar)
        base_output = self.base_output(base)
        base_delta = base_output[:, :3]
        if self.base_reference_dt is not None:
            # STRICT_LOCAL_SCALAR_NAMES[1] is dt / 0.01.  The explicit RHS of
            # a first-order solver step scales with dt, while the centered
            # Robin response already carries its own dt-sized amplitude.
            dt_scale = (
                scalar[:, 1:2] * (0.01 / self.base_reference_dt)
            )[..., None, None]
            base_delta = base_delta * dt_scale
        response_output = self.response_output(response)
        response_delta = response_output[:, :3]
        if self.response_reference_dt is not None:
            # The communicated correction is the difference between two
            # candidate endpoints of the same native step.  It therefore
            # vanishes with that step just like the particular update.  This
            # explicit factor carries the leading cadence dependence, while
            # the context gates learn the remaining modified-Stokes response.
            response_dt_scale = (
                scalar[:, 1:2] * (0.01 / self.response_reference_dt)
            )[..., None, None]
            response_delta = response_delta * response_dt_scale
        delta = base_delta + response_delta
        state = (
            physical[:, :3] + delta * self.residual_scale.to(delta)
        ) * physical[:, -1:]
        if self.projection_channels == 0:
            return state
        # The auxiliary channels are the standardized cell-centered pressure
        # gradient that incflo stores as part of its next Markov state.  They
        # are not transmitted by Schwarz; they are accepted together with the
        # final local velocity/pressure response for the following time step.
        projection = (
            base_output[:, 3:] + response_output[:, 3:]
        ) * physical[:, -1:]
        return torch.cat((state, projection), dim=1)


def build_stokes_robin_tile_solver(
    *, response_architecture: str = "entangled", **kwargs,
) -> nn.Module:
    if response_architecture == "entangled":
        return StokesRobinTileSolver2D(**kwargs)
    if response_architecture == "separable":
        return SeparableStokesRobinTileSolver2D(**kwargs)
    raise ValueError(f"unknown Stokes Robin response architecture: {response_architecture}")


def stokes_robin_solver_from_payload(
    payload: dict[str, object], device: torch.device,
) -> nn.Module:
    """Rebuild a strict-local optimized-Stokes endpoint from its checkpoint."""
    if payload.get("contract") != CONTRACT:
        raise ValueError("checkpoint is not an optimized-Stokes Airfoil endpoint")
    args = dict(payload["args"])
    statistics = dict(payload["statistics"])
    residual_scale = tuple(
        float(delta) / float(state)
        for delta, state in zip(
            statistics["delta_std"], statistics["state_std"],
        )
    )
    architecture = str(payload.get("response_architecture", "entangled"))
    backbone = str(payload.get("backbone", args.get("backbone", "cnn")))
    if backbone == "point":
        from .absolute_robin_sni import AbsoluteRobinPointTileSolver2D

        solver = AbsoluteRobinPointTileSolver2D(
            physical_channels=int(payload.get("physical_channel_count", 13)),
            d_model=int(args.get("point_d_model", 320)),
            surface_anchors=int(args.get("point_surface_anchors", 256)),
            volume_anchors=int(args.get("point_volume_anchors", 512)),
            geometry_anchors=int(args.get("point_geometry_anchors", 128)),
            depth=int(args.get("point_depth", 5)),
            decoder_depth=int(args.get("point_decoder_depth", 3)),
            heads=int(args.get("point_heads", 8)),
            robin_channels=8,
            projection_channels=int(payload.get("projection_channels", 0)),
            interface_anchor_fraction=float(args.get(
                "point_interface_anchor_fraction", 0.0,
            )),
            port_input_scale=float(args.get("point_port_input_scale", 1.0)),
            residual_scale=residual_scale,
        ).to(device)
        solver.load_state_dict(payload["solver"], strict=True)
        return solver
    if backbone != "cnn":
        raise ValueError(f"unknown optimized-Stokes backbone: {backbone}")
    optional = {}
    if architecture == "separable":
        optional["projection_channels"] = int(
            payload.get("projection_channels", 0)
        )
        reference_dt = float(args.get("base_reference_dt", 0.0))
        if reference_dt > 0.0:
            optional["base_reference_dt"] = reference_dt
        response_reference_dt = float(args.get("response_reference_dt", 0.0))
        if response_reference_dt > 0.0:
            optional["response_reference_dt"] = response_reference_dt
        optional["response_global_attention_layers"] = int(
            args.get("response_global_attention_layers", 0)
        )
        optional["response_attention_grid"] = int(
            args.get("response_attention_grid", 8)
        )
        optional["response_attention_heads"] = int(
            args.get("response_attention_heads", 8)
        )
        optional["base_face_role_dilation"] = int(
            args.get("base_face_role_dilation", 0)
        )
    solver = build_stokes_robin_tile_solver(
        response_architecture=architecture,
        physical_channels=int(payload.get("physical_channel_count", 13)),
        width=int(args["width"]),
        depth=int(args["depth"]),
        residual_scale=residual_scale,
        **optional,
    ).to(device)
    solver.load_state_dict(payload["solver"], strict=True)
    return solver


@dataclass(frozen=True)
class StokesRobinSweep2D:
    field: Tensor
    local_candidates: Tensor
    auxiliary_field: Tensor | None
    local_auxiliary_candidates: Tensor | None
    imposed_port: Tensor
    next_port: Tensor
    reference_port: Tensor | None
    interface_rms: Tensor
    port_defect_rms: Tensor
    realization_rms: Tensor
    pressure_reference_mean: Tensor | None = None
    pressure_overlap_rms_before: Tensor | None = None
    pressure_overlap_rms_after: Tensor | None = None


def stokes_robin_sni_sweep_2d(
    solver: nn.Module,
    physical_tiles: Tensor,
    scalars: Tensor,
    current: Tensor,
    cover,
    *,
    state_mean: Tensor,
    state_std: Tensor,
    robin_scale: Tensor,
    dx: float,
    viscosity: Tensor | float,
    normal_impedance: Tensor | float,
    tangential_impedance: Tensor | float,
    relaxation: float = 0.5,
    face_width: int = 1,
    chunk_tiles: int | None = None,
    neighbor_faces: Tensor | None = None,
    imposed_port: Tensor | None = None,
    reference_port: Tensor | None = None,
    defect_coordinate: bool = False,
    assembly: str = "normalized_overlap",
    symmetric_stress: bool = False,
    analytic_retraction: bool = False,
    pressure_coarse_correction: bool = False,
    pressure_reference_mean: Tensor | None = None,
) -> StokesRobinSweep2D:
    """Apply one parallel local solve and receiver-normal Robin refresh."""
    if not 0.0 < relaxation <= 1.5:
        raise ValueError("relaxation must lie in (0,1.5]")
    current_tiles = cover.tile(current, padding_mode="replicate")
    if physical_tiles.shape[0] != current_tiles.shape[0]:
        raise ValueError("physical tiles and current cover disagree")
    if neighbor_faces is None:
        neighbor_faces = neighboring_face_mask_2d(
            cover.grid_shape,
            batch=current.shape[0],
            device=current.device,
            dtype=current.dtype,
        )
    port_kwargs = {
        "state_mean": state_mean,
        "state_std": state_std,
        "robin_scale": robin_scale,
        "dx": dx,
        "viscosity": viscosity,
        "normal_impedance": normal_impedance,
        "tangential_impedance": tangential_impedance,
        "face_width": face_width,
        "neighbor_faces": neighbor_faces,
        "valid": physical_tiles[:, -1:],
        "symmetric_stress": symmetric_stress,
    }
    base_tiles = physical_tiles[:, :3]
    initialized, roles = normalized_stokes_robin_residual_port_2d(
        current_tiles, base_tiles, robin_increment_scale=robin_scale,
        **{key: value for key, value in port_kwargs.items() if key != "robin_scale"},
    )
    port = (
        torch.zeros_like(initialized)
        if defect_coordinate and imposed_port is None
        else initialized if imposed_port is None else imposed_port
    )
    if port.shape != initialized.shape:
        raise ValueError("imposed Robin port has the wrong shape")
    active = roles.repeat_interleave(2, dim=1)
    port = port * active
    chunk = current_tiles.shape[0] if chunk_tiles is None else int(chunk_tiles)
    if chunk <= 0:
        raise ValueError("chunk_tiles must be positive")
    candidates = []
    for start in range(0, current_tiles.shape[0], chunk):
        stop = min(start + chunk, current_tiles.shape[0])
        candidates.append(solver(
            physical_tiles[start:stop], scalars[start:stop],
            port[start:stop], roles[start:stop],
        ))
    local_full = torch.cat(candidates)
    if local_full.shape[1] < 3:
        raise ValueError("local Stokes solver must return at least (u,v,p)")
    local = local_full[:, :3]
    if analytic_retraction:
        if face_width != 1:
            raise ValueError("analytic Robin retraction requires face_width=1")
        if not defect_coordinate or reference_port is not None:
            desired_residual = (
                port + reference_port
                if defect_coordinate else port
            )
            local = retract_normalized_stokes_robin_residual_2d(
                local, base_tiles, desired_residual,
                state_mean=state_mean,
                state_std=state_std,
                robin_increment_scale=robin_scale,
                dx=dx,
                viscosity=viscosity,
                normal_impedance=normal_impedance,
                tangential_impedance=tangential_impedance,
                neighbor_faces=neighbor_faces,
                valid=physical_tiles[:, -1:],
                symmetric_stress=symmetric_stress,
            )
    # The graph solve below recovers the otherwise free pressure constants for
    # the assembled field.  It is deliberately *not* part of the Robin map.
    # Bui--Japhet--Omnes (2024, Rem. 10.1) make the same separation: pressure
    # recovery must not alter the velocity iteration or its transmission data.
    # Keep the locally solved state for realization and neighbor refresh, then
    # correct a separate tensor only for assembly/reporting.
    transmission_local = local
    pressure_overlap_before = None
    pressure_overlap_after = None
    if pressure_coarse_correction:
        pressure_correction = overlap_pressure_constant_correction_2d(
            local, cover, current.shape[0],
            valid_tiles=physical_tiles[:, -1:],
            reference_mean=pressure_reference_mean,
        )
        local = pressure_correction.patches
        pressure_reference_mean = pressure_correction.reference_mean
        pressure_overlap_before = pressure_correction.overlap_rms_before
        pressure_overlap_after = pressure_correction.overlap_rms_after
    local_auxiliary = (
        local_full[:, 3:] if local_full.shape[1] > 3 else None
    )
    field = assemble_subdomains_2d(
        cover, local, current.shape[0], assembly,
    )
    exchange = assemble_subdomains_2d(
        cover, local, current.shape[0], "disjoint_core",
    )
    transmission_exchange = assemble_subdomains_2d(
        cover, transmission_local, current.shape[0], "disjoint_core",
    )
    valid = assemble_subdomains_2d(
        cover, physical_tiles[:, -1:], current.shape[0], assembly,
    )
    field = field * valid
    auxiliary_field = (
        assemble_subdomains_2d(
            cover, local_auxiliary, current.shape[0], assembly,
        ) * valid
        if local_auxiliary is not None else None
    )
    consensus = cover.tile(exchange * valid, padding_mode="replicate")
    transmission_consensus = cover.tile(
        transmission_exchange * valid, padding_mode="replicate",
    )
    collar = (roles.sum(1, keepdim=True) > 0.0).to(local)
    interface_rms = (
        ((local - consensus).square() * collar).sum()
        / (collar.sum() * local.shape[1]).clamp_min(1.0)
    ).sqrt()
    residual_kwargs = {
        key: value for key, value in port_kwargs.items() if key != "robin_scale"
    }
    realized, _ = normalized_stokes_robin_residual_port_2d(
        transmission_local, base_tiles,
        robin_increment_scale=robin_scale, **residual_kwargs,
    )
    if defect_coordinate:
        if reference_port is None:
            # The first zero-defect solve is Local/Core.  Its Robin trace is
            # the fixed particular-solution reference for all later sweeps.
            reference_port = realized.detach()
        if reference_port.shape != realized.shape:
            raise ValueError("reference Robin port has the wrong shape")
        realized_coordinate = realized - reference_port
    else:
        realized_coordinate = realized
    realization_rms = (
        ((realized_coordinate - port).square() * active).sum()
        / active.sum().clamp_min(1.0)
    ).sqrt()
    neighbor_proposal, _ = normalized_stokes_robin_residual_port_2d(
        transmission_consensus, base_tiles,
        robin_increment_scale=robin_scale,
        **residual_kwargs,
    )
    if defect_coordinate:
        assert reference_port is not None
        neighbor_proposal = neighbor_proposal - reference_port
    residual = (neighbor_proposal - port) * active
    port_defect_rms = (
        residual.square().sum() / active.sum().clamp_min(1.0)
    ).sqrt()
    next_port = (port + relaxation * residual) * active
    return StokesRobinSweep2D(
        field=field,
        local_candidates=local,
        auxiliary_field=auxiliary_field,
        local_auxiliary_candidates=local_auxiliary,
        imposed_port=port,
        next_port=next_port,
        reference_port=reference_port,
        interface_rms=interface_rms,
        port_defect_rms=port_defect_rms,
        realization_rms=realization_rms,
        pressure_reference_mean=pressure_reference_mean,
        pressure_overlap_rms_before=pressure_overlap_before,
        pressure_overlap_rms_after=pressure_overlap_after,
    )
