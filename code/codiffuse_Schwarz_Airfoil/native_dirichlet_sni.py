"""Strict-local native-step Airfoil Dirichlet Schwarz operator.

This is the overlapping-Dirichlet control for the optimized-Robin experiment.
The zero-message particular solution is identical, but artificial-face state
is imposed exactly and exchanged from the neighboring RAS owner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .airfoil_sni import _FiLMBlock2D, _groups
from .airfoil_flux_conditioned_sni import directional_face_roles_2d
from .airfoil_trace_sni import assemble_subdomains_2d, neighboring_face_mask_2d
from .stokes_robin_sni import (
    _ZeroPreservingGlobalResponseAttention2D,
    _ZeroPreservingResponseBlock2D,
)
from .pressure_coarse_correction import (
    overlap_pressure_constant_correction_2d,
)


Tensor = torch.Tensor
CONTRACT = "airfoil_native_dirichlet_local_step_v1"
VELOCITY_CONTRACT = "airfoil_native_velocity_dirichlet_local_step_v1"


def normalized_dirichlet_trace_2d(
    state: Tensor,
    *,
    face_width: int = 1,
    neighbor_faces: Tensor | None = None,
    valid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return four face blocks of standardized ``(u,v,p)`` state."""
    if state.ndim != 4 or state.shape[1] != 3:
        raise ValueError("Dirichlet state must have shape B,3,H,W")
    roles = directional_face_roles_2d(
        *state.shape[-2:], face_width=face_width,
        device=state.device, dtype=state.dtype,
    ).expand(state.shape[0], -1, -1, -1)
    if neighbor_faces is not None:
        if neighbor_faces.shape != (state.shape[0], 4, 1, 1):
            raise ValueError("neighbor face mask must have shape B,4,1,1")
        roles = roles * neighbor_faces
    if valid is not None:
        if valid.shape != state[:, :1].shape:
            raise ValueError("valid mask must have shape B,1,H,W")
        roles = roles * valid
    return torch.cat([
        state * roles[:, face : face + 1] for face in range(4)
    ], dim=1), roles


def normalized_velocity_dirichlet_trace_2d(
    state: Tensor,
    *,
    face_width: int = 1,
    neighbor_faces: Tensor | None = None,
    valid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the four velocity traces of a Stokes Dirichlet problem."""
    full, roles = normalized_dirichlet_trace_2d(
        state, face_width=face_width, neighbor_faces=neighbor_faces,
        valid=valid,
    )
    full = full.reshape(state.shape[0], 4, 3, *state.shape[-2:])
    return full[:, :, :2].reshape(
        state.shape[0], 8, *state.shape[-2:],
    ), roles


def impose_normalized_dirichlet_trace_2d(
    candidate: Tensor, trace: Tensor, roles: Tensor,
) -> Tensor:
    """Strongly impose represented face values, averaging only at corners."""
    if candidate.ndim != 4 or candidate.shape[1] != 3:
        raise ValueError("Dirichlet candidate must have shape B,3,H,W")
    if trace.shape != (candidate.shape[0], 12, *candidate.shape[-2:]):
        raise ValueError("Dirichlet trace must contain four three-channel blocks")
    if roles.shape != (candidate.shape[0], 4, *candidate.shape[-2:]):
        raise ValueError("Dirichlet roles do not match candidate")
    values = trace.reshape(
        candidate.shape[0], 4, 3, *candidate.shape[-2:],
    )
    count = roles.sum(1, keepdim=True)
    imposed = values.sum(1) / count.clamp_min(1.0)
    active = (count > 0.0).to(candidate)
    return candidate * (1.0 - active) + imposed * active


def impose_normalized_velocity_dirichlet_trace_2d(
    candidate: Tensor, trace: Tensor, roles: Tensor,
) -> Tensor:
    """Impose interface velocity while leaving local pressure unconstrained."""
    if candidate.ndim != 4 or candidate.shape[1] != 3:
        raise ValueError("velocity Dirichlet candidate must have B,3,H,W")
    if trace.shape != (candidate.shape[0], 8, *candidate.shape[-2:]):
        raise ValueError("velocity trace must contain four two-channel blocks")
    if roles.shape != (candidate.shape[0], 4, *candidate.shape[-2:]):
        raise ValueError("velocity Dirichlet roles do not match candidate")
    values = trace.reshape(
        candidate.shape[0], 4, 2, *candidate.shape[-2:],
    )
    count = roles.sum(1, keepdim=True)
    imposed = values.sum(1) / count.clamp_min(1.0)
    active = (count > 0.0).to(candidate)
    velocity = candidate[:, :2] * (1.0 - active) + imposed * active
    return torch.cat((velocity, candidate[:, 2:3]), dim=1)


class SeparableNativeDirichletTileSolver2D(nn.Module):
    """Frozen particular solution plus zero-at-zero Dirichlet response."""

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
        dirichlet_channels: int = 3,
    ) -> None:
        super().__init__()
        if min(physical_channels, width, depth) <= 0 or scalar_dim < 0:
            raise ValueError("invalid native Dirichlet solver dimensions")
        if len(residual_scale) != 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in residual_scale
        ):
            raise ValueError("residual scale must contain three positives")
        if projection_channels not in (0, 2):
            raise ValueError("projection channels must be zero or two")
        if response_global_attention_layers < 0 or base_face_role_dilation < 0:
            raise ValueError("invalid Dirichlet response controls")
        if dirichlet_channels not in (2, 3):
            raise ValueError("Dirichlet response requires velocity or full state")
        self.physical_channels = int(physical_channels)
        self.scalar_dim = int(scalar_dim)
        self.projection_channels = int(projection_channels)
        self.base_reference_dt = (
            None if base_reference_dt is None else float(base_reference_dt)
        )
        self.response_reference_dt = (
            None if response_reference_dt is None
            else float(response_reference_dt)
        )
        self.base_face_role_dilation = int(base_face_role_dilation)
        self.dirichlet_channels = int(dirichlet_channels)
        for name, reference_dt in (
            ("base", self.base_reference_dt),
            ("response", self.response_reference_dt),
        ):
            if reference_dt is not None and (
                not math.isfinite(reference_dt) or reference_dt <= 0.0
            ):
                raise ValueError(f"{name} reference dt must be positive")
        self.register_buffer(
            "residual_scale",
            torch.tensor(residual_scale, dtype=torch.float32).reshape(1, 3, 1, 1),
            persistent=False,
        )
        cycle = (1, 2, 4, 1, 2, 4)
        # Keep base_* names and shapes exactly compatible with the selected
        # optimized-Robin Local/Core checkpoint.
        self.base_stem = nn.Conv2d(physical_channels + 4, width, 5, padding=2)
        self.base_blocks = nn.ModuleList(
            _FiLMBlock2D(width, scalar_dim, cycle[index % len(cycle)])
            for index in range(depth)
        )
        self.base_output = nn.Sequential(
            nn.GroupNorm(_groups(width), width), nn.SiLU(),
            nn.Conv2d(width, 3 + projection_channels, 3, padding=1),
        )
        self.response_context = nn.Conv2d(
            physical_channels + 4, width, 3, padding=1,
        )
        self.response_stem = nn.Conv2d(
            4 * self.dirichlet_channels, width, 5, padding=2, bias=False,
        )
        self.response_blocks = nn.ModuleList(
            _ZeroPreservingResponseBlock2D(
                width, scalar_dim, cycle[index % len(cycle)],
            )
            for index in range(depth)
        )
        self.response_global_attention = nn.ModuleList(
            _ZeroPreservingGlobalResponseAttention2D(
                width, scalar_dim, response_attention_heads,
                response_attention_grid,
            )
            for _ in range(response_global_attention_layers)
        )
        self.response_output = nn.Conv2d(
            width, 3 + projection_channels, 3, padding=1, bias=False,
        )
        for output in (self.base_output[-1], self.response_output):
            nn.init.zeros_(output.weight)
        nn.init.zeros_(self.base_output[-1].bias)

    def forward(
        self, physical: Tensor, scalar: Tensor,
        boundary_defect: Tensor, face_roles: Tensor,
    ) -> Tensor:
        shape = physical.shape[-2:]
        if physical.ndim != 4 or physical.shape[1] != self.physical_channels:
            raise ValueError("physical condition has the wrong shape")
        if boundary_defect.shape != (
            physical.shape[0], 4 * self.dirichlet_channels, *shape,
        ):
            raise ValueError("Dirichlet defect must have four state blocks")
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
        response = self.response_stem(boundary_defect)
        for block in self.response_blocks:
            response = block(response, context, scalar)
        for block in self.response_global_attention:
            response = block(response, context, scalar)
        base_output = self.base_output(base)
        response_output = self.response_output(response)
        base_delta = base_output[:, :3]
        response_delta = response_output[:, :3]
        if self.base_reference_dt is not None:
            scale = scalar[:, 1:2] * (0.01 / self.base_reference_dt)
            base_delta = base_delta * scale[..., None, None]
        if self.response_reference_dt is not None:
            scale = scalar[:, 1:2] * (0.01 / self.response_reference_dt)
            response_delta = response_delta * scale[..., None, None]
        state = (
            physical[:, :3]
            + (base_delta + response_delta) * self.residual_scale.to(base_delta)
        ) * physical[:, -1:]
        if self.projection_channels == 0:
            return state
        projection = (
            base_output[:, 3:] + response_output[:, 3:]
        ) * physical[:, -1:]
        return torch.cat((state, projection), dim=1)


@dataclass(frozen=True)
class NativeDirichletSweep2D:
    field: Tensor
    local_candidates: Tensor
    auxiliary_field: Tensor | None
    local_auxiliary_candidates: Tensor | None
    imposed_port: Tensor
    next_port: Tensor
    reference_port: Tensor
    interface_rms: Tensor
    port_defect_rms: Tensor
    realization_rms: Tensor
    pressure_reference_mean: Tensor | None = None
    pressure_overlap_rms_before: Tensor | None = None
    pressure_overlap_rms_after: Tensor | None = None


def native_dirichlet_sni_sweep_2d(
    solver: nn.Module,
    physical_tiles: Tensor,
    scalars: Tensor,
    current: Tensor,
    cover,
    *,
    relaxation: float = 1.0,
    face_width: int = 1,
    neighbor_faces: Tensor | None = None,
    imposed_port: Tensor | None = None,
    reference_port: Tensor | None = None,
    assembly: str = "normalized_overlap",
    strong_trace: bool = True,
    velocity_only: bool = False,
    pressure_coarse_correction: bool = False,
    pressure_reference_mean: Tensor | None = None,
    cached_particular: Tensor | None = None,
    transmission_valid: Tensor | None = None,
    correction_valid: Tensor | None = None,
) -> NativeDirichletSweep2D:
    """Apply one parallel overlapping-Dirichlet local solve and refresh."""
    if not 0.0 < relaxation <= 1.5:
        raise ValueError("relaxation must lie in (0,1.5]")
    current_tiles = cover.tile(current, padding_mode="replicate")
    if physical_tiles.shape[0] != current_tiles.shape[0]:
        raise ValueError("physical tiles and cover disagree")
    if neighbor_faces is None:
        neighbor_faces = neighboring_face_mask_2d(
            cover.grid_shape, batch=current.shape[0],
            device=current.device, dtype=current.dtype,
        )
    trace_valid = (
        physical_tiles[:, -1:]
        if transmission_valid is None else transmission_valid
    )
    if trace_valid.shape != physical_tiles[:, -1:].shape:
        raise ValueError("transmission-valid mask must be crop-aligned")
    correction_mask = (
        trace_valid if correction_valid is None else correction_valid
    )
    if correction_mask.shape != physical_tiles[:, -1:].shape:
        raise ValueError("correction-valid mask must be crop-aligned")
    common = {
        "face_width": face_width,
        "neighbor_faces": neighbor_faces,
        "valid": trace_valid,
    }
    trace_operator = (
        normalized_velocity_dirichlet_trace_2d
        if velocity_only else normalized_dirichlet_trace_2d
    )
    impose_operator = (
        impose_normalized_velocity_dirichlet_trace_2d
        if velocity_only else impose_normalized_dirichlet_trace_2d
    )
    initialized, roles = trace_operator(
        current_tiles, **common,
    )
    port = torch.zeros_like(initialized) if imposed_port is None else imposed_port
    active = roles.repeat_interleave(2 if velocity_only else 3, dim=1)
    correction_active = active * correction_mask
    port = port * correction_active
    if cached_particular is not None:
        if not bool(getattr(solver, "supports_cached_particular", False)):
            raise ValueError("cached particular supplied to an incompatible solver")
        local_full = solver.compose_from_particular(
            cached_particular, physical_tiles, scalars, port, roles,
        )
    else:
        local_full = solver(physical_tiles, scalars, port, roles)
    local = local_full[:, :3]
    if reference_port is None:
        reference_port, _ = trace_operator(local, **common)
    elif strong_trace:
        local = impose_operator(
            local, reference_port + port, roles,
        )
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
    local_auxiliary = local_full[:, 3:] if local_full.shape[1] > 3 else None
    field = assemble_subdomains_2d(
        cover, local, current.shape[0], assembly,
    )
    exchange = assemble_subdomains_2d(
        cover, local, current.shape[0], "disjoint_core",
    )
    valid = assemble_subdomains_2d(
        cover, physical_tiles[:, -1:], current.shape[0], assembly,
    )
    field = field * valid
    auxiliary = (
        assemble_subdomains_2d(
            cover, local_auxiliary, current.shape[0], assembly,
        ) * valid
        if local_auxiliary is not None else None
    )
    consensus = cover.tile(exchange * valid, padding_mode="replicate")
    proposal_absolute, _ = trace_operator(
        consensus, **common,
    )
    proposal = (proposal_absolute - reference_port) * correction_active
    residual = proposal - port
    next_port = (port + relaxation * residual) * correction_active
    realized_absolute, _ = trace_operator(local, **common)
    realized = realized_absolute - reference_port
    realization_rms = (
        ((realized - port).square() * correction_active).sum()
        / correction_active.sum().clamp_min(1.0)
    ).sqrt()
    port_defect_rms = (
        (residual.square() * correction_active).sum()
        / correction_active.sum().clamp_min(1.0)
    ).sqrt()
    collar = (roles.sum(1, keepdim=True) > 0.0).to(local)
    interface_rms = (
        ((local - cover.tile(exchange, padding_mode="replicate")).square()
         * collar).sum()
        / (collar.sum() * 3).clamp_min(1.0)
    ).sqrt()
    return NativeDirichletSweep2D(
        field=field,
        local_candidates=local,
        auxiliary_field=auxiliary,
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


def native_dirichlet_solver_from_payload(
    payload: dict[str, object], device: torch.device,
) -> nn.Module:
    if payload.get("contract") not in (CONTRACT, VELOCITY_CONTRACT):
        raise ValueError("checkpoint is not a native Dirichlet endpoint")
    args = dict(payload["args"])
    statistics = dict(payload["statistics"])
    residual_scale = tuple(
        float(delta) / float(state)
        for delta, state in zip(
            statistics["delta_std"], statistics["state_std"],
        )
    )
    backbone = str(payload.get("backbone", args.get("backbone", "cnn")))
    if backbone == "point":
        from .absolute_robin_sni import AbsoluteRobinPointTileSolver2D
        from .point_boundary_input_response import (
            PointBoundaryInputResponse2D,
            PointBoundaryInputWithAttachedResponse2D,
            PointBoundaryInputWithIncfloResponse2D,
            validate_point_boundary_input_metadata,
        )

        solver = AbsoluteRobinPointTileSolver2D(
            physical_channels=int(payload.get("physical_channel_count", 13)),
            d_model=int(args.get("point_d_model", 320)),
            surface_anchors=int(args.get("point_surface_anchors", 256)),
            volume_anchors=int(args.get("point_volume_anchors", 512)),
            geometry_anchors=int(args.get("point_geometry_anchors", 128)),
            depth=int(args.get("point_depth", 5)),
            decoder_depth=int(args.get("point_decoder_depth", 3)),
            heads=int(args.get("point_heads", 8)),
            robin_channels=(
                8 if payload.get("contract") == VELOCITY_CONTRACT else 12
            ),
            projection_channels=int(payload.get("projection_channels", 0)),
            interface_anchor_fraction=float(args.get(
                "point_interface_anchor_fraction", 0.0,
            )),
            port_input_scale=float(args.get("point_port_input_scale", 1.0)),
            residual_scale=residual_scale,
        ).to(device)
        solver.load_state_dict(payload["solver"], strict=True)
        rebuilt: nn.Module = solver
        response = payload.get("deterministic_modified_stokes_response")
        learned_response = payload.get("native_learned_boundary_response")
        if learned_response is not None and not isinstance(learned_response, dict):
            raise ValueError("invalid native learned-response declaration")
        if learned_response is not None and response is not None and not bool(
            learned_response.get("residual_over_fixed_response", False)
        ):
            raise ValueError(
                "a learned response may accompany a deterministic response "
                "only as an explicitly declared residual"
            )
        if response is not None:
            if not isinstance(response, dict) or not bool(
                response.get("enabled", False)
            ):
                raise ValueError(
                    "invalid deterministic modified-Stokes response metadata"
                )
            if payload.get("contract") != VELOCITY_CONTRACT:
                raise ValueError(
                    "modified-Stokes response requires velocity transmission"
                )
        if isinstance(response, dict) and response.get(
            "kind"
        ) == "incflo_helmholtz_amrex_q1_inflow_v1":
            from .native_incflo_response import (
                PointParticularWithIncfloResponse2D,
            )

            projection_statistics = payload.get("projection_statistics")
            if not isinstance(projection_statistics, dict):
                raise ValueError("incflo response omits projection statistics")
            rebuilt = PointParticularWithIncfloResponse2D(
                solver,
                state_std=torch.tensor(
                    statistics["state_std"], dtype=torch.float32,
                ),
                projection_std=torch.tensor(
                    projection_statistics["state_std"], dtype=torch.float32,
                ),
                dx=float(response["dx"]),
                helmholtz_iterations=int(response["helmholtz_iterations"]),
                projection_iterations=int(response["projection_iterations"]),
                tolerance=float(response["tolerance"]),
                trace_reference_kind=str(response.get(
                    "trace_reference_kind", "lagged_state",
                )),
                response_output_kind=str(response.get(
                    "response_output_kind", "five_state",
                )),
            ).to(device)
        elif isinstance(response, dict):
            if response.get("kind") != "collocated_augmented_modified_stokes_v1":
                raise ValueError("unknown deterministic response kind")
            from .discrete_modified_stokes_response import (
                PointParticularWithModifiedStokesResponse2D,
                RESPONSE_DOMAIN_SDF_FLUID,
            )

            rebuilt = PointParticularWithModifiedStokesResponse2D(
                solver,
                state_std=torch.tensor(
                    statistics["state_std"], dtype=torch.float32,
                ),
                state_mean=torch.tensor(
                    statistics["state_mean"], dtype=torch.float32,
                ),
                dx=float(response["dx"]),
                divergence_penalty=float(response["divergence_penalty"]),
                iterations=int(response["iterations"]),
                tolerance=float(response["tolerance"]),
                response_gain=float(response["response_gain"]),
                pressure_gain=float(response["pressure_gain"]),
                transport_gain=float(response.get("transport_gain", 0.0)),
                response_domain=str(response.get(
                    "response_domain", RESPONSE_DOMAIN_SDF_FLUID,
                )),
            ).to(device)

        if learned_response is not None:
            from .native_learned_boundary_response import (
                PointParticularWithLearnedBoundaryResponse2D,
                build_native_learned_response,
            )

            learned = build_native_learned_response(
                learned_response,
                physical_channels=solver.physical_channels,
                scalar_dim=solver.scalar_dim,
            ).to(device)
            state = payload.get("native_learned_boundary_response_state")
            if not isinstance(state, dict):
                raise ValueError("native learned response omits its state dictionary")
            learned.load_state_dict(state, strict=True)
            rebuilt = PointParticularWithLearnedBoundaryResponse2D(
                solver, learned,
                fixed_response=(rebuilt if response is not None else None),
            ).to(device)

        input_response = validate_point_boundary_input_metadata(
            payload.get("point_boundary_input_response")
        )
        if input_response is None:
            return rebuilt
        if learned_response is not None:
            raise ValueError(
                "boundary-input reruns cannot be combined with learned response"
            )
        if payload.get("contract") != VELOCITY_CONTRACT:
            raise ValueError("Point boundary-input response requires velocity ports")
        mode = str(input_response["mode"])
        gain = float(input_response["gain"])
        collar_width = int(input_response["collar_width"])
        if mode == "direct":
            return PointBoundaryInputResponse2D(
                solver, gain=gain, collar_width=collar_width,
            ).to(device)
        if response is None:
            raise ValueError("attached boundary-input mode requires a response chart")
        if isinstance(response, dict) and response.get(
            "kind"
        ) == "incflo_helmholtz_amrex_q1_inflow_v1":
            return PointBoundaryInputWithIncfloResponse2D(
                rebuilt, gain=gain, collar_width=collar_width,
            ).to(device)
        return PointBoundaryInputWithAttachedResponse2D(
            rebuilt, gain=gain, collar_width=collar_width,
        ).to(device)
    if backbone != "cnn":
        raise ValueError(f"unknown native-Dirichlet backbone: {backbone}")
    optional: dict[str, object] = {
        "projection_channels": int(payload.get("projection_channels", 0)),
        "response_global_attention_layers": int(
            args.get("response_global_attention_layers", 0)
        ),
        "response_attention_grid": int(args.get("response_attention_grid", 8)),
        "response_attention_heads": int(args.get("response_attention_heads", 8)),
        "base_face_role_dilation": int(args.get("base_face_role_dilation", 0)),
        "dirichlet_channels": (
            2 if payload.get("contract") == VELOCITY_CONTRACT else 3
        ),
    }
    for name in ("base_reference_dt", "response_reference_dt"):
        value = float(args.get(name, 0.0))
        if value > 0.0:
            optional[name] = value
    solver = SeparableNativeDirichletTileSolver2D(
        physical_channels=int(payload.get("physical_channel_count", 13)),
        width=int(args["width"]), depth=int(args["depth"]),
        residual_scale=residual_scale, **optional,
    ).to(device)
    solver.load_state_dict(payload["solver"], strict=True)
    return solver
