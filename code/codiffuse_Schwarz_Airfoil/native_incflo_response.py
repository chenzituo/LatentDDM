"""Production Airfoil local response matched to the native incflo split.

The heavy Point/AB-UPT endpoint represents the trace-independent particular
solution.  This module adds the unit-scale homogeneous response executed by
the data generator: implicit component diffusion followed by AMReX's regular
Q1 nodal projection with inflow boundary data.  Communication is performed in
the zero-net-flux velocity-trace space required by a divergence-free local
problem.  The projected field is never overwritten after projection.

The response is target-free.  Recorded targets enter only the separate
particular-label construction used during supervised training.
"""

from __future__ import annotations

from dataclasses import replace
import math

import torch
from torch import Tensor, nn

from .incflo_helmholtz_projection_response import (
    project_port_to_zero_net_flux,
    source_audited_helmholtz_nodal_response,
)
from .native_dirichlet_sni import (
    native_dirichlet_sni_sweep_2d,
    normalized_velocity_dirichlet_trace_2d,
)


KIND = "incflo_helmholtz_amrex_q1_inflow_v1"
TRACE_COORDINATE = "minimum_physical_l2_zero_net_flux_velocity_defect"
TRACE_REFERENCE_LAGGED = "lagged_state"
TRACE_REFERENCE_EXTRAPOLATED = "first_order_state_extrapolation"
TRACE_REFERENCE_LOCAL_PREDICTION = "local_prediction"
TRACE_REFERENCE_KINDS = (
    TRACE_REFERENCE_LAGGED,
    TRACE_REFERENCE_EXTRAPOLATED,
    TRACE_REFERENCE_LOCAL_PREDICTION,
)
RESPONSE_OUTPUT_FIVE_STATE = "five_state"
RESPONSE_OUTPUT_VELOCITY_ONLY = "velocity_only"
RESPONSE_OUTPUT_VELOCITY_PROJECTION = "velocity_projection"
RESPONSE_OUTPUT_KINDS = (
    RESPONSE_OUTPUT_FIVE_STATE,
    RESPONSE_OUTPUT_VELOCITY_ONLY,
    RESPONSE_OUTPUT_VELOCITY_PROJECTION,
)
STATE_TRACE_REFERENCE_KINDS = (
    TRACE_REFERENCE_LAGGED,
    TRACE_REFERENCE_EXTRAPOLATED,
)


def velocity_trace_reference_2d(
    example: dict[str, object],
    *,
    face_width: int,
    kind: str = TRACE_REFERENCE_LAGGED,
) -> tuple[Tensor, Tensor]:
    """Return a target-free absolute velocity reference on local faces.

    The lagged reference is the trace of ``q_n``.  The extrapolated reference
    is

    ``q_n + (dt_n / dt_(n-1)) (q_n - q_(n-1))``

    restricted to the same local faces.  Both states are already expressed in
    one common standardization chart.  The latter is the lowest-order
    time-level interface predictor used by non-iterative transient DDMs; it
    changes only the origin of the communicated defect and uses no endpoint
    target.  Flux compatibility is deliberately applied to the *defect* by
    the response wrapper, not to this absolute reference.
    """
    if kind not in STATE_TRACE_REFERENCE_KINDS:
        raise ValueError(f"unknown state-only velocity trace reference: {kind}")
    current, roles = normalized_velocity_dirichlet_trace_2d(
        example["physical"][:, :3], face_width=int(face_width),
        neighbor_faces=example["neighbor_faces"],
        valid=example["valid_tiles"],
    )
    if kind == TRACE_REFERENCE_LAGGED:
        return current, roles
    if "previous_previous" not in example or "previous_dt" not in example:
        raise ValueError("extrapolated trace reference requires two time levels")
    previous_tiles = example["cover"].tile(
        example["previous_previous"], padding_mode="replicate",
    )
    selected = example.get("training_tile_indices")
    if selected is not None:
        index = torch.as_tensor(
            selected, device=previous_tiles.device, dtype=torch.long,
        )
        previous_tiles = previous_tiles.index_select(0, index)
    previous, previous_roles = normalized_velocity_dirichlet_trace_2d(
        previous_tiles, face_width=int(face_width),
        neighbor_faces=example["neighbor_faces"],
        valid=example["valid_tiles"],
    )
    if not torch.equal(roles, previous_roles):
        raise RuntimeError("current and previous velocity traces use different roles")
    previous_dt = float(example["previous_dt"])
    ratio = (
        float(example["dt"]) / previous_dt
        if math.isfinite(previous_dt) and previous_dt > 1.0e-12 else 1.0
    )
    active = roles.repeat_interleave(2, dim=1)
    return (current + ratio * (current - previous)) * active, roles


class PointParticularWithIncfloResponse2D(nn.Module):
    """Add the source-audited unit-scale response to a Point particular."""

    supports_cached_particular = True
    postsolve_strong_trace = False
    flux_compatible_velocity_trace = True

    def __init__(
        self,
        particular: nn.Module,
        *,
        state_std: Tensor,
        projection_std: Tensor,
        dx: float,
        helmholtz_iterations: int = 160,
        projection_iterations: int = 320,
        tolerance: float = 1.0e-7,
        trace_reference_kind: str = TRACE_REFERENCE_LAGGED,
        response_output_kind: str = RESPONSE_OUTPUT_FIVE_STATE,
    ) -> None:
        super().__init__()
        state = torch.as_tensor(state_std, dtype=torch.float32).flatten()
        projection = torch.as_tensor(
            projection_std, dtype=torch.float32,
        ).flatten()
        if (
            state.numel() != 3 or projection.numel() != 2
            or bool(torch.any(state <= 0.0))
            or bool(torch.any(projection <= 0.0))
            or not math.isfinite(dx) or dx <= 0.0
            or min(helmholtz_iterations, projection_iterations) <= 0
            or not math.isfinite(tolerance) or tolerance <= 0.0
            or trace_reference_kind not in TRACE_REFERENCE_KINDS
            or response_output_kind not in RESPONSE_OUTPUT_KINDS
        ):
            raise ValueError("invalid incflo response controls")
        self.particular = particular
        self.register_buffer("state_std", state)
        self.register_buffer("projection_std", projection)
        self.dx = float(dx)
        self.helmholtz_iterations = int(helmholtz_iterations)
        self.projection_iterations = int(projection_iterations)
        self.tolerance = float(tolerance)
        self.trace_reference_kind = str(trace_reference_kind)
        self.response_output_kind = str(response_output_kind)

    @property
    def residual_scale(self) -> Tensor:
        return self.particular.residual_scale

    @property
    def physical_channels(self) -> int:
        return int(self.particular.physical_channels)

    @property
    def scalar_dim(self) -> int:
        return int(self.particular.scalar_dim)

    @property
    def projection_channels(self) -> int:
        return int(self.particular.projection_channels)

    @property
    def dirichlet_channels(self) -> int:
        return 2

    def compatible_port(self, port: Tensor, roles: Tensor) -> Tensor:
        """Project a normalized velocity trace onto zero total normal flux."""
        projected, _ = project_port_to_zero_net_flux(
            port, roles, state_std=self.state_std, dx=self.dx,
        )
        return projected

    def reference_trace(
        self,
        example: dict[str, object],
        *,
        face_width: int,
        cached_particular: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the checkpoint-declared target-free trace origin."""
        if self.trace_reference_kind == TRACE_REFERENCE_LOCAL_PREDICTION:
            initialized, roles = normalized_velocity_dirichlet_trace_2d(
                example["physical"][:, :3], face_width=int(face_width),
                neighbor_faces=example["neighbor_faces"],
                valid=example["valid_tiles"],
            )
            particular = cached_particular
            if particular is None:
                particular = self.particular(
                    example["physical"], example["scalar"],
                    torch.zeros_like(initialized), roles,
                )
            if particular.shape != (
                example["physical"].shape[0], 5,
                *example["physical"].shape[-2:],
            ):
                raise ValueError(
                    "local-prediction reference requires a five-state particular"
                )
            reference, reference_roles = normalized_velocity_dirichlet_trace_2d(
                particular[:, :3], face_width=int(face_width),
                neighbor_faces=example["neighbor_faces"],
                valid=example["valid_tiles"],
            )
            if not torch.equal(roles, reference_roles):
                raise RuntimeError("particular and endpoint face roles differ")
            return reference, roles
        return velocity_trace_reference_2d(
            example, face_width=int(face_width),
            kind=self.trace_reference_kind,
        )

    def homogeneous_response(
        self,
        physical: Tensor,
        scalar: Tensor,
        boundary_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        """Evaluate the unit-scale five-state homogeneous response."""
        if not bool(torch.any(boundary_defect != 0.0)):
            return physical.new_zeros(
                physical.shape[0], 5, *physical.shape[-2:],
            )
        compatible = self.compatible_port(boundary_defect, face_roles)
        dt = scalar[:, 1] * 0.01
        reynolds = torch.exp(scalar[:, 0]) * 10_000.0
        response, _ = source_audited_helmholtz_nodal_response(
            compatible, face_roles, physical[:, -1:],
            state_std=self.state_std, projection_std=self.projection_std,
            dt=dt, dx=self.dx, viscosity=reynolds.reciprocal(),
            flux_compatibility=False,
            helmholtz_iterations=self.helmholtz_iterations,
            projection_iterations=self.projection_iterations,
            tolerance=self.tolerance,
        )
        if self.response_output_kind == RESPONSE_OUTPUT_VELOCITY_ONLY:
            response = torch.cat((
                response[:, :2], torch.zeros_like(response[:, 2:]),
            ), dim=1)
        elif self.response_output_kind == RESPONSE_OUTPUT_VELOCITY_PROJECTION:
            response = torch.cat((
                response[:, :2], torch.zeros_like(response[:, 2:3]),
                response[:, 3:],
            ), dim=1)
        return response

    def compose_from_particular(
        self,
        particular: Tensor,
        physical: Tensor,
        scalar: Tensor,
        boundary_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        if particular.shape != (
            physical.shape[0], 5, *physical.shape[-2:]
        ):
            raise ValueError("incflo particular must contain five local states")
        return particular + self.homogeneous_response(
            physical, scalar, boundary_defect, face_roles,
        )

    def forward(
        self,
        physical: Tensor,
        scalar: Tensor,
        boundary_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        zero = torch.zeros_like(boundary_defect)
        particular = self.particular(physical, scalar, zero, face_roles)
        return self.compose_from_particular(
            particular, physical, scalar, boundary_defect, face_roles,
        )

    def target_defect(
        self,
        example: dict[str, object],
        *,
        face_width: int,
        cached_particular: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return the compatible target defect, roles, and trace reference."""
        target, roles = normalized_velocity_dirichlet_trace_2d(
            example["target_tiles"], face_width=face_width,
            neighbor_faces=example["neighbor_faces"],
            valid=example["valid_tiles"],
        )
        reference, reference_roles = self.reference_trace(
            example, face_width=face_width,
            cached_particular=cached_particular,
        )
        if not torch.equal(roles, reference_roles):
            raise RuntimeError("target and reference trace roles differ")
        active = roles.repeat_interleave(2, dim=1)
        defect = self.compatible_port((target - reference) * active, roles)
        declared_reference = (
            reference * active
            if self.trace_reference_kind == TRACE_REFERENCE_LOCAL_PREDICTION
            else self.compatible_port(reference * active, roles)
        )
        return defect, roles, declared_reference

    def oracle_prediction(
        self,
        example: dict[str, object],
        args,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Target-informed validation oracle; never used at inference."""
        initialized, roles = normalized_velocity_dirichlet_trace_2d(
            example["physical"][:, :3], face_width=int(args.face_width),
            neighbor_faces=example["neighbor_faces"],
            valid=example["valid_tiles"],
        )
        zero = torch.zeros_like(initialized)
        particular = self.particular(
            example["physical"], example["scalar"], zero, roles,
        )
        defect, target_roles, reference = self.target_defect(
            example, face_width=int(args.face_width),
            cached_particular=particular,
        )
        if not torch.equal(roles, target_roles):
            raise RuntimeError("particular and target face roles differ")
        prediction = self.compose_from_particular(
            particular, example["physical"], example["scalar"], defect, roles,
        )
        return prediction, defect, roles, particular, reference

    def native_sni_sweep(
        self,
        example: dict[str, object],
        args,
        current: Tensor,
        *,
        imposed_port: Tensor | None = None,
        reference_port: Tensor | None = None,
        pressure_reference_mean: Tensor | None = None,
        pressure_coarse_correction: bool = False,
        cached_particular: Tensor | None = None,
    ):
        """Apply one compatible-trace Schwarz refresh without trace overwrite."""
        if pressure_reference_mean is not None and not pressure_coarse_correction:
            raise ValueError(
                "a pressure reference requires the pressure coarse correction"
            )
        if reference_port is None:
            default_reference, roles = self.reference_trace(
                example, face_width=int(args.face_width),
                cached_particular=cached_particular,
            )
        else:
            _, roles = normalized_velocity_dirichlet_trace_2d(
                example["physical"][:, :3], face_width=int(args.face_width),
                neighbor_faces=example["neighbor_faces"],
                valid=example["valid_tiles"],
            )
            default_reference = reference_port
        port = (
            None if imposed_port is None
            else self.compatible_port(imposed_port, roles)
        )
        active = roles.repeat_interleave(2, dim=1)
        # The communicated *defect* must have zero net flux.  A model-owned
        # absolute reference need not: projecting it would move sweep zero away
        # from the frozen Local/Core prediction that defines this coordinate.
        reference = (
            default_reference * active
            if self.trace_reference_kind == TRACE_REFERENCE_LOCAL_PREDICTION
            else self.compatible_port(default_reference, roles)
        )
        # An undamped generic sweep exposes the raw realized proposal.  The
        # compatible projection precedes the declared outer relaxation.
        raw = native_dirichlet_sni_sweep_2d(
            self, example["physical"], example["scalar"], current,
            example["cover"], relaxation=1.0,
            face_width=int(args.face_width),
            neighbor_faces=example["neighbor_faces"],
            imposed_port=port, reference_port=reference,
            assembly=str(args.assembly), strong_trace=False,
            velocity_only=True,
            pressure_coarse_correction=pressure_coarse_correction,
            pressure_reference_mean=pressure_reference_mean,
            cached_particular=cached_particular,
            transmission_valid=example["valid_tiles"],
            correction_valid=example["valid_tiles"],
        )
        proposal = self.compatible_port(raw.next_port, roles)
        residual = proposal - raw.imposed_port
        next_port = (
            raw.imposed_port + float(args.relaxation) * residual
        ) * active
        defect_rms = (
            (residual.square() * active).sum()
            / active.sum().clamp_min(1.0)
        ).sqrt()
        return replace(
            raw, next_port=next_port, port_defect_rms=defect_rms,
            reference_port=(reference if reference is not None else raw.reference_port),
        )


def response_metadata(
    *,
    dx: float,
    helmholtz_iterations: int,
    projection_iterations: int,
    tolerance: float,
    trace_reference_kind: str = TRACE_REFERENCE_LAGGED,
    response_output_kind: str = RESPONSE_OUTPUT_FIVE_STATE,
) -> dict[str, object]:
    """Return the complete provenance contract stored in a checkpoint."""
    if trace_reference_kind not in TRACE_REFERENCE_KINDS:
        raise ValueError("unknown incflo trace reference")
    if response_output_kind not in RESPONSE_OUTPUT_KINDS:
        raise ValueError("unknown incflo response-output contract")
    corrected_state = {
        RESPONSE_OUTPUT_FIVE_STATE: ["u", "v", "p", "gpx", "gpy"],
        RESPONSE_OUTPUT_VELOCITY_ONLY: ["u", "v"],
        RESPONSE_OUTPUT_VELOCITY_PROJECTION: ["u", "v", "gpx", "gpy"],
    }[response_output_kind]
    return {
        "kind": KIND,
        "enabled": True,
        "dx": float(dx),
        "response_scale": 1.0,
        "helmholtz_iterations": int(helmholtz_iterations),
        "projection_iterations": int(projection_iterations),
        "tolerance": float(tolerance),
        "response_domain": "all_regular_valid",
        "trace_coordinate": TRACE_COORDINATE,
        "trace_reference_kind": str(trace_reference_kind),
        "response_output_kind": str(response_output_kind),
        "flux_compatibility_projection": True,
        "postsolve_strong_trace": False,
        "corrected_state": corrected_state,
        "learned_parameter_count": 0,
        "target_used_at_inference": False,
        "discretization_qualification": (
            "unit-scale component Helmholtz plus AMReX Q1 inflow nodal "
            "projection on the all-regular volume-penalized domain; retained "
            f"response outputs: {response_output_kind}"
        ),
    }
