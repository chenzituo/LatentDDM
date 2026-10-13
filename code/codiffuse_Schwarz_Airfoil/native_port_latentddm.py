"""Two-round LatentDDM for any strict-local Airfoil face-port contract.

The first call is exactly the configured Local/Core solve.  Tokens summarize
only crop-local conditions, tentative fields, and target-free interface
defects.  Relative-geometry attention predicts one low-order face-port update,
after which the byte-identical frozen local operator is called once more.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from codiffuse_Schwarz.boundary_trace_basis import decode_trace_code_2d

from .absolute_robin_latentddm import AirfoilRobinLatentDDM
from .train_absolute_robin_bvp import (
    active_robin_mask,
    declared_reference_port_and_roles,
    defect_sni_sweep,
    realized_port_and_roles,
)


Tensor = torch.Tensor
CONTRACT = "airfoil_native_port_latentddm_v1"


@dataclass(frozen=True)
class AirfoilNativePortLatentOutput:
    field: Tensor
    local_field: Tensor
    auxiliary_field: Tensor | None
    local_auxiliary_field: Tensor | None
    final_tiles: Tensor
    local_tiles: Tensor
    initial_port: Tensor
    corrected_port: Tensor
    proposed_neighbor_port: Tensor
    realized_local_port: Tensor
    reference_port: Tensor | None
    face_roles: Tensor
    standardized_correction_code: Tensor
    correction_code: Tensor
    interface_rms: Tensor
    port_defect_rms: Tensor
    realization_rms: Tensor
    communication: object


class AirfoilNativePortLatentDDM(AirfoilRobinLatentDDM):
    """Amortize one target-free native transmission fixed-point search."""

    def forward(
        self,
        example: dict[str, object],
        *,
        graph_mask: Tensor | None = None,
    ) -> AirfoilNativePortLatentOutput:
        physical = example["physical"]
        scalar = example["scalar"]
        cover = example["cover"]
        configuration = example["configuration"]
        grid_shape = tuple(int(value) for value in cover.grid_shape)
        batch = int(example["previous"].shape[0])
        relaxation = float(configuration.relaxation)
        if not 0.0 < relaxation <= 1.5:
            raise ValueError("LatentDDM received an invalid SNI relaxation")

        initial_reference = None
        if configuration.port_coordinate == "lagged_defect":
            initial_reference, _ = declared_reference_port_and_roles(
                self.local_solver, example, configuration,
            )
        initial = defect_sni_sweep(
            self.local_solver, example, configuration, example["previous"],
            imposed_port=None, reference_port=initial_reference,
            pressure_coarse_correction=False,
        )
        local = initial.local_candidates[:, :3]
        local_field = initial.field[:, :3]
        initial_port = initial.imposed_port
        reference_port = getattr(initial, "reference_port", None)
        realized_absolute, roles = realized_port_and_roles(
            local, example, configuration,
        )
        realized_coordinate = (
            realized_absolute - reference_port
            if reference_port is not None else realized_absolute
        )
        # Undo the declared relaxation so the token sees the neighbor proposal
        # itself rather than an arbitrary fraction of it.
        proposed_port = initial_port + (
            initial.next_port - initial_port
        ) / relaxation
        tokens = self._tokens(
            physical, scalar, local, initial_port, realized_coordinate,
            proposed_port, roles, grid_shape,
        )
        communication = self.communication(
            tokens,
            grid_shape=grid_shape,
            graph_mask=graph_mask,
            centered=self.communication_mode == "centered_delta",
        )
        standardized_tokens = (
            communication.correction_code
            if self.communication_mode == "centered_delta" else
            communication.global_code
        )
        standardized = standardized_tokens.reshape(
            local.shape[0], self.code_width,
        )
        decoded_code = (
            standardized * self.code_scale.to(standardized)
            + self.code_center.to(standardized)
        )
        decoded_port = decode_trace_code_2d(
            decoded_code, roles, channels=self.channels_per_face,
            modes=self.trace_modes,
        )
        active = active_robin_mask(roles, configuration)
        corrected_port = (
            decoded_port if self.communication_mode == "full_absolute"
            else initial_port + decoded_port
        ) * active
        final = defect_sni_sweep(
            self.local_solver, example, configuration, local_field,
            imposed_port=corrected_port,
            reference_port=reference_port,
            pressure_reference_mean=initial.pressure_reference_mean,
            pressure_coarse_correction=bool(getattr(
                configuration, "pressure_coarse_correction", False,
            )),
        )
        if local.shape[0] != batch * math.prod(grid_shape):
            raise RuntimeError("native LatentDDM changed the cover batch")
        return AirfoilNativePortLatentOutput(
            field=final.field[:, :3],
            local_field=local_field,
            auxiliary_field=getattr(final, "auxiliary_field", None),
            local_auxiliary_field=getattr(initial, "auxiliary_field", None),
            final_tiles=final.local_candidates[:, :3],
            local_tiles=local,
            initial_port=initial_port,
            corrected_port=corrected_port,
            proposed_neighbor_port=proposed_port,
            realized_local_port=realized_coordinate,
            reference_port=reference_port,
            face_roles=roles,
            standardized_correction_code=standardized,
            correction_code=decoded_code,
            interface_rms=final.interface_rms,
            port_defect_rms=final.port_defect_rms,
            realization_rms=final.realization_rms,
            communication=communication,
        )
