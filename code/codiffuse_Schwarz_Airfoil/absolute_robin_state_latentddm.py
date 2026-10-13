"""Airfoil LatentDDM that communicates through the frozen solver's state input.

The first call is the strict-local Point/Robin prediction.  Relative attention
then produces a low-order correction to the local previous state, and the same
frozen operator is called again.  This preserves the reusable local-compute
claim while avoiding reliance on a Robin response direction that may be flat.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from codiffuse_Schwarz.boundary_trace_basis import decode_trace_code_2d
from codiffuse_Schwarz.field_token_composer import FieldTokenComposition

from .absolute_robin_latentddm import AirfoilRobinLatentDDM
from .absolute_robin_sni import absolute_robin_port_2d
from .airfoil_trace_sni import assemble_subdomains_2d


Tensor = torch.Tensor
CONTRACT = "airfoil_absolute_robin_state_latentddm_v1"


@dataclass(frozen=True)
class AirfoilRobinStateLatentOutput:
    """Auditable result of one message between two frozen local calls."""

    field: Tensor
    local_field: Tensor
    final_tiles: Tensor
    local_tiles: Tensor
    initial_port: Tensor
    corrected_port: Tensor
    proposed_neighbor_port: Tensor
    realized_local_port: Tensor
    face_roles: Tensor
    standardized_correction_code: Tensor
    correction_code: Tensor
    state_input_correction: Tensor
    communication: FieldTokenComposition


class AirfoilRobinStateLatentDDM(AirfoilRobinLatentDDM):
    """Correct local state conditions and reevaluate a frozen local operator."""

    def __init__(
        self,
        local_solver,
        *,
        width: int = 64,
        heads: int = 4,
        depth: int = 4,
        trace_modes: int = 4,
        spatial_bins: int = 4,
        code_scale: Tensor | None = None,
        correction_limit: float = 0.5,
        recompute_port: bool = True,
        centered_communication: bool = False,
        state_chart: str = "collar",
        state_bins: int = 4,
    ) -> None:
        if not math.isfinite(correction_limit) or correction_limit <= 0.0:
            raise ValueError("state-input correction limit must be positive")
        if state_chart not in ("collar", "tile") or state_bins <= 0:
            raise ValueError("unknown state-input response chart")
        super().__init__(
            local_solver,
            width=width,
            heads=heads,
            depth=depth,
            trace_modes=trace_modes,
            spatial_bins=spatial_bins,
            code_scale=code_scale,
            communication_mode=(
                "centered_delta" if centered_communication else "full_delta"
            ),
        )
        self.correction_limit = float(correction_limit)
        self.recompute_port = bool(recompute_port)
        self.centered_communication = bool(centered_communication)
        self.state_chart = str(state_chart)
        self.state_bins = int(state_bins)
        if self.state_chart == "tile" and self.code_width != 3 * self.state_bins**2:
            raise ValueError(
                "tile-state chart requires code_width = 3 * state_bins**2"
            )

    def _decode_state_correction(
        self, standardized: Tensor, roles: Tensor, valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        code = standardized * self.code_scale.to(standardized)
        if self.state_chart == "tile":
            if valid is None or valid.shape != roles[:, :1].shape:
                raise ValueError("tile-state correction requires a valid-cell mask")
            coarse = code.reshape(-1, 3, self.state_bins, self.state_bins)
            dense = F.interpolate(
                coarse, size=roles.shape[-2:], mode="bilinear",
                align_corners=False,
            )
            return code, self.correction_limit * torch.tanh(
                dense / self.correction_limit,
            ) * valid.to(dense)
        faces = decode_trace_code_2d(
            code, roles, channels=3, modes=self.trace_modes,
        ).reshape(roles.shape[0], 4, 3, *roles.shape[-2:])
        role_count = roles.sum(dim=1, keepdim=True).clamp_min(1.0)
        active = (roles.sum(dim=1, keepdim=True) > 0).to(faces)
        dense = self.correction_limit * torch.tanh(
            faces.sum(dim=1) / role_count / self.correction_limit,
        ) * active
        return code, dense

    def forward(
        self,
        example: dict[str, object],
        *,
        graph_mask: Tensor | None = None,
    ) -> AirfoilRobinStateLatentOutput:
        physical = example["physical"]
        scalar = example["scalar"]
        cover = example["cover"]
        grid_shape = tuple(int(value) for value in cover.grid_shape)
        current_tiles = cover.tile(
            example["previous"], padding_mode="replicate",
        )
        kwargs = self._port_kwargs(example)
        initial_port, roles = absolute_robin_port_2d(current_tiles, **kwargs)
        local = self.local_solver(physical, scalar, initial_port, roles)
        local_field = assemble_subdomains_2d(
            cover, local, example["previous"].shape[0], "normalized_overlap",
        )
        exchange = assemble_subdomains_2d(
            cover, local, example["previous"].shape[0], "disjoint_core",
        )
        consensus = cover.tile(exchange, padding_mode="replicate")
        realized_port, _ = absolute_robin_port_2d(local, **kwargs)
        proposed_port, _ = absolute_robin_port_2d(consensus, **kwargs)
        tokens = self._tokens(
            physical, scalar, local, initial_port, realized_port,
            proposed_port, roles, grid_shape,
        )
        communication = self.communication(
            tokens, grid_shape=grid_shape, graph_mask=graph_mask,
            centered=self.centered_communication,
        )
        standardized = communication.correction_code.reshape(
            local.shape[0], self.code_width,
        )
        correction_code, state_correction = self._decode_state_correction(
            standardized, roles, physical[:, -1:],
        )
        corrected_physical = physical.clone()
        corrected_physical[:, :3] += state_correction
        if self.recompute_port:
            corrected_current = current_tiles + state_correction
            corrected_port, _ = absolute_robin_port_2d(
                corrected_current, **kwargs,
            )
        else:
            corrected_port = initial_port
        final_tiles = self.local_solver(
            corrected_physical, scalar, corrected_port, roles,
        )
        field = assemble_subdomains_2d(
            cover, final_tiles, example["previous"].shape[0],
            "normalized_overlap",
        )
        return AirfoilRobinStateLatentOutput(
            field=field,
            local_field=local_field,
            final_tiles=final_tiles,
            local_tiles=local,
            initial_port=initial_port,
            corrected_port=corrected_port,
            proposed_neighbor_port=proposed_port,
            realized_local_port=realized_port,
            face_roles=roles,
            standardized_correction_code=standardized,
            correction_code=correction_code,
            state_input_correction=state_correction,
            communication=communication,
        )
