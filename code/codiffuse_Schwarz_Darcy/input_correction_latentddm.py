"""Darcy LatentDDM that communicates through local Dirichlet inputs."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from codiffuse_Schwarz.boundary_trace_basis import decode_trace_code_2d
from codiffuse_Schwarz.field_token_composer import (
    CenteredFieldTokenTransformer,
    FieldTokenComposition,
)

from .adapter import FaceDarcyAdapter, face_condition_features
from .field_token_composer import DarcyFieldTokenEncoder, coarse_spatial_moments
from .train_residual_port_chart import local_model_call


Tensor = torch.Tensor
class DarcyInputCorrectionLatentDDM(nn.Module):
    """Encode tentative local solutions and predict one boundary update."""

    contract = "darcy_input_correction_latentddm_v1"

    def __init__(
        self,
        physical_channels: int,
        scalar_dim: int,
        *,
        token_width: int = 128,
        heads: int = 4,
        communication_depth: int = 4,
        trace_modes: int = 2,
        spatial_token_bins: int = 0,
    ) -> None:
        super().__init__()
        if trace_modes not in (2, 4, 8):
            raise ValueError("Darcy trace modes must be 2, 4, or 8")
        if spatial_token_bins not in (0, 2, 3, 4):
            raise ValueError("spatial token bins must be 0, 2, 3, or 4")
        self.trace_modes = int(trace_modes)
        self.trace_code_width = 4 * self.trace_modes
        self.spatial_token_bins = int(spatial_token_bins)
        self.encoder = DarcyFieldTokenEncoder(
            1, physical_channels, scalar_dim, token_width,
        )
        summarized_channels = physical_channels + 1
        spatial_width = 2 * summarized_channels * self.spatial_token_bins**2
        self.spatial_encoder = (
            nn.Sequential(
                nn.LayerNorm(spatial_width),
                nn.Linear(spatial_width, token_width),
                nn.SiLU(),
                nn.Linear(token_width, token_width, bias=False),
            )
            if self.spatial_token_bins else None
        )
        if self.spatial_encoder is not None:
            final = self.spatial_encoder[-1]
            assert isinstance(final, nn.Linear)
            nn.init.normal_(final.weight, std=1.0e-3)
        self.communication = CenteredFieldTokenTransformer(
            token_width, self.trace_code_width, heads, dimension=2,
            depth=communication_depth,
        )
        self.communication_depth = communication_depth

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())

    def forward(
        self,
        local: Tensor,
        physical: Tensor,
        scalar: Tensor,
        valid: Tensor,
        grid_shape: tuple[int, int],
    ) -> FieldTokenComposition:
        tile_count = math.prod(grid_shape)
        if local.shape[0] % tile_count:
            raise ValueError("Darcy local fields and grid shape disagree")
        tokens = self.encoder(local, physical, scalar, valid, grid_shape)
        if self.spatial_encoder is not None:
            summarized = torch.cat((physical, local), dim=1)
            spatial = coarse_spatial_moments(
                summarized, valid, self.spatial_token_bins,
            ).flatten(1)
            tokens = tokens + self.spatial_encoder(spatial).reshape_as(tokens)
        return self.communication(tokens, grid_shape=grid_shape)


@dataclass(frozen=True)
class DarcyBoundaryInputContext:
    physical: Tensor
    scalar: Tensor
    face_mask: Tensor
    face_value: Tensor
    face_role: Tensor
    domain_mask: Tensor
    valid: Tensor
    roles: Tensor
    target_delta: Tensor
    cover: object
    batch_size: int
    trace_modes: int


def boundary_input_context(
    adapter: FaceDarcyAdapter,
    batch: dict[str, Tensor],
    conditions,
    cover,
    *,
    include_target: bool = True,
    trace_modes: int = 2,
) -> DarcyBoundaryInputContext:
    """Build the homogeneous initial trace and state-only training target."""
    roles = (
        (conditions.dirichlet_face_role >= 1.5)
        & (conditions.dirichlet_face_mask > .5)
    ).to(conditions.physical.dtype)
    # This optional supervised coordinate is extracted only from the solution
    # field.  It does not evaluate a Darcy residual or a flux law.
    if include_target:
        target_tiles = cover.tile(
            batch["solution"].to(conditions.physical.dtype),
        )
        target_face_value = target_tiles.expand(-1, 4, -1, -1)
        target_delta = (
            (target_face_value - conditions.dirichlet_face_value)
            / float(adapter.statistics.bc_std)
        ) * roles
    else:
        target_delta = torch.zeros_like(roles)
    if trace_modes not in (2, 4, 8):
        raise ValueError("Darcy trace modes must be 2, 4, or 8")
    return DarcyBoundaryInputContext(
        physical=conditions.physical,
        scalar=conditions.scalar,
        face_mask=conditions.dirichlet_face_mask,
        face_value=conditions.dirichlet_face_value,
        face_role=conditions.dirichlet_face_role,
        domain_mask=conditions.mask,
        valid=conditions.valid,
        roles=roles,
        target_delta=target_delta,
        cover=cover,
        batch_size=int(batch["solution"].shape[0]),
        trace_modes=int(trace_modes),
    )


def corrected_local_conditions(
    adapter: FaceDarcyAdapter,
    context: DarcyBoundaryInputContext,
    correction_code: Tensor,
) -> tuple[Tensor, Tensor]:
    """Decode a normalized face-value update into the local model input."""
    flat_code = correction_code.reshape(-1, correction_code.shape[-1])
    delta = decode_trace_code_2d(
        flat_code, context.roles, channels=1,
        modes=context.trace_modes,
    )
    physical = corrected_local_conditions_from_delta(
        adapter, context, delta,
    )
    return physical, delta


def corrected_local_conditions_from_delta(
    adapter: FaceDarcyAdapter,
    context: DarcyBoundaryInputContext,
    delta: Tensor,
) -> Tensor:
    """Apply a dense normalized face-value update to local conditions."""
    if delta.shape != context.face_value.shape:
        raise ValueError("dense Darcy boundary update does not align")
    delta = delta * context.roles
    face_value = (
        context.face_value
        + delta * float(adapter.statistics.bc_std)
    )
    count = context.face_mask.sum(1, keepdim=True)
    bc_value = (
        (face_value * context.face_mask).sum(1, keepdim=True)
        / count.clamp_min(1.0)
    )
    physical = context.physical.clone()
    # The first 12 generic Darcy channels end in cellwise BC value.  The next
    # 16 channels are the directional face features; tile validity is last.
    physical[:, 4:5] = (count > .5).to(physical) * context.domain_mask
    physical[:, 5:6] = (
        (bc_value - float(adapter.statistics.bc_mean))
        / float(adapter.statistics.bc_std)
    ) * context.domain_mask
    physical[:, 12:28] = face_condition_features(
        context.face_mask, face_value, context.face_role,
        context.domain_mask, adapter.statistics,
    )
    return physical


def solve_corrected_boundary_input(
    local_operator: nn.Module,
    adapter: FaceDarcyAdapter,
    context: DarcyBoundaryInputContext,
    correction_code: Tensor,
) -> tuple[Tensor, Tensor]:
    physical, delta = corrected_local_conditions(
        adapter, context, correction_code,
    )
    local = local_model_call(
        local_operator, physical, context.scalar, context.cover,
    )
    return local * context.domain_mask * context.valid, delta


def solve_corrected_boundary_trace(
    local_operator: nn.Module,
    adapter: FaceDarcyAdapter,
    context: DarcyBoundaryInputContext,
    delta: Tensor,
) -> Tensor:
    """Evaluate the frozen local BVP map for a dense boundary update."""
    physical = corrected_local_conditions_from_delta(adapter, context, delta)
    local = local_model_call(
        local_operator, physical, context.scalar, context.cover,
    )
    return local * context.domain_mask * context.valid
