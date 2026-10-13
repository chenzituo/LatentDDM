"""Two-level Cartesian layout plus experimental generative integration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .adapter import PhysicalTransition, PhysicsAdapter
from .cover import CartesianCover
from .model import CoDiffuseOperator
from .schwarz import FlowSample, PreparedLevel, _tile_time, local_velocity, schwarz_velocity


Tensor = torch.Tensor


def restrict(field: Tensor, factor: int) -> Tensor:
    return F.avg_pool2d(field, factor, factor)


def prolong(field: Tensor, shape: tuple[int, int]) -> Tensor:
    return F.interpolate(field, size=shape, mode="bilinear", align_corners=False)


def _refinement_weight(
    indicator: Tensor, cover: CartesianCover, fraction: float,
) -> tuple[Tensor, Tensor]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fine fraction must lie in (0,1]")
    patches = cover.tile(indicator)
    h, c = cover.halo, cover.core
    scores = patches[:, :, h:h + c, h:h + c].flatten(2).quantile(.95, dim=2)[:, 0]
    scores = scores.reshape(indicator.shape[0], *cover.grid_shape)
    count = scores.shape[-2] * scores.shape[-1]
    selected_count = max(1, min(count, round(fraction * count)))
    threshold = scores.flatten(1).topk(selected_count, dim=1).values[:, -1, None, None]
    selected = (scores >= threshold).to(indicator.dtype)
    selected = F.max_pool2d(selected[:, None], 3, stride=1, padding=1)[:, 0]
    local = torch.zeros(
        indicator.shape[0] * count, 1, cover.tile_size, cover.tile_size,
        device=indicator.device, dtype=indicator.dtype,
    )
    local[:, :, h:h + c, h:h + c] = selected.reshape(-1, 1, 1, 1)
    binary = cover.assemble_ras(local, indicator.shape[0])
    weight = binary
    for _ in range(2):
        weight = F.avg_pool2d(weight, 7, stride=1, padding=3)
    return binary, weight


@dataclass
class TwoLevelHierarchy:
    fine: PreparedLevel
    coarse: PreparedLevel
    refinement_mask: Tensor
    refinement_weight: Tensor
    factor: int

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.fine.shape


def build_two_level_hierarchy(
    adapter: PhysicsAdapter, transition: PhysicalTransition,
    core: int = 32, halo: int = 4, factor: int = 2,
    fine_fraction: float = .35, seed: int = 0,
) -> TwoLevelHierarchy:
    """Build a model-owned hierarchy independent of a numerical solver mesh."""
    height, width = transition.spatial_shape
    fine_cover = CartesianCover(height, width, core, halo).random_shift(seed)
    coarse_transition = adapter.coarsen(transition, factor)
    coarse_height, coarse_width = coarse_transition.spatial_shape
    coarse_cover = CartesianCover(
        coarse_height, coarse_width, core, halo
    ).random_shift(seed + 1_000_003)
    indicator = adapter.refinement_indicator(transition)
    mask, weight = _refinement_weight(indicator, fine_cover, fine_fraction)
    return TwoLevelHierarchy(
        fine=PreparedLevel(transition, adapter, fine_cover, 1.0),
        coarse=PreparedLevel(coarse_transition, adapter, coarse_cover, 1.0 / factor),
        refinement_mask=mask, refinement_weight=weight, factor=factor,
    )


def multilevel_velocity(
    model: CoDiffuseOperator, state: Tensor, hierarchy: TwoLevelHierarchy,
    flow_time: Tensor, schwarz_iterations: int = 1, relaxation: float = .7,
) -> tuple[Tensor, dict[str, Tensor]]:
    coarse_state = restrict(state, hierarchy.factor)
    if schwarz_iterations > 1:
        fine_velocity, fine_residual = schwarz_velocity(
            model, state, hierarchy.fine, flow_time, schwarz_iterations, relaxation
        )
        coarse_velocity, coarse_residual = schwarz_velocity(
            model, coarse_state, hierarchy.coarse, flow_time,
            schwarz_iterations, relaxation,
        )
    else:
        fine_velocity = local_velocity(model, state, hierarchy.fine, flow_time)[0]
        coarse_velocity = local_velocity(
            model, coarse_state, hierarchy.coarse, flow_time
        )[0]
        fine_residual = state.new_empty(0)
        coarse_residual = state.new_empty(0)
    prolonged = prolong(coarse_velocity, state.shape[-2:])
    result = prolonged + hierarchy.refinement_weight * (fine_velocity - prolonged)
    return result, {
        "fine_trace_residual": fine_residual,
        "coarse_trace_residual": coarse_residual,
    }


def integrate_multilevel_flow(
    model: CoDiffuseOperator, hierarchy: TwoLevelHierarchy, initial_noise: Tensor,
    steps: int = 8, schwarz_iterations: int = 1, relaxation: float = .7,
) -> FlowSample:
    if steps <= 0 or tuple(initial_noise.shape) != hierarchy.shape:
        raise ValueError("noise must match hierarchy shape and steps must be positive")
    state, residuals = initial_noise, []
    for step in range(steps):
        time = torch.full(
            (state.shape[0],), (step + .5) / steps,
            device=state.device, dtype=state.dtype,
        )
        velocity, details = multilevel_velocity(
            model, state, hierarchy, time, schwarz_iterations, relaxation
        )
        if details["fine_trace_residual"].numel():
            residuals.append(details["fine_trace_residual"])
            residuals.append(details["coarse_trace_residual"])
        state = state + velocity / steps
    return FlowSample(
        state, torch.cat(residuals) if residuals else state.new_empty(0)
    )


def pnp_multilevel_sample(
    model: CoDiffuseOperator, hierarchy: TwoLevelHierarchy,
    fine_noise_sequence: Tensor, relaxation: float = .7,
) -> FlowSample:
    """Coarse/fine endpoint consensus inside every denoise-renoise iteration."""
    if fine_noise_sequence.ndim != 5 \
            or tuple(fine_noise_sequence.shape[1:]) != hierarchy.shape:
        raise ValueError("fine noise sequence shape does not match hierarchy")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must lie in (0,1]")
    fine, coarse = hierarchy.fine, hierarchy.coarse
    channels = hierarchy.shape[1]
    fine_tiles = torch.zeros(
        fine.batch_size * fine.cover.tile_count, channels,
        fine.cover.tile_size, fine.cover.tile_size,
        device=fine_noise_sequence.device, dtype=fine_noise_sequence.dtype,
    )
    coarse_tiles = torch.zeros(
        coarse.batch_size * coarse.cover.tile_count, channels,
        coarse.cover.tile_size, coarse.cover.tile_size,
        device=fine_noise_sequence.device, dtype=fine_noise_sequence.dtype,
    )
    residuals = []
    steps = fine_noise_sequence.shape[0]
    for step in range(steps):
        fine_global = fine.cover.assemble_weighted(fine_tiles, fine.batch_size)
        coarse_global = coarse.cover.assemble_weighted(coarse_tiles, coarse.batch_size)
        prolonged = prolong(coarse_global, fine_global.shape[-2:])
        composite = prolonged + hierarchy.refinement_weight * (fine_global - prolonged)
        fine_projected = fine.cover.tile(composite)
        coarse_projected = coarse.cover.tile(restrict(composite, hierarchy.factor))
        fine_mismatch = fine_projected - fine_tiles
        coarse_mismatch = coarse_projected - coarse_tiles
        residuals.append(torch.sqrt(
            fine_mismatch.square().mean() + coarse_mismatch.square().mean()
        ))
        fine_tiles = fine_tiles + relaxation * fine_mismatch
        coarse_tiles = coarse_tiles + relaxation * coarse_mismatch

        value = (step + 1.0) / steps
        time = torch.full(
            (fine.batch_size,), value,
            device=fine_noise_sequence.device, dtype=fine_noise_sequence.dtype,
        )
        fine_physical, fine_scalar = fine.conditions(time)
        coarse_physical, coarse_scalar = coarse.conditions(time)
        fine_noise = fine.cover.tile(fine_noise_sequence[step])
        coarse_noise = coarse.cover.tile(
            restrict(fine_noise_sequence[step], hierarchy.factor)
        )
        fine_noisy = (1.0 - value) * fine_noise + value * fine_tiles
        coarse_noisy = (1.0 - value) * coarse_noise + value * coarse_tiles
        fine_velocity = model(
            fine_noisy, fine_physical, fine_scalar,
            _tile_time(time, fine.cover.tile_count),
        )
        coarse_velocity = model(
            coarse_noisy, coarse_physical, coarse_scalar,
            _tile_time(time, coarse.cover.tile_count),
        )
        fine_tiles = model.project_endpoint(
            fine_noisy + (1.0 - value) * fine_velocity,
            fine_physical, fine_scalar, fine.cover.tile_count,
        )
        coarse_tiles = model.project_endpoint(
            coarse_noisy + (1.0 - value) * coarse_velocity,
            coarse_physical, coarse_scalar, coarse.cover.tile_count,
        )
    fine_global = fine.cover.assemble_weighted(fine_tiles, fine.batch_size)
    coarse_global = coarse.cover.assemble_weighted(coarse_tiles, coarse.batch_size)
    prolonged = prolong(coarse_global, fine_global.shape[-2:])
    endpoint = prolonged + hierarchy.refinement_weight * (fine_global - prolonged)
    return FlowSample(endpoint, torch.stack(residuals))
