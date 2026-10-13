"""Experimental Schwarz communication embedded in Flow integration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .adapter import PhysicalTransition, PhysicsAdapter
from .cover import CartesianCover
from .model import CoDiffuseOperator


Tensor = torch.Tensor


@dataclass
class PreparedLevel:
    transition: PhysicalTransition
    adapter: PhysicsAdapter
    cover: CartesianCover
    level_scale: float = 1.0

    @property
    def batch_size(self) -> int:
        return self.transition.batch_size

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(self.transition.current.shape)  # type: ignore[return-value]

    def conditions(self, flow_time: Tensor) -> tuple[Tensor, Tensor]:
        condition = self.adapter.conditions(
            self.transition, flow_time, self.level_scale
        )
        return condition.tiled(self.cover)


@dataclass
class FlowSample:
    endpoint: Tensor
    overlap_residuals: Tensor


def _tile_time(time: Tensor, tile_count: int) -> Tensor:
    return time[:, None].expand(-1, tile_count).reshape(-1)


def local_velocity(
    model: CoDiffuseOperator, state: Tensor, level: PreparedLevel,
    flow_time: Tensor, weighted_writeback: bool = False,
) -> tuple[Tensor, Tensor]:
    physical, scalar = level.conditions(flow_time)
    local_state = level.cover.tile(state)
    velocity_tiles = model(
        local_state, physical, scalar, _tile_time(flow_time, level.cover.tile_count)
    )
    assemble = level.cover.assemble_weighted if weighted_writeback else level.cover.assemble_ras
    return assemble(velocity_tiles, level.batch_size), velocity_tiles


def schwarz_velocity(
    model: CoDiffuseOperator, state: Tensor, level: PreparedLevel,
    flow_time: Tensor, iterations: int = 2, relaxation: float = .7,
) -> tuple[Tensor, Tensor]:
    """Repeat local solves while relaxing duplicate clean-endpoint traces.

    Flow time and Schwarz iteration are separate clocks. The local Flow state is
    held at one generative time while endpoint traces are exchanged.
    """
    if iterations <= 0 or not 0.0 < relaxation <= 1.0:
        raise ValueError("iterations must be positive and relaxation in (0,1]")
    physical, scalar = level.conditions(flow_time)
    local_state = level.cover.tile(state)
    valid = level.cover.valid_tiles(level.batch_size, state.device, state.dtype)
    h, c = level.cover.halo, level.cover.core
    owned = torch.zeros_like(valid)
    owned[:, :, h:h + c, h:h + c] = 1.0
    trace = valid * (1.0 - owned)
    tile_time = _tile_time(flow_time, level.cover.tile_count)
    residuals = []
    velocity_tiles = None
    for iteration in range(iterations):
        velocity_tiles = model(local_state, physical, scalar, tile_time)
        endpoint = local_state + (1.0 - tile_time)[:, None, None, None] * velocity_tiles
        consensus = level.cover.assemble_weighted(endpoint, level.batch_size)
        mismatch = level.cover.tile(consensus) - endpoint
        denominator = trace.sum().clamp_min(1.0) * state.shape[1]
        residuals.append(((mismatch.square() * trace).sum() / denominator).sqrt())
        if iteration + 1 < iterations:
            local_state = local_state + relaxation * mismatch * trace
    assert velocity_tiles is not None
    velocity = level.cover.assemble_ras(velocity_tiles, level.batch_size)
    return velocity, torch.stack(residuals)


def _project_global_endpoint(
    model: CoDiffuseOperator, endpoint: Tensor, level: PreparedLevel,
) -> Tensor:
    one = torch.ones(
        level.batch_size, device=endpoint.device, dtype=endpoint.dtype
    )
    physical, scalar = level.conditions(one)
    corrected = model.project_endpoint(
        level.cover.tile(endpoint), physical, scalar, level.cover.tile_count
    )
    return level.cover.assemble_weighted(corrected, level.batch_size)


def integrate_flow(
    model: CoDiffuseOperator, level: PreparedLevel, initial_noise: Tensor,
    steps: int = 8, schwarz_iterations: int = 1, relaxation: float = .7,
    project_endpoint: bool = True,
) -> FlowSample:
    """Integrate the probability-flow ODE with optional trace communication."""
    if steps <= 0 or tuple(initial_noise.shape) != level.shape:
        raise ValueError("positive steps and noise matching the transition are required")
    state = initial_noise
    residuals = []
    for step in range(steps):
        time = torch.full(
            (level.batch_size,), (step + .5) / steps,
            device=state.device, dtype=state.dtype,
        )
        if schwarz_iterations > 1:
            velocity, current = schwarz_velocity(
                model, state, level, time, schwarz_iterations, relaxation
            )
            residuals.append(current)
        else:
            velocity, _ = local_velocity(model, state, level, time)
        state = state + velocity / steps
    if project_endpoint:
        state = _project_global_endpoint(model, state, level)
    stacked = torch.cat(residuals) if residuals else state.new_empty(0)
    return FlowSample(state, stacked)


def pnp_schwarz_sample(
    model: CoDiffuseOperator, level: PreparedLevel, noise_sequence: Tensor,
    relaxation: float = .7,
) -> FlowSample:
    """Project clean endpoints, then re-noise onto the next Flow path.

    `noise_sequence[k]` is a hierarchy-consistent global Gaussian realization.
    Projection is applied to clean local endpoint estimates; the projected
    endpoint is then mixed with noise at the next Flow time. This is the
    transferable PnP-Flow/Schwarz mechanism used by CoDiffuse.
    """
    if noise_sequence.ndim != 5 or tuple(noise_sequence.shape[1:]) != level.shape:
        raise ValueError("noise sequence must have shape [steps,B,C,H,W]")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must lie in (0,1]")
    steps = noise_sequence.shape[0]
    tile_shape = (
        level.batch_size * level.cover.tile_count,
        level.transition.current.shape[1], level.cover.tile_size, level.cover.tile_size,
    )
    endpoint_tiles = torch.zeros(
        tile_shape, device=noise_sequence.device, dtype=noise_sequence.dtype
    )
    valid = level.cover.valid_tiles(
        level.batch_size, noise_sequence.device, noise_sequence.dtype
    )
    residuals = []
    for step in range(steps):
        consensus = level.cover.assemble_weighted(endpoint_tiles, level.batch_size)
        projected = level.cover.tile(consensus)
        mismatch = (projected - endpoint_tiles) * valid
        residuals.append(
            (mismatch.square().sum() / valid.sum().clamp_min(1.0)).sqrt()
        )
        endpoint_tiles = endpoint_tiles + relaxation * mismatch

        time_value = (step + 1.0) / steps
        time = torch.full(
            (level.batch_size,), time_value,
            device=noise_sequence.device, dtype=noise_sequence.dtype,
        )
        physical, scalar = level.conditions(time)
        local_noise = level.cover.tile(noise_sequence[step])
        noisy = (1.0 - time_value) * local_noise + time_value * endpoint_tiles
        velocity = model(
            noisy, physical, scalar, _tile_time(time, level.cover.tile_count)
        )
        endpoint_tiles = noisy + (1.0 - time_value) * velocity
        endpoint_tiles = model.project_endpoint(
            endpoint_tiles, physical, scalar, level.cover.tile_count
        )
    endpoint = level.cover.assemble_weighted(endpoint_tiles, level.batch_size)
    return FlowSample(endpoint, torch.stack(residuals))
