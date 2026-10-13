"""Experimental Flow-Matching losses; not the retained direct training path.

No physics residual is included. Physical constraints enter as supervised
conditions and targets supplied by the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import CoDiffuseOperator
from .schwarz import PreparedLevel, integrate_flow, local_velocity


Tensor = torch.Tensor


@dataclass(frozen=True)
class LossReport:
    total: Tensor
    flow: Tensor
    endpoint: Tensor
    overlap: Tensor


def flow_matching_loss(
    model: CoDiffuseOperator, level: PreparedLevel,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Straight-path conditional Flow Matching on the assembled global field."""
    target = level.adapter.normalized_target(level.transition)
    noise = torch.randn_like(target)
    time = torch.rand(
        target.shape[0], device=target.device, dtype=target.dtype
    )
    mix = time[:, None, None, None]
    state = (1.0 - mix) * noise + mix * target
    target_velocity = target - noise
    prediction, local_prediction = local_velocity(
        model, state, level, time, weighted_writeback=False
    )
    loss = (prediction - target_velocity).square().mean()
    return loss, {
        "target": target, "noise": noise, "time": time,
        "state": state, "prediction": prediction,
        "local_prediction": local_prediction,
    }


def endpoint_projection_loss(
    model: CoDiffuseOperator, level: PreparedLevel,
    candidate: Tensor, target: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Supervise neural EoS correction and duplicate-overlap agreement."""
    one = torch.ones(
        level.batch_size, device=candidate.device, dtype=candidate.dtype
    )
    physical, scalar = level.conditions(one)
    local_candidate = level.cover.tile(candidate)
    local_corrected = model.project_endpoint(
        local_candidate, physical, scalar, level.cover.tile_count
    )
    corrected = level.cover.assemble_weighted(
        local_corrected, level.batch_size
    )
    endpoint = (corrected - target).square().mean()
    duplicate_target = level.cover.tile(corrected)
    overlap = (local_corrected - duplicate_target).square().mean()
    return endpoint + overlap, endpoint, overlap


def combined_loss(
    model: CoDiffuseOperator, level: PreparedLevel,
    endpoint_weight: float = 1.0, overlap_weight: float = .25,
) -> LossReport:
    """Experimental joint Flow and projection training for one transition."""
    flow, details = flow_matching_loss(model, level)
    # The path's one-shot clean estimate supplies realistic projection errors
    # without backpropagating through a long ODE solve.
    time = details["time"]
    candidate = details["state"] + (
        1.0 - time[:, None, None, None]
    ) * details["prediction"]
    target = details["target"]
    _, endpoint, overlap = endpoint_projection_loss(
        model, level, candidate, target
    )
    total = flow + endpoint_weight * endpoint + overlap_weight * overlap
    return LossReport(total, flow, endpoint, overlap)


@torch.no_grad()
def predict_next_state(
    model: CoDiffuseOperator, level: PreparedLevel, steps: int = 8,
    seed: int = 0, schwarz_iterations: int = 1,
) -> Tensor:
    """Generate a normalized endpoint and return the physical next state."""
    generator = torch.Generator(device=level.transition.current.device).manual_seed(seed)
    noise = torch.randn(
        level.shape, generator=generator, device=level.transition.current.device,
        dtype=level.transition.current.dtype,
    )
    sample = integrate_flow(
        model, level, noise, steps=steps,
        schwarz_iterations=schwarz_iterations,
    )
    return level.adapter.apply_endpoint(level.transition, sample.endpoint)
