from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from codiffuse_Schwarz.core import (
    ConditionedResidualBlock,
    Tensor,
    coordinate_grid,
    relative_l2,
    relative_residual,
    timestep_embedding,
)
from neural_eos.block_structured_amr import AMRBox, build_block_structured_hierarchy


@dataclass(frozen=True)
class TwoLevelAMRConfig:
    fine_height: int = 32
    fine_width: int = 32
    refinement: int = 2
    halo: int = 4
    tag_threshold: float = 0.48
    blocking: int = 4
    buffer_cells: int = 1
    max_component_blocks: int = 2

    @property
    def coarse_shape(self) -> tuple[int, int]:
        return self.fine_height // self.refinement, self.fine_width // self.refinement


@dataclass(frozen=True)
class SampleAMRLayout:
    fine_boxes: tuple[tuple[int, int, int, int], ...]


@dataclass
class AMRSampleResult:
    field: Tensor
    trajectory: list[dict[str, float]]


class AMRFlowModel(nn.Module):
    """One fully convolutional local flow shared by coarse and fine levels."""

    def __init__(self, width: int = 32, depth: int = 4):
        super().__init__()
        self.width = width
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        # state, mask, source, reaction, x, y, and normalized cell width
        self.stem = nn.Conv2d(7, width, 3, padding=1)
        self.blocks = nn.ModuleList([ConditionedResidualBlock(width, width) for _ in range(depth)])
        self.out = nn.Sequential(nn.GroupNorm(min(8, width), width), nn.SiLU(), nn.Conv2d(width, 1, 3, padding=1))

    def forward(
        self,
        state: Tensor,
        mask: Tensor,
        source: Tensor,
        reaction: Tensor,
        coords: Tensor,
        time: Tensor,
        cell_scale: Tensor,
    ) -> Tensor:
        height, width = state.shape[-2:]
        reaction_image = reaction[:, :, None, None].expand(-1, -1, height, width)
        scale_image = cell_scale[:, :, None, None].expand(-1, -1, height, width)
        inputs = torch.cat([state, mask, source, reaction_image, coords, scale_image], dim=1)
        condition = self.time_mlp(timestep_embedding(time, self.width))
        hidden = self.stem(inputs)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.out(hidden) * mask


def _neighbor_boundary(mask: Tensor) -> Tensor:
    padded = F.pad(mask, (1, 1, 1, 1))
    center = padded[:, :, 1:-1, 1:-1]
    difference = (
        (center - padded[:, :, :-2, 1:-1]).abs()
        + (center - padded[:, :, 2:, 1:-1]).abs()
        + (center - padded[:, :, 1:-1, :-2]).abs()
        + (center - padded[:, :, 1:-1, 2:]).abs()
    )
    return (difference > 0).float()


def complexity_indicator(mask: Tensor, source: Tensor) -> Tensor:
    source_scale = source.abs().flatten(1).amax(dim=1).clamp_min(1.0e-8)[:, None, None, None]
    source_signal = (source.abs() / source_scale).clamp(0.0, 1.0)
    source_dx = F.pad((source_signal[:, :, :, 1:] - source_signal[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    source_dy = F.pad((source_signal[:, :, 1:, :] - source_signal[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    geometry = _neighbor_boundary(mask)
    # The physical outer boundary spans the whole domain and would cluster into
    # one full-domain child box. Keep embedded boundaries as refinement tags,
    # but let the coarse level represent the regular exterior boundary.
    geometry[:, :, :3, :] = 0.0
    geometry[:, :, -3:, :] = 0.0
    geometry[:, :, :, :3] = 0.0
    geometry[:, :, :, -3:] = 0.0
    return (0.55 * source_signal + 0.20 * (source_dx + source_dy).clamp_max(1.0) + 0.65 * geometry).clamp_max(1.0)


def build_amr_layouts(mask: Tensor, source: Tensor, reaction: Tensor, config: TwoLevelAMRConfig) -> list[SampleAMRLayout]:
    indicator = complexity_indicator(mask, source)
    layouts: list[SampleAMRLayout] = []
    for sample in range(mask.shape[0]):
        active = mask[sample, 0].detach().cpu().numpy() > 0.5
        rhs = source[sample, 0].detach().cpu().numpy().astype(np.float64)
        hierarchy = build_block_structured_hierarchy(
            fine_active=active,
            fine_rhs=rhs,
            fine_fixed=np.zeros_like(active, dtype=bool),
            fine_fixed_values=np.zeros_like(rhs),
            fine_indicator=indicator[sample, 0].detach().cpu().numpy(),
            fine_reaction=float(reaction[sample, 0]),
            max_level=1,
            refinement=config.refinement,
            tag_thresholds=(config.tag_threshold,),
            blocking=config.blocking,
            buffer_cells=config.buffer_cells,
            max_component_blocks=config.max_component_blocks,
        )
        boxes = tuple((box.y0, box.y1, box.x0, box.x1) for box in hierarchy.levels[1].boxes)
        layouts.append(SampleAMRLayout(fine_boxes=boxes))
    return layouts


def restrict_weighted(field: Tensor, mask: Tensor, factor: int) -> tuple[Tensor, Tensor]:
    numerator = F.avg_pool2d(field * mask, factor, factor) * (factor * factor)
    denominator = F.avg_pool2d(mask, factor, factor) * (factor * factor)
    coarse_mask = (denominator > 0).to(field.dtype)
    return numerator / denominator.clamp_min(1.0), coarse_mask


def coarse_problem(
    field: Tensor,
    noise: Tensor,
    mask: Tensor,
    source: Tensor,
    reaction: Tensor,
    factor: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    coarse_field, coarse_mask = restrict_weighted(field, mask, factor)
    coarse_noise, _ = restrict_weighted(noise, mask, factor)
    coarse_source, _ = restrict_weighted(source, mask, factor)
    coarse_source = coarse_source * float(factor * factor)
    coarse_reaction = reaction * float(factor * factor)
    return coarse_field, coarse_noise, coarse_mask, coarse_source, coarse_reaction


def prolong(field: Tensor, shape: tuple[int, int]) -> Tensor:
    return F.interpolate(field, size=shape, mode="bilinear", align_corners=False)


def coverage_masks(layouts: Sequence[SampleAMRLayout], shape: tuple[int, int], device: torch.device) -> Tensor:
    coverage = torch.zeros(len(layouts), 1, shape[0], shape[1], device=device)
    for sample, layout in enumerate(layouts):
        for y0, y1, x0, x1 in layout.fine_boxes:
            coverage[sample, :, y0:y1, x0:x1] = 1.0
    return coverage


def _patch_records(layouts: Sequence[SampleAMRLayout]) -> dict[tuple[int, int], list[tuple[int, tuple[int, int, int, int]]]]:
    buckets: dict[tuple[int, int], list[tuple[int, tuple[int, int, int, int]]]] = {}
    for sample, layout in enumerate(layouts):
        for box in layout.fine_boxes:
            y0, y1, x0, x1 = box
            buckets.setdefault((y1 - y0, x1 - x0), []).append((sample, box))
    return buckets


def predict_fine_patch_velocity(
    model: AMRFlowModel,
    state: Tensor,
    mask: Tensor,
    source: Tensor,
    reaction: Tensor,
    coords: Tensor,
    time: Tensor,
    layouts: Sequence[SampleAMRLayout],
    halo: int,
) -> tuple[Tensor, Tensor]:
    batch, _, height, width = state.shape
    padded_state = F.pad(state, (halo, halo, halo, halo))
    padded_mask = F.pad(mask, (halo, halo, halo, halo))
    padded_source = F.pad(source, (halo, halo, halo, halo))
    padded_coords = F.pad(coords, (halo, halo, halo, halo))
    accumulated = torch.zeros_like(state)
    weight = torch.zeros_like(state)

    for (core_h, core_w), records in _patch_records(layouts).items():
        states = []
        masks = []
        sources = []
        coordinates = []
        reactions = []
        times = []
        for sample, (y0, y1, x0, x1) in records:
            states.append(padded_state[sample : sample + 1, :, y0 : y1 + 2 * halo, x0 : x1 + 2 * halo])
            masks.append(padded_mask[sample : sample + 1, :, y0 : y1 + 2 * halo, x0 : x1 + 2 * halo])
            sources.append(padded_source[sample : sample + 1, :, y0 : y1 + 2 * halo, x0 : x1 + 2 * halo])
            coordinates.append(padded_coords[sample : sample + 1, :, y0 : y1 + 2 * halo, x0 : x1 + 2 * halo])
            reactions.append(reaction[sample : sample + 1])
            times.append(time[sample : sample + 1])
        local_state = torch.cat(states)
        count = local_state.shape[0]
        prediction = model(
            local_state,
            torch.cat(masks),
            torch.cat(sources),
            torch.cat(reactions),
            torch.cat(coordinates),
            torch.cat(times),
            torch.ones(count, 1, device=state.device),
        )
        core_prediction = prediction[:, :, halo : halo + core_h, halo : halo + core_w]
        for local_index, (sample, (y0, y1, x0, x1)) in enumerate(records):
            accumulated[sample, :, y0:y1, x0:x1] += core_prediction[local_index]
            weight[sample, :, y0:y1, x0:x1] += 1.0
    return accumulated / weight.clamp_min(1.0), (weight > 0).to(state.dtype)


def prepare_training_states(
    target: Tensor,
    noise: Tensor,
    mask: Tensor,
    source: Tensor,
    reaction: Tensor,
    time: Tensor,
    layouts: Sequence[SampleAMRLayout],
    config: TwoLevelAMRConfig,
) -> dict[str, Tensor]:
    factor = config.refinement
    coarse_target, coarse_noise, coarse_mask, coarse_source, coarse_reaction = coarse_problem(
        target, noise, mask, source, reaction, factor
    )
    time_image = time[:, None, None, None]
    coarse_state = ((1.0 - time_image) * coarse_target + time_image * coarse_noise) * coarse_mask
    coarse_velocity = (coarse_noise - coarse_target) * coarse_mask
    fine_state_true = ((1.0 - time_image) * target + time_image * noise) * mask
    fine_velocity = (noise - target) * mask
    coverage = coverage_masks(layouts, target.shape[-2:], target.device) * mask
    fine_state = prolong(coarse_state, target.shape[-2:])
    fine_state = fine_state * (1.0 - coverage) + fine_state_true * coverage
    return {
        "coarse_target": coarse_target,
        "coarse_state": coarse_state,
        "coarse_velocity": coarse_velocity,
        "coarse_mask": coarse_mask,
        "coarse_source": coarse_source,
        "coarse_reaction": coarse_reaction,
        "fine_state": fine_state,
        "fine_velocity": fine_velocity,
        "coverage": coverage,
    }


def amr_flow_loss(
    model: AMRFlowModel,
    target: Tensor,
    mask: Tensor,
    source: Tensor,
    reaction: Tensor,
    layouts: Sequence[SampleAMRLayout],
    config: TwoLevelAMRConfig,
    endpoint_weight: float = 0.15,
    consistency_weight: float = 0.10,
) -> tuple[Tensor, dict[str, float]]:
    batch = target.shape[0]
    time = torch.rand(batch, device=target.device).clamp(1.0e-3, 1.0 - 1.0e-3)
    noise = torch.randn_like(target) * mask
    states = prepare_training_states(target, noise, mask, source, reaction, time, layouts, config)
    coarse_coords = coordinate_grid(*config.coarse_shape, target.device).expand(batch, -1, -1, -1)
    fine_coords = coordinate_grid(config.fine_height, config.fine_width, target.device).expand(batch, -1, -1, -1)
    coarse_prediction = model(
        states["coarse_state"],
        states["coarse_mask"],
        states["coarse_source"],
        states["coarse_reaction"],
        coarse_coords,
        time,
        torch.full((batch, 1), float(config.refinement), device=target.device),
    )
    fine_prediction, prediction_coverage = predict_fine_patch_velocity(
        model,
        states["fine_state"],
        mask,
        source,
        reaction,
        fine_coords,
        time,
        layouts,
        config.halo,
    )
    fine_weight = states["coverage"] * prediction_coverage
    coarse_denominator = states["coarse_mask"].sum().clamp_min(1.0)
    fine_denominator = fine_weight.sum().clamp_min(1.0)
    coarse_velocity_loss = (
        (coarse_prediction - states["coarse_velocity"]).square() * states["coarse_mask"]
    ).sum() / coarse_denominator
    fine_velocity_loss = ((fine_prediction - states["fine_velocity"]).square() * fine_weight).sum() / fine_denominator
    time_image = time[:, None, None, None]
    coarse_endpoint = states["coarse_state"] - time_image * coarse_prediction
    fine_endpoint = states["fine_state"] - time_image * fine_prediction
    coarse_endpoint_loss = (
        (coarse_endpoint - states["coarse_target"]).square() * states["coarse_mask"]
    ).sum() / coarse_denominator
    fine_endpoint_loss = ((fine_endpoint - target).square() * fine_weight).sum() / fine_denominator
    restricted_fine_prediction, coarse_coverage = restrict_weighted(
        fine_prediction, fine_weight, config.refinement
    )
    consistency_denominator = coarse_coverage.sum().clamp_min(1.0)
    consistency_loss = (
        (restricted_fine_prediction - coarse_prediction).square() * coarse_coverage
    ).sum() / consistency_denominator
    velocity_loss = 0.5 * (coarse_velocity_loss + fine_velocity_loss)
    endpoint_loss = 0.5 * (coarse_endpoint_loss + fine_endpoint_loss)
    loss = velocity_loss + endpoint_weight * endpoint_loss + consistency_weight * consistency_loss
    return loss, {
        "loss": float(loss.detach()),
        "coarse_velocity_loss": float(coarse_velocity_loss.detach()),
        "fine_velocity_loss": float(fine_velocity_loss.detach()),
        "endpoint_loss": float(endpoint_loss.detach()),
        "consistency_loss": float(consistency_loss.detach()),
    }


def _average_down_covered(coarse: Tensor, fine: Tensor, coverage: Tensor, fine_mask: Tensor, factor: int) -> Tensor:
    restricted, valid = restrict_weighted(fine, coverage * fine_mask, factor)
    return torch.where(valid.bool(), restricted, coarse)


@torch.no_grad()
def sample_amr_flow(
    model: AMRFlowModel,
    mask: Tensor,
    source: Tensor,
    reaction: Tensor,
    target: Tensor,
    noise: Tensor,
    layouts: Sequence[SampleAMRLayout],
    config: TwoLevelAMRConfig,
    steps: int,
) -> AMRSampleResult:
    batch = target.shape[0]
    _, coarse_noise, coarse_mask, coarse_source, coarse_reaction = coarse_problem(
        target, noise, mask, source, reaction, config.refinement
    )
    coarse_state = coarse_noise * coarse_mask
    coverage = coverage_masks(layouts, target.shape[-2:], target.device) * mask
    fine_state = prolong(coarse_state, target.shape[-2:])
    fine_state = fine_state * (1.0 - coverage) + noise * coverage
    coarse_coords = coordinate_grid(*config.coarse_shape, target.device).expand(batch, -1, -1, -1)
    fine_coords = coordinate_grid(config.fine_height, config.fine_width, target.device).expand(batch, -1, -1, -1)
    times = torch.linspace(1.0, 0.0, steps + 1, device=target.device)
    trajectory: list[dict[str, float]] = []

    for step in range(steps):
        time = torch.full((batch,), float(times[step]), device=target.device)
        coarse_velocity = model(
            coarse_state,
            coarse_mask,
            coarse_source,
            coarse_reaction,
            coarse_coords,
            time,
            torch.full((batch, 1), float(config.refinement), device=target.device),
        )
        fine_velocity, fine_velocity_mask = predict_fine_patch_velocity(
            model, fine_state, mask, source, reaction, fine_coords, time, layouts, config.halo
        )
        dt = times[step + 1] - times[step]
        coarse_state = (coarse_state + dt * coarse_velocity).clamp(-1.75, 1.75) * coarse_mask
        fine_updated = (fine_state + dt * fine_velocity).clamp(-1.75, 1.75) * mask
        written = coverage * fine_velocity_mask
        coarse_state = _average_down_covered(coarse_state, fine_updated, written, mask, config.refinement)
        background = prolong(coarse_state, target.shape[-2:]) * mask
        fine_state = background * (1.0 - written) + fine_updated * written
        trajectory.append(
            {
                "step": float(step),
                "time": float(times[step + 1]),
                "relative_l2": float(relative_l2(fine_state, target, mask).mean()),
                "residual": float(relative_residual(fine_state, source, mask, reaction).mean()),
            }
        )
    return AMRSampleResult(field=fine_state, trajectory=trajectory)


@torch.no_grad()
def sample_coarse_flow(
    model: AMRFlowModel,
    mask: Tensor,
    source: Tensor,
    reaction: Tensor,
    target: Tensor,
    noise: Tensor,
    config: TwoLevelAMRConfig,
    steps: int,
) -> AMRSampleResult:
    batch = target.shape[0]
    _, coarse_noise, coarse_mask, coarse_source, coarse_reaction = coarse_problem(
        target, noise, mask, source, reaction, config.refinement
    )
    state = coarse_noise * coarse_mask
    coords = coordinate_grid(*config.coarse_shape, target.device).expand(batch, -1, -1, -1)
    times = torch.linspace(1.0, 0.0, steps + 1, device=target.device)
    trajectory = []
    for step in range(steps):
        time = torch.full((batch,), float(times[step]), device=target.device)
        velocity = model(
            state,
            coarse_mask,
            coarse_source,
            coarse_reaction,
            coords,
            time,
            torch.full((batch, 1), float(config.refinement), device=target.device),
        )
        state = (state + (times[step + 1] - times[step]) * velocity).clamp(-1.75, 1.75) * coarse_mask
        composite = prolong(state, target.shape[-2:]) * mask
        trajectory.append(
            {
                "step": float(step),
                "relative_l2": float(relative_l2(composite, target, mask).mean()),
                "residual": float(relative_residual(composite, source, mask, reaction).mean()),
            }
        )
    return AMRSampleResult(field=prolong(state, target.shape[-2:]) * mask, trajectory=trajectory)


@torch.no_grad()
def sample_uniform_fine_flow(
    model: AMRFlowModel,
    mask: Tensor,
    source: Tensor,
    reaction: Tensor,
    target: Tensor,
    noise: Tensor,
    config: TwoLevelAMRConfig,
    steps: int,
) -> AMRSampleResult:
    batch = target.shape[0]
    boxes = ((0, 16, 0, 16), (0, 16, 16, 32), (16, 32, 0, 16), (16, 32, 16, 32))
    layouts = [SampleAMRLayout(boxes) for _ in range(batch)]
    state = noise * mask
    coords = coordinate_grid(config.fine_height, config.fine_width, target.device).expand(batch, -1, -1, -1)
    times = torch.linspace(1.0, 0.0, steps + 1, device=target.device)
    trajectory = []
    for step in range(steps):
        time = torch.full((batch,), float(times[step]), device=target.device)
        velocity, _ = predict_fine_patch_velocity(
            model, state, mask, source, reaction, coords, time, layouts, config.halo
        )
        state = (state + (times[step + 1] - times[step]) * velocity).clamp(-1.75, 1.75) * mask
        trajectory.append(
            {
                "step": float(step),
                "relative_l2": float(relative_l2(state, target, mask).mean()),
                "residual": float(relative_residual(state, source, mask, reaction).mean()),
            }
        )
    return AMRSampleResult(field=state, trajectory=trajectory)


def layout_costs(layouts: Sequence[SampleAMRLayout], config: TwoLevelAMRConfig) -> dict[str, float]:
    fine_cells = []
    processed = []
    box_counts = []
    for layout in layouts:
        coverage = np.zeros((config.fine_height, config.fine_width), dtype=bool)
        process = config.coarse_shape[0] * config.coarse_shape[1]
        for y0, y1, x0, x1 in layout.fine_boxes:
            coverage[y0:y1, x0:x1] = True
            process += (y1 - y0 + 2 * config.halo) * (x1 - x0 + 2 * config.halo)
        fine_cells.append(int(coverage.sum()))
        processed.append(process)
        box_counts.append(len(layout.fine_boxes))
    uniform_processed = 4 * (16 + 2 * config.halo) ** 2
    return {
        "mean_fine_cells": float(np.mean(fine_cells)),
        "mean_stored_cells": float(config.coarse_shape[0] * config.coarse_shape[1] + np.mean(fine_cells)),
        "stored_fraction_vs_uniform_fine": float(
            (config.coarse_shape[0] * config.coarse_shape[1] + np.mean(fine_cells))
            / (config.fine_height * config.fine_width)
        ),
        "mean_boxes": float(np.mean(box_counts)),
        "mean_processed_cells_per_step": float(np.mean(processed)),
        "uniform_processed_cells_per_step": float(uniform_processed),
        "processed_fraction_vs_uniform": float(np.mean(processed) / uniform_processed),
    }
