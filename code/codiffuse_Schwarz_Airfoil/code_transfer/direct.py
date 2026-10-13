"""Deterministic overlap-local residual learning used by the retained model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .hierarchy import TwoLevelHierarchy, prolong
from .model import DirectEndpointOperator
from .schwarz import PreparedLevel


Tensor = torch.Tensor


@dataclass(frozen=True)
class DirectSample:
    """One deterministic physical transition and its interface diagnostic."""

    endpoint: Tensor
    prediction: Tensor
    overlap_residual: Tensor
    selected_tiles: Tensor


@dataclass(frozen=True)
class DirectMultilevelSample:
    """Adaptive fine result together with the coarse endpoint prediction."""

    endpoint: Tensor
    prediction: Tensor
    overlap_residual: Tensor
    selected_tiles: Tensor
    coarse: DirectSample


@dataclass(frozen=True)
class DirectLossReport:
    total: Tensor
    endpoint: Tensor
    local_endpoint: Tensor
    overlap: Tensor


def _endpoint_conditions(level: PreparedLevel) -> tuple[Tensor, Tensor]:
    phase = torch.ones(
        level.batch_size,
        device=level.transition.current.device,
        dtype=level.transition.current.dtype,
    )
    return level.conditions(phase)


def _overlap_square(
    local_endpoint: Tensor, global_endpoint: Tensor, level: PreparedLevel,
) -> Tensor:
    consensus = level.cover.tile(global_endpoint)
    valid = level.cover.valid_tiles(
        level.batch_size, local_endpoint.device, local_endpoint.dtype
    )
    channels = local_endpoint.shape[1]
    return (
        (local_endpoint - consensus).square() * valid
    ).sum() / (valid.sum().clamp_min(1.0) * channels)


def direct_endpoint_loss(
    model: DirectEndpointOperator,
    level: PreparedLevel,
    local_weight: float = 1.0,
    overlap_weight: float = 0.25,
) -> DirectLossReport:
    """Train clean local residuals and endpoint compatibility jointly."""
    if local_weight < 0.0 or overlap_weight < 0.0:
        raise ValueError("loss weights must be nonnegative")
    target = level.adapter.normalized_target(level.transition)
    physical, scalar = _endpoint_conditions(level)
    local = model(physical, scalar, level.cover.tile_count)
    assembled = level.cover.assemble_weighted(local, level.batch_size)
    target_tiles = level.cover.tile(target)
    valid = level.cover.valid_tiles(
        level.batch_size, target.device, target.dtype
    )
    channels = target.shape[1]
    local_endpoint = (
        (local - target_tiles).square() * valid
    ).sum() / (valid.sum().clamp_min(1.0) * channels)
    endpoint = (assembled - target).square().mean()
    overlap = _overlap_square(local, assembled, level)
    total = endpoint + local_weight * local_endpoint + overlap_weight * overlap
    return DirectLossReport(total, endpoint, local_endpoint, overlap)


@torch.no_grad()
def predict_direct(
    model: DirectEndpointOperator,
    level: PreparedLevel,
) -> DirectSample:
    """Predict one transition with one parallel local endpoint evaluation."""
    physical, scalar = _endpoint_conditions(level)
    local = model(physical, scalar, level.cover.tile_count)
    endpoint = level.cover.assemble_weighted(local, level.batch_size)
    prediction = level.adapter.apply_endpoint(level.transition, endpoint)
    residual = _overlap_square(local, endpoint, level).sqrt()
    selected = torch.ones(
        level.batch_size, level.cover.tile_count,
        device=endpoint.device, dtype=torch.bool,
    )
    return DirectSample(endpoint, prediction, residual, selected)


@torch.no_grad()
def predict_direct_multilevel(
    model: DirectEndpointOperator,
    hierarchy: TwoLevelHierarchy,
) -> DirectMultilevelSample:
    """Run coarse cells everywhere and fine cells only in the causal AMR cover.

    The prolonged coarse physical prediction defines the fine-level baseline.
    Selected fine cells replace that baseline through weighted overlap
    assembly, after which the hierarchy's transition band restores exact
    coarse ownership outside the refined region.
    """
    coarse = predict_direct(model, hierarchy.coarse)
    fine = hierarchy.fine
    prolonged_coarse = prolong(coarse.prediction, fine.transition.spatial_shape)
    baseline = fine.adapter.normalize_endpoint(fine.transition, prolonged_coarse)
    local = fine.cover.tile(baseline)

    physical, scalar = _endpoint_conditions(fine)
    local_mask = fine.cover.tile(hierarchy.refinement_mask)
    selected = local_mask.flatten(1).amax(1) > 0.0
    if selected.any():
        local = local.clone()
        # A sparse subset cannot use dense cross-cell attention.  The retained
        # model likewise communicates selected cells at endpoint assembly.
        local[selected] = model(physical[selected], scalar[selected], tile_count=None)

    assembled = fine.cover.assemble_weighted(local, fine.batch_size)
    endpoint = baseline + hierarchy.refinement_weight * (assembled - baseline)
    prediction = fine.adapter.apply_endpoint(fine.transition, endpoint)
    residual = _overlap_square(local, endpoint, fine).sqrt()
    return DirectMultilevelSample(
        endpoint=endpoint,
        prediction=prediction,
        overlap_residual=residual,
        selected_tiles=selected.reshape(fine.batch_size, fine.cover.tile_count),
        coarse=coarse,
    )
