"""Shared-weight paired-level inference and AMR-composite error metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from codiffuse_Schwarz_Airfoil.cartesian_cover import PlacementMode
from codiffuse_Schwarz_Airfoil.coupled_trajectory import CyclicWindowDataset
from codiffuse_Schwarz_Airfoil.dataset import FieldStatistics, denormalize_delta
from codiffuse_Schwarz_Airfoil.multiresolution import (
    AdaptiveGridController,
    adaptive_refinement_mask,
    constant_prolongation,
    decode_composite,
    synchronize_levels,
)
from codiffuse_Schwarz_Airfoil.train_one_step_multilevel import (
    interior_vorticity,
    random_uniform_transition,
)
from codiffuse_Schwarz_Airfoil.train_robust_vorticity_projection import projected_next


Tensor = torch.Tensor


@dataclass(frozen=True)
class MultiresolutionPrediction:
    coarse: Tensor
    fine: Tensor
    refinement_mask: Tensor
    composite_fine_view: Tensor
    executed_fine_tiles: Tensor


def normalize_delta(delta: Tensor, statistics: FieldStatistics) -> Tensor:
    mean = torch.tensor(
        statistics.delta_mean, device=delta.device, dtype=delta.dtype
    )[None, :, None, None]
    std = torch.tensor(
        statistics.delta_std, device=delta.device, dtype=delta.dtype
    )[None, :, None, None]
    return (delta - mean) / std.clamp_min(1.0e-12)


def sparse_fine_projection(
    model: torch.nn.Module,
    fine_batch: dict[str, Tensor],
    fine_dataset: CyclicWindowDataset,
    coarse_prediction: Tensor,
    coarse_refinement_mask: Tensor,
    statistics: FieldStatistics,
    *,
    halo: int,
    seed: int,
    device: torch.device,
    placement_mode: PlacementMode = "random",
) -> tuple[Tensor, Tensor]:
    """Evaluate only fine tiles intersecting selected parent cells.

    Unselected fine tiles carry a prolonged coarse prediction. Selected tiles
    run the same direct learned-projection endpoint used by uniform inference,
    and weighted overlap assembly provides the fine/coarse transition band.
    """
    if not bool(getattr(model, "endpoint_learned_projection", False)):
        raise ValueError("sparse executor requires a direct learned projection")
    if not bool(getattr(model, "endpoint_eos_direct", False)):
        raise ValueError("sparse executor requires a direct endpoint EoS")
    if any(
        getattr(model, name, None) is not None
        for name in ("projection_attention", "projection_spatial_attention")
    ):
        raise ValueError("sparse tile subsets do not yet support cross-tile attention")
    transition = random_uniform_transition(
        fine_batch, statistics, fine_dataset, device, seed, halo,
        placement_mode,
    )
    fine_q_n = fine_batch["q_n"].to(device)
    prolonged_coarse = constant_prolongation(coarse_prediction)
    if prolonged_coarse.shape != fine_q_n.shape:
        raise ValueError("coarse prediction is incompatible with fine state")
    baseline_delta = normalize_delta(prolonged_coarse - fine_q_n, statistics)
    local_delta = transition.cover.tile(baseline_delta)

    fine_mask = constant_prolongation(
        coarse_refinement_mask.to(device=device, dtype=fine_q_n.dtype)
    )
    local_mask = transition.cover.tile(fine_mask)
    # Dependency closure: any tile whose halo contributes to a selected cell
    # must run the model, otherwise weighted assembly would mix a coarse
    # baseline into the interior of the nominally refined region.
    selected = local_mask.flatten(1).amax(1) > 0.0
    if selected.any():
        flow_time = torch.ones(
            transition.batch_size, device=device, dtype=fine_q_n.dtype
        )
        physical, scalar = transition.conditions(flow_time)
        motion_physical, motion_scalar = transition.endpoint_conditions()
        endpoint = torch.zeros_like(local_delta[selected])
        selected_tiles = selected.reshape(
            transition.batch_size, transition.cover.tile_count,
        )
        projected, _ = model.correct_endpoint_fields(  # type: ignore[attr-defined]
            endpoint, physical[selected], scalar[selected], tile_count=None,
            tile_exchange=lambda hidden: transition.cover.exchange_selected_weighted(
                hidden, selected_tiles, transition.batch_size,
            ),
            motion_physical=motion_physical[selected],
            motion_scalar=motion_scalar[selected],
        )
        local_delta = local_delta.clone()
        local_delta[selected] = projected
    # A compatibility-coded operator communicates after the frozen endpoint.
    # Sparse execution has no rectangular selected-tile batch, so the wrapper
    # waits until the prolonged-coarse baseline has filled every inactive tile,
    # then performs its one message on the complete fine cover before the
    # unchanged weighted POU realization below.
    if hasattr(model, "correct_post_endpoint_compatibility"):
        flow_time = torch.ones(
            transition.batch_size, device=device, dtype=fine_q_n.dtype,
        )
        physical, scalar = transition.conditions(flow_time)
        local_delta = model.correct_post_endpoint_compatibility(  # type: ignore[attr-defined]
            local_delta, physical, scalar, transition.cover.grid_shape,
        )
    assembled_delta = transition.cover.assemble_weighted(
        local_delta, transition.batch_size
    )
    prediction = fine_q_n + denormalize_delta(assembled_delta, statistics)
    return prediction, selected.reshape(
        transition.batch_size, transition.cover.tile_count
    )


def full_uniform_projection(
    model: torch.nn.Module,
    batch: dict[str, Tensor],
    dataset: CyclicWindowDataset,
    statistics: FieldStatistics,
    *,
    halo: int,
    seed: int,
    device: torch.device,
    placement_mode: PlacementMode = "random",
) -> Tensor:
    """Full-fine direct endpoint reference using the same shifted cover."""
    if not bool(getattr(model, "endpoint_learned_projection", False)):
        raise ValueError("reference executor requires a learned projection")
    if not bool(getattr(model, "endpoint_eos_direct", False)):
        raise ValueError("reference executor requires a direct endpoint EoS")
    transition = random_uniform_transition(
        batch, statistics, dataset, device, seed, halo, placement_mode,
    )
    flow_time = torch.ones(
        transition.batch_size, device=device, dtype=transition.q_n.dtype
    )
    physical, scalar = transition.conditions(flow_time)
    motion_physical, motion_scalar = transition.endpoint_conditions()
    endpoint = torch.zeros_like(transition.cover.tile(transition.target))
    projected, _ = model.correct_endpoint_fields(  # type: ignore[attr-defined]
        endpoint, physical, scalar, transition.cover.tile_count,
        tile_exchange=lambda hidden: transition.cover.tile(
            transition.cover.assemble_weighted(hidden, transition.batch_size)
        ),
        tile_grid_shape=transition.cover.grid_shape,
        motion_physical=motion_physical,
        motion_scalar=motion_scalar,
    )
    normalized_delta = transition.cover.assemble_weighted(
        projected, transition.batch_size
    )
    return transition.q_n + denormalize_delta(normalized_delta, statistics)


def predict_multiresolution_step(
    model: torch.nn.Module,
    coarse_batch: dict[str, Tensor],
    fine_batch: dict[str, Tensor],
    coarse_dataset: CyclicWindowDataset,
    fine_dataset: CyclicWindowDataset,
    statistics: FieldStatistics,
    *,
    fine_fraction: float,
    halo: int,
    seed: int,
    device: torch.device,
    fine_trust: float = 1.0,
    fine_core_cells: int = 32,
    vorticity_amplitude_fraction: float = .10,
    grid_controller: AdaptiveGridController | None = None,
    placement_ensemble: int = 1,
    placement_mode: PlacementMode = "random",
    refinement_mask_override: Tensor | None = None,
) -> MultiresolutionPrediction:
    """Advance adjacent levels with shared weights and synchronize predictions.

    This correctness-first reference evaluates every fine patch. The returned
    refinement mask determines the composite field and provides the contract
    for a later sparse fine-patch executor without changing synchronization.
    """
    if placement_ensemble <= 0:
        raise ValueError("placement ensemble must be positive")
    if not torch.isclose(
        torch.tensor(coarse_dataset.dx), torch.tensor(2.0 * fine_dataset.dx)
    ):
        raise ValueError("datasets must be adjacent Cartesian levels")
    if refinement_mask_override is not None:
        expected = (
            coarse_batch["q_n"].shape[0], 1,
            *coarse_batch["q_n"].shape[-2:],
        )
        if refinement_mask_override.shape != expected:
            raise ValueError("refinement mask override has incompatible shape")
        refinement_mask = refinement_mask_override.to(device=device).bool()
    elif grid_controller is None:
        refinement_mask = adaptive_refinement_mask(
            coarse_batch["q_n"].to(device), coarse_batch["sdf_n"].to(device),
            coarse_batch["sdf_np1"].to(device), dx=coarse_dataset.dx,
            previous_state=coarse_batch.get("q_nm1", coarse_batch["q_n"]).to(device),
            patch_cells=fine_core_cells // 2,
            vorticity_amplitude_fraction=vorticity_amplitude_fraction,
        )
    else:
        refinement_mask = grid_controller.select(
            coarse_batch.get("q_nm1", coarse_batch["q_n"]).to(device),
            coarse_batch["q_n"].to(device), coarse_batch["sdf_n"].to(device),
            coarse_batch["sdf_np1"].to(device),
        )

    shared_operator = getattr(model, "shared_operator", model)
    role_operator = model if (
        hasattr(model, "geometry_operator") or hasattr(model, "geometry_stem")
    ) else shared_operator
    coarse_operator = getattr(model, "coarse_operator", role_operator)
    fine_operator = getattr(model, "fine_operator", role_operator)

    # Different shifted Cartesian covers are independent domain decompositions
    # of the same physical update. Average them before the Schwarz level
    # synchronization so cover placement cannot become a recurrent state.
    coarse_members = []
    fine_members = []
    executed_members = []
    for member in range(placement_ensemble):
        member_seed = seed + member * 1_000_003
        coarse_prediction, _ = projected_next(
            coarse_operator, coarse_batch, coarse_dataset, statistics,
            fine_fraction, halo,
            member_seed, device, placement_mode,
        )
        if hasattr(model, "correct_surface_level"):
            coarse_prediction = model.correct_surface_level(
                "coarse", coarse_prediction, coarse_batch, coarse_dataset,
            )
        fine_prediction, executed_fine_tiles = sparse_fine_projection(
            fine_operator, fine_batch, fine_dataset, coarse_prediction,
            refinement_mask,
            statistics, halo=halo, seed=member_seed + 10_000_019, device=device,
            placement_mode=placement_mode,
        )
        if hasattr(model, "correct_surface_level"):
            fine_prediction = model.correct_surface_level(
                "fine", fine_prediction, fine_batch, fine_dataset,
            )
        coarse_members.append(coarse_prediction)
        fine_members.append(fine_prediction)
        executed_members.append(executed_fine_tiles)
    coarse_stack = torch.stack(coarse_members)
    fine_stack = torch.stack(fine_members)
    if hasattr(model, "correct_level_consensus"):
        coarse_prediction = model.correct_level_consensus(
            "coarse", coarse_stack, coarse_batch, dx=coarse_dataset.dx,
        )
        fine_prediction = model.correct_level_consensus(
            "fine", fine_stack, fine_batch, dx=fine_dataset.dx,
        )
    else:
        coarse_prediction = coarse_stack.mean(dim=0)
        fine_prediction = fine_stack.mean(dim=0)
    # Shifted covers can have different boundary tile counts. Concatenation
    # preserves the aggregate executed fraction used for compute accounting.
    executed_fine_tiles = torch.cat(executed_members, dim=1)
    synchronized_coarse, synchronized_fine = synchronize_levels(
        coarse_prediction, fine_prediction, refinement_mask,
        fine_trust=fine_trust,
    )
    composite = decode_composite(
        synchronized_coarse, synchronized_fine, refinement_mask,
    )
    return MultiresolutionPrediction(
        synchronized_coarse, synchronized_fine, refinement_mask, composite,
        executed_fine_tiles,
    )


def composite_vorticity_relative_l2(
    coarse_prediction: Tensor,
    fine_prediction: Tensor,
    coarse_target: Tensor,
    fine_target: Tensor,
    refinement_mask: Tensor,
    *,
    coarse_dx: float,
    fine_dx: float,
) -> Tensor:
    """AMR composite norm: coarse outside refinement, fine inside it."""
    if not torch.isclose(torch.tensor(coarse_dx), torch.tensor(2.0 * fine_dx)):
        raise ValueError("spacing must describe adjacent factor-two levels")
    coarse_predicted_vorticity = interior_vorticity(coarse_prediction, coarse_dx)
    coarse_target_vorticity = interior_vorticity(coarse_target, coarse_dx)
    fine_predicted_vorticity = interior_vorticity(fine_prediction, fine_dx)
    fine_target_vorticity = interior_vorticity(fine_target, fine_dx)
    if refinement_mask.shape != (
        coarse_prediction.shape[0], 1, *coarse_prediction.shape[-2:]
    ):
        raise ValueError("refinement mask must be B1HW on the coarse level")
    coarse_refined = refinement_mask[:, 0, 1:-1, 1:-1].to(
        coarse_prediction.dtype
    )
    fine_refined = refinement_mask.to(fine_prediction.dtype).repeat_interleave(
        2, dim=-2
    ).repeat_interleave(2, dim=-1)[:, 0, 1:-1, 1:-1]
    coarse_active = 1.0 - coarse_refined
    coarse_error = coarse_predicted_vorticity - coarse_target_vorticity
    fine_error = fine_predicted_vorticity - fine_target_vorticity
    numerator = (
        coarse_dx**2 * (coarse_error.square() * coarse_active).sum()
        + fine_dx**2 * (fine_error.square() * fine_refined).sum()
    )
    denominator = (
        coarse_dx**2 * (coarse_target_vorticity.square() * coarse_active).sum()
        + fine_dx**2 * (fine_target_vorticity.square() * fine_refined).sum()
    ).clamp_min(1.0e-20)
    return torch.sqrt(numerator / denominator)
