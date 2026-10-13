"""Two-level random Cartesian coupling during Flow integration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from codiffuse_Schwarz_Airfoil.cartesian_cover import (
    CartesianCover,
    PlacementMode,
    choose_offset,
    vorticity_magnitude,
)
from codiffuse_Schwarz_Airfoil.coupled_trajectory import (
    ControlledTransition,
    CyclicWindowDataset,
    controlled_velocity,
    endpoint_trace_velocity,
)
from codiffuse_Schwarz_Airfoil.dataset import FieldStatistics


Tensor = torch.Tensor


@dataclass(frozen=True)
class CoarseDatasetView:
    """Geometry and nondimensional scales for one coarsened model level."""

    dx: float
    core_size: int
    window: tuple[float, float, float, float]
    physical_parameters: dict[str, object]
    pitch_start_time: float
    level_fraction: float
    history_steps: int = 1


@dataclass
class MultilevelTransition:
    fine: ControlledTransition
    coarse: ControlledTransition
    refinement_mask: Tensor
    refinement_weight: Tensor
    factor: int

    @property
    def batch_size(self) -> int:
        return self.fine.batch_size

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(self.fine.q_n.shape)  # type: ignore[return-value]


def restrict(field: Tensor, factor: int) -> Tensor:
    return F.avg_pool2d(field, kernel_size=factor, stride=factor)


def prolong(field: Tensor, shape: tuple[int, int]) -> Tensor:
    return F.interpolate(field, size=shape, mode="bilinear", align_corners=False)


def _random_cover(
    q_n: Tensor,
    sdf_n: Tensor,
    sdf_np1: Tensor,
    core: int,
    halo: int,
    dx: float,
    seed: int,
    placement_mode: PlacementMode = "random",
) -> CartesianCover:
    oy, ox, _ = choose_offset(
        placement_mode, q_n, sdf_n, sdf_np1, core, halo, dx, seed
    )
    return CartesianCover(q_n.shape[-2], q_n.shape[-1], core, halo, oy, ox)


def adaptive_refinement_mask(
    q_n: Tensor,
    sdf_n: Tensor,
    sdf_np1: Tensor,
    cover: CartesianCover,
    dx: float,
    fine_fraction: float,
    seed: int,
) -> Tensor:
    """Return buffered AMR-like fine blocks tagged by body and vorticity."""
    if not 0.0 < fine_fraction <= 1.0:
        raise ValueError("fine_fraction must lie in (0, 1]")
    body = torch.maximum(
        torch.exp(-sdf_n[:, :1].abs() / (6.0 * dx)),
        torch.exp(-sdf_np1[:, :1].abs() / (6.0 * dx)),
    )
    vort = vorticity_magnitude(q_n, dx)[:, None]
    scale = vort.flatten(2).quantile(0.99, dim=2).clamp_min(1.0e-6)[:, :, None, None]
    indicator = body + 0.45 * (vort / scale).clamp_max(2.0)
    tiles = cover.tile(indicator)
    h, c = cover.halo, cover.core
    scores = tiles[:, :, h : h + c, h : h + c].flatten(2).quantile(0.95, dim=2)[:, 0]
    scores = scores.reshape(q_n.shape[0], *cover.grid_shape)
    generator = torch.Generator(device=q_n.device).manual_seed(seed)
    scores = scores + 1.0e-4 * torch.rand(
        scores.shape, generator=generator, device=q_n.device, dtype=scores.dtype
    )
    tile_count = scores.shape[-2] * scores.shape[-1]
    selected_count = max(1, min(tile_count, round(fine_fraction * tile_count)))
    threshold = scores.flatten(1).topk(selected_count, dim=1).values[:, -1, None, None]
    selected = (scores >= threshold).to(q_n.dtype)
    # AMReX-style one-block tag buffer prevents isolated high-gradient boxes.
    selected = F.max_pool2d(selected[:, None], 3, stride=1, padding=1)[:, 0]
    patches = torch.zeros(
        q_n.shape[0] * tile_count, 1, c + 2 * h, c + 2 * h,
        device=q_n.device, dtype=q_n.dtype,
    )
    patches[:, :, h : h + c, h : h + c] = selected.reshape(-1, 1, 1, 1)
    mask = cover.assemble_ras(patches, q_n.shape[0])
    # Smooth only the transmission strip; the binary mask remains available for viz.
    weight = mask
    for _ in range(2):
        weight = F.avg_pool2d(weight, 7, stride=1, padding=3)
    return torch.cat((mask, weight), dim=1)


def build_multilevel_transition(
    batch: dict[str, Tensor],
    statistics: FieldStatistics,
    dataset: CyclicWindowDataset,
    device: torch.device,
    seed: int,
    fine_fraction: float = 0.35,
    factor: int = 2,
    halo: int = 4,
    placement_mode: PlacementMode = "random",
) -> MultilevelTransition:
    if factor <= 1:
        raise ValueError("coarsening factor must exceed one")
    fine_batch = {key: value.to(device) for key, value in batch.items()}
    q_n = fine_batch["q_n"]
    fine_cover = _random_cover(
        q_n, fine_batch["sdf_n"], fine_batch["sdf_np1"], dataset.core_size,
        halo, dataset.dx, seed, placement_mode,
    )
    fine = ControlledTransition(fine_batch, statistics, dataset, fine_cover, device)

    spatial = {
        "q_nm1", "q_n", "q_np1", "sdf_nm1", "sdf_n", "sdf_np1",
        "right_ghost_q_nm1", "right_ghost_q_n",
    }
    coarse_batch = {
        key: restrict(value, factor) if key in spatial else value
        for key, value in fine_batch.items()
    }
    coarse_view = CoarseDatasetView(
        dx=dataset.dx * factor,
        core_size=dataset.core_size,
        window=dataset.window,
        physical_parameters=dataset.physical_parameters,
        pitch_start_time=dataset.pitch_start_time,
        level_fraction=max(0.0, (dataset.level - 1) / max(dataset.level, 1)),
        history_steps=dataset.history_steps,
    )
    coarse_cover = _random_cover(
        coarse_batch["q_n"], coarse_batch["sdf_n"], coarse_batch["sdf_np1"],
        dataset.core_size, halo, coarse_view.dx, seed + 1_000_003,
        placement_mode,
    )
    coarse = ControlledTransition(
        coarse_batch, statistics, coarse_view, coarse_cover, device  # type: ignore[arg-type]
    )
    masks = adaptive_refinement_mask(
        q_n, fine_batch["sdf_n"], fine_batch["sdf_np1"], fine_cover,
        dataset.dx, fine_fraction, seed + 2_000_003,
    )
    return MultilevelTransition(fine, coarse, masks[:, :1], masks[:, 1:], factor)


def multilevel_velocity(
    model: torch.nn.Module,
    state: Tensor,
    transition: MultilevelTransition,
    flow_time: Tensor,
    schwarz_iterations: int = 1,
    schwarz_relaxation: float = 0.7,
) -> tuple[Tensor, dict[str, Tensor]]:
    """One coarse/fine Schwarz exchange at the current Flow time."""
    coarse_state = restrict(state, transition.factor)
    if schwarz_iterations > 1:
        coarse_velocity, coarse_tiles, _, coarse_residuals = endpoint_trace_velocity(
            model, coarse_state, transition.coarse, flow_time,
            schwarz_iterations, schwarz_relaxation,
        )
        fine_velocity, fine_tiles, _, fine_residuals = endpoint_trace_velocity(
            model, state, transition.fine, flow_time,
            schwarz_iterations, schwarz_relaxation,
        )
    else:
        coarse_velocity, coarse_tiles, _ = controlled_velocity(
            model, coarse_state, transition.coarse, flow_time, "ras"
        )
        fine_velocity, fine_tiles, _ = controlled_velocity(
            model, state, transition.fine, flow_time, "ras"
        )
        coarse_residuals = []
        fine_residuals = []
    coarse_prolonged = prolong(coarse_velocity, state.shape[-2:])
    weight = transition.refinement_weight
    velocity = coarse_prolonged + weight * (fine_velocity - coarse_prolonged)
    return velocity, {
        "fine_velocity": fine_velocity,
        "coarse_velocity": coarse_velocity,
        "coarse_prolonged": coarse_prolonged,
        "fine_tiles": fine_tiles,
        "coarse_tiles": coarse_tiles,
        "fine_trace_residuals": torch.stack(fine_residuals) if fine_residuals else state.new_empty(0),
        "coarse_trace_residuals": torch.stack(coarse_residuals) if coarse_residuals else state.new_empty(0),
    }


def integrate_multilevel(
    model: torch.nn.Module,
    transition: MultilevelTransition,
    steps: int,
    initial_noise: Tensor,
    schwarz_iterations: int = 1,
    schwarz_relaxation: float = 0.7,
) -> Tensor:
    if steps <= 0:
        raise ValueError("Flow integration steps must be positive")
    state = initial_noise
    if tuple(state.shape) != transition.shape:
        raise ValueError("initial noise and transition shapes differ")
    for step in range(steps):
        flow_time = torch.full(
            (transition.batch_size,), (step + 0.5) / steps,
            device=state.device, dtype=state.dtype,
        )
        velocity, _ = multilevel_velocity(
            model, state, transition, flow_time,
            schwarz_iterations, schwarz_relaxation,
        )
        state = state + velocity / steps
    return state


def pnp_multilevel_sample(
    model: torch.nn.Module,
    transition: MultilevelTransition,
    noise_sequence: Tensor,
    relaxation: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Co-trainable PnP denoise-project-renoise schedule on two AMR levels."""
    if noise_sequence.ndim != 5 or tuple(noise_sequence.shape[1:]) != transition.shape:
        raise ValueError("PnP noise sequence has the wrong multilevel shape")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("PnP relaxation must lie in (0, 1]")
    fine = transition.fine
    coarse = transition.coarse
    fine_valid = fine.cover.valid_tiles(
        fine.batch_size, fine.device, fine.q_n.dtype
    )
    coarse_valid = coarse.cover.valid_tiles(
        coarse.batch_size, coarse.device, coarse.q_n.dtype
    )
    fine_endpoint = torch.zeros(
        fine.batch_size * fine.cover.tile_count, 3,
        fine.cover.core + 2 * fine.cover.halo,
        fine.cover.core + 2 * fine.cover.halo,
        device=fine.device, dtype=fine.q_n.dtype,
    )
    coarse_endpoint = torch.zeros(
        coarse.batch_size * coarse.cover.tile_count, 3,
        coarse.cover.core + 2 * coarse.cover.halo,
        coarse.cover.core + 2 * coarse.cover.halo,
        device=coarse.device, dtype=coarse.q_n.dtype,
    )
    fine_residuals = []
    coarse_residuals = []
    steps = noise_sequence.shape[0]
    for step in range(steps):
        fine_global = fine.cover.assemble_weighted(fine_endpoint, fine.batch_size)
        coarse_global = coarse.cover.assemble_weighted(
            coarse_endpoint, coarse.batch_size
        )
        coarse_up = prolong(coarse_global, fine_global.shape[-2:])
        composite = coarse_up + transition.refinement_weight * (fine_global - coarse_up)
        fine_consensus = fine.cover.tile(composite)
        coarse_consensus = coarse.cover.tile(restrict(composite, transition.factor))
        fine_mismatch = (fine_consensus - fine_endpoint) * fine_valid
        coarse_mismatch = (coarse_consensus - coarse_endpoint) * coarse_valid
        fine_residuals.append(
            (fine_mismatch.square().sum() /
             (fine_valid.sum().clamp_min(1.0) * 3.0)).sqrt()
        )
        coarse_residuals.append(
            (coarse_mismatch.square().sum() /
             (coarse_valid.sum().clamp_min(1.0) * 3.0)).sqrt()
        )
        fine_projected = fine_endpoint + relaxation * fine_mismatch
        coarse_projected = coarse_endpoint + relaxation * coarse_mismatch

        flow_time = torch.full(
            (fine.batch_size,), step / steps,
            device=fine.device, dtype=fine.q_n.dtype,
        )
        fine_tile_time = flow_time[:, None].expand(-1, fine.cover.tile_count).reshape(-1)
        coarse_tile_time = flow_time[:, None].expand(-1, coarse.cover.tile_count).reshape(-1)
        fine_t = fine_tile_time[:, None, None, None]
        coarse_t = coarse_tile_time[:, None, None, None]
        fine_noise = noise_sequence[step]
        coarse_noise = restrict(fine_noise, transition.factor)
        fine_noisy = (
            (1.0 - fine_t) * fine.cover.tile(fine_noise) + fine_t * fine_projected
        ) * fine_valid
        coarse_noisy = (
            (1.0 - coarse_t) * coarse.cover.tile(coarse_noise)
            + coarse_t * coarse_projected
        ) * coarse_valid
        fine_physical, fine_scalar = fine.conditions(flow_time)
        coarse_physical, coarse_scalar = coarse.conditions(flow_time)
        fine_velocity = model(
            fine_noisy, fine_physical, fine_scalar, fine_tile_time
        )
        coarse_velocity = model(
            coarse_noisy, coarse_physical, coarse_scalar, coarse_tile_time
        )
        fine_endpoint = (
            fine_noisy + (1.0 - fine_t) * fine_velocity
        ) * fine_valid
        coarse_endpoint = (
            coarse_noisy + (1.0 - coarse_t) * coarse_velocity
        ) * coarse_valid

    fine_global = fine.cover.assemble_ras(fine_endpoint, fine.batch_size)
    coarse_global = coarse.cover.assemble_ras(coarse_endpoint, coarse.batch_size)
    coarse_up = prolong(coarse_global, fine_global.shape[-2:])
    composite = coarse_up + transition.refinement_weight * (fine_global - coarse_up)
    return composite, {
        "fine_trace_residuals": torch.stack(fine_residuals),
        "coarse_trace_residuals": torch.stack(coarse_residuals),
    }
