"""Identify a weak physical Airfoil residual before allowing it into training.

Only validation trajectories are opened.  The target is used to test whether
the proposed residual is actually better aligned with the reference solution;
it is never consumed by the residual or the matrix-free Jacobian probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .airfoil_compatibility_experiment import load_candidate, residual_port_example
from .airfoil_defect_analyzer_experiment import (
    _force, _force_components, _paired_bootstrap_interval, _physical_prediction,
)
from .airfoil_physical_residual import physical_consistency_residual
from .airfoil_two_level import centered_gradient, velocity_divergence
from .train_multitrajectory_multiresolution import PairedTrajectoryCorpus


Tensor = torch.Tensor


def _laplacian(field: Tensor, dx: float) -> Tensor:
    padded = F.pad(field[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
    return (
        padded[:, 1:-1, 2:] + padded[:, 1:-1, :-2]
        + padded[:, 2:, 1:-1] + padded[:, :-2, 1:-1]
        - 4.0 * field
    ) / dx**2


def _weak_blocks(example, state: Tensor) -> dict[str, Tensor]:
    """Return separately testable mass, momentum, and global closure blocks."""
    *_, transition, dataset = example
    previous = transition.q_n.to(state)
    sdf = transition.batch["sdf_np1"].to(state)
    previous_sdf = transition.batch["sdf_n"].to(state)
    motion = transition.batch["motion_np1"].to(state)
    dt = (
        transition.batch["time_np1"] - transition.batch["time_n"]
    ).to(state).flatten()
    viscosity = state.new_full(
        (state.shape[0],), 1.0 / float(dataset.physical_parameters["reynolds"]),
    )
    u, v, pressure = state[:, 0], state[:, 1], state[:, 2]
    du_dx, du_dy = centered_gradient(u, dataset.dx)
    dv_dx, dv_dy = centered_gradient(v, dataset.dx)
    dp_dx, dp_dy = centered_gradient(pressure, dataset.dx)
    solid = (sdf[:, 0] <= 0.0).to(state)
    x0, _, y0, _ = dataset.window
    x = x0 + (
        torch.arange(state.shape[-1], device=state.device, dtype=state.dtype) + .5
    ) * dataset.dx
    y = y0 + (
        torch.arange(state.shape[-2], device=state.device, dtype=state.dtype) + .5
    ) * dataset.dx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    omega = motion[:, 1, None, None]
    wall_u, wall_v = -omega * yy, omega * xx
    momentum_x = (
        (u - previous[:, 0]) / dt[:, None, None]
        + u * du_dx + v * du_dy + dp_dx
        - viscosity[:, None, None] * _laplacian(u, dataset.dx)
        - solid * (wall_u - u) / dataset.penalty_time
    )
    momentum_y = (
        (v - previous[:, 1]) / dt[:, None, None]
        + u * dv_dx + v * dv_dy + dp_dy
        - viscosity[:, None, None] * _laplacian(v, dataset.dx)
        - solid * (wall_v - v) / dataset.penalty_time
    )
    divergence = velocity_divergence(state, dataset.dx)
    fields = torch.stack((divergence, momentum_x, momentum_y), dim=1)
    tiles = transition.cover.tile(fields)
    valid = transition.cover.valid_tiles(
        state.shape[0], state.device, state.dtype,
    )
    coordinate = torch.linspace(
        -1.0, 1.0, tiles.shape[-1], device=state.device, dtype=state.dtype,
    )
    cy, cx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    basis = torch.stack((torch.ones_like(cx), cx, cy))[None, None]
    weight = valid[:, None]
    denominator = weight.sum((-2, -1)).clamp_min(1.0)
    moments = (
        (tiles[:, :, None] * basis * weight).sum((-2, -1)) / denominator
    ).reshape(
        state.shape[0], transition.cover.tile_count, 3, 3,
    )
    global_balance = physical_consistency_residual(
        previous, state, sdf, motion, dt=dt, dx=dataset.dx,
        window=dataset.window, penalty_time=dataset.penalty_time,
        rotation_sign=dataset.rotation_sign, viscosity=viscosity,
        previous_sdf=previous_sdf, momentum_identity="all_box",
    )
    global_signed = torch.stack((
        global_balance.momentum_x, global_balance.momentum_y,
    ), dim=1)
    return {
        "tile_divergence": moments[:, :, 0].reshape(state.shape[0], -1),
        "tile_momentum": moments[:, :, 1:].reshape(state.shape[0], -1),
        "global_momentum": global_signed,
    }


def _weak_vector(example, state: Tensor) -> Tensor:
    blocks = _weak_blocks(example, state)
    return torch.cat(tuple(blocks.values()), dim=1)


def _oracle_corrected(model, example) -> Tensor:
    local, _, physical, _, raw, grid_shape, transition, _ = example
    code = (raw / model.composer.port_std).reshape(
        transition.batch_size, transition.cover.tile_count, -1,
    )
    semantics = model.composer.semantic_features(local, physical)
    return model.composer.decode(
        local, semantics, code, physical[:, -1:].to(local), grid_shape,
    )


def _decode(model, example, code: Tensor) -> Tensor:
    local, _, physical, _, _, grid_shape, transition, _ = example
    semantics = model.composer.semantic_features(local, physical)
    corrected = model.composer.decode(
        local, semantics, code, physical[:, -1:].to(local), grid_shape,
    )
    normalized = transition.cover.assemble_weighted(
        corrected, transition.batch_size,
    )
    # Use registered tensors directly inside torch.func transforms; creating a
    # new tensor from Python statistics under a GradWrapper dispatch key fails.
    mean = model.composer.delta_mean[None, :, None, None]
    std = model.composer.delta_std[None, :, None, None]
    return transition.q_n + normalized * std + mean


def _relative_field(prediction: Tensor, target: Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1.0e-20)
    )


def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    chart, base, model, statistics = load_candidate(
        argparse.Namespace(candidate=args.chart_checkpoint, graph_radius=None), device,
    )
    if not chart["validation"].get("chart_gate", False):
        raise ValueError("response chart did not pass its oracle gate")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    corpus = PairedTrajectoryCorpus(
        args.data_root, base["split"]["validation"], "all",
    )
    rows = []
    try:
        for trajectory, paired in enumerate(corpus.paired):
            count = min(args.samples_per_trajectory, len(paired))
            first = 1 if args.exclude_first and len(paired) > 1 else 0
            count = min(count, len(paired) - first)
            indices = np.linspace(first, len(paired) - 1, count, dtype=int)
            for ordinal, index in enumerate(indices):
                example = residual_port_example(
                    model, paired[int(index)], corpus.fine[trajectory], "fine",
                    statistics, halo=args.halo,
                    seed=args.seed + trajectory * 1_000_003 + ordinal,
                    device=device,
                )
                local, _, _, _, _, _, transition, _ = example
                zero = _physical_prediction(
                    model, example, local, statistics, apply_surface=False,
                )
                target = transition.q_np1
                oracle = _physical_prediction(
                    model, example, _oracle_corrected(model, example), statistics,
                    apply_surface=False,
                )
                zero_blocks = _weak_blocks(example, zero)
                target_blocks = _weak_blocks(example, target)
                oracle_blocks = _weak_blocks(example, oracle)
                # Nondimensional benchmark has U=L=rho=1.  Fixed unit scales
                # avoid target-dependent normalization and RMS makes blocks
                # independent of coordinate count.
                def block_norm(value: Tensor) -> Tensor:
                    return value.square().mean(1).sqrt()

                code_shape = (
                    transition.batch_size, transition.cover.tile_count,
                    model.composer.code_dim,
                )
                code_zero = zero.new_zeros(code_shape)
                block_names = tuple(zero_blocks)
                block_sizes = {name: zero_blocks[name].numel() for name in block_names}
                def residual_from_code(code: Tensor) -> tuple[Tensor, ...]:
                    blocks = _weak_blocks(example, _decode(model, example, code))
                    return tuple(blocks[name].reshape(-1) for name in block_names)
                probes = {name: [] for name in block_names}
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + 71_000_003 * trajectory + ordinal
                )
                for _ in range(args.rank_probes):
                    direction = torch.randn(
                        code_shape, generator=generator, device=device,
                        dtype=zero.dtype,
                    )
                    direction = direction / torch.linalg.vector_norm(direction)
                    response = torch.func.jvp(
                        residual_from_code, (code_zero,), (direction,),
                    )[1]
                    for name, value in zip(block_names, response):
                        probes[name].append(value)
                zero_drag, zero_lift = _force_components(zero, example)
                oracle_drag, oracle_lift = _force_components(oracle, example)
                row = {
                    "zero_field": _relative_field(zero, target),
                    "oracle_field": _relative_field(oracle, target),
                    "zero_force": float(_force(zero, example)),
                    "oracle_force": float(_force(oracle, example)),
                    "zero_drag": float(zero_drag), "oracle_drag": float(oracle_drag),
                    "zero_lift": float(zero_lift), "oracle_lift": float(oracle_lift),
                }
                for name in block_names:
                    singular = torch.linalg.svdvals(
                        torch.stack(probes[name], dim=1).float()
                    )
                    relative = singular / singular[0].clamp_min(1.0e-20)
                    row.update({
                        f"zero_{name}": float(block_norm(zero_blocks[name]).mean()),
                        f"target_{name}": float(block_norm(target_blocks[name]).mean()),
                        f"oracle_{name}": float(block_norm(oracle_blocks[name]).mean()),
                        f"jvp_rank_{name}": int(
                            (relative >= args.rank_relative_cutoff).sum()
                        ),
                        f"jvp_condition_{name}": float(
                            singular[0] / singular[-1].clamp_min(1.0e-20)
                        ),
                        f"jvp_leading_{name}": float(singular[0]),
                    })
                rows.append(row)
    finally:
        corpus.close()
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    paired = {}
    for ordinal, name in enumerate((
        "tile_divergence", "tile_momentum", "global_momentum",
    )):
        for candidate, offset in (("target", 11), ("oracle", 23)):
            paired[f"{candidate}_minus_zero_{name}"] = _paired_bootstrap_interval(
                np.asarray([
                    row[f"{candidate}_{name}"] - row[f"zero_{name}"]
                    for row in rows
                ]), seed=args.seed + 101 * ordinal + offset,
                draws=args.bootstrap_draws,
            )
    for name, left, right, offset in (
        ("oracle_minus_zero_field", "oracle_field", "zero_field", 401),
        ("oracle_minus_zero_force", "oracle_force", "zero_force", 409),
    ):
        paired[name] = _paired_bootstrap_interval(
            np.asarray([row[left] - row[right] for row in rows]),
            seed=args.seed + offset, draws=args.bootstrap_draws,
        )
    block_gates = {}
    for name in ("tile_divergence", "tile_momentum", "global_momentum"):
        block_gates[name] = {
            "target_materially_lower": (
                paired[f"target_minus_zero_{name}"]["ci95_high"] < 0.0
                and metrics[f"target_{name}"] <= .9 * metrics[f"zero_{name}"]
            ),
            "oracle_lower": paired[
                f"oracle_minus_zero_{name}"
            ]["ci95_high"] < 0.0,
            "usable_jacobian_rank": metrics[f"jvp_rank_{name}"] >= 2.0,
        }
        block_gates[name]["eligible"] = all(block_gates[name].values())
    eligible_blocks = [name for name, value in block_gates.items() if value["eligible"]]
    gate = {
        "blocks": block_gates,
        "eligible_blocks": eligible_blocks,
        "oracle_field_improves": paired[
            "oracle_minus_zero_field"
        ]["ci95_high"] < 0.0,
        "oracle_force_nonregression": paired[
            "oracle_minus_zero_force"
        ]["ci95_high"] <= 0.0,
    }
    gate["eligible_training_residual"] = (
        bool(eligible_blocks) and gate["oracle_field_improves"]
        and gate["oracle_force_nonregression"]
    )
    gate["full_concatenation_eligible"] = all(
        value["eligible"] for value in block_gates.values()
    ) and gate["oracle_field_improves"] and gate["oracle_force_nonregression"]
    result = {
        "contract": "airfoil_weak_physical_residual_preflight_v1",
        "split": "validation", "samples": len(rows), "metrics": metrics,
        "paired_bootstrap": paired, "gate": gate,
        "residual": {
            "global_momentum_identity": (
                "d/dt integral_Omega u + outer [uu+pI-nu(grad u+grad u^T)]n "
                "+ raw Brinkman reaction"
            ),
            "weak_coordinates": (
                "per-tile constant/x/y moments of divergence and strong "
                "backward-Euler momentum plus two signed global balances"
            ),
            "block_scaling": "fixed nondimensional unit scales; RMS within block",
            "tile_momentum_role": (
                "diagnostic weak moments of a finite-difference strong residual; "
                "not claimed to reproduce the AMReX discretization"
            ),
            "viscous_outer_stress_included": True,
            "target_used_by_residual_or_jacobian": False,
        },
        "rank_probes": args.rank_probes,
        "rank_relative_cutoff": args.rank_relative_cutoff,
        "index_policy": (
            "disjoint_confirmation_excluding_first_per_trajectory"
            if args.exclude_first else "includes_first_per_trajectory"
        ),
        "protected_test_trajectories_opened": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not gate["eligible_training_residual"]:
        raise SystemExit(2)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-trajectory", type=int, default=1)
    parser.add_argument("--exclude-first", action="store_true")
    parser.add_argument("--rank-probes", type=int, default=8)
    parser.add_argument("--rank-relative-cutoff", type=float, default=1e-3)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
