"""Target-free latent-Schur ablation with the validated tile-momentum block.

The frozen local endpoint and nonlinear response chart are identical to the
q_Gamma-only reference.  For a calibration-selected nonnegative ``beta`` this
script solves the offline reference problem

  min_z ||q_Gamma(v + L(z))||^2 + beta ||q_mom(v + L(z))||^2
        + ridge ||z||^2.

``q_mom`` contains only the independently preflighted per-tile momentum
moments.  It has fixed nondimensional scaling and uses no target.  Targets are
opened only after both target-free solves to compare beta=0 with beta>0.
The protected test split is never available through this entry point.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from codiffuse_Schwarz.compatibility_defect import residual_minimizing_damping
from .airfoil_compatibility_experiment import load_candidate, residual_port_example
from .airfoil_defect_analyzer_experiment import (
    _force, _force_components, _paired_bootstrap_interval,
)
from .airfoil_fixed_point_analyzer import compatibility_face_support
from .airfoil_latent_schur_reference import _conjugate_gradient, _decode24, _q24
from .airfoil_physical_residual_preflight import _weak_blocks
from .train_multitrajectory_multiresolution import PairedTrajectoryCorpus


Tensor = torch.Tensor
CONTRACT = "airfoil_target_free_latent_schur_tile_momentum_ablation_v1"


def _physical_prediction(model, example, broken: Tensor) -> Tensor:
    """Map broken normalized updates to the strict no-T_X physical endpoint."""
    *_, transition, _ = example
    normalized = transition.cover.assemble_weighted(
        broken, transition.batch_size,
    )
    # Registered frozen-chart buffers are safe under torch.func transforms.
    mean = model.composer.delta_mean[None, :, None, None]
    std = model.composer.delta_std[None, :, None, None]
    return transition.q_n + normalized * std + mean


def _components(model, example, broken: Tensor, support: Tensor) -> tuple[Tensor, Tensor]:
    q_gamma = (_q24(model, example, broken) * support).reshape(
        example[-2].batch_size, -1,
    )
    q_momentum = _weak_blocks(
        example, _physical_prediction(model, example, broken),
    )["tile_momentum"].reshape(example[-2].batch_size, -1)
    return q_gamma, q_momentum


def _weighted_vector(q_gamma: Tensor, q_momentum: Tensor, beta: float) -> Tensor:
    # beta is the objective weight: concatenating sqrt(beta) q_momentum gives
    # exactly ||q_Gamma||^2 + beta ||q_momentum||^2.  No target-dependent scale.
    if beta < 0.0:
        raise ValueError("tile-momentum weight beta must be nonnegative")
    return torch.cat((q_gamma, math.sqrt(beta) * q_momentum), dim=1)


def _objective_vector(
    model, example, broken: Tensor, support: Tensor, beta: float,
) -> Tensor:
    # Keep beta=0 an exact computational q_Gamma-only control: do not evaluate
    # the physical block and multiply it by zero after the fact.
    q_gamma = (_q24(model, example, broken) * support).reshape(
        example[-2].batch_size, -1,
    )
    if beta == 0.0:
        return q_gamma
    momentum = _weak_blocks(
        example, _physical_prediction(model, example, broken),
    )["tile_momentum"].reshape(example[-2].batch_size, -1)
    return _weighted_vector(q_gamma, momentum, beta)


def solve(
    model, example, *, beta: float, ridge: float,
    cg_iterations: int, cg_tolerance: float, line_search_points: int = 17,
) -> dict:
    local, _, physical, _, _, grid_shape, transition, _ = example
    support = compatibility_face_support(
        physical[:, -1:].to(local), grid_shape,
    )
    zero = local.new_zeros(transition.batch_size, transition.cover.tile_count, 24)

    def residual_function(code: Tensor) -> Tensor:
        broken = _decode24(model, example, code * support)
        return _objective_vector(model, example, broken, support, beta)

    residual_zero, vjp = torch.func.vjp(residual_function, zero)
    right = -vjp(residual_zero)[0].reshape(-1)

    def normal(vector: Tensor) -> Tensor:
        shaped = vector.reshape_as(zero)
        j_vector = torch.func.jvp(residual_function, (zero,), (shaped,))[1]
        return (vjp(j_vector)[0] + ridge * shaped).reshape(-1)

    raw_code, used, linear_ratio = _conjugate_gradient(
        normal, right, cg_iterations, cg_tolerance,
    )
    raw_code = raw_code.reshape_as(zero) * support

    # Decode the proposed student direction exactly once.  Damping is then
    # solved in broken-field space using the actual nonlinear-chart proposal;
    # evaluating either residual block below does not invoke L again.
    proposed = _decode24(model, example, raw_code)
    proposed_residual = _objective_vector(
        model, example, proposed, support, beta,
    )
    local_view = local.reshape(
        transition.batch_size, transition.cover.tile_count, *local.shape[1:],
    )
    proposed_view = proposed.reshape_as(local_view)
    if beta == 0.0:
        # q_Gamma is linear in the broken field, so this closed form is exact.
        damping = residual_minimizing_damping(
            residual_zero,
            proposed_residual - residual_zero,
            lower=0.0, upper=1.0,
        )
        alpha = damping.scale
        corrected = (
            local_view + alpha[:, None, None, None, None]
            * (proposed_view - local_view)
        ).reshape_as(local)
        q_zero = residual_zero
        q_corrected = q_zero + alpha[:, None] * (
            proposed_residual - q_zero
        )
        # Momentum is diagnostic for the q-only control and is not evaluated
        # inside its solver/Jacobian.
        momentum_zero = _weak_blocks(
            example, _physical_prediction(model, example, local),
        )["tile_momentum"].reshape(transition.batch_size, -1)
        momentum_corrected = _weak_blocks(
            example, _physical_prediction(model, example, corrected),
        )["tile_momentum"].reshape(transition.batch_size, -1)
        total_corrected = q_corrected
        damping_kind = "exact_linear_q_gamma_closed_form"
        line_search_residual_evaluations = 0
        solver_physical_residual_evaluations = 0
        diagnostic_physical_residual_evaluations = 2
    else:
        if line_search_points < 2:
            raise ValueError("nonlinear line search must include at least alpha=0,1")
        q_dimension = transition.cover.tile_count * 24
        q_zero = residual_zero[:, :q_dimension]
        momentum_zero = residual_zero[:, q_dimension:] / math.sqrt(beta)
        q_proposed = proposed_residual[:, :q_dimension]
        momentum_proposed = proposed_residual[:, q_dimension:] / math.sqrt(beta)
        alpha_grid = torch.linspace(
            0.0, 1.0, line_search_points,
            device=local.device, dtype=local.dtype,
        )
        candidate_residuals = [residual_zero]
        candidate_q = [q_zero]
        candidate_momentum = [momentum_zero]
        for candidate_alpha in alpha_grid[1:-1]:
            candidate = (
                local_view + candidate_alpha * (proposed_view - local_view)
            ).reshape_as(local)
            q_value, momentum_value = _components(
                model, example, candidate, support,
            )
            candidate_q.append(q_value)
            candidate_momentum.append(momentum_value)
            candidate_residuals.append(
                _weighted_vector(q_value, momentum_value, beta)
            )
        candidate_residuals.append(proposed_residual)
        candidate_q.append(q_proposed)
        candidate_momentum.append(momentum_proposed)
        residual_stack = torch.stack(candidate_residuals, dim=1)
        objective = residual_stack.square().sum(2)
        selected = objective.argmin(1)
        alpha = alpha_grid[selected]
        corrected = (
            local_view + alpha[:, None, None, None, None]
            * (proposed_view - local_view)
        ).reshape_as(local)
        batch_index = torch.arange(
            transition.batch_size, device=local.device,
        )
        total_corrected = residual_stack[batch_index, selected]
        q_corrected = torch.stack(candidate_q, dim=1)[batch_index, selected]
        momentum_corrected = torch.stack(
            candidate_momentum, dim=1,
        )[batch_index, selected]
        damping_kind = "actual_combined_residual_frozen_grid_line_search"
        line_search_residual_evaluations = line_search_points
        # alpha=0 is reused from the VJP forward and alpha=1 from the single
        # proposal evaluation; only interior points add residual evaluations.
        solver_physical_residual_evaluations = (
            used + 2 + max(line_search_points - 2, 0)
        )
        diagnostic_physical_residual_evaluations = 0

    def ratio(after: Tensor, before: Tensor) -> Tensor:
        return torch.sqrt(
            after.square().sum(1) / before.square().sum(1).clamp_min(1.0e-12)
        )

    return {
        "code": raw_code.detach(), "corrected": corrected.detach(),
        "q24_zero": q_zero.detach().reshape(
            transition.batch_size, transition.cover.tile_count, 24,
        ),
        "qmom_zero": momentum_zero.detach().reshape(
            transition.batch_size, transition.cover.tile_count, 6,
        ),
        "alpha": alpha.detach(), "damping_kind": damping_kind,
        "line_search_points_including_zero": (
            line_search_points if beta > 0.0 else 0
        ),
        "line_search_residual_evaluations": line_search_residual_evaluations,
        "compatibility_cycle": ratio(q_corrected, q_zero).detach(),
        "tile_momentum_cycle": ratio(
            momentum_corrected, momentum_zero,
        ).detach(),
        "combined_cycle": ratio(total_corrected, residual_zero).detach(),
        "cg_iterations": used,
        "cg_relative_residual": float(linear_ratio),
        "teacher_autodiff_chart_evaluations": used + 1,
        "teacher_proposal_chart_decodes": 1,
        "teacher_total_chart_evaluations": used + 2,
        "solver_physical_residual_evaluations": (
            solver_physical_residual_evaluations
        ),
        "diagnostic_physical_residual_evaluations": (
            diagnostic_physical_residual_evaluations
        ),
        "production_student_chart_decodes": 1,
        "production_field_damping_extra_chart_decodes": 0,
    }


def _relative_field(prediction: Tensor, target: Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1.0e-20)
    )


def _metric_row(model, example, solved: dict, prefix: str, elapsed: float) -> dict:
    *_, transition, _ = example
    prediction = _physical_prediction(model, example, solved["corrected"])
    drag, lift = _force_components(prediction, example)
    return {
        f"{prefix}_field": _relative_field(prediction, transition.q_np1),
        f"{prefix}_force": float(_force(prediction, example)),
        f"{prefix}_drag": float(drag),
        f"{prefix}_lift": float(lift),
        f"{prefix}_compatibility_cycle": float(
            solved["compatibility_cycle"].mean()
        ),
        f"{prefix}_tile_momentum_cycle": float(
            solved["tile_momentum_cycle"].mean()
        ),
        f"{prefix}_combined_cycle": float(solved["combined_cycle"].mean()),
        f"{prefix}_alpha": float(solved["alpha"].mean()),
        f"{prefix}_cg_iterations": float(solved["cg_iterations"]),
        f"{prefix}_teacher_chart_evaluations": float(
            solved["teacher_total_chart_evaluations"]
        ),
        f"{prefix}_solver_physical_residual_evaluations": float(
            solved["solver_physical_residual_evaluations"]
        ),
        f"{prefix}_diagnostic_physical_residual_evaluations": float(
            solved["diagnostic_physical_residual_evaluations"]
        ),
        f"{prefix}_line_search_residual_evaluations": float(
            solved["line_search_residual_evaluations"]
        ),
        f"{prefix}_wall_seconds": elapsed,
    }


def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.block_weight <= 0.0:
        raise ValueError("ablation block weight must be positive; beta=0 is internal")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    chart, base, model, statistics = load_candidate(
        argparse.Namespace(candidate=args.chart_checkpoint, graph_radius=None), device,
    )
    # load_candidate refreshes endpoint statistics after its initial .to(); the
    # refresh replaces registered buffers from Python tuples on CPU.
    model = model.to(device)
    if not chart["validation"].get("chart_gate", False):
        raise ValueError("frozen chart did not pass its oracle capacity gate")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    corpus = PairedTrajectoryCorpus(
        args.data_root, base["split"]["validation"], "all",
    )
    rows: list[dict] = []
    records: list[dict] = []
    try:
        for trajectory, paired in enumerate(corpus.paired):
            count = min(args.samples_per_trajectory, len(paired))
            first = 1 if args.exclude_first and len(paired) > 1 else 0
            count = min(count, len(paired) - first)
            indices = np.linspace(first, len(paired) - 1, count, dtype=int)
            for ordinal, index in enumerate(indices):
                seed = args.seed + trajectory * 1_000_003 + ordinal
                example = residual_port_example(
                    model, paired[int(index)], corpus.fine[trajectory], "fine",
                    statistics, halo=args.halo, seed=seed, device=device,
                )
                local = example[0]
                transition = example[-2]
                started = time.perf_counter()
                q_only = solve(
                    model, example, beta=0.0, ridge=args.ridge,
                    cg_iterations=args.cg_iterations,
                    cg_tolerance=args.cg_tolerance,
                    line_search_points=args.line_search_points,
                )
                q_elapsed = time.perf_counter() - started
                started = time.perf_counter()
                augmented = solve(
                    model, example, beta=args.block_weight, ridge=args.ridge,
                    cg_iterations=args.cg_iterations,
                    cg_tolerance=args.cg_tolerance,
                    line_search_points=args.line_search_points,
                )
                augmented_elapsed = time.perf_counter() - started
                zero_prediction = _physical_prediction(model, example, local)
                zero_drag, zero_lift = _force_components(zero_prediction, example)
                row = {
                    "zero_field": _relative_field(
                        zero_prediction, transition.q_np1,
                    ),
                    "zero_force": float(_force(zero_prediction, example)),
                    "zero_drag": float(zero_drag), "zero_lift": float(zero_lift),
                }
                row.update(_metric_row(
                    model, example, q_only, "q_only", q_elapsed,
                ))
                row.update(_metric_row(
                    model, example, augmented, "augmented", augmented_elapsed,
                ))
                rows.append(row)
                records.append({
                    "trajectory": trajectory, "index": int(index), "seed": seed,
                    "q24": augmented["q24_zero"].cpu(),
                    "qmom6": augmented["qmom_zero"].cpu(),
                    "z24": augmented["code"].cpu(),
                })
    finally:
        corpus.close()

    metrics = {
        key: float(np.mean([row[key] for row in rows])) for key in rows[0]
    }
    paired = {}
    for ordinal, metric in enumerate((
        "field", "force", "drag", "lift", "compatibility_cycle",
        "tile_momentum_cycle", "combined_cycle", "wall_seconds",
        "teacher_chart_evaluations",
        "solver_physical_residual_evaluations",
        )):
        paired[f"augmented_minus_q_only_{metric}"] = _paired_bootstrap_interval(
            np.asarray([
                row[f"augmented_{metric}"] - row[f"q_only_{metric}"]
                for row in rows
            ]), seed=args.seed + 101 + 17 * ordinal,
            draws=args.bootstrap_draws,
        )
    for method_offset, method in enumerate(("q_only", "augmented")):
        for ordinal, metric in enumerate(("field", "force", "drag", "lift")):
            paired[f"{method}_minus_zero_{metric}"] = _paired_bootstrap_interval(
                np.asarray([
                    row[f"{method}_{metric}"] - row[f"zero_{metric}"]
                    for row in rows
                ]), seed=args.seed + 701 + 101 * method_offset + 17 * ordinal,
                draws=args.bootstrap_draws,
            )
    for ordinal, metric in enumerate((
        "compatibility_cycle", "tile_momentum_cycle", "combined_cycle",
    )):
        paired[f"augmented_{metric}_minus_one"] = _paired_bootstrap_interval(
            np.asarray([
                row[f"augmented_{metric}"] - 1.0 for row in rows
            ]), seed=args.seed + 401 + 17 * ordinal,
            draws=args.bootstrap_draws,
        )
    gate = {
        "combined_residual_nonexpansion": (
            max(row["augmented_combined_cycle"] for row in rows) <= 1.0 + 1e-6
            and paired["augmented_combined_cycle_minus_one"]["ci95_high"] <= 0.0
        ),
        # q-only is optimized specifically for q_Gamma, so requiring the
        # augmented multi-objective solve to beat that specialist on q_Gamma
        # would preclude any genuine tradeoff.  The theorem requires each
        # retained residual to contract relative to the uncorrected endpoint;
        # the delta versus q-only remains reported, not hidden.
        "compatibility_residual_nonexpansion": paired[
            "augmented_compatibility_cycle_minus_one"
        ]["ci95_high"] <= 0.0,
        "tile_momentum_residual_nonexpansion": paired[
            "augmented_tile_momentum_cycle_minus_one"
        ]["ci95_high"] <= 0.0,
        "field_nonregression_vs_q_only": paired[
            "augmented_minus_q_only_field"
        ]["ci95_high"] <= 0.0,
        "force_nonregression_vs_q_only": paired[
            "augmented_minus_q_only_force"
        ]["ci95_high"] <= 0.0,
        "drag_nonregression_vs_q_only": paired[
            "augmented_minus_q_only_drag"
        ]["ci95_high"] <= 0.0,
        "lift_nonregression_vs_q_only": paired[
            "augmented_minus_q_only_lift"
        ]["ci95_high"] <= 0.0,
        **{
            f"{metric}_nonregression_vs_zero": paired[
                f"augmented_minus_zero_{metric}"
            ]["ci95_high"] <= 0.0
            for metric in ("field", "force", "drag", "lift")
        },
        "finite": all(np.isfinite(list(metrics.values()))),
    }
    gate["reference_passes"] = all(gate.values())
    result = {
        "contract": CONTRACT,
        "chart_checkpoint": str(args.chart_checkpoint.resolve()),
        "split": "validation", "samples": len(rows),
        "block": "tile_momentum_only",
        "block_definition": (
            "per-tile constant/x/y weak moments of the finite-difference "
            "backward-Euler momentum residual; fixed nondimensional unit scale"
        ),
        "block_weight_objective_beta": args.block_weight,
        "ridge": args.ridge,
        "metrics": metrics, "paired_bootstrap": paired, "gate": gate,
        "index_policy": (
            "disjoint_confirmation_excluding_first_per_trajectory"
            if args.exclude_first else "one_calibration_example_per_trajectory"
        ),
        "target_used_by_solver": False,
        "target_used_only_for_postsolve_selection_or_evaluation": True,
        "reference_cost_role": "offline_teacher_not_deployable_one_pass",
        "production_contract": {
            "messages_per_step": 1,
            "student_chart_decodes": 1,
            "field_damping_extra_chart_decodes": 0,
            "field_damping": (
                "frozen target-free actual-residual line search including alpha=0"
            ),
            "line_search_points": args.line_search_points,
            "residual_blocks_share_one_global_message": True,
        },
        "protected_test_trajectories_opened": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.teacher_output is not None and gate["reference_passes"]:
        torch.save({
            "contract": "airfoil_target_free_latent_schur_tile_momentum_teacher_v1",
            "chart_checkpoint": str(args.chart_checkpoint.resolve()),
            "block_weight_objective_beta": args.block_weight,
            "records": records, "split": "validation",
            "protected_test_trajectories_opened": 0,
        }, args.teacher_output)
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-output", type=Path)
    parser.add_argument("--block-weight", type=float, required=True)
    parser.add_argument("--samples-per-trajectory", type=int, default=1)
    parser.add_argument("--exclude-first", action="store_true")
    parser.add_argument("--ridge", type=float, default=.1)
    parser.add_argument("--cg-iterations", type=int, default=16)
    parser.add_argument("--cg-tolerance", type=float, default=1e-3)
    parser.add_argument("--line-search-points", type=int, default=17)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
