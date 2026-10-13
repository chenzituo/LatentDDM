"""Target-free reduced Gauss--Newton reference for the frozen Airfoil chart.

For the 24D-per-tile compatibility coordinate, this evaluates the matrix-free
linearization

    J = d q(v + L(z)) / dz |_(z=0),
    z_GN = argmin_z ||q(v) + J z||^2 + lambda ||z||^2,

then applies residual-only scalar damping.  Targets are used only after the
solve to evaluate the validation reference.  There is no protected-test path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from codiffuse_Schwarz.compatibility_defect import residual_minimizing_damping
from codiffuse_Schwarz.residual_port import ResidualPortSpec, project_residual_ports
from .airfoil_compatibility_experiment import load_candidate, residual_port_example
from .airfoil_defect_analyzer_experiment import (
    _force, _force_components, _paired_bootstrap_interval, _physical_prediction,
)
from .airfoil_fixed_point_analyzer import compatibility_face_support
from .train_multitrajectory_multiresolution import PairedTrajectoryCorpus


Tensor = torch.Tensor
TRACE_SPEC = ResidualPortSpec(
    dimension=2, field_channels=3, polynomial_degree=1,
    maximum_modes_per_face=2, appended_functionals=0,
)


def _q24(model, example, fields: Tensor) -> Tensor:
    _, _, physical, _, _, grid_shape, transition, _ = example
    consensus = transition.cover.tile(
        transition.cover.assemble_weighted(fields, transition.batch_size),
    )
    valid = physical[:, -1:].to(fields).clamp(0.0, 1.0)
    raw = project_residual_ports(
        (consensus - fields) * valid, valid, TRACE_SPEC,
    )
    return (raw / model.composer.port_std[:24]).reshape(
        transition.batch_size, transition.cover.tile_count, 24,
    )


def _decode24(model, example, code24: Tensor) -> Tensor:
    local, _, physical, _, _, grid_shape, transition, _ = example
    valid = physical[:, -1:].to(local)
    semantics = model.composer.semantic_features(local, physical)
    auxiliary = code24.new_zeros(
        transition.batch_size, transition.cover.tile_count, 8,
    )
    return model.composer.decode(
        local, semantics, torch.cat((code24, auxiliary), dim=-1),
        valid, grid_shape,
    )


def _conjugate_gradient(matvec, right: Tensor, iterations: int, tolerance: float):
    solution = torch.zeros_like(right)
    residual = right - matvec(solution)
    direction = residual.clone()
    squared = torch.dot(residual, residual)
    initial = squared.detach().clamp_min(1.0e-30)
    used = 0
    for used in range(1, iterations + 1):
        image = matvec(direction)
        denominator = torch.dot(direction, image).clamp_min(1.0e-30)
        step = squared / denominator
        solution = solution + step * direction
        residual = residual - step * image
        next_squared = torch.dot(residual, residual)
        if float(torch.sqrt(next_squared / initial)) <= tolerance:
            squared = next_squared
            break
        direction = residual + (next_squared / squared.clamp_min(1.0e-30)) * direction
        squared = next_squared
    return solution, used, torch.sqrt(squared / initial)


def latent_schur_solve(model, example, *, ridge: float, cg_iterations: int, cg_tolerance: float):
    local, _, physical, _, _, grid_shape, transition, _ = example
    support = compatibility_face_support(
        physical[:, -1:].to(local), grid_shape,
    )
    zero = local.new_zeros(transition.batch_size, transition.cover.tile_count, 24)

    def residual_function(code: Tensor) -> Tensor:
        return _q24(model, example, _decode24(model, example, code * support)) * support

    q_zero, vjp = torch.func.vjp(residual_function, zero)
    right = -vjp(q_zero)[0].reshape(-1)

    def normal(vector: Tensor) -> Tensor:
        shaped = vector.reshape_as(zero)
        j_vector = torch.func.jvp(
            residual_function, (zero,), (shaped,),
        )[1]
        return (vjp(j_vector)[0] + ridge * shaped).reshape(-1)

    solution, used, linear_ratio = _conjugate_gradient(
        normal, right, cg_iterations, cg_tolerance,
    )
    solution = solution.reshape_as(zero) * support
    j_solution = torch.func.jvp(
        residual_function, (zero,), (solution,),
    )[1]
    linear_damping = residual_minimizing_damping(
        q_zero.reshape(transition.batch_size, -1),
        j_solution.reshape(transition.batch_size, -1),
        lower=0.0, upper=1.0,
    )
    # The chart ends in tanh and is nonlinear in z. Decode the proposed
    # direction exactly once, then damp in broken-field space. q24 is linear in
    # that field, so closed-form alpha with [0,1] (including alpha=0) proves
    # actual q nonexpansion without pretending L(alpha z)=alpha L(z).
    proposed = _decode24(model, example, solution)
    proposed_q = _q24(model, example, proposed) * support
    actual_damping = residual_minimizing_damping(
        q_zero.reshape(transition.batch_size, -1),
        (proposed_q - q_zero).reshape(transition.batch_size, -1),
        lower=0.0, upper=1.0,
    )
    tile_count = transition.cover.tile_count
    local_view = local.reshape(
        transition.batch_size, tile_count, *local.shape[1:],
    )
    proposed_view = proposed.reshape_as(local_view)
    corrected = (
        local_view + actual_damping.scale[:, None, None, None, None]
        * (proposed_view - local_view)
    ).reshape_as(local)
    q_corrected = _q24(model, example, corrected) * support
    cycle = torch.sqrt(
        q_corrected.square().sum((1, 2))
        / q_zero.square().sum((1, 2)).clamp_min(1.0e-12)
    )
    return {
        "code": solution.detach(), "corrected": corrected.detach(),
        "q_zero": q_zero.detach(), "q_corrected": q_corrected.detach(),
        "cycle_ratio": cycle.detach(), "alpha": actual_damping.scale.detach(),
        "linearized_alpha": linear_damping.scale.detach(),
        "field_level_damping": True,
        # This direct Gauss--Newton solve is an offline reference, not the
        # deployable one-message algorithm.  The VJP evaluates the chart once,
        # every CG matvec evaluates one JVP, and the post-CG linear diagnostic
        # evaluates one more JVP.  The proposed field is then decoded once.
        "teacher_autodiff_chart_evaluations": used + 2,
        "teacher_proposal_chart_decodes": 1,
        "teacher_total_chart_evaluations": used + 3,
        # At deployment a distilled student proposes z with one kernel message;
        # its chart is decoded once and field-level damping reuses that field.
        "production_student_chart_decodes": 1,
        "production_field_damping_extra_chart_decodes": 0,
        "cg_iterations": used, "cg_relative_residual": float(linear_ratio),
    }


def evaluate(args: argparse.Namespace) -> None:
    if args.output.exists() or args.teacher_output.exists():
        raise FileExistsError("reference output already exists")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    chart, base, model, statistics = load_candidate(
        argparse.Namespace(candidate=args.chart_checkpoint, graph_radius=None), device,
    )
    if not chart["validation"].get("chart_gate", False):
        raise ValueError("frozen chart did not pass its oracle capacity gate")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    corpus = PairedTrajectoryCorpus(
        args.data_root, base["split"]["validation"], "all",
    )
    rows = []
    teachers = []
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
                solved = latent_schur_solve(
                    model, example, ridge=args.ridge,
                    cg_iterations=args.cg_iterations,
                    cg_tolerance=args.cg_tolerance,
                )
                local, target, _, _, _, _, transition, _ = example
                zero_prediction = _physical_prediction(
                    model, example, local, statistics, apply_surface=False,
                )
                gn_prediction = _physical_prediction(
                    model, example, solved["corrected"], statistics,
                    apply_surface=False,
                )
                zero_drag, zero_lift = _force_components(zero_prediction, example)
                gn_drag, gn_lift = _force_components(gn_prediction, example)
                rows.append({
                    "zero_field": float(
                        torch.linalg.vector_norm(zero_prediction - transition.q_np1)
                        / torch.linalg.vector_norm(transition.q_np1).clamp_min(1e-20)
                    ),
                    "gn_field": float(
                        torch.linalg.vector_norm(gn_prediction - transition.q_np1)
                        / torch.linalg.vector_norm(transition.q_np1).clamp_min(1e-20)
                    ),
                    "zero_force": float(_force(zero_prediction, example)),
                    "gn_force": float(_force(gn_prediction, example)),
                    "zero_drag": float(zero_drag), "gn_drag": float(gn_drag),
                    "zero_lift": float(zero_lift), "gn_lift": float(gn_lift),
                    "cycle_ratio": float(solved["cycle_ratio"].mean()),
                    "alpha": float(solved["alpha"].mean()),
                    "cg_iterations": float(solved["cg_iterations"]),
                    "teacher_autodiff_chart_evaluations": float(
                        solved["teacher_autodiff_chart_evaluations"]
                    ),
                    "teacher_total_chart_evaluations": float(
                        solved["teacher_total_chart_evaluations"]
                    ),
                    "cg_relative_residual": solved["cg_relative_residual"],
                })
                teachers.append({
                    "trajectory": trajectory, "index": int(index),
                    "seed": args.seed + trajectory * 1_000_003 + ordinal,
                    "q24": solved["q_zero"].cpu(),
                    "z24": solved["code"].cpu(),
                })
    finally:
        corpus.close()

    metrics = {
        key: float(np.mean([row[key] for row in rows])) for key in rows[0]
    }
    paired = {
        metric: _paired_bootstrap_interval(
            np.asarray([row[f"gn_{metric}"] - row[f"zero_{metric}"] for row in rows]),
            seed=args.seed + offset, draws=args.bootstrap_draws,
        )
        for metric, offset in (("field", 11), ("force", 23), ("drag", 37), ("lift", 41))
    }
    cycle_ci = _paired_bootstrap_interval(
        np.asarray([row["cycle_ratio"] - 1.0 for row in rows]),
        seed=args.seed + 53, draws=args.bootstrap_draws,
    )
    gate = {
        "cycle_residual_decreases": cycle_ci["ci95_high"] <= 0.0,
        "field_nonregression": paired["field"]["ci95_high"] <= 0.0,
        "force_nonregression": paired["force"]["ci95_high"] <= 0.0,
        "drag_nonregression": paired["drag"]["ci95_high"] <= 0.0,
        "lift_nonregression": paired["lift"]["ci95_high"] <= 0.0,
        "finite_solution": all(np.isfinite(list(metrics.values()))),
    }
    gate["teacher_eligible"] = all(gate.values())
    result = {
        "contract": "airfoil_target_free_latent_schur_reference_v1",
        "chart_checkpoint": str(args.chart_checkpoint.resolve()),
        "split": "validation", "samples": len(rows),
        "linearization": "J=d q(v+L(z))/dz at z=0; 24D compatibility only",
        "solver": (
            "matrix-free regularized normal equations + one-decode "
            "actual-q field-level residual-only damping"
        ),
        "field_level_damping": True,
        "reference_cost_accounting": {
            "role": "offline_teacher_not_deployable_one_pass",
            "autodiff_chart_evaluations_mean": metrics[
                "teacher_autodiff_chart_evaluations"
            ],
            "total_chart_evaluations_mean": metrics[
                "teacher_total_chart_evaluations"
            ],
            "production_student_chart_decodes": 1,
            "production_field_damping_extra_chart_decodes": 0,
        },
        "ridge": args.ridge, "metrics": metrics,
        "index_policy": (
            "disjoint_confirmation_excluding_first_per_trajectory"
            if args.exclude_first else "includes_first_per_trajectory"
        ),
        "paired_bootstrap": paired, "cycle_minus_one_bootstrap": cycle_ci,
        "gate": gate, "target_used_by_solver": False,
        "primary_endpoint": "strict_noT_after_POU",
        "protected_test_trajectories_opened": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if gate["teacher_eligible"]:
        torch.save({
            "contract": "airfoil_target_free_latent_schur_teacher_v1",
            "chart_checkpoint": str(args.chart_checkpoint.resolve()),
            "records": teachers, "split": "validation",
            "protected_test_trajectories_opened": 0,
        }, args.teacher_output)
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-output", type=Path, required=True)
    parser.add_argument("--samples-per-trajectory", type=int, default=1)
    parser.add_argument(
        "--exclude-first", action="store_true",
        help="exclude the one-example-per-trajectory ridge-calibration subset",
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--cg-iterations", type=int, default=16)
    parser.add_argument("--cg-tolerance", type=float, default=1e-3)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
