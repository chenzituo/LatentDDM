"""Declared validation rollout for a screened fixed-point Airfoil analyzer.

This evaluator has no protected-test option.  It advances a full fine-grid
single cover autoregressively for 20 steps and compares zero, predicted, and
spatially shuffled codes.  Separate zero/predicted trajectories omit the
retained post-POU closure T_X, making its effect explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .airfoil_compatibility_experiment import load_candidate
from .airfoil_defect_analyzer_experiment import _paired_bootstrap_interval
from .airfoil_fixed_point_analyzer import FixedPointSplitAnalyzer
from .airfoil_fixed_point_analyzer_experiment import CONTRACT
from .airfoil_two_level import fluid_divergence_rms
from .force_diagnostics import penalty_force_coefficients
from .interface_trace import interface_trace_relative_square
from .train_codiffuse_schwarz_rollout import relative_l2
from .train_multiresolution_projection import batched
from .train_multitrajectory_multiresolution import PairedTrajectoryCorpus
from .train_one_step_multilevel import interior_vorticity, random_uniform_transition


Tensor = torch.Tensor


def _relative_field(prediction: Tensor, target: Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1.0e-20)
    )


def _relative_vorticity(prediction: Tensor, target: Tensor, dx: float) -> float:
    actual = interior_vorticity(prediction, dx)
    expected = interior_vorticity(target, dx)
    return float(
        torch.linalg.vector_norm(actual - expected)
        / torch.linalg.vector_norm(expected).clamp_min(1.0e-20)
    )


def _load(args, device):
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("contract") != CONTRACT:
        raise ValueError("rollout requires a screened fixed-point split checkpoint")
    chart_path = Path(payload["chart_checkpoint"])
    if hashlib.sha256(chart_path.read_bytes()).hexdigest() != payload["chart_sha256"]:
        raise ValueError("frozen response chart identity changed")
    _, base, model, statistics = load_candidate(
        argparse.Namespace(candidate=chart_path, graph_radius=None), device,
    )
    analyzer = FixedPointSplitAnalyzer(
        model, heads=int(payload["args"].get("heads", 4)),
    ).to(device)
    analyzer.load_state_dict(payload["analyzer"])
    analyzer.eval()
    return payload, base, analyzer, statistics


@torch.no_grad()
def _step(
    analyzer: FixedPointSplitAnalyzer,
    sample: dict[str, Any],
    previous: Tensor,
    current: Tensor,
    dataset,
    statistics,
    *,
    mode: str,
    apply_surface: bool,
    halo: int,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    batch = batched(sample, "fine")
    batch["q_nm1"] = previous
    batch["q_n"] = current
    transition = random_uniform_transition(
        batch, statistics, dataset, device, seed, halo, "fixed",
    )
    one = torch.ones(
        transition.batch_size, device=device, dtype=transition.q_n.dtype,
    )
    physical, scalar = transition.conditions(one)
    motion_physical, motion_scalar = transition.endpoint_conditions()
    endpoint = torch.zeros_like(transition.cover.tile(transition.target))
    local, _ = analyzer.model._local_operator.correct_endpoint_fields(
        endpoint, physical, scalar, transition.cover.tile_count,
        tile_grid_shape=transition.cover.grid_shape,
        motion_physical=motion_physical, motion_scalar=motion_scalar,
    )
    consensus = transition.cover.tile(
        transition.cover.assemble_weighted(local, transition.batch_size),
    )
    baseline_defect = analyzer._defect(
        local, consensus, physical, transition.cover.grid_shape,
        force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
        lift_sign=dataset.rotation_sign,
    )[..., :analyzer.compatibility_dim]
    oracle_shape = (
        transition.batch_size, transition.cover.tile_count,
        analyzer.composer.code_dim,
    )
    translated_mode = {
        "zero": "zero", "predicted": "full", "shuffled": "shuffle_all",
    }[mode]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = analyzer(
        local, consensus, physical, scalar, transition.cover.grid_shape,
        mode=translated_mode, oracle_code=local.new_zeros(oracle_shape),
        force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
        lift_sign=dataset.rotation_sign,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    message_ms = 1.0e3 * (time.perf_counter() - start)
    normalized = transition.cover.assemble_weighted(
        output.corrected, transition.batch_size,
    )
    corrected_consensus = transition.cover.tile(normalized)
    corrected_defect = analyzer._defect(
        output.corrected, corrected_consensus, physical,
        transition.cover.grid_shape,
        force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
        lift_sign=dataset.rotation_sign,
    )[..., :analyzer.compatibility_dim]
    cycle_ratio = torch.sqrt(
        corrected_defect.square().sum((1, 2))
        / baseline_defect.square().sum((1, 2)).clamp_min(1.0e-12)
    ).mean()
    mean = normalized.new_tensor(statistics.delta_mean)[None, :, None, None]
    std = normalized.new_tensor(statistics.delta_std)[None, :, None, None]
    prediction = transition.q_n + normalized * std + mean
    if apply_surface:
        prediction = analyzer.model.correct_surface_level(
            "fine", prediction, batch, dataset,
        )
    target = transition.q_np1
    motion = batch["motion_np1"].to(device)
    trace = torch.sqrt(interface_trace_relative_square(
        prediction, target, motion, dx=dataset.dx, window=dataset.window,
        thickness=float(dataset.physical_parameters["thickness"]),
        pivot_fraction=float(dataset.physical_parameters["pivot_fraction"]),
        stations=64,
    ).clamp_min(0.0))
    predicted_force = penalty_force_coefficients(
        prediction, batch["sdf_np1"].to(device), motion,
        dx=dataset.dx, window=dataset.window,
        penalty_time=dataset.penalty_time, rotation_sign=dataset.rotation_sign,
    )
    target_force = penalty_force_coefficients(
        target, batch["sdf_np1"].to(device), motion,
        dx=dataset.dx, window=dataset.window,
        penalty_time=dataset.penalty_time, rotation_sign=dataset.rotation_sign,
    )
    metrics = {
        "field": _relative_field(prediction, target),
        "vorticity": _relative_vorticity(prediction, target, dataset.dx),
        "interface": float(trace),
        "divergence": float(fluid_divergence_rms(
            prediction, batch["sdf_np1"].to(device), dataset.dx,
        ).mean()),
        "cycle_ratio": float(cycle_ratio),
        "drag_prediction": float(predicted_force.drag),
        "drag_reference": float(target_force.drag),
        "lift_prediction": float(predicted_force.lift),
        "lift_reference": float(target_force.lift),
        "message_calls": float(output.message_calls),
        "message_and_decode_ms": message_ms,
    }
    return prediction, metrics


@torch.no_grad()
def _fixed_point_audit(analyzer, example, dataset, statistics, args, device) -> dict[str, Any]:
    transition = random_uniform_transition(
        batched(example, "fine"), statistics, dataset, device,
        args.seed + 991, args.halo, "fixed",
    )
    one = torch.ones(transition.batch_size, device=device)
    physical, scalar = transition.conditions(one)
    motion_physical, motion_scalar = transition.endpoint_conditions()
    endpoint = torch.zeros_like(transition.cover.tile(transition.target))
    local, _ = analyzer.model._local_operator.correct_endpoint_fields(
        endpoint, physical, scalar, transition.cover.tile_count,
        tile_grid_shape=transition.cover.grid_shape,
        motion_physical=motion_physical, motion_scalar=motion_scalar,
    )
    shape = (transition.batch_size, transition.cover.tile_count, 32)
    output = analyzer(
        local, local, physical, scalar, transition.cover.grid_shape,
        mode="full", oracle_code=local.new_zeros(shape),
        force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
        lift_sign=dataset.rotation_sign,
    )
    return {
        "full_code_nonzeros": int(torch.count_nonzero(output.code)),
        "corrected_bitwise_identity": bool(torch.equal(output.corrected, local)),
        "message_calls": output.message_calls,
    }


def _sequence_summary(rows: list[dict[str, float]], horizon: int) -> dict[str, float]:
    result = {}
    for metric in ("field", "vorticity", "interface", "divergence", "cycle_ratio"):
        result[f"all_step_{metric}"] = float(np.mean([row[metric] for row in rows]))
        result[f"step{horizon}_{metric}"] = rows[-1][metric]
    for force in ("drag", "lift"):
        result[f"all_step_{force}"] = relative_l2(
            [row[f"{force}_prediction"] for row in rows],
            [row[f"{force}_reference"] for row in rows],
        )
        result[f"step{horizon}_{force}"] = relative_l2(
            [rows[-1][f"{force}_prediction"]],
            [rows[-1][f"{force}_reference"]],
        )
    result["message_calls_mean"] = float(np.mean([
        row["message_calls"] for row in rows
    ]))
    result["message_and_decode_ms_mean"] = float(np.mean([
        row["message_and_decode_ms"] for row in rows
    ]))
    result["message_and_decode_ms_p90"] = float(np.quantile([
        row["message_and_decode_ms"] for row in rows
    ], .9))
    return result


def evaluate(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    payload, base, analyzer, statistics = _load(args, device)
    corpus = PairedTrajectoryCorpus(
        args.data_root, base["split"]["validation"], "all",
    )
    variants = {
        "zero_T_X": ("zero", True),
        "predicted_T_X": ("predicted", True),
        "shuffled_T_X": ("shuffled", True),
        "zero_noT": ("zero", False),
        "predicted_noT": ("predicted", False),
        "shuffled_noT": ("shuffled", False),
    }
    sequence_records = {variant: [] for variant in variants}
    try:
        fixed_point = _fixed_point_audit(
            analyzer, corpus.paired[0][0], corpus.fine[0], statistics,
            args, device,
        )
        for trajectory, paired in enumerate(corpus.paired):
            maximum = len(paired) - args.horizon
            starts = np.linspace(
                0, maximum, args.starts_per_trajectory, dtype=int,
            )
            for start_index, start in enumerate(starts):
                for variant, (mode, apply_surface) in variants.items():
                    sample0 = paired[int(start)]
                    previous = sample0["fine_q_nm1"][None].to(device)
                    current = sample0["fine_q_n"][None].to(device)
                    rows = []
                    for step in range(args.horizon):
                        prediction, metrics = _step(
                            analyzer, paired[int(start) + step], previous, current,
                            corpus.fine[trajectory], statistics, mode=mode,
                            apply_surface=apply_surface, halo=args.halo,
                            seed=(args.seed + trajectory * 10_000_019
                                  + start_index * 100_003 + step),
                            device=device,
                        )
                        previous, current = current, prediction
                        rows.append(metrics)
                    sequence_records[variant].append(_sequence_summary(rows, args.horizon))
    finally:
        corpus.close()

    aggregate = {
        variant: {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        for variant, rows in sequence_records.items()
    }
    compared_metrics = [
        f"{time_scope}_{metric}"
        for time_scope in ("all_step", f"step{args.horizon}")
        for metric in ("field", "vorticity", "interface", "divergence", "drag", "lift")
    ]
    cycle_metrics = ["all_step_cycle_ratio", f"step{args.horizon}_cycle_ratio"]
    paired = {}
    for index, metric in enumerate(compared_metrics + cycle_metrics):
        for method, reference, label in (
            ("predicted_T_X", "zero_T_X", "predicted_minus_zero"),
            ("shuffled_T_X", "predicted_T_X", "shuffled_minus_predicted"),
            ("predicted_noT", "zero_noT", "noT_predicted_minus_zero"),
            ("shuffled_noT", "predicted_noT", "noT_shuffled_minus_predicted"),
            ("predicted_T_X", "predicted_noT", "T_X_minus_noT_predicted"),
        ):
            differences = np.asarray([
                left[metric] - right[metric]
                for left, right in zip(sequence_records[method], sequence_records[reference])
            ])
            paired[f"{label}_{metric}"] = _paired_bootstrap_interval(
                differences, seed=args.seed + index * 101 + len(paired),
                draws=args.bootstrap_draws,
            )
    required = compared_metrics
    nonregression = {
        metric: paired[f"noT_predicted_minus_zero_{metric}"]["ci95_high"] <= 0.0
        for metric in required
    }
    field_causality = {}
    for metric in (
        "all_step_field", f"step{args.horizon}_field",
        "all_step_vorticity", f"step{args.horizon}_vorticity",
    ):
        gain = aggregate["zero_noT"][metric] - aggregate["predicted_noT"][metric]
        field_causality[metric] = (
            gain > 0.0
            and aggregate["shuffled_noT"][metric] - aggregate["predicted_noT"][metric]
            >= .5 * gain
            and paired[f"noT_shuffled_minus_predicted_{metric}"]["ci95_low"] >= 0.0
        )
    cycle_nonregression = {
        metric: (
            aggregate["predicted_noT"][metric] <= aggregate["zero_noT"][metric]
            and paired[f"noT_predicted_minus_zero_{metric}"]["ci95_high"] <= 0.0
        )
        for metric in cycle_metrics
    }
    gate = {
        "all_declared_metrics_nonregress_with_paired_ci": all(nonregression.values()),
        "field_assignment_causal": all(field_causality.values()),
        "real_defect_cycle_nonregression": all(cycle_nonregression.values()),
        "fixed_point_exact": fixed_point["corrected_bitwise_identity"]
            and fixed_point["full_code_nonzeros"] == 0,
        "exact_single_message": all(
            row["message_calls_mean"] == 1.0
            for variant in (
                "predicted_T_X", "shuffled_T_X", "predicted_noT", "shuffled_noT",
            )
            for row in sequence_records[variant]
            if not math.isnan(row["message_calls_mean"])
        ),
    }
    gate["accepted"] = all(gate.values())
    result = {
        "contract": "airfoil_fixed_point_full_fine_h20_validation_v1",
        "candidate": str(args.checkpoint.resolve()),
        "candidate_epoch": payload["epoch"],
        "split": "validation", "executor": "full_fine_single_cover",
        "horizon": args.horizon,
        "starts_per_trajectory": args.starts_per_trajectory,
        "paired_sequences": len(next(iter(sequence_records.values()))),
        "bootstrap_draws": args.bootstrap_draws,
        "aggregate": aggregate, "paired_bootstrap": paired,
        "metric_nonregression": nonregression,
        "field_causality": field_causality,
        "real_defect_cycle_nonregression": cycle_nonregression,
        "fixed_point_audit": fixed_point,
        "T_X_accounting": {
            "retained_variants": ["zero_T_X", "predicted_T_X", "shuffled_T_X"],
            "strict_noT_variants": ["zero_noT", "predicted_noT", "shuffled_noT"],
            "additional_global_learned_closure_calls_when_retained": 1,
        },
        "primary_endpoint": "strict_noT_after_POU",
        "retained_T_X_role": "explicitly costed practical ablation only",
        "gate": gate, "protected_test_trajectories_opened": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--starts-per-trajectory", type=int, default=2)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
