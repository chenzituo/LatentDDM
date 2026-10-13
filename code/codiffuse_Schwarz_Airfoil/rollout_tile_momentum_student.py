"""Strict validation-only 20-step rollout of the screened Airfoil student."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .airfoil_compatibility_experiment import load_candidate
from .airfoil_defect_analyzer_experiment import _paired_bootstrap_interval
from .airfoil_fixed_point_rollout import (
    _relative_field, _relative_vorticity, _sequence_summary,
)
from .airfoil_tile_momentum_student import SharedCompatibilityMomentumKernel, infer
from .airfoil_two_level import fluid_divergence_rms
from .force_diagnostics import penalty_force_coefficients
from .interface_trace import interface_trace_relative_square
from .train_multiresolution_projection import batched
from .train_multitrajectory_multiresolution import PairedTrajectoryCorpus
from .train_one_step_multilevel import random_uniform_transition
from .train_tile_momentum_student import (
    CONTRACT, FULL_CHART_CONTRACT, RESIDUAL_FIELD_CONTRACT,
    GLOBAL_RESPONSE_LIFT_CONTRACT, RESPONSE_LIFT_CONTRACT,
    STATE_GATED_CONTRACT,
    load_analysis_state,
)


Tensor = torch.Tensor


def _load(args, device):
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("contract") not in (
        CONTRACT, STATE_GATED_CONTRACT, RESIDUAL_FIELD_CONTRACT,
        FULL_CHART_CONTRACT, RESPONSE_LIFT_CONTRACT,
        GLOBAL_RESPONSE_LIFT_CONTRACT,
    ) or not payload.get(
        "validation", {},
    ).get("gate", {}).get("accepted", False):
        raise ValueError("rollout requires a strict-validation-accepted student")
    chart_path = Path(payload["chart_checkpoint"])
    if hashlib.sha256(chart_path.read_bytes()).hexdigest() != payload["chart_sha256"]:
        raise ValueError("student response chart identity changed")
    _, base, model, statistics = load_candidate(
        argparse.Namespace(candidate=chart_path, graph_radius=None), device,
    )
    student = SharedCompatibilityMomentumKernel(
        model, beta=float(payload["beta"]),
        heads=int(payload["args"].get("heads", 4)),
        drive_scale=payload["drive_scale"],
        state_value_rank=int(payload["args"].get("state_value_rank", 0)),
        residual_sketch_width=int(
            payload["args"].get("residual_sketch_width", 0)
        ),
        residual_sketch_full_chart=bool(
            payload["args"].get("residual_sketch_full_chart", False)
        ),
        response_lift_rank=int(
            payload["args"].get("response_lift_rank", 0)
        ),
        response_lift_global_only=bool(
            payload["args"].get("response_lift_global_only", False)
        ),
    ).to(device)
    load_analysis_state(student, payload["analysis_state"])
    student.eval()
    return payload, base, student, statistics


@torch.no_grad()
def _step(
    student, sample: dict[str, Any], previous: Tensor, current: Tensor,
    dataset, statistics, *, mode: str, halo: int, seed: int,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    batch = batched(sample, "fine")
    batch["q_nm1"] = previous; batch["q_n"] = current
    transition = random_uniform_transition(
        batch, statistics, dataset, device, seed, halo, "random",
    )
    one = torch.ones(transition.batch_size, device=device, dtype=current.dtype)
    physical, scalar = transition.conditions(one)
    motion_physical, motion_scalar = transition.endpoint_conditions()
    endpoint = torch.zeros_like(transition.cover.tile(transition.target))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    local, _ = student.model._local_operator.correct_endpoint_fields(
        endpoint, physical, scalar, transition.cover.tile_count,
        tile_grid_shape=transition.cover.grid_shape,
        motion_physical=motion_physical, motion_scalar=motion_scalar,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    local_ms = 1.0e3 * (time.perf_counter() - started)
    example = (
        local, transition.cover.tile(transition.target), physical, scalar,
        local.new_zeros(local.shape[0], 32), transition.cover.grid_shape,
        transition, dataset,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    if mode == "zero":
        corrected = local
        compatibility_cycle = tile_momentum_cycle = combined_cycle = local.new_ones(
            transition.batch_size
        )
        alpha = local.new_zeros(transition.batch_size)
        message_calls = 0
    else:
        output = infer(student, example, shuffle=(mode == "shuffled"))
        corrected = output.corrected
        compatibility_cycle = output.compatibility_cycle
        tile_momentum_cycle = output.tile_momentum_cycle
        combined_cycle = output.combined_cycle
        alpha = output.alpha; message_calls = output.message_calls
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    correction_ms = 1.0e3 * (time.perf_counter() - started)
    normalized = transition.cover.assemble_weighted(
        corrected, transition.batch_size,
    )
    mean = normalized.new_tensor(statistics.delta_mean)[None, :, None, None]
    std = normalized.new_tensor(statistics.delta_std)[None, :, None, None]
    prediction = transition.q_n + normalized * std + mean
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
    return prediction, {
        "field": _relative_field(prediction, target),
        "vorticity": _relative_vorticity(prediction, target, dataset.dx),
        "interface": float(trace),
        "divergence": float(fluid_divergence_rms(
            prediction, batch["sdf_np1"].to(device), dataset.dx,
        ).mean()),
        "cycle_ratio": float(combined_cycle.mean()),
        "compatibility_cycle": float(compatibility_cycle.mean()),
        "tile_momentum_cycle": float(tile_momentum_cycle.mean()),
        "alpha": float(alpha.mean()),
        "drag_prediction": float(predicted_force.drag),
        "drag_reference": float(target_force.drag),
        "lift_prediction": float(predicted_force.lift),
        "lift_reference": float(target_force.lift),
        "message_calls": float(message_calls),
        "message_and_decode_ms": correction_ms,
        "local_endpoint_ms": local_ms,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    payload, base, student, statistics = _load(args, device)
    corpus = PairedTrajectoryCorpus(
        args.data_root, base["split"]["validation"], "all",
    )
    variants = ("zero", "predicted", "shuffled")
    sequences = {variant: [] for variant in variants}
    step_records = {variant: [] for variant in variants}
    fixed_identity = True
    try:
        for trajectory, paired in enumerate(corpus.paired):
            maximum = len(paired) - args.horizon
            starts = np.linspace(0, maximum, args.starts_per_trajectory, dtype=int)
            for start_ordinal, start in enumerate(starts):
                for variant in variants:
                    first = paired[int(start)]
                    previous = first["fine_q_nm1"][None].to(device)
                    current = first["fine_q_n"][None].to(device)
                    rows = []
                    for step in range(args.horizon):
                        prediction, metrics = _step(
                            student, paired[int(start) + step], previous, current,
                            corpus.fine[trajectory], statistics, mode=variant,
                            halo=args.halo,
                            seed=(args.seed + trajectory * 10_000_019
                                  + start_ordinal * 100_003 + step),
                            device=device,
                        )
                        previous, current = current, prediction
                        rows.append(metrics)
                    summary = _sequence_summary(rows, args.horizon)
                    summary["local_endpoint_ms_mean"] = float(np.mean([
                        row["local_endpoint_ms"] for row in rows
                    ]))
                    summary["compatibility_cycle_max"] = float(max(
                        row["compatibility_cycle"] for row in rows
                    ))
                    summary["tile_momentum_cycle_max"] = float(max(
                        row["tile_momentum_cycle"] for row in rows
                    ))
                    summary["combined_cycle_max"] = float(max(
                        row["cycle_ratio"] for row in rows
                    ))
                    sequences[variant].append(summary)
                    step_records[variant].append(rows)
        # Algebraic fixed-point audit does not require a data target.
        example = paired[0]
        # Accepted one-step checkpoint already contains a 40-case exact audit;
        # preserve that authoritative result rather than opening another split.
        fixed_identity = bool(payload["validation"]["gate"][
            "zero_residual_fixed_point_exact"
        ])
    finally:
        corpus.close()
    aggregate = {
        variant: {
            key: float(np.mean([row[key] for row in values]))
            for key in values[0]
        } for variant, values in sequences.items()
    }
    compared = [
        f"{scope}_{metric}"
        for scope in ("all_step", f"step{args.horizon}")
        for metric in ("field", "vorticity", "interface", "divergence", "drag", "lift")
    ]
    paired = {}
    for offset, metric in enumerate(compared):
        for left, right, label in (
            ("predicted", "zero", "predicted_minus_zero"),
            ("shuffled", "predicted", "shuffled_minus_predicted"),
        ):
            paired[f"{label}_{metric}"] = _paired_bootstrap_interval(
                np.asarray([
                    a[metric] - b[metric]
                    for a, b in zip(sequences[left], sequences[right])
                ]), seed=args.seed + 101 * offset + len(paired),
                draws=args.bootstrap_draws,
            )
    nonregression = {
        metric: paired[f"predicted_minus_zero_{metric}"]["ci95_high"] <= 0.0
        for metric in compared
    }
    causal = {}
    for metric in (
        "all_step_field", f"step{args.horizon}_field",
        "all_step_vorticity", f"step{args.horizon}_vorticity",
    ):
        gain = aggregate["zero"][metric] - aggregate["predicted"][metric]
        causal[metric] = (
            gain > 0.0
            and aggregate["shuffled"][metric] - aggregate["predicted"][metric]
            >= .5 * gain
            and paired[f"shuffled_minus_predicted_{metric}"]["ci95_low"] >= 0.0
        )
    gate = {
        "all_field_vorticity_interface_divergence_force_metrics_nonregress": all(
            nonregression.values()
        ),
        "field_and_vorticity_assignment_causal": all(causal.values()),
        "all_combined_residual_cycles_nonexpansive": max(
            row["combined_cycle_max"] for row in sequences["predicted"]
        ) <= 1.0 + 2.0e-5,
        "fixed_point_exact": fixed_identity,
        "exact_single_message": all(
            row["message_calls_mean"] == 1.0 for row in sequences["predicted"]
        ),
    }
    gate["accepted"] = all(gate.values())
    result = {
        "contract": "airfoil_qgamma_qmom_h20_validation_v1",
        "candidate": str(args.checkpoint.resolve()),
        "candidate_epoch": payload["epoch"],
        "split": "validation", "horizon": args.horizon,
        "starts_per_trajectory": args.starts_per_trajectory,
        "paired_sequences": len(sequences["predicted"]),
        "aggregate": aggregate, "paired_bootstrap": paired,
        "metric_nonregression": nonregression,
        "assignment_causality": causal, "gate": gate,
        "production_contract": payload["validation"]["production_contract"],
        "model_owned_rollout": True,
        "primary_endpoint": "strict_noT_after_normalized_POU",
        "target_used_by_communicator_or_damping": False,
        "protected_test_trajectories_opened": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


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
