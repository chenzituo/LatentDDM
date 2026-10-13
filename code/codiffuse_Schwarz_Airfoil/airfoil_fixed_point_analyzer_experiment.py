"""Train and validate the fixed-point split Airfoil analyzer.

Phase A identifies the frozen 24D compatibility response by target-free
probe--invert cycles.  Phase B fine-tunes on real provisional POU defects and
oracle chart codes.  The 8D transport/load auxiliary is excluded from Phase A
and is reported with its own ablations in validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from codiffuse_Schwarz.compatibility_defect import residual_minimizing_damping

from .airfoil_compatibility_experiment import load_candidate, residual_port_example
from .airfoil_defect_analyzer_experiment import (
    _balanced_force, _force, _force_components, _paired_bootstrap_interval,
    _physical_prediction, fixed_validation_examples,
)
from .airfoil_fixed_point_analyzer import (
    FixedPointSplitAnalyzer, airfoil_probe_invert_loss,
)
from .airfoil_physical_residual import physical_consistency_residual
from .train_multitrajectory_multiresolution import PairedTrajectoryCorpus


Tensor = torch.Tensor
CONTRACT = "airfoil_fixed_point_split_analyzer_v1"


def _codes(analyzer: FixedPointSplitAnalyzer, example):
    local, _, _, _, raw, _, transition, _ = example
    oracle = (raw / analyzer.composer.port_std).reshape(
        transition.batch_size, transition.cover.tile_count, -1,
    )
    consensus = transition.cover.tile(
        transition.cover.assemble_weighted(local, transition.batch_size),
    )
    return oracle, consensus


def _metric_row(prediction: Tensor, corrected: Tensor, target: Tensor, example) -> dict[str, float]:
    *_, transition, _ = example
    drag, lift = _force_components(prediction, example)
    return {
        "field": float(
            torch.linalg.vector_norm(prediction - transition.q_np1)
            / torch.linalg.vector_norm(transition.q_np1).clamp_min(1.0e-20)
        ),
        "delta": float(
            torch.linalg.vector_norm(corrected - target)
            / torch.linalg.vector_norm(target).clamp_min(1.0e-20)
        ),
        "force": float(_force(prediction, example)),
        "drag": float(drag), "lift": float(lift),
    }


def _real_defect_cycle_ratio(
    analyzer: FixedPointSplitAnalyzer,
    example,
    corrected: Tensor,
    baseline_defect: Tensor,
) -> Tensor:
    """Return ||q(v+L(C(q)))||/||q(v)|| in the 24D compatibility port."""
    *_, transition, dataset = example
    corrected_consensus = transition.cover.tile(
        transition.cover.assemble_weighted(corrected, transition.batch_size),
    )
    corrected_defect = analyzer._defect(
        corrected, corrected_consensus, example[2], example[5],
        force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
        lift_sign=dataset.rotation_sign,
    )[..., :analyzer.compatibility_dim]
    baseline = baseline_defect[..., :analyzer.compatibility_dim]
    return torch.sqrt(
        corrected_defect.square().sum((1, 2))
        / baseline.square().sum((1, 2)).clamp_min(1.0e-12)
    )


@torch.no_grad()
def evaluate(
    analyzer: FixedPointSplitAnalyzer,
    examples: list[tuple[Any, ...]],
    statistics,
    *,
    primary_mode: str = "full",
    apply_field_level_damping: bool = False,
) -> dict[str, Any]:
    if primary_mode not in ("full", "compatibility_only"):
        raise ValueError("primary_mode must be full or compatibility_only")
    shuffled_primary = (
        "shuffle_all" if primary_mode == "full" else "shuffle_compatibility"
    )
    analyzer.eval()
    modes = {
        "zero": ("zero", None),
        "predicted": (primary_mode, None),
        "compatibility_only": ("compatibility_only", None),
        "auxiliary_only": ("auxiliary_only", None),
        "shuffle_compatibility": ("shuffle_compatibility", None),
        "shuffle_auxiliary": ("shuffle_auxiliary", None),
        "shuffled": (shuffled_primary, None),
        "radius0": (primary_mode, 0), "radius1": (primary_mode, 1),
        "global": (primary_mode, None), "oracle": ("oracle", None),
    }
    records = []
    for example in examples:
        local, target, physical, scalar, _, grid_shape, transition, dataset = example
        oracle, consensus = _codes(analyzer, example)
        baseline_defect = analyzer._defect(
            local, consensus, physical, grid_shape,
            force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
            lift_sign=dataset.rotation_sign,
        )
        row: dict[str, float] = {}
        viscosity = 1.0 / float(dataset.physical_parameters["reynolds"])
        for label, (mode, radius) in modes.items():
            output = analyzer(
                local, consensus, physical, scalar, grid_shape,
                mode=mode, oracle_code=oracle, graph_radius=radius,
                force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
                lift_sign=dataset.rotation_sign,
            )
            corrected = output.corrected
            damping_alpha = corrected.new_ones(transition.batch_size)
            if apply_field_level_damping and label not in ("zero",):
                candidate_consensus = transition.cover.tile(
                    transition.cover.assemble_weighted(
                        corrected, transition.batch_size,
                    ),
                )
                candidate_q = analyzer._defect(
                    corrected, candidate_consensus, physical, grid_shape,
                    force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
                    lift_sign=dataset.rotation_sign,
                )[..., :analyzer.compatibility_dim]
                q0 = baseline_defect[..., :analyzer.compatibility_dim]
                damping = residual_minimizing_damping(
                    q0.reshape(transition.batch_size, -1),
                    (candidate_q - q0).reshape(transition.batch_size, -1),
                    lower=0.0, upper=1.0,
                )
                tile_count = transition.cover.tile_count
                local_view = local.reshape(
                    transition.batch_size, tile_count, *local.shape[1:],
                )
                candidate_view = corrected.reshape_as(local_view)
                corrected = (
                    local_view + damping.scale[:, None, None, None, None]
                    * (candidate_view - local_view)
                ).reshape_as(local)
                damping_alpha = damping.scale
            prediction = _physical_prediction(
                analyzer, example, corrected, statistics,
            )
            for metric, value in _metric_row(
                prediction, corrected, target, example,
            ).items():
                row[f"{label}_{metric}"] = value
            no_t_prediction = _physical_prediction(
                analyzer, example, corrected, statistics,
                apply_surface=False,
            )
            for metric, value in _metric_row(
                no_t_prediction, corrected, target, example,
            ).items():
                row[f"noT_{label}_{metric}"] = value
            dt = (
                transition.batch["time_np1"] - transition.batch["time_n"]
            ).to(no_t_prediction)
            physical_residual = physical_consistency_residual(
                transition.q_n, no_t_prediction,
                transition.batch["sdf_np1"].to(no_t_prediction.device),
                transition.batch["motion_np1"].to(no_t_prediction.device),
                dt=dt, dx=dataset.dx, window=dataset.window,
                penalty_time=dataset.penalty_time,
                rotation_sign=dataset.rotation_sign,
                viscosity=viscosity,
                previous_sdf=transition.batch["sdf_n"].to(
                    no_t_prediction.device
                ),
                momentum_identity="all_box",
            )
            row[f"noT_{label}_mass_divergence"] = float(
                physical_residual.mass_divergence_rms.mean()
            )
            row[f"noT_{label}_mass_boundary_flux_abs"] = float(
                physical_residual.mass_boundary_flux.abs().mean()
            )
            row[f"noT_{label}_momentum_force_residual"] = float(
                physical_residual.normalized_momentum_force.mean()
            )
            row[f"{label}_calls"] = float(output.message_calls)
            row[f"{label}_field_damping_alpha"] = float(
                damping_alpha.mean()
            )
            row[f"{label}_cycle_ratio"] = float(_real_defect_cycle_ratio(
                analyzer, example, corrected, baseline_defect,
            ).mean())
            if label in ("predicted", "global"):
                row[f"{label}_code"] = float((output.code - oracle).square().mean())
        target_residual = physical_consistency_residual(
            transition.q_n, transition.q_np1,
            transition.batch["sdf_np1"].to(transition.q_np1.device),
            transition.batch["motion_np1"].to(transition.q_np1.device),
            dt=(transition.batch["time_np1"] - transition.batch["time_n"]).to(
                transition.q_np1
            ),
            dx=dataset.dx, window=dataset.window,
            penalty_time=dataset.penalty_time,
            rotation_sign=dataset.rotation_sign, viscosity=viscosity,
            previous_sdf=transition.batch["sdf_n"].to(
                transition.q_np1.device
            ),
            momentum_identity="all_box",
        )
        row["qphys_target_mass_divergence"] = float(
            target_residual.mass_divergence_rms.mean()
        )
        row["qphys_target_momentum_force_residual"] = float(
            target_residual.normalized_momentum_force.mean()
        )
        row["qphys_target_viscous_outer_fraction"] = float(
            (
                target_residual.viscous_outer_x.square()
                + target_residual.viscous_outer_y.square()
            ).sqrt().mean()
            / (
                target_residual.momentum_x.square()
                + target_residual.momentum_y.square()
            ).sqrt().mean().clamp_min(1.0e-12)
        )
        # The complete 32D analyzer must preserve an already compatible state
        # exactly; neither the theorem-facing nor auxiliary block may move it.
        fixed = analyzer(
            local, local, physical, scalar, grid_shape,
            mode="full", oracle_code=oracle,
            force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
            lift_sign=dataset.rotation_sign,
        )
        row["fixed_point_exact"] = float(torch.equal(fixed.corrected, local))
        records.append(row)

    metrics = {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
    }
    paired: dict[str, dict[str, float]] = {}
    comparisons = {
        "noT_predicted_minus_zero_field": (
            "noT_predicted_field", "noT_zero_field", 11,
        ),
        "noT_predicted_minus_zero_force": (
            "noT_predicted_force", "noT_zero_force", 13,
        ),
        "noT_predicted_minus_zero_drag": (
            "noT_predicted_drag", "noT_zero_drag", 17,
        ),
        "noT_predicted_minus_zero_lift": (
            "noT_predicted_lift", "noT_zero_lift", 19,
        ),
        "noT_shuffle_compatibility_minus_predicted_field": (
            "noT_shuffle_compatibility_field", "noT_predicted_field", 23,
        ),
        "noT_shuffled_minus_predicted_field": (
            "noT_shuffled_field", "noT_predicted_field", 31,
        ),
        "noT_radius0_minus_global_field": (
            "noT_radius0_field", "noT_global_field", 47,
        ),
        "noT_radius1_minus_global_field": (
            "noT_radius1_field", "noT_global_field", 59,
        ),
        "predicted_minus_zero_cycle_ratio": (
            "predicted_cycle_ratio", "zero_cycle_ratio", 67,
        ),
        "noT_shuffle_auxiliary_minus_predicted_drag": (
            "noT_shuffle_auxiliary_drag", "noT_predicted_drag", 71,
        ),
        "noT_shuffle_auxiliary_minus_predicted_lift": (
            "noT_shuffle_auxiliary_lift", "noT_predicted_lift", 83,
        ),
        "noT_predicted_minus_zero_mass_divergence": (
            "noT_predicted_mass_divergence", "noT_zero_mass_divergence", 89,
        ),
        "noT_predicted_minus_zero_momentum_force_residual": (
            "noT_predicted_momentum_force_residual",
            "noT_zero_momentum_force_residual", 97,
        ),
        "qphys_target_minus_zero_mass_divergence": (
            "qphys_target_mass_divergence", "noT_zero_mass_divergence", 101,
        ),
        "qphys_target_minus_zero_momentum_force_residual": (
            "qphys_target_momentum_force_residual",
            "noT_zero_momentum_force_residual", 103,
        ),
        "qphys_oracle_minus_zero_mass_divergence": (
            "noT_oracle_mass_divergence", "noT_zero_mass_divergence", 107,
        ),
        "qphys_oracle_minus_zero_momentum_force_residual": (
            "noT_oracle_momentum_force_residual",
            "noT_zero_momentum_force_residual", 109,
        ),
    }
    for name, (method, reference, offset) in comparisons.items():
        paired[name] = _paired_bootstrap_interval(
            np.asarray([record[method] - record[reference] for record in records]),
            seed=20260806 + offset,
        )
    gain = metrics["noT_zero_field"] - metrics["noT_predicted_field"]
    relative_gain = gain / max(metrics["noT_zero_field"], 1.0e-20)
    oracle_gap = metrics["noT_zero_field"] - metrics["noT_oracle_field"]
    gate = {
        "field_improvement_ci95_nonpositive": (
            paired["noT_predicted_minus_zero_field"]["ci95_high"] <= 0.0
        ),
        "field_relative_gain_at_least_0.1_percent": relative_gain >= .001,
        "force_nonregression": (
            metrics["noT_predicted_force"] <= metrics["noT_zero_force"]
            and paired["noT_predicted_minus_zero_force"]["ci95_high"] <= 0.0
        ),
        "drag_nonregression": (
            metrics["noT_predicted_drag"] <= metrics["noT_zero_drag"]
            and paired["noT_predicted_minus_zero_drag"]["ci95_high"] <= 0.0
        ),
        "lift_nonregression": (
            metrics["noT_predicted_lift"] <= metrics["noT_zero_lift"]
            and paired["noT_predicted_minus_zero_lift"]["ci95_high"] <= 0.0
        ),
        "compatibility_assignment_is_causal": (
            paired["noT_shuffle_compatibility_minus_predicted_field"]["ci95_low"] >= 0.0
            and metrics["noT_shuffle_compatibility_field"] - metrics["noT_predicted_field"]
            >= .5 * gain
        ),
        "all_code_assignment_is_causal": (
            paired["noT_shuffled_minus_predicted_field"]["ci95_low"] >= 0.0
            and metrics["noT_shuffled_field"] - metrics["noT_predicted_field"] >= .5 * gain
        ),
        "global_beats_radius0": (
            paired["noT_radius0_minus_global_field"]["ci95_low"] >= 0.0
        ),
        "global_beats_radius1": (
            paired["noT_radius1_minus_global_field"]["ci95_low"] >= 0.0
        ),
        "real_defect_cycle_nonregression": (
            metrics["predicted_cycle_ratio"] <= metrics["zero_cycle_ratio"]
            and paired["predicted_minus_zero_cycle_ratio"]["ci95_high"] <= 0.0
        ),
        "oracle_field_improves": metrics["noT_oracle_field"] < metrics["noT_zero_field"],
        "oracle_force_nonregression": metrics["noT_oracle_force"] <= metrics["noT_zero_force"],
        "oracle_drag_nonregression": metrics["noT_oracle_drag"] <= metrics["noT_zero_drag"],
        "oracle_lift_nonregression": metrics["noT_oracle_lift"] <= metrics["noT_zero_lift"],
        "fixed_point_exact": metrics["fixed_point_exact"] == 1.0,
        "exact_single_message": metrics["global_calls"] == 1.0,
    }
    gate["accepted"] = all(gate.values())
    return {
        "samples": len(records), "metrics": metrics,
        "paired_bootstrap": paired, "relative_field_gain": relative_gain,
        "oracle_field_gap_fraction_closed": (
            gain / oracle_gap if oracle_gap > 0.0 else float("nan")
        ),
        "gate": gate,
        "qoi_auxiliary": {
            "dimension": 8,
            "theorem_role": "excluded from the compatibility inverse claim",
            "force_claim_rule": (
                "force is a hard non-regression gate; no global-message QoI gain "
                "is claimed unless the separate auxiliary shuffle CI is positive"
            ),
        },
        "target_free_physical_residual": {
            "uses_target_drag_or_lift": False,
            "mass_divergence_nonregression": (
                paired["noT_predicted_minus_zero_mass_divergence"]["ci95_high"]
                <= 0.0
            ),
            "momentum_force_nonregression": (
                paired[
                    "noT_predicted_minus_zero_momentum_force_residual"
                ]["ci95_high"] <= 0.0
            ),
            "identification_preflight": {
                "target_mass_lower_than_local": paired[
                    "qphys_target_minus_zero_mass_divergence"
                ]["ci95_high"] < 0.0,
                "target_momentum_lower_than_local": paired[
                    "qphys_target_minus_zero_momentum_force_residual"
                ]["ci95_high"] < 0.0,
                "oracle_reduces_mass_or_momentum": (
                    paired["qphys_oracle_minus_zero_mass_divergence"][
                        "ci95_high"
                    ] < 0.0
                    or paired[
                        "qphys_oracle_minus_zero_momentum_force_residual"
                    ]["ci95_high"] < 0.0
                ),
                "viscous_outer_stress_included": True,
                "target_viscous_outer_fraction": metrics[
                    "qphys_target_viscous_outer_fraction"
                ],
            },
            "role": (
                "validation-only identification diagnostic; cannot become a "
                "training residual unless target and oracle preflight gates pass"
            ),
        },
        "primary_endpoint": "strict_noT_after_POU",
        "primary_analyzer_mode": primary_mode,
        "retained_T_X_role": "explicitly costed practical ablation only",
        "protected_test_trajectories_opened": 0,
        "field_level_damping": apply_field_level_damping,
        "chart_decodes_per_candidate": 1,
    }


def _field_loss(corrected: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    return ((corrected - target).square() * valid).sum() / (
        valid.sum().clamp_min(1.0) * corrected.shape[1]
    )


def train(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    chart, base, model, statistics = load_candidate(
        argparse.Namespace(candidate=args.chart_checkpoint, graph_radius=None), device,
    )
    if not chart["validation"].get("chart_gate", False):
        raise ValueError("the frozen response chart did not pass its oracle gate")
    analyzer = FixedPointSplitAnalyzer(model, heads=args.heads).to(device)
    train_corpus = PairedTrajectoryCorpus(args.data_root, base["split"]["train"], "all")
    validation_corpus = PairedTrajectoryCorpus(
        args.data_root, base["split"]["validation"], "all",
    )
    validation_examples = fixed_validation_examples(
        model, validation_corpus, statistics, args, device,
    )
    optimizer = torch.optim.AdamW(
        analyzer.analysis_parameters, lr=args.lr, weight_decay=args.weight_decay,
    )
    generator = random.Random(args.seed + 17_000_003)
    history: list[dict[str, Any]] = []
    best = float("inf")
    try:
        # Phase A: no target field and no QoI coordinates are consumed.
        for epoch in range(1, args.probe_epochs + 1):
            analyzer.train(); losses = []
            for local_index in range(args.samples_per_epoch):
                _, _, fine, sample = train_corpus.random_sample(generator)
                example = residual_port_example(
                    model, sample, fine, "fine", statistics, halo=args.halo,
                    seed=args.seed + epoch * 13_000_033 + local_index,
                    device=device,
                )
                local, _, physical, scalar, _, grid_shape, transition, dataset = example
                excitation = args.probe_scale * torch.randn(
                    transition.batch_size, transition.cover.tile_count, 24,
                    device=device, dtype=local.dtype,
                )
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = airfoil_probe_invert_loss(
                    analyzer, local, physical, scalar, grid_shape,
                    transition.cover, transition.batch_size, excitation,
                    force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
                    lift_sign=dataset.rotation_sign,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(analyzer.analysis_parameters, 1.0)
                optimizer.step(); losses.append(float(loss.detach()))
            record = {
                "phase": "target_free_probe_invert", "epoch": epoch,
                "inverse_cycle_relative_square": float(np.mean(losses)),
                "qoi_coordinates_used": 0,
            }
            if args.probe_validation_every > 0 \
                    and epoch % args.probe_validation_every == 0:
                validation = evaluate(
                    analyzer, validation_examples, statistics,
                    primary_mode="compatibility_only",
                )
                record["validation"] = validation
                score = validation["metrics"]["noT_predicted_field"]
                payload = {
                    "contract": CONTRACT,
                    "phase": "target_free_probe_invert",
                    "chart_checkpoint": str(args.chart_checkpoint.resolve()),
                    "chart_sha256": hashlib.sha256(
                        args.chart_checkpoint.read_bytes()
                    ).hexdigest(),
                    "analyzer": analyzer.state_dict(), "epoch": epoch,
                    "validation": validation, "args": vars(args),
                    "split": base["split"], "messages_per_physical_step": 1,
                    "compatibility_dimension": 24,
                    "qoi_auxiliary_dimension": 8,
                    "qoi_coordinates_used": 0,
                    "protected_test_trajectories_opened": 0,
                }
                torch.save(payload, args.output_dir / "last_probe_checkpoint.pt")
                if validation["gate"]["accepted"] and score < best:
                    best = score
                    torch.save(payload, args.output_dir / "checkpoint.pt")
            history.append(record); print(json.dumps(record), flush=True)

        # Phase B: real provisional defects and fixed oracle chart coordinates.
        for epoch in range(1, args.finetune_epochs + 1):
            analyzer.train(); losses = []
            for local_index in range(args.samples_per_epoch):
                _, _, fine, sample = train_corpus.random_sample(generator)
                example = residual_port_example(
                    model, sample, fine, "fine", statistics, halo=args.halo,
                    seed=args.seed + 500_000_003 + epoch * 13_000_033 + local_index,
                    device=device,
                )
                endpoint, target, physical, scalar, _, grid_shape, transition, dataset = example
                oracle, consensus = _codes(analyzer, example)
                optimizer.zero_grad(set_to_none=True)
                output = analyzer(
                    endpoint, consensus, physical, scalar, grid_shape,
                    mode="full", oracle_code=oracle,
                    force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
                    lift_sign=dataset.rotation_sign,
                )
                valid = physical[:, -1:].to(endpoint)
                field = _field_loss(output.corrected, target, valid)
                compatibility_code = (
                    output.compatibility_code - oracle[..., :24]
                ).square().mean()
                auxiliary_code = (
                    output.auxiliary_code - oracle[..., 24:]
                ).square().mean()
                prediction = _physical_prediction(
                    analyzer, example, output.corrected, statistics,
                    apply_surface=False,
                )
                force = _balanced_force(prediction, example)
                cycle = _real_defect_cycle_ratio(
                    analyzer, example, output.corrected, output.defect_code,
                ).square().mean()

                shuffled_code = analyzer._shuffle(output.code)
                shuffled_fields = analyzer.composer.decode(
                    endpoint, analyzer.composer.semantic_features(endpoint, physical),
                    shuffled_code, valid, grid_shape,
                )
                shuffled_field_loss = _field_loss(shuffled_fields, target, valid)
                shuffled_prediction = _physical_prediction(
                    analyzer, example, shuffled_fields, statistics,
                    apply_surface=False,
                )
                shuffled_force = _balanced_force(shuffled_prediction, example)
                placement = torch.relu(
                    field + args.field_placement_margin - shuffled_field_loss
                ) + torch.relu(
                    force + args.force_placement_margin - shuffled_force
                )

                # Maintain the target-free inverse while fine-tuning on real q.
                excitation = args.probe_scale * torch.randn(
                    transition.batch_size, transition.cover.tile_count, 24,
                    device=device, dtype=endpoint.dtype,
                )
                inverse, _, _ = airfoil_probe_invert_loss(
                    analyzer, endpoint, physical, scalar, grid_shape,
                    transition.cover, transition.batch_size, excitation,
                    force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
                    lift_sign=dataset.rotation_sign,
                )
                objective = (
                    args.field_weight * field
                    + args.compatibility_code_weight * compatibility_code
                    + args.auxiliary_code_weight * auxiliary_code
                    + args.force_weight * force
                    + args.placement_weight * placement
                    + args.inverse_weight * inverse
                    + args.cycle_weight * cycle
                )
                objective.backward()
                nn.utils.clip_grad_norm_(analyzer.analysis_parameters, 1.0)
                optimizer.step()
                losses.append(tuple(float(value.detach()) for value in (
                    field, compatibility_code, auxiliary_code, force, placement,
                    inverse, cycle,
                )))
            validation = evaluate(analyzer, validation_examples, statistics)
            record = {
                "phase": "real_defect_oracle_finetune", "epoch": epoch,
                "train": {
                    name: float(np.mean([row[index] for row in losses]))
                    for index, name in enumerate((
                        "field", "compatibility_code", "auxiliary_code", "force",
                        "placement", "inverse_cycle", "real_defect_cycle_square",
                    ))
                },
                "validation": validation,
            }
            history.append(record); print(json.dumps(record), flush=True)
            score = (
                validation["metrics"]["noT_predicted_field"]
                + args.selection_force_weight
                * validation["metrics"]["noT_predicted_force"]
            )
            payload = {
                "contract": CONTRACT,
                "chart_checkpoint": str(args.chart_checkpoint.resolve()),
                "chart_sha256": hashlib.sha256(args.chart_checkpoint.read_bytes()).hexdigest(),
                "analyzer": analyzer.state_dict(), "epoch": epoch,
                "validation": validation, "args": vars(args), "split": base["split"],
                "messages_per_physical_step": 1,
                "compatibility_dimension": 24, "qoi_auxiliary_dimension": 8,
                "protected_test_trajectories_opened": 0,
            }
            torch.save(payload, args.output_dir / "last_checkpoint.pt")
            if validation["gate"]["accepted"] and score < best:
                best = score
                torch.save(payload, args.output_dir / "checkpoint.pt")
    finally:
        train_corpus.close(); validation_corpus.close()
    (args.output_dir / "validation_history.json").write_text(json.dumps({
        "contract": CONTRACT, "history": history,
        "protected_test_trajectories_opened": 0,
    }, indent=2) + "\n")
    if not (args.output_dir / "checkpoint.pt").exists():
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-epochs", type=int, default=2)
    parser.add_argument(
        "--probe-validation-every", type=int, default=0,
        help="validate the compatibility-only q branch every N probe epochs",
    )
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--samples-per-epoch", type=int, default=256)
    parser.add_argument("--validation-samples-per-trajectory", type=int, default=8)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-scale", type=float, default=.5)
    parser.add_argument("--field-weight", type=float, default=1.0)
    parser.add_argument("--compatibility-code-weight", type=float, default=.5)
    parser.add_argument("--auxiliary-code-weight", type=float, default=.5)
    parser.add_argument("--force-weight", type=float, default=.5)
    parser.add_argument("--placement-weight", type=float, default=.1)
    parser.add_argument("--field-placement-margin", type=float, default=1e-5)
    parser.add_argument("--force-placement-margin", type=float, default=1e-4)
    parser.add_argument("--inverse-weight", type=float, default=.1)
    parser.add_argument("--cycle-weight", type=float, default=.1)
    parser.add_argument("--selection-force-weight", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
