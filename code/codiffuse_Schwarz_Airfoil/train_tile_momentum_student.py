"""Train one q_Gamma+momentum residual message on a frozen local endpoint.

Only train and validation trajectories are addressable.  The student receives
the complete target-free residual once, applies one fixed-context-linear global
kernel, decodes one chart direction, and uses an exact quartic field-space
damping safeguard.  The default objective distills a target-free latent-Schur
teacher.  An explicit control can instead use the exact supervised response
objective used by the same-cover Patch Transformer baseline; checkpoint
provenance records which training signal was used.
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
import torch.nn.functional as F

from .airfoil_compatibility_experiment import load_candidate, residual_port_example
from .airfoil_defect_analyzer_experiment import (
    _force, _force_components, _paired_bootstrap_interval,
)
from .airfoil_latent_schur_physical_ablation import _physical_prediction
from .airfoil_tile_momentum_student import (
    SharedCompatibilityMomentumKernel, complete_residual_drive, infer,
    pou_field_defect, pou_transport_load_defect,
)
from .force_diagnostics import penalty_force_coefficients
from .matched_patch_response_objective import (
    matched_patch_response_objective,
)
from .train_multitrajectory_multiresolution import PairedTrajectoryCorpus


CONTRACT = "airfoil_qgamma_qmom_one_message_student_v1"
STATE_GATED_CONTRACT = "airfoil_qgamma_qmom_state_gated_one_message_student_v2"
RESIDUAL_FIELD_CONTRACT = (
    "airfoil_qgamma_qmom_residual_field_one_message_student_v3"
)
FULL_CHART_CONTRACT = (
    "airfoil_complete_residual_full_chart_one_message_student_v4"
)
RESPONSE_LIFT_CONTRACT = (
    "airfoil_augmented_response_chart_one_message_student_v5"
)
GLOBAL_RESPONSE_LIFT_CONTRACT = (
    "airfoil_global_residual_response_chart_one_message_student_v6"
)
TEACHER_CONTRACT = "airfoil_target_free_latent_schur_tile_momentum_train_teacher_v2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _drive_scale(records: list[dict[str, Any]], beta: float) -> torch.Tensor:
    total = torch.zeros(30, dtype=torch.float64)
    count = 0
    for record in records:
        drive = torch.cat((
            record["q24"].reshape(-1, 24).double(),
            np.sqrt(beta) * record["qmom6"].reshape(-1, 6).double(),
        ), dim=1)
        total += drive.square().sum(0)
        count += drive.shape[0]
    return (total / count).sqrt().clamp_min(1.0e-4).float()


def reception_identifiability_loss(
    global_code: torch.Tensor, radius0_code: torch.Tensor,
    radius1_code: torch.Tensor, teacher_code: torch.Tensor, *, margin: float,
) -> torch.Tensor:
    """Require the full message to fit a target-free teacher better locally.

    A radius-limited communicator cannot drive this loss below ``2*margin``
    when all three codes coincide.  The complete validation gate still decides
    whether the learned nonlocal reception improves fields rather than merely
    making the radius controls worse.
    """
    if margin <= 0.0:
        raise ValueError("reception margin must be positive")
    errors = [
        F.smooth_l1_loss(code, teacher_code, reduction="none").mean((1, 2))
        for code in (global_code, radius0_code, radius1_code)
    ]
    return (
        torch.relu(errors[0] + margin - errors[1])
        + torch.relu(errors[0] + margin - errors[2])
    ).mean()


def _analysis_state(student: SharedCompatibilityMomentumKernel) -> dict[str, Any]:
    state = {
        "encoder": student.composer.encoder.state_dict(),
        "pool": student.composer.pool.state_dict(),
        "scalar_encoder": student.composer.scalar_encoder.state_dict(),
        "residual_map": student.residual_map.state_dict(),
    }
    if student.residual_sketch is not None:
        state["residual_sketch"] = student.residual_sketch.state_dict()
    if student.response_lift is not None:
        state["response_lift"] = student.response_lift.state_dict()
    return state


def load_analysis_state(
    student, state: dict[str, Any], *, identity_lift: bool = False,
    residual_sketch_lift: bool = False,
) -> dict[str, list[str]]:
    student.composer.encoder.load_state_dict(state["encoder"])
    student.composer.pool.load_state_dict(state["pool"])
    student.composer.scalar_encoder.load_state_dict(state["scalar_encoder"])
    incompatible = student.residual_map.load_state_dict(
        state["residual_map"], strict=not identity_lift,
    )
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if identity_lift:
        expected = {
            "global_kernel.state_value_coefficients.weight",
            "global_kernel.state_value_basis",
        }
        if set(missing) != expected or unexpected:
            raise ValueError(
                "state-gated identity lift has unexpected state mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
    elif missing or unexpected:
        raise ValueError(
            f"analysis state mismatch: missing={missing}, unexpected={unexpected}"
        )
    source_has_sketch = "residual_sketch" in state
    target_has_sketch = student.residual_sketch is not None
    if source_has_sketch and target_has_sketch:
        student.residual_sketch.load_state_dict(
            state["residual_sketch"], strict=True,
        )
    elif residual_sketch_lift and target_has_sketch and not source_has_sketch:
        pass
    elif source_has_sketch != target_has_sketch:
        raise ValueError(
            "source and target residual-field sketch states are incompatible"
        )
    source_has_response_lift = "response_lift" in state
    target_has_response_lift = student.response_lift is not None
    if source_has_response_lift and target_has_response_lift:
        student.response_lift.load_state_dict(
            state["response_lift"], strict=True,
        )
    elif residual_sketch_lift and target_has_response_lift \
            and not source_has_response_lift:
        pass
    elif source_has_response_lift != target_has_response_lift:
        raise ValueError(
            "source and target response-chart lift states are incompatible"
        )
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "residual_sketch_identity_lift": bool(residual_sketch_lift),
        "response_chart_identity_lift": bool(
            residual_sketch_lift and target_has_response_lift
            and not source_has_response_lift
        ),
    }


def _relative_field(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(prediction - target)
        / torch.linalg.vector_norm(target).clamp_min(1.0e-20)
    )


def _field_metrics(model, example, corrected: torch.Tensor) -> dict[str, float]:
    prediction = _physical_prediction(model, example, corrected)
    target = example[-2].q_np1
    drag, lift = _force_components(prediction, example)
    return {
        "field": _relative_field(prediction, target),
        "force": float(_force(prediction, example)),
        "drag": float(drag), "lift": float(lift),
    }


@torch.no_grad()
def evaluate(
    student: SharedCompatibilityMomentumKernel, corpus, statistics,
    args: argparse.Namespace, device: torch.device,
) -> dict[str, Any]:
    student.eval()
    rows: list[dict[str, float]] = []
    fixed_identity = True
    for trajectory, paired in enumerate(corpus.paired):
        count = min(args.validation_samples_per_trajectory, max(len(paired) - 1, 0))
        if count <= 0:
            continue
        indices = np.linspace(1, len(paired) - 1, count, dtype=int)
        for ordinal, index in enumerate(indices):
            example = residual_port_example(
                student.model, paired[int(index)], corpus.fine[trajectory], "fine",
                statistics, halo=args.halo,
                seed=args.seed + trajectory * 1_000_003 + ordinal,
                device=device,
            )
            local, _, physical, scalar, _, grid_shape, transition, _ = example
            row: dict[str, float] = {}
            for key, value in _field_metrics(student.model, example, local).items():
                row[f"zero_{key}"] = value
            q24, qmom6, _, _ = complete_residual_drive(student.model, example)
            residual_field = (
                pou_field_defect(example, local)
                if student.residual_sketch is not None else None
            )
            auxiliary_residual = (
                pou_transport_load_defect(student.model, example, local)
                if student.residual_sketch_full_chart else None
            )
            zero_code = student.code(
                local, physical, scalar, grid_shape,
                torch.zeros_like(q24), torch.zeros_like(qmom6),
                residual_field=(
                    torch.zeros_like(residual_field)
                    if residual_field is not None else None
                ),
                auxiliary_residual=(
                    torch.zeros_like(auxiliary_residual)
                    if auxiliary_residual is not None else None
                ),
            )
            fixed_identity &= bool(torch.count_nonzero(zero_code) == 0)
            fixed_identity &= bool(torch.equal(
                student.decode(local, physical, grid_shape, zero_code), local,
            ))
            modes = {
                "predicted": (None, False), "shuffled": (None, True),
                "radius0": (0, False), "radius1": (1, False),
            }
            for label, (radius, shuffled) in modes.items():
                output = infer(
                    student, example, graph_radius=radius, shuffle=shuffled,
                )
                for key, value in _field_metrics(
                    student.model, example, output.corrected,
                ).items():
                    row[f"{label}_{key}"] = value
                row[f"{label}_compatibility_cycle"] = float(
                    output.compatibility_cycle.mean()
                )
                row[f"{label}_tile_momentum_cycle"] = float(
                    output.tile_momentum_cycle.mean()
                )
                row[f"{label}_combined_cycle"] = float(
                    output.combined_cycle.mean()
                )
                row[f"{label}_alpha"] = float(output.alpha.mean())
                row[f"{label}_message_calls"] = float(output.message_calls)
                row[f"{label}_qmom_analyses"] = float(output.qmom_analyses)
            rows.append(row)
    if not rows:
        raise ValueError("validation split supplied no disjoint examples")
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    paired: dict[str, dict[str, float]] = {}
    comparisons = []
    for metric in ("field", "force", "drag", "lift"):
        comparisons.append((f"predicted_minus_zero_{metric}",
                            f"predicted_{metric}", f"zero_{metric}"))
    comparisons.extend((
        ("shuffled_minus_predicted_field", "shuffled_field", "predicted_field"),
        ("radius0_minus_predicted_field", "radius0_field", "predicted_field"),
        ("radius1_minus_predicted_field", "radius1_field", "predicted_field"),
    ))
    for offset, (name, left, right) in enumerate(comparisons):
        paired[name] = _paired_bootstrap_interval(
            np.asarray([row[left] - row[right] for row in rows]),
            seed=args.seed + 101 + 17 * offset, draws=args.bootstrap_draws,
        )
    for offset, metric in enumerate((
        "compatibility_cycle", "tile_momentum_cycle", "combined_cycle",
    )):
        paired[f"predicted_{metric}_minus_one"] = _paired_bootstrap_interval(
            np.asarray([row[f"predicted_{metric}"] - 1.0 for row in rows]),
            seed=args.seed + 401 + 17 * offset, draws=args.bootstrap_draws,
        )
    gain = metrics["zero_field"] - metrics["predicted_field"]
    gate = {
        **{
            f"{metric}_nonregression": paired[
                f"predicted_minus_zero_{metric}"
            ]["ci95_high"] <= 0.0
            for metric in ("field", "force", "drag", "lift")
        },
        "field_relative_gain_at_least_0.1_percent": (
            gain / max(metrics["zero_field"], 1.0e-20) >= 0.001
        ),
        **{
            f"{metric}_residual_nonexpansion": (
                paired[f"predicted_{metric}_minus_one"]["ci95_high"] <= 0.0
                and max(row[f"predicted_{metric}"] for row in rows) <= 1.0 + 2.0e-5
            ) for metric in (
                "compatibility_cycle", "tile_momentum_cycle", "combined_cycle",
            )
        },
        "spatial_assignment_is_causal": (
            paired["shuffled_minus_predicted_field"]["ci95_low"] >= 0.0
            and metrics["shuffled_field"] - metrics["predicted_field"] >= 0.5 * gain
        ),
        "global_beats_radius0": paired[
            "radius0_minus_predicted_field"
        ]["ci95_low"] >= 0.0,
        "global_beats_radius1": paired[
            "radius1_minus_predicted_field"
        ]["ci95_low"] >= 0.0,
        "zero_residual_fixed_point_exact": fixed_identity,
        "exact_single_message": metrics["predicted_message_calls"] == 1.0,
        "four_target_free_momentum_analyses": metrics[
            "predicted_qmom_analyses"
        ] == 4.0,
        "finite": bool(np.isfinite(list(metrics.values())).all()),
    }
    gate["accepted"] = all(gate.values())
    return {
        "contract": student.contract, "split": "validation", "samples": len(rows),
        "index_policy": "disjoint_confirmation_excluding_first_per_trajectory",
        "metrics": metrics, "paired_bootstrap": paired, "gate": gate,
        "records": rows,
        "production_contract": {
            "local_endpoint_calls_per_step": 1,
            "global_residual_messages_per_step": 1,
            "chart_decodes_per_step": 1,
            "damping": "target-free exact quartic minimization on alpha in [0,1]",
            "momentum_residual_analyses_including_initial_and_audit": 4,
            "POU_assembly": "normalized weighted cover assembly",
        },
        "target_used_by_communicator_or_damping": False,
        "target_used_only_for_validation_metrics": True,
        "protected_test_trajectories_opened": 0,
    }


def train(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    teacher = torch.load(args.teacher, map_location="cpu", weights_only=False)
    if teacher.get("contract") != TEACHER_CONTRACT or teacher.get("split") != "train":
        raise ValueError("a train-only q_Gamma+momentum teacher is required")
    if teacher.get("target_used_by_solver") is not False or teacher.get(
        "protected_test_trajectories_opened",
    ) != 0 or teacher.get("cover_offsets_explicit") is not True \
            or not teacher.get(
                "target_replacement_invariance_audit", {}
            ).get("passed", False) \
            or not all(teacher.get("strict_confirmation_gate", {}).values()):
        raise ValueError("teacher provenance or confirmation gate failed")
    chart, base, model, statistics = load_candidate(
        argparse.Namespace(candidate=args.chart_checkpoint, graph_radius=None), device,
    )
    if teacher.get("chart_sha256") != _sha(args.chart_checkpoint):
        raise ValueError("teacher and student chart identities differ")
    records = list(teacher["records"])
    beta = float(teacher["block_weight_objective_beta"])
    # The residual coordinate is part of checkpoint identity.  A smoke run may
    # optimize only a prefix of the records, but it must retain the train-split
    # normalization used by the accepted parent and the full production run.
    scale = _drive_scale(records, beta)
    if args.max_train_records is not None:
        records = records[:args.max_train_records]
    student = SharedCompatibilityMomentumKernel(
        model, beta=beta, heads=args.heads, drive_scale=scale,
        state_value_rank=args.state_value_rank,
        residual_sketch_width=args.residual_sketch_width,
        residual_sketch_full_chart=args.residual_sketch_full_chart,
        response_lift_rank=args.response_lift_rank,
        response_lift_global_only=args.response_lift_global_only,
    ).to(device)
    identity_lift = None
    if args.initial_analysis_checkpoint is not None:
        initial = torch.load(
            args.initial_analysis_checkpoint, map_location="cpu", weights_only=False,
        )
        if initial.get("contract") not in (
            CONTRACT, STATE_GATED_CONTRACT, RESIDUAL_FIELD_CONTRACT,
            FULL_CHART_CONTRACT, RESPONSE_LIFT_CONTRACT,
            GLOBAL_RESPONSE_LIFT_CONTRACT,
        ):
            raise ValueError("initial analysis checkpoint has the wrong contract")
        if initial.get("chart_sha256") != _sha(args.chart_checkpoint):
            raise ValueError("initial analysis checkpoint uses a different chart")
        if initial.get("teacher_sha256") != _sha(args.teacher):
            raise ValueError("initial analysis checkpoint uses a different teacher")
        initial_rank = int(initial.get("args", {}).get("state_value_rank", 0))
        initial_sketch = int(
            initial.get("args", {}).get("residual_sketch_width", 0)
        )
        initial_full_chart = bool(
            initial.get("args", {}).get("residual_sketch_full_chart", False)
        )
        initial_response_lift_rank = int(
            initial.get("args", {}).get("response_lift_rank", 0)
        )
        initial_response_lift_global_only = bool(
            initial.get("args", {}).get("response_lift_global_only", False)
        )
        if initial_rank not in (0, args.state_value_rank):
            raise ValueError("initial state-value rank is incompatible")
        if initial_sketch not in (0, args.residual_sketch_width):
            raise ValueError("initial residual-sketch width is incompatible")
        if initial_sketch and initial_full_chart != args.residual_sketch_full_chart:
            raise ValueError("initial residual-sketch response width is incompatible")
        if initial_response_lift_rank not in (0, args.response_lift_rank):
            raise ValueError("initial response-chart lift rank is incompatible")
        if initial_response_lift_rank and (
            initial_response_lift_global_only != args.response_lift_global_only
        ):
            raise ValueError("initial response-chart routing is incompatible")
        initial_scale = torch.as_tensor(initial["drive_scale"]).float()
        if not torch.equal(initial_scale, scale.cpu()):
            raise ValueError("initial analysis checkpoint uses a different drive scale")
        identity_lift = {
            "checkpoint": str(args.initial_analysis_checkpoint.resolve()),
            "sha256": _sha(args.initial_analysis_checkpoint),
            "source_contract": initial["contract"],
            "source_epoch": int(initial["epoch"]),
            "source_state_value_rank": initial_rank,
            "source_residual_sketch_width": initial_sketch,
            "source_residual_sketch_full_chart": initial_full_chart,
            "source_response_lift_rank": initial_response_lift_rank,
            "source_response_lift_global_only": (
                initial_response_lift_global_only
            ),
            **load_analysis_state(
                student, initial["analysis_state"],
                identity_lift=(initial_rank == 0 and args.state_value_rank > 0),
                residual_sketch_lift=(
                    initial_sketch == 0 and args.residual_sketch_width > 0
                ),
            ),
        }
    if args.freeze_base_on_lift:
        for name, parameter in student.named_parameters():
            if args.residual_sketch_width:
                parameter.requires_grad_(
                    name.startswith("residual_sketch.")
                    or name.startswith("response_lift.")
                )
            else:
                parameter.requires_grad_(name.startswith(
                    "residual_map.global_kernel.state_value_"
                ))
    trainable_parameter_count = sum(
        parameter.numel() for parameter in student.trainable_parameters
    )
    if trainable_parameter_count <= 0:
        raise RuntimeError("the selected Airfoil analysis phase has no trainable parameters")
    optimizer = torch.optim.AdamW(
        student.trainable_parameters, lr=args.lr, weight_decay=args.weight_decay,
    )
    train_corpus = PairedTrajectoryCorpus(args.data_root, base["split"]["train"], "all")
    validation_corpus = PairedTrajectoryCorpus(
        args.data_root, base["split"]["validation"], "all",
    )
    history: list[dict[str, Any]] = []
    best = float("inf")
    parent_validation = None
    if identity_lift is not None:
        parent_validation = evaluate(
            student, validation_corpus, statistics, args, device,
        )
        if not parent_validation["gate"]["accepted"]:
            raise ValueError(
                "identity-lifted parent no longer passes its validation gate"
            )
        best = (
            parent_validation["metrics"]["predicted_field"]
            + args.selection_force_weight
            * parent_validation["metrics"]["predicted_force"]
        )
        history.append({
            "epoch": 0,
            "role": "exact identity-lifted parent",
            "selection_score": best,
            "validation": parent_validation,
        })
    generator = random.Random(args.seed + 17_000_003)
    drive_audit = None
    try:
        for epoch in range(1, args.epochs + 1):
            generator.shuffle(records)
            student.train(); losses = []
            for record in records:
                trajectory = int(record["trajectory"]); index = int(record["index"])
                example = residual_port_example(
                    model, train_corpus.paired[trajectory][index],
                    train_corpus.fine[trajectory], "fine", statistics,
                    halo=args.halo, seed=int(record["seed"]), device=device,
                    cover_offset=tuple(record["cover_offset"]),
                )
                local, _, physical, scalar, _, grid_shape, transition, dataset = example
                q24 = record["q24"].to(device=device, dtype=local.dtype)
                qmom6 = record["qmom6"].to(device=device, dtype=local.dtype)
                target_code = record["z24"].to(device=device, dtype=local.dtype)
                residual_field = (
                    pou_field_defect(example, local)
                    if student.residual_sketch is not None else None
                )
                auxiliary_residual = (
                    pou_transport_load_defect(model, example, local)
                    if student.residual_sketch_full_chart else None
                )
                if drive_audit is None:
                    actual_q, actual_m, _, _ = complete_residual_drive(model, example)
                    drive_audit = {
                        "qgamma_max_abs_delta": float((actual_q - q24).abs().max()),
                        "qmom_max_abs_delta": float((actual_m - qmom6).abs().max()),
                    }
                    if max(drive_audit.values()) > 1.0e-5:
                        raise ValueError(
                            f"explicit-cover teacher drive mismatch: {drive_audit}"
                        )
                optimizer.zero_grad(set_to_none=True)
                predicted_code = student.code(
                    local, physical, scalar, grid_shape, q24, qmom6,
                    residual_field=residual_field,
                    auxiliary_residual=auxiliary_residual,
                )
                radius0_code = radius1_code = None
                if args.reception_weight > 0.0 \
                        or args.matched_reception_weight > 0.0:
                    radius0_code = student.code(
                        local, physical, scalar, grid_shape, q24, qmom6,
                        graph_radius=0, residual_field=residual_field,
                        auxiliary_residual=auxiliary_residual,
                    )
                    radius1_code = student.code(
                        local, physical, scalar, grid_shape, q24, qmom6,
                        graph_radius=1, residual_field=residual_field,
                        auxiliary_residual=auxiliary_residual,
                    )
                reception_loss = predicted_code.new_zeros(())
                if args.reception_weight > 0.0:
                    if radius0_code is None or radius1_code is None:
                        raise RuntimeError("code reception controls were not built")
                    reception_loss = reception_identifiability_loss(
                        predicted_code[..., :24], radius0_code[..., :24],
                        radius1_code[..., :24], target_code,
                        margin=args.reception_margin,
                    )
                code_loss = F.smooth_l1_loss(
                    predicted_code[..., :24], target_code,
                )
                predicted_field = student.decode(
                    local, physical, grid_shape, predicted_code,
                )
                valid = physical[:, -1:].to(local)
                predicted_state = _physical_prediction(
                    model, example, predicted_field,
                )
                field_loss = predicted_code.new_zeros(())
                teacher_force_loss = predicted_code.new_zeros(())
                if args.distillation_weight > 0.0:
                    with torch.no_grad():
                        teacher_field = student.decode(
                            local, physical, grid_shape, target_code,
                        )
                    field_loss = (
                        (predicted_field - teacher_field).square() * valid
                    ).sum() / (
                        valid.sum().clamp_min(1.0) * local.shape[1]
                    )
                    with torch.no_grad():
                        teacher_state = _physical_prediction(
                            model, example, teacher_field,
                        )
                    force_arguments = dict(
                        sdf=transition.batch["sdf_np1"].to(predicted_state),
                        motion=transition.batch["motion_np1"].to(predicted_state),
                        dx=dataset.dx, window=dataset.window,
                        penalty_time=dataset.penalty_time,
                        rotation_sign=dataset.rotation_sign,
                    )
                    predicted_force = penalty_force_coefficients(
                        predicted_state, **force_arguments,
                    )
                    with torch.no_grad():
                        teacher_force = penalty_force_coefficients(
                            teacher_state, **force_arguments,
                        )
                    teacher_force_loss = (
                        (predicted_force.drag - teacher_force.drag).square()
                        + (predicted_force.lift - teacher_force.lift).square()
                    ).mean() / .1**2
                distillation_loss = (
                    code_loss + args.field_weight * field_loss
                    + args.teacher_force_weight * teacher_force_loss
                )
                matched_loss = predicted_code.new_zeros(())
                matched_reception_loss = predicted_code.new_zeros(())
                matched_terms: dict[str, torch.Tensor] = {}
                if args.matched_patch_objective_weight > 0.0:
                    matched_loss, matched_terms = matched_patch_response_objective(
                        predicted_field, predicted_state, example,
                    )
                if args.matched_reception_weight > 0.0:
                    if radius0_code is None or radius1_code is None \
                            or "field" not in matched_terms:
                        raise RuntimeError(
                            "matched reception requires the matched field objective"
                        )
                    radius_field_terms = []
                    for radius_code in (radius0_code, radius1_code):
                        radius_field = student.decode(
                            local, physical, grid_shape, radius_code,
                        )
                        radius_state = _physical_prediction(
                            model, example, radius_field,
                        )
                        _, radius_terms = matched_patch_response_objective(
                            radius_field, radius_state, example,
                        )
                        radius_field_terms.append(radius_terms["field"])
                    matched_reception_loss = sum(
                        torch.relu(
                            matched_terms["field"]
                            + args.matched_reception_margin - radius_field
                        )
                        for radius_field in radius_field_terms
                    )
                objective = (
                    args.distillation_weight * distillation_loss
                    + args.matched_patch_objective_weight * matched_loss
                    + args.reception_weight * reception_loss
                    + args.matched_reception_weight * matched_reception_loss
                )
                objective.backward()
                torch.nn.utils.clip_grad_norm_(student.trainable_parameters, 1.0)
                optimizer.step()
                losses.append({
                    "code": float(code_loss.detach()),
                    "teacher_field": float(field_loss.detach()),
                    "teacher_force": float(teacher_force_loss.detach()),
                    "reception": float(reception_loss.detach()),
                    "matched": float(matched_loss.detach()),
                    "matched_reception": float(
                        matched_reception_loss.detach()
                    ),
                    **{
                        f"matched_{name}": float(value.detach())
                        for name, value in matched_terms.items()
                    },
                })
            row: dict[str, Any] = {
                "epoch": epoch,
                "train_code_smooth_l1": float(np.mean([
                    x["code"] for x in losses
                ])),
                "train_decoded_teacher_mse": float(np.mean([
                    x["teacher_field"] for x in losses
                ])),
                "train_teacher_force_square": float(np.mean([
                    x["teacher_force"] for x in losses
                ])),
                "train_reception_identifiability": float(np.mean([
                    x["reception"] for x in losses
                ])),
                "train_matched_reception": float(np.mean([
                    x["matched_reception"] for x in losses
                ])),
                "train_matched_patch_objective": float(np.mean([
                    x["matched"] for x in losses
                ])),
            }
            for name in (
                "field", "local", "consensus", "near_body", "surface",
                "interface", "force",
            ):
                key = f"matched_{name}"
                if key in losses[0]:
                    row[f"train_{key}"] = float(np.mean([
                        x[key] for x in losses
                    ]))
            if epoch % args.validate_every == 0 or epoch == args.epochs:
                validation = evaluate(
                    student, validation_corpus, statistics, args, device,
                )
                row["validation"] = validation
                score = validation["metrics"]["predicted_field"] \
                    + args.selection_force_weight * validation["metrics"]["predicted_force"]
                payload = {
                    "contract": student.contract, "epoch": epoch,
                    "chart_checkpoint": str(args.chart_checkpoint.resolve()),
                    "chart_sha256": _sha(args.chart_checkpoint),
                    "teacher": str(args.teacher.resolve()),
                    "teacher_sha256": _sha(args.teacher),
                    "teacher_contract": teacher["contract"],
                    "teacher_target_used_by_solver": False,
                    "teacher_records_contain_target_tensor": False,
                    "training_target_used": bool(
                        args.matched_patch_objective_weight > 0.0
                    ),
                    "training_objective": {
                        "target_free_distillation_weight": args.distillation_weight,
                        "matched_patch_response_weight": (
                            args.matched_patch_objective_weight
                        ),
                        "matched_field_reception_weight": (
                            args.matched_reception_weight
                        ),
                    },
                    "drive_recomputation_audit": drive_audit,
                    "beta": beta, "drive_scale": scale,
                    "analysis_state": _analysis_state(student),
                    "identity_lift": identity_lift,
                    "identity_lifted_parent_validation": parent_validation,
                    "trainable_parameter_count": trainable_parameter_count,
                    "validation": validation, "args": vars(args),
                    "split": base["split"],
                    "protected_test_trajectories_opened": 0,
                }
                torch.save(payload, args.output_dir / "last_checkpoint.pt")
                if validation["gate"]["accepted"] and score < best:
                    best = score
                    torch.save(payload, args.output_dir / "checkpoint.pt")
            history.append(row); print(json.dumps(row), flush=True)
    finally:
        train_corpus.close(); validation_corpus.close()
    summary = {
        "contract": student.contract, "history": history,
        "teacher_provenance": {
            "split": "train", "target_used_by_solver": False,
            "records_contain_target_tensor": False,
            "strict_confirmation_gate": teacher["strict_confirmation_gate"],
            "drive_recomputation_audit": drive_audit,
        },
        "drive_scale_source": "train-only teacher residual RMS",
        "identity_lift": identity_lift,
        "identity_lifted_parent_validation": parent_validation,
        "state_value_rank": args.state_value_rank,
        "residual_sketch_width": args.residual_sketch_width,
        "residual_sketch_full_chart": args.residual_sketch_full_chart,
        "response_lift_rank": args.response_lift_rank,
        "response_lift_global_only": args.response_lift_global_only,
        "freeze_base_on_lift": args.freeze_base_on_lift,
        "trainable_parameter_count": trainable_parameter_count,
        "training_target_used": bool(
            args.matched_patch_objective_weight > 0.0
        ),
        "training_objective": {
            "target_free_distillation_weight": args.distillation_weight,
            "matched_patch_response_weight": args.matched_patch_objective_weight,
            "matched_field_reception_weight": args.matched_reception_weight,
        },
        "protected_test_trajectories_opened": 0,
    }
    (args.output_dir / "validation_history.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if not (args.output_dir / "checkpoint.pt").exists():
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--validate-every", type=int, default=2)
    parser.add_argument("--validation-samples-per-trajectory", type=int, default=8)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--state-value-rank", type=int, default=0,
        help=(
            "low-rank executable-state gate on residual values; zero keeps "
            "the accepted residual kernel"
        ),
    )
    parser.add_argument(
        "--residual-sketch-width", type=int, default=0,
        help=(
            "width of a centered physical field-defect token; zero keeps the "
            "accepted semantic residual coordinate"
        ),
    )
    parser.add_argument(
        "--residual-sketch-full-chart", action="store_true",
        help=(
            "decode all 32 centered response coordinates and append the "
            "eight target-free transport/load residuals to the field sketch"
        ),
    )
    parser.add_argument(
        "--response-lift-rank", type=int, default=0,
        help=(
            "number of appended centered response coordinates and spatial "
            "modes; requires the complete residual-field sketch"
        ),
    )
    parser.add_argument(
        "--response-lift-global-only", action="store_true",
        help=(
            "make appended response coordinates depend only on the globally "
            "communicated residual value, never the local residual branch"
        ),
    )
    parser.add_argument("--initial-analysis-checkpoint", type=Path)
    parser.add_argument(
        "--freeze-base-on-lift", action="store_true",
        help="optimize only the identity-lifted state-value branch",
    )
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--field-weight", type=float, default=10.0)
    parser.add_argument("--teacher-force-weight", type=float, default=.1)
    parser.add_argument(
        "--distillation-weight", type=float, default=1.0,
        help="weight on the train-only target-free latent-Schur objective",
    )
    parser.add_argument(
        "--matched-patch-objective-weight", type=float, default=0.0,
        help=(
            "weight on the exact supervised field/load response objective "
            "used by the same-cover Patch Transformer control"
        ),
    )
    parser.add_argument("--reception-weight", type=float, default=0.0)
    parser.add_argument("--reception-margin", type=float, default=.001)
    parser.add_argument(
        "--matched-reception-weight", type=float, default=0.0,
        help=(
            "hinge weight requiring full reception to beat radius-zero and "
            "radius-one on the matched supervised field objective"
        ),
    )
    parser.add_argument(
        "--matched-reception-margin", type=float, default=1.0e-3,
    )
    parser.add_argument("--selection-force-weight", type=float, default=.25)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.state_value_rank < 0:
        parser.error("--state-value-rank must be nonnegative")
    if args.residual_sketch_width < 0:
        parser.error("--residual-sketch-width must be nonnegative")
    if args.response_lift_rank < 0:
        parser.error("--response-lift-rank must be nonnegative")
    if args.residual_sketch_full_chart and args.residual_sketch_width == 0:
        parser.error("--residual-sketch-full-chart requires a nonzero sketch width")
    if args.response_lift_rank and not args.residual_sketch_full_chart:
        parser.error("--response-lift-rank requires --residual-sketch-full-chart")
    if args.response_lift_global_only and not args.response_lift_rank:
        parser.error("--response-lift-global-only requires --response-lift-rank")
    if args.distillation_weight < 0.0 \
            or args.matched_patch_objective_weight < 0.0 \
            or args.matched_reception_weight < 0.0:
        parser.error("training objective weights must be nonnegative")
    if args.distillation_weight == 0.0 \
            and args.matched_patch_objective_weight == 0.0:
        parser.error("at least one training objective weight must be positive")
    if args.matched_reception_weight > 0.0 \
            and args.matched_patch_objective_weight == 0.0:
        parser.error(
            "--matched-reception-weight requires a matched response objective"
        )
    if args.matched_reception_margin <= 0.0:
        parser.error("--matched-reception-margin must be positive")
    if args.state_value_rank and args.residual_sketch_width:
        parser.error("test state-value and residual-sketch lifts separately")
    if args.initial_analysis_checkpoint is not None \
            and args.state_value_rank == 0 and args.residual_sketch_width == 0:
        parser.error(
            "--initial-analysis-checkpoint is reserved for an identity lift"
        )
    if args.freeze_base_on_lift and args.initial_analysis_checkpoint is None:
        parser.error("--freeze-base-on-lift requires an initial checkpoint")
    return args


if __name__ == "__main__":
    train(parse_args())
