"""Train and validate the frozen-endpoint Airfoil compatibility composer.

Only the checkpoint's declared training trajectories are opened for fitting and
only its validation trajectories are opened for model selection/evaluation.
This command deliberately has no protected-test option.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .airfoil_compatibility import (
    AirfoilCompatibilityComposer,
    CompatibilityCodedAirfoilOperator,
)
from .cartesian_cover import CartesianCover
from .dataset import FieldStatistics
from .evaluate_feature_schwarz_ablation import evaluation_namespace
from .interface_operator import load_interface_trace_checkpoint
from .multiresolution_inference import predict_multiresolution_step
from .force_diagnostics import force_coefficient_relative_square
from .train_codiffuse_schwarz_rollout import (
    evaluate as evaluate_rollout,
    selection_score as rollout_selection_score,
)
from .train_multiresolution_projection import batched
from .train_multitrajectory_multiresolution import (
    PairedTrajectoryCorpus,
    evaluate as evaluate_one_step,
    prediction_and_losses,
    selection_score as one_step_selection_score,
)
from .train_one_step_multilevel import load_pretrained, random_uniform_transition


Tensor = torch.Tensor
CONTRACT = "airfoil_post_endpoint_compatibility_code_v1"


def checkpoint_digest(state: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        # Hash raw storage bytes so uncommon checkpoint dtypes (notably
        # bfloat16) do not depend on NumPy dtype support.
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_retained(checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    if bool(checkpoint.get("model_spec", {}).get("interface_trace_operator", False)):
        model, _ = load_interface_trace_checkpoint(checkpoint, device)
    else:
        model, _ = load_pretrained(checkpoint, device, history_steps=2)
    return model


def make_composer(
    statistics: FieldStatistics,
    args: argparse.Namespace,
    device: torch.device,
) -> AirfoilCompatibilityComposer:
    return AirfoilCompatibilityComposer(
        width=args.compatibility_width, code_dim=args.code_dim,
        rank=args.decoder_rank, heads=args.heads,
        correction_scale=args.correction_scale,
        max_graph_radius=(None if args.graph_radius < 0 else args.graph_radius),
        port_functional_kind=getattr(
            args, "port_functional_kind", "pressure_traction",
        ),
        state_mean=statistics.state_mean, state_std=statistics.state_std,
        delta_mean=statistics.delta_mean, delta_std=statistics.delta_std,
    ).to(device)


def runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        validation_samples_per_trajectory=args.validation_samples_per_trajectory,
        fine_fraction=args.fine_fraction, halo=args.halo,
        fine_trust=args.fine_trust,
        vorticity_amplitude_fraction=args.vorticity_amplitude_fraction,
        coarse_state_weight=1.0, fine_state_weight=1.0,
        composite_vorticity_weight=args.vorticity_weight,
        near_body_weight=args.near_body_weight,
        far_field_weight=args.far_field_weight,
        domain_boundary_vorticity_weight=0.0,
        force_weight=args.force_weight,
        selection_force_weight=args.selection_force_weight,
        selection_domain_boundary_weight=0.0,
        seed=args.seed,
    )


def loss_value(losses: dict[str, Tensor], args: argparse.Namespace) -> Tensor:
    return (
        losses["coarse_state"] + losses["fine_state"]
        + args.vorticity_weight * losses["composite_vorticity"]
        + args.near_body_weight * losses["near_body_velocity"]
        + args.far_field_weight * losses["far_field_vorticity"]
        + args.force_weight * losses["force"]
    )


def residual_port_example(
    model: CompatibilityCodedAirfoilOperator,
    sample: dict[str, Any],
    dataset,
    level: str,
    statistics: FieldStatistics,
    *,
    halo: int,
    seed: int,
    device: torch.device,
    cover_offset: tuple[int, int] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, tuple[int, int], Any, Any]:
    """Construct one target-labelled port example from a training transition."""
    transition = random_uniform_transition(
        batched(sample, level), statistics, dataset, device, seed, halo, "random",
    )
    if cover_offset is not None:
        offset_y, offset_x = (int(value) for value in cover_offset)
        transition.cover = CartesianCover(
            transition.q_n.shape[-2], transition.q_n.shape[-1],
            dataset.core_size, halo, offset_y, offset_x,
        )
    one = torch.ones(
        transition.batch_size, device=device, dtype=transition.q_n.dtype,
    )
    physical, scalar = transition.conditions(one)
    motion_physical, motion_scalar = transition.endpoint_conditions()
    endpoint = torch.zeros_like(transition.cover.tile(transition.target))
    with torch.no_grad():
        local_endpoint, _ = model._local_operator.correct_endpoint_fields(
            endpoint, physical, scalar, transition.cover.tile_count,
            tile_grid_shape=transition.cover.grid_shape,
            motion_physical=motion_physical, motion_scalar=motion_scalar,
        )
    target_tiles = transition.cover.tile(transition.target)
    raw_code = model.composer.oracle_residual_port_code(
        local_endpoint, target_tiles, physical, standardized=False,
        force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
        lift_sign=dataset.rotation_sign,
    )
    return (
        local_endpoint, target_tiles, physical, scalar, raw_code,
        transition.cover.grid_shape, transition, dataset,
    )


def fit_training_port_statistics(
    model: CompatibilityCodedAirfoilOperator,
    corpus: PairedTrajectoryCorpus,
    statistics: FieldStatistics,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Fit fixed residual-port normalization using training trajectories only."""
    generator = random.Random(args.seed + 71_000_003)
    total = torch.zeros(model.composer.code_dim, dtype=torch.float64, device=device)
    square = torch.zeros_like(total)
    count = 0
    for index in range(args.port_stat_samples):
        _, coarse, fine, sample = corpus.random_sample(generator)
        if index % 2:
            dataset, level = fine, "fine"
        else:
            dataset, level = coarse, "coarse"
        example = residual_port_example(
            model, sample, dataset, level, statistics, halo=args.halo,
            seed=args.seed + 73_000_019 + index, device=device,
        )
        raw = example[4]
        values = raw.to(torch.float64)
        total += values.sum(0); square += values.square().sum(0)
        count += values.shape[0]
    # The shared contract uses an RMS scale without centering so the physical
    # zero residual remains exactly z=0.
    scale = (square / count).sqrt().clamp_min(1.0e-5)
    model.composer.set_port_scale(scale.float())
    return {
        "source_split": "train", "sampled_transitions": args.port_stat_samples,
        "tile_count": count, "zero_preserving": True,
        "minimum_scale": float(scale.min()),
        "maximum_scale": float(scale.max()),
    }


def port_losses(
    model: CompatibilityCodedAirfoilOperator,
    example: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, tuple[int, int], Any, Any],
    *,
    oracle_decoder: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    (
        local_endpoint, target_tiles, physical, scalar, raw, grid_shape,
        transition, dataset,
    ) = example
    batch = local_endpoint.shape[0] // int(np.prod(grid_shape))
    oracle = (raw / model.composer.port_std).reshape(
        batch, int(np.prod(grid_shape)), -1,
    )
    valid = physical[:, -1:].to(local_endpoint)
    if oracle_decoder:
        corrected = model.composer(
            local_endpoint, physical, scalar, grid_shape,
            code_mode="oracle", oracle_code=oracle,
        ).corrected
        code_loss = local_endpoint.new_zeros(())
    else:
        result = model.composer(
            local_endpoint, physical, scalar, grid_shape, code_mode="predicted",
        )
        corrected = result.corrected
        code_loss = (result.predicted_code - oracle).square().mean()
    field_loss = (
        (corrected - target_tiles).square() * valid
    ).sum() / (valid.sum().clamp_min(1.0) * local_endpoint.shape[1])
    normalized_delta = transition.cover.assemble_weighted(
        corrected, transition.batch_size,
    )
    delta_mean = local_endpoint.new_tensor(
        transition.statistics.delta_mean,
    )[None, :, None, None]
    delta_std = local_endpoint.new_tensor(
        transition.statistics.delta_std,
    )[None, :, None, None]
    prediction = transition.q_n + normalized_delta * delta_std + delta_mean
    force_loss = force_coefficient_relative_square(
        prediction, transition.q_np1,
        transition.batch["sdf_np1"].to(local_endpoint.device),
        transition.batch["motion_np1"].to(local_endpoint.device),
        dx=dataset.dx, window=dataset.window,
        penalty_time=dataset.penalty_time, rotation_sign=dataset.rotation_sign,
    )
    return field_loss, code_loss, force_loss


def save_candidate(
    path: Path,
    composer: AirfoilCompatibilityComposer,
    base_path: Path,
    base: dict[str, Any],
    args: argparse.Namespace,
    epoch: int,
    validation: dict[str, Any],
) -> None:
    torch.save({
        "contract": CONTRACT,
        "composer": composer.state_dict(),
        "composer_spec": {
            "width": composer.width, "code_dim": composer.code_dim,
            "rank": composer.rank, "heads": composer.global_message.heads,
            "correction_scale": composer.correction_scale,
            "max_graph_radius": composer.max_graph_radius,
            "semantic_channels": composer.semantic_channels,
            "case_scalar_dim": composer.scalar_dim,
            "global_messages_per_complete_cover": 1,
            "absolute_tile_coordinates_consumed": False,
            "residual_port_spec": (
                composer.residual_port_spec.to_json()
                if hasattr(composer, "residual_port_spec") else None
            ),
            "port_functional_kind": composer.port_functional_kind,
            "port_scale_fit_split": "train" if args.residual_port else None,
            "training_factorization": (
                "oracle_chart_select_freeze_then_analyzer_only"
                if args.residual_port else "legacy"
            ),
            "response_chart_frozen_during_predictor_training": bool(
                args.residual_port
            ),
        },
        "base_checkpoint": str(base_path.resolve()),
        "base_state_sha256": checkpoint_digest(base["model"]),
        "base_tensor_count": len(base["model"]),
        "statistics": base["statistics"], "split": base["split"],
        "epoch": epoch, "validation": validation, "args": vars(args),
    }, path)


def train(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    base = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    if not {"train", "validation", "test"}.issubset(base.get("split", {})):
        raise ValueError("retained checkpoint does not carry the protected split")
    statistics = FieldStatistics.from_json(base["statistics"])
    retained = load_retained(base, device)
    retained.set_endpoint_statistics(
        statistics.state_mean, statistics.state_std, statistics.delta_std,
        statistics.delta_mean,
    )
    composer = make_composer(statistics, args, device)
    if args.residual_port and composer.code_dim != 32:
        raise ValueError("the shared Airfoil residual-port contract requires code_dim=32")
    model = CompatibilityCodedAirfoilOperator(retained, composer).to(device)
    train_corpus = PairedTrajectoryCorpus(
        args.data_root, base["split"]["train"], "all",
    )
    validation_corpus = PairedTrajectoryCorpus(
        args.data_root, base["split"]["validation"], "all",
    )
    run_args = runtime_args(args)
    history: list[dict[str, Any]] = []
    try:
        port_statistics = None
        chart_gate = True
        if args.residual_port:
            port_statistics = fit_training_port_statistics(
                model, train_corpus, statistics, args, device,
            )
            # Phase 1: learn and validation-select only the response chart from
            # fixed, target-defined oracle ports.  The analyzer is untouched.
            chart_parameters = list(composer.decoder.parameters()) + list(
                composer.code_coefficients.parameters()
            )
            chart_optimizer = torch.optim.AdamW(
                chart_parameters, lr=args.lr, weight_decay=args.weight_decay,
            )
            warmup_generator = random.Random(args.seed + 79_000_003)
            best_chart = copy.deepcopy({
                "decoder": composer.decoder.state_dict(),
                "coefficients": composer.code_coefficients.state_dict(),
            })
            best_oracle = float("inf")
            for warmup in range(1, args.oracle_warmup_epochs + 1):
                model.train()
                warmup_losses = []
                for local in range(args.samples_per_epoch):
                    _, coarse, fine, sample = train_corpus.random_sample(
                        warmup_generator,
                    )
                    dataset, level = (
                        (fine, "fine") if local % 2 else (coarse, "coarse")
                    )
                    example = residual_port_example(
                        model, sample, dataset, level, statistics,
                        halo=args.halo,
                        seed=args.seed + warmup * 11_000_003 + local,
                        device=device,
                    )
                    chart_optimizer.zero_grad(set_to_none=True)
                    field_loss, _, chart_force = port_losses(
                        model, example, oracle_decoder=True,
                    )
                    chart_objective = (
                        field_loss + args.chart_force_weight * chart_force
                    )
                    chart_objective.backward()
                    torch.nn.utils.clip_grad_norm_(chart_parameters, 1.0)
                    chart_optimizer.step()
                    warmup_losses.append((
                        float(field_loss.detach()), float(chart_force.detach()),
                    ))
                model.eval()
                oracle_validation = oracle_capacity(
                    model, validation_corpus, statistics, args, device,
                )
                oracle_score = oracle_validation["oracle_relative_l2"]
                if oracle_score < best_oracle:
                    best_oracle = oracle_score
                    best_chart = copy.deepcopy({
                        "decoder": composer.decoder.state_dict(),
                        "coefficients": composer.code_coefficients.state_dict(),
                    })
                warmup_record = {
                    "stage": "oracle_decoder_warmup", "warmup_epoch": warmup,
                    "oracle_port_field_loss": float(np.mean(
                        [row[0] for row in warmup_losses]
                    )),
                    "oracle_port_force_loss": float(np.mean(
                        [row[1] for row in warmup_losses]
                    )),
                    "validation_fixed_oracle_relative_l2": oracle_score,
                }
                history.append(warmup_record)
                print(json.dumps(warmup_record), flush=True)
            composer.decoder.load_state_dict(best_chart["decoder"])
            composer.code_coefficients.load_state_dict(best_chart["coefficients"])
            chart_validation = theorem_single_cover_validation(
                model, validation_corpus, statistics, args, device,
            )
            chart_field_gate = (
                chart_validation["oracle_relative_l2"]
                < chart_validation["zero_relative_l2"]
            )
            chart_force_gate = (
                chart_validation["oracle_force"]
                <= chart_validation["zero_force"]
            )
            chart_gate = chart_field_gate and chart_force_gate
            for parameter in chart_parameters:
                parameter.requires_grad_(False)
            chart_record = {
                "stage": "response_chart_frozen",
                "selected_fixed_oracle_relative_l2": best_oracle,
                "zero_code_relative_l2": oracle_validation["zero_relative_l2"],
                "single_cover_validation": chart_validation,
                "chart_field_gate": chart_field_gate,
                "chart_force_gate": chart_force_gate,
                "chart_gate": chart_gate,
            }
            history.append(chart_record)
            save_candidate(
                args.output_dir / "last_checkpoint.pt", composer,
                args.pretrained, base, args, 0, chart_record,
            )
            if args.chart_only:
                if chart_gate:
                    save_candidate(
                        args.output_dir / "chart_checkpoint.pt", composer,
                        args.pretrained, base, args, 0, chart_record,
                    )
                (args.output_dir / "training_summary.json").write_text(json.dumps({
                    "contract": CONTRACT,
                    "stage": "chart_only",
                    "protected_test_trajectories_opened": 0,
                    "residual_port_statistics": port_statistics,
                    "history": history,
                }, indent=2) + "\n")
                print(json.dumps(chart_record), flush=True)
                return
            if not chart_gate:
                (args.output_dir / "training_summary.json").write_text(json.dumps({
                    "contract": CONTRACT,
                    "stage": "chart_rejected_before_analyzer",
                    "protected_test_trajectories_opened": 0,
                    "residual_port_statistics": port_statistics,
                    "history": history,
                }, indent=2) + "\n")
                print(json.dumps(chart_record), flush=True)
                return
        # Phase 2: only analysis/token communication/code inference is trained.
        analysis_modules = (
            composer.encoder, composer.pool, composer.scalar_encoder,
            composer.global_message, composer.code_head,
        )
        analysis_parameters = [
            parameter for module in analysis_modules
            for parameter in module.parameters()
        ]
        optimizer = torch.optim.AdamW(
            analysis_parameters, lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, max(args.epochs, 1), eta_min=args.min_lr,
        )
        model.set_code_override("zero")
        zero = evaluate_one_step(
            model, validation_corpus, statistics, run_args, device,
        )
        model.set_code_override("predicted")
        predicted = evaluate_one_step(
            model, validation_corpus, statistics, run_args, device,
        )
        baseline_score = one_step_selection_score(
            zero, args.selection_force_weight,
        )
        record = {
            "epoch": 0, "zero": zero, "predicted": predicted,
            "selection_score": one_step_selection_score(
                predicted, args.selection_force_weight,
            ),
        }
        history.append(record)
        save_candidate(
            args.output_dir / "last_checkpoint.pt", composer, args.pretrained,
            base, args, 0, record,
        )
        print(json.dumps(record), flush=True)
        best = float("inf")
        for epoch in range(1, args.epochs + 1):
            model.train(); model.set_code_override("predicted")
            generator = random.Random(args.seed + epoch * 1_000_003)
            train_losses = []
            for local in range(args.samples_per_epoch):
                _, coarse, fine, sample = train_corpus.random_sample(generator)
                optimizer.zero_grad(set_to_none=True)
                _, losses, _ = prediction_and_losses(
                    model, sample, coarse, fine, statistics, run_args,
                    args.seed + epoch * 10_000_019 + local, device,
                )
                objective = loss_value(losses, args)
                port_field = objective.new_zeros(())
                code_loss = objective.new_zeros(())
                port_force = objective.new_zeros(())
                if args.residual_port:
                    dataset, level = (
                        (fine, "fine") if local % 2 else (coarse, "coarse")
                    )
                    example = residual_port_example(
                        model, sample, dataset, level, statistics,
                        halo=args.halo,
                        seed=args.seed + epoch * 13_000_019 + local,
                        device=device,
                    )
                    port_field, code_loss, port_force = port_losses(
                        model, example, oracle_decoder=False,
                    )
                    objective = (
                        objective + args.port_field_weight * port_field
                        + args.code_supervision_weight * code_loss
                        + args.port_force_weight * port_force
                    )
                objective.backward()
                torch.nn.utils.clip_grad_norm_(analysis_parameters, 1.0)
                optimizer.step()
                train_losses.append((
                    float(objective.detach()), float(port_field.detach()),
                    float(code_loss.detach()), float(port_force.detach()),
                ))
            scheduler.step(); model.eval()
            model.set_code_override("predicted")
            predicted = evaluate_one_step(
                model, validation_corpus, statistics, run_args, device,
            )
            model.set_code_override("shuffled")
            shuffled = evaluate_one_step(
                model, validation_corpus, statistics, run_args, device,
            )
            score = one_step_selection_score(
                predicted, args.selection_force_weight,
            )
            force_gate = predicted["force_mean"] <= zero["force_mean"]
            accepted = score <= baseline_score and force_gate and chart_gate
            record = {
                "epoch": epoch,
                "train_loss": float(np.mean([row[0] for row in train_losses])),
                "train_port_field_loss": float(np.mean([row[1] for row in train_losses])),
                "train_code_supervision_loss": float(np.mean([row[2] for row in train_losses])),
                "train_port_force_loss": float(np.mean([row[3] for row in train_losses])),
                "zero": zero, "predicted": predicted, "shuffled": shuffled,
                "selection_score": score, "force_gate": force_gate,
                "fixed_oracle_chart_gate": chart_gate,
                "accepted_one_step": accepted,
            }
            history.append(record)
            save_candidate(
                args.output_dir / "last_checkpoint.pt", composer, args.pretrained,
                base, args, epoch, record,
            )
            if accepted and score < best:
                best = score
                save_candidate(
                    args.output_dir / "checkpoint.pt", composer, args.pretrained,
                    base, args, epoch, record,
                )
            print(json.dumps(record), flush=True)
    finally:
        train_corpus.close(); validation_corpus.close()
    (args.output_dir / "training_summary.json").write_text(json.dumps({
        "contract": CONTRACT,
        "data_access": {
            "training_trajectories": len(base["split"]["train"]),
            "validation_trajectories": len(base["split"]["validation"]),
            "protected_test_trajectories_opened": 0,
        },
        "history": history,
        "residual_port_statistics": port_statistics,
    }, indent=2) + "\n")


def load_candidate(
    args: argparse.Namespace, device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], CompatibilityCodedAirfoilOperator, FieldStatistics]:
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    if candidate.get("contract") != CONTRACT:
        raise ValueError("candidate is not an Airfoil compatibility checkpoint")
    base_path = Path(candidate["base_checkpoint"])
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    if checkpoint_digest(base["model"]) != candidate["base_state_sha256"]:
        raise ValueError("retained endpoint differs from the candidate's frozen base")
    if base["split"] != candidate["split"] or base["statistics"] != candidate["statistics"]:
        raise ValueError("candidate changed the protected split or normalization")
    statistics = FieldStatistics.from_json(base["statistics"])
    retained = load_retained(base, device)
    spec = candidate["composer_spec"]
    composer_args = argparse.Namespace(
        compatibility_width=int(spec["width"]), code_dim=int(spec["code_dim"]),
        decoder_rank=int(spec["rank"]), heads=int(spec["heads"]),
        correction_scale=float(spec["correction_scale"]),
        graph_radius=(
            -1 if spec.get("max_graph_radius") is None
            else int(spec["max_graph_radius"])
        ),
        port_functional_kind=spec.get(
            "port_functional_kind", "pressure_traction",
        ),
    )
    composer = make_composer(statistics, composer_args, device)
    missing, unexpected = composer.load_state_dict(candidate["composer"], strict=False)
    allowed_missing = {"port_mean", "port_std"}
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(
            f"candidate composer mismatch: missing={missing}, unexpected={unexpected}"
        )
    model = CompatibilityCodedAirfoilOperator(retained, composer).to(device)
    if args.graph_radius is not None:
        model.set_graph_radius(None if args.graph_radius < 0 else args.graph_radius)
    model.set_endpoint_statistics(
        statistics.state_mean, statistics.state_std, statistics.delta_std,
        statistics.delta_mean,
    )
    return candidate, base, model, statistics


@torch.no_grad()
def _relative_l2(prediction: Tensor, target: Tensor) -> float:
    return float(torch.linalg.vector_norm(prediction - target) /
                 torch.linalg.vector_norm(target).clamp_min(1.0e-20))


def oracle_capacity(
    model: CompatibilityCodedAirfoilOperator,
    corpus: PairedTrajectoryCorpus,
    statistics: FieldStatistics,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    records = []
    composer = model.composer
    for trajectory, paired in enumerate(corpus.paired):
        count = min(args.oracle_samples_per_trajectory, len(paired))
        for local, item in enumerate(np.linspace(0, len(paired) - 1, count, dtype=int)):
            batch = batched(paired[int(item)], "coarse")
            transition = random_uniform_transition(
                batch, statistics, corpus.coarse[trajectory], device,
                args.seed + trajectory * 1_000_003 + local, args.halo, "fixed",
            )
            one = torch.ones(transition.batch_size, device=device, dtype=transition.q_n.dtype)
            physical, scalar = transition.conditions(one)
            motion_physical, motion_scalar = transition.endpoint_conditions()
            endpoint = torch.zeros_like(transition.cover.tile(transition.target))
            with torch.no_grad():
                local_endpoint, _ = model._local_operator.correct_endpoint_fields(
                    endpoint, physical, scalar, transition.cover.tile_count,
                    tile_grid_shape=transition.cover.grid_shape,
                    motion_physical=motion_physical, motion_scalar=motion_scalar,
                )
            target_tiles = transition.cover.tile(transition.target)
            oracle = composer.oracle_residual_port_code(
                local_endpoint, target_tiles, physical,
                force_area_scale=(
                    2.0 * corpus.coarse[trajectory].dx**2
                    / corpus.coarse[trajectory].penalty_time
                ),
                lift_sign=corpus.coarse[trajectory].rotation_sign,
            ).reshape(
                transition.batch_size, transition.cover.tile_count, -1,
            )
            fields = {}
            calls = {}
            for mode in ("zero", "predicted", "shuffled"):
                composer.reset_attention_calls()
                fields[mode] = composer(
                    local_endpoint, physical, scalar, transition.cover.grid_shape,
                    code_mode=mode,
                ).corrected
                calls[mode] = composer.attention_calls
            composer.reset_attention_calls()
            fields["oracle"] = composer(
                local_endpoint, physical, scalar, transition.cover.grid_shape,
                code_mode="oracle", oracle_code=oracle,
            ).corrected
            calls["oracle"] = composer.attention_calls
            original_radius = composer.max_graph_radius
            radius_fields = {}
            for label, radius in (("self", 0), ("neighbor", 1), ("global", None)):
                composer.max_graph_radius = radius
                radius_fields[label] = composer(
                    local_endpoint, physical, scalar, transition.cover.grid_shape,
                    code_mode="predicted",
                ).corrected
            composer.max_graph_radius = original_radius
            target = transition.target
            records.append({
                mode: _relative_l2(
                    transition.cover.assemble_weighted(field, transition.batch_size),
                    target,
                )
                for mode, field in fields.items()
            } | {
                f"radius_{label}": _relative_l2(
                    transition.cover.assemble_weighted(field, transition.batch_size),
                    target,
                )
                for label, field in radius_fields.items()
            } | {f"{mode}_calls": value for mode, value in calls.items()})
    return {
        f"{mode}_relative_l2": float(np.mean([row[mode] for row in records]))
        for mode in ("zero", "predicted", "shuffled", "oracle")
    } | {
        "samples": len(records),
        "attention_calls_per_cover": {
            mode: float(np.mean([row[f"{mode}_calls"] for row in records]))
            for mode in ("zero", "predicted", "shuffled", "oracle")
        },
        "oracle_definition": "fixed_identifiable_residual_port",
        "radius_ablation_relative_l2": {
            label: float(np.mean([row[f"radius_{label}"] for row in records]))
            for label in ("self", "neighbor", "global")
        },
    }


def theorem_single_cover_validation(
    model: CompatibilityCodedAirfoilOperator,
    corpus: PairedTrajectoryCorpus,
    statistics: FieldStatistics,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a theorem-facing full-fine step with exactly one cover/message."""
    records = []
    delta_mean = torch.tensor(
        statistics.delta_mean, device=device,
    )[None, :, None, None]
    delta_std = torch.tensor(
        statistics.delta_std, device=device,
    )[None, :, None, None]
    for trajectory, paired in enumerate(corpus.paired):
        count = min(args.oracle_samples_per_trajectory, len(paired))
        for local, item in enumerate(np.linspace(0, len(paired) - 1, count, dtype=int)):
            batch = batched(paired[int(item)], "fine")
            transition = random_uniform_transition(
                batch, statistics, corpus.fine[trajectory], device,
                args.seed + 101_000_003 + trajectory * 1_000_003 + local,
                args.halo, "fixed",
            )
            one = torch.ones(
                transition.batch_size, device=device, dtype=transition.q_n.dtype,
            )
            physical, scalar = transition.conditions(one)
            motion_physical, motion_scalar = transition.endpoint_conditions()
            target_tiles = transition.cover.tile(transition.target)
            endpoint = torch.zeros_like(target_tiles)
            with torch.no_grad():
                local_endpoint, _ = model._local_operator.correct_endpoint_fields(
                    endpoint, physical, scalar, transition.cover.tile_count,
                    tile_grid_shape=transition.cover.grid_shape,
                    motion_physical=motion_physical,
                    motion_scalar=motion_scalar,
                )
            oracle = model.composer.oracle_residual_port_code(
                local_endpoint, target_tiles, physical,
                force_area_scale=(
                    2.0 * corpus.fine[trajectory].dx**2
                    / corpus.fine[trajectory].penalty_time
                ),
                lift_sign=corpus.fine[trajectory].rotation_sign,
            ).reshape(transition.batch_size, transition.cover.tile_count, -1)
            row = {}
            for mode in ("zero", "predicted", "shuffled", "oracle"):
                model.composer.reset_attention_calls()
                output = model.composer(
                    local_endpoint, physical, scalar, transition.cover.grid_shape,
                    code_mode=mode,
                    oracle_code=oracle if mode == "oracle" else None,
                ).corrected
                normalized_delta = transition.cover.assemble_weighted(
                    output, transition.batch_size,
                )
                prediction = transition.q_n + normalized_delta * delta_std + delta_mean
                prediction = model.correct_surface_level(
                    "fine", prediction, batch, corpus.fine[trajectory],
                )
                target = transition.q_np1
                row[f"{mode}_relative_l2"] = _relative_l2(prediction, target)
                row[f"{mode}_force"] = float(force_coefficient_relative_square(
                    prediction, target, batch["sdf_np1"].to(device),
                    batch["motion_np1"].to(device), dx=corpus.fine[trajectory].dx,
                    window=corpus.fine[trajectory].window,
                    penalty_time=corpus.fine[trajectory].penalty_time,
                    rotation_sign=corpus.fine[trajectory].rotation_sign,
                ))
                row[f"{mode}_attention_calls"] = model.composer.attention_calls
            records.append(row)
    return {
        "executor": "full_fine_single_cover",
        "cover_messages_per_physical_step": 1,
        "samples": len(records),
        **{
            key: float(np.mean([record[key] for record in records]))
            for key in records[0]
        },
    }


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    candidate, base, model, statistics = load_candidate(args, device)
    # Only the validation names are opened. There is intentionally no --split.
    corpus = PairedTrajectoryCorpus(
        args.data_root, base["split"]["validation"], "all",
    )
    rollout_args = evaluation_namespace(args)
    try:
        # Count calls on an actual adaptive physical step.  The current
        # executor has distinct coarse and fine covers, hence two messages.
        audit_sample = corpus.paired[0][0]
        model.set_code_override("predicted")
        model.reset_attention_calls()
        predict_multiresolution_step(
            model, batched(audit_sample, "coarse"), batched(audit_sample, "fine"),
            corpus.coarse[0], corpus.fine[0], statistics,
            fine_fraction=args.fine_fraction, halo=args.halo, seed=args.seed,
            device=device, fine_trust=args.fine_trust,
            vorticity_amplitude_fraction=args.vorticity_amplitude_fraction,
            placement_mode="fixed",
        )
        multiresolution_attention_calls = model.attention_calls
        metrics = {}
        for mode in ("zero", "predicted", "shuffled"):
            model.set_code_override(mode)  # type: ignore[arg-type]
            metrics[mode] = evaluate_rollout(
                model, corpus, statistics, rollout_args, device,
            )
        capacity = oracle_capacity(model, corpus, statistics, args, device)
        theorem_single_cover = theorem_single_cover_validation(
            model, corpus, statistics, args, device,
        )
    finally:
        corpus.close()
    scores = {
        mode: rollout_selection_score(metrics[mode], args.validation_horizon)
        for mode in metrics
    }
    zero, predicted, shuffled = scores["zero"], scores["predicted"], scores["shuffled"]
    improvement = zero - predicted
    gate = {
        "rollout_selection_nonregression": predicted <= zero,
        "drag_nonregression": (
            metrics["predicted"]["c_drag_relative_l2"]
            <= metrics["zero"]["c_drag_relative_l2"]
        ),
        "lift_nonregression": (
            metrics["predicted"]["c_lift_relative_l2"]
            <= metrics["zero"]["c_lift_relative_l2"]
        ),
        "shuffle_loses_half_gain": (
            improvement > 0.0 and shuffled - predicted >= 0.5 * improvement
        ),
        "oracle_improves_local_chart": (
            capacity["oracle_relative_l2"] < capacity["zero_relative_l2"]
        ),
        "predicted_code_improves_local_chart": (
            capacity["predicted_relative_l2"] < capacity["zero_relative_l2"]
        ),
    }
    gate["accepted"] = all(gate.values())
    result = {
        "contract": "airfoil_compatibility_validation_gate_v1",
        "candidate": str(args.candidate.resolve()),
        "retained_endpoint": candidate["base_checkpoint"],
        "split": "validation",
        "protected_test_trajectories_opened": 0,
        "rollout": metrics, "selection_scores": scores,
        "oracle_chart_capacity": capacity, "acceptance_gate": gate,
        "attention_call_audit": {
            "adaptive_multiresolution_calls_per_physical_step":
                multiresolution_attention_calls,
            "reason": "one complete coarse cover plus one complete fine cover",
        },
        "theorem_single_cover_validation": theorem_single_cover,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--fine-fraction", type=float, default=.35)
    parser.add_argument("--fine-trust", type=float, default=.5)
    parser.add_argument("--vorticity-amplitude-fraction", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cpu", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train")
    common(training)
    training.add_argument("--pretrained", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--epochs", type=int, default=1)
    training.add_argument("--samples-per-epoch", type=int, default=64)
    training.add_argument("--validation-samples-per-trajectory", type=int, default=4)
    training.add_argument("--lr", type=float, default=3.0e-4)
    training.add_argument("--min-lr", type=float, default=3.0e-5)
    training.add_argument("--weight-decay", type=float, default=1.0e-4)
    training.add_argument("--compatibility-width", type=int, default=64)
    training.add_argument("--code-dim", type=int, default=32)
    training.add_argument("--decoder-rank", type=int, default=8)
    training.add_argument("--heads", type=int, default=4)
    training.add_argument("--correction-scale", type=float, default=.1)
    training.add_argument("--residual-port", action="store_true")
    training.add_argument(
        "--port-functional-kind",
        choices=("pressure_traction", "brinkman_force"),
        default="pressure_traction",
    )
    training.add_argument("--chart-only", action="store_true")
    training.add_argument("--port-stat-samples", type=int, default=64)
    training.add_argument("--oracle-warmup-epochs", type=int, default=1)
    training.add_argument("--oracle-samples-per-trajectory", type=int, default=2)
    training.add_argument("--port-field-weight", type=float, default=1.0)
    training.add_argument("--code-supervision-weight", type=float, default=.2)
    training.add_argument("--chart-force-weight", type=float, default=.2)
    training.add_argument("--port-force-weight", type=float, default=.2)
    training.add_argument(
        "--graph-radius", type=int, default=-1,
        help="Chebyshev tile radius; -1 is one globally receptive message",
    )
    training.add_argument("--vorticity-weight", type=float, default=6.0)
    training.add_argument("--near-body-weight", type=float, default=4.0)
    training.add_argument("--far-field-weight", type=float, default=4.0)
    training.add_argument("--force-weight", type=float, default=.2)
    training.add_argument("--selection-force-weight", type=float, default=.5)

    validation = subparsers.add_parser("evaluate")
    common(validation)
    validation.add_argument("--candidate", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True)
    validation.add_argument("--validation-horizon", type=int, default=20)
    validation.add_argument("--validation-starts-per-trajectory", type=int, default=2)
    validation.add_argument("--validation-samples-per-trajectory", type=int, default=4)
    validation.add_argument("--endpoint-relaxation", type=float, default=1.0)
    validation.add_argument("--history-momentum", type=float, default=0.0)
    validation.add_argument("--oracle-samples-per-trajectory", type=int, default=2)
    validation.add_argument("--oracle-steps", type=int, default=64)
    validation.add_argument("--oracle-lr", type=float, default=.1)
    validation.add_argument(
        "--graph-radius", type=int, default=None,
        help="optional candidate override: 0 self-only, 1 neighbors, -1 global",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
