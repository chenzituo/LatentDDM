"""Matched full-fine Airfoil analyzer ablation on a frozen response chart.

The generic and defect-aware tasks have identical modules and optimization.
The sole switch is whether the analyzer receives the label-free provisional
POU consensus defect.  Validation indices and cover offsets are deterministic,
and no protected-test option exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from codiffuse_Schwarz.compatibility_defect import residual_minimizing_damping
from .airfoil_compatibility import CompatibilityCodedAirfoilOperator
from .airfoil_compatibility_experiment import (
    load_candidate, residual_port_example,
)
from .airfoil_defect_ports import provisional_defect_port_code
from .force_diagnostics import (
    balanced_force_coefficient_relative_square,
    force_coefficient_relative_square,
    penalty_force_coefficients,
)
from .train_multiresolution_projection import batched
from .train_multitrajectory_multiresolution import PairedTrajectoryCorpus


Tensor = torch.Tensor
Mode = Literal["zero", "predicted", "shuffled", "oracle"]
CONTRACT = "airfoil_frozen_chart_single_cover_analyzer_v1"


def _paired_bootstrap_interval(
    differences: np.ndarray, *, seed: int, draws: int = 10_000,
) -> dict[str, float]:
    """Paired sample bootstrap for a mean method-minus-reference difference."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or draws <= 0:
        raise ValueError("paired bootstrap requires a nonempty vector and positive draws")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(draws, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, .025)),
        "ci95_high": float(np.quantile(means, .975)),
        "probability_lower": float(np.mean(means < 0.0)),
    }


class FrozenChartAnalyzer(nn.Module):
    """One-message code analyzer with an optional provisional-defect input."""

    def __init__(
        self, model: CompatibilityCodedAirfoilOperator, *, use_defect: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.composer = model.composer
        self.use_defect = use_defect
        self.defect_encoder = nn.Sequential(
            nn.LayerNorm(self.composer.code_dim),
            nn.Linear(self.composer.code_dim, self.composer.width, bias=False),
            nn.SiLU(),
            nn.Linear(self.composer.width, self.composer.width, bias=False),
        ).to(self.composer.port_scale if hasattr(self.composer, "port_scale") else self.composer.port_std)
        for parameter in self.composer.parameters():
            parameter.requires_grad_(False)
        for module in (
            self.composer.encoder, self.composer.pool,
            self.composer.scalar_encoder, self.composer.global_message,
            self.composer.code_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        # The module exists in both arms for exact parameter-count matching.
        for parameter in self.defect_encoder.parameters():
            parameter.requires_grad_(True)

    @property
    def analysis_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def forward(
        self,
        local_endpoint: Tensor,
        consensus_endpoint: Tensor,
        physical: Tensor,
        scalar: Tensor,
        grid_shape: tuple[int, int],
        *,
        mode: Mode,
        oracle_code: Tensor,
        graph_radius: int | None = None,
        force_area_scale: float = 1.0,
        lift_sign: float = 1.0,
    ) -> tuple[Tensor, Tensor, int]:
        tile_count = int(np.prod(grid_shape))
        batch = local_endpoint.shape[0] // tile_count
        valid = physical[:, -1:].to(local_endpoint)
        semantics = self.composer.semantic_features(local_endpoint, physical)
        shape = (batch, tile_count, self.composer.code_dim)
        calls = 0
        if mode in ("predicted", "shuffled"):
            tokens = self.composer._tokens(
                local_endpoint, semantics, scalar, valid, grid_shape,
            )
            if self.use_defect:
                defect = provisional_defect_port_code(
                    local_endpoint, consensus_endpoint, physical,
                    state_mean=tuple(float(x) for x in self.composer.state_mean),
                    state_std=tuple(float(x) for x in self.composer.state_std),
                    delta_mean=tuple(float(x) for x in self.composer.delta_mean),
                    delta_std=tuple(float(x) for x in self.composer.delta_std),
                    port_scale=self.composer.port_std,
                    functional_kind=self.composer.port_functional_kind,
                    force_area_scale=force_area_scale,
                    lift_sign=lift_sign,
                ).reshape(batch, tile_count, -1)
                tokens = tokens + self.defect_encoder(defect)
            message = self.composer.global_message(
                tokens, grid_shape, max_graph_radius=graph_radius,
            )
            calls = 1
            code = self.composer.code_head(torch.cat((tokens, message), dim=-1))
            if mode == "shuffled":
                code = self.composer._shuffle_code(code, None)
        elif mode == "zero":
            code = local_endpoint.new_zeros(shape)
        elif mode == "oracle":
            if oracle_code.shape != shape:
                raise ValueError("oracle code has incompatible shape")
            code = oracle_code
        else:
            raise ValueError(mode)
        corrected = self.composer.decode(
            local_endpoint, semantics, code, valid, grid_shape,
        )
        return corrected, code, calls


def _physical_prediction(
    analyzer, example, corrected: Tensor, statistics, *, apply_surface: bool = True,
) -> Tensor:
    *_, transition, dataset = example
    normalized = transition.cover.assemble_weighted(
        corrected, transition.batch_size,
    )
    mean = corrected.new_tensor(statistics.delta_mean)[None, :, None, None]
    std = corrected.new_tensor(statistics.delta_std)[None, :, None, None]
    prediction = transition.q_n + normalized * std + mean
    return (
        analyzer.model.correct_surface_level(
            "fine", prediction, transition.batch, dataset,
        )
        if apply_surface else prediction
    )


def _force(prediction: Tensor, example) -> Tensor:
    *_, transition, dataset = example
    return force_coefficient_relative_square(
        prediction, transition.q_np1,
        transition.batch["sdf_np1"].to(prediction.device),
        transition.batch["motion_np1"].to(prediction.device),
        dx=dataset.dx, window=dataset.window,
        penalty_time=dataset.penalty_time, rotation_sign=dataset.rotation_sign,
    )


def _force_components(prediction: Tensor, example) -> tuple[Tensor, Tensor]:
    *_, transition, dataset = example
    arguments = dict(
        sdf=transition.batch["sdf_np1"].to(prediction.device),
        motion=transition.batch["motion_np1"].to(prediction.device),
        dx=dataset.dx, window=dataset.window,
        penalty_time=dataset.penalty_time, rotation_sign=dataset.rotation_sign,
    )
    actual = penalty_force_coefficients(prediction, **arguments)
    expected = penalty_force_coefficients(transition.q_np1, **arguments)
    return (
        ((actual.drag - expected.drag).square() / (expected.drag.square() + .1**2)).mean(),
        ((actual.lift - expected.lift).square() / (expected.lift.square() + .1**2)).mean(),
    )


def _balanced_force(prediction: Tensor, example) -> Tensor:
    *_, transition, dataset = example
    return balanced_force_coefficient_relative_square(
        prediction, transition.q_np1,
        transition.batch["sdf_np1"].to(prediction.device),
        transition.batch["motion_np1"].to(prediction.device),
        dx=dataset.dx, window=dataset.window,
        penalty_time=dataset.penalty_time, rotation_sign=dataset.rotation_sign,
        component_floor=.1,
    )


def _example_codes(analyzer: FrozenChartAnalyzer, example):
    local, target, physical, _, raw, grid_shape, transition, _ = example
    batch = transition.batch_size
    oracle = (raw / analyzer.composer.port_std).reshape(
        batch, transition.cover.tile_count, -1,
    )
    consensus = transition.cover.tile(
        transition.cover.assemble_weighted(local, batch),
    )
    return oracle, consensus


def _compatibility_residual_vector(
    analyzer: FrozenChartAnalyzer, example, local_fields: Tensor,
) -> Tensor:
    """Stack label-free consensus, port-trace, and divergence residuals."""
    _, _, physical, _, _, _, transition, dataset = example
    batch = transition.batch_size
    consensus = transition.cover.tile(
        transition.cover.assemble_weighted(local_fields, batch),
    )
    valid = physical[:, -1:].to(local_fields)
    consensus_defect = (consensus - local_fields) * valid
    defect_block = consensus_defect.reshape(batch, -1) / np.sqrt(
        consensus_defect[0].numel() * transition.cover.tile_count,
    )
    port = provisional_defect_port_code(
        local_fields, consensus, physical,
        state_mean=tuple(float(x) for x in analyzer.composer.state_mean),
        state_std=tuple(float(x) for x in analyzer.composer.state_std),
        delta_mean=tuple(float(x) for x in analyzer.composer.delta_mean),
        delta_std=tuple(float(x) for x in analyzer.composer.delta_std),
        port_scale=analyzer.composer.port_std,
        functional_kind=analyzer.composer.port_functional_kind,
        force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
        lift_sign=dataset.rotation_sign,
    ).reshape(batch, -1) / np.sqrt(
        transition.cover.tile_count * analyzer.composer.code_dim,
    )
    normalized = transition.cover.assemble_weighted(local_fields, batch)
    mean = local_fields.new_tensor(
        transition.statistics.delta_mean,
    )[None, :, None, None]
    std = local_fields.new_tensor(
        transition.statistics.delta_std,
    )[None, :, None, None]
    state = transition.q_n + normalized * std + mean
    divergence = (
        (state[:, 0:1, 1:-1, 2:] - state[:, 0:1, 1:-1, :-2])
        + (state[:, 1:2, 2:, 1:-1] - state[:, 1:2, :-2, 1:-1])
    ) / (2.0 * dataset.dx)
    divergence_block = divergence.reshape(batch, -1) / np.sqrt(
        divergence[0].numel(),
    )
    return torch.cat((defect_block, port, divergence_block), dim=1)


def fixed_validation_examples(
    model, corpus, statistics, args, device,
) -> list[tuple[Any, ...]]:
    examples = []
    for trajectory, paired in enumerate(corpus.paired):
        count = min(args.validation_samples_per_trajectory, len(paired))
        indices = np.linspace(0, len(paired) - 1, count, dtype=int)
        for ordinal, index in enumerate(indices):
            sample = paired[int(index)]
            examples.append(residual_port_example(
                model, sample, corpus.fine[trajectory], "fine", statistics,
                halo=args.halo,
                seed=args.seed + trajectory * 1_000_003 + ordinal,
                device=device,
            ))
    return examples


@torch.no_grad()
def evaluate(analyzer, examples, statistics) -> dict[str, Any]:
    analyzer.eval()
    modes = {
        "zero": ("zero", None),
        "predicted": ("predicted", None),
        "shuffled": ("shuffled", None),
        "radius0": ("predicted", 0),
        "radius1": ("predicted", 1),
        "global": ("predicted", None),
        "oracle": ("oracle", None),
    }
    records = []
    for example in examples:
        local, target, physical, scalar, _, grid_shape, transition, dataset = example
        oracle, consensus = _example_codes(analyzer, example)
        row = {}
        corrected_fields = {}
        for label, (mode, radius) in modes.items():
            corrected, code, calls = analyzer(
                local, consensus, physical, scalar, grid_shape,
                mode=mode, oracle_code=oracle, graph_radius=radius,
                force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
                lift_sign=dataset.rotation_sign,
            )
            prediction = _physical_prediction(analyzer, example, corrected, statistics)
            corrected_fields[label] = corrected
            row[f"{label}_field"] = float(
                torch.linalg.vector_norm(prediction - transition.q_np1)
                / torch.linalg.vector_norm(transition.q_np1).clamp_min(1e-20)
            )
            row[f"{label}_delta"] = float(
                torch.linalg.vector_norm(corrected - target)
                / torch.linalg.vector_norm(target).clamp_min(1e-20)
            )
            row[f"{label}_force"] = float(_force(prediction, example))
            drag, lift = _force_components(prediction, example)
            row[f"{label}_drag"] = float(drag)
            row[f"{label}_lift"] = float(lift)
            row[f"{label}_calls"] = calls
            if label in ("predicted", "global"):
                row[f"{label}_code"] = float((code - oracle).square().mean())
            no_t = _physical_prediction(
                analyzer, example, corrected, statistics,
                apply_surface=False,
            )
            row[f"noT_{label}_field"] = float(
                torch.linalg.vector_norm(no_t - transition.q_np1)
                / torch.linalg.vector_norm(transition.q_np1).clamp_min(1e-20)
            )
            row[f"noT_{label}_force"] = float(_force(no_t, example))
            no_t_drag, no_t_lift = _force_components(no_t, example)
            row[f"noT_{label}_drag"] = float(no_t_drag)
            row[f"noT_{label}_lift"] = float(no_t_lift)
        residual_zero = _compatibility_residual_vector(
            analyzer, example, corrected_fields["zero"],
        )
        residual_predicted = _compatibility_residual_vector(
            analyzer, example, corrected_fields["global"],
        )
        damping = residual_minimizing_damping(
            residual_zero, residual_predicted - residual_zero,
            lower=0.0, upper=1.0,
        )
        tile_view = (
            damping.scale.shape[0], 1, 1, 1, 1,
        )
        tile_count = transition.cover.tile_count
        zero_tiles = corrected_fields["zero"].reshape(
            transition.batch_size, tile_count,
            *corrected_fields["zero"].shape[1:],
        )
        predicted_tiles = corrected_fields["global"].reshape_as(zero_tiles)
        damped = zero_tiles + damping.scale.reshape(tile_view) * (
            predicted_tiles - zero_tiles
        )
        damped = damped.reshape_as(corrected_fields["zero"])
        damped_prediction = _physical_prediction(
            analyzer, example, damped, statistics,
        )
        row["damped_field"] = float(
            torch.linalg.vector_norm(damped_prediction - transition.q_np1)
            / torch.linalg.vector_norm(transition.q_np1).clamp_min(1e-20)
        )
        row["damped_delta"] = float(
            torch.linalg.vector_norm(damped - target)
            / torch.linalg.vector_norm(target).clamp_min(1e-20)
        )
        row["damped_force"] = float(_force(damped_prediction, example))
        damped_drag, damped_lift = _force_components(damped_prediction, example)
        row["damped_drag"] = float(damped_drag)
        row["damped_lift"] = float(damped_lift)
        row["damping_alpha"] = float(damping.scale.mean())
        row["damping_residual_ratio"] = float(
            torch.sqrt(
                damping.damped_squared_norm
                / damping.baseline_squared_norm.clamp_min(1e-20)
            ).mean()
        )
        fixed, fixed_code, fixed_calls = analyzer(
            local, local, physical, scalar, grid_shape,
            mode="predicted", oracle_code=oracle, graph_radius=None,
            force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
            lift_sign=dataset.rotation_sign,
        )
        row["fixed_point_exact"] = float(torch.equal(fixed, local))
        row["fixed_point_code_norm"] = float(torch.linalg.vector_norm(fixed_code))
        row["fixed_point_calls"] = float(fixed_calls)
        records.append(row)
    metrics = {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0]
    }
    gains = {
        metric: metrics[f"noT_zero_{metric}"] - metrics[f"noT_predicted_{metric}"]
        for metric in ("field", "drag", "lift")
    }
    paired = {}
    seed_offset = 0
    for metric in ("field", "drag", "lift"):
        paired[f"predicted_minus_zero_{metric}"] = _paired_bootstrap_interval(
            np.asarray([
                record[f"noT_predicted_{metric}"] - record[f"noT_zero_{metric}"]
                for record in records
            ]), seed=20260806 + seed_offset,
        )
        paired[f"shuffled_minus_predicted_{metric}"] = _paired_bootstrap_interval(
            np.asarray([
                record[f"noT_shuffled_{metric}"] - record[f"noT_predicted_{metric}"]
                for record in records
            ]), seed=20260837 + seed_offset,
        )
        seed_offset += 101
    for radius, offset in (("radius0", 701), ("radius1", 907)):
        paired[f"{radius}_minus_global_field"] = _paired_bootstrap_interval(
            np.asarray([
                record[f"noT_{radius}_field"] - record["noT_global_field"]
                for record in records
            ]), seed=20260806 + offset,
        )
    relative_field_gain = gains["field"] / max(metrics["noT_zero_field"], 1.0e-20)
    oracle_gap = metrics["noT_zero_field"] - metrics["noT_oracle_field"]
    oracle_gap_fraction_closed = (
        gains["field"] / oracle_gap if oracle_gap > 0.0 else float("nan")
    )
    gate = {
        "field_improvement_ci95_nonpositive": (
            paired["predicted_minus_zero_field"]["ci95_high"] <= 0.0
        ),
        "field_relative_gain_at_least_0.1_percent": relative_field_gain >= .001,
        "force_nonregression": metrics["noT_predicted_force"] <= metrics["noT_zero_force"],
        "drag_nonregression": metrics["noT_predicted_drag"] <= metrics["noT_zero_drag"],
        "lift_nonregression": metrics["noT_predicted_lift"] <= metrics["noT_zero_lift"],
        "shuffle_loses_half_field_gain": (
            gains["field"] > 0.0
            and metrics["noT_shuffled_field"] - metrics["noT_predicted_field"]
            >= .5 * gains["field"]
            and paired["shuffled_minus_predicted_field"]["ci95_low"] >= 0.0
        ),
        "global_beats_radius0": (
            metrics["noT_global_field"] < metrics["noT_radius0_field"]
            and paired["radius0_minus_global_field"]["ci95_low"] >= 0.0
        ),
        "global_beats_radius1": (
            metrics["noT_global_field"] < metrics["noT_radius1_field"]
            and paired["radius1_minus_global_field"]["ci95_low"] >= 0.0
        ),
        "oracle_field_improves": metrics["noT_oracle_field"] < metrics["noT_zero_field"],
        "oracle_force_nonregression": metrics["noT_oracle_force"] <= metrics["noT_zero_force"],
        "oracle_drag_nonregression": metrics["noT_oracle_drag"] <= metrics["noT_zero_drag"],
        "oracle_lift_nonregression": metrics["noT_oracle_lift"] <= metrics["noT_zero_lift"],
        "exact_single_message": metrics["global_calls"] == 1.0,
        "zero_defect_fixed_point_exact": metrics["fixed_point_exact"] == 1.0,
    }
    gate["accepted"] = all(gate.values())
    alpha = np.asarray([record["damping_alpha"] for record in records])
    damping_report = {
        "label_free": True, "additional_neural_calls": 0,
        "alpha_mean": float(alpha.mean()), "alpha_median": float(np.median(alpha)),
        "alpha_p10": float(np.quantile(alpha, .1)),
        "alpha_p90": float(np.quantile(alpha, .9)),
        "alpha_zero_fraction": float(np.mean(alpha <= 1e-8)),
        "alpha_one_fraction": float(np.mean(alpha >= 1.0 - 1e-8)),
        "raw_global": {
            key: metrics[f"global_{key}"]
            for key in ("field", "delta", "force", "drag", "lift")
        },
        "damped": {
            key: metrics[f"damped_{key}"]
            for key in ("field", "delta", "force", "drag", "lift")
        },
        "residual_ratio": metrics["damping_residual_ratio"],
        "selection_role": "diagnostic_only",
    }
    return {
        "samples": len(records), "metrics": metrics, "gate": gate,
        "paired_bootstrap": paired,
        "relative_field_gain": relative_field_gain,
        "oracle_field_gap_fraction_closed": oracle_gap_fraction_closed,
        "qoi_causality_diagnostic": {
            "drag_shuffle_loses_half_gain": (
                gains["drag"] > 0.0
                and metrics["noT_shuffled_drag"] - metrics["noT_predicted_drag"]
                >= .5 * gains["drag"]
            ),
            "lift_shuffle_loses_half_gain": (
                gains["lift"] > 0.0
                and metrics["noT_shuffled_lift"] - metrics["noT_predicted_lift"]
                >= .5 * gains["lift"]
            ),
            "claim_role": (
                "diagnostic only; force gains are not attributed to global "
                "messaging unless their separate shuffle diagnostics pass"
            ),
        },
        "shuffle_strategy": (
            "within-cover cyclic tile-code permutation for batch size one; "
            "preserves the exact empirical code marginal and breaks spatial assignment"
        ),
        "predicted_damping_control": damping_report,
        "primary_endpoint": "strict_noT_after_POU",
        "retained_T_X_role": "explicitly costed practical ablation only",
    }


def train(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    load_args = argparse.Namespace(candidate=args.chart_checkpoint, graph_radius=None)
    chart, base, model, statistics = load_candidate(load_args, device)
    if not chart["validation"].get("chart_gate", False):
        raise ValueError("response chart did not pass fixed-oracle field/force gate")
    analyzer = FrozenChartAnalyzer(model, use_defect=args.use_defect).to(device)
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
    history = []
    best = float("inf")
    try:
        baseline = evaluate(analyzer, validation_examples, statistics)
        history.append({"epoch": 0, "validation": baseline})
        generator = random.Random(args.seed + 17_000_003)
        for epoch in range(1, args.epochs + 1):
            analyzer.train(); losses = []
            for local in range(args.samples_per_epoch):
                _, _, fine, sample = train_corpus.random_sample(generator)
                example = residual_port_example(
                    model, sample, fine, "fine", statistics, halo=args.halo,
                    seed=args.seed + epoch * 10_000_019 + local, device=device,
                )
                endpoint, target, physical, scalar, _, grid_shape, _, dataset = example
                oracle, consensus = _example_codes(analyzer, example)
                optimizer.zero_grad(set_to_none=True)
                corrected, code, calls = analyzer(
                    endpoint, consensus, physical, scalar, grid_shape,
                    mode="predicted", oracle_code=oracle, graph_radius=None,
                    force_area_scale=2.0 * dataset.dx**2 / dataset.penalty_time,
                    lift_sign=dataset.rotation_sign,
                )
                if calls != 1:
                    raise RuntimeError("analyzer violated the one-message contract")
                valid = physical[:, -1:].to(endpoint)
                field = ((corrected - target).square() * valid).sum() / (
                    valid.sum().clamp_min(1) * endpoint.shape[1]
                )
                code_loss = (code - oracle).square().mean()
                prediction = _physical_prediction(
                    analyzer, example, corrected, statistics,
                )
                force = _balanced_force(prediction, example)
                objective = (
                    args.field_weight * field + args.code_weight * code_loss
                    + args.force_weight * force
                )
                objective.backward()
                torch.nn.utils.clip_grad_norm_(analyzer.analysis_parameters, 1.0)
                optimizer.step()
                losses.append((float(field), float(code_loss), float(force)))
            validation = evaluate(analyzer, validation_examples, statistics)
            record = {
                "epoch": epoch,
                "train_field": float(np.mean([x[0] for x in losses])),
                "train_code": float(np.mean([x[1] for x in losses])),
                "train_force": float(np.mean([x[2] for x in losses])),
                "validation": validation,
            }
            history.append(record); print(json.dumps(record), flush=True)
            score = (
                validation["metrics"]["predicted_field"]
                + args.selection_force_weight
                * validation["metrics"]["predicted_force"]
            )
            payload = {
                "contract": CONTRACT, "chart_checkpoint": str(args.chart_checkpoint.resolve()),
                "chart_sha256": hashlib.sha256(args.chart_checkpoint.read_bytes()).hexdigest(),
                "use_provisional_defect": args.use_defect,
                "analyzer": analyzer.state_dict(), "epoch": epoch,
                "validation": validation, "args": vars(args),
                "split": base["split"], "protected_test_trajectories_opened": 0,
                "validation_indices": "linspace_fixed_per_trajectory",
                "messages_per_physical_step": 1,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-defect", action="store_true")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples-per-epoch", type=int, default=256)
    parser.add_argument("--validation-samples-per-trajectory", type=int, default=8)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--field-weight", type=float, default=1.0)
    parser.add_argument("--code-weight", type=float, default=.5)
    parser.add_argument("--force-weight", type=float, default=.5)
    parser.add_argument("--selection-force-weight", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
