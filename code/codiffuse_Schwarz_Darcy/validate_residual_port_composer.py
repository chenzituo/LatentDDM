"""Validation-only causal ladder for the phase-separated Darcy composer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz.compatibility_defect import residual_minimizing_damping
from codiffuse_Schwarz.residual_port import scale_port_code

from .adapter import H5DarcyDataset
from .evaluate import move_batch, relative_l2
from .residual_port_composer import (
    ResidualPortCommunicator, SignedSourceLinearResidualPortCommunicator,
    SourceLinearResidualPortCommunicator,
)
from .train_residual_port_chart import frozen_context, load_frozen_local, sha256
from .train_residual_port_communicator import (
    CANONICAL_ANALYZER_CONTRACT, CONTRACT, communicator_features,
    compatibility_cycle_metrics, source_linear_features,
    source_linear_signal_width,
    forward_composer, interface_losses, load_frozen_chart, physics_losses,
)
from .validate_latent_schur import measured_q, q_from_corrected


Tensor = torch.Tensor


def _compatibility_blocks(
    corrected: Tensor,
    context,
    batch: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-case value/flux residual blocks before fixed normalization."""
    case_count = context.target_normalized.shape[0]
    tile_count = context.cover.tile_count
    assembled = context.cover.assemble_weighted(corrected, case_count) * batch["mask"]
    consensus = context.cover.tile(assembled)
    halo, core = context.cover.halo, context.cover.core
    owned = torch.zeros_like(context.valid)
    owned[..., halo:halo + core, halo:halo + core] = 1.0
    trace = context.valid * (1.0 - owned)
    value = (corrected - consensus) * trace
    dx = 2.0 / max(context.cover.height, context.cover.width)
    local_dx = (corrected[..., 2:] - corrected[..., :-2]) / (2.0 * dx)
    consensus_dx = (consensus[..., 2:] - consensus[..., :-2]) / (2.0 * dx)
    local_dy = (corrected[..., 2:, :] - corrected[..., :-2, :]) / (2.0 * dx)
    consensus_dy = (consensus[..., 2:, :] - consensus[..., :-2, :]) / (2.0 * dx)
    permeability = torch.pow(
        10.0, context.cover.tile(batch["logk"].to(corrected.dtype)).clamp(-6, 6),
    )
    flux_x = permeability[..., 1:-1] * (local_dx - consensus_dx) * trace[..., 1:-1]
    flux_y = permeability[..., 1:-1, :] * (local_dy - consensus_dy) * trace[..., 1:-1, :]
    def grouped(value: Tensor) -> Tensor:
        return value.reshape(case_count, tile_count, -1).flatten(1)
    return grouped(value), grouped(flux_x), grouped(flux_y), assembled


def damp_affine_compatibility_direction(
    baseline: Tensor,
    corrected: Tensor,
    context,
    batch: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Retract one field direction against a target-free compatibility norm.

    ``baseline`` may be the frozen local endpoint or a previously accepted
    one-message field.  The returned field lies on the affine segment from
    ``baseline`` to ``corrected`` and cannot increase the measured normalized
    value+flux compatibility norm relative to that baseline.  The operation
    intentionally remains differentiable: training an incremental correction
    must receive gradients through the safeguarded deployed field.  Validation
    callers already invoke it under their surrounding ``no_grad`` context.
    """
    if baseline.shape != corrected.shape:
        raise ValueError("affine retraction fields must have matching shapes")
    zero_blocks = _compatibility_blocks(baseline, context, batch)[:3]
    raw_blocks = _compatibility_blocks(corrected, context, batch)[:3]
    normalized_zero, normalized_direction = [], []
    for zero, raw in zip(zero_blocks, raw_blocks):
        scale = zero.square().mean(1, keepdim=True).sqrt().clamp_min(1.0e-6)
        normalized_zero.append(zero / scale)
        normalized_direction.append((raw - zero) / scale)
    residual_zero = torch.cat(normalized_zero, 1)
    residual_direction = torch.cat(normalized_direction, 1)
    damping = residual_minimizing_damping(
        residual_zero, residual_direction, lower=0.0, upper=1.0,
    )
    tile_count = context.cover.tile_count
    alpha = damping.scale.repeat_interleave(tile_count).reshape(
        -1, *((1,) * (corrected.ndim - 1)),
    )
    damped = baseline + alpha * (corrected - baseline)
    assembled = context.cover.assemble_weighted(
        damped, context.target_normalized.shape[0],
    ) * batch["mask"]
    ratio = damping.damped_squared_norm / damping.baseline_squared_norm.clamp_min(1.0e-12)
    return damped, assembled, damping.scale, ratio


def damp_compatibility_direction(
    corrected: Tensor,
    context,
    batch: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Backward-compatible retraction about the frozen local endpoint."""
    return damp_affine_compatibility_direction(
        context.local, corrected, context, batch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radii", nargs="+", default=("0", "1", "2", "global"))
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-validation-cases", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _radii(values: list[str]) -> list[int | None]:
    result = []
    for value in values:
        if value == "global":
            result.append(None)
        else:
            radius = int(value)
            if radius < 0:
                raise ValueError("causal radii must be nonnegative")
            result.append(radius)
    return result


def _metrics(
    normalized: Tensor,
    corrected: Tensor,
    context,
    batch: dict[str, Tensor],
    solution_std: float,
) -> dict[str, float]:
    prediction = normalized * solution_std * batch["mask"]
    l2 = relative_l2(prediction, batch["solution"], batch["mask"]).mean()
    energy, residual = physics_losses(
        normalized, batch, solution_std,
    )
    value, flux = interface_losses(corrected, normalized, context, batch)
    return {
        "relative_l2": float(l2),
        "energy_relative_square": float(energy),
        "residual_relative_square": float(residual),
        "interface_value_square": float(value),
        "interface_flux_square": float(flux),
    }


@torch.no_grad()
def validate_ladder(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError("causal validation requires the phase-separated communicator")
    if checkpoint["source_local_sha256"] != sha256(args.local_checkpoint):
        raise ValueError("validation local checkpoint identity differs")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart_path = Path(checkpoint["source_chart_checkpoint"])
    if checkpoint["source_chart_sha256"] != sha256(chart_path):
        raise ValueError("selected response chart identity differs")
    chart, spec, scale, _ = load_frozen_chart(chart_path, device)
    communicator_spec = checkpoint["communicator_spec"]
    analyzer_mode = communicator_spec.get("analyzer_mode", "generic")
    if analyzer_mode in ("source_linear", "value_flux"):
        if communicator_spec.get("canonical_analyzer_contract") != \
                CANONICAL_ANALYZER_CONTRACT:
            raise ValueError(
                "legacy source_linear alias lacks the canonical homogeneous "
                "compatibility-residual contract"
            )
        expected_signal_width = source_linear_signal_width(spec) + (
            8 if analyzer_mode == "value_flux" else 0
        )
        if int(communicator_spec.get("source_linear_signal_width", -1)) != \
                expected_signal_width:
            raise ValueError(
                "source-linear checkpoint width does not match its port degree"
            )
        communicator_type = (
            SignedSourceLinearResidualPortCommunicator
            if communicator_spec.get("kernel_type", "softmax") == "signed_l1_tanh"
            else SourceLinearResidualPortCommunicator
        )
        communicator = communicator_type(
            int(communicator_spec["physical_channels"]),
            int(communicator_spec["scalar_dim"]), spec.code_width,
            expected_signal_width,
            token_width=int(communicator_spec["token_width"]),
            heads=int(communicator_spec["heads"]),
        ).to(device)
        communicator.analyzer_mode = analyzer_mode
    else:
        communicator = ResidualPortCommunicator(
            1, int(communicator_spec["physical_channels"]),
            int(communicator_spec["scalar_dim"]), spec.code_width,
            token_width=int(communicator_spec["token_width"]),
            heads=int(communicator_spec["heads"]),
            aligned_width=int(communicator_spec.get("aligned_width", 0)),
        ).to(device)
        communicator.analyzer_mode = communicator_spec.get(
            "analyzer_mode", "generic",
        )
    communicator.load_state_dict(checkpoint["communicator"], strict=True)
    communicator.eval(); chart.eval(); local.eval()
    dataset = H5DarcyDataset(args.data, "validation")
    if not dataset.has_face_dirichlet:
        raise ValueError("causal validation requires face-resolved validation data")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    radii = _radii(list(args.radii))
    rows: list[dict[str, Any]] = []
    processed = 0
    for host_batch in loader:
        if processed >= args.max_validation_cases:
            break
        batch = move_batch(host_batch, device)
        original = batch["solution"].shape[0]
        keep = min(original, args.max_validation_cases - processed)
        for key, value in list(batch.items()):
            if isinstance(value, Tensor) and value.shape[0] == original:
                batch[key] = value[:keep]
        if device.type == "cuda":
            torch.cuda.synchronize()
        local_started = time.perf_counter()
        context = frozen_context(
            local, adapter, batch, spec, args.core, args.halo,
        )
        aligned = (
            measured_q(context, batch, spec, scale, "value_flux")[0]
            if analyzer_mode == "value_flux"
            else communicator_features(
                communicator, context, batch, spec, scale, adapter,
            )
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        local_seconds = (time.perf_counter() - local_started) / keep
        batch_size = batch["solution"].shape[0]
        baseline_q = (
            aligned if analyzer_mode == "value_flux"
            else source_linear_features(context, spec, scale)
        )
        def cycle(corrected: Tensor) -> dict[str, float]:
            if analyzer_mode == "value_flux":
                corrected_q = q_from_corrected(
                    corrected, context, batch, spec, scale, "value_flux",
                )
                axes = tuple(range(1, baseline_q.ndim))
                ratio_square_per_case = corrected_q.square().sum(axes) \
                    / baseline_q.square().sum(axes).clamp_min(1.0e-12)
                ratio = ratio_square_per_case.sqrt().mean()
                ratio_square = ratio_square_per_case.mean()
            else:
                ratio, ratio_square = compatibility_cycle_metrics(
                    corrected, context, spec, scale, baseline_q=baseline_q,
                )
            return {
                "real_defect_cycle_ratio": float(ratio),
                "real_defect_cycle_relative_square": float(ratio_square),
            }

        local_assembled = context.cover.assemble_weighted(
            context.local, batch_size,
        ) * batch["mask"]
        rows.append({
            "method": "local_zero_code", "radius": "none",
            **_metrics(
                local_assembled, context.local, context, batch,
                adapter.statistics.solution_std,
            ),
            **cycle(context.local),
            "wall_seconds": local_seconds,
            "local_nfe": context.cover.tile_count,
            "batched_local_calls": 1,
            "communication_rounds": 0,
            "cases": keep,
            "damping_alpha_mean": float("nan"),
            "damping_alpha_min": float("nan"),
            "damping_alpha_max": float("nan"),
            "compatibility_norm_ratio": float("nan"),
        })

        predicted_codes: dict[str, Tensor] = {}
        for radius in radii:
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            assembled, predicted, _, detail = forward_composer(
                communicator, chart, context, scale, batch,
                max_graph_radius=radius,
                aligned_features=aligned,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = local_seconds + (time.perf_counter() - started) / keep
            label = "global" if radius is None else str(radius)
            predicted_codes[label] = predicted
            rows.append({
                "method": "predicted", "radius": label,
                **_metrics(
                    assembled, detail["corrected"], context, batch,
                    adapter.statistics.solution_std,
                ),
                **cycle(detail["corrected"]),
                "wall_seconds": elapsed,
                "local_nfe": context.cover.tile_count,
                "batched_local_calls": 1,
                "communication_rounds": 1,
                "cases": keep,
                "damping_alpha_mean": float("nan"),
                "damping_alpha_min": float("nan"),
                "damping_alpha_max": float("nan"),
                "compatibility_norm_ratio": float("nan"),
            })
            if device.type == "cuda":
                torch.cuda.synchronize()
            damping_started = time.perf_counter()
            damped_corrected, damped_assembled, alpha, ratio = (
                damp_compatibility_direction(
                    detail["corrected"], context, batch,
                )
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            damped_elapsed = elapsed + (
                time.perf_counter() - damping_started
            ) / keep
            rows.append({
                "method": "predicted_damped", "radius": label,
                **_metrics(
                    damped_assembled, damped_corrected, context, batch,
                    adapter.statistics.solution_std,
                ),
                **cycle(damped_corrected),
                "wall_seconds": damped_elapsed,
                "local_nfe": context.cover.tile_count,
                "batched_local_calls": 1,
                "communication_rounds": 1,
                "cases": keep,
                "damping_alpha_mean": float(alpha.mean()),
                "damping_alpha_min": float(alpha.min()),
                "damping_alpha_max": float(alpha.max()),
                "compatibility_norm_ratio": float(ratio.mean()),
            })

        oracle = scale_port_code(context.oracle_code, scale)
        oracle_assembled, _, _, oracle_detail = forward_composer(
            communicator, chart, context, scale, batch, code_override=oracle,
        )
        rows.append({
            "method": "oracle_chart", "radius": "oracle",
            **_metrics(
                oracle_assembled, oracle_detail["corrected"], context, batch,
                adapter.statistics.solution_std,
            ),
            **cycle(oracle_detail["corrected"]),
            "wall_seconds": float("nan"),
            "local_nfe": context.cover.tile_count,
            "batched_local_calls": 1,
            "communication_rounds": 0,
            "cases": keep,
            "damping_alpha_mean": float("nan"),
            "damping_alpha_min": float("nan"),
            "damping_alpha_max": float("nan"),
            "compatibility_norm_ratio": float("nan"),
        })
        if "global" in predicted_codes:
            shuffled = torch.roll(predicted_codes["global"], shifts=1, dims=1)
            shuffled_assembled, _, _, shuffled_detail = forward_composer(
                communicator, chart, context, scale, batch,
                code_override=shuffled,
            )
            rows.append({
                "method": "shuffled_code", "radius": "global_shuffled",
                **_metrics(
                    shuffled_assembled, shuffled_detail["corrected"], context,
                    batch, adapter.statistics.solution_std,
                ),
                **cycle(shuffled_detail["corrected"]),
                "wall_seconds": float("nan"),
                "local_nfe": context.cover.tile_count,
                "batched_local_calls": 1,
                "communication_rounds": 1,
                "cases": keep,
                "damping_alpha_mean": float("nan"),
                "damping_alpha_min": float("nan"),
                "damping_alpha_max": float("nan"),
                "compatibility_norm_ratio": float("nan"),
            })
        processed += keep

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "causal_ladder.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summaries: dict[str, dict[str, float]] = {}
    for method, radius in sorted({(row["method"], row["radius"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method and row["radius"] == radius]
        summaries[f"{method}/radius_{radius}"] = {
            key: float(np.average(
                [float(row[key]) for row in selected],
                weights=[int(row["cases"]) for row in selected],
            ))
            for key in (
                "relative_l2", "energy_relative_square",
                "residual_relative_square", "interface_value_square",
                "interface_flux_square", "local_nfe", "batched_local_calls",
                "communication_rounds", "real_defect_cycle_ratio",
                "real_defect_cycle_relative_square",
            )
        }
        finite_wall = [
            row for row in selected if np.isfinite(float(row["wall_seconds"]))
        ]
        summaries[f"{method}/radius_{radius}"]["wall_seconds"] = (
            float(np.average(
                [float(row["wall_seconds"]) for row in finite_wall],
                weights=[int(row["cases"]) for row in finite_wall],
            )) if finite_wall else None
        )
        for key in (
            "damping_alpha_mean", "damping_alpha_min", "damping_alpha_max",
            "compatibility_norm_ratio",
        ):
            finite = [row for row in selected if np.isfinite(float(row[key]))]
            summaries[f"{method}/radius_{radius}"][key] = (
                float(np.average(
                    [float(row[key]) for row in finite],
                    weights=[int(row["cases"]) for row in finite],
                )) if finite else None
            )
    result = {
        "contract": "darcy_residual_port_validation_ladder_v1",
        "selection_split": "validation",
        "test_subsets_opened": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "local_checkpoint": str(args.local_checkpoint.resolve()),
        "source_local_sha256": checkpoint["source_local_sha256"],
        "source_chart_checkpoint": str(chart_path.resolve()),
        "source_chart_sha256": checkpoint["source_chart_sha256"],
        "chart_type": checkpoint.get("chart_spec", {}).get(
            "chart_type", "nonlinear_12mode",
        ),
        "analyzer_mode": communicator_spec.get("analyzer_mode", "generic"),
        "radii": ["global" if value is None else value for value in radii],
        "cases": processed,
        "oracle_uses_target_for_diagnostic_only": True,
        "timing_excludes_physics_diagnostics": True,
        "local_nfe_definition": "one frozen local evaluation per tile",
        "damping": {
            "optional_post_communicator_scalar": True,
            "target_or_pde_solve_used": False,
            "residual": "per-case normalized POU consensus plus permeability-weighted flux compatibility",
            "interval": [0.0, 1.0],
            "cannot_replace": ["code", "shuffle", "field", "physics"],
        },
        "summary": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(validate_ladder(parse_args()), indent=2))
