"""Identify the one-pass composition error terms on held-out Darcy domains.

The trained composer decodes one scalar compatibility code per constant/linear
mode and tile.  Because this basis is fixed, a weighted least-squares oracle
code is identifiable from the held-out target.  This evaluator separates:

* the uncommunicated local endpoint error;
* the fixed-chart representation floor (oracle code);
* the learned compatibility-code error; and
* the amplification induced by partition-of-unity assembly.

The oracle is an evaluation diagnostic only.  It is never used by inference
or training and does not call the discrete Darcy operator.  Energy and finite-
volume residuals are computed only after predictions have been formed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz.learned_composer import canonical_modes
from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import (
    DarcyAdapter, DarcyStatistics, H5DarcyDataset, MODEL_SPEC,
    batch_to_transition, build_model, tiled_darcy_conditions,
)
from .evaluate import energy_relative_error, move_batch, relative_l2, residual_rms
from .train_operator_composer import CONTRACT, OperatorOnlyDarcyComposer


Tensor = torch.Tensor
VARIANTS = ("local", "predicted_constant", "predicted_affine",
            "oracle_constant", "oracle_affine")


def oracle_mode_coefficients(
    local: Tensor, target: Tensor, weight: Tensor, modes: Tensor, *,
    ridge: float = 1.0e-6,
) -> Tensor:
    """Weighted least-squares code for ``target - local`` in ``modes``.

    The solve is independent for every tile and field channel.  A trace-scaled
    ridge makes fully masked tiles return zero without changing populated
    least-squares solutions appreciably.
    """
    if local.shape != target.shape or weight.shape != (local.shape[0], 1, *local.shape[2:]):
        raise ValueError("local, target, and weight have incompatible shapes")
    if modes.ndim != local.ndim - 1 or tuple(modes.shape[1:]) != tuple(local.shape[2:]):
        raise ValueError("mode basis and local spatial shape differ")
    work_dtype = torch.float64 if local.dtype == torch.float64 else torch.float32
    residual = (target - local).to(work_dtype).flatten(2)
    pixel_weight = weight.to(work_dtype).flatten(2)[:, 0]
    basis = modes.to(work_dtype).flatten(1)
    gram = torch.einsum("np,mp,kp->nmk", pixel_weight, basis, basis)
    rhs = torch.einsum("np,ncp,mp->ncm", pixel_weight, residual, basis)
    mode_count = basis.shape[0]
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1.0)
    identity = torch.eye(mode_count, device=gram.device, dtype=gram.dtype)
    regularized = gram + (ridge * scale)[:, None, None] * identity
    coefficient = torch.linalg.solve(regularized, rhs.transpose(1, 2)).transpose(1, 2)
    return coefficient.to(local.dtype)


def decode_modes(coefficients: Tensor, modes: Tensor, valid: Tensor) -> Tensor:
    return torch.einsum("bcm,m...->bc...", coefficients, modes) * valid


def _local_norms(error: Tensor, target: Tensor, weight: Tensor,
                 batch: int, tiles: int) -> tuple[Tensor, Tensor]:
    axes = tuple(range(1, error.ndim))
    numerator = (error.square() * weight).sum(axes).reshape(batch, tiles).sum(1)
    denominator = (target.square() * weight).sum(axes).reshape(batch, tiles).sum(1)
    count = weight.sum(axes).reshape(batch, tiles).sum(1).clamp_min(1.0)
    return (numerator / count).sqrt(), (numerator / denominator.clamp_min(1.0e-12)).sqrt()


def _global_rms(error: Tensor, mask: Tensor) -> Tensor:
    axes = tuple(range(1, error.ndim))
    return ((error.square() * mask).sum(axes) / mask.sum(axes).clamp_min(1.0)).sqrt()


def _physical_metrics(prediction: Tensor, batch: dict[str, Any]) -> dict[str, Tensor]:
    return {
        "relative_l2": relative_l2(prediction, batch["solution"], batch["mask"]),
        "energy_relative_error": energy_relative_error(
            prediction, batch["solution"], batch["logk"], batch["mask"],
        ),
        "residual_rms": residual_rms(prediction, batch),
    }


def _mean_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float]:
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def _plot(variant_rows: list[dict[str, Any]], theorem_rows: list[dict[str, Any]],
          output_dir: Path) -> None:
    means = {
        variant: np.mean([row["relative_l2"] for row in variant_rows
                          if row["variant"] == variant])
        for variant in VARIANTS
    }
    labels = ("local", "pred. const.", "pred. affine", "oracle const.", "oracle affine")
    colors = ("#777777", "#4c78a8", "#2b5b84", "#f58518", "#b85c00")
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.25))
    axes[0].bar(np.arange(len(VARIANTS)), [means[name] for name in VARIANTS], color=colors)
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=24, ha="right")
    axes[0].set_ylabel("relative $L^2$ error")
    axes[0].set_title("Code-width and oracle decomposition")
    axes[0].grid(axis="y", alpha=.25)

    oracle = np.asarray([row["oracle_affine_broken_rms"] for row in theorem_rows])
    code = np.asarray([row["decoded_code_error_broken_rms"] for row in theorem_rows])
    total = np.asarray([row["predicted_affine_broken_rms"] for row in theorem_rows])
    x = np.arange(3)
    values = (oracle.mean(), code.mean(), total.mean())
    axes[1].bar(x, values, color=("#f58518", "#54a24b", "#2b5b84"))
    axes[1].set_xticks(x, ("chart floor", "code gap", "total"), rotation=18, ha="right")
    axes[1].set_ylabel("broken-space RMS")
    axes[1].set_title("Terms in the one-pass bound")
    axes[1].grid(axis="y", alpha=.25)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"theorem_decomposition.{suffix}", dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "val", "A", "B", "C", "D"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--offsets", nargs="+", default=("0,0", "8,8", "16,16"))
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CONTRACT or not checkpoint.get("operator_free"):
        raise ValueError("checkpoint is not the operator-only Darcy composer")
    statistics = DarcyStatistics.from_json(checkpoint["statistics"])
    adapter = DarcyAdapter(statistics, use_discrete_residual=False)
    model = OperatorOnlyDarcyComposer(build_model(adapter, communication_stage="none"), adapter).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = H5DarcyDataset(args.data, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    offsets = [tuple(int(piece) for piece in value.split(",")) for value in args.offsets]
    if any(len(value) != 2 or min(value) < 0 or max(value) >= MODEL_SPEC["core"]
           for value in offsets):
        raise ValueError("offsets must be y,x pairs inside the core")

    variant_rows: list[dict[str, Any]] = []
    theorem_rows: list[dict[str, Any]] = []
    processed = 0
    for host_batch in loader:
        if args.max_cases is not None and processed >= args.max_cases:
            break
        batch = move_batch(host_batch, device)
        original_batch = batch["solution"].shape[0]
        if args.max_cases is not None:
            keep = min(original_batch, args.max_cases - processed)
            for key, value in list(batch.items()):
                if isinstance(value, Tensor) and value.shape[0] == original_batch:
                    batch[key] = value[:keep]
        batch_size = batch["solution"].shape[0]
        current = torch.zeros_like(batch["solution"])
        transition = batch_to_transition(batch, current, target=batch["solution"])
        normalized_target = adapter.normalized_target(transition).float()
        mask = batch["mask"].float()

        for offset_y, offset_x in offsets:
            cover = CartesianCover(normalized_target.shape[-2], normalized_target.shape[-1],
                                   MODEL_SPEC["core"], MODEL_SPEC["halo"], offset_y, offset_x)
            physical, scalar = tiled_darcy_conditions(adapter, transition, cover)
            valid = cover.valid_tiles(batch_size, device, normalized_target.dtype)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                _, local_raw, predicted_coefficients_raw = model(
                    physical, scalar, valid, cover.grid_shape,
                )
            local = local_raw.float()
            predicted_coefficients = predicted_coefficients_raw.float()
            local_target = cover.tile(normalized_target)
            local_weight = cover.tile(mask) * valid
            modes = canonical_modes(local.shape[2:], device, local.dtype)
            oracle_coefficients = oracle_mode_coefficients(
                local, local_target, local_weight, modes, ridge=args.ridge,
            )
            oracle_constant_coefficients = oracle_mode_coefficients(
                local, local_target, local_weight, modes[:1], ridge=args.ridge,
            )

            local_correction = torch.zeros_like(local)
            pred_constant = decode_modes(predicted_coefficients[..., :1], modes[:1], valid)
            pred_affine = decode_modes(predicted_coefficients, modes, valid)
            oracle_constant = decode_modes(
                oracle_constant_coefficients, modes[:1], valid,
            )
            oracle_affine = decode_modes(oracle_coefficients, modes, valid)
            corrections = {
                "local": local_correction,
                "predicted_constant": pred_constant,
                "predicted_affine": pred_affine,
                "oracle_constant": oracle_constant,
                "oracle_affine": oracle_affine,
            }
            local_errors: dict[str, tuple[Tensor, Tensor]] = {}
            assembled_errors: dict[str, Tensor] = {}
            for variant, correction in corrections.items():
                endpoint = (local + correction) * valid
                assembled = cover.assemble_weighted(endpoint, batch_size)
                prediction = adapter.apply_endpoint(transition, assembled)
                metrics = _physical_metrics(prediction, batch)
                local_error = endpoint - local_target
                local_errors[variant] = _local_norms(
                    local_error, local_target, local_weight, batch_size, cover.tile_count,
                )
                assembled_errors[variant] = assembled - normalized_target
                global_rms = _global_rms(assembled_errors[variant], mask)
                broken_rms, broken_relative = local_errors[variant]
                for item in range(batch_size):
                    variant_rows.append({
                        "case_index": int(batch["case_index"][item]),
                        "offset_y": offset_y, "offset_x": offset_x,
                        "variant": variant,
                        "mode_count": {"local": 0, "predicted_constant": 1,
                                       "predicted_affine": 3, "oracle_constant": 1,
                                       "oracle_affine": 3}[variant],
                        "relative_l2": float(metrics["relative_l2"][item]),
                        "energy_relative_error": float(metrics["energy_relative_error"][item]),
                        "residual_rms": float(metrics["residual_rms"][item]),
                        "broken_relative_l2": float(broken_relative[item]),
                        "broken_rms": float(broken_rms[item]),
                        "global_rms": float(global_rms[item]),
                        "assembly_amplification": float(
                            global_rms[item] / broken_rms[item].clamp_min(1.0e-12)
                        ),
                    })

            code_correction_error = pred_affine - oracle_affine
            code_broken_rms, _ = _local_norms(
                code_correction_error, local_target, local_weight,
                batch_size, cover.tile_count,
            )
            assembled_code_error = cover.assemble_weighted(code_correction_error, batch_size)
            code_global_rms = _global_rms(assembled_code_error, mask)
            predicted_broken = local_errors["predicted_affine"][0]
            oracle_broken = local_errors["oracle_affine"][0]
            predicted_global = _global_rms(assembled_errors["predicted_affine"], mask)
            oracle_global = _global_rms(assembled_errors["oracle_affine"], mask)
            coefficient_delta = predicted_coefficients - oracle_coefficients
            coefficient_rmse = coefficient_delta.square().mean((1, 2)).sqrt().reshape(
                batch_size, cover.tile_count,
            ).mean(1)
            coefficient_relative = (
                coefficient_delta.square().sum((1, 2)).reshape(batch_size, cover.tile_count).sum(1).sqrt()
                / oracle_coefficients.square().sum((1, 2)).reshape(
                    batch_size, cover.tile_count,
                ).sum(1).sqrt().clamp_min(1.0e-12)
            )
            for item in range(batch_size):
                theorem_rows.append({
                    "case_index": int(batch["case_index"][item]),
                    "offset_y": offset_y, "offset_x": offset_x,
                    "oracle_affine_broken_rms": float(oracle_broken[item]),
                    "decoded_code_error_broken_rms": float(code_broken_rms[item]),
                    "predicted_affine_broken_rms": float(predicted_broken[item]),
                    "broken_triangle_ratio": float(
                        predicted_broken[item]
                        / (oracle_broken[item] + code_broken_rms[item]).clamp_min(1.0e-12)
                    ),
                    "oracle_affine_global_rms": float(oracle_global[item]),
                    "decoded_code_error_global_rms": float(code_global_rms[item]),
                    "predicted_affine_global_rms": float(predicted_global[item]),
                    "global_triangle_ratio": float(
                        predicted_global[item]
                        / (oracle_global[item] + code_global_rms[item]).clamp_min(1.0e-12)
                    ),
                    "coefficient_rmse": float(coefficient_rmse[item]),
                    "coefficient_relative_l2": float(coefficient_relative[item]),
                    "oracle_assembly_amplification": float(
                        oracle_global[item] / oracle_broken[item].clamp_min(1.0e-12)
                    ),
                    "code_assembly_amplification": float(
                        code_global_rms[item] / code_broken_rms[item].clamp_min(1.0e-12)
                    ),
                    "predicted_assembly_amplification": float(
                        predicted_global[item] / predicted_broken[item].clamp_min(1.0e-12)
                    ),
                })
        processed += batch_size

    if not variant_rows or not theorem_rows:
        raise ValueError("evaluation selected no cases")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("variant_metrics.csv", variant_rows),
                       ("theorem_metrics.csv", theorem_rows)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metric_keys = ("relative_l2", "energy_relative_error", "residual_rms",
                   "broken_relative_l2", "broken_rms", "global_rms",
                   "assembly_amplification")
    theorem_keys = tuple(key for key in theorem_rows[0]
                         if key not in ("case_index", "offset_y", "offset_x"))
    result = {
        "contract": "darcy_one_pass_theorem_decomposition_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "data": str(args.data.resolve()), "split": args.split,
        "cases": processed, "offsets": offsets,
        "inference_operator_free": True,
        "oracle_used_for_evaluation_only": True,
        "oracle_chart": "constant_plus_axis_linear",
        "nonlinear_tangent_test_applicable": False,
        "nonlinear_tangent_reason": "the present decoder is affine in its fixed compatibility code",
        "variant_summary": {
            variant: _mean_rows([row for row in variant_rows if row["variant"] == variant], metric_keys)
            for variant in VARIANTS
        },
        "theorem_summary": _mean_rows(theorem_rows, theorem_keys),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    _plot(variant_rows, theorem_rows, args.output_dir)
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
