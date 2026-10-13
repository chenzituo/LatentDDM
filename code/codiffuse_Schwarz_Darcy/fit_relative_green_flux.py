"""Analytically fit a signed relative Green flux correction on train data.

For each receiver tile, the eight flux compatibility coordinates of every
source tile are placed in one of 25 receiver-minus-source displacement slots.
The resulting 200-dimensional feature is mapped to the difference between the
train-only value+flux Schur teacher code and the immutable accepted value-only
code.  A 200x4 ridge solve therefore fits all 800 signed Green coefficients;
no absolute tile identity, target field, or discrete PDE operator enters the
fit.  Ridge is selected on a deterministic train-corpus holdout (case mod 10).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .adapter import H5DarcyDataset
from .evaluate import move_batch
from .export_value_flux_teacher import CONTRACT as TEACHER_CONTRACT
from .latent_schur_teacher import inference_only_context
from .residual_port_composer import (
    RelativeGreenFluxResidualPortCommunicator,
    SourceLinearResidualPortCommunicator,
)
from .train import seed_everything
from .train_residual_port_chart import frozen_context, load_frozen_local, sha256
from .train_residual_port_communicator import (
    CANONICAL_ANALYZER_CONTRACT, CONTRACT, load_frozen_chart,
    source_linear_signal_width,
)
from .validate_latent_schur import measured_q


Tensor = torch.Tensor
RUN_CONTRACT = "darcy_relative_green_flux_ridge_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-cases", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument(
        "--ridge", type=float, nargs="+",
        default=(0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def relative_design(flux: Tensor, grid_shape: tuple[int, int]) -> Tensor:
    """Place q_flux_j in receiver/source displacement slots for every i."""
    if flux.ndim != 3 or flux.shape[-1] != 8 or grid_shape != (3, 3):
        raise ValueError("relative Green fitter requires B,9,8 flux on a 3x3 cover")
    axes = [torch.arange(size, device=flux.device) for size in grid_shape]
    coordinates = torch.stack(
        torch.meshgrid(*axes, indexing="ij"), -1,
    ).reshape(-1, 2)
    delta = coordinates[:, None] - coordinates[None, :]
    slot = (delta[..., 0] + 2) * 5 + delta[..., 1] + 2
    design = flux.new_zeros(flux.shape[0], 9, 25, 8)
    for receiver in range(9):
        design[:, receiver, slot[receiver]] = flux
    return design.flatten(2)


def _load_base(
    path: Path, local_path: Path, chart_path: Path, spec, device: torch.device,
) -> tuple[SourceLinearResidualPortCommunicator, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError("base checkpoint is not a one-message communicator")
    if checkpoint.get("source_local_sha256") != sha256(local_path) or \
            checkpoint.get("source_chart_sha256") != sha256(chart_path):
        raise ValueError("base communicator lineage differs")
    cfg = checkpoint["communicator_spec"]
    if cfg.get("analyzer_mode") != "source_linear" or \
            cfg.get("canonical_analyzer_contract") != CANONICAL_ANALYZER_CONTRACT:
        raise ValueError("base checkpoint is not the accepted homogeneous value map")
    width = source_linear_signal_width(spec)
    base = SourceLinearResidualPortCommunicator(
        int(cfg["physical_channels"]), int(cfg["scalar_dim"]), spec.code_width,
        width, token_width=int(cfg["token_width"]), heads=int(cfg["heads"]),
    ).to(device)
    base.load_state_dict(checkpoint["communicator"], strict=True)
    base.eval()
    return base, checkpoint


def _normal_equations(width: int, outputs: int) -> dict[str, Tensor | int]:
    return {
        "xtx": torch.zeros(width, width, dtype=torch.float64),
        "xty": torch.zeros(width, outputs, dtype=torch.float64),
        "yty": torch.zeros((), dtype=torch.float64),
        "rows": 0,
    }


def _accumulate(stats: dict[str, Tensor | int], design: Tensor, target: Tensor) -> None:
    x = design.reshape(-1, design.shape[-1]).double().cpu()
    y = target.reshape(-1, target.shape[-1]).double().cpu()
    stats["xtx"].add_(x.T @ x)
    stats["xty"].add_(x.T @ y)
    stats["yty"].add_(y.square().sum())
    stats["rows"] += x.shape[0]


def _sse(stats: dict[str, Tensor | int], weight: Tensor) -> float:
    value = stats["yty"] - 2.0 * (weight * stats["xty"]).sum() \
        + (weight * (stats["xtx"] @ weight)).sum()
    return float(value.clamp_min(0.0))


@torch.no_grad()
def fit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite relative Green fit: {args.output}")
    if args.maximum_cases <= 1 or any(value < 0 for value in args.ridge):
        raise ValueError("fit extent/ridge candidates are invalid")
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, chart_checkpoint = load_frozen_chart(
        args.chart_checkpoint, device,
    )
    del chart
    if spec.code_width != 4:
        raise ValueError("relative Green fit is predeclared for degree-0 four-code chart")
    if chart_checkpoint["source_local_sha256"] != sha256(args.local_checkpoint):
        raise ValueError("chart/local lineage differs")
    base, base_checkpoint = _load_base(
        args.base_checkpoint, args.local_checkpoint, args.chart_checkpoint,
        spec, device,
    )
    with h5py.File(args.teacher, "r") as handle:
        if handle.attrs.get("contract") != TEACHER_CONTRACT or \
                handle.attrs.get("source_split") != "train" or \
                bool(handle.attrs.get("validation_or_test_opened", True)) or \
                bool(handle.attrs.get("target_or_pde_labels_used", True)):
            raise ValueError("teacher is not the approved train-only target-free corpus")
        if handle.attrs.get("source_data_sha256") != sha256(args.data) or \
                handle.attrs.get("local_checkpoint_sha256") != sha256(
                    args.local_checkpoint
                ) or handle.attrs.get("chart_checkpoint_sha256") != sha256(
                    args.chart_checkpoint
                ):
            raise ValueError("teacher lineage differs")
        indices = np.asarray(handle["train/case_index"], dtype=np.int64)
        teacher = np.asarray(handle["train/teacher_code"], dtype=np.float32)
    count = min(args.maximum_cases, len(indices))
    indices, teacher = indices[:count], teacher[:count]
    if len(np.unique(indices)) != count:
        raise ValueError("teacher case indices are not unique")
    lookup = {int(index): row for row, index in enumerate(indices)}
    dataset = H5DarcyDataset(args.data, "train")
    subset = Subset(dataset, [int(index) for index in indices])
    loader = DataLoader(
        subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    train_stats = _normal_equations(200, spec.code_width)
    holdout_stats = _normal_equations(200, spec.code_width)
    invariance_max_abs = None
    prefix_width = source_linear_signal_width(spec)
    processed = 0
    for host in loader:
        batch = move_batch(host, device)
        context = inference_only_context(frozen_context(
            local, adapter, batch, spec, args.core, args.halo,
        ))
        q = measured_q(context, batch, spec, scale, "value_flux")[0]
        base_code, _ = base(
            context.local, context.physical, context.scalar, context.valid,
            context.cover.grid_shape, aligned_features=q[..., :prefix_width],
        )
        rows = [lookup[int(value)] for value in batch["case_index"].cpu().numpy()]
        target = torch.from_numpy(teacher[rows]).to(device) - base_code
        design = relative_design(q[..., prefix_width:], context.cover.grid_shape)
        case_indices = batch["case_index"].reshape(-1)
        holdout = case_indices.remainder(10).eq(0)
        if holdout.any():
            _accumulate(holdout_stats, design[holdout], target[holdout])
        if (~holdout).any():
            _accumulate(train_stats, design[~holdout], target[~holdout])
        if invariance_max_abs is None:
            changed = dict(batch)
            changed["solution"] = torch.full_like(batch["solution"], 37.0)
            changed_context = inference_only_context(frozen_context(
                local, adapter, changed, spec, args.core, args.halo,
            ))
            changed_q = measured_q(
                changed_context, changed, spec, scale, "value_flux",
            )[0]
            invariance_max_abs = float((changed_q - q).abs().max())
            if invariance_max_abs > 1e-6:
                raise RuntimeError("relative Green design depends on replaced targets")
        processed += batch["solution"].shape[0]
        if processed % 256 == 0 or processed == count:
            print(json.dumps({"processed": processed, "cases": count}), flush=True)
    dataset.close()
    if train_stats["rows"] == 0 or holdout_stats["rows"] == 0:
        raise RuntimeError("empty relative Green fit/holdout partition")

    xtx, xty = train_stats["xtx"], train_stats["xty"]
    ridge_scale = float(torch.trace(xtx) / xtx.shape[0])
    identity = torch.eye(xtx.shape[0], dtype=xtx.dtype)
    candidates = []
    best_weight = None; best_error = float("inf"); best_ridge = None
    for ridge in args.ridge:
        matrix = xtx + float(ridge) * ridge_scale * identity
        try:
            weight = torch.linalg.solve(matrix, xty)
        except torch.linalg.LinAlgError:
            continue
        error = _sse(holdout_stats, weight) / int(holdout_stats["rows"])
        relative = _sse(holdout_stats, weight) / max(
            float(holdout_stats["yty"]), 1e-30,
        )
        candidates.append({
            "ridge": float(ridge), "holdout_mse": error,
            "holdout_relative_square": relative,
        })
        if error < best_error:
            best_error, best_weight, best_ridge = error, weight, float(ridge)
    if best_weight is None:
        raise RuntimeError("every relative Green ridge solve failed")

    cfg = base_checkpoint["communicator_spec"]
    communicator = RelativeGreenFluxResidualPortCommunicator(
        int(cfg["physical_channels"]), int(cfg["scalar_dim"]), spec.code_width,
        prefix_width, token_width=int(cfg["token_width"]),
        heads=int(cfg["heads"]), max_offset=2,
    ).to(device)
    communicator.base.load_state_dict(base.state_dict(), strict=True)
    kernel = best_weight.reshape(5, 5, 8, spec.code_width).permute(0, 1, 3, 2)
    communicator.flux_green.kernel.copy_(kernel.to(device))
    # Algebraic implementation check against the fitted design matrix.
    check_flux = torch.randn(2, 9, 8, device=device)
    check_design = relative_design(check_flux, (3, 3))
    expected = check_design @ best_weight.to(device, dtype=check_design.dtype)
    actual = communicator.flux_green(check_flux, (3, 3))
    implementation_error = float((expected - actual).abs().max())
    if implementation_error > 2e-5:
        raise RuntimeError("relative Green tensor layout differs from ridge design")

    output_cfg = dict(cfg)
    output_cfg.update({
        "analyzer_mode": "value_flux_relative_green",
        "kernel_type": "accepted_softmax_plus_relative_green_5x5",
        "source_linear_signal_width": prefix_width + 8,
        "aligned_width": prefix_width + 8,
        "value_prefix_width": prefix_width, "flux_suffix_width": 8,
        "relative_green_max_offset": 2,
        "relative_green_parameters": 5 * 5 * 4 * 8,
        "relative_attention_count": 1,
        "one_global_communication_stage": True,
        "signed_offset_specific_flux_action": True,
        "absolute_tile_identifier": False,
        "fixed_context_q_linearity": True,
        "zero_signal_zero_increment_bitwise": True,
        "base_attention_plus_relative_convolution_disclosed": True,
        "deployment_affine_retraction": "accepted_value_to_flux_corrected",
    })
    payload = dict(base_checkpoint)
    payload.update({
        "contract": CONTRACT, "phase": RUN_CONTRACT,
        "communicator": communicator.state_dict(),
        "communicator_spec": output_cfg,
        "source_local_checkpoint": str(args.local_checkpoint.resolve()),
        "source_local_sha256": sha256(args.local_checkpoint),
        "source_chart_checkpoint": str(args.chart_checkpoint.resolve()),
        "source_chart_sha256": sha256(args.chart_checkpoint),
        "ordinary_reference_checkpoint": str(args.base_checkpoint.resolve()),
        "ordinary_reference_sha256": sha256(args.base_checkpoint),
        "teacher_corpus": str(args.teacher.resolve()),
        "teacher_corpus_sha256": sha256(args.teacher),
        "training_splits": ["train"],
        "ridge_selection_partition": "train case_index mod 10 == 0",
        "selected_relative_ridge": best_ridge,
        "ridge_scale": ridge_scale,
        "ridge_candidates": candidates,
        "target_or_pde_labels_used": False,
        "exact_global_solution_used_at_inference": False,
        "discrete_operator_used_at_inference": False,
        "test_subsets_opened": False,
        "solution_replacement_invariance_max_abs": invariance_max_abs,
        "relative_green_implementation_max_abs_error": implementation_error,
        "args": vars(args),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    result = {
        "contract": RUN_CONTRACT, "checkpoint": str(args.output.resolve()),
        "checkpoint_sha256": sha256(args.output), "cases": count,
        "fit_rows": int(train_stats["rows"]),
        "holdout_rows": int(holdout_stats["rows"]),
        "selected_ridge": best_ridge,
        "selected_holdout_mse": best_error,
        "selected_holdout_relative_square": next(
            row["holdout_relative_square"] for row in candidates
            if row["ridge"] == best_ridge
        ),
        "ridge_scale": ridge_scale, "ridge_candidates": candidates,
        "solution_replacement_invariance_max_abs": invariance_max_abs,
        "relative_green_implementation_max_abs_error": implementation_error,
        "test_subsets_opened": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(fit(parse_args()), indent=2, default=str))
