"""Visualize matched Darcy solution fields across one-pass and iterative DDMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from codiffuse_Schwarz.learned_composer import canonical_modes
from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import (
    DarcyAdapter, DarcyStatistics, H5DarcyDataset, MODEL_SPEC,
    batch_to_transition, build_model, enforce_domain_boundary,
    tiled_darcy_conditions,
)
from .evaluate import composition_step, energy_relative_error, relative_l2, residual_rms
from .evaluate_theorem_decomposition import decode_modes, oracle_mode_coefficients
from .train_operator_composer import CONTRACT, OperatorOnlyDarcyComposer
from .two_level import galerkin_correct, tile_pou_basis


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("A", "B", "C", "D"), default="A")
    parser.add_argument("--case-index", type=int, default=55)
    parser.add_argument("--offset", default="8,8")
    parser.add_argument("--sweeps", type=int, default=64)
    parser.add_argument("--sni-relaxation", type=float, default=.1)
    parser.add_argument(
        "--include-sni", action="store_true",
        help="disabled: iterative fields require a separately audited face-local checkpoint/data contract",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _to_batch(item: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value[None].to(device) if isinstance(value, Tensor) else value
        for key, value in item.items()
    }


def _metrics(field: Tensor, batch: dict[str, object]) -> dict[str, float]:
    return {
        "relative_l2": float(relative_l2(field, batch["solution"], batch["mask"])[0]),  # type: ignore[index]
        "energy_relative_error": float(energy_relative_error(
            field, batch["solution"], batch["logk"], batch["mask"],  # type: ignore[arg-type]
        )[0]),
        "residual_rms": float(residual_rms(field, batch)[0]),
    }


@torch.no_grad()
def _iterate(
    local_model: torch.nn.Module, adapter: DarcyAdapter,
    batch: dict[str, object], cover: CartesianCover,
    method: str, sweeps: int, relaxation: float,
) -> tuple[Tensor, list[dict[str, float]]]:
    current = torch.zeros_like(batch["solution"])  # type: ignore[arg-type]
    previous = current.clone()
    curve = []
    for sweep in range(1, sweeps + 1):
        candidate, _ = composition_step(
            local_model, adapter, batch, current, previous, cover,
            method, 1.0,
        )
        next_state = enforce_domain_boundary(
            batch, current + relaxation * (candidate - current),  # type: ignore[arg-type]
        )
        curve.append({"sweep": sweep, "local_nfe": sweep * cover.tile_count,
                      **_metrics(next_state, batch)})
        previous, current = current, next_state
    return current, curve


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError("checkpoint is not the operator-only composer")
    if args.include_sni:
        raise ValueError(
            "iterative SNI is intentionally disabled in this legacy composer "
            "visualizer: its embedded local is not the frozen face-local BVP "
            "solver. Use the separately gated Huang/POU validation artifacts; "
            "Robin SNAP additionally requires a v3 Robin-local checkpoint."
        )
    statistics = DarcyStatistics.from_json(checkpoint["statistics"])
    adapter = DarcyAdapter(statistics, use_discrete_residual=False)
    model = OperatorOnlyDarcyComposer(
        build_model(adapter, communication_stage="none"), adapter,
    ).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    dataset = H5DarcyDataset(args.data, args.split)
    if not 0 <= args.case_index < len(dataset):
        raise IndexError("case index lies outside the requested split")
    batch = _to_batch(dataset[args.case_index], device)
    dataset.close()
    target: Tensor = batch["solution"]  # type: ignore[assignment]
    oy, ox = (int(piece) for piece in args.offset.split(","))
    cover = CartesianCover(target.shape[-2], target.shape[-1], MODEL_SPEC["core"],
                           MODEL_SPEC["halo"], oy, ox)
    current = torch.zeros_like(target)
    transition = batch_to_transition(batch, current, target=target)  # type: ignore[arg-type]
    physical, scalar = tiled_darcy_conditions(adapter, transition, cover)
    valid = cover.valid_tiles(1, device, target.dtype)
    corrected, local, _ = model(physical, scalar, valid, cover.grid_shape)
    normalized_target = adapter.normalized_target(transition)
    local_target = cover.tile(normalized_target)
    local_weight = cover.tile(batch["mask"].to(target.dtype)) * valid  # type: ignore[union-attr]
    modes = canonical_modes(local.shape[2:], device, local.dtype)
    oracle_coefficient = oracle_mode_coefficients(
        local.float(), local_target.float(), local_weight.float(), modes.float(),
    )
    oracle_local = local.float() + decode_modes(
        oracle_coefficient, modes.float(), valid.float(),
    )
    local_field = adapter.apply_endpoint(
        transition, cover.assemble_weighted(local.float(), 1),
    )
    learned_field = adapter.apply_endpoint(
        transition, cover.assemble_weighted(corrected.float(), 1),
    )
    oracle_field = adapter.apply_endpoint(
        transition, cover.assemble_weighted(oracle_local, 1),
    )
    basis = tile_pou_basis(cover, degree=1, device=device, dtype=target.dtype)
    galerkin_field, _ = galerkin_correct(
        local_field, batch, basis, relative_cutoff=1.0e-8,  # type: ignore[arg-type]
    )
    fields = {
        "reference": target,
        "local_pou": local_field,
        "learned_affine": learned_field,
        "oracle_affine": oracle_field,
        "galerkin_oracle": galerkin_field,
    }
    curves: dict[str, list[dict[str, float]]] = {}
    metrics = {name: _metrics(field, batch) for name, field in fields.items()
               if name != "reference"}
    labels = {
        "oracle_affine": "target-fitted evaluation oracle",
        "galerkin_oracle": "exact FV-operator Galerkin diagnostic",
    }
    result: dict[str, object] = {
        "contract": "darcy_solution_field_ddm_comparison_v1",
        "checkpoint": str(args.checkpoint.resolve()), "split": args.split,
        "case_index": args.case_index, "offset": [oy, ox],
        "tile_count": cover.tile_count, "sweeps": args.sweeps,
        "metrics": metrics, "curves": curves,
        "labels": labels,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "darcy_solution_ddm_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n",
    )

    mask = batch["mask"][0, 0].detach().cpu().numpy() > .5  # type: ignore[index]
    target_np = target[0, 0].detach().cpu().numpy()
    solution_values = target_np[mask]
    vmin, vmax = np.quantile(solution_values, (.01, .99))
    method_order = ["reference", "local_pou", "learned_affine", "oracle_affine",
                    "galerkin_oracle"]
    titles = ["Reference", "Local POU", "Learned one-pass", "Affine oracle",
              "Galerkin oracle"]
    errors = []
    arrays = {}
    for name in method_order:
        array = fields[name][0, 0].detach().cpu().numpy()
        arrays[name] = np.where(mask, array, np.nan)
        if name != "reference":
            errors.append(np.abs(array - target_np)[mask])
    # Keep the informative one-pass/SNI errors on a readable common scale.
    scale_methods = tuple(name for name in method_order if name != "reference")
    error_max = max(float(np.quantile(
        np.abs(fields[name][0, 0].detach().cpu().numpy() - target_np)[mask], .98,
    )) for name in scale_methods)
    figure, axes = plt.subplots(2, len(method_order), figsize=(13.6, 4.25),
                               constrained_layout=True)
    field_image = None; error_image = None
    for column, (name, title) in enumerate(zip(method_order, titles)):
        field_image = axes[0, column].imshow(
            arrays[name], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
        )
        suffix = ""
        if name != "reference":
            suffix = f"\n$L^2$={metrics[name]['relative_l2']:.3f}"
        axes[0, column].set_title(title + suffix, fontsize=9)
        if name == "reference":
            coefficient = batch["logk"][0, 0].detach().cpu().numpy()  # type: ignore[index]
            error_image = axes[1, column].imshow(
                np.where(mask, coefficient, np.nan), origin="lower", cmap="magma",
            )
            axes[1, column].set_title(r"Input $\log_{10}\kappa$", fontsize=9)
        else:
            error_image = axes[1, column].imshow(
                np.where(mask, np.abs(arrays[name] - target_np), np.nan),
                origin="lower", cmap="inferno", vmin=0.0, vmax=error_max,
            )
            axes[1, column].set_title("Absolute error", fontsize=9)
        for row in range(2):
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
    assert field_image is not None and error_image is not None
    figure.colorbar(field_image, ax=axes[0].tolist(), shrink=.78, pad=.012,
                    label="solution $u$")
    # Error panels share a scale; the first lower panel is explicitly an input map.
    figure.colorbar(axes[1, 1].images[0], ax=axes[1, 1:].tolist(), shrink=.78,
                    pad=.012, label=r"$|\hat u-u|$")
    for suffix in ("pdf", "png"):
        figure.savefig(args.output_dir / f"darcy_solution_ddm_comparison.{suffix}", dpi=240)
    plt.close(figure)

    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
