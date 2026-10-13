"""Visualize the locked 128^2 Darcy input-correction comparison.

The case is selected before rendering as the validation field whose LatentDDM
error is closest to the median over all 128 cases.  Every local method uses the
same normalized-overlap Local/Core field; Init-SNI is shown at the first sweep
whose aggregate mean crosses LatentDDM (20 sweeps).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codiffuse_mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import torch

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import H5DarcyDataset
from .evaluate import move_batch, relative_l2
from .evaluate_input_correction_latentddm import sha256
from .input_correction_latentddm import (
    DarcyInputCorrectionLatentDDM,
    solve_corrected_boundary_input,
)
from .train_input_correction_latentddm import prepared_context
from .train_residual_port_chart import load_frozen_local
from .validate_face_sni import face_sni_step
from .visualize_five_method_comparison import global_prediction


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "codiffuse_Schwarz_Darcy/data/darcy_domain_extension_128_v1.h5",
    )
    parser.add_argument(
        "--local-checkpoint", type=Path,
        default=ROOT / "codiffuse_Schwarz_Darcy/outputs/sni_face_local_seed20260806/checkpoint.pt",
    )
    parser.add_argument(
        "--composer-checkpoint", type=Path,
        default=ROOT / (
            "codiffuse_Schwarz_Darcy/outputs/input_correction_domain_extension_v1/"
            "budget64/seed20260811/checkpoint_best.pt"
        ),
    )
    parser.add_argument(
        "--matched-evaluation", type=Path,
        default=ROOT / (
            "codiffuse_Schwarz_Darcy/outputs/input_correction_domain_extension_v1/"
            "matched_normalized_overlap_sni_128_fine.json"
        ),
    )
    base = ROOT / "codiffuse_Schwarz_Darcy/outputs/domain_extension_128_v1"
    parser.add_argument(
        "--point-checkpoint", type=Path,
        default=base / "point_full_reuse_v1/budget64/point_seed20260811/checkpoint.pt",
    )
    parser.add_argument(
        "--slice-checkpoint", type=Path,
        default=base / "adaptation_scaling_v1/budget64/transolver_seed20260811/checkpoint.pt",
    )
    parser.add_argument(
        "--patch-checkpoint", type=Path,
        default=base / (
            "patch_core32_capacitymatched_v1/budget64/"
            "patch_seed20260811/checkpoint.pt"
        ),
    )
    parser.add_argument("--sni-sweeps", type=int, default=20)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / (
            "codiffuse_Schwarz/figures/unified_report/"
            "darcy_input_correction_seven_method.pdf"
        ),
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def selected_case(payload: dict) -> tuple[int, dict]:
    rows = [row for row in payload["rows"] if row["method"] == "latentddm"]
    if len(rows) != 128:
        raise ValueError("median visualization requires all 128 validation rows")
    median = float(np.median([row["relative_l2"] for row in rows]))
    row = min(
        rows,
        key=lambda item: (abs(float(item["relative_l2"]) - median),
                          int(item["case_index"])),
    )
    return int(row["case_index"]), {
        "selection_rule": "LatentDDM relative-L2 closest to validation median",
        "median": median,
        "selected_row": row,
        "visual_appearance_used": False,
    }


def dataset_case(dataset: H5DarcyDataset, case_index: int, device) -> dict:
    for index in range(len(dataset)):
        item = dataset[index]
        if int(item["case_index"]) == case_index:
            return move_batch({
                name: value[None] if isinstance(value, torch.Tensor) else value
                for name, value in item.items()
            }, device)
    raise KeyError(f"validation case_index={case_index} was not found")


@torch.inference_mode()
def predict(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict]:
    if args.sni_sweeps <= 0:
        raise ValueError("the displayed SNI operating point must be positive")
    matched = json.loads(args.matched_evaluation.read_text(encoding="utf-8"))
    if (
        matched.get("contract") != "darcy_input_correction_matched_validation_v1"
        or matched.get("assembly") != "normalized_overlap"
        or matched.get("zero_message_local_core_max_abs") != 0.0
        or f"init_sni_{args.sni_sweeps}" not in matched["summary"]
    ):
        raise ValueError("the matched Darcy evaluation is incompatible")
    case_index, selection = selected_case(matched)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda",
    )
    local_operator, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    saved = torch.load(
        args.composer_checkpoint, map_location="cpu", weights_only=False,
    )
    if (
        saved.get("contract") != DarcyInputCorrectionLatentDDM.contract
        or saved.get("local_checkpoint_sha256") != sha256(args.local_checkpoint)
    ):
        raise ValueError("Darcy composer lineage does not match the local BVP")
    config = saved["configuration"]
    model = DarcyInputCorrectionLatentDDM(
        29, 4, token_width=int(config["token_width"]),
        heads=int(config["heads"]),
        communication_depth=int(config["communication_depth"]),
        trace_modes=int(config.get("trace_modes", 2)),
        spatial_token_bins=int(config.get("spatial_token_bins", 0)),
    ).to(device)
    model.load_state_dict(saved["composer"], strict=True)
    model.eval().requires_grad_(False)

    dataset = H5DarcyDataset(args.data, "validation")
    try:
        batch = dataset_case(dataset, case_index, device)
    finally:
        dataset.close()
    cover = CartesianCover(
        batch["solution"].shape[-2], batch["solution"].shape[-1],
        int(config["core"]), int(config["halo"]),
    )
    local, context = prepared_context(
        local_operator, adapter, batch, cover, include_target=False,
        trace_modes=int(config.get("trace_modes", 2)),
    )
    scale = float(adapter.statistics.solution_std)
    local_field = cover.assemble_weighted(local, 1) * scale * batch["mask"]
    latent = model(
        local, context.physical, context.scalar,
        context.domain_mask * context.valid, cover.grid_shape,
    )
    final_local, _ = solve_corrected_boundary_input(
        local_operator, adapter, context, latent.correction_code,
    )
    latent_field = cover.assemble_weighted(final_local, 1) * scale * batch["mask"]
    sni = local_field
    for _ in range(args.sni_sweeps):
        sni = face_sni_step(
            local_operator, adapter, batch, sni, cover,
            "weighted_pou", 1.0,
        )
    point, _ = global_prediction(args.point_checkpoint, batch, device)
    sliced, _ = global_prediction(args.slice_checkpoint, batch, device)
    patch, patch_saved = global_prediction(args.patch_checkpoint, batch, device)
    if tuple(patch_saved.get("model_spec", {}).get("patch_size", ())) != (32, 32):
        raise ValueError("Darcy visualization Patch is not owned-core matched")
    tensors = {
        "Ground truth": batch["solution"],
        "Local/Core": local_field,
        "LatentDDM": latent_field,
        f"Init-SNI ({args.sni_sweeps})": sni,
        "Point": point,
        "Slice": sliced,
        "Patch": patch,
    }
    mask = batch["mask"]
    arrays = {
        name: value[0, 0].detach().cpu().numpy()
        for name, value in tensors.items()
    }
    metadata = {
        "contract": "darcy_input_correction_seven_method_figure_v1",
        "split": "validation",
        "protected_test_cases_opened": 0,
        "case_index": case_index,
        "selection": selection,
        "assembly": "normalized_overlap",
        "tile_count": cover.tile_count,
        "sni_sweeps": args.sni_sweeps,
        "patch_size": [32, 32],
        "metrics": {
            name: float(relative_l2(value, batch["solution"], mask)[0])
            for name, value in tensors.items() if name != "Ground truth"
        },
        "sources": {
            name: str(path.resolve()) for name, path in {
                "data": args.data,
                "local": args.local_checkpoint,
                "composer": args.composer_checkpoint,
                "matched_evaluation": args.matched_evaluation,
                "point": args.point_checkpoint,
                "slice": args.slice_checkpoint,
                "patch": args.patch_checkpoint,
            }.items()
        },
        "source_sha256": {
            name: sha256(path) for name, path in {
                "local": args.local_checkpoint,
                "composer": args.composer_checkpoint,
                "matched_evaluation": args.matched_evaluation,
                "point": args.point_checkpoint,
                "slice": args.slice_checkpoint,
                "patch": args.patch_checkpoint,
            }.items()
        },
        "logk": batch["logk"][0, 0].detach().cpu().numpy(),
        "mask": mask[0, 0].detach().cpu().numpy(),
    }
    return arrays, metadata


def render(arrays: dict[str, np.ndarray], metadata: dict, output: Path) -> None:
    labels = tuple(arrays)
    truth = arrays["Ground truth"]
    mask = metadata.pop("mask")
    logk = metadata.pop("logk")
    valid = mask > 0.5
    vmin, vmax = np.quantile(truth[valid], (.01, .99))
    signed_errors = {
        name: arrays[name] - truth for name in labels[1:]
    }
    residual_values = np.concatenate([
        np.abs(error[valid]) for error in signed_errors.values()
    ])
    residual_limit = max(float(np.quantile(residual_values, .99)), 1.0e-4)
    residual_linthresh = max(.01 * residual_limit, 1.0e-5)
    error_norm = SymLogNorm(
        linthresh=residual_linthresh, linscale=1.0,
        vmin=-residual_limit, vmax=residual_limit, base=10,
    )
    metadata["residual_visualization"] = {
        "quantity": "signed prediction error on one shared symmetric-log scale",
        "shared_symmetric_limit": residual_limit,
        "linear_threshold": residual_linthresh,
        "interpolation": "nearest",
    }

    plt.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42})
    figure, axes = plt.subplots(2, len(labels), figsize=(8.15, 2.78))
    figure.subplots_adjust(
        left=.022, right=.965, bottom=.055, top=.77,
        wspace=.065, hspace=.23,
    )
    best_method = min(metadata["metrics"], key=metadata["metrics"].get)
    field_image = None
    error_image = None
    for column, name in enumerate(labels):
        field = np.ma.masked_where(~valid, arrays[name])
        field_image = axes[0, column].imshow(
            field, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
            interpolation="bilinear", rasterized=True,
        )
        title = name
        if name != "Ground truth":
            value = f"{metadata['metrics'][name]:.4f}"
            if name == best_method:
                value = rf"\mathbf{{{value}}}"
            title += "\n" + rf"$e_u={value}$"
        axes[0, column].set_title(
            title, fontsize=7.25, pad=2.0,
        )
        if column == 0:
            axes[1, column].imshow(
                np.ma.masked_where(~valid, logk), origin="lower", cmap="magma",
                interpolation="nearest", rasterized=True,
            )
            axes[1, column].set_title(
                r"Input $\log_{10}\kappa$", fontsize=7.25,
            )
        else:
            error = np.ma.masked_where(~valid, signed_errors[name])
            error_image = axes[1, column].imshow(
                error, origin="lower", cmap="coolwarm", norm=error_norm,
                interpolation="nearest", rasterized=True,
            )
            axes[1, column].set_title("Signed error", fontsize=7.25)
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    assert field_image is not None and error_image is not None
    figure.colorbar(
        field_image, ax=axes[0].tolist(), shrink=.76, pad=.006, label="$u$",
    )
    error_bar = figure.colorbar(
        error_image, ax=axes[1, 1:].tolist(), shrink=.76, pad=.006,
        label=r"$\widehat u-u$ (symmetric log scale)",
    )
    exponent_min = int(np.ceil(np.log10(residual_linthresh)))
    exponent_max = int(np.floor(np.log10(residual_limit)))
    positive_ticks = 10.0 ** np.arange(exponent_min, exponent_max + 1)
    largest_tick = positive_ticks[-1]
    residual_ticks = np.asarray((-largest_tick, 0.0, largest_tick))
    error_bar.set_ticks(residual_ticks)
    error_bar.set_ticklabels([f"{value:g}" for value in residual_ticks])

    figure.canvas.draw()
    for group, first, last, color in (
        ("LOCAL / DECOMPOSED", 1, 3, "#B44B4B"),
        ("GLOBAL TOKEN MODELS", 4, 6, "#3568A8"),
    ):
        left = axes[0, first].get_position().x0
        right = axes[0, last].get_position().x1
        y = max(
            axes[0, column].get_position().y1
            for column in range(first, last + 1)
        ) + .175
        figure.text(
            (left + right) / 2, y, group,
            ha="center", va="bottom", fontsize=9.1,
            fontweight="bold", color=color,
        )
        figure.add_artist(plt.Line2D(
            (left, right), (y - .012, y - .012),
            transform=figure.transFigure, color=color, lw=1.25,
        ))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    arrays, metadata = predict(args)
    render(arrays, metadata, args.output)
    print(json.dumps({
        "output": str(args.output),
        "case_index": metadata["case_index"],
        "metrics": metadata["metrics"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
