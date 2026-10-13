"""Aligned source-retention and target-adaptation fields for the Darcy paper.

Both representative validation cases are selected by the same scalar rule:
minimum distance to the median LatentDDM field error.  The target-adapted
checkpoints are evaluated unchanged on the 3x3 source and 4x4 target covers.
No rendered appearance or protected test field enters selection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codiffuse_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import torch

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import H5DarcyDataset
from .benchmark_field_token_decomposed_methods import target_free_context
from .evaluate import relative_l2
from .validate_face_sni import face_sni_step
from .visualize_five_method_comparison import (
    COMPOSER,
    DATA as TARGET_DATA,
    OUTPUT as LEGACY_OUTPUT,
    PATCH,
    POINT,
    SELECTION as TARGET_SELECTION,
    SLICE,
    SNI_ARTIFACT,
    global_prediction,
    load_composer,
    one_case,
    select_representative_case,
    selected_sni_sweeps,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "codiffuse_Schwarz_Darcy/data/darcy_full_domain_face_96_v2.h5"
SOURCE_SELECTION = ROOT / (
    "codiffuse_Schwarz_Darcy/outputs/field_token_domain_extension_degree1_v1/"
    "depth4/budget64/figure3_source_median_latent_selection_seed20260811.json"
)
OUTPUT = LEGACY_OUTPUT.with_name("darcy_source_target_adaptation.pdf")

Tensor = torch.Tensor
METHODS = ("LatentDDM", "SNI", "Point", "Slice", "Patch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=SOURCE_DATA)
    parser.add_argument("--target-data", type=Path, default=TARGET_DATA)
    parser.add_argument("--composer-checkpoint", type=Path, default=COMPOSER)
    parser.add_argument("--point-checkpoint", type=Path, default=POINT)
    parser.add_argument("--slice-checkpoint", type=Path, default=SLICE)
    parser.add_argument("--patch-checkpoint", type=Path, default=PATCH)
    parser.add_argument("--sni-artifact", type=Path, default=SNI_ARTIFACT)
    parser.add_argument("--source-selection", type=Path, default=SOURCE_SELECTION)
    parser.add_argument("--target-selection", type=Path, default=TARGET_SELECTION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--sni-tau", type=float, default=1.0)
    parser.add_argument("--source-selection-cases", type=int, default=250)
    parser.add_argument("--target-selection-cases", type=int, default=128)
    parser.add_argument("--selection-batch-size", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def selection_namespace(
    args: argparse.Namespace, data: Path, selection: Path, count: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        composer_checkpoint=args.composer_checkpoint,
        local_checkpoint=None,
        chart_checkpoint=None,
        selection_output=selection,
        selection_cases=count,
        selection_batch_size=args.selection_batch_size,
        overwrite_selection=False,
        core=args.core,
        halo=args.halo,
    )


@torch.inference_mode()
def predict_domain(
    data: Path,
    dataset_index: int,
    *,
    local: torch.nn.Module,
    adapter: Any,
    chart: torch.nn.Module,
    spec: Any,
    composer: torch.nn.Module,
    point_checkpoint: Path,
    slice_checkpoint: Path,
    patch_checkpoint: Path,
    sni_sweeps: int,
    sni_tau: float,
    core: int,
    halo: int,
    device: torch.device,
) -> dict[str, Any]:
    dataset = H5DarcyDataset(data, "validation")
    try:
        with_target = one_case(dataset, dataset_index, device)
    finally:
        dataset.close()
    target = with_target["solution"]
    batch = {key: value for key, value in with_target.items() if key != "solution"}
    cover = CartesianCover(
        target.shape[-2], target.shape[-1], core, halo,
    )
    context = target_free_context(local, adapter, batch, cover, spec.code_width)
    local_field = (
        cover.assemble_ras(context.local, 1)
        * batch["mask"] * adapter.statistics.solution_std
    )
    latent = composer(chart, context, batch["mask"]).assembled
    latent = latent * adapter.statistics.solution_std
    sni = local_field.clone()
    for _ in range(sni_sweeps):
        sni = face_sni_step(
            local, adapter, batch, sni, cover, "core_ras", sni_tau,
        )
    point, point_saved = global_prediction(point_checkpoint, batch, device)
    sliced, slice_saved = global_prediction(slice_checkpoint, batch, device)
    patch, patch_saved = global_prediction(patch_checkpoint, batch, device)
    if tuple(patch_saved["model_spec"].get("patch_size", ())) != (core, core):
        raise ValueError("Patch token size does not match the local owned core")
    tensors = {
        "Ground truth": target,
        "LatentDDM": latent,
        "SNI": sni,
        "Point": point,
        "Slice": sliced,
        "Patch": patch,
    }
    mask = batch["mask"]
    return {
        "fields": {
            name: value[0, 0].detach().cpu().numpy()
            for name, value in tensors.items()
        },
        "relative_l2": {
            name: float(relative_l2(value, target, mask)[0])
            for name, value in tensors.items() if name != "Ground truth"
        },
        "logk": batch["logk"][0, 0].detach().cpu().numpy(),
        "mask": mask[0, 0].detach().cpu().numpy(),
        "case_index": int(with_target["case_index"][0]),
        "dataset_index": int(dataset_index),
        "cover": {
            "grid_shape_yx": list(cover.grid_shape),
            "subdomains": int(cover.tile_count),
            "core": int(core),
            "halo_context": int(halo),
        },
        "global_model_specs": {
            "Point": point_saved["model_spec"],
            "Slice": slice_saved["model_spec"],
            "Patch": patch_saved["model_spec"],
        },
    }


def plot(domains: dict[str, dict[str, Any]], metadata: dict[str, Any]) -> plt.Figure:
    names = ("Ground truth",) + METHODS
    target_values = np.concatenate([
        domain["fields"]["Ground truth"][domain["mask"] > 0.5]
        for domain in domains.values()
    ])
    field_min, field_max = np.quantile(target_values, (0.01, 0.99))
    errors: dict[tuple[str, str], np.ndarray] = {}
    absolute_errors = []
    for domain_name, domain in domains.items():
        mask = domain["mask"] > 0.5
        target = domain["fields"]["Ground truth"]
        for method in METHODS:
            error = domain["fields"][method] - target
            errors[(domain_name, method)] = error
            absolute_errors.append(np.abs(error[mask]))
    error_limit = max(float(np.quantile(np.concatenate(absolute_errors), .99)), 1e-4)
    linear_threshold = max(.01 * error_limit, 1e-5)
    error_norm = SymLogNorm(
        linthresh=linear_threshold, linscale=1.0,
        vmin=-error_limit, vmax=error_limit, base=10,
    )
    metadata["visual_scales"] = {
        "solution_quantile_range": [float(field_min), float(field_max)],
        "signed_error_symmetric_limit": error_limit,
        "signed_error_linear_threshold": linear_threshold,
    }

    figure, axes = plt.subplots(4, len(names), figsize=(8.05, 5.0))
    figure.subplots_adjust(
        left=.079, right=.885, bottom=.04, top=.90,
        wspace=.045, hspace=.12,
    )
    field_image = error_image = None
    row = 0
    for domain_name, domain in domains.items():
        mask = domain["mask"] > .5
        best = min(domain["relative_l2"], key=domain["relative_l2"].get)
        for column, name in enumerate(names):
            field_image = axes[row, column].imshow(
                np.where(mask, domain["fields"][name], np.nan),
                origin="lower", cmap="viridis", vmin=field_min, vmax=field_max,
                interpolation="bilinear", rasterized=True,
            )
            if name != "Ground truth":
                weight = "bold" if name == best else "normal"
                axes[row, column].text(
                    .04, .94, rf"$e_u={domain['relative_l2'][name]:.3f}$",
                    transform=axes[row, column].transAxes,
                    ha="left", va="top", fontsize=5.7, fontweight=weight,
                    bbox={"facecolor": "white", "alpha": .74, "pad": .5,
                          "edgecolor": "none"},
                )
            if row == 0:
                axes[row, column].set_title(name, fontsize=7.2, pad=2.5)
            if name == "Ground truth":
                axes[row + 1, column].imshow(
                    np.where(mask, domain["logk"], np.nan),
                    origin="lower", cmap="magma", interpolation="nearest",
                    rasterized=True,
                )
            else:
                error_image = axes[row + 1, column].imshow(
                    np.where(mask, errors[(domain_name, name)], np.nan),
                    origin="lower", cmap="coolwarm", norm=error_norm,
                    interpolation="nearest", rasterized=True,
                )
        row += 2
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    row_labels = (
        r"Source $3\!\times\!3$ field", r"Source input / error",
        r"Target $4\!\times\!4$ field", r"Target input / error",
    )
    for index, label in enumerate(row_labels):
        axes[index, 0].set_ylabel(label, fontsize=6.6, labelpad=3.5)
    assert field_image is not None and error_image is not None
    field_axis = figure.add_axes((.905, .555, .012, .29))
    error_axis = figure.add_axes((.905, .105, .012, .29))
    field_bar = figure.colorbar(field_image, cax=field_axis)
    field_bar.set_label("$u$", fontsize=7)
    field_bar.ax.tick_params(labelsize=6, length=2)
    error_bar = figure.colorbar(error_image, cax=error_axis)
    error_bar.set_label(r"$\widehat u-u$ (symmetric log)", fontsize=7)
    error_bar.ax.tick_params(labelsize=6, length=2)
    figure.canvas.draw()
    for label, first, last, color in (
        ("LOCAL", 1, 2, "#B44B4B"),
        ("GLOBAL", 3, 5, "#3568A8"),
    ):
        left = axes[0, first].get_position().x0
        right = axes[0, last].get_position().x1
        y = axes[0, first].get_position().y1 + .075
        figure.text(
            (left + right) / 2, y, label, ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color=color,
        )
        figure.add_artist(plt.Line2D(
            (left, right), (y - .008, y - .008),
            transform=figure.transFigure, color=color, lw=1.0,
        ))
    return figure


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    local, adapter, chart, spec, composer, saved, local_path, chart_path = (
        load_composer(args.composer_checkpoint, None, None, device)
    )
    source_ns = selection_namespace(
        args, args.source_data, args.source_selection,
        args.source_selection_cases,
    )
    target_ns = selection_namespace(
        args, args.target_data, args.target_selection,
        args.target_selection_cases,
    )
    source_index, source_selection = select_representative_case(
        source_ns, local, adapter, chart, spec, composer,
        local_path, chart_path, device,
    )
    target_index, target_selection = select_representative_case(
        target_ns, local, adapter, chart, spec, composer,
        local_path, chart_path, device,
    )
    target_sni_ns = SimpleNamespace(
        sni_artifact=args.sni_artifact,
        data=args.target_data,
        sni_sweeps=None,
        sni_tau=args.sni_tau,
    )
    sni_sweeps, sni_metadata = selected_sni_sweeps(
        target_sni_ns, local_path,
    )
    common = dict(
        local=local, adapter=adapter, chart=chart, spec=spec,
        composer=composer, point_checkpoint=args.point_checkpoint,
        slice_checkpoint=args.slice_checkpoint,
        patch_checkpoint=args.patch_checkpoint, sni_sweeps=sni_sweeps,
        sni_tau=args.sni_tau, core=args.core, halo=args.halo, device=device,
    )
    domains = {
        "source": predict_domain(args.source_data, source_index, **common),
        "target": predict_domain(args.target_data, target_index, **common),
    }
    metadata: dict[str, Any] = {
        "contract": "darcy_source_target_five_method_visual_v1",
        "protected_test_cases_opened": 0,
        "method_order": list(METHODS),
        "source": {
            **{key: value for key, value in domains["source"].items()
               if key not in ("fields", "logk", "mask")},
            "selection_rule": source_selection["selection_rule"],
            "selection_path": str(args.source_selection.resolve()),
        },
        "target": {
            **{key: value for key, value in domains["target"].items()
               if key not in ("fields", "logk", "mask")},
            "selection_rule": target_selection["selection_rule"],
            "selection_path": str(args.target_selection.resolve()),
        },
        "sni": {
            "sweeps": sni_sweeps, "tau": args.sni_tau, **sni_metadata,
        },
        "communication_depth": int(
            saved["composer_spec"]["communication_depth"]
        ),
        "assembly": "disjoint_owned_core",
        "prediction_target_read": False,
    }
    figure = plot(domains, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(
        args.output.with_suffix(".png"), dpi=220,
        bbox_inches="tight", facecolor="white",
    )
    plt.close(figure)
    metadata["output_pdf"] = str(args.output.resolve())
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
