"""Render the final Darcy four-scale five-method extension figure.

All validation cases are chosen from input geometry statistics only.  One
predeclared target-adapted seed is evaluated unchanged at each scale; neither
prediction error nor rendered appearance participates in case selection.  The
4x4-adapted models are also replayed on the 3x3 source domain to show retention.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codiffuse_mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
DATA_ROOT = ROOT / "codiffuse_Schwarz_Darcy/data"
LOCAL = ROOT / (
    "codiffuse_Schwarz_Darcy/outputs/sni_face_local_seed20260806/checkpoint.pt"
)
EXTENSION_ROOT = ROOT / (
    "codiffuse_Schwarz_Darcy/outputs/domain_extension_fixed_dx_v1"
)
SEED = 20260811
OUTPUT = ROOT / (
    "codiffuse_Schwarz/workshop/submission/Two_case/figures/"
    "darcy_source_target_adaptation.pdf"
)

METHODS = ("LatentDDM", "SNI-16", "Point", "Slice", "Patch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--extension-root", type=Path, default=EXTENSION_ROOT)
    parser.add_argument("--scales", type=int, nargs=4, default=(3, 4, 8, 16))
    parser.add_argument("--local-checkpoint", type=Path, default=LOCAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sni-sweeps", type=int, default=16)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def representative_input_case(dataset: H5DarcyDataset) -> tuple[int, dict[str, Any]]:
    """Choose the multivariate input medoid without reading solution values."""
    rows = []
    for index in range(len(dataset)):
        item = dataset[index]
        mask = item["mask"][0].numpy() > 0.5
        logk = item["logk"][0].numpy()
        horizontal = np.logical_xor(mask[:, 1:], mask[:, :-1]).mean()
        vertical = np.logical_xor(mask[1:, :], mask[:-1, :]).mean()
        active = logk[mask]
        rows.append({
            "dataset_index": index,
            "case_index": int(item["case_index"]),
            "active_fraction": float(mask.mean()),
            "boundary_density": float(horizontal + vertical),
            "active_logk_std": float(active.std()),
        })
    features = np.asarray([
        [row["active_fraction"], row["boundary_density"], row["active_logk_std"]]
        for row in rows
    ])
    center = np.median(features, axis=0)
    scale = np.median(np.abs(features - center), axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    distance = np.linalg.norm((features - center) / scale, axis=1)
    selected = min(
        range(len(rows)), key=lambda index: (float(distance[index]), index),
    )
    metadata = {
        "rule": "input medoid in active fraction, boundary density, and log-k std",
        "solution_or_prediction_used": False,
        "feature_median": center.tolist(),
        "feature_mad": scale.tolist(),
        "selected_distance": float(distance[selected]),
        "selected_features": rows[selected],
    }
    return int(rows[selected]["dataset_index"]), metadata


def one_case(dataset: H5DarcyDataset, index: int, device: torch.device) -> dict:
    item = dataset[index]
    return move_batch({
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
        for key, value in item.items()
    }, device)


def build_composer(
    checkpoint: Path, local_checkpoint: Path, device: torch.device,
) -> tuple[DarcyInputCorrectionLatentDDM, dict[str, Any]]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        saved.get("contract") != DarcyInputCorrectionLatentDDM.contract
        or saved.get("local_checkpoint_sha256") != sha256(local_checkpoint)
    ):
        raise ValueError("composer and local checkpoint lineage differ")
    config = saved["configuration"]
    model = DarcyInputCorrectionLatentDDM(
        29, 4,
        token_width=int(config["token_width"]),
        heads=int(config["heads"]),
        communication_depth=int(config["communication_depth"]),
        trace_modes=int(config.get("trace_modes", 2)),
        spatial_token_bins=int(config.get("spatial_token_bins", 0)),
    ).to(device)
    model.load_state_dict(saved["composer"], strict=True)
    model.eval().requires_grad_(False)
    return model, saved


@torch.inference_mode()
def predict_domain(
    data: Path,
    *,
    local_operator: torch.nn.Module,
    adapter: Any,
    composer: DarcyInputCorrectionLatentDDM,
    composer_saved: dict[str, Any],
    point_checkpoint: Path,
    slice_checkpoint: Path,
    patch_checkpoint: Path,
    sni_sweeps: int,
    core: int,
    halo: int,
    device: torch.device,
) -> dict[str, Any]:
    dataset = H5DarcyDataset(data, "validation")
    try:
        index, selection = representative_input_case(dataset)
        batch = one_case(dataset, index, device)
    finally:
        dataset.close()
    config = composer_saved["configuration"]
    cover = CartesianCover(
        batch["solution"].shape[-2], batch["solution"].shape[-1], core, halo,
    )
    local, context = prepared_context(
        local_operator, adapter, batch, cover, include_target=False,
        trace_modes=int(config.get("trace_modes", 2)),
    )
    scale = float(adapter.statistics.solution_std)
    local_field = cover.assemble_weighted(local, 1) * scale * batch["mask"]
    communication = composer(
        local, context.physical, context.scalar,
        context.domain_mask * context.valid, cover.grid_shape,
    )
    corrected, _ = solve_corrected_boundary_input(
        local_operator, adapter, context, communication.correction_code,
    )
    latent = cover.assemble_weighted(corrected, 1) * scale * batch["mask"]
    sni = local_field
    for _ in range(sni_sweeps):
        sni = face_sni_step(
            local_operator, adapter, batch, sni, cover, "weighted_pou", 1.0,
        )
    point, point_saved = global_prediction(point_checkpoint, batch, device)
    sliced, slice_saved = global_prediction(slice_checkpoint, batch, device)
    patch, patch_saved = global_prediction(patch_checkpoint, batch, device)
    if tuple(patch_saved["model_spec"].get("patch_size", ())) != (core, core):
        raise ValueError("Patch size differs from the fixed 32-cell unit")
    tensors = {
        "Ground truth": batch["solution"],
        "LatentDDM": latent,
        "SNI-16": sni,
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
        "metrics": {
            name: float(relative_l2(value, batch["solution"], mask)[0])
            for name, value in tensors.items() if name != "Ground truth"
        },
        "mask": mask[0, 0].detach().cpu().numpy(),
        "logk": batch["logk"][0, 0].detach().cpu().numpy(),
        "case_index": int(batch["case_index"][0]),
        "selection": selection,
        "grid": int(batch["solution"].shape[-1]),
        "tile_count": int(cover.tile_count),
        "global_model_specs": {
            "Point": point_saved["model_spec"],
            "Slice": slice_saved["model_spec"],
            "Patch": patch_saved["model_spec"],
        },
    }


def render(domains: dict[str, dict[str, Any]], metadata: dict[str, Any]) -> plt.Figure:
    labels = ("Ground truth",) + METHODS
    active_truth = np.concatenate([
        domain["fields"]["Ground truth"][domain["mask"] > 0.5]
        for domain in domains.values()
    ])
    field_min, field_max = np.quantile(active_truth, (.01, .99))
    metadata["visual_scales"] = {
        "solution_quantiles": [float(field_min), float(field_max)],
    }

    plt.rcParams.update({"font.size": 7.3, "pdf.fonttype": 42})
    figure, axes = plt.subplots(len(domains), len(labels), figsize=(8.05, 4.15))
    figure.subplots_adjust(
        left=.058, right=.915, bottom=.035, top=.865,
        wspace=.045, hspace=.085,
    )
    field_image = None
    for row, (domain_name, domain) in enumerate(domains.items()):
        valid = domain["mask"] > .5
        best = min(domain["metrics"], key=domain["metrics"].get)
        for column, name in enumerate(labels):
            field_image = axes[row, column].imshow(
                np.ma.masked_where(~valid, domain["fields"][name]),
                origin="lower", cmap="viridis", vmin=field_min, vmax=field_max,
                interpolation="bilinear", rasterized=True,
            )
            if name != "Ground truth":
                value = domain["metrics"][name]
                axes[row, column].text(
                    .04, .94, rf"$e_u={value:.3f}$",
                    transform=axes[row, column].transAxes,
                    va="top", ha="left", fontsize=5.8,
                    fontweight="bold" if name == best else "normal",
                    bbox={"facecolor": "white", "alpha": .76,
                          "pad": .45, "edgecolor": "none"},
                )
            if row == 0:
                axes[row, column].set_title(name, fontsize=7.4, pad=2.5)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    for row, domain_name in enumerate(domains):
        cover = domain_name.replace("x", r"\!\times\!")
        axes[row, 0].set_ylabel(rf"${cover}$", fontsize=7.0, labelpad=3.0)
    assert field_image is not None
    field_axis = figure.add_axes((.930, .105, .011, .675))
    field_bar = figure.colorbar(field_image, cax=field_axis)
    field_bar.set_label("$u$", fontsize=7)
    field_bar.ax.tick_params(labelsize=6, length=2)

    figure.canvas.draw()
    for label, first, last, color in (
        ("LOCAL COMPOSITION", 1, 2, "#B44B4B"),
        ("GLOBAL METHODS", 3, 5, "#3568A8"),
    ):
        left = axes[0, first].get_position().x0
        right = axes[0, last].get_position().x1
        y = axes[0, first].get_position().y1 + .070
        figure.text(
            (left + right) / 2, y, label, ha="center", va="bottom",
            fontsize=7.4, fontweight="bold", color=color,
        )
        figure.add_artist(plt.Line2D(
            (left, right), (y - .008, y - .008),
            transform=figure.transFigure, color=color, lw=1.0,
        ))
    return figure


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.sni_sweeps != 16:
        raise ValueError("final Darcy figure uses the fixed SNI-16 endpoint")
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda",
    )
    local_operator, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    domains: dict[str, dict[str, Any]] = {}
    sources: dict[str, Any] = {
        "local": str(args.local_checkpoint.resolve()),
        "scales": {},
    }
    for blocks in args.scales:
        if blocks not in (3, 4, 8, 16):
            raise ValueError("the final figure is frozen to 3x3, 4x4, 8x8, and 16x16")
        label = f"{blocks}x{blocks}"
        checkpoint_label = "4x4" if blocks == 3 else label
        scale_root = args.extension_root / checkpoint_label
        data = (
            args.data_root / "darcy_full_domain_face_96_v2.h5"
            if blocks == 3 else
            args.data_root / f"darcy_domain_extension_fixed_dx_{label}_v1.h5"
        )
        paths = {
            "composer": scale_root / (
                f"input_latent_budget16/seed{args.seed}/checkpoint_best.pt"
            ),
            "point": scale_root / (
                f"global_budget16/point_seed{args.seed}/checkpoint.pt"
            ),
            "slice": scale_root / (
                f"global_budget16/slice_seed{args.seed}/checkpoint.pt"
            ),
            "patch": scale_root / (
                f"global_budget16/patch_seed{args.seed}/checkpoint.pt"
            ),
        }
        missing = [str(path) for path in (data, *paths.values()) if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        composer, saved = build_composer(
            paths["composer"], args.local_checkpoint, device,
        )
        domains[label] = predict_domain(
            data,
            local_operator=local_operator,
            adapter=adapter,
            composer=composer,
            composer_saved=saved,
            point_checkpoint=paths["point"],
            slice_checkpoint=paths["slice"],
            patch_checkpoint=paths["patch"],
            sni_sweeps=args.sni_sweeps,
            core=args.core,
            halo=args.halo,
            device=device,
        )
        sources["scales"][label] = {
            "data": str(data.resolve()),
            "checkpoint_scale": checkpoint_label,
            **{name: str(path.resolve()) for name, path in paths.items()},
        }
    metadata: dict[str, Any] = {
        "contract": "darcy_fixed_dx_four_scale_five_method_visual_v1",
        "split": "validation",
        "protected_test_cases_opened": 0,
        "method_order": list(METHODS),
        "adaptation_seed_shared_by_learned_methods": args.seed,
        "assembly": "cosine normalized overlap",
        "source_retention": "3x3 source data replayed with 4x4-adapted checkpoints",
        "sni": {"tau": 1.0, "sweeps": 16},
        "sources": sources,
        "domains": {
            name: {
                key: value for key, value in domain.items()
                if key not in ("fields", "mask", "logk")
            }
            for name, domain in domains.items()
        },
    }
    figure = render(domains, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(
        args.output.with_suffix(".png"), dpi=220,
        bbox_inches="tight", facecolor="white",
    )
    plt.close(figure)
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "domains": metadata["domains"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
