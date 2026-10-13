"""Paper visualization for the 128^2 Darcy adaptation comparison.

The plot uses the 64-label endpoint of the target-adaptation curve in Figure 2(a).
The displayed validation case is fixed by a numerical rule: choose the case
whose LatentDDM error is closest to the validation median.  Visual appearance
does not enter selection.  The local methods share the frozen local neural
operator and disjoint core assembly.  SNI starts exactly from Local/Core.
Point, Slice, and Patch are independently adapted full-domain controls.  No
target field or discrete Darcy operator enters any prediction path.
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
from matplotlib.colors import SymLogNorm
import numpy as np
import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import DarcyAdapter, DarcyStatistics, H5DarcyDataset
from .benchmark_field_token_decomposed_methods import (
    physical_zero,
    target_free_context,
)
from .evaluate import move_batch, relative_l2
from .evaluate_global import build_from_checkpoint
from .field_token_composer import DarcyFieldTokenComposer
from .train_field_token_composer import CONTRACT
from .train_global import global_step
from .train_residual_port_chart import load_frozen_local, sha256
from .train_residual_port_communicator import load_frozen_chart
from .validate_face_sni import face_sni_step


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "codiffuse_Schwarz_Darcy/data/darcy_domain_extension_128_v1.h5"
COMPOSER = ROOT / (
    "codiffuse_Schwarz_Darcy/outputs/field_token_domain_extension_degree1_v1/"
    "depth4/budget64/seed20260811/checkpoint.pt"
)
SNI_ARTIFACT = ROOT / (
    "codiffuse_Schwarz_Darcy/outputs/domain_extension_128_v1/"
    "core_sni_local_init_validation_128.json"
)
POINT = ROOT / (
    "codiffuse_Schwarz_Darcy/outputs/domain_extension_128_v1/"
    "point_full_reuse_v1/budget64/point_seed20260811/checkpoint.pt"
)
SLICE = ROOT / (
    "codiffuse_Schwarz_Darcy/outputs/domain_extension_128_v1/"
    "adaptation_scaling_v1/budget64/transolver_seed20260811/checkpoint.pt"
)
PATCH = ROOT / (
    "codiffuse_Schwarz_Darcy/outputs/domain_extension_128_v1/"
    "patch_core32_capacitymatched_v1/budget64/"
    "patch_seed20260811/checkpoint.pt"
)
SELECTION = ROOT / (
    "codiffuse_Schwarz_Darcy/outputs/field_token_domain_extension_degree1_v1/"
    "depth4/budget64/figure3_median_latent_selection_seed20260811.json"
)
OUTPUT = ROOT / (
    "codiffuse_Schwarz/figures/unified_report/"
    "darcy_six_method_comparison.pdf"
)


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--composer-checkpoint", type=Path, default=COMPOSER)
    parser.add_argument("--local-checkpoint", type=Path)
    parser.add_argument("--chart-checkpoint", type=Path)
    parser.add_argument("--sni-artifact", type=Path, default=SNI_ARTIFACT)
    parser.add_argument("--point-checkpoint", type=Path, default=POINT)
    parser.add_argument("--slice-checkpoint", type=Path, default=SLICE)
    parser.add_argument("--patch-checkpoint", type=Path, default=PATCH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--selection-output", type=Path, default=SELECTION)
    parser.add_argument("--selection-cases", type=int, default=128)
    parser.add_argument("--selection-batch-size", type=int, default=4)
    parser.add_argument("--overwrite-selection", action="store_true")
    parser.add_argument(
        "--case-index", type=int,
        help="optional dataset position; default uses the frozen median rule",
    )
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument(
        "--sni-sweeps", type=int,
        help="override the validation-selected sweep from --sni-artifact",
    )
    parser.add_argument("--sni-tau", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def one_case(dataset: H5DarcyDataset, index: int, device: torch.device) -> dict:
    if index < 0 or index >= len(dataset):
        raise IndexError(f"case index {index} is outside [0,{len(dataset)})")
    host = dataset[index]
    return move_batch({
        key: value[None] if isinstance(value, Tensor) else value
        for key, value in host.items()
    }, device)


def load_composer(
    checkpoint: Path,
    local_override: Path | None,
    chart_override: Path | None,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, torch.nn.Module, Any, DarcyFieldTokenComposer, dict, Path, Path]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("contract") != CONTRACT:
        raise ValueError("composer checkpoint has the wrong contract")
    local_path = Path(
        local_override or saved["source_local_checkpoint"]
    ).resolve()
    chart_path = Path(
        chart_override or saved["source_chart_checkpoint"]
    ).resolve()
    if saved.get("source_local_sha256") != sha256(local_path):
        raise ValueError("composer/local checkpoint lineage mismatch")
    if saved.get("source_chart_sha256") != sha256(chart_path):
        raise ValueError("composer/chart checkpoint lineage mismatch")

    local, adapter, _ = load_frozen_local(local_path, device)
    chart, spec, _, chart_saved = load_frozen_chart(chart_path, device)
    if chart_saved.get("source_local_sha256") != sha256(local_path):
        raise ValueError("response chart/local checkpoint lineage mismatch")
    configuration = saved["composer_spec"]
    if configuration.get("assembly") != "disjoint_owned_core":
        raise ValueError("selected composer is not an owned-core model")
    composer = DarcyFieldTokenComposer(
        int(configuration["physical_channels"]),
        int(configuration["scalar_dim"]),
        int(configuration["code_width"]),
        token_width=int(configuration["token_width"]),
        heads=int(configuration["heads"]),
        communication_depth=int(configuration["communication_depth"]),
        spatial_port_bins=int(configuration.get("spatial_port_bins", 0)),
    ).to(device)
    composer.load_state_dict(saved["composer"], strict=True)
    for module in (local, chart, composer):
        module.eval().requires_grad_(False)
    return (
        local, adapter, chart, spec, composer, saved, local_path, chart_path,
    )


def global_prediction(
    checkpoint: Path,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[Tensor, dict]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    statistics = DarcyStatistics.from_json(saved["statistics"])
    adapter = DarcyAdapter(statistics)
    model = build_from_checkpoint(saved).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval().requires_grad_(False)
    current = physical_zero(batch)
    return global_step(model, adapter, batch, current, current, 1.0), saved


def _selection_identity(
    args: argparse.Namespace,
    local_path: Path,
    chart_path: Path,
) -> dict[str, Any]:
    return {
        "data": str(args.data.resolve()),
        "composer_checkpoint": str(args.composer_checkpoint.resolve()),
        "composer_checkpoint_sha256": sha256(args.composer_checkpoint),
        "local_checkpoint_sha256": sha256(local_path),
        "chart_checkpoint_sha256": sha256(chart_path),
        "selection_cases": int(args.selection_cases),
        "core": int(args.core),
        "halo": int(args.halo),
    }


@torch.inference_mode()
def select_representative_case(
    args: argparse.Namespace,
    local: torch.nn.Module,
    adapter: Any,
    chart: torch.nn.Module,
    spec: Any,
    composer: DarcyFieldTokenComposer,
    local_path: Path,
    chart_path: Path,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    """Freeze a representative case without looking at rendered images."""
    identity = _selection_identity(args, local_path, chart_path)
    if args.selection_output.exists() and not args.overwrite_selection:
        saved = json.loads(args.selection_output.read_text(encoding="utf-8"))
        if (
            saved.get("contract")
            != "darcy_128_median_latent_validation_selection_v1"
            or saved.get("identity") != identity
            or saved.get("visual_appearance_used_for_selection") is not False
        ):
            raise ValueError(
                "existing representative-case selection has stale lineage; "
                "use --overwrite-selection after freezing the new checkpoint"
            )
        return int(saved["selected"]["dataset_index"]), saved

    dataset = H5DarcyDataset(args.data, "validation")
    rows: list[dict[str, float | int]] = []
    try:
        dataset.length = min(dataset.length, args.selection_cases)
        loader = DataLoader(
            dataset, batch_size=args.selection_batch_size, shuffle=False,
            num_workers=0,
        )
        position = 0
        for host in loader:
            with_target = move_batch(host, device)
            target = with_target["solution"]
            batch = {
                key: value for key, value in with_target.items()
                if key != "solution"
            }
            cover = CartesianCover(
                target.shape[-2], target.shape[-1], args.core, args.halo,
            )
            context = target_free_context(
                local, adapter, batch, cover, spec.code_width,
            )
            local_field = cover.assemble_ras(
                context.local, target.shape[0],
            ) * batch["mask"] * adapter.statistics.solution_std
            latent_field = composer(
                chart, context, batch["mask"],
            ).assembled * adapter.statistics.solution_std
            local_error = relative_l2(
                local_field, target, batch["mask"],
            ).detach().cpu()
            latent_error = relative_l2(
                latent_field, target, batch["mask"],
            ).detach().cpu()
            case_ids = with_target["case_index"].detach().cpu()
            for item in range(target.shape[0]):
                rows.append({
                    "dataset_index": position + item,
                    "case_index": int(case_ids[item]),
                    "local_core_relative_l2": float(local_error[item]),
                    "latentddm_relative_l2": float(latent_error[item]),
                    "paired_error_change": float(
                        latent_error[item] - local_error[item]
                    ),
                })
            position += target.shape[0]
    finally:
        dataset.close()
    if not rows:
        raise RuntimeError("representative-case selection evaluated no cases")
    median = float(np.median([
        float(row["latentddm_relative_l2"]) for row in rows
    ]))
    selected = min(
        rows,
        key=lambda row: (
            abs(float(row["latentddm_relative_l2"]) - median),
            int(row["dataset_index"]),
        ),
    )
    result: dict[str, Any] = {
        "contract": "darcy_128_median_latent_validation_selection_v1",
        "split": "validation",
        "protected_test_cases_opened": 0,
        "selection_rule": (
            "minimum absolute deviation from the validation median "
            "LatentDDM relative L2; ties use the lower dataset index"
        ),
        "visual_appearance_used_for_selection": False,
        "identity": identity,
        "median_latentddm_relative_l2": median,
        "selected": selected,
        "rows": rows,
    }
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return int(selected["dataset_index"]), result


def selected_sni_sweeps(
    args: argparse.Namespace,
    local_path: Path,
) -> tuple[int, dict[str, Any]]:
    payload = json.loads(args.sni_artifact.read_text(encoding="utf-8"))
    if (
        payload.get("contract")
        != "darcy_core_ras_sni_local_initialization_v2"
        or payload.get("split") != "validation"
        or payload.get("protected_test_cases_opened") != 0
        or payload.get("assembly") != "core_ras"
        or payload.get("initialization") != "local_core"
        or payload.get("local_initialization_equals_sni_sweep0_max_abs") != 0.0
        or payload.get("local_checkpoint_sha256") != sha256(local_path)
        or Path(payload.get("data", "")).resolve() != args.data.resolve()
    ):
        raise ValueError("SNI artifact does not match the 128^2 figure protocol")
    if args.sni_sweeps is None:
        configuration = str(payload["best_validation_field"]["configuration"])
        sweeps = int(configuration.rsplit("sweep_", 1)[1])
        source = "best validation field checkpoint"
    else:
        sweeps = int(args.sni_sweeps)
        source = "command-line override"
    key = f"init_local_core/tau_{args.sni_tau:.3f}/sweep_{sweeps:04d}"
    if key not in payload["summary"]:
        raise ValueError(f"SNI artifact does not contain {key}")
    return sweeps, {
        "artifact": str(args.sni_artifact.resolve()),
        "artifact_contract": payload["contract"],
        "selection_source": source,
        "validation_summary": payload["summary"][key],
    }


@torch.inference_mode()
def predict(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict]:
    if (
        min(args.core, args.selection_cases, args.selection_batch_size) <= 0
        or args.halo < 0
        or args.sni_sweeps is not None and args.sni_sweeps <= 0
        or not 0.0 < args.sni_tau <= 1.0
    ):
        raise ValueError("invalid visualization configuration")
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    (
        local, adapter, chart, spec, composer, composer_saved,
        local_path, chart_path,
    ) = load_composer(
        args.composer_checkpoint, args.local_checkpoint,
        args.chart_checkpoint, device,
    )
    if args.case_index is None:
        dataset_index, selection = select_representative_case(
            args, local, adapter, chart, spec, composer,
            local_path, chart_path, device,
        )
    else:
        dataset_index = int(args.case_index)
        selection = {
            "selection_rule": "explicit dataset index",
            "visual_appearance_used_for_selection": None,
            "selected": {"dataset_index": dataset_index},
        }
    sni_sweeps, sni_selection = selected_sni_sweeps(args, local_path)

    dataset = H5DarcyDataset(args.data, "validation")
    try:
        batch_with_target = one_case(dataset, dataset_index, device)
    finally:
        dataset.close()
    target = batch_with_target["solution"]
    # Keep the target physically absent from every prediction call.
    batch = {
        key: value for key, value in batch_with_target.items()
        if key != "solution"
    }
    cover = CartesianCover(
        batch["mask"].shape[-2], batch["mask"].shape[-1],
        args.core, args.halo,
    )
    context = target_free_context(
        local, adapter, batch, cover, spec.code_width,
    )
    local_field = cover.assemble_ras(context.local, 1) * batch["mask"]
    local_field = local_field * adapter.statistics.solution_std
    latent_field = composer(chart, context, batch["mask"]).assembled
    latent_field = latent_field * adapter.statistics.solution_std

    init_sni = local_field.clone()
    for _ in range(sni_sweeps):
        init_sni = face_sni_step(
            local, adapter, batch, init_sni, cover,
            "core_ras", args.sni_tau,
        )
    point, point_saved = global_prediction(
        args.point_checkpoint, batch, device,
    )
    sliced, slice_saved = global_prediction(
        args.slice_checkpoint, batch, device,
    )
    patched, patch_saved = global_prediction(
        args.patch_checkpoint, batch, device,
    )
    if tuple(patch_saved["model_spec"].get("patch_size", ())) != (32, 32):
        raise ValueError("paper Patch checkpoint must use the 32x32 local-core size")

    tensor_fields = {
        "Ground truth": target,
        "Local/Core": local_field,
        "LatentDDM": latent_field,
        "SNI": init_sni,
        "Point": point,
        "Slice": sliced,
        "Patch": patched,
    }
    mask = batch["mask"]
    metrics = {
        name: float(relative_l2(field, target, mask)[0])
        for name, field in tensor_fields.items()
        if name != "Ground truth"
    }
    fields = {
        name: field[0, 0].detach().cpu().numpy()
        for name, field in tensor_fields.items()
    }
    checkpoints = {
        "local": local_path,
        "response_chart": chart_path,
        "latentddm": args.composer_checkpoint,
        "point": args.point_checkpoint,
        "slice": args.slice_checkpoint,
        "patch": args.patch_checkpoint,
    }
    metadata = {
        "contract": "darcy_128_adaptation_seven_column_visual_v4",
        "split": "validation",
        "protected_test_cases_opened": 0,
        "dataset_index": dataset_index,
        "case_index": int(batch_with_target["case_index"][0].item()),
        "adaptation_setting": {
            "source_grid": [96, 96],
            "target_grid": [128, 128],
            "target_training_labels": 64,
            "validation_cases": args.selection_cases,
        },
        "assembly": "disjoint_owned_core",
        "overlap_averaging_or_pou": False,
        "method_order": [
            "Local/Core", "LatentDDM", "SNI", "Point", "Slice", "Patch",
        ],
        "cover": {
            "grid_shape_yx": list(cover.grid_shape),
            "subdomains": cover.tile_count,
            "core": cover.core,
            "halo_context": cover.halo,
        },
        "communication_depth": int(
            composer_saved["composer_spec"]["communication_depth"]
        ),
        "representative_case_selection": {
            "path": (
                str(args.selection_output.resolve())
                if args.case_index is None else None
            ),
            "selection_rule": selection["selection_rule"],
            "visual_appearance_used_for_selection": selection[
                "visual_appearance_used_for_selection"
            ],
        },
        "sni": {
            "tau": args.sni_tau,
            "sweeps": sni_sweeps,
            "initialization": "exact Local/Core field",
            "sweep0_is_exact_local_core": True,
            "assembly_each_sweep": "disjoint_owned_core",
            **sni_selection,
        },
        "relative_l2": metrics,
        "prediction_target_read": False,
        "discrete_operator_used_by_prediction": False,
        "method_definition": {
            "Local/Core": "one frozen local-operator call; no communication",
            "LatentDDM": "same local operator plus one feed-forward communication module",
            "SNI": (
                "same local operator in a recurrence initialized at Local/Core"
            ),
            "Point": "independently trained full-domain anchor attention",
            "Slice": "independently trained full-domain Transolver",
            "Patch": "independently trained full-domain patch Transformer",
        },
        "checkpoints": {
            name: {
                "path": str(Path(path).resolve()),
                "sha256": sha256(Path(path)),
            }
            for name, path in checkpoints.items()
        },
        "global_model_specs": {
            "Point": point_saved["model_spec"],
            "Slice": slice_saved["model_spec"],
            "Patch": patch_saved["model_spec"],
        },
        "logk": batch["logk"][0, 0].detach().cpu().numpy(),
        "mask": mask[0, 0].detach().cpu().numpy(),
    }
    return fields, metadata


def plot(fields: dict[str, np.ndarray], metadata: dict) -> plt.Figure:
    names = tuple(fields)
    mask = metadata["mask"] > 0.5
    target = fields["Ground truth"]
    vmin, vmax = np.quantile(target[mask], (0.01, 0.99))
    signed_errors = {
        name: fields[name] - target
        for name in names[1:]
    }
    residual_values = np.concatenate([
        np.abs(error[mask]) for error in signed_errors.values()
    ])
    residual_limit = max(float(np.quantile(residual_values, 0.99)), 1.0e-4)
    residual_linthresh = max(0.01 * residual_limit, 1.0e-5)
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
    best = min(metadata["relative_l2"], key=metadata["relative_l2"].get)

    figure, axes = plt.subplots(2, len(names), figsize=(8.15, 2.78))
    figure.subplots_adjust(
        left=0.022, right=0.965, bottom=0.055, top=0.77,
        wspace=0.065, hspace=0.23,
    )
    field_image = None
    error_image = None
    for column, name in enumerate(names):
        field_image = axes[0, column].imshow(
            np.where(mask, fields[name], np.nan),
            origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
            interpolation="bilinear", rasterized=True,
        )
        if name == "Ground truth":
            title = "Ground truth"
        else:
            value = f"{metadata['relative_l2'][name]:.4f}"
            if name == best:
                value = rf"\mathbf{{{value}}}"
            title = name + "\n" + rf"$e_u={value}$"
        axes[0, column].set_title(title, fontsize=7.25, pad=2.0)

        if name == "Ground truth":
            axes[1, column].imshow(
                np.where(mask, metadata["logk"], np.nan),
                origin="lower", cmap="magma", interpolation="nearest",
                rasterized=True,
            )
            axes[1, column].set_title(r"Input $\log_{10}\kappa$", fontsize=7.25)
        else:
            error_image = axes[1, column].imshow(
                np.where(mask, signed_errors[name], np.nan),
                origin="lower", cmap="coolwarm", norm=error_norm,
                interpolation="nearest", rasterized=True,
            )
            axes[1, column].set_title("Signed error", fontsize=7.25)
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

    assert field_image is not None and error_image is not None
    figure.colorbar(
        field_image, ax=axes[0].tolist(), shrink=0.76, pad=0.006,
        label="$u$",
    )
    error_colorbar = figure.colorbar(
        error_image, ax=axes[1, 1:].tolist(), shrink=0.76, pad=0.006,
        label=r"$\widehat u-u$ (symmetric log scale)",
    )
    exponent_min = int(np.ceil(np.log10(residual_linthresh)))
    exponent_max = int(np.floor(np.log10(residual_limit)))
    positive_ticks = 10.0 ** np.arange(exponent_min, exponent_max + 1)
    # Three ticks keep the narrow paper colorbar legible; the complete
    # symmetric-log transform remains recorded in the JSON sidecar.
    largest_tick = positive_ticks[-1]
    residual_ticks = np.asarray((-largest_tick, 0.0, largest_tick))
    error_colorbar.set_ticks(residual_ticks)
    error_colorbar.set_ticklabels([f"{value:g}" for value in residual_ticks])
    figure.canvas.draw()
    for label, first, last, color in (
        ("LOCAL METHODS", 1, 3, "#B44B4B"),
        ("GLOBAL METHODS", 4, 6, "#3568A8"),
    ):
        left = axes[0, first].get_position().x0
        right = axes[0, last].get_position().x1
        y = max(
            axes[0, column].get_position().y1
            for column in range(first, last + 1)
        ) + 0.175
        figure.text(
            (left + right) / 2, y, label,
            ha="center", va="bottom", fontsize=9.1,
            fontweight="bold", color=color,
        )
        figure.add_artist(plt.Line2D(
            (left, right), (y - 0.012, y - 0.012),
            transform=figure.transFigure, color=color, lw=1.25,
        ))
    return figure


def main() -> None:
    args = parse_args()
    fields, metadata = predict(args)
    figure = plot(fields, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output, dpi=300, bbox_inches="tight", facecolor="white",
    )
    figure.savefig(
        args.output.with_suffix(".png"), dpi=220,
        bbox_inches="tight", facecolor="white",
    )
    plt.close(figure)
    serializable = {
        key: value for key, value in metadata.items()
        if key not in ("logk", "mask")
    }
    serializable["output_pdf"] = str(args.output.resolve())
    args.output.with_suffix(".json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(serializable, indent=2), flush=True)


if __name__ == "__main__":
    main()
