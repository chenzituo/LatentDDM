"""Visualize a representative active Darcy LatentDDM validation case.

The displayed case is chosen deterministically as the median positive
value-to-dynamic field improvement among the first ``selection_cases``
validation examples.  Identity-fallback and negative-response counts, the
selection rule, and all per-case errors are saved next to the figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .adapter import H5DarcyDataset
from .evaluate import move_batch, relative_l2
from .latent_schur_teacher import inference_only_context
from .train_incremental_flux_remote_schur import (
    _decomposed_fields,
    _load_communicator,
    _reference_fields,
)
from .train_residual_port_chart import frozen_context, load_frozen_local, sha256
from .train_residual_port_communicator import load_frozen_chart


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--dynamic-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-cases", type=int, default=60)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _one_case(dataset: H5DarcyDataset, index: int, device: torch.device):
    host = dataset[index]
    return move_batch({
        key: value[None] if isinstance(value, torch.Tensor) else value
        for key, value in host.items()
    }, device)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, _ = load_frozen_chart(args.chart_checkpoint, device)
    value, value_payload = _load_communicator(
        args.value_checkpoint, spec, device,
    )
    dynamic, dynamic_payload = _load_communicator(
        args.dynamic_checkpoint, spec, device,
    )
    if getattr(dynamic, "analyzer_mode", None) != \
            "value_flux_dynamic_relative_green":
        raise ValueError("dynamic checkpoint is not the frozen value--flux kernel")
    for payload in (value_payload, dynamic_payload):
        if payload["source_local_sha256"] != sha256(args.local_checkpoint):
            raise ValueError("communicator/local lineage mismatch")
        if payload["source_chart_sha256"] != sha256(args.chart_checkpoint):
            raise ValueError("communicator/chart lineage mismatch")
    for module in (local, chart, value, dynamic):
        module.eval()

    dataset = H5DarcyDataset(args.data, "validation")
    count = min(args.selection_cases, len(dataset))
    rows: list[dict[str, float | int]] = []
    cached: dict[int, tuple[dict[str, torch.Tensor], object, torch.Tensor,
                            torch.Tensor, torch.Tensor]] = {}
    for index in range(count):
        batch = _one_case(dataset, index, device)
        supervised = frozen_context(
            local, adapter, batch, spec, args.core, args.halo,
        )
        context = inference_only_context(supervised)
        value_field, _, _ = _reference_fields(
            value, chart, context, batch, spec, scale,
        )
        dynamic_field = _decomposed_fields(
            dynamic, chart, context, batch, spec, scale,
        )["assembled"]
        local_field = context.cover.assemble_weighted(
            context.local, batch["solution"].shape[0],
        ) * batch["mask"]
        physical = {
            "local": local_field * adapter.statistics.solution_std,
            "value": value_field * adapter.statistics.solution_std,
            "dynamic": dynamic_field * adapter.statistics.solution_std,
        }
        errors = {
            name: float(relative_l2(
                field, batch["solution"], batch["mask"],
            )[0]) for name, field in physical.items()
        }
        improvement = errors["value"] - errors["dynamic"]
        rows.append({"case_index": index, **errors, "improvement": improvement})
        if args.case_index is not None and index == args.case_index:
            cached[index] = (batch, context, *physical.values())

    if args.case_index is None:
        active = [row for row in rows if float(row["improvement"]) > 1.0e-10]
        ordered = sorted(active or rows, key=lambda row: float(row["improvement"]))
        selected = int(ordered[len(ordered) // 2]["case_index"])
    else:
        selected = args.case_index
        if not 0 <= selected < len(dataset):
            raise IndexError("case index lies outside validation split")
    if selected in cached:
        batch, context, local_field, value_field, dynamic_field = cached[selected]
    else:
        batch = _one_case(dataset, selected, device)
        supervised = frozen_context(
            local, adapter, batch, spec, args.core, args.halo,
        )
        context = inference_only_context(supervised)
        value_normalized, _, _ = _reference_fields(
            value, chart, context, batch, spec, scale,
        )
        dynamic_normalized = _decomposed_fields(
            dynamic, chart, context, batch, spec, scale,
        )["assembled"]
        local_field = context.cover.assemble_weighted(
            context.local, 1,
        ) * batch["mask"] * adapter.statistics.solution_std
        value_field = value_normalized * adapter.statistics.solution_std
        dynamic_field = dynamic_normalized * adapter.statistics.solution_std
    dataset.close()

    target = batch["solution"]
    mask = batch["mask"][0, 0].cpu().numpy() > 0.5
    target_np = target[0, 0].cpu().numpy()
    fields = {
        "Reference": target_np,
        "Local POU": local_field[0, 0].cpu().numpy(),
        "Value latent": value_field[0, 0].cpu().numpy(),
        "Value+flux LatentDDM": dynamic_field[0, 0].cpu().numpy(),
    }
    metrics = {
        name: float(relative_l2(
            torch.as_tensor(array, device=device)[None, None],
            target, batch["mask"],
        )[0]) for name, array in fields.items() if name != "Reference"
    }
    vmin, vmax = np.quantile(target_np[mask], (0.01, 0.99))
    error_max = max(float(np.quantile(
        np.abs(array - target_np)[mask], 0.98,
    )) for name, array in fields.items() if name != "Reference")
    figure, axes = plt.subplots(2, 4, figsize=(11.8, 5.0), constrained_layout=True)
    field_image = None
    error_image = None
    for column, (name, array) in enumerate(fields.items()):
        field_image = axes[0, column].imshow(
            np.where(mask, array, np.nan), origin="lower", cmap="viridis",
            vmin=vmin, vmax=vmax,
        )
        suffix = "" if name == "Reference" else f"\n$L^2$={metrics[name]:.4f}"
        axes[0, column].set_title(name + suffix, fontsize=9)
        if name == "Reference":
            axes[1, column].imshow(
                np.where(mask, batch["logk"][0, 0].cpu().numpy(), np.nan),
                origin="lower", cmap="magma",
            )
            axes[1, column].set_title(r"Input $\log_{10}\kappa$", fontsize=9)
        else:
            error_image = axes[1, column].imshow(
                np.where(mask, np.abs(array - target_np), np.nan),
                origin="lower", cmap="inferno", vmin=0.0, vmax=error_max,
            )
            axes[1, column].set_title("Absolute error", fontsize=9)
        for row in range(2):
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
            for boundary in (32, 64):
                axes[row, column].axhline(boundary - 0.5, color="white",
                                          lw=0.45, ls="--", alpha=0.55)
                axes[row, column].axvline(boundary - 0.5, color="white",
                                          lw=0.45, ls="--", alpha=0.55)
    assert field_image is not None and error_image is not None
    figure.colorbar(field_image, ax=axes[0].tolist(), shrink=0.78, pad=0.012,
                    label="solution $u$")
    figure.colorbar(error_image, ax=axes[1, 1:].tolist(), shrink=0.78, pad=0.012,
                    label=r"$|\widehat u-u|$")
    figure.suptitle(
        f"Frozen Darcy LatentDDM: median active value-to-flux correction "
        f"(validation case {selected})", fontsize=11,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "darcy_frozen_latentddm_median_case"
    figure.savefig(stem.with_suffix(".pdf"), dpi=240, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)
    selected_row = next(row for row in rows if row["case_index"] == selected)
    positive = sum(float(row["improvement"]) > 1.0e-10 for row in rows)
    negative = sum(float(row["improvement"]) < -1.0e-10 for row in rows)
    identity = len(rows) - positive - negative
    stem.with_suffix(".json").write_text(json.dumps({
        "contract": "darcy_frozen_latentddm_representative_visual_v1",
        "selection_rule": (
            "median positive value-minus-dynamic relative-L2 improvement; "
            "active cases only"
        ),
        "selection_cases": count,
        "field_response_counts": {
            "positive": positive, "negative": negative, "identity": identity,
        },
        "selected": selected_row,
        "display_metrics": metrics,
        "dynamic_checkpoint": str(args.dynamic_checkpoint.resolve()),
        "dynamic_checkpoint_sha256": sha256(args.dynamic_checkpoint),
        "value_checkpoint": str(args.value_checkpoint.resolve()),
        "target_or_discrete_operator_used_by_inference": False,
        "validation_cases": rows,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
