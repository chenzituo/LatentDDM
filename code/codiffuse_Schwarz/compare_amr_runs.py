from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from codiffuse_Schwarz.amr_flow import TwoLevelAMRConfig, build_amr_layouts
from codiffuse_Schwarz.run_amr_toy import level_map, masked


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compare matched adaptive AMR flow runs.")
    parser.add_argument("--control", type=Path, default=root / "outputs/amr_flow_nosync_60ep")
    parser.add_argument("--sync", type=Path, default=root / "outputs/amr_flow_sync_v2")
    parser.add_argument("--out-dir", type=Path, default=root / "outputs/amr_comparison_v2")
    parser.add_argument("--samples", type=int, default=4)
    return parser.parse_args()


def load_run(path: Path) -> tuple[dict, dict]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    predictions = torch.load(path / "predictions.pt", map_location="cpu", weights_only=False)
    return summary, predictions


def draw_boxes(axis: plt.Axes, boxes: tuple[tuple[int, int, int, int], ...]) -> None:
    for y0, y1, x0, x1 in boxes:
        axis.add_patch(
            plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec="white", lw=0.8)
        )


def plot_fields(control: dict, sync: dict, layouts: list, count: int, path: Path) -> None:
    test = sync["test"]
    count = min(count, test["field"].shape[0])
    titles = (
        "AMR levels",
        "source",
        "reference",
        "coarse (sync model)",
        "AMR, no consistency",
        "AMR, level consistency",
        "no-consistency error",
        "consistency error",
    )
    fig, axes = plt.subplots(count, len(titles), figsize=(21, 2.75 * count), squeeze=False)
    for row in range(count):
        mask = test["mask"][row]
        reference = test["field"][row]
        control_field = control["predictions"]["amr"][row]
        sync_field = sync["predictions"]["amr"][row]
        values = (
            level_map(layouts[row], reference.shape[-2:]),
            masked(test["source"][row], mask),
            masked(reference, mask),
            masked(sync["predictions"]["coarse"][row], mask),
            masked(control_field, mask),
            masked(sync_field, mask),
            masked((control_field - reference).abs(), mask),
            masked((sync_field - reference).abs(), mask),
        )
        field_limit = max(float(torch.nan_to_num(reference.abs()).max()), 1.0e-6)
        error_limit = max(float((control_field - reference).abs().max()), float((sync_field - reference).abs().max()))
        for column, (title, value) in enumerate(zip(titles, values)):
            if column == 0:
                image = axes[row, column].imshow(value, origin="lower", cmap="viridis", vmin=0, vmax=1)
            elif column >= 6:
                image = axes[row, column].imshow(value, origin="lower", cmap="magma", vmin=0, vmax=error_limit)
            else:
                image = axes[row, column].imshow(
                    value, origin="lower", cmap="coolwarm", vmin=-field_limit, vmax=field_limit
                )
            axes[row, column].set_title(title if row == 0 else f"sample {row}", fontsize=9)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if column in {0, 4, 5}:
                draw_boxes(axes[row, column], layouts[row].fine_boxes)
            fig.colorbar(image, ax=axes[row, column], fraction=0.044, pad=0.02)
    fig.suptitle("Adaptive Cartesian flow: matched 60-epoch comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_metrics(control_summary: dict, sync_summary: dict, path: Path) -> None:
    series = (
        ("control coarse", control_summary["metrics"]["coarse"]),
        ("control AMR", control_summary["metrics"]["amr"]),
        ("sync coarse", sync_summary["metrics"]["coarse"]),
        ("sync AMR", sync_summary["metrics"]["amr"]),
    )
    metrics = (
        ("relative_l2_mean", "relative L2"),
        ("relative_residual_mean", "relative residual"),
        ("interface_error_ratio_mean", "interface error ratio"),
    )
    colors = ("#8da0cb", "#4c78a8", "#f2a65a", "#d1495b")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, (key, label) in zip(axes, metrics):
        values = [entry[key] for _, entry in series]
        axis.bar(np.arange(len(series)), values, color=colors)
        axis.set_xticks(np.arange(len(series)), [name for name, _ in series], rotation=24, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    processed = sync_summary["layout_costs"]["processed_fraction_vs_uniform"] * 100.0
    fig.suptitle(f"Matched AMR ablation; adaptive processing estimate = {processed:.1f}% of uniform tiled fine")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    control_summary, control = load_run(args.control)
    sync_summary, sync = load_run(args.sync)
    config = TwoLevelAMRConfig(**sync_summary["config"])
    test = sync["test"]
    layouts = build_amr_layouts(test["mask"], test["source"], test["reaction"], config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_fields(control, sync, layouts, args.samples, args.out_dir / "paired_fields.png")
    plot_metrics(control_summary, sync_summary, args.out_dir / "metrics.png")


if __name__ == "__main__":
    main()
