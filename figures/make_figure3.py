"""Rebuild both manuscript Figure 3 components from portable local data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/two_case_figure_making_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator
import numpy as np


HERE = Path(__file__).resolve().parent
SCALES = ("3x3", "4x4", "8x8", "16x16")
LABELS = ("Ground truth", "LatentDDM", "SNI-16", "Point", "Slice", "Patch")
METHODS = ("LatentDDM", "Point", "Slice", "Patch")
COLORS = {
    "LatentDDM": "#B44B4B", "SNI": "#D8844B", "Point": "#3568A8",
    "Slice": "#2E8B8B", "Patch": "#7761A8",
}
MARKERS = {"LatentDDM": "D", "Point": "o", "Slice": "s", "Patch": "^"}


def render_fields(npz_path: Path) -> plt.Figure:
    archive = np.load(npz_path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"]))
    if metadata.get("contract") != "two_case_figure3a_plot_data_v1":
        raise ValueError("unexpected Figure 3(a) data contract")
    if int(metadata.get("protected_test_cases_opened", -1)) != 0:
        raise ValueError("Figure 3(a) data is not validation-only")
    active_truth = np.concatenate([
        archive[f"{scale}__Ground truth"][archive[f"{scale}__mask"] > 0]
        for scale in SCALES
    ])
    field_min, field_max = np.quantile(active_truth, (.01, .99))

    plt.rcParams.update({"font.size": 7.3, "pdf.fonttype": 42})
    figure, axes = plt.subplots(len(SCALES), len(LABELS), figsize=(8.05, 4.15))
    figure.subplots_adjust(
        left=.058, right=.915, bottom=.035, top=.865,
        wspace=.045, hspace=.085,
    )
    field_image = None
    for row, scale in enumerate(SCALES):
        valid = archive[f"{scale}__mask"] > 0
        metrics = metadata["metrics"][scale]
        best = min(metrics, key=metrics.get)
        for column, label in enumerate(LABELS):
            field_image = axes[row, column].imshow(
                np.ma.masked_where(~valid, archive[f"{scale}__{label}"]),
                origin="lower", cmap="viridis", vmin=field_min, vmax=field_max,
                interpolation="bilinear", rasterized=True,
            )
            if label != "Ground truth":
                value = float(metrics[label])
                axes[row, column].text(
                    .04, .94, rf"$e_u={value:.3f}$",
                    transform=axes[row, column].transAxes,
                    va="top", ha="left", fontsize=5.8,
                    fontweight="bold" if label == best else "normal",
                    bbox={"facecolor": "white", "alpha": .76,
                          "pad": .45, "edgecolor": "none"},
                )
            if row == 0:
                axes[row, column].set_title(label, fontsize=7.4, pad=2.5)
    for axis in axes.flat:
        axis.set_xticks([]); axis.set_yticks([])
    for row, scale in enumerate(SCALES):
        cover = scale.replace("x", r"\!\times\!")
        axes[row, 0].set_ylabel(
            f"${cover}$",
            fontsize=7.0, labelpad=3.0,
        )
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


def render_accuracy_time(json_path: Path) -> plt.Figure:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("contract") != "darcy_four_scale_accuracy_time_validation_v1":
        raise ValueError("unexpected Figure 3(b) data contract")
    if int(payload.get("protected_test_cases_opened", -1)) != 0:
        raise ValueError("Figure 3(b) data is not validation-only")

    plt.rcParams.update({
        "font.size": 7.3, "pdf.fonttype": 42, "axes.linewidth": .65,
        "xtick.major.width": .55, "ytick.major.width": .55,
    })
    figure, axes = plt.subplots(1, 4, figsize=(7.73, 1.656), sharey=True)
    figure.subplots_adjust(left=.074, right=.993, bottom=.285, top=.765, wspace=.14)
    for index, scale in enumerate(SCALES):
        axis = axes[index]
        row = payload["scales"][scale]
        sni = row["sni_accuracy_time_curve"]
        axis.loglog(
            [1000.0 * float(point["seconds_per_case_mean"]) for point in sni],
            [float(point["relative_l2_mean"]) for point in sni],
            color=COLORS["SNI"], lw=1.25, marker=".", ms=3.3,
            label="SNI sweeps", zorder=2,
        )
        for method in METHODS:
            method_row = row["methods"][method]
            axis.scatter(
                1000.0 * float(method_row["seconds"]["mean"]),
                float(method_row["error"]["mean"]),
                s=28 if method == "LatentDDM" else 23,
                marker=MARKERS[method], color=COLORS[method], edgecolor="white",
                linewidth=.45, label=method, zorder=4,
            )
        axis.grid(True, which="major", color="#D9DDE5", lw=.45, alpha=.8)
        axis.grid(True, which="minor", color="#ECEEF2", lw=.30, alpha=.65)
        axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.yaxis.set_major_locator(FixedLocator((.02, .05, .1, .2, .5)))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        title = scale.replace("x", r"\!\times\!")
        axis.set_title(
            r"$3\!\times\!3$ replay" if scale == "3x3" else f"${title}$",
            fontsize=7.4, pad=3.0,
        )
        axis.set_xlabel("Inference time / case (ms)", fontsize=7.0, labelpad=1.5)
        axis.tick_params(labelsize=6.0, length=2.1, pad=1.3)
        axis.set_ylim(.014, .72)
        x_values = [
            1000.0 * float(point["seconds_per_case_mean"]) for point in sni
        ] + [
            1000.0 * float(row["methods"][method]["seconds"]["mean"])
            for method in METHODS
        ]
        axis.set_xlim(min(x_values) * .72, max(x_values) * 1.35)
    axes[0].set_ylabel(r"Relative $L^2$ error", fontsize=7.0, labelpad=2.0)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(name) for name in (
        "LatentDDM", "SNI sweeps", "Point", "Slice", "Patch",
    )]
    figure.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="upper center", bbox_to_anchor=(.535, .995), ncol=5,
        frameon=False, fontsize=7.4, handletextpad=.35, columnspacing=1.1,
    )
    return figure


def save(figure: plt.Figure, path: Path, png_dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(path.with_suffix(".png"), dpi=png_dpi,
                   bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "build")
    args = parser.parse_args()
    save(
        render_fields(args.data_dir / "figure3a_fields.npz"),
        args.output_dir / "darcy_source_target_adaptation.pdf", 220,
    )
    save(
        render_accuracy_time(args.data_dir / "figure3b_accuracy_time.json"),
        args.output_dir / "darcy_four_scale_accuracy_time.pdf", 240,
    )


if __name__ == "__main__":
    main()
