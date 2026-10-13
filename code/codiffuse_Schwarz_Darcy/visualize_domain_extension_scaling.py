"""Visualize the source and fixed-resolution Darcy extension domains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--targets", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-index", type=int, default=0)
    return parser.parse_args()


def record(path: Path, index: int) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        group = handle["val"]
        grid = int(group.attrs.get("grid_size", handle.attrs["grid_size"]))
        extent = float(group.attrs.get(
            "physical_extent", handle.attrs.get("physical_extent", 2.0),
        ))
        spacing = float(group.attrs.get(
            "grid_spacing", handle.attrs.get("grid_spacing", extent / grid),
        ))
        mask = np.asarray(group["mask"][index, 0], dtype=bool)
        solution = np.asarray(group["solution"][index, 0], dtype=np.float64)
        family = group["family"][index]
        if isinstance(family, bytes):
            family = family.decode("utf-8")
        return {
            "path": str(path.resolve()),
            "grid": grid,
            "extent": extent,
            "spacing": spacing,
            "blocks": grid // 32,
            "mask": mask,
            "solution": solution,
            "family": str(family),
            "case_index": index,
        }


def _grid_lines(axis, extent: float, blocks: int) -> None:
    edges = np.linspace(-0.5 * extent, 0.5 * extent, blocks + 1)[1:-1]
    for edge in edges:
        axis.axvline(edge, color="white", lw=0.35, alpha=0.42)
        axis.axhline(edge, color="white", lw=0.35, alpha=0.42)


def main() -> None:
    args = parse_args()
    entries = [record(args.source, args.case_index)] + [
        record(path, args.case_index) for path in args.targets
    ]
    active_values = np.concatenate([
        entry["solution"][entry["mask"]]
        for entry in entries
    ])
    lower, upper = np.quantile(active_values, (0.01, 0.99))
    figure, axes = plt.subplots(2, 4, figsize=(12.0, 5.7))
    titles = ("$3\\times3$", "$4\\times4$", "$8\\times8$", "$16\\times16$")
    image = None
    for column, (entry, title) in enumerate(zip(entries, titles, strict=True)):
        extent = float(entry["extent"])
        plot_extent = (-0.5 * extent, 0.5 * extent, -0.5 * extent, 0.5 * extent)
        mask = entry["mask"]
        axes[0, column].imshow(
            mask, origin="lower", extent=plot_extent, cmap="Greys",
            vmin=0.0, vmax=1.0, interpolation="nearest",
        )
        _grid_lines(axes[0, column], extent, int(entry["blocks"]))
        axes[0, column].set_title(title)
        shown = np.where(mask, entry["solution"], np.nan)
        image = axes[1, column].imshow(
            shown, origin="lower", extent=plot_extent, cmap="viridis",
            vmin=lower, vmax=upper, interpolation="nearest",
        )
        _grid_lines(axes[1, column], extent, int(entry["blocks"]))
        for row in range(2):
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_ylabel("geometry")
    axes[1, 0].set_ylabel("solution")
    assert image is not None
    figure.subplots_adjust(left=0.055, right=0.925, top=0.91, bottom=0.04,
                           wspace=0.08, hspace=0.10)
    color_axis = figure.add_axes((0.94, 0.075, 0.012, 0.385))
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label("$u$")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(
            args.output_dir / f"domain_extension_scaling.{suffix}",
            dpi=240, bbox_inches="tight",
        )
    plt.close(figure)
    metadata = {
        "contract": "darcy_fixed_dx_extension_visualization_v1",
        "split": "validation",
        "protected_test_cases_opened": 0,
        "solution_color_limits": [float(lower), float(upper)],
        "entries": [{
            key: value for key, value in entry.items()
            if key not in ("mask", "solution")
        } for entry in entries],
    }
    (args.output_dir / "domain_extension_scaling.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
