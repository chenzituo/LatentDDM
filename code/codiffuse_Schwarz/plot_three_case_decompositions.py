"""Dedicated decomposition figure for the three LatentDDM case studies."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codiffuse_mplconfig")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DARCY = ROOT / "codiffuse_Schwarz_Darcy/data/darcy_domain_extension_128_v1.h5"
AIRFOIL = Path(
    "/orcd/scratch/orcd/003/zituo/AMR_Diffusion_Foundation/"
    "codiffuse_airfoil/cyclic_002_level6_window.h5"
)
OUTPUT = ROOT / "codiffuse_Schwarz/figures/unified_report/three_case_decompositions.pdf"
PURPLE = "#5B21B6"


def car_lattice_segments(nz: int = 5, ny: int = 3, nx: int = 28):
    def project(ix: int, iy: int, iz: int) -> tuple[float, float]:
        x, y, z = ix / nz, iy / nz, iz / nz
        return x + .36 * y, z + .26 * y

    segments = []
    for iy in range(ny + 1):
        for iz in range(nz + 1):
            segments.append((project(0, iy, iz), project(nx, iy, iz)))
    for ix in range(nx + 1):
        for iz in range(nz + 1):
            segments.append((project(ix, 0, iz), project(ix, ny, iz)))
    for ix in range(nx + 1):
        for iy in range(ny + 1):
            segments.append((project(ix, iy, 0), project(ix, iy, nz)))
    return segments


def main() -> None:
    with h5py.File(DARCY, "r") as handle:
        mask = np.asarray(handle["val/mask"][39, 0], dtype=bool)
        logk = np.asarray(handle["val/logk"][39, 0], dtype=np.float32)
    with h5py.File(AIRFOIL, "r") as handle:
        sdf = np.asarray(handle["sdf"][0, 0], dtype=np.float32)

    figure, axes = plt.subplots(
        1, 3, figsize=(7.0, 1.52), constrained_layout=True,
        gridspec_kw={"width_ratios": (1.0, 1.45, 2.0)},
    )

    # Darcy: actual target geometry with the exact 4 x 4 core partition.
    axes[0].imshow(
        np.ma.array(logk, mask=~mask), origin="lower", cmap="magma",
        interpolation="nearest", rasterized=True,
    )
    for boundary in (31.5, 63.5, 95.5):
        axes[0].axhline(boundary, color="white", lw=.65, ls="--", alpha=.9)
        axes[0].axvline(boundary, color="white", lw=.65, ls="--", alpha=.9)
    axes[0].contour(mask.astype(float), levels=(.5,), colors="black", linewidths=.55)
    axes[0].set_title("Darcy", fontsize=8.0)
    axes[0].set_aspect("equal")

    # DrivAerNet++: every edge of the full 28 x 3 x 5 cube topology.
    axes[1].add_collection(LineCollection(
        car_lattice_segments(), colors=PURPLE, linewidths=.28, alpha=.58,
    ))
    axes[1].set_xlim(-.04, 28 / 5 + .36 * 3 / 5 + .04)
    axes[1].set_ylim(-.04, 1.0 + .26 * 3 / 5 + .04)
    axes[1].set_aspect("equal")
    axes[1].set_title("DrivAerNet++", fontsize=8.0)

    # Airfoil: actual solid boundary under the exact shifted 25 x 13 cover.
    height, width = sdf.shape
    axes[2].set_facecolor("#F7F7FA")
    axes[2].contourf(
        np.arange(width), np.arange(height), sdf < 0,
        levels=(-.5, .5, 1.5), colors=("#F7F7FA", "#262626"),
    )
    axes[2].contour(sdf, levels=(0,), colors="black", linewidths=.55)
    core, offset_x, offset_y = 32, 25, 24
    for ix in range(26):
        boundary = ix * core - offset_x
        if 0 <= boundary <= width:
            axes[2].axvline(boundary, color=PURPLE, lw=.28, alpha=.48)
    for iy in range(14):
        boundary = iy * core - offset_y
        if 0 <= boundary <= height:
            axes[2].axhline(boundary, color=PURPLE, lw=.28, alpha=.48)
    axes[2].set_xlim(0, width - 1)
    axes[2].set_ylim(0, height - 1)
    axes[2].set_aspect("equal")
    axes[2].set_title("Airfoil", fontsize=8.0)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(.65)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(
        OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    metadata = {
        "contract": "three_case_decomposition_visual_v1",
        "darcy": {"grid_shape": [4, 4], "tiles": 16, "core": 32, "halo": 4},
        "drivaernet": {
            "grid_shape_zyx": [5, 3, 28], "tiles": 420,
            "core": 16, "halo": 4,
        },
        "airfoil": {
            "grid_shape_yx": [13, 25], "tiles": 325,
            "core": 32, "halo": 4, "offset_yx": [24, 25],
        },
        "output_pdf": str(OUTPUT.resolve()),
    }
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
