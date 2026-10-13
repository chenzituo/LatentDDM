"""Visual audit of every stored Darcy modality on held-out A--D domains."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 10, figsize=(25, 10), constrained_layout=True)
    with h5py.File(args.data, "r") as handle:
        for row, domain in enumerate(("A", "B", "C", "D")):
            group = handle[f"domains/{domain}"]
            index = min(args.case, group["solution"].shape[0] - 1)
            mask = group["mask"][index, 0]
            normals = group["normals"][index]
            fields = (
                (mask, "mask"),
                (group["logk"][index, 0], "log10 k"),
                (group["source"][index, 0], "source"),
                (group["bc_mask"][index, 0], "boundary mask"),
                (group["bc_value"][index, 0], "boundary value"),
                (group["sdf"][index, 0], "SDF"),
                (normals[0], "normal x"),
                (normals[1], "normal y"),
                (group["bc_role"][index, 0], "boundary role"),
                (group["solution"][index, 0], "FV solution"),
            )
            for column, (field, name) in enumerate(fields):
                image = axes[row, column].imshow(field, origin="lower", cmap="viridis")
                axes[row, column].set_title(f"{domain}: {name}")
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                figure.colorbar(image, ax=axes[row, column], shrink=0.65)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
