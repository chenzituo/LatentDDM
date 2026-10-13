"""Create the workshop-facing accuracy/interface-versus-NFE figure."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.input_dir / "iteration_metrics.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise SystemExit(f"no records in {csv_path}")
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[(record["method"], int(record["sweep"]))].append(record)
    metrics = (
        ("relative_l2", "relative L2"),
        ("residual_rms", "relative PDE residual"),
        ("value_jump", "overlap value RMS"),
        ("flux_jump", "overlap flux RMS"),
    )
    for x_name, x_label, basename in (
        ("local_nfe", "local neural evaluations (NFE)", "composition_curves"),
        ("wall_seconds", "cumulative wall time per case (s)", "composition_curves_time"),
    ):
        figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for method in sorted({key[0] for key in grouped}):
            entries = sorted(
                ((sweep, values) for (name, sweep), values in grouped.items() if name == method),
            )
            x_values = [
                np.mean([float(value[x_name]) for value in values])
                for _, values in entries
            ]
            for axis, (metric, title) in zip(axes.flat, metrics):
                mean = [
                    np.mean([float(value[metric]) for value in values])
                    for _, values in entries
                ]
                axis.plot(x_values, mean, marker="o", linewidth=1.5, label=method)
                axis.set_title(title)
                axis.set_xlabel(x_label)
                axis.grid(True, alpha=0.25)
        for axis in axes.flat:
            if any(np.any(np.asarray(line.get_ydata()) > 0.0) for line in axis.lines):
                axis.set_yscale("log")
        axes.flat[0].legend(frameon=False)
        for suffix in ("png", "pdf"):
            figure.savefig(args.input_dir / f"{basename}.{suffix}", dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    main()
