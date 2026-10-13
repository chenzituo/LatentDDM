"""Aggregate the exact-offset one-pass versus iterative Darcy frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("pou", "ras", "sni_additive", "pou_aitken")
METRICS = ("relative_l2", "energy_relative_error", "residual_rms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composer-dir", type=Path, required=True)
    parser.add_argument("--theorem-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0]
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in keys if isinstance(rows[0][key], (int, float))
    }


def main(args: argparse.Namespace) -> dict[str, object]:
    domains = "ABCD"
    iterative = {}
    sources = {}
    for domain in domains:
        source = (
            args.composer_dir
            / f"matched_offsets_iterative_baselines_{domain}"
            / "summary.json"
        )
        if not source.exists():
            raise FileNotFoundError(source)
        payload = json.loads(source.read_text())
        if payload.get("huang_assembly") != "raw_zero_extension_sum" \
                or payload.get("snap_boundary_contract") != "robin_v3":
            raise ValueError(
                f"{source} is a quarantined legacy iterative artifact: exact "
                "Huang requires raw zero-extension assembly and neural SNAP "
                "requires a distinct Robin-v3 local checkpoint"
            )
        iterative[domain] = payload["summary"]
        sources[domain] = str(source.resolve())

    evaluation_rows = []
    for domain in domains:
        source = args.composer_dir / f"evaluation_{domain}" / "summary.json"
        evaluation_rows.append(json.loads(source.read_text())["summary"])
    one_pass_runtime = mean_dict(evaluation_rows)
    theorem = json.loads(args.theorem_summary.read_text())["variant_summary"]

    curves: dict[str, list[dict[str, float]]] = {}
    for method in METHODS:
        curve = []
        for sweep in range(1, 65):
            key = f"{method}/sweep_{sweep:03d}"
            curve.append(mean_dict([iterative[domain][key] for domain in domains]))
        curves[method] = curve

    one_pass = {
        "local_pou": {
            **{metric: float(theorem["local"][metric]) for metric in METRICS},
            "local_nfe": float(curves["pou"][0]["local_nfe"]),
            "wall_seconds": float(curves["pou"][0]["wall_seconds"]),
        },
        "learned_affine": {
            **{metric: float(theorem["predicted_affine"][metric]) for metric in METRICS},
            "local_nfe": float(one_pass_runtime["local_nfe"]),
            "wall_seconds": float(one_pass_runtime["wall_seconds"]),
        },
        "target_fitted_affine_oracle": {
            **{metric: float(theorem["oracle_affine"][metric]) for metric in METRICS},
            "local_nfe": float(one_pass_runtime["local_nfe"]),
        },
    }

    result: dict[str, object] = {
        "contract": "darcy_exact_offset_ddm_frontier_v1",
        "domains": list(domains),
        "offsets": [[0, 0], [8, 8], [16, 16]],
        "equal_domain_weighting": True,
        "same_embedded_local_endpoint": True,
        "one_pass": one_pass,
        "iterative_curves": curves,
        "labels": {
            "sni_additive": "literal Huang raw-additive SNI, validation-selected tau",
            "ras": (
                "Dirichlet neural RAS; SNAP boundary-contract ablation only, "
                "not Mao et al.'s Robin-OSM reproduction"
            ),
            "pou_aitken": "Aitken-accelerated repeated POU",
            "target_fitted_affine_oracle": "evaluation-only chart oracle",
        },
        "sources": sources,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "darcy_exact_offset_ddm_frontier.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    labels = {
        "pou": "repeated POU",
        "ras": "Dirichlet neural RAS",
        "sni_additive": "additive SNI",
        "pou_aitken": "Aitken POU",
    }
    colors = {
        "pou": "#9d755d", "ras": "#72b7b2",
        "sni_additive": "#e45756", "pou_aitken": "#b279a2",
    }
    ylabels = {
        "relative_l2": r"relative $L^2$",
        "energy_relative_error": "relative energy error",
        "residual_rms": "FV residual RMS",
    }
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.15), constrained_layout=True)
    for axis, metric in zip(axes, METRICS):
        for method in METHODS:
            curve = curves[method]
            axis.plot(
                [row["local_nfe"] for row in curve],
                [row[metric] for row in curve],
                color=colors[method], linewidth=1.5, label=labels[method],
            )
        axis.scatter(
            [one_pass["local_pou"]["local_nfe"]],
            [one_pass["local_pou"][metric]], marker="o", s=42,
            color="#6b7280", label="local POU (one pass)", zorder=6,
        )
        axis.scatter(
            [one_pass["learned_affine"]["local_nfe"]],
            [one_pass["learned_affine"][metric]], marker="s", s=48,
            color="#2563eb", label="learned token (one pass)", zorder=7,
        )
        axis.scatter(
            [one_pass["target_fitted_affine_oracle"]["local_nfe"]],
            [one_pass["target_fitted_affine_oracle"][metric]], marker="D", s=36,
            facecolors="none", edgecolors="#f59e0b", linewidths=1.4,
            label="target-fitted oracle", zorder=7,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("cumulative local neural evaluations")
        axis.set_ylabel(ylabels[metric])
        axis.grid(alpha=.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=4,
                  frameon=False, fontsize=8)
    for suffix in ("pdf", "png"):
        figure.savefig(args.output_dir / f"darcy_exact_offset_ddm_frontier.{suffix}",
                       dpi=240, bbox_inches="tight")
    plt.close(figure)
    return result


if __name__ == "__main__":
    print(json.dumps(main(parse_args()), indent=2))
