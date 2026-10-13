"""Aggregate and plot the three-seed Darcy two-level experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "outputs"
SEEDS = (20260805, 20260806, 20260807)
DOMAINS = tuple("ABCD")
METHODS = (
    ("local_pou", "Local POU", "#64748b"),
    ("scalar_energy", "Scalar energy", "#f59e0b"),
    ("qkv", "QKV communication", "#a855f7"),
    ("galerkin_constant", "Constant coarse", "#0ea5e9"),
    ("galerkin_linear", "Linear coarse", "#059669"),
)


def main() -> None:
    qkv = json.loads((ROOT / "darcy_proposal_communication_multiseed.json").read_text())
    values: dict[str, dict[str, dict[str, list[float]]]] = {
        domain: {method: {} for method, _, _ in METHODS} for domain in DOMAINS
    }
    metrics = ("relative_l2", "energy_relative_error", "residual_rms", "total_seconds")
    for domain in DOMAINS:
        for seed in SEEDS:
            summary = json.loads((
                ROOT / f"two_level_galerkin_seed{seed}" / f"domain_{domain}" / "summary.json"
            ).read_text())["summary"]
            for method, _, _ in METHODS:
                if method == "qkv":
                    continue
                for metric in metrics:
                    values[domain][method].setdefault(metric, []).append(
                        summary[method][metric]
                    )
        qkv_domain = qkv["domains"][domain]["communication"]
        values[domain]["qkv"] = {
            "relative_l2": [qkv_domain["relative_l2_mean"]],
            "energy_relative_error": [qkv_domain["energy_relative_error_mean"]],
            "residual_rms": [qkv_domain["residual_rms_mean"]],
            "total_seconds": [qkv_domain["wall_seconds_mean"]],
        }

    output = {"contract": "darcy_two_level_three_seed_aggregate_v1", "domains": {}}
    for domain in DOMAINS:
        output["domains"][domain] = {}
        local = np.mean(values[domain]["local_pou"]["relative_l2"])
        for method, _, _ in METHODS:
            output["domains"][domain][method] = {
                metric: {
                    "mean": float(np.mean(values[domain][method][metric])),
                    "std": float(np.std(values[domain][method][metric])),
                }
                for metric in metrics
            }
            mean = output["domains"][domain][method]["relative_l2"]["mean"]
            output["domains"][domain][method]["relative_l2_change_from_local"] = mean / local - 1.0
    output["equal_domain"] = {}
    for method, _, _ in METHODS:
        output["equal_domain"][method] = {
            metric: float(np.mean([
                output["domains"][domain][method][metric]["mean"] for domain in DOMAINS
            ]))
            for metric in metrics
        }
    base = output["equal_domain"]["local_pou"]["relative_l2"]
    for method, _, _ in METHODS:
        output["equal_domain"][method]["relative_l2_change_from_local"] = (
            output["equal_domain"][method]["relative_l2"] / base - 1.0
        )
    output["energy_nonincrease_fraction"] = {
        method: float(np.mean([
            json.loads((
                ROOT / f"two_level_galerkin_seed{seed}" / f"domain_{domain}" / "summary.json"
            ).read_text())["summary"][method]["energy_nonincrease_fraction"]
            for seed in SEEDS for domain in DOMAINS
        ]))
        for method in ("scalar_energy", "galerkin_constant", "galerkin_linear")
    }
    (ROOT / "darcy_two_level_galerkin_multiseed.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    x = np.arange(len(DOMAINS))
    width = 0.16
    for index, (method, label, color) in enumerate(METHODS):
        means = [output["domains"][domain][method]["relative_l2"]["mean"] for domain in DOMAINS]
        stds = [output["domains"][domain][method]["relative_l2"]["std"] for domain in DOMAINS]
        axes[0].bar(x + (index - 2) * width, means, width, yerr=stds, label=label, color=color)
    axes[0].set_xticks(x, DOMAINS)
    axes[0].set_ylabel("One-pass relative L2")
    axes[0].set_xlabel("Held-out topology")
    axes[0].set_title("Structured correction removes topology-B regression")
    axes[0].legend(fontsize=8, ncol=2)

    selected = ("local_pou", "qkv", "galerkin_constant", "galerkin_linear")
    labels = [next(label for key, label, _ in METHODS if key == method) for method in selected]
    colors = [next(color for key, _, color in METHODS if key == method) for method in selected]
    metric_labels = ("Field L2", "Energy error", "FV residual")
    metric_keys = ("relative_l2", "energy_relative_error", "residual_rms")
    positions = np.arange(len(metric_keys))
    for index, method in enumerate(selected):
        axes[1].bar(
            positions + (index - 1.5) * 0.2,
            [output["equal_domain"][method][metric] for metric in metric_keys],
            0.2, label=labels[index], color=colors[index],
        )
    axes[1].set_xticks(positions, metric_labels)
    axes[1].set_ylabel("Equal-domain mean (lower is better)")
    axes[1].set_title("Exact coarse solve improves every physics metric")
    axes[1].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(ROOT / f"darcy_two_level_galerkin_multiseed.{suffix}", dpi=220)
    plt.close(figure)
    print(json.dumps(output["equal_domain"], indent=2))


if __name__ == "__main__":
    main()

