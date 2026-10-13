#!/usr/bin/env python3
"""Summarize the matched local/communicated full-domain Darcy result."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "outputs"
RUNS = {
    "local_pou": ROOT / "full_domain_local_pou_seed20260805",
    "one_message": ROOT / "full_domain_relative_comm_seed20260805",
}
LEGACY = {"A": 0.5692215995, "B": 0.4323344054, "C": 0.4444648928, "D": 0.3917832296}
METRICS = (
    "relative_l2", "energy_relative_error", "residual_rms",
    "value_jump", "flux_jump", "wall_seconds",
)


def load_records(run: Path, domain: str) -> list[dict[str, str]]:
    path = run / f"domain_{domain}_composition" / "iteration_metrics.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarize() -> dict:
    result = {
        "contract": "darcy_matched_full96_two_stage_communication_v2",
        "communication_call_sites": ["proposal", "projector"],
        "schema_note": "one_message keys are retained for artifact compatibility",
        "cases_per_domain": 100,
        "cover_offsets": [[0, 0], [8, 8], [16, 16]],
        "domains": {},
    }
    for domain in "ABCD":
        domain_result = {"legacy_local_one_pass_relative_l2": LEGACY[domain]}
        for label, run in RUNS.items():
            records = load_records(run, domain)
            first = [row for row in records if int(row["sweep"]) == 1]
            metrics = {
                metric: float(np.mean([float(row[metric]) for row in first]))
                for metric in METRICS
            }
            offset_l2 = {
                key: float(np.mean([
                    float(row["relative_l2"]) for row in first
                    if (int(row["offset_y"]), int(row["offset_x"])) == offset
                ]))
                for key, offset in {
                    "0,0": (0, 0), "8,8": (8, 8), "16,16": (16, 16),
                }.items()
            }
            metrics["offset_relative_l2"] = offset_l2
            metrics["offset_std"] = float(np.std(list(offset_l2.values())))
            curves = {}
            for sweep in range(1, 17):
                selected = [row for row in records if int(row["sweep"]) == sweep]
                curves[str(sweep)] = float(np.mean([
                    float(row["relative_l2"]) for row in selected
                ]))
            metrics["relative_l2_by_sweep"] = curves
            domain_result[label] = metrics
        local = domain_result["local_pou"]["relative_l2"]
        communicated = domain_result["one_message"]["relative_l2"]
        domain_result["one_message_relative_change"] = communicated / local - 1.0
        result["domains"][domain] = domain_result
    for label in RUNS:
        result[f"mean_{label}_one_pass_relative_l2"] = float(np.mean([
            result["domains"][domain][label]["relative_l2"] for domain in "ABCD"
        ]))
    result["mean_one_message_relative_change"] = (
        result["mean_one_message_one_pass_relative_l2"]
        / result["mean_local_pou_one_pass_relative_l2"] - 1.0
    )
    return result


def plot(result: dict, output: Path) -> None:
    domains = list("ABCD")
    colors = {"legacy": "#aab2bd", "local": "#4c78a8", "comm": "#e45756"}
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), constrained_layout=True)
    x = np.arange(4); width = 0.25
    axes[0, 0].bar(x - width, [LEGACY[d] for d in domains], width, label="legacy local", color=colors["legacy"])
    axes[0, 0].bar(x, [result["domains"][d]["local_pou"]["relative_l2"] for d in domains], width, label="full96 local POU", color=colors["local"])
    axes[0, 0].bar(x + width, [result["domains"][d]["one_message"]["relative_l2"] for d in domains], width, label="two-stage comm. + POU", color=colors["comm"])
    axes[0, 0].set(title="One-pass held-out error", ylabel="relative $L_2$", xticks=x, xticklabels=domains)
    axes[0, 0].legend(frameon=False, fontsize=8)

    for metric, title, axis in (
        ("energy_relative_error", "Discrete energy error", axes[0, 1]),
        ("residual_rms", "Relative FV residual", axes[1, 0]),
    ):
        axis.bar(x - width / 2, [result["domains"][d]["local_pou"][metric] for d in domains], width, label="local POU", color=colors["local"])
        axis.bar(x + width / 2, [result["domains"][d]["one_message"][metric] for d in domains], width, label="two-stage comm.", color=colors["comm"])
        axis.set(title=title, xticks=x, xticklabels=domains)
        axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    palette = dict(zip(domains, ("#4c78a8", "#f58518", "#54a24b", "#b279a2")))
    sweeps = np.arange(1, 17)
    for domain in domains:
        for label, linestyle in (("local_pou", "--"), ("one_message", "-")):
            values = result["domains"][domain][label]["relative_l2_by_sweep"]
            axis.plot(sweeps, [values[str(s)] for s in sweeps], linestyle, color=palette[domain], linewidth=1.6, label=f"{domain} {'local' if label == 'local_pou' else 'two-stage'}")
    axis.set(title="Post-hoc repeated composition", xlabel="sweep", ylabel="relative $L_2$", xticks=(1, 4, 8, 12, 16))
    axis.legend(ncol=2, frameon=False, fontsize=7)
    for axis in axes.flat:
        axis.grid(alpha=0.18, axis="y")
    figure.savefig(output, dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    result = summarize()
    output = ROOT / "darcy_communication_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot(result, ROOT / "darcy_communication_comparison.png")
    print(json.dumps({
        "output": str(output),
        "mean_local": result["mean_local_pou_one_pass_relative_l2"],
        "mean_message": result["mean_one_message_one_pass_relative_l2"],
        "relative_change": result["mean_one_message_relative_change"],
    }, indent=2))
