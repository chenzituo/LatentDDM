"""Build the locked four-panel LatentDDM evidence summary.

The script consumes only the boundary-input protocol: normalized overlapping
assembly, two fixed local rounds for LatentDDM, and Init-SNI starting from the
identical Local/Core field.  Airfoil uncertainty is clustered by trajectory;
Car uncertainty is clustered by vehicle design; Darcy adaptation uncertainty
is across the three predeclared training seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "codiffuse_Schwarz/figures/latentddm_claim_summary_v3"
CAR_SELECTED = (
    "codiffuse_Schwarz_Car/outputs/"
    "input_correction_response_selected_v1"
)
AIR_SELECTED = (
    "codiffuse_Schwarz_Airfoil/outputs/state_input_selected_final_v3"
)

METHODS = ("Local/Core", "LatentDDM", "Init-SNI", "Point", "Slice", "Patch")
COLORS = {
    "Local/Core": "#8A8A8A",
    "LatentDDM": "#D94F4F",
    "Init-SNI": "#3E78B2",
    "Point": "#E69F32",
    "Slice": "#4F9D69",
    "Patch": "#7651A8",
}


def load(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def darcy_adaptation():
    local = load(
        "codiffuse_Schwarz_Darcy/outputs/input_correction_domain_extension_v1/"
        "aggregate.json"
    )
    global_ = load(
        "codiffuse_Schwarz_Darcy/outputs/field_token_domain_extension_degree1_v1/"
        "depth4/comparison_aggregate_with_point.json"
    )
    budgets = np.asarray((4, 16, 64), dtype=np.float64)
    curves = {
        "LatentDDM": (
            np.asarray([local["budgets"][str(int(b))]["relative_l2_mean"]
                        for b in budgets]),
            np.asarray([local["budgets"][str(int(b))]["relative_l2_sample_std"]
                        for b in budgets]),
        ),
    }
    names = {
        "Point Transformer full fine-tune": "Point",
        "Transolver full fine-tune": "Slice",
    }
    for source, label in names.items():
        rows = sorted(
            (row for row in global_["rows"] if row["method"] == source),
            key=lambda row: int(row["budget"]),
        )
        curves[label] = (
            np.asarray([row["adapted_relative_l2_mean"] for row in rows]),
            np.asarray([row["adapted_relative_l2_sample_std"] for row in rows]),
        )
    # The old aggregate used an 8x8 ViT. The locked comparison instead uses
    # the already-completed capacity-matched 32x32 Patch Transformer, equal to
    # the owned local-core size on the source domain.
    patch_mean, patch_std = [], []
    for budget in budgets.astype(int):
        values = []
        for seed in (20260810, 20260811, 20260812):
            payload = load(
                "codiffuse_Schwarz_Darcy/outputs/domain_extension_128_v1/"
                f"patch_core32_capacitymatched_v1/budget{budget}/"
                f"patch_seed{seed}/validation/summary.json"
            )
            if payload["model_spec"]["patch_size"] != [32, 32]:
                raise ValueError("Darcy Patch adaptation is not core-size matched")
            values.append(float(
                payload["summary"]["validation/sweep_001"]["relative_l2"]
            ))
        patch_mean.append(float(np.mean(values)))
        patch_std.append(float(np.std(values, ddof=1)))
    curves["Patch"] = (np.asarray(patch_mean), np.asarray(patch_std))
    return budgets, curves, float(local["local_core_relative_l2"])


def car_records():
    matched = load(f"{CAR_SELECTED}/matched_validation_sni_v1.json")
    frontier = load(f"{CAR_SELECTED}/paper_boundary_input_frontier_v1.json")
    if frontier.get("contract") != "car_input_correction_six_method_frontier_v1":
        raise ValueError("Car plot requires the strict boundary-input frontier")
    records = {}
    for label in METHODS:
        metric = frontier["summary"][label]["velocity_relative_l2"]
        records[label] = {
            "mean": float(metric["mean"]),
            "ci": tuple(metric["design_bootstrap_ci95"]),
        }
    return records, matched, int(frontier["selected_sni_sweeps"])


def _force_relative(rows: list[dict[str, Any]]) -> float:
    prediction = np.asarray([
        [row["drag_prediction"], row["lift_prediction"]] for row in rows
    ])
    reference = np.asarray([
        [row["drag_reference"], row["lift_reference"]] for row in rows
    ])
    return float(
        np.linalg.norm(prediction - reference)
        / max(np.linalg.norm(reference), 1.0e-20)
    )


def airfoil_clusters():
    local = load(f"{AIR_SELECTED}/final_holdout_h20_five_starts_v3.json")
    global_ = load(f"{AIR_SELECTED}/global_final_holdout_h20_five_starts_v3.json")
    manifest = load(
        "codiffuse_Schwarz_Airfoil/outputs/"
        "state_input_final_holdout_v3_manifest.json"
    )
    names = list(manifest["selected"])
    metrics = ("field", "vorticity", "interface", "divergence", "force")
    clustered: dict[str, dict[str, dict[str, float]]] = {}
    local_map = {
        "Local/Core": "local_core", "LatentDDM": "latentddm",
        "Init-SNI": "init_sni_1",
    }
    for label, key in local_map.items():
        clustered[label] = {metric: {} for metric in metrics}
        for trajectory, name in enumerate(names):
            windows = [row[key] for row in local["per_window"]
                       if int(row["trajectory"]) == trajectory]
            for metric in metrics:
                clustered[label][metric][name] = float(np.mean([
                    window["summary"][f"all_step_{metric}"] for window in windows
                ]))

    global_map = {
        "Point": "point_anchor", "Slice": "transolver",
        "Patch": "patch_transformer",
    }
    rows = global_["rows"]["rollout"]
    for label, key in global_map.items():
        clustered[label] = {metric: {} for metric in metrics}
        for name in names:
            trajectory_rows = [row for row in rows
                               if row["method"] == key and row["trajectory"] ==
                               Path(name).stem]
            starts = sorted({int(row["start"]) for row in trajectory_rows})
            windows = []
            for start in starts:
                window = sorted(
                    (row for row in trajectory_rows if int(row["start"]) == start),
                    key=lambda row: int(row["step"]),
                )
                windows.append({
                    "field": float(np.mean([row["field_relative_l2"] for row in window])),
                    "vorticity": float(np.mean([
                        row["vorticity_relative_l2"] for row in window
                    ])),
                    "interface": float(np.mean([
                        row["interface_relative_l2"] for row in window
                    ])),
                    "divergence": float(np.mean([
                        row["divergence_rms"] for row in window
                    ])),
                    "force": _force_relative(window),
                })
            for metric in metrics:
                clustered[label][metric][name] = float(np.mean([
                    window[metric] for window in windows
                ]))
    return clustered, names, local


def ratio_interval(
    method: dict[str, float], local: dict[str, float], names: list[str],
    *, seed: int, draws: int = 20_000,
):
    numerator = np.asarray([method[name] for name in names])
    denominator = np.asarray([local[name] for name in names])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(names), size=(draws, len(names)))
    ratios = numerator[indices].mean(1) / denominator[indices].mean(1)
    mean = float(numerator.mean() / denominator.mean())
    return mean, tuple(np.quantile(ratios, (.025, .975)))


def sni_path(summary: dict[str, Any], metric: str, suffix: str):
    records = []
    for name, row in summary.items():
        if not name.startswith("init_sni_"):
            continue
        sweep = int(name.rsplit("_", 1)[1])
        records.append((sweep + 1, float(row[f"{metric}_{suffix}"])))
    return tuple(zip(*sorted(records)))


def make_figure() -> dict[str, Any]:
    budgets, adaptation, local_darcy = darcy_adaptation()
    car, car_matched, car_sweep = car_records()
    air, trajectory_names, air_local = airfoil_clusters()

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.2))
    ax = axes[0, 0]
    for label in ("LatentDDM", "Point", "Slice", "Patch"):
        mean, std = adaptation[label]
        ax.errorbar(
            budgets, mean, yerr=std, marker="o", linewidth=2, capsize=2.5,
            color=COLORS[label], label=label,
        )
    ax.axhline(local_darcy, color=COLORS["Local/Core"], linestyle="--",
               linewidth=1.5, label="Local/Core")
    ax.set(xscale="log", yscale="log", xticks=budgets,
           xlabel="target-domain simulations", ylabel="validation relative $L^2$",
           title="a  Reuse on enlarged Darcy")
    ax.set_xticklabels([str(int(value)) for value in budgets])
    ax.grid(alpha=.2, which="both")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[0, 1]
    positions = np.arange(len(METHODS))
    means = np.asarray([car[name]["mean"] for name in METHODS])
    lower = means - np.asarray([car[name]["ci"][0] for name in METHODS])
    upper = np.asarray([car[name]["ci"][1] for name in METHODS]) - means
    ax.axvspan(-.52, 2.48, color="#F8EAEA", alpha=.75, zorder=0)
    ax.axvspan(2.52, 5.52, color="#EAF1FA", alpha=.8, zorder=0)
    bars = ax.bar(positions, means, color=[COLORS[name] for name in METHODS],
                  yerr=np.vstack((lower, upper)), capsize=2.5, width=.72)
    best = int(np.argmin(means))
    bars[best].set_edgecolor("black"); bars[best].set_linewidth(1.8)
    for index, value in enumerate(means):
        ax.text(index, value + upper[index] + .00025, f"{value:.4f}",
                ha="center", va="bottom", fontsize=7,
                fontweight="bold" if index == best else "normal")
    ax.text(1.0, .96, "LOCAL METHODS", transform=ax.get_xaxis_transform(),
            color="#B44B4B", ha="center", va="top", fontsize=8,
            fontweight="bold")
    ax.text(4.0, .96, "GLOBAL METHODS", transform=ax.get_xaxis_transform(),
            color="#376AA5", ha="center", va="top", fontsize=8,
            fontweight="bold")
    ax.set_xticks(positions, ["Local/Core", "LatentDDM", f"SNI-{car_sweep}",
                             "Point", "Slice", "Patch"], rotation=25)
    ax.set(ylabel="velocity relative $L^2$", title="b  Car: paired 25-design audit")
    ax.grid(axis="y", alpha=.2)

    ax = axes[1, 0]
    metric_labels = ("field", "vorticity", "interface", "divergence", "force")
    display_metrics = ("field", r"$\omega$", "trace", "div.", "force")
    ratios_payload: dict[str, dict[str, float]] = {}
    ratio_ci_payload: dict[str, dict[str, list[float]]] = {}
    ratio_matrix = np.empty((len(METHODS), len(metric_labels)))
    for method_index, method in enumerate(METHODS):
        for metric_index, metric in enumerate(metric_labels):
            value, ci = ratio_interval(
                air[method][metric], air["Local/Core"][metric], trajectory_names,
                seed=20260811 + method_index * 17 + metric_index,
            )
            ratio_matrix[method_index, metric_index] = value
            ratio_ci_payload.setdefault(method, {})[metric] = list(map(float, ci))
        ratios_payload[method] = dict(zip(
            metric_labels, ratio_matrix[method_index].tolist(),
        ))
    image = ax.imshow(
        ratio_matrix,
        aspect="auto",
        cmap="RdYlGn_r",
        norm=TwoSlopeNorm(vmin=min(.45, float(ratio_matrix.min())), vcenter=1.0,
                          vmax=max(1.45, float(ratio_matrix.max()))),
    )
    best_by_metric = np.argmin(ratio_matrix, axis=0)
    for row in range(ratio_matrix.shape[0]):
        for column in range(ratio_matrix.shape[1]):
            value = ratio_matrix[row, column]
            ax.text(
                column, row, f"{value:.2f}", ha="center", va="center",
                fontsize=8, color="white" if value < .62 or value > 1.35 else "#222222",
                fontweight="bold" if row == best_by_metric[column] else "normal",
            )
    ax.set_xticks(np.arange(len(display_metrics)), display_metrics)
    ax.set_yticks(np.arange(len(METHODS)), METHODS)
    ax.get_yticklabels()[1].set_color(COLORS["LatentDDM"])
    ax.get_yticklabels()[1].set_fontweight("bold")
    ax.set_title("c  Airfoil: held-out 20-step error / Local-Core")
    colorbar = fig.colorbar(image, ax=ax, fraction=.035, pad=.03)
    colorbar.set_label("error ratio", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    ax = axes[1, 1]
    darcy_sni = load(
        "codiffuse_Schwarz_Darcy/outputs/input_correction_domain_extension_v1/"
        "matched_normalized_overlap_sni_128.json"
    )
    cases = {
        "Darcy": (darcy_sni["summary"], "relative_l2", "mean",
                  float(darcy_sni["summary"]["latentddm"]["relative_l2_mean"])),
        "Car": (car_matched["summary"], "velocity_relative_l2", "mean",
                float(car_matched["summary"]["latentddm"]["velocity_relative_l2_mean"])),
        "Airfoil": (air_local["summary"], "all_step_field", "mean",
                    float(air_local["summary"]["latentddm"]["all_step_field_mean"])),
    }
    case_colors = {"Darcy": "#355C9A", "Car": "#4F9D69", "Airfoil": "#C55A7A"}
    path_payload = {}
    for case, (summary, metric, suffix, latent) in cases.items():
        rounds, errors = sni_path(summary, metric, suffix)
        errors = np.asarray(errors)
        relative_to_latent = errors / latent
        ax.plot(rounds, relative_to_latent, marker="o", markersize=3.5,
                color=case_colors[case], label=case)
        path_payload[case] = {
            "rounds": list(map(int, rounds)),
            "sni_error_over_latentddm": relative_to_latent.tolist(),
            "latent_rounds": 2, "latent_error_over_latentddm": 1.0,
        }
    ax.scatter(
        [2], [1.0], marker="D", s=45, color="0.25", edgecolor="black",
        linewidth=.6, zorder=4,
    )
    ax.axhline(1.0, color="0.45", linewidth=.8, linestyle="--", zorder=0)
    ax.set(xscale="log", yscale="log", xlabel="dependent local-operator rounds",
           ylabel="field error / LatentDDM error",
           title="d  SNI convergence relative to one pass")
    ax.grid(alpha=.2, which="both")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D(
        [], [], color="0.25", marker="D", linestyle="None", markersize=5,
        markeredgecolor="black", markeredgewidth=.6,
    ))
    labels.append("LatentDDM (2 rounds)")
    ax.legend(handles, labels, frameon=False, fontsize=7, ncol=2)

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
        axis.title.set_fontsize(11)
        axis.title.set_fontweight("bold")
    fig.tight_layout(pad=.7, w_pad=1.1, h_pad=1.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    payload = {
        "contract": "latentddm_boundary_input_claim_summary_v6_frozen_final",
        "darcy_adaptation": {
            label: {
                "budgets": budgets.astype(int).tolist(),
                "relative_l2_mean": mean.tolist(),
                "relative_l2_sample_std": std.tolist(),
                **(
                    {"patch_size": [32, 32]}
                    if label == "Patch" else {}
                ),
            }
            for label, (mean, std) in adaptation.items()
        },
        "airfoil_trajectories": trajectory_names,
        "airfoil_ratios": ratios_payload,
        "airfoil_ratio_trajectory_bootstrap_ci95": ratio_ci_payload,
        "car": car,
        "car_sni_sweep": car_sweep,
        "dependent_round_paths": path_payload,
    }
    OUT.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(make_figure(), indent=2))
