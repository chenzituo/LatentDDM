"""Build the Darcy--Airfoil evidence summary for the two-case submission.

The figure intentionally separates the two physical regimes: target-domain
adaptation for elliptic Darcy and model-owned rollout for transport-dominated
Airfoil.  It consumes the frozen Darcy artifacts and the prospectively sealed
absolute-Robin Airfoil evaluation shared by all six methods.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

from codiffuse_Schwarz.plot_latentddm_claim_summary_v3 import (
    COLORS,
    darcy_adaptation,
    load,
    ratio_interval,
    sni_path,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "codiffuse_Schwarz/workshop/submission/Two_case/figures/"
    "two_case_claim_summary"
)
AIR_METHODS = ("Local/Core", "LatentDDM", "SNI", "Point", "Slice", "Patch")
AIR_DIR = (
    ROOT / "codiffuse_Schwarz_Airfoil/outputs/"
    "airfoil_absolute_robin_selected_v1/final_holdout"
)
AIR_NATIVE = (
    ROOT / "codiffuse_Schwarz_Airfoil/outputs/"
    "airfoil_native_h20_final_latent_relax025_v1/"
    "local_latent_sni_point_validation_h20_s3_calls1_2_4_8_16.json"
)


def native_rollout_curve(
    payload: dict[str, object], method: str, metric: str, *,
    replicates: int = 2000, seed: int = 20260814,
) -> dict[str, list[float]]:
    """Trajectory-cluster bootstrap curve for one locked native method."""
    records = payload["methods"][method]["records"]
    trajectories = sorted({int(row["trajectory"]) for row in records})
    steps = sorted({int(row["step"]) for row in records})
    values = np.asarray([
        [
            np.mean([
                float(row[metric]) for row in records
                if int(row["trajectory"]) == trajectory
                and int(row["step"]) == step
            ])
            for step in steps
        ]
        for trajectory in trajectories
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, len(steps)), dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(trajectories), size=len(trajectories))
        samples[index] = values[selected].mean(axis=0)
    return {
        "step": list(map(float, steps)),
        "mean": values.mean(axis=0).tolist(),
        "lower95": np.quantile(samples, 0.025, axis=0).tolist(),
        "upper95": np.quantile(samples, 0.975, axis=0).tolist(),
    }


def _window_force(rows: list[dict[str, object]]) -> float:
    drag_key = "c_drag_prediction" if "c_drag_prediction" in rows[0] \
        else "drag_prediction"
    lift_key = "c_lift_prediction" if "c_lift_prediction" in rows[0] \
        else "lift_prediction"
    drag_reference_key = "c_drag_reference" if "c_drag_reference" in rows[0] \
        else "drag_reference"
    lift_reference_key = "c_lift_reference" if "c_lift_reference" in rows[0] \
        else "lift_reference"
    prediction = np.asarray([
        [row[drag_key], row[lift_key]] for row in rows
    ], dtype=np.float64)
    reference = np.asarray([
        [row[drag_reference_key], row[lift_reference_key]] for row in rows
    ], dtype=np.float64)
    return float(
        np.linalg.norm(prediction - reference)
        / max(np.linalg.norm(reference), 1.0e-20)
    )


def airfoil_absolute_robin_clusters():
    """Read the prospectively paired absolute-Robin Airfoil evaluation."""
    manifest = json.loads((AIR_DIR.parent / "final_holdout_manifest.json").read_text())
    names = list(manifest["selected"])
    local_sources = {
        "Local/Core": "local_s1_h20_n5.json",
        "LatentDDM": "latent_h20_n5.json",
        "SNI": "sni_s8_tau025_h20_n5.json",
    }
    clustered: dict[str, dict[str, dict[str, float]]] = {}
    paths: dict[str, object] = {}
    for label, filename in local_sources.items():
        payload = json.loads((AIR_DIR / filename).read_text())
        paths[label] = payload
        rows = payload["records"]
        clustered[label] = {key: {} for key in ("field", "vorticity", "force")}
        for trajectory_index, name in enumerate(names):
            selected = [
                row for row in rows if int(row["trajectory"]) == trajectory_index
            ]
            final = [row for row in selected if int(row["step"]) == 20]
            clustered[label]["field"][name] = float(np.mean([
                row["field_relative_l2"] for row in final
            ]))
            clustered[label]["vorticity"][name] = float(np.mean([
                row["vorticity_relative_l2"] for row in final
            ]))
            starts = sorted({int(row["start"]) for row in selected})
            clustered[label]["force"][name] = float(np.mean([
                _window_force(sorted(
                    (row for row in selected if int(row["start"]) == start),
                    key=lambda row: int(row["step"]),
                )) for start in starts
            ]))

    global_payload = json.loads((AIR_DIR / "global_h20_n5.json").read_text())
    global_names = {
        "Point": "point_anchor",
        "Slice": "transolver",
        "Patch": "patch_transformer",
    }
    rows = global_payload["rows"]["rollout"]
    for label, method in global_names.items():
        clustered[label] = {key: {} for key in ("field", "vorticity", "force")}
        for name in names:
            stem = Path(name).stem
            selected = [
                row for row in rows
                if row["method"] == method and row["trajectory"] == stem
            ]
            final = [row for row in selected if int(row["step"]) == 20]
            clustered[label]["field"][name] = float(np.mean([
                row["field_relative_l2"] for row in final
            ]))
            clustered[label]["vorticity"][name] = float(np.mean([
                row["vorticity_relative_l2"] for row in final
            ]))
            starts = sorted({int(row["start"]) for row in selected})
            clustered[label]["force"][name] = float(np.mean([
                _window_force(sorted(
                    (row for row in selected if int(row["start"]) == start),
                    key=lambda row: int(row["step"]),
                )) for start in starts
            ]))
    return clustered, names, paths, global_payload


def make_figure() -> dict[str, object]:
    budgets, adaptation, local_darcy = darcy_adaptation()
    native = json.loads(AIR_NATIVE.read_text(encoding="utf-8"))

    figure = plt.figure(figsize=(15.0, 4.25))
    grid = figure.add_gridspec(
        1, 3, width_ratios=(1.0, 1.08, 1.22), wspace=0.34,
    )
    axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2]),
    )

    # Elliptic regime: adapt only the communicator on the enlarged domain.
    axis = axes[0]
    for label in ("LatentDDM", "Point", "Slice", "Patch"):
        mean, standard_deviation = adaptation[label]
        axis.errorbar(
            budgets,
            mean,
            yerr=standard_deviation,
            marker="o",
            linewidth=2,
            capsize=2.5,
            color=COLORS[label],
            label=label,
        )
    axis.axhline(
        local_darcy,
        color=COLORS["Local/Core"],
        linestyle="--",
        linewidth=1.5,
        label="Local/Core",
    )
    axis.set(
        xscale="log",
        yscale="log",
        xticks=budgets,
        xlabel="target-domain simulations",
        ylabel="validation relative $L^2$",
        title="a  Darcy domain extension",
    )
    axis.set_xticklabels([str(int(value)) for value in budgets])
    axis.grid(alpha=0.2, which="both")
    axis.legend(frameon=False, fontsize=8, ncol=2)

    # Transport regime: display the complete model-owned trajectory rather
    # than an ambiguous aggregate that can be mistaken for one-step error.
    axis = axes[1]
    native_labels = {
        "local_core": "Local/Core",
        "latentddm": "LatentDDM",
        "sni_calls_16": "SNI-16",
        "global_point": "Point",
    }
    native_colors = {
        "Local/Core": COLORS["Local/Core"],
        "LatentDDM": COLORS["LatentDDM"],
        "SNI-16": COLORS["Init-SNI"],
        "Point": COLORS["Point"],
    }
    native_curves: dict[str, dict[str, list[float]]] = {}
    for method_index, (method, label) in enumerate(native_labels.items()):
        curve = native_rollout_curve(
            native, method, "field_relative_l2",
            seed=20260814 + 31 * method_index,
        )
        native_curves[label] = curve
        step = np.asarray(curve["step"])
        mean = np.asarray(curve["mean"])
        lower = np.asarray(curve["lower95"])
        upper = np.asarray(curve["upper95"])
        axis.plot(
            step, mean, linewidth=2.1, color=native_colors[label],
            linestyle="--" if label == "Local/Core" else "-", label=label,
        )
        axis.fill_between(
            step, lower, upper, color=native_colors[label], alpha=0.10,
            linewidth=0,
        )
    axis.set(
        yscale="log",
        xlim=(1, 20),
        xticks=(1, 5, 10, 15, 20),
        xlabel="model-owned rollout step",
        ylabel="field relative $L^2$",
        title="b  Airfoil transport rollout",
    )
    axis.grid(alpha=0.2, which="both")
    axis.legend(frameon=False, fontsize=7.5, ncol=2)

    # The common DDM question: how many dependent local calls are needed to
    # reach the two-round learned composer?  The curves intentionally show the
    # contrasting fixed-point behavior of the two regimes.
    axis = axes[2]
    darcy_sni = load(
        "codiffuse_Schwarz_Darcy/outputs/input_correction_domain_extension_v1/"
        "matched_normalized_overlap_sni_128.json"
    )
    cases = {
        "Darcy": (
            darcy_sni["summary"],
            "relative_l2",
            "mean",
            float(darcy_sni["summary"]["latentddm"]["relative_l2_mean"]),
        ),
    }
    case_colors = {"Darcy": "#355C9A", "Airfoil": "#C55A7A"}
    paths: dict[str, object] = {}
    for case, (summary, metric, suffix, latent_error) in cases.items():
        rounds, errors = sni_path(summary, metric, suffix)
        relative = np.asarray(errors) / latent_error
        axis.plot(
            rounds,
            relative,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=case_colors[case],
            label=f"{case} SNI",
        )
        paths[case] = {
            "rounds": list(map(int, rounds)),
            "sni_error_over_latentddm": relative.tolist(),
        }
    air_latent_error = float(
        native["methods"]["latentddm"]["summary"]["step_horizon"][
            "field_relative_l2"
        ]
    )
    air_rounds = np.asarray((1, 2, 4, 8, 16), dtype=np.float64)
    air_errors = np.asarray([
        native["methods"][method]["summary"]["step_horizon"][
            "field_relative_l2"
        ]
        for method in (
            "local_core", "sni_calls_2", "sni_calls_4", "sni_calls_8",
            "sni_calls_16",
        )
    ], dtype=np.float64)
    air_relative = air_errors / air_latent_error
    axis.plot(
        air_rounds,
        air_relative,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        color=case_colors["Airfoil"],
        label="Airfoil SNI",
    )
    paths["Airfoil"] = {
        "rounds": air_rounds.astype(int).tolist(),
        "sni_error_over_latentddm": air_relative.tolist(),
    }
    axis.scatter(
        [2],
        [1.0],
        marker="D",
        s=45,
        color=COLORS["LatentDDM"],
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
    )
    axis.axhline(1.0, color="0.45", linewidth=0.8, linestyle="--", zorder=0)
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="sequential response stages",
        ylabel="field error / LatentDDM error",
        title="c  Interface search",
    )
    axis.grid(alpha=0.2, which="both")
    handles, labels = axis.get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            color=COLORS["LatentDDM"],
            marker="D",
            linestyle="None",
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.6,
        )
    )
    labels.append("LatentDDM (one exchange)")
    axis.legend(handles, labels, frameon=False, fontsize=7.5, ncol=1)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8)
        axis.title.set_fontsize(11)
        axis.title.set_fontweight("bold")
    figure.tight_layout(pad=0.55)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)

    payload: dict[str, object] = {
        "contract": "latentddm_two_case_locked_evidence_v1",
        "darcy_adaptation": {
            label: {
                "budgets": budgets.astype(int).tolist(),
                "relative_l2_mean": mean.tolist(),
                "relative_l2_sample_std": standard_deviation.tolist(),
            }
            for label, (mean, standard_deviation) in adaptation.items()
        },
        "airfoil_native_source": str(AIR_NATIVE),
        "airfoil_h1_h20_trajectory_bootstrap_curves": native_curves,
        "dependent_round_paths": paths,
    }
    OUT.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(make_figure(), indent=2))
