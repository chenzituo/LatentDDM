"""Plot the verified Darcy adaptation and native Airfoil H20 frontier.

The Airfoil panel accepts only the matched native artifact containing the
selected Local/Core endpoint, its LatentDDM composer, target-free SNI, and the
independent whole-domain Point model.  It is intentionally not runnable from
the older coarse-observation or six-method artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

from codiffuse_Schwarz.plot_latentddm_claim_summary_v3 import (
    COLORS,
    darcy_adaptation,
)
from codiffuse_Schwarz_Airfoil.verify_native_final_h20 import (
    verify as verify_airfoil_h20,
)


AIRFOIL_CONTRACT = "airfoil_native_point_matched_h20_v1"
DISPLAY = {
    "local_core": "Local/Core",
    "latentddm": "LatentDDM",
    "global_point": "Point",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(payload: dict[str, object], method: str) -> dict[str, float]:
    return payload["methods"][method]["summary"]["all_steps"]


def _terminal(payload: dict[str, object], method: str) -> dict[str, float]:
    return payload["methods"][method]["summary"]["step_horizon"]


def _validate(payload: dict[str, object]) -> str:
    if payload.get("contract") != AIRFOIL_CONTRACT:
        raise ValueError("figure requires the matched native Airfoil H20 artifact")
    if payload.get("protected_test_trajectories_opened") != 0:
        raise ValueError("figure input opened the protected Airfoil test split")
    if payload.get("reference_used_by_inference") is not False:
        raise ValueError("figure input used a reference field during inference")
    if payload.get("same_data_split_statistics_and_validation_windows") is not True:
        raise ValueError("figure input violates the matched-window contract")
    for method in ("local_core", "latentddm", "global_point"):
        if method not in payload.get("methods", {}):
            raise ValueError(f"figure input omits {method}")
    acceptance = payload.get("sni_acceptance", {})
    if acceptance.get("benchmark_goal_field_and_convergence_passed") is not True:
        raise ValueError("figure input has no accepted target-free SNI point")
    accepted = acceptance.get("accepted_sni_method")
    if accepted not in payload["methods"]:
        raise ValueError("accepted SNI method is missing")
    return str(accepted)


def make_figure(airfoil_path: Path, output_base: Path) -> dict[str, object]:
    verification = verify_airfoil_h20(airfoil_path)
    payload = json.loads(airfoil_path.read_text(encoding="utf-8"))
    accepted_sni = _validate(payload)
    budgets, adaptation, local_darcy = darcy_adaptation()

    plt.rcParams.update({
        "font.size": 8.2,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.4,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure = plt.figure(figsize=(10.0, 2.75))
    grid = figure.add_gridspec(1, 3, width_ratios=(1.08, 1.0, 1.08), wspace=0.36)
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]

    axis = axes[0]
    for label in ("LatentDDM", "Point", "Slice", "Patch"):
        mean, deviation = adaptation[label]
        axis.errorbar(
            budgets, mean, yerr=deviation, marker="o", markersize=3.2,
            linewidth=1.5, capsize=2.0, color=COLORS[label], label=label,
        )
    axis.axhline(
        local_darcy, color=COLORS["Local/Core"], linestyle="--",
        linewidth=1.2, label="Local/Core",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(budgets, [str(int(value)) for value in budgets])
    axis.set_xlabel("target-domain simulations")
    axis.set_ylabel("relative $L^2$")
    axis.set_title("a  Darcy domain extension", loc="left", fontweight="bold")
    axis.legend(frameon=False, ncol=2, columnspacing=0.8)

    methods = ("local_core", "latentddm", accepted_sni, "global_point")
    labels = (
        "Local/Core", "LatentDDM",
        f"SNI-{accepted_sni.removeprefix('sni_calls_')}", "Point",
    )
    metric_specs = (
        ("field", lambda method: _summary(payload, method)["field_relative_l2"]),
        ("step 20", lambda method: _terminal(payload, method)["field_relative_l2"]),
        (r"$\omega$", lambda method: _summary(payload, method)["vorticity_relative_l2"]),
        ("drag", lambda method: _summary(payload, method)["c_drag_relative_l2"]),
        ("lift", lambda method: _summary(payload, method)["c_lift_relative_l2"]),
    )
    values = np.asarray([
        [float(accessor(method)) for _, accessor in metric_specs]
        for method in methods
    ])
    ratios = values / values[0:1].clip(min=1.0e-12)
    axis = axes[1]
    finite = ratios[np.isfinite(ratios)]
    image = axis.imshow(
        ratios, aspect="auto", cmap="RdYlGn_r",
        norm=TwoSlopeNorm(
            vmin=min(0.45, float(finite.min())), vcenter=1.0,
            vmax=max(1.45, float(finite.max())),
        ),
    )
    best = np.nanargmin(ratios, axis=0)
    for row in range(ratios.shape[0]):
        for column in range(ratios.shape[1]):
            value = ratios[row, column]
            axis.text(
                column, row, f"{value:.2f}", ha="center", va="center",
                fontsize=7.4,
                color="white" if value < 0.62 or value > 1.45 else "#222222",
                fontweight="bold" if row == best[column] else "normal",
            )
    axis.set_xticks(np.arange(len(metric_specs)), [name for name, _ in metric_specs])
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.get_yticklabels()[1].set_color(COLORS["LatentDDM"])
    axis.get_yticklabels()[1].set_fontweight("bold")
    axis.set_title("b  Airfoil H20 / Local", loc="left", fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    colorbar.set_label("error ratio", fontsize=7.5)
    colorbar.ax.tick_params(labelsize=7)

    sni_names = sorted(
        (name for name in payload["methods"] if name.startswith("sni_calls_")),
        key=lambda name: int(name.removeprefix("sni_calls_")),
    )
    path_names = ["local_core", *sni_names]
    latent_error = float(_summary(payload, "latentddm")["field_relative_l2"])
    axis = axes[2]
    times = np.asarray([
        float(_summary(payload, name)["inference_seconds_mean"])
        for name in path_names
    ])
    errors = np.asarray([
        float(_summary(payload, name)["field_relative_l2"])
        for name in path_names
    ]) / latent_error
    axis.plot(
        times, errors, marker="o", markersize=3.4, linewidth=1.5,
        color="#355C9A", label="target-free SNI",
    )
    for name, marker, color, label in (
        ("latentddm", "D", COLORS["LatentDDM"], "LatentDDM"),
        ("global_point", "s", COLORS["Point"], "Point"),
    ):
        axis.scatter(
            [_summary(payload, name)["inference_seconds_mean"]],
            [_summary(payload, name)["field_relative_l2"] / latent_error],
            marker=marker, s=34, color=color, edgecolor="black",
            linewidth=0.5, zorder=4, label=label,
        )
    for name, x, y in zip(path_names, times, errors):
        calls = 1 if name == "local_core" else int(name.removeprefix("sni_calls_"))
        axis.annotate(str(calls), (x, y), xytext=(2, 3), textcoords="offset points", fontsize=6.8)
    axis.axhline(1.0, color="0.45", linewidth=0.8, linestyle="--")
    axis.set_xscale("log")
    axis.set_xlabel("measured seconds / step")
    axis.set_ylabel("field error / LatentDDM")
    axis.set_title("c  Airfoil accuracy--cost", loc="left", fontweight="bold")
    axis.legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=0.18, which="both")
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(pad=0.45)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)

    result = {
        "contract": "latentddm_two_case_native_verified_figure_v1",
        "airfoil_artifact": str(airfoil_path.resolve()),
        "airfoil_artifact_sha256": _sha256(airfoil_path),
        "accepted_sni_method": accepted_sni,
        "airfoil_methods": list(methods),
        "airfoil_metric_names": [name for name, _ in metric_specs],
        "airfoil_absolute_metrics": values.tolist(),
        "airfoil_ratios_to_local": ratios.tolist(),
        "airfoil_accuracy_cost": {
            "methods": path_names,
            "seconds_per_step": times.tolist(),
            "field_error_over_latentddm": errors.tolist(),
        },
        "protected_test_trajectories_opened": 0,
        "final_h20_verification": verification,
    }
    output_base.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airfoil-h20", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_figure(args.airfoil_h20, args.output_base), indent=2))


if __name__ == "__main__":
    main()
