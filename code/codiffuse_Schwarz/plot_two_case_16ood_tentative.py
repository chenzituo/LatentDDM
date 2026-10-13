"""Render a tentative Figure 2 with 16-OOD-shot Airfoil adaptation.

Panel (a) retains the protected Darcy domain-extension comparison. Panel (b)
uses only the Airfoil OOD target-validation aggregate: 16 OOD trajectories
adapt each source model, and a disjoint set of 16 OOD trajectories evaluates
the raw adapted checkpoints. The protected Airfoil test is never read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_two_case_transfer import (
    COLORS,
    MARKERS,
    METHOD_KEYS,
    interval,
    load_darcy,
    plot_method,
)


ROOT = Path(__file__).resolve().parents[1]
LEARNED_METHODS = ("LatentDDM", "Point", "Slice", "Patch")
AIRFOIL_KEYS = {
    "LatentDDM": "latent",
    "Point": "point",
    "Slice": "slice",
    "Patch": "patch",
}
UPDATE_SCOPE = {
    "LatentDDM": "adapter only",
    "Point": "all weights",
    "Slice": "all weights",
    "Patch": "all weights",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--darcy-validation", type=Path,
        default=ROOT / (
            "codiffuse_Schwarz_Darcy/outputs/domain_extension_fixed_dx_v1/"
            "paired_validation_summary/summary.json"
        ),
    )
    parser.add_argument(
        "--darcy-test", type=Path,
        default=ROOT / (
            "codiffuse_Schwarz_Darcy/outputs/domain_extension_fixed_dx_v1/"
            "protected_test_summary/summary.json"
        ),
    )
    parser.add_argument(
        "--airfoil-validation", type=Path,
        default=ROOT / (
            "codiffuse_Schwarz_Airfoil/outputs/pitch_rate_ood_v1/"
            "evaluation_v2/aggregate_validation.json"
        ),
    )
    parser.add_argument(
        "--airfoil-selection", type=Path,
        default=ROOT / (
            "codiffuse_Schwarz_Airfoil/outputs/pitch_rate_ood_v1/"
            "evaluation_v2/adaptation_selection_validation.json"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / (
            "codiffuse_Schwarz/workshop/submission/Two_case/figures/"
            "two_case_adaptation_16ood_tentative.pdf"
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_airfoil_validation(
    aggregate_path: Path, selection_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    aggregate = read_json(aggregate_path)
    selection = read_json(selection_path)
    if (
        aggregate.get("contract") != "airfoil_pitch_rate_ood_rollout_aggregate_v1"
        or not bool(aggregate.get("complete"))
        or aggregate.get("report_split") != "validation"
        or bool(aggregate.get("protected_test_evaluation"))
        or int(aggregate.get("horizon", -1)) != 20
        or 16 not in aggregate.get("budgets", ())
    ):
        raise ValueError("Airfoil OOD validation aggregate failed its audit contract")
    if (
        selection.get("contract")
        != "airfoil_pitch_rate_ood_adaptation_selection_v1"
        or selection.get("selection_partition") != "target_validation"
        or int(selection.get("protected_test_trajectories_opened", -1)) != 0
    ):
        raise ValueError("Airfoil adaptation selection failed its audit contract")
    if set(AIRFOIL_KEYS.values()) - set(aggregate.get("methods", ())):
        raise ValueError("Airfoil aggregate is missing a learned comparator")
    return aggregate, selection


def render(args: argparse.Namespace) -> None:
    darcy_validation, darcy_test = load_darcy(
        args.darcy_validation, args.darcy_test,
    )
    airfoil, selection = load_airfoil_validation(
        args.airfoil_validation, args.airfoil_selection,
    )

    plt.rcParams.update({
        "font.size": 7.5,
        "axes.titlesize": 8.4,
        "axes.labelsize": 7.7,
        "legend.fontsize": 6.7,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure = plt.figure(figsize=(7.15, 2.28))
    grid = figure.add_gridspec(
        1, 2, width_ratios=(.92, 1.32),
        left=.08, right=.985, top=.91, bottom=.25, wspace=.31,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    exported: dict[str, object] = {
        "contract": "two_case_16ood_tentative_figure_v1",
        "status": "validation-only tentative; protected Airfoil test not opened",
        "darcy": {},
        "airfoil": {
            "adaptation_partition": "OOD target adaptation",
            "adaptation_trajectories_per_seed": 16,
            "evaluation_partition": "OOD target validation",
            "evaluation_trajectories": 16,
            "adaptation_seeds": list(selection["seeds"]),
            "selection_horizon": 20,
            "protected_test_trajectories_opened": 0,
            "methods": {},
        },
        "inputs": {
            "airfoil_validation": str(args.airfoil_validation),
            "airfoil_validation_sha256": sha256(args.airfoil_validation),
            "airfoil_selection": str(args.airfoil_selection),
            "airfoil_selection_sha256": sha256(args.airfoil_selection),
        },
    }

    # (a) Preserve the current protected Darcy comparison.
    ax = axes[0]
    x = np.asarray((3, 4, 8, 16), dtype=float)
    source = darcy_validation["source_error"]
    for method in LEARNED_METHODS:
        key = METHOD_KEYS[method]
        zero = [float(source[key])]
        adapted = [float(source[key])]
        low = [float(source[key])]
        high = [float(source[key])]
        for blocks in (4, 8, 16):
            entry = darcy_test["scales"][f"{blocks}x{blocks}"]["methods"][key]
            zero.append(float(entry["zero_shot_error"]["mean"]))
            mean, lo, hi = interval(entry["adapted_error"])
            adapted.append(mean)
            low.append(lo)
            high.append(hi)
        ax.plot(
            x, zero, color=COLORS[method], linestyle=":", linewidth=1.0,
            marker=MARKERS[method], markersize=3.8, markerfacecolor="white",
            alpha=.72,
        )
        plot_method(
            ax, x, np.asarray(adapted), np.asarray(low), np.asarray(high), method,
        )
        exported["darcy"][key] = {
            "blocks_per_axis": x.tolist(),
            "zero_shot": zero,
            "adapted_16": adapted,
        }
    sni_x = np.asarray((4, 8, 16), dtype=float)
    sni_entries = [
        darcy_test["scales"][f"{blocks}x{blocks}"]["sni"]["sweep16_error"]
        for blocks in (4, 8, 16)
    ]
    plot_method(
        ax, sni_x,
        np.asarray([float(row["mean"]) for row in sni_entries]),
        np.asarray([float(row["ci95_low"]) for row in sni_entries]),
        np.asarray([float(row["ci95_high"]) for row in sni_entries]),
        "SNI", linestyle="--",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x, labels=("3x3", "4x4", "8x8", "16x16"))
    ax.set_title("(a) Darcy: larger domains", loc="left", fontweight="bold")
    ax.set_xlabel("subdomain cover")
    ax.set_ylabel(r"relative $L_2$")
    ax.legend(
        handles=(
            Line2D([], [], color="#555555", linestyle=":", marker="o",
                   markerfacecolor="white", markersize=3.5, label="zero-shot"),
            Line2D([], [], color="#555555", linestyle="-", marker="o",
                   markersize=3.5, label="+16 simulations"),
        ),
        loc="lower left", frameon=False, handlelength=1.5,
        borderpad=.1, labelspacing=.2,
    )

    # (b) Raw OOD validation change after matched 16-trajectory adaptation.
    ax = axes[1]
    condition_x = np.asarray((0.0, 1.0))
    for method in LEARNED_METHODS:
        key = AIRFOIL_KEYS[method]
        source_entry = airfoil["curves"]["ood"][key]["0"]["field"]
        adapted_entry = airfoil["curves"]["ood"][key]["16"]["field"]
        source_mean, source_low, source_high = interval(source_entry)
        adapted_mean, adapted_low, adapted_high = interval(adapted_entry)
        means = np.asarray((source_mean, adapted_mean))
        errors = np.asarray((
            (source_mean - source_low, source_high - source_mean),
            (adapted_mean - adapted_low, adapted_high - adapted_mean),
        )).T
        ax.plot(
            condition_x, means, color=COLORS[method],
            linewidth=2.25 if method == "LatentDDM" else 1.35,
            marker=MARKERS[method], markersize=5.0,
            markerfacecolor="white", markeredgewidth=1.0,
            label=f"{method} ({UPDATE_SCOPE[method]})",
            zorder=7 if method == "LatentDDM" else 4,
        )
        ax.errorbar(
            condition_x, means, yerr=errors, fmt="none", color=COLORS[method],
            elinewidth=.9, capsize=2.2, capthick=.8, alpha=.85,
            zorder=6 if method == "LatentDDM" else 3,
        )
        ax.scatter(
            (1.0,), (adapted_mean,), color=COLORS[method],
            marker=MARKERS[method], s=23, zorder=8,
        )
        paired_gain = airfoil["curves"]["ood"][key]["16"][
            "paired_relative_gain"
        ]["field"]
        exported["airfoil"]["methods"][key] = {
            "display_name": method,
            "update_scope": UPDATE_SCOPE[method],
            "trainable_parameter_count": int(
                airfoil["curves"]["ood"][key]["16"]["adaptation_contract"][
                    "trainable_parameter_count"
                ]
            ),
            "source": dict(source_entry),
            "adapted_16": dict(adapted_entry),
            "paired_relative_gain": dict(paired_gain),
            "raw_mean_relative_change": adapted_mean / source_mean - 1.0,
        }
    ax.set_xlim(-.18, 1.18)
    ax.set_ylim(.27, .60)
    ax.set_xticks(condition_x, labels=("zero-shot", "+16 OOD"))
    ax.set_title("(b) Airfoil: 16 OOD simulations", loc="left", fontweight="bold")
    ax.set_ylabel(r"OOD-validation H20 field $L_2$")
    ax.text(
        .985, .965, "validation only", transform=ax.transAxes,
        ha="right", va="top", color="#596574", fontsize=6.6,
        fontweight="bold",
    )

    for axis in axes:
        axis.grid(True, which="major", alpha=.18, linewidth=.55)
        axis.grid(True, which="minor", alpha=.07, linewidth=.4)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(.5, 0.0), handlelength=1.75, columnspacing=1.1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    args.output.with_suffix(".json").write_text(
        json.dumps(exported, indent=2) + "\n", encoding="utf-8",
    )
    plt.close(figure)


def main() -> None:
    render(parse_args())


if __name__ == "__main__":
    main()
