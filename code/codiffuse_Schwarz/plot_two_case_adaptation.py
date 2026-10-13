"""Render the final three-panel two-case adaptation figure.

Panel (a) separates zero-shot domain extension from matched 16-label target
adaptation.  Panel (b) reports protected Airfoil H20 OOD adaptation.  Panel
(c) is a validation-only mechanism diagnostic: it compares the sequential SNI
depth needed to reach a fixed two-call LatentDDM response in the elliptic and
transport regimes.  Test artifacts never select models or SNI depths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("LatentDDM", "SNI", "Point", "Slice", "Patch")
LEARNED_METHODS = ("LatentDDM", "Point", "Slice", "Patch")
METHOD_KEYS = {
    "LatentDDM": "latentddm",
    "Point": "point",
    "Slice": "slice",
    "Patch": "patch",
}
AIRFOIL_KEYS = {
    "LatentDDM": "latent",
    "Point": "point",
    "Slice": "slice",
    "Patch": "patch",
}
COLORS = {
    "LatentDDM": "#D64B46",
    "SNI": "#3978B5",
    "Point": "#F28E2B",
    "Slice": "#4E9B5F",
    "Patch": "#8064A2",
}
MARKERS = {
    "LatentDDM": "o", "SNI": "D", "Point": "s",
    "Slice": "^", "Patch": "P",
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
        "--airfoil", type=Path,
        default=ROOT / (
            "codiffuse_Schwarz_Airfoil/outputs/pitch_rate_ood_v1/"
            "evaluation_v2/aggregate.json"
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
        "--airfoil-adaptation-selection", type=Path,
        default=ROOT / (
            "codiffuse_Schwarz_Airfoil/outputs/pitch_rate_ood_v1/"
            "evaluation_v2/adaptation_selection_validation.json"
        ),
    )
    parser.add_argument(
        "--airfoil-sni-depth-root", type=Path,
        default=ROOT / (
            "codiffuse_Schwarz_Airfoil/outputs/pitch_rate_ood_v1/"
            "evaluation_v2/sni_depth_validation"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "codiffuse_Schwarz/figures/two_case_adaptation.pdf",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def interval(entry: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(entry["mean"]),
        float(entry.get("ci95_low", entry["mean"])),
        float(entry.get("ci95_high", entry["mean"])),
    )


def load_darcy(
    validation_path: Path, test_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    validation = read_json(validation_path)
    test = read_json(test_path)
    if (
        validation.get("contract")
        != "darcy_fixed_dx_extension_paired_validation_v2"
        or validation.get("split") != "validation"
        or int(validation.get("protected_test_cases_opened", -1)) != 0
    ):
        raise ValueError("Darcy validation artifact failed its audit contract")
    if (
        test.get("contract") != "darcy_fixed_dx_extension_protected_test_v1"
        or bool(test.get("selection_uses_protected_test", True))
        or int(test.get("protected_test_cases_per_scale", -1)) != 64
    ):
        raise ValueError("Darcy protected-test artifact failed its audit contract")
    selected = {
        key: value["best_global"]
        for key, value in validation["scales"].items()
    }
    reported = {
        key: value["validation_selected_best_global"]
        for key, value in test["scales"].items()
    }
    if selected != reported:
        raise ValueError("Darcy test report changed a validation-selected family")
    return validation, test


def load_airfoil(path: Path, *, split: str) -> dict[str, object]:
    payload = read_json(path)
    expected_protected = split == "test"
    if (
        payload.get("contract") != "airfoil_pitch_rate_ood_rollout_aggregate_v1"
        or not bool(payload.get("complete"))
        or payload.get("report_split") != split
        or bool(payload.get("protected_test_evaluation")) != expected_protected
    ):
        raise ValueError(f"Airfoil {split} aggregate failed its audit contract")
    return payload


def selected_validation_field_mean(
    path: Path, *, method: str, budget: int,
) -> float:
    """Return the validation mean of the checkpoint frozen for deployment."""

    payload = read_json(path)
    if (
        payload.get("contract")
        != "airfoil_pitch_rate_ood_adaptation_selection_v1"
        or payload.get("selection_partition") != "target_validation"
        or int(payload.get("protected_test_trajectories_opened", -1)) != 0
    ):
        raise ValueError("Airfoil adaptation selection failed its audit contract")
    rows = payload["selections"][method][f"budget{budget:02d}"]
    values = []
    for seed in payload["seeds"]:
        row = rows[str(seed)]
        if row["decision"] == "adapted":
            values.append(float(row["adapted_field_h20"]))
        elif row["decision"] == "source_retained":
            values.append(float(row["source_field_h20"]))
        else:
            raise ValueError(f"unknown adaptation decision: {row['decision']}")
    if not values:
        raise ValueError("Airfoil adaptation selection contains no seeds")
    return float(np.mean(values))


def clustered_h20_mean(path: Path) -> tuple[int, float]:
    """Read one validation SNI rollout and average starts within trajectory."""

    payload = read_json(path)
    if (
        payload.get("contract") != "airfoil_local_sni_rollout_v2"
        or payload.get("split") != "validation"
        or int(payload.get("protected_test_trajectories_opened", -1)) != 0
        or int(payload.get("horizon", -1)) != 20
    ):
        raise ValueError(f"invalid validation-only SNI depth artifact: {path}")
    grouped: dict[str, list[float]] = {}
    for row in payload["records"]:
        if int(row["step"]) == 20:
            grouped.setdefault(str(row["trajectory"]), []).append(
                float(row["field_relative_l2"])
            )
    if not grouped:
        raise ValueError(f"SNI depth artifact has no H20 rows: {path}")
    mean = float(np.mean([np.mean(values) for values in grouped.values()]))
    return int(payload["sweeps_per_physical_step"]), mean


def plot_method(
    ax, x: np.ndarray, mean: np.ndarray, low: np.ndarray, high: np.ndarray,
    method: str, *, dashed: bool = False, label: str | None = None,
) -> None:
    color = COLORS[method]
    ax.plot(
        x, mean, color=color, marker=MARKERS[method], markersize=4.2,
        linewidth=2.05 if method == "LatentDDM" else 1.4,
        linestyle="--" if dashed else "-", label=label or method,
        zorder=6 if method == "LatentDDM" else 4,
    )
    if np.any(high > low):
        ax.fill_between(x, low, high, color=color, alpha=.12, linewidth=0)


def render(args: argparse.Namespace) -> None:
    darcy_validation, darcy_test = load_darcy(
        args.darcy_validation, args.darcy_test,
    )
    airfoil = load_airfoil(args.airfoil, split="test")
    airfoil_validation = load_airfoil(args.airfoil_validation, split="validation")

    plt.rcParams.update({
        "font.size": 7.6,
        "axes.titlesize": 8.3,
        "axes.labelsize": 7.7,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.24))
    exported: dict[str, object] = {
        "contract": "two_case_adaptation_figure_v2",
        "darcy": {}, "airfoil": {}, "sni_depth": {},
    }

    # (a) Structural extension. Open markers show direct zero-shot reuse;
    # filled markers show the same source models after 16 target simulations.
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
        zero_array = np.asarray(zero)
        adapted_array = np.asarray(adapted)
        ax.plot(
            x, zero_array, color=COLORS[method], linestyle=":", linewidth=1.0,
            marker=MARKERS[method], markersize=4.0, markerfacecolor="white",
            markeredgewidth=1.0, alpha=.70, zorder=2,
        )
        for index in range(1, len(x)):
            ax.plot(
                (x[index], x[index]),
                (zero_array[index], adapted_array[index]),
                color=COLORS[method], alpha=.22, linewidth=.8, zorder=1,
            )
        plot_method(
            ax, x, adapted_array, np.asarray(low), np.asarray(high), method,
        )
        exported["darcy"][key] = {
            "blocks_per_axis": x.tolist(),
            "zero_shot": zero_array.tolist(),
            "adapted_16": adapted_array.tolist(),
        }
    sni_x = np.asarray((4, 8, 16), dtype=float)
    sni_entries = [
        darcy_test["scales"][f"{blocks}x{blocks}"]["sni"]["sweep16_error"]
        for blocks in (4, 8, 16)
    ]
    sni_values = np.asarray([float(entry["mean"]) for entry in sni_entries])
    plot_method(
        ax, sni_x, sni_values,
        np.asarray([float(entry["ci95_low"]) for entry in sni_entries]),
        np.asarray([float(entry["ci95_high"]) for entry in sni_entries]),
        "SNI", dashed=True,
    )
    exported["darcy"]["sni16"] = {
        "blocks_per_axis": sni_x.tolist(), "error": sni_values.tolist(),
    }
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x, labels=("3x3", "4x4", "8x8", "16x16"))
    ax.set_title("(a) Darcy: larger domains", loc="left", fontweight="bold")
    ax.set_xlabel("subdomain cover")
    ax.set_ylabel(r"relative $L_2$")
    style_legend = ax.legend(
        handles=(
            Line2D([], [], color="#555555", linestyle=":", marker="o",
                   markerfacecolor="white", markersize=3.7, label="zero-shot"),
            Line2D([], [], color="#555555", linestyle="-", marker="o",
                   markersize=3.7, label="+16 simulations"),
        ),
        loc="lower left", frameon=False, handlelength=1.5,
        borderpad=.1, labelspacing=.25,
    )
    ax.add_artist(style_legend)

    # (b) Model-owned H20 adaptation on the high-pitch-rate OOD split.
    ax = axes[1]
    budgets = (0, 1, 2, 4, 8, 16)
    budget_x = np.arange(len(budgets), dtype=float)
    ood = airfoil["curves"]["ood"]
    for method in LEARNED_METHODS:
        rows = [ood[AIRFOIL_KEYS[method]][str(value)]["field"] for value in budgets]
        plot_method(
            ax, budget_x,
            np.asarray([float(row["mean"]) for row in rows]),
            np.asarray([float(row["ci95_low"]) for row in rows]),
            np.asarray([float(row["ci95_high"]) for row in rows]),
            method,
        )
        exported["airfoil"][AIRFOIL_KEYS[method]] = {
            "target_trajectories": list(budgets),
            "h20_field": [float(row["mean"]) for row in rows],
        }
    sni = ood["sni"]["0"]["field"]
    plot_method(
        ax, budget_x, np.full(len(budgets), float(sni["mean"])),
        np.full(len(budgets), float(sni["ci95_low"])),
        np.full(len(budgets), float(sni["ci95_high"])),
        "SNI", dashed=True,
    )
    ax.set_yscale("log")
    ax.set_xticks(budget_x, labels=tuple(map(str, budgets)))
    ax.set_title("(b) Airfoil: new regime", loc="left", fontweight="bold")
    ax.set_xlabel("target trajectories")
    ax.set_ylabel(r"OOD H20 field $L_2$")

    # (c) Validation-only mechanism diagnostic. Darcy SNI uses one initial
    # local response plus each refresh; Airfoil's reported sweep count already
    # equals its number of dependent local calls.
    ax = axes[2]
    darcy_colors = ("#9ecae1", "#5b9ec9", "#2171b5")
    for color, blocks in zip(darcy_colors, (4, 8, 16), strict=True):
        points = darcy_validation["scales"][f"{blocks}x{blocks}"][
            "sni_accuracy_time_curve"
        ]
        calls = np.asarray([int(point["sweep"]) + 1 for point in points])
        ratio = np.asarray([
            float(point["error_ratio_to_latentddm"]) for point in points
        ])
        ax.plot(
            calls, ratio, color=color, linewidth=1.25, marker="o",
            markersize=2.8, label=f"Darcy {blocks}x{blocks}",
        )
        exported["sni_depth"][f"darcy_{blocks}x{blocks}"] = {
            "local_calls": calls.tolist(), "error_over_latentddm": ratio.tolist(),
        }
    latent_validation = selected_validation_field_mean(
        args.airfoil_adaptation_selection, method="latent", budget=16,
    )
    airfoil_points = [
        clustered_h20_mean(
            args.airfoil_sni_depth_root / f"ood_sweeps{sweep}.json"
        )
        for sweep in (1, 2, 4, 8, 16)
    ]
    airfoil_calls = np.asarray([point[0] for point in airfoil_points])
    airfoil_ratio = np.asarray([
        point[1] / max(latent_validation, 1.0e-12) for point in airfoil_points
    ])
    ax.plot(
        airfoil_calls, airfoil_ratio, color="#D64B46", linewidth=1.65,
        marker="D", markersize=3.2, label="Airfoil OOD",
    )
    exported["sni_depth"]["airfoil_ood"] = {
        "local_calls": airfoil_calls.tolist(),
        "error_over_latentddm": airfoil_ratio.tolist(),
        "latentddm_validation_selected_h20": latent_validation,
    }
    ax.axhline(1.0, color="#333333", linewidth=.9, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks((1, 2, 4, 8, 16), labels=("1", "2", "4", "8", "16"))
    ax.set_title("(c) Iteration vs. one pass", loc="left", fontweight="bold")
    ax.set_xlabel("sequential local calls")
    ax.set_ylabel("SNI error / LatentDDM")
    ax.legend(loc="upper right", frameon=False, handlelength=1.6)

    for ax in axes:
        ax.grid(True, which="major", alpha=.18, linewidth=.55)
        ax.grid(True, which="minor", alpha=.07, linewidth=.4)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    ordered = [labels.index(name) for name in METHODS]
    figure.legend(
        [handles[index] for index in ordered], METHODS,
        loc="lower center", ncol=5, frameon=False,
        bbox_to_anchor=(.5, -.01), handlelength=1.9, columnspacing=1.25,
    )
    figure.tight_layout(rect=(0, .09, 1, 1), w_pad=1.15)
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
