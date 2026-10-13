"""Render the three-panel two-case transfer figure used by the workshop paper.

The Darcy panel reports protected domain-extension tests.  The Airfoil panels
report the immutable, zero-shot ID/OOD H1--H20 protocol.  All checkpoint and
split selection must have happened before the Airfoil protected aggregate is
created; this script only renders that aggregate.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("LatentDDM", "SNI", "Point", "Slice", "Patch")
METHOD_KEYS = {
    "LatentDDM": "latentddm",
    "Point": "point",
    "Slice": "slice",
    "Patch": "patch",
}
AIRFOIL_KEYS = {
    "LatentDDM": "latent",
    "SNI": "sni",
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
    "LatentDDM": "o",
    "SNI": "D",
    "Point": "s",
    "Slice": "^",
    "Patch": "P",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--darcy-validation",
        type=Path,
        default=ROOT / (
            "codiffuse_Schwarz_Darcy/outputs/domain_extension_fixed_dx_v1/"
            "paired_validation_summary/summary.json"
        ),
    )
    parser.add_argument(
        "--darcy-test",
        type=Path,
        default=ROOT / (
            "codiffuse_Schwarz_Darcy/outputs/domain_extension_fixed_dx_v1/"
            "protected_test_summary/summary.json"
        ),
    )
    parser.add_argument(
        "--airfoil-test",
        type=Path,
        default=ROOT / (
            "codiffuse_Schwarz_Airfoil/outputs/pitch_rate_ood_v1/"
            "evaluation_v2/aggregate_zero_shot_test.json"
        ),
    )
    parser.add_argument(
        "--airfoil-assignment",
        type=Path,
        default=ROOT / (
            "codiffuse_Schwarz_Airfoil/layouts/pitch_rate_ood_v1/"
            "trajectory_assignment.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / (
            "codiffuse_Schwarz/workshop/submission/Two_case/figures/"
            "two_case_adaptation.pdf"
        ),
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
        key: value["best_global"] for key, value in validation["scales"].items()
    }
    reported = {
        key: value["validation_selected_best_global"]
        for key, value in test["scales"].items()
    }
    if selected != reported:
        raise ValueError("Darcy test report changed a validation-selected family")
    return validation, test


def load_airfoil(path: Path) -> dict[str, object]:
    payload = read_json(path)
    if (
        payload.get("contract") != "airfoil_pitch_rate_ood_rollout_aggregate_v1"
        or not bool(payload.get("complete"))
        or payload.get("report_split") != "test"
        or not bool(payload.get("protected_test_evaluation"))
        or list(payload.get("budgets", [])) != [0]
        or int(payload.get("horizon", -1)) != 20
    ):
        raise ValueError("Airfoil protected zero-shot aggregate failed its contract")
    if set(payload.get("methods", ())) != {"latent", "sni", "point", "slice", "patch"}:
        raise ValueError("Airfoil aggregate does not contain the frozen five-method comparison")
    return payload


def plot_method(
    ax,
    x: np.ndarray,
    mean: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    method: str,
    *,
    linestyle: str = "-",
) -> None:
    ax.plot(
        x,
        mean,
        color=COLORS[method],
        marker=MARKERS[method],
        markevery=(0, 4),
        markersize=3.6,
        linewidth=2.25 if method == "LatentDDM" else 1.25,
        linestyle=linestyle,
        label=method,
        zorder=7 if method == "LatentDDM" else 4,
    )
    ax.fill_between(x, low, high, color=COLORS[method], alpha=.10, linewidth=0)


def airfoil_curve(
    payload: dict[str, object], split: str, method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = payload["rollout"][split][AIRFOIL_KEYS[method]]
    entries = [rows[f"step_{step:02d}"]["field"] for step in range(1, 21)]
    return tuple(
        np.asarray([float(row[key]) for row in entries])
        for key in ("mean", "ci95_low", "ci95_high")
    )


def _trajectory_id(name: str) -> str:
    return name.split("_level", 1)[0]


def _record_path(evaluation_root: Path, method: str, split: str) -> Path:
    key = AIRFOIL_KEYS[method]
    if key == "sni":
        return evaluation_root / "sni" / f"{split}_test.json"
    candidates = sorted((evaluation_root / key / "budget00").glob(
        f"seed*/{split}_test.json"
    ))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one frozen {method}/{split} record file, found {len(candidates)}"
        )
    return candidates[0]


def load_airfoil_severity(
    aggregate: dict[str, object], assignment_path: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    """Load trajectory-level H20 errors against metadata-only pitch severity."""
    assignment: dict[str, dict[str, object]] = {}
    with assignment_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            assignment[row["id"]] = {
                "q_pitch": float(row["q_pitch"]),
                "pool": row["pool"],
                "partition": row["partition"],
            }
    source_max = max(
        float(row["q_pitch"]) for row in assignment.values()
        if row["pool"] == "source"
    )
    evaluation_root = Path(str(aggregate["evaluation_root"]))
    output: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        q_values: list[float] = []
        errors: list[float] = []
        regimes: list[str] = []
        for split, pool in (("id", "source"), ("ood", "target")):
            payload = read_json(_record_path(evaluation_root, method, split))
            grouped: dict[str, list[float]] = defaultdict(list)
            records = payload.get("records")
            if records is None:
                records = payload.get("rows", {}).get("rollout", ())
            for row in records:
                if int(row["step"]) != 20:
                    continue
                name = row.get("trajectory_name", row.get("trajectory"))
                if name is None:
                    raise ValueError(f"missing trajectory identity in {method}/{split}")
                grouped[_trajectory_id(str(name))].append(
                    float(row["field_relative_l2"])
                )
            for trajectory, values in sorted(grouped.items()):
                metadata = assignment.get(trajectory)
                if metadata is None or metadata["pool"] != pool \
                        or metadata["partition"] != "test":
                    raise ValueError(
                        f"trajectory metadata mismatch for {trajectory}/{split}"
                    )
                q_values.append(float(metadata["q_pitch"]) / source_max)
                errors.append(float(np.mean(values)))
                regimes.append(split)
        output[method] = {
            "severity": np.asarray(q_values),
            "error": np.asarray(errors),
            "regime": np.asarray(regimes),
        }
    reference = output["LatentDDM"]
    for method, row in output.items():
        if not np.allclose(row["severity"], reference["severity"]) \
                or not np.array_equal(row["regime"], reference["regime"]):
            raise ValueError(f"severity support differs for {method}")
    return output, source_max


def severity_bins(
    row: dict[str, np.ndarray], *, seed: int = 20260815,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Three trajectory-level bins per regime with a fixed bootstrap interval."""
    rng = np.random.default_rng(seed)
    centers: list[float] = []
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    regimes: list[str] = []
    for regime in ("id", "ood"):
        indices = np.flatnonzero(row["regime"] == regime)
        indices = indices[np.argsort(row["severity"][indices])]
        for chunk in np.array_split(indices, 3):
            values = row["error"][chunk]
            bootstrap = np.mean(
                rng.choice(values, size=(2000, len(values)), replace=True), axis=1,
            )
            centers.append(float(np.mean(row["severity"][chunk])))
            means.append(float(np.mean(values)))
            lows.append(float(np.quantile(bootstrap, .025)))
            highs.append(float(np.quantile(bootstrap, .975)))
            regimes.append(regime)
    return tuple(np.asarray(values) for values in (
        centers, means, lows, highs, regimes,
    ))


def render(args: argparse.Namespace) -> None:
    darcy_validation, darcy_test = load_darcy(
        args.darcy_validation, args.darcy_test,
    )
    airfoil = load_airfoil(args.airfoil_test)
    severity, source_severity_max = load_airfoil_severity(
        airfoil, args.airfoil_assignment,
    )

    plt.rcParams.update({
        "font.size": 7.5,
        "axes.titlesize": 8.4,
        "axes.labelsize": 7.7,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure = plt.figure(figsize=(7.15, 2.28))
    grid = figure.add_gridspec(
        1, 2, width_ratios=(.92, 1.48),
        left=.08, right=.985, top=.91, bottom=.24, wspace=.29,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    exported: dict[str, object] = {
        "contract": "two_case_transfer_figure_v2",
        "darcy": {},
        "airfoil": {
            "severity_definition": "q_pitch / maximum source-regime q_pitch",
            "source_q_pitch_max": source_severity_max,
            "summary": "trajectory-level H20 means, three bins per regime",
        },
    }

    ax = axes[0]
    x = np.asarray((3, 4, 8, 16), dtype=float)
    source = darcy_validation["source_error"]
    for method in ("LatentDDM", "Point", "Slice", "Patch"):
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
            x,
            zero,
            color=COLORS[method],
            linestyle=":",
            linewidth=1.0,
            marker=MARKERS[method],
            markersize=3.8,
            markerfacecolor="white",
            alpha=.72,
        )
        plot_method(
            ax,
            x,
            np.asarray(adapted),
            np.asarray(low),
            np.asarray(high),
            method,
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
        ax,
        sni_x,
        np.asarray([float(row["mean"]) for row in sni_entries]),
        np.asarray([float(row["ci95_low"]) for row in sni_entries]),
        np.asarray([float(row["ci95_high"]) for row in sni_entries]),
        "SNI",
        linestyle="--",
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
        loc="lower left",
        frameon=False,
        handlelength=1.5,
        borderpad=.1,
        labelspacing=.2,
    )

    ax = axes[1]
    all_x = np.concatenate([row["severity"] for row in severity.values()])
    ax.axvspan(1.0, float(all_x.max()) * 1.035, color="#E9EEF5", zorder=0)
    ax.axvline(1.0, color="#5C6470", linewidth=.8, linestyle="--", zorder=1)
    for method in METHODS:
        centers, mean, low, high, regimes = severity_bins(severity[method])
        for regime, filled in (("id", False), ("ood", True)):
            selected = regimes == regime
            ax.fill_between(
                centers[selected], low[selected], high[selected],
                color=COLORS[method], alpha=.09, linewidth=0,
            )
            ax.plot(
                centers[selected], mean[selected],
                color=COLORS[method], marker=MARKERS[method],
                markerfacecolor=COLORS[method] if filled else "white",
                markeredgewidth=.8, markersize=4.0,
                linewidth=2.25 if method == "LatentDDM" else 1.2,
                linestyle="--" if method == "SNI" else "-",
                label=method if regime == "id" else None,
                zorder=7 if method == "LatentDDM" else 4,
            )
        exported["airfoil"][AIRFOIL_KEYS[method]] = {
            "normalized_pitch_severity": centers.tolist(),
            "h20_field_mean": mean.tolist(),
            "h20_field_ci95_low": low.tolist(),
            "h20_field_ci95_high": high.tolist(),
            "regime": regimes.tolist(),
        }
    ax.set_yscale("log")
    ax.set_xlim(float(all_x.min()) * .975, float(all_x.max()) * 1.025)
    ax.set_title(
        "(b) Airfoil: error across pitch-rate severity",
        loc="left", fontweight="bold",
    )
    ax.set_xlabel(r"normalized pitch severity $q_{\rm pitch}/q_{\rm split}$")
    ax.set_ylabel(r"H20 field relative $L_2$")
    ymin, ymax = ax.get_ylim()
    ax.text(.985, .965, "OOD", transform=ax.transAxes, ha="right", va="top",
            color="#4F5968", fontsize=6.8, fontweight="bold")
    ax.text(.015, .965, "ID", transform=ax.transAxes, ha="left", va="top",
            color="#4F5968", fontsize=6.8, fontweight="bold")

    for ax in axes:
        ax.grid(True, which="major", alpha=.18, linewidth=.55)
        ax.grid(True, which="minor", alpha=.07, linewidth=.4)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    order = [labels.index(name) for name in METHODS]
    figure.legend(
        [handles[index] for index in order],
        METHODS,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(.5, 0.0),
        handlelength=1.9,
        columnspacing=1.25,
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
