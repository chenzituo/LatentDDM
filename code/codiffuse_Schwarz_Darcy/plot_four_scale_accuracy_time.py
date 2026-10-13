"""Plot Darcy accuracy against measured inference time at four cover sizes.

The 3x3 panel replays the 4x4-adapted checkpoints on the source validation
domain, matching the first row of the qualitative figure.  The target panels
read the frozen validation ledger used to select all protected-test operating
points.  Every time includes a complete domain prediction with local tiles
batched within a round; SNI rounds remain sequential.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codiffuse_mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "codiffuse_Schwarz_Darcy/outputs/domain_extension_fixed_dx_v1"
OUTPUT = ROOT / (
    "codiffuse_Schwarz/workshop/submission/Two_case/figures/"
    "darcy_four_scale_accuracy_time.pdf"
)
SEEDS = (20260810, 20260811, 20260812)
METHODS = ("LatentDDM", "Point", "Slice", "Patch")
COLORS = {
    "LatentDDM": "#B44B4B",
    "SNI": "#D8844B",
    "Point": "#3568A8",
    "Slice": "#2E8B8B",
    "Patch": "#7761A8",
}
MARKERS = {"LatentDDM": "D", "Point": "o", "Slice": "s", "Patch": "^"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def mean_interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    half = 1.96 * float(array.std(ddof=1)) / np.sqrt(array.size)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(array.mean() - half),
        "ci95_high": float(array.mean() + half),
        "seed_count": int(array.size),
    }


def source_replay() -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for name in METHODS:
        errors: list[float] = []
        seconds: list[float] = []
        for seed in SEEDS:
            if name == "LatentDDM":
                payload = read_json(EVIDENCE / "source_3x3" / (
                    f"replay_4x4_latent_seed{seed}_validation250.json"
                ))["summary"]["latentddm"]
                errors.append(float(payload["relative_l2_mean"]))
                seconds.append(float(payload["seconds_mean"]))
            else:
                payload = read_json(EVIDENCE / "source_3x3" / (
                    f"replay_4x4_{name.lower()}_seed{seed}_validation250/summary.json"
                ))["summary"]["validation/sweep_001"]
                errors.append(float(payload["relative_l2"]))
                seconds.append(float(payload["wall_seconds"]))
        methods[name] = {
            "error": mean_interval(errors),
            "seconds": mean_interval(seconds),
        }

    sni_payload = read_json(
        EVIDENCE / "source_3x3/sni_weighted_validation250_tau1_v1.json"
    )["summary"]
    sni = []
    for sweep in range(1, 17):
        row = sni_payload[f"init_local_core/tau_1.000/sweep_{sweep:04d}"]
        sni.append({
            "sweep": sweep,
            "relative_l2_mean": float(row["relative_l2_mean"]),
            "seconds_per_case_mean": float(row["seconds_mean"]),
        })
    return {
        "evaluation": "validation",
        "cases": 250,
        "checkpoint_policy": "4x4-adapted checkpoints replayed on 3x3",
        "methods": methods,
        "sni_accuracy_time_curve": sni,
    }


def target_scales() -> dict[str, Any]:
    ledger = read_json(EVIDENCE / "paired_validation_summary/summary.json")
    result: dict[str, Any] = {}
    for scale in ("4x4", "8x8", "16x16"):
        source = ledger["scales"][scale]
        methods = {}
        for name in METHODS:
            key = "latentddm" if name == "LatentDDM" else name.lower()
            methods[name] = {
                "error": source["methods"][key]["adapted_error"],
                "seconds": source["methods"][key]["inference_seconds_per_case"],
            }
        result[scale] = {
            "evaluation": "validation",
            "cases": 64,
            "checkpoint_policy": "scale-matched adaptation with 16 labels",
            "methods": methods,
            "sni_accuracy_time_curve": source["sni_accuracy_time_curve"],
        }
    return result


def render(scales: dict[str, Any]) -> plt.Figure:
    plt.rcParams.update({
        # Use the same typography hierarchy as the qualitative panel above it:
        # 7.3 pt body text, 7.4 pt headings, and 6 pt compact annotations.
        "font.size": 7.3,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
    })
    # Match the tight PDF canvas width of the qualitative panel.  LaTeX then
    # applies the same scale factor to both panels, keeping their typography
    # and effective plotting widths consistent.
    figure, axes = plt.subplots(1, 4, figsize=(7.73, 1.656), sharey=True)
    figure.subplots_adjust(left=.074, right=.993, bottom=.285, top=.765, wspace=.14)

    for index, (scale, payload) in enumerate(scales.items()):
        axis = axes[index]
        sni = payload["sni_accuracy_time_curve"]
        axis.loglog(
            [1000.0 * float(row["seconds_per_case_mean"]) for row in sni],
            [float(row["relative_l2_mean"]) for row in sni],
            color=COLORS["SNI"], lw=1.25, marker=".", ms=3.3,
            label="SNI sweeps", zorder=2,
        )
        for name in METHODS:
            row = payload["methods"][name]
            axis.scatter(
                1000.0 * float(row["seconds"]["mean"]),
                float(row["error"]["mean"]),
                s=28 if name == "LatentDDM" else 23,
                marker=MARKERS[name], color=COLORS[name], edgecolor="white",
                linewidth=.45, label=name, zorder=4,
            )
        axis.grid(True, which="major", color="#D9DDE5", lw=.45, alpha=.8)
        axis.grid(True, which="minor", color="#ECEEF2", lw=.30, alpha=.65)
        axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.yaxis.set_major_locator(FixedLocator((.02, .05, .1, .2, .5)))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        scale_title = scale.replace("x", r"\!\times\!")
        axis.set_title(
            (r"$3\!\times\!3$ replay" if scale == "3x3" else
             f"${scale_title}$"), fontsize=7.4, pad=3.0,
        )
        axis.set_xlabel("Inference time / case (ms)", fontsize=7.0, labelpad=1.5)
        axis.tick_params(labelsize=6.0, length=2.1, pad=1.3)
        axis.set_ylim(.014, .72)
        x_values = [
            1000.0 * float(row["seconds_per_case_mean"]) for row in sni
        ] + [
            1000.0 * float(payload["methods"][name]["seconds"]["mean"])
            for name in METHODS
        ]
        axis.set_xlim(min(x_values) * .72, max(x_values) * 1.35)
    axes[0].set_ylabel(r"Relative $L^2$ error", fontsize=7.0, labelpad=2.0)

    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(name) for name in (
        "LatentDDM", "SNI sweeps", "Point", "Slice", "Patch",
    )]
    figure.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="upper center", bbox_to_anchor=(.535, .995), ncol=5,
        frameon=False, fontsize=7.4, handletextpad=.35, columnspacing=1.1,
    )
    return figure


def main() -> None:
    scales = {"3x3": source_replay(), **target_scales()}
    metadata = {
        "contract": "darcy_four_scale_accuracy_time_validation_v1",
        "timing": (
            "CUDA-synchronized complete-domain seconds per case; local tiles "
            "batched within each round; SNI sweeps sequential"
        ),
        "protected_test_cases_opened": 0,
        "scales": scales,
    }
    figure = render(scales)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(
        OUTPUT.with_suffix(".png"), dpi=240, bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), "scales": list(scales)}, indent=2))


if __name__ == "__main__":
    main()
