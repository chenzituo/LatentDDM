"""Plot low-data Transolver diagnostics for Car and Darcy.

The plot keeps unlike objectives separate.  Car training/validation histories
are normalized to their own epoch-zero values because the training objective
contains a near-body term while validation is endpoint MSE.  Darcy shows the
actual physical validation relative-L2 histories for three predeclared seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--car-log", type=Path, required=True)
    parser.add_argument("--car-comparison", type=Path, required=True)
    parser.add_argument("--car-promoted-metrics", type=Path, required=True)
    parser.add_argument("--darcy-aggregate", type=Path, required=True)
    parser.add_argument("--darcy-runs-root", type=Path, required=True)
    parser.add_argument("--darcy-local-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def main() -> None:
    args = parse_args()
    car_rows = jsonl(args.car_log)
    car = json.loads(args.car_comparison.read_text())
    car_promoted = json.loads(args.car_promoted_metrics.read_text())
    darcy = json.loads(args.darcy_aggregate.read_text())
    local = json.loads(args.darcy_local_metrics.read_text())

    car_epoch = np.asarray([row["epoch"] for row in car_rows], dtype=int)
    car_train = np.asarray([row["train_loss"] for row in car_rows], dtype=float)
    car_validation = np.asarray(
        [row["validation_loss"] for row in car_rows], dtype=float,
    )
    car_selected = int(car_epoch[np.argmin(car_validation)])
    if car_selected != 5:
        raise ValueError(f"unexpected frozen Car Transolver epoch: {car_selected}")

    seeds = [int(seed) for seed in darcy["seeds"]]
    darcy_histories = []
    selected_epochs = []
    for record in darcy["records"]:
        seed = int(record["seed"])
        path = (
            args.darcy_runs_root
            / f"global_transolver_face_v2_full_only_seed{seed}"
            / "summary.json"
        )
        payload = json.loads(path.read_text())
        history = payload["history"]
        darcy_histories.append(np.asarray(
            [row["sweep1_relative_l2"] for row in history], dtype=float,
        ))
        selected_epochs.append(int(record["selected_epoch"]))
    if len({len(curve) for curve in darcy_histories}) != 1:
        raise ValueError("Darcy seed histories have different lengths")
    darcy_curves = np.stack(darcy_histories)
    darcy_epoch = np.arange(1, darcy_curves.shape[1] + 1)

    colors = {
        "global": "#b2182b", "local": "#2166ac", "latent": "#1b7837",
        "neutral": "#666666",
    }
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 6.5))

    axis = axes[0, 0]
    axis.plot(
        car_epoch, car_train / car_train[0], color=colors["global"],
        linewidth=1.9, label="training composite / epoch 0",
    )
    axis.plot(
        car_epoch, car_validation / car_validation[0], color=colors["local"],
        linewidth=1.9, label="validation endpoint MSE / epoch 0",
    )
    axis.scatter(
        [car_selected], [car_validation[car_selected] / car_validation[0]],
        marker="*", s=62, color=colors["local"], zorder=5,
    )
    axis.annotate(
        "selected e5", (car_selected, car_validation[car_selected] / car_validation[0]),
        xytext=(7, 7), textcoords="offset points", fontsize=8,
        color=colors["local"],
    )
    axis.set_title("Car volume Transolver: widening generalization gap")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Ratio to epoch 0")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[0, 1]
    car_values = [
        car["global_vs_local"]["baseline_mean"],
        car["global_vs_local"]["candidate_mean"],
        car_promoted["mean"]["damped_native_magnitude_rel_l2"],
    ]
    car_labels = ["Global\nTransolver", "Local cube\nPOU", "LatentDDM\ncube"]
    bars = axis.bar(
        car_labels, car_values,
        color=[colors["global"], colors["local"], colors["latent"]],
    )
    axis.bar_label(bars, labels=[f"{value:.4f}" for value in car_values],
                   padding=3, fontsize=8)
    axis.set_ylim(0.0, max(car_values) * 1.22)
    axis.set_ylabel(r"Native-CFD $|U|$ relative $L_2$")
    axis.set_title("Car: same 121/25 designs and 131,072 queries")

    axis = axes[1, 0]
    for seed, curve, selected in zip(seeds, darcy_curves, selected_epochs):
        axis.plot(darcy_epoch, curve, linewidth=0.9, alpha=0.42,
                  label=f"seed {seed}")
        axis.scatter([selected], [curve[selected - 1]], marker="*", s=35,
                     color=colors["global"], zorder=5)
    mean = darcy_curves.mean(axis=0)
    std = darcy_curves.std(axis=0)
    axis.plot(darcy_epoch, mean, color=colors["global"], linewidth=2.1,
              label="three-seed mean")
    axis.fill_between(darcy_epoch, mean - std, mean + std,
                      color=colors["global"], alpha=0.13)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(r"Validation relative $L_2$")
    axis.set_title("Darcy mask-aware Transolver learns the global map")
    axis.legend(frameon=False, fontsize=7, ncol=2)

    axis = axes[1, 1]
    global_one = float(darcy["aggregate"]["1"]["relative_l2_mean"])
    global_one_std = float(
        darcy["aggregate"]["1"]["relative_l2_std_population"]
    )
    global_four = float(darcy["aggregate"]["4"]["relative_l2_mean"])
    global_four_std = float(
        darcy["aggregate"]["4"]["relative_l2_std_population"]
    )
    value = float(local["ordinary_reference"]["relative_l2"])
    dynamic = float(local["candidate_ordinary"]["relative_l2"])
    darcy_values = [global_one, global_four, value, dynamic]
    labels = ["Transolver\n1 call", "Transolver\n4 calls",
              "Value latent\n1 pass", "Value+flux\n1 pass"]
    bars = axis.bar(
        labels, darcy_values,
        yerr=[global_one_std, global_four_std, 0.0, 0.0], capsize=3,
        color=[colors["global"], "#d6604d", colors["local"], colors["latent"]],
    )
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in darcy_values],
                   padding=3, fontsize=7.5)
    axis.set_ylim(0.0, max(darcy_values) * 1.25)
    axis.set_ylabel(r"Validation relative $L_2$")
    axis.set_title("Darcy: validation diagnostic (selection overlap)")

    for axis in axes.flat:
        axis.grid(True, axis="y", linewidth=0.45, alpha=0.28)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Low-data global-operator audit: Transolver works, but local reuse generalizes better",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "low_data_transolver_car_darcy"
    figure.savefig(stem.with_suffix(".pdf"), dpi=240, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)

    audit = {
        "contract": "low_data_global_operator_car_darcy_audit_v1",
        "car": {
            "train_designs": 121,
            "validation_designs": 25,
            "epochs": len(car_rows),
            "selected_epoch": car_selected,
            "parameter_count": 11_635_014,
            "field_training_seconds": float(car_train.size and sum(
                float(row["seconds"]) for row in car_rows
            )),
            "train_loss_final_over_initial": float(car_train[-1] / car_train[0]),
            "validation_final_over_selected": float(
                car_validation[-1] / car_validation[car_selected]
            ),
            "native_cfd_magnitude_relative_l2": {
                "global_transolver": float(car_values[0]),
                "local_cube_pou": float(car_values[1]),
                "latentddm": float(car_values[2]),
            },
            "fairness_boundary": [
                "same 121/25 designs, train statistics, 30 epochs, and equalized point-slot budget",
                "global receives deterministic normals and ground height, which favors the global control",
                "local endpoint was initialized from prior local pretraining; learning curves are not compute-matched",
                "global training composite and validation endpoint MSE are normalized separately and not equated",
                "single seed; validation-only result",
            ],
            "promoted_latentddm_source": str(args.car_promoted_metrics.resolve()),
        },
        "darcy": {
            "train_designs": 4000,
            "validation_designs": 250,
            "seeds": seeds,
            "selected_epochs": selected_epochs,
            "global_parameter_count": int(darcy["parameter_count"]),
            "face_local_parameter_count": int(darcy["face_local_parameter_count"]),
            "one_call_relative_l2_mean": global_one,
            "one_call_relative_l2_std_population": global_one_std,
            "four_call_relative_l2_mean": global_four,
            "four_call_relative_l2_std_population": global_four_std,
            "value_latent_relative_l2": value,
            "value_flux_latentddm_relative_l2": dynamic,
            "fairness_boundary": [
                "global model uses random initialization and full-domain training only",
                "local route additionally uses 40,000 local-BVP pretraining examples",
                "first-60 evaluation overlaps checkpoint-selection validation",
                "three seeds are aggregated without post-hoc seed selection",
                "protected A-D/test groups remain unopened by this fresh audit",
            ],
        },
    }
    stem.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
