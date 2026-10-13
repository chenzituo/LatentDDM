"""Held-out head-to-head evaluation of CoDiffuse-Cartesian and global AB-UPT."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.neural_eos.abupt import ABUPTScalarOperator
from codiffuse_Schwarz_Airfoil.coupled_trajectory import CyclicWindowDataset
from codiffuse_Schwarz_Airfoil.dataset import FieldStatistics
from codiffuse_Schwarz_Airfoil.force_diagnostics import (
    corrected_force_coefficients,
    penalty_force_coefficients,
)
from codiffuse_Schwarz_Airfoil.train_global_abupt import (
    INPUT_CHANNELS,
    make_inputs,
    physical_state,
    predict as predict_abupt,
)
from codiffuse_Schwarz_Airfoil.train_one_step_multilevel import load_pretrained
from codiffuse_Schwarz_Airfoil.train_robust_vorticity_projection import projected_next
from codiffuse_Schwarz_Airfoil.vorticity_projection import periodic_vorticity


METHODS = ("codiffuse", "abupt")


def batch_sample(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value[None] for key, value in sample.items()}


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(
        (prediction - target).square().sum()
        / target.square().sum().clamp_min(1.0e-20)
    ))


def field_metrics(
    prediction: torch.Tensor, target: torch.Tensor, dx: float,
) -> dict[str, float]:
    error = prediction - target
    gauge_error = error[:, 2] - error[:, 2].mean(dim=(-1, -2), keepdim=True)
    pressure_target = target[:, 2] - target[:, 2].mean(dim=(-1, -2), keepdim=True)
    return {
        "state_l2": relative_l2(prediction, target),
        "velocity_l2": relative_l2(prediction[:, :2], target[:, :2]),
        "u_l2": relative_l2(prediction[:, :1], target[:, :1]),
        "v_l2": relative_l2(prediction[:, 1:2], target[:, 1:2]),
        "pressure_l2": relative_l2(prediction[:, 2:], target[:, 2:]),
        "pressure_gauge_l2": float(torch.sqrt(
            gauge_error.square().sum() / pressure_target.square().sum().clamp_min(1.0e-20)
        )),
        "vorticity_l2": relative_l2(
            periodic_vorticity(prediction, dx), periodic_vorticity(target, dx)
        ),
    }


def force_metrics(
    prediction: torch.Tensor, target: torch.Tensor,
    batch: dict[str, torch.Tensor], dataset: CyclicWindowDataset,
    reference_batch: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    """Compare raw and momentum-corrected penalty-reaction coefficients."""
    device = prediction.device
    reference_batch = batch if reference_batch is None else reference_batch
    common = {
        "dx": dataset.dx,
        "window": dataset.window,
        "penalty_time": dataset.penalty_time,
        "rotation_sign": dataset.rotation_sign,
    }
    predicted_previous = penalty_force_coefficients(
        batch["q_n"].to(device), batch["sdf_n"].to(device),
        batch["motion_n"].to(device), **common,
    )
    reference_previous = penalty_force_coefficients(
        reference_batch["q_n"].to(device), reference_batch["sdf_n"].to(device),
        reference_batch["motion_n"].to(device), **common,
    )
    predicted = penalty_force_coefficients(
        prediction, batch["sdf_np1"].to(device),
        batch["motion_np1"].to(device), **common,
    )
    reference = penalty_force_coefficients(
        target, batch["sdf_np1"].to(device),
        batch["motion_np1"].to(device), **common,
    )
    dt = batch["time_np1"] - batch["time_n"]
    predicted_corrected = corrected_force_coefficients(
        predicted, predicted_previous, dt, rotation_sign=dataset.rotation_sign,
    )
    reference_corrected = corrected_force_coefficients(
        reference, reference_previous, dt, rotation_sign=dataset.rotation_sign,
    )
    values = {
        "c_drag_prediction": predicted.drag,
        "c_drag_reference": reference.drag,
        "c_lift_prediction": predicted.lift,
        "c_lift_reference": reference.lift,
        "c_drag_corrected_prediction": predicted_corrected[0],
        "c_drag_corrected_reference": reference_corrected[0],
        "c_lift_corrected_prediction": predicted_corrected[1],
        "c_lift_corrected_reference": reference_corrected[1],
    }
    result = {key: float(value[0]) for key, value in values.items()}
    for coefficient in ("c_drag", "c_lift", "c_drag_corrected", "c_lift_corrected"):
        result[f"{coefficient}_absolute_error"] = abs(
            result[f"{coefficient}_prediction"] - result[f"{coefficient}_reference"]
        )
    return result


def load_abupt(checkpoint: dict, device: torch.device) -> ABUPTScalarOperator:
    args = checkpoint["args"]
    model = ABUPTScalarOperator(
        len(INPUT_CHANNELS), int(args["d_model"]), int(args["anchors"]),
        int(args["depth"]), int(args["heads"]), out_channels=3,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


@torch.no_grad()
def predict_models(
    codiffuse: torch.nn.Module, abupt: ABUPTScalarOperator,
    batch: dict[str, torch.Tensor], dataset: CyclicWindowDataset,
    statistics: FieldStatistics, args: argparse.Namespace, seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    codiffuse_prediction, _ = projected_next(
        codiffuse, batch, dataset, statistics, args.fine_fraction, args.halo,
        seed, device,
    )
    inputs, _ = make_inputs(batch, dataset, statistics, device)
    abupt_delta = predict_abupt(abupt, inputs, args.context_stride).float()
    return {
        "codiffuse": codiffuse_prediction,
        "abupt": physical_state(abupt_delta, batch, statistics, device),
    }


@torch.no_grad()
def one_step_evaluation(
    codiffuse, abupt, datasets, statistics, args, device,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trajectory_index, dataset in enumerate(datasets):
        for item in range(0, len(dataset), args.one_step_stride):
            batch = batch_sample(dataset[item])
            predictions = predict_models(
                codiffuse, abupt, batch, dataset, statistics, args,
                args.seed + trajectory_index * 1_000_003 + item, device,
            )
            target = batch["q_np1"].to(device)
            for method, prediction in predictions.items():
                rows.append({
                    "regime": "one_step", "trajectory": dataset.path.stem,
                    "item": item, "frame": int(batch["frame"][0]),
                    "method": method,
                } | field_metrics(prediction, target, dataset.dx)
                  | force_metrics(prediction, target, batch, dataset))
    return rows


@torch.no_grad()
def rollout_evaluation(
    codiffuse, abupt, datasets, statistics, args, device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    endpoints: list[dict[str, object]] = []
    for trajectory_index, dataset in enumerate(datasets):
        maximum = len(dataset) - args.horizon
        starts = sorted(set((0, maximum // 2, maximum)))
        for start_index, start in enumerate(starts):
            initial = dataset[start]
            histories = {
                method: (initial["q_nm1"][None].to(device), initial["q_n"][None].to(device))
                for method in METHODS
            }
            for step in range(args.horizon):
                item = start + step
                base = batch_sample(dataset[item])
                target = base["q_np1"].to(device)
                predictions = {}
                for method in METHODS:
                    previous, current = histories[method]
                    controlled = dict(base)
                    controlled["q_nm1"] = previous
                    controlled["q_n"] = current
                    if method == "codiffuse":
                        prediction, _ = projected_next(
                            codiffuse, controlled, dataset, statistics,
                            args.fine_fraction, args.halo,
                            args.seed + trajectory_index * 10_000_019
                            + start_index * 100_003 + step, device,
                        )
                    else:
                        inputs, _ = make_inputs(controlled, dataset, statistics, device)
                        delta = predict_abupt(
                            abupt, inputs, args.context_stride
                        ).float()
                        prediction = physical_state(delta, controlled, statistics, device)
                    predictions[method] = prediction
                    histories[method] = (current, prediction)
                    rows.append({
                        "regime": "rollout", "trajectory": dataset.path.stem,
                        "start": start, "step": step + 1, "method": method,
                        "central_start": start_index == 1,
                        "angle_deg": abs(float(base["motion_np1"][0, 0])) * 180.0 / np.pi,
                    } | field_metrics(prediction, target, dataset.dx)
                      | force_metrics(prediction, target, controlled, dataset))
                if step + 1 == args.horizon and start_index == 1:
                    endpoints.append({
                        "trajectory": dataset.path.stem,
                        "target": target.cpu(),
                        "codiffuse": predictions["codiffuse"].cpu(),
                        "abupt": predictions["abupt"].cpu(),
                        "dx": dataset.dx,
                    })
    return rows, endpoints


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    result = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        result[method] = {}
        for metric in (
            "state_l2", "velocity_l2", "u_l2", "v_l2", "pressure_l2",
            "pressure_gauge_l2", "vorticity_l2",
        ):
            values = np.asarray([float(row[metric]) for row in selected])
            result[method][f"{metric}_mean"] = float(values.mean())
            result[method][f"{metric}_p90"] = float(np.quantile(values, 0.9))
            result[method][f"{metric}_max"] = float(values.max())
        for coefficient in (
            "c_drag", "c_lift", "c_drag_corrected", "c_lift_corrected",
        ):
            prediction = np.asarray([
                float(row[f"{coefficient}_prediction"]) for row in selected
            ])
            reference = np.asarray([
                float(row[f"{coefficient}_reference"]) for row in selected
            ])
            error = prediction - reference
            result[method][f"{coefficient}_relative_l2"] = float(
                np.linalg.norm(error) / max(np.linalg.norm(reference), 1.0e-20)
            )
            result[method][f"{coefficient}_mae"] = float(np.mean(np.abs(error)))
    return result


def save_force_histories(
    rows: list[dict[str, object]], output: Path, *, corrected: bool = False,
) -> None:
    selected = [row for row in rows if bool(row["central_start"])]
    trajectories = sorted({str(row["trajectory"]) for row in selected})
    figure, axes = plt.subplots(
        len(trajectories), 2, figsize=(11, 2.8 * len(trajectories)), squeeze=False,
        sharex="row",
    )
    colors = {"codiffuse": "#1769aa", "abupt": "#c62828"}
    labels = {"codiffuse": "Cartesian Patch-EoS", "abupt": "Global AB-UPT"}
    suffix = "_corrected" if corrected else ""
    legend_handles = None
    for row_index, trajectory in enumerate(trajectories):
        trajectory_rows = [row for row in selected if row["trajectory"] == trajectory]
        reference_rows = sorted(
            (row for row in trajectory_rows if row["method"] == "codiffuse"),
            key=lambda row: int(row["step"]),
        )
        step = [int(row["step"]) for row in reference_rows]
        for column, coefficient in enumerate((f"c_drag{suffix}", f"c_lift{suffix}")):
            axis = axes[row_index, column]
            axis.plot(
                step, [float(row[f"{coefficient}_reference"]) for row in reference_rows],
                color="#222222", linewidth=2.0, label="Reference",
            )
            for method in METHODS:
                method_rows = sorted(
                    (row for row in trajectory_rows if row["method"] == method),
                    key=lambda row: int(row["step"]),
                )
                axis.plot(
                    step, [float(row[f"{coefficient}_prediction"]) for row in method_rows],
                    color=colors[method], linewidth=1.5, label=labels[method],
                )
            axis.grid(True, color="#dddddd", linewidth=.7)
            axis.set_ylabel(r"$C_D$" if column == 0 else r"$C_L$")
            axis.set_xlabel("Autoregressive step")
            if row_index == 0:
                axis.set_title("Drag" if column == 0 else "Lift")
                if column == 0:
                    legend_handles = axis.get_legend_handles_labels()
        axes[row_index, 0].text(
            .01, .96, trajectory.replace("_level5_window", ""),
            transform=axes[row_index, 0].transAxes, va="top", fontsize=9,
        )
    qualifier = " with fictitious-fluid correction" if corrected else ""
    figure.suptitle(
        f"Held-out 20-step penalty-reaction force histories{qualifier}",
        fontsize=14, y=.995,
    )
    if legend_handles is not None:
        figure.legend(
            *legend_handles, loc="upper center", bbox_to_anchor=(.5, .972),
            frameon=False, fontsize=8, ncol=3,
        )
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def save_loss_curves(codiffuse_path: Path, abupt_path: Path, output: Path) -> None:
    histories = {
        "Cartesian Patch-EoS": json.loads(
            (codiffuse_path.parent / "summary.json").read_text(encoding="utf-8")
        )["history"],
        "Global AB-UPT": json.loads(
            (abupt_path.parent / "summary.json").read_text(encoding="utf-8")
        )["history"],
    }
    colors = {"Cartesian Patch-EoS": "#1769aa", "Global AB-UPT": "#c62828"}
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for axis, metric, title in zip(
        axes, ("state_l2", "vorticity_l2"), ("State", "Vorticity"),
    ):
        for label, history in histories.items():
            epochs = [int(row["epoch"]) for row in history]
            axis.plot(
                epochs, [float(row[f"train_{metric}"]) for row in history],
                color=colors[label], linestyle="--", alpha=.7,
                label=f"{label} train",
            )
            validation = [
                (int(row["epoch"]), float(row[f"validation_{metric}_mean"]))
                for row in history if f"validation_{metric}_mean" in row
            ]
            axis.plot(
                [item[0] for item in validation], [item[1] for item in validation],
                color=colors[label], linewidth=2.0, label=f"{label} validation",
            )
        axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Relative L2")
        axis.set_title(f"{title} convergence")
        axis.grid(True, which="both", color="#dddddd", linewidth=.7)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def save_fields(endpoints: list[dict[str, object]], output: Path, field: str) -> None:
    figure, axes = plt.subplots(len(endpoints), 5, figsize=(18, 3.3 * len(endpoints)), squeeze=False)
    for row, endpoint in enumerate(endpoints):
        target = endpoint["target"]
        codiffuse = endpoint["codiffuse"]
        abupt = endpoint["abupt"]
        if field == "vorticity":
            dx = float(endpoint["dx"])
            target_field = periodic_vorticity(target, dx)[0]
            codiffuse_field = periodic_vorticity(codiffuse, dx)[0]
            abupt_field = periodic_vorticity(abupt, dx)[0]
            cmap = "RdBu_r"
        else:
            target_field = target[0, :2].square().sum(0).sqrt()
            codiffuse_field = codiffuse[0, :2].square().sum(0).sqrt()
            abupt_field = abupt[0, :2].square().sum(0).sqrt()
            cmap = "viridis"
        scale = float(target_field.abs().quantile(0.995))
        error_scale = max(
            float((codiffuse_field - target_field).abs().quantile(0.995)),
            float((abupt_field - target_field).abs().quantile(0.995)), 1.0e-8,
        )
        panels = (
            (target_field, "Reference", cmap, -scale if field == "vorticity" else 0.0, scale),
            (codiffuse_field, "CoDiffuse-Cartesian", cmap, -scale if field == "vorticity" else 0.0, scale),
            ((codiffuse_field-target_field).abs(), "CoDiffuse error", "magma", 0.0, error_scale),
            (abupt_field, "Global AB-UPT", cmap, -scale if field == "vorticity" else 0.0, scale),
            ((abupt_field-target_field).abs(), "AB-UPT error", "magma", 0.0, error_scale),
        )
        for column, (values, title, color, low, high) in enumerate(panels):
            axis = axes[row, column]
            image = axis.imshow(values, origin="lower", cmap=color, vmin=low, vmax=high, aspect="auto")
            axis.set_title(title if row == 0 else "", fontsize=10)
            if column == 0:
                axis.set_ylabel(str(endpoint["trajectory"]).replace("_level5_window", ""))
            axis.set_xticks([]); axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=.025, pad=.015)
    figure.suptitle(f"Held-out 20-step rollout: {field}", fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--codiffuse", type=Path, required=True)
    parser.add_argument("--abupt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--one-step-stride", type=int, default=1)
    parser.add_argument("--fine-fraction", type=float, default=.35)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--context-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    codiffuse_checkpoint = torch.load(args.codiffuse, map_location="cpu", weights_only=False)
    abupt_checkpoint = torch.load(args.abupt, map_location="cpu", weights_only=False)
    if codiffuse_checkpoint["split"] != abupt_checkpoint["split"]:
        raise RuntimeError("models were not trained on the same trajectory split")
    split = codiffuse_checkpoint["split"]
    statistics = FieldStatistics.from_json(codiffuse_checkpoint["statistics"])
    codiffuse, _ = load_pretrained(codiffuse_checkpoint, device)
    codiffuse.eval()
    abupt = load_abupt(abupt_checkpoint, device)
    datasets = [
        CyclicWindowDataset(args.data_root / name, "all", history_steps=2)
        for name in split["test"]
    ]
    one_step = one_step_evaluation(
        codiffuse, abupt, datasets, statistics, args, device
    )
    rollout, endpoints = rollout_evaluation(
        codiffuse, abupt, datasets, statistics, args, device
    )
    rows = one_step + rollout
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "test_trajectories": split["test"],
        "one_step_samples_per_method": len(one_step) // 2,
        "rollout_samples_per_method": len(rollout) // 2,
        "horizon": args.horizon,
        "one_step": summarize(one_step),
        "rollout_all_steps": summarize(rollout),
        "rollout_final_step": summarize([
            row for row in rollout if row["step"] == args.horizon
        ]),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    save_fields(endpoints, args.output_dir / "vorticity_fields.png", "vorticity")
    save_fields(endpoints, args.output_dir / "velocity_fields.png", "velocity")
    save_force_histories(rollout, args.output_dir / "force_coefficients.png")
    save_force_histories(
        rollout, args.output_dir / "force_coefficients_corrected.png", corrected=True,
    )
    save_loss_curves(args.codiffuse, args.abupt, args.output_dir / "loss_curves.png")
    for dataset in datasets:
        dataset.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
