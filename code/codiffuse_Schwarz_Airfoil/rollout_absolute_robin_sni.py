"""Autoregressive rollout for a strict-local absolute or defect-port solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from .absolute_robin_sni import (
    CONTRACT as ABSOLUTE_CONTRACT,
    absolute_robin_port_2d,
    absolute_robin_sni_sweep_2d,
    absolute_robin_solver_from_payload,
)
from .dataset import FieldStatistics
from .evaluate_global_validation_protocol import rollout_starts
from .force_diagnostics import (
    corrected_force_coefficients,
    penalty_force_coefficients,
)
from .train_absolute_robin_bvp import (
    defect_sni_sweep,
    local_solver_from_payload,
    make_example,
    sha256,
    target_port,
    uses_defect_coordinate,
)
from .train_absolute_robin_latentddm import spatial_level
from .train_multitrajectory_multiresolution import PairedTrajectoryCorpus


Tensor = torch.Tensor


def force_record(
    prediction: Tensor,
    target: Tensor,
    predicted_previous: Tensor,
    reference_previous: Tensor,
    sample: dict[str, Tensor],
    dataset,
    level: str = "fine",
) -> dict[str, float]:
    """Return raw and fictitious-fluid-corrected load coefficients."""
    common = {
        "dx": float(dataset.dx),
        "window": tuple(dataset.window),
        "penalty_time": float(dataset.penalty_time),
        "rotation_sign": float(dataset.rotation_sign),
    }
    device = prediction.device
    predicted = penalty_force_coefficients(
        prediction, sample[f"{level}_sdf_np1"][None].to(device),
        sample["motion_np1"][None].to(device), **common,
    )
    reference = penalty_force_coefficients(
        target, sample[f"{level}_sdf_np1"][None].to(device),
        sample["motion_np1"][None].to(device), **common,
    )
    predicted_prior = penalty_force_coefficients(
        predicted_previous, sample[f"{level}_sdf_n"][None].to(device),
        sample["motion_n"][None].to(device), **common,
    )
    reference_prior = penalty_force_coefficients(
        reference_previous, sample[f"{level}_sdf_n"][None].to(device),
        sample["motion_n"][None].to(device), **common,
    )
    dt = (sample["time_np1"] - sample["time_n"])[None].to(device)
    predicted_corrected = corrected_force_coefficients(
        predicted, predicted_prior, dt,
        rotation_sign=float(dataset.rotation_sign),
    )
    reference_corrected = corrected_force_coefficients(
        reference, reference_prior, dt,
        rotation_sign=float(dataset.rotation_sign),
    )
    pairs = {
        "c_drag": (predicted.drag, reference.drag),
        "c_lift": (predicted.lift, reference.lift),
        "c_drag_corrected": (predicted_corrected[0], reference_corrected[0]),
        "c_lift_corrected": (predicted_corrected[1], reference_corrected[1]),
    }
    result: dict[str, float] = {}
    for name, (value, expected) in pairs.items():
        result[f"{name}_prediction"] = float(value[0])
        result[f"{name}_reference"] = float(expected[0])
        result[f"{name}_absolute_error"] = abs(
            result[f"{name}_prediction"] - result[f"{name}_reference"]
        )
    return result


def scalar_relative_l2(
    rows: list[dict[str, float]], prediction: str, reference: str,
) -> float:
    values = np.asarray([row[prediction] for row in rows], dtype=np.float64)
    expected = np.asarray([row[reference] for row in rows], dtype=np.float64)
    return float(
        np.linalg.norm(values - expected)
        / max(np.linalg.norm(expected), 1.0e-12)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectory-manifest", type=Path,
        help="optional sealed reporting manifest; otherwise use validation",
    )
    parser.add_argument(
        "--split-manifest", type=Path,
        help="source/target protocol split used for explicit validation or test replay",
    )
    parser.add_argument(
        "--split-part", choices=("validation", "test"), default="validation",
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--starts-per-trajectory", type=int, default=3)
    parser.add_argument(
        "--start-policy", choices=("linspace", "middle"), default="linspace",
        help="evaluate evenly spaced windows or one central validation window",
    )
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--relaxation", type=float, default=0.25)
    parser.add_argument(
        "--port-initialization",
        choices=("reset", "carry", "extrapolate", "state-extrapolate"),
        default="reset",
        help=(
            "initialize each physical step from its predicted state, carry "
            "the preceding accepted Robin state, or extrapolate the two "
            "preceding accepted states; state-extrapolate instead uses the "
            "two model-owned physical states"
        ),
    )
    parser.add_argument(
        "--cover-evolution",
        choices=("fixed", "resample"),
        default="fixed",
        help=(
            "hold one decomposition fixed throughout a rollout or reproduce "
            "the legacy behavior that resamples its offset every time step"
        ),
    )
    parser.add_argument(
        "--oracle-port", action="store_true",
        help=(
            "diagnostic only: supply the reference endpoint Robin state at "
            "every physical step"
        ),
    )
    return parser.parse_args()


def physical_relative_l2(
    prediction: Tensor, target: Tensor, fluid: Tensor,
) -> float:
    weight = fluid.expand_as(prediction)
    return float(
        ((prediction - target).square() * weight).sum().sqrt()
        / (target.square() * weight).sum().sqrt().clamp_min(1.0e-8)
    )


def vorticity_relative_l2(
    prediction: Tensor, target: Tensor, fluid: Tensor, dx: float,
) -> float:
    pred_vort = torch.gradient(prediction[:, 1], spacing=dx, dim=-1)[0] \
        - torch.gradient(prediction[:, 0], spacing=dx, dim=-2)[0]
    true_vort = torch.gradient(target[:, 1], spacing=dx, dim=-1)[0] \
        - torch.gradient(target[:, 0], spacing=dx, dim=-2)[0]
    weight = fluid[:, 0]
    return float(
        ((pred_vort - true_vort).square() * weight).sum().sqrt()
        / (true_vort.square() * weight).sum().sqrt().clamp_min(1.0e-8)
    )


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.horizon, args.starts_per_trajectory, args.sweeps) <= 0 \
            or not 0.0 < args.relaxation <= 1.5:
        raise ValueError("invalid Airfoil rollout controls")
    if args.oracle_port and args.port_initialization != "reset":
        raise ValueError("oracle-port diagnostics do not use temporal prediction")
    if args.trajectory_manifest is not None and args.split_manifest is not None:
        raise ValueError("trajectory-manifest and split-manifest are exclusive")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    reference = payload.get("reference_checkpoint")
    if reference is not None:
        reference_path = Path(reference)
        if payload.get("reference_checkpoint_sha256") != sha256(reference_path):
            raise ValueError("Airfoil reference lineage changed")
    elif payload.get("reference_checkpoint_sha256") is not None:
        raise ValueError("Airfoil checkpoint has an incomplete reference lineage")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = dict(payload["args"])
    configuration = SimpleNamespace(**saved)
    defect_coordinate = uses_defect_coordinate(configuration)
    if not defect_coordinate and payload.get("contract") != ABSOLUTE_CONTRACT:
        raise ValueError("checkpoint is not a supported Airfoil local operator")
    if defect_coordinate and (
        args.oracle_port or args.port_initialization != "reset"
    ):
        raise ValueError(
            "defect-coordinate rollout currently requires target-free reset "
            "initialization"
        )
    if defect_coordinate:
        # ``defect_sni_sweep`` reads relaxation from the checkpoint-owned
        # configuration. The source calibration is allowed to choose a
        # different value, so bind that frozen operating point explicitly.
        configuration.relaxation = args.relaxation
    solver = (
        local_solver_from_payload(payload, device)
        if defect_coordinate else
        absolute_robin_solver_from_payload(payload, device)
    )
    solver.eval().requires_grad_(False)
    statistics = FieldStatistics.from_json(payload["statistics"])
    level = spatial_level(configuration)
    manifest = None
    names = payload["split"]["validation"]
    split_label = "validation"
    protected_opened = 0
    if args.split_manifest is not None:
        manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        names = list(manifest.get(args.split_part, ()))
        if not names:
            raise ValueError(f"split manifest has no {args.split_part} trajectories")
        split_label = f"manifest_{args.split_part}"
        protected_opened = len(names) if args.split_part == "test" else 0
    elif args.trajectory_manifest is not None:
        manifest = json.loads(args.trajectory_manifest.read_text(encoding="utf-8"))
        names = list(manifest.get("selected", ()))
        if not names or set(names) & {
            value for values in payload["split"].values() for value in values
        }:
            raise ValueError("reporting manifest is empty or overlaps source split")
        split_label = "prospective_manifest"
    history = int(getattr(configuration, "history_steps", 2))
    corpus = PairedTrajectoryCorpus(args.data_root, names, "all", history)
    levels = getattr(corpus, level)
    records = []
    try:
        for trajectory, paired in enumerate(corpus.paired):
            trajectory_name = Path(names[trajectory]).stem
            maximum_start = len(paired) - args.horizon
            starts = (
                np.asarray((maximum_start // 2,), dtype=int)
                if args.start_policy == "middle" else
                np.asarray(rollout_starts(
                    len(paired), args.horizon, args.starts_per_trajectory,
                ), dtype=int)
            )
            dataset = levels[trajectory]
            for start in starts:
                first = paired[int(start)]
                previous_physical = first[f"{level}_q_nm1"][None].to(device)
                current_physical = first[f"{level}_q_n"][None].to(device)
                previous_accepted_port = None
                current_accepted_port = None
                fixed_cover_seed = (
                    int(saved["seed"])
                    + trajectory * 10_000_019
                    + int(start)
                )
                for offset in range(args.horizon):
                    index = int(start) + offset
                    reference_sample = dict(paired[index])
                    sample = dict(reference_sample)
                    sample[f"{level}_q_nm1"] = previous_physical[0]
                    sample[f"{level}_q_n"] = current_physical[0]
                    example = make_example(
                        sample,
                        dataset,
                        statistics,
                        configuration,
                        (
                            fixed_cover_seed
                            if args.cover_evolution == "fixed" else
                            int(saved["seed"])
                            + trajectory * 10_000_019 + index
                        ),
                        device,
                        projection_statistics=payload.get(
                            "projection_statistics"
                        ),
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_started = time.perf_counter()
                    current = example["previous"]
                    oracle_state = (
                        target_port(example, configuration)[0]
                        if args.oracle_port else None
                    )
                    if oracle_state is not None:
                        port_state = oracle_state
                    elif args.port_initialization == "state-extrapolate":
                        prior_normalized = (
                            previous_physical - example["mean"]
                        ) / example["std"]
                        current_tiles = example["cover"].tile(
                            current, padding_mode="replicate",
                        )
                        prior_tiles = example["cover"].tile(
                            prior_normalized, padding_mode="replicate",
                        )
                        port_kwargs = {
                            "state_mean": example["mean"].flatten(),
                            "state_std": example["std"].flatten(),
                            "flux_scale": example["flux_scale"],
                            "dx": example["dx"],
                            "viscosity": example["viscosity"],
                            "impedance": (
                                float(saved["mass_impedance"]),
                                float(saved["momentum_impedance"]),
                            ),
                            "face_width": int(saved["face_width"]),
                            "neighbor_faces": example["neighbor_faces"],
                            "valid": example["valid_tiles"],
                        }
                        current_port, _ = absolute_robin_port_2d(
                            current_tiles, **port_kwargs,
                        )
                        prior_port, _ = absolute_robin_port_2d(
                            prior_tiles, **port_kwargs,
                        )
                        port_state = 2.0 * current_port - prior_port
                    elif args.port_initialization == "reset" \
                            or current_accepted_port is None:
                        port_state = None
                    elif args.port_initialization == "carry" \
                            or previous_accepted_port is None:
                        port_state = current_accepted_port
                    else:
                        port_state = (
                            2.0 * current_accepted_port
                            - previous_accepted_port
                        )
                    initial_port = None
                    last_interface = float("nan")
                    last_port = float("nan")
                    last_realization = float("nan")
                    reference_port = None
                    pressure_reference_mean = None
                    for _ in range(args.sweeps):
                        if defect_coordinate:
                            result = defect_sni_sweep(
                                solver, example, configuration, current,
                                imposed_port=port_state,
                                reference_port=reference_port,
                                pressure_reference_mean=pressure_reference_mean,
                                pressure_coarse_correction=bool(getattr(
                                    configuration,
                                    "pressure_coarse_correction", False,
                                )),
                            )
                        else:
                            result = absolute_robin_sni_sweep_2d(
                                solver,
                                example["physical"],
                                example["scalar"],
                                current,
                                example["cover"],
                                state_mean=example["mean"].flatten(),
                                state_std=example["std"].flatten(),
                                flux_scale=example["flux_scale"],
                                dx=example["dx"],
                                viscosity=example["viscosity"],
                                impedance=(
                                    float(saved["mass_impedance"]),
                                    float(saved["momentum_impedance"]),
                                ),
                                relaxation=args.relaxation,
                                face_width=int(saved["face_width"]),
                                neighbor_faces=example["neighbor_faces"],
                                imposed_port=port_state,
                                assembly=str(saved["assembly"]),
                            )
                        if initial_port is None:
                            initial_port = result.imposed_port
                        current = result.field[:, :3]
                        port_state = (
                            oracle_state
                            if oracle_state is not None else result.next_port
                        )
                        reference_port = getattr(
                            result, "reference_port", reference_port,
                        )
                        pressure_reference_mean = getattr(
                            result, "pressure_reference_mean",
                            pressure_reference_mean,
                        )
                        last_interface = float(result.interface_rms)
                        last_port = float(result.port_defect_rms)
                        last_realization = float(getattr(
                            result, "realization_rms", float("nan"),
                        ))
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    inference_seconds = time.perf_counter() - inference_started
                    mean = example["mean"]
                    std = example["std"]
                    prediction_physical = current * std + mean
                    target_physical = example["target"] * std + mean
                    fluid = (
                        sample[f"{level}_sdf_np1"][None].to(device) > 0.0
                    ).to(prediction_physical)
                    records.append({
                        "trajectory": trajectory,
                        "trajectory_name": trajectory_name,
                        "start": int(start),
                        "step": offset + 1,
                        "frame": index,
                        "field_relative_l2": physical_relative_l2(
                            prediction_physical, target_physical, fluid,
                        ),
                        "vorticity_relative_l2": vorticity_relative_l2(
                            prediction_physical, target_physical, fluid,
                            float(dataset.dx),
                        ),
                        "interface_rms": last_interface,
                        "port_defect_rms": last_port,
                        "realization_rms": last_realization,
                        "inference_seconds": inference_seconds,
                        **force_record(
                            prediction_physical,
                            target_physical,
                            current_physical,
                            reference_sample[f"{level}_q_n"][None].to(device),
                            reference_sample, dataset, level,
                        ),
                    })
                    if current_accepted_port is None:
                        previous_accepted_port = initial_port
                    else:
                        previous_accepted_port = current_accepted_port
                    current_accepted_port = port_state
                    previous_physical = current_physical
                    current_physical = prediction_physical
    finally:
        corpus.close()
    summary = {}
    for step in range(1, args.horizon + 1):
        selected = [row for row in records if int(row["step"]) == step]
        summary[f"step_{step:02d}"] = {
            "windows": len(selected),
            **{
                f"{name}_{stat}": float(function([
                    float(row[name]) for row in selected
                ]))
                for name in (
                    "field_relative_l2", "vorticity_relative_l2",
                    "interface_rms", "port_defect_rms", "realization_rms",
                )
                for stat, function in (
                    ("mean", np.nanmean),
                    ("q10", lambda x: np.nanquantile(x, 0.1)),
                    ("q90", lambda x: np.nanquantile(x, 0.9)),
                )
            },
        }
        summary[f"step_{step:02d}"].update({
            f"{name}_relative_l2": scalar_relative_l2(
                selected, f"{name}_prediction", f"{name}_reference",
            )
            for name in (
                "c_drag", "c_lift", "c_drag_corrected", "c_lift_corrected",
            )
        })
        summary[f"step_{step:02d}"].update({
            f"{name}_absolute_error_{stat}": float(function([
                float(row[f"{name}_absolute_error"]) for row in selected
            ]))
            for name in (
                "c_drag", "c_lift", "c_drag_corrected", "c_lift_corrected",
            )
            for stat, function in (
                ("mean", np.nanmean),
                ("q10", lambda x: np.nanquantile(x, 0.1)),
                ("q90", lambda x: np.nanquantile(x, 0.9)),
            )
        })
    result = {
        "contract": "airfoil_local_sni_rollout_v2",
        "local_contract": payload.get("contract"),
        "port_coordinate": saved.get("port_coordinate", "absolute"),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "split": split_label,
        "trajectory_manifest": (
            str(args.trajectory_manifest.resolve())
            if args.trajectory_manifest is not None else None
        ),
        "trajectory_manifest_sha256": (
            sha256(args.trajectory_manifest)
            if args.trajectory_manifest is not None else None
        ),
        "split_manifest": (
            str(args.split_manifest.resolve())
            if args.split_manifest is not None else None
        ),
        "split_manifest_sha256": (
            sha256(args.split_manifest)
            if args.split_manifest is not None else None
        ),
        "protected_test_trajectories_opened": protected_opened,
        "evaluation_trajectories": list(names),
        "reference_used_by_inference": bool(args.oracle_port),
        "model_owned_rollout": True,
        "history_update": "previous <- current; current <- prediction",
        "locality_contract": payload["locality_contract"],
        "horizon": args.horizon,
        "starts_per_trajectory": args.starts_per_trajectory,
        "start_policy": args.start_policy,
        "sweeps_per_physical_step": args.sweeps,
        "relaxation": args.relaxation,
        "port_initialization": args.port_initialization,
        "cover_evolution": args.cover_evolution,
        "oracle_port": bool(args.oracle_port),
        "timing_scope": (
            "one physical step from temporal Robin initialization through "
            "parallel local solves and neighbor exchange; data loading, metric "
            "evaluation, and force reconstruction excluded"
        ),
        "seconds_per_physical_step_mean": float(np.mean([
            row["inference_seconds"] for row in records
        ])),
        "seconds_per_rollout_mean": float(np.mean([
            sum(
                row["inference_seconds"] for row in records
                if row["trajectory"] == trajectory and row["start"] == start
            )
            for trajectory, start in sorted({
                (row["trajectory"], row["start"]) for row in records
            })
        ])),
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    result = evaluate(parse_args())
    print(json.dumps({
        "output": result["contract"],
        "step_20": result["summary"].get("step_20"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
