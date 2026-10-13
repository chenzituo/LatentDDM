"""Model-owned rollout for absolute or native-port Airfoil LatentDDM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .absolute_robin_latentddm import (
    CONTRACT as ABSOLUTE_CONTRACT,
    AirfoilRobinLatentDDM,
)
from .absolute_robin_sni import absolute_robin_port_2d
from .absolute_robin_state_latentddm import (
    CONTRACT as STATE_INPUT_CONTRACT,
    AirfoilRobinStateLatentDDM,
)
from .airfoil_trace_sni import assemble_subdomains_2d
from .dataset import FieldStatistics
from .evaluate_global_validation_protocol import rollout_starts
from .rollout_absolute_robin_sni import (
    force_record,
    physical_relative_l2,
    scalar_relative_l2,
    vorticity_relative_l2,
)
from .train_absolute_robin_bvp import make_example, sha256
from .train_absolute_robin_latentddm import (
    attach_configuration,
    load_local,
    spatial_level,
)
from .native_port_latentddm import CONTRACT as NATIVE_PORT_CONTRACT
from .train_native_port_latentddm import load_model as load_native_port_model
from .train_multitrajectory_multiresolution import PairedTrajectoryCorpus


Tensor = torch.Tensor


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
    )
    return parser.parse_args()


def interface_diagnostics(
    model: AirfoilRobinLatentDDM,
    example: dict[str, object],
    output,
) -> tuple[float, float, float]:
    cover = example["cover"]
    batch = example["previous"].shape[0]
    exchange = assemble_subdomains_2d(
        cover, output.final_tiles, batch, "disjoint_core",
    )
    consensus = cover.tile(exchange, padding_mode="replicate")
    collar = (output.face_roles.sum(1, keepdim=True) > 0.0).to(consensus)
    interface = (
        ((output.final_tiles - consensus).square() * collar).sum()
        / (collar.sum() * output.final_tiles.shape[1]).clamp_min(1.0)
    ).sqrt()
    kwargs = model._port_kwargs(example)
    realized, _ = absolute_robin_port_2d(output.final_tiles, **kwargs)
    proposed, _ = absolute_robin_port_2d(consensus, **kwargs)
    active = output.face_roles.repeat_interleave(3, dim=1)
    realization = (
        ((realized - output.corrected_port).square() * active).sum()
        / active.sum().clamp_min(1.0)
    ).sqrt()
    defect = (
        ((proposed - output.corrected_port).square() * active).sum()
        / active.sum().clamp_min(1.0)
    ).sqrt()
    return float(interface), float(defect), float(realization)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.horizon, args.starts_per_trajectory) <= 0:
        raise ValueError("invalid Airfoil LatentDDM rollout controls")
    if args.trajectory_manifest is not None and args.split_manifest is not None:
        raise ValueError("trajectory-manifest and split-manifest are exclusive")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    composer_contract = payload.get("contract")
    if composer_contract not in (
        ABSOLUTE_CONTRACT, STATE_INPUT_CONTRACT, NATIVE_PORT_CONTRACT,
    ):
        raise ValueError("checkpoint is not a supported Airfoil LatentDDM")
    local_path = Path(payload["local_checkpoint"])
    if payload.get("local_checkpoint_sha256") != sha256(local_path):
        raise ValueError("frozen Airfoil local-operator lineage changed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if composer_contract == NATIVE_PORT_CONTRACT:
        model, payload, local_payload, configuration = load_native_port_model(
            args.checkpoint, device,
        )
    elif composer_contract == STATE_INPUT_CONTRACT:
        solver, local_payload, configuration = load_local(local_path, device)
        saved = payload["args"]
        model = AirfoilRobinStateLatentDDM(
            solver,
            width=int(saved["width"]), heads=int(saved["heads"]),
            depth=int(saved["depth"]), trace_modes=int(saved["trace_modes"]),
            spatial_bins=int(saved["spatial_bins"]),
            code_scale=payload["composer"]["code_scale"],
            correction_limit=float(saved["correction_limit"]),
            recompute_port=bool(saved["recompute_port"]),
            centered_communication=bool(saved.get("centered_communication", True)),
            state_chart=str(saved.get("state_chart", "collar")),
            state_bins=int(saved.get("state_bins", 4)),
        ).to(device)
        model.load_composer_state_dict(payload["composer"])
    else:
        solver, local_payload, configuration = load_local(local_path, device)
        saved = payload["args"]
        model = AirfoilRobinLatentDDM(
            solver,
            width=int(saved["width"]), heads=int(saved["heads"]),
            depth=int(saved["depth"]), trace_modes=int(saved["trace_modes"]),
            spatial_bins=int(saved["spatial_bins"]),
            code_scale=payload["composer"]["code_scale"],
            code_center=payload.get("code_center"),
            communication_mode=str(payload.get(
                "communication_mode", saved.get(
                    "communication_mode", "centered_delta",
                ),
            )),
        ).to(device)
        model.load_composer_state_dict(payload["composer"])
    model.eval()
    saved = payload["args"]
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
                fixed_cover_seed = (
                    int(configuration.seed)
                    + trajectory * 10_000_019 + int(start)
                )
                for offset in range(args.horizon):
                    index = int(start) + offset
                    reference_sample = dict(paired[index])
                    sample = dict(reference_sample)
                    sample[f"{level}_q_nm1"] = previous_physical[0]
                    sample[f"{level}_q_n"] = current_physical[0]
                    example = attach_configuration(make_example(
                        sample, dataset, statistics, configuration,
                        fixed_cover_seed, device,
                        projection_statistics=local_payload.get(
                            "projection_statistics"
                        ),
                    ), configuration)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    output = model(example)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    if composer_contract == NATIVE_PORT_CONTRACT:
                        interface = float(output.interface_rms)
                        port_defect = float(output.port_defect_rms)
                        realization = float(output.realization_rms)
                    else:
                        interface, port_defect, realization = interface_diagnostics(
                            model, example, output,
                        )
                    prediction_physical = (
                        output.field * example["std"] + example["mean"]
                    )
                    target_physical = (
                        example["target"] * example["std"] + example["mean"]
                    )
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
                        "interface_rms": interface,
                        "port_defect_rms": port_defect,
                        "realization_rms": realization,
                        "inference_seconds": elapsed,
                        **force_record(
                            prediction_physical, target_physical,
                            current_physical,
                            reference_sample[f"{level}_q_n"][None].to(device),
                            reference_sample, dataset, level,
                        ),
                    })
                    previous_physical = current_physical
                    current_physical = prediction_physical
    finally:
        corpus.close()
    summary = {}
    for step in range(1, args.horizon + 1):
        selected = [row for row in records if int(row["step"]) == step]
        entry = {
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
        entry.update({
            f"{name}_relative_l2": scalar_relative_l2(
                selected, f"{name}_prediction", f"{name}_reference",
            )
            for name in (
                "c_drag", "c_lift", "c_drag_corrected", "c_lift_corrected",
            )
        })
        entry.update({
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
        summary[f"step_{step:02d}"] = entry
    rollouts = sorted({
        (row["trajectory"], row["start"]) for row in records
    })
    result = {
        "contract": "airfoil_latentddm_rollout_v2",
        "composer_contract": composer_contract,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "local_checkpoint": str(local_path.resolve()),
        "local_checkpoint_sha256": payload["local_checkpoint_sha256"],
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
        "reference_used_by_inference": False,
        "model_owned_rollout": True,
        "history_update": "previous <- current; current <- prediction",
        "locality_contract": payload["locality_contract"],
        "horizon": args.horizon,
        "starts_per_trajectory": args.starts_per_trajectory,
        "start_policy": args.start_policy,
        "local_operator_calls_per_physical_step": 2,
        "communication_layers": int(saved["depth"]),
        "communication_mode": (
            "local_state_input"
            if composer_contract == STATE_INPUT_CONTRACT else
            str(payload.get(
                "communication_mode", saved.get(
                    "communication_mode", "centered_delta",
                ),
            ))
        ),
        "timing_scope": (
            "two parallel local-operator batches plus one relative-attention "
            "stage; data loading, diagnostics, and force reconstruction excluded"
        ),
        "seconds_per_physical_step_mean": float(np.mean([
            row["inference_seconds"] for row in records
        ])),
        "seconds_per_rollout_mean": float(np.mean([
            sum(
                row["inference_seconds"] for row in records
                if row["trajectory"] == trajectory and row["start"] == start
            )
            for trajectory, start in rollouts
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
