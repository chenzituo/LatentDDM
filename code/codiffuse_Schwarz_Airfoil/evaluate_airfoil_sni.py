"""Validation-only Airfoil LatentDDM versus interface-conditioned SNI frontier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .airfoil_sni import BoundaryConditionedTileSolver2D, airfoil_sni_sweep
from .airfoil_fixed_point_rollout import _relative_field, _relative_vorticity
from .airfoil_tile_momentum_student import infer
from .airfoil_two_level import fluid_divergence_rms
from .force_diagnostics import penalty_force_coefficients
from .interface_trace import interface_trace_relative_square
from .rollout_tile_momentum_student import _load as load_frozen_student
from .train_multiresolution_projection import batched
from .train_multitrajectory_multiresolution import PairedTrajectoryCorpus
from .train_one_step_multilevel import random_uniform_transition


Tensor = torch.Tensor
CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64)


def state_affine(statistics, reference):
    mean = reference.new_tensor(statistics.state_mean)[None, :, None, None]
    std = reference.new_tensor(statistics.state_std)[None, :, None, None]
    return mean, std


def load_solver(path: Path, student_path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("contract") != "airfoil_boundary_conditioned_sni_local_v1":
        raise ValueError("not an Airfoil boundary-conditioned SNI checkpoint")
    if Path(payload["endpoint"]).resolve() != student_path.resolve():
        raise ValueError("SNI solver and requested student checkpoints differ")
    args = payload["args"]
    solver = BoundaryConditionedTileSolver2D(
        scalar_dim=4, width=int(args["width"]), depth=int(args["depth"]),
    )
    solver.load_state_dict(payload["solver"])
    return solver.to(device).eval().requires_grad_(False), payload


@torch.no_grad()
def prepare_step(student, sample, previous, current, dataset, statistics, args, seed, device):
    batch = batched(sample, "fine")
    batch["q_nm1"] = previous; batch["q_n"] = current
    transition = random_uniform_transition(
        batch, statistics, dataset, device, seed, args.halo, "random",
    )
    one = torch.ones(transition.batch_size, device=device, dtype=current.dtype)
    physical, scalar = transition.conditions(one)
    motion_physical, motion_scalar = transition.endpoint_conditions()
    endpoint = torch.zeros_like(transition.cover.tile(transition.target))
    if device.type == "cuda": torch.cuda.synchronize(device)
    started = time.perf_counter()
    local, _ = student.model._local_operator.correct_endpoint_fields(
        endpoint, physical, scalar, transition.cover.tile_count,
        tile_grid_shape=transition.cover.grid_shape,
        motion_physical=motion_physical, motion_scalar=motion_scalar,
    )
    normalized = transition.cover.assemble_weighted(local, transition.batch_size)
    delta_mean = normalized.new_tensor(statistics.delta_mean)[None, :, None, None]
    delta_std = normalized.new_tensor(statistics.delta_std)[None, :, None, None]
    local_state_tiles = transition.cover.tile(
        transition.q_n, padding_mode="replicate",
    ) + local * delta_std + delta_mean
    local_field = transition.q_n + normalized * delta_std + delta_mean
    if device.type == "cuda": torch.cuda.synchronize(device)
    local_seconds = time.perf_counter() - started
    replay = transition.cover.assemble_weighted(
        local_state_tiles, transition.batch_size,
    )
    replay_max_abs = float((replay - local_field).abs().max())
    if replay_max_abs > 2.0e-5:
        raise RuntimeError(
            "original local endpoint tiles do not replay the Local-POU field: "
            f"max_abs={replay_max_abs:.3e}"
        )

    example = (
        local, transition.cover.tile(transition.target), physical, scalar,
        local.new_zeros(local.shape[0], 32), transition.cover.grid_shape,
        transition, dataset,
    )
    if device.type == "cuda": torch.cuda.synchronize(device)
    started = time.perf_counter()
    corrected = infer(student, example).corrected
    latent_normalized = transition.cover.assemble_weighted(
        corrected, transition.batch_size,
    )
    latent_field = transition.q_n + latent_normalized * delta_std + delta_mean
    if device.type == "cuda": torch.cuda.synchronize(device)
    latent_seconds = local_seconds + time.perf_counter() - started
    return {
        "batch": batch, "transition": transition, "local": local_field,
        "local_delta_tiles": local,
        "local_state_tiles": local_state_tiles,
        "local_tile_replay_max_abs": replay_max_abs,
        "latent_delta_tiles": corrected,
        "latentddm": latent_field, "local_seconds": local_seconds,
        "latent_seconds": latent_seconds,
    }


@torch.no_grad()
def sni_frontier(prepared, solver, statistics, args, tau, assembly, device):
    transition = prepared["transition"]
    batch = prepared["batch"]
    cover = transition.cover
    mean, std = state_affine(statistics, prepared["local"])
    base = (prepared["local"] - mean) / std
    state = (transition.q_n - mean) / std
    sdf = (batch["sdf_np1"].to(device).float() / .25).clamp(-4, 4)
    base_tiles = cover.tile(base, padding_mode="replicate")
    state_tiles = cover.tile(state, padding_mode="replicate")
    sdf_tiles = cover.tile(sdf, padding_mode="replicate")
    scalar = batch["motion_np1"].to(device).float().expand(cover.tile_count, -1)
    current = base
    cumulative = float(prepared["local_seconds"])
    outputs = {}
    for sweep in range(1, max(CHECKPOINTS) + 1):
        if device.type == "cuda": torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = airfoil_sni_sweep(
            solver, base_tiles, state_tiles, sdf_tiles, scalar, current, cover,
            tau=tau, assembly=assembly, chunk_tiles=args.chunk_tiles,
            face_width=args.face_width,
        )
        current = result.field
        if device.type == "cuda": torch.cuda.synchronize(device)
        cumulative += time.perf_counter() - started
        if sweep in CHECKPOINTS:
            outputs[sweep] = {
                "field": current * std + mean,
                "seconds": cumulative,
                "correction_norm": result.correction_norm,
                "interface_norm": result.interface_norm,
                "local_nfe": (sweep + 1) * cover.tile_count,
                "batched_local_calls": sweep + 1,
            }
    return outputs


def metrics(prediction, target, batch, dataset):
    motion = batch["motion_np1"].to(prediction)
    predicted_force = penalty_force_coefficients(
        prediction, batch["sdf_np1"].to(prediction), motion,
        dx=dataset.dx, window=dataset.window,
        penalty_time=dataset.penalty_time, rotation_sign=dataset.rotation_sign,
    )
    target_force = penalty_force_coefficients(
        target, batch["sdf_np1"].to(prediction), motion,
        dx=dataset.dx, window=dataset.window,
        penalty_time=dataset.penalty_time, rotation_sign=dataset.rotation_sign,
    )
    force_error = np.hypot(
        float(predicted_force.drag - target_force.drag),
        float(predicted_force.lift - target_force.lift),
    ) / max(np.hypot(float(target_force.drag), float(target_force.lift)), 1e-12)
    trace = torch.sqrt(interface_trace_relative_square(
        prediction, target, motion, dx=dataset.dx, window=dataset.window,
        thickness=float(dataset.physical_parameters["thickness"]),
        pivot_fraction=float(dataset.physical_parameters["pivot_fraction"]),
        stations=64,
    ).clamp_min(0))
    return {
        "field_relative_l2": _relative_field(prediction, target),
        "vorticity_relative_l2": _relative_vorticity(prediction, target, dataset.dx),
        "interface_relative_l2": float(trace),
        "divergence_rms": float(fluid_divergence_rms(
            prediction, batch["sdf_np1"].to(prediction), dataset.dx,
        ).mean()),
        "force_relative_l2": float(force_error),
        # Retain the signed coefficients so rollout evaluators can compute one
        # relative norm over a complete force trajectory.  Averaging the
        # per-step relative errors above is ill-conditioned when the physical
        # load crosses zero and is not comparable with ``sequence_summary``.
        "drag_prediction": float(predicted_force.drag),
        "drag_reference": float(target_force.drag),
        "lift_prediction": float(predicted_force.lift),
        "lift_reference": float(target_force.lift),
        "drag_relative_abs": abs(float(predicted_force.drag - target_force.drag))
            / max(abs(float(target_force.drag)), 1e-12),
        "lift_relative_abs": abs(float(predicted_force.lift - target_force.lift))
            / max(abs(float(target_force.lift)), 1e-12),
    }


def aggregate(rows):
    output = {}
    groups = sorted({(row["method"], int(row["sweep"])) for row in rows})
    numeric = sorted({key for row in rows for key, value in row.items()
                      if isinstance(value, (int, float))} - {"sweep", "trajectory", "item"})
    for method, sweep in groups:
        selected = [row for row in rows if row["method"] == method and row["sweep"] == sweep]
        summary = {"examples": len(selected)}
        for key in numeric:
            values = np.asarray([float(row[key]) for row in selected if key in row])
            if values.size:
                summary[f"{key}_mean"] = float(values.mean())
                summary[f"{key}_std_population"] = float(values.std())
        output[f"{method}/sweep_{sweep:03d}"] = summary
    return output


def calibration_examples(corpus, count_per_trajectory):
    for trajectory, paired in enumerate(corpus.paired):
        indices = np.linspace(0, len(paired) - 1, count_per_trajectory, dtype=int)
        for ordinal, index in enumerate(indices):
            yield trajectory, ordinal, int(index), paired[int(index)]


def select(candidates):
    ranked = []
    for name, rows in candidates.items():
        summary = aggregate(rows)
        for sweep in CHECKPOINTS:
            row = summary[f"sni/sweep_{sweep:03d}"]
            ranked.append((row["field_relative_l2_mean"], row["seconds_mean"], name, sweep))
    return min(ranked)[2:]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--sni-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--face-width", type=int, default=1)
    parser.add_argument("--chunk-tiles", type=int, default=64)
    parser.add_argument("--selection-samples-per-trajectory", type=int, default=1)
    parser.add_argument("--validation-samples-per-trajectory", type=int, default=8)
    parser.add_argument("--raw-taus", type=float, nargs="+", default=(.05, .1, .2))
    parser.add_argument("--pou-taus", type=float, nargs="+", default=(.5, .75, 1.0))
    parser.add_argument("--seed", type=int, default=20260806)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_payload, base, student, statistics = load_frozen_student(
        argparse.Namespace(checkpoint=args.student), device,
    )
    solver, solver_payload = load_solver(args.sni_checkpoint, args.student, device)
    corpus = PairedTrajectoryCorpus(args.data_root, base["split"]["validation"], "all")
    configs = [
        (f"huang_raw_additive_tau_{tau:g}", "huang_raw_additive", tau)
        for tau in args.raw_taus
    ] + [
        (f"normalized_pou_tau_{tau:g}", "normalized_pou", tau)
        for tau in args.pou_taus
    ]
    try:
        prepared_calibration = []
        for trajectory, ordinal, index, sample in calibration_examples(
            corpus, args.selection_samples_per_trajectory,
        ):
            dataset = corpus.fine[trajectory]
            previous = sample["fine_q_nm1"][None].to(device)
            current = sample["fine_q_n"][None].to(device)
            seed = args.seed + trajectory * 1_000_003 + ordinal
            prepared_calibration.append((
                trajectory, index, dataset,
                prepare_step(student, sample, previous, current, dataset,
                             statistics, args, seed, device),
            ))
        candidates = {}
        for name, assembly, tau in configs:
            rows = []
            for trajectory, index, dataset, prepared in prepared_calibration:
                target = prepared["transition"].q_np1
                for sweep, output in sni_frontier(
                    prepared, solver, statistics, args, tau, assembly, device,
                ).items():
                    rows.append({
                        "method": "sni", "sweep": sweep,
                        "trajectory": trajectory, "item": index,
                        "seconds": output["seconds"],
                        **metrics(output["field"], target, prepared["batch"], dataset),
                    })
            candidates[name] = rows
        selected_name, selected_sweep = select(candidates)
        _, selected_assembly, selected_tau = next(
            item for item in configs if item[0] == selected_name
        )

        final_rows = []
        for trajectory, ordinal, index, sample in calibration_examples(
            corpus, args.validation_samples_per_trajectory,
        ):
            dataset = corpus.fine[trajectory]
            previous = sample["fine_q_nm1"][None].to(device)
            current = sample["fine_q_n"][None].to(device)
            seed = args.seed + trajectory * 1_000_003 + ordinal
            prepared = prepare_step(
                student, sample, previous, current, dataset, statistics,
                args, seed, device,
            )
            target = prepared["transition"].q_np1
            base_common = {"trajectory": trajectory, "item": index, "sweep": 0}
            tile_count = prepared["transition"].cover.tile_count
            final_rows.append({
                **base_common, "method": "local", "seconds": prepared["local_seconds"],
                "local_nfe": tile_count, "batched_local_calls": 1,
                **metrics(prepared["local"], target, prepared["batch"], dataset),
            })
            final_rows.append({
                **base_common, "method": "latentddm", "seconds": prepared["latent_seconds"],
                "local_nfe": tile_count, "batched_local_calls": 1,
                **metrics(prepared["latentddm"], target, prepared["batch"], dataset),
            })
            for sweep, output in sni_frontier(
                prepared, solver, statistics, args, selected_tau,
                selected_assembly, device,
            ).items():
                final_rows.append({
                    "trajectory": trajectory, "item": index, "method": "sni",
                    "sweep": sweep, "seconds": output["seconds"],
                    "local_nfe": output["local_nfe"],
                    "batched_local_calls": output["batched_local_calls"],
                    "correction_norm": output["correction_norm"],
                    "sni_interface_norm": output["interface_norm"],
                    **metrics(output["field"], target, prepared["batch"], dataset),
                })
    finally:
        corpus.close()
    fields = sorted({key for row in final_rows for key in row})
    with (args.output_dir / "one_step_frontier.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(final_rows)
    payload = {
        "contract": "airfoil_huang_style_boundary_conditioned_sni_validation_v1",
        "split": "validation", "protected_test_trajectories_opened": 0,
        "student": str(args.student.resolve()),
        "student_epoch": student_payload["epoch"],
        "sni_checkpoint": str(args.sni_checkpoint.resolve()),
        "selected": {"configuration": selected_name, "assembly": selected_assembly,
                     "tau": selected_tau, "validation_selected_sweep": selected_sweep},
        "candidate_summary": {name: aggregate(rows) for name, rows in candidates.items()},
        "frontier": aggregate(final_rows),
        "selection_overlap_warning": (
            "calibration examples are a subset of the reported validation frontier; "
            "protected test remains closed"
        ),
        "training_boundary_contract": solver_payload["training_boundary_contract"],
        "rollout_status": "pending selected-sweep model-owned rollout",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["selected"], indent=2), flush=True)


if __name__ == "__main__":
    main()
