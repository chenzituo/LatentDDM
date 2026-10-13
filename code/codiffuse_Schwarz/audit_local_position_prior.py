"""Audit global-position and frozen-prior leakage in the three local maps.

This is a validation-only mechanism diagnostic.  It asks three distinct
questions which are easy to conflate:

1. Does a nominally local endpoint receive explicit absolute coordinates?
2. Can a coordinate-free endpoint distinguish geometry-free cubes anyway?
3. Is the learned boundary map invariant to a change of response-chart base
   when the *absolute* imposed boundary state is held fixed?

The third condition is automatic for an absolute boundary-to-solution map,
as used by the Darcy local BVP.  A large defect exposes anchoring to the
frozen initial prediction and explains why repeated SNI need not reproduce a
classical Schwarz fixed-point map.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from types import SimpleNamespace

import torch


Tensor = torch.Tensor


def relative_rms(difference: Tensor, reference: Tensor, weight: Tensor) -> float:
    weight = weight.to(difference)
    while weight.ndim < difference.ndim:
        weight = weight.unsqueeze(1)
    numerator = (difference.square() * weight).sum()
    denominator = (reference.square() * weight).sum().clamp_min(1.0e-12)
    return float((numerator / denominator).sqrt())


def weighted_rms(field: Tensor, weight: Tensor) -> float:
    weight = weight.to(field)
    while weight.ndim < field.ndim:
        weight = weight.unsqueeze(1)
    denominator = (weight.sum() * field.shape[1]).clamp_min(1.0)
    return float(((field.square() * weight).sum() / denominator).sqrt())


@torch.inference_mode()
def audit_car(args: argparse.Namespace, device: torch.device) -> dict:
    from codiffuse_Schwarz_Car.boundary_local_backbone import (
        CarBoundaryLocalBackbone,
    )
    from codiffuse_Schwarz_Car.car_flux_conditioned_sni import (
        directional_state_trace_3d,
        neighboring_face_mask_3d,
    )
    from codiffuse_Schwarz_Car.car_flux_data import GroupedFluxStateDataset
    from codiffuse_Schwarz_Car.cover3d import CartesianCover3D
    from codiffuse_Schwarz_Car.data import (
        VelocityStatistics,
        discover_volumes,
        load_volume,
        volume_tensors,
    )

    backbone = CarBoundaryLocalBackbone(args.car_boundary_checkpoint, device)
    endpoint = backbone.endpoint
    velocity_statistics = VelocityStatistics.from_dict(
        backbone.payload["statistics"],
    )
    validation_paths = discover_volumes(args.car_data_root, "val")
    if not validation_paths:
        validation_paths = discover_volumes(args.car_data_root, "validation")
    if not validation_paths:
        raise FileNotFoundError("Car validation volumes were not found")

    # Complete-cover invariance: truly feature-free cubes have exactly the
    # same coordinate-free input and therefore cannot carry an x/y/z prior.
    data = load_volume(validation_paths[0])
    physical, target, scalar = volume_tensors(
        data, velocity_statistics, include_coordinates=False,
    )
    cover = CartesianCover3D(*physical.shape[-3:], args.car_core, args.car_halo)
    tiles = cover.tile(physical[None]).to(device)
    tiles[:, -1:] = cover.valid_tiles(1, device, tiles.dtype)
    outputs = []
    controls = scalar[None].expand(cover.tile_count, -1).to(device)
    for start in range(0, cover.tile_count, args.chunk_tiles):
        stop = start + args.chunk_tiles
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs.append(endpoint(tiles[start:stop], controls[start:stop]).float())
    local = torch.cat(outputs)
    feature_free = (
        (tiles[:, :1] - 1.0).abs().flatten(1).amax(1) < 1.0e-6
    ) & (
        (tiles[:, 1:2] - 1.0).abs().flatten(1).amax(1) < 1.0e-6
    ) & (
        (tiles[:, 2:3] - 1.0).abs().flatten(1).amax(1) < 1.0e-6
    )
    indices = torch.nonzero(feature_free, as_tuple=False).flatten()
    free_input_variation = free_output_variation = float("nan")
    free_target_variation = float("nan")
    if len(indices) >= 2:
        free_inputs = tiles[indices]
        free_outputs = local[indices]
        target_tiles = cover.tile(target[None].to(device))[indices]
        free_input_variation = weighted_rms(
            free_inputs - free_inputs.mean(0, keepdim=True),
            torch.ones_like(free_inputs[:, :1]),
        )
        free_output_variation = weighted_rms(
            free_outputs - free_outputs.mean(0, keepdim=True),
            torch.ones_like(free_outputs[:, :1]),
        )
        free_target_variation = weighted_rms(
            target_tiles - target_tiles.mean(0, keepdim=True),
            torch.ones_like(target_tiles[:, :1]),
        )

    # Response-chart gauge test on a deterministic connected validation group.
    config = backbone.payload["args"]
    group_shape = tuple(int(v) for v in config["group_shape"])
    dataset = GroupedFluxStateDataset(
        validation_paths,
        args.car_pressure_root,
        velocity_statistics,
        __import__(
            "codiffuse_Schwarz_Car.car_flux_data", fromlist=["PressureStatistics"]
        ).PressureStatistics.from_dict(backbone.payload["pressure_statistics"]),
        int(config["core"]),
        int(config["halo"]),
        group_shape,
        1,
        args.seed,
        sampling="index",
    )
    from codiffuse_Schwarz_Car.train_car_trace_sni import prepare_batch

    raw_group = {
        key: (value[None] if isinstance(value, Tensor) else value)
        for key, value in dataset[0].items()
    }
    batch = prepare_batch(
        endpoint,
        raw_group,
        group_shape,
        int(config["core"]),
        int(config["halo"]),
        backbone.state_mean,
        backbone.state_std,
        backbone.flux_scale,
        device,
        loss_collar_regions=int(config.get("loss_collar_regions", 1)),
        assembly=str(backbone.payload.get("assembly", "normalized_overlap")),
    )
    base = batch["base"]
    target_tiles = batch["target_tiles"]
    neighbor_faces = neighboring_face_mask_3d(
        group_shape, batch=1, device=device, dtype=base.dtype,
    )
    common = {
        "state_mean": backbone.state_mean,
        "state_std": backbone.state_std,
        "face_width": int(config["face_width"]),
        "incoming_temperature": float(config["incoming_temperature"]),
        "neighbor_faces": neighbor_faces,
    }
    trace, roles, incoming = directional_state_trace_3d(
        target_tiles, base, **common,
    )
    original = backbone.solver(
        base, batch["physical"], batch["scalars"], trace, roles, incoming,
    )
    gauge = base.new_tensor((0.25, -0.15, 0.10, 0.20)).reshape(1, 4, 1, 1, 1)
    shifted_base = base + gauge * batch["physical"][:, -1:]
    shifted_trace, shifted_roles, shifted_incoming = directional_state_trace_3d(
        target_tiles, shifted_base, **common,
    )
    shifted = backbone.solver(
        shifted_base,
        batch["physical"],
        batch["scalars"],
        shifted_trace,
        shifted_roles,
        shifted_incoming,
    )
    valid = batch["bvp_valid"]
    return {
        "endpoint_contract": backbone.endpoint_payload.get("contract"),
        "explicit_absolute_coordinate_channels": False,
        "complete_cover_grid_shape": list(cover.grid_shape),
        "complete_cover_tiles": cover.tile_count,
        "feature_free_tiles": int(len(indices)),
        "feature_free_fraction": float(len(indices) / cover.tile_count),
        "feature_free_input_variation_rms": free_input_variation,
        "feature_free_endpoint_variation_rms": free_output_variation,
        "feature_free_target_velocity_variation_rms": free_target_variation,
        "gauge_shift_rms": weighted_rms(
            shifted_base - base, batch["physical"][:, -1:],
        ),
        "gauge_prediction_defect_relative_to_original": relative_rms(
            shifted - original, original, valid,
        ),
        "oracle_bvp_error_original": relative_rms(
            original - target_tiles, target_tiles, valid,
        ),
        "oracle_bvp_error_after_equivalent_base_shift": relative_rms(
            shifted - target_tiles, target_tiles, valid,
        ),
        "absolute_boundary_state_preserved_max_abs": float((
            (
                shifted_base.repeat(1, 6, 1, 1, 1) + shifted_trace
                - base.repeat(1, 6, 1, 1, 1) - trace
            ) * roles.repeat_interleave(4, dim=1)
        ).abs().max()),
    }


def _airfoil_local_next(
    endpoint,
    physical: Tensor,
    scalar: Tensor,
    motion_physical: Tensor,
    motion_scalar: Tensor,
    cover,
    state_mean: Tensor,
    state_std: Tensor,
    delta_mean: Tensor,
    delta_std: Tensor,
) -> tuple[Tensor, Tensor]:
    seed = torch.zeros_like(physical[:, :3])
    raw, _ = endpoint.correct_endpoint_fields(
        seed,
        physical,
        scalar,
        cover.tile_count,
        tile_grid_shape=cover.grid_shape,
        motion_physical=motion_physical,
        motion_scalar=motion_scalar,
    )
    mean = state_mean.reshape(1, 3, 1, 1)
    std = state_std.reshape(1, 3, 1, 1)
    dmean = delta_mean.reshape(1, 3, 1, 1)
    dstd = delta_std.reshape(1, 3, 1, 1)
    previous = physical[:, :3] * std + mean
    return raw.float(), previous + raw.float() * dstd + dmean


@torch.inference_mode()
def audit_airfoil(args: argparse.Namespace, device: torch.device) -> dict:
    from codiffuse_Schwarz_Airfoil.airfoil_trace_sni import (
        TraceConditionedTileSolver2D,
        directional_state_trace_2d,
    )
    from codiffuse_Schwarz_Airfoil.evaluate_state_input_rollout import (
        load_model,
        statistics_for,
    )
    from codiffuse_Schwarz_Airfoil.field_token_response_source import (
        load_local_endpoint_source,
    )
    from codiffuse_Schwarz_Airfoil.train_airfoil_trace_sni import example
    from codiffuse_Schwarz_Airfoil.train_multiresolution_projection import batched
    from codiffuse_Schwarz_Airfoil.train_multitrajectory_multiresolution import (
        PairedTrajectoryCorpus,
    )
    from codiffuse_Schwarz_Airfoil.train_one_step_multilevel import (
        random_uniform_transition,
    )

    selected, split_base, model, statistics = load_model(
        args.airfoil_composer_checkpoint, device,
    )
    corpus = PairedTrajectoryCorpus(
        args.airfoil_data_root, split_base["split"]["validation"], "all",
    )
    try:
        sample = corpus.paired[0][0]
        dataset = corpus.fine[0]
        batch = batched(sample, "fine")
        previous = sample["fine_q_nm1"][None].to(device)
        current = sample["fine_q_n"][None].to(device)
        batch["q_nm1"] = previous
        batch["q_n"] = current
        transition = random_uniform_transition(
            batch, statistics, dataset, device, args.seed, args.airfoil_halo,
            "random",
        )
        one = torch.ones(
            transition.batch_size, device=device, dtype=current.dtype,
        )
        physical, scalar = transition.conditions(one)
        motion_physical, motion_scalar = transition.endpoint_conditions()
        endpoint = model.response_student.model._local_operator
        probe = torch.zeros_like(physical[:, :3])
        state_mean, state_std, delta_mean, delta_std = statistics_for(model, probe)
        raw, local_next = _airfoil_local_next(
            endpoint, physical, scalar, motion_physical, motion_scalar,
            transition.cover, state_mean, state_std, delta_mean, delta_std,
        )
        original_field = transition.cover.assemble_weighted(
            local_next, transition.batch_size,
        )

        # Convert absolute spatial and scalar positions to a canonical
        # tile-centered chart while preserving q_n, SDF, wall motion, and valid.
        if physical.shape[1] not in (12, 15) or scalar.shape[1] < 2:
            raise RuntimeError(
                f"unexpected Airfoil condition shapes {physical.shape}/{scalar.shape}"
            )
        canonical_physical = physical.clone()
        canonical_motion = motion_physical.clone()
        canonical_scalar = scalar.clone()
        canonical_motion_scalar = motion_scalar.clone()
        canonical_physical[:, 9:10] -= scalar[:, -2:-1, None, None]
        canonical_physical[:, 10:11] -= scalar[:, -1:, None, None]
        canonical_motion[:, 9:10] -= motion_scalar[:, -2:-1, None, None]
        canonical_motion[:, 10:11] -= motion_scalar[:, -1:, None, None]
        canonical_scalar[:, -2:] = 0.0
        canonical_motion_scalar[:, -2:] = 0.0
        canonical_raw, canonical_next = _airfoil_local_next(
            endpoint,
            canonical_physical,
            canonical_scalar,
            canonical_motion,
            canonical_motion_scalar,
            transition.cover,
            state_mean,
            state_std,
            delta_mean,
            delta_std,
        )
        canonical_field = transition.cover.assemble_weighted(
            canonical_next, transition.batch_size,
        )
        target = transition.q_np1
        position_probe = {
            "physical_channels": int(physical.shape[1]),
            "scalar_channels": int(scalar.shape[1]),
            "explicit_spatial_xy_channels": [9, 10],
            "explicit_tile_center_scalar_channels": [
                int(scalar.shape[1] - 2), int(scalar.shape[1] - 1),
            ],
            "canonical_position_raw_delta_change_relative": relative_rms(
                canonical_raw - raw, raw, torch.ones_like(raw[:, :1]),
            ),
            "canonical_position_next_state_change_relative": relative_rms(
                canonical_field - original_field,
                original_field,
                torch.ones_like(original_field[:, :1]),
            ),
            "original_local_field_relative_l2": relative_rms(
                original_field - target, target, torch.ones_like(target[:, :1]),
            ),
            "canonical_position_local_field_relative_l2": relative_rms(
                canonical_field - target, target, torch.ones_like(target[:, :1]),
            ),
        }

        # Gauge test the independently pretrained boundary response map.
        boundary_saved = torch.load(
            args.airfoil_boundary_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        boundary_args = SimpleNamespace(**boundary_saved["args"])
        _, boundary_base, boundary_endpoint, boundary_stats, _ = (
            load_local_endpoint_source(
                Path(boundary_saved["endpoint"]), device,
            )
        )
        boundary_corpus = PairedTrajectoryCorpus(
            args.airfoil_data_root,
            boundary_base["split"]["validation"],
            "all",
        )
        try:
            boundary_batch = example(
                boundary_endpoint,
                boundary_corpus.paired[0][0],
                boundary_corpus.fine[0],
                boundary_stats,
                boundary_args,
                args.seed,
                device,
            )
        finally:
            boundary_corpus.close()
        solver = TraceConditionedTileSolver2D(
            width=int(boundary_saved["args"]["width"]),
            depth=int(boundary_saved["args"]["depth"]),
        ).to(device)
        solver.load_state_dict(boundary_saved["solver"], strict=True)
        solver.eval().requires_grad_(False)
        base = boundary_batch["base"]
        target_tiles = boundary_batch["target_tiles"]
        common = {
            "state_mean": boundary_batch["mean"].flatten(),
            "state_std": boundary_batch["std"].flatten(),
            "face_width": int(boundary_saved["args"]["face_width"]),
            "incoming_temperature": float(
                boundary_saved["args"]["incoming_temperature"]
            ),
            "neighbor_faces": boundary_batch["neighbor_faces"],
        }
        trace, roles, incoming = directional_state_trace_2d(
            target_tiles, base, **common,
        )
        original = solver(
            base,
            boundary_batch["previous"],
            boundary_batch["sdf"],
            boundary_batch["scalar"],
            trace,
            roles,
            incoming,
        )
        gauge = base.new_tensor((0.25, -0.15, 0.20)).reshape(1, 3, 1, 1)
        shifted_base = base + gauge
        shifted_trace, shifted_roles, shifted_incoming = directional_state_trace_2d(
            target_tiles, shifted_base, **common,
        )
        shifted = solver(
            shifted_base,
            boundary_batch["previous"],
            boundary_batch["sdf"],
            boundary_batch["scalar"],
            shifted_trace,
            shifted_roles,
            shifted_incoming,
        )
        gauge_probe = {
            "gauge_shift_rms": weighted_rms(
                shifted_base - base, torch.ones_like(base[:, :1]),
            ),
            "gauge_prediction_defect_relative_to_original": relative_rms(
                shifted - original, original, torch.ones_like(original[:, :1]),
            ),
            "oracle_bvp_error_original": relative_rms(
                original - target_tiles,
                target_tiles,
                torch.ones_like(target_tiles[:, :1]),
            ),
            "oracle_bvp_error_after_equivalent_base_shift": relative_rms(
                shifted - target_tiles,
                target_tiles,
                torch.ones_like(target_tiles[:, :1]),
            ),
            "absolute_boundary_state_preserved_max_abs": float((
                (
                    shifted_base.repeat(1, 4, 1, 1) + shifted_trace
                    - base.repeat(1, 4, 1, 1) - trace
                ) * roles.repeat_interleave(3, dim=1)
            ).abs().max()),
        }
        return {
            "endpoint_contract": selected.get("contract"),
            "position_probe": position_probe,
            "boundary_chart_gauge_probe": gauge_probe,
        }
    finally:
        corpus.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--car-data-root", type=Path,
        default=Path("/orcd/pool/003/zituo/DrivAerNetPlusPlus/derived/cartesian_0p1m"),
    )
    parser.add_argument(
        "--car-pressure-root", type=Path,
        default=Path("/orcd/pool/003/zituo/DrivAerNetPlusPlus/derived/cartesian_pressure_0p1m"),
    )
    parser.add_argument(
        "--car-boundary-checkpoint", type=Path,
        default=Path(
            "codiffuse_Schwarz_Car/outputs/"
            "car_boundary_local_state_face4_warm_seed20260811/checkpoint_best.pt"
        ),
    )
    parser.add_argument("--car-core", type=int, default=16)
    parser.add_argument("--car-halo", type=int, default=4)
    parser.add_argument(
        "--airfoil-data-root", type=Path,
        default=Path(
            "/orcd/scratch/orcd/003/zituo/AMR_Diffusion_Foundation/"
            "codiffuse_airfoil"
        ),
    )
    parser.add_argument(
        "--airfoil-composer-checkpoint", type=Path,
        default=Path(
            "codiffuse_Schwarz_Airfoil/outputs/"
            "state_input_boundary_selected_v2/checkpoint_best.pt"
        ),
    )
    parser.add_argument(
        "--airfoil-boundary-checkpoint", type=Path,
        default=Path(
            "codiffuse_Schwarz_Airfoil/outputs/"
            "airfoil_boundary_local_state_overlap_seed20260811/checkpoint_best.pt"
        ),
    )
    parser.add_argument("--airfoil-halo", type=int, default=4)
    parser.add_argument("--chunk-tiles", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "contract": "three_case_local_position_prior_audit_v1",
        "device": str(device),
        "darcy": {
            "explicit_global_tile_position": False,
            "local_coordinates_reset_per_crop": True,
            "boundary_semantics": "absolute Dirichlet trace",
            "response_chart_base": None,
            "chart_gauge_invariance": "by construction",
        },
        "car": audit_car(args, device),
        "airfoil": audit_airfoil(args, device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
