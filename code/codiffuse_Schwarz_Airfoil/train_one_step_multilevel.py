"""One-step fine-tuning with random Cartesian Schwarz Flow integration."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from codiffuse_Schwarz_Airfoil.cartesian_cover import (
    CartesianCover,
    PlacementMode,
    choose_offset,
)
from codiffuse_Schwarz_Airfoil.coupled_trajectory import (
    ControlledTransition,
    CyclicWindowDataset,
    controlled_velocity,
    endpoint_trace_velocity,
    pnp_endpoint_trace_sample,
)
from codiffuse_Schwarz_Airfoil.dataset import FieldStatistics, denormalize_delta
from codiffuse_Schwarz_Airfoil.model import LocalAirfoilFlow, MPPFlow
from codiffuse_Schwarz_Airfoil.multilevel_coupling import (
    MultilevelTransition,
    build_multilevel_transition,
    integrate_multilevel,
    multilevel_velocity,
    pnp_multilevel_sample,
    restrict,
)
from codiffuse_Schwarz_Airfoil.vorticity_projection import (
    periodic_vorticity,
    project_velocity_curl,
)


Tensor = torch.Tensor
DATA = Path(
    "/orcd/scratch/orcd/003/zituo/AMR_Diffusion_Foundation/"
    "codiffuse_airfoil/cyclic_000_level5_window.h5"
)
PRETRAINED = Path(
    "codiffuse_Schwarz_Airfoil/outputs/local_pretrain/"
    "p0_mpp_transport_h200_v1/checkpoint.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--pretrained", type=Path, default=PRETRAINED)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("uniform", "multilevel"), default="multilevel")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3.0e-5)
    parser.add_argument("--min-lr", type=float, default=3.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--fine-fraction", type=float, default=0.35)
    parser.add_argument("--train-flow-steps", type=int, default=2)
    parser.add_argument("--eval-flow-steps", type=int, default=8)
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--flow-weight", type=float, default=1.0)
    parser.add_argument("--coarse-weight", type=float, default=0.25)
    parser.add_argument("--trace-weight", type=float, default=0.10)
    parser.add_argument("--vorticity-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-consistency-weight", type=float, default=0.0)
    parser.add_argument(
        "--endpoint-eos", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--endpoint-eos-width", type=int, default=64)
    parser.add_argument("--endpoint-eos-depth", type=int, default=6)
    parser.add_argument(
        "--endpoint-eos-direct", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--endpoint-eos-warp", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--endpoint-eos-kinematic-warp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--endpoint-vorticity", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--endpoint-vorticity-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--endpoint-vorticity-dedicated",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--endpoint-eos-iterations", type=int, default=1)
    parser.add_argument("--input-noise", type=float, default=0.02)
    parser.add_argument(
        "--teacher-error-augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--teacher-error-scale", type=float, default=1.0)
    parser.add_argument(
        "--rollout-error-augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--rollout-error-horizon", type=int, default=20)
    parser.add_argument("--rollout-error-stride", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument(
        "--training-scope", choices=("cycle1", "cycle2", "all"), default="cycle1"
    )
    parser.add_argument("--ensemble-size", type=int, default=2)
    parser.add_argument("--placement-ensemble-size", type=int, default=1)
    parser.add_argument("--schwarz-iterations", type=int, default=1)
    parser.add_argument("--schwarz-relaxation", type=float, default=0.7)
    parser.add_argument("--sampler", choices=("ode", "pnp"), default="pnp")
    parser.add_argument("--max-train-transitions", type=int)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--rollout-start-stride", type=int, default=20)
    parser.add_argument(
        "--selection-metric",
        choices=("state", "vorticity"),
        default="state",
    )
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def load_pretrained(
    checkpoint: dict[str, object],
    device: torch.device,
    endpoint_eos: bool = False,
    endpoint_eos_width: int = 64,
    endpoint_eos_depth: int = 6,
    endpoint_eos_direct: bool = False,
    endpoint_eos_warp: bool = False,
    endpoint_eos_kinematic_warp: bool = False,
    endpoint_vorticity: bool = False,
    endpoint_vorticity_residual: bool = False,
    endpoint_vorticity_dedicated: bool = False,
    endpoint_vorticity_phase_only: bool = False,
    endpoint_vorticity_scale: float = 1.0,
    endpoint_learned_projection: bool = False,
    endpoint_projection_attention: bool = False,
    endpoint_projection_spatial_attention: bool = False,
    endpoint_multilevel_projection: bool = False,
    endpoint_spectral_shaping: bool = False,
    endpoint_boundary_memory: bool = False,
    endpoint_boundary_memory_width: int = 32,
    endpoint_boundary_memory_depth: int = 3,
    endpoint_motion_transport_adapter: bool = False,
    endpoint_motion_transport_width: int = 32,
    endpoint_motion_transport_depth: int = 2,
    endpoint_geometry_latent: bool = False,
    endpoint_geometry_latent_width: int = 32,
    endpoint_geometry_latent_depth: int = 2,
    endpoint_prescribed_motion_warp: bool = False,
    endpoint_prescribed_motion_residual: bool = False,
    endpoint_prescribed_motion_band_cells: float = 8.0,
    endpoint_domain_boundary_adapter: bool = False,
    endpoint_domain_boundary_width: int = 32,
    endpoint_domain_boundary_depth: int = 2,
    endpoint_crop_ghost_feature: bool = False,
    endpoint_rollout_closure: bool = False,
    endpoint_rollout_closure_width: int = 32,
    endpoint_curl_memory: bool = False,
    endpoint_curl_memory_width: int = 32,
    endpoint_curl_memory_depth: int = 2,
    endpoint_feature_schwarz: bool = False,
    endpoint_tile_communication: bool = False,
    history_steps: int | None = None,
) -> tuple[torch.nn.Module, dict[str, object]]:
    state = dict(checkpoint["model"])
    args = checkpoint.get("args", {})
    saved_spec = checkpoint.get("model_spec", {})
    if history_steps is None:
        history_steps = int(args.get("history_steps", saved_spec.get("history_steps", 1)))
    if history_steps not in (1, 2):
        raise ValueError("history_steps must be one or two")
    if "embed.0.weight" in state:
        checkpoint_physical_channels = int(state["embed.0.weight"].shape[1]) - 3
        physical_channels = 15 if history_steps == 2 else checkpoint_physical_channels
        if physical_channels < checkpoint_physical_channels:
            raise ValueError("cannot load a two-step checkpoint into a one-step model")
        if physical_channels > checkpoint_physical_channels:
            extra = physical_channels - checkpoint_physical_channels
            for key, prefix in (
                ("embed.0.weight", 3), ("transport_stem.weight", 3),
                ("endpoint_stem.weight", 3), ("vorticity_stem.weight", 0),
                ("projection_stem.weight", 0),
            ):
                if key not in state:
                    continue
                old = state[key]
                expanded = old.new_zeros(
                    old.shape[0], old.shape[1] + extra, *old.shape[2:]
                )
                valid_index = prefix + checkpoint_physical_channels - 1
                expanded[:, :valid_index] = old[:, :valid_index]
                expanded[:, valid_index + extra] = old[:, valid_index]
                expanded[:, valid_index + extra + 1:] = old[:, valid_index + 1:]
                state[key] = expanded
        width = int(state["embed.0.weight"].shape[0])
        depth = len({key.split(".")[1] for key in state if key.startswith("blocks.")})
        heads = int(args.get("heads", saved_spec.get("heads", 8)))
        transport = "transport_stem.weight" in state
        checkpoint_has_eos = "endpoint_stem.weight" in state
        checkpoint_eos_direct = bool(
            args.get("endpoint_eos_direct", saved_spec.get("endpoint_eos_direct", False))
        )
        checkpoint_eos_warp = (
            "endpoint_displacement.weight" in state
            or bool(args.get("endpoint_eos_warp", saved_spec.get("endpoint_eos_warp", False)))
        )
        checkpoint_eos_kinematic_warp = bool(
            args.get(
                "endpoint_eos_kinematic_warp",
                saved_spec.get("endpoint_eos_kinematic_warp", False),
            )
        )
        checkpoint_endpoint_vorticity = (
            "endpoint_vorticity_output.weight" in state
            or "vorticity_stem.weight" in state
        )
        checkpoint_vorticity_residual = bool(
            args.get(
                "endpoint_vorticity_residual",
                saved_spec.get("endpoint_vorticity_residual", False),
            )
        )
        checkpoint_vorticity_dedicated = "vorticity_stem.weight" in state
        checkpoint_vorticity_phase_only = bool(
            args.get(
                "endpoint_vorticity_phase_only",
                saved_spec.get("endpoint_vorticity_phase_only", False),
            )
        )
        checkpoint_vorticity_scale = float(
            args.get(
                "endpoint_vorticity_scale",
                saved_spec.get("endpoint_vorticity_scale", 1.0),
            )
        )
        checkpoint_learned_projection = (
            "projection_stem.weight" in state
            or bool(args.get(
                "endpoint_learned_projection",
                saved_spec.get("endpoint_learned_projection", False),
            ))
        )
        checkpoint_projection_attention = (
            "projection_attention.in_proj_weight" in state
            or bool(args.get(
                "endpoint_projection_attention",
                saved_spec.get("endpoint_projection_attention", False),
            ))
        )
        checkpoint_projection_spatial_attention = (
            "projection_spatial_attention.in_proj_weight" in state
            or bool(args.get(
                "endpoint_projection_spatial_attention",
                saved_spec.get("endpoint_projection_spatial_attention", False),
            ))
        )
        checkpoint_multilevel_projection = bool(
            args.get(
                "endpoint_multilevel_projection",
                saved_spec.get("endpoint_multilevel_projection", False),
            )
        )
        checkpoint_spectral_shaping = (
            any(key.endswith("spectral_shaping_gate") for key in state)
            or bool(args.get(
                "endpoint_spectral_shaping",
                saved_spec.get("endpoint_spectral_shaping", False),
            ))
        )
        checkpoint_boundary_memory = (
            "boundary_memory_stem.weight" in state
            or bool(args.get(
                "endpoint_boundary_memory",
                saved_spec.get("endpoint_boundary_memory", False),
            ))
        )
        checkpoint_domain_boundary = (
            "domain_boundary_stem.weight" in state
            or bool(args.get(
                "endpoint_domain_boundary_adapter",
                saved_spec.get("endpoint_domain_boundary_adapter", False),
            ))
        )
        checkpoint_crop_ghost = bool(
            args.get(
                "endpoint_crop_ghost_feature",
                saved_spec.get("endpoint_crop_ghost_feature", False),
            )
        )
        checkpoint_motion_transport = (
            "motion_transport_stem.weight" in state
            or bool(args.get(
                "endpoint_motion_transport_adapter",
                saved_spec.get("endpoint_motion_transport_adapter", False),
            ))
        )
        checkpoint_geometry_latent = (
            "geometry_latent_stem.weight" in state
            or bool(args.get(
                "endpoint_geometry_latent",
                saved_spec.get("endpoint_geometry_latent", False),
            ))
        )
        checkpoint_prescribed_motion_warp = bool(
            args.get(
                "endpoint_prescribed_motion_warp",
                saved_spec.get("endpoint_prescribed_motion_warp", False),
            )
        )
        checkpoint_prescribed_motion_residual = bool(
            args.get(
                "endpoint_prescribed_motion_residual",
                saved_spec.get("endpoint_prescribed_motion_residual", False),
            )
        )
        if checkpoint_prescribed_motion_warp:
            endpoint_prescribed_motion_band_cells = float(
                args.get(
                    "endpoint_prescribed_motion_band_cells",
                    saved_spec.get(
                        "endpoint_prescribed_motion_band_cells", 8.0,
                    ),
                )
            )
        checkpoint_rollout_closure = (
            "rollout_closure_stem.weight" in state
            or bool(args.get(
                "endpoint_rollout_closure",
                saved_spec.get("endpoint_rollout_closure", False),
            ))
        )
        checkpoint_curl_memory = (
            "curl_memory_stem.weight" in state
            or bool(args.get(
                "endpoint_curl_memory",
                saved_spec.get("endpoint_curl_memory", False),
            ))
        )
        checkpoint_feature_schwarz = (
            "feature_schwarz_gate" in state
            or bool(args.get(
                "endpoint_feature_schwarz",
                saved_spec.get("endpoint_feature_schwarz", False),
            ))
        )
        checkpoint_tile_communication = (
            any(key.startswith("endpoint_tile_communicator.") for key in state)
            or bool(args.get(
                "endpoint_tile_communication",
                saved_spec.get("endpoint_tile_communication", False),
            ))
        )
        if checkpoint_rollout_closure:
            endpoint_rollout_closure_width = int(
                state["rollout_closure_stem.weight"].shape[0]
            )
        if checkpoint_curl_memory:
            endpoint_curl_memory_width = int(
                state["curl_memory_stem.weight"].shape[0]
            )
            endpoint_curl_memory_depth = len({
                key.split(".")[1]
                for key in state if key.startswith("curl_memory_blocks.")
            })
        if checkpoint_boundary_memory:
            endpoint_boundary_memory_width = int(
                state["boundary_memory_stem.weight"].shape[0]
            )
            endpoint_boundary_memory_depth = len({
                key.split(".")[1]
                for key in state if key.startswith("boundary_memory_blocks.")
            })
        if checkpoint_domain_boundary:
            endpoint_domain_boundary_width = int(
                state["domain_boundary_stem.weight"].shape[0]
            )
            endpoint_domain_boundary_depth = len({
                key.split(".")[1]
                for key in state if key.startswith("domain_boundary_blocks.")
            })
        if checkpoint_motion_transport:
            endpoint_motion_transport_width = int(
                state["motion_transport_stem.weight"].shape[0]
            )
            endpoint_motion_transport_depth = len({
                key.split(".")[1]
                for key in state if key.startswith("motion_transport_blocks.")
            })
        if checkpoint_geometry_latent:
            endpoint_geometry_latent_width = int(
                state["geometry_latent_stem.weight"].shape[0]
            )
            endpoint_geometry_latent_depth = len({
                key.split(".")[1]
                for key in state if key.startswith("geometry_latent_blocks.")
            })
        if checkpoint_has_eos:
            endpoint_eos_width = int(state["endpoint_stem.weight"].shape[0])
            endpoint_eos_depth = len({
                key.split(".")[1]
                for key in state if key.startswith("endpoint_blocks.")
            })
        model = MPPFlow(
            physical_channels=physical_channels,
            width=width, depth=depth, heads=heads, transport_skip=transport,
            endpoint_eos=checkpoint_has_eos,
            endpoint_eos_width=endpoint_eos_width,
            endpoint_eos_depth=endpoint_eos_depth,
            endpoint_eos_direct=checkpoint_eos_direct or endpoint_eos_direct,
            endpoint_eos_warp=checkpoint_eos_warp or endpoint_eos_warp,
            endpoint_eos_kinematic_warp=(
                checkpoint_eos_kinematic_warp or endpoint_eos_kinematic_warp
            ),
            endpoint_vorticity=(
                checkpoint_endpoint_vorticity or endpoint_vorticity
            ),
            endpoint_vorticity_residual=(
                checkpoint_vorticity_residual or endpoint_vorticity_residual
            ),
            endpoint_vorticity_dedicated=(
                checkpoint_vorticity_dedicated or endpoint_vorticity_dedicated
            ),
            endpoint_vorticity_phase_only=(
                checkpoint_vorticity_phase_only or endpoint_vorticity_phase_only
            ),
            endpoint_vorticity_scale=(
                checkpoint_vorticity_scale
                if checkpoint_endpoint_vorticity else endpoint_vorticity_scale
            ),
            endpoint_learned_projection=(
                checkpoint_learned_projection or endpoint_learned_projection
            ),
            endpoint_projection_attention=(
                checkpoint_projection_attention or endpoint_projection_attention
            ),
            endpoint_projection_spatial_attention=(
                checkpoint_projection_spatial_attention
                or endpoint_projection_spatial_attention
            ),
            endpoint_multilevel_projection=(
                checkpoint_multilevel_projection or endpoint_multilevel_projection
            ),
            endpoint_spectral_shaping=(
                checkpoint_spectral_shaping or endpoint_spectral_shaping
            ),
            endpoint_boundary_memory=(
                checkpoint_boundary_memory or endpoint_boundary_memory
            ),
            endpoint_boundary_memory_width=endpoint_boundary_memory_width,
            endpoint_boundary_memory_depth=endpoint_boundary_memory_depth,
            endpoint_motion_transport_adapter=(
                checkpoint_motion_transport or endpoint_motion_transport_adapter
            ),
            endpoint_motion_transport_width=endpoint_motion_transport_width,
            endpoint_motion_transport_depth=endpoint_motion_transport_depth,
            endpoint_geometry_latent=(
                checkpoint_geometry_latent or endpoint_geometry_latent
            ),
            endpoint_geometry_latent_width=endpoint_geometry_latent_width,
            endpoint_geometry_latent_depth=endpoint_geometry_latent_depth,
            endpoint_prescribed_motion_warp=(
                checkpoint_prescribed_motion_warp
                or endpoint_prescribed_motion_warp
            ),
            endpoint_prescribed_motion_residual=(
                checkpoint_prescribed_motion_residual
                or endpoint_prescribed_motion_residual
            ),
            endpoint_prescribed_motion_band_cells=(
                endpoint_prescribed_motion_band_cells
            ),
            endpoint_domain_boundary_adapter=(
                checkpoint_domain_boundary or endpoint_domain_boundary_adapter
            ),
            endpoint_domain_boundary_width=endpoint_domain_boundary_width,
            endpoint_domain_boundary_depth=endpoint_domain_boundary_depth,
            endpoint_crop_ghost_feature=(
                checkpoint_crop_ghost or endpoint_crop_ghost_feature
            ),
            endpoint_rollout_closure=(
                checkpoint_rollout_closure or endpoint_rollout_closure
            ),
            endpoint_rollout_closure_width=endpoint_rollout_closure_width,
            endpoint_curl_memory=(
                checkpoint_curl_memory or endpoint_curl_memory
            ),
            endpoint_curl_memory_width=endpoint_curl_memory_width,
            endpoint_curl_memory_depth=endpoint_curl_memory_depth,
            endpoint_feature_schwarz=(
                checkpoint_feature_schwarz or endpoint_feature_schwarz
            ),
            endpoint_tile_communication=(
                checkpoint_tile_communication or endpoint_tile_communication
            ),
        )
        incompatible = model.load_state_dict(state, strict=False)
        allowed_missing = {
            "endpoint_displacement.weight", "endpoint_displacement.bias",
            "endpoint_warp_embed.weight", "endpoint_warp_embed.bias",
        } if endpoint_eos_warp and not checkpoint_eos_warp else set()
        if endpoint_vorticity and not checkpoint_endpoint_vorticity:
            if endpoint_vorticity_dedicated:
                allowed_missing |= {
                    key for key in model.state_dict()
                    if key.startswith((
                        "vorticity_stem", "vorticity_condition",
                        "vorticity_blocks", "vorticity_output",
                    ))
                }
            else:
                allowed_missing |= {
                    "endpoint_vorticity_output.weight",
                    "endpoint_vorticity_output.bias",
                }
        if endpoint_learned_projection and not checkpoint_learned_projection:
            allowed_missing |= {
                key for key in model.state_dict() if key.startswith("projection_")
            }
        if endpoint_projection_attention and not checkpoint_projection_attention:
            allowed_missing |= {
                key for key in model.state_dict()
                if key.startswith("projection_attention")
            }
        if (
            endpoint_projection_spatial_attention
            and not checkpoint_projection_spatial_attention
        ):
            allowed_missing |= {
                key for key in model.state_dict()
                if key.startswith("projection_spatial")
            }
        if endpoint_spectral_shaping and not checkpoint_spectral_shaping:
            allowed_missing |= {
                key for key in model.state_dict()
                if key.endswith("spectral_shaping_gate")
            }
        if endpoint_boundary_memory and not checkpoint_boundary_memory:
            allowed_missing |= {
                key for key in model.state_dict()
                if key.startswith("boundary_memory_")
            }
        if endpoint_domain_boundary_adapter and not checkpoint_domain_boundary:
            allowed_missing |= {
                key for key in model.state_dict()
                if key.startswith("domain_boundary_")
            }
        if (
            endpoint_crop_ghost_feature
            and "domain_boundary_ghost_embed.weight" not in state
            and model.domain_boundary_ghost_embed is not None
        ):
            allowed_missing.add("domain_boundary_ghost_embed.weight")
        if endpoint_motion_transport_adapter and not checkpoint_motion_transport:
            allowed_missing |= {
                key for key in model.state_dict()
                if key.startswith("motion_transport_")
            }
        if endpoint_geometry_latent and not checkpoint_geometry_latent:
            allowed_missing |= {
                key for key in model.state_dict()
                if key.startswith("geometry_latent_")
            }
        if (
            endpoint_prescribed_motion_warp
            and "motion_transport_warp_embed.weight" not in state
            and model.motion_transport_warp_embed is not None
        ):
            allowed_missing.add("motion_transport_warp_embed.weight")
        if (
            endpoint_prescribed_motion_residual
            and not checkpoint_prescribed_motion_residual
        ):
            allowed_missing.add("prescribed_motion_residual_gate")
        if endpoint_rollout_closure and not checkpoint_rollout_closure:
            allowed_missing |= {
                key for key in model.state_dict()
                if key.startswith("rollout_closure_")
            }
        if endpoint_curl_memory and not checkpoint_curl_memory:
            allowed_missing |= {
                key for key in model.state_dict()
                if key.startswith("curl_memory_")
            }
        if endpoint_feature_schwarz and not checkpoint_feature_schwarz:
            allowed_missing.add("feature_schwarz_gate")
        if endpoint_tile_communication and not checkpoint_tile_communication:
            allowed_missing |= {
                key for key in model.state_dict()
                if key.startswith("endpoint_tile_communicator.")
            }
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if incompatible.unexpected_keys or unexpected_missing:
            raise RuntimeError(
                "incompatible MPP checkpoint: "
                f"missing={sorted(unexpected_missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
        if endpoint_eos and not checkpoint_has_eos:
            model.enable_endpoint_eos(endpoint_eos_width, endpoint_eos_depth)
        if checkpoint_has_eos or endpoint_eos:
            checkpoint_statistics = FieldStatistics.from_json(checkpoint["statistics"])
            model.set_endpoint_statistics(
                checkpoint_statistics.state_mean,
                checkpoint_statistics.state_std,
                checkpoint_statistics.delta_std,
                checkpoint_statistics.delta_mean,
            )
        spec = {
            "architecture": "mpp",
            "width": width,
            "depth": depth,
            "heads": heads,
            "transport_skip": transport,
            "endpoint_eos": checkpoint_has_eos or endpoint_eos,
            "endpoint_eos_width": endpoint_eos_width,
            "endpoint_eos_depth": endpoint_eos_depth,
            "endpoint_eos_direct": checkpoint_eos_direct or endpoint_eos_direct,
            "endpoint_eos_warp": checkpoint_eos_warp or endpoint_eos_warp,
            "endpoint_eos_kinematic_warp": (
                checkpoint_eos_kinematic_warp or endpoint_eos_kinematic_warp
            ),
            "endpoint_vorticity": (
                checkpoint_endpoint_vorticity or endpoint_vorticity
            ),
            "endpoint_vorticity_residual": (
                checkpoint_vorticity_residual or endpoint_vorticity_residual
            ),
            "endpoint_vorticity_dedicated": (
                checkpoint_vorticity_dedicated or endpoint_vorticity_dedicated
            ),
            "endpoint_vorticity_phase_only": (
                checkpoint_vorticity_phase_only or endpoint_vorticity_phase_only
            ),
            "endpoint_vorticity_scale": (
                checkpoint_vorticity_scale
                if checkpoint_endpoint_vorticity else endpoint_vorticity_scale
            ),
            "endpoint_learned_projection": (
                checkpoint_learned_projection or endpoint_learned_projection
            ),
            "endpoint_projection_attention": (
                checkpoint_projection_attention or endpoint_projection_attention
            ),
            "endpoint_projection_spatial_attention": (
                checkpoint_projection_spatial_attention
                or endpoint_projection_spatial_attention
            ),
            "endpoint_multilevel_projection": (
                checkpoint_multilevel_projection or endpoint_multilevel_projection
            ),
            "endpoint_spectral_shaping": (
                checkpoint_spectral_shaping or endpoint_spectral_shaping
            ),
            "endpoint_boundary_memory": (
                checkpoint_boundary_memory or endpoint_boundary_memory
            ),
            "endpoint_boundary_memory_width": endpoint_boundary_memory_width,
            "endpoint_boundary_memory_depth": endpoint_boundary_memory_depth,
            "endpoint_motion_transport_adapter": (
                checkpoint_motion_transport or endpoint_motion_transport_adapter
            ),
            "endpoint_motion_transport_width": endpoint_motion_transport_width,
            "endpoint_motion_transport_depth": endpoint_motion_transport_depth,
            "endpoint_geometry_latent": (
                checkpoint_geometry_latent or endpoint_geometry_latent
            ),
            "endpoint_geometry_latent_width": endpoint_geometry_latent_width,
            "endpoint_geometry_latent_depth": endpoint_geometry_latent_depth,
            "endpoint_prescribed_motion_warp": (
                checkpoint_prescribed_motion_warp
                or endpoint_prescribed_motion_warp
            ),
            "endpoint_prescribed_motion_residual": (
                checkpoint_prescribed_motion_residual
                or endpoint_prescribed_motion_residual
            ),
            "endpoint_prescribed_motion_band_cells": (
                endpoint_prescribed_motion_band_cells
            ),
            "endpoint_domain_boundary_adapter": (
                checkpoint_domain_boundary or endpoint_domain_boundary_adapter
            ),
            "endpoint_domain_boundary_width": endpoint_domain_boundary_width,
            "endpoint_domain_boundary_depth": endpoint_domain_boundary_depth,
            "endpoint_crop_ghost_feature": (
                checkpoint_crop_ghost or endpoint_crop_ghost_feature
            ),
            "endpoint_rollout_closure": (
                checkpoint_rollout_closure or endpoint_rollout_closure
            ),
            "endpoint_rollout_closure_width": endpoint_rollout_closure_width,
            "endpoint_curl_memory": (
                checkpoint_curl_memory or endpoint_curl_memory
            ),
            "endpoint_curl_memory_width": endpoint_curl_memory_width,
            "endpoint_curl_memory_depth": endpoint_curl_memory_depth,
            "endpoint_feature_schwarz": (
                checkpoint_feature_schwarz or endpoint_feature_schwarz
            ),
            "endpoint_tile_communication": (
                checkpoint_tile_communication or endpoint_tile_communication
            ),
            "history_steps": history_steps,
            "physical_channels": physical_channels,
        }
    else:
        width = int(state["stem.weight"].shape[0])
        depth = len({key.split(".")[1] for key in state if key.startswith("blocks.")})
        if endpoint_eos:
            raise ValueError("endpoint EoS currently requires the MPP architecture")
        model = LocalAirfoilFlow(width=width, depth=depth)
        model.load_state_dict(state)
        spec = {"architecture": "residual", "width": width, "depth": depth}
    return model.to(device), spec


def random_uniform_transition(
    batch: dict[str, Tensor],
    statistics: FieldStatistics,
    dataset: CyclicWindowDataset,
    device: torch.device,
    seed: int,
    halo: int,
    placement_mode: PlacementMode = "random",
) -> ControlledTransition:
    batch = {key: value.to(device) for key, value in batch.items()}
    q_n = batch["q_n"]
    oy, ox, _ = choose_offset(
        placement_mode, q_n, batch["sdf_n"], batch["sdf_np1"], dataset.core_size,
        halo, dataset.dx, seed,
    )
    cover = CartesianCover(
        q_n.shape[-2], q_n.shape[-1], dataset.core_size, halo, oy, ox
    )
    return ControlledTransition(batch, statistics, dataset, cover, device)


def perturb_input(
    batch: dict[str, Tensor],
    statistics: FieldStatistics,
    magnitude: float,
    device: torch.device,
) -> dict[str, Tensor]:
    result = {key: value.to(device) for key, value in batch.items()}
    if magnitude <= 0.0:
        return result
    q_n = result["q_n"]
    low_shape = (max(2, q_n.shape[-2] // 16), max(2, q_n.shape[-1] // 16))
    perturbation = F.interpolate(
        torch.randn(q_n.shape[0], 3, *low_shape, device=device),
        size=q_n.shape[-2:], mode="bilinear", align_corners=False,
    )
    rms = perturbation.square().mean(dim=(-1, -2), keepdim=True).sqrt().clamp_min(1e-6)
    scale = torch.tensor(
        statistics.state_std, device=device, dtype=q_n.dtype
    )[None, :, None, None]
    result["q_n"] = q_n + magnitude * scale * perturbation / rms
    return result


def transition_target(
    transition: ControlledTransition | MultilevelTransition,
) -> Tensor:
    return transition.target if isinstance(transition, ControlledTransition) else transition.fine.target


def transition_shape(
    transition: ControlledTransition | MultilevelTransition,
) -> tuple[int, int, int, int]:
    q_n = transition.q_n if isinstance(transition, ControlledTransition) else transition.fine.q_n
    return tuple(q_n.shape)  # type: ignore[return-value]


def coupled_step_velocity(
    model: torch.nn.Module,
    state: Tensor,
    transition: ControlledTransition | MultilevelTransition,
    flow_time: Tensor,
    schwarz_iterations: int = 1,
    schwarz_relaxation: float = 0.7,
) -> tuple[Tensor, dict[str, Tensor]]:
    if isinstance(transition, MultilevelTransition):
        return multilevel_velocity(
            model, state, transition, flow_time,
            schwarz_iterations, schwarz_relaxation,
        )
    if schwarz_iterations > 1:
        velocity, tiles, physical, residuals = endpoint_trace_velocity(
            model, state, transition, flow_time,
            schwarz_iterations, schwarz_relaxation,
        )
        return velocity, {
            "fine_velocity": velocity, "fine_tiles": tiles, "physical": physical,
            "fine_trace_residuals": torch.stack(residuals),
        }
    velocity, tiles, physical = controlled_velocity(
        model, state, transition, flow_time, "ras"
    )
    return velocity, {"fine_velocity": velocity, "fine_tiles": tiles, "physical": physical}


def integrate(
    model: torch.nn.Module,
    transition: ControlledTransition | MultilevelTransition,
    steps: int,
    noise: Tensor,
    schwarz_iterations: int = 1,
    schwarz_relaxation: float = 0.7,
) -> Tensor:
    if isinstance(transition, MultilevelTransition):
        return integrate_multilevel(
            model, transition, steps, noise,
            schwarz_iterations, schwarz_relaxation,
        )
    state = noise
    for step in range(steps):
        flow_time = torch.full(
            (state.shape[0],), (step + 0.5) / steps,
            device=state.device, dtype=state.dtype,
        )
        velocity, _ = coupled_step_velocity(
            model, state, transition, flow_time,
            schwarz_iterations, schwarz_relaxation,
        )
        state = state + velocity / steps
    return state


def generate_endpoint(
    model: torch.nn.Module,
    transition: ControlledTransition | MultilevelTransition,
    noise_sequence: Tensor,
    sampler: str,
    schwarz_iterations: int,
    schwarz_relaxation: float,
    endpoint_eos_iterations: int = 0,
) -> Tensor:
    return generate_endpoint_fields(
        model, transition, noise_sequence, sampler, schwarz_iterations,
        schwarz_relaxation, endpoint_eos_iterations,
    )[0]


def generate_endpoint_fields(
    model: torch.nn.Module,
    transition: ControlledTransition | MultilevelTransition,
    noise_sequence: Tensor,
    sampler: str,
    schwarz_iterations: int,
    schwarz_relaxation: float,
    endpoint_eos_iterations: int = 0,
) -> tuple[Tensor, Tensor | None]:
    if sampler == "pnp":
        if isinstance(transition, MultilevelTransition):
            endpoint, _ = pnp_multilevel_sample(
                model, transition, noise_sequence, schwarz_relaxation
            )
        else:
            endpoint, _ = pnp_endpoint_trace_sample(
                model, transition, noise_sequence, schwarz_relaxation
            )
    else:
        endpoint = integrate(
            model, transition, noise_sequence.shape[0], noise_sequence[0],
            schwarz_iterations, schwarz_relaxation,
        )
    return apply_endpoint_eos_fields(
        model, endpoint, transition, endpoint_eos_iterations
    )


def apply_endpoint_eos(
    model: torch.nn.Module,
    endpoint: Tensor,
    transition: ControlledTransition | MultilevelTransition,
    iterations: int,
) -> Tensor:
    return apply_endpoint_eos_fields(
        model, endpoint, transition, iterations
    )[0]


def apply_endpoint_eos_fields(
    model: torch.nn.Module,
    endpoint: Tensor,
    transition: ControlledTransition | MultilevelTransition,
    iterations: int,
) -> tuple[Tensor, Tensor | None]:
    if iterations <= 0 or not hasattr(model, "correct_endpoint"):
        return endpoint, None
    multilevel = transition if isinstance(transition, MultilevelTransition) else None
    fine = transition if isinstance(transition, ControlledTransition) else transition.fine
    flow_time = torch.ones(
        fine.batch_size, device=endpoint.device, dtype=endpoint.dtype,
    )
    physical, scalar = fine.conditions(flow_time)
    motion_physical, motion_scalar = (
        fine.endpoint_conditions()
        if hasattr(fine, "endpoint_conditions")
        else (physical, scalar)
    )
    corrected = endpoint
    global_vorticity = None
    for _ in range(iterations):
        local = fine.cover.tile(corrected)
        if hasattr(model, "correct_endpoint_fields"):
            local, local_vorticity = model.correct_endpoint_fields(  # type: ignore[attr-defined]
                local, physical, scalar, fine.cover.tile_count,
                tile_exchange=lambda hidden: fine.cover.tile(
                    fine.cover.assemble_weighted(hidden, fine.batch_size)
                ),
                tile_grid_shape=fine.cover.grid_shape,
                motion_physical=motion_physical,
                motion_scalar=motion_scalar,
            )
        else:
            local = model.correct_endpoint(  # type: ignore[attr-defined]
                local, physical, scalar
            )
            local_vorticity = None
        corrected = fine.cover.assemble_weighted(local, fine.batch_size)
        if local_vorticity is not None:
            global_vorticity = fine.cover.assemble_weighted(
                local_vorticity[:, None], fine.batch_size
            )[:, 0]
    if global_vorticity is not None:
        if multilevel is not None:
            coarse = multilevel.coarse
            coarse_endpoint = restrict(corrected, multilevel.factor)
            coarse_time = torch.ones(
                coarse.batch_size, device=endpoint.device, dtype=endpoint.dtype,
            )
            coarse_physical, coarse_scalar = coarse.conditions(coarse_time)
            coarse_motion_physical, coarse_motion_scalar = (
                coarse.endpoint_conditions()
                if hasattr(coarse, "endpoint_conditions")
                else (coarse_physical, coarse_scalar)
            )
            coarse_local = coarse.cover.tile(coarse_endpoint)
            coarse_corrected_tiles, coarse_vorticity_tiles = model.correct_endpoint_fields(  # type: ignore[attr-defined]
                coarse_local, coarse_physical, coarse_scalar,
                coarse.cover.tile_count,
                tile_exchange=lambda hidden: coarse.cover.tile(
                    coarse.cover.assemble_weighted(hidden, coarse.batch_size)
                ),
                tile_grid_shape=coarse.cover.grid_shape,
                motion_physical=coarse_motion_physical,
                motion_scalar=coarse_motion_scalar,
            )
            if bool(getattr(model, "endpoint_multilevel_projection", False)):
                coarse_corrected = coarse.cover.assemble_weighted(
                    coarse_corrected_tiles, coarse.batch_size
                )
                coarse_corrected = F.interpolate(
                    coarse_corrected, size=corrected.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
                endpoint_weight = multilevel.refinement_weight
                corrected = coarse_corrected + endpoint_weight * (
                    corrected - coarse_corrected
                )
            if coarse_vorticity_tiles is not None:
                coarse_vorticity = coarse.cover.assemble_weighted(
                    coarse_vorticity_tiles[:, None], coarse.batch_size
                )
                coarse_vorticity = F.interpolate(
                    coarse_vorticity,
                    size=global_vorticity.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
                weight = multilevel.refinement_weight[:, 0]
                global_vorticity = (
                    coarse_vorticity + weight * (global_vorticity - coarse_vorticity)
                )
        if bool(getattr(model, "endpoint_vorticity_residual", False)):
            global_vorticity = global_vorticity + periodic_vorticity(
                fine.q_n, fine.dataset.dx
            )
    return corrected, global_vorticity


def make_transition(
    mode: str,
    batch: dict[str, Tensor],
    statistics: FieldStatistics,
    dataset: CyclicWindowDataset,
    device: torch.device,
    seed: int,
    fine_fraction: float,
    halo: int,
    placement_mode: PlacementMode = "random",
) -> ControlledTransition | MultilevelTransition:
    if mode == "multilevel":
        return build_multilevel_transition(
            batch, statistics, dataset, device, seed, fine_fraction, halo=halo,
            placement_mode=placement_mode,
        )
    return random_uniform_transition(
        batch, statistics, dataset, device, seed, halo, placement_mode,
    )


def one_step_loss(
    model: torch.nn.Module,
    transition: ControlledTransition | MultilevelTransition,
    train_steps: int,
    endpoint_weight: float,
    flow_weight: float,
    coarse_weight: float,
    trace_weight: float,
    vorticity_weight: float,
    endpoint_consistency_weight: float,
    endpoint_eos_iterations: int,
    schwarz_iterations: int,
    schwarz_relaxation: float,
    sampler: str,
) -> tuple[Tensor, dict[str, float]]:
    target = transition_target(transition)
    noise = torch.randn_like(target)
    flow_time = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
    mix = flow_time[:, None, None, None]
    state = (1.0 - mix) * noise + mix * target
    target_velocity = target - noise
    prediction, details = coupled_step_velocity(
        model, state, transition, flow_time,
        schwarz_iterations, schwarz_relaxation,
    )
    flow_loss = (prediction - target_velocity).square().mean()

    coarse_loss = torch.zeros((), device=target.device)
    if isinstance(transition, MultilevelTransition):
        coarse_target = restrict(target_velocity, transition.factor)
        coarse_loss = (details["coarse_velocity"] - coarse_target).square().mean()

    fine = transition if isinstance(transition, ControlledTransition) else transition.fine
    tile_target = fine.cover.tile(target_velocity)
    valid = fine.cover.valid_tiles(target.shape[0], target.device, target.dtype)
    trace_loss = (
        (details["fine_tiles"] - tile_target).square() * valid
    ).sum() / (valid.sum().clamp_min(1.0) * target.shape[1])

    endpoint_noise = torch.randn(
        train_steps, *target.shape, device=target.device, dtype=target.dtype
    )
    endpoint_positive, vorticity_positive = generate_endpoint_fields(
        model, transition, endpoint_noise, sampler,
        schwarz_iterations, schwarz_relaxation,
        endpoint_eos_iterations,
    )
    needs_antithetic = (
        endpoint_consistency_weight > 0.0 or vorticity_weight > 0.0
    ) and not bool(getattr(model, "endpoint_eos_direct", False))
    if needs_antithetic:
        endpoint_negative, vorticity_negative = generate_endpoint_fields(
            model, transition, -endpoint_noise, sampler,
            schwarz_iterations, schwarz_relaxation,
            endpoint_eos_iterations,
        )
        endpoint = 0.5 * (endpoint_positive + endpoint_negative)
        endpoint_vorticity = (
            None if vorticity_positive is None else
            0.5 * (vorticity_positive + vorticity_negative)
        )
    else:
        endpoint_negative = endpoint_positive
        endpoint = endpoint_positive
        vorticity_negative = vorticity_positive
        endpoint_vorticity = vorticity_positive
    endpoint_loss = (endpoint - target).square().mean()

    fine = transition if isinstance(transition, ControlledTransition) else transition.fine
    predicted_next = fine.q_n + denormalize_delta(endpoint.float(), fine.statistics)
    target_next = fine.q_np1
    target_periodic_vorticity = periodic_vorticity(target_next, fine.dataset.dx)
    if endpoint_vorticity is not None:
        predicted_next = project_velocity_curl(
            predicted_next, endpoint_vorticity, fine.dataset.dx
        )
    predicted_vorticity = interior_vorticity(predicted_next, fine.dataset.dx)
    target_vorticity = interior_vorticity(target_next, fine.dataset.dx)
    vorticity_scale = target_vorticity.square().mean().detach().clamp_min(1.0e-8)
    vorticity_loss = (
        (predicted_vorticity - target_vorticity).square().mean() / vorticity_scale
    )
    positive_next = fine.q_n + denormalize_delta(
        endpoint_positive.float(), fine.statistics
    )
    negative_next = fine.q_n + denormalize_delta(
        endpoint_negative.float(), fine.statistics
    )
    consistency_loss = (
        interior_vorticity(positive_next, fine.dataset.dx)
        - interior_vorticity(negative_next, fine.dataset.dx)
    ).square().mean() / vorticity_scale
    vorticity_head_loss = torch.zeros((), device=target.device)
    if endpoint_vorticity is not None:
        vorticity_head_loss = (
            (endpoint_vorticity - target_periodic_vorticity).square().sum()
            / target_periodic_vorticity.square().sum().clamp_min(1.0e-20)
        )
    loss = (
        flow_weight * flow_loss + endpoint_weight * endpoint_loss
        + coarse_weight * coarse_loss + trace_weight * trace_loss
        + vorticity_weight * vorticity_loss
        + endpoint_consistency_weight * consistency_loss
    )
    return loss, {
        "flow": float(flow_loss.detach()),
        "endpoint": float(endpoint_loss.detach()),
        "coarse": float(coarse_loss.detach()),
        "trace": float(trace_loss.detach()),
        "vorticity": float(vorticity_loss.detach()),
        "consistency": float(consistency_loss.detach()),
        "vorticity_head": float(vorticity_head_loss.detach()),
    }


def global_l2(prediction: Tensor, target: Tensor) -> float:
    return float(torch.sqrt(
        (prediction - target).square().sum()
        / target.square().sum().clamp_min(1.0e-20)
    ))


def interior_vorticity(field: Tensor, dx: float) -> Tensor:
    """Second-order curl on interior cells, suitable for training and metrics."""
    dv_dx = (field[:, 1, 1:-1, 2:] - field[:, 1, 1:-1, :-2]) / (2.0 * dx)
    du_dy = (field[:, 0, 2:, 1:-1] - field[:, 0, :-2, 1:-1]) / (2.0 * dx)
    return dv_dx - du_dy


class TeacherErrorDataset(Dataset[dict[str, Tensor]]):
    """Exact pairs plus detached preceding one-step error perturbations."""

    def __init__(
        self,
        base: CyclicWindowDataset,
        indices: list[int],
        input_errors: dict[int, Tensor],
        scale: float,
    ) -> None:
        self.base = base
        self.records = [
            (item, corrected)
            for item in indices
            for corrected in (False, True)
            if not corrected or item in input_errors
        ]
        self.input_errors = input_errors
        self.scale = scale

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item, corrected = self.records[index]
        sample = self.base[item]
        if corrected:
            sample["q_n"] = sample["q_n"] + self.scale * self.input_errors[item]
        return sample


@torch.no_grad()
def predict_next(
    model: torch.nn.Module,
    batch: dict[str, Tensor],
    dataset: CyclicWindowDataset,
    statistics: FieldStatistics,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> Tensor:
    generator = torch.Generator(device=device).manual_seed(seed + 7_000_001)
    predictions = []
    vorticity_predictions = []
    for placement in range(args.placement_ensemble_size):
        transition = make_transition(
            args.mode, batch, statistics, dataset, device,
            seed + placement * 10_000_019, args.fine_fraction, args.halo,
        )
        target = transition_target(transition)
        base_noise = None
        for member in range(args.ensemble_size):
            if member % 2 == 0:
                base_noise = torch.randn(
                    (args.eval_flow_steps, *target.shape), generator=generator,
                    device=device, dtype=target.dtype
                )
                noise = base_noise
            else:
                assert base_noise is not None
                noise = -base_noise
            endpoint, endpoint_vorticity = generate_endpoint_fields(
                model, transition, noise, args.sampler,
                args.schwarz_iterations, args.schwarz_relaxation,
                args.endpoint_eos_iterations,
            )
            predictions.append(endpoint)
            if endpoint_vorticity is not None:
                vorticity_predictions.append(endpoint_vorticity)
    normalized_delta = torch.stack(predictions).mean(dim=0)
    q_n = batch["q_n"].to(device)
    prediction = q_n + denormalize_delta(normalized_delta.float(), statistics)
    if vorticity_predictions and not bool(
        getattr(model, "endpoint_learned_projection", False)
    ):
        target_vorticity = torch.stack(vorticity_predictions).mean(dim=0)
        projected = project_velocity_curl(prediction, target_vorticity, dataset.dx)
        relaxation = float(getattr(args, "vorticity_projection_relaxation", 1.0))
        prediction = prediction + relaxation * (projected - prediction)
    return prediction


@torch.no_grad()
def collect_teacher_input_errors(
    model: torch.nn.Module,
    dataset: CyclicWindowDataset,
    indices: list[int],
    statistics: FieldStatistics,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> dict[int, Tensor]:
    """Map transition i to the detached error produced for its input frame."""
    model.eval()
    selected = set(indices)
    errors: dict[int, Tensor] = {}
    for item in indices:
        next_item = item + 1
        if next_item not in selected or next_item >= len(dataset):
            continue
        sample = dataset[item]
        batch = {key: value[None] for key, value in sample.items()}
        prediction = predict_next(
            model, batch, dataset, statistics, args, seed + item, device
        )
        errors[next_item] = (
            prediction[0].cpu() - sample["q_np1"]
        ).detach()
    return errors


@torch.no_grad()
def collect_rollout_input_errors(
    model: torch.nn.Module,
    dataset: CyclicWindowDataset,
    indices: list[int],
    statistics: FieldStatistics,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> dict[int, Tensor]:
    """Collect detached deployment-state errors from overlapping free rollouts."""
    model.eval()
    selected = set(indices)
    horizon = args.rollout_error_horizon
    stride = args.rollout_error_stride
    starts = list(range(0, max(len(dataset) - horizon + 1, 1), stride))
    final_start = max(0, len(dataset) - horizon)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    accumulated: dict[int, list[Tensor]] = {}
    for start in starts:
        current = dataset[start]["q_n"][None].to(device)
        for step in range(min(horizon, len(dataset) - start)):
            item = start + step
            sample = dataset[item]
            batch = {key: value[None] for key, value in sample.items()}
            batch["q_n"] = current
            current = predict_next(
                model, batch, dataset, statistics, args,
                seed + start * 1009 + step, device,
            )
            next_item = item + 1
            if next_item in selected:
                accumulated.setdefault(next_item, []).append(
                    (current[0].cpu() - sample["q_np1"]).detach()
                )
    return {
        item: torch.stack(values).mean(dim=0)
        for item, values in accumulated.items()
    }


@torch.no_grad()
def evaluate_one_step(
    model: torch.nn.Module,
    dataset: CyclicWindowDataset,
    indices: list[int],
    statistics: FieldStatistics,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> float:
    model.eval()
    numerator = torch.zeros((), device=device)
    denominator = torch.zeros((), device=device)
    for local, item in enumerate(indices):
        sample = dataset[item]
        batch = {key: value[None] for key, value in sample.items()}
        prediction = predict_next(
            model, batch, dataset, statistics, args, seed + local, device
        )
        target = batch["q_np1"].to(device)
        numerator += (prediction - target).square().sum()
        denominator += target.square().sum()
    return float(torch.sqrt(numerator / denominator.clamp_min(1e-20)))


@torch.no_grad()
def evaluate_rollout(
    model: torch.nn.Module,
    dataset: CyclicWindowDataset,
    start: int,
    horizon: int,
    statistics: FieldStatistics,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    model.eval()
    initial = dataset[start]
    previous = initial["q_nm1"][None].to(device)
    current = initial["q_n"][None].to(device)
    rows = []
    for step in range(horizon):
        sample = dataset[start + step]
        batch = {key: value[None] for key, value in sample.items()}
        if dataset.history_steps == 2:
            batch["q_nm1"] = previous
        batch["q_n"] = current
        prediction = predict_next(
            model, batch, dataset, statistics, args, seed + step, device
        )
        previous, current = current, prediction
        target = batch["q_np1"].to(device)
        channel = torch.sqrt(
            (current - target).square().sum(dim=(0, 2, 3))
            / target.square().sum(dim=(0, 2, 3)).clamp_min(1e-20)
        )
        predicted_vorticity = interior_vorticity(current, dataset.dx)
        target_vorticity = interior_vorticity(target, dataset.dx)
        vorticity_l2 = torch.sqrt(
            (predicted_vorticity - target_vorticity).square().sum()
            / target_vorticity.square().sum().clamp_min(1e-20)
        )
        rows.append({
            "step": step + 1,
            "l2": global_l2(current, target),
            "u": float(channel[0]), "v": float(channel[1]), "p": float(channel[2]),
            "vorticity": float(vorticity_l2),
        })
    return {
        "step20_l2": rows[-1]["l2"],
        "max_first20_l2": max(row["l2"] for row in rows),
        "mean_first20_l2": float(np.mean([row["l2"] for row in rows])),
        "step20_vorticity_l2": rows[-1]["vorticity"],
        "max_first20_vorticity_l2": max(row["vorticity"] for row in rows),
    }, rows


@torch.no_grad()
def evaluate_multi_offset_rollouts(
    model: torch.nn.Module,
    dataset: CyclicWindowDataset,
    horizon: int,
    stride: int,
    statistics: FieldStatistics,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    if stride <= 0:
        raise ValueError("rollout start stride must be positive")
    starts = list(range(0, len(dataset) - horizon + 1, stride))
    if starts[-1] != len(dataset) - horizon:
        starts.append(len(dataset) - horizon)
    rows = []
    for start in starts:
        metrics, _ = evaluate_rollout(
            model, dataset, start, horizon, statistics, args,
            seed + start * 1009, device,
        )
        rows.append({"start": start} | metrics)
    return {
        "offset_count": len(rows),
        "mean_step20_l2": float(np.mean([row["step20_l2"] for row in rows])),
        "worst_step20_l2": max(row["step20_l2"] for row in rows),
        "mean_step20_vorticity_l2": float(np.mean(
            [row["step20_vorticity_l2"] for row in rows]
        )),
        "worst_step20_vorticity_l2": max(
            row["step20_vorticity_l2"] for row in rows
        ),
    }, rows


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    statistics = FieldStatistics.from_json(checkpoint["statistics"])
    model, model_spec = load_pretrained(
        checkpoint, device, args.endpoint_eos,
        args.endpoint_eos_width, args.endpoint_eos_depth,
        args.endpoint_eos_direct,
        args.endpoint_eos_warp,
        args.endpoint_eos_kinematic_warp,
        args.endpoint_vorticity,
        args.endpoint_vorticity_residual,
        args.endpoint_vorticity_dedicated,
    )
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("the pretrained model is not fully unfrozen")

    training_data = CyclicWindowDataset(args.data, args.training_scope)
    split = int(round(len(training_data) * (1.0 - args.validation_fraction)))
    train_indices = (
        list(range(split)) if args.training_scope == "cycle1"
        else list(range(len(training_data)))
    )
    if args.max_train_transitions is not None:
        train_indices = train_indices[: args.max_train_transitions]
    validation_indices = list(range(split, len(training_data)))
    if args.teacher_error_augmentation or args.rollout_error_augmentation:
        collector = (
            collect_rollout_input_errors
            if args.rollout_error_augmentation else collect_teacher_input_errors
        )
        teacher_errors = collector(
            model, training_data, train_indices, statistics, args,
            args.seed + 700_000, device,
        )
        train_set: Dataset[dict[str, Tensor]] = TeacherErrorDataset(
            training_data, train_indices, teacher_errors, args.teacher_error_scale
        )
    else:
        teacher_errors = {}
        train_set = Subset(training_data, train_indices)
    loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.min_lr
    )
    history = []
    best = float("inf")
    best_epoch = 0
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        components = []
        for batch_index, batch in enumerate(loader):
            batch = perturb_input(batch, statistics, args.input_noise, device)
            transition = make_transition(
                args.mode, batch, statistics, training_data, device,
                args.seed + epoch * 100_000 + batch_index,
                args.fine_fraction, args.halo,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                loss, values = one_step_loss(
                    model, transition, args.train_flow_steps, args.endpoint_weight,
                    args.flow_weight,
                    args.coarse_weight, args.trace_weight,
                    args.vorticity_weight, args.endpoint_consistency_weight,
                    args.endpoint_eos_iterations,
                    args.schwarz_iterations, args.schwarz_relaxation,
                    args.sampler,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            components.append({"loss": float(loss.detach())} | values)
        scheduler.step()
        record = {
            "epoch": epoch,
            **{key: float(np.mean([value[key] for value in components]))
               for key in (
                   "loss", "flow", "endpoint", "coarse", "trace",
                   "vorticity", "consistency",
                   "vorticity_head",
               )},
            "lr": optimizer.param_groups[0]["lr"],
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            if args.training_scope == "cycle1":
                evaluation_data = training_data
                evaluation_indices = validation_indices
                evaluation_start = split
            else:
                evaluation_data = CyclicWindowDataset(args.data, "cycle2")
                evaluation_indices = list(range(min(20, len(evaluation_data))))
                evaluation_start = 0
            one_step = evaluate_one_step(
                model, evaluation_data, evaluation_indices, statistics, args,
                args.seed + 800_000, device,
            )
            rollout, _ = evaluate_rollout(
                model, evaluation_data, evaluation_start, 20, statistics,
                args, args.seed + 850_000, device,
            )
            multi_offset = None
            if args.selection_metric == "vorticity":
                multi_offset, _ = evaluate_multi_offset_rollouts(
                    model, evaluation_data, 20, args.rollout_start_stride,
                    statistics, args, args.seed + 875_000, device,
                )
            if evaluation_data is not training_data:
                evaluation_data.close()
            record |= {"validation_one_step_l2": one_step} | {
                f"validation_{key}": value for key, value in rollout.items()
            }
            if multi_offset is not None:
                record |= {
                    f"validation_multi_offset_{key}": value
                    for key, value in multi_offset.items()
                }
            selection = (
                rollout["step20_l2"] if args.selection_metric == "state"
                else multi_offset["worst_step20_vorticity_l2"]
            )
            if selection < best:
                best = selection
                best_epoch = epoch
                save_checkpoint(args.output_dir / "checkpoint.pt", {
                    "model": model.state_dict(), "statistics": statistics.to_json(),
                    "model_spec": model_spec, "args": vars(args), "epoch": epoch,
                    "validation": {"one_step_l2": one_step} | rollout | (
                        {"multi_offset": multi_offset} if multi_offset is not None else {}
                    ),
                })
        history.append(record)
        print(json.dumps(record), flush=True)

    best_checkpoint = torch.load(
        args.output_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(best_checkpoint["model"])
    cycle2 = CyclicWindowDataset(args.data, "cycle2")
    cycle2_metrics, rows = evaluate_rollout(
        model, cycle2, 0, 20, statistics, args, args.seed + 900_000, device
    )
    with (args.output_dir / "cycle2_first20.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    final_multi_offset = None
    if args.selection_metric == "vorticity":
        final_multi_offset, offset_rows = evaluate_multi_offset_rollouts(
            model, cycle2, 20, args.rollout_start_stride, statistics,
            args, args.seed + 925_000, device,
        )
        with (args.output_dir / "cycle2_multi_offset.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=offset_rows[0].keys())
            writer.writeheader()
            writer.writerows(offset_rows)
    summary = {
        "training_contract": "one_step_pairs_only",
        "training_scope": args.training_scope,
        "evaluation_scope": (
            "held_cycle2" if args.training_scope == "cycle1"
            else "in_sample_trajectory_capacity"
        ),
        "all_pretrained_weights_unfrozen": True,
        "teacher_error_records": len(teacher_errors),
        "model_spec": model_spec,
        "mode": args.mode,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_validation_selection": best,
        "cycle2_first20": cycle2_metrics,
        "cycle2_multi_offset": final_multi_offset,
        "gate_passed": (
            cycle2_metrics["step20_l2"] < 0.1
            if args.selection_metric == "state"
            else final_multi_offset["worst_step20_vorticity_l2"] < 0.01
        ),
        "history": history,
        "wall_seconds": time.perf_counter() - start_time,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)
    training_data.close()
    cycle2.close()


if __name__ == "__main__":
    main()
