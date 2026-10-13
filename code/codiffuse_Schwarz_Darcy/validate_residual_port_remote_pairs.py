"""Validation-only remote-pair gate for the phase-separated Darcy composer.

The target tile and its radius-zero input are held fixed while a physical
Dirichlet segment in a maximally separated tile is perturbed.  This directly
tests whether the one-message compatibility code transmits a response that a
strictly local endpoint cannot represent.  Gate thresholds below are declared
in code before inspecting this candidate's paired-world results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import H5DarcyDataset
from .evaluate import move_batch
from .evaluate_locality_obstruction import _boundary_bump, _tile_pair, _weighted_rms
from .physics import solve_darcy
from .residual_port_composer import (
    ResidualPortCommunicator, SignedSourceLinearResidualPortCommunicator,
    SourceLinearResidualPortCommunicator,
)
from .train_residual_port_chart import frozen_context, load_frozen_local, sha256
from .train_residual_port_communicator import (
    CANONICAL_ANALYZER_CONTRACT, CONTRACT, communicator_features,
    forward_composer, load_frozen_chart, source_linear_signal_width,
)
from .validate_latent_schur import measured_q


Tensor = torch.Tensor
PAIR_GATE = {
    "target_input_max_delta_at_most": 1.0e-7,
    "local_response_fraction_at_most": 1.0e-5,
    "global_code_rms_delta_at_least": 1.0e-5,
    "global_target_error_fraction_of_radius0_at_most": 0.95,
    "global_full_error_fraction_of_radius0_at_most": 0.95,
    "global_target_gain_min": 0.05,
    "global_target_gain_max": 1.95,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=(0.5, 1.0))
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _paired_batch(
    item: dict[str, Any], delta_bc: np.ndarray, modified_solution: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in item.items():
        if isinstance(value, Tensor):
            result[key] = torch.stack((value, value)).to(device)
        else:
            result[key] = value
    delta = torch.from_numpy(delta_bc.astype(np.float32)).to(device)
    result["bc_value"][1] += delta[None]
    result["solution"][1] = torch.from_numpy(
        modified_solution.astype(np.float32),
    )[None].to(device)
    if "dirichlet_face_value" in result:
        face_mask = result["dirichlet_face_mask"][1].to(delta.dtype)
        result["dirichlet_face_value"][1] += delta[None] * face_mask
    return result


def _load_models(args: argparse.Namespace, device: torch.device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError("remote-pair gate requires a phase-separated checkpoint")
    if checkpoint["source_local_sha256"] != sha256(args.local_checkpoint):
        raise ValueError("local checkpoint hash differs from communicator lineage")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart_path = Path(checkpoint["source_chart_checkpoint"])
    if checkpoint["source_chart_sha256"] != sha256(chart_path):
        raise ValueError("chart checkpoint hash differs from communicator lineage")
    chart, spec, scale, _ = load_frozen_chart(chart_path, device)
    cfg = checkpoint["communicator_spec"]
    analyzer_mode = cfg.get("analyzer_mode", "generic")
    if analyzer_mode in ("source_linear", "value_flux"):
        if cfg.get("canonical_analyzer_contract") != CANONICAL_ANALYZER_CONTRACT:
            raise ValueError("checkpoint lacks homogeneous compatibility contract")
        communicator_type = (
            SignedSourceLinearResidualPortCommunicator
            if cfg.get("kernel_type", "softmax") == "signed_l1_tanh"
            else SourceLinearResidualPortCommunicator
        )
        communicator = communicator_type(
            int(cfg["physical_channels"]), int(cfg["scalar_dim"]),
            spec.code_width, source_linear_signal_width(spec) + (
                8 if analyzer_mode == "value_flux" else 0
            ),
            token_width=int(cfg["token_width"]), heads=int(cfg["heads"]),
        ).to(device)
        communicator.analyzer_mode = analyzer_mode
    else:
        communicator = ResidualPortCommunicator(
            1, int(cfg["physical_channels"]), int(cfg["scalar_dim"]),
            spec.code_width, token_width=int(cfg["token_width"]),
            heads=int(cfg["heads"]),
            aligned_width=int(cfg.get("aligned_width", 0)),
        ).to(device)
        communicator.analyzer_mode = cfg.get("analyzer_mode", "generic")
    communicator.load_state_dict(checkpoint["communicator"], strict=True)
    local.eval(); chart.eval(); communicator.eval()
    return checkpoint, local, adapter, chart, spec, scale, communicator


def _relative_rms(error: Tensor, reference: Tensor, weight: Tensor) -> float:
    return _weighted_rms(error, weight) / max(_weighted_rms(reference, weight), 1.0e-12)


@torch.no_grad()
def validate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint, local, adapter, chart, spec, scale, communicator = _load_models(
        args, device,
    )
    dataset = H5DarcyDataset(args.data, "validation")
    if not dataset.has_face_dirichlet:
        raise ValueError("paired gate requires the face-resolved v2 corpus")
    count = min(args.cases, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    rows: list[dict[str, Any]] = []
    radii: tuple[int | None, ...] = (0, 1, 2, None)
    for case_index in indices:
        item = dataset[int(case_index)]
        mask = item["mask"][0].numpy().astype(bool)
        boundary = item["bc_mask"][0].numpy().astype(bool) & mask
        target_tile, remote_tile = _tile_pair(mask, boundary, args.core)
        cover = CartesianCover(*mask.shape, args.core, args.halo)
        target_index = target_tile[0] * cover.grid_shape[1] + target_tile[1]
        source = item["source"][0].numpy().astype(np.float64)
        logk = item["logk"][0].numpy().astype(np.float64)
        base_bc = item["bc_value"][0].numpy().astype(np.float64)
        face_mask = item["dirichlet_face_mask"].numpy().astype(bool)
        base_face_value = item["dirichlet_face_value"].numpy().astype(np.float64)
        for amplitude in args.amplitudes:
            delta_bc = _boundary_bump(
                boundary, remote_tile, args.core,
                float(amplitude) * adapter.statistics.bc_std,
            )
            modified_face_value = base_face_value + delta_bc[None] * face_mask
            modified, _ = solve_darcy(
                mask, np.power(10.0, np.clip(logk, -6.0, 6.0)), source,
                base_bc + delta_bc, dx=2.0 / max(mask.shape),
                dirichlet_face_mask=face_mask,
                dirichlet_face_value=modified_face_value,
            )
            batch = _paired_batch(item, delta_bc, modified, device)
            context = frozen_context(local, adapter, batch, spec, args.core, args.halo)
            aligned = (
                measured_q(context, batch, spec, scale, "value_flux")[0]
                if getattr(communicator, "analyzer_mode", "generic") == "value_flux"
                else communicator_features(
                    communicator, context, batch, spec, scale, adapter,
                )
            )
            exact_tiles = cover.tile(batch["solution"].float())
            tile_count = cover.tile_count
            base_id, changed_id = target_index, tile_count + target_index
            target_weight = cover.tile(batch["mask"].float())[base_id]
            exact_target = exact_tiles[changed_id] - exact_tiles[base_id]
            exact_full = batch["solution"][1] - batch["solution"][0]
            local_target = adapter.statistics.solution_std * (
                context.local[changed_id] - context.local[base_id]
            )
            target_input_delta = max(
                float((context.physical[changed_id] - context.physical[base_id]).abs().max()),
                float((context.scalar[changed_id] - context.scalar[base_id]).abs().max()),
            )
            row: dict[str, Any] = {
                "case_index": int(case_index), "amplitude": float(amplitude),
                "target_tile_y": target_tile[0], "target_tile_x": target_tile[1],
                "remote_tile_y": remote_tile[0], "remote_tile_x": remote_tile[1],
                "target_input_max_delta": target_input_delta,
                "target_exact_response_rms": _weighted_rms(exact_target, target_weight),
                "local_target_response_fraction": _relative_rms(
                    local_target, exact_target, target_weight,
                ),
            }
            for radius in radii:
                assembled, codes, _, detail = forward_composer(
                    communicator, chart, context, scale, batch,
                    max_graph_radius=radius, aligned_features=aligned,
                )
                label = "global" if radius is None else f"radius{radius}"
                predicted_target = adapter.statistics.solution_std * (
                    detail["corrected"][changed_id] - detail["corrected"][base_id]
                )
                predicted_full = adapter.statistics.solution_std * (
                    assembled[1] - assembled[0]
                )
                truth = (exact_target * target_weight).flatten()
                pred = (predicted_target * target_weight).flatten()
                row[f"{label}_target_response_relative_error"] = _relative_rms(
                    predicted_target - exact_target, exact_target, target_weight,
                )
                row[f"{label}_full_response_relative_error"] = _relative_rms(
                    predicted_full - exact_full, exact_full, batch["mask"][0],
                )
                row[f"{label}_target_response_gain"] = float(
                    torch.dot(truth, pred) / truth.square().sum().clamp_min(1.0e-12)
                )
                row[f"{label}_code_rms_delta"] = float(
                    (codes[1, target_index] - codes[0, target_index]).square().mean().sqrt()
                )
            rows.append(row)
    dataset.close()
    mean = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in rows[0] if key not in (
            "case_index", "target_tile_y", "target_tile_x",
            "remote_tile_y", "remote_tile_x",
        )
    }
    gates = {
        "target_input_identical": mean["target_input_max_delta"]
            <= PAIR_GATE["target_input_max_delta_at_most"],
        "local_endpoint_blind": mean["local_target_response_fraction"]
            <= PAIR_GATE["local_response_fraction_at_most"],
        "global_code_receives_remote_change": mean["global_code_rms_delta"]
            >= PAIR_GATE["global_code_rms_delta_at_least"],
        "global_improves_target_over_radius0":
            mean["global_target_response_relative_error"]
            <= PAIR_GATE["global_target_error_fraction_of_radius0_at_most"]
            * mean["radius0_target_response_relative_error"],
        "global_improves_full_response_over_radius0":
            mean["global_full_response_relative_error"]
            <= PAIR_GATE["global_full_error_fraction_of_radius0_at_most"]
            * mean["radius0_full_response_relative_error"],
        "global_target_gain_physical":
            PAIR_GATE["global_target_gain_min"] <= mean["global_target_response_gain"]
            <= PAIR_GATE["global_target_gain_max"],
    }
    result = {
        "contract": "darcy_residual_port_remote_pair_gate_v1",
        "selection_split": "validation", "test_subsets_opened": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_local_sha256": checkpoint["source_local_sha256"],
        "pairs": len(rows), "predeclared_thresholds": PAIR_GATE,
        "summary": mean, "gates": gates, "clears": all(gates.values()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "pair_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), indent=2))
