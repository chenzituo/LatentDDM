"""Remote-different/local-identical Darcy pairs for the locality obstruction.

A boundary-data bump is placed in a tile as far as possible from a selected target
tile while the geometry, coefficient, boundary data, and source in the target
neighborhood remain unchanged.  The exact elliptic solution is recomputed.
Any deterministic radius-r architecture with r smaller than the nearest
changed input tile must return the same target output for the pair, although
the reference target generally changes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import (
    DarcyAdapter, DarcyStatistics, H5DarcyDataset, MODEL_SPEC,
    batch_to_transition, build_model, tiled_darcy_conditions,
)
from .physics import solve_darcy
from .train_operator_composer import CONTRACT, OperatorOnlyDarcyComposer


Tensor = torch.Tensor


def _tile_pair(mask: np.ndarray, boundary: np.ndarray,
               core: int) -> tuple[tuple[int, int], tuple[int, int]]:
    height, width = boundary.shape
    ny, nx = (height + core - 1) // core, (width + core - 1) // core
    candidates = []
    for iy in range(ny):
        for ix in range(nx):
            active = mask[iy * core:min((iy + 1) * core, height),
                          ix * core:min((ix + 1) * core, width)]
            if active.size and active.mean() >= 0.12:
                candidates.append((iy, ix, float(active.mean())))
    if len(candidates) < 2:
        raise ValueError("domain has fewer than two sufficiently active core tiles")
    remote_candidates = []
    for iy, ix, fraction in candidates:
        patch = boundary[iy * core:min((iy + 1) * core, height),
                         ix * core:min((ix + 1) * core, width)]
        if np.any(patch):
            remote_candidates.append((iy, ix, fraction))
    pairs = [(abs(a[0] - b[0]) + abs(a[1] - b[1]), a[2], a[:2], b[:2])
             for a in candidates for b in remote_candidates if a[:2] != b[:2]]
    if not pairs:
        raise ValueError("domain has no separated boundary and target tiles")
    _, _, target, remote = max(pairs)
    return target, remote


def _boundary_bump(boundary: np.ndarray, remote: tuple[int, int], core: int,
                   amplitude: float) -> np.ndarray:
    height, width = boundary.shape
    y0, x0 = remote[0] * core, remote[1] * core
    y1, x1 = min(y0 + core, height), min(x0 + core, width)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cy, cx = .5 * (y0 + y1 - 1), .5 * (x0 + x1 - 1)
    sigma = max(core / 5.0, 1.0)
    bump = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
    result = np.zeros((height, width), dtype=np.float64)
    result[y0:y1, x0:x1] = amplitude * bump * boundary[y0:y1, x0:x1]
    if not np.any(result):
        raise ValueError("remote boundary bump has empty support")
    return result


def _pair_batch(item: dict[str, Any], bc_value: np.ndarray,
                solution: np.ndarray, device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in item.items():
        if isinstance(value, Tensor) and value.ndim >= 3:
            result[key] = torch.stack((value, value)).to(device)
        elif key == "case_index":
            result[key] = torch.stack((value, value)).to(device)
        else:
            result[key] = value
    result["bc_value"][1] = torch.from_numpy(bc_value.astype(np.float32))[None].to(device)
    result["solution"][1] = torch.from_numpy(solution.astype(np.float32))[None].to(device)
    return result


def _weighted_rms(field: Tensor, weight: Tensor) -> float:
    return float(((field.square() * weight).sum() / weight.sum().clamp_min(1.0)).sqrt())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("A", "B", "C", "D"))
    parser.add_argument("--cases-per-split", type=int, default=4)
    parser.add_argument("--boundary-amplitudes", type=float, nargs="+", default=(0.5, 1.0))
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CONTRACT or not checkpoint.get("operator_free"):
        raise ValueError("checkpoint is not the operator-only Darcy composer")
    statistics = DarcyStatistics.from_json(checkpoint["statistics"])
    adapter = DarcyAdapter(statistics, use_discrete_residual=False)
    model = OperatorOnlyDarcyComposer(build_model(adapter, communication_stage="none"), adapter).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    rows: list[dict[str, Any]] = []
    core, halo = MODEL_SPEC["core"], MODEL_SPEC["halo"]

    for split in args.splits:
        dataset = H5DarcyDataset(args.data, split)
        indices = np.linspace(0, len(dataset) - 1,
                              min(args.cases_per_split, len(dataset)), dtype=int)
        for case_index in indices:
            item = dataset[int(case_index)]
            mask = item["mask"][0].numpy().astype(bool)
            boundary = item["bc_mask"][0].numpy().astype(bool) & mask
            target_tile, remote_tile = _tile_pair(mask, boundary, core)
            height, width = mask.shape
            cover = CartesianCover(height, width, core, halo)
            target_index = target_tile[0] * cover.grid_shape[1] + target_tile[1]
            source = item["source"][0].numpy().astype(np.float64)
            logk = item["logk"][0].numpy().astype(np.float64)
            bc_value = item["bc_value"][0].numpy().astype(np.float64)
            base_solution = item["solution"][0].numpy().astype(np.float64)
            for amplitude_multiplier in args.boundary_amplitudes:
                delta_bc = _boundary_bump(
                    boundary, remote_tile, core,
                    amplitude_multiplier * statistics.bc_std,
                )
                modified_bc = bc_value + delta_bc
                modified_solution, _ = solve_darcy(
                    mask, np.power(10.0, np.clip(logk, -6.0, 6.0)),
                    source, modified_bc, dx=2.0 / max(height, width),
                )
                batch = _pair_batch(item, modified_bc, modified_solution, device)
                current = torch.zeros_like(batch["solution"])
                transition = batch_to_transition(batch, current, target=batch["solution"])
                physical, scalar = tiled_darcy_conditions(adapter, transition, cover)
                valid = cover.valid_tiles(2, device, current.dtype)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    corrected_raw, local_raw, coefficients_raw = model(
                        physical, scalar, valid, cover.grid_shape,
                    )
                corrected, local = corrected_raw.float(), local_raw.float()
                coefficients = coefficients_raw.float()
                tile_count = cover.tile_count
                base_id, modified_id = target_index, tile_count + target_index
                target_weight = cover.tile(batch["mask"].float())[base_id]
                exact_tiles = cover.tile(batch["solution"].float())
                exact_delta = exact_tiles[modified_id] - exact_tiles[base_id]
                local_delta = statistics.solution_std * (
                    local[modified_id] - local[base_id]
                )
                communicated_delta = statistics.solution_std * (
                    corrected[modified_id] - corrected[base_id]
                )
                physical_delta = physical[modified_id] - physical[base_id]
                scalar_delta = scalar[modified_id] - scalar[base_id]
                changed_input_tiles = cover.tile(
                    torch.from_numpy(delta_bc.astype(np.float32))[None, None].to(device)
                )
                changed = (changed_input_tiles.abs().flatten(1).amax(1) > 0).nonzero().flatten().tolist()
                tx, ty = target_tile[1], target_tile[0]
                distances = [abs(ty - index // cover.grid_shape[1])
                             + abs(tx - index % cover.grid_shape[1]) for index in changed]
                unchanged_radius = min(distances) - 1
                true_rms = _weighted_rms(exact_delta, target_weight)
                local_rms = _weighted_rms(local_delta, target_weight)
                communicated_rms = _weighted_rms(communicated_delta, target_weight)
                response_error = _weighted_rms(communicated_delta - exact_delta, target_weight)
                true_flat = (exact_delta * target_weight).flatten()
                pred_flat = (communicated_delta * target_weight).flatten()
                dot = torch.dot(true_flat, pred_flat)
                cosine = dot / (
                    true_flat.norm().clamp_min(1.0e-12)
                    * pred_flat.norm().clamp_min(1.0e-12)
                )
                gain = dot / true_flat.square().sum().clamp_min(1.0e-12)
                code_delta = coefficients[modified_id] - coefficients[base_id]
                rows.append({
                    "split": split, "case_index": int(case_index),
                    "boundary_amplitude_multiplier": amplitude_multiplier,
                    "target_tile_y": target_tile[0], "target_tile_x": target_tile[1],
                    "remote_tile_y": remote_tile[0], "remote_tile_x": remote_tile[1],
                    "nearest_changed_tile_distance": min(distances),
                    "largest_provably_blind_radius": unchanged_radius,
                    "target_physical_input_max_delta": float(physical_delta.abs().max()),
                    "target_scalar_input_max_delta": float(scalar_delta.abs().max()),
                    "target_local_output_rms_delta": local_rms,
                    "target_exact_response_rms": true_rms,
                    "radius_blind_worst_case_error_lower_bound_rms": .5 * true_rms,
                    "global_composer_response_rms": communicated_rms,
                    "global_composer_response_relative_error": response_error / max(true_rms, 1.0e-12),
                    "global_composer_response_cosine": float(cosine),
                    "global_composer_response_gain": float(gain),
                    "target_predicted_code_rms_delta": float(code_delta.square().mean().sqrt()),
                })
        dataset.close()

    if not rows:
        raise ValueError("no locality pairs were evaluated")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "pair_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    numeric = [key for key, value in rows[0].items()
               if key not in ("split", "case_index") and isinstance(value, (int, float))]
    summary = {key: float(np.mean([float(row[key]) for row in rows])) for key in numeric}
    result = {
        "contract": "darcy_remote_pair_locality_obstruction_v1",
        "checkpoint": str(args.checkpoint.resolve()), "pairs": len(rows),
        "operator_free_model": True, "exact_solver_used_for_labels_only": True,
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    figure, axis = plt.subplots(figsize=(5.2, 3.5))
    true = np.asarray([row["target_exact_response_rms"] for row in rows])
    pred = np.asarray([row["global_composer_response_rms"] for row in rows])
    lower = .5 * true
    order = np.argsort(true)
    axis.plot(true[order], lower[order], label="radius-blind lower bound", color="#e45756")
    axis.scatter(true, pred, label="global composer response", color="#4c78a8", alpha=.8)
    axis.set_xlabel("exact remote response RMS on target tile")
    axis.set_ylabel("response / unavoidable error RMS")
    axis.grid(alpha=.25); axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(args.output_dir / f"locality_obstruction.{suffix}", dpi=220)
    plt.close(figure)
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
