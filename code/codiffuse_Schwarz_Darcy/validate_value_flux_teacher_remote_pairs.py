"""Untouched validation remote-pair gate for the value+flux Schur teacher."""

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
from .evaluate_locality_obstruction import _boundary_bump, _tile_pair, _weighted_rms
from .latent_schur_teacher import inference_only_context, solve_value_flux_teacher
from .physics import solve_darcy
from .train_residual_port_chart import frozen_context, load_frozen_local, sha256
from .train_residual_port_communicator import load_frozen_chart
from .validate_latent_schur import chart_corrected
from .validate_residual_port_remote_pairs import PAIR_GATE, _paired_batch


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=(0.5, 1.0))
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--damping", type=float, nargs="+", default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _relative_rms(error: Tensor, reference: Tensor, weight: Tensor) -> float:
    return _weighted_rms(error, weight) / max(_weighted_rms(reference, weight), 1.0e-12)


def _one(batch: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        key: value[index:index + 1] if isinstance(value, Tensor) and value.shape[0] == 2 else value
        for key, value in batch.items()
    }


@torch.no_grad()
def validate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, chart_checkpoint = load_frozen_chart(args.chart_checkpoint, device)
    if chart_checkpoint["source_local_sha256"] != sha256(args.local_checkpoint):
        raise ValueError("teacher chart/local lineage differs")
    dataset = H5DarcyDataset(args.data, "validation")
    if not dataset.has_face_dirichlet:
        raise ValueError("remote teacher gate requires face-resolved data")
    indices = np.linspace(0, len(dataset) - 1, min(args.cases, len(dataset)), dtype=int)
    rows = []
    for case_index in indices:
        item = dataset[int(case_index)]
        mask = item["mask"][0].numpy().astype(bool)
        boundary = item["bc_mask"][0].numpy().astype(bool) & mask
        target_tile, remote_tile = _tile_pair(mask, boundary, args.core)
        cover = CartesianCover(*mask.shape, args.core, args.halo)
        target_index = target_tile[0] * cover.grid_shape[1] + target_tile[1]
        logk = item["logk"][0].numpy().astype(np.float64)
        source = item["source"][0].numpy().astype(np.float64)
        base_bc = item["bc_value"][0].numpy().astype(np.float64)
        face_mask = item["dirichlet_face_mask"].numpy().astype(bool)
        base_face = item["dirichlet_face_value"].numpy().astype(np.float64)
        for amplitude in args.amplitudes:
            delta = _boundary_bump(
                boundary, remote_tile, args.core,
                float(amplitude) * adapter.statistics.bc_std,
            )
            modified, _ = solve_darcy(
                mask, np.power(10.0, np.clip(logk, -6.0, 6.0)), source,
                base_bc + delta, dx=2.0 / max(mask.shape),
                dirichlet_face_mask=face_mask,
                dirichlet_face_value=base_face + delta[None] * face_mask,
            )
            pair = _paired_batch(item, delta, modified, device)
            assembled = []; corrected = []; contexts = []; q_squares = []
            for state in range(2):
                one = _one(pair, state)
                context = frozen_context(local, adapter, one, spec, args.core, args.halo)
                context = inference_only_context(context)
                code, _, q_square = solve_value_flux_teacher(
                    chart, context, one, spec, scale, ridge=args.ridge,
                    damping=tuple(args.damping),
                )
                tile = chart_corrected(chart, context, code)
                assembled.append(context.cover.assemble_weighted(tile, 1) * one["mask"])
                corrected.append(tile); contexts.append(context); q_squares.append(q_square)
            local_response = adapter.statistics.solution_std * (
                contexts[1].cover.assemble_weighted(contexts[1].local, 1)
                - contexts[0].cover.assemble_weighted(contexts[0].local, 1)
            )
            teacher_response = adapter.statistics.solution_std * (assembled[1] - assembled[0])
            exact_response = pair["solution"][1:2] - pair["solution"][0:1]
            target_weight = cover.tile(pair["mask"][:1])[target_index]
            exact_target = cover.tile(exact_response)[target_index]
            local_target = cover.tile(local_response)[target_index]
            teacher_target = cover.tile(teacher_response)[target_index]
            truth = (exact_target * target_weight).flatten()
            prediction = (teacher_target * target_weight).flatten()
            rows.append({
                "case_index": int(case_index), "amplitude": float(amplitude),
                "target_input_max_delta": max(
                    float((contexts[1].physical[target_index] - contexts[0].physical[target_index]).abs().max()),
                    float((contexts[1].scalar[target_index] - contexts[0].scalar[target_index]).abs().max()),
                ),
                "target_exact_response_rms": _weighted_rms(exact_target, target_weight),
                "local_target_response_relative_error": _relative_rms(
                    local_target - exact_target, exact_target, target_weight,
                ),
                "teacher_target_response_relative_error": _relative_rms(
                    teacher_target - exact_target, exact_target, target_weight,
                ),
                "local_full_response_relative_error": _relative_rms(
                    local_response - exact_response, exact_response, pair["mask"][:1],
                ),
                "teacher_full_response_relative_error": _relative_rms(
                    teacher_response - exact_response, exact_response, pair["mask"][:1],
                ),
                "teacher_target_response_gain": float(
                    torch.dot(truth, prediction) / truth.square().sum().clamp_min(1.0e-12)
                ),
                "teacher_complete_q_cycle_square": float(np.mean(q_squares)),
            })
    dataset.close()
    mean = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in rows[0] if key != "case_index"
    }
    gates = {
        "target_input_identical": mean["target_input_max_delta"]
            <= PAIR_GATE["target_input_max_delta_at_most"],
        "teacher_improves_target_over_local":
            mean["teacher_target_response_relative_error"]
            <= PAIR_GATE["global_target_error_fraction_of_radius0_at_most"]
            * mean["local_target_response_relative_error"],
        "teacher_improves_full_response_over_local":
            mean["teacher_full_response_relative_error"]
            <= PAIR_GATE["global_full_error_fraction_of_radius0_at_most"]
            * mean["local_full_response_relative_error"],
        "teacher_target_gain_physical":
            PAIR_GATE["global_target_gain_min"] <= mean["teacher_target_response_gain"]
            <= PAIR_GATE["global_target_gain_max"],
        "teacher_complete_q_nonregression": mean["teacher_complete_q_cycle_square"] <= 1.0,
    }
    result = {
        "contract": "darcy_value_flux_teacher_remote_pair_gate_v1",
        "selection_split": "validation", "test_subsets_opened": False,
        "teacher_used_target_or_pde_labels": False,
        "ridge": args.ridge, "pairs": len(rows), "summary": mean,
        "gates": gates, "clears": all(gates.values()),
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
