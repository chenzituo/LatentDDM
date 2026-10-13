"""Evaluate one-pass physics-defined Galerkin corrections on held-out Darcy domains."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import (
    DarcyAdapter,
    DarcyStatistics,
    H5DarcyDataset,
    MODEL_SPEC,
    build_model,
    checkpoint_communication_stage,
)
from .evaluate import (
    composition_step,
    energy_factor,
    energy_relative_error,
    initial_state,
    move_batch,
    relative_l2,
    residual_rms,
)
from .two_level import galerkin_correct, tile_pou_basis


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(
        "codiffuse_Schwarz_Darcy/data/darcy_complex_v1.h5"
    ))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--offsets", nargs="+", default=("0,0", "8,8", "16,16"))
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--relative-cutoff", type=float, default=1.0e-8)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def parse_offsets(values: list[str]) -> list[tuple[int, int]]:
    offsets = []
    for value in values:
        y, x = (int(piece) for piece in value.split(","))
        if not 0 <= y < MODEL_SPEC["core"] or not 0 <= x < MODEL_SPEC["core"]:
            raise ValueError(f"invalid cover offset {value}")
        offsets.append((y, x))
    return offsets


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["method"]].append(record)
    metrics = (
        "relative_l2", "energy_relative_error", "residual_rms",
        "coarse_seconds", "total_seconds", "coarse_rank", "coarse_condition",
        "correction_energy", "correction_relative_l2", "local_nfe",
    )
    summary = {}
    for method, values in grouped.items():
        summary[method] = {}
        for metric in metrics:
            samples = np.asarray([float(value[metric]) for value in values])
            finite = samples[np.isfinite(samples)]
            summary[method][metric] = float(finite.mean()) if len(finite) else float("nan")
        summary[method]["records"] = len(values)
    local_energy = np.asarray([
        record["energy_relative_error"] for record in records
        if record["method"] == "local_pou"
    ])
    for method in ("scalar_energy", "galerkin_constant", "galerkin_linear"):
        corrected_energy = np.asarray([
            record["energy_relative_error"] for record in records
            if record["method"] == method
        ])
        if len(corrected_energy) == len(local_energy) and len(local_energy):
            summary[method]["energy_nonincrease_fraction"] = float(
                np.mean(corrected_energy <= local_energy + 1.0e-7)
            )
    return summary


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    statistics = DarcyStatistics.from_json(checkpoint["statistics"])
    adapter = DarcyAdapter(statistics)
    communication_stage = checkpoint_communication_stage(checkpoint)
    model = build_model(adapter, communication_stage=communication_stage).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = H5DarcyDataset(args.data, args.split)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    offsets = parse_offsets(args.offsets)
    records: list[dict[str, Any]] = []
    basis_cache: dict[tuple[int, int, int, int, int], Tensor] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    seen = 0
    for host_batch in loader:
        if args.max_cases is not None and seen >= args.max_cases:
            break
        batch = move_batch(host_batch, device)
        if args.max_cases is not None and seen + batch["solution"].shape[0] > args.max_cases:
            keep = args.max_cases - seen
            batch = {
                key: value[:keep] if isinstance(value, Tensor) and value.shape[0] == batch["solution"].shape[0] else value
                for key, value in batch.items()
            }
        batch_size = batch["solution"].shape[0]
        seen += batch_size
        for offset_y, offset_x in offsets:
            cover = CartesianCover(
                batch["solution"].shape[-2], batch["solution"].shape[-1],
                MODEL_SPEC["core"], MODEL_SPEC["halo"], offset_y, offset_x,
            )
            current = initial_state(batch, statistics, "zero")
            if device.type == "cuda":
                torch.cuda.synchronize()
            local_start = time.perf_counter()
            preliminary, _ = composition_step(
                model, adapter, batch, current, current, cover, "pou", 1.0,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            local_seconds = (time.perf_counter() - local_start) / batch_size

            candidates: list[tuple[str, Tensor, float, Tensor, Tensor, Tensor]] = []
            nan = torch.full((batch_size,), torch.nan, device=device)
            zero = torch.zeros(batch_size, device=device)
            candidates.append(("local_pou", preliminary, 0.0, nan, nan, zero))

            factor = energy_factor(current, preliminary - current, batch)
            scalar = current + factor[:, None, None, None] * (preliminary - current)
            candidates.append(("scalar_energy", scalar, 0.0, torch.ones_like(factor), nan, zero))

            for degree, method in ((0, "galerkin_constant"), (1, "galerkin_linear")):
                key = (cover.height, cover.width, offset_y, offset_x, degree)
                if key not in basis_cache:
                    basis_cache[key] = tile_pou_basis(
                        cover, degree=degree, device=device, dtype=preliminary.dtype,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                coarse_start = time.perf_counter()
                corrected, diagnostics = galerkin_correct(
                    preliminary, batch, basis_cache[key],
                    relative_cutoff=args.relative_cutoff,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                coarse_seconds = (time.perf_counter() - coarse_start) / batch_size
                candidates.append((
                    method, corrected, coarse_seconds, diagnostics.rank,
                    diagnostics.condition, diagnostics.correction_energy,
                ))

            for method, prediction, coarse_seconds, rank, condition, correction_energy in candidates:
                l2 = relative_l2(prediction, batch["solution"], batch["mask"])
                energy = energy_relative_error(
                    prediction, batch["solution"], batch["logk"], batch["mask"],
                )
                residual = residual_rms(prediction, batch)
                correction_l2 = relative_l2(prediction, preliminary, batch["mask"])
                for item in range(batch_size):
                    records.append({
                        "case_index": int(batch["case_index"][item]),
                        "method": method,
                        "offset_y": offset_y,
                        "offset_x": offset_x,
                        "tile_count": cover.tile_count,
                        "relative_l2": float(l2[item]),
                        "energy_relative_error": float(energy[item]),
                        "residual_rms": float(residual[item]),
                        "coarse_seconds": float(coarse_seconds),
                        "total_seconds": float(local_seconds + coarse_seconds),
                        "coarse_rank": float(rank[item]) if torch.isfinite(rank[item]) else float("nan"),
                        "coarse_condition": float(condition[item]) if torch.isfinite(condition[item]) else float("nan"),
                        "correction_energy": float(correction_energy[item]),
                        "correction_relative_l2": float(correction_l2[item]),
                        "local_nfe": cover.tile_count,
                    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = {
        "contract": "darcy_one_pass_two_level_galerkin_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_contract": checkpoint.get("contract"),
        "communication_stage": communication_stage,
        "split": args.split,
        "configuration": {
            "offsets": offsets,
            "relative_cutoff": args.relative_cutoff,
            "max_cases": args.max_cases,
        },
        "records": len(records),
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "summary": aggregate(records),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


def main() -> None:
    result = evaluate(parse_args())
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
