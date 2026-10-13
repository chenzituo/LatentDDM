"""Evaluate one-step operator-only Darcy composition on held-out geometries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import (
    DarcyAdapter, DarcyStatistics, FaceDarcyAdapter, H5DarcyDataset,
    MODEL_SPEC, build_model,
)
from .evaluate import (
    energy_relative_error, move_batch, relative_l2, residual_rms,
)
from .train_operator_composer import CONTRACT, OperatorOnlyDarcyComposer, one_step


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "val", "A", "B", "C", "D"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--offsets", nargs="+", default=("0,0", "8,8", "16,16"))
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CONTRACT or not checkpoint.get("operator_free"):
        raise ValueError("checkpoint is not the operator-only Darcy composer")
    statistics = DarcyStatistics.from_json(checkpoint["statistics"])
    adapter = (
        FaceDarcyAdapter(statistics)
        if checkpoint.get("source_contract")
        == "darcy_sni_face_boundary_to_solution_v1"
        else DarcyAdapter(statistics, use_discrete_residual=False)
    )
    local = build_model(adapter, communication_stage="none")
    model = OperatorOnlyDarcyComposer(local, adapter).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = H5DarcyDataset(args.data, args.split)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    offsets = [tuple(int(piece) for piece in value.split(",")) for value in args.offsets]
    if any(len(value) != 2 or min(value) < 0 or max(value) >= MODEL_SPEC["core"] for value in offsets):
        raise ValueError("offsets must be y,x pairs inside the core")
    records: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    processed = 0
    for host_batch in loader:
        if args.max_cases is not None and processed >= args.max_cases:
            break
        batch = move_batch(host_batch, device)
        original_batch = batch["solution"].shape[0]
        if args.max_cases is not None:
            keep = min(original_batch, args.max_cases - processed)
            for key, value in list(batch.items()):
                if isinstance(value, Tensor) and value.shape[0] == original_batch:
                    batch[key] = value[:keep]
        target = batch["solution"]
        for offset_y, offset_x in offsets:
            cover = CartesianCover(
                target.shape[-2], target.shape[-1], MODEL_SPEC["core"],
                MODEL_SPEC["halo"], offset_y, offset_x,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction, _ = one_step(
                    model, adapter, batch, MODEL_SPEC["core"], MODEL_SPEC["halo"], cover,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds = (time.perf_counter() - started) / target.shape[0]
            # Discrete diagnostics occur after the timed operator-only method;
            # they are evaluation measurements, never model inputs or updates.
            l2 = relative_l2(prediction, target, batch["mask"])
            energy = energy_relative_error(prediction, target, batch["logk"], batch["mask"])
            residual = residual_rms(prediction, batch)
            for item in range(target.shape[0]):
                records.append({
                    "case_index": int(batch["case_index"][item]),
                    "offset_y": offset_y, "offset_x": offset_x,
                    "relative_l2": float(l2[item]),
                    "energy_relative_error": float(energy[item]),
                    "residual_rms": float(residual[item]),
                    "wall_seconds": seconds,
                    "local_nfe": cover.tile_count,
                    "communication_rounds": 1,
                })
        processed += target.shape[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        key: float(np.mean([record[key] for record in records]))
        for key in (
            "relative_l2", "energy_relative_error", "residual_rms",
            "wall_seconds", "local_nfe", "communication_rounds",
        )
    }
    result = {
        "contract": "darcy_operator_only_one_step_evaluation_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "operator_free_inference": True,
        "discrete_diagnostics_outside_timing": True,
        "cases": processed,
        "offsets": offsets,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
