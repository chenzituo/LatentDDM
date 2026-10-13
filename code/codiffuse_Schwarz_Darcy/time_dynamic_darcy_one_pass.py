"""Hardware-matched timing for the frozen dynamic Darcy one-pass solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from torch.utils.data import DataLoader

from .adapter import H5DarcyDataset
from .evaluate import move_batch
from .latent_schur_teacher import inference_only_context
from .train_incremental_flux_remote_schur import (
    _decomposed_fields, _load_communicator,
)
from .train_residual_port_chart import frozen_context, load_frozen_local, sha256
from .train_residual_port_communicator import load_frozen_chart


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    return parser.parse_args()


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.inference_mode()
def time_solver(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("matched timing requires CUDA")
    if min(args.batch_size, args.warmup, args.repeats) <= 0:
        raise ValueError("timing extent must be positive")
    device = torch.device("cuda")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, _ = load_frozen_chart(args.chart_checkpoint, device)
    communicator, payload = _load_communicator(args.checkpoint, spec, device)
    if getattr(communicator, "analyzer_mode", None) != \
            "value_flux_dynamic_relative_green":
        raise ValueError("timing checkpoint is not the frozen dynamic candidate")
    for module in (local, chart, communicator):
        module.eval()
    dataset = H5DarcyDataset(args.data, "validation")
    host = next(iter(DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )))
    batch = move_batch(host, device)

    def endpoint_only():
        return frozen_context(
            local, adapter, batch, spec, args.core, args.halo,
        )

    def complete():
        supervised = endpoint_only()
        context = inference_only_context(supervised)
        return _decomposed_fields(
            communicator, chart, context, batch, spec, scale,
        )["assembled"]

    for _ in range(args.warmup):
        endpoint_only(); complete()
    _synchronize()

    def measure(function):
        samples = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            _synchronize(); started = time.perf_counter()
            function(); _synchronize()
            samples.append((time.perf_counter() - started) / args.batch_size)
        return {
            "mean_seconds_per_case": statistics.mean(samples),
            "median_seconds_per_case": statistics.median(samples),
            "stdev_seconds_per_case": statistics.stdev(samples),
            "minimum_seconds_per_case": min(samples),
            "maximum_seconds_per_case": max(samples),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }

    endpoint = measure(endpoint_only)
    full = measure(complete)
    result = {
        "contract": "darcy_dynamic_one_pass_h200_timing_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "local_checkpoint": str(args.local_checkpoint.resolve()),
        "local_checkpoint_sha256": sha256(args.local_checkpoint),
        "device": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
        "warmup": args.warmup, "repeats": args.repeats,
        "tile_count": 9, "local_nfe": 9,
        "batched_local_calls": 1, "global_communication_stages": 1,
        "endpoint": endpoint, "full_one_pass": full,
        "communication_and_chart_overhead_fraction": (
            full["mean_seconds_per_case"] / endpoint["mean_seconds_per_case"] - 1.0
        ),
        "physics_diagnostics_in_timed_region": False,
        "target_or_discrete_operator_used_by_inference": False,
        "test_subsets_opened": False,
        "checkpoint_frozen": bool(
            payload.get("communicator_spec", {}).get("frozen_candidate", False)
        ),
    }
    dataset.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(time_solver(parse_args()), indent=2))
