"""Same-process H200 timing for Local POU, LatentDDM, and POU-SNI.

All methods start from the same resident batch and frozen face-local operator.
The timed paths do not form diagnostics or access target fields.  POU-SNI-k
includes the endpoint call and k further weighted-POU correction sweeps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import H5DarcyDataset, tiled_sni_conditions
from .evaluate import initial_state, move_batch, relative_l2
from .train_incremental_flux_remote_schur import (
    _decomposed_fields,
    _load_communicator,
)
from .train_residual_port_chart import (
    FrozenContext,
    local_model_call,
    load_frozen_local,
    sha256,
)
from .train_residual_port_communicator import (
    load_frozen_chart,
)
from .validate_face_sni import face_sni_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sweeps", type=int, nargs="+", default=(1, 4, 8, 16, 128))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    return parser.parse_args()


def synchronize() -> None:
    torch.cuda.synchronize()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("matched timing requires CUDA")
    if min(args.sweeps) <= 0 or min(args.batch_size, args.warmup, args.repeats) <= 0:
        raise ValueError("timing extents must be positive")

    device = torch.device("cuda")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, _ = load_frozen_chart(args.chart_checkpoint, device)
    communicator, communicator_payload = _load_communicator(
        args.latent_checkpoint, spec, device,
    )
    for module in (local, chart, communicator):
        module.eval().requires_grad_(False)

    dataset = H5DarcyDataset(args.data, "validation")
    host = next(iter(DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )))
    batch = move_batch(host, device)
    cover = CartesianCover(
        batch["mask"].shape[-2], batch["mask"].shape[-1],
        args.core, args.halo,
    )
    zero = initial_state(batch, adapter.statistics, "zero")

    def target_free_context() -> FrozenContext:
        physical, scalar = tiled_sni_conditions(adapter, batch, zero, cover)
        endpoint = local_model_call(local, physical, scalar, cover)
        valid = cover.tile(batch["mask"].to(endpoint.dtype)) * cover.valid_tiles(
            batch["mask"].shape[0], endpoint.device, endpoint.dtype,
        )
        endpoint = endpoint * valid
        empty_global = torch.zeros_like(batch["mask"])
        empty_tiles = cover.tile(empty_global) * valid
        empty_code = torch.zeros(
            batch["mask"].shape[0], cover.tile_count, spec.code_width,
            device=device, dtype=endpoint.dtype,
        )
        return FrozenContext(
            endpoint, physical, scalar, valid, empty_global, empty_tiles,
            empty_code, cover,
        )

    def local_pou():
        return face_sni_step(
            local, adapter, batch, zero, cover, "weighted_pou", 1.0,
        )

    def latentddm():
        context = target_free_context()
        return _decomposed_fields(
            communicator, chart, context, batch, spec, scale,
        )["assembled"] * adapter.statistics.solution_std

    def pou_sni(sweeps: int):
        # The first call constructs the shared Local-POU initializer; each
        # requested sweep is one additional trace refresh and local solve.
        current = local_pou()
        for _ in range(sweeps):
            current = face_sni_step(
                local, adapter, batch, current, cover, "weighted_pou", 1.0,
            )
        return current

    methods = {"local_pou": local_pou, "latentddm": latentddm}
    methods.update({
        f"pou_sni_{sweep}": (lambda sweep=sweep: pou_sni(sweep))
        for sweep in sorted(set(args.sweeps))
    })

    method_names = list(methods)
    for _ in range(args.warmup):
        for name in method_names:
            methods[name]()
    samples = {name: [] for name in method_names}
    peaks = {name: [] for name in method_names}
    # Rotate and reverse the order so clock/thermal drift cannot systematically
    # favor Local POU, LatentDDM, or a particular SNI depth.
    for repeat in range(args.repeats):
        shift = repeat % len(method_names)
        order = method_names[shift:] + method_names[:shift]
        if repeat % 2:
            order = list(reversed(order))
        for name in order:
            synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            methods[name]()
            synchronize()
            samples[name].append(
                (time.perf_counter() - started) / args.batch_size
            )
            peaks[name].append(torch.cuda.max_memory_allocated() / 2**30)

    timings = {
        name: {
            "mean_seconds_per_case": statistics.mean(samples[name]),
            "median_seconds_per_case": statistics.median(samples[name]),
            "stdev_seconds_per_case": statistics.stdev(samples[name]),
            "minimum_seconds_per_case": min(samples[name]),
            "maximum_seconds_per_case": max(samples[name]),
            "peak_allocated_gib": max(peaks[name]),
        }
        for name in method_names
    }
    for name, timing in timings.items():
        sweep = int(name.rsplit("_", 1)[1]) if name.startswith("pou_sni_") else 0
        timing["parallel_local_rounds"] = 1 + sweep
        timing["within_round_execution"] = (
            "one measured batched GPU call over every local crop"
        )
    fields = {name: function() for name, function in methods.items()}
    audits = {
        name: {
            "relative_l2_first_batch_mean": float(relative_l2(
                field, batch["solution"], batch["mask"],
            ).mean()),
            "local_nfe": (
                cover.tile_count
                if name in ("local_pou", "latentddm")
                else (int(name.rsplit("_", 1)[1]) + 1) * cover.tile_count
            ),
        }
        for name, field in fields.items()
    }
    result = {
        "contract": "darcy_same_process_weighted_pou_sni_timing_v1",
        "split": "validation",
        "protected_test_cases_opened": 0,
        "target_used_by_timed_paths": False,
        "diagnostics_in_timed_paths": False,
        "device": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "tile_count": cover.tile_count,
        "tau": 1.0,
        "assembly": "cosine-weighted normalized POU",
        "local_checkpoint": str(args.local_checkpoint.resolve()),
        "local_checkpoint_sha256": sha256(args.local_checkpoint),
        "chart_checkpoint": str(args.chart_checkpoint.resolve()),
        "chart_checkpoint_sha256": sha256(args.chart_checkpoint),
        "latent_checkpoint": str(args.latent_checkpoint.resolve()),
        "latent_checkpoint_sha256": sha256(args.latent_checkpoint),
        "latent_checkpoint_frozen": bool(
            communicator_payload.get("communicator_spec", {}).get(
                "frozen_candidate", False,
            )
        ),
        "cost_definition": "POU-SNI-k includes (k+1)N local evaluations",
        "parallel_execution_contract": (
            "all N crops execute in one measured batched GPU call per round; "
            "SNI rounds remain sequential"
        ),
        "analytic_time_division_or_projection": False,
        "method_order": "rotated and reversed across repeats",
        "timings": timings,
        "field_audit_outside_timed_region": audits,
    }
    dataset.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
