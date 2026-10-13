"""Same-process timing for Darcy Local/Core, LatentDDM, and Core-SNI.

All timed methods start from one resident validation batch.  A local round is
one GPU-batched evaluation over all nine subdomains; no latency is divided by
the number of tiles.  Targets are used only for the post-timing accuracy audit.
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

from .adapter import H5DarcyDataset, enforce_domain_boundary, tiled_sni_conditions
from .evaluate import move_batch, relative_l2
from .field_token_composer import DarcyFieldTokenComposer
from .train_field_token_composer import CONTRACT
from .train_residual_port_chart import (
    FrozenContext,
    local_model_call,
    load_frozen_local,
    sha256,
)
from .train_residual_port_communicator import load_frozen_chart
from .validate_face_sni import face_sni_step


Tensor = torch.Tensor


def physical_zero(batch: dict[str, Tensor]) -> Tensor:
    return enforce_domain_boundary(batch, torch.zeros_like(batch["mask"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--composer-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sni-sweeps", type=int, nargs="+", default=(1, 4, 8))
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    return parser.parse_args()


def target_free_context(
    local: torch.nn.Module,
    adapter,
    batch: dict[str, Tensor],
    cover: CartesianCover,
    code_width: int,
) -> FrozenContext:
    zero = physical_zero(batch)
    physical, scalar = tiled_sni_conditions(adapter, batch, zero, cover)
    endpoint = local_model_call(local, physical, scalar, cover)
    valid = cover.tile(batch["mask"].to(endpoint.dtype)) * cover.valid_tiles(
        batch["mask"].shape[0], endpoint.device, endpoint.dtype,
    )
    endpoint = endpoint * valid
    dummy_global = torch.zeros_like(batch["mask"])
    dummy_tiles = cover.tile(dummy_global) * valid
    dummy_code = torch.zeros(
        batch["mask"].shape[0], cover.tile_count, code_width,
        device=endpoint.device, dtype=endpoint.dtype,
    )
    return FrozenContext(
        endpoint, physical, scalar, valid,
        dummy_global, dummy_tiles, dummy_code, cover,
    )


def synchronize() -> None:
    torch.cuda.synchronize()


@torch.inference_mode()
def main(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("matched timing requires CUDA")
    if (
        min(args.sni_sweeps) <= 0
        or min(args.batch_size, args.warmups, args.repeats, args.core) <= 0
        or args.halo < 0 or not 0.0 < args.tau <= 1.0
    ):
        raise ValueError("invalid matched timing configuration")

    device = torch.device("cuda")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, _, chart_payload = load_frozen_chart(
        args.chart_checkpoint, device,
    )
    saved = torch.load(
        args.composer_checkpoint, map_location="cpu", weights_only=False,
    )
    if saved.get("contract") != CONTRACT:
        raise ValueError("wrong Darcy field-token checkpoint contract")
    if saved.get("source_local_sha256") != sha256(args.local_checkpoint):
        raise ValueError("field-token checkpoint and local endpoint differ")
    if saved.get("source_chart_sha256") != sha256(args.chart_checkpoint):
        raise ValueError("field-token checkpoint and response chart differ")
    if chart_payload.get("source_local_sha256") != sha256(args.local_checkpoint):
        raise ValueError("response chart and local endpoint differ")
    configuration = saved["composer_spec"]
    composer = DarcyFieldTokenComposer(
        int(configuration["physical_channels"]),
        int(configuration["scalar_dim"]),
        int(configuration["code_width"]),
        token_width=int(configuration["token_width"]),
        heads=int(configuration["heads"]),
        communication_depth=int(configuration["communication_depth"]),
        spatial_port_bins=int(configuration.get("spatial_port_bins", 0)),
    ).to(device)
    composer.load_state_dict(saved["composer"], strict=True)
    for module in (local, chart, composer):
        module.eval().requires_grad_(False)

    dataset = H5DarcyDataset(args.data, "validation")
    try:
        host = next(iter(DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        )))
        batch = move_batch(host, device)
        cover = CartesianCover(
            batch["mask"].shape[-2], batch["mask"].shape[-1],
            args.core, args.halo,
        )
        zero = physical_zero(batch)

        def local_core() -> Tensor:
            context = target_free_context(
                local, adapter, batch, cover, spec.code_width,
            )
            normalized = cover.assemble_ras(
                context.local, batch["mask"].shape[0],
            ) * batch["mask"]
            return normalized * adapter.statistics.solution_std

        def latentddm() -> Tensor:
            context = target_free_context(
                local, adapter, batch, cover, spec.code_width,
            )
            return composer(chart, context, batch["mask"]).assembled \
                * adapter.statistics.solution_std

        def core_sni(sweeps: int) -> Tensor:
            current = local_core()
            for _ in range(sweeps):
                current = face_sni_step(
                    local, adapter, batch, current, cover,
                    "core_ras", args.tau,
                )
            return current

        methods = {"local_core": local_core, "latentddm": latentddm}
        methods.update({
            f"init_sni_{sweep}": (
                lambda sweep=sweep: core_sni(sweep)
            )
            for sweep in sorted(set(args.sni_sweeps))
        })
        names = list(methods)
        for _ in range(args.warmups):
            for name in names:
                methods[name]()
        samples = {name: [] for name in names}
        peaks = {name: [] for name in names}
        for repeat in range(args.repeats):
            shift = repeat % len(names)
            order = names[shift:] + names[:shift]
            if repeat % 2:
                order = list(reversed(order))
            for name in order:
                synchronize(); torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                methods[name]()
                synchronize()
                samples[name].append(
                    (time.perf_counter() - started) / args.batch_size
                )
                peaks[name].append(
                    torch.cuda.max_memory_allocated(device) / 2**30
                )
        timing = {
            name: {
                "mean_seconds_per_case": statistics.mean(values),
                "median_seconds_per_case": statistics.median(values),
                "sample_std_seconds_per_case": statistics.stdev(values),
                "minimum_seconds_per_case": min(values),
                "maximum_seconds_per_case": max(values),
                "peak_allocated_gib": max(peaks[name]),
                "sequential_local_rounds": (
                    1 + int(name.rsplit("_", 1)[1])
                    if name.startswith("init_sni_") else 1
                ),
            }
            for name, values in samples.items()
        }
        fields = {name: function() for name, function in methods.items()}
        audit = {
            name: {
                "relative_l2_first_batch_mean": float(relative_l2(
                    field, batch["solution"], batch["mask"],
                ).mean()),
                "local_field_evaluations": (
                    cover.tile_count * (1 + int(name.rsplit("_", 1)[1]))
                    if name.startswith("init_sni_") else cover.tile_count
                ),
            }
            for name, field in fields.items()
        }
        result = {
            "contract": "darcy_field_token_core_sni_same_process_timing_v1",
            "split": "validation",
            "protected_test_cases_opened": 0,
            "target_used_by_timed_paths": False,
            "diagnostics_in_timed_paths": False,
            "device": torch.cuda.get_device_name(),
            "batch_size": args.batch_size,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "tile_count": cover.tile_count,
            "tau": args.tau,
            "assembly": "disjoint_owned_core",
            "analytic_time_division_or_projection": False,
            "within_round_execution": (
                "one measured GPU-batched call over every local crop"
            ),
            "composer_checkpoint": str(args.composer_checkpoint.resolve()),
            "composer_checkpoint_sha256": sha256(args.composer_checkpoint),
            "communication_depth": int(configuration["communication_depth"]),
            "timing": timing,
            "field_audit_outside_timed_region": audit,
        }
    finally:
        dataset.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
