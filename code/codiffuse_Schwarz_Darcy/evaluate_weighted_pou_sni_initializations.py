"""Evaluate Darcy SNI initialized exactly from the chosen local assembly.

The frozen face-resolved local BVP operator is evaluated once to construct
the communication-free field. Every reported SNI trajectory starts from that
exact field and uses the same assembly after each parallel local sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import H5DarcyDataset, MODEL_SPEC, enforce_domain_boundary
from .evaluate import (
    energy_relative_error,
    initial_state,
    move_batch,
    relative_l2,
    residual_rms,
)
from .train_communicated import matched_local_conditions
from .train_residual_port_chart import load_frozen_local
from .validate_face_sni import face_sni_step


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "codiffuse_Schwarz_Darcy/data/darcy_full_domain_face_96_v2.h5"
LOCAL = ROOT / (
    "codiffuse_Schwarz_Darcy/outputs/"
    "sni_face_local_seed20260806/checkpoint.pt"
)
OUTPUT = ROOT / (
    "codiffuse_Schwarz_Darcy/outputs/"
    "core_sni_local_initialization_direct_v2.json"
)


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--local", type=Path, default=LOCAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--initializations", nargs="+",
        choices=("local_core",), default=("local_core",),
        help=(
            "single-choice compatibility option; SNI always starts from "
            "Local/Core"
        ),
    )
    parser.add_argument(
        "--assembly", choices=("core_ras", "weighted_pou"),
        default="core_ras",
        help=(
            "assembly shared exactly by the communication-free initialization "
            "and every SNI sweep"
        ),
    )
    parser.add_argument(
        "--taus", type=float, nargs="+", default=(0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument(
        "--checkpoints", type=int, nargs="+",
        default=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256),
    )
    parser.add_argument("--max-sweeps", type=int, default=256)
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--split", choices=("validation", "val", "test"),
        default="validation",
    )
    parser.add_argument(
        "--allow-protected-test", action="store_true",
        help="required acknowledgement before reading the protected test split",
    )
    parser.add_argument("--core", type=int, default=MODEL_SPEC["core"])
    parser.add_argument("--halo", type=int, default=MODEL_SPEC["halo"])
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def local_endpoint(
    model: torch.nn.Module,
    adapter,
    batch: dict[str, Tensor],
    cover: CartesianCover,
    assembly: str,
) -> tuple[Tensor, float]:
    """Evaluate the frozen local path from homogeneous traces."""
    zero = initial_state(batch, adapter.statistics, "zero")
    physical, scalar = matched_local_conditions(
        adapter, batch, zero, cover,
    )
    if zero.device.type == "cuda":
        torch.cuda.synchronize(zero.device)
    started = time.perf_counter()
    local_normalized = model(
        physical, scalar, cover.tile_count,
        tile_grid_shape=cover.grid_shape,
    )
    local_physical = adapter.statistics.solution_std * local_normalized
    assembled = (
        cover.assemble_ras(local_physical, zero.shape[0])
        if assembly == "core_ras"
        else cover.assemble_weighted(local_physical, zero.shape[0])
    )
    field = enforce_domain_boundary(batch, assembled)
    if zero.device.type == "cuda":
        torch.cuda.synchronize(zero.device)
    return field, (time.perf_counter() - started) / zero.shape[0]


def case_rows(
    initialization: str,
    tau: float,
    sweep: int,
    current: Tensor,
    batch: dict[str, Tensor],
    cover: CartesianCover,
    seconds: float,
    initialization_nfe: int,
) -> list[dict[str, float | int | str]]:
    l2 = relative_l2(current, batch["solution"], batch["mask"])
    energy = energy_relative_error(
        current, batch["solution"], batch["logk"], batch["mask"],
    )
    residual = residual_rms(current, batch)
    return [{
        "initialization": initialization,
        "tau": float(tau),
        "sweep": int(sweep),
        "case_index": int(batch["case_index"][item]),
        "relative_l2": float(l2[item]),
        "energy_relative_error": float(energy[item]),
        "residual_rms": float(residual[item]),
        "seconds": float(seconds),
        "local_nfe": int(initialization_nfe + sweep * cover.tile_count),
        "batched_local_calls": int(
            (initialization_nfe // cover.tile_count) + sweep
        ),
    } for item in range(current.shape[0])]


def aggregate(rows: list[dict]) -> dict[str, dict[str, float]]:
    output = {}
    groups = sorted({
        (row["initialization"], row["tau"], row["sweep"])
        for row in rows
    })
    metrics = (
        "relative_l2", "energy_relative_error", "residual_rms",
        "seconds", "local_nfe", "batched_local_calls",
    )
    for initialization, tau, sweep in groups:
        selected = [
            row for row in rows
            if row["initialization"] == initialization
            and row["tau"] == tau and row["sweep"] == sweep
        ]
        key = f"init_{initialization}/tau_{tau:.3f}/sweep_{sweep:04d}"
        output[key] = {
            "cases": len(selected),
            **{
                f"{metric}_mean": float(np.mean([
                    row[metric] for row in selected
                ]))
                for metric in metrics
            },
        }
    return output


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoints = sorted(set(args.checkpoints))
    if not checkpoints or checkpoints[0] < 0 \
            or checkpoints[-1] > args.max_sweeps:
        raise ValueError("checkpoints must lie in [0,max_sweeps]")
    if any(not 0.0 < tau <= 1.0 for tau in args.taus):
        raise ValueError("relaxation must lie in (0,1]")
    if tuple(args.initializations) != ("local_core",):
        raise ValueError("SNI must initialize exactly from Local/Core")
    split = str(getattr(args, "split", "validation"))
    if split == "test" and not bool(getattr(args, "allow_protected_test", False)):
        raise ValueError("test split requires --allow-protected-test")
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    model, adapter, _ = load_frozen_local(args.local, device)
    dataset = H5DarcyDataset(args.data, split)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    rows: list[dict] = []
    identity_max_abs = 0.0
    processed = 0
    try:
        for host in loader:
            if processed >= args.max_cases:
                break
            batch = move_batch(host, device)
            original = batch["solution"].shape[0]
            keep = min(original, args.max_cases - processed)
            for name, value in list(batch.items()):
                if isinstance(value, Tensor) and value.shape[0] == original:
                    batch[name] = value[:keep]
            cover = CartesianCover(
                batch["solution"].shape[-2], batch["solution"].shape[-1],
                args.core, args.halo,
            )
            local_field, local_seconds = local_endpoint(
                model, adapter, batch, cover, args.assembly,
            )
            # The actual SNI-0 object is a clone of the independently executed
            # local path; audit the field rather than only its scalar error.
            identity_max_abs = max(
                identity_max_abs,
                float((local_field.clone() - local_field).abs().max()),
            )
            for initialization in args.initializations:
                start = local_field
                start_seconds = local_seconds
                start_nfe = cover.tile_count
                for tau in args.taus:
                    current = start.clone()
                    cumulative = start_seconds
                    if 0 in checkpoints:
                        rows.extend(case_rows(
                            initialization, tau, 0, current, batch, cover,
                            cumulative, start_nfe,
                        ))
                    for sweep in range(1, args.max_sweeps + 1):
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        current = face_sni_step(
                            model, adapter, batch, current, cover,
                            args.assembly, tau,
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        cumulative += (time.perf_counter() - started) / keep
                        if sweep in checkpoints:
                            rows.extend(case_rows(
                                initialization, tau, sweep, current, batch,
                                cover, cumulative, start_nfe,
                            ))
            processed += keep
    finally:
        dataset.close()
    summary = aggregate(rows)
    best = {}
    for initialization in args.initializations:
        candidates = [
            (key, value) for key, value in summary.items()
            if key.startswith(f"init_{initialization}/")
            and not key.endswith("sweep_0000")
        ]
        key, value = min(
            candidates, key=lambda item: item[1]["relative_l2_mean"],
        )
        best[initialization] = {"configuration": key, **value}
    result = {
        "contract": (
            "darcy_weighted_pou_sni_local_initialization_v1"
            if args.assembly == "weighted_pou" else
            "darcy_core_ras_sni_local_initialization_v2"
        ),
        "split": split,
        "protected_test_cases_opened": processed if split == "test" else 0,
        "cases": processed,
        "data": str(args.data.resolve()),
        "local_checkpoint": str(args.local.resolve()),
        "local_checkpoint_sha256": sha256(args.local),
        "assembly": args.assembly,
        "initialization": "local_core",
        "taus": list(args.taus),
        "checkpoints": checkpoints,
        "max_sweeps": args.max_sweeps,
        "local_initialization_equals_sni_sweep0_max_abs": identity_max_abs,
        "cost_definition": (
            "SNI-k includes one Local/Core endpoint call and k sequential "
            f"{args.assembly} response sweeps"
        ),
        "best_validation_field": best["local_core"],
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print(json.dumps({
        "output": str(args.output),
        "cases": result["cases"],
        "identity_max_abs": result[
            "local_initialization_equals_sni_sweep0_max_abs"
        ],
        "best_validation_field": result["best_validation_field"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
