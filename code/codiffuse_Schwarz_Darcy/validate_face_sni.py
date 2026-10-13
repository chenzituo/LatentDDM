"""Select exact-Huang or normalized-POU neural-SNI validation controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import (
    DarcyStatistics, FaceDarcyAdapter, H5DarcyDataset, MODEL_SPEC, build_model,
    enforce_domain_boundary, tiled_sni_conditions,
)
from .evaluate import (
    initial_state, move_batch, relative_l2, residual_rms,
)
from .train_face_local import CONTRACT


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--assembly", choices=(
            "huang_raw_additive", "normalized_pou", "weighted_pou",
            "core_ras",
        ),
        required=True,
    )
    parser.add_argument("--taus", type=float, nargs="+", default=(0.2, 0.4, 0.6, 0.8, 1.0))
    parser.add_argument(
        "--residual-tolerances", type=float, nargs="+",
        default=(0.5, 0.2, 0.1, 0.05),
    )
    parser.add_argument("--max-sweeps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-validation-cases", type=int)
    parser.add_argument("--offset", default="0,0")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _offset(value: str) -> tuple[int, int]:
    pieces = tuple(int(piece) for piece in value.split(","))
    if len(pieces) != 2 or min(pieces) < 0 or max(pieces) >= MODEL_SPEC["core"]:
        raise ValueError("offset must be y,x inside the local core")
    return pieces


def validate_controls(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_sweeps <= 0 or not args.taus or not args.residual_tolerances:
        raise ValueError("positive sweeps and nonempty control grids are required")
    if any(not 0.0 < value <= 1.5 for value in args.taus):
        raise ValueError("all SNI relaxation values must lie in (0,1.5]")
    if any(value <= 0.0 for value in args.residual_tolerances):
        raise ValueError("residual stopping tolerances must be positive")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != CONTRACT:
        raise ValueError("validation control selection requires the frozen face solver")
    statistics = DarcyStatistics.from_json(checkpoint["statistics"])
    adapter = FaceDarcyAdapter(statistics)
    model = build_model(adapter, communication_stage="none").to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    # This script has no split argument by design: it cannot open test or A-D.
    dataset = H5DarcyDataset(args.data, "validation")
    if not dataset.has_face_dirichlet:
        raise ValueError("SNI validation domains require face-resolved boundary fields")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    offset_y, offset_x = _offset(args.offset)
    rows: list[dict[str, float | int]] = []
    for tau in args.taus:
        processed_for_tau = 0
        for host_batch in loader:
            if args.max_validation_cases is not None \
                    and processed_for_tau >= args.max_validation_cases:
                break
            batch = move_batch(host_batch, device)
            if args.max_validation_cases is not None:
                keep = min(
                    batch["solution"].shape[0],
                    args.max_validation_cases - processed_for_tau,
                )
                if keep <= 0:
                    break
                for key, value in list(batch.items()):
                    if isinstance(value, Tensor) and value.shape[0] == batch["solution"].shape[0]:
                        batch[key] = value[:keep]
            target = batch["solution"]
            cover = CartesianCover(
                target.shape[-2], target.shape[-1],
                MODEL_SPEC["core"], MODEL_SPEC["halo"], offset_y, offset_x,
            )
            if args.assembly == "huang_raw_additive" \
                    and not tau < 1.0 / cover.tile_count:
                raise ValueError(
                    "literal Huang raw-additive tau must satisfy 0 < tau < 1/K; "
                    f"got tau={tau}, K={cover.tile_count}"
                )
            current = initial_state(batch, statistics, "zero")
            previous = current.clone()
            cumulative = 0.0
            for sweep in range(1, args.max_sweeps + 1):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                next_state = face_sni_step(
                    model, adapter, batch, current, cover, args.assembly, tau,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                cumulative += (time.perf_counter() - started) / target.shape[0]
                l2 = relative_l2(next_state, target, batch["mask"])
                residual = residual_rms(next_state, batch)
                for item in range(target.shape[0]):
                    rows.append({
                        "tau": float(tau),
                        "case_index": int(batch["case_index"][item]),
                        "sweep": sweep,
                        "relative_l2": float(l2[item]),
                        "residual_rms": float(residual[item]),
                        "local_nfe": sweep * cover.tile_count,
                        "batched_local_calls": sweep,
                        "wall_seconds": cumulative,
                    })
                previous, current = current, next_state
            processed_for_tau += target.shape[0]

    candidates: list[dict[str, float]] = []
    for tau in args.taus:
        tau_rows = [row for row in rows if float(row["tau"]) == float(tau)]
        cases = sorted({int(row["case_index"]) for row in tau_rows})
        if not cases:
            raise RuntimeError(f"no validation cases were evaluated for tau={tau}")
        for tolerance in args.residual_tolerances:
            stopped = []
            for case in cases:
                trajectory = [
                    row for row in tau_rows if int(row["case_index"]) == case
                ]
                chosen = next(
                    (row for row in trajectory if float(row["residual_rms"]) <= tolerance),
                    trajectory[-1],
                )
                stopped.append(chosen)
            mean_l2 = float(np.mean([float(row["relative_l2"]) for row in stopped]))
            mean_residual = float(np.mean([float(row["residual_rms"]) for row in stopped]))
            mean_nfe = float(np.mean([float(row["local_nfe"]) for row in stopped]))
            candidates.append({
                "tau": float(tau),
                "residual_tolerance": float(tolerance),
                "mean_relative_l2": mean_l2,
                "mean_residual_rms": mean_residual,
                "mean_local_nfe": mean_nfe,
                "selection_score": mean_l2 + 0.1 * mean_residual + 1.0e-5 * mean_nfe,
            })
    selected = min(candidates, key=lambda row: row["selection_score"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "validation_trajectories.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = {
        "contract": (
            "darcy_huang_raw_additive_sni_validation_v1"
            if args.assembly == "huang_raw_additive"
            else (
                "darcy_weighted_pou_sni_validation_v1"
                if args.assembly == "weighted_pou"
                else (
                    "darcy_core_ras_sni_validation_v1"
                    if args.assembly == "core_ras"
                    else "darcy_normalized_pou_sni_validation_v1"
                )
            )
        ),
        "assembly": args.assembly,
        "exact_huang_equation_2": args.assembly == "huang_raw_additive",
        "overlap_normalized": args.assembly in ("normalized_pou", "weighted_pou"),
        "same_cosine_pou_as_local_endpoint": args.assembly == "weighted_pou",
        "selection_split": "validation",
        "test_subsets_opened": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "data": str(args.data.resolve()),
        "candidate_controls": candidates,
        "selected_control": selected,
        "max_sweeps": args.max_sweeps,
        "local_nfe_definition": "sweeps times overlapping tile count per case",
        "batched_local_call_definition": "one frozen local-network call per sweep",
    }
    (args.output_dir / "selected_control.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


@torch.no_grad()
def face_sni_step(
    model: torch.nn.Module,
    adapter: FaceDarcyAdapter,
    batch: dict[str, Tensor],
    current: Tensor,
    cover: CartesianCover,
    assembly: str,
    tau: float,
) -> Tensor:
    """Apply one audited recurrence with no ambiguity in overlap assembly."""
    physical, scalar = tiled_sni_conditions(adapter, batch, current, cover)
    local_normalized = model(
        physical, scalar, cover.tile_count,
        tile_grid_shape=cover.grid_shape,
    )
    local_solution = adapter.statistics.solution_std * local_normalized
    active = cover.tile(batch["mask"].to(local_solution.dtype))
    if assembly == "huang_raw_additive":
        # Huang Eq. (2): u+tau sum_i R_i^T(D_i(P_i u)-R_i u).
        correction = local_solution - cover.tile(current)
        update = cover.assemble_extension_sum(
            correction, current.shape[0], active,
        )
        candidate = current + tau * update
    elif assembly == "normalized_pou":
        patched = cover.assemble_uniform(
            local_solution, current.shape[0], active,
        )
        candidate = current + tau * (patched - current)
    elif assembly == "weighted_pou":
        # Controlled paper recurrence: use the exact cosine POU that defines
        # the frozen Local-POU endpoint.  Initialization is handled by the
        # caller so SNI-0 can be audited against that endpoint bitwise.
        patched = cover.assemble_weighted(local_solution, current.shape[0])
        candidate = current + tau * (patched - current)
    elif assembly == "core_ras":
        # Matched no-averaging recurrence used by the field-token comparison.
        # The halo remains an input trace, but every output cell has one owner.
        patched = cover.assemble_ras(local_solution, current.shape[0])
        candidate = current + tau * (patched - current)
    else:
        raise ValueError(f"unknown SNI overlap assembly: {assembly}")
    return enforce_domain_boundary(batch, candidate)


if __name__ == "__main__":
    print(json.dumps(validate_controls(parse_args()), indent=2))
