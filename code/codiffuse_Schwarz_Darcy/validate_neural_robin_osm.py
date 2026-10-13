"""Validation-only neural Robin recurrence and matched iterative frontier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .physics import (
    assemble_darcy_system, relative_energy_error, relative_l2,
    relative_residual,
)
from .robin_osm import load_neural_robin_local, neural_robin_osm_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--huang-frontier", type=Path, required=True)
    parser.add_argument("--exact-robin-summary", type=Path, required=True)
    parser.add_argument("--maximum-cases", type=int, default=64)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--impedance-ratio", type=float, default=0.5)
    parser.add_argument("--relaxation", type=float, default=0.8)
    parser.add_argument(
        "--checkpoints", type=int, nargs="+",
        default=(1, 2, 4, 8, 16, 32, 64),
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    output = {}
    for sweep in sorted({int(row["sweep"]) for row in rows}):
        selected = [row for row in rows if int(row["sweep"]) == sweep]
        output[f"checkpoint_{sweep:04d}"] = {
            name: float(np.mean([float(row[name]) for row in selected]))
            for name in (
                "relative_l2", "energy_relative_error", "residual_rms",
                "local_nfe", "batched_local_calls", "wall_seconds",
            )
        }
        output[f"checkpoint_{sweep:04d}"]["failed_cases"] = float(sum(
            bool(row["failed"]) for row in selected
        ))
    return output


def _matched_huang(path: Path, cases: int, checkpoints: list[int]) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if (
            abs(float(row["tau"]) - 0.1) < 1.0e-10
            and int(row["case_index"]) < cases
            and int(row["requested_checkpoint"]) in checkpoints
        )]
    mapped = [{
        "sweep": int(row["requested_checkpoint"]), "failed": False,
        **{name: float(row[name]) for name in (
            "relative_l2", "energy_relative_error", "residual_rms",
            "local_nfe", "batched_local_calls", "wall_seconds",
        )},
    } for row in rows]
    return _aggregate(mapped)


def validate(args: argparse.Namespace) -> dict[str, object]:
    if args.maximum_cases <= 0 or sorted(set(args.checkpoints)) != list(args.checkpoints):
        raise ValueError("invalid case count or checkpoint sequence")
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda",
    )
    model, adapter, checkpoint = load_neural_robin_local(args.checkpoint, device)
    rows: list[dict[str, object]] = []
    with h5py.File(args.data, "r") as handle:
        group = handle["val"]
        cases = min(args.maximum_cases, int(group["solution"].shape[0]))
        for case in range(cases):
            mask = group["mask"][case, 0] > 0
            kappa = 10.0 ** group["logk"][case, 0].astype(np.float64)
            source = group["source"][case, 0].astype(np.float64)
            sdf = group["sdf"][case, 0].astype(np.float64)
            normals = group["normals"][case].astype(np.float64)
            face_mask = group["dirichlet_face_mask"][case] > 0
            face_value = group["dirichlet_face_value"][case].astype(np.float64)
            target = group["solution"][case, 0].astype(np.float64)
            cover = CartesianCover(
                mask.shape[0], mask.shape[1], args.core, args.halo,
            )
            system = assemble_darcy_system(
                mask, kappa, source, np.zeros_like(source),
                dirichlet_face_mask=face_mask,
                dirichlet_face_value=face_value,
            )
            current = np.zeros_like(target)
            elapsed = 0.0
            failed = False
            update = float("nan")
            for sweep in range(1, max(args.checkpoints) + 1):
                if device.type == "cuda": torch.cuda.synchronize()
                started = time.perf_counter()
                report = neural_robin_osm_sweep(
                    model, adapter, mask, kappa, source, sdf, normals,
                    face_mask, face_value, current, cover,
                    impedance_ratio=args.impedance_ratio,
                    relaxation=args.relaxation, device=device,
                )
                if device.type == "cuda": torch.cuda.synchronize()
                elapsed += time.perf_counter() - started
                current = report.field; update = report.relative_update
                scale = max(np.linalg.norm(target[mask]), 1.0e-12)
                failed = (
                    not np.isfinite(current[mask]).all()
                    or np.linalg.norm(current[mask]) > 1.0e6 * scale
                )
                if sweep not in args.checkpoints:
                    continue
                rows.append({
                    "case_index": case, "sweep": sweep, "failed": failed,
                    "relative_update": update,
                    "relative_l2": float("inf") if failed else
                        relative_l2(current, target, mask),
                    "energy_relative_error": float("inf") if failed else
                        relative_energy_error(current, target, system),
                    "residual_rms": float("inf") if failed else
                        relative_residual(current, system),
                    "local_nfe": sweep * cover.tile_count,
                    "batched_local_calls": sweep,
                    "wall_seconds": elapsed,
                })
                if failed:
                    break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "frontier.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    neural = _aggregate(rows)
    exact = json.loads(args.exact_robin_summary.read_text())["summary"]
    exact = {
        key.split("/")[-1]: value for key, value in exact.items()
        if key.startswith("p_ratio_0.5000/")
    }
    huang = _matched_huang(
        args.huang_frontier, cases, list(args.checkpoints),
    )
    result = {
        "contract": "darcy_neural_robin_osm_matched_frontier_v1",
        "selection_split": "validation",
        "test_subsets_opened": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "cases": cases,
        "impedance_ratio": args.impedance_ratio,
        "relaxation": args.relaxation,
        "local_nfe_definition": "actual sweeps times nine overlapping tiles",
        "batched_calls_definition": "one all-tile neural call per sweep",
        "timing_excludes_physics_diagnostics": True,
        "frontiers": {
            "neural_robin_osm": neural,
            "exact_robin_osm_selected_64_case": exact,
            "literal_huang_raw_additive_matched_first_cases": huang,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), indent=2))
