"""Validation-only exact mixed-Robin OSM frontier and control selection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import h5py
import numpy as np

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .physics import (
    assemble_darcy_system, relative_energy_error, relative_l2,
    relative_residual,
)
from .robin_osm import exact_robin_osm_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-cases", type=int, default=8)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument(
        "--impedance-ratios", type=float, nargs="+",
        default=(0.2, 0.3, 0.5, 1.0, 3.0),
    )
    parser.add_argument("--relaxation", type=float, default=0.8)
    parser.add_argument(
        "--checkpoints", type=int, nargs="+",
        default=(1, 2, 4, 8, 16, 32, 64),
    )
    return parser.parse_args()


def validate(args: argparse.Namespace) -> dict[str, object]:
    if args.maximum_cases <= 0 or min(args.checkpoints) <= 0:
        raise ValueError("case count and checkpoints must be positive")
    if sorted(set(args.checkpoints)) != list(args.checkpoints):
        raise ValueError("checkpoints must be strictly increasing and unique")
    if any(value <= 0.0 for value in args.impedance_ratios):
        raise ValueError("impedance ratios must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with h5py.File(args.data, "r") as handle:
        if "val" not in handle:
            raise ValueError("Robin OSM selection requires the validation group")
        group = handle["val"]
        count = min(args.maximum_cases, int(group["solution"].shape[0]))
        for impedance_ratio in args.impedance_ratios:
            for case in range(count):
                mask = group["mask"][case, 0] > 0
                kappa = 10.0 ** group["logk"][case, 0].astype(np.float64)
                source = group["source"][case, 0].astype(np.float64)
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
                started = time.perf_counter()
                failed = False
                update = float("nan")
                for sweep in range(1, max(args.checkpoints) + 1):
                    if not failed:
                        report = exact_robin_osm_sweep(
                            mask, kappa, source, face_mask, face_value,
                            current, cover, impedance_ratio=impedance_ratio,
                            relaxation=args.relaxation,
                        )
                        current = report.field
                        update = report.relative_update
                        scale = max(np.linalg.norm(target[mask]), 1.0e-12)
                        failed = (
                            not np.isfinite(current[mask]).all()
                            or np.linalg.norm(current[mask]) > 1.0e6 * scale
                        )
                    if sweep not in args.checkpoints:
                        continue
                    elapsed = time.perf_counter() - started
                    row = {
                        "impedance_ratio": impedance_ratio,
                        "relaxation": args.relaxation,
                        "case_index": case,
                        "sweep": sweep,
                        "failed": failed,
                        "relative_update": update,
                        "relative_l2": float("inf") if failed else relative_l2(
                            current, target, mask,
                        ),
                        "energy_relative_error": float("inf") if failed else
                            relative_energy_error(current, target, system),
                        "residual_rms": float("inf") if failed else
                            relative_residual(current, system),
                        "local_solves": sweep * cover.tile_count,
                        "batched_local_calls": sweep,
                        "wall_seconds": elapsed,
                    }
                    rows.append(row)
    with (args.output_dir / "frontier.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    aggregated: dict[str, dict[str, float]] = {}
    for ratio in args.impedance_ratios:
        for checkpoint in args.checkpoints:
            selected = [
                row for row in rows
                if row["impedance_ratio"] == ratio and row["sweep"] == checkpoint
            ]
            key = f"p_ratio_{ratio:.4f}/checkpoint_{checkpoint:04d}"
            aggregated[key] = {
                name: float(np.mean([float(row[name]) for row in selected]))
                for name in (
                    "relative_l2", "energy_relative_error", "residual_rms",
                    "local_solves", "batched_local_calls", "wall_seconds",
                )
            }
            aggregated[key]["failed_cases"] = float(sum(
                bool(row["failed"]) for row in selected
            ))
    final_checkpoint = max(args.checkpoints)
    finite = [
        (ratio, aggregated[f"p_ratio_{ratio:.4f}/checkpoint_{final_checkpoint:04d}"])
        for ratio in args.impedance_ratios
        if aggregated[f"p_ratio_{ratio:.4f}/checkpoint_{final_checkpoint:04d}"]["failed_cases"] == 0
    ]
    if not finite:
        raise RuntimeError("all Robin controls diverged")
    selected_ratio, selected_metrics = min(
        finite,
        key=lambda item: (
            item[1]["relative_l2"] + 0.1 * item[1]["energy_relative_error"]
            + 0.1 * item[1]["residual_rms"]
        ),
    )
    result = {
        "contract": "darcy_exact_consensus_robin_osm_validation_v1",
        "baseline_label": "Mao-style Robin OSM (reimplementation)",
        "selection_split": "validation",
        "test_subsets_opened": False,
        "local_solver": "exact sparse mixed-Robin FV",
        "exchange": "previous normalized-POU consensus",
        "cases": count,
        "tile_count": cover.tile_count,
        "impedance_ratios": list(args.impedance_ratios),
        "relaxation": args.relaxation,
        "checkpoints": list(args.checkpoints),
        "summary": aggregated,
        "selected_control": {
            "impedance_ratio": selected_ratio,
            "checkpoint": final_checkpoint,
            **selected_metrics,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), indent=2))
