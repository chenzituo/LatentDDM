"""Target-free spectrum audit for Darcy trace versus trace+flux ports.

The response chart and local endpoint are frozen.  For each validation problem
this script forms the exact linearized reduced map

    J_X = d q_X(v + L_X(z)) / dz,

without using the reference solution, then reports the supported singular
spectrum and fixed-ridge residual contraction.  It is a theorem diagnostic,
not an inference method or a checkpoint-selection stage.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .adapter import H5DarcyDataset
from .evaluate import move_batch
from .latent_schur_teacher import inference_only_context
from .train import make_loader
from .train_residual_port_chart import frozen_context, load_frozen_local, sha256
from .train_residual_port_communicator import load_frozen_chart
from .validate_latent_schur import linearized_q


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-case", type=int, default=4)
    parser.add_argument("--maximum-cases", type=int, default=60)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--relative-rank-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def _one_spectrum(jacobian: torch.Tensor, q: torch.Tensor, ridge: float,
                  relative_rank_tolerance: float) -> dict[str, float]:
    j = jacobian.double()
    q = q.double()
    singular = torch.linalg.svdvals(j)
    maximum = singular[0].clamp_min(1.0e-30)
    supported = singular > relative_rank_tolerance * maximum
    rank = int(supported.sum())
    if rank == 0:
        raise RuntimeError("reduced response has no supported singular direction")
    minimum = singular[supported][-1]
    identity = torch.eye(j.shape[1], device=j.device, dtype=j.dtype)
    direction = torch.linalg.solve(
        j.T @ j + ridge * identity, -(j.T @ q),
    )
    residual_ratio = torch.linalg.vector_norm(q + j @ direction) \
        / torch.linalg.vector_norm(q).clamp_min(1.0e-30)
    return {
        "active_rows": float(j.shape[0]),
        "code_columns": float(j.shape[1]),
        "supported_rank": float(rank),
        "maximum_singular_value": float(maximum),
        "minimum_supported_singular_value": float(minimum),
        "supported_condition_number": float(maximum / minimum),
        "stable_rank": float(singular.square().sum() / maximum.square()),
        "ridge_linear_residual_ratio": float(residual_ratio),
    }


@torch.no_grad()
def analyze(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.start_case < 0 or args.maximum_cases <= 0 or args.ridge <= 0.0 \
            or not 0.0 < args.relative_rank_tolerance < 1.0:
        raise ValueError("invalid spectrum-audit arguments")
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda",
    )
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, chart_payload = load_frozen_chart(
        args.chart_checkpoint, device,
    )
    if chart_payload["source_local_sha256"] != sha256(args.local_checkpoint):
        raise ValueError("chart and local endpoint lineage differ")
    dataset = H5DarcyDataset(args.data, "validation")
    loader = make_loader(dataset, 1, False, args.workers, device)
    rows: list[dict[str, float | int | str]] = []
    seen = 0
    try:
        for host in loader:
            if seen < args.start_case:
                seen += 1
                continue
            if len(rows) // 2 >= args.maximum_cases:
                break
            batch = move_batch(host, device)
            # The label tensor is replaced before context construction.  Its
            # shape remains available, but no solution value enters v, q, or J.
            inference_batch = dict(batch)
            inference_batch["solution"] = torch.zeros_like(batch["solution"])
            context = inference_only_context(frozen_context(
                local, adapter, inference_batch, spec, args.core, args.halo,
            ))
            case_index = int(batch["case_index"].reshape(-1)[0])
            for analysis in ("value", "value_flux"):
                q, jacobian, weight = linearized_q(
                    chart, context, inference_batch, spec, scale, analysis,
                )
                active = weight > 0.5
                metrics = _one_spectrum(
                    jacobian[active], q[active], args.ridge,
                    args.relative_rank_tolerance,
                )
                rows.append({
                    "case_index": case_index,
                    "analysis": analysis,
                    **metrics,
                })
            seen += 1
    finally:
        dataset.close()

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "cases.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_analysis = {
        name: [row for row in rows if row["analysis"] == name]
        for name in ("value", "value_flux")
    }
    metric_names = [
        key for key in rows[0] if key not in ("case_index", "analysis")
    ]
    summary: dict[str, object] = {}
    for name, selected in by_analysis.items():
        summary[name] = {
            metric: {
                "mean": float(np.mean([float(row[metric]) for row in selected])),
                "median": float(np.median([float(row[metric]) for row in selected])),
                "minimum": float(np.min([float(row[metric]) for row in selected])),
                "maximum": float(np.max([float(row[metric]) for row in selected])),
            }
            for metric in metric_names
        }
    paired = []
    for value, value_flux in zip(
        by_analysis["value"], by_analysis["value_flux"], strict=True,
    ):
        if value["case_index"] != value_flux["case_index"]:
            raise RuntimeError("paired spectrum rows lost case alignment")
        paired.append({
            "condition_ratio_value_flux_over_value":
                float(value_flux["supported_condition_number"])
                / max(float(value["supported_condition_number"]), 1.0e-30),
            "minimum_singular_ratio_value_flux_over_value":
                float(value_flux["minimum_supported_singular_value"])
                / max(float(value["minimum_supported_singular_value"]), 1.0e-30),
        })
    result = {
        "contract": "darcy_target_free_cauchy_port_spectrum_audit_v1",
        "split": "validation",
        "start_case": args.start_case,
        "cases": len(paired),
        "protected_test_opened": False,
        "target_values_used": False,
        "local_checkpoint_sha256": sha256(args.local_checkpoint),
        "chart_checkpoint_sha256": sha256(args.chart_checkpoint),
        "ridge": args.ridge,
        "relative_rank_tolerance": args.relative_rank_tolerance,
        "summary": summary,
        "paired": {
            key: {
                "mean": float(np.mean([row[key] for row in paired])),
                "median": float(np.median([row[key] for row in paired])),
                "fraction_at_most_one": float(np.mean([
                    row[key] <= 1.0 for row in paired
                ])),
            }
            for key in paired[0]
        },
        "interpretation": (
            "Only nonzero supported singular directions are conditioned; "
            "structural code null directions are reported through rank."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(parse_args()), indent=2))
