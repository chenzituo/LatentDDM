"""Validation-only oracle gate for a frozen Darcy residual-port chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from codiffuse_Schwarz.residual_port import scale_port_code

from .adapter import H5DarcyDataset
from .evaluate import move_batch, relative_l2
from .train import make_loader
from .train_residual_port_chart import (
    frozen_context, interface_losses, load_frozen_local, physics_losses, sha256,
)
from .train_residual_port_communicator import load_frozen_chart


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maximum-cases", type=int, default=64)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def metrics(normalized, corrected, context, batch, solution_std) -> dict[str, float]:
    physical = normalized * solution_std * batch["mask"]
    energy, residual = physics_losses(normalized, batch, solution_std)
    value, flux = interface_losses(corrected, normalized, context, batch)
    return {
        "relative_l2": float(relative_l2(
            physical, batch["solution"], batch["mask"],
        ).mean()),
        "energy_relative_square": float(energy),
        "residual_relative_square": float(residual),
        "interface_value_square": float(value),
        "interface_flux_square": float(flux),
    }


@torch.no_grad()
def validate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, checkpoint = load_frozen_chart(
        args.chart_checkpoint, device,
    )
    if checkpoint["source_local_sha256"] != sha256(args.local_checkpoint):
        raise ValueError("chart and local checkpoint identities differ")
    dataset = H5DarcyDataset(args.data, "validation")
    if not dataset.has_face_dirichlet:
        raise ValueError("oracle chart gate requires face-resolved validation data")
    loader = make_loader(dataset, args.batch_size, False, args.workers, device)
    rows = []
    cases = 0
    for host_batch in loader:
        if cases >= args.maximum_cases:
            break
        batch = move_batch(host_batch, device)
        original = batch["solution"].shape[0]
        keep = min(original, args.maximum_cases - cases)
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor) and value.shape[0] == original:
                batch[key] = value[:keep]
        context = frozen_context(
            local, adapter, batch, spec, args.core, args.halo,
        )
        local_assembled = context.cover.assemble_weighted(
            context.local, keep,
        ) * batch["mask"]
        oracle_code = scale_port_code(context.oracle_code, scale).reshape(
            -1, spec.code_width,
        )
        output = chart(
            context.local, context.physical, context.scalar, context.valid,
            oracle_code,
        )
        oracle_assembled = context.cover.assemble_weighted(
            output.corrected, keep,
        ) * batch["mask"]
        rows.append({
            "cases": keep,
            "local": metrics(
                local_assembled, context.local, context, batch,
                adapter.statistics.solution_std,
            ),
            "oracle": metrics(
                oracle_assembled, output.corrected, context, batch,
                adapter.statistics.solution_std,
            ),
        })
        cases += keep
    if not rows:
        raise RuntimeError("oracle chart gate saw no validation cases")
    names = tuple(rows[0]["local"])
    summary = {}
    for method in ("local", "oracle"):
        summary[method] = {
            name: float(np.average(
                [row[method][name] for row in rows],
                weights=[row["cases"] for row in rows],
            ))
            for name in names
        }
    improvement = {
        name: summary["local"][name] - summary["oracle"][name]
        for name in names
    }
    result = {
        "contract": "darcy_oracle_response_chart_gate_v1",
        "selection_split": "validation",
        "test_subsets_opened": False,
        "oracle_uses_target_for_diagnostic_only": True,
        "cases": cases,
        "chart_type": checkpoint["chart_spec"].get(
            "chart_type", "nonlinear_12mode",
        ),
        "chart_checkpoint": str(args.chart_checkpoint.resolve()),
        "port_width": spec.code_width,
        "chart_parameters": int(checkpoint["chart_spec"]["parameter_count"]),
        "source_local_sha256": checkpoint["source_local_sha256"],
        "source_chart_sha256": sha256(args.chart_checkpoint),
        "summary": summary,
        "improvement_local_minus_oracle": improvement,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), indent=2))
