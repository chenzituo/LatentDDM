"""Measure the frozen-local family floor and learned communication gap.

This validation-only diagnostic evaluates three predictions on identical
cases: the communication-free local/core field, LatentDDM, and an oracle
response-chart code fitted from the target.  The oracle is never an inference
method; it estimates the best response available to the frozen local family
and therefore distinguishes a communication error from a local-family error.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from codiffuse_Schwarz.residual_port import scale_port_code

from .adapter import H5DarcyDataset
from .evaluate import move_batch, relative_l2
from .field_token_composer import DarcyFieldTokenComposer
from .train import make_loader
from .train_field_token_composer import CONTRACT
from .train_residual_port_chart import frozen_context, load_frozen_local, sha256
from .train_residual_port_communicator import load_frozen_chart


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--composer-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "val"), default="validation")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=64)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _case_metadata(batch: dict[str, Any], item: int) -> dict[str, Any]:
    result: dict[str, Any] = {"case_index": int(batch["case_index"][item])}
    if "seed" in batch:
        result["case_seed"] = int(batch["seed"][item])
    if "family" in batch:
        result["family"] = str(batch["family"][item])
    if "physical_extent" in batch:
        result["physical_extent"] = float(batch["physical_extent"][item])
    if "grid_spacing" in batch:
        result["grid_spacing"] = float(batch["grid_spacing"][item])
    return result


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.batch_size, args.max_cases, args.core) <= 0 or args.halo < 0:
        raise ValueError("invalid family-floor evaluation extent")
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, chart_payload = load_frozen_chart(
        args.chart_checkpoint, device,
    )
    saved = torch.load(
        args.composer_checkpoint, map_location="cpu", weights_only=False,
    )
    if saved.get("contract") != CONTRACT:
        raise ValueError("composer checkpoint has the wrong contract")
    if saved.get("source_local_sha256") != sha256(args.local_checkpoint):
        raise ValueError("composer and frozen local operator differ")
    if saved.get("source_chart_sha256") != sha256(args.chart_checkpoint):
        raise ValueError("composer and response chart differ")
    if chart_payload.get("source_local_sha256") != sha256(args.local_checkpoint):
        raise ValueError("response chart and frozen local operator differ")
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
        module.eval()

    dataset = H5DarcyDataset(args.data, args.split)
    dataset.length = min(dataset.length, args.max_cases)
    loader = make_loader(dataset, args.batch_size, False, args.workers, device)
    rows: list[dict[str, Any]] = []
    try:
        for host in loader:
            batch = move_batch(host, device)
            context = frozen_context(
                local, adapter, batch, spec, args.core, args.halo,
            )
            batch_size = batch["solution"].shape[0]
            local_normalized = context.cover.assemble_ras(
                context.local, batch_size,
            ) * batch["mask"]
            latent_normalized = composer(
                chart, context, batch["mask"],
            ).assembled
            oracle_code = scale_port_code(
                context.oracle_code, scale,
            ).reshape(-1, spec.code_width)
            oracle_tiles = chart(
                context.local, context.physical, context.scalar,
                context.valid, oracle_code,
            ).corrected
            oracle_normalized = context.cover.assemble_ras(
                oracle_tiles, batch_size,
            ) * batch["mask"]
            predictions = {
                "local_core": local_normalized * adapter.statistics.solution_std,
                "latentddm": latent_normalized * adapter.statistics.solution_std,
                "oracle_family": oracle_normalized * adapter.statistics.solution_std,
            }
            errors = {
                name: relative_l2(value, batch["solution"], batch["mask"])
                for name, value in predictions.items()
            }
            for item in range(batch_size):
                local_error = float(errors["local_core"][item])
                latent_error = float(errors["latentddm"][item])
                oracle_error = float(errors["oracle_family"][item])
                denominator = max(local_error, 1.0e-12)
                oracle_gap = max(latent_error - oracle_error, 0.0)
                available = max(local_error - oracle_error, 1.0e-12)
                rows.append({
                    **_case_metadata(batch, item),
                    "local_core_relative_l2": local_error,
                    "latentddm_relative_l2": latent_error,
                    "oracle_family_relative_l2": oracle_error,
                    "latent_gain_over_local": (
                        local_error - latent_error
                    ) / denominator,
                    "oracle_gain_over_local": (
                        local_error - oracle_error
                    ) / denominator,
                    "communication_gap_to_oracle": latent_error - oracle_error,
                    "fraction_available_gain_recovered": 1.0 - oracle_gap / available,
                })
    finally:
        dataset.close()
    if not rows:
        raise RuntimeError("family-floor evaluation selected no cases")

    metric_names = tuple(
        key for key in rows[0]
        if key not in (
            "case_index", "case_seed", "family", "physical_extent",
            "grid_spacing",
        )
    )
    summary = {
        name: {
            "mean": float(np.mean([float(row[name]) for row in rows])),
            "sample_std": float(np.std(
                [float(row[name]) for row in rows], ddof=1,
            )) if len(rows) > 1 else 0.0,
        }
        for name in metric_names
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "case_metrics.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "contract": "darcy_field_token_family_floor_v1",
        "split": "validation",
        "test_cases_opened": 0,
        "data": str(args.data.resolve()),
        "local_checkpoint": str(args.local_checkpoint.resolve()),
        "chart_checkpoint": str(args.chart_checkpoint.resolve()),
        "composer_checkpoint": str(args.composer_checkpoint.resolve()),
        "cases": len(rows),
        "oracle_role": (
            "evaluation-only projection of target onto the frozen local response family"
        ),
        "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
