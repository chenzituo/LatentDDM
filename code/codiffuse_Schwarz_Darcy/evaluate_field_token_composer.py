"""Evaluate a frozen Darcy field-token composer on a declared validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .adapter import H5DarcyDataset
from .field_token_composer import DarcyFieldTokenComposer
from .train import make_loader
from .train_field_token_composer import CONTRACT, validate
from .train_residual_port_chart import load_frozen_local, sha256
from .train_residual_port_communicator import load_frozen_chart


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--composer-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--max-validation-cases", type=int, default=128)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.batch_size, args.validation_batches,
        args.max_validation_cases, args.core,
    ) <= 0 or args.halo < 0:
        raise ValueError("invalid evaluation extent")
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
        raise ValueError("composer and endpoint differ")
    if saved.get("source_chart_sha256") != sha256(args.chart_checkpoint):
        raise ValueError("composer and response chart differ")
    if chart_payload.get("source_local_sha256") != sha256(args.local_checkpoint):
        raise ValueError("response chart and endpoint differ")
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
    composer.eval(); chart.eval(); local.eval()

    dataset = H5DarcyDataset(args.data, "validation")
    try:
        dataset.length = min(dataset.length, args.max_validation_cases)
        loader = make_loader(
            dataset, args.batch_size, False, args.workers, device,
        )
        metrics = validate(
            composer, chart, local, adapter, loader, device, spec, scale,
            SimpleNamespace(
                validation_batches=args.validation_batches,
                core=args.core,
                halo=args.halo,
                tile_weight=0.25,
                trace_weight=0.05,
                seam_weight=0.0,
                code_weight=0.10,
            ),
        )
        evaluated = min(
            dataset.length,
            args.batch_size * args.validation_batches,
        )
    finally:
        dataset.close()
    result: dict[str, object] = {
        "contract": "darcy_field_token_validation_v1",
        "model_contract": CONTRACT,
        "data": str(args.data.resolve()),
        "composer_checkpoint": str(args.composer_checkpoint.resolve()),
        "composer_checkpoint_sha256": sha256(args.composer_checkpoint),
        "communication_depth": int(configuration["communication_depth"]),
        "spatial_port_bins": int(configuration.get("spatial_port_bins", 0)),
        "assembly": "disjoint_owned_core",
        "cases": evaluated,
        "test_subsets_opened": False,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
