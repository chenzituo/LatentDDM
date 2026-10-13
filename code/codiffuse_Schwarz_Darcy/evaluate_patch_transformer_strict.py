"""Strict 60-case Darcy Local POU versus same-cover Patch Transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import (
    H5DarcyDataset,
    build_model,
    checkpoint_communication_stage,
)
from .evaluate import move_batch, relative_l2
from .train_communicated import matched_local_conditions
from .train_residual_port_chart import load_frozen_local


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "codiffuse_Schwarz_Darcy/data/darcy_full_domain_face_96_v2.h5"
LOCAL = ROOT / "codiffuse_Schwarz_Darcy/outputs/sni_face_local_seed20260806/checkpoint.pt"
PATCH = ROOT / (
    "codiffuse_Schwarz_Darcy/outputs/"
    "patch_transformer_face_v2_frozen_local_v2/checkpoint.pt"
)
OUTPUT = ROOT / (
    "codiffuse_Schwarz_Darcy/outputs/patch_transformer_face_v2_frozen_local_v2/"
    "strict_validation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--local", type=Path, default=LOCAL)
    parser.add_argument("--patch-transformer", type=Path, default=PATCH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    return parser.parse_args()


def overlap_rms(
    tiles: torch.Tensor, assembled: torch.Tensor, valid: torch.Tensor,
    cover: CartesianCover,
) -> torch.Tensor:
    batch = assembled.shape[0]
    consensus = cover.tile(assembled)
    valid_tiles = cover.tile(valid) * cover.valid_tiles(
        batch, tiles.device, tiles.dtype,
    )
    numerator = ((tiles - consensus).square() * valid_tiles).flatten(1).sum(1)
    denominator = valid_tiles.flatten(1).sum(1).clamp_min(1.0)
    return (numerator / denominator).sqrt()


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local, adapter, _ = load_frozen_local(args.local, device)
    patch_payload = torch.load(
        args.patch_transformer, map_location="cpu", weights_only=False,
    )
    if patch_payload.get("local_input_contract") != \
            "face_resolved_sni_directional_trace_v1":
        raise ValueError(
            "Darcy Patch Transformer did not preserve the face-local input contract"
        )
    stage = checkpoint_communication_stage(patch_payload)
    if stage != "proposal":
        raise ValueError("Patch Transformer must use proposal-stage communication")
    patch = build_model(adapter, communication_stage=stage).to(device)
    patch.load_state_dict(patch_payload["model"])
    patch.eval().requires_grad_(False)
    dataset = H5DarcyDataset(args.data, "validation")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    records = []
    warmed = False
    try:
        for host in loader:
            if len(records) >= args.max_cases:
                break
            batch = move_batch(host, device)
            keep = min(batch["solution"].shape[0], args.max_cases - len(records))
            if keep < batch["solution"].shape[0]:
                for name, value in list(batch.items()):
                    if isinstance(value, torch.Tensor) and \
                            value.shape[0] == batch["solution"].shape[0]:
                        batch[name] = value[:keep]
            zero = torch.zeros_like(batch["solution"])
            cover = CartesianCover(
                zero.shape[-2], zero.shape[-1], args.core, args.halo,
            )
            physical, scalar = matched_local_conditions(
                adapter, batch, zero, cover,
            )
            if not warmed:
                for model in (local, patch):
                    warm_tiles = model(
                        physical, scalar, cover.tile_count,
                        tile_grid_shape=cover.grid_shape,
                    )
                    cover.assemble_weighted(
                        warm_tiles, batch["solution"].shape[0],
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                warmed = True
            outputs = {}
            for name, model in (("local", local), ("patch_transformer", patch)):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                tiles = model(
                    physical, scalar, cover.tile_count,
                    tile_grid_shape=cover.grid_shape,
                )
                normalized = cover.assemble_weighted(
                    tiles, batch["solution"].shape[0],
                )
                field = (
                    normalized * adapter.statistics.solution_std * batch["mask"]
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                outputs[name] = {
                    "field": field,
                    "seconds": (time.perf_counter() - started) / field.shape[0],
                    "overlap": overlap_rms(
                        tiles, normalized, batch["mask"], cover,
                    ),
                }
            for item in range(batch["solution"].shape[0]):
                row = {"case_index": int(batch["case_index"][item])}
                for name, output in outputs.items():
                    row[name] = {
                        "relative_l2": float(relative_l2(
                            output["field"][item:item + 1],
                            batch["solution"][item:item + 1],
                            batch["mask"][item:item + 1],
                        )[0]),
                        "seconds": float(output["seconds"]),
                        "overlap_rms_normalized": float(output["overlap"][item]),
                    }
                records.append(row)
    finally:
        dataset.close()
    aggregate = {}
    for name in ("local", "patch_transformer"):
        aggregate[name] = {
            key: float(np.mean([row[name][key] for row in records]))
            for key in records[0][name]
        }
    return {
        "contract": "darcy_patch_transformer_strict_validation_v1",
        "split": "validation",
        "protected_test_cases_opened": 0,
        "cases": len(records),
        "cover": {
            "grid_shape": list(CartesianCover(96, 96, args.core, args.halo).grid_shape),
            "core": args.core, "halo": args.halo,
        },
        "local_input_contract": "face_resolved_sni_directional_trace_v1",
        "local": str(args.local.resolve()),
        "patch_transformer": str(args.patch_transformer.resolve()),
        "aggregate": aggregate,
        "records": records,
    }


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "cases": result["cases"], "aggregate": result["aggregate"],
        "output": str(args.output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
