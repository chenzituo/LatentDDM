"""Audit whether Darcy LatentDDM depends on overlap averaging.

The local crops retain the halo used during pretraining.  We change only the
writeback map: ``pou`` blends every duplicate overlap value, whereas ``core``
keeps one disjoint owned core per crop (RAS writeback).  The latter is a
no-overlap *output* composition, not a claim that the frozen endpoint was
trained without halo context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .adapter import H5DarcyDataset
from .evaluate import move_batch, relative_l2
from .train_incremental_flux_remote_schur import (
    _decomposed_fields,
    _load_communicator,
)
from .train_residual_port_chart import frozen_context, load_frozen_local, sha256
from .train_residual_port_communicator import load_frozen_chart
from .validate_residual_port_composer import interface_losses, physics_losses


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--latent-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _interface_strip(cover, field: Tensor) -> Tensor:
    """Return a two-cell strip on every owned-core boundary."""
    mask = torch.zeros_like(field[:, :1])
    for location in range(cover.core - cover.offset_y, cover.height, cover.core):
        if 0 < location < cover.height:
            mask[..., max(location - 1, 0):min(location + 1, cover.height), :] = 1
    for location in range(cover.core - cover.offset_x, cover.width, cover.core):
        if 0 < location < cover.width:
            mask[..., :, max(location - 1, 0):min(location + 1, cover.width)] = 1
    return mask


def _weighted_mean(rows: list[dict[str, float]], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    weights = np.asarray([row["cases"] for row in rows], dtype=np.float64)
    return float(np.average(values, weights=weights))


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.max_cases, args.batch_size, args.core) <= 0 or args.halo < 0:
        raise ValueError("invalid evaluation extent")
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, _ = load_frozen_chart(args.chart_checkpoint, device)
    communicator, communicator_payload = _load_communicator(
        args.latent_checkpoint, spec, device,
    )
    for module in (local, chart, communicator):
        module.eval().requires_grad_(False)

    dataset = H5DarcyDataset(args.data, "validation")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    rows: list[dict[str, float]] = []
    processed = 0
    replay_max_abs = 0.0
    try:
        for host in loader:
            if processed >= args.max_cases:
                break
            batch = move_batch(host, device)
            original = int(batch["solution"].shape[0])
            keep = min(original, args.max_cases - processed)
            for key, value in list(batch.items()):
                if isinstance(value, Tensor) and value.shape[0] == original:
                    batch[key] = value[:keep]
            context = frozen_context(
                local, adapter, batch, spec, args.core, args.halo,
            )
            output = _decomposed_fields(
                communicator, chart, context, batch, spec, scale,
            )
            weighted_replay = context.cover.assemble_weighted(
                output["corrected"], keep,
            ) * batch["mask"]
            replay_max_abs = max(
                replay_max_abs,
                float((weighted_replay - output["assembled"]).abs().max()),
            )
            strip = _interface_strip(context.cover, batch["solution"])
            target = batch["solution"]
            for method, tiles in (
                ("local", context.local),
                ("latentddm", output["corrected"]),
            ):
                for assembly in ("pou", "core"):
                    assemble = (
                        context.cover.assemble_weighted
                        if assembly == "pou" else context.cover.assemble_ras
                    )
                    normalized = assemble(tiles, keep) * batch["mask"]
                    prediction = normalized * adapter.statistics.solution_std
                    field_error = relative_l2(
                        prediction, target, batch["mask"],
                    )
                    strip_error = relative_l2(
                        prediction, target, batch["mask"] * strip,
                    )
                    value, flux = interface_losses(
                        tiles, normalized, context, batch,
                    )
                    energy, residual = physics_losses(
                        normalized, batch, adapter.statistics.solution_std,
                    )
                    rows.append({
                        "cases": float(keep),
                        "method": method,
                        "assembly": assembly,
                        "relative_l2": float(field_error.mean()),
                        "core_boundary_strip_relative_l2": float(
                            strip_error.mean()
                        ),
                        "interface_value_square": float(value),
                        "interface_flux_square": float(flux),
                        "energy_relative_square": float(energy),
                        "residual_relative_square": float(residual),
                    })
            processed += keep
    finally:
        dataset.close()

    metrics = (
        "relative_l2", "core_boundary_strip_relative_l2",
        "interface_value_square", "interface_flux_square",
        "energy_relative_square", "residual_relative_square",
    )
    summary = {}
    for method in ("local", "latentddm"):
        for assembly in ("pou", "core"):
            selected = [
                row for row in rows
                if row["method"] == method and row["assembly"] == assembly
            ]
            summary[f"{method}/{assembly}"] = {
                "cases": processed,
                **{key: _weighted_mean(selected, key) for key in metrics},
            }
    result = {
        "contract": "darcy_core_owned_composition_ablation_v1",
        "split": "validation",
        "protected_test_cases_opened": 0,
        "cases": processed,
        "cover": {"core": args.core, "halo": args.halo},
        "assembly_contract": {
            "pou": "normalized cosine least-squares consensus in overlap",
            "core": "disjoint owned-core/RAS writeback; no overlap averaging",
            "qualification": (
                "both arms retain the same pretrained halo-extended local crops"
            ),
        },
        "local_checkpoint": str(args.local_checkpoint.resolve()),
        "local_checkpoint_sha256": sha256(args.local_checkpoint),
        "chart_checkpoint": str(args.chart_checkpoint.resolve()),
        "chart_checkpoint_sha256": sha256(args.chart_checkpoint),
        "latent_checkpoint": str(args.latent_checkpoint.resolve()),
        "latent_checkpoint_sha256": sha256(args.latent_checkpoint),
        "latent_checkpoint_frozen": bool(
            communicator_payload.get("communicator_spec", {}).get(
                "frozen_candidate", False,
            )
        ),
        "weighted_latent_replay_max_abs": replay_max_abs,
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    return result


if __name__ == "__main__":
    evaluate(parse_args())
