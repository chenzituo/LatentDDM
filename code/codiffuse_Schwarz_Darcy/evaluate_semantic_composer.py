"""Evaluate the semantic Darcy composer and its causal code ladder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz.compatibility_composer import CompatibilityOutput
from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import (
    DarcyAdapter, DarcyStatistics, FaceDarcyAdapter, H5DarcyDataset,
    MODEL_SPEC, build_model,
)
from .evaluate import (
    _interface_diagnostics, energy_relative_error, move_batch, relative_l2,
    residual_rms,
)
from .train_semantic_composer import (
    CONTRACT, SemanticDarcyComposer, assembled_prediction, prepare,
    semantic_trace_code,
)


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("validation", "val", "A", "B", "C", "D"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--offsets", nargs="+", default=("0,0", "8,8", "16,16"))
    parser.add_argument("--max-graph-radius", type=int)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def _instantiate(
    checkpoint: dict[str, Any], device: torch.device,
) -> tuple[SemanticDarcyComposer, DarcyAdapter]:
    if checkpoint.get("contract") != CONTRACT or not checkpoint.get("operator_free"):
        raise ValueError("checkpoint is not a semantic operator-free Darcy composer")
    statistics = DarcyStatistics.from_json(checkpoint["statistics"])
    adapter = (
        FaceDarcyAdapter(statistics)
        if checkpoint.get("source_contract")
        == "darcy_sni_face_boundary_to_solution_v1"
        else DarcyAdapter(statistics, use_discrete_residual=False)
    )
    local = build_model(adapter, communication_stage="none")
    spec = checkpoint["model_spec"]
    model = SemanticDarcyComposer(
        local, adapter,
        bins_per_face=int(spec["bins_per_face"]),
        response_modes=int(spec["response_modes"]),
        token_width=int(spec["token_width"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, adapter


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, adapter = _instantiate(checkpoint, device)
    dataset = H5DarcyDataset(args.data, args.split)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    offsets = [tuple(int(piece) for piece in value.split(",")) for value in args.offsets]
    if any(len(value) != 2 or min(value) < 0 or max(value) >= MODEL_SPEC["core"]
           for value in offsets):
        raise ValueError("offsets must be y,x pairs inside the core")
    records: list[dict[str, Any]] = []
    processed = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for host_batch in loader:
        if args.max_cases is not None and processed >= args.max_cases:
            break
        batch = move_batch(host_batch, device)
        original_batch = batch["solution"].shape[0]
        if args.max_cases is not None:
            keep = min(original_batch, args.max_cases - processed)
            for key, value in list(batch.items()):
                if isinstance(value, Tensor) and value.shape[0] == original_batch:
                    batch[key] = value[:keep]
        for offset_y, offset_x in offsets:
            cover = CartesianCover(
                batch["solution"].shape[-2], batch["solution"].shape[-1],
                MODEL_SPEC["core"], MODEL_SPEC["halo"], offset_y, offset_x,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                local, physical, scalar, valid, target_normalized, current, cover, transition = prepare(
                    model, adapter, batch, MODEL_SPEC["core"], MODEL_SPEC["halo"], cover,
                )
                predicted = model.compose(
                    local, physical, scalar, valid, cover.grid_shape,
                    max_graph_radius=args.max_graph_radius,
                )
                predicted_field, _ = assembled_prediction(
                    adapter, transition, cover, predicted.corrected,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            predicted_seconds = (
                time.perf_counter() - started
            ) / batch["solution"].shape[0]

            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                oracle_code = semantic_trace_code(
                    target_normalized, batch["mask"], cover, model.bins_per_face,
                )
                zero = torch.zeros_like(predicted.predicted_code)
                outputs: dict[str, CompatibilityOutput] = {
                    "local": model.compose(
                        local, physical, scalar, valid, cover.grid_shape,
                        code_override=zero,
                        max_graph_radius=args.max_graph_radius,
                    ),
                    "predicted": predicted,
                    "shuffled": model.compose(
                        local, physical, scalar, valid, cover.grid_shape,
                        code_override=torch.roll(
                            predicted.predicted_code, shifts=1, dims=1,
                        ),
                        max_graph_radius=args.max_graph_radius,
                    ),
                    "oracle": model.compose(
                        local, physical, scalar, valid, cover.grid_shape,
                        code_override=oracle_code,
                        max_graph_radius=args.max_graph_radius,
                    ),
                }
                fields = {"predicted": predicted_field}
                assembled_normalized = {
                    "predicted": cover.assemble_weighted(
                        predicted.corrected, transition.current.shape[0],
                    ),
                }
                for name in ("local", "shuffled", "oracle"):
                    fields[name], assembled_normalized[name] = assembled_prediction(
                        adapter, transition, cover, outputs[name].corrected,
                    )

            for method, field in fields.items():
                l2 = relative_l2(field, batch["solution"], batch["mask"])
                energy = energy_relative_error(
                    field, batch["solution"], batch["logk"], batch["mask"],
                )
                residual = residual_rms(field, batch)
                value_jump, flux_jump = _interface_diagnostics(
                    outputs[method].corrected,
                    assembled_normalized[method],
                    current,
                    batch["logk"],
                    cover,
                    adapter.statistics.solution_std,
                )
                for item in range(field.shape[0]):
                    records.append({
                        "case_index": int(batch["case_index"][item]),
                        "offset_y": offset_y,
                        "offset_x": offset_x,
                        "method": method,
                        "relative_l2": float(l2[item]),
                        "energy_relative_error": float(energy[item]),
                        "residual_rms": float(residual[item]),
                        "interface_value_jump": float(value_jump[item]),
                        "interface_flux_jump": float(flux_jump[item]),
                        "predicted_wall_seconds": predicted_seconds,
                        "local_nfe": cover.tile_count,
                        "communication_rounds": 1 if method == "predicted" else 0,
                    })
        processed += batch["solution"].shape[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "case_metrics.csv").open(
        "w", newline="", encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    summaries = {}
    for method in ("local", "predicted", "shuffled", "oracle"):
        selected = [record for record in records if record["method"] == method]
        summaries[method] = {
            key: float(np.mean([record[key] for record in selected]))
            for key in (
                "relative_l2", "energy_relative_error", "residual_rms",
                "interface_value_jump", "interface_flux_jump",
                "predicted_wall_seconds", "local_nfe", "communication_rounds",
            )
        }
    result = {
        "contract": "darcy_semantic_trace_theorem_ladder_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "cases": processed,
        "offsets": offsets,
        "max_graph_radius": args.max_graph_radius,
        "semantic_oracle_uses_targets_for_diagnostics_only": True,
        "shuffled_code_control": "cyclic permutation over tile index",
        "operator_free_predicted_inference": True,
        "discrete_diagnostics_outside_timing": True,
        "peak_gpu_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        ),
        "summary": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))
