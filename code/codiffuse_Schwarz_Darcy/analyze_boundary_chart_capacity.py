"""Measure how much Darcy accuracy a boundary correction chart can express.

This is an oracle diagnostic, not a deployable method: held-out solution traces
are projected onto each chart and passed through the same frozen local BVP map.
It separates boundary-chart capacity from communication-model optimization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from codiffuse_Schwarz.boundary_trace_basis import (
    decode_trace_code_2d,
    project_trace_code_2d,
)
from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .adapter import H5DarcyDataset, MODEL_SPEC
from .evaluate import move_batch, relative_l2
from .input_correction_latentddm import (
    solve_corrected_boundary_trace,
)
from .train_input_correction_latentddm import prepared_context
from .train_residual_port_chart import load_frozen_local


Tensor = torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--max-cases", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--core", type=int, default=MODEL_SPEC["core"])
    parser.add_argument("--halo", type=int, default=MODEL_SPEC["halo"])
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    modes = sorted(set(args.modes))
    if not modes or min(modes) <= 0 or args.max_cases <= 0:
        raise ValueError("invalid chart-capacity controls")
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda",
    )
    local_operator, adapter, _ = load_frozen_local(
        args.local_checkpoint, device,
    )
    dataset = H5DarcyDataset(args.data, "validation")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    errors: dict[str, list[float]] = {
        "local_core": [], "exact_trace_oracle": [],
    }
    trace_errors: dict[str, list[float]] = {}
    rows: list[dict[str, float | int]] = []
    for mode in modes:
        errors[f"projected_trace_modes_{mode}"] = []
        trace_errors[f"modes_{mode}"] = []
    processed = 0
    try:
        for host in loader:
            if processed >= args.max_cases:
                break
            batch = move_batch(host, device)
            original = int(batch["solution"].shape[0])
            keep = min(original, args.max_cases - processed)
            for name, value in tuple(batch.items()):
                if isinstance(value, Tensor) and value.shape[0] == original:
                    batch[name] = value[:keep]
            cover = CartesianCover(
                batch["solution"].shape[-2], batch["solution"].shape[-1],
                args.core, args.halo,
            )
            local, context = prepared_context(
                local_operator, adapter, batch, cover,
                include_target=True, trace_modes=2,
            )
            scale = float(adapter.statistics.solution_std)
            local_field = cover.assemble_weighted(local, keep) * scale
            exact_local = solve_corrected_boundary_trace(
                local_operator, adapter, context, context.target_delta,
            )
            exact_field = cover.assemble_weighted(exact_local, keep) * scale
            mask = batch["mask"]
            local_values = relative_l2(
                local_field * mask, batch["solution"], mask,
            ).detach().cpu().tolist()
            exact_values = relative_l2(
                exact_field * mask, batch["solution"], mask,
            ).detach().cpu().tolist()
            errors["local_core"].extend(local_values)
            errors["exact_trace_oracle"].extend(exact_values)
            batch_rows = [{
                "case_index": int(batch["case_index"][item]),
                "local_core_relative_l2": float(local_values[item]),
                "exact_trace_oracle_relative_l2": float(exact_values[item]),
            } for item in range(keep)]
            for mode in modes:
                code = project_trace_code_2d(
                    context.target_delta, context.roles,
                    channels=1, modes=mode,
                ).reshape(keep, cover.tile_count, 4 * mode)
                projected = decode_trace_code_2d(
                    code.flatten(0, 1), context.roles,
                    channels=1, modes=mode,
                )
                # The context's mode is only used by the code decoder, so
                # evaluate the projected dense trace directly here.
                projected_local = solve_corrected_boundary_trace(
                    local_operator, adapter, context, projected,
                )
                projected_field = (
                    cover.assemble_weighted(projected_local, keep) * scale
                )
                field_values = relative_l2(
                    projected_field * mask, batch["solution"], mask,
                ).detach().cpu().tolist()
                errors[f"projected_trace_modes_{mode}"].extend(field_values)
                active = context.roles.repeat_interleave(1, dim=1)
                numerator = (
                    (projected - context.target_delta).square() * active
                ).flatten(1).sum(1)
                denominator = (
                    context.target_delta.square() * active
                ).flatten(1).sum(1).clamp_min(1.0e-8)
                per_tile = (numerator / denominator).reshape(
                    keep, cover.tile_count,
                ).mean(1)
                trace_values = per_tile.detach().cpu().tolist()
                trace_errors[f"modes_{mode}"].extend(trace_values)
                for item, row in enumerate(batch_rows):
                    row[f"projected_trace_modes_{mode}_relative_l2"] = float(
                        field_values[item]
                    )
                    row[f"trace_modes_{mode}_relative_square"] = float(
                        trace_values[item]
                    )
            rows.extend(batch_rows)
            processed += keep
    finally:
        dataset.close()
    result = {
        "contract": "darcy_boundary_chart_capacity_oracle_v1",
        "split": "validation",
        "protected_test_cases_opened": 0,
        "cases": processed,
        "core": args.core,
        "halo": args.halo,
        "local_checkpoint": str(args.local_checkpoint.resolve()),
        "field_relative_l2": {
            name: summarize(value) for name, value in errors.items()
        },
        "trace_relative_square": {
            name: summarize(value) for name, value in trace_errors.items()
        },
        "rows": rows,
        "oracle_uses_held_out_solution_trace": True,
        "pde_residual_used": False,
        "flux_law_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
