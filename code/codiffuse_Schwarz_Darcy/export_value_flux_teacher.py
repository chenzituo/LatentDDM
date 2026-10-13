"""Export fixed-ridge value+flux latent-Schur codes from train data only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from .adapter import H5DarcyDataset
from .evaluate import move_batch
from .latent_schur_teacher import inference_only_context, solve_value_flux_teacher
from .train import make_loader
from .train_residual_port_chart import frozen_context, load_frozen_local, sha256
from .train_residual_port_communicator import load_frozen_chart


CONTRACT = "darcy_value_flux_latent_schur_teacher_train_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--chart-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-cases", type=int, default=4000)
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--halo", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--damping", type=float, nargs="+", default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def export(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite teacher corpus: {args.output}")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    local, adapter, _ = load_frozen_local(args.local_checkpoint, device)
    chart, spec, scale, chart_checkpoint = load_frozen_chart(args.chart_checkpoint, device)
    if chart_checkpoint["source_local_sha256"] != sha256(args.local_checkpoint):
        raise ValueError("teacher chart/local lineage differs")
    dataset = H5DarcyDataset(args.data, "train")
    loader = make_loader(dataset, 1, False, args.workers, device)
    count = min(args.maximum_cases, len(dataset))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale teacher partial exists: {temporary}")
    ratios = []; dampings = []; invariance_max_abs = None
    with h5py.File(temporary, "w") as handle:
        handle.attrs["contract"] = CONTRACT
        handle.attrs["source_split"] = "train"
        handle.attrs["validation_or_test_opened"] = False
        handle.attrs["target_or_pde_labels_used"] = False
        handle.attrs["analysis"] = "value_plus_artificial_face_flux"
        handle.attrs["ridge"] = args.ridge
        handle.attrs["source_data_sha256"] = sha256(args.data)
        handle.attrs["local_checkpoint_sha256"] = sha256(args.local_checkpoint)
        handle.attrs["chart_checkpoint_sha256"] = sha256(args.chart_checkpoint)
        group = handle.create_group("train")
        indices = group.create_dataset("case_index", (count,), dtype="i8")
        codes = group.create_dataset(
            "teacher_code", (count, 9, spec.code_width), dtype="f4",
            compression="gzip", compression_opts=4, shuffle=True,
        )
        alpha_ds = group.create_dataset("selected_damping", (count,), dtype="f4")
        ratio_ds = group.create_dataset("complete_q_cycle_square", (count,), dtype="f4")
        for row, host in enumerate(loader):
            if row >= count:
                break
            batch = move_batch(host, device)
            context = frozen_context(
                local, adapter, batch, spec, args.core, args.halo,
            )
            context = inference_only_context(context)
            code, alpha, ratio = solve_value_flux_teacher(
                chart, context, batch, spec, scale,
                ridge=args.ridge, damping=tuple(args.damping),
            )
            if row == 0:
                changed = dict(batch)
                changed["solution"] = torch.full_like(batch["solution"], 37.0)
                changed_context = inference_only_context(frozen_context(
                    local, adapter, changed, spec, args.core, args.halo,
                ))
                if torch.count_nonzero(changed_context.local - context.local) \
                        or torch.count_nonzero(changed_context.physical - context.physical) \
                        or torch.count_nonzero(changed_context.scalar - context.scalar):
                    raise RuntimeError("solution replacement altered teacher observables")
                changed_code, _, _ = solve_value_flux_teacher(
                    chart, changed_context, changed, spec, scale,
                    ridge=args.ridge, damping=tuple(args.damping),
                )
                invariance_max_abs = float((changed_code - code).abs().max())
                if invariance_max_abs > 1.0e-6:
                    raise RuntimeError("teacher code depends on the replaced solution target")
            indices[row] = int(batch["case_index"].reshape(-1)[0])
            codes[row] = code[0].cpu().numpy().astype(np.float32)
            alpha_ds[row] = alpha; ratio_ds[row] = ratio
            ratios.append(ratio); dampings.append(alpha)
        handle.attrs["solution_replacement_invariance_checked"] = True
        handle.attrs["solution_replacement_invariance_max_abs"] = invariance_max_abs
    dataset.close(); os.replace(temporary, args.output)
    result = {
        "contract": CONTRACT, "source_split": "train",
        "validation_or_test_opened": False, "target_or_pde_labels_used": False,
        "cases": count, "ridge": args.ridge,
        "mean_complete_q_cycle_square": float(np.mean(ratios)),
        "maximum_complete_q_cycle_square": float(np.max(ratios)),
        "mean_selected_damping": float(np.mean(dampings)),
        "solution_replacement_invariance_checked": True,
        "solution_replacement_invariance_max_abs": invariance_max_abs,
        "sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result
if __name__ == "__main__":
    print(json.dumps(export(parse_args()), indent=2))
