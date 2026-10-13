"""Generate train-split-only remote Dirichlet response pairs offline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from .adapter import H5DarcyDataset
from .evaluate_locality_obstruction import _boundary_bump, _tile_pair
from .physics import relative_residual, solve_darcy
from .train_residual_port_chart import sha256


CONTRACT = "darcy_remote_dirichlet_pair_train_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-cases", type=int, default=1024)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=(0.5, 1.0))
    parser.add_argument("--core", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--residual-checks", type=int, default=128)
    return parser.parse_args()


def generate(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite paired corpus: {args.output}")
    if args.base_cases <= 0 or not args.amplitudes or min(args.amplitudes) <= 0:
        raise ValueError("base-cases and amplitudes must be positive")
    dataset = H5DarcyDataset(args.data, "train")
    if not dataset.has_face_dirichlet:
        raise ValueError("remote pairs require face-resolved Darcy data")
    rng = np.random.default_rng(args.seed)
    base_count = min(args.base_cases, len(dataset))
    base_indices = rng.choice(len(dataset), base_count, replace=False)
    pair_count = base_count * len(args.amplitudes)
    sample = dataset[int(base_indices[0])]
    height, width = sample["mask"].shape[-2:]
    normalization = json.loads(str(dataset.attrs["normalization"]))
    bc_std = float(normalization["bc_value"]["std"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial corpus exists: {temporary}")
    checked_residuals: list[float] = []
    with h5py.File(temporary, "w") as handle:
        handle.attrs["contract"] = CONTRACT
        handle.attrs["source_data"] = str(args.data.resolve())
        handle.attrs["source_data_sha256"] = sha256(args.data)
        handle.attrs["source_split"] = "train"
        handle.attrs["validation_or_test_opened"] = False
        handle.attrs["seed"] = args.seed
        group = handle.create_group("train_pairs")
        base_ds = group.create_dataset("base_index", (pair_count,), dtype="i8")
        amplitude_ds = group.create_dataset("amplitude", (pair_count,), dtype="f4")
        target_ds = group.create_dataset("target_tile", (pair_count, 2), dtype="i2")
        remote_ds = group.create_dataset("remote_tile", (pair_count, 2), dtype="i2")
        delta_ds = group.create_dataset(
            "delta_bc", (pair_count, height, width), dtype="f4",
            chunks=(1, height, width), compression="gzip", compression_opts=4,
            shuffle=True,
        )
        response_ds = group.create_dataset(
            "solution_response", (pair_count, height, width), dtype="f4",
            chunks=(1, height, width), compression="gzip", compression_opts=4,
            shuffle=True,
        )
        check_rows = set(np.linspace(
            0, pair_count - 1, min(args.residual_checks, pair_count), dtype=int,
        ).tolist())
        row = 0
        for base_index in base_indices:
            item = dataset[int(base_index)]
            mask = item["mask"][0].numpy().astype(bool)
            boundary = item["bc_mask"][0].numpy().astype(bool) & mask
            target, remote = _tile_pair(mask, boundary, args.core)
            logk = item["logk"][0].numpy().astype(np.float64)
            source = item["source"][0].numpy().astype(np.float64)
            base_bc = item["bc_value"][0].numpy().astype(np.float64)
            base_solution = item["solution"][0].numpy().astype(np.float64)
            face_mask = item["dirichlet_face_mask"].numpy().astype(bool)
            face_value = item["dirichlet_face_value"].numpy().astype(np.float64)
            for amplitude in args.amplitudes:
                delta = _boundary_bump(
                    boundary, remote, args.core,
                    float(amplitude) * bc_std,
                )
                modified_face = face_value + delta[None] * face_mask
                modified, system = solve_darcy(
                    mask, np.power(10.0, np.clip(logk, -6.0, 6.0)), source,
                    base_bc + delta, dx=2.0 / max(height, width),
                    dirichlet_face_mask=face_mask,
                    dirichlet_face_value=modified_face,
                )
                response = modified - base_solution
                if not np.isfinite(response).all() or not np.any(response[mask]):
                    raise FloatingPointError("remote pair has invalid zero/nonfinite response")
                if row in check_rows:
                    checked_residuals.append(relative_residual(modified, system))
                base_ds[row] = int(base_index)
                amplitude_ds[row] = float(amplitude)
                target_ds[row] = target
                remote_ds[row] = remote
                delta_ds[row] = delta.astype(np.float32)
                response_ds[row] = response.astype(np.float32)
                row += 1
        group.attrs["pairs"] = row
        group.attrs["base_cases"] = base_count
        group.attrs["amplitudes"] = np.asarray(args.amplitudes, dtype=np.float32)
        group.attrs["max_checked_relative_residual"] = max(checked_residuals, default=0.0)
    dataset.close()
    os.replace(temporary, args.output)
    result = {
        "contract": CONTRACT, "output": str(args.output.resolve()),
        "source_split": "train", "validation_or_test_opened": False,
        "base_cases": base_count, "pairs": pair_count,
        "max_checked_relative_residual": max(checked_residuals, default=0.0),
        "sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(generate(parse_args()), indent=2))
