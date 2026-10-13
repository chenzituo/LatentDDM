"""Convert v2 local Dirichlet BVPs into paired mixed Robin-v3 BVPs.

Only train/validation groups are accepted and written. Artificial Dirichlet
traces are converted exactly to Robin data using the stored reference solution;
physical geometry faces remain Dirichlet. No test or A--D group is opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .dataset import SCHEMA_VERSION
from .physics import relative_residual
from .robin_v3 import (
    ROBIN_SCHEMA, assemble_mixed_robin_system, robin_data_from_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--maximum-train", type=int)
    parser.add_argument("--maximum-validation", type=int)
    parser.add_argument("--validation-checks-per-split", type=int, default=64)
    return parser.parse_args()


def convert(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    maximum = {"train": args.maximum_train, "val": args.maximum_validation}
    if args.validation_checks_per_split <= 0:
        raise ValueError("validation-checks-per-split must be positive")
    residual_max = 0.0
    residual_checks = 0
    counts = {}
    with h5py.File(args.source, "r") as source, h5py.File(temporary, "w") as target:
        if str(source.attrs.get("schema_version")) != SCHEMA_VERSION:
            raise ValueError("Robin conversion requires v2 face-Dirichlet source")
        target.attrs.update(source.attrs)
        target.attrs["schema_version"] = ROBIN_SCHEMA
        target.attrs["training_only"] = True
        target.attrs["source_v2"] = str(args.source.resolve())
        target.attrs["pde"] = "-div(kappa grad u)=source; physical Dirichlet + artificial Robin"
        target.attrs["robin_normal_convention"] = "outward: kappa*d_n u + p*u_face = g"
        for split in ("train", "val"):
            if split not in source:
                raise KeyError(f"source lacks {split}")
            source_group = source[split]
            count = source_group["solution"].shape[0]
            if maximum[split] is not None:
                count = min(count, int(maximum[split]))
            counts["validation" if split == "val" else split] = count
            group = target.create_group(split)
            for name, dataset in source_group.items():
                if name in (
                    "dirichlet_face_mask", "dirichlet_face_value",
                    "dirichlet_face_role", "bc_mask", "bc_value", "bc_role",
                ):
                    continue
                kwargs = {}
                if dataset.ndim > 0:
                    kwargs["shape"] = (count, *dataset.shape[1:])
                    kwargs["dtype"] = dataset.dtype
                    if dataset.compression is not None:
                        kwargs["compression"] = dataset.compression
                    if dataset.shuffle:
                        kwargs["shuffle"] = True
                copied = group.create_dataset(name, **kwargs)
                if dataset.ndim > 0:
                    copied[...] = dataset[:count]
                else:
                    copied[()] = dataset[()]
            face_shape = (count, 4, *source_group["solution"].shape[-2:])
            for name, dtype in (
                ("dirichlet_face_mask", "u1"),
                ("dirichlet_face_value", "f4"),
                ("dirichlet_face_role", "u1"),
                ("robin_face_mask", "u1"),
                ("robin_face_value", "f4"),
                ("robin_face_impedance", "f4"),
                ("robin_face_trace", "f4"),
            ):
                group.create_dataset(
                    name, shape=face_shape, dtype=dtype, compression="lzf",
                    shuffle=dtype != "u1", chunks=(1, *face_shape[1:]),
                )
            scalar_shape = (count, 1, *source_group["solution"].shape[-2:])
            for name, dtype in (("bc_mask", "u1"), ("bc_value", "f4"), ("bc_role", "u1")):
                group.create_dataset(
                    name, shape=scalar_shape, dtype=dtype, compression="lzf",
                    shuffle=dtype != "u1", chunks=(1, *scalar_shape[1:]),
                )
            group.attrs.update(source_group.attrs)
            group.attrs["count"] = count
            rng = np.random.default_rng(args.seed + (0 if split == "train" else 1_000_000))
            checked_indices = set(np.linspace(
                0, count - 1,
                num=min(count, args.validation_checks_per_split),
                dtype=np.int64,
            ).tolist())
            for index in range(count):
                mask = source_group["mask"][index, 0] > 0
                kappa = 10.0 ** source_group["logk"][index, 0].astype(np.float64)
                source_field = source_group["source"][index, 0].astype(np.float64)
                solution = source_group["solution"][index, 0].astype(np.float64)
                role = source_group["dirichlet_face_role"][index]
                old_value = source_group["dirichlet_face_value"][index].astype(np.float64)
                physical = (role & 1) > 0
                artificial = (role & 2) > 0
                dx = 2.0 / mask.shape[0]
                center_k = np.broadcast_to(kappa, old_value.shape)
                center_u = np.broadcast_to(solution, old_value.shape)
                theta = np.exp(rng.uniform(np.log(0.1), np.log(10.0), size=old_value.shape))
                impedance = theta * 2.0 * center_k / dx
                robin_value = robin_data_from_trace(
                    center_u, old_value, center_k, impedance, dx,
                )
                dvalue = np.where(physical, old_value, 0.0)
                rvalue = np.where(artificial, robin_value, 0.0)
                pvalue = np.where(artificial, impedance, 0.0)
                trace = np.where(artificial, old_value, 0.0)
                group["dirichlet_face_mask"][index] = physical
                group["dirichlet_face_value"][index] = dvalue
                group["dirichlet_face_role"][index] = physical.astype(np.uint8)
                group["robin_face_mask"][index] = artificial
                group["robin_face_value"][index] = rvalue
                group["robin_face_impedance"][index] = pvalue
                group["robin_face_trace"][index] = trace
                count_face = physical.sum(0)
                cell_bc = (dvalue * physical).sum(0) / np.maximum(count_face, 1)
                group["bc_mask"][index, 0] = count_face > 0
                group["bc_value"][index, 0] = cell_bc
                group["bc_role"][index, 0] = (count_face > 0).astype(np.uint8)
                if index in checked_indices:
                    system = assemble_mixed_robin_system(
                        mask, kappa, source_field, physical, dvalue,
                        artificial, rvalue, pvalue, dx=dx,
                    )
                    residual = relative_residual(solution, system)
                    residual_checks += 1
                    residual_max = max(residual_max, residual)
                    if residual > 2.0e-5:
                        raise RuntimeError(
                            f"paired Robin conversion residual {residual:.3e} "
                            f"at {split}/{index}"
                        )
    temporary.replace(args.output)
    result = {
        "contract": "darcy_robin_v3_paired_corpus_v1",
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "splits": counts,
        "test_subsets_opened": False,
        "reference_residual_checks": residual_checks,
        "maximum_reference_relative_residual": residual_max,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(convert(parse_args()), indent=2))
