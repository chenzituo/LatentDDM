"""Shared full-domain Darcy fine-tuning dataset.

This dataset is deliberately separate from the local-BVP pretraining file.
It contains only geometry-disjoint simple-polygon ``train`` and ``val``
splits.  Controlled A--D domains stay in the evaluation file, so it is not
possible to leak those geometries into either the communicated local model or
the matched global baseline during fine-tuning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import h5py
import numpy as np

from .dataset import (
    _RunningStatistics,
    _empty_group,
    _sample_polygon_mask,
    _write_sample,
    boundary_face_roles,
    boundary_roles,
    signed_distance,
)


SCHEMA_VERSION = "darcy_full_domain_cartesian_v2_face_dirichlet"


@dataclass(frozen=True)
class DarcyFullDomainConfig:
    """Configuration for shared global-domain fine-tuning records."""

    grid: int = 96
    train_shapes: int = 4000
    train_conditions: int = 1
    val_shapes: int = 250
    val_conditions: int = 1
    coefficient_coarse_grid: int = 7
    kappa_floor: float = 1.0e-3
    seed: int = 20260805
    compression: str | None = "lzf"

    def __post_init__(self) -> None:
        if min(
            self.grid,
            self.train_shapes,
            self.train_conditions,
            self.val_shapes,
            self.val_conditions,
            self.coefficient_coarse_grid,
        ) <= 0:
            raise ValueError("grid and dataset counts must be positive")
        if not 0.0 < self.kappa_floor < 1.0:
            raise ValueError("kappa_floor must lie strictly between zero and one")

    @classmethod
    def smoke(cls, seed: int = 20260805) -> "DarcyFullDomainConfig":
        return cls(
            grid=16,
            train_shapes=3,
            train_conditions=2,
            val_shapes=2,
            val_conditions=1,
            coefficient_coarse_grid=4,
            seed=seed,
            compression=None,
        )


def generate_full_domain_hdf5(
    path: Path | str, config: DarcyFullDomainConfig,
) -> Path:
    """Write train/validation-only full-domain data with physical boundaries."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    statistics = _RunningStatistics()
    split_specs = (
        ("train", config.train_shapes, config.train_conditions, 0),
        ("val", config.val_shapes, config.val_conditions, 1_000_000_000),
    )
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["dataset_role"] = "shared_full_domain_finetune"
        handle.attrs["pde"] = "-div(kappa grad u)=source; Dirichlet boundary"
        handle.attrs["grid_size"] = config.grid
        handle.attrs["core"] = config.grid
        handle.attrs["halo"] = 0
        handle.attrs["boundary_role"] = "physical_geometry_only"
        handle.attrs["held_out_domains"] = "A,B,C,D (stored separately)"
        handle.attrs["kappa_floor"] = config.kappa_floor
        handle.attrs["logk_base"] = 10
        handle.attrs["sdf_sign"] = "negative_inside"
        handle.attrs["config"] = json.dumps(asdict(config), sort_keys=True)

        for split, shapes, conditions, offset in split_specs:
            group = _empty_group(
                handle, split, shapes * conditions, config.grid,
                config.compression,
            )
            group.attrs["geometry_count"] = shapes
            group.attrs["conditions_per_geometry"] = conditions
            group.attrs["geometry_disjoint_from_other_splits"] = True
            group.attrs["boundary_role"] = "physical"
            cursor = 0
            for shape_index in range(shapes):
                if shape_index % 250 == 0:
                    print(f"[{split}] full geometry {shape_index}/{shapes}", flush=True)
                geometry_seed = config.seed + offset + shape_index
                mask, family = _sample_polygon_mask(
                    config.grid, np.random.default_rng(geometry_seed),
                )
                dx = 2.0 / config.grid
                sdf = signed_distance(mask, dx)
                # A full-domain record has no crop/halo boundary. Every
                # boundary represented by the mask is a physical boundary.
                bc_mask, bc_role = boundary_roles(
                    mask, outer_is_artificial=False,
                )
                _, face_role = boundary_face_roles(
                    mask, outer_is_artificial=False,
                )
                for condition_index in range(conditions):
                    condition_seed = (
                        config.seed + offset + 10_000_000
                        + shape_index * conditions + condition_index
                    )
                    _write_sample(
                        group, cursor, mask, sdf, bc_mask, bc_role, face_role, family,
                        condition_seed, config,
                        training_range=(split == "train"),
                        statistics=statistics if split == "train" else None,
                    )
                    cursor += 1

        handle.attrs["normalization"] = json.dumps(
            statistics.payload(), sort_keys=True,
        )
        handle.flush()
    temporary.replace(path)
    return path


def inspect_full_domain_dataset(path: Path | str) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        return {
            "schema_version": str(handle.attrs["schema_version"]),
            "dataset_role": str(handle.attrs["dataset_role"]),
            "config": json.loads(str(handle.attrs["config"])),
            "normalization": json.loads(str(handle.attrs["normalization"])),
            "splits": {
                name: {
                    "count": int(handle[name].attrs["count"]),
                    "geometry_count": int(handle[name].attrs["geometry_count"]),
                    "conditions_per_geometry": int(
                        handle[name].attrs["conditions_per_geometry"]
                    ),
                    "grid_size": int(handle[name].attrs["grid_size"]),
                }
                for name in ("train", "val")
            },
        }
