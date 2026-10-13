"""Fixed-resolution Darcy domains for compositional scale extension.

The 96x96 source benchmark contains a 3x3 cover with a 32-cell core.  This
module holds the source grid spacing fixed and grows the physical square to
4x4, 8x8, or 16x16 covers.  Consequently every local call retains the same
physical footprint while the number of reusable local units changes.

Target geometries are connected perforated and concave domains.  Their holes
and boundary features have approximately fixed physical size, so increasing
the box creates more local geometric structure instead of magnifying one
shape.  Train, validation, and test geometries use disjoint deterministic
seeds; the test split is generated but is not used for model selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage

from .dataset import (
    _RunningStatistics,
    _empty_group,
    _write_sample,
    boundary_face_roles,
    boundary_roles,
    signed_distance,
)


Array = np.ndarray
SCHEMA_VERSION = "darcy_domain_extension_cartesian_v2_fixed_dx"
DATASET_ROLE = "shared_full_domain_finetune"


@dataclass(frozen=True)
class DarcyDomainExtensionConfig:
    """Configuration for one target cover scale."""

    blocks: int
    core: int = 32
    source_blocks: int = 3
    source_extent: float = 2.0
    train_shapes: int = 64
    train_conditions: int = 1
    val_shapes: int = 64
    val_conditions: int = 1
    test_shapes: int = 64
    test_conditions: int = 1
    coefficient_coarse_grid: int | None = None
    kappa_floor: float = 1.0e-3
    seed: int = 20260815
    compression: str | None = "lzf"

    def __post_init__(self) -> None:
        if self.blocks not in (4, 8, 16):
            raise ValueError("extension blocks must be one of 4, 8, or 16")
        if min(
            self.core, self.source_blocks, self.train_shapes,
            self.train_conditions, self.val_shapes, self.val_conditions,
            self.test_shapes, self.test_conditions,
        ) <= 0:
            raise ValueError("grid and split counts must be positive")
        if self.source_extent <= 0.0:
            raise ValueError("source extent must be positive")
        if not 0.0 < self.kappa_floor < 1.0:
            raise ValueError("kappa_floor must lie strictly between zero and one")
        if self.coefficient_coarse_grid is not None \
                and self.coefficient_coarse_grid <= 1:
            raise ValueError("coefficient coarse grid must exceed one")

    @property
    def grid(self) -> int:
        return self.blocks * self.core

    @property
    def source_grid(self) -> int:
        return self.source_blocks * self.core

    @property
    def grid_spacing(self) -> float:
        return self.source_extent / self.source_grid

    @property
    def physical_extent(self) -> float:
        return self.grid * self.grid_spacing

    @property
    def effective_coarse_grid(self) -> int:
        if self.coefficient_coarse_grid is not None:
            return self.coefficient_coarse_grid
        # Preserve the source field's approximate physical correlation length.
        return max(3, int(round(7 * self.blocks / self.source_blocks)))

    @classmethod
    def smoke(
        cls, blocks: int = 4, seed: int = 20260815,
    ) -> "DarcyDomainExtensionConfig":
        return cls(
            blocks=blocks, core=4, source_blocks=3,
            train_shapes=2, val_shapes=1, test_shapes=1,
            coefficient_coarse_grid=4, seed=seed, compression=None,
        )


def _physical_grid(config: DarcyDomainExtensionConfig) -> tuple[Array, Array]:
    extent = config.physical_extent
    axis = -0.5 * extent + (
        np.arange(config.grid, dtype=np.float64) + 0.5
    ) * config.grid_spacing
    return np.meshgrid(axis, axis, indexing="ij")


def _largest_component(mask: Array) -> tuple[Array, int]:
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask, 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    label = int(np.argmax(sizes))
    return labels == label, count


def _outer_domain(
    yy: Array, xx: Array, extent: float, rng: np.random.Generator,
) -> Array:
    """A connected corrugated outer boundary with one or two concave bays."""

    half = 0.5 * extent
    angle = np.arctan2(yy, xx)
    radius = np.hypot(xx / half, yy / half)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    frequency = int(rng.integers(4, 8))
    outer = radius < 0.90 + 0.035 * np.cos(frequency * angle + phase)

    # Open notches create concavity without introducing a new component.  The
    # feature depth is fixed in physical units across target scales.
    notch_width = float(rng.uniform(0.18, 0.30))
    notch_depth = float(rng.uniform(0.28, 0.46))
    notch_y = float(rng.uniform(-0.28 * extent, 0.28 * extent))
    right_notch = (
        (xx > half * 0.90 - notch_depth)
        & (np.abs(yy - notch_y) < notch_width)
    )
    outer &= ~right_notch
    if rng.random() < 0.5:
        notch_y2 = float(rng.uniform(-0.28 * extent, 0.28 * extent))
        left_notch = (
            (xx < -half * 0.90 + notch_depth)
            & (np.abs(yy - notch_y2) < 0.8 * notch_width)
        )
        outer &= ~left_notch
    return outer


def _carve_fixed_scale_holes(
    mask: Array,
    yy: Array,
    xx: Array,
    config: DarcyDomainExtensionConfig,
    rng: np.random.Generator,
    family_index: int,
) -> Array:
    """Carve separated fixed-physical-size holes on a jittered lattice."""

    extent = config.physical_extent
    spacing = float(rng.uniform(0.58, 0.72))
    margin = 0.42
    centers = np.arange(
        -0.5 * extent + margin, 0.5 * extent - margin + 1.0e-12, spacing,
    )
    holes = np.zeros_like(mask)
    for row, center_y in enumerate(centers):
        for column, center_x in enumerate(centers):
            # Stagger and subsample the lattice to vary global topology while
            # retaining the same local geometric scale.
            if (row + 2 * column + family_index) % 5 == 0:
                continue
            jitter_x, jitter_y = rng.uniform(-0.055, 0.055, size=2)
            cx = center_x + (0.5 * spacing if row % 2 else 0.0) + jitter_x
            cy = center_y + jitter_y
            if abs(cx) > 0.5 * extent - margin or abs(cy) > 0.5 * extent - margin:
                continue
            rx = float(rng.uniform(0.075, 0.125))
            ry = float(rng.uniform(0.075, 0.125))
            if family_index == 0:
                hole = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1.0
            elif family_index == 1:
                rotation = float(rng.uniform(-0.7, 0.7))
                xr = np.cos(rotation) * (xx - cx) + np.sin(rotation) * (yy - cy)
                yr = -np.sin(rotation) * (xx - cx) + np.cos(rotation) * (yy - cy)
                hole = (np.abs(xr) / rx + np.abs(yr) / ry) < 1.0
            else:
                theta = np.arctan2(yy - cy, xx - cx)
                radial = np.hypot((xx - cx) / rx, (yy - cy) / ry)
                hole = radial < 0.82 + 0.16 * np.cos(4.0 * theta)
            holes |= hole
    return mask & ~holes


def sample_extension_mask(
    config: DarcyDomainExtensionConfig,
    rng: np.random.Generator,
) -> tuple[Array, str]:
    """Sample a connected target geometry with scale-local boundary features."""

    yy, xx = _physical_grid(config)
    family_names = (
        "perforated_ellipse",
        "perforated_diamond",
        "perforated_star",
    )
    for _ in range(50):
        family_index = int(rng.integers(0, len(family_names)))
        mask = _outer_domain(yy, xx, config.physical_extent, rng)
        mask = _carve_fixed_scale_holes(
            mask, yy, xx, config, rng, family_index,
        )
        largest, components = _largest_component(mask)
        # Removed material can create isolated one-cell fragments near the
        # corrugated exterior.  Retaining only the principal connected fluid
        # region is deterministic and enforces the PDE contract.
        mask = largest
        active_fraction = float(mask.mean())
        if (
            components >= 1
            and 0.42 <= active_fraction <= 0.90
            and bool(ndimage.binary_erosion(mask).any())
        ):
            return mask, family_names[family_index]
    raise RuntimeError("failed to sample a valid fixed-scale extension domain")


def _split_specs(config: DarcyDomainExtensionConfig):
    return (
        ("train", config.train_shapes, config.train_conditions, 0),
        ("val", config.val_shapes, config.val_conditions, 1_000_000_000),
        ("test", config.test_shapes, config.test_conditions, 2_000_000_000),
    )


def generate_domain_extension_hdf5(
    path: Path | str,
    config: DarcyDomainExtensionConfig,
) -> Path:
    """Write one fixed-resolution target scale atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    statistics = _RunningStatistics()
    condition_config = type("ConditionConfig", (), {
        "coefficient_coarse_grid": config.effective_coarse_grid,
        "kappa_floor": config.kappa_floor,
    })()
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["dataset_role"] = DATASET_ROLE
        handle.attrs["pde"] = "-div(kappa grad u)=source; Dirichlet boundary"
        handle.attrs["scaling_mode"] = "fixed_grid_spacing_physical_growth"
        handle.attrs["source_grid"] = config.source_grid
        handle.attrs["source_blocks"] = config.source_blocks
        handle.attrs["source_extent"] = config.source_extent
        handle.attrs["grid_size"] = config.grid
        handle.attrs["blocks"] = config.blocks
        handle.attrs["core"] = config.core
        handle.attrs["halo"] = 0
        handle.attrs["grid_spacing"] = config.grid_spacing
        handle.attrs["physical_extent"] = config.physical_extent
        handle.attrs["patch_size"] = config.core
        handle.attrs["geometry_policy"] = (
            "connected corrugated domain with fixed-physical-scale perforations"
        )
        handle.attrs["split_policy"] = "geometry-disjoint deterministic seeds"
        handle.attrs["test_policy"] = "generated protected split; validation selects models"
        handle.attrs["kappa_floor"] = config.kappa_floor
        handle.attrs["logk_base"] = 10
        handle.attrs["sdf_sign"] = "negative_inside"
        handle.attrs["config"] = json.dumps(asdict(config), sort_keys=True)

        for split, shapes, conditions, offset in _split_specs(config):
            group = _empty_group(
                handle, split, shapes * conditions, config.grid,
                config.compression,
            )
            group.attrs["geometry_count"] = shapes
            group.attrs["conditions_per_geometry"] = conditions
            group.attrs["geometry_disjoint_from_other_splits"] = True
            group.attrs["boundary_role"] = "physical"
            group.attrs["grid_spacing"] = config.grid_spacing
            group.attrs["physical_extent"] = config.physical_extent
            cursor = 0
            for shape_index in range(shapes):
                if shape_index % 16 == 0:
                    print(
                        f"[{split}] extension geometry {shape_index}/{shapes}",
                        flush=True,
                    )
                geometry_seed = config.seed + offset + shape_index
                mask, family = sample_extension_mask(
                    config, np.random.default_rng(geometry_seed),
                )
                sdf = signed_distance(mask, config.grid_spacing)
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
                        group, cursor, mask, sdf, bc_mask, bc_role, face_role,
                        family, condition_seed, condition_config,
                        training_range=(split == "train"),
                        statistics=statistics if split == "train" else None,
                        dx=config.grid_spacing,
                    )
                    cursor += 1

        handle.attrs["normalization"] = json.dumps(
            statistics.payload(), sort_keys=True,
        )
        handle.flush()
    temporary.replace(path)
    return path


def inspect_domain_extension_dataset(path: Path | str) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        return {
            "schema_version": str(handle.attrs["schema_version"]),
            "dataset_role": str(handle.attrs["dataset_role"]),
            "scaling_mode": str(handle.attrs["scaling_mode"]),
            "blocks": int(handle.attrs["blocks"]),
            "grid_size": int(handle.attrs["grid_size"]),
            "grid_spacing": float(handle.attrs["grid_spacing"]),
            "physical_extent": float(handle.attrs["physical_extent"]),
            "normalization": json.loads(str(handle.attrs["normalization"])),
            "splits": {
                name: {
                    "count": int(handle[name].attrs["count"]),
                    "geometry_count": int(handle[name].attrs["geometry_count"]),
                    "grid_size": int(handle[name].attrs["grid_size"]),
                }
                for name in ("train", "val", "test")
            },
        }
