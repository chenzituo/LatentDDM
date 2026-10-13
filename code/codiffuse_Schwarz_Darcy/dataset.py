"""Deterministic complex-boundary Darcy dataset generation.

Training samples are local Cartesian boundary-value problems on simple
3--12-vertex polygons, matching the basic-shape protocol of Huang et al. while
using the Airfoil case's 32-cell core plus four-cell halo.  Evaluation domains
A--D hold out holes, corners, concavity, and graph diameter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
from scipy import ndimage

from codiffuse_Schwarz_Darcy.physics import FACE_OFFSETS, solve_darcy


Array = np.ndarray
SCHEMA_VERSION = "darcy_complex_cartesian_v2_face_dirichlet"


@dataclass(frozen=True)
class DarcyDatasetConfig:
    core: int = 32
    halo: int = 4
    train_shapes: int = 4000
    train_conditions: int = 10
    val_shapes: int = 250
    val_conditions: int = 10
    test_shapes: int = 100
    test_conditions: int = 1
    domain_conditions: int = 100
    eval_grid: int = 128
    large_eval_grid: int = 192
    coefficient_coarse_grid: int = 7
    kappa_floor: float = 1.0e-3
    seed: int = 20260805
    compression: str | None = "lzf"

    @property
    def local_grid(self) -> int:
        return self.core + 2 * self.halo

    def __post_init__(self) -> None:
        integer_fields = (
            self.core, self.halo, self.train_shapes, self.train_conditions,
            self.val_shapes, self.val_conditions, self.test_shapes,
            self.test_conditions, self.domain_conditions, self.eval_grid,
            self.large_eval_grid, self.coefficient_coarse_grid,
        )
        if self.core <= 0 or self.halo < 0 or any(value <= 0 for value in integer_fields[2:]):
            raise ValueError("dataset sizes must be positive and halo nonnegative")
        if not 0.0 < self.kappa_floor < 1.0:
            raise ValueError("kappa_floor must lie strictly between zero and one")

    @classmethod
    def smoke(cls, seed: int = 20260805) -> "DarcyDatasetConfig":
        return cls(
            core=12, halo=2,
            train_shapes=3, train_conditions=2,
            val_shapes=2, val_conditions=1,
            test_shapes=2, test_conditions=1,
            domain_conditions=1, eval_grid=24, large_eval_grid=28,
            coefficient_coarse_grid=4, seed=seed, compression=None,
        )


def coordinate_grid(size: int) -> tuple[Array, Array]:
    # Cell centers on a length-two canonical box.  This agrees with the
    # finite-volume spacing dx=2/size used by the reference operator.
    axis = -1.0 + (np.arange(size, dtype=np.float64) + 0.5) * (2.0 / size)
    return np.meshgrid(axis, axis, indexing="ij")


def _points_inside_polygon(y: Array, x: Array, vertices: Array) -> Array:
    inside = np.zeros(y.shape, dtype=bool)
    vx = vertices[:, 0]
    vy = vertices[:, 1]
    previous = len(vertices) - 1
    for current in range(len(vertices)):
        yi = vy[current]
        yj = vy[previous]
        xi = vx[current]
        xj = vx[previous]
        crossing = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1.0e-30) + xi
        )
        inside ^= crossing
        previous = current
    return inside


def _boundary_mask(mask: Array) -> Array:
    padded = np.pad(mask, 1, constant_values=False)
    interior_neighbors = (
        padded[:-2, 1:-1] & padded[2:, 1:-1]
        & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    return mask & ~interior_neighbors


def signed_distance(mask: Array, dx: float) -> Array:
    # Negative is inside, matching the geometry convention in the protocol.
    return (
        ndimage.distance_transform_edt(~mask)
        - ndimage.distance_transform_edt(mask)
    ) * dx


def boundary_normals(sdf: Array, bc_mask: Array, dx: float) -> Array:
    edge_order = 2 if min(sdf.shape) >= 3 else 1
    grad_y, grad_x = np.gradient(sdf, dx, dx, edge_order=edge_order)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    normals = np.stack((grad_x, grad_y), axis=0) / np.maximum(magnitude[None], 1.0e-12)
    normals *= bc_mask[None]
    return normals


def _sample_polygon_mask(size: int, rng: np.random.Generator) -> tuple[Array, str]:
    y, x = coordinate_grid(size)
    for _ in range(100):
        vertices = int(rng.integers(3, 13))
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, vertices))
        radii = rng.uniform(0.55, 1.15, vertices)
        scale_x = rng.uniform(0.75, 1.25)
        scale_y = rng.uniform(0.75, 1.25)
        center_x, center_y = rng.uniform(-0.30, 0.30, size=2)
        polygon = np.stack((
            center_x + scale_x * radii * np.cos(angles),
            center_y + scale_y * radii * np.sin(angles),
        ), axis=1)
        mask = _points_inside_polygon(y, x, polygon)
        active_fraction = float(mask.mean())
        labels, components = ndimage.label(mask)
        del labels
        if 0.20 <= active_fraction <= 0.94 and components == 1 and _boundary_mask(mask).any():
            return mask, f"polygon_{vertices}"
    raise RuntimeError("failed to sample a valid connected polygon")


def _linear_random_field(
    size: int, coarse_size: int, rng: np.random.Generator,
) -> Array:
    coarse = rng.uniform(0.0, 1.0, size=(coarse_size, coarse_size))
    coarse_axis = np.arange(coarse_size, dtype=np.float64)
    fine_axis = np.linspace(0.0, coarse_size - 1.0, size)
    along_x = np.stack([
        np.interp(fine_axis, coarse_axis, row) for row in coarse
    ], axis=0)
    return np.stack([
        np.interp(fine_axis, coarse_axis, along_x[:, column])
        for column in range(size)
    ], axis=1)


def boundary_roles(mask: Array, *, outer_is_artificial: bool) -> tuple[Array, Array]:
    """Return boundary mask and role bits: 1 physical, 2 artificial."""
    height, width = mask.shape
    physical = np.zeros_like(mask)
    padded = np.pad(mask, 1, constant_values=False)
    # Inactive neighbors within the represented window are physical geometry.
    physical |= mask & ~padded[:-2, 1:-1]
    physical |= mask & ~padded[2:, 1:-1]
    physical |= mask & ~padded[1:-1, :-2]
    physical |= mask & ~padded[1:-1, 2:]
    artificial = np.zeros_like(mask)
    outer = np.zeros_like(mask)
    outer[0] = outer[-1] = True
    outer[:, 0] = outer[:, -1] = True
    outer &= mask
    if outer_is_artificial:
        artificial |= outer
        # Padding-created missing neighbors are artificial, not a solid wall.
        physical &= ~outer
    else:
        physical |= outer
    role = physical.astype(np.uint8) + 2 * artificial.astype(np.uint8)
    return role > 0, role


def boundary_face_roles(mask: Array, *, outer_is_artificial: bool) -> tuple[Array, Array]:
    """Return face-resolved Dirichlet masks and physical/artificial role bits.

    Channels are north, south, west, east.  Role bit 1 denotes a physical
    geometry face and bit 2 an artificial window face.  Unlike the legacy
    cellwise representation, corners retain both directional traces.
    """
    domain = np.asarray(mask, dtype=bool)
    if domain.ndim != 2:
        raise ValueError("boundary face roles require a two-dimensional mask")
    height, width = domain.shape
    result = np.zeros((4, height, width), dtype=np.uint8)
    for face, (dy, dx) in enumerate(FACE_OFFSETS):
        neighbor = np.zeros_like(domain)
        if dy == -1:
            neighbor[1:] = domain[:-1]
        elif dy == 1:
            neighbor[:-1] = domain[1:]
        elif dx == -1:
            neighbor[:, 1:] = domain[:, :-1]
        else:
            neighbor[:, :-1] = domain[:, 1:]
        exposed = domain & ~neighbor
        out_of_grid = np.zeros_like(domain)
        if dy == -1:
            out_of_grid[0] = True
        elif dy == 1:
            out_of_grid[-1] = True
        elif dx == -1:
            out_of_grid[:, 0] = True
        else:
            out_of_grid[:, -1] = True
        artificial = exposed & out_of_grid & outer_is_artificial
        physical = exposed & ~artificial
        result[face, physical] = 1
        result[face, artificial] = 2
    return result > 0, result


def _sample_conditions(
    mask: Array,
    face_role: Array,
    rng: np.random.Generator,
    config: DarcyDatasetConfig,
    *,
    training_range: bool,
) -> tuple[Array, Array, Array, Array, Array]:
    size = mask.shape[0]
    raw_kappa = _linear_random_field(size, config.coefficient_coarse_grid, rng)
    kappa = config.kappa_floor + (1.0 - config.kappa_floor) * raw_kappa
    source = _linear_random_field(size, config.coefficient_coarse_grid, rng) * mask
    physical_bc = _linear_random_field(size, config.coefficient_coarse_grid, rng)
    artificial_bc = np.stack(tuple(
        _linear_random_field(size, config.coefficient_coarse_grid, rng)
        for _ in FACE_OFFSETS
    ))
    amplitude = float(rng.uniform(0.3, 1.0)) if training_range else 1.0
    physical_bc *= amplitude
    artificial_bc *= amplitude
    physical_face = (face_role & 1) > 0
    artificial_face = (face_role & 2) > 0
    face_value = np.zeros(face_role.shape, dtype=np.float64)
    face_value[physical_face] = np.broadcast_to(
        physical_bc, face_role.shape,
    )[physical_face]
    face_value[artificial_face] = artificial_bc[artificial_face]
    # The legacy channel is retained for old checkpoints.  It is the mean of
    # all exposed directional traces at a cell and is not used by the v2 FV
    # solve.  Face-aware callers consume ``face_value`` directly.
    face_count = (face_role > 0).sum(0)
    face_sum = (face_value * (face_role > 0)).sum(0)
    bc_value = face_sum / np.maximum(face_count, 1)
    return kappa, source, bc_value, face_role > 0, face_value


def _complex_domain(name: str, size: int) -> tuple[Array, str]:
    y, x = coordinate_grid(size)
    if name == "A":
        left = ((x + 0.36) / 0.58) ** 2 + (y / 0.72) ** 2 < 1.0
        right = ((x - 0.36) / 0.58) ** 2 + (y / 0.72) ** 2 < 1.0
        waist = (np.abs(x) < 0.52) & (np.abs(y) < 0.30)
        mask = left | right | waist
        family = "A_simply_connected_concave"
    elif name == "B":
        outer = (np.abs(x) < 0.90) & (np.abs(y) < 0.76)
        holes = (
            ((x + 0.38) / 0.18) ** 2 + ((y + 0.04) / 0.25) ** 2 < 1.0
        ) | (
            ((x - 0.38) / 0.18) ** 2 + ((y - 0.04) / 0.25) ** 2 < 1.0
        )
        mask = outer & ~holes
        family = "B_two_holes"
    elif name == "C":
        radius = np.hypot(x, y)
        angle = np.arctan2(y, x)
        outer = radius < 0.76 + 0.12 * np.cos(5.0 * angle)
        xr = 0.86 * (x - 0.12) + 0.50 * (y + 0.04)
        yr = -0.50 * (x - 0.12) + 0.86 * (y + 0.04)
        cornered_hole = (np.abs(xr) < 0.20) & (np.abs(yr) < 0.14)
        mask = outer & ~cornered_hole
        family = "C_cornered_hole"
    elif name == "D":
        spine = (np.abs(y) < 0.24) & (np.abs(x) < 0.94)
        upper = (x < -0.52) & (y > -0.24) & (y < 0.72)
        lower = (x > 0.52) & (y > -0.72) & (y < 0.24)
        outer = spine | upper | lower
        holes = np.zeros_like(outer)
        for center_x in (-0.66, -0.22, 0.22, 0.66):
            holes |= (x - center_x) ** 2 + y**2 < 0.105**2
        mask = outer & ~holes
        family = "D_large_diameter_four_holes"
    else:
        raise ValueError(f"unknown complex domain {name!r}")
    labels, components = ndimage.label(mask)
    del labels
    if components != 1:
        raise RuntimeError(f"complex domain {name} is disconnected")
    return mask, family


def _empty_group(
    handle: h5py.File,
    name: str,
    count: int,
    size: int,
    compression: str | None,
) -> h5py.Group:
    group = handle.create_group(name)
    single = (count, 1, size, size)
    for key, dtype in (
        ("mask", "u1"), ("logk", "f4"), ("source", "f4"),
        ("bc_mask", "u1"), ("bc_value", "f4"), ("solution", "f4"),
        ("sdf", "f4"), ("bc_role", "u1"),
    ):
        group.create_dataset(
            key, shape=single, dtype=dtype, compression=compression,
            chunks=(1, 1, size, size),
        )
    group.create_dataset(
        "normals", shape=(count, 2, size, size), dtype="f4", compression=compression,
        chunks=(1, 2, size, size),
    )
    face_shape = (count, 4, size, size)
    for key, dtype in (
        ("dirichlet_face_mask", "u1"),
        ("dirichlet_face_value", "f4"),
        ("dirichlet_face_role", "u1"),
    ):
        group.create_dataset(
            key, shape=face_shape, dtype=dtype, compression=compression,
            chunks=(1, 4, size, size),
        )
    group.create_dataset("seed", shape=(count,), dtype="i8")
    group.create_dataset("family", shape=(count,), dtype=h5py.string_dtype("utf-8"))
    group.attrs["count"] = count
    group.attrs["grid_size"] = size
    return group


class _RunningStatistics:
    def __init__(self) -> None:
        self.values = {
            key: [0, 0.0, 0.0]
            for key in ("logk", "source", "bc_value", "solution")
        }

    def update(self, key: str, field: Array, selection: Array) -> None:
        values = np.asarray(field, dtype=np.float64)[np.asarray(selection, dtype=bool)]
        if values.size == 0:
            return
        record = self.values[key]
        record[0] += int(values.size)
        record[1] += float(values.sum())
        record[2] += float((values * values).sum())

    def payload(self) -> dict[str, dict[str, float]]:
        result = {}
        for key, (count, total, total_sq) in self.values.items():
            mean = total / max(count, 1)
            variance = max(total_sq / max(count, 1) - mean * mean, 1.0e-24)
            result[key] = {
                "mean": float(mean), "std": float(np.sqrt(variance)),
                "count": int(count),
            }
        return result


def _write_sample(
    group: h5py.Group,
    index: int,
    mask: Array,
    sdf: Array,
    bc_mask: Array,
    bc_role: Array,
    face_role: Array,
    family: str,
    seed: int,
    config: DarcyDatasetConfig,
    *,
    training_range: bool,
    statistics: _RunningStatistics | None,
    dx: float | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    kappa, source, bc_value, face_mask, face_value = _sample_conditions(
        mask, face_role, rng, config, training_range=training_range,
    )
    dx = 2.0 / mask.shape[0] if dx is None else float(dx)
    if dx <= 0.0:
        raise ValueError("sample grid spacing must be positive")
    solution, system = solve_darcy(
        mask, kappa, source, bc_value, dx=dx,
        dirichlet_face_mask=face_mask,
        dirichlet_face_value=face_value,
    )
    discrete_residual = np.linalg.norm(
        system.rhs - system.matrix @ solution[mask]
    ) / max(np.linalg.norm(system.rhs), 1.0e-12)
    if discrete_residual > 1.0e-8:
        raise RuntimeError(f"reference residual {discrete_residual:.3e} is too large")
    logk = np.log10(kappa)
    normals = boundary_normals(sdf, bc_mask, dx)
    fields = {
        "mask": mask.astype(np.uint8)[None],
        "logk": logk.astype(np.float32)[None],
        "source": source.astype(np.float32)[None],
        "bc_mask": bc_mask.astype(np.uint8)[None],
        "bc_value": bc_value.astype(np.float32)[None],
        "solution": solution.astype(np.float32)[None],
        "sdf": sdf.astype(np.float32)[None],
        "bc_role": bc_role.astype(np.uint8)[None],
        "normals": normals.astype(np.float32),
        "dirichlet_face_mask": face_mask.astype(np.uint8),
        "dirichlet_face_value": face_value.astype(np.float32),
        "dirichlet_face_role": face_role.astype(np.uint8),
    }
    for key, value in fields.items():
        group[key][index] = value
    group["seed"][index] = int(seed)
    group["family"][index] = family
    if statistics is not None:
        statistics.update("logk", logk, mask)
        statistics.update("source", source, mask)
        statistics.update("bc_value", bc_value, bc_mask)
        statistics.update("solution", solution, mask)


def _split_spec(
    config: DarcyDatasetConfig, *, training_only: bool = False,
) -> Iterator[tuple[str, int, int, int]]:
    yield "train", config.train_shapes, config.train_conditions, 0
    yield "val", config.val_shapes, config.val_conditions, 1_000_000_000
    if not training_only:
        yield "test", config.test_shapes, config.test_conditions, 2_000_000_000


def generate_hdf5(
    path: Path | str, config: DarcyDatasetConfig, *, training_only: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    statistics = _RunningStatistics()
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["training_only"] = bool(training_only)
        handle.attrs["pde"] = "-div(kappa grad u)=source; Dirichlet boundary"
        handle.attrs["grid_size"] = config.local_grid
        handle.attrs["core"] = config.core
        handle.attrs["halo"] = config.halo
        handle.attrs["kappa_floor"] = config.kappa_floor
        handle.attrs["logk_base"] = 10
        handle.attrs["sdf_sign"] = "negative_inside"
        handle.attrs["config"] = json.dumps(asdict(config), sort_keys=True)

        for split, shapes, conditions, offset in _split_spec(
            config, training_only=training_only,
        ):
            count = shapes * conditions
            group = _empty_group(
                handle, split, count, config.local_grid, config.compression,
            )
            cursor = 0
            for shape_index in range(shapes):
                if shape_index % 250 == 0:
                    print(
                        f"[{split}] geometry {shape_index}/{shapes}", flush=True,
                    )
                geometry_seed = config.seed + offset + shape_index
                mask, family = _sample_polygon_mask(
                    config.local_grid, np.random.default_rng(geometry_seed),
                )
                dx = 2.0 / config.local_grid
                sdf = signed_distance(mask, dx)
                bc_mask, bc_role = boundary_roles(mask, outer_is_artificial=True)
                _, face_role = boundary_face_roles(mask, outer_is_artificial=True)
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

        if not training_only:
            domains = handle.create_group("domains")
            for domain_index, name in enumerate(("A", "B", "C", "D")):
                size = config.large_eval_grid if name == "D" else config.eval_grid
                group = _empty_group(
                    domains, name, config.domain_conditions, size, config.compression,
                )
                mask, family = _complex_domain(name, size)
                dx = 2.0 / size
                sdf = signed_distance(mask, dx)
                bc_mask, bc_role = boundary_roles(mask, outer_is_artificial=False)
                _, face_role = boundary_face_roles(mask, outer_is_artificial=False)
                for condition_index in range(config.domain_conditions):
                    condition_seed = (
                        config.seed + 3_000_000_000 + domain_index * 10_000_000
                        + condition_index
                    )
                    _write_sample(
                        group, condition_index, mask, sdf, bc_mask, bc_role,
                        face_role, family, condition_seed, config,
                        training_range=False, statistics=None,
                    )
                group.attrs["official_huang_geometry"] = False
                group.attrs["geometry_note"] = (
                    "Controlled topology analogue; replace with official SNI mesh "
                    "rasterization for the exact Huang A/B/C comparison."
                )

        handle.attrs["normalization"] = json.dumps(
            statistics.payload(), sort_keys=True,
        )
        handle.flush()
    temporary.replace(path)
    return path


def inspect_dataset(path: Path | str) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        split_names = tuple(
            name for name in ("train", "val", "test") if name in handle
        )
        return {
            "schema_version": str(handle.attrs["schema_version"]),
            "config": json.loads(str(handle.attrs["config"])),
            "normalization": json.loads(str(handle.attrs["normalization"])),
            "splits": {
                name: {
                    "count": int(handle[name].attrs["count"]),
                    "grid_size": int(handle[name].attrs["grid_size"]),
                }
                for name in split_names
            },
            "domains": ({
                name: {
                    "count": int(handle[f"domains/{name}"].attrs["count"]),
                    "grid_size": int(handle[f"domains/{name}"].attrs["grid_size"]),
                }
                for name in ("A", "B", "C", "D")
            } if "domains" in handle else {}),
        }
