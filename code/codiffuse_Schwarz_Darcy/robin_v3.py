"""Distinct mixed Dirichlet/Robin finite-volume and neural-input contract.

Robin channels use the outward-normal convention

    kappa * partial_n u + p * u_face = g,

at a boundary face half a cell from the active cell center.  This module is
deliberately separate from the v2 Dirichlet contract: a Dirichlet checkpoint
cannot load into the wider Robin adapter/model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
import torch

from codiffuse_Schwarz_Airfoil.code_transfer.adapter import (
    ConditionBatch, PhysicalTransition,
)

from .adapter import (
    DarcyAdapter, DarcyStatistics, FACE_DIRICHLET_FIELDS, H5DarcyDataset,
    REQUIRED_FIELDS, _face_field, face_condition_features,
)
from .physics import DarcySystem, FACE_OFFSETS, harmonic_mean


Array = np.ndarray
Tensor = torch.Tensor
ROBIN_SCHEMA = "darcy_complex_cartesian_v3_mixed_robin"
ROBIN_FIELDS = (
    "robin_face_mask", "robin_face_value", "robin_face_impedance",
    "robin_face_trace",
)
ROBIN_LOCAL_CONTRACT = "darcy_robin_v3_boundary_to_solution_v1"


def assemble_mixed_robin_system(
    mask: Array,
    kappa: Array,
    source: Array,
    dirichlet_face_mask: Array,
    dirichlet_face_value: Array,
    robin_face_mask: Array,
    robin_face_value: Array,
    robin_face_impedance: Array,
    *,
    dx: float | None = None,
) -> DarcySystem:
    """Assemble SPD FV system with disjoint Dirichlet and Robin faces."""
    mask = np.asarray(mask, dtype=bool)
    kappa = np.asarray(kappa, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    arrays = [
        np.asarray(value, dtype=np.float64) for value in (
            dirichlet_face_mask, dirichlet_face_value, robin_face_mask,
            robin_face_value, robin_face_impedance,
        )
    ]
    dmask, dvalue, rmask, rvalue, impedance = arrays
    expected = (4, *mask.shape)
    if any(value.shape != expected for value in arrays):
        raise ValueError(f"mixed face tensors must have shape {expected}")
    dmask, rmask = dmask > 0.5, rmask > 0.5
    if np.any(dmask & rmask):
        raise ValueError("Dirichlet and Robin face masks must be disjoint")
    if np.any(impedance[rmask] <= 0.0) or not all(
        np.isfinite(value).all() for value in (kappa, source, dvalue, rvalue, impedance)
    ):
        raise ValueError("mixed Robin fields must be finite and impedance positive")
    height, width = mask.shape
    dx = 2.0 / max(height, width) if dx is None else float(dx)
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    coordinates = np.argwhere(mask)
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[mask] = np.arange(len(coordinates))
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    rhs = np.zeros(len(coordinates), dtype=np.float64)
    inv_h2 = 1.0 / dx**2
    for row, (yy, xx) in enumerate(coordinates):
        y, x = int(yy), int(xx)
        center_k = float(kappa[y, x])
        diagonal = 0.0
        rhs[row] = float(source[y, x])
        for face, (dy, ddx) in enumerate(FACE_OFFSETS):
            yn, xn = y + dy, x + ddx
            if 0 <= yn < height and 0 <= xn < width and mask[yn, xn]:
                transmissibility = harmonic_mean(center_k, float(kappa[yn, xn])) * inv_h2
                diagonal += transmissibility
                rows.append(row); columns.append(int(index[yn, xn])); values.append(-transmissibility)
                continue
            transmissibility = 2.0 * center_k * inv_h2
            if dmask[face, y, x]:
                diagonal += transmissibility
                rhs[row] += transmissibility * float(dvalue[face, y, x])
            elif rmask[face, y, x]:
                p = float(impedance[face, y, x])
                denominator = p + 2.0 * center_k / dx
                diagonal += transmissibility * p / denominator
                rhs[row] += transmissibility * float(rvalue[face, y, x]) / denominator
            else:
                raise ValueError(
                    f"uncovered exposed mixed face {face} at ({y},{x})"
                )
        rows.append(row); columns.append(row); values.append(diagonal)
    matrix = sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(coordinates), len(coordinates)),
    )
    return DarcySystem(matrix, rhs, index, coordinates, dx)


def solve_mixed_robin(**kwargs) -> tuple[Array, DarcySystem]:
    system = assemble_mixed_robin_system(**kwargs)
    values = sparse_linalg.spsolve(system.matrix, system.rhs)
    if not np.isfinite(values).all():
        raise RuntimeError("mixed Robin solve produced non-finite values")
    solution = np.zeros(system.index.shape, dtype=np.float64)
    solution[system.index >= 0] = values
    return solution, system


def robin_data_from_trace(
    center_value: Array,
    face_trace: Array,
    center_kappa: Array,
    impedance: Array,
    dx: float,
) -> Array:
    """Return g for a known face trace and adjacent cell-center solution."""
    return (
        2.0 * np.asarray(center_kappa) / dx
        * (np.asarray(face_trace) - np.asarray(center_value))
        + np.asarray(impedance) * np.asarray(face_trace)
    )


def robin_exchange_from_neighbor(
    center_value: Array,
    neighbor_value: Array,
    center_kappa: Array,
    neighbor_kappa: Array,
    impedance: Array,
    dx: float,
) -> tuple[Array, Array]:
    """Discrete Mao-style neighbor trace and Robin datum for one face."""
    denominator = np.asarray(center_kappa) + np.asarray(neighbor_kappa)
    trace = (
        np.asarray(center_kappa) * np.asarray(center_value)
        + np.asarray(neighbor_kappa) * np.asarray(neighbor_value)
    ) / np.maximum(denominator, np.finfo(np.float64).tiny)
    return trace, robin_data_from_trace(
        center_value, trace, center_kappa, impedance, dx,
    )


class H5RobinDarcyDataset(H5DarcyDataset):
    def __init__(self, path: Path | str, split: str) -> None:
        super().__init__(path, split)
        if str(self.attrs.get("schema_version")) != ROBIN_SCHEMA:
            raise ValueError("Robin dataset requires the v3 mixed schema")
        with h5py.File(self.path, "r") as handle:
            group = handle[self.split]
            missing = [name for name in ROBIN_FIELDS if name not in group]
            if missing:
                raise KeyError(f"Robin split is missing {missing}")
            if any(group[name].shape[0] != self.length for name in ROBIN_FIELDS):
                raise ValueError("Robin fields have inconsistent case counts")
        self.has_robin = True

    def __getitem__(self, index: int) -> dict[str, Any]:
        values = super().__getitem__(index)
        group = self._group()
        for name in ROBIN_FIELDS:
            values[name] = torch.from_numpy(np.ascontiguousarray(
                _face_field(group[name][index], name),
            ))
        values["robin_face_mask"] = (
            values["robin_face_mask"] > 0.5
        ).to(torch.float32)
        return values


def robin_condition_features(
    mask: Tensor,
    value: Tensor,
    impedance: Tensor,
    domain: Tensor,
    statistics: DarcyStatistics,
) -> Tensor:
    dtype = domain.dtype
    active = (mask > 0.5).to(dtype) * domain
    kappa_reference = torch.ones_like(value)
    dx = 2.0 / max(domain.shape[-2:])
    flux_scale = max(statistics.bc_std, 1.0e-6) * 2.0 * kappa_reference / dx
    scaled_value = value.to(dtype) / flux_scale
    scaled_impedance = torch.log1p(
        impedance.to(dtype).clamp_min(0.0) * dx / (2.0 * kappa_reference)
    )
    return torch.cat((
        active, scaled_value * active, scaled_impedance * active,
    ), 1) * domain


class RobinDarcyAdapter(DarcyAdapter):
    spatial_condition_channels = DarcyAdapter.spatial_condition_channels + 16 + 12

    def __init__(self, statistics: DarcyStatistics) -> None:
        super().__init__(statistics, use_discrete_residual=False)

    def conditions(
        self, transition: PhysicalTransition, endpoint_phase: Tensor,
        level_scale: float,
    ) -> ConditionBatch:
        base = super().conditions(transition, endpoint_phase, level_scale)
        payload = transition.payload
        missing = [
            name for name in (*FACE_DIRICHLET_FIELDS, *ROBIN_FIELDS)
            if name not in payload
        ]
        if missing:
            raise ValueError(f"Robin-v3 checkpoint requires fields {missing}")
        domain = payload["mask"].to(base.spatial.dtype)
        dirichlet = face_condition_features(
            payload["dirichlet_face_mask"].to(base.spatial.dtype),
            payload["dirichlet_face_value"].to(base.spatial.dtype),
            payload["dirichlet_face_role"].to(base.spatial.dtype),
            domain, self.statistics,
        )
        robin = robin_condition_features(
            payload["robin_face_mask"], payload["robin_face_value"],
            payload["robin_face_impedance"], domain, self.statistics,
        )
        return ConditionBatch(
            spatial=torch.cat((base.spatial, dirichlet, robin), 1),
            scalars=base.scalars,
        )


def robin_batch_to_transition(
    batch: dict[str, Tensor], current: Tensor, target: Tensor | None = None,
) -> PhysicalTransition:
    payload = {
        name: batch[name]
        for name in (*REQUIRED_FIELDS, *FACE_DIRICHLET_FIELDS, *ROBIN_FIELDS)
        if name != "solution"
    }
    return PhysicalTransition(
        previous=current, current=current,
        target=batch.get("solution") if target is None else target,
        dt=torch.ones(current.shape[0], device=current.device, dtype=current.dtype),
        payload=payload, metadata={"case_index": batch.get("case_index")},
    )


def raw_mixed_robin_residual(
    solution: Tensor,
    kappa: Tensor,
    source: Tensor,
    mask: Tensor,
    dirichlet_mask: Tensor,
    dirichlet_value: Tensor,
    robin_mask: Tensor,
    robin_value: Tensor,
    impedance: Tensor,
) -> Tensor:
    """Torch residual matching :func:`assemble_mixed_robin_system`."""
    dx = 2.0 / max(solution.shape[-2:])
    inv_h2 = 1.0 / dx**2
    result = -source.clone()
    for face, (dy, ddx) in enumerate(FACE_OFFSETS):
        shifted_u = torch.roll(solution, shifts=(-dy, -ddx), dims=(-2, -1))
        shifted_k = torch.roll(kappa, shifts=(-dy, -ddx), dims=(-2, -1))
        shifted_mask = torch.roll(mask, shifts=(-dy, -ddx), dims=(-2, -1))
        # ``roll(..., shifts=(-dy, -ddx))`` makes output cell (y,x)
        # receive its (y+dy,x+ddx) neighbor.  Clear the wrapped edge on the
        # *receiving* side so the tensor implementation matches the sparse FV
        # row assembly exactly.
        if dy == -1: shifted_mask[..., 0, :] = 0
        elif dy == 1: shifted_mask[..., -1, :] = 0
        elif ddx == -1: shifted_mask[..., :, 0] = 0
        else: shifted_mask[..., :, -1] = 0
        neighbor = shifted_mask > 0.5
        harmonic = 2.0 * kappa * shifted_k / (kappa + shifted_k).clamp_min(1.0e-12)
        interior = harmonic * (solution - shifted_u) * inv_h2
        transmissibility = 2.0 * kappa * inv_h2
        dterm = transmissibility * (
            solution - dirichlet_value[:, face:face + 1]
        )
        p = impedance[:, face:face + 1]
        denominator = p + 2.0 * kappa / dx
        rterm = (
            transmissibility * p / denominator * solution
            - transmissibility * robin_value[:, face:face + 1] / denominator
        )
        exposed = (mask > 0.5) & ~neighbor
        covered_d = dirichlet_mask[:, face:face + 1] > 0.5
        covered_r = robin_mask[:, face:face + 1] > 0.5
        if torch.any(exposed & ~(covered_d | covered_r)):
            raise ValueError("mixed Robin residual has uncovered exposed face")
        boundary = torch.where(covered_d, dterm, rterm)
        result = result + torch.where(neighbor, interior, boundary)
    return result * mask
