"""Discrete Darcy operator and paper-facing physical diagnostics.

The reference problem is

    -div(kappa grad u) = f in Omega,   u = g on boundary(Omega).

It is discretized on active cells of a Cartesian mask.  Interior face
conductivities use harmonic averaging, and an inactive/out-of-grid neighbor is
treated as a Dirichlet face half a cell from the active cell center.  Keeping
the operator and all diagnostics here prevents training and evaluation from
silently using different PDE conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


Array = np.ndarray
FACE_NAMES = ("north", "south", "west", "east")
FACE_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class DarcySystem:
    matrix: sparse.csr_matrix
    rhs: Array
    index: Array
    active_coordinates: Array
    dx: float


def _validate_fields(
    mask: Array, kappa: Array, source: Array, bc_value: Array,
) -> tuple[Array, Array, Array, Array]:
    mask = np.asarray(mask, dtype=bool)
    kappa = np.asarray(kappa, dtype=np.float64)
    source = np.asarray(source, dtype=np.float64)
    bc_value = np.asarray(bc_value, dtype=np.float64)
    if mask.ndim != 2:
        raise ValueError("Darcy fields must be two-dimensional")
    if not (kappa.shape == source.shape == bc_value.shape == mask.shape):
        raise ValueError("mask, kappa, source, and bc_value shapes must agree")
    if not mask.any():
        raise ValueError("Darcy domain contains no active cells")
    if not np.isfinite(kappa[mask]).all() or np.any(kappa[mask] <= 0.0):
        raise ValueError("permeability must be finite and strictly positive")
    if not np.isfinite(source[mask]).all() or not np.isfinite(bc_value).all():
        raise ValueError("source and boundary values must be finite")
    return mask, kappa, source, bc_value


def exposed_dirichlet_faces(mask: Array) -> Array:
    """Return the four exposed cell faces in north/south/west/east order."""
    domain = np.asarray(mask, dtype=bool)
    if domain.ndim != 2:
        raise ValueError("Darcy mask must be two-dimensional")
    padded = np.pad(domain, 1, constant_values=False)
    neighbors = (
        padded[:-2, 1:-1],
        padded[2:, 1:-1],
        padded[1:-1, :-2],
        padded[1:-1, 2:],
    )
    return np.stack(tuple(domain & ~neighbor for neighbor in neighbors), axis=0)


def _validate_face_dirichlet(
    mask: Array,
    dirichlet_face_mask: Array | None,
    dirichlet_face_value: Array | None,
) -> tuple[Array | None, Array | None]:
    if dirichlet_face_mask is None and dirichlet_face_value is None:
        return None, None
    if dirichlet_face_mask is None or dirichlet_face_value is None:
        raise ValueError(
            "dirichlet_face_mask and dirichlet_face_value must be provided together"
        )
    face_mask = np.asarray(dirichlet_face_mask, dtype=bool)
    face_value = np.asarray(dirichlet_face_value, dtype=np.float64)
    expected = (4, *mask.shape)
    if face_mask.shape != expected or face_value.shape != expected:
        raise ValueError(
            f"face Dirichlet tensors must have shape {expected}, got "
            f"{face_mask.shape} and {face_value.shape}"
        )
    if not np.isfinite(face_value).all():
        raise ValueError("face Dirichlet values must be finite")
    exposed = exposed_dirichlet_faces(mask)
    missing = exposed & ~face_mask
    if missing.any():
        face, y, x = np.argwhere(missing)[0]
        raise ValueError(
            "pure-Dirichlet Darcy problem has an uncovered exposed face: "
            f"{FACE_NAMES[int(face)]} at cell ({int(y)},{int(x)})"
        )
    return face_mask, face_value


def harmonic_mean(left: float, right: float) -> float:
    return 2.0 * left * right / max(left + right, np.finfo(np.float64).tiny)


def assemble_darcy_system(
    mask: Array,
    kappa: Array,
    source: Array,
    bc_value: Array,
    *,
    dx: float | None = None,
    dirichlet_face_mask: Array | None = None,
    dirichlet_face_value: Array | None = None,
) -> DarcySystem:
    """Assemble the symmetric positive definite masked finite-volume system.

    ``bc_value`` is the legacy cellwise boundary contract.  New callers should
    additionally supply the two four-channel face tensors.  Their channel order
    is north, south, west, east, allowing a corner cell to carry two distinct
    traces.  Omitting them preserves the historical behavior exactly.
    """
    mask, kappa, source, bc_value = _validate_fields(
        mask, kappa, source, bc_value,
    )
    face_mask, face_value = _validate_face_dirichlet(
        mask, dirichlet_face_mask, dirichlet_face_value,
    )
    height, width = mask.shape
    if dx is None:
        dx = 2.0 / max(height, width)
    if dx <= 0.0:
        raise ValueError("dx must be positive")

    coordinates = np.argwhere(mask)
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[mask] = np.arange(len(coordinates), dtype=np.int64)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    rhs = np.empty(len(coordinates), dtype=np.float64)
    inverse_h2 = 1.0 / (dx * dx)

    for row, (yy, xx) in enumerate(coordinates):
        y = int(yy)
        x = int(xx)
        center_kappa = float(kappa[y, x])
        diagonal = 0.0
        rhs[row] = float(source[y, x])
        for face, (dy, ddx) in enumerate(FACE_OFFSETS):
            yn = y + dy
            xn = x + ddx
            if 0 <= yn < height and 0 <= xn < width and mask[yn, xn]:
                transmissibility = harmonic_mean(
                    center_kappa, float(kappa[yn, xn]),
                ) * inverse_h2
                diagonal += transmissibility
                rows.append(row)
                columns.append(int(index[yn, xn]))
                values.append(-transmissibility)
            else:
                # The Dirichlet face lies half a cell from the unknown center.
                transmissibility = 2.0 * center_kappa * inverse_h2
                diagonal += transmissibility
                boundary_value = (
                    float(bc_value[y, x])
                    if face_value is None
                    else float(face_value[face, y, x])
                )
                rhs[row] += transmissibility * boundary_value
        rows.append(row)
        columns.append(row)
        values.append(diagonal)

    matrix = sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(coordinates), len(coordinates)),
    )
    return DarcySystem(matrix, rhs, index, coordinates, float(dx))


def solve_darcy(
    mask: Array,
    kappa: Array,
    source: Array,
    bc_value: Array,
    *,
    dx: float | None = None,
    dirichlet_face_mask: Array | None = None,
    dirichlet_face_value: Array | None = None,
) -> tuple[Array, DarcySystem]:
    system = assemble_darcy_system(
        mask, kappa, source, bc_value, dx=dx,
        dirichlet_face_mask=dirichlet_face_mask,
        dirichlet_face_value=dirichlet_face_value,
    )
    values = sparse_linalg.spsolve(system.matrix, system.rhs)
    if not np.isfinite(values).all():
        raise RuntimeError("Darcy reference solve produced non-finite values")
    solution = np.zeros(np.asarray(mask).shape, dtype=np.float64)
    solution[np.asarray(mask, dtype=bool)] = values
    return solution, system


def active_vector(field: Array, system: DarcySystem) -> Array:
    field = np.asarray(field, dtype=np.float64)
    if field.shape != system.index.shape:
        raise ValueError("field and Darcy system shapes differ")
    return field[system.index >= 0]


def apply_system(field: Array, system: DarcySystem) -> Array:
    result = np.zeros(system.index.shape, dtype=np.float64)
    result[system.index >= 0] = system.matrix @ active_vector(field, system)
    return result


def relative_l2(prediction: Array, target: Array, mask: Array) -> float:
    active = np.asarray(mask, dtype=bool)
    error = np.asarray(prediction, dtype=np.float64)[active] - np.asarray(
        target, dtype=np.float64,
    )[active]
    denominator = np.linalg.norm(np.asarray(target, dtype=np.float64)[active])
    return float(np.linalg.norm(error) / max(denominator, 1.0e-12))


def relative_residual(prediction: Array, system: DarcySystem) -> float:
    residual = system.rhs - system.matrix @ active_vector(prediction, system)
    return float(np.linalg.norm(residual) / max(np.linalg.norm(system.rhs), 1.0e-12))


def relative_energy_error(
    prediction: Array, target: Array, system: DarcySystem,
) -> float:
    error = active_vector(prediction, system) - active_vector(target, system)
    truth = active_vector(target, system)
    numerator = float(error @ (system.matrix @ error))
    denominator = float(truth @ (system.matrix @ truth))
    return float(np.sqrt(max(numerator, 0.0) / max(denominator, 1.0e-24)))


def optimal_energy_relaxation(
    current: Array,
    proposal_update: Array,
    system: DarcySystem,
    *,
    lower: float = 0.0,
    upper: float = 1.5,
) -> float:
    """Energy-minimizing scalar step along an assembled update direction."""
    if lower > upper:
        raise ValueError("relaxation bounds are reversed")
    u = active_vector(current, system)
    direction = active_vector(proposal_update, system)
    residual = system.rhs - system.matrix @ u
    denominator = float(direction @ (system.matrix @ direction))
    if denominator <= 1.0e-24:
        return float(lower)
    value = float((residual @ direction) / denominator)
    return float(np.clip(value, lower, upper))


def aitken_relaxation(
    previous_update: Array,
    current_update: Array,
    previous_relaxation: float,
    *,
    lower: float = 0.05,
    upper: float = 1.5,
) -> float:
    """Scalar dynamic Aitken relaxation for successive assembled updates."""
    previous = np.asarray(previous_update, dtype=np.float64).ravel()
    current = np.asarray(current_update, dtype=np.float64).ravel()
    difference = current - previous
    denominator = float(difference @ difference)
    if denominator <= 1.0e-24:
        return float(np.clip(previous_relaxation, lower, upper))
    value = -float(previous_relaxation) * float(previous @ difference) / denominator
    return float(np.clip(value, lower, upper))


def _gradient(field: Array, dx: float) -> tuple[Array, Array]:
    edge_order = 2 if min(np.asarray(field).shape) >= 3 else 1
    grad_y, grad_x = np.gradient(
        np.asarray(field, dtype=np.float64), dx, dx, edge_order=edge_order,
    )
    return grad_y, grad_x


def interface_value_flux_jump(
    patches: Array,
    origins: Iterable[tuple[int, int]],
    masks: Array,
    kappas: Array,
    *,
    dx: float,
) -> dict[str, float]:
    """Measure pairwise value and normal-flux disagreement in patch overlaps.

    The normal for a patch pair is the dominant center-to-center direction.
    This matches regular Cartesian covers and remains well-defined for shifted
    covers.  Each unordered active overlap pair is counted once.
    """
    patches = np.asarray(patches, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    kappas = np.asarray(kappas, dtype=np.float64)
    origins = list(origins)
    if patches.ndim != 3 or masks.shape != patches.shape or kappas.shape != patches.shape:
        raise ValueError("patches, masks, and kappas must have shape K,H,W")
    if len(origins) != patches.shape[0]:
        raise ValueError("one origin is required per patch")

    fluxes = []
    for patch, permeability in zip(patches, kappas):
        grad_y, grad_x = _gradient(patch, dx)
        fluxes.append((-permeability * grad_y, -permeability * grad_x))

    value_sq: list[Array] = []
    value_scale_sq: list[Array] = []
    flux_sq: list[Array] = []
    flux_scale_sq: list[Array] = []
    count = 0
    patch_h, patch_w = patches.shape[-2:]
    for left in range(len(origins)):
        ly, lx = origins[left]
        for right in range(left + 1, len(origins)):
            ry, rx = origins[right]
            y0 = max(ly, ry)
            y1 = min(ly + patch_h, ry + patch_h)
            x0 = max(lx, rx)
            x1 = min(lx + patch_w, rx + patch_w)
            if y0 >= y1 or x0 >= x1:
                continue
            left_slice = (slice(y0 - ly, y1 - ly), slice(x0 - lx, x1 - lx))
            right_slice = (slice(y0 - ry, y1 - ry), slice(x0 - rx, x1 - rx))
            shared = masks[left][left_slice] & masks[right][right_slice]
            if not shared.any():
                continue
            left_value = patches[left][left_slice][shared]
            right_value = patches[right][right_slice][shared]
            value_sq.append((left_value - right_value) ** 2)
            value_scale_sq.append(0.5 * (left_value**2 + right_value**2))

            if abs(rx - lx) >= abs(ry - ly):
                left_flux = fluxes[left][1][left_slice][shared]
                right_flux = fluxes[right][1][right_slice][shared]
            else:
                left_flux = fluxes[left][0][left_slice][shared]
                right_flux = fluxes[right][0][right_slice][shared]
            flux_sq.append((left_flux - right_flux) ** 2)
            flux_scale_sq.append(0.5 * (left_flux**2 + right_flux**2))
            count += int(shared.sum())

    if count == 0:
        return {
            "overlap_points": 0.0,
            "value_jump_rms": 0.0,
            "value_jump_relative": 0.0,
            "normal_flux_jump_rms": 0.0,
            "normal_flux_jump_relative": 0.0,
        }
    value_mse = float(np.concatenate(value_sq).mean())
    value_scale = float(np.concatenate(value_scale_sq).mean())
    flux_mse = float(np.concatenate(flux_sq).mean())
    flux_scale = float(np.concatenate(flux_scale_sq).mean())
    return {
        "overlap_points": float(count),
        "value_jump_rms": float(np.sqrt(value_mse)),
        "value_jump_relative": float(np.sqrt(value_mse / max(value_scale, 1.0e-24))),
        "normal_flux_jump_rms": float(np.sqrt(flux_mse)),
        "normal_flux_jump_relative": float(np.sqrt(flux_mse / max(flux_scale, 1.0e-24))),
    }
