"""Production-inspired AMR benchmark utilities.

This module is intentionally smaller than PeleC/PeleLMeX, but borrows the
pieces that matter for learned Cartesian AMR:

- cut-cell-like geometry channels: volume fraction, face apertures, normals;
- face-weighted finite-volume pressure projection;
- vector-valued Cartesian solid mechanics instead of a scalar anti-plane toy;
- AMR refinement indicators driven by geometry, sources, and solution features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from neural_eos.block_structured_amr import (
    AMRHierarchy,
    build_block_structured_hierarchy,
    composite_level_map,
    draw_amr_grid,
    equivalent_cell_count,
)
from neural_eos.classical_amr_ddm import Array, normalize01


@dataclass
class CutCellGeometry:
    vol_frac: Array
    area_x: Array
    area_y: Array
    normal_x: Array
    normal_y: Array
    active: Array

    @property
    def shape(self) -> tuple[int, int]:
        return self.vol_frac.shape


@dataclass
class SolidCase:
    geometry: CutCellGeometry
    force_x: Array
    force_y: Array
    fixed: Array
    indicator: Array
    hierarchy: AMRHierarchy


@dataclass
class SolidResult:
    ux: Array
    uy: Array
    disp_mag: Array
    strain_xx: Array
    strain_yy: Array
    strain_xy: Array
    von_mises: Array
    residual: float
    dofs: int


@dataclass
class FluidCase:
    geometry: CutCellGeometry
    u_cell: Array
    v_cell: Array
    u_face: Array
    v_face: Array
    divergence: Array
    rhs: Array
    fixed: Array
    fixed_values: Array
    indicator: Array
    hierarchy: AMRHierarchy


@dataclass
class FluidResult:
    pressure: Array
    divergence_after: Array
    residual: float
    dofs: int


def meshgrid01(height: int, width: int) -> tuple[Array, Array]:
    y = (np.arange(height, dtype=np.float64) + 0.5)[:, None] / height
    x = (np.arange(width, dtype=np.float64) + 0.5)[None, :] / width
    return np.broadcast_to(y, (height, width)), np.broadcast_to(x, (height, width))


def gaussian(y: Array, x: Array, cy: float, cx: float, sy: float, sx: float) -> Array:
    return np.exp(-0.5 * (((y - cy) / sy) ** 2 + ((x - cx) / sx) ** 2))


def sample_cutcell_geometry(
    height: int,
    width: int,
    inside_fn,
    samples: int = 5,
    threshold: float = 1.0e-3,
) -> CutCellGeometry:
    offsets = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    y0 = np.arange(height, dtype=np.float64)[:, None] / height
    x0 = np.arange(width, dtype=np.float64)[None, :] / width
    vol = np.zeros((height, width), dtype=np.float64)
    for oy in offsets:
        for ox in offsets:
            yy = y0 + oy / height
            xx = x0 + ox / width
            vol += inside_fn(yy, xx).astype(np.float64)
    vol /= float(samples * samples)

    area_x = np.zeros((height, width + 1), dtype=np.float64)
    y_faces = (np.arange(height, dtype=np.float64)[:, None] + offsets[None, :]) / height
    for ix in range(width + 1):
        xx = np.full_like(y_faces, ix / width)
        area_x[:, ix] = inside_fn(y_faces, xx).mean(axis=1)

    area_y = np.zeros((height + 1, width), dtype=np.float64)
    x_faces = (np.arange(width, dtype=np.float64)[None, :] + offsets[:, None]) / width
    for iy in range(height + 1):
        yy = np.full_like(x_faces, iy / height)
        area_y[iy, :] = inside_fn(yy, x_faces).mean(axis=0)

    gy, gx = np.gradient(vol)
    norm = np.sqrt(gx * gx + gy * gy) + 1.0e-12
    # The gradient points from solid/covered to fluid/active when volume
    # fraction increases. Use the opposite as an approximate outward normal.
    normal_x = -gx / norm
    normal_y = -gy / norm
    active = vol > threshold
    return CutCellGeometry(vol, area_x, area_y, normal_x, normal_y, active)


def _edge_add(
    rows: list[int],
    cols: list[int],
    data: list[float],
    rhs: Array,
    free_index: dict[tuple[int, int, int], int],
    fixed_value,
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    stiffness: float,
) -> None:
    a_free = a in free_index
    b_free = b in free_index
    if a_free:
        ia = free_index[a]
        rows.append(ia)
        cols.append(ia)
        data.append(stiffness)
        if b_free:
            rows.append(ia)
            cols.append(free_index[b])
            data.append(-stiffness)
        else:
            rhs[ia] += stiffness * fixed_value(*b)
    if b_free:
        ib = free_index[b]
        rows.append(ib)
        cols.append(ib)
        data.append(stiffness)
        if a_free:
            rows.append(ib)
            cols.append(free_index[a])
            data.append(-stiffness)
        else:
            rhs[ib] += stiffness * fixed_value(*a)


def solve_vector_elasticity(
    geom: CutCellGeometry,
    force_x: Array,
    force_y: Array,
    fixed: Array,
    fixed_ux: Array | None = None,
    fixed_uy: Array | None = None,
    normal_stiffness: float = 320.0,
    shear_stiffness: float = 100.0,
    reaction: float = 1.0e-5,
) -> SolidResult:
    active = geom.active
    fixed_ux = np.zeros_like(force_x) if fixed_ux is None else fixed_ux
    fixed_uy = np.zeros_like(force_y) if fixed_uy is None else fixed_uy
    free_coords = np.argwhere(active & ~fixed)
    free_index: dict[tuple[int, int, int], int] = {}
    for cell_id, (yy, xx) in enumerate(free_coords):
        y = int(yy)
        x = int(xx)
        free_index[(y, x, 0)] = 2 * cell_id
        free_index[(y, x, 1)] = 2 * cell_id + 1

    n = 2 * len(free_coords)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(n, dtype=np.float64)
    for cell_id, (yy, xx) in enumerate(free_coords):
        y = int(yy)
        x = int(xx)
        vol = max(float(geom.vol_frac[y, x]), 1.0e-6)
        rhs[2 * cell_id] = force_x[y, x] * vol
        rhs[2 * cell_id + 1] = force_y[y, x] * vol

    def fixed_value(y: int, x: int, comp: int) -> float:
        if not active[y, x]:
            return 0.0
        if comp == 0:
            return float(fixed_ux[y, x])
        return float(fixed_uy[y, x])

    h, w = active.shape
    for y in range(h):
        for x in range(w):
            if not active[y, x]:
                continue
            vol = max(float(geom.vol_frac[y, x]), 1.0e-6)
            for comp in (0, 1):
                key = (y, x, comp)
                if key in free_index:
                    row = free_index[key]
                    rows.append(row)
                    cols.append(row)
                    data.append(reaction * vol)
            if x + 1 < w and active[y, x + 1]:
                aperture = float(geom.area_x[y, x + 1])
                if aperture > 1.0e-8:
                    _edge_add(
                        rows,
                        cols,
                        data,
                        rhs,
                        free_index,
                        fixed_value,
                        (y, x, 0),
                        (y, x + 1, 0),
                        normal_stiffness * aperture,
                    )
                    _edge_add(
                        rows,
                        cols,
                        data,
                        rhs,
                        free_index,
                        fixed_value,
                        (y, x, 1),
                        (y, x + 1, 1),
                        shear_stiffness * aperture,
                    )
            if y + 1 < h and active[y + 1, x]:
                aperture = float(geom.area_y[y + 1, x])
                if aperture > 1.0e-8:
                    _edge_add(
                        rows,
                        cols,
                        data,
                        rhs,
                        free_index,
                        fixed_value,
                        (y, x, 0),
                        (y + 1, x, 0),
                        shear_stiffness * aperture,
                    )
                    _edge_add(
                        rows,
                        cols,
                        data,
                        rhs,
                        free_index,
                        fixed_value,
                        (y, x, 1),
                        (y + 1, x, 1),
                        normal_stiffness * aperture,
                    )

    mat = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    sol = sparse_linalg.spsolve(mat, rhs)
    residual = float(np.linalg.norm(mat @ sol - rhs) / (np.linalg.norm(rhs) + 1.0e-12))

    ux = np.zeros_like(force_x, dtype=np.float64)
    uy = np.zeros_like(force_y, dtype=np.float64)
    ux[fixed & active] = fixed_ux[fixed & active]
    uy[fixed & active] = fixed_uy[fixed & active]
    for cell_id, (yy, xx) in enumerate(free_coords):
        y = int(yy)
        x = int(xx)
        ux[y, x] = sol[2 * cell_id]
        uy[y, x] = sol[2 * cell_id + 1]
    ux[~active] = 0.0
    uy[~active] = 0.0

    strain_xx, strain_yy, strain_xy, von_mises = compute_strain_von_mises(ux, uy, active)
    return SolidResult(
        ux=ux,
        uy=uy,
        disp_mag=np.sqrt(ux * ux + uy * uy) * active,
        strain_xx=strain_xx,
        strain_yy=strain_yy,
        strain_xy=strain_xy,
        von_mises=von_mises,
        residual=residual,
        dofs=n,
    )


def _masked_central_gradient(a: Array, active: Array) -> tuple[Array, Array]:
    h, w = a.shape
    dx = np.zeros_like(a, dtype=np.float64)
    dy = np.zeros_like(a, dtype=np.float64)
    for y in range(h):
        for x in range(w):
            if not active[y, x]:
                continue
            xm = x - 1 if x > 0 and active[y, x - 1] else x
            xp = x + 1 if x + 1 < w and active[y, x + 1] else x
            ym = y - 1 if y > 0 and active[y - 1, x] else y
            yp = y + 1 if y + 1 < h and active[y + 1, x] else y
            dx[y, x] = (a[y, xp] - a[y, xm]) / max(1, xp - xm)
            dy[y, x] = (a[yp, x] - a[ym, x]) / max(1, yp - ym)
    return dx, dy


def compute_strain_von_mises(ux: Array, uy: Array, active: Array, young: float = 1.0, nu: float = 0.30) -> tuple[Array, Array, Array, Array]:
    du_dx, du_dy = _masked_central_gradient(ux, active)
    dv_dx, dv_dy = _masked_central_gradient(uy, active)
    exx = du_dx * active
    eyy = dv_dy * active
    exy = 0.5 * (du_dy + dv_dx) * active
    coeff = young / (1.0 - nu * nu)
    sxx = coeff * (exx + nu * eyy)
    syy = coeff * (eyy + nu * exx)
    txy = young / (1.0 + nu) * exy
    vm = np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * txy * txy, 0.0)) * active
    return exx, eyy, exy, vm


def cell_to_face_velocity(u: Array, v: Array, active: Array) -> tuple[Array, Array]:
    h, w = u.shape
    u_face = np.zeros((h, w + 1), dtype=np.float64)
    v_face = np.zeros((h + 1, w), dtype=np.float64)
    for x in range(w + 1):
        if x == 0:
            u_face[:, x] = u[:, 0]
        elif x == w:
            u_face[:, x] = u[:, w - 1]
        else:
            mask = active[:, x - 1] & active[:, x]
            u_face[mask, x] = 0.5 * (u[mask, x - 1] + u[mask, x])
    for y in range(h + 1):
        if y == 0:
            v_face[y, :] = v[0, :]
        elif y == h:
            v_face[y, :] = v[h - 1, :]
        else:
            mask = active[y - 1, :] & active[y, :]
            v_face[y, mask] = 0.5 * (v[y - 1, mask] + v[y, mask])
    return u_face, v_face


def weighted_divergence_from_faces(geom: CutCellGeometry, u_face: Array, v_face: Array) -> Array:
    vol = np.maximum(geom.vol_frac, 1.0e-8)
    flux = (
        geom.area_x[:, 1:] * u_face[:, 1:]
        - geom.area_x[:, :-1] * u_face[:, :-1]
        + geom.area_y[1:, :] * v_face[1:, :]
        - geom.area_y[:-1, :] * v_face[:-1, :]
    )
    out = np.zeros_like(geom.vol_frac, dtype=np.float64)
    out[geom.active] = flux[geom.active] / vol[geom.active]
    return out


def assemble_weighted_poisson(
    geom: CutCellGeometry,
    rhs: Array,
    fixed: Array,
    fixed_values: Array,
    reaction: float = 0.0,
) -> tuple[sparse.csr_matrix, Array, dict[tuple[int, int], int]]:
    active = geom.active
    free_coords = np.argwhere(active & ~fixed)
    index = {(int(y), int(x)): i for i, (y, x) in enumerate(free_coords)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b = np.zeros(len(free_coords), dtype=np.float64)
    h, w = active.shape

    for row, (yy, xx) in enumerate(free_coords):
        y = int(yy)
        x = int(xx)
        diag = reaction * max(float(geom.vol_frac[y, x]), 1.0e-8)
        b[row] = rhs[y, x]
        for dy, dx, aperture in (
            (0, 1, geom.area_x[y, x + 1] if x + 1 <= w else 0.0),
            (0, -1, geom.area_x[y, x] if x >= 0 else 0.0),
            (1, 0, geom.area_y[y + 1, x] if y + 1 <= h else 0.0),
            (-1, 0, geom.area_y[y, x] if y >= 0 else 0.0),
        ):
            yn = y + dy
            xn = x + dx
            if yn < 0 or yn >= h or xn < 0 or xn >= w or not active[yn, xn]:
                continue
            weight = float(aperture)
            if weight <= 1.0e-10:
                continue
            diag += weight
            if fixed[yn, xn]:
                b[row] += weight * fixed_values[yn, xn]
            else:
                rows.append(row)
                cols.append(index[(yn, xn)])
                data.append(-weight)
        rows.append(row)
        cols.append(row)
        data.append(diag)

    mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(free_coords), len(free_coords)))
    return mat, b, index


def weighted_poisson_apply(geom: CutCellGeometry, pressure: Array, reaction: float = 0.0) -> Array:
    active = geom.active
    out = reaction * geom.vol_frac * pressure
    h, w = active.shape
    for y in range(h):
        for x in range(w):
            if not active[y, x]:
                continue
            val = 0.0
            if x + 1 < w and active[y, x + 1]:
                val += geom.area_x[y, x + 1] * (pressure[y, x] - pressure[y, x + 1])
            if x - 1 >= 0 and active[y, x - 1]:
                val += geom.area_x[y, x] * (pressure[y, x] - pressure[y, x - 1])
            if y + 1 < h and active[y + 1, x]:
                val += geom.area_y[y + 1, x] * (pressure[y, x] - pressure[y + 1, x])
            if y - 1 >= 0 and active[y - 1, x]:
                val += geom.area_y[y, x] * (pressure[y, x] - pressure[y - 1, x])
            out[y, x] += val
    out[~active] = 0.0
    return out


def solve_weighted_pressure_projection(
    geom: CutCellGeometry,
    u_face: Array,
    v_face: Array,
    fixed: Array,
    fixed_values: Array,
    reaction: float = 0.0,
) -> FluidResult:
    div = weighted_divergence_from_faces(geom, u_face, v_face)
    rhs = div * geom.vol_frac
    rhs[~geom.active] = 0.0
    mat, b, index = assemble_weighted_poisson(geom, rhs, fixed, fixed_values, reaction)
    sol = sparse_linalg.spsolve(mat, b)
    pressure = np.zeros_like(rhs, dtype=np.float64)
    pressure[fixed & geom.active] = fixed_values[fixed & geom.active]
    for (y, x), i in index.items():
        pressure[y, x] = sol[i]
    pressure[~geom.active] = 0.0
    applied = weighted_poisson_apply(geom, pressure, reaction)
    free = geom.active & ~fixed
    residual = float(np.linalg.norm((rhs - applied)[free]) / (np.linalg.norm(rhs[free]) + 1.0e-12))
    div_after = np.zeros_like(div)
    vol = np.maximum(geom.vol_frac, 1.0e-8)
    div_after[geom.active] = (rhs[geom.active] - applied[geom.active]) / vol[geom.active]
    return FluidResult(pressure=pressure, divergence_after=div_after, residual=residual, dofs=len(index))


def bracket_inside(y: Array, x: Array) -> Array:
    beam = (x > 0.06) & (x < 0.82) & (y > 0.26) & (y < 0.74)
    lug = ((x - 0.78) / 0.25) ** 2 + ((y - 0.50) / 0.28) ** 2 < 1.0
    top_rib = (x > 0.18) & (x < 0.70) & (y > 0.18) & (y < 0.32)
    bot_rib = (x > 0.18) & (x < 0.70) & (y > 0.68) & (y < 0.82)
    outer = beam | lug | top_rib | bot_rib
    left_hole_top = ((x - 0.18) / 0.055) ** 2 + ((y - 0.38) / 0.055) ** 2 < 1.0
    left_hole_bot = ((x - 0.18) / 0.055) ** 2 + ((y - 0.62) / 0.055) ** 2 < 1.0
    eye_hole = ((x - 0.78) / 0.105) ** 2 + ((y - 0.50) / 0.105) ** 2 < 1.0
    slot = ((x - 0.47) / 0.12) ** 2 + ((y - 0.50) / 0.065) ** 2 < 1.0
    return outer & ~(left_hole_top | left_hole_bot | eye_hole | slot)


def fluid_inside(y: Array, x: Array) -> Array:
    channel = (y > 0.12) & (y < 0.88)
    inlet_step = (x < 0.20) & (y < 0.36)
    ramp_wall = (x > 0.18) & (x < 0.38) & (y < 0.18 + 0.45 * (x - 0.18))
    cylinder = ((x - 0.37) / 0.075) ** 2 + ((y - 0.52) / 0.115) ** 2 < 1.0
    splitter = (x > 0.66) & (x < 0.70) & (y > 0.28) & (y < 0.68)
    rounded_tip = ((x - 0.70) / 0.055) ** 2 + ((y - 0.68) / 0.055) ** 2 < 1.0
    return channel & ~(inlet_step | ramp_wall | cylinder | splitter | rounded_tip)


def _resolution(preset: str, case: str) -> tuple[int, int]:
    if case == "solid":
        if preset == "smoke":
            return 72, 112
        if preset == "large":
            return 128, 224
        return 96, 160
    if preset == "smoke":
        return 80, 120
    if preset == "large":
        return 160, 256
    return 128, 192


def _hierarchy_for(
    geom: CutCellGeometry,
    indicator: Array,
    fixed: Array,
    rhs: Array,
    reaction: float,
    case: str,
) -> AMRHierarchy:
    active = geom.active
    fixed_values = np.zeros_like(rhs)
    if case == "fluid":
        thresholds = (2.40, 1.20)
    else:
        thresholds = (0.16, 0.36)
    return build_block_structured_hierarchy(
        active,
        rhs,
        fixed,
        fixed_values,
        indicator,
        reaction,
        max_level=2,
        refinement=2,
        tag_thresholds=thresholds,
        blocking=4,
        buffer_cells=1,
        max_component_blocks=3 if case == "fluid" else 4,
    )


def make_mature_solid_case(preset: str = "medium") -> SolidCase:
    height, width = _resolution(preset, "solid")
    geom = sample_cutcell_geometry(height, width, bracket_inside, samples=5)
    yy, xx = meshgrid01(height, width)
    fixed = geom.active & (xx < 0.085)
    load_region = gaussian(yy, xx, 0.50, 0.96, 0.13, 0.035) + 0.35 * gaussian(yy, xx, 0.30, 0.87, 0.08, 0.06)
    force_x = -0.10 * load_region * geom.vol_frac
    force_y = -1.35 * load_region * geom.vol_frac
    cutness = 4.0 * geom.vol_frac * (1.0 - geom.vol_frac)
    force_mag = np.sqrt(force_x * force_x + force_y * force_y)
    indicator = 2.2 * normalize01(cutness) + 1.4 * normalize01(force_mag, geom.active) + 0.8 * fixed.astype(np.float64)
    hierarchy = _hierarchy_for(geom, indicator, fixed, force_mag, 1.0e-7, "solid")
    return SolidCase(geom, force_x, force_y, fixed, indicator, hierarchy)


def make_mature_fluid_case(preset: str = "medium") -> FluidCase:
    height, width = _resolution(preset, "fluid")
    geom = sample_cutcell_geometry(height, width, fluid_inside, samples=5)
    yy, xx = meshgrid01(height, width)
    wake = gaussian(yy, xx, 0.52, 0.52, 0.17, 0.20)
    shear = gaussian(yy, xx, 0.34, 0.30, 0.10, 0.16) - gaussian(yy, xx, 0.70, 0.72, 0.13, 0.18)
    inlet_profile = 1.0 - 2.2 * (yy - 0.50) ** 2
    u_cell = (0.95 * inlet_profile - 0.55 * wake + 0.16 * shear) * geom.active
    v_cell = (0.33 * (xx - 0.42) * wake - 0.23 * (yy - 0.58) * shear) * geom.active
    u_face, v_face = cell_to_face_velocity(u_cell, v_cell, geom.active)
    inlet_flux = float(np.sum(geom.area_x[:, 0] * u_face[:, 0]))
    outlet_area = float(np.sum(geom.area_x[:, -1]))
    outlet_flux = float(np.sum(geom.area_x[:, -1] * u_face[:, -1]))
    if outlet_area > 1.0e-12:
        u_face[:, -1] += (inlet_flux - outlet_flux) / outlet_area
    divergence = weighted_divergence_from_faces(geom, u_face, v_face)
    rhs = divergence * geom.vol_frac
    fixed = np.zeros((height, width), dtype=bool)
    gauge_y = int(0.18 * height)
    gauge_x = int(0.88 * width)
    if geom.active[gauge_y, gauge_x]:
        fixed[gauge_y, gauge_x] = True
    else:
        coords = np.argwhere(geom.active)
        nearest = coords[np.argmin((coords[:, 0] - gauge_y) ** 2 + (coords[:, 1] - gauge_x) ** 2)]
        fixed[int(nearest[0]), int(nearest[1])] = True
    fixed_values = np.zeros((height, width), dtype=np.float64)
    cutness = 4.0 * geom.vol_frac * (1.0 - geom.vol_frac)
    speed = np.sqrt(u_cell * u_cell + v_cell * v_cell)
    indicator = 2.0 * normalize01(cutness) + 1.4 * normalize01(np.abs(divergence), geom.active) + 0.7 * normalize01(speed, geom.active)
    hierarchy = _hierarchy_for(geom, indicator, fixed, np.abs(rhs), 0.0, "fluid")
    return FluidCase(geom, u_cell, v_cell, u_face, v_face, divergence, rhs, fixed, fixed_values, indicator, hierarchy)


def level_and_grid(hierarchy: AMRHierarchy) -> tuple[Array, Array, dict[str, int | float]]:
    return composite_level_map(hierarchy), draw_amr_grid(hierarchy), equivalent_cell_count(hierarchy)
