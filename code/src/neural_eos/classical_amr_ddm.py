from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


Array = np.ndarray


@dataclass(frozen=True)
class DDMBlock:
    """Axis-aligned Cartesian subdomain used by the Schwarz solver."""

    y0: int
    y1: int
    x0: int
    x1: int
    level: int = 0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def area(self) -> int:
        return self.height * self.width

    def expanded(self, halo: int, height: int, width: int) -> "DDMBlock":
        return DDMBlock(
            max(0, self.y0 - halo),
            min(height, self.y1 + halo),
            max(0, self.x0 - halo),
            min(width, self.x1 + halo),
            self.level,
        )


@dataclass(frozen=True)
class SchwarzConfig:
    max_sweeps: int = 80
    local_iters: int = 24
    halo: int = 4
    damping: float = 0.9
    tolerance: float = 1.0e-6
    coarse_every: int = 0
    coarse_iters: int = 80
    coarse_factor: int = 4
    coarse_weight: float = 0.35


@dataclass
class SchwarzResult:
    field: Array
    residual_history: list[float]
    blocks: list[DDMBlock]
    sweeps: int


def _shift(a: Array, dy: int, dx: int, fill: float = 0.0) -> Array:
    out = np.full_like(a, fill)
    h, w = a.shape
    ys_src0 = max(0, -dy)
    ys_src1 = min(h, h - dy)
    xs_src0 = max(0, -dx)
    xs_src1 = min(w, w - dx)
    ys_dst0 = max(0, dy)
    ys_dst1 = min(h, h + dy)
    xs_dst0 = max(0, dx)
    xs_dst1 = min(w, w + dx)
    if ys_src1 > ys_src0 and xs_src1 > xs_src0:
        out[ys_dst0:ys_dst1, xs_dst0:xs_dst1] = a[ys_src0:ys_src1, xs_src0:xs_src1]
    return out


def neighbor_count(active: Array) -> Array:
    active_f = active.astype(np.float64)
    return (
        _shift(active_f, 0, 1)
        + _shift(active_f, 0, -1)
        + _shift(active_f, 1, 0)
        + _shift(active_f, -1, 0)
    )


def graph_laplacian_apply(field: Array, active: Array, reaction: float = 0.0) -> Array:
    count = neighbor_count(active)
    neigh = (
        _shift(field * active, 0, 1)
        + _shift(field * active, 0, -1)
        + _shift(field * active, 1, 0)
        + _shift(field * active, -1, 0)
    )
    return ((count + reaction) * field - neigh) * active


def residual_norm(field: Array, rhs: Array, active: Array, fixed: Array, reaction: float = 0.0) -> float:
    free = active & ~fixed
    res = (rhs - graph_laplacian_apply(field, active, reaction))[free]
    denom = np.linalg.norm(rhs[free]) + 1.0e-12
    return float(np.linalg.norm(res) / denom)


def jacobi_relax(
    field: Array,
    rhs: Array,
    active: Array,
    fixed: Array,
    fixed_values: Array,
    reaction: float,
    iterations: int,
    omega: float = 0.8,
) -> Array:
    u = field.copy()
    active = active.astype(bool)
    fixed = fixed.astype(bool)
    free = active & ~fixed
    count = neighbor_count(active)
    denom = np.maximum(count + reaction, 1.0e-12)
    for _ in range(iterations):
        neigh = (
            _shift(u * active, 0, 1)
            + _shift(u * active, 0, -1)
            + _shift(u * active, 1, 0)
            + _shift(u * active, -1, 0)
        )
        proposal = (rhs + neigh) / denom
        u[free] = (1.0 - omega) * u[free] + omega * proposal[free]
        u[fixed] = fixed_values[fixed]
        u[~active] = 0.0
    return u


def assemble_scalar_system(
    active: Array,
    rhs: Array,
    fixed: Array,
    fixed_values: Array,
    reaction: float = 0.0,
) -> tuple[sparse.csr_matrix, Array, dict[tuple[int, int], int]]:
    active = active.astype(bool)
    fixed = fixed.astype(bool)
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
        diag = reaction
        b[row] = rhs[y, x]
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            yn = y + dy
            xn = x + dx
            if yn < 0 or yn >= h or xn < 0 or xn >= w or not active[yn, xn]:
                continue
            diag += 1.0
            if fixed[yn, xn]:
                b[row] += fixed_values[yn, xn]
            else:
                rows.append(row)
                cols.append(index[(yn, xn)])
                data.append(-1.0)
        rows.append(row)
        cols.append(row)
        data.append(diag)

    mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(free_coords), len(free_coords)))
    return mat, b, index


def direct_scalar_solve(
    active: Array,
    rhs: Array,
    fixed: Array,
    fixed_values: Array,
    reaction: float = 0.0,
) -> Array:
    mat, b, index = assemble_scalar_system(active, rhs, fixed, fixed_values, reaction)
    sol = sparse_linalg.spsolve(mat, b)
    field = np.zeros_like(rhs, dtype=np.float64)
    field[fixed] = fixed_values[fixed]
    for (y, x), i in index.items():
        field[y, x] = sol[i]
    field[~active] = 0.0
    return field


def normalize01(a: Array, mask: Array | None = None) -> Array:
    values = a[mask] if mask is not None else a.reshape(-1)
    if values.size == 0:
        return np.zeros_like(a, dtype=np.float64)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi - lo < 1.0e-12:
        return np.zeros_like(a, dtype=np.float64)
    return (a - lo) / (hi - lo)


def boundary_indicator(active: Array) -> Array:
    active_f = active.astype(np.float64)
    count = neighbor_count(active)
    return ((count < 4.0) & active).astype(np.float64) + (
        (_shift(active_f, 0, 1) + _shift(active_f, 0, -1) + _shift(active_f, 1, 0) + _shift(active_f, -1, 0))
        > 0
    ).astype(np.float64) * (~active).astype(np.float64)


def dilate(mask: Array, radius: int) -> Array:
    out = mask.astype(bool).copy()
    for _ in range(radius):
        expanded = out.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            expanded |= _shift(out.astype(np.float64), dy, dx).astype(bool)
        out = expanded
    return out


def adaptive_blocks(
    height: int,
    width: int,
    indicator: Array,
    min_block: int = 12,
    max_block: int = 40,
    threshold: float = 0.2,
    max_level: int = 4,
) -> list[DDMBlock]:
    if indicator.shape != (height, width):
        raise ValueError("indicator shape must match grid shape")
    indicator = normalize01(indicator)
    blocks: list[DDMBlock] = []

    def split(y0: int, y1: int, x0: int, x1: int, level: int) -> None:
        h = y1 - y0
        w = x1 - x0
        region = indicator[y0:y1, x0:x1]
        # Keep large unfeatured regions coarse. This is closer to an AMR leaf
        # layout than a uniform box partition: only very large boxes are forced
        # to split, while feature-bearing boxes split down to the requested
        # minimum scale.
        forced = h > 3 * max_block or w > 3 * max_block
        hot_fraction = float(np.mean(region >= threshold)) if region.size else 0.0
        featured = hot_fraction >= 0.025
        can_split = level < max_level and h > 2 * min_block and w > 2 * min_block
        if can_split and (forced or featured):
            ym = y0 + h // 2
            xm = x0 + w // 2
            split(y0, ym, x0, xm, level + 1)
            split(y0, ym, xm, x1, level + 1)
            split(ym, y1, x0, xm, level + 1)
            split(ym, y1, xm, x1, level + 1)
            return
        blocks.append(DDMBlock(y0, y1, x0, x1, level))

    split(0, height, 0, width, 0)
    return blocks


def uniform_blocks(height: int, width: int, block: int = 32) -> list[DDMBlock]:
    out: list[DDMBlock] = []
    for y0 in range(0, height, block):
        for x0 in range(0, width, block):
            out.append(DDMBlock(y0, min(height, y0 + block), x0, min(width, x0 + block), 0))
    return out


def draw_block_edges(shape: tuple[int, int], blocks: list[DDMBlock]) -> Array:
    h, w = shape
    edges = np.zeros((h, w), dtype=np.float64)
    for block in blocks:
        y0, y1, x0, x1 = block.y0, block.y1, block.x0, block.x1
        edges[y0:y1, x0] = 1.0
        edges[y0:y1, x1 - 1] = 1.0
        edges[y0, x0:x1] = 1.0
        edges[y1 - 1, x0:x1] = 1.0
    return edges


def _local_fixed_for_patch(global_fixed: Array, ext: DDMBlock) -> Array:
    local_fixed = global_fixed[ext.y0 : ext.y1, ext.x0 : ext.x1].copy()
    if ext.y0 > 0:
        local_fixed[0, :] = True
    if ext.y1 < global_fixed.shape[0]:
        local_fixed[-1, :] = True
    if ext.x0 > 0:
        local_fixed[:, 0] = True
    if ext.x1 < global_fixed.shape[1]:
        local_fixed[:, -1] = True
    return local_fixed


def restrict_mean(a: Array, factor: int) -> Array:
    h, w = a.shape
    hc = h // factor
    wc = w // factor
    trimmed = a[: hc * factor, : wc * factor]
    return trimmed.reshape(hc, factor, wc, factor).mean(axis=(1, 3))


def restrict_bool_any(a: Array, factor: int) -> Array:
    h, w = a.shape
    hc = h // factor
    wc = w // factor
    trimmed = a[: hc * factor, : wc * factor]
    return trimmed.reshape(hc, factor, wc, factor).any(axis=(1, 3))


def prolong_nearest(a: Array, factor: int, shape: tuple[int, int]) -> Array:
    out = np.repeat(np.repeat(a, factor, axis=0), factor, axis=1)
    h, w = shape
    if out.shape[0] < h or out.shape[1] < w:
        padded = np.zeros(shape, dtype=out.dtype)
        padded[: out.shape[0], : out.shape[1]] = out
        if out.shape[0] < h:
            padded[out.shape[0] :, : out.shape[1]] = out[-1:, :]
        if out.shape[1] < w:
            padded[:, out.shape[1] :] = padded[:, out.shape[1] - 1 : out.shape[1]]
        out = padded
    return out[:h, :w]


def coarse_residual_correction(
    field: Array,
    rhs: Array,
    active: Array,
    fixed: Array,
    reaction: float,
    factor: int,
    iters: int,
    weight: float,
) -> Array:
    if factor <= 1 or min(field.shape) < 2 * factor:
        return field
    residual = (rhs - graph_laplacian_apply(field, active, reaction)) * (active & ~fixed)
    coarse_active = restrict_bool_any(active, factor)
    coarse_fixed = restrict_bool_any(fixed, factor)
    # For near-Poisson pressure solves, the graph Laplacian is the h^2-scaled
    # finite-volume operator; after coarsening by r, the coarse residual needs
    # an r^2 scale. For screened/reaction problems in this prototype, the
    # reaction coefficient is already a grid-level stabilizer, so the unscaled
    # residual is the more robust two-level correction.
    coarse_scale = float(factor * factor) if reaction <= 1.0e-7 else 1.0
    coarse_rhs = restrict_mean(residual, factor) * coarse_scale
    coarse_fixed_values = np.zeros_like(coarse_rhs)
    if coarse_active.sum() == coarse_fixed.sum():
        return field
    try:
        corr = direct_scalar_solve(
            coarse_active,
            coarse_rhs,
            coarse_fixed,
            coarse_fixed_values,
            reaction=max(reaction, 1.0e-8),
        )
    except Exception:
        corr = jacobi_relax(
            np.zeros_like(coarse_rhs),
            coarse_rhs,
            coarse_active,
            coarse_fixed,
            coarse_fixed_values,
            reaction=max(reaction, 1.0e-8),
            iterations=iters,
            omega=0.75,
        )
    fine_corr = prolong_nearest(corr, factor, field.shape)
    out = field.copy()
    out[active & ~fixed] += weight * fine_corr[active & ~fixed]
    return out


def overlapping_schwarz_solve(
    active: Array,
    rhs: Array,
    fixed: Array,
    fixed_values: Array,
    blocks: list[DDMBlock],
    reaction: float = 0.0,
    config: SchwarzConfig | None = None,
    callback: Callable[[int, Array, float], None] | None = None,
) -> SchwarzResult:
    cfg = config or SchwarzConfig()
    h, w = active.shape
    u = np.zeros_like(rhs, dtype=np.float64)
    u[fixed] = fixed_values[fixed]
    history: list[float] = []

    for sweep in range(1, cfg.max_sweeps + 1):
        accum = np.zeros_like(u)
        weight = np.zeros_like(u)
        for block in blocks:
            ext = block.expanded(cfg.halo, h, w)
            local_u = u[ext.y0 : ext.y1, ext.x0 : ext.x1].copy()
            local_rhs = rhs[ext.y0 : ext.y1, ext.x0 : ext.x1]
            local_active = active[ext.y0 : ext.y1, ext.x0 : ext.x1]
            local_fixed = _local_fixed_for_patch(fixed, ext)
            local_fixed_values = u[ext.y0 : ext.y1, ext.x0 : ext.x1].copy()
            local_fixed_values[fixed[ext.y0 : ext.y1, ext.x0 : ext.x1]] = fixed_values[
                ext.y0 : ext.y1, ext.x0 : ext.x1
            ][fixed[ext.y0 : ext.y1, ext.x0 : ext.x1]]
            local_u = jacobi_relax(
                local_u,
                local_rhs,
                local_active,
                local_fixed,
                local_fixed_values,
                reaction,
                cfg.local_iters,
                omega=0.82,
            )
            sy0 = block.y0 - ext.y0
            sy1 = block.y1 - ext.y0
            sx0 = block.x0 - ext.x0
            sx1 = block.x1 - ext.x0
            core = local_u[sy0:sy1, sx0:sx1]
            core_active = active[block.y0 : block.y1, block.x0 : block.x1] & ~fixed[
                block.y0 : block.y1, block.x0 : block.x1
            ]
            target = accum[block.y0 : block.y1, block.x0 : block.x1]
            target_w = weight[block.y0 : block.y1, block.x0 : block.x1]
            target[core_active] += core[core_active]
            target_w[core_active] += 1.0

        updated = u.copy()
        valid = weight > 0
        updated[valid] = accum[valid] / weight[valid]
        free = active & ~fixed
        u[free] = (1.0 - cfg.damping) * u[free] + cfg.damping * updated[free]
        u[fixed] = fixed_values[fixed]
        u[~active] = 0.0
        if cfg.coarse_every and sweep % cfg.coarse_every == 0:
            u = coarse_residual_correction(
                u,
                rhs,
                active,
                fixed,
                reaction,
                factor=cfg.coarse_factor,
                iters=cfg.coarse_iters,
                weight=cfg.coarse_weight,
            )
            u[fixed] = fixed_values[fixed]
            u[~active] = 0.0
        res = residual_norm(u, rhs, active, fixed, reaction)
        history.append(res)
        if callback is not None:
            callback(sweep, u, res)
        if res <= cfg.tolerance:
            break

    return SchwarzResult(field=u, residual_history=history, blocks=blocks, sweeps=len(history))


def relative_l2(a: Array, b: Array, mask: Array) -> float:
    diff = (a - b)[mask]
    denom = np.linalg.norm(b[mask]) + 1.0e-12
    return float(np.linalg.norm(diff) / denom)


def divergence(u: Array, v: Array, active: Array) -> Array:
    uu = u * active
    vv = v * active
    div = 0.5 * (_shift(uu, 0, -1) - _shift(uu, 0, 1) + _shift(vv, -1, 0) - _shift(vv, 1, 0))
    return div * active


def gradient(p: Array, active: Array) -> tuple[Array, Array]:
    pe = _shift(p, 0, -1)
    pw = _shift(p, 0, 1)
    ps = _shift(p, -1, 0)
    pn = _shift(p, 1, 0)
    me = _shift(active.astype(np.float64), 0, -1).astype(bool)
    mw = _shift(active.astype(np.float64), 0, 1).astype(bool)
    ms = _shift(active.astype(np.float64), -1, 0).astype(bool)
    mn = _shift(active.astype(np.float64), 1, 0).astype(bool)
    pe = np.where(me, pe, p)
    pw = np.where(mw, pw, p)
    ps = np.where(ms, ps, p)
    pn = np.where(mn, pn, p)
    return 0.5 * (pe - pw) * active, 0.5 * (ps - pn) * active


def save_npz(path: Path, **arrays: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
