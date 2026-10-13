"""Block-structured Cartesian AMR baseline.

Literature anchors used while implementing and debugging this file:

- Berger and Colella (1989): nested rectangular patch hierarchy, refinement
  levels, and explicit coarse/fine synchronization.
- Aftosmis (1997) and Aftosmis, Berger, and Melton (1998): Cartesian grids as a
  practical complex-geometry scaffold, with refinement around embedded geometry.
- Zhang et al. (2019, 2021) AMReX: box/patch-oriented block-structured AMR data
  model and GPU-friendly rectangular-grid iteration.
- Projection/cut-cell AMR papers such as Martin, Colella, and Graves (2008),
  Min and Gibou (2006), Johansen and Colella (1998), and later embedded-boundary
  finite-volume work motivate the face/interface bookkeeping.

This is still a compact reproduction baseline, not a full production AMR code:
it does not implement time subcycling, refluxing, high-order embedded-boundary
geometry, or AMR-consistent velocity projection. Those omissions are deliberate
and documented in papers/AMR_Cartesian/block_structured_amr_reproduction.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import linalg as sparse_linalg

from neural_eos.classical_amr_ddm import (
    Array,
    direct_scalar_solve,
    graph_laplacian_apply,
    jacobi_relax,
    residual_norm,
)


@dataclass(frozen=True)
class AMRBox:
    """A rectangular patch in the index space of one AMR level."""

    level: int
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def area(self) -> int:
        return self.height * self.width

    def child_extent(self, refinement: int) -> tuple[int, int, int, int]:
        return (
            self.y0 * refinement,
            self.y1 * refinement,
            self.x0 * refinement,
            self.x1 * refinement,
        )


@dataclass
class AMRLevel:
    level: int
    shape: tuple[int, int]
    active: Array
    rhs: Array
    fixed: Array
    fixed_values: Array
    indicator: Array
    reaction: float
    boxes: list[AMRBox]


@dataclass
class AMRHierarchy:
    levels: list[AMRLevel]
    refinement: int
    fine_active: Array
    fine_rhs: Array
    fine_fixed: Array
    fine_fixed_values: Array
    fine_indicator: Array
    fine_reaction: float

    @property
    def max_level(self) -> int:
        return len(self.levels) - 1

    @property
    def fine_shape(self) -> tuple[int, int]:
        return self.fine_active.shape


@dataclass(frozen=True)
class AMRSolveConfig:
    cycles: int = 18
    coarse_iters: int = 80
    patch_iters: int = 80
    restriction_weight: float = 0.45
    tolerance: float = 2.0e-6
    use_direct_patch_solve: bool = True


@dataclass
class AMRSolveResult:
    fields: list[Array]
    composite: Array
    residual_history: list[float]
    cycles: int


@dataclass
class CompositeAMRDirectResult:
    field: Array
    leaf_level_map: Array
    leaf_count: int
    free_count: int
    operator_residual: float


def _require_divisible(shape: tuple[int, int], factor: int) -> None:
    if shape[0] % factor != 0 or shape[1] % factor != 0:
        raise ValueError(f"shape {shape} must be divisible by refinement factor {factor}")


def _block_view(a: Array, factor: int) -> Array:
    h, w = a.shape
    _require_divisible((h, w), factor)
    return a.reshape(h // factor, factor, w // factor, factor)


def restrict_mean(a: Array, factor: int) -> Array:
    if factor == 1:
        return a.copy()
    return _block_view(a, factor).mean(axis=(1, 3))


def restrict_sum(a: Array, factor: int) -> Array:
    if factor == 1:
        return a.copy()
    return _block_view(a, factor).sum(axis=(1, 3))


def restrict_max(a: Array, factor: int) -> Array:
    if factor == 1:
        return a.copy()
    return _block_view(a, factor).max(axis=(1, 3))


def restrict_bool_any(a: Array, factor: int) -> Array:
    if factor == 1:
        return a.astype(bool).copy()
    return _block_view(a.astype(bool), factor).any(axis=(1, 3))


def restrict_weighted_mean(a: Array, weight_mask: Array, factor: int) -> tuple[Array, Array]:
    if factor == 1:
        valid = weight_mask.astype(bool).copy()
        out = np.zeros_like(a, dtype=np.float64)
        out[valid] = a[valid]
        return out, valid
    weight = weight_mask.astype(np.float64)
    denom = restrict_sum(weight, factor)
    numer = restrict_sum(a * weight, factor)
    out = np.zeros_like(denom, dtype=np.float64)
    valid = denom > 0.0
    out[valid] = numer[valid] / denom[valid]
    return out, valid


def prolong_bilinear(a: Array, factor: int, shape: tuple[int, int] | None = None) -> Array:
    """Cell-centered bilinear prolongation from a coarse Cartesian grid."""

    if factor == 1:
        out = a.copy()
        return out if shape is None else out[: shape[0], : shape[1]]
    h, w = a.shape
    out_shape = shape or (h * factor, w * factor)
    hf, wf = out_shape

    yy = (np.arange(hf, dtype=np.float64) + 0.5) / factor - 0.5
    xx = (np.arange(wf, dtype=np.float64) + 0.5) / factor - 0.5
    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    ty = yy - y0
    tx = xx - x0
    y0 = np.clip(y0, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)

    top = (1.0 - tx)[None, :] * a[y0[:, None], x0[None, :]] + tx[None, :] * a[y0[:, None], x1[None, :]]
    bot = (1.0 - tx)[None, :] * a[y1[:, None], x0[None, :]] + tx[None, :] * a[y1[:, None], x1[None, :]]
    return (1.0 - ty)[:, None] * top + ty[:, None] * bot


def boxes_to_mask(shape: tuple[int, int], boxes: Sequence[AMRBox]) -> Array:
    mask = np.zeros(shape, dtype=bool)
    for box in boxes:
        mask[box.y0 : box.y1, box.x0 : box.x1] = True
    return mask


def _align_down(value: int, multiple: int, lower: int) -> int:
    return max(lower, (value // multiple) * multiple)


def _align_up(value: int, multiple: int, upper: int) -> int:
    return min(upper, ((value + multiple - 1) // multiple) * multiple)


def _expand_small_interval(
    start: int,
    end: int,
    lower: int,
    upper: int,
    minimum: int,
    blocking: int,
) -> tuple[int, int]:
    if end - start >= minimum:
        return start, end
    while end - start < minimum:
        can_grow_left = start - blocking >= lower
        can_grow_right = end + blocking <= upper
        if can_grow_left:
            start -= blocking
        elif can_grow_right:
            end += blocking
        else:
            break
    return start, end


def cluster_tagged_child_boxes(
    tags: Array,
    parent_boxes: Sequence[AMRBox],
    child_shape: tuple[int, int],
    child_level: int,
    refinement: int = 2,
    blocking: int = 4,
    buffer_cells: int = 1,
    max_component_blocks: int | None = None,
) -> list[AMRBox]:
    """Cluster tagged parent-level cells into nested child-level patch boxes."""

    boxes: list[AMRBox] = []
    structure = np.ones((3, 3), dtype=bool)
    for parent in parent_boxes:
        sub = tags[parent.y0 : parent.y1, parent.x0 : parent.x1].astype(bool)
        if not np.any(sub):
            continue
        if buffer_cells > 0:
            sub = ndimage.binary_dilation(sub, structure=structure, iterations=buffer_cells)
        labels, count = ndimage.label(sub, structure=structure)
        py0, py1, px0, px1 = parent.child_extent(refinement)

        def append_extent(y0: int, y1: int, x0: int, x1: int) -> None:
            cy0 = y0 * refinement
            cy1 = y1 * refinement
            cx0 = x0 * refinement
            cx1 = x1 * refinement
            cy0 = _align_down(cy0, blocking, py0)
            cy1 = _align_up(cy1, blocking, py1)
            cx0 = _align_down(cx0, blocking, px0)
            cx1 = _align_up(cx1, blocking, px1)
            cy0, cy1 = _expand_small_interval(
                cy0, cy1, py0, py1, min(blocking * 2, py1 - py0), blocking
            )
            cx0, cx1 = _expand_small_interval(
                cx0, cx1, px0, px1, min(blocking * 2, px1 - px0), blocking
            )

            cy0 = max(0, min(child_shape[0], cy0))
            cy1 = max(0, min(child_shape[0], cy1))
            cx0 = max(0, min(child_shape[1], cx0))
            cx1 = max(0, min(child_shape[1], cx1))
            if cy1 > cy0 and cx1 > cx0:
                boxes.append(AMRBox(child_level, cy0, cy1, cx0, cx1))

        for component in range(1, count + 1):
            coords = np.argwhere(labels == component)
            if coords.size == 0:
                continue
            y0 = int(coords[:, 0].min()) + parent.y0
            y1 = int(coords[:, 0].max()) + parent.y0 + 1
            x0 = int(coords[:, 1].min()) + parent.x0
            x1 = int(coords[:, 1].max()) + parent.x0 + 1

            if max_component_blocks is None:
                append_extent(y0, y1, x0, x1)
                continue

            tile = max(1, max_component_blocks * blocking)
            if y1 - y0 <= tile and x1 - x0 <= tile:
                append_extent(y0, y1, x0, x1)
                continue

            component_mask = labels == component
            for ty0 in range(y0, y1, tile):
                ty1 = min(y1, ty0 + tile)
                for tx0 in range(x0, x1, tile):
                    tx1 = min(x1, tx0 + tile)
                    local = component_mask[
                        ty0 - parent.y0 : ty1 - parent.y0,
                        tx0 - parent.x0 : tx1 - parent.x0,
                    ]
                    if not np.any(local):
                        continue
                    local_coords = np.argwhere(local)
                    sy0 = int(local_coords[:, 0].min()) + ty0
                    sy1 = int(local_coords[:, 0].max()) + ty0 + 1
                    sx0 = int(local_coords[:, 1].min()) + tx0
                    sx1 = int(local_coords[:, 1].max()) + tx0 + 1
                    append_extent(sy0, sy1, sx0, sx1)

    unique: dict[tuple[int, int, int, int, int], AMRBox] = {}
    for box in boxes:
        unique[(box.level, box.y0, box.y1, box.x0, box.x1)] = box
    return sorted(unique.values(), key=lambda b: (b.level, b.y0, b.x0, b.y1, b.x1))


def build_block_structured_hierarchy(
    fine_active: Array,
    fine_rhs: Array,
    fine_fixed: Array,
    fine_fixed_values: Array,
    fine_indicator: Array,
    fine_reaction: float,
    max_level: int = 2,
    refinement: int = 2,
    tag_thresholds: Sequence[float] = (0.20, 0.42),
    blocking: int = 4,
    buffer_cells: int = 1,
    max_component_blocks: int | None = None,
) -> AMRHierarchy:
    if max_level < 1:
        raise ValueError("max_level must be at least 1 for a block-structured AMR hierarchy")
    total_factor = refinement**max_level
    _require_divisible(fine_active.shape, total_factor)
    if len(tag_thresholds) < max_level:
        raise ValueError("tag_thresholds must provide one threshold for each coarse-to-fine transition")

    fine_active = fine_active.astype(bool)
    fine_fixed = fine_fixed.astype(bool)
    levels: list[AMRLevel] = []
    for level in range(max_level + 1):
        factor = refinement ** (max_level - level)
        active = restrict_bool_any(fine_active, factor)
        fixed = restrict_bool_any(fine_fixed, factor) & active
        rhs, rhs_valid = restrict_weighted_mean(fine_rhs.astype(np.float64), fine_active, factor)
        rhs = rhs * float(factor * factor)
        rhs[~rhs_valid] = 0.0
        fixed_values, fixed_valid = restrict_weighted_mean(
            fine_fixed_values.astype(np.float64),
            fine_fixed,
            factor,
        )
        fixed_values[~fixed_valid] = 0.0
        indicator = restrict_max(fine_indicator.astype(np.float64), factor)
        reaction = float(fine_reaction) * float(factor * factor)
        shape = active.shape
        boxes = [AMRBox(0, 0, shape[0], 0, shape[1])] if level == 0 else []
        levels.append(
            AMRLevel(
                level=level,
                shape=shape,
                active=active,
                rhs=rhs,
                fixed=fixed,
                fixed_values=fixed_values,
                indicator=indicator,
                reaction=reaction,
                boxes=boxes,
            )
        )

    for parent_level in range(max_level):
        parent = levels[parent_level]
        child = levels[parent_level + 1]
        parent_patch_mask = boxes_to_mask(parent.shape, parent.boxes)
        tags = (parent.indicator >= float(tag_thresholds[parent_level])) & parent.active & parent_patch_mask
        child.boxes = cluster_tagged_child_boxes(
            tags,
            parent.boxes,
            child.shape,
            child_level=parent_level + 1,
            refinement=refinement,
            blocking=blocking,
            buffer_cells=buffer_cells,
            max_component_blocks=max_component_blocks,
        )
        if not child.boxes:
            # Keep the hierarchy valid even for very smooth manufactured cases.
            y0 = child.shape[0] // 4
            y1 = 3 * child.shape[0] // 4
            x0 = child.shape[1] // 4
            x1 = 3 * child.shape[1] // 4
            child.boxes = [AMRBox(parent_level + 1, y0, y1, x0, x1)]

    return AMRHierarchy(
        levels=levels,
        refinement=refinement,
        fine_active=fine_active,
        fine_rhs=fine_rhs.astype(np.float64),
        fine_fixed=fine_fixed,
        fine_fixed_values=fine_fixed_values.astype(np.float64),
        fine_indicator=fine_indicator.astype(np.float64),
        fine_reaction=float(fine_reaction),
    )


def _patch_interface_fixed(global_fixed: Array, box: AMRBox) -> Array:
    local_fixed = global_fixed[box.y0 : box.y1, box.x0 : box.x1].copy()
    h, w = global_fixed.shape
    if box.y0 > 0:
        local_fixed[0, :] = True
    if box.y1 < h:
        local_fixed[-1, :] = True
    if box.x0 > 0:
        local_fixed[:, 0] = True
    if box.x1 < w:
        local_fixed[:, -1] = True
    return local_fixed


def _solve_level_patch(
    current: Array,
    background: Array,
    level: AMRLevel,
    box: AMRBox,
    iterations: int,
    use_direct: bool,
) -> Array:
    local_active = level.active[box.y0 : box.y1, box.x0 : box.x1]
    if not np.any(local_active):
        return current[box.y0 : box.y1, box.x0 : box.x1].copy()
    local_rhs = level.rhs[box.y0 : box.y1, box.x0 : box.x1]
    local_fixed = _patch_interface_fixed(level.fixed, box)
    local_fixed_values = background[box.y0 : box.y1, box.x0 : box.x1].copy()
    original_fixed = level.fixed[box.y0 : box.y1, box.x0 : box.x1]
    original_fixed_values = level.fixed_values[box.y0 : box.y1, box.x0 : box.x1]
    local_fixed_values[original_fixed] = original_fixed_values[original_fixed]

    if np.all(local_fixed[local_active]):
        out = current[box.y0 : box.y1, box.x0 : box.x1].copy()
        out[local_fixed & local_active] = local_fixed_values[local_fixed & local_active]
        return out
    if use_direct:
        try:
            return direct_scalar_solve(local_active, local_rhs, local_fixed, local_fixed_values, level.reaction)
        except Exception:
            pass
    local_initial = current[box.y0 : box.y1, box.x0 : box.x1].copy()
    local_initial[local_fixed & local_active] = local_fixed_values[local_fixed & local_active]
    return jacobi_relax(
        local_initial,
        local_rhs,
        local_active,
        local_fixed,
        local_fixed_values,
        level.reaction,
        iterations=iterations,
        omega=0.82,
    )


def _restrict_child_field_to_parent(
    parent_field: Array,
    parent_level: AMRLevel,
    child_field: Array,
    child_level: AMRLevel,
    child_box: AMRBox,
    refinement: int,
    weight: float,
) -> None:
    py0 = child_box.y0 // refinement
    py1 = child_box.y1 // refinement
    px0 = child_box.x0 // refinement
    px1 = child_box.x1 // refinement
    cy0 = py0 * refinement
    cy1 = py1 * refinement
    cx0 = px0 * refinement
    cx1 = px1 * refinement
    child_values = child_field[cy0:cy1, cx0:cx1]
    child_active = child_level.active[cy0:cy1, cx0:cx1]
    restricted, valid = restrict_weighted_mean(child_values, child_active, refinement)
    target_active = parent_level.active[py0:py1, px0:px1] & ~parent_level.fixed[py0:py1, px0:px1] & valid
    target = parent_field[py0:py1, px0:px1]
    target[target_active] = (1.0 - weight) * target[target_active] + weight * restricted[target_active]


def composite_to_finest(hierarchy: AMRHierarchy, fields: Sequence[Array]) -> Array:
    out = prolong_bilinear(fields[0], hierarchy.refinement**hierarchy.max_level, hierarchy.fine_shape)
    for level_idx in range(1, hierarchy.max_level + 1):
        level = hierarchy.levels[level_idx]
        factor = hierarchy.refinement ** (hierarchy.max_level - level_idx)
        for box in level.boxes:
            patch = fields[level_idx][box.y0 : box.y1, box.x0 : box.x1]
            if factor > 1:
                patch_fine = prolong_bilinear(patch, factor, (box.height * factor, box.width * factor))
            else:
                patch_fine = patch.copy()
            y0 = box.y0 * factor
            y1 = box.y1 * factor
            x0 = box.x0 * factor
            x1 = box.x1 * factor
            out[y0:y1, x0:x1] = patch_fine
    out[~hierarchy.fine_active] = 0.0
    return out


def solve_block_structured_amr(
    hierarchy: AMRHierarchy,
    config: AMRSolveConfig | None = None,
) -> AMRSolveResult:
    cfg = config or AMRSolveConfig()
    fields = [np.zeros(level.shape, dtype=np.float64) for level in hierarchy.levels]
    for field, level in zip(fields, hierarchy.levels):
        field[level.fixed] = level.fixed_values[level.fixed]
        field[~level.active] = 0.0
    coarse = hierarchy.levels[0]
    try:
        fields[0] = direct_scalar_solve(
            coarse.active,
            coarse.rhs,
            coarse.fixed,
            coarse.fixed_values,
            coarse.reaction,
        )
    except Exception:
        pass

    residual_history: list[float] = []
    for cycle in range(1, cfg.cycles + 1):
        coarse = hierarchy.levels[0]
        fields[0] = jacobi_relax(
            fields[0],
            coarse.rhs,
            coarse.active,
            coarse.fixed,
            coarse.fixed_values,
            coarse.reaction,
            iterations=cfg.coarse_iters,
            omega=0.78,
        )

        for level_idx in range(1, hierarchy.max_level + 1):
            parent = hierarchy.levels[level_idx - 1]
            level = hierarchy.levels[level_idx]
            background = prolong_bilinear(fields[level_idx - 1], hierarchy.refinement, level.shape)
            patch_mask = boxes_to_mask(level.shape, level.boxes) & level.active
            fields[level_idx][~patch_mask] = background[~patch_mask]
            fields[level_idx][level.fixed] = level.fixed_values[level.fixed]

            updated = fields[level_idx].copy()
            for box in level.boxes:
                patch = _solve_level_patch(
                    fields[level_idx],
                    background,
                    level,
                    box,
                    iterations=cfg.patch_iters,
                    use_direct=cfg.use_direct_patch_solve,
                )
                local_active = level.active[box.y0 : box.y1, box.x0 : box.x1]
                target = updated[box.y0 : box.y1, box.x0 : box.x1]
                target[local_active] = patch[local_active]
            fields[level_idx] = updated
            fields[level_idx][level.fixed] = level.fixed_values[level.fixed]
            fields[level_idx][~level.active] = 0.0

            for box in level.boxes:
                _restrict_child_field_to_parent(
                    fields[level_idx - 1],
                    parent,
                    fields[level_idx],
                    level,
                    box,
                    hierarchy.refinement,
                    cfg.restriction_weight,
                )

        composite = composite_to_finest(hierarchy, fields)
        res = residual_norm(
            composite,
            hierarchy.fine_rhs,
            hierarchy.fine_active,
            hierarchy.fine_fixed,
            hierarchy.fine_reaction,
        )
        residual_history.append(res)
        if res <= cfg.tolerance:
            break

    return AMRSolveResult(
        fields=fields,
        composite=composite_to_finest(hierarchy, fields),
        residual_history=residual_history,
        cycles=len(residual_history),
    )


def composite_level_map(hierarchy: AMRHierarchy) -> Array:
    level_map = np.zeros(hierarchy.fine_shape, dtype=np.int16)
    for level_idx in range(1, hierarchy.max_level + 1):
        level = hierarchy.levels[level_idx]
        factor = hierarchy.refinement ** (hierarchy.max_level - level_idx)
        for box in level.boxes:
            y0 = box.y0 * factor
            y1 = box.y1 * factor
            x0 = box.x0 * factor
            x1 = box.x1 * factor
            level_map[y0:y1, x0:x1] = level_idx
    level_map[~hierarchy.fine_active] = -1
    return level_map


def draw_amr_grid(hierarchy: AMRHierarchy, draw_interior: bool = True) -> Array:
    h, w = hierarchy.fine_shape
    grid = np.zeros((h, w), dtype=np.float64)
    for level_idx, level in enumerate(hierarchy.levels):
        factor = hierarchy.refinement ** (hierarchy.max_level - level_idx)
        value = 0.25 + 0.75 * (level_idx / max(1, hierarchy.max_level))
        for box in level.boxes:
            y0 = box.y0 * factor
            y1 = box.y1 * factor
            x0 = box.x0 * factor
            x1 = box.x1 * factor
            grid[y0:y1, x0] = np.maximum(grid[y0:y1, x0], value)
            grid[y0:y1, x1 - 1] = np.maximum(grid[y0:y1, x1 - 1], value)
            grid[y0, x0:x1] = np.maximum(grid[y0, x0:x1], value)
            grid[y1 - 1, x0:x1] = np.maximum(grid[y1 - 1, x0:x1], value)
            if draw_interior and factor > 1:
                ys = np.arange(y0, y1, factor)
                xs = np.arange(x0, x1, factor)
                grid[ys, x0:x1] = np.maximum(grid[ys, x0:x1], value * 0.6)
                grid[y0:y1, xs] = np.maximum(grid[y0:y1, xs], value * 0.6)
    return grid


def equivalent_cell_count(hierarchy: AMRHierarchy) -> dict[str, int | float]:
    level_map = composite_level_map(hierarchy)
    counts: dict[str, int | float] = {}
    total_equiv = 0.0
    for level_idx in range(hierarchy.max_level + 1):
        fine_cells = int(np.sum(level_map == level_idx))
        factor = hierarchy.refinement ** (hierarchy.max_level - level_idx)
        equiv = fine_cells / float(factor * factor)
        counts[f"level_{level_idx}_fine_cells"] = fine_cells
        counts[f"level_{level_idx}_stored_cells"] = int(round(equiv))
        total_equiv += equiv
    counts["equivalent_stored_cells"] = int(round(total_equiv))
    counts["uniform_fine_cells"] = int(np.prod(hierarchy.fine_shape))
    counts["stored_cell_fraction_vs_uniform_fine"] = total_equiv / float(np.prod(hierarchy.fine_shape))
    return counts


def composite_residual_field(hierarchy: AMRHierarchy, composite: Array) -> Array:
    residual = hierarchy.fine_rhs - graph_laplacian_apply(
        composite,
        hierarchy.fine_active,
        hierarchy.fine_reaction,
    )
    residual[~(hierarchy.fine_active & ~hierarchy.fine_fixed)] = 0.0
    return residual


def _child_coverage_on_parent(hierarchy: AMRHierarchy, parent_level_idx: int) -> Array:
    parent = hierarchy.levels[parent_level_idx]
    covered = np.zeros(parent.shape, dtype=bool)
    if parent_level_idx >= hierarchy.max_level:
        return covered
    for box in hierarchy.levels[parent_level_idx + 1].boxes:
        y0 = box.y0 // hierarchy.refinement
        y1 = box.y1 // hierarchy.refinement
        x0 = box.x0 // hierarchy.refinement
        x1 = box.x1 // hierarchy.refinement
        covered[y0:y1, x0:x1] = True
    return covered


def solve_composite_amr_direct(hierarchy: AMRHierarchy) -> CompositeAMRDirectResult:
    """Solve the scalar elliptic problem on the AMR leaf-cell finite-volume mesh.

    This assembles one unknown per leaf cell. Coarse-fine interface coupling uses
    face-overlap conductances, so a coarse face communicates with all adjacent
    finer cells along that face.
    """

    cells: list[dict[str, int | float | bool]] = []
    owner = -np.ones(hierarchy.fine_shape, dtype=np.int64)
    leaf_level_map = -np.ones(hierarchy.fine_shape, dtype=np.int16)
    h_fine, w_fine = hierarchy.fine_shape

    for level_idx, level in enumerate(hierarchy.levels):
        patch_mask = boxes_to_mask(level.shape, level.boxes)
        child_covered = _child_coverage_on_parent(hierarchy, level_idx)
        leaf_mask = patch_mask & ~child_covered & level.active
        scale = hierarchy.refinement ** (hierarchy.max_level - level_idx)
        for yy, xx in np.argwhere(leaf_mask):
            y = int(yy)
            x = int(xx)
            fy0 = y * scale
            fy1 = (y + 1) * scale
            fx0 = x * scale
            fx1 = (x + 1) * scale
            fine_active = hierarchy.fine_active[fy0:fy1, fx0:fx1]
            if not np.any(fine_active):
                continue
            fine_fixed = hierarchy.fine_fixed[fy0:fy1, fx0:fx1] & fine_active
            active_count = int(fine_active.sum())
            rhs_sum = float(np.sum(hierarchy.fine_rhs[fy0:fy1, fx0:fx1][fine_active]))
            fixed = bool(np.any(fine_fixed))
            fixed_value = 0.0
            if fixed:
                fixed_value = float(np.mean(hierarchy.fine_fixed_values[fy0:fy1, fx0:fx1][fine_fixed]))
            cell_id = len(cells)
            owner_region = owner[fy0:fy1, fx0:fx1]
            owner_region[fine_active] = cell_id
            leaf_region = leaf_level_map[fy0:fy1, fx0:fx1]
            leaf_region[fine_active] = level_idx
            cells.append(
                {
                    "level": level_idx,
                    "scale": scale,
                    "y0": fy0,
                    "y1": fy1,
                    "x0": fx0,
                    "x1": fx1,
                    "active_count": active_count,
                    "rhs": rhs_sum,
                    "fixed": fixed,
                    "fixed_value": fixed_value,
                }
            )

    free_ids = [i for i, cell in enumerate(cells) if not bool(cell["fixed"])]
    free_index = {cell_id: row for row, cell_id in enumerate(free_ids)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b = np.zeros(len(free_ids), dtype=np.float64)

    pair_weights: dict[tuple[int, int], float] = {}

    def add_pair(i: int, j: int, weight: float) -> None:
        if i == j or weight <= 0.0:
            return
        a, b_id = (i, j) if i < j else (j, i)
        pair_weights[(a, b_id)] = pair_weights.get((a, b_id), 0.0) + weight

    for i, cell in enumerate(cells):
        y0 = int(cell["y0"])
        y1 = int(cell["y1"])
        x0 = int(cell["x0"])
        x1 = int(cell["x1"])
        scale_i = float(cell["scale"])
        if x1 < w_fine:
            neigh = owner[y0:y1, x1]
            for j in np.unique(neigh[neigh >= 0]):
                scale_j = float(cells[int(j)]["scale"])
                overlap = float(np.sum(neigh == j))
                add_pair(i, int(j), overlap / (0.5 * (scale_i + scale_j)))
        if y1 < h_fine:
            neigh = owner[y1, x0:x1]
            for j in np.unique(neigh[neigh >= 0]):
                scale_j = float(cells[int(j)]["scale"])
                overlap = float(np.sum(neigh == j))
                add_pair(i, int(j), overlap / (0.5 * (scale_i + scale_j)))

    diag = np.zeros(len(free_ids), dtype=np.float64)
    for cell_id in free_ids:
        row = free_index[cell_id]
        cell = cells[cell_id]
        b[row] = float(cell["rhs"])
        diag[row] += hierarchy.fine_reaction * float(cell["active_count"])

    for (i, j), weight in pair_weights.items():
        i_fixed = bool(cells[i]["fixed"])
        j_fixed = bool(cells[j]["fixed"])
        if not i_fixed:
            row_i = free_index[i]
            diag[row_i] += weight
            if j_fixed:
                b[row_i] += weight * float(cells[j]["fixed_value"])
            else:
                rows.append(row_i)
                cols.append(free_index[j])
                data.append(-weight)
        if not j_fixed:
            row_j = free_index[j]
            diag[row_j] += weight
            if i_fixed:
                b[row_j] += weight * float(cells[i]["fixed_value"])
            else:
                rows.append(row_j)
                cols.append(free_index[i])
                data.append(-weight)

    for row, value in enumerate(diag):
        rows.append(row)
        cols.append(row)
        data.append(value)

    if len(free_ids) == 0:
        solution = np.zeros(0, dtype=np.float64)
    else:
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(free_ids), len(free_ids)))
        solution = sparse_linalg.spsolve(matrix, b)

    cell_values = np.zeros(len(cells), dtype=np.float64)
    for i, cell in enumerate(cells):
        if bool(cell["fixed"]):
            cell_values[i] = float(cell["fixed_value"])
        else:
            cell_values[i] = solution[free_index[i]]

    residual_sq = 0.0
    rhs_sq = 0.0
    for cell_id in free_ids:
        row = free_index[cell_id]
        value = diag[row] * cell_values[cell_id]
        for (i, j), weight in pair_weights.items():
            if i == cell_id:
                value -= weight * cell_values[j]
            elif j == cell_id:
                value -= weight * cell_values[i]
        residual = float(cells[cell_id]["rhs"]) - value
        residual_sq += residual * residual
        rhs_sq += float(cells[cell_id]["rhs"]) ** 2

    field = np.zeros(hierarchy.fine_shape, dtype=np.float64)
    for i, cell in enumerate(cells):
        y0 = int(cell["y0"])
        y1 = int(cell["y1"])
        x0 = int(cell["x0"])
        x1 = int(cell["x1"])
        region_active = hierarchy.fine_active[y0:y1, x0:x1]
        target = field[y0:y1, x0:x1]
        target[region_active] = cell_values[i]
    field[~hierarchy.fine_active] = 0.0

    operator_residual = float(np.sqrt(residual_sq) / (np.sqrt(rhs_sq) + 1.0e-12))
    return CompositeAMRDirectResult(
        field=field,
        leaf_level_map=leaf_level_map,
        leaf_count=len(cells),
        free_count=len(free_ids),
        operator_residual=operator_residual,
    )
