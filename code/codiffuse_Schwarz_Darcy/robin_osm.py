"""Exact mixed-Robin overlapping Schwarz recurrence for Darcy.

This is a transparent Mao-style Robin OSM reimplementation, not an official
SNAP-DDM code release.  At every sweep, artificial local faces receive Robin
data from the previous global POU consensus; true geometry faces retain the
problem's physical Dirichlet data.  Every local update is an exact sparse FV
solve, so it is a reference recurrence rather than the neural approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover

from .physics import FACE_OFFSETS, exposed_dirichlet_faces
from .robin_v3 import robin_exchange_from_neighbor, solve_mixed_robin
from .robin_v3 import (
    ROBIN_LOCAL_CONTRACT, RobinDarcyAdapter, robin_batch_to_transition,
)
from .adapter import DarcyStatistics, build_model


Array = np.ndarray


@dataclass(frozen=True)
class RobinSweep:
    field: Array
    local_fields: Array
    relative_update: float
    local_solves: int


def load_neural_robin_local(
    checkpoint_path, device: torch.device,
) -> tuple[torch.nn.Module, RobinDarcyAdapter, dict]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )
    if checkpoint.get("contract") != ROBIN_LOCAL_CONTRACT:
        raise ValueError("neural Robin recurrence requires a Robin-v3 checkpoint")
    adapter = RobinDarcyAdapter(
        DarcyStatistics.from_json(checkpoint["statistics"]),
    )
    model = build_model(adapter, communication_stage="none").to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, adapter, checkpoint


def _tile_origin(cover: CartesianCover, tile: int) -> tuple[int, int]:
    ny, nx = cover.grid_shape
    if not 0 <= tile < ny * nx:
        raise IndexError("tile index out of range")
    iy, ix = divmod(tile, nx)
    return (
        iy * cover.core - cover.halo - cover.offset_y,
        ix * cover.core - cover.halo - cover.offset_x,
    )


def _extract_tile(field: Array, cover: CartesianCover, tile: int) -> Array:
    field = np.asarray(field)
    y0, x0 = _tile_origin(cover, tile)
    size = cover.tile_size
    output = np.zeros((size, size), dtype=field.dtype)
    gy0, gy1 = max(y0, 0), min(y0 + size, cover.height)
    gx0, gx1 = max(x0, 0), min(x0 + size, cover.width)
    if gy1 > gy0 and gx1 > gx0:
        output[
            gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0,
        ] = field[gy0:gy1, gx0:gx1]
    return output


def mixed_tile_boundary_data(
    global_mask: Array,
    global_kappa: Array,
    current: Array,
    physical_face_mask: Array,
    physical_face_value: Array,
    cover: CartesianCover,
    tile: int,
    *,
    impedance_ratio: float,
    dx: float,
) -> tuple[Array, Array, Array, Array, Array]:
    """Build disjoint physical-Dirichlet and artificial-Robin tile faces."""
    if impedance_ratio <= 0.0:
        raise ValueError("impedance_ratio must be positive")
    mask = _extract_tile(np.asarray(global_mask, dtype=np.uint8), cover, tile) > 0
    exposed = exposed_dirichlet_faces(mask)
    shape = (4, cover.tile_size, cover.tile_size)
    dmask = np.zeros(shape, dtype=bool)
    dvalue = np.zeros(shape, dtype=np.float64)
    rmask = np.zeros(shape, dtype=bool)
    rvalue = np.zeros(shape, dtype=np.float64)
    impedance = np.zeros(shape, dtype=np.float64)
    y0, x0 = _tile_origin(cover, tile)
    for face, (dy, ddx) in enumerate(FACE_OFFSETS):
        for ly, lx in np.argwhere(exposed[face]):
            gy, gx = y0 + int(ly), x0 + int(lx)
            if not (0 <= gy < cover.height and 0 <= gx < cover.width):
                raise RuntimeError("active tile cell mapped outside global grid")
            if physical_face_mask[face, gy, gx]:
                dmask[face, ly, lx] = True
                dvalue[face, ly, lx] = physical_face_value[face, gy, gx]
                continue
            gn_y, gn_x = gy + dy, gx + ddx
            artificial = (
                0 <= gn_y < cover.height and 0 <= gn_x < cover.width
                and bool(global_mask[gn_y, gn_x])
            )
            if not artificial:
                raise ValueError(
                    "exposed local face is neither declared physical nor backed "
                    f"by an active global neighbor: tile={tile}, face={face}, "
                    f"global=({gy},{gx})"
                )
            center_k = float(global_kappa[gy, gx])
            neighbor_k = float(global_kappa[gn_y, gn_x])
            p = impedance_ratio * 2.0 * center_k / dx
            _, g = robin_exchange_from_neighbor(
                float(current[gy, gx]), float(current[gn_y, gn_x]),
                center_k, neighbor_k, p, dx,
            )
            rmask[face, ly, lx] = True
            rvalue[face, ly, lx] = float(g)
            impedance[face, ly, lx] = p
    if np.any(exposed & ~(dmask | rmask)) or np.any(dmask & rmask):
        raise RuntimeError("mixed tile face coverage is not a disjoint partition")
    return dmask, dvalue, rmask, rvalue, impedance


def exact_robin_osm_sweep(
    mask: Array,
    kappa: Array,
    source: Array,
    physical_face_mask: Array,
    physical_face_value: Array,
    current: Array,
    cover: CartesianCover,
    *,
    impedance_ratio: float,
    relaxation: float = 1.0,
    dx: float | None = None,
) -> RobinSweep:
    """Perform one parallel exact local solve and normalized-POU writeback."""
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must lie in (0,1]")
    mask = np.asarray(mask, dtype=bool)
    current = np.asarray(current, dtype=np.float64)
    if current.shape != mask.shape or mask.shape != (cover.height, cover.width):
        raise ValueError("global fields and cover dimensions differ")
    dx = 2.0 / max(mask.shape) if dx is None else float(dx)
    local_fields = []
    local_masks = []
    for tile in range(cover.tile_count):
        local_mask = _extract_tile(mask.astype(np.uint8), cover, tile) > 0
        local_kappa = _extract_tile(kappa, cover, tile).astype(np.float64)
        local_source = _extract_tile(source, cover, tile).astype(np.float64)
        if not local_mask.any():
            local_fields.append(np.zeros_like(local_kappa))
            local_masks.append(local_mask)
            continue
        dmask, dvalue, rmask, rvalue, impedance = mixed_tile_boundary_data(
            mask, kappa, current, physical_face_mask, physical_face_value,
            cover, tile, impedance_ratio=impedance_ratio, dx=dx,
        )
        local, _ = solve_mixed_robin(
            mask=local_mask, kappa=local_kappa, source=local_source,
            dirichlet_face_mask=dmask, dirichlet_face_value=dvalue,
            robin_face_mask=rmask, robin_face_value=rvalue,
            robin_face_impedance=impedance, dx=dx,
        )
        local_fields.append(local); local_masks.append(local_mask)
    tiles = torch.from_numpy(np.stack(local_fields)[:, None])
    weights = torch.from_numpy(np.stack(local_masks)[:, None].astype(np.float64))
    assembled = cover.assemble_uniform(tiles, 1, weights=weights)[0, 0].numpy()
    updated = ((1.0 - relaxation) * current + relaxation * assembled) * mask
    relative_update = float(
        np.linalg.norm((updated - current)[mask])
        / max(np.linalg.norm(updated[mask]), 1.0e-12)
    )
    return RobinSweep(
        updated, np.stack(local_fields), relative_update, cover.tile_count,
    )


@torch.no_grad()
def neural_robin_osm_sweep(
    model: torch.nn.Module,
    adapter: RobinDarcyAdapter,
    mask: Array,
    kappa: Array,
    source: Array,
    sdf: Array,
    normals: Array,
    physical_face_mask: Array,
    physical_face_value: Array,
    current: Array,
    cover: CartesianCover,
    *,
    impedance_ratio: float,
    relaxation: float = 1.0,
    dx: float | None = None,
    device: torch.device | None = None,
) -> RobinSweep:
    """One parallel neural mixed-Robin local call and normalized POU update."""
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must lie in (0,1]")
    device = next(model.parameters()).device if device is None else device
    mask = np.asarray(mask, dtype=bool)
    current = np.asarray(current, dtype=np.float64)
    dx = 2.0 / max(mask.shape) if dx is None else float(dx)
    records: dict[str, list[Array]] = {
        name: [] for name in (
            "mask", "logk", "source", "bc_mask", "bc_value", "bc_role",
            "sdf", "normals", "dirichlet_face_mask",
            "dirichlet_face_value", "dirichlet_face_role", "robin_face_mask",
            "robin_face_value", "robin_face_impedance", "robin_face_trace",
        )
    }
    local_masks = []
    for tile in range(cover.tile_count):
        local_mask = _extract_tile(mask.astype(np.uint8), cover, tile) > 0
        local_kappa = _extract_tile(kappa, cover, tile).astype(np.float64)
        local_source = _extract_tile(source, cover, tile).astype(np.float64)
        dmask, dvalue, rmask, rvalue, impedance = mixed_tile_boundary_data(
            mask, kappa, current, physical_face_mask, physical_face_value,
            cover, tile, impedance_ratio=impedance_ratio, dx=dx,
        ) if local_mask.any() else tuple(
            np.zeros((4, cover.tile_size, cover.tile_size), dtype=dtype)
            for dtype in (bool, np.float64, bool, np.float64, np.float64)
        )
        physical_count = dmask.sum(0)
        cell_bc = (dvalue * dmask).sum(0) / np.maximum(physical_count, 1)
        records["mask"].append(local_mask[None])
        records["logk"].append(np.log10(np.maximum(local_kappa, 1.0e-12))[None])
        records["source"].append(local_source[None])
        records["bc_mask"].append((physical_count > 0)[None])
        records["bc_value"].append(cell_bc[None])
        records["bc_role"].append((physical_count > 0)[None])
        records["sdf"].append(_extract_tile(sdf, cover, tile)[None])
        records["normals"].append(np.stack((
            _extract_tile(normals[0], cover, tile),
            _extract_tile(normals[1], cover, tile),
        )))
        records["dirichlet_face_mask"].append(dmask)
        records["dirichlet_face_value"].append(dvalue)
        records["dirichlet_face_role"].append(dmask.astype(np.uint8))
        records["robin_face_mask"].append(rmask)
        records["robin_face_value"].append(rvalue)
        records["robin_face_impedance"].append(impedance)
        records["robin_face_trace"].append(np.zeros_like(rvalue))
        local_masks.append(local_mask)
    batch = {
        name: torch.from_numpy(np.stack(values)).to(device=device, dtype=torch.float32)
        for name, values in records.items()
    }
    zero = torch.zeros(
        cover.tile_count, 1, cover.tile_size, cover.tile_size,
        device=device, dtype=torch.float32,
    )
    transition = robin_batch_to_transition(batch, zero, target=None)
    phase = torch.ones(cover.tile_count, device=device)
    local_cover = CartesianCover(
        cover.tile_size, cover.tile_size, cover.tile_size, 0,
    )
    condition = adapter.conditions(transition, phase, 1.0)
    physical, scalar = condition.tiled(local_cover)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        normalized = model(
            physical, scalar, 1, tile_grid_shape=(1, 1),
        )
        prediction = adapter.apply_endpoint(transition, normalized)
    tiles = prediction.float().cpu()
    weights = torch.from_numpy(
        np.stack(local_masks)[:, None].astype(np.float32),
    )
    assembled = cover.assemble_uniform(tiles, 1, weights=weights)[0, 0].numpy()
    updated = ((1.0 - relaxation) * current + relaxation * assembled) * mask
    relative_update = float(
        np.linalg.norm((updated - current)[mask])
        / max(np.linalg.norm(updated[mask]), 1.0e-12)
    )
    return RobinSweep(
        updated, tiles[:, 0].numpy(), relative_update, cover.tile_count,
    )


def exact_robin_osm(
    mask: Array,
    kappa: Array,
    source: Array,
    physical_face_mask: Array,
    physical_face_value: Array,
    cover: CartesianCover,
    *,
    sweeps: int,
    initial: Array | None = None,
    impedance_ratio: float,
    relaxation: float = 1.0,
    dx: float | None = None,
) -> tuple[Array, list[float], int]:
    """Run a fixed number of exact sweeps and archive the update trajectory."""
    if sweeps <= 0:
        raise ValueError("sweeps must be positive")
    mask = np.asarray(mask, dtype=bool)
    current = np.zeros(mask.shape, dtype=np.float64) if initial is None \
        else np.asarray(initial, dtype=np.float64).copy()
    trajectory = []
    local_solves = 0
    for _ in range(sweeps):
        report = exact_robin_osm_sweep(
            mask, kappa, source, physical_face_mask, physical_face_value,
            current, cover, impedance_ratio=impedance_ratio,
            relaxation=relaxation, dx=dx,
        )
        current = report.field
        trajectory.append(report.relative_update)
        local_solves += report.local_solves
    return current, trajectory, local_solves
