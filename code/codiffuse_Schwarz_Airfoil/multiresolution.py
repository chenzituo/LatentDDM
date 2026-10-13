"""Paired Cartesian levels and conservative coarse/fine synchronization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from codiffuse_Schwarz_Airfoil.cartesian_cover import vorticity_magnitude
from codiffuse_Schwarz_Airfoil.dataset import (
    FieldStatistics,
    usable_transition_stop,
)


Tensor = torch.Tensor


@dataclass(frozen=True)
class NestedLevelBox:
    """Parent-level index box covered by one nested fine-level window."""

    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0


def nested_level_box(
    coarse_window: tuple[float, float, float, float],
    fine_window: tuple[float, float, float, float],
    *,
    coarse_dx: float,
    coarse_shape: tuple[int, int],
    fine_shape: tuple[int, int],
    factor: int = 2,
) -> NestedLevelBox:
    """Return the aligned coarse box covered by a nested fine grid."""
    if coarse_dx <= 0.0 or factor <= 0:
        raise ValueError("spacing and refinement factor must be positive")
    cx0, cx1, cy0, cy1 = coarse_window
    fx0, fx1, fy0, fy1 = fine_window
    if not (cx0 <= fx0 < fx1 <= cx1 and cy0 <= fy0 < fy1 <= cy1):
        raise ValueError("fine window must be nested inside coarse window")
    indices = torch.tensor([
        (fy0 - cy0) / coarse_dx, (fy1 - cy0) / coarse_dx,
        (fx0 - cx0) / coarse_dx, (fx1 - cx0) / coarse_dx,
    ], dtype=torch.float64)
    rounded = indices.round().to(torch.int64)
    if not torch.allclose(indices, rounded.to(indices.dtype), atol=1.0e-9, rtol=0.0):
        raise ValueError("nested window is not aligned to the coarse grid")
    box = NestedLevelBox(*map(int, rounded.tolist()))
    if box.y0 < 0 or box.x0 < 0 or box.y1 > coarse_shape[0] or box.x1 > coarse_shape[1]:
        raise ValueError("nested box lies outside the coarse array")
    expected = (factor * box.shape[0], factor * box.shape[1])
    if expected != fine_shape:
        raise ValueError(f"fine shape {fine_shape} does not match nested box {expected}")
    return box


def proper_nesting_mask(
    batch: int,
    parent_shape: tuple[int, int],
    *,
    buffer_cells: int | tuple[int, int, int, int],
    device: torch.device,
) -> Tensor:
    """Select fine-owned cells after symmetric or directional guard margins.

    Tuple margins are ``(left, right, bottom, top)`` in parent-level cells.
    """
    if isinstance(buffer_cells, int):
        margins = (buffer_cells,) * 4
    else:
        if len(buffer_cells) != 4:
            raise ValueError("directional nesting needs four margins")
        margins = tuple(int(value) for value in buffer_cells)
    if batch <= 0 or min(margins) < 0:
        raise ValueError("batch must be positive and buffers nonnegative")
    height, width = parent_shape
    left, right, bottom, top = margins
    if left + right >= width or bottom + top >= height:
        raise ValueError("proper-nesting buffer removes the complete fine region")
    mask = torch.zeros(batch, 1, height, width, device=device, dtype=torch.bool)
    y_stop = None if top == 0 else -top
    x_stop = None if right == 0 else -right
    mask[..., bottom:y_stop, left:x_stop] = True
    return mask


def synchronize_nested_levels(
    coarse: Tensor,
    fine: Tensor,
    box: NestedLevelBox,
    refinement_weight: Tensor,
    *,
    factor: int = 2,
    fine_trust: float = 1.0,
    replace_inactive_fine: bool = True,
) -> tuple[Tensor, Tensor]:
    """Synchronize a nested fine window with its parent-level subarray."""
    parent = coarse[..., box.y0:box.y1, box.x0:box.x1]
    synchronized_parent, synchronized_fine = synchronize_levels(
        parent, fine, refinement_weight, factor=factor, fine_trust=fine_trust,
    )
    if replace_inactive_fine:
        active = constant_prolongation(
            refinement_weight.to(device=fine.device, dtype=torch.bool), factor,
        )
        parent_view = constant_prolongation(synchronized_parent, factor)
        synchronized_fine = torch.where(active, synchronized_fine, parent_view)
    synchronized_coarse = coarse.clone()
    synchronized_coarse[..., box.y0:box.y1, box.x0:box.x1] = synchronized_parent
    return synchronized_coarse, synchronized_fine


def conservative_restrict(fine: Tensor, factor: int = 2) -> Tensor:
    """Cell-average a fine cell-centered field onto its parent Cartesian level."""
    if fine.ndim != 4:
        raise ValueError("fine field must have shape (batch, channels, height, width)")
    if factor <= 0 or fine.shape[-2] % factor or fine.shape[-1] % factor:
        raise ValueError("fine dimensions must be divisible by the positive factor")
    return F.avg_pool2d(fine, kernel_size=factor, stride=factor)


def constant_prolongation(coarse: Tensor, factor: int = 2) -> Tensor:
    """Prolong a parent-cell correction without changing its block mean."""
    if coarse.ndim != 4 or factor <= 0:
        raise ValueError("coarse field must be BCHW and factor must be positive")
    return coarse.repeat_interleave(factor, dim=-2).repeat_interleave(factor, dim=-1)


def synchronize_levels(
    coarse: Tensor,
    fine: Tensor,
    refinement_weight: Tensor,
    *,
    factor: int = 2,
    fine_trust: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Synchronize predicted levels with exact refined-cell average consistency.

    ``fine_trust=1`` is classical AMR average-down. Intermediate values form a
    confidence-weighted consensus between the independently predicted levels.
    Fine corrections are parent-cell constants, so they introduce no subcell
    oscillation and make restriction equal the synchronized coarse value.
    """
    restricted = conservative_restrict(fine, factor)
    if coarse.shape != restricted.shape:
        raise ValueError("coarse shape must match conservative restriction of fine")
    if refinement_weight.shape != (coarse.shape[0], 1, *coarse.shape[-2:]):
        raise ValueError("refinement weight must have shape (batch, 1, coarse H, coarse W)")
    if not 0.0 <= fine_trust <= 1.0:
        raise ValueError("fine_trust must lie in [0, 1]")
    weight = refinement_weight.to(device=coarse.device, dtype=coarse.dtype).clamp(0.0, 1.0)
    consensus = coarse + fine_trust * weight * (restricted - coarse)
    active = (weight > 0.0).to(coarse.dtype)
    correction = (consensus - restricted) * active
    synchronized_fine = fine + constant_prolongation(correction, factor)
    return consensus, synchronized_fine


def decode_composite(
    coarse: Tensor,
    fine: Tensor,
    refinement_mask: Tensor,
    *,
    factor: int = 2,
) -> Tensor:
    """Decode a fine-lattice view using fine values only in refined parents."""
    restricted = conservative_restrict(fine, factor)
    if coarse.shape != restricted.shape:
        raise ValueError("coarse and fine levels are incompatible")
    if refinement_mask.shape != (coarse.shape[0], 1, *coarse.shape[-2:]):
        raise ValueError("refinement mask must be B1HW on the coarse level")
    coarse_view = constant_prolongation(coarse, factor)
    fine_mask = constant_prolongation(refinement_mask.to(coarse.dtype), factor) > 0.0
    return torch.where(fine_mask, fine, coarse_view)


def restriction_consistency_loss(
    coarse: Tensor,
    fine: Tensor,
    refinement_mask: Tensor,
    *,
    factor: int = 2,
) -> Tensor:
    """Relative coarse/fine disagreement restricted to active refined parents."""
    restricted = conservative_restrict(fine, factor)
    if coarse.shape != restricted.shape:
        raise ValueError("coarse and fine levels are incompatible")
    mask = refinement_mask.to(device=coarse.device, dtype=coarse.dtype)
    if mask.shape != (coarse.shape[0], 1, *coarse.shape[-2:]):
        raise ValueError("refinement mask must be B1HW on the coarse level")
    error = (restricted - coarse) * mask
    scale = (restricted.square() * mask).sum().clamp_min(1.0e-20)
    return error.square().sum() / scale


def patch_aligned_mask(mask: Tensor, patch_cells: int) -> Tensor:
    """Expand a cell mask to complete Cartesian patch cores."""
    if mask.ndim != 4 or mask.shape[1] != 1 or patch_cells <= 0:
        raise ValueError("mask must be B1HW and patch_cells must be positive")
    height, width = mask.shape[-2:]
    pad_h = (-height) % patch_cells
    pad_w = (-width) % patch_cells
    padded = F.pad(mask.float(), (0, pad_w, 0, pad_h))
    selected = F.max_pool2d(padded, patch_cells, patch_cells)
    expanded = F.interpolate(selected, scale_factor=patch_cells, mode="nearest")
    return expanded[..., :height, :width] > 0.0


def adaptive_refinement_mask(
    state: Tensor,
    sdf_n: Tensor,
    sdf_np1: Tensor,
    *,
    dx: float,
    previous_state: Tensor | None = None,
    previous_refinement_mask: Tensor | None = None,
    patch_cells: int = 16,
    geometry_band_cells: float = 12.0,
    vorticity_fraction: float = .15,
    vorticity_amplitude_fraction: float = .10,
    coarsen_amplitude_fraction: float = .05,
    buffer_cells: int = 2,
) -> Tensor:
    """Select complete fine-patch footprints from swept geometry and vorticity."""
    if state.ndim != 4 or state.shape[1] < 2:
        raise ValueError("state must be BCHW with at least two velocity channels")
    expected = (state.shape[0], 1, *state.shape[-2:])
    if sdf_n.shape != expected or sdf_np1.shape != expected:
        raise ValueError("SDF fields must match the state batch and grid")
    if (
        dx <= 0.0 or not 0.0 <= vorticity_fraction <= 1.0
        or vorticity_amplitude_fraction < 0.0
        or not 0.0 <= coarsen_amplitude_fraction <= vorticity_amplitude_fraction
        or buffer_cells < 0
    ):
        raise ValueError("invalid spacing, vorticity fraction, or buffer")
    swept_geometry = torch.minimum(sdf_n.abs(), sdf_np1.abs()) <= geometry_band_cells * dx
    omega = vorticity_magnitude(state, dx)[:, None]
    indicator = omega
    if previous_state is not None:
        if previous_state.shape != state.shape:
            raise ValueError("previous state must match current state")
        previous_omega = vorticity_magnitude(previous_state, dx)[:, None]
        indicator = torch.maximum(omega, (2.0 * omega - previous_omega).abs())
    if vorticity_fraction == 0.0:
        vortical = torch.zeros_like(swept_geometry)
    else:
        quantile = torch.quantile(
            indicator.flatten(2), 1.0 - vorticity_fraction, dim=2,
            keepdim=True,
        )[..., None]
        resolved_amplitude = torch.quantile(
            indicator.flatten(2), .99, dim=2, keepdim=True,
        )[..., None]
        threshold = torch.maximum(
            quantile, vorticity_amplitude_fraction * resolved_amplitude
        )
        vortical = indicator > threshold
    retained = torch.zeros_like(swept_geometry)
    if previous_refinement_mask is not None:
        if previous_refinement_mask.shape != swept_geometry.shape:
            raise ValueError("previous refinement mask must be B1HW")
        resolved_amplitude = torch.quantile(
            indicator.flatten(2), .99, dim=2, keepdim=True,
        )[..., None]
        retain_threshold = coarsen_amplitude_fraction * resolved_amplitude
        retained = previous_refinement_mask.bool() & (indicator > retain_threshold)
    selected = swept_geometry | vortical | retained
    if buffer_cells:
        selected = F.max_pool2d(
            selected.float(), 2 * buffer_cells + 1, stride=1,
            padding=buffer_cells,
        ) > 0.0
    return patch_aligned_mask(selected, patch_cells)


@dataclass
class AdaptiveGridController:
    """Stateful refine/coarsen policy with hysteresis across physical steps."""

    dx: float
    patch_cells: int = 16
    geometry_band_cells: float = 12.0
    vorticity_fraction: float = .15
    refine_amplitude_fraction: float = .10
    coarsen_amplitude_fraction: float = .05
    buffer_cells: int = 2
    previous_mask: Tensor | None = None

    def reset(self) -> None:
        self.previous_mask = None

    def select(
        self,
        q_nm1: Tensor,
        q_n: Tensor,
        sdf_n: Tensor,
        sdf_np1: Tensor,
    ) -> Tensor:
        mask = adaptive_refinement_mask(
            q_n, sdf_n, sdf_np1, dx=self.dx, previous_state=q_nm1,
            previous_refinement_mask=self.previous_mask,
            patch_cells=self.patch_cells,
            geometry_band_cells=self.geometry_band_cells,
            vorticity_fraction=self.vorticity_fraction,
            vorticity_amplitude_fraction=self.refine_amplitude_fraction,
            coarsen_amplitude_fraction=self.coarsen_amplitude_fraction,
            buffer_cells=self.buffer_cells,
        )
        self.previous_mask = mask.detach()
        return mask


def remap_hierarchy_state(
    coarse: Tensor,
    fine: Tensor,
    old_refinement_mask: Tensor,
    new_refinement_mask: Tensor,
    *,
    factor: int = 2,
) -> tuple[Tensor, Tensor]:
    """Conservatively remap one state when the adaptive hierarchy changes."""
    averaged_coarse, consistent_fine = synchronize_levels(
        coarse, fine, old_refinement_mask, factor=factor, fine_trust=1.0,
    )
    prolonged = constant_prolongation(averaged_coarse, factor)
    persistent = old_refinement_mask.bool() & new_refinement_mask.bool()
    persistent_fine = constant_prolongation(persistent.to(fine.dtype), factor) > 0.0
    remapped_fine = torch.where(persistent_fine, consistent_fine, prolonged)
    return averaged_coarse, remapped_fine


class PairedCartesianLevelDataset(Dataset[dict[str, Tensor]]):
    """Lazy synchronized transitions from two adjacent dense Cartesian levels."""

    def __init__(
        self,
        coarse_path: Path,
        fine_path: Path,
        scope: Literal["cycle1", "cycle2", "all"] = "all",
        history_steps: int = 1,
    ) -> None:
        if history_steps not in (1, 2):
            raise ValueError("history_steps must be one or two")
        self.coarse_path = coarse_path.resolve()
        self.fine_path = fine_path.resolve()
        with h5py.File(self.coarse_path, "r") as coarse, h5py.File(
            self.fine_path, "r"
        ) as fine:
            for handle, path in ((coarse, self.coarse_path), (fine, self.fine_path)):
                if int(handle.attrs.get("complete", 0)) != 1:
                    raise RuntimeError(f"incomplete Cartesian dataset: {path}")
            self.frames = int(coarse["state"].shape[0])
            if self.frames != int(fine["state"].shape[0]):
                raise ValueError("paired levels have different frame counts")
            self.coarse_level = int(coarse.attrs["level"])
            self.fine_level = int(fine.attrs["level"])
            self.coarse_dx = float(coarse.attrs["dx"])
            self.fine_dx = float(fine.attrs["dx"])
            self.window = tuple(float(x) for x in coarse.attrs["window"])
            self.boundary = int(coarse.attrs["cycle_boundary_frame"])
            fine_boundary = int(fine.attrs["cycle_boundary_frame"])
            coarse_time = np.asarray(coarse["time"])
            fine_time = np.asarray(fine["time"])
            coarse_stop = usable_transition_stop(
                coarse_time,
                microstep_aligned=bool(coarse.attrs.get("microstep_aligned", 0)),
            )
            fine_stop = usable_transition_stop(
                fine_time,
                microstep_aligned=bool(fine.attrs.get("microstep_aligned", 0)),
            )
            fine_window = tuple(float(x) for x in fine.attrs["window"])
            self.has_native_refinement_mask = "native_refinement_mask" in fine
            self.has_native_level = (
                "native_level" in coarse and "native_level" in fine
            )
            self.has_gradp = "gradp" in coarse and "gradp" in fine
            self.physical_parameters = json.loads(
                coarse.attrs["source_manifest_json"]
            )["physical_parameters"]
        if self.fine_level != self.coarse_level + 1:
            raise ValueError("paired files must contain adjacent Cartesian levels")
        if not np.isclose(self.coarse_dx, 2.0 * self.fine_dx):
            raise ValueError("paired level spacing must differ by factor two")
        if self.boundary != fine_boundary or self.window != fine_window:
            raise ValueError("paired levels have inconsistent cycle/window metadata")
        np.testing.assert_allclose(coarse_time, fine_time, rtol=0.0, atol=1e-10)
        if coarse_stop != fine_stop:
            raise ValueError("paired levels disagree on usable transition count")
        self.usable_transition_stop = coarse_stop
        self.terminal_stop_time_step_dropped = coarse_stop < self.frames - 1
        if scope == "cycle1":
            self.frame_start = 0
            self.frame_stop = min(self.boundary, self.usable_transition_stop)
        elif scope == "cycle2":
            self.frame_start = self.boundary
            self.frame_stop = self.usable_transition_stop
        elif scope == "all":
            self.frame_start = 0
            self.frame_stop = self.usable_transition_stop
        else:
            raise ValueError(scope)
        self.transitions = list(range(self.frame_start, self.frame_stop))
        if history_steps == 2:
            self.transitions = [frame for frame in self.transitions if frame > 0]
            if not self.transitions:
                raise ValueError("two-step history leaves no complete transitions")
        self._coarse: h5py.File | None = None
        self._fine: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.transitions)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_coarse"] = None
        state["_fine"] = None
        return state

    def _files(self) -> tuple[h5py.File, h5py.File]:
        if self._coarse is None:
            self._coarse = h5py.File(self.coarse_path, "r")
            self._fine = h5py.File(self.fine_path, "r")
        assert self._fine is not None
        return self._coarse, self._fine

    def close(self) -> None:
        for handle in (self._coarse, self._fine):
            if handle is not None:
                handle.close()
        self._coarse = None
        self._fine = None

    @staticmethod
    def _level_sample(handle: h5py.File, frame: int) -> dict[str, Tensor]:
        previous = max(frame - 1, 0)
        result = {
            "q_nm1": torch.from_numpy(np.asarray(handle["state"][previous], dtype=np.float32)),
            "q_n": torch.from_numpy(np.asarray(handle["state"][frame], dtype=np.float32)),
            "q_np1": torch.from_numpy(np.asarray(handle["state"][frame + 1], dtype=np.float32)),
            "sdf_nm1": torch.from_numpy(np.asarray(handle["sdf"][previous], dtype=np.float32)),
            "sdf_n": torch.from_numpy(np.asarray(handle["sdf"][frame], dtype=np.float32)),
            "sdf_np1": torch.from_numpy(np.asarray(handle["sdf"][frame + 1], dtype=np.float32)),
        }
        if "gradp" in handle:
            result.update({
                "gradp_nm1": torch.from_numpy(np.asarray(
                    handle["gradp"][previous], dtype=np.float32,
                )),
                "gradp_n": torch.from_numpy(np.asarray(
                    handle["gradp"][frame], dtype=np.float32,
                )),
                "gradp_np1": torch.from_numpy(np.asarray(
                    handle["gradp"][frame + 1], dtype=np.float32,
                )),
            })
        if "native_level" in handle:
            result.update({
                "native_level_nm1": torch.from_numpy(np.asarray(
                    handle["native_level"][previous], dtype=np.float32,
                )),
                "native_level_n": torch.from_numpy(np.asarray(
                    handle["native_level"][frame], dtype=np.float32,
                )),
                "native_level_np1": torch.from_numpy(np.asarray(
                    handle["native_level"][frame + 1], dtype=np.float32,
                )),
            })
        return result

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        frame = self.transitions[item]
        coarse, fine = self._files()
        result: dict[str, Tensor] = {"frame": torch.tensor(frame, dtype=torch.int64)}
        for prefix, values in (
            ("coarse", self._level_sample(coarse, frame)),
            ("fine", self._level_sample(fine, frame)),
        ):
            result.update({f"{prefix}_{key}": value for key, value in values.items()})
        result["motion_n"] = torch.from_numpy(
            np.asarray(coarse["motion"][frame], dtype=np.float32)
        )
        previous = max(frame - 1, 0)
        result["motion_nm1"] = torch.from_numpy(
            np.asarray(coarse["motion"][previous], dtype=np.float32)
        )
        result["motion_np1"] = torch.from_numpy(
            np.asarray(coarse["motion"][frame + 1], dtype=np.float32)
        )
        result["time_n"] = torch.tensor(float(coarse["time"][frame]), dtype=torch.float32)
        result["time_nm1"] = torch.tensor(
            float(coarse["time"][previous]), dtype=torch.float32
        )
        result["time_np1"] = torch.tensor(float(coarse["time"][frame + 1]), dtype=torch.float32)
        if self.has_native_refinement_mask:
            result["native_refinement_mask"] = torch.from_numpy(
                np.asarray(fine["native_refinement_mask"][frame], dtype=np.bool_)
            )
            result["native_refinement_mask_np1"] = torch.from_numpy(
                np.asarray(fine["native_refinement_mask"][frame + 1], dtype=np.bool_)
            )
        return result


class PairedTemporalStrideDataset(Dataset[dict[str, Tensor]]):
    """Synchronized two-level transitions between Cartesian time anchors.

    A sample centered at frame ``n`` represents
    ``q_(n-stride), q_n -> q_(n+stride)``.  The center range is restricted so
    neither history nor target crosses the parent dataset's cycle split.
    """

    def __init__(
        self, parent: PairedCartesianLevelDataset, stride: int,
    ) -> None:
        if stride <= 0:
            raise ValueError("temporal stride must be positive")
        self.parent = parent
        self.stride = int(stride)
        self.centers = list(range(
            parent.frame_start + self.stride,
            parent.frame_stop - self.stride + 1,
        ))
        if not self.centers:
            raise ValueError("temporal stride leaves no complete anchor triplets")

    def __len__(self) -> int:
        return len(self.centers)

    @staticmethod
    def _level_sample(
        handle: h5py.File, history: int, center: int, endpoint: int,
    ) -> dict[str, Tensor]:
        return {
            "q_nm1": torch.from_numpy(
                np.asarray(handle["state"][history], dtype=np.float32)
            ),
            "q_n": torch.from_numpy(
                np.asarray(handle["state"][center], dtype=np.float32)
            ),
            "q_np1": torch.from_numpy(
                np.asarray(handle["state"][endpoint], dtype=np.float32)
            ),
            "sdf_nm1": torch.from_numpy(
                np.asarray(handle["sdf"][history], dtype=np.float32)
            ),
            "sdf_n": torch.from_numpy(
                np.asarray(handle["sdf"][center], dtype=np.float32)
            ),
            "sdf_np1": torch.from_numpy(
                np.asarray(handle["sdf"][endpoint], dtype=np.float32)
            ),
        }

    def sample_at_frame(self, center: int) -> dict[str, Tensor]:
        if center not in self.centers:
            raise IndexError(
                f"frame {center} is not a complete stride-{self.stride} center"
            )
        history = center - self.stride
        endpoint = center + self.stride
        coarse, fine = self.parent._files()
        result: dict[str, Tensor] = {
            "frame": torch.tensor(center, dtype=torch.int64),
            "history_frame": torch.tensor(history, dtype=torch.int64),
            "endpoint_frame": torch.tensor(endpoint, dtype=torch.int64),
        }
        for prefix, values in (
            ("coarse", self._level_sample(coarse, history, center, endpoint)),
            ("fine", self._level_sample(fine, history, center, endpoint)),
        ):
            result.update({f"{prefix}_{key}": value for key, value in values.items()})
        for name, frame in (("nm1", history), ("n", center), ("np1", endpoint)):
            result[f"motion_{name}"] = torch.from_numpy(
                np.asarray(coarse["motion"][frame], dtype=np.float32)
            )
            result[f"time_{name}"] = torch.tensor(
                float(coarse["time"][frame]), dtype=torch.float32,
            )
        if self.parent.has_native_refinement_mask:
            result["native_refinement_mask"] = torch.from_numpy(
                np.asarray(fine["native_refinement_mask"][center], dtype=np.bool_)
            )
            result["native_refinement_mask_np1"] = torch.from_numpy(
                np.asarray(fine["native_refinement_mask"][endpoint], dtype=np.bool_)
            )
        return result

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        return self.sample_at_frame(self.centers[item])


class GuardedCartesianHierarchyDataset(Dataset[dict[str, Tensor]]):
    """Synchronized nested L4 guard and paired L5/L6 observation levels."""

    def __init__(
        self,
        guard_path: Path,
        coarse_path: Path,
        fine_path: Path,
        scope: Literal["cycle1", "cycle2", "all"] = "all",
    ) -> None:
        self.guard_path = guard_path.resolve()
        self.inner = PairedCartesianLevelDataset(
            coarse_path, fine_path, scope,
        )
        with h5py.File(self.guard_path, "r") as guard, h5py.File(
            self.inner.coarse_path, "r",
        ) as coarse:
            if int(guard.attrs.get("complete", 0)) != 1:
                raise RuntimeError(f"incomplete guard dataset: {self.guard_path}")
            self.guard_level = int(guard.attrs["level"])
            self.guard_dx = float(guard.attrs["dx"])
            self.guard_window = tuple(float(x) for x in guard.attrs["window"])
            self.guard_shape = tuple(int(x) for x in guard["state"].shape[-2:])
            self.guard_core_size = int(guard.attrs["core_size"])
            self.inner_coarse_shape = tuple(
                int(x) for x in coarse["state"].shape[-2:]
            )
            guard_frames = int(guard["state"].shape[0])
            guard_boundary = int(guard.attrs["cycle_boundary_frame"])
            guard_time = np.asarray(guard["time"])
            guard_motion = np.asarray(guard["motion"])
            coarse_motion = np.asarray(coarse["motion"])
        if self.guard_level + 1 != self.inner.coarse_level:
            raise ValueError("guard must be one level below the inner coarse grid")
        if not np.isclose(self.guard_dx, 2.0 * self.inner.coarse_dx):
            raise ValueError("guard and inner coarse spacing must differ by factor two")
        if guard_frames != self.inner.frames or guard_boundary != self.inner.boundary:
            raise ValueError("guard and inner hierarchy frame metadata differ")
        with h5py.File(self.inner.coarse_path, "r") as coarse:
            np.testing.assert_allclose(
                guard_time, np.asarray(coarse["time"]), rtol=0.0, atol=1.0e-10,
            )
        np.testing.assert_allclose(
            guard_motion, coarse_motion, rtol=0.0, atol=1.0e-12,
        )
        self.inner_box = nested_level_box(
            self.guard_window, self.inner.window,
            coarse_dx=self.guard_dx, coarse_shape=self.guard_shape,
            fine_shape=self.inner_coarse_shape,
        )
        self.transitions = self.inner.transitions
        self._guard: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.inner)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_guard"] = None
        return state

    def _file(self) -> h5py.File:
        if self._guard is None:
            self._guard = h5py.File(self.guard_path, "r")
        return self._guard

    def close(self) -> None:
        self.inner.close()
        if self._guard is not None:
            self._guard.close()
            self._guard = None

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        result = self.inner[item]
        frame = int(result["frame"])
        guard = self._file()
        result.update({
            f"guard_{key}": value
            for key, value in PairedCartesianLevelDataset._level_sample(
                guard, frame,
            ).items()
        })
        if "native_refinement_mask" in guard:
            result["guard_native_refinement_mask"] = torch.from_numpy(
                np.asarray(
                    guard["native_refinement_mask"][frame], dtype=np.bool_,
                )
            )
        return result


def level_batch(
    sample: dict[str, Tensor], level: Literal["guard", "coarse", "fine"],
) -> dict[str, Tensor]:
    """Convert one paired sample into the batch keys expected by transition builders."""
    result = {
        key: sample[f"{level}_{key}"]
        for key in ("q_nm1", "q_n", "q_np1", "sdf_nm1", "sdf_n", "sdf_np1")
    }
    result.update({
        key: sample[key]
        for key in (
            "motion_nm1", "motion_n", "motion_np1",
            "time_nm1", "time_n", "time_np1", "frame",
        )
    })
    for key in (
        "gradp_nm1", "gradp_n", "gradp_np1",
        "native_level_nm1", "native_level_n", "native_level_np1",
    ):
        prefixed = f"{level}_{key}"
        if prefixed in sample:
            result[key] = sample[prefixed]
    return result


def compute_balanced_paired_statistics(
    dataset: PairedCartesianLevelDataset,
    max_transitions: int,
    seed: int,
) -> FieldStatistics:
    """Compute shared moments with equal weight per transition and level."""
    if max_transitions <= 0:
        raise ValueError("max_transitions must be positive")
    count = min(max_transitions, len(dataset))
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(dataset), size=count, replace=False))
    state_sum = np.zeros(3, dtype=np.float64)
    state_square = np.zeros(3, dtype=np.float64)
    delta_sum = np.zeros(3, dtype=np.float64)
    delta_square = np.zeros(3, dtype=np.float64)
    records = 0
    coarse, fine = dataset._files()
    for item in selected:
        frame = dataset.transitions[int(item)]
        for handle in (coarse, fine):
            q_n = np.asarray(handle["state"][frame], dtype=np.float64)
            delta = np.asarray(handle["state"][frame + 1], dtype=np.float64) - q_n
            state_sum += q_n.mean(axis=(1, 2))
            state_square += np.square(q_n).mean(axis=(1, 2))
            delta_sum += delta.mean(axis=(1, 2))
            delta_square += np.square(delta).mean(axis=(1, 2))
            records += 1
    state_mean = state_sum / records
    delta_mean = delta_sum / records
    state_variance = np.maximum(state_square / records - state_mean**2, 1e-10)
    delta_variance = np.maximum(delta_square / records - delta_mean**2, 1e-10)
    return FieldStatistics(
        state_mean=tuple(state_mean.tolist()),
        state_std=tuple(np.sqrt(state_variance).tolist()),
        delta_mean=tuple(delta_mean.tolist()),
        delta_std=tuple(np.sqrt(delta_variance).tolist()),
        sampled_records=records,
    )
