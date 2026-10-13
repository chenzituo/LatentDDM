"""Fixed-hierarchy data and Schwarz-coupled Flow operations for one trajectory."""

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

from codiffuse_Schwarz_Airfoil.dataset import (
    FieldStatistics,
    denormalize_delta,
    usable_transition_stop,
)
from codiffuse_Schwarz_Airfoil.cartesian_cover import CartesianCover
from codiffuse_Schwarz_Airfoil.motion_control import (
    CyclicAirfoilControl,
    signed_distance_proxy_torch,
)


Tensor = torch.Tensor
AssemblyMode = Literal["independent", "ras", "average", "weighted"]


class CyclicWindowDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self, path: Path, split: Literal["cycle1", "cycle2", "all"],
        history_steps: int = 1,
    ) -> None:
        if history_steps not in (1, 2):
            raise ValueError("history_steps must be one or two")
        self.history_steps = history_steps
        self.path = path.resolve()
        with h5py.File(self.path, "r") as handle:
            if int(handle.attrs.get("complete", 0)) != 1:
                raise RuntimeError(f"incomplete synchronized dataset: {self.path}")
            self.frames = int(handle["state"].shape[0])
            self.boundary = int(handle.attrs["cycle_boundary_frame"])
            self.transition_stop = usable_transition_stop(
                np.asarray(handle["time"]),
                microstep_aligned=bool(handle.attrs.get("microstep_aligned", 0)),
            )
            self.terminal_stop_time_step_dropped = (
                self.transition_stop < self.frames - 1
            )
            self.dx = float(handle.attrs["dx"])
            self.level = int(handle.attrs["level"])
            self.core_size = int(handle.attrs["core_size"])
            self.window = tuple(float(value) for value in handle.attrs["window"])
            self.has_gradp = "gradp" in handle
            self.has_native_level = "native_level" in handle
            manifest = json.loads(handle.attrs["source_manifest_json"])
            pitch_start_time = float(
                handle.attrs.get(
                    "pitch_start_time", manifest.get("pitch_start_time", 2.0),
                )
            )
        self.physical_parameters = manifest["physical_parameters"]
        pitch_run = next(
            (
                run for run in manifest["runs"]
                if run.get("stage") == "pitch"
            ),
            next(
                run for run in manifest["runs"]
                if "airfoil_ib.penalty_time" in run.get("overrides", {})
            ),
        )
        self.penalty_time = float(
            pitch_run["overrides"]["airfoil_ib.penalty_time"]
        )
        self.rotation_sign = float(
            self.physical_parameters.get("rotation_sign", 1.0)
        )
        self.pitch_start_time = pitch_start_time
        if split == "cycle1":
            self.transitions = list(range(0, min(
                self.boundary, self.transition_stop,
            )))
        elif split == "cycle2":
            self.transitions = list(range(
                self.boundary, self.transition_stop,
            ))
        elif split == "all":
            self.transitions = list(range(self.transition_stop))
        else:
            raise ValueError(split)
        if history_steps == 2:
            # Each phase capture starts at frame zero and does not retain the
            # preceding native plotfile.  Do not fabricate q_(n-1)=q_n for the
            # optional two-history ablation.
            self.transitions = [frame for frame in self.transitions if frame > 0]
            if not self.transitions:
                raise ValueError("two-step history leaves no complete transitions")
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.transitions)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        frame = self.transitions[item]
        previous = max(frame - 1, 0)
        handle = self._file()
        result = {
            "q_nm1": torch.from_numpy(np.asarray(handle["state"][previous], dtype=np.float32)),
            "q_n": torch.from_numpy(np.asarray(handle["state"][frame], dtype=np.float32)),
            "q_np1": torch.from_numpy(np.asarray(handle["state"][frame + 1], dtype=np.float32)),
            "sdf_nm1": torch.from_numpy(np.asarray(handle["sdf"][previous], dtype=np.float32)),
            "sdf_n": torch.from_numpy(np.asarray(handle["sdf"][frame], dtype=np.float32)),
            "sdf_np1": torch.from_numpy(np.asarray(handle["sdf"][frame + 1], dtype=np.float32)),
            "motion_nm1": torch.from_numpy(np.asarray(handle["motion"][previous], dtype=np.float32)),
            "motion_n": torch.from_numpy(np.asarray(handle["motion"][frame], dtype=np.float32)),
            "motion_np1": torch.from_numpy(np.asarray(handle["motion"][frame + 1], dtype=np.float32)),
            "time_nm1": torch.tensor(float(handle["time"][previous]), dtype=torch.float32),
            "time_n": torch.tensor(float(handle["time"][frame]), dtype=torch.float32),
            "time_np1": torch.tensor(float(handle["time"][frame + 1]), dtype=torch.float32),
            "frame": torch.tensor(frame, dtype=torch.int64),
        }
        if self.has_gradp:
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
        if self.has_native_level:
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


def tile_tensor(field: Tensor, core: int, halo: int) -> Tensor:
    """Return B*tiles halo crops in row-major tile order."""
    batch, channels, height, width = field.shape
    if height % core or width % core:
        raise ValueError("global field must be divisible by the core size")
    padded = F.pad(field, (halo, halo, halo, halo))
    patches = padded.unfold(2, core + 2 * halo, core).unfold(3, core + 2 * halo, core)
    ny, nx = patches.shape[2:4]
    return patches.permute(0, 2, 3, 1, 4, 5).reshape(
        batch * ny * nx, channels, core + 2 * halo, core + 2 * halo
    )


def assemble_ras(patches: Tensor, batch: int, shape: tuple[int, int], core: int, halo: int) -> Tensor:
    height, width = shape
    ny, nx = height // core, width // core
    channels = patches.shape[1]
    cores = patches[:, :, halo : halo + core, halo : halo + core]
    cores = cores.reshape(batch, ny, nx, channels, core, core)
    return cores.permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, height, width)


def assemble_average(
    patches: Tensor, batch: int, shape: tuple[int, int], core: int, halo: int
) -> Tensor:
    height, width = shape
    size = core + 2 * halo
    channels = patches.shape[1]
    ny, nx = height // core, width // core
    columns = patches.reshape(batch, ny * nx, channels * size * size).transpose(1, 2)
    accumulated = F.fold(
        columns, output_size=(height + 2 * halo, width + 2 * halo),
        kernel_size=size, stride=core,
    )
    weights = F.fold(
        torch.ones(batch, size * size, ny * nx, device=patches.device, dtype=patches.dtype),
        output_size=(height + 2 * halo, width + 2 * halo), kernel_size=size, stride=core,
    )
    averaged = accumulated / weights.clamp_min(1.0)
    return averaged[:, :, halo : halo + height, halo : halo + width]


def patch_confidence_window(
    core: int, halo: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    if halo == 0:
        return torch.ones(1, 1, core, core, device=device, dtype=dtype)
    ramp = 0.5 - 0.5 * torch.cos(
        torch.pi * torch.arange(1, halo + 1, device=device, dtype=dtype) / (halo + 1)
    )
    one_dimensional = torch.cat((ramp, torch.ones(core, device=device, dtype=dtype), ramp.flip(0)))
    return (one_dimensional[:, None] * one_dimensional[None, :])[None, None]


def assemble_weighted(
    patches: Tensor, batch: int, shape: tuple[int, int], core: int, halo: int
) -> Tensor:
    """Confidence-tapered least-squares consensus for direct Cartesian crops."""
    height, width = shape
    size = core + 2 * halo
    channels = patches.shape[1]
    ny, nx = height // core, width // core
    confidence = patch_confidence_window(core, halo, patches.device, patches.dtype)
    weighted = patches * confidence
    columns = weighted.reshape(batch, ny * nx, channels * size * size).transpose(1, 2)
    accumulated = F.fold(
        columns, output_size=(height + 2 * halo, width + 2 * halo),
        kernel_size=size, stride=core,
    )
    weight_columns = confidence.expand(batch * ny * nx, 1, size, size).reshape(
        batch, ny * nx, size * size
    ).transpose(1, 2)
    weights = F.fold(
        weight_columns, output_size=(height + 2 * halo, width + 2 * halo),
        kernel_size=size, stride=core,
    )
    averaged = accumulated / weights.clamp_min(torch.finfo(patches.dtype).eps)
    return averaged[:, :, halo : halo + height, halo : halo + width]


def zero_patch_halos(patches: Tensor, core: int, halo: int) -> Tensor:
    result = torch.zeros_like(patches)
    result[:, :, halo : halo + core, halo : halo + core] = patches[
        :, :, halo : halo + core, halo : halo + core
    ]
    return result


def make_physical_and_scalars(
    batch: dict[str, Tensor], statistics: FieldStatistics, dataset: CyclicWindowDataset,
    core: int, halo: int, device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    q_n = batch["q_n"].to(device)
    sdf_n = batch["sdf_n"].to(device)
    sdf_np1 = batch["sdf_np1"].to(device)
    motion_n = batch["motion_n"].to(device)
    motion_np1 = batch["motion_np1"].to(device)
    batch_size, _, height, width = q_n.shape
    mean = torch.tensor(statistics.state_mean, device=device)[None, :, None, None]
    std = torch.tensor(statistics.state_std, device=device)[None, :, None, None]
    q_normalized = (q_n - mean) / std
    sdf_scale = dataset.dx * core
    sdf = torch.cat((sdf_n, sdf_np1), dim=1).clamp(-4 * sdf_scale, 4 * sdf_scale)
    sdf = sdf / (4 * sdf_scale)
    x0, _, y0, _ = dataset.window
    x = x0 + (torch.arange(width, device=device) + 0.5) * dataset.dx
    y = y0 + (torch.arange(height, device=device) + 0.5) * dataset.dx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx[None].expand(batch_size, -1, -1)
    yy = yy[None].expand(batch_size, -1, -1)
    band_n = torch.exp(-sdf_n[:, 0].to(device).abs() / (4 * dataset.dx))
    band_np1 = torch.exp(-sdf_np1[:, 0].to(device).abs() / (4 * dataset.dx))
    omega_n = motion_n[:, 1, None, None]
    omega_np1 = motion_np1[:, 1, None, None]
    wall = torch.stack(
        (-omega_n * yy * band_n, omega_n * xx * band_n,
         -omega_np1 * yy * band_np1, omega_np1 * xx * band_np1),
        dim=1,
    )
    valid = torch.ones(batch_size, 1, height, width, device=device)
    physical_global = torch.cat(
        (q_normalized, sdf, wall, xx[:, None] / 10.0, yy[:, None] / 10.0, valid), dim=1
    )
    physical_tiles = tile_tensor(physical_global, core, halo)

    ny, nx = height // core, width // core
    centers_x = x0 + (torch.arange(nx, device=device) + 0.5) * core * dataset.dx
    centers_y = y0 + (torch.arange(ny, device=device) + 0.5) * core * dataset.dx
    center_y, center_x = torch.meshgrid(centers_y, centers_x, indexing="ij")
    centers = torch.stack((center_x.flatten() / 10.0, center_y.flatten() / 10.0), dim=1)
    params = dataset.physical_parameters
    dt = (batch["time_np1"] - batch["time_n"]).to(device)
    viscosity = 1.0 / float(params["reynolds"])
    common = torch.stack(
        (
            torch.full_like(dt, np.log(float(params["reynolds"]) / 10_000.0)),
            torch.full_like(dt, (float(params["thickness"]) - 0.15) / 0.05),
            torch.full_like(dt, (float(params["pivot_fraction"]) - 0.25) / 0.10),
            motion_n[:, 0] / torch.pi,
            motion_n[:, 1],
            motion_np1[:, 0] / torch.pi,
            motion_np1[:, 1],
            motion_n[:, 2] / 3.0,
            motion_np1[:, 2] / 3.0,
            dt / 0.01,
            torch.full_like(dt, float(getattr(self.dataset, "level_fraction", 1.0))),
            dt / dataset.dx,
            100.0 * viscosity * dt / (dataset.dx**2),
        ), dim=1,
    )
    scalar = torch.cat(
        (
            common[:, None, :].expand(-1, ny * nx, -1),
            centers[None].expand(batch_size, -1, -1),
        ), dim=2,
    ).reshape(batch_size * ny * nx, -1)
    target = batch["q_np1"].to(device) - q_n
    delta_mean = torch.tensor(statistics.delta_mean, device=device)[None, :, None, None]
    delta_std = torch.tensor(statistics.delta_std, device=device)[None, :, None, None]
    target_normalized = (target - delta_mean) / delta_std
    return physical_tiles, scalar, target_normalized


def coupled_velocity(
    model: torch.nn.Module, state: Tensor, physical_tiles: Tensor, scalar: Tensor,
    flow_time: Tensor, core: int, halo: int, mode: AssemblyMode,
) -> tuple[Tensor, Tensor]:
    batch, _, height, width = state.shape
    state_tiles = tile_tensor(state, core, halo)
    if mode == "independent":
        state_tiles = zero_patch_halos(state_tiles, core, halo)
    tile_time = flow_time[:, None].expand(-1, (height // core) * (width // core)).reshape(-1)
    tile_velocity = model(state_tiles, physical_tiles, scalar, tile_time)
    if mode == "average":
        global_velocity = assemble_average(tile_velocity, batch, (height, width), core, halo)
    elif mode == "weighted":
        global_velocity = assemble_weighted(tile_velocity, batch, (height, width), core, halo)
    else:
        global_velocity = assemble_ras(tile_velocity, batch, (height, width), core, halo)
    return global_velocity, tile_velocity


@torch.no_grad()
def sample_coupled(
    model: torch.nn.Module, physical_tiles: Tensor, scalar: Tensor,
    shape: tuple[int, int], batch: int, steps: int, seed: int,
    core: int, halo: int, mode: AssemblyMode,
    initial_noise: Tensor | None = None,
) -> Tensor:
    if initial_noise is None:
        generator = torch.Generator(device=physical_tiles.device).manual_seed(seed)
        state = torch.randn(batch, 3, *shape, device=physical_tiles.device, generator=generator)
    else:
        expected = (batch, 3, *shape)
        if tuple(initial_noise.shape) != expected:
            raise ValueError(f"initial noise shape {tuple(initial_noise.shape)} != {expected}")
        state = initial_noise.clone()
    for step in range(steps):
        flow_time = torch.full(
            (batch,), (step + 0.5) / steps, device=state.device, dtype=state.dtype
        )
        velocity, _ = coupled_velocity(
            model, state, physical_tiles, scalar, flow_time, core, halo, mode
        )
        state = state + velocity / steps
    return state


def predicted_next_state(q_n: Tensor, normalized_delta: Tensor, statistics: FieldStatistics) -> Tensor:
    return q_n + denormalize_delta(normalized_delta, statistics)


@dataclass
class ControlledTransition:
    """One physical transition conditioned by an analytic prescribed motion."""

    batch: dict[str, Tensor]
    statistics: FieldStatistics
    dataset: CyclicWindowDataset
    cover: CartesianCover
    device: torch.device

    def __post_init__(self) -> None:
        self.q_n = self.batch["q_n"].to(self.device)
        self.q_nm1 = self.batch.get("q_nm1", self.batch["q_n"]).to(self.device)
        self.q_np1 = self.batch["q_np1"].to(self.device)
        self.time_n = self.batch["time_n"].to(self.device)
        self.time_np1 = self.batch["time_np1"].to(self.device)
        self.control = CyclicAirfoilControl.from_parameters(
            self.dataset.physical_parameters, self.dataset.pitch_start_time
        )
        batch_size, _, height, width = self.q_n.shape
        if (height, width) != (self.cover.height, self.cover.width):
            raise ValueError("transition field and Cartesian cover shapes differ")
        x0, _, y0, _ = self.dataset.window
        x = x0 + (torch.arange(width, device=self.device) + 0.5) * self.dataset.dx
        y = y0 + (torch.arange(height, device=self.device) + 0.5) * self.dataset.dx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.xx = xx[None].expand(batch_size, -1, -1)
        self.yy = yy[None].expand(batch_size, -1, -1)
        state_mean = torch.tensor(
            self.statistics.state_mean, device=self.device, dtype=self.q_n.dtype
        )[None, :, None, None]
        state_std = torch.tensor(
            self.statistics.state_std, device=self.device, dtype=self.q_n.dtype
        )[None, :, None, None]
        self.q_normalized = (self.q_n - state_mean) / state_std
        self.gradp_increment_normalized: Tensor | None = None
        self.gradp_n: Tensor | None = None
        self.gradp_np1: Tensor | None = None
        if "gradp_n" in self.batch:
            gradp_n = self.batch["gradp_n"].to(self.device)
            gradp_np1 = self.batch["gradp_np1"].to(self.device)
            if gradp_n.shape != self.q_n[:, :2].shape:
                raise ValueError("lagged pressure gradient must have shape B,2,H,W")
            if gradp_np1.shape != gradp_n.shape:
                raise ValueError("endpoint pressure gradient must have shape B,2,H,W")
            self.gradp_n = gradp_n
            self.gradp_np1 = gradp_np1
            dt = (self.time_np1 - self.time_n)[:, None, None, None]
            # incflo adds -dt*grad(p) to its velocity RHS.  Expressing that
            # known local contribution in velocity-standard-deviation units
            # avoids introducing a corpus-specific gradient normalization.
            self.gradp_increment_normalized = (
                -dt * gradp_n / state_std[:, :2]
            )
        self.native_level_n: Tensor | None = None
        self.native_level_np1: Tensor | None = None
        if "native_level_n" in self.batch:
            native_n = self.batch["native_level_n"].to(self.device)
            native_np1 = self.batch["native_level_np1"].to(self.device)
            expected = self.q_n[:, :1].shape
            if native_n.shape != expected or native_np1.shape != expected:
                raise ValueError("native AMR level must have shape B,1,H,W")
            if bool(torch.any(native_n < 0.0)) or bool(torch.any(
                native_np1 < 0.0
            )):
                raise ValueError("native AMR levels must be nonnegative")
            self.native_level_n = native_n
            self.native_level_np1 = native_np1
        delta_mean = torch.tensor(
            self.statistics.delta_mean, device=self.device, dtype=self.q_n.dtype
        )[None, :, None, None]
        delta_std = torch.tensor(
            self.statistics.delta_std, device=self.device, dtype=self.q_n.dtype
        )[None, :, None, None]
        self.target = (self.q_np1 - self.q_n - delta_mean) / delta_std
        self.history_normalized = (self.q_n - self.q_nm1) / delta_std
        self.right_ghost_q_normalized: Tensor | None = None
        self.right_ghost_history_normalized: Tensor | None = None
        if "right_ghost_q_n" in self.batch:
            ghost_n = self.batch["right_ghost_q_n"].to(self.device)
            ghost_nm1 = self.batch.get("right_ghost_q_nm1", ghost_n).to(self.device)
            expected = (batch_size, 3, height)
            if (
                ghost_n.ndim != 4 or ghost_n.shape[:3] != expected
                or ghost_nm1.shape != ghost_n.shape
            ):
                raise ValueError("right ghost histories must have shape B,3,H,G")
            self.right_ghost_q_normalized = (ghost_n - state_mean) / state_std
            self.right_ghost_history_normalized = (ghost_n - ghost_nm1) / delta_std

    @property
    def batch_size(self) -> int:
        return self.q_n.shape[0]

    def motion(self, fraction: Tensor) -> tuple[Tensor, Tensor]:
        current = self.control.between(self.time_n, self.time_np1, fraction)
        endpoint = self.control.evaluate(self.time_np1)
        return current, endpoint

    def geometry(self, fraction: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        current, endpoint = self.motion(fraction)
        sdf_current = signed_distance_proxy_torch(
            self.xx, self.yy, current[:, 0], self.control.thickness,
            self.control.pivot_fraction,
        )
        sdf_endpoint = signed_distance_proxy_torch(
            self.xx, self.yy, endpoint[:, 0], self.control.thickness,
            self.control.pivot_fraction,
        )
        return current, endpoint, sdf_current, sdf_endpoint

    def _conditions(
        self, current_fraction: Tensor, phase_fraction: Tensor,
    ) -> tuple[Tensor, Tensor]:
        current, endpoint, sdf_current, sdf_endpoint = self.geometry(
            current_fraction
        )
        sdf_scale = self.dataset.dx * self.cover.core
        sdf = torch.stack((sdf_current, sdf_endpoint), dim=1)
        sdf = sdf.clamp(-4 * sdf_scale, 4 * sdf_scale) / (4 * sdf_scale)
        band_current = torch.exp(-sdf_current.abs() / (4 * self.dataset.dx))
        band_endpoint = torch.exp(-sdf_endpoint.abs() / (4 * self.dataset.dx))
        omega_current = current[:, 1, None, None]
        omega_endpoint = endpoint[:, 1, None, None]
        wall = torch.stack(
            (
                -omega_current * self.yy * band_current,
                omega_current * self.xx * band_current,
                -omega_endpoint * self.yy * band_endpoint,
                omega_endpoint * self.xx * band_endpoint,
            ), dim=1,
        )
        valid = torch.ones(
            self.batch_size, 1, self.cover.height, self.cover.width,
            device=self.device, dtype=self.q_n.dtype,
        )
        fields = [
            self.q_normalized, sdf, wall,
            self.xx[:, None] / 10.0, self.yy[:, None] / 10.0,
        ]
        if getattr(self.dataset, "history_steps", 1) == 2:
            fields.append(self.history_normalized)
        physical_global = torch.cat((*fields, valid), dim=1)
        padding_mode = getattr(self.dataset, "boundary_padding_mode", "constant")
        if self.right_ghost_q_normalized is None:
            physical = self.cover.tile(
                physical_global, padding_mode=padding_mode,
            )
            # Replace the padded validity channel with exact cover validity.
            physical[:, -1:] = self.cover.valid_tiles(
                self.batch_size, self.device, physical.dtype
            )
        else:
            assert self.right_ghost_history_normalized is not None
            ghost_width = self.right_ghost_q_normalized.shape[-1]
            ghost_x = self.dataset.window[1] + (
                torch.arange(
                    ghost_width, device=self.device, dtype=self.q_n.dtype,
                ) + .5
            ) * self.dataset.dx
            ghost_y = self.dataset.window[2] + (
                torch.arange(
                    self.cover.height, device=self.device, dtype=self.q_n.dtype,
                ) + .5
            ) * self.dataset.dx
            ghost_yy, ghost_xx = torch.meshgrid(ghost_y, ghost_x, indexing="ij")
            ghost_fields = [
                self.right_ghost_q_normalized,
                torch.ones(
                    self.batch_size, 2, self.cover.height, ghost_width,
                    device=self.device, dtype=self.q_n.dtype,
                ),
                torch.zeros(
                    self.batch_size, 4, self.cover.height, ghost_width,
                    device=self.device, dtype=self.q_n.dtype,
                ),
                ghost_xx[None, None].expand(self.batch_size, -1, -1, -1) / 10.0,
                ghost_yy[None, None].expand(self.batch_size, -1, -1, -1) / 10.0,
            ]
            if getattr(self.dataset, "history_steps", 1) == 2:
                ghost_fields.append(self.right_ghost_history_normalized)
            ghost_fields.append(torch.ones(
                self.batch_size, 1, self.cover.height, ghost_width,
                device=self.device, dtype=self.q_n.dtype,
            ))
            physical = self.cover.tile_with_right_ghost(
                physical_global, torch.cat(ghost_fields, dim=1),
                padding_mode=padding_mode,
            )

        params = self.dataset.physical_parameters
        dt = self.time_np1 - self.time_n
        viscosity = 1.0 / float(params["reynolds"])
        cycle_period = float(params.get("cycle_period", 1.0))
        absolute_phase = (
            self.time_n + phase_fraction * dt - self.dataset.pitch_start_time
        ) / cycle_period
        common = torch.stack(
            (
                absolute_phase,
                torch.full_like(dt, (float(params["thickness"]) - 0.15) / 0.05),
                torch.full_like(dt, (float(params["pivot_fraction"]) - 0.25) / 0.10),
                current[:, 0] / torch.pi,
                current[:, 1],
                endpoint[:, 0] / torch.pi,
                endpoint[:, 1],
                current[:, 2] / 3.0,
                endpoint[:, 2] / 3.0,
                dt / 0.01,
                torch.ones_like(dt),
                dt / self.dataset.dx,
                100.0 * viscosity * dt / (self.dataset.dx**2),
            ), dim=1,
        )
        centers = self.cover.centers(
            self.dataset.window[0], self.dataset.window[2], self.dataset.dx, self.device
        ) / 10.0
        scalar = torch.cat(
            (
                common[:, None].expand(-1, self.cover.tile_count, -1),
                centers[None].expand(self.batch_size, -1, -1),
            ), dim=2,
        ).reshape(self.batch_size * self.cover.tile_count, -1)
        return physical, scalar

    def conditions(self, fraction: Tensor) -> tuple[Tensor, Tensor]:
        """Conditions along the Flow-Matching interpolation path."""
        return self._conditions(fraction, fraction)

    def endpoint_conditions(self) -> tuple[Tensor, Tensor]:
        """Source/destination conditions for a clean one-step endpoint map.

        The endpoint projector predicts q_(n+1)-q_n directly. It therefore
        needs the pose and wall velocity at both physical endpoints. Calling
        ``conditions(1)`` collapses those two channel groups to the same
        destination pose and discards the explicit geometric displacement.
        """
        source = torch.zeros_like(self.time_n)
        endpoint = torch.ones_like(self.time_n)
        return self._conditions(source, endpoint)


def controlled_velocity(
    model: torch.nn.Module,
    state: Tensor,
    transition: ControlledTransition,
    flow_time: Tensor,
    mode: AssemblyMode,
) -> tuple[Tensor, Tensor, Tensor]:
    physical, scalar = transition.conditions(flow_time)
    state_tiles = transition.cover.tile(state)
    if mode == "independent":
        state_tiles = zero_patch_halos(state_tiles, transition.cover.core, transition.cover.halo)
    tile_time = flow_time[:, None].expand(-1, transition.cover.tile_count).reshape(-1)
    tile_velocity = model(state_tiles, physical, scalar, tile_time)
    if mode == "average":
        velocity = transition.cover.assemble_average(tile_velocity, transition.batch_size)
    elif mode == "weighted":
        velocity = transition.cover.assemble_weighted(tile_velocity, transition.batch_size)
    else:
        velocity = transition.cover.assemble_ras(tile_velocity, transition.batch_size)
    return velocity, tile_velocity, physical


def endpoint_trace_velocity(
    model: torch.nn.Module,
    state: Tensor,
    transition: ControlledTransition,
    flow_time: Tensor,
    iterations: int,
    relaxation: float,
) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
    """Re-solve local patches after clean-endpoint overlap projection."""
    if iterations <= 0:
        raise ValueError("Schwarz iterations must be positive")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("Schwarz relaxation must lie in (0, 1]")
    physical, scalar = transition.conditions(flow_time)
    local_state = transition.cover.tile(state)
    valid = transition.cover.valid_tiles(
        transition.batch_size, state.device, state.dtype
    )
    h, c = transition.cover.halo, transition.cover.core
    owned = torch.zeros_like(valid)
    owned[:, :, h : h + c, h : h + c] = 1.0
    trace_mask = valid * (1.0 - owned)
    tile_time = flow_time[:, None].expand(
        -1, transition.cover.tile_count
    ).reshape(-1)
    residuals: list[Tensor] = []
    tile_velocity: Tensor | None = None
    for iteration in range(iterations):
        tile_velocity = model(local_state, physical, scalar, tile_time)
        remaining = (1.0 - tile_time)[:, None, None, None]
        local_endpoint = local_state + remaining * tile_velocity
        endpoint_consensus = transition.cover.assemble_weighted(
            local_endpoint, transition.batch_size
        )
        mismatch = transition.cover.tile(endpoint_consensus) - local_endpoint
        trace_cells = trace_mask.sum().clamp_min(1.0) * state.shape[1]
        residuals.append(
            ((mismatch.square() * trace_mask).sum() / trace_cells).sqrt()
        )
        if iteration + 1 < iterations:
            local_state = local_state + relaxation * mismatch * trace_mask
    assert tile_velocity is not None
    velocity = transition.cover.assemble_ras(
        tile_velocity, transition.batch_size
    )
    return velocity, tile_velocity, physical, residuals


def pnp_endpoint_trace_sample(
    model: torch.nn.Module,
    transition: ControlledTransition,
    noise_sequence: Tensor,
    relaxation: float,
) -> tuple[Tensor, list[Tensor]]:
    """PnP-Flow clean-endpoint consensus with path reprojection."""
    if noise_sequence.ndim != 5 or tuple(noise_sequence.shape[1:]) != (
        transition.batch_size, 3, transition.cover.height, transition.cover.width
    ):
        raise ValueError("PnP noise sequence has the wrong shape")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("PnP relaxation must lie in (0, 1]")
    steps = noise_sequence.shape[0]
    valid = transition.cover.valid_tiles(
        transition.batch_size, transition.device, transition.q_n.dtype
    )
    local_endpoint = torch.zeros(
        transition.batch_size * transition.cover.tile_count, 3,
        transition.cover.core + 2 * transition.cover.halo,
        transition.cover.core + 2 * transition.cover.halo,
        device=transition.device, dtype=transition.q_n.dtype,
    )
    endpoint_physical = endpoint_scalar = None
    motion_physical = motion_scalar = None
    if bool(getattr(model, "pnp_endpoint_eos_each_step", False)):
        if not hasattr(model, "correct_endpoint_fields"):
            raise ValueError("per-step PnP EoS requires endpoint field correction")
        endpoint_time = torch.ones(
            transition.batch_size, device=transition.device,
            dtype=transition.q_n.dtype,
        )
        endpoint_physical, endpoint_scalar = transition.conditions(endpoint_time)
        motion_physical, motion_scalar = transition.endpoint_conditions()
    residuals: list[Tensor] = []
    for step in range(steps):
        flow_time = torch.full(
            (transition.batch_size,), step / steps,
            device=transition.device, dtype=transition.q_n.dtype,
        )
        consensus = transition.cover.assemble_weighted(
            local_endpoint, transition.batch_size
        )
        consensus_tiles = transition.cover.tile(consensus)
        mismatch = (consensus_tiles - local_endpoint) * valid
        cells = valid.sum().clamp_min(1.0) * local_endpoint.shape[1]
        residuals.append((mismatch.square().sum() / cells).sqrt())
        projected = local_endpoint + relaxation * mismatch
        noise_tiles = transition.cover.tile(noise_sequence[step])
        tile_time = flow_time[:, None].expand(
            -1, transition.cover.tile_count
        ).reshape(-1)
        t = tile_time[:, None, None, None]
        noisy = ((1.0 - t) * noise_tiles + t * projected) * valid
        physical, scalar = transition.conditions(flow_time)
        velocity = model(noisy, physical, scalar, tile_time)
        local_endpoint = (noisy + (1.0 - t) * velocity) * valid
        if endpoint_physical is not None:
            assert endpoint_scalar is not None
            local_endpoint, _ = model.correct_endpoint_fields(
                local_endpoint, endpoint_physical, endpoint_scalar,
                transition.cover.tile_count,
                tile_exchange=lambda hidden: transition.cover.tile(
                    transition.cover.assemble_weighted(
                        hidden, transition.batch_size,
                    )
                ),
                tile_grid_shape=transition.cover.grid_shape,
                motion_physical=motion_physical,
                motion_scalar=motion_scalar,
            )
            local_endpoint = local_endpoint * valid
    return transition.cover.assemble_ras(
        local_endpoint, transition.batch_size
    ), residuals


@torch.no_grad()
def sample_controlled(
    model: torch.nn.Module,
    transition: ControlledTransition,
    steps: int,
    seed: int,
    mode: AssemblyMode,
    initial_noise: Tensor | None = None,
) -> Tensor:
    shape = (transition.batch_size, 3, transition.cover.height, transition.cover.width)
    if initial_noise is None:
        generator = torch.Generator(device=transition.device).manual_seed(seed)
        state = torch.randn(*shape, device=transition.device, generator=generator)
    else:
        if tuple(initial_noise.shape) != shape:
            raise ValueError("controlled sample initial-noise shape mismatch")
        state = initial_noise.clone()
    for step in range(steps):
        flow_time = torch.full(
            (transition.batch_size,), (step + 0.5) / steps,
            device=transition.device, dtype=state.dtype,
        )
        velocity, _, _ = controlled_velocity(model, state, transition, flow_time, mode)
        state = state + velocity / steps
    return state


def integrate_controlled(
    model: torch.nn.Module,
    transition: ControlledTransition,
    steps: int,
    mode: AssemblyMode,
    initial_noise: Tensor | None = None,
) -> Tensor:
    """Differentiable Flow integration for deterministic endpoint training."""
    shape = (transition.batch_size, 3, transition.cover.height, transition.cover.width)
    state = (
        torch.zeros(*shape, device=transition.device, dtype=transition.q_n.dtype)
        if initial_noise is None else initial_noise
    )
    if tuple(state.shape) != shape:
        raise ValueError("controlled integration initial-noise shape mismatch")
    for step in range(steps):
        flow_time = torch.full(
            (transition.batch_size,), (step + 0.5) / steps,
            device=transition.device, dtype=state.dtype,
        )
        velocity, _, _ = controlled_velocity(model, state, transition, flow_time, mode)
        state = state + velocity / steps
    return state
