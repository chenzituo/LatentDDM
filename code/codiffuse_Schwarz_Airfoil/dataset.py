"""HDF5 block-transition data for local airfoil Flow-Matching pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


DOMAIN_LO = -10.0
DOMAIN_WIDTH = 20.0
BASE_CELLS = 64
STATE_CHANNELS = ("u", "v", "p")
SCALAR_CONDITIONS = (
    "log_reynolds_ratio",
    "thickness_offset",
    "pivot_offset",
    "alpha_n_over_pi",
    "omega_n",
    "alpha_np1_over_pi",
    "omega_np1",
    "alpha_accel_n_scaled",
    "alpha_accel_np1_scaled",
    "dt_ratio",
    "level_fraction",
    "dt_over_dx",
    "viscous_dt_over_dx2_scaled",
    "block_center_x",
    "block_center_y",
)


def usable_transition_stop(
    times: np.ndarray, *, microstep_aligned: bool,
) -> int:
    """Return the exclusive transition stop after removing a stop-time sliver.

    An incflo capture terminated by ``stop_time`` may contain one final step
    shortened solely to land on that requested time.  Such a step is a valid
    numerical update, but it is not sampled from the adaptive-CFL cadence that
    defines the intended solver-step dataset.  For explicitly microstep-aligned
    captures, discard only a terminal step shorter than half the median of all
    preceding steps.  Interior adaptive steps are never filtered.
    """
    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("times must be a finite one-dimensional frame array")
    increments = np.diff(values)
    if np.any(increments <= 0.0):
        raise ValueError("frame times must be strictly increasing")
    stop = values.size - 1
    if microstep_aligned and increments.size >= 4:
        reference = float(np.median(increments[:-1]))
        if increments[-1] < 0.5 * reference:
            stop -= 1
    return int(stop)


@dataclass(frozen=True)
class FieldStatistics:
    state_mean: tuple[float, float, float]
    state_std: tuple[float, float, float]
    delta_mean: tuple[float, float, float]
    delta_std: tuple[float, float, float]
    sampled_records: int

    def to_json(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json(cls, values: dict[str, object]) -> "FieldStatistics":
        return cls(
            state_mean=tuple(float(value) for value in values["state_mean"]),
            state_std=tuple(float(value) for value in values["state_std"]),
            delta_mean=tuple(float(value) for value in values["delta_mean"]),
            delta_std=tuple(float(value) for value in values["delta_std"]),
            sampled_records=int(values["sampled_records"]),
        )


@dataclass(frozen=True)
class ShardMetadata:
    path: Path
    records: int
    core_size: int
    halo: int
    max_level: int
    sample_dt: float
    reynolds: float
    thickness: float
    pivot_fraction: float
    omega0: float
    t0: float
    rotation_sign: float
    trajectory_id: str
    precursor_id: str


def read_metadata(path: Path) -> ShardMetadata:
    with h5py.File(path, "r") as handle:
        precursor = json.loads(handle.attrs["precursor_json"])
        trajectory = json.loads(handle.attrs["trajectory_json"])
        return ShardMetadata(
            path=path.resolve(),
            records=int(handle["fields"].shape[0]),
            core_size=int(handle.attrs["core_size"]),
            halo=int(handle.attrs["halo"]),
            max_level=int(handle.attrs["max_level"]),
            sample_dt=float(handle.attrs["sample_dt"]),
            reynolds=float(precursor["reynolds"]),
            thickness=float(precursor["thickness"]),
            pivot_fraction=float(precursor["pivot_fraction"]),
            omega0=float(trajectory["omega0"]),
            t0=float(trajectory["t0"]),
            rotation_sign=float(handle.attrs.get("rotation_sign", 1.0)),
            trajectory_id=str(trajectory["id"]),
            precursor_id=str(trajectory["precursor_id"]),
        )


def discover_shards(root: Path) -> list[Path]:
    paths = sorted(root.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"no HDF5 shards under {root}")
    return paths


def split_partial_shards(paths: Sequence[Path]) -> tuple[list[Path], list[Path]]:
    """Keep complete positive/negative precursor pairs in one split."""
    by_precursor: dict[str, list[Path]] = {}
    for path in paths:
        metadata = read_metadata(path)
        by_precursor.setdefault(metadata.precursor_id, []).append(path)
    precursor_ids = sorted(by_precursor)
    if len(precursor_ids) < 2:
        raise ValueError("at least two precursor groups are required for a clean split")
    validation_id = precursor_ids[-1]
    train = sorted(path for key in precursor_ids[:-1] for path in by_precursor[key])
    validation = sorted(by_precursor[validation_id])
    return train, validation


def _owned_mask(index: np.ndarray, valid: np.ndarray, core_size: int, halo: int) -> np.ndarray:
    nx, ny = int(index[4]), int(index[5])
    ownership = np.zeros_like(valid, dtype=np.float32)
    ownership[:, halo : halo + min(ny, core_size), halo : halo + min(nx, core_size)] = 1.0
    return ownership * valid.astype(np.float32)


def compute_statistics(paths: Sequence[Path], max_records: int, seed: int) -> FieldStatistics:
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    metadata = [read_metadata(path) for path in paths]
    total = sum(item.records for item in metadata)
    count = min(max_records, total)
    rng = np.random.default_rng(seed)
    flat_indices = np.sort(rng.choice(total, size=count, replace=False))
    offsets = np.cumsum([0] + [item.records for item in metadata])
    state_sum = np.zeros(3, dtype=np.float64)
    state_square = np.zeros(3, dtype=np.float64)
    delta_sum = np.zeros(3, dtype=np.float64)
    delta_square = np.zeros(3, dtype=np.float64)
    cell_count = 0.0

    for shard_index, item in enumerate(metadata):
        selected = flat_indices[(flat_indices >= offsets[shard_index])
                                & (flat_indices < offsets[shard_index + 1])]
        if not len(selected):
            continue
        local = selected - offsets[shard_index]
        with h5py.File(item.path, "r") as handle:
            # h5py requires monotonically increasing fancy indices.
            fields = np.asarray(handle["fields"][local], dtype=np.float64)
            valid = np.asarray(handle["valid"][local], dtype=np.float64)
            index = np.asarray(handle["index"][local])
        for record in range(len(local)):
            weight = _owned_mask(index[record], valid[record], item.core_size, item.halo)
            q_n = fields[record, :3]
            delta = fields[record, 3:6] - q_n
            cells = float(weight.sum())
            state_sum += (q_n * weight).sum(axis=(1, 2))
            state_square += (np.square(q_n) * weight).sum(axis=(1, 2))
            delta_sum += (delta * weight).sum(axis=(1, 2))
            delta_square += (np.square(delta) * weight).sum(axis=(1, 2))
            cell_count += cells
    if cell_count <= 0:
        raise RuntimeError("statistics found no owned valid cells")
    state_mean = state_sum / cell_count
    delta_mean = delta_sum / cell_count
    state_var = np.maximum(state_square / cell_count - state_mean**2, 1.0e-10)
    delta_var = np.maximum(delta_square / cell_count - delta_mean**2, 1.0e-10)
    return FieldStatistics(
        state_mean=tuple(state_mean.tolist()),
        state_std=tuple(np.sqrt(state_var).tolist()),
        delta_mean=tuple(delta_mean.tolist()),
        delta_std=tuple(np.sqrt(delta_var).tolist()),
        sampled_records=count,
    )


def _pitch_acceleration(relative_time: float, omega0: float, t0: float) -> float:
    return 4.6 * omega0 / t0 * math.exp(-4.6 * max(relative_time, 0.0) / t0)


class AirfoilBlockDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        paths: Sequence[Path],
        statistics: FieldStatistics,
        max_records: int | None = None,
    ) -> None:
        if not paths:
            raise ValueError("dataset needs at least one shard")
        self.shards = [read_metadata(path) for path in paths]
        self.statistics = statistics
        self.records: list[tuple[int, int]] = [
            (shard, record)
            for shard, metadata in enumerate(self.shards)
            for record in range(metadata.records)
        ]
        if max_records is not None:
            self.records = self.records[:max_records]
        self._handles: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def _handle(self, shard: int) -> h5py.File:
        if shard not in self._handles:
            self._handles[shard] = h5py.File(self.shards[shard].path, "r")
        return self._handles[shard]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        shard_index, record = self.records[item]
        metadata = self.shards[shard_index]
        handle = self._handle(shard_index)
        fields = np.asarray(handle["fields"][record], dtype=np.float32)
        valid = np.asarray(handle["valid"][record], dtype=np.float32)
        index = np.asarray(handle["index"][record], dtype=np.int64)
        time = np.asarray(handle["time"][record], dtype=np.float64)
        ownership = _owned_mask(index, valid, metadata.core_size, metadata.halo)
        q_n = fields[:3]
        q_np1 = fields[3:6]
        delta = q_np1 - q_n

        state_mean = np.asarray(self.statistics.state_mean, dtype=np.float32)[:, None, None]
        state_std = np.asarray(self.statistics.state_std, dtype=np.float32)[:, None, None]
        delta_mean = np.asarray(self.statistics.delta_mean, dtype=np.float32)[:, None, None]
        delta_std = np.asarray(self.statistics.delta_std, dtype=np.float32)[:, None, None]
        q_n_normalized = (q_n - state_mean) / state_std * valid
        delta_normalized = (delta - delta_mean) / delta_std * valid

        level, i0, j0 = int(index[1]), int(index[2]), int(index[3])
        dx = DOMAIN_WIDTH / (BASE_CELLS * 2**level)
        size = fields.shape[-1]
        start_x = i0 - metadata.halo
        start_y = j0 - metadata.halo
        x = DOMAIN_LO + (start_x + np.arange(size, dtype=np.float32) + 0.5) * dx
        y = DOMAIN_LO + (start_y + np.arange(size, dtype=np.float32) + 0.5) * dx
        yy, xx = np.meshgrid(y, x, indexing="ij")
        coordinates = np.stack((xx / 10.0, yy / 10.0)).astype(np.float32)

        sdf_scale = max(dx * metadata.core_size, 1.0e-8)
        sdf = np.clip(fields[6:8] / sdf_scale, -4.0, 4.0) / 4.0
        geometry_band_n = np.exp(-np.abs(fields[6]) / max(4.0 * dx, 1.0e-8))
        geometry_band_np1 = np.exp(-np.abs(fields[7]) / max(4.0 * dx, 1.0e-8))
        alpha_n, omega_n, alpha_np1, omega_np1 = map(float, time[2:6])
        geometric_omega_n = metadata.rotation_sign * omega_n
        geometric_omega_np1 = metadata.rotation_sign * omega_np1
        wall = np.stack(
            (
                -geometric_omega_n * yy * geometry_band_n,
                geometric_omega_n * xx * geometry_band_n,
                -geometric_omega_np1 * yy * geometry_band_np1,
                geometric_omega_np1 * xx * geometry_band_np1,
            )
        ).astype(np.float32)
        physical = np.concatenate((q_n_normalized, sdf, wall, coordinates, valid), axis=0)
        physical *= valid

        dt = float(time[1] - time[0])
        relative_n = float(time[0] - 2.0)
        relative_np1 = float(time[1] - 2.0)
        acceleration_n = _pitch_acceleration(relative_n, metadata.omega0, metadata.t0)
        acceleration_np1 = _pitch_acceleration(relative_np1, metadata.omega0, metadata.t0)
        center_x = DOMAIN_LO + (i0 + 0.5 * int(index[4])) * dx
        center_y = DOMAIN_LO + (j0 + 0.5 * int(index[5])) * dx
        viscosity = 1.0 / metadata.reynolds
        scalar = np.asarray(
            (
                math.log(metadata.reynolds / 10_000.0),
                (metadata.thickness - 0.15) / 0.05,
                (metadata.pivot_fraction - 0.25) / 0.10,
                alpha_n / math.pi,
                omega_n,
                alpha_np1 / math.pi,
                omega_np1,
                acceleration_n / 3.0,
                acceleration_np1 / 3.0,
                dt / 0.01,
                level / max(metadata.max_level, 1),
                dt / dx,
                100.0 * viscosity * dt / (dx * dx),
                center_x / 10.0,
                center_y / 10.0,
            ),
            dtype=np.float32,
        )
        body_intersection = np.asarray(
            [float(fields[6].min() <= 0.0 <= fields[6].max()
                   or fields[7].min() <= 0.0 <= fields[7].max())],
            dtype=np.float32,
        )
        return {
            "physical": torch.from_numpy(physical),
            "scalar": torch.from_numpy(scalar),
            "target": torch.from_numpy(delta_normalized),
            "q_n": torch.from_numpy(q_n),
            "q_np1": torch.from_numpy(q_np1),
            "ownership": torch.from_numpy(ownership),
            "valid": torch.from_numpy(valid),
            "level": torch.tensor(level, dtype=torch.int64),
            "direction": torch.tensor(1 if metadata.omega0 >= 0.0 else -1, dtype=torch.int64),
            "body_intersection": torch.from_numpy(body_intersection),
        }


def denormalize_delta(delta: torch.Tensor, statistics: FieldStatistics) -> torch.Tensor:
    mean = torch.tensor(statistics.delta_mean, device=delta.device, dtype=delta.dtype)[None, :, None, None]
    std = torch.tensor(statistics.delta_std, device=delta.device, dtype=delta.dtype)[None, :, None, None]
    return delta * std + mean
