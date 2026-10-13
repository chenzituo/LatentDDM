"""Trajectory-clean splits and randomized batches for the cyclic airfoil corpus."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from codiffuse_Schwarz_Airfoil.coupled_trajectory import CyclicWindowDataset


SPLIT_VERSION = 2
PARAMETERS = (
    "reynolds", "thickness", "pivot_fraction", "cycle_amplitude_deg",
    "cycle_period", "t0",
)
RESERVED_VALIDATION = (
    "cyclic_002_level5_window.h5",
    "cyclic_017_level5_window.h5",
    "cyclic_028_level5_window.h5",
)


def discover_trajectories(root: Path) -> list[Path]:
    paths = sorted(root.resolve().glob("cyclic_*_level5_window.h5"))
    if not paths:
        raise FileNotFoundError(f"no synchronized trajectories under {root}")
    return paths


def _parameters(path: Path) -> dict[str, float]:
    with h5py.File(path, "r") as handle:
        manifest = json.loads(handle.attrs["source_manifest_json"])
    return {
        name: float(manifest["physical_parameters"][name])
        for name in PARAMETERS
    }


def make_trajectory_split(paths: list[Path]) -> dict[str, list[str] | int]:
    """Hold parameter extremes out for test and central cases for validation."""
    if len(paths) < 12:
        raise ValueError("trajectory-clean split requires at least 12 cases")
    values = np.asarray([
        [_parameters(path)[name] for name in PARAMETERS] for path in paths
    ])
    lower, upper = values.min(0), values.max(0)
    normalized = (values - lower) / np.maximum(upper - lower, 1.0e-12)
    extreme_score = np.abs(normalized - 0.5).max(1)
    count = max(2, len(paths) // 8)
    names = [path.name for path in paths]
    canonical = names.index("cyclic_000_level5_window.h5")
    eligible_test = np.asarray([index for index in range(len(paths)) if index != canonical])
    test_indices = eligible_test[
        np.argsort(extreme_score[eligible_test], kind="stable")[-count:]
    ]
    remaining = np.setdiff1d(np.arange(len(paths)), test_indices)
    reserved = [names.index(name) for name in RESERVED_VALIDATION]
    if any(index in test_indices for index in reserved):
        raise RuntimeError("reserved validation trajectory selected as parameter-extreme test")
    validation_indices = np.asarray(reserved)
    training_indices = np.setdiff1d(remaining, validation_indices)
    return {
        "format_version": SPLIT_VERSION,
        "train": [names[index] for index in training_indices],
        "validation": [names[index] for index in validation_indices],
        "test": [names[index] for index in test_indices],
    }


def write_or_validate_split(
    root: Path, output: Path, trajectory_count: int | None = None,
) -> dict[str, list[str] | int]:
    paths = discover_trajectories(root)
    if trajectory_count is not None:
        if trajectory_count <= 0:
            raise ValueError("trajectory_count must be positive")
        paths = [
            path for path in paths
            if int(path.name.removeprefix("cyclic_").split("_", 1)[0])
            < trajectory_count
        ]
        if len(paths) != trajectory_count:
            raise RuntimeError(
                f"expected {trajectory_count} trajectories, found {len(paths)}"
            )
    proposed = make_trajectory_split(paths)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != proposed:
            raise RuntimeError(f"existing split does not match deterministic split: {output}")
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proposed, indent=2) + "\n", encoding="utf-8")
    return proposed


@dataclass
class TrajectoryCorpus:
    root: Path
    names: list[str]
    scope: str = "all"
    history_steps: int = 2

    def __post_init__(self) -> None:
        self.datasets = [
            CyclicWindowDataset(
                self.root / name, self.scope, history_steps=self.history_steps
            )
            for name in self.names
        ]
        self.concatenated = _TrajectoryDataset(self.datasets)

    def loader(
        self, batch_size: int, shuffle: bool, seed: int,
        num_workers: int = 0,
    ) -> DataLoader:
        sampler = HomogeneousTrajectoryBatchSampler(
            [len(dataset) for dataset in self.datasets], batch_size, shuffle, seed
        )
        return DataLoader(
            self.concatenated, batch_sampler=sampler, num_workers=num_workers,
            pin_memory=True, persistent_workers=num_workers > 0,
        )

    def random_batch(
        self, batch_size: int, rng: random.Random,
    ) -> tuple[CyclicWindowDataset, dict[str, torch.Tensor]]:
        """Draw one trajectory then independent frames, avoiding mixed metadata."""
        dataset = self.datasets[rng.randrange(len(self.datasets))]
        indices = [rng.randrange(len(dataset)) for _ in range(batch_size)]
        samples = [dataset[index] for index in indices]
        return dataset, {
            key: torch.stack([sample[key] for sample in samples])
            for key in samples[0]
        }

    @property
    def transition_count(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def close(self) -> None:
        for dataset in self.datasets:
            dataset.close()


class _TrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, datasets: list[CyclicWindowDataset]) -> None:
        self.datasets = datasets
        self.offsets = np.cumsum([0] + [len(dataset) for dataset in datasets])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        trajectory = int(np.searchsorted(self.offsets, index, side="right") - 1)
        item = index - int(self.offsets[trajectory])
        sample = self.datasets[trajectory][item]
        sample["trajectory_index"] = torch.tensor(trajectory, dtype=torch.int64)
        return sample


class HomogeneousTrajectoryBatchSampler(Sampler[list[int]]):
    """Shuffle every transition once while keeping each batch trajectory-pure."""

    def __init__(
        self, lengths: list[int], batch_size: int, shuffle: bool, seed: int,
    ) -> None:
        if batch_size <= 0 or not lengths or min(lengths) <= 0:
            raise ValueError("positive batch size and trajectory lengths are required")
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.offsets = np.cumsum([0] + lengths).tolist()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return sum(math.ceil(length / self.batch_size) for length in self.lengths)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches = []
        for trajectory, length in enumerate(self.lengths):
            order = (
                torch.randperm(length, generator=generator).tolist()
                if self.shuffle else list(range(length))
            )
            offset = self.offsets[trajectory]
            batches.extend(
                [offset + item for item in order[start:start + self.batch_size]]
                for start in range(0, length, self.batch_size)
            )
        if self.shuffle:
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in permutation]
        yield from batches
