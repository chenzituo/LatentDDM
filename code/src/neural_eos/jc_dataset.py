"""Learning-data projection for the AMReX Johansen--Colella benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neural_eos.johansen_colella import FlowerGeometry, HarmonicField, cell_centers
from neural_eos.mature_amr_cases import sample_cutcell_geometry


INPUT_CHANNELS = (
    "operator_rhs",
    "volume_fraction",
    "boundary_area",
    "dirichlet",
    "aperture_x",
    "aperture_y",
    "boundary_normal_x",
    "boundary_normal_y",
    "x",
    "y",
    "rhs_to_solution_scale",
)


def _blocks(array: np.ndarray, resolution: int) -> np.ndarray:
    height, width = array.shape
    if height != width or height % resolution:
        raise ValueError(f"cannot project shape {array.shape} to {resolution}x{resolution}")
    factor = height // resolution
    return array.reshape(resolution, factor, resolution, factor)


def block_mean(array: np.ndarray, resolution: int) -> np.ndarray:
    return _blocks(array, resolution).mean(axis=(1, 3))


def block_max(array: np.ndarray, resolution: int) -> np.ndarray:
    return _blocks(array, resolution).max(axis=(1, 3))


def weighted_block_mean(array: np.ndarray, weight: np.ndarray, resolution: int) -> np.ndarray:
    numerator = _blocks(array * weight, resolution).sum(axis=(1, 3))
    denominator = _blocks(weight, resolution).sum(axis=(1, 3))
    return numerator / np.maximum(denominator, 1.0e-12)


def load_learning_sample(path: Path, resolution: int = 128) -> dict[str, torch.Tensor | str]:
    with np.load(path) as archived:
        metadata = str(archived["metadata_json"])

    parameters = json.loads(metadata)
    geometry = FlowerGeometry(**parameters["geometry"])
    field = HarmonicField(**parameters["field"])
    cut_geometry = sample_cutcell_geometry(
        resolution, resolution, geometry.inside, samples=8, threshold=1.0e-8
    )
    volume = cut_geometry.vol_frac
    active = cut_geometry.active
    aperture_x = 0.5 * (cut_geometry.area_x[:, :-1] + cut_geometry.area_x[:, 1:])
    aperture_y = 0.5 * (cut_geometry.area_y[:-1, :] + cut_geometry.area_y[1:, :])
    delta_x = cut_geometry.area_x[:, :-1] - cut_geometry.area_x[:, 1:]
    delta_y = cut_geometry.area_y[:-1, :] - cut_geometry.area_y[1:, :]
    boundary_area = np.hypot(delta_x, delta_y)
    normal_x = cut_geometry.normal_x * (boundary_area > 0.0)
    normal_y = cut_geometry.normal_y * (boundary_area > 0.0)
    y, x = cell_centers(resolution, resolution)
    solution = field.value(y, x, geometry)
    rhs = field.operator_rhs(y, x, geometry)
    dirichlet = field.value(y, x, geometry) * (boundary_area > 0.0)
    solution[~active] = 0.0
    rhs[~active] = 0.0
    solution_scale = max(float(np.max(np.abs(solution[active]), initial=0.0)), 1.0e-6)
    rhs_scale = max(float(np.max(np.abs(rhs[active]), initial=0.0)), 1.0e-6)
    solution = solution / solution_scale
    rhs = rhs / rhs_scale
    dirichlet = dirichlet / solution_scale
    scale_ratio = np.log1p(rhs_scale / solution_scale) / 10.0

    coordinates = (np.arange(resolution, dtype=np.float64) + 0.5) / resolution
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    inputs = np.stack(
        (
            rhs,
            volume,
            boundary_area,
            dirichlet,
            aperture_x,
            aperture_y,
            normal_x,
            normal_y,
            2.0 * xx - 1.0,
            2.0 * yy - 1.0,
            np.full_like(volume, scale_ratio),
        ),
        axis=0,
    ).astype(np.float32)
    return {
        "input": torch.from_numpy(inputs),
        "target": torch.from_numpy(solution.astype(np.float32))[None],
        "weight": torch.from_numpy(volume.astype(np.float32))[None],
        "solution_scale": torch.tensor(solution_scale, dtype=torch.float32),
        "metadata_json": metadata,
        "path": str(path),
    }


def load_learning_corpus(root: Path, resolution: int = 128) -> dict[str, dict[str, torch.Tensor | list[str]]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, torch.Tensor | str]]] = {}
    for item in manifest["samples"]:
        grouped.setdefault(item["split"], []).append(load_learning_sample(root / item["path"], resolution))

    result: dict[str, dict[str, torch.Tensor | list[str]]] = {}
    for split, samples in grouped.items():
        result[split] = {
            "input": torch.stack([sample["input"] for sample in samples]),
            "target": torch.stack([sample["target"] for sample in samples]),
            "weight": torch.stack([sample["weight"] for sample in samples]),
            "solution_scale": torch.stack([sample["solution_scale"] for sample in samples]),
            "metadata_json": [str(sample["metadata_json"]) for sample in samples],
            "path": [str(sample["path"]) for sample in samples],
        }
    return result
