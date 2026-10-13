"""Johansen--Colella embedded-boundary Poisson benchmark definitions.

The analytic definitions in this module are independent of any discretization.
AMReX supplies the authoritative cut-cell geometry and numerical reference
fields; the rasterizer here is intended for tests, previews, and neural input
prototyping only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from neural_eos.mature_amr_cases import CutCellGeometry, sample_cutcell_geometry

Array = np.ndarray


@dataclass(frozen=True)
class FlowerGeometry:
    radius: float = 0.30
    amplitude: float = 0.15
    petals: int = 6
    rotation: float = 0.0
    center_x: float = 0.5
    center_y: float = 0.5

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        if self.amplitude < 0.0 or self.amplitude >= self.radius:
            raise ValueError("amplitude must satisfy 0 <= amplitude < radius")
        if self.petals < 2:
            raise ValueError("petals must be at least two")
        extent = self.radius + self.amplitude
        if not (extent < self.center_x < 1.0 - extent and extent < self.center_y < 1.0 - extent):
            raise ValueError("flower must lie strictly inside the unit square")

    def polar(self, y: Array, x: Array) -> tuple[Array, Array]:
        dx = np.asarray(x) - self.center_x
        dy = np.asarray(y) - self.center_y
        return np.hypot(dx, dy), np.arctan2(dy, dx)

    def boundary_radius(self, theta: Array) -> Array:
        return self.radius + self.amplitude * np.cos(self.petals * (theta - self.rotation))

    def level_set(self, y: Array, x: Array) -> Array:
        radius, theta = self.polar(y, x)
        return radius - self.boundary_radius(theta)

    def inside(self, y: Array, x: Array) -> Array:
        return self.level_set(y, x) <= 0.0


@dataclass(frozen=True)
class HarmonicField:
    radial_power: int = 4
    angular_mode: int = 3
    radial_scale: float = 1.0
    phase: float = 0.0
    cosine_amplitude: float = 1.0
    sine_amplitude: float = 0.0

    def __post_init__(self) -> None:
        if self.angular_mode < 0:
            raise ValueError("angular_mode must be nonnegative")
        if self.radial_power < max(2, self.angular_mode):
            raise ValueError("radial_power must be at least max(2, angular_mode)")
        if self.radial_scale <= 0.0:
            raise ValueError("radial_scale must be positive")
        if abs(self.cosine_amplitude) + abs(self.sine_amplitude) == 0.0:
            raise ValueError("at least one field amplitude must be nonzero")

    def angular_value(self, theta: Array) -> Array:
        angle = self.angular_mode * (theta - self.phase)
        return self.cosine_amplitude * np.cos(angle) + self.sine_amplitude * np.sin(angle)

    def angular_derivative(self, theta: Array) -> Array:
        angle = self.angular_mode * (theta - self.phase)
        return self.angular_mode * (
            -self.cosine_amplitude * np.sin(angle) + self.sine_amplitude * np.cos(angle)
        )

    def value(self, y: Array, x: Array, geometry: FlowerGeometry) -> Array:
        radius, theta = geometry.polar(y, x)
        return (radius / self.radial_scale) ** self.radial_power * self.angular_value(theta)

    def laplacian(self, y: Array, x: Array, geometry: FlowerGeometry) -> Array:
        radius, theta = geometry.polar(y, x)
        coefficient = self.radial_power**2 - self.angular_mode**2
        return (
            coefficient
            * radius ** (self.radial_power - 2)
            / self.radial_scale**self.radial_power
            * self.angular_value(theta)
        )

    def operator_rhs(self, y: Array, x: Array, geometry: FlowerGeometry) -> Array:
        """Right-hand side for AMReX's positive operator, ``-Laplacian(u)``."""
        return -self.laplacian(y, x, geometry)

    def gradient(self, y: Array, x: Array, geometry: FlowerGeometry) -> tuple[Array, Array]:
        radius, theta = geometry.polar(y, x)
        angular = self.angular_value(theta)
        angular_derivative = self.angular_derivative(theta)
        scale = self.radial_scale**self.radial_power
        radial_term = self.radial_power * radius ** (self.radial_power - 1) * angular / scale
        tangential_term = radius ** (self.radial_power - 1) * angular_derivative / scale
        grad_x = np.cos(theta) * radial_term - np.sin(theta) * tangential_term
        grad_y = np.sin(theta) * radial_term + np.cos(theta) * tangential_term
        return grad_y, grad_x


@dataclass(frozen=True)
class BenchmarkSample:
    geometry: FlowerGeometry
    field: HarmonicField


def canonical_sample() -> BenchmarkSample:
    return BenchmarkSample(FlowerGeometry(), HarmonicField())


def sample_problem(seed: int, split: str = "train") -> BenchmarkSample:
    rng = np.random.default_rng(seed)
    if split not in {"train", "interpolation", "geometry_extrapolation"}:
        raise ValueError(f"unknown split: {split}")

    if split == "geometry_extrapolation":
        radius = rng.choice((rng.uniform(0.20, 0.24), rng.uniform(0.31, 0.325)))
        relative_amplitude = rng.uniform(0.42, 0.48)
        petals = int(rng.choice((7, 8)))
        offset_limit = min(0.12, 0.5 - radius * (1.0 + relative_amplitude) - 0.01)
    else:
        radius = rng.uniform(0.24, 0.32)
        relative_amplitude = rng.uniform(0.12, 0.40)
        petals = int(rng.choice((3, 4, 5, 6)))
        offset_limit = min(0.08, 0.5 - radius * (1.0 + relative_amplitude) - 0.01)

    offset_limit = max(0.0, offset_limit)
    geometry = FlowerGeometry(
        radius=float(radius),
        amplitude=float(radius * relative_amplitude),
        petals=petals,
        rotation=float(rng.uniform(0.0, 2.0 * math.pi)),
        center_x=float(0.5 + rng.uniform(-offset_limit, offset_limit)),
        center_y=float(0.5 + rng.uniform(-offset_limit, offset_limit)),
    )

    angular_mode = int(rng.integers(0, 5))
    radial_power = angular_mode + 2 * int(rng.integers(1, 4))
    field_scale = float(rng.uniform(0.5, 1.5))
    if angular_mode == 0:
        cosine_amplitude = field_scale * float(rng.choice((-1.0, 1.0)))
        sine_amplitude = 0.0
    else:
        mixing_angle = float(rng.uniform(0.0, 2.0 * math.pi))
        cosine_amplitude = field_scale * math.cos(mixing_angle)
        sine_amplitude = field_scale * math.sin(mixing_angle)
    field = HarmonicField(
        radial_power=radial_power,
        angular_mode=angular_mode,
        radial_scale=geometry.radius,
        phase=float(rng.uniform(0.0, 2.0 * math.pi)),
        cosine_amplitude=cosine_amplitude,
        sine_amplitude=sine_amplitude,
    )
    return BenchmarkSample(geometry, field)


def cell_centers(height: int, width: int) -> tuple[Array, Array]:
    y = (np.arange(height, dtype=np.float64) + 0.5)[:, None] / height
    x = (np.arange(width, dtype=np.float64) + 0.5)[None, :] / width
    return np.broadcast_to(y, (height, width)), np.broadcast_to(x, (height, width))


def rasterize_sample(
    sample: BenchmarkSample,
    height: int,
    width: int,
    subsamples: int = 8,
) -> dict[str, Array | CutCellGeometry]:
    geometry = sample_cutcell_geometry(height, width, sample.geometry.inside, samples=subsamples)
    y, x = cell_centers(height, width)
    exact = sample.field.value(y, x, sample.geometry)
    source = sample.field.laplacian(y, x, sample.geometry)
    operator_rhs = -source
    exact[~geometry.active] = 0.0
    source[~geometry.active] = 0.0
    operator_rhs[~geometry.active] = 0.0
    cut = (geometry.vol_frac > 0.0) & (geometry.vol_frac < 1.0)
    return {
        "geometry": geometry,
        "source": source,
        "operator_rhs": operator_rhs,
        "exact": exact,
        "cut_cells": cut,
    }
