"""Darcy data and adapter for the shared Airfoil direct-endpoint core.

The HDF5 contract is deliberately small.  Each split is a group named
``train``, ``validation`` (``val`` is accepted), or ``test`` containing
``mask``, ``logk``, ``source``, ``bc_mask``, ``bc_value``, ``solution``,
``sdf``, and ``normals``.  The v2 contract additionally provides
``dirichlet_face_mask``, ``dirichlet_face_value``, and
``dirichlet_face_role`` in north/south/west/east order. Scalar fields may be
``[N,H,W]`` or ``[N,1,H,W]``; normals and face fields may be channel-first or
channel-last. The v1 cellwise contract remains loadable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from codiffuse_Schwarz_Airfoil.code_transfer.adapter import (
    ConditionBatch,
    PhysicalTransition,
    PhysicsAdapter,
)
from codiffuse_Schwarz_Airfoil.code_transfer.cover import CartesianCover
from codiffuse_Schwarz_Airfoil.code_transfer.model import (
    DirectEndpointOperator,
    DirectEndpointRegressor,
    NeuralEndpointProjector,
)


Tensor = torch.Tensor
REQUIRED_FIELDS = (
    "mask", "logk", "source", "bc_mask", "bc_value", "solution", "sdf", "normals",
)
FACE_DIRICHLET_FIELDS = (
    "dirichlet_face_mask", "dirichlet_face_value", "dirichlet_face_role",
)
OPTIONAL_PHYSICAL_METADATA = (
    "grid_spacing", "physical_extent",
)
MODEL_SPEC = {
    "proposal_width": 128,
    "proposal_depth": 8,
    "projector_width": 96,
    "projector_depth": 6,
    "cell_attention": False,
    "core": 32,
    "halo": 4,
}
CommunicationStage = Literal["none", "proposal", "both"]
COMMUNICATION_STAGES: tuple[CommunicationStage, ...] = (
    "none", "proposal", "both",
)
COMMUNICATION_CONTRACTS: dict[CommunicationStage, str] = {
    "none": "darcy_full_domain_local_pou_v3",
    "proposal": "darcy_full_domain_proposal_communication_v3",
    "both": "darcy_full_domain_both_stage_communication_v3",
}
LEGACY_COMMUNICATION_CONTRACTS: dict[str, CommunicationStage] = {
    # From-scratch endpoint used as the frozen base of the operator-only
    # composer. It has no cross-unity communicator.
    "darcy_operator_free_local_endpoint_v1": "none",
    "darcy_sni_cellwise_boundary_to_solution_experimental_v1": "none",
    "darcy_sni_face_boundary_to_solution_v1": "none",
    "darcy_arbitrary_current_direct_endpoint_v1": "none",
    "darcy_full_domain_local_pou_v2": "none",
    # Historical "relative communication" enabled grid communicators in both
    # the proposal and endpoint projector. Keep that architecture loadable but
    # label it explicitly as the two-stage ablation from now on.
    "darcy_full_domain_relative_communication_v2": "both",
}


def discrete_darcy_residual(
    state: Tensor,
    logk: Tensor,
    source: Tensor,
    mask: Tensor,
    bc_value: Tensor,
) -> Tensor:
    """Causal dimensionless residual of ``-div(k grad u)=source``.

    ``logk`` follows the dataset's documented base-10 convention. Inactive and
    out-of-grid neighbors use the same half-cell Dirichlet-face convention as
    :mod:`codiffuse_Schwarz_Darcy.physics`.
    """
    if state.ndim != 4 or state.shape[1] != 1:
        raise ValueError("Darcy residual expects scalar BCHW fields")
    height, width = state.shape[-2:]
    dx = 2.0 / max(height, width, 1)
    coefficient = torch.pow(10.0, logk.clamp(-6.0, 6.0))
    residual = raw_darcy_residual(
        state, coefficient, source, mask, bc_value, dx=dx,
    )
    # Fixed physical scaling retains convergence amplitude while keeping the
    # input range comparable across permeability fields and resolutions.
    coefficient_scale = (
        (coefficient * mask).flatten(1).sum(1)
        / mask.flatten(1).sum(1).clamp_min(1.0)
    ).clamp_min(1.0e-6)
    scale = coefficient_scale / (dx * dx)
    return (residual / scale[:, None, None, None]).clamp(-10.0, 10.0)


def raw_darcy_residual(
    state: Tensor,
    coefficient: Tensor,
    source: Tensor,
    mask: Tensor,
    bc_value: Tensor,
    *,
    dx: float | Tensor | None = None,
) -> Tensor:
    """Return ``A u - b`` for the exact masked finite-volume convention."""
    height, width = state.shape[-2:]
    if dx is None:
        inverse_h2: float | Tensor = (
            max(height, width, 1) / 2.0
        ) ** 2
    elif isinstance(dx, Tensor):
        spacing = dx.to(device=state.device, dtype=state.dtype).reshape(-1)
        if spacing.numel() not in (1, state.shape[0]) or bool((spacing <= 0).any()):
            raise ValueError("grid spacing must be one positive value per batch")
        if spacing.numel() == 1:
            spacing = spacing.expand(state.shape[0])
        inverse_h2 = spacing.reciprocal().square()[:, None, None]
    else:
        spacing_value = float(dx)
        if spacing_value <= 0.0:
            raise ValueError("grid spacing must be positive")
        inverse_h2 = 1.0 / (spacing_value * spacing_value)
    u = state[:, 0]
    k = coefficient[:, 0]
    domain = mask[:, 0] > 0.5
    boundary_value = bc_value[:, 0]
    u_pad = F.pad(u, (1, 1, 1, 1))
    k_pad = F.pad(k, (1, 1, 1, 1))
    mask_pad = F.pad(domain, (1, 1, 1, 1), value=False)

    def harmonic(left: Tensor, right: Tensor) -> Tensor:
        return 2.0 * left * right / (left + right).clamp_min(1.0e-12)

    result = -source[:, 0].clone()
    neighbor_views = (
        (u_pad[:, 1:-1, 2:], k_pad[:, 1:-1, 2:], mask_pad[:, 1:-1, 2:]),
        (u_pad[:, 1:-1, :-2], k_pad[:, 1:-1, :-2], mask_pad[:, 1:-1, :-2]),
        (u_pad[:, 2:, 1:-1], k_pad[:, 2:, 1:-1], mask_pad[:, 2:, 1:-1]),
        (u_pad[:, :-2, 1:-1], k_pad[:, :-2, 1:-1], mask_pad[:, :-2, 1:-1]),
    )
    for neighbor_u, neighbor_k, neighbor_active in neighbor_views:
        interior = harmonic(k, neighbor_k) * (u - neighbor_u)
        boundary = 2.0 * k * (u - boundary_value)
        result = result + torch.where(neighbor_active, interior, boundary) * inverse_h2
    return result[:, None] * mask


def raw_face_darcy_residual(
    state: Tensor,
    coefficient: Tensor,
    source: Tensor,
    mask: Tensor,
    dirichlet_face_mask: Tensor,
    dirichlet_face_value: Tensor,
    *,
    dx: float | Tensor | None = None,
) -> Tensor:
    """Return ``A u-b`` for the N/S/W/E face-resolved FV contract."""
    if state.ndim != 4 or state.shape[1] != 1:
        raise ValueError("face Darcy residual expects scalar BCHW states")
    expected = (state.shape[0], 4, *state.shape[-2:])
    if tuple(dirichlet_face_mask.shape) != expected \
            or tuple(dirichlet_face_value.shape) != expected:
        raise ValueError(f"face Darcy fields must have shape {expected}")
    height, width = state.shape[-2:]
    if dx is None:
        inverse_h2: float | Tensor = (
            max(height, width, 1) / 2.0
        ) ** 2
    elif isinstance(dx, Tensor):
        spacing = dx.to(device=state.device, dtype=state.dtype).reshape(-1)
        if spacing.numel() not in (1, state.shape[0]) or bool((spacing <= 0).any()):
            raise ValueError("grid spacing must be one positive value per batch")
        if spacing.numel() == 1:
            spacing = spacing.expand(state.shape[0])
        inverse_h2 = spacing.reciprocal().square()[:, None, None]
    else:
        spacing_value = float(dx)
        if spacing_value <= 0.0:
            raise ValueError("grid spacing must be positive")
        inverse_h2 = 1.0 / (spacing_value * spacing_value)
    u = state[:, 0]
    k = coefficient[:, 0]
    domain = mask[:, 0] > 0.5
    u_pad = F.pad(u, (1, 1, 1, 1))
    k_pad = F.pad(k, (1, 1, 1, 1))
    mask_pad = F.pad(domain, (1, 1, 1, 1), value=False)
    neighbors = (
        (u_pad[:, :-2, 1:-1], k_pad[:, :-2, 1:-1], mask_pad[:, :-2, 1:-1]),
        (u_pad[:, 2:, 1:-1], k_pad[:, 2:, 1:-1], mask_pad[:, 2:, 1:-1]),
        (u_pad[:, 1:-1, :-2], k_pad[:, 1:-1, :-2], mask_pad[:, 1:-1, :-2]),
        (u_pad[:, 1:-1, 2:], k_pad[:, 1:-1, 2:], mask_pad[:, 1:-1, 2:]),
    )
    result = -source[:, 0].clone()
    for face, (neighbor_u, neighbor_k, neighbor_active) in enumerate(neighbors):
        harmonic = 2.0 * k * neighbor_k / (k + neighbor_k).clamp_min(1.0e-12)
        interior = harmonic * (u - neighbor_u)
        boundary = 2.0 * k * (u - dirichlet_face_value[:, face])
        exposed = domain & ~neighbor_active
        covered = dirichlet_face_mask[:, face] > 0.5
        if torch.any(exposed & ~covered):
            raise ValueError("face-resolved residual has an uncovered exposed face")
        result = result + torch.where(neighbor_active, interior, boundary) * inverse_h2
    return result[:, None] * mask


def canonical_split(handle: h5py.File, split: str) -> str:
    aliases = {
        "train": ("train",),
        "validation": ("validation", "val", "valid"),
        "test": ("test",),
        "A": ("domains/A",),
        "B": ("domains/B",),
        "C": ("domains/C",),
        "D": ("domains/D",),
    }
    requested = "validation" if split in ("val", "valid") else split
    if requested not in aliases:
        raise ValueError(f"unknown split {split!r}")
    for candidate in aliases[requested]:
        if candidate in handle:
            return candidate
    raise KeyError(f"HDF5 file has no group for split {split!r}")


def _scalar_field(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 2:
        return value[None]
    if value.ndim == 3 and value.shape[0] == 1:
        return value
    raise ValueError(f"{name} must be [H,W] or [1,H,W], got {value.shape}")


def _normal_field(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3 and value.shape[0] == 2:
        return value
    if value.ndim == 3 and value.shape[-1] == 2:
        return value.transpose(2, 0, 1)
    raise ValueError(f"normals must be [2,H,W] or [H,W,2], got {value.shape}")


def _face_field(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3 and value.shape[0] == 4:
        return value
    if value.ndim == 3 and value.shape[-1] == 4:
        return value.transpose(2, 0, 1)
    raise ValueError(f"{name} must be [4,H,W] or [H,W,4], got {value.shape}")


@dataclass(frozen=True)
class DarcyStatistics:
    solution_mean: float
    solution_std: float
    logk_mean: float
    logk_std: float
    source_mean: float
    source_std: float
    bc_mean: float
    bc_std: float
    sampled_cases: int

    def to_json(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_json(cls, values: dict[str, Any]) -> "DarcyStatistics":
        return cls(**{
            key: int(value) if key == "sampled_cases" else float(value)
            for key, value in values.items()
        })


class H5DarcyDataset(Dataset[dict[str, Any]]):
    """Lazy reader for one geometry-disjoint split."""

    def __init__(self, path: Path | str, split: str) -> None:
        self.path = Path(path).resolve()
        with h5py.File(self.path, "r") as handle:
            self.split = canonical_split(handle, split)
            group = handle[self.split]
            missing = [name for name in REQUIRED_FIELDS if name not in group]
            if missing:
                raise KeyError(f"split {self.split!r} is missing {missing}")
            self.length = int(group["solution"].shape[0])
            if any(int(group[name].shape[0]) != self.length for name in REQUIRED_FIELDS):
                raise ValueError("Darcy fields have inconsistent case counts")
            present_faces = [name for name in FACE_DIRICHLET_FIELDS if name in group]
            if present_faces and len(present_faces) != len(FACE_DIRICHLET_FIELDS):
                raise KeyError(
                    "face-resolved Darcy data must provide mask, value, and role together"
                )
            self.has_face_dirichlet = bool(present_faces)
            if self.has_face_dirichlet and any(
                int(group[name].shape[0]) != self.length
                for name in FACE_DIRICHLET_FIELDS
            ):
                raise ValueError("face Dirichlet fields have inconsistent case counts")
            self.attrs = {key: value for key, value in handle.attrs.items()}
            self.attrs.update({key: value for key, value in group.attrs.items()})
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _group(self) -> h5py.Group:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle[self.split]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if hasattr(self, "case_indices"):
            index = self.case_indices[index]
        group = self._group()
        values: dict[str, Any] = {}
        for name in REQUIRED_FIELDS:
            raw = group[name][index]
            array = _normal_field(raw) if name == "normals" else _scalar_field(raw, name)
            values[name] = torch.from_numpy(np.ascontiguousarray(array))
        if self.has_face_dirichlet:
            for name in FACE_DIRICHLET_FIELDS:
                array = _face_field(group[name][index], name)
                values[name] = torch.from_numpy(np.ascontiguousarray(array))
        values["mask"] = (values["mask"] > 0.5).to(torch.float32)
        values["bc_mask"] = (values["bc_mask"] > 0.5).to(torch.float32)
        if self.has_face_dirichlet:
            values["dirichlet_face_mask"] = (
                values["dirichlet_face_mask"] > 0.5
            ).to(torch.float32)
        values["case_index"] = torch.tensor(index, dtype=torch.int64)
        if "seed" in group:
            values["seed"] = torch.tensor(int(group["seed"][index]), dtype=torch.int64)
        if "family" in group:
            family = group["family"][index]
            values["family"] = family.decode("utf-8") if isinstance(family, bytes) else str(family)
        # Older files implicitly represent a length-two square.  New domain-
        # extension files state their physical scale explicitly, allowing a
        # larger grid to mean a larger domain rather than a finer raster of
        # the same domain.  Group attributes may override file attributes.
        grid_size = int(group.attrs.get(
            "grid_size", self.attrs.get("grid_size", values["mask"].shape[-1]),
        ))
        physical_extent = float(group.attrs.get(
            "physical_extent", self.attrs.get("physical_extent", 2.0),
        ))
        grid_spacing = float(group.attrs.get(
            "grid_spacing", self.attrs.get(
                "grid_spacing", physical_extent / max(grid_size, 1),
            ),
        ))
        values["physical_extent"] = torch.tensor(
            physical_extent, dtype=torch.float32,
        )
        values["grid_spacing"] = torch.tensor(
            grid_spacing, dtype=torch.float32,
        )
        return values


def compute_statistics(
    dataset: H5DarcyDataset, max_cases: int = 512, seed: int = 20260805,
) -> DarcyStatistics:
    """Compute train-only masked statistics from a deterministic case sample."""
    if max_cases <= 0 or len(dataset) <= 0:
        raise ValueError("statistics require positive case counts")
    generator = np.random.default_rng(seed)
    count = min(max_cases, len(dataset))
    indices = np.sort(generator.choice(len(dataset), count, replace=False))
    accum: dict[str, list[np.ndarray]] = {name: [] for name in ("solution", "logk", "source", "bc_value")}
    for index in indices:
        item = dataset[int(index)]
        domain = item["mask"].numpy() > 0.5
        boundary = (item["bc_mask"].numpy() > 0.5) & domain
        for name in ("solution", "logk", "source"):
            selected = item[name].numpy()[domain]
            if selected.size:
                accum[name].append(selected.astype(np.float64, copy=False))
        selected_bc = item["bc_value"].numpy()[boundary]
        if selected_bc.size:
            accum["bc_value"].append(selected_bc.astype(np.float64, copy=False))

    def moments(name: str, fallback: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
        if not accum[name]:
            return fallback
        values = np.concatenate(accum[name])
        return float(values.mean()), float(max(values.std(), 1.0e-6))

    solution_mean, solution_std = moments("solution")
    logk_mean, logk_std = moments("logk")
    source_mean, source_std = moments("source")
    bc_mean, bc_std = moments("bc_value", (solution_mean, solution_std))
    return DarcyStatistics(
        solution_mean, solution_std, logk_mean, logk_std,
        source_mean, source_std, bc_mean, bc_std, count,
    )


class DarcyAdapter(PhysicsAdapter):
    """Static Darcy solution represented as an arbitrary-current correction.

    The same network can therefore be applied once or repeatedly.  A fixed
    point is represented by a zero normalized correction, exactly matching the
    residual endpoint semantics used by the Airfoil direct model.
    """

    state_channels = 1
    spatial_condition_channels = 12
    scalar_condition_dim = 4

    def __init__(
        self, statistics: DarcyStatistics, *, use_discrete_residual: bool = True,
    ) -> None:
        self.statistics = statistics
        # ``True`` preserves legacy checkpoints.  The workshop production
        # method sets this to False: the final channel remains in place for
        # architectural compatibility but is identically zero and no discrete
        # Darcy operator is constructed anywhere in its inference path.
        self.use_discrete_residual = bool(use_discrete_residual)

    def conditions(
        self, transition: PhysicalTransition, endpoint_phase: Tensor, level_scale: float,
    ) -> ConditionBatch:
        del endpoint_phase
        current = transition.current
        payload = transition.payload
        batch, _, height, width = current.shape
        dtype, device = current.dtype, current.device
        stats = self.statistics
        extent = payload.get("physical_extent")
        if extent is None:
            extent = torch.full(
                (batch,), 2.0, dtype=dtype, device=device,
            )
        else:
            extent = extent.to(device=device, dtype=dtype).reshape(batch)
        y_unit = (
            torch.arange(height, dtype=dtype, device=device) + 0.5
        ) / max(height, 1) - 0.5
        x_unit = (
            torch.arange(width, dtype=dtype, device=device) + 0.5
        ) / max(width, 1) - 0.5
        yy_unit, xx_unit = torch.meshgrid(y_unit, x_unit, indexing="ij")
        coordinates = torch.stack((xx_unit, yy_unit))[None] * extent[:, None, None, None]
        mask = payload["mask"].to(dtype)
        sdf = payload["sdf"]
        sdf_scale = sdf.abs().flatten(1).amax(1).clamp_min(1.0e-6)
        sdf_normalized = sdf / sdf_scale[:, None, None, None]
        if self.use_discrete_residual:
            coefficient = torch.pow(10.0, payload["logk"].clamp(-6.0, 6.0))
            raw_residual = raw_darcy_residual(
                current, coefficient, payload["source"], mask, payload["bc_value"],
                dx=payload.get("grid_spacing"),
            )
            spacing = payload.get("grid_spacing")
            if spacing is None:
                dx_squared = (2.0 / max(height, width, 1)) ** 2
            else:
                spacing = spacing.to(device=device, dtype=dtype).reshape(batch)
                dx_squared = spacing.square()[:, None, None, None]
            coefficient_scale = (
                (coefficient * mask).flatten(1).sum(1)
                / mask.flatten(1).sum(1).clamp_min(1.0)
            ).clamp_min(1.0e-6)
            residual = (
                raw_residual * dx_squared / coefficient_scale[:, None, None, None]
            ).clamp(-10.0, 10.0)
        else:
            residual = torch.zeros_like(current)
        spatial = torch.cat((
            (current - stats.solution_mean) / stats.solution_std,
            (payload["logk"] - stats.logk_mean) / stats.logk_std,
            (payload["source"] - stats.source_mean) / stats.source_std,
            mask,
            payload["bc_mask"].to(dtype),
            (payload["bc_value"] - stats.bc_mean) / stats.bc_std,
            sdf_normalized,
            payload["normals"],
            coordinates,
            residual,
        ), dim=1) * mask
        correction_rms = (
            (((current - stats.solution_mean) / stats.solution_std).square() * mask)
            .flatten(1).sum(1)
            / mask.flatten(1).sum(1).clamp_min(1.0)
        ).sqrt()
        logk_mean = (
            (payload["logk"] * mask).flatten(1).sum(1)
            / mask.flatten(1).sum(1).clamp_min(1.0)
            - stats.logk_mean
        ) / stats.logk_std
        domain_fraction = mask.flatten(1).mean(1)
        scalars = torch.stack((
            torch.full_like(correction_rms, float(level_scale)),
            correction_rms,
            logk_mean,
            domain_fraction,
        ), dim=1)
        return ConditionBatch(spatial=spatial, scalars=scalars)

    def normalized_target(self, transition: PhysicalTransition) -> Tensor:
        if transition.target is None:
            raise ValueError("Darcy correction training requires a solution target")
        mask = transition.payload["mask"].to(transition.current.dtype)
        return (
            (transition.target - transition.current) / self.statistics.solution_std
        ) * mask

    def apply_endpoint(
        self, transition: PhysicalTransition, normalized_endpoint: Tensor,
    ) -> Tensor:
        payload = transition.payload
        mask = payload["mask"].to(transition.current.dtype)
        prediction = transition.current + self.statistics.solution_std * normalized_endpoint
        return prediction * mask

    def refinement_indicator(self, transition: PhysicalTransition) -> Tensor:
        payload = transition.payload
        geometry = torch.exp(-8.0 * payload["sdf"].abs())
        coefficient = (payload["logk"] - payload["logk"].mean((-1, -2), keepdim=True)).abs()
        return (geometry + coefficient + payload["source"].abs()) * payload["mask"]


FACE_FEATURE_CHANNELS = 16


def face_condition_features(
    face_mask: Tensor,
    face_value: Tensor,
    face_role: Tensor,
    domain_mask: Tensor,
    statistics: DarcyStatistics,
) -> Tensor:
    """Encode directional Dirichlet data without collapsing corner traces.

    The channel blocks are face mask, normalized face value, physical-role
    indicator, and artificial-role indicator, each in N/S/W/E order.
    """
    expected = (face_mask.shape[0], 4, *domain_mask.shape[-2:])
    if tuple(face_mask.shape) != expected:
        raise ValueError(f"face mask must have shape {expected}")
    if tuple(face_value.shape) != expected or tuple(face_role.shape) != expected:
        raise ValueError("face mask, value, and role shapes must agree")
    dtype = domain_mask.dtype
    directional_mask = (face_mask > 0.5).to(dtype)
    role = face_role.to(dtype)
    physical = ((role.remainder(2.0)) > 0.5).to(dtype) * directional_mask
    artificial = (role >= 1.5).to(dtype) * directional_mask
    normalized_value = (
        (face_value.to(dtype) - statistics.bc_mean) / statistics.bc_std
    ) * directional_mask
    return torch.cat((
        directional_mask, normalized_value, physical, artificial,
    ), dim=1) * domain_mask


class FaceDarcyAdapter(DarcyAdapter):
    """Dedicated face-resolved local Dirichlet-BVP adapter for neural SNI."""

    spatial_condition_channels = DarcyAdapter.spatial_condition_channels \
        + FACE_FEATURE_CHANNELS

    def __init__(self, statistics: DarcyStatistics) -> None:
        super().__init__(statistics, use_discrete_residual=False)

    def conditions(
        self, transition: PhysicalTransition, endpoint_phase: Tensor,
        level_scale: float,
    ) -> ConditionBatch:
        base = super().conditions(transition, endpoint_phase, level_scale)
        payload = transition.payload
        missing = [name for name in FACE_DIRICHLET_FIELDS if name not in payload]
        if missing:
            raise ValueError(
                "face-resolved local BVP checkpoint requires v2 fields; "
                f"missing {missing}"
            )
        features = face_condition_features(
            payload["dirichlet_face_mask"].to(base.spatial.dtype),
            payload["dirichlet_face_value"].to(base.spatial.dtype),
            payload["dirichlet_face_role"].to(base.spatial.dtype),
            payload["mask"].to(base.spatial.dtype),
            self.statistics,
        )
        return ConditionBatch(
            spatial=torch.cat((base.spatial, features), dim=1),
            scalars=base.scalars,
        )


def resolve_communication_stage(
    communication_stage: CommunicationStage | str | None = None,
    grid_communication: bool | None = None,
) -> CommunicationStage:
    """Resolve the explicit stage while preserving the legacy Boolean API."""

    if communication_stage is None:
        return "both" if grid_communication else "none"
    if communication_stage not in COMMUNICATION_STAGES:
        raise ValueError(
            f"communication_stage must be one of {COMMUNICATION_STAGES}, "
            f"got {communication_stage!r}"
        )
    stage = communication_stage
    if grid_communication is not None:
        legacy_stage: CommunicationStage = "both" if grid_communication else "none"
        if stage != legacy_stage:
            raise ValueError(
                "communication_stage conflicts with legacy grid_communication"
            )
    return stage  # type: ignore[return-value]


def checkpoint_communication_stage(checkpoint: dict[str, Any]) -> CommunicationStage:
    """Infer old and new checkpoint architecture without rewriting the file."""

    explicit = checkpoint.get("communication_stage")
    if explicit is None:
        explicit = checkpoint.get("model_spec", {}).get("communication_stage")
    if explicit is not None:
        return resolve_communication_stage(str(explicit))
    contract = str(checkpoint.get("contract", ""))
    for stage, current_contract in COMMUNICATION_CONTRACTS.items():
        if contract == current_contract:
            return stage
    if contract in LEGACY_COMMUNICATION_CONTRACTS:
        return LEGACY_COMMUNICATION_CONTRACTS[contract]
    raise ValueError(
        f"cannot infer Darcy communication architecture from contract {contract!r}"
    )


def build_model(
    adapter: DarcyAdapter, *,
    communication_stage: CommunicationStage | str | None = None,
    grid_communication: bool | None = None,
) -> DirectEndpointOperator:
    """Construct the frozen cross-case architecture contract.

    Width, depth, and communication choices intentionally have no CLI override:
    changing them would break the workshop's same-model-design comparison.
    """
    stage = resolve_communication_stage(communication_stage, grid_communication)
    physical_channels = adapter.spatial_condition_channels + 1
    proposal = DirectEndpointRegressor(
        adapter.state_channels,
        physical_channels,
        adapter.scalar_condition_dim,
        width=MODEL_SPEC["proposal_width"],
        depth=MODEL_SPEC["proposal_depth"],
        cell_attention=MODEL_SPEC["cell_attention"],
        grid_communication=stage in ("proposal", "both"),
    )
    projector = NeuralEndpointProjector(
        adapter.state_channels,
        physical_channels,
        adapter.scalar_condition_dim,
        width=MODEL_SPEC["projector_width"],
        depth=MODEL_SPEC["projector_depth"],
        cell_attention=MODEL_SPEC["cell_attention"],
        grid_communication=stage == "both",
    )
    return DirectEndpointOperator(proposal, projector)


def batch_to_transition(
    batch: dict[str, Tensor], current: Tensor, *, previous: Tensor | None = None,
    target: Tensor | None = None,
) -> PhysicalTransition:
    """Move a collated batch into the system-independent transition object."""
    if previous is None:
        previous = current
    if target is None:
        target = batch.get("solution")
    payload = {name: batch[name] for name in REQUIRED_FIELDS if name != "solution"}
    payload.update({
        name: batch[name] for name in FACE_DIRICHLET_FIELDS if name in batch
    })
    payload.update({
        name: batch[name] for name in OPTIONAL_PHYSICAL_METADATA if name in batch
    })
    return PhysicalTransition(
        previous=previous,
        current=current,
        target=target,
        dt=torch.ones(current.shape[0], device=current.device, dtype=current.dtype),
        payload=payload,
        metadata={"case_index": batch.get("case_index")},
    )


def enforce_domain_boundary(batch: dict[str, Tensor], state: Tensor) -> Tensor:
    mask = batch["mask"].to(state.dtype)
    # Boundary values live at exterior half-cell faces; active boundary-cell
    # centers remain unknowns in the reference finite-volume system.
    return state * mask


def tiled_darcy_conditions(
    adapter: DarcyAdapter,
    transition: PhysicalTransition,
    cover: CartesianCover,
    *,
    level_scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Build local conditions and inject the current artificial trace.

    Stored 40x40 training BVPs carry exact artificial-window Dirichlet data on
    their outer cells.  A full-domain cover must construct the analogous data
    at inference time: physical-boundary values remain prescribed, while each
    local patch's outer active cells receive the current global iterate as
    their artificial trace.  Without this step repeated composition would not
    actually communicate the interface state seen during local training.
    """
    phase = torch.ones(
        transition.batch_size,
        device=transition.current.device,
        dtype=transition.current.dtype,
    )
    condition = adapter.conditions(transition, phase, level_scale)
    physical, scalar = condition.tiled(cover)
    if cover.halo == 0:
        return physical, scalar

    # Adapter channel contract: current=0, domain mask=3, boundary mask=4,
    # normalized boundary value=5. The final generic channel is tile validity.
    active = physical[:, 3:4] > 0.5
    patch_edge = torch.zeros_like(active)
    patch_edge[..., 0, :] = True
    patch_edge[..., -1, :] = True
    patch_edge[..., :, 0] = True
    patch_edge[..., :, -1] = True
    artificial = patch_edge & active & (physical[:, 4:5] < 0.5)
    current_value = (
        physical[:, 0:1] * adapter.statistics.solution_std
        + adapter.statistics.solution_mean
    )
    current_as_boundary = (
        current_value - adapter.statistics.bc_mean
    ) / adapter.statistics.bc_std
    physical = physical.clone()
    physical[:, 5:6] = torch.where(
        artificial, current_as_boundary, physical[:, 5:6],
    )
    physical[:, 4:5] = torch.where(
        artificial, torch.ones_like(physical[:, 4:5]), physical[:, 4:5],
    )
    return physical, scalar


@dataclass(frozen=True)
class FaceResolvedSNIConditions:
    """Canonical neural inputs plus the exact physical local-FV contract."""

    physical: Tensor
    scalar: Tensor
    mask: Tensor
    kappa: Tensor
    source: Tensor
    dirichlet_face_mask: Tensor
    dirichlet_face_value: Tensor
    dirichlet_face_role: Tensor
    valid: Tensor
    dx: float


def _shift_global(field: Tensor, face: int) -> Tensor:
    """Align a global north/south/west/east neighbor with each center cell."""
    if face == 0:
        return F.pad(field, (0, 0, 1, 0))[..., :-1, :]
    if face == 1:
        return F.pad(field, (0, 0, 0, 1))[..., 1:, :]
    if face == 2:
        return F.pad(field, (1, 0, 0, 0))[..., :, :-1]
    if face == 3:
        return F.pad(field, (0, 1, 0, 0))[..., :, 1:]
    raise ValueError("Darcy face index must be in north/south/west/east order")


def _legacy_global_face_dirichlet(
    batch: dict[str, Tensor], dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Lift the historical cellwise data to four physical face channels."""
    domain = batch["mask"].to(dtype) > 0.5
    values = batch["bc_value"].to(dtype)
    masks = []
    for face in range(4):
        masks.append(domain & ~(_shift_global(domain.to(dtype), face) > 0.5))
    face_mask = torch.cat(masks, dim=1)
    face_value = values.expand(-1, 4, -1, -1).clone()
    face_role = face_mask.to(dtype)  # role bit 1: physical boundary
    return face_mask.to(dtype), face_value, face_role


def tiled_sni_face_conditions(
    adapter: DarcyAdapter,
    batch: dict[str, Tensor],
    current: Tensor,
    cover: CartesianCover,
) -> FaceResolvedSNIConditions:
    """Build face-resolved local BVPs without exposing the interior iterate.

    The exact fields retain physical units and the global ``dx``.  ``physical``
    and ``scalar`` retain the historical network width by collapsing directional
    values to a cellwise mean; a future face-aware network must consume the
    explicit four-channel tensors instead.  Old checkpoints continue to use
    :func:`tiled_darcy_conditions` unchanged.
    """
    batch_size = current.shape[0]
    device, dtype = current.device, current.dtype
    tile_count, size = cover.tile_count, cover.tile_size
    nlocal = batch_size * tile_count
    stats = adapter.statistics

    mask = cover.tile(batch["mask"].to(dtype))
    logk = cover.tile(batch["logk"].to(dtype))
    # Mapping a physical patch to the canonical length-two local-training
    # square multiplies the Darcy source by the square of the length scale.
    # The historical expression ``size / global_height`` is only correct when
    # every global record spans a length-two box.  Explicit ``grid_spacing``
    # keeps the local physical footprint fixed during genuine domain growth.
    spacing = batch.get("grid_spacing")
    if spacing is None:
        global_dx = 2.0 / max(cover.height, cover.width)
    else:
        spacing_values = spacing.to(device=device, dtype=dtype).reshape(-1)
        if not torch.allclose(
            spacing_values, spacing_values[:1], rtol=1.0e-6, atol=1.0e-8,
        ):
            raise ValueError("one tiled SNI batch must use a common grid spacing")
        global_dx = float(spacing_values[0].item())
    local_training_dx = 2.0 / size
    spatial_scale = global_dx / local_training_dx
    source_physical = cover.tile(batch["source"].to(dtype))
    source = source_physical * spatial_scale**2
    sdf = cover.tile(batch["sdf"].to(dtype))
    normals = cover.tile(batch["normals"].to(dtype))
    current_tiles = cover.tile(current)
    valid = cover.valid_tiles(batch_size, device, dtype)

    active = mask > 0.5
    global_kappa = torch.pow(10.0, batch["logk"].to(dtype).clamp(-6.0, 6.0))
    if all(name in batch for name in FACE_DIRICHLET_FIELDS):
        global_face_mask = batch["dirichlet_face_mask"].to(dtype)
        global_face_value = batch["dirichlet_face_value"].to(dtype)
        global_face_role = batch["dirichlet_face_role"].to(dtype)
    else:
        global_face_mask, global_face_value, global_face_role = (
            _legacy_global_face_dirichlet(batch, dtype)
        )
    face_mask = cover.tile(global_face_mask)
    face_value = cover.tile(global_face_value)
    face_role = cover.tile(global_face_role)
    center_kappa = torch.pow(10.0, logk.clamp(-6.0, 6.0))
    global_domain = batch["mask"].to(dtype)
    for direction in range(4):
        patch_edge = torch.zeros_like(active, dtype=torch.bool)
        if direction == 0:
            patch_edge[..., 0, :] = True
        elif direction == 1:
            patch_edge[..., -1, :] = True
        elif direction == 2:
            patch_edge[..., :, 0] = True
        else:
            patch_edge[..., :, -1] = True
        neighbor_tile = cover.tile(_shift_global(current, direction))
        neighbor_active = cover.tile(
            _shift_global(global_domain, direction)
        ) > 0.5
        neighbor_kappa = cover.tile(_shift_global(global_kappa, direction))
        current_face = (
            patch_edge & active & (valid > 0.5) & neighbor_active
            & ~(face_mask[:, direction:direction + 1] > 0.5)
        )
        # Choose the face trace so the local half-cell Dirichlet flux equals
        # the global harmonic two-cell flux.  This reduces to midpoint
        # interpolation for equal permeability and makes the exact discrete
        # solution a fixed point of the subdomain map.
        denominator = (center_kappa + neighbor_kappa).clamp_min(1.0e-12)
        trace = (
            center_kappa / denominator * current_tiles
            + neighbor_kappa / denominator * neighbor_tile
        )
        face_mask[:, direction:direction + 1] = torch.where(
            current_face, torch.ones_like(current_face, dtype=dtype),
            face_mask[:, direction:direction + 1],
        )
        face_value[:, direction:direction + 1] = torch.where(
            current_face, trace, face_value[:, direction:direction + 1],
        )
        face_role[:, direction:direction + 1] = torch.where(
            current_face, torch.full_like(trace, 2.0),
            face_role[:, direction:direction + 1],
        )

    face_count = face_mask.sum(1, keepdim=True)
    bc_mask = face_count > 0.5
    bc_value = (face_value * face_mask).sum(1, keepdim=True) \
        / face_count.clamp_min(1.0)

    coordinate = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    coordinates = torch.stack((xx, yy))[None].expand(nlocal, -1, -1, -1)
    sdf_scale = (
        (sdf.abs() * mask).flatten(1).amax(1).clamp_min(1.0e-6)
    )
    sdf_normalized = sdf / sdf_scale[:, None, None, None]
    zero_current = torch.full_like(mask, -stats.solution_mean / stats.solution_std)
    residual = torch.zeros_like(mask)
    spatial_core = torch.cat((
        zero_current,
        (logk - stats.logk_mean) / stats.logk_std,
        (source - stats.source_mean) / stats.source_std,
        mask,
        bc_mask,
        (bc_value - stats.bc_mean) / stats.bc_std,
        sdf_normalized,
        normals,
        coordinates,
        residual,
    ), dim=1) * mask
    if isinstance(adapter, FaceDarcyAdapter):
        spatial_core = torch.cat((
            spatial_core,
            face_condition_features(
                face_mask, face_value, face_role, mask, stats,
            ),
        ), dim=1)
    spatial = torch.cat((spatial_core, valid), dim=1)

    count = mask.flatten(1).sum(1).clamp_min(1.0)
    zero_rms = (
        (zero_current.square() * mask).flatten(1).sum(1) / count
    ).sqrt()
    logk_mean = (
        (logk * mask).flatten(1).sum(1) / count - stats.logk_mean
    ) / stats.logk_std
    domain_fraction = mask.flatten(1).mean(1)
    scalar = torch.stack((
        torch.ones_like(zero_rms), zero_rms, logk_mean, domain_fraction,
    ), dim=1)
    return FaceResolvedSNIConditions(
        physical=spatial,
        scalar=scalar,
        mask=mask,
        kappa=center_kappa,
        source=source_physical,
        dirichlet_face_mask=face_mask,
        dirichlet_face_value=face_value,
        dirichlet_face_role=face_role,
        valid=valid,
        dx=global_dx,
    )


def tiled_sni_conditions(
    adapter: DarcyAdapter,
    batch: dict[str, Tensor],
    current: Tensor,
    cover: CartesianCover,
) -> tuple[Tensor, Tensor]:
    """Compatibility view of the face-resolved SNI boundary contract."""
    conditions = tiled_sni_face_conditions(adapter, batch, current, cover)
    return conditions.physical, conditions.scalar
