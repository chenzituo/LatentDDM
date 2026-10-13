"""Identifiable residual-port coordinates for one-pass local composition.

The ideal compatibility state is a fixed projection of the broken correction

    e_i = R_i u - u_tilde_i

on the oriented faces of each local tile.  Unlike an unconstrained latent,
this coordinate has the same meaning across checkpoints, admits a direct
oracle diagnostic, and makes the zero code the unchanged local endpoint.

Only local canonical face coordinates are used.  No absolute tile coordinate
or discrete PDE operator enters this projection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import torch


Tensor = torch.Tensor


def _multi_indices(face_dimension: int, degree: int) -> list[tuple[int, ...]]:
    indices = [
        index for index in product(range(degree + 1), repeat=face_dimension)
        if sum(index) <= degree
    ]
    return sorted(indices, key=lambda index: (sum(index), index[::-1]))


def _legendre_values(degree: int, coordinate: Tensor) -> list[Tensor]:
    values = [torch.ones_like(coordinate)]
    if degree == 0:
        return values
    values.append(coordinate)
    for order in range(2, degree + 1):
        values.append(
            ((2 * order - 1) * coordinate * values[-1]
             - (order - 1) * values[-2]) / order
        )
    return values


def canonical_face_basis(
    face_shape: tuple[int, ...],
    degree: int,
    maximum_modes: int | None,
    reference: Tensor,
) -> Tensor:
    """Return a total-degree tensor Legendre basis as ``M,*face_shape``."""
    if len(face_shape) not in (1, 2) or min(face_shape) <= 0 or degree < 0:
        raise ValueError("face basis requires a positive 1D/2D shape and degree")
    coordinates = [
        torch.linspace(-1.0, 1.0, length, device=reference.device,
                       dtype=reference.dtype)
        for length in face_shape
    ]
    grids = torch.meshgrid(*coordinates, indexing="ij")
    values = [_legendre_values(degree, grid) for grid in grids]
    indices = _multi_indices(len(face_shape), degree)
    if maximum_modes is not None:
        if maximum_modes <= 0:
            raise ValueError("maximum_modes must be positive when specified")
        indices = indices[:maximum_modes]
    modes = []
    for index in indices:
        mode = torch.ones_like(grids[0])
        for axis, order in enumerate(index):
            mode = mode * values[axis][order]
        modes.append(mode)
    return torch.stack(modes)


@dataclass(frozen=True)
class ResidualPortSpec:
    """Fixed trace-coordinate contract shared by all cases.

    Face order follows tensor spatial-axis order, negative then positive.  It
    is ``(y-,y+,x-,x+)`` in 2D and ``(z-,z+,y-,y+,x-,x+)`` for ``D,H,W`` 3D
    tensors.
    """

    dimension: int
    field_channels: int
    polynomial_degree: int = 1
    maximum_modes_per_face: int | None = None
    appended_functionals: int = 0
    ridge: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.dimension not in (2, 3):
            raise ValueError("ResidualPortSpec dimension must be 2 or 3")
        if self.field_channels <= 0 or self.polynomial_degree < 0:
            raise ValueError("invalid residual-port field dimensions")
        if self.maximum_modes_per_face is not None and self.maximum_modes_per_face <= 0:
            raise ValueError("maximum_modes_per_face must be positive")
        if self.appended_functionals < 0 or self.ridge <= 0.0:
            raise ValueError("invalid residual-port functional/ridge setting")

    @property
    def modes_per_face(self) -> int:
        count = len(_multi_indices(self.dimension - 1, self.polynomial_degree))
        return count if self.maximum_modes_per_face is None else min(
            count, self.maximum_modes_per_face,
        )

    @property
    def trace_width(self) -> int:
        return 2 * self.dimension * self.field_channels * self.modes_per_face

    @property
    def code_width(self) -> int:
        return self.trace_width + self.appended_functionals

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ResidualPortSpec":
        return cls(**value)


def _weighted_projection(
    face: Tensor, validity: Tensor, basis: Tensor, ridge: float,
) -> Tensor:
    """Masked least-squares coefficients, returned as ``B,C,M``."""
    batch, channels = face.shape[:2]
    flattened = face.reshape(batch, channels, -1).float()
    weight = validity.reshape(batch, 1, -1)[:, 0].float().clamp(0.0, 1.0)
    phi = basis.reshape(basis.shape[0], -1).float()
    gram = torch.einsum("ml,bl,nl->bmn", phi, weight, phi)
    rhs = torch.einsum("bcl,bl,ml->bcm", flattened, weight, phi)
    diagonal_scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1.0)
    identity = torch.eye(phi.shape[0], device=face.device, dtype=torch.float32)
    regularized = gram + ridge * diagonal_scale[:, None, None] * identity
    # ``einsum`` participates in CUDA autocast even when its inputs were
    # explicitly promoted above. The small least-squares system is both more
    # stable and dtype-consistent when its solve boundary is forced to FP32.
    coefficient = torch.linalg.solve(
        regularized.float(), rhs.float().transpose(1, 2),
    ).transpose(1, 2)
    active = (weight.sum(-1) > 0)[:, None, None]
    return torch.where(active, coefficient, torch.zeros_like(coefficient)).to(face.dtype)


def project_residual_ports(
    residual: Tensor,
    validity: Tensor,
    spec: ResidualPortSpec,
    *,
    functionals: Tensor | None = None,
) -> Tensor:
    """Project oriented correction traces and append optional functionals.

    Args:
        residual: ``B,C,H,W`` or ``B,C,D,H,W`` broken correction tiles.
        validity: matching one-channel mask.
        spec: immutable port definition.
        functionals: optional ``B,F`` conservation/load residuals.
    """
    expected_rank = spec.dimension + 2
    if residual.ndim != expected_rank or residual.shape[1] != spec.field_channels:
        raise ValueError("residual tensor does not match the port specification")
    if validity.shape != (residual.shape[0], 1, *residual.shape[2:]):
        raise ValueError("validity must be a matching one-channel field")
    coefficients = []
    for spatial_axis in range(spec.dimension):
        tensor_axis = spatial_axis + 2
        for index in (0, -1):
            face = residual.select(tensor_axis, index)
            face_valid = validity.select(tensor_axis, index)
            basis = canonical_face_basis(
                tuple(face.shape[2:]), spec.polynomial_degree,
                spec.maximum_modes_per_face, residual,
            )
            coefficients.append(
                _weighted_projection(face, face_valid, basis, spec.ridge)
            )
    trace = torch.stack(coefficients, dim=1).reshape(residual.shape[0], -1)
    if spec.appended_functionals:
        if functionals is None or functionals.shape != (
            residual.shape[0], spec.appended_functionals,
        ):
            raise ValueError("port specification requires matching functionals")
        trace = torch.cat((trace, functionals.to(trace)), dim=-1)
    elif functionals is not None:
        raise ValueError("functionals were supplied to a trace-only port specification")
    if trace.shape[1] != spec.code_width:
        raise RuntimeError("internal residual-port width mismatch")
    return trace


def fit_zero_preserving_scale(codes: Tensor, floor: float = 1.0e-5) -> Tensor:
    """Fit per-coordinate RMS scaling without shifting the physical zero."""
    if codes.ndim != 2 or codes.shape[0] == 0 or floor <= 0.0:
        raise ValueError("codes must be a nonempty matrix and floor positive")
    return codes.float().square().mean(0).sqrt().clamp_min(floor)


def scale_port_code(code: Tensor, scale: Tensor) -> Tensor:
    if scale.ndim != 1 or code.shape[-1] != scale.shape[0]:
        raise ValueError("port scale is incompatible with the code")
    return code / scale.to(code)


def unscale_port_code(code: Tensor, scale: Tensor) -> Tensor:
    if scale.ndim != 1 or code.shape[-1] != scale.shape[0]:
        raise ValueError("port scale is incompatible with the code")
    return code * scale.to(code)
