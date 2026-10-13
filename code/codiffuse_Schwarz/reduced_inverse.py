"""Target-free probes for a frozen reduced compatibility response.

Case adapters own the field decoder and compatibility analysis.  This module
fixes the sign, masking, and normalization of their shared phase-2
probe--invert contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class ReducedInverseProbe:
    """One synthetic reduced-inverse training pair."""

    excitation: Tensor
    response: Tensor
    inverse_rhs: Tensor
    supported_excitation: Tensor


def build_reduced_inverse_probe(
    baseline_residual: Tensor,
    perturbed_residual: Tensor,
    excitation: Tensor,
    *,
    supported_coordinates: Tensor | None = None,
) -> ReducedInverseProbe:
    """Build ``C(c, -[q(v+L(xi))-q(v)]) ~= xi``.

    The minus sign belongs to the *synthetic probe input*.  If
    ``Delta q = A xi``, these pairs identify ``C ~= -A^dagger``.  At normal
    inference the communicator therefore consumes the measured baseline
    compatibility coordinate ``q(v)`` itself, not ``-q(v)``: then
    ``z=C(q(v))`` approximately solves ``A z=-q(v)``.  Keeping this distinction
    explicit prevents applying the reduced-inverse sign twice.

    ``baseline_residual`` and ``perturbed_residual`` may have a different final
    width from ``excitation``.  ``supported_coordinates`` masks only code
    coordinates; it is used to remove physical faces absent from a tile and
    QoI coordinates that are invisible to the compatibility operator.
    """
    if baseline_residual.shape != perturbed_residual.shape:
        raise ValueError("baseline and perturbed residuals must have the same shape")
    if baseline_residual.ndim < 2 or excitation.ndim < 2:
        raise ValueError("probe tensors must retain batch and coordinate axes")
    if baseline_residual.shape[:-1] != excitation.shape[:-1]:
        raise ValueError("residual and excitation token layouts must match")
    if not all(torch.isfinite(value).all() for value in (
        baseline_residual, perturbed_residual, excitation,
    )):
        raise ValueError("probe tensors must be finite")
    if supported_coordinates is None:
        supported = torch.ones_like(excitation)
    else:
        try:
            supported = torch.broadcast_to(
                supported_coordinates.to(excitation), excitation.shape,
            )
        except RuntimeError as error:
            raise ValueError("supported-coordinate mask is not broadcastable") from error
        if torch.any((supported < 0.0) | (supported > 1.0)):
            raise ValueError("supported-coordinate mask must lie in [0,1]")
    excitation_supported = excitation * supported
    response = perturbed_residual - baseline_residual
    return ReducedInverseProbe(
        excitation=excitation,
        response=response,
        inverse_rhs=-response,
        supported_excitation=excitation_supported,
    )


def inverse_cycle_relative_square(
    predicted_excitation: Tensor,
    probe: ReducedInverseProbe,
    *,
    supported_coordinates: Tensor | None = None,
    floor: float = 1.0e-8,
) -> Tensor:
    """Return batchwise relative-square inverse-cycle error."""
    if predicted_excitation.shape != probe.excitation.shape or floor <= 0.0:
        raise ValueError("predicted excitation shape/floor is incompatible")
    if supported_coordinates is None:
        supported = torch.ones_like(probe.excitation)
    else:
        try:
            supported = torch.broadcast_to(
                supported_coordinates.to(probe.excitation), probe.excitation.shape,
            )
        except RuntimeError as error:
            raise ValueError("supported-coordinate mask is not broadcastable") from error
    axes = tuple(range(1, predicted_excitation.ndim))
    difference = (predicted_excitation - probe.supported_excitation) * supported
    numerator = difference.square().sum(axes)
    denominator = probe.supported_excitation.square().sum(axes).clamp_min(floor)
    return numerator / denominator
