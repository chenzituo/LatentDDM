"""Label-free provisional compatibility defects for the Airfoil analyzer.

This module is intentionally separate from response-chart training.  Given
frozen local endpoints ``v_i``, the caller first realizes their zero-code POU
consensus and re-restricts it to obtain ``R_i A_chi(v)``.  The defect

    d_i = R_i A_chi(v) - v_i

is projected in exactly the same fixed residual-port coordinate used by the
oracle response chart.  No target field or discrete PDE residual is used.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from codiffuse_Schwarz.residual_port import (
    ResidualPortSpec,
    project_residual_ports,
    scale_port_code,
)
from .airfoil_compatibility import AIRFOIL_RESIDUAL_PORT_SPEC, PortFunctionalKind


Tensor = torch.Tensor


def _statistics_tensor(values: tuple[float, float, float], reference: Tensor) -> Tensor:
    return reference.new_tensor(values)[None, :, None, None]


def _masked_mean(field: Tensor, weight: Tensor) -> Tensor:
    return (field * weight).sum((-2, -1)) / weight.sum(
        (-2, -1),
    ).clamp_min(1.0)


def provisional_defect_port_code(
    local_endpoint: Tensor,
    consensus_endpoint: Tensor,
    physical: Tensor,
    *,
    state_mean: tuple[float, float, float],
    state_std: tuple[float, float, float],
    delta_mean: tuple[float, float, float],
    delta_std: tuple[float, float, float],
    port_scale: Tensor | None = None,
    functional_kind: PortFunctionalKind = "pressure_traction",
    force_area_scale: float | Tensor = 1.0,
    lift_sign: float = 1.0,
    trace_modes: int = 2,
) -> Tensor:
    """Project consensus defects into the same fixed oracle port coordinate.

    Coordinates 0:24 are the shared oriented degree-one face projection of
    ``consensus_endpoint-local_endpoint``. Coordinates 24:30 are the chart's
    positive/negative streamwise-upwind residual moments. Coordinates 30:32
    are its load residuals. Thus this deployment-time code has exactly the
    frozen chart's train-time coordinate semantics; only its provisional
    target (neighbor consensus instead of a supervised field) differs.
    """
    if consensus_endpoint.shape != local_endpoint.shape:
        raise ValueError("consensus and local endpoints must have identical shape")
    if trace_modes not in (2, 4, 8):
        raise ValueError("defect trace modes must be 2, 4, or 8")
    if physical.ndim != 4 or physical.shape[0] != local_endpoint.shape[0]:
        raise ValueError("physical fields must share the endpoint tile batch")
    valid = physical[:, -1:].to(local_endpoint).clamp(0.0, 1.0)
    defect = (consensus_endpoint - local_endpoint) * valid
    current = (
        physical[:, :3] * _statistics_tensor(state_std, local_endpoint)
        + _statistics_tensor(state_mean, local_endpoint)
    )
    increment = (
        local_endpoint * _statistics_tensor(delta_std, local_endpoint)
        + _statistics_tensor(delta_mean, local_endpoint)
    )
    predicted = current + increment

    incoming_left = F.relu(predicted[:, :1]) * valid
    incoming_right = F.relu(-predicted[:, :1]) * valid
    incoming = torch.cat((
        _masked_mean(defect, incoming_left),
        _masked_mean(defect, incoming_right),
    ), dim=1)

    defect_physical = defect * _statistics_tensor(delta_std, defect)
    destination_sdf = physical[:, 4:5]
    grad_y, grad_x = torch.gradient(destination_sdf, dim=(-2, -1))
    magnitude = torch.sqrt(grad_x.square() + grad_y.square()).clamp_min(1.0e-6)
    surface = torch.exp(-8.0 * destination_sdf.abs()) * valid
    if functional_kind == "pressure_traction":
        pressure_residual = defect_physical[:, 2:3]
        terminal = torch.cat((
            _masked_mean(-pressure_residual * grad_x / magnitude, surface),
            _masked_mean(-pressure_residual * grad_y / magnitude, surface),
        ), dim=1)
    elif functional_kind == "brinkman_force":
        solid = (destination_sdf <= 0.0).to(defect) * valid
        scale = torch.as_tensor(
            force_area_scale, device=defect.device, dtype=defect.dtype,
        )
        if scale.ndim == 1:
            scale = scale[:, None]
        if scale.ndim not in (0, 2) or (
            scale.ndim == 2 and scale.shape != (defect.shape[0], 1)
        ):
            raise ValueError("force_area_scale must be scalar or one value per tile")
        terminal = torch.cat((
            scale * (defect_physical[:, 0:1] * solid).sum((-2, -1)),
            -lift_sign * scale
            * (defect_physical[:, 1:2] * solid).sum((-2, -1)),
        ), dim=1)
    else:
        raise ValueError(f"unknown Airfoil port functional kind: {functional_kind}")
    specification = (
        AIRFOIL_RESIDUAL_PORT_SPEC
        if trace_modes == 2 else ResidualPortSpec(
            dimension=2, field_channels=3,
            polynomial_degree=trace_modes - 1,
            maximum_modes_per_face=trace_modes,
            appended_functionals=8,
        )
    )
    code = project_residual_ports(
        defect, valid, specification,
        functionals=torch.cat((incoming, terminal), dim=1),
    )
    if port_scale is not None:
        code = scale_port_code(code, port_scale)
    return code
