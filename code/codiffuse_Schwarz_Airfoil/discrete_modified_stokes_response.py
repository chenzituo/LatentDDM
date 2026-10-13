"""Parameter-free local response for a time-discrete incompressible step.

The learned Airfoil endpoint supplies a particular solution.  This module
supplies a diagnostic homogeneous boundary response without consulting a
neighbouring field or a reference solution.  On one fixed native time step we
approximate

    dt**-1 du - nu Laplacian(du) + grad(dp) = 0,
    div(du) = 0,

with prescribed velocity defects on artificial faces.  Incompressibility is
imposed by an augmented-Lagrangian penalty; the resulting symmetric
positive-definite system is solved matrix-free.  The pressure response is the
associated penalty multiplier.  The solve domain is explicit: historical
checkpoints treated negative SDF as a hole, while the actual Airfoil corpus
uses incflo's all-regular geometry with volume penalization and therefore
uses every valid crop cell.  This remains a numerical diagnostic, not an
assertion that the collocated stencil is identical to incflo's nodal
projection.

The solve is homogeneous and target-free.  A zero boundary defect returns an
exactly zero response before any iteration, which keeps Local/Core unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


Tensor = torch.Tensor


RESPONSE_DOMAIN_SDF_FLUID = "sdf_fluid"
RESPONSE_DOMAIN_ALL_REGULAR_VALID = "all_regular_valid"
RESPONSE_DOMAINS = (
    RESPONSE_DOMAIN_SDF_FLUID,
    RESPONSE_DOMAIN_ALL_REGULAR_VALID,
)


def response_domain_mask(physical: Tensor, response_domain: str) -> Tensor:
    """Return the declared homogeneous-response solve domain.

    The native Airfoil channel contract stores endpoint SDF in channel four
    and crop validity in the final channel.  ``sdf_fluid`` is retained solely
    for byte-compatible reconstruction of historical checkpoints.  The
    generated Airfoil data use ``incflo.geometry = all_regular`` and volume
    penalization, so ``all_regular_valid`` is the discretization-consistent
    choice even in cells whose endpoint SDF is negative.
    """
    if physical.ndim != 4 or physical.shape[1] < 5:
        raise ValueError("physical input omits the native Airfoil mask channels")
    if response_domain == RESPONSE_DOMAIN_SDF_FLUID:
        return (physical[:, 4:5] > 0.0).to(physical)
    if response_domain == RESPONSE_DOMAIN_ALL_REGULAR_VALID:
        return (physical[:, -1:] > 0.5).to(physical)
    raise ValueError(f"unknown homogeneous response domain: {response_domain}")


@dataclass(frozen=True)
class ModifiedStokesResponseDiagnostics:
    """Per-crop residual and incompressibility diagnostics."""

    relative_linear_residual: Tensor
    physical_divergence_rms: Tensor
    iterations: int


def _velocity_face_values(
    boundary_defect: Tensor,
    face_roles: Tensor,
) -> tuple[Tensor, Tensor]:
    """Collapse four oriented velocity blocks to one boundary value."""
    if boundary_defect.ndim != 4 or boundary_defect.shape[1] != 8:
        raise ValueError("velocity boundary defect must have shape B,8,H,W")
    if face_roles.shape != (
        boundary_defect.shape[0], 4, *boundary_defect.shape[-2:],
    ):
        raise ValueError("face roles do not match the velocity defect")
    values = boundary_defect.reshape(
        boundary_defect.shape[0], 4, 2, *boundary_defect.shape[-2:],
    )
    count = face_roles.sum(1, keepdim=True)
    value = values.sum(1) / count.clamp_min(1.0)
    return value, count > 0.0


def _edge_mask(reference: Tensor) -> Tensor:
    mask = torch.zeros_like(reference, dtype=torch.bool)
    mask[..., 0, :] = True
    mask[..., -1, :] = True
    mask[..., :, 0] = True
    mask[..., :, -1] = True
    return mask


def _gradient_energy_operator(field: Tensor, fluid: Tensor) -> Tensor:
    """Return the adjoint-forward gradient operator with wall edge weights."""
    result = torch.zeros_like(field)
    fluid_vector = fluid.expand(-1, field.shape[1], -1, -1)

    difference_x = field[..., 1:] - field[..., :-1]
    weight_x = torch.maximum(
        fluid_vector[..., 1:], fluid_vector[..., :-1],
    )
    weighted_x = difference_x * weight_x
    result[..., :-1] -= weighted_x
    result[..., 1:] += weighted_x

    difference_y = field[..., 1:, :] - field[..., :-1, :]
    weight_y = torch.maximum(
        fluid_vector[..., 1:, :], fluid_vector[..., :-1, :],
    )
    weighted_y = difference_y * weight_y
    result[..., :-1, :] -= weighted_y
    result[..., 1:, :] += weighted_y
    return result


def _backward_divergence_index(velocity: Tensor) -> Tensor:
    """Dimensionless backward divergence on lower-right cell corners."""
    return (
        velocity[:, 0, 1:, 1:] - velocity[:, 0, 1:, :-1]
        + velocity[:, 1, 1:, 1:] - velocity[:, 1, :-1, 1:]
    )


def _divergence_adjoint_index(residual: Tensor, shape: tuple[int, int]) -> Tensor:
    """Adjoint of :func:`_backward_divergence_index`."""
    height, width = shape
    result = residual.new_zeros(residual.shape[0], 2, height, width)
    result[:, 0, 1:, 1:] += residual
    result[:, 0, 1:, :-1] -= residual
    result[:, 1, 1:, 1:] += residual
    result[:, 1, :-1, 1:] -= residual
    return result


def _divergence_weight(fluid: Tensor) -> Tensor:
    """Use divergence equations whose complete collocated stencil is fluid."""
    label = fluid[:, 0]
    return (
        label[:, 1:, 1:] * label[:, 1:, :-1]
        * label[:, :-1, 1:] * label[:, :-1, :-1]
    )


def _batch_inner(left: Tensor, right: Tensor) -> Tensor:
    return (left * right).flatten(1).sum(1)


def augmented_modified_stokes_velocity_response(
    boundary_velocity: Tensor,
    boundary_active: Tensor,
    fluid: Tensor,
    *,
    diffusive_cfl: Tensor | float,
    divergence_penalty: float = 100.0,
    iterations: int = 160,
    tolerance: float = 1.0e-7,
) -> tuple[Tensor, Tensor, Tensor]:
    """Solve the homogeneous augmented modified-Stokes velocity system.

    ``boundary_velocity`` is physical velocity, while ``diffusive_cfl`` is
    ``nu * dt / dx**2``.  All four crop edges are fixed: represented
    artificial faces take the supplied value and other edges take zero.  Solid
    and invalid cells are also fixed to zero.  A matrix-free conjugate-gradient
    solve is used because local masks differ near the moving airfoil.

    Returns physical velocity response, dimensionless divergence, and the
    relative linear residual for each crop.
    """
    if boundary_velocity.ndim != 4 or boundary_velocity.shape[1] != 2:
        raise ValueError("boundary velocity must have shape B,2,H,W")
    if boundary_active.shape != boundary_velocity[:, :1].shape:
        raise ValueError("boundary activity must have shape B,1,H,W")
    if fluid.shape != boundary_velocity[:, :1].shape:
        raise ValueError("fluid mask must have shape B,1,H,W")
    if iterations <= 0 or not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("iterations and tolerance must be positive")
    if not math.isfinite(divergence_penalty) or divergence_penalty <= 0.0:
        raise ValueError("divergence penalty must be positive")

    work_dtype = (
        torch.float64
        if boundary_velocity.dtype == torch.float64 else torch.float32
    )
    boundary = boundary_velocity.to(work_dtype)
    fluid_work = (fluid > 0.5).to(work_dtype)
    active = boundary_active.to(work_dtype)
    edge = _edge_mask(fluid_work).to(work_dtype)
    fixed = edge * fluid_work
    prescribed = boundary * active * fixed
    free = fluid_work * (1.0 - edge)
    free_vector = free.expand(-1, 2, -1, -1)
    divergence_weight = _divergence_weight(fluid_work)
    mu = torch.as_tensor(
        diffusive_cfl, device=boundary.device, dtype=work_dtype,
    ).flatten()
    if mu.numel() == 1:
        mu = mu.expand(boundary.shape[0])
    if mu.numel() != boundary.shape[0] or bool(torch.any(mu < 0.0)):
        raise ValueError("diffusive CFL must be nonnegative and crop-aligned")
    mu = mu[:, None, None, None]
    penalty = float(divergence_penalty)

    def operator(value: Tensor) -> Tensor:
        mass = value * fluid_work
        diffusion = _gradient_energy_operator(value, fluid_work)
        divergence = _backward_divergence_index(value) * divergence_weight
        grad_div = _divergence_adjoint_index(
            divergence, value.shape[-2:],
        )
        return mass + mu * diffusion + penalty * grad_div

    # Write u = prescribed + free*x and solve the free equations.  Crops with
    # no active artificial face have an exactly zero right-hand side and are
    # kept bitwise at zero.
    right_hand_side = -operator(prescribed) * free_vector
    rhs_norm = _batch_inner(right_hand_side, right_hand_side).sqrt()
    solution = torch.zeros_like(boundary)
    residual = right_hand_side.clone()
    direction = residual.clone()
    residual_square = _batch_inner(residual, residual)
    threshold = tolerance * rhs_norm.clamp_min(1.0e-30)
    live = rhs_norm > 0.0
    completed = 0
    for index in range(iterations):
        applied = operator(direction * free_vector) * free_vector
        denominator = _batch_inner(direction, applied)
        safe = live & (denominator > 1.0e-30)
        alpha = torch.where(
            safe,
            residual_square / denominator.clamp_min(1.0e-30),
            torch.zeros_like(denominator),
        )
        solution = solution + alpha[:, None, None, None] * direction
        residual = residual - alpha[:, None, None, None] * applied
        next_square = _batch_inner(residual, residual)
        completed = index + 1
        next_live = live & (next_square.sqrt() > threshold)
        if not bool(torch.any(next_live)):
            residual_square = next_square
            break
        beta = torch.where(
            next_live,
            next_square / residual_square.clamp_min(1.0e-30),
            torch.zeros_like(next_square),
        )
        direction = residual + beta[:, None, None, None] * direction
        residual_square = next_square
        live = next_live

    velocity = (prescribed + solution * free_vector) * fluid_work
    relative = residual_square.sqrt() / rhs_norm.clamp_min(1.0e-30)
    relative = torch.where(rhs_norm > 0.0, relative, torch.zeros_like(relative))
    divergence = _backward_divergence_index(velocity) * divergence_weight
    return velocity.to(boundary_velocity.dtype), divergence, relative


def augmented_modified_stokes_response(
    boundary_defect: Tensor,
    face_roles: Tensor,
    fluid: Tensor,
    *,
    state_std: Tensor,
    dt: Tensor | float,
    dx: float,
    viscosity: Tensor | float,
    divergence_penalty: float = 100.0,
    iterations: int = 160,
    tolerance: float = 1.0e-7,
    pressure_gain: float = 1.0,
) -> tuple[Tensor, ModifiedStokesResponseDiagnostics]:
    """Return standardized ``(du,dv,dp)`` from a velocity face defect."""
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be positive")
    if not math.isfinite(pressure_gain):
        raise ValueError("pressure gain must be finite")
    normalized_boundary, active = _velocity_face_values(
        boundary_defect, face_roles,
    )
    std = torch.as_tensor(
        state_std, device=boundary_defect.device, dtype=boundary_defect.dtype,
    ).flatten()
    if std.numel() != 3 or bool(torch.any(std <= 0.0)):
        raise ValueError("state_std must contain three positive values")
    dt_tensor = torch.as_tensor(
        dt, device=boundary_defect.device, dtype=boundary_defect.dtype,
    ).flatten()
    viscosity_tensor = torch.as_tensor(
        viscosity, device=boundary_defect.device, dtype=boundary_defect.dtype,
    ).flatten()
    batch = boundary_defect.shape[0]
    if dt_tensor.numel() == 1:
        dt_tensor = dt_tensor.expand(batch)
    if viscosity_tensor.numel() == 1:
        viscosity_tensor = viscosity_tensor.expand(batch)
    if dt_tensor.numel() != batch or viscosity_tensor.numel() != batch:
        raise ValueError("dt and viscosity must be scalar or crop-aligned")
    if bool(torch.any(dt_tensor <= 0.0)) or bool(torch.any(viscosity_tensor < 0.0)):
        raise ValueError("dt must be positive and viscosity nonnegative")

    physical_boundary = normalized_boundary * std[:2].reshape(1, 2, 1, 1)
    diffusive_cfl = viscosity_tensor * dt_tensor / (dx * dx)
    velocity, divergence_index, residual = (
        augmented_modified_stokes_velocity_response(
            physical_boundary, active, fluid,
            diffusive_cfl=diffusive_cfl,
            divergence_penalty=divergence_penalty,
            iterations=iterations,
            tolerance=tolerance,
        )
    )
    # In the augmented formulation p = -lambda div(u), with
    # lambda = penalty*dx**2/dt and div(u)=div_index/dx.
    pressure_corner = (
        -float(divergence_penalty) * dx
        / dt_tensor[:, None, None].to(divergence_index)
    ) * divergence_index
    pressure = velocity.new_zeros(batch, 1, *velocity.shape[-2:])
    pressure[:, 0, 1:, 1:] = pressure_corner.to(velocity)
    pressure[:, 0, 0, 1:] = pressure[:, 0, 1, 1:]
    pressure[:, 0, 1:, 0] = pressure[:, 0, 1:, 1]
    pressure[:, 0, 0, 0] = pressure[:, 0, 1, 1]
    fluid_work = (fluid > 0.5).to(pressure)
    pressure_mean = (
        (pressure * fluid_work).flatten(1).sum(1)
        / fluid_work.flatten(1).sum(1).clamp_min(1.0)
    )
    pressure = (
        pressure - pressure_mean[:, None, None, None]
    ) * fluid_work * float(pressure_gain)
    standardized = torch.cat((
        velocity / std[:2].reshape(1, 2, 1, 1),
        pressure / std[2].reshape(1, 1, 1, 1),
    ), dim=1)
    divergence_rms = (
        divergence_index.square().flatten(1).mean(1).sqrt() / dx
    )
    return standardized, ModifiedStokesResponseDiagnostics(
        relative_linear_residual=residual,
        physical_divergence_rms=divergence_rms,
        iterations=iterations,
    )


def semi_lagrangian_transport_velocity_response(
    response: Tensor,
    boundary_defect: Tensor,
    face_roles: Tensor,
    fluid: Tensor,
    base_velocity: Tensor,
    *,
    state_mean: Tensor,
    state_std: Tensor,
    dt: Tensor | float,
    dx: float,
    transport_gain: float,
) -> Tensor:
    """Apply a target-free lagged-advection split to a Stokes response.

    The modified-Stokes chart captures the instantaneous elliptic response.
    This optional split backtraces that response through the known local
    velocity at time ``n``.  It is a first-order semi-Lagrangian approximation
    to the lagged Oseen term, not a claim of equivalence to incflo's EB
    Godunov discretization.  Artificial-face values are reimposed exactly.

    ``transport_gain=0`` is a bitwise identity, preserving every historical
    checkpoint and Local/Core result.
    """
    if not math.isfinite(transport_gain) or transport_gain < 0.0:
        raise ValueError("transport gain must be finite and nonnegative")
    if transport_gain == 0.0:
        return response
    if response.ndim != 4 or response.shape[1] != 3:
        raise ValueError("response must have shape B,3,H,W")
    if base_velocity.shape != response[:, :2].shape:
        raise ValueError("base velocity must have shape B,2,H,W")
    if fluid.shape != response[:, :1].shape:
        raise ValueError("fluid mask must have shape B,1,H,W")
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx must be positive")
    mean = torch.as_tensor(
        state_mean, device=response.device, dtype=response.dtype,
    ).flatten()
    std = torch.as_tensor(
        state_std, device=response.device, dtype=response.dtype,
    ).flatten()
    if mean.numel() != 3 or std.numel() != 3 or bool(torch.any(std <= 0.0)):
        raise ValueError("state affine coordinates require three channels")
    dt_tensor = torch.as_tensor(
        dt, device=response.device, dtype=response.dtype,
    ).flatten()
    if dt_tensor.numel() == 1:
        dt_tensor = dt_tensor.expand(response.shape[0])
    if dt_tensor.numel() != response.shape[0] or bool(torch.any(dt_tensor <= 0.0)):
        raise ValueError("dt must be positive and crop-aligned")

    physical_base = (
        base_velocity * std[:2].reshape(1, 2, 1, 1)
        + mean[:2].reshape(1, 2, 1, 1)
    )
    height, width = response.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=response.device, dtype=response.dtype),
        torch.linspace(-1.0, 1.0, width, device=response.device, dtype=response.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1)[None].expand(
        response.shape[0], -1, -1, -1,
    ).clone()
    cell_shift = (
        float(transport_gain) * dt_tensor[:, None, None, None]
        * physical_base / float(dx)
    )
    grid[..., 0] -= 2.0 * cell_shift[:, 0] / max(width - 1, 1)
    grid[..., 1] -= 2.0 * cell_shift[:, 1] / max(height - 1, 1)
    velocity = F.grid_sample(
        response[:, :2], grid, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    )
    fluid_work = (fluid > 0.5).to(response)
    edge = _edge_mask(fluid_work).to(response)
    fixed = edge * fluid_work
    normalized_boundary, active = _velocity_face_values(
        boundary_defect, face_roles,
    )
    prescribed = normalized_boundary * active.to(response) * fixed
    velocity = velocity * fluid_work * (1.0 - fixed) + prescribed
    return torch.cat((velocity, response[:, 2:3] * fluid_work), dim=1)


class PointParticularWithModifiedStokesResponse2D(nn.Module):
    """Use a frozen local Point endpoint only as the particular solution."""

    def __init__(
        self,
        particular: nn.Module,
        *,
        state_std: Tensor,
        state_mean: Tensor | None = None,
        dx: float,
        divergence_penalty: float = 100.0,
        iterations: int = 160,
        tolerance: float = 1.0e-7,
        response_gain: float = 1.0,
        pressure_gain: float = 1.0,
        transport_gain: float = 0.0,
        response_domain: str = RESPONSE_DOMAIN_SDF_FLUID,
    ) -> None:
        super().__init__()
        if not math.isfinite(response_gain):
            raise ValueError("response gain must be finite")
        self.particular = particular
        self.register_buffer(
            "state_std", torch.as_tensor(state_std, dtype=torch.float32).flatten(),
        )
        self.register_buffer(
            "state_mean",
            torch.zeros(3, dtype=torch.float32) if state_mean is None
            else torch.as_tensor(state_mean, dtype=torch.float32).flatten(),
        )
        self.dx = float(dx)
        self.divergence_penalty = float(divergence_penalty)
        self.iterations = int(iterations)
        self.tolerance = float(tolerance)
        self.response_gain = float(response_gain)
        self.pressure_gain = float(pressure_gain)
        if not math.isfinite(transport_gain) or transport_gain < 0.0:
            raise ValueError("transport gain must be finite and nonnegative")
        self.transport_gain = float(transport_gain)
        if response_domain not in RESPONSE_DOMAINS:
            raise ValueError(f"unknown homogeneous response domain: {response_domain}")
        self.response_domain = str(response_domain)

    @property
    def residual_scale(self) -> Tensor:
        return self.particular.residual_scale

    @property
    def physical_channels(self) -> int:
        return int(self.particular.physical_channels)

    @property
    def scalar_dim(self) -> int:
        return int(self.particular.scalar_dim)

    @property
    def projection_channels(self) -> int:
        return int(self.particular.projection_channels)

    @property
    def dirichlet_channels(self) -> int:
        # The response consumes two velocity components on every oriented face;
        # pressure remains owned by the particular/projection state.
        return 2

    @property
    def supports_cached_particular(self) -> bool:
        return True

    def compose_from_particular(
        self,
        particular: Tensor,
        physical: Tensor,
        scalar: Tensor,
        boundary_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        """Apply only the cheap homogeneous response to a cached local call."""
        if particular.ndim != 4 or particular.shape[0] != physical.shape[0]:
            raise ValueError("cached particular is not crop-aligned")
        if not bool(torch.any(boundary_defect != 0.0)):
            return particular
        dt = scalar[:, 1] * 0.01
        reynolds = torch.exp(scalar[:, 0]) * 10_000.0
        viscosity = reynolds.reciprocal()
        solve_domain = response_domain_mask(physical, self.response_domain)
        response, _ = augmented_modified_stokes_response(
            boundary_defect, face_roles, solve_domain,
            state_std=self.state_std, dt=dt, dx=self.dx,
            viscosity=viscosity,
            divergence_penalty=self.divergence_penalty,
            iterations=self.iterations, tolerance=self.tolerance,
            pressure_gain=self.pressure_gain,
        )
        response = semi_lagrangian_transport_velocity_response(
            response, boundary_defect, face_roles,
            solve_domain, physical[:, :2],
            state_mean=self.state_mean, state_std=self.state_std,
            dt=dt, dx=self.dx, transport_gain=self.transport_gain,
        )
        corrected_state = particular[:, :3] + self.response_gain * response
        if particular.shape[1] == 3:
            return corrected_state
        return torch.cat((corrected_state, particular[:, 3:]), dim=1)

    def forward(
        self,
        physical: Tensor,
        scalar: Tensor,
        boundary_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        zero = torch.zeros_like(boundary_defect)
        base = self.particular(physical, scalar, zero, face_roles)
        return self.compose_from_particular(
            base, physical, scalar, boundary_defect, face_roles,
        )
