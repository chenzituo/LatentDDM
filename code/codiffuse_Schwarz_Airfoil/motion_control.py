"""Analytic prescribed airfoil motion and geometry control."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class CyclicAirfoilControl:
    amplitude: float
    period: float
    startup_time: float
    pitch_start_time: float
    rotation_sign: float
    thickness: float
    pivot_fraction: float

    @classmethod
    def from_parameters(
        cls, parameters: dict[str, object], pitch_start_time: float
    ) -> "CyclicAirfoilControl":
        return cls(
            amplitude=math.radians(float(parameters["cycle_amplitude_deg"])),
            period=float(parameters["cycle_period"]),
            startup_time=float(parameters["t0"]),
            pitch_start_time=pitch_start_time,
            rotation_sign=float(parameters.get("rotation_sign", 1.0)),
            thickness=float(parameters["thickness"]),
            pivot_fraction=float(parameters["pivot_fraction"]),
        )

    def evaluate(self, time: Tensor) -> Tensor:
        tau = (time - self.pitch_start_time).clamp_min(0.0)
        decay_rate = 4.6 / self.startup_time
        decay = torch.exp(-decay_rate * tau)
        envelope = 1.0 - decay
        envelope_rate = decay_rate * decay
        envelope_acceleration = -(decay_rate**2) * decay
        phase_rate = 2.0 * math.pi / self.period
        phase = phase_rate * tau
        one_minus_cosine = 1.0 - torch.cos(phase)
        angle = 0.5 * self.amplitude * envelope * one_minus_cosine
        omega = 0.5 * self.amplitude * (
            envelope_rate * one_minus_cosine
            + envelope * phase_rate * torch.sin(phase)
        )
        acceleration = 0.5 * self.amplitude * (
            envelope_acceleration * one_minus_cosine
            + 2.0 * envelope_rate * phase_rate * torch.sin(phase)
            + envelope * phase_rate**2 * torch.cos(phase)
        )
        controlled = torch.stack((angle, omega, acceleration, phase), dim=-1)
        controlled[..., :3] *= self.rotation_sign
        return controlled

    def between(self, time_n: Tensor, time_np1: Tensor, fraction: Tensor) -> Tensor:
        time = time_n + fraction * (time_np1 - time_n)
        return self.evaluate(time)


def signed_distance_proxy_torch(
    x: Tensor,
    y: Tensor,
    angle: Tensor,
    thickness: float,
    pivot_fraction: float,
) -> Tensor:
    """Torch equivalent of the airfoil dataset's analytic NACA distance proxy."""
    while angle.ndim < x.ndim:
        angle = angle.unsqueeze(-1)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    x_body = cosine * x + sine * y
    y_body = -sine * x + cosine * y
    xi = x_body + pivot_fraction
    xi_clipped = xi.clamp(0.0, 1.0)
    half = 5.0 * thickness * (
        0.2969 * torch.sqrt(xi_clipped)
        - 0.1260 * xi_clipped
        - 0.3516 * xi_clipped.square()
        + 0.2843 * xi_clipped.pow(3)
        - 0.1036 * xi_clipped.pow(4)
    )
    axial_outside = (xi - xi_clipped).abs()
    vertical = y_body.abs() - half
    outside = torch.sqrt(axial_outside.square() + vertical.clamp_min(0.0).square())
    inside = (xi >= 0.0) & (xi <= 1.0) & (vertical <= 0.0)
    endpoint_clearance = torch.minimum(xi, 1.0 - xi)
    interior_distance = torch.minimum(-vertical, endpoint_clearance)
    return torch.where(inside, -interior_distance, outside)
