"""Zero-message-preserving boundary-input response for a local Point solve.

The wrapper interprets the communicated velocity defect as a correction to
the artificial-boundary values in the local time-n state and reruns the same
frozen local Point/AB-UPT operator.  At zero defect it is exactly the original
Local/Core function.  No tile index or global coordinate is introduced.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .native_incflo_response import PointParticularWithIncfloResponse2D


Tensor = torch.Tensor
METADATA_KIND = "point_boundary_input_rerun_v1"
METADATA_MODES = ("direct", "attached")


def validate_point_boundary_input_metadata(
    value: object,
) -> dict[str, object] | None:
    """Validate the immutable checkpoint declaration for this response.

    Returning ``None`` keeps historical checkpoints unchanged.  A present but
    malformed declaration fails closed rather than silently rebuilding a
    different local family.
    """
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("enabled") is not True:
        raise ValueError("invalid Point boundary-input response metadata")
    if value.get("kind") != METADATA_KIND:
        raise ValueError("unknown Point boundary-input response kind")
    mode = str(value.get("mode", ""))
    if mode not in METADATA_MODES:
        raise ValueError("invalid Point boundary-input response mode")
    gain = float(value.get("gain", float("nan")))
    collar_width = int(value.get("collar_width", 0))
    relaxation = float(value.get("relaxation", float("nan")))
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError("invalid Point boundary-input response gain")
    if collar_width <= 0:
        raise ValueError("invalid Point boundary-input collar width")
    if not math.isfinite(relaxation) or not 0.0 < relaxation <= 1.5:
        raise ValueError("invalid Point boundary-input SNI relaxation")
    required_false = (
        "target_used", "learned_weights_changed", "global_coordinates_used",
    )
    if any(value.get(name) is not False for name in required_false):
        raise ValueError("boundary-input response has invalid provenance")
    if value.get("zero_message_equals_local_core") is not True:
        raise ValueError("boundary-input response does not preserve Local/Core")
    return value


def velocity_defect_to_local_input_2d(
    defect: Tensor,
    face_roles: Tensor,
    *,
    collar_width: int = 1,
) -> Tensor:
    """Extend four face-wise velocity defects into a local input collar."""
    if defect.ndim != 4 or defect.shape[1] != 8:
        raise ValueError("velocity defect must have four two-channel blocks")
    if face_roles.shape != (defect.shape[0], 4, *defect.shape[-2:]):
        raise ValueError("face roles are incompatible with velocity defect")
    height, width = defect.shape[-2:]
    if collar_width <= 0 or 2 * collar_width > min(height, width):
        raise ValueError("invalid input-correction collar width")

    values = defect.reshape(defect.shape[0], 4, 2, height, width)
    correction = defect.new_zeros(defect.shape[0], 2, height, width)
    count = defect.new_zeros(defect.shape[0], 1, height, width)

    north_value = values[:, 0, :, :1, :].expand(-1, -1, collar_width, -1)
    north_role = face_roles[:, 0:1, :1, :].expand(
        -1, -1, collar_width, -1,
    )
    correction[:, :, :collar_width, :] += north_value * north_role
    count[:, :, :collar_width, :] += north_role

    south_value = values[:, 1, :, -1:, :].expand(-1, -1, collar_width, -1)
    south_role = face_roles[:, 1:2, -1:, :].expand(
        -1, -1, collar_width, -1,
    )
    correction[:, :, -collar_width:, :] += south_value * south_role
    count[:, :, -collar_width:, :] += south_role

    west_value = values[:, 2, :, :, :1].expand(-1, -1, -1, collar_width)
    west_role = face_roles[:, 2:3, :, :1].expand(
        -1, -1, -1, collar_width,
    )
    correction[:, :, :, :collar_width] += west_value * west_role
    count[:, :, :, :collar_width] += west_role

    east_value = values[:, 3, :, :, -1:].expand(-1, -1, -1, collar_width)
    east_role = face_roles[:, 3:4, :, -1:].expand(
        -1, -1, -1, collar_width,
    )
    correction[:, :, :, -collar_width:] += east_value * east_role
    count[:, :, :, -collar_width:] += east_role

    return correction / count.clamp_min(1.0)


class PointBoundaryInputResponse2D(nn.Module):
    """Rerun one frozen local Point unit after a boundary-input correction."""

    reruns_particular_per_response = True

    def __init__(
        self,
        particular: nn.Module,
        *,
        gain: float = 1.0,
        collar_width: int = 1,
    ) -> None:
        super().__init__()
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("boundary-input gain must be finite and nonnegative")
        if collar_width <= 0:
            raise ValueError("boundary-input collar width must be positive")
        if int(getattr(particular, "robin_channels", -1)) != 8:
            raise ValueError("boundary-input response requires velocity ports")
        self.particular = particular
        self.gain = float(gain)
        self.collar_width = int(collar_width)
        self.physical_channels = int(particular.physical_channels)
        self.scalar_dim = int(particular.scalar_dim)
        self.robin_channels = 8
        self.projection_channels = int(particular.projection_channels)
        self.postsolve_strong_trace = True
        for parameter in self.particular.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        physical: Tensor,
        scalar: Tensor,
        boundary_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        if physical.ndim != 4 or physical.shape[1] != self.physical_channels:
            raise ValueError("physical condition has the wrong shape")
        correction = velocity_defect_to_local_input_2d(
            boundary_defect, face_roles, collar_width=self.collar_width,
        )
        corrected = physical.clone()
        corrected[:, :2] = corrected[:, :2] + self.gain * correction
        zero_port = torch.zeros_like(boundary_defect)
        return self.particular(corrected, scalar, zero_port, face_roles)

    def architecture_signature(self):
        return self.particular.architecture_signature()


class PointBoundaryInputWithIncfloResponse2D(
    PointParticularWithIncfloResponse2D,
):
    """Combine a corrected-input Point rerun with the source-audited response."""

    reruns_particular_per_response = True

    def __init__(
        self,
        source: PointParticularWithIncfloResponse2D,
        *,
        gain: float = 1.0,
        collar_width: int = 1,
    ) -> None:
        if not isinstance(source, PointParticularWithIncfloResponse2D):
            raise TypeError("combined input response needs an incflo source wrapper")
        super().__init__(
            source.particular,
            state_std=source.state_std.detach().clone(),
            projection_std=source.projection_std.detach().clone(),
            dx=source.dx,
            helmholtz_iterations=source.helmholtz_iterations,
            projection_iterations=source.projection_iterations,
            tolerance=source.tolerance,
            trace_reference_kind=source.trace_reference_kind,
            response_output_kind=source.response_output_kind,
        )
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("boundary-input gain must be finite and nonnegative")
        if collar_width <= 0:
            raise ValueError("boundary-input collar width must be positive")
        self.gain = float(gain)
        self.collar_width = int(collar_width)
        for parameter in self.particular.parameters():
            parameter.requires_grad_(False)

    def compose_from_particular(
        self,
        particular: Tensor,
        physical: Tensor,
        scalar: Tensor,
        boundary_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        if particular.shape != (
            physical.shape[0], 5, *physical.shape[-2:]
        ):
            raise ValueError("incflo particular must contain five local states")
        if not bool(torch.any(boundary_defect)):
            return particular
        correction = velocity_defect_to_local_input_2d(
            boundary_defect, face_roles, collar_width=self.collar_width,
        )
        corrected = physical.clone()
        corrected[:, :2] = corrected[:, :2] + self.gain * correction
        zero = torch.zeros_like(boundary_defect)
        rerun = self.particular(corrected, scalar, zero, face_roles)
        # Writing this as a centered update makes the exact zero-message
        # invariant explicit even when a cached particular is supplied.
        input_response = rerun - particular
        return particular + input_response + self.homogeneous_response(
            physical, scalar, boundary_defect, face_roles,
        )


class PointBoundaryInputWithAttachedResponse2D(nn.Module):
    """Combine an input rerun with an existing deterministic response chart."""

    supports_cached_particular = True
    reruns_particular_per_response = True

    def __init__(
        self,
        source: nn.Module,
        *,
        gain: float = 1.0,
        collar_width: int = 1,
    ) -> None:
        super().__init__()
        if not hasattr(source, "particular") or not callable(
            getattr(source, "compose_from_particular", None)
        ):
            raise TypeError("attached response must expose a Point particular chart")
        if callable(getattr(source, "native_sni_sweep", None)):
            raise ValueError(
                "custom-sweep responses require their dedicated combined wrapper"
            )
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("boundary-input gain must be finite and nonnegative")
        if collar_width <= 0:
            raise ValueError("boundary-input collar width must be positive")
        self.source = source
        self.gain = float(gain)
        self.collar_width = int(collar_width)
        self.postsolve_strong_trace = bool(
            getattr(source, "postsolve_strong_trace", True)
        )
        for parameter in self.source.parameters():
            parameter.requires_grad_(False)

    @property
    def particular(self) -> nn.Module:
        return self.source.particular

    @property
    def physical_channels(self) -> int:
        return int(self.source.physical_channels)

    @property
    def scalar_dim(self) -> int:
        return int(self.source.scalar_dim)

    @property
    def projection_channels(self) -> int:
        return int(self.source.projection_channels)

    @property
    def residual_scale(self) -> Tensor:
        return self.source.residual_scale

    def compose_from_particular(
        self,
        particular: Tensor,
        physical: Tensor,
        scalar: Tensor,
        boundary_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        # Sweep zero is exactly the cached Local/Core evaluation.  Avoid a
        # redundant heavy Point rerun in this structurally zero-message call;
        # every later nonzero response is intentionally rerun on corrected
        # physical input and is counted as a heavy local-operator evaluation.
        if not bool(torch.any(boundary_defect)):
            return self.source.compose_from_particular(
                particular, physical, scalar, boundary_defect, face_roles,
            )
        correction = velocity_defect_to_local_input_2d(
            boundary_defect, face_roles, collar_width=self.collar_width,
        )
        corrected = physical.clone()
        corrected[:, :2] = corrected[:, :2] + self.gain * correction
        zero = torch.zeros_like(boundary_defect)
        rerun = self.particular(corrected, scalar, zero, face_roles)
        return self.source.compose_from_particular(
            rerun, physical, scalar, boundary_defect, face_roles,
        )

    def forward(
        self,
        physical: Tensor,
        scalar: Tensor,
        boundary_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        zero = torch.zeros_like(boundary_defect)
        particular = self.particular(physical, scalar, zero, face_roles)
        return self.compose_from_particular(
            particular, physical, scalar, boundary_defect, face_roles,
        )
