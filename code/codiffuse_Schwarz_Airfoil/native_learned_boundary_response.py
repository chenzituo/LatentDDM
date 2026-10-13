"""Identifiable learned boundary response around a frozen native Point unit.

The frozen Point/AB-UPT endpoint supplies the zero-message particular field.
Only a zero-at-zero boundary-response operator is trained.  This prevents a
whole-field objective from silently removing dependence on the communicated
trace while preserving Local/Core exactly.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .boundary_geometry_abupt import (
    BoundaryGeometryABUPTResponse2D,
    LinearBoundaryGeometryABUPTResponse2D,
)


Tensor = torch.Tensor
KIND = "native_learned_boundary_response_v1"


class PointParticularWithLearnedBoundaryResponse2D(nn.Module):
    """Compose a frozen five-state Point particular with learned ``H(x,g)``."""

    supports_cached_particular = True
    postsolve_strong_trace = True
    flux_compatible_velocity_trace = False

    def __init__(
        self,
        particular: nn.Module,
        response: nn.Module,
        *,
        fixed_response: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if int(getattr(particular, "projection_channels", -1)) != 2:
            raise ValueError("native learned response requires five-state Point output")
        if int(getattr(particular, "robin_channels", -1)) != 8:
            raise ValueError("native learned response requires velocity face ports")
        if int(getattr(response, "base_channels", -1)) != 5 \
                or int(getattr(response, "output_channels", -1)) != 5 \
                or int(getattr(response, "robin_channels", -1)) != 8:
            raise ValueError("learned response must implement an 8 -> 5 native chart")
        if int(getattr(response, "physical_channels", -1)) != int(
            particular.physical_channels
        ):
            raise ValueError("particular and response physical inputs disagree")
        if int(getattr(response, "scalar_dim", -1)) != int(particular.scalar_dim):
            raise ValueError("particular and response scalar inputs disagree")
        self.particular = particular
        self.response = response
        if fixed_response is not None and not callable(
            getattr(fixed_response, "compose_from_particular", None)
        ):
            raise ValueError("fixed response omits cached-particular composition")
        self.fixed_response = fixed_response

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
        return 2

    @property
    def dirichlet_channels(self) -> int:
        return 2

    def homogeneous_response(
        self,
        physical: Tensor,
        scalar: Tensor,
        particular: Tensor,
        boundary_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        if not bool(torch.any(boundary_defect != 0.0)):
            return particular.new_zeros(particular.shape)
        learned = self.response(
            physical, scalar, particular, boundary_defect, face_roles,
        )
        if self.fixed_response is None:
            return learned
        fixed = self.fixed_response.compose_from_particular(
            particular, physical, scalar, boundary_defect, face_roles,
        ) - particular
        return fixed + learned

    def compose_from_particular(
        self,
        particular: Tensor,
        physical: Tensor,
        scalar: Tensor,
        boundary_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        if particular.shape != (
            physical.shape[0], 5, *physical.shape[-2:],
        ):
            raise ValueError("cached native particular must contain five states")
        return particular + self.homogeneous_response(
            physical, scalar, particular, boundary_defect, face_roles,
        )

    def forward(
        self,
        physical: Tensor,
        scalar: Tensor,
        boundary_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        particular = self.particular(
            physical, scalar, torch.zeros_like(boundary_defect), face_roles,
        )
        return self.compose_from_particular(
            particular, physical, scalar, boundary_defect, face_roles,
        )


def build_native_learned_response(
    metadata: dict[str, Any], *, physical_channels: int, scalar_dim: int,
) -> nn.Module:
    """Construct the response declared by immutable checkpoint metadata."""
    if metadata.get("kind") != KIND or not bool(metadata.get("enabled", False)):
        raise ValueError("invalid native learned-response metadata")
    response_map = str(metadata.get("response_map", "linear"))
    response_class = (
        LinearBoundaryGeometryABUPTResponse2D
        if response_map == "linear" else
        BoundaryGeometryABUPTResponse2D
        if response_map == "positive_homogeneous" else None
    )
    if response_class is None:
        raise ValueError(f"unknown native response map: {response_map}")
    controls = dict(
        physical_channels=int(physical_channels),
        scalar_dim=int(scalar_dim),
        base_channels=5,
        robin_channels=8,
        output_channels=5,
        d_model=int(metadata["d_model"]),
        interface_anchors=int(metadata["interface_anchors"]),
        volume_anchors=int(metadata["volume_anchors"]),
        geometry_anchors=int(metadata["geometry_anchors"]),
        depth=int(metadata["depth"]),
        decoder_depth=int(metadata["decoder_depth"]),
        heads=int(metadata["heads"]),
        response_scale=(1.0,) * 5,
    )
    if response_class is LinearBoundaryGeometryABUPTResponse2D:
        controls["geometry_context"] = bool(
            metadata.get("geometry_context", True)
        )
    return response_class(**controls)


def learned_response_metadata(
    *, response_map: str, d_model: int, interface_anchors: int,
    volume_anchors: int, geometry_anchors: int, depth: int,
    decoder_depth: int, heads: int, geometry_context: bool,
    learned_parameter_count: int,
    residual_over_fixed_response: bool = False,
) -> dict[str, object]:
    if learned_parameter_count <= 0 or not all(
        math.isfinite(float(value)) and int(value) > 0
        for value in (
            d_model, interface_anchors, volume_anchors, geometry_anchors,
            depth, decoder_depth, heads,
        )
    ):
        raise ValueError("invalid native learned-response architecture")
    return {
        "kind": KIND,
        "enabled": True,
        "response_map": str(response_map),
        "d_model": int(d_model),
        "interface_anchors": int(interface_anchors),
        "volume_anchors": int(volume_anchors),
        "geometry_anchors": int(geometry_anchors),
        "depth": int(depth),
        "decoder_depth": int(decoder_depth),
        "heads": int(heads),
        "geometry_context": bool(geometry_context),
        "base_channels": 5,
        "robin_channels": 8,
        "output_channels": 5,
        "trace_reference_kind": "local_prediction",
        "zero_message_equals_local_core": True,
        "target_used_at_inference": False,
        "learned_parameter_count": int(learned_parameter_count),
        "residual_over_fixed_response": bool(residual_over_fixed_response),
    }
