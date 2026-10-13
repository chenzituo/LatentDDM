"""Boundary--geometry anchored response operator for local Airfoil BVPs.

The module deliberately predicts only the field response to an artificial-
interface Robin *defect*.  A separately pretrained local operator supplies the
base field.  Hence the response is exactly zero when communication is removed,
and response training cannot silently redefine Local/Core.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from src.neural_eos.abupt import DynamicABUPTOperator


Tensor = torch.Tensor


class BoundaryGeometryABUPTResponse2D(nn.Module):
    """Geometry-conditioned, positively homogeneous Robin response.

    Physical-geometry anchors are selected from the local signed-distance
    field, while the surface-anchor bank is reserved for active artificial
    interfaces.  The defect is normalized per local problem and the decoded
    response is multiplied by its original amplitude.  Consequently
    ``H(x, 0) = 0`` exactly and ``H(x, a b) = a H(x, b)`` for ``a > 0`` at
    fixed local context, matching the affine boundary response of one
    time-discrete modified-Stokes solve without hard-coding that PDE.
    """

    def __init__(
        self,
        *,
        physical_channels: int = 13,
        scalar_dim: int = 4,
        base_channels: int = 3,
        robin_channels: int = 12,
        output_channels: int = 3,
        d_model: int = 128,
        interface_anchors: int = 64,
        volume_anchors: int = 128,
        geometry_anchors: int = 64,
        depth: int = 3,
        decoder_depth: int = 2,
        heads: int = 4,
        response_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
        amplitude_floor: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if min(
            physical_channels, base_channels, robin_channels, output_channels,
            d_model, interface_anchors, volume_anchors,
            geometry_anchors, depth, decoder_depth, heads,
        ) <= 0 or scalar_dim < 0:
            raise ValueError("invalid boundary--geometry response dimensions")
        if robin_channels % 4:
            raise ValueError("Robin channels must form four oriented face blocks")
        if len(response_scale) != output_channels or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in response_scale
        ):
            raise ValueError("response scale must contain three positives")
        if not math.isfinite(amplitude_floor) or amplitude_floor <= 0.0:
            raise ValueError("amplitude floor must be positive and finite")
        self.physical_channels = int(physical_channels)
        self.scalar_dim = int(scalar_dim)
        self.base_channels = int(base_channels)
        self.robin_channels = int(robin_channels)
        self.output_channels = int(output_channels)
        self.channels_per_face = self.robin_channels // 4
        self.d_model = int(d_model)
        self.interface_anchors = int(interface_anchors)
        self.volume_anchors = int(volume_anchors)
        self.geometry_anchors = int(geometry_anchors)
        self.depth = int(depth)
        self.decoder_depth = int(decoder_depth)
        self.heads = int(heads)
        self.amplitude_floor = float(amplitude_floor)
        self.register_buffer(
            "response_scale",
            torch.tensor(response_scale, dtype=torch.float32).reshape(
                1, self.output_channels, 1, 1,
            ),
            persistent=False,
        )

        # local condition + frozen base prediction + normalized Robin defect
        # + four face roles + broadcast scalars + relative coordinates.
        in_channels = (
            self.physical_channels + self.base_channels
            + self.robin_channels + 4 + self.scalar_dim + 2
        )
        role_start = (
            self.physical_channels + self.base_channels + self.robin_channels
        )
        self.operator = DynamicABUPTOperator(
            in_channels=in_channels,
            d_model=self.d_model,
            surface_anchors=self.interface_anchors,
            volume_anchors=self.volume_anchors,
            geometry_anchors=self.geometry_anchors,
            depth=self.depth,
            decoder_depth=self.decoder_depth,
            heads=self.heads,
            out_channels=self.output_channels,
            # The endpoint geometry at channel four is the relevant local
            # body for q^{n+1}; this matches the capacity-matched endpoint.
            sdf_channel=4,
            coordinate_channels=(in_channels - 2, in_channels - 1),
            surface_mask_channels=tuple(range(role_start, role_start + 4)),
            surface_mask_fraction=1.0,
        )

    @staticmethod
    def _relative_coordinates(reference: Tensor) -> Tensor:
        height, width = reference.shape[-2:]
        y = torch.linspace(
            -1.0, 1.0, height, device=reference.device,
            dtype=reference.dtype,
        )
        x = torch.linspace(
            -1.0, 1.0, width, device=reference.device,
            dtype=reference.dtype,
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0)[None].expand(
            reference.shape[0], -1, -1, -1,
        )

    def forward(
        self,
        physical: Tensor,
        scalar: Tensor,
        base_prediction: Tensor,
        robin_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        """Return a correction to ``base_prediction`` in state coordinates."""
        if physical.ndim != 4 or physical.shape[1] != self.physical_channels:
            raise ValueError("physical condition has the wrong shape")
        shape = physical.shape[-2:]
        if base_prediction.shape != (
            physical.shape[0], self.base_channels, *shape,
        ):
            raise ValueError("base prediction has the wrong channel count")
        if robin_defect.shape != (
            physical.shape[0], self.robin_channels, *shape,
        ):
            raise ValueError("Robin defect has the wrong channel count")
        if face_roles.shape != (physical.shape[0], 4, *shape):
            raise ValueError("face roles have the wrong shape")
        if scalar.shape != (physical.shape[0], self.scalar_dim):
            raise ValueError("scalar condition has the wrong shape")

        active = face_roles.repeat_interleave(self.channels_per_face, dim=1)
        amplitude = self.defect_amplitude(robin_defect, face_roles)
        normalized_defect = robin_defect / amplitude.clamp_min(
            self.amplitude_floor,
        )
        normalized_defect = normalized_defect * active
        scalar_field = scalar[:, :, None, None].expand(-1, -1, *shape)
        condition = torch.cat((
            physical,
            base_prediction,
            normalized_defect,
            face_roles,
            scalar_field,
            self._relative_coordinates(physical),
        ), dim=1)
        unit_response = self.operator(condition)
        valid = physical[:, -1:]
        return (
            unit_response
            * amplitude
            * self.response_scale.to(unit_response)
            * valid
        )

    def defect_amplitude(self, robin_defect: Tensor, face_roles: Tensor) -> Tensor:
        """Return per-problem RMS over active artificial-boundary entries."""
        if robin_defect.ndim != 4 or robin_defect.shape[1] != self.robin_channels:
            raise ValueError("Robin defect has the wrong channel count")
        if face_roles.shape != (
            robin_defect.shape[0], 4, *robin_defect.shape[-2:],
        ):
            raise ValueError("face roles are incompatible with Robin defect")
        active = face_roles.repeat_interleave(self.channels_per_face, dim=1)
        squared_amplitude = (
            robin_defect.square() * active
        ).sum(dim=(1, 2, 3), keepdim=True) / active.sum(
            dim=(1, 2, 3), keepdim=True,
        ).clamp_min(1.0)
        return squared_amplitude.sqrt()

    def composed_field(
        self,
        physical: Tensor,
        scalar: Tensor,
        base_prediction: Tensor,
        robin_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        """Return the frozen local prediction plus its communicated response."""
        return base_prediction + self(
            physical, scalar, base_prediction, robin_defect, face_roles,
        )

    def architecture_signature(self) -> dict[str, int | str]:
        return {
            "variant": "boundary_geometry_abupt_response_v1",
            "d_model": self.d_model,
            "interface_anchors": self.interface_anchors,
            "volume_anchors": self.volume_anchors,
            "geometry_anchors": self.geometry_anchors,
            "depth": self.depth,
            "decoder_depth": self.decoder_depth,
            "heads": self.heads,
            "base_channels": self.base_channels,
            "robin_channels": self.robin_channels,
            "output_channels": self.output_channels,
        }


class _ContextLinearTransport(nn.Module):
    """Attention whose weights depend on context and values on response.

    For fixed query/anchor context this block is a linear map of
    ``anchor_response``.  No normalization, bias, or activation is applied to
    the response path.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("linear transport width must divide its heads")
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.query_norm = nn.LayerNorm(dim)
        self.anchor_norm = nn.LayerNorm(dim)
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim, bias=False)
        self.output_projection = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        query_context: Tensor,
        anchor_context: Tensor,
        anchor_response: Tensor,
    ) -> Tensor:
        if anchor_context.shape != anchor_response.shape:
            raise ValueError("anchor context and response must share a shape")
        batch = query_context.shape[0]
        query = self.query_projection(
            self.query_norm(query_context),
        ).reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        key = self.key_projection(
            self.anchor_norm(anchor_context),
        ).reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        value = self.value_projection(anchor_response).reshape(
            batch, -1, self.heads, self.head_dim,
        ).transpose(1, 2)
        transported = F.scaled_dot_product_attention(
            query, key, value,
        ).transpose(1, 2).reshape(batch, query_context.shape[1], -1)
        return self.output_projection(transported)


class LinearBoundaryGeometryABUPTResponse2D(nn.Module):
    """A geometry-conditioned response exactly linear in Robin defects.

    Geometry and the frozen local prediction determine attention queries and
    keys.  Sparse Robin values are transported from artificial-boundary
    anchors through volume anchors to every query point.  Because the defect
    appears only on bias-free value paths, the map is additive and homogeneous
    for every real scalar at fixed local context.
    """

    def __init__(
        self,
        *,
        physical_channels: int = 13,
        scalar_dim: int = 4,
        base_channels: int = 3,
        robin_channels: int = 12,
        output_channels: int = 3,
        d_model: int = 128,
        interface_anchors: int = 64,
        volume_anchors: int = 128,
        geometry_anchors: int = 64,
        depth: int = 3,
        decoder_depth: int = 2,
        heads: int = 4,
        geometry_context: bool = False,
        response_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
        amplitude_floor: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if min(
            physical_channels, base_channels, robin_channels, output_channels,
            d_model, interface_anchors, volume_anchors,
            geometry_anchors, depth, decoder_depth, heads,
        ) <= 0 or scalar_dim < 0 or d_model % heads:
            raise ValueError("invalid linear BG-ABUPT dimensions")
        if robin_channels % 4:
            raise ValueError("Robin channels must form four oriented face blocks")
        if len(response_scale) != output_channels or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in response_scale
        ):
            raise ValueError("response scale must contain three positives")
        self.physical_channels = int(physical_channels)
        self.scalar_dim = int(scalar_dim)
        self.base_channels = int(base_channels)
        self.robin_channels = int(robin_channels)
        self.output_channels = int(output_channels)
        self.channels_per_face = self.robin_channels // 4
        self.d_model = int(d_model)
        self.interface_anchors = int(interface_anchors)
        self.volume_anchors = int(volume_anchors)
        self.geometry_anchors = int(geometry_anchors)
        self.depth = int(depth)
        self.decoder_depth = int(decoder_depth)
        self.heads = int(heads)
        self.geometry_context = bool(geometry_context)
        self.amplitude_floor = float(amplitude_floor)
        self.register_buffer(
            "response_scale",
            torch.tensor(response_scale, dtype=torch.float32).reshape(
                1, self.output_channels, 1, 1,
            ),
            persistent=False,
        )
        context_channels = (
            self.physical_channels + self.base_channels
            + 4 + self.scalar_dim + 2
        )
        self.context_embedding = nn.Sequential(
            nn.Linear(context_channels, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.geometry_embedding = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.position_embedding = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.geometry_self = nn.MultiheadAttention(
            d_model, heads, batch_first=True,
        )
        self.context_from_geometry = nn.MultiheadAttention(
            d_model, heads, batch_first=True,
        )
        self.boundary_value = nn.Linear(self.robin_channels, d_model, bias=False)
        self.volume_from_boundary = nn.ModuleList([
            _ContextLinearTransport(d_model, heads)
            for _ in range(depth)
        ])
        self.boundary_from_volume = nn.ModuleList([
            _ContextLinearTransport(d_model, heads)
            for _ in range(depth)
        ])
        self.decoder = nn.ModuleList([
            _ContextLinearTransport(d_model, heads)
            for _ in range(decoder_depth)
        ])
        self.output_gate = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Sigmoid(),
        )
        self.output = nn.Linear(d_model, self.output_channels, bias=False)
        nn.init.normal_(self.output.weight, std=0.01)

    @staticmethod
    def _relative_coordinates(reference: Tensor) -> Tensor:
        return BoundaryGeometryABUPTResponse2D._relative_coordinates(reference)

    def defect_amplitude(self, robin_defect: Tensor, face_roles: Tensor) -> Tensor:
        if robin_defect.ndim != 4 or robin_defect.shape[1] != self.robin_channels:
            raise ValueError("Robin defect has the wrong channel count")
        if face_roles.shape != (
            robin_defect.shape[0], 4, *robin_defect.shape[-2:],
        ):
            raise ValueError("face roles are incompatible with Robin defect")
        active = face_roles.repeat_interleave(self.channels_per_face, dim=1)
        return (
            (robin_defect.square() * active).sum(
                dim=(1, 2, 3), keepdim=True,
            )
            / active.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
        ).sqrt()

    @staticmethod
    def _gather(tokens: Tensor, indices: Tensor) -> Tensor:
        return torch.gather(
            tokens, 1,
            indices[..., None].expand(-1, -1, tokens.shape[-1]),
        )

    @staticmethod
    def _volume_indices(
        height: int, width: int, count: int, device: torch.device,
    ) -> Tensor:
        count = min(count, height * width)
        rows = max(1, round((count * height / width) ** 0.5))
        columns = max(1, (count + rows - 1) // rows)
        y = torch.linspace(0, height - 1, rows, device=device).round().long()
        x = torch.linspace(0, width - 1, columns, device=device).round().long()
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        indices = (yy * width + xx).flatten()
        if indices.numel() < count:
            supplement = torch.linspace(
                0, height * width - 1, count, device=device,
            ).round().long()
            indices = torch.cat((indices, supplement))
        return indices[:count]

    def _indices(
        self, physical: Tensor, face_roles: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, _, height, width = physical.shape
        cells = height * width
        distance = physical[:, 4].abs().flatten(1)
        geometry_count = min(self.geometry_anchors, cells)
        geometry = torch.topk(
            distance, k=geometry_count, largest=False,
        ).indices
        has_surface = distance.amin(1) < 1.0 - 1.0e-6
        geometry_fallback = self._volume_indices(
            height, width, geometry_count, physical.device,
        )[None].expand(batch, -1)
        geometry = torch.where(
            has_surface[:, None], geometry, geometry_fallback,
        )

        interface_count = min(self.interface_anchors, cells)
        active = face_roles.amax(1).flatten(1) > 0.5
        all_indices = torch.arange(cells, device=physical.device)
        hash_value = (
            (all_indices * 1_103_515_245 + 12_345) % 2_147_483_647
        ).to(physical.dtype) / 2_147_483_647.0
        interface_score = torch.where(
            active, hash_value[None],
            physical.new_full((batch, cells), -1.0),
        )
        interface = torch.topk(
            interface_score, k=interface_count, dim=1,
        ).indices
        volume = self._volume_indices(
            height, width, self.volume_anchors, physical.device,
        )[None].expand(batch, -1)
        return geometry, interface, volume

    def forward(
        self,
        physical: Tensor,
        scalar: Tensor,
        base_prediction: Tensor,
        robin_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        if physical.ndim != 4 or physical.shape[1] != self.physical_channels:
            raise ValueError("physical condition has the wrong shape")
        batch, _, height, width = physical.shape
        shape = (height, width)
        if base_prediction.shape != (batch, self.base_channels, *shape):
            raise ValueError("base prediction has the wrong channel count")
        if robin_defect.shape != (batch, self.robin_channels, *shape):
            raise ValueError("Robin defect has the wrong channel count")
        if face_roles.shape != (batch, 4, *shape):
            raise ValueError("face roles have the wrong shape")
        if scalar.shape != (batch, self.scalar_dim):
            raise ValueError("scalar condition has the wrong shape")

        coordinates = self._relative_coordinates(physical)
        scalar_field = scalar[:, :, None, None].expand(-1, -1, *shape)
        context_field = torch.cat((
            physical, base_prediction, face_roles, scalar_field, coordinates,
        ), dim=1)
        context = self.context_embedding(
            context_field.flatten(2).transpose(1, 2),
        )
        positions = coordinates.flatten(2).transpose(1, 2)
        position_tokens = self.position_embedding(positions)
        context = context + position_tokens
        geometry_indices, interface_indices, volume_indices = self._indices(
            physical, face_roles,
        )
        geometry_position = self._gather(positions, geometry_indices)
        geometry_sdf = self._gather(
            physical[:, 4:5].flatten(2).transpose(1, 2),
            geometry_indices,
        )
        geometry = self.geometry_embedding(torch.cat((
            geometry_position, geometry_sdf,
        ), dim=-1)) + self._gather(position_tokens, geometry_indices)
        # A signed distance and position describe shape but not a moving-wall
        # BVP.  When enabled, body anchors also carry the strictly local state
        # and wall data gathered from the context field.  This remains
        # independent of the Robin defect, so the response map is still
        # exactly linear in its communicated boundary values.
        if self.geometry_context:
            geometry = geometry + self._gather(context, geometry_indices)
        geometry_update, _ = self.geometry_self(
            geometry, geometry, geometry, need_weights=False,
        )
        geometry = geometry + geometry_update
        context_update, _ = self.context_from_geometry(
            context, geometry, geometry, need_weights=False,
        )
        context = context + context_update
        interface_context = self._gather(context, interface_indices)
        volume_context = self._gather(context, volume_indices)
        boundary_values = self.boundary_value(self._gather(
            robin_defect.flatten(2).transpose(1, 2), interface_indices,
        ))
        volume_response = boundary_values.new_zeros(
            batch, volume_context.shape[1], self.d_model,
        )
        boundary_response = boundary_values
        for to_volume, to_boundary in zip(
            self.volume_from_boundary, self.boundary_from_volume,
        ):
            volume_response = volume_response + to_volume(
                volume_context, interface_context, boundary_response,
            )
            boundary_response = boundary_response + to_boundary(
                interface_context, volume_context, volume_response,
            )
        anchor_context = torch.cat((interface_context, volume_context), dim=1)
        anchor_response = torch.cat((boundary_response, volume_response), dim=1)
        query_response = boundary_values.new_zeros(
            batch, context.shape[1], self.d_model,
        )
        for decoder in self.decoder:
            query_response = query_response + decoder(
                context, anchor_context, anchor_response,
            )
        query_response = query_response * self.output_gate(context)
        correction = self.output(query_response).transpose(1, 2).reshape(
            batch, self.output_channels, height, width,
        )
        return correction * self.response_scale.to(correction) * physical[:, -1:]

    def composed_field(
        self,
        physical: Tensor,
        scalar: Tensor,
        base_prediction: Tensor,
        robin_defect: Tensor,
        face_roles: Tensor,
    ) -> Tensor:
        return base_prediction + self(
            physical, scalar, base_prediction, robin_defect, face_roles,
        )

    def architecture_signature(self) -> dict[str, int | str]:
        return {
            "variant": (
                "linear_boundary_geometry_context_abupt_response_v2"
                if self.geometry_context else
                "linear_boundary_geometry_abupt_response_v1"
            ),
            "d_model": self.d_model,
            "interface_anchors": self.interface_anchors,
            "volume_anchors": self.volume_anchors,
            "geometry_anchors": self.geometry_anchors,
            "depth": self.depth,
            "decoder_depth": self.decoder_depth,
            "heads": self.heads,
            "geometry_context": self.geometry_context,
            "base_channels": self.base_channels,
            "robin_channels": self.robin_channels,
            "output_channels": self.output_channels,
        }


def boundary_geometry_response_from_payload(
    payload: dict[str, Any],
    base_payload: dict[str, Any],
    device: torch.device | str,
) -> nn.Module:
    """Rebuild a saved boundary--geometry response with strict dimensions."""
    saved = SimpleNamespace(**dict(payload["args"]))
    response_class = (
        LinearBoundaryGeometryABUPTResponse2D
        if getattr(saved, "response_map", "homogeneous") == "linear"
        else BoundaryGeometryABUPTResponse2D
    )
    response = response_class(
        physical_channels=int(base_payload["physical_channel_count"]),
        scalar_dim=len(base_payload["scalar_channels"]),
        base_channels=int(getattr(saved, "base_channels", 3)),
        robin_channels=int(getattr(saved, "robin_channels", 12)),
        output_channels=int(getattr(saved, "output_channels", 3)),
        d_model=int(saved.d_model),
        interface_anchors=int(saved.interface_anchors),
        volume_anchors=int(saved.volume_anchors),
        geometry_anchors=int(saved.geometry_anchors),
        depth=int(saved.depth),
        decoder_depth=int(saved.decoder_depth),
        heads=int(saved.heads),
        **(
            {"geometry_context": bool(getattr(saved, "geometry_context", False))}
            if response_class is LinearBoundaryGeometryABUPTResponse2D else {}
        ),
        response_scale=tuple(
            1.0 for _ in range(int(getattr(saved, "output_channels", 3)))
        ),
    ).to(device)
    response.load_state_dict(payload["response"], strict=True)
    return response
