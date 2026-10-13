"""Anchor-based Universal Physics Transformer components."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ABUPTScalarOperator(nn.Module):
    """Geometry-aware point-token encoder, anchor processor, and query decoder."""

    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        n_anchors: int = 96,
        depth: int = 4,
        heads: int = 4,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.anchors = nn.Parameter(torch.randn(n_anchors, d_model) * 0.02)
        self.context_embedding = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.query_embedding = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.pool = nn.MultiheadAttention(d_model, heads, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.processor = nn.TransformerEncoder(layer, num_layers=depth)
        self.decode = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels),
        )

    def forward(self, context: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        batch = context.shape[0]
        context_tokens = self.context_embedding(context)
        anchors = self.anchors[None].expand(batch, -1, -1)
        pooled, _ = self.pool(anchors, context_tokens, context_tokens, need_weights=False)
        latent = self.processor(anchors + pooled)
        query_tokens = self.query_embedding(query)
        decoded, _ = self.decode(query_tokens, latent, latent, need_weights=False)
        return self.output(query_tokens + decoded)


class _AnchorBlock(nn.Module):
    """Pre-norm attention from query tokens to a bounded anchor set."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % (4 * heads):
            raise ValueError("anchor attention requires head dimension divisible by four")
        self.heads = heads
        self.head_dim = dim // heads
        self.query_norm = nn.LayerNorm(dim)
        self.anchor_norm = nn.LayerNorm(dim)
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim),
        )

    def _rope(self, tokens: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        batch, _, count, _ = tokens.shape
        axis_dim = self.head_dim // 2
        frequency = torch.exp(
            -torch.log(torch.tensor(10_000.0, device=tokens.device))
            * torch.arange(0, axis_dim, 2, device=tokens.device) / axis_dim
        ).to(tokens.dtype)
        parts = []
        for axis in range(2):
            values = tokens[..., axis * axis_dim:(axis + 1) * axis_dim]
            phase = position[:, None, :, axis, None].to(tokens.dtype) * frequency
            cosine, sine = phase.cos(), phase.sin()
            even, odd = values[..., 0::2], values[..., 1::2]
            rotated = torch.stack((
                even * cosine - odd * sine,
                even * sine + odd * cosine,
            ), dim=-1).flatten(-2)
            parts.append(rotated)
        result = torch.cat(parts, dim=-1)
        return result.reshape(batch, self.heads, count, self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        anchors: torch.Tensor,
        query_position: torch.Tensor,
        anchor_position: torch.Tensor,
    ) -> torch.Tensor:
        normalized_query = self.query_norm(query)
        normalized_anchors = self.anchor_norm(anchors)
        batch = query.shape[0]
        projected_query = self.query_projection(normalized_query).reshape(
            batch, -1, self.heads, self.head_dim,
        ).transpose(1, 2)
        projected_key = self.key_projection(normalized_anchors).reshape(
            batch, -1, self.heads, self.head_dim,
        ).transpose(1, 2)
        projected_value = self.value_projection(normalized_anchors).reshape(
            batch, -1, self.heads, self.head_dim,
        ).transpose(1, 2)
        projected_query = self._rope(projected_query, query_position)
        projected_key = self._rope(projected_key, anchor_position)
        update = F.scaled_dot_product_attention(
            projected_query, projected_key, projected_value,
        ).transpose(1, 2).reshape(batch, query.shape[1], -1)
        update = self.output_projection(update)
        query = query + update
        return query + self.mlp(self.mlp_norm(query))


class _ContinuousPositionEmbedding(nn.Module):
    def __init__(self, dim: int, frequencies: int = 24) -> None:
        super().__init__()
        scales = torch.pi * 2.0 ** torch.linspace(0.0, 7.0, frequencies)
        self.register_buffer("scales", scales, persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(4 * frequencies, dim), nn.GELU(), nn.Linear(dim, dim),
        )

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        phase = position[..., None] * self.scales
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        return self.projection(embedding)


class DynamicABUPTOperator(nn.Module):
    """Two-branch AB-UPT adaptation for temporal Cartesian flow operators.

    Near-surface and volume tokens communicate through shared-weight anchor
    attention. A separate geometry encoding conditions both branches, while all
    Cartesian cells remain query locations for the volume decoder.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int = 320,
        surface_anchors: int = 256,
        volume_anchors: int = 512,
        geometry_anchors: int = 128,
        depth: int = 5,
        decoder_depth: int = 3,
        heads: int = 8,
        out_channels: int = 3,
        sdf_channel: int = 7,
        coordinate_channels: tuple[int, int] = (16, 17),
        surface_mask_channels: tuple[int, ...] | None = None,
        surface_mask_fraction: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        if min(surface_anchors, volume_anchors, geometry_anchors) <= 0:
            raise ValueError("anchor counts must be positive")
        if not 0.0 <= surface_mask_fraction <= 1.0:
            raise ValueError("surface-mask anchor fraction must lie in [0,1]")
        if surface_mask_fraction > 0.0 and not surface_mask_channels:
            raise ValueError(
                "positive surface-mask fraction requires mask channels"
            )
        self.surface_anchors = surface_anchors
        self.volume_anchors = volume_anchors
        self.geometry_anchors = geometry_anchors
        self.sdf_channel = sdf_channel
        self.coordinate_channels = coordinate_channels
        self.surface_mask_channels = (
            tuple(int(channel) for channel in surface_mask_channels)
            if surface_mask_channels is not None else ()
        )
        self.surface_mask_fraction = float(surface_mask_fraction)

        self.position_embedding = _ContinuousPositionEmbedding(d_model)
        self.geometry_embedding = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.surface_embedding = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.volume_embedding = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.geometry_block = _AnchorBlock(d_model, heads)
        self.geometry_perceiver = _AnchorBlock(d_model, heads)
        self.physics_blocks = nn.ModuleList([
            _AnchorBlock(d_model, heads) for _ in range(depth)
        ])
        self.decoder_blocks = nn.ModuleList([
            _AnchorBlock(d_model, heads) for _ in range(decoder_depth)
        ])
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_channels))
        self.apply(self._initialize_linear)
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    @staticmethod
    def _initialize_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _gather(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            tokens, 1, indices[..., None].expand(-1, -1, tokens.shape[-1]),
        )

    @staticmethod
    def _volume_indices(
        height: int, width: int, count: int, device: torch.device,
    ) -> torch.Tensor:
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

    def _surface_indices(
        self,
        distance: torch.Tensor,
        field: torch.Tensor,
        *,
        height: int,
        width: int,
        count: int,
    ) -> torch.Tensor:
        """Select physical- and artificial-boundary anchors without new weights.

        Whole-domain AB-UPT uses the cells nearest the physical SDF zero set.
        A strict-local BVP additionally carries data on artificial crop faces.
        When mask channels are configured, a fixed fraction of the *same*
        surface-anchor budget is drawn evenly from those faces and the rest is
        drawn from the physical surface.  Geometry anchors remain unchanged.
        """
        near_order = torch.topk(
            distance, k=count, largest=False,
        ).indices
        has_local_surface = distance.amin(dim=1) < 1.0 - 1.0e-6
        fallback = self._volume_indices(
            height, width, count, field.device,
        )
        if not self.surface_mask_channels or self.surface_mask_fraction <= 0.0:
            return torch.where(
                has_local_surface[:, None],
                near_order,
                fallback[None].expand(field.shape[0], -1),
            )

        if min(self.surface_mask_channels) < 0 or \
                max(self.surface_mask_channels) >= field.shape[1]:
            raise ValueError("surface-mask channel lies outside the input field")
        mask = field[:, self.surface_mask_channels].abs().amax(dim=1).flatten(1)
        interface_budget = min(
            count, max(1, round(count * self.surface_mask_fraction)),
        )
        all_indices = torch.arange(height * width, device=field.device)
        # Stable pseudo-random ranks spread selected cells around all active
        # faces.  This avoids both raster-corner concentration and Python/GPU
        # synchronization inside every local call.
        hash_value = (
            (all_indices * 1_103_515_245 + 12_345) % 2_147_483_647
        ).to(distance.dtype) / 2_147_483_647.0
        active = mask > 0.5
        active_score = torch.where(
            active, hash_value[None], mask.new_full(mask.shape, -1.0),
        )
        interface_order = torch.topk(
            active_score, k=interface_budget, dim=1,
        ).indices
        active_count = active.sum(dim=1).clamp_max(interface_budget)
        selected_values = (
            torch.arange(interface_budget, device=field.device)[None]
            < active_count[:, None]
        )
        selected_mask = torch.zeros_like(active)
        selected_mask.scatter_(1, interface_order, selected_values)

        distance_min = distance.amin(dim=1, keepdim=True)
        distance_range = (
            distance.amax(dim=1, keepdim=True) - distance_min
        ).clamp_min(1.0e-12)
        near_surface_score = 1.0 - (distance - distance_min) / distance_range
        fallback_mask = torch.zeros_like(active)
        fallback_mask[:, fallback] = True
        physical_score = torch.where(
            has_local_surface[:, None],
            near_surface_score,
            fallback_mask.to(distance.dtype),
        )
        # Selected artificial faces outrank physical candidates; the small
        # hash term only breaks otherwise exact ties deterministically.
        priority = physical_score + 1.0e-5 * hash_value[None]
        priority = torch.where(
            selected_mask, 2.0 + hash_value[None], priority,
        )
        return torch.topk(priority, k=count, dim=1).indices

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        if field.ndim != 4:
            raise ValueError("field must have shape (batch, channels, height, width)")
        batch, _, height, width = field.shape
        tokens = field.flatten(2).transpose(1, 2)
        position = field[:, self.coordinate_channels].flatten(2).transpose(1, 2)
        signed_sdf = field[:, self.sdf_channel].flatten(1)
        distance = signed_sdf.abs()

        surface_count = min(self.surface_anchors, tokens.shape[1])
        geometry_count = min(self.geometry_anchors, tokens.shape[1])
        near_order = torch.topk(
            distance, k=max(surface_count, geometry_count), largest=False,
        ).indices
        surface_indices = self._surface_indices(
            distance, field, height=height, width=width, count=surface_count,
        )
        geometry_indices = near_order[:, :geometry_count]
        # A strict-local crop may contain no body at all.  Its truncated SDF is
        # then identically +/-1, so ``topk`` ties would concentrate all
        # nominal surface/geometry anchors in an arbitrary raster corner.
        # Fall back to an even local chart in that case; no non-local geometry
        # is inferred, and full domains containing a surface are unchanged.
        has_local_surface = distance.amin(dim=1, keepdim=True) < 1.0 - 1.0e-6
        fallback_count = max(surface_count, geometry_count)
        fallback = self._volume_indices(
            height, width, fallback_count, field.device,
        )[None].expand(batch, -1)
        geometry_indices = torch.where(
            has_local_surface, geometry_indices, fallback[:, :geometry_count],
        )
        volume_indices = self._volume_indices(
            height, width, self.volume_anchors, field.device,
        )[None].expand(batch, -1)

        position_tokens = self.position_embedding(position)
        geometry_position = self._gather(position, geometry_indices)
        geometry_sdf = self._gather(signed_sdf[..., None], geometry_indices)
        geometry = self.geometry_embedding(torch.cat((geometry_position, geometry_sdf), -1))
        geometry = geometry + self._gather(position_tokens, geometry_indices)
        geometry = self.geometry_block(
            geometry, geometry, geometry_position, geometry_position,
        )

        surface_position = self._gather(position, surface_indices)
        surface = self.surface_embedding(self._gather(tokens, surface_indices))
        surface = surface + self._gather(position_tokens, surface_indices)
        volume_anchor_position = self._gather(position, volume_indices)
        volume_anchor = self.volume_embedding(self._gather(tokens, volume_indices))
        volume_anchor = volume_anchor + self._gather(position_tokens, volume_indices)
        volume_query = self.volume_embedding(tokens) + position_tokens

        surface = self.geometry_perceiver(
            surface, geometry, surface_position, geometry_position,
        )
        volume_all = torch.cat((volume_anchor, volume_query), dim=1)
        volume_position = torch.cat((volume_anchor_position, position), dim=1)
        volume_all = self.geometry_perceiver(
            volume_all, geometry, volume_position, geometry_position,
        )

        for index, block in enumerate(self.physics_blocks):
            volume_anchor = volume_all[:, :volume_anchor.shape[1]]
            if index % 2 == 0:
                surface = block(
                    surface, surface, surface_position, surface_position,
                )
                volume_all = block(
                    volume_all, volume_anchor,
                    volume_position, volume_anchor_position,
                )
            else:
                previous_surface = surface
                surface = block(
                    surface, volume_anchor,
                    surface_position, volume_anchor_position,
                )
                volume_all = block(
                    volume_all, previous_surface,
                    volume_position, surface_position,
                )

        for block in self.decoder_blocks:
            volume_anchor = volume_all[:, :volume_anchor.shape[1]]
            volume_all = block(
                volume_all, volume_anchor,
                volume_position, volume_anchor_position,
            )
        volume_query = volume_all[:, volume_anchor.shape[1]:]
        output = self.output(volume_query)
        return output.transpose(1, 2).reshape(batch, -1, height, width)
