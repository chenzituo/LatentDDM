"""Compositional Cartesian flow matching with EoS-conditioned patch exchange."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def timestep_embedding(timestep: Tensor, dimension: int) -> Tensor:
    half = dimension // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timestep.device, dtype=timestep.dtype)
        / max(1, half - 1)
    )
    angle = timestep[:, None] * frequency[None]
    embedding = torch.cat((angle.sin(), angle.cos()), dim=1)
    return F.pad(embedding, (0, dimension % 2))


def patch_cores(height: int, width: int, patch_size: int) -> list[tuple[int, int, int, int]]:
    if height % patch_size or width % patch_size:
        raise ValueError("height and width must be divisible by patch size")
    return [
        (y, y + patch_size, x, x + patch_size)
        for y in range(0, height, patch_size)
        for x in range(0, width, patch_size)
    ]


def stack_crops(tensor: Tensor, cores: list[tuple[int, int, int, int]], halo: int) -> Tensor:
    padded = F.pad(tensor, (halo, halo, halo, halo)) if halo else tensor
    return torch.stack(
        [padded[:, :, y0 : y1 + 2 * halo, x0 : x1 + 2 * halo] for y0, y1, x0, x1 in cores],
        dim=1,
    )


def patch_geometry(
    cores: list[tuple[int, int, int, int]],
    weight: Tensor,
) -> Tensor:
    _, _, height, width = weight.shape
    geometry = []
    for y0, y1, x0, x1 in cores:
        center_y = (y0 + y1) / height - 1.0
        center_x = (x0 + x1) / width - 1.0
        area = (y1 - y0) * (x1 - x0) / float(height * width)
        active_fraction = weight[:, :, y0:y1, x0:x1].mean(dim=(1, 2, 3))
        fixed = torch.tensor((center_x, center_y, area), device=weight.device, dtype=weight.dtype)
        geometry.append(torch.cat((fixed[None].expand(weight.shape[0], -1), active_fraction[:, None]), dim=1))
    return torch.stack(geometry, dim=1)


class ExtensiveEoS(nn.Module):
    def __init__(self, in_channels: int, m_dim: int, hidden: int) -> None:
        super().__init__()
        self.density = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, m_dim, 1),
        )
        self.potential_input = nn.Linear(m_dim, hidden)
        self.potential_weight = nn.Parameter(torch.randn(hidden) / math.sqrt(hidden))
        self.quadratic = nn.Parameter(torch.full((m_dim,), -2.0))

    def extensive(self, state: Tensor, weight: Tensor, normalizer: int) -> Tensor:
        density = self.density(state) * weight
        return density.sum(dim=(-1, -2)) / float(normalizer)

    def intensive(self, extensive: Tensor) -> Tensor:
        hidden = self.potential_input(extensive)
        slope = torch.sigmoid(hidden) * self.potential_weight
        gradient = slope @ self.potential_input.weight
        return gradient + F.softplus(self.quadratic) * extensive


class ConditionedBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        groups = min(8, width)
        self.norm1 = nn.GroupNorm(groups, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.condition = nn.Linear(width, 2 * width)

    def forward(self, state: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=1)
        hidden = self.norm1(state)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv1(F.silu(hidden))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return state + hidden


class CartesianFlowEoS(nn.Module):
    def __init__(
        self,
        condition_channels: int,
        width: int = 64,
        depth: int = 4,
        m_dim: int = 24,
        heads: int = 4,
        use_eos: bool = True,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        if use_attention and not use_eos:
            raise ValueError("patch attention currently consumes extensive EoS states")
        self.condition_channels = condition_channels
        self.width = width
        self.use_eos = use_eos
        self.use_attention = use_attention
        state_channels = 1 + condition_channels
        self.eos = ExtensiveEoS(state_channels, m_dim, width) if use_eos else None
        self.eos_projection = nn.Sequential(nn.Linear(m_dim, width), nn.SiLU(), nn.Linear(width, width)) \
            if use_eos else None
        self.token_projection = nn.Sequential(
            nn.Linear(m_dim + 4, width), nn.SiLU(), nn.Linear(width, width)
        ) if use_attention else None
        self.token_norm = nn.LayerNorm(width) if use_attention else None
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True) if use_attention else None
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        ) if use_attention else None
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.stem = nn.Conv2d(state_channels, width, 3, padding=1)
        self.blocks = nn.ModuleList(ConditionedBlock(width) for _ in range(depth))
        self.output = nn.Sequential(
            nn.GroupNorm(min(8, width), width), nn.SiLU(), nn.Conv2d(width, 1, 3, padding=1)
        )

    def token_context(self, patch_extensive: Tensor, geometry: Tensor, intensive: Tensor) -> Tensor:
        assert self.token_projection is not None
        assert self.token_norm is not None
        assert self.attention is not None
        assert self.token_mlp is not None
        assert self.eos_projection is not None
        token = self.token_projection(torch.cat((patch_extensive, geometry), dim=-1))
        token = token + self.eos_projection(intensive)[:, None]
        normalized = self.token_norm(token)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        token = token + attended
        return token + self.token_mlp(token)

    def local_velocity(
        self,
        noisy: Tensor,
        physical: Tensor,
        timestep: Tensor,
        intensive: Tensor | None,
        token_context: Tensor | None,
    ) -> Tensor:
        condition = self.time(timestep_embedding(timestep, self.width))
        if intensive is not None:
            assert self.eos_projection is not None
            condition = condition + self.eos_projection(intensive)
        if token_context is not None:
            condition = condition + token_context
        hidden = self.stem(torch.cat((noisy, physical), dim=1))
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(hidden)

    def patch_velocity(
        self,
        noisy: Tensor,
        physical: Tensor,
        weight: Tensor,
        timestep: Tensor,
        patch_size: int,
        halo: int,
    ) -> Tensor:
        batch, _, height, width = noisy.shape
        cores = patch_cores(height, width, patch_size)
        count = len(cores)
        intensive = None
        contexts = None
        if self.eos is not None:
            core_state = stack_crops(torch.cat((noisy, physical), dim=1), cores, 0)
            core_weight = stack_crops(weight, cores, 0)
            patch_extensive = []
            for index in range(count):
                patch_extensive.append(
                    self.eos.extensive(core_state[:, index], core_weight[:, index], height * width)
                )
            patch_extensive_tensor = torch.stack(patch_extensive, dim=1)
            intensive = self.eos.intensive(patch_extensive_tensor.sum(dim=1))
            if self.use_attention:
                contexts = self.token_context(
                    patch_extensive_tensor, patch_geometry(cores, weight), intensive
                )

        noisy_crops = stack_crops(noisy, cores, halo)
        physical_crops = stack_crops(physical, cores, halo)
        crop_height, crop_width = noisy_crops.shape[-2:]
        noisy_flat = noisy_crops.reshape(batch * count, 1, crop_height, crop_width)
        physical_flat = physical_crops.reshape(
            batch * count, physical.shape[1], crop_height, crop_width
        )
        timestep_flat = timestep[:, None].expand(-1, count).reshape(-1)
        intensive_flat = intensive[:, None].expand(-1, count, -1).reshape(batch * count, -1) \
            if intensive is not None else None
        context_flat = contexts.reshape(batch * count, -1) if contexts is not None else None
        local = self.local_velocity(
            noisy_flat, physical_flat, timestep_flat, intensive_flat, context_flat
        ).reshape(batch, count, 1, crop_height, crop_width)
        assembled = torch.zeros_like(noisy)
        for index, (y0, y1, x0, x1) in enumerate(cores):
            assembled[:, :, y0:y1, x0:x1] = local[
                :, index, :, halo : halo + patch_size, halo : halo + patch_size
            ]
        return assembled * (weight > 0.0)

    @torch.no_grad()
    def sample(
        self,
        physical: Tensor,
        weight: Tensor,
        patch_size: int,
        halo: int,
        steps: int,
        seed: int = 0,
        noise_scale: float = 1.0,
        integrator: str = "midpoint_time",
    ) -> Tensor:
        if integrator not in {"midpoint_time", "euler", "heun"}:
            raise ValueError(f"unknown flow integrator: {integrator}")
        generator = torch.Generator(device=physical.device).manual_seed(seed)
        state = torch.randn(
            (physical.shape[0], 1, physical.shape[-2], physical.shape[-1]),
            device=physical.device,
            dtype=physical.dtype,
            generator=generator,
        ) * noise_scale * (weight > 0.0)
        for step in range(steps):
            time_value = (step + 0.5) / steps if integrator == "midpoint_time" else step / steps
            timestep = torch.full(
                (physical.shape[0],), time_value,
                device=physical.device, dtype=physical.dtype,
            )
            velocity = self.patch_velocity(
                state, physical, weight, timestep, patch_size=patch_size, halo=halo
            )
            if integrator == "heun":
                predictor = (state + velocity / steps) * (weight > 0.0)
                next_timestep = torch.full(
                    (physical.shape[0],), (step + 1.0) / steps,
                    device=physical.device, dtype=physical.dtype,
                )
                next_velocity = self.patch_velocity(
                    predictor,
                    physical,
                    weight,
                    next_timestep,
                    patch_size=patch_size,
                    halo=halo,
                )
                state = (state + 0.5 * (velocity + next_velocity) / steps) * (weight > 0.0)
            else:
                state = (state + velocity / steps) * (weight > 0.0)
        return state
