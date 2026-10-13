from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
SamplerMode = Literal["global", "independent", "schwarz"]
AssemblyMode = Literal["ras", "average"]


@dataclass(frozen=True)
class ToyConfig:
    height: int = 32
    width: int = 32
    patch_size: int = 16
    halo: int = 4
    reaction_min: float = 0.035
    reaction_max: float = 0.09
    solve_iters: int = 650


@dataclass
class SampleResult:
    field: Tensor
    trajectory: list[dict[str, float]]


def coordinate_grid(height: int, width: int, device: torch.device | str = "cpu") -> Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=0).unsqueeze(0)


def neighbor_sum(field: Tensor) -> Tensor:
    # Direct slices avoid the large launch overhead of tiny 3x3 convolutions
    # during CPU dataset generation while preserving the exact stencil.
    neighbors = torch.zeros_like(field)
    neighbors[:, :, 1:, :] += field[:, :, :-1, :]
    neighbors[:, :, :-1, :] += field[:, :, 1:, :]
    neighbors[:, :, :, 1:] += field[:, :, :, :-1]
    neighbors[:, :, :, :-1] += field[:, :, :, 1:]
    return neighbors


def operator_apply(field: Tensor, mask: Tensor, reaction: Tensor) -> Tensor:
    return ((4.0 + reaction[:, :, None, None]) * field - neighbor_sum(field)) * mask


def residual_field(field: Tensor, source: Tensor, mask: Tensor, reaction: Tensor) -> Tensor:
    return (source - operator_apply(field, mask, reaction)) * mask


def relative_residual(field: Tensor, source: Tensor, mask: Tensor, reaction: Tensor) -> Tensor:
    residual = residual_field(field, source, mask, reaction).flatten(1)
    rhs = (source * mask).flatten(1)
    return residual.norm(dim=1) / rhs.norm(dim=1).clamp_min(1.0e-8)


def relative_l2(field: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    error = ((field - target) * mask).flatten(1).norm(dim=1)
    denom = (target * mask).flatten(1).norm(dim=1).clamp_min(1.0e-8)
    return error / denom


def jacobi_step(field: Tensor, source: Tensor, mask: Tensor, reaction: Tensor, omega: float = 0.8) -> Tensor:
    proposal = (neighbor_sum(field * mask) + source) / (4.0 + reaction[:, :, None, None])
    return ((1.0 - omega) * field + omega * proposal) * mask


def solve_screened_poisson(
    mask: Tensor,
    source: Tensor,
    reaction: Tensor,
    iterations: int,
    omega: float = 0.9,
) -> Tensor:
    field = torch.zeros_like(source)
    for _ in range(iterations):
        field = jacobi_step(field, source, mask, reaction, omega=omega)
    return field


def _sample_mask(height: int, width: int, rng: random.Random) -> Tensor:
    mask = torch.ones(1, height, width)
    mask[:, 0, :] = 0.0
    mask[:, -1, :] = 0.0
    mask[:, :, 0] = 0.0
    mask[:, :, -1] = 0.0
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    for _ in range(rng.randint(0, 3)):
        if rng.random() < 0.55:
            cy = rng.randint(height // 4, 3 * height // 4)
            cx = rng.randint(width // 4, 3 * width // 4)
            radius = rng.randint(2, max(2, min(height, width) // 7))
            obstacle = (yy - cy).square() + (xx - cx).square() <= radius * radius
        else:
            h = rng.randint(3, max(3, height // 4))
            w = rng.randint(3, max(3, width // 4))
            y0 = rng.randint(3, max(3, height - h - 3))
            x0 = rng.randint(3, max(3, width - w - 3))
            obstacle = torch.zeros(height, width, dtype=torch.bool)
            obstacle[y0 : y0 + h, x0 : x0 + w] = True
        mask[:, obstacle] = 0.0
    return mask


def _sample_source(mask: Tensor, rng: random.Random) -> Tensor:
    _, height, width = mask.shape
    y = torch.linspace(-1.0, 1.0, height)[:, None]
    x = torch.linspace(-1.0, 1.0, width)[None, :]
    source = torch.zeros_like(mask)
    active = mask[0].bool()
    ys, xs = torch.where(active)
    for _ in range(rng.randint(2, 5)):
        index = rng.randrange(int(ys.numel()))
        cy = float(y[ys[index], 0])
        cx = float(x[0, xs[index]])
        sy = rng.uniform(0.07, 0.22)
        sx = rng.uniform(0.07, 0.22)
        amplitude = rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 1.2)
        source[0] += amplitude * torch.exp(-0.5 * (((y - cy) / sy).square() + ((x - cx) / sx).square()))
    return source * mask


def generate_dataset(count: int, config: ToyConfig, seed: int, batch_size: int = 64) -> dict[str, Tensor]:
    rng = random.Random(seed)
    masks: list[Tensor] = []
    sources: list[Tensor] = []
    reactions: list[float] = []
    for _ in range(count):
        mask = _sample_mask(config.height, config.width, rng)
        masks.append(mask)
        sources.append(_sample_source(mask, rng))
        reactions.append(rng.uniform(config.reaction_min, config.reaction_max))

    mask_all = torch.stack(masks)
    source_all = torch.stack(sources)
    reaction_all = torch.tensor(reactions, dtype=torch.float32)[:, None]
    fields: list[Tensor] = []
    scaled_sources: list[Tensor] = []
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        mask = mask_all[start:stop]
        source = source_all[start:stop]
        reaction = reaction_all[start:stop]
        field = solve_screened_poisson(mask, source, reaction, config.solve_iters)
        scale = field.abs().flatten(1).amax(dim=1).clamp_min(1.0e-4)[:, None, None, None]
        fields.append(field / scale)
        scaled_sources.append(source / scale)
    return {
        "mask": mask_all,
        "source": torch.cat(scaled_sources),
        "reaction": reaction_all,
        "field": torch.cat(fields),
    }


def save_dataset(path: Path, config: ToyConfig, train: int, val: int, test: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.__dict__,
        "train": generate_dataset(train, config, seed),
        "val": generate_dataset(val, config, seed + 10_000),
        "test": generate_dataset(test, config, seed + 20_000),
    }
    torch.save(payload, path)


def timestep_embedding(t: Tensor, dimension: int) -> Tensor:
    half = dimension // 2
    frequency = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device) / max(1, half - 1))
    angle = t.float()[:, None] * frequency[None]
    embedding = torch.cat([angle.sin(), angle.cos()], dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ExtensiveEoS(nn.Module):
    """Additive cell density followed by an analytic potential gradient."""

    def __init__(self, m_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.m_dim = m_dim
        self.density = nn.Sequential(nn.Conv2d(6, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, m_dim, 1))
        self.potential_in = nn.Linear(m_dim, hidden)
        self.potential_weight = nn.Parameter(torch.randn(hidden) / math.sqrt(hidden))
        self.quadratic = nn.Parameter(torch.full((m_dim,), -2.0))

    def extensive(
        self,
        field: Tensor,
        residual: Tensor,
        source: Tensor,
        mask: Tensor,
        coords: Tensor,
        normalizer: float,
    ) -> Tensor:
        features = torch.cat([field, residual, source, mask, coords], dim=1)
        density = self.density(features) * mask
        return density.sum(dim=(-1, -2)) / float(normalizer)

    def intensive(self, extensive: Tensor) -> Tensor:
        hidden = self.potential_in(extensive)
        slope = torch.sigmoid(hidden) * self.potential_weight
        gradient = slope @ self.potential_in.weight
        return gradient + F.softplus(self.quadratic) * extensive


class ConditionedResidualBlock(nn.Module):
    def __init__(self, width: int, condition_dim: int):
        super().__init__()
        groups = min(8, width)
        self.norm1 = nn.GroupNorm(groups, width)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.condition = nn.Linear(condition_dim, 2 * width)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=1)
        h = self.norm1(x)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv1(F.silu(h))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class CoDiffuseSchwarz(nn.Module):
    def __init__(
        self,
        width: int = 32,
        depth: int = 3,
        m_dim: int = 16,
        heads: int = 4,
        use_eos: bool = False,
        use_attention: bool = False,
    ):
        super().__init__()
        if use_attention and not use_eos:
            raise ValueError("the optional patch attention currently requires EoS patch states")
        self.width = width
        self.use_eos = use_eos
        self.use_attention = use_attention
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.eos = ExtensiveEoS(m_dim=m_dim, hidden=width) if use_eos else None
        self.eos_project = (
            nn.Sequential(nn.Linear(m_dim, width), nn.SiLU(), nn.Linear(width, width)) if use_eos else None
        )
        self.patch_project = (
            nn.Sequential(nn.Linear(m_dim + 3, width), nn.SiLU(), nn.Linear(width, width))
            if use_attention
            else None
        )
        self.patch_attention = nn.MultiheadAttention(width, heads, batch_first=True) if use_attention else None
        self.patch_norm = nn.LayerNorm(width) if use_attention else None
        self.patch_ff = (
            nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))
            if use_attention
            else None
        )
        self.stem = nn.Conv2d(6, width, 3, padding=1)
        self.blocks = nn.ModuleList([ConditionedResidualBlock(width, width) for _ in range(depth)])
        self.out = nn.Sequential(nn.GroupNorm(min(8, width), width), nn.SiLU(), nn.Conv2d(width, 1, 3, padding=1))

    def patch_context(self, patch_m: Tensor, geometry: Tensor, global_g: Tensor) -> Tensor:
        if not self.use_attention:
            raise RuntimeError("patch attention is disabled")
        assert self.patch_project is not None
        assert self.eos_project is not None
        assert self.patch_attention is not None
        assert self.patch_norm is not None
        assert self.patch_ff is not None
        token = self.patch_project(torch.cat([patch_m, geometry], dim=-1))
        token = token + self.eos_project(global_g)[:, None, :]
        attended, _ = self.patch_attention(self.patch_norm(token), self.patch_norm(token), self.patch_norm(token))
        token = token + attended
        return token + self.patch_ff(self.patch_norm(token))

    def forward(
        self,
        noisy: Tensor,
        mask: Tensor,
        source: Tensor,
        reaction: Tensor,
        coords: Tensor,
        timestep: Tensor,
        global_g: Tensor | None = None,
        patch_context: Tensor | None = None,
    ) -> Tensor:
        reaction_image = reaction[:, :, None, None].expand(-1, -1, noisy.shape[-2], noisy.shape[-1])
        inputs = torch.cat([noisy, mask, source, reaction_image, coords], dim=1)
        condition = self.time_mlp(timestep_embedding(timestep, self.width))
        if global_g is not None:
            assert self.eos_project is not None
            condition = condition + self.eos_project(global_g)
        if patch_context is not None:
            condition = condition + patch_context
        h = self.stem(inputs)
        for block in self.blocks:
            h = block(h, condition)
        return self.out(h) * mask


def patch_cores(height: int, width: int, patch_size: int) -> list[tuple[int, int, int, int]]:
    if height % patch_size or width % patch_size:
        raise ValueError("toy grid dimensions must be divisible by patch size")
    return [
        (y, y + patch_size, x, x + patch_size)
        for y in range(0, height, patch_size)
        for x in range(0, width, patch_size)
    ]


def _stack_crops(tensor: Tensor, cores: list[tuple[int, int, int, int]], halo: int) -> Tensor:
    padded = F.pad(tensor, (halo, halo, halo, halo)) if halo else tensor
    crops = [padded[:, :, y0 : y1 + 2 * halo, x0 : x1 + 2 * halo] for y0, y1, x0, x1 in cores]
    return torch.stack(crops, dim=1)


def _patch_geometry(cores: list[tuple[int, int, int, int]], height: int, width: int, device: torch.device) -> Tensor:
    values = []
    for y0, y1, x0, x1 in cores:
        cy = ((y0 + y1) / max(1, height) - 1.0)
        cx = ((x0 + x1) / max(1, width) - 1.0)
        area = (y1 - y0) * (x1 - x0) / float(height * width)
        values.append((cx, cy, area))
    return torch.tensor(values, device=device)


def predict_patch_velocity(
    model: CoDiffuseSchwarz,
    noisy: Tensor,
    mask: Tensor,
    source: Tensor,
    reaction: Tensor,
    coords: Tensor,
    timestep: Tensor,
    patch_size: int,
    halo: int,
    connected: bool,
    assembly: AssemblyMode = "ras",
) -> Tensor:
    batch, _, height, width = noisy.shape
    cores = patch_cores(height, width, patch_size)
    global_g: Tensor | None = None
    contexts: Tensor | None = None
    if model.eos is not None:
        residual = residual_field(noisy, source, mask, reaction)
        normalizer = height * width
        core_noisy = _stack_crops(noisy, cores, 0)
        core_residual = _stack_crops(residual, cores, 0)
        core_source = _stack_crops(source, cores, 0)
        core_mask = _stack_crops(mask, cores, 0)
        core_coords = _stack_crops(coords.expand(batch, -1, -1, -1), cores, 0)
        patch_m = []
        for index in range(len(cores)):
            patch_m.append(
                model.eos.extensive(
                    core_noisy[:, index],
                    core_residual[:, index],
                    core_source[:, index],
                    core_mask[:, index],
                    core_coords[:, index],
                    normalizer,
                )
            )
        patch_m_tensor = torch.stack(patch_m, dim=1)
        if connected:
            global_g = model.eos.intensive(patch_m_tensor.sum(dim=1))
            if model.use_attention:
                geometry = _patch_geometry(cores, height, width, noisy.device)[None].expand(batch, -1, -1)
                contexts = model.patch_context(patch_m_tensor, geometry, global_g)
        else:
            global_g = model.eos.intensive(patch_m_tensor.reshape(-1, patch_m_tensor.shape[-1]))

    noisy_crops = _stack_crops(noisy, cores, halo)
    mask_crops = _stack_crops(mask, cores, halo)
    source_crops = _stack_crops(source, cores, halo)
    coords_crops = _stack_crops(coords.expand(batch, -1, -1, -1), cores, halo)
    count = len(cores)
    crop_h, crop_w = noisy_crops.shape[-2:]

    def flatten(x: Tensor) -> Tensor:
        return x.reshape(batch * count, x.shape[2], crop_h, crop_w)

    reaction_flat = reaction[:, None].expand(-1, count, -1).reshape(batch * count, -1)
    timestep_flat = timestep[:, None].expand(-1, count).reshape(-1)
    if connected and global_g is not None:
        g_flat = global_g[:, None, :].expand(-1, count, -1).reshape(batch * count, -1)
        context_flat = contexts.reshape(batch * count, -1) if contexts is not None else None
    else:
        g_flat = global_g
        context_flat = None
    prediction = model(
        flatten(noisy_crops),
        flatten(mask_crops),
        flatten(source_crops),
        reaction_flat,
        flatten(coords_crops),
        timestep_flat,
        g_flat,
        context_flat,
    ).reshape(batch, count, 1, crop_h, crop_w)

    assembled = torch.zeros_like(noisy)
    if assembly == "ras":
        for index, (y0, y1, x0, x1) in enumerate(cores):
            core = prediction[:, index, :, halo : halo + patch_size, halo : halo + patch_size]
            assembled[:, :, y0:y1, x0:x1] = core
        return assembled * mask
    if assembly != "average":
        raise ValueError(f"unknown assembly mode: {assembly}")

    weight = torch.zeros_like(noisy)
    for index, (y0, y1, x0, x1) in enumerate(cores):
        global_y0 = max(0, y0 - halo)
        global_y1 = min(height, y1 + halo)
        global_x0 = max(0, x0 - halo)
        global_x1 = min(width, x1 + halo)
        local_y0 = global_y0 - (y0 - halo)
        local_y1 = local_y0 + (global_y1 - global_y0)
        local_x0 = global_x0 - (x0 - halo)
        local_x1 = local_x0 + (global_x1 - global_x0)
        assembled[:, :, global_y0:global_y1, global_x0:global_x1] += prediction[
            :, index, :, local_y0:local_y1, local_x0:local_x1
        ]
        weight[:, :, global_y0:global_y1, global_x0:global_x1] += 1.0
    return (assembled / weight.clamp_min(1.0)) * mask


def predict_global_velocity(
    model: CoDiffuseSchwarz,
    noisy: Tensor,
    mask: Tensor,
    source: Tensor,
    reaction: Tensor,
    coords: Tensor,
    timestep: Tensor,
) -> Tensor:
    global_g = None
    if model.eos is not None:
        residual = residual_field(noisy, source, mask, reaction)
        extensive = model.eos.extensive(noisy, residual, source, mask, coords, noisy.shape[-2] * noisy.shape[-1])
        global_g = model.eos.intensive(extensive)
    return model(noisy, mask, source, reaction, coords, timestep, global_g)


def boundary_error_ratio(field: Tensor, target: Tensor, mask: Tensor, patch_size: int) -> Tensor:
    error = (field - target).abs() * mask
    boundary = torch.zeros_like(mask)
    for x in range(patch_size, field.shape[-1], patch_size):
        boundary[..., x - 1 : x + 1] = 1.0
    for y in range(patch_size, field.shape[-2], patch_size):
        boundary[..., y - 1 : y + 1, :] = 1.0
    boundary = boundary * mask
    interface_error = (error * boundary).sum(dim=(-1, -2, -3)) / boundary.sum(dim=(-1, -2, -3)).clamp_min(1.0)
    global_error = error.sum(dim=(-1, -2, -3)) / mask.sum(dim=(-1, -2, -3)).clamp_min(1.0)
    return interface_error / global_error.clamp_min(1.0e-8)


@torch.no_grad()
def sample_flow(
    model: CoDiffuseSchwarz,
    mask: Tensor,
    source: Tensor,
    reaction: Tensor,
    target: Tensor | None,
    config: ToyConfig,
    mode: SamplerMode,
    inference_steps: int,
    initial_noise: Tensor,
    assembly: AssemblyMode = "ras",
) -> SampleResult:
    device = mask.device
    times = torch.linspace(1.0, 0.0, inference_steps + 1, device=device)
    coords = coordinate_grid(config.height, config.width, device).expand(mask.shape[0], -1, -1, -1)
    field = initial_noise * mask
    trajectory: list[dict[str, float]] = []

    for step_index in range(inference_steps):
        time_value = times[step_index]
        timestep = torch.full((mask.shape[0],), float(time_value), device=device)
        if mode == "global":
            predicted_velocity = predict_global_velocity(model, field, mask, source, reaction, coords, timestep)
        else:
            predicted_velocity = predict_patch_velocity(
                model,
                field,
                mask,
                source,
                reaction,
                coords,
                timestep,
                config.patch_size,
                config.halo if mode != "independent" else 0,
                connected=mode != "independent",
                assembly=assembly,
            )

        dt = times[step_index + 1] - time_value
        field = (field + dt * predicted_velocity).clamp(-1.75, 1.75) * mask

        row = {
            "step": float(step_index),
            "time": float(times[step_index + 1]),
            "residual": float(relative_residual(field, source, mask, reaction).mean()),
        }
        if target is not None:
            row["relative_l2"] = float(relative_l2(field, target, mask).mean())
            row["interface_error_ratio"] = float(boundary_error_ratio(field, target, mask, config.patch_size).mean())
        trajectory.append(row)

    return SampleResult(field=field, trajectory=trajectory)
