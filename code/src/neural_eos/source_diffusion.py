from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from neural_eos.cavity_flow import coordinate_channels, wall_mask
from neural_eos.geometry import sample_geometry_mask_hw


@dataclass(frozen=True)
class SourceDiffusionConfig:
    height: int
    width: int
    base: int = 4
    iters: int = 700
    reaction: float = 0.03
    min_blobs: int = 3
    max_blobs: int = 7


def diffusion_mask(height: int, width: int, base: int, family: str, rng: random.Random, device: torch.device) -> torch.Tensor:
    if family == "empty":
        mask = torch.ones(1, 1, height, width, device=device)
    else:
        mask = sample_geometry_mask_hw(height, width, base, family, rng, device)
    mask[..., 0, :] = 1.0
    mask[..., -1, :] = 1.0
    mask[..., :, 0] = 1.0
    mask[..., :, -1] = 1.0
    return mask


def zero_lid(mask: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(mask)


def interior_mask(mask: torch.Tensor) -> torch.Tensor:
    lid = zero_lid(mask)
    return mask * (1.0 - wall_mask(mask, lid))


def sample_source(mask: torch.Tensor, cfg: SourceDiffusionConfig, rng: random.Random) -> torch.Tensor:
    _, _, h, w = mask.shape
    device = mask.device
    active = interior_mask(mask)[0, 0].bool()
    ys, xs = torch.where(active)
    source = torch.zeros_like(mask)
    if ys.numel() == 0:
        return source

    y_grid = torch.linspace(0.0, 1.0, h, device=device)[None, None, :, None]
    x_grid = torch.linspace(0.0, 1.0, w, device=device)[None, None, None, :]
    count = rng.randint(cfg.min_blobs, cfg.max_blobs)
    for _ in range(count):
        pick = rng.randrange(int(ys.numel()))
        cy = float(ys[pick].item()) / max(1, h - 1)
        cx = float(xs[pick].item()) / max(1, w - 1)
        amp = rng.uniform(0.45, 1.35)
        sigma_y = rng.uniform(0.025, 0.085)
        sigma_x = rng.uniform(0.025, 0.085)
        blob = amp * torch.exp(-0.5 * (((y_grid - cy) / sigma_y).pow(2) + ((x_grid - cx) / sigma_x).pow(2)))
        source = source + blob

    source = source * active[None, None].float()
    max_val = source.amax(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-8)
    return source / max_val


def solve_screened_diffusion(mask: torch.Tensor, source: torch.Tensor, iters: int, reaction: float) -> torch.Tensor:
    active = interior_mask(mask)
    u = torch.zeros_like(source)
    kernel = torch.tensor(
        [[[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]],
        dtype=source.dtype,
        device=source.device,
    )
    rhs = source * active
    denom = 4.0 + float(reaction)
    for _ in range(iters):
        neighbor_sum = F.conv2d(u, kernel, padding=1)
        u = ((neighbor_sum + rhs) / denom) * active
    max_val = u.amax(dim=(-1, -2, -3), keepdim=True).clamp_min(1e-8)
    return (u / max_val) * active


def model_inputs(mask: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    lid = zero_lid(mask)
    wall = wall_mask(mask, lid)
    x, y = coordinate_channels(mask)
    return torch.cat([mask, source * mask, wall, x * mask, y * mask], dim=1)


def generate_source_diffusion_samples(
    count: int,
    cfg: SourceDiffusionConfig,
    families: list[str],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    rng = random.Random(seed)
    inputs = []
    fields = []
    sources = []
    masks = []
    lids = []
    names = []
    generated = 0
    while generated < count:
        current = min(batch_size, count - generated)
        batch_masks = []
        batch_sources = []
        for _ in range(current):
            family = rng.choice(families)
            mask = diffusion_mask(cfg.height, cfg.width, cfg.base, family, rng, device)
            source = sample_source(mask, cfg, rng)
            batch_masks.append(mask)
            batch_sources.append(source)
            names.append(family)
        mask = torch.cat(batch_masks, dim=0)
        source = torch.cat(batch_sources, dim=0)
        field = solve_screened_diffusion(mask, source, cfg.iters, cfg.reaction)
        inputs.append(model_inputs(mask, source).detach().cpu())
        fields.append(field.detach().cpu())
        sources.append(source.detach().cpu())
        masks.append(mask.detach().cpu())
        lids.append(zero_lid(mask).detach().cpu())
        generated += current
    return torch.cat(inputs), torch.cat(fields), torch.cat(sources), torch.cat(masks), torch.cat(lids), names
