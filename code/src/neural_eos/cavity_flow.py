from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from neural_eos.geometry import sample_geometry_mask_hw


@dataclass(frozen=True)
class CavityFlowConfig:
    height: int = 64
    width: int = 64
    base: int = 8
    lid_velocity: float = 1.0
    viscosity: float = 0.8
    dt: float = 0.18
    steps: int = 650
    pressure_iters: int = 45
    domain_area: float = 1.0
    advection: bool = True


def cavity_mask(
    height: int,
    width: int,
    base: int,
    family: str,
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    if family == "empty":
        mask = torch.ones(1, 1, height, width, device=device)
    else:
        mask = sample_geometry_mask_hw(height, width, base, family, rng, device)
    # For this cavity case the outer container is a full enclosed box; families
    # add internal walls/obstacles rather than deleting the outer walls.
    mask[..., 0, :] = 1
    mask[..., -1, :] = 1
    mask[..., :, 0] = 1
    mask[..., :, -1] = 1
    return mask


def lid_mask(mask: torch.Tensor) -> torch.Tensor:
    lid = torch.zeros_like(mask)
    lid[..., 0, :] = mask[..., 0, :]
    return lid


def wall_mask(mask: torch.Tensor, lid: torch.Tensor) -> torch.Tensor:
    active = mask.bool()
    wall = torch.zeros_like(active)
    wall[..., 0, :] = True
    wall[..., -1, :] = True
    wall[..., :, 0] = True
    wall[..., :, -1] = True
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        neigh = torch.roll(active, shifts=(dy, dx), dims=(-2, -1))
        if dx == 1:
            neigh[..., :, 0] = False
        elif dx == -1:
            neigh[..., :, -1] = False
        elif dy == 1:
            neigh[..., 0, :] = False
        elif dy == -1:
            neigh[..., -1, :] = False
        wall = wall | (active & ~neigh)
    return (wall & active & ~lid.bool()).float()


def coordinate_channels(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, _, h, w = mask.shape
    y = torch.linspace(0.0, 1.0, h, device=mask.device)[None, None, :, None].expand(b, 1, h, w)
    x = torch.linspace(0.0, 1.0, w, device=mask.device)[None, None, None, :].expand(b, 1, h, w)
    return x, y


def model_inputs(mask: torch.Tensor, lid: torch.Tensor) -> torch.Tensor:
    wall = wall_mask(mask, lid)
    x, y = coordinate_channels(mask)
    return torch.cat([mask, lid, wall, x * mask, y * mask], dim=1)


def apply_velocity_bc(
    u: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    lid: torch.Tensor,
    lid_velocity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    wall = wall_mask(mask, lid)
    u = u * mask
    v = v * mask
    u = torch.where(wall.bool(), torch.zeros_like(u), u)
    v = torch.where(wall.bool(), torch.zeros_like(v), v)
    u = torch.where(lid.bool(), torch.full_like(u, lid_velocity), u)
    v = torch.where(lid.bool(), torch.zeros_like(v), v)
    return u * mask, v * mask


def _rolled_neighbor(x: torch.Tensor, mask: torch.Tensor, dy: int, dx: int) -> tuple[torch.Tensor, torch.Tensor]:
    x_j = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
    m_j = torch.roll(mask, shifts=(dy, dx), dims=(-2, -1))
    if dx == 1:
        m_j[..., :, 0] = 0
    elif dx == -1:
        m_j[..., :, -1] = 0
    elif dy == 1:
        m_j[..., 0, :] = 0
    elif dy == -1:
        m_j[..., -1, :] = 0
    return x_j, m_j


def velocity_laplacian(
    field: torch.Tensor,
    mask: torch.Tensor,
    lid: torch.Tensor,
    lid_velocity: float,
    component: str,
) -> torch.Tensor:
    out = torch.zeros_like(field)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        f_j, m_j = _rolled_neighbor(field, mask, dy, dx)
        bc = torch.zeros_like(field)
        if component == "u" and dy == 1:
            bc = torch.where(lid.bool(), torch.full_like(field, lid_velocity), bc)
        neighbor = torch.where(m_j.bool(), f_j, bc)
        out = out + mask * (neighbor - field)
    return out * mask


def velocity_derivatives(
    field: torch.Tensor,
    mask: torch.Tensor,
    component: str,
    lid: torch.Tensor,
    lid_velocity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    east, m_e = _rolled_neighbor(field, mask, 0, -1)
    west, m_w = _rolled_neighbor(field, mask, 0, 1)
    south, m_s = _rolled_neighbor(field, mask, -1, 0)
    north, m_n = _rolled_neighbor(field, mask, 1, 0)
    zero = torch.zeros_like(field)
    east = torch.where(m_e.bool(), east, zero)
    west = torch.where(m_w.bool(), west, zero)
    south = torch.where(m_s.bool(), south, zero)
    north_bc = torch.full_like(field, lid_velocity) if component == "u" else zero
    north = torch.where(m_n.bool(), north, north_bc * lid)
    return 0.5 * (east - west) * mask, 0.5 * (south - north) * mask


def divergence(u: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    u_e, m_e = _rolled_neighbor(u, mask, 0, -1)
    u_w, m_w = _rolled_neighbor(u, mask, 0, 1)
    v_s, m_s = _rolled_neighbor(v, mask, -1, 0)
    v_n, m_n = _rolled_neighbor(v, mask, 1, 0)
    u_e = torch.where(m_e.bool(), u_e, torch.zeros_like(u_e))
    u_w = torch.where(m_w.bool(), u_w, torch.zeros_like(u_w))
    v_s = torch.where(m_s.bool(), v_s, torch.zeros_like(v_s))
    v_n = torch.where(m_n.bool(), v_n, torch.zeros_like(v_n))
    return 0.5 * (u_e - u_w + v_s - v_n) * mask


def vorticity(u: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, lid: torch.Tensor, lid_velocity: float) -> torch.Tensor:
    _, du_dy = velocity_derivatives(u, mask, "u", lid, lid_velocity)
    dv_dx, _ = velocity_derivatives(v, mask, "v", lid, lid_velocity)
    return (dv_dx - du_dy) * mask


def pressure_poisson(rhs: torch.Tensor, mask: torch.Tensor, iters: int) -> torch.Tensor:
    p = torch.zeros_like(rhs)
    active_count = mask.sum(dim=(-1, -2, -3), keepdim=True).clamp_min(1.0)
    for _ in range(iters):
        total = torch.zeros_like(p)
        count = torch.zeros_like(p)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            p_j, m_j = _rolled_neighbor(p, mask, dy, dx)
            total = total + p_j * m_j
            count = count + m_j
        p = ((total - rhs) / count.clamp_min(1.0)) * mask
        p = (p - (p * mask).sum(dim=(-1, -2, -3), keepdim=True) / active_count) * mask
    return p


def pressure_gradient(p: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p_e, m_e = _rolled_neighbor(p, mask, 0, -1)
    p_w, m_w = _rolled_neighbor(p, mask, 0, 1)
    p_s, m_s = _rolled_neighbor(p, mask, -1, 0)
    p_n, m_n = _rolled_neighbor(p, mask, 1, 0)
    p_e = torch.where(m_e.bool(), p_e, p)
    p_w = torch.where(m_w.bool(), p_w, p)
    p_s = torch.where(m_s.bool(), p_s, p)
    p_n = torch.where(m_n.bool(), p_n, p)
    return 0.5 * (p_e - p_w) * mask, 0.5 * (p_s - p_n) * mask


def cavity_step(
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    mask: torch.Tensor,
    lid: torch.Tensor,
    cfg: CavityFlowConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, v = apply_velocity_bc(u, v, mask, lid, cfg.lid_velocity)
    if cfg.advection:
        du_dx, du_dy = velocity_derivatives(u, mask, "u", lid, cfg.lid_velocity)
        dv_dx, dv_dy = velocity_derivatives(v, mask, "v", lid, cfg.lid_velocity)
        adv_u = u * du_dx + v * du_dy
        adv_v = u * dv_dx + v * dv_dy
    else:
        adv_u = torch.zeros_like(u)
        adv_v = torch.zeros_like(v)
    u_star = u + cfg.dt * (-adv_u + cfg.viscosity * velocity_laplacian(u, mask, lid, cfg.lid_velocity, "u"))
    v_star = v + cfg.dt * (-adv_v + cfg.viscosity * velocity_laplacian(v, mask, lid, cfg.lid_velocity, "v"))
    u_star, v_star = apply_velocity_bc(u_star, v_star, mask, lid, cfg.lid_velocity)
    rhs = divergence(u_star, v_star, mask) / max(cfg.dt, 1e-6)
    p = pressure_poisson(rhs, mask, cfg.pressure_iters)
    grad_x, grad_y = pressure_gradient(p, mask)
    u_next = u_star - cfg.dt * grad_x
    v_next = v_star - cfg.dt * grad_y
    return (*apply_velocity_bc(u_next, v_next, mask, lid, cfg.lid_velocity), p)


def solve_cavity(mask: torch.Tensor, lid: torch.Tensor, cfg: CavityFlowConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = torch.zeros_like(mask)
    v = torch.zeros_like(mask)
    p = torch.zeros_like(mask)
    u, v = apply_velocity_bc(u, v, mask, lid, cfg.lid_velocity)
    for _ in range(cfg.steps):
        u, v, p = cavity_step(u, v, p, mask, lid, cfg)
    return u * mask, v * mask, p * mask


def generate_cavity_samples(
    num_samples: int,
    cfg: CavityFlowConfig,
    families: tuple[str, ...],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    rng = random.Random(seed)
    inputs = []
    targets = []
    masks = []
    lids = []
    names: list[str] = []
    batches = math.ceil(num_samples / batch_size)
    for batch_idx in range(batches):
        current_bs = min(batch_size, num_samples - batch_idx * batch_size)
        batch_masks = []
        batch_names = []
        for _ in range(current_bs):
            family = rng.choice(families)
            batch_names.append(family)
            batch_masks.append(cavity_mask(cfg.height, cfg.width, cfg.base, family, rng, device))
        mask = torch.cat(batch_masks, dim=0)
        lid = lid_mask(mask)
        u, v, p = solve_cavity(mask, lid, cfg)
        inputs.append(model_inputs(mask, lid).detach().cpu())
        targets.append(torch.cat([u, v, p], dim=1).detach().cpu())
        masks.append(mask.detach().cpu())
        lids.append(lid.detach().cpu())
        names.extend(batch_names)
    return torch.cat(inputs), torch.cat(targets), torch.cat(masks), torch.cat(lids), names
