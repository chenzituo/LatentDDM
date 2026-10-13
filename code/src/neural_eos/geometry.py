from __future__ import annotations

import random

import torch


GEOMETRY_FAMILIES = (
    "rect",
    "notch",
    "obstacle",
    "multi_obstacle",
    "lshape",
    "ushape",
    "rooms",
    "labyrinth",
    "patch_rooms",
    "random_blocks",
)


def _blocks_to_mask(blocks: torch.Tensor, base: int, device: torch.device) -> torch.Tensor:
    mask = blocks.float().repeat_interleave(base, dim=0).repeat_interleave(base, dim=1)
    return mask[None, None].to(device)


def _rect_blocks_hw(rng: random.Random, ny: int, nx: int) -> torch.Tensor:
    blocks = torch.zeros(ny, nx, dtype=torch.bool)
    h = rng.randint(max(1, ny // 2), ny)
    w = rng.randint(max(1, nx // 2), nx)
    y0 = rng.randint(0, ny - h)
    x0 = rng.randint(0, nx - w)
    blocks[y0 : y0 + h, x0 : x0 + w] = True
    return blocks


def _notch_blocks_hw(rng: random.Random, ny: int, nx: int) -> torch.Tensor:
    blocks = torch.ones(ny, nx, dtype=torch.bool)
    notch_h = rng.randint(max(1, ny // 4), max(1, ny // 2))
    notch_w = rng.randint(max(1, nx // 4), max(1, nx // 2))
    side = rng.choice(["top", "bottom", "left", "right"])
    if side == "top":
        x0 = rng.randint(0, nx - notch_w)
        blocks[:notch_h, x0 : x0 + notch_w] = False
    elif side == "bottom":
        x0 = rng.randint(0, nx - notch_w)
        blocks[ny - notch_h :, x0 : x0 + notch_w] = False
    elif side == "left":
        y0 = rng.randint(0, ny - notch_h)
        blocks[y0 : y0 + notch_h, :notch_w] = False
    else:
        y0 = rng.randint(0, ny - notch_h)
        blocks[y0 : y0 + notch_h, nx - notch_w :] = False
    return blocks


def _obstacle_blocks_hw(rng: random.Random, ny: int, nx: int, count: int = 1) -> torch.Tensor:
    blocks = torch.ones(ny, nx, dtype=torch.bool)
    for _ in range(count):
        h_max = max(1, min(ny - 2, max(2, ny // 4)))
        w_max = max(1, min(nx - 2, max(2, nx // 4)))
        h_min = max(1, min(h_max, max(1, ny // 8)))
        w_min = max(1, min(w_max, max(1, nx // 8)))
        h = rng.randint(h_min, h_max)
        w = rng.randint(w_min, w_max)
        y0 = rng.randint(1, ny - h - 1)
        x0 = rng.randint(1, nx - w - 1)
        blocks[y0 : y0 + h, x0 : x0 + w] = False
    return blocks


def _lshape_blocks_hw(rng: random.Random, ny: int, nx: int) -> torch.Tensor:
    blocks = torch.zeros(ny, nx, dtype=torch.bool)
    thick_y = rng.randint(max(1, ny // 3), max(1, ny // 2))
    thick_x = rng.randint(max(1, nx // 3), max(1, nx // 2))
    if rng.random() < 0.5:
        blocks[:, :thick_x] = True
        blocks[ny - thick_y :, :] = True
    else:
        blocks[:thick_y, :] = True
        blocks[:, :thick_x] = True
    if rng.random() < 0.5:
        blocks = torch.flip(blocks, dims=(0,))
    if rng.random() < 0.5:
        blocks = torch.flip(blocks, dims=(1,))
    return blocks


def _ushape_blocks_hw(rng: random.Random, ny: int, nx: int) -> torch.Tensor:
    blocks = torch.ones(ny, nx, dtype=torch.bool)
    margin_y = rng.randint(1, max(1, ny // 4))
    margin_x = rng.randint(1, max(1, nx // 4))
    gap_w = rng.randint(max(1, nx // 4), max(1, nx // 2))
    x_high = max(margin_x, nx - margin_x - gap_w)
    x0 = rng.randint(margin_x, x_high)
    blocks[margin_y : ny - margin_y, x0 : min(nx, x0 + gap_w)] = False
    blocks[ny - margin_y :, :] = True
    if rng.random() < 0.5:
        blocks = torch.flip(blocks, dims=(0,))
    if rng.random() < 0.5:
        blocks = torch.flip(blocks, dims=(1,))
    return blocks


def _rooms_blocks_hw(rng: random.Random, ny: int, nx: int) -> torch.Tensor:
    blocks = torch.ones(ny, nx, dtype=torch.bool)
    if rng.random() < 0.5:
        wall = rng.randint(max(1, nx // 3), min(nx - 2, max(1, 2 * nx // 3)))
        channel = rng.randint(max(1, ny // 3), min(ny - 2, max(1, 2 * ny // 3)))
        channel_half = max(1, ny // 10)
        blocks[:, wall] = False
        blocks[max(0, channel - channel_half) : min(ny, channel + channel_half + 1), wall] = True
    else:
        wall = rng.randint(max(1, ny // 3), min(ny - 2, max(1, 2 * ny // 3)))
        channel = rng.randint(max(1, nx // 3), min(nx - 2, max(1, 2 * nx // 3)))
        channel_half = max(1, nx // 10)
        blocks[wall, :] = False
        blocks[wall, max(0, channel - channel_half) : min(nx, channel + channel_half + 1)] = True
    return blocks


def _labyrinth_blocks_hw(rng: random.Random, ny: int, nx: int) -> torch.Tensor:
    blocks = torch.zeros(ny, nx, dtype=torch.bool)
    corridor_w = 1 if min(ny, nx) < 16 else 2
    ys = sorted({ny // 5, ny // 2, 4 * ny // 5})
    xs = sorted({nx // 6, nx // 3, nx // 2, 2 * nx // 3, 5 * nx // 6})

    for y in ys:
        blocks[max(0, y - corridor_w) : min(ny, y + corridor_w + 1), 1 : nx - 1] = True
    for x in xs:
        blocks[1 : ny - 1, max(0, x - corridor_w) : min(nx, x + corridor_w + 1)] = True

    for y in ys:
        for x in xs:
            if rng.random() < 0.72:
                room_h = rng.randint(max(2, ny // 10), max(3, ny // 5))
                room_w = rng.randint(max(2, nx // 12), max(3, nx // 5))
                y0 = max(1, min(ny - room_h - 1, y - room_h // 2 + rng.randint(-1, 1)))
                x0 = max(1, min(nx - room_w - 1, x - room_w // 2 + rng.randint(-1, 1)))
                blocks[y0 : y0 + room_h, x0 : x0 + room_w] = True

    for _ in range(max(2, (ny * nx) // 120)):
        h = rng.choice([2, 3, 4])
        w = rng.choice([2, 3, 4, 5])
        if h >= ny - 2 or w >= nx - 2:
            continue
        y0 = rng.randint(1, ny - h - 1)
        x0 = rng.randint(1, nx - w - 1)
        blocks[y0 : y0 + h, x0 : x0 + w] = False

    for y in ys:
        blocks[max(0, y - corridor_w) : min(ny, y + corridor_w + 1), 1 : nx - 1] = True
    for x in xs:
        blocks[1 : ny - 1, max(0, x - corridor_w) : min(nx, x + corridor_w + 1)] = True
    return blocks


def _patch_rooms_blocks_hw(rng: random.Random, ny: int, nx: int) -> torch.Tensor:
    blocks = torch.ones(ny, nx, dtype=torch.bool)
    wall_w = 1 if min(ny, nx) < 16 else 2
    xs = sorted({nx // 5, 2 * nx // 5, 3 * nx // 5, 4 * nx // 5})
    ys = sorted({ny // 4, ny // 2, 3 * ny // 4})

    for x in xs:
        x0, x1 = max(1, x - wall_w // 2), min(nx - 1, x + wall_w)
        blocks[1 : ny - 1, x0:x1] = False
        for y in ys:
            door_h = rng.choice([2, 3, 4])
            y0 = max(1, y - door_h // 2 + rng.randint(-1, 1))
            y1 = min(ny - 1, y0 + door_h)
            blocks[y0:y1, x0:x1] = True

    for y in ys:
        y0, y1 = max(1, y - wall_w // 2), min(ny - 1, y + wall_w)
        blocks[y0:y1, 1 : nx - 1] = False
        for x in xs:
            door_w = rng.choice([2, 3, 4])
            x0 = max(1, x - door_w // 2 + rng.randint(-1, 1))
            x1 = min(nx - 1, x0 + door_w)
            blocks[y0:y1, x0:x1] = True

    for _ in range(max(3, (ny * nx) // 96)):
        h = rng.choice([1, 2, 3])
        w = rng.choice([1, 2, 3, 4])
        if h >= ny - 2 or w >= nx - 2:
            continue
        y0 = rng.randint(2, ny - h - 2)
        x0 = rng.randint(2, nx - w - 2)
        blocks[y0 : y0 + h, x0 : x0 + w] = False
    return blocks


def _random_blocks_hw(rng: random.Random, ny: int, nx: int) -> torch.Tensor:
    blocks = torch.ones(ny, nx, dtype=torch.bool)
    count = rng.randint(max(3, (ny * nx) // 80), max(4, (ny * nx) // 40))
    for _ in range(count):
        h = rng.randint(1, max(1, min(4, ny // 4)))
        w = rng.randint(1, max(1, min(5, nx // 4)))
        if h >= ny - 2 or w >= nx - 2:
            continue
        y0 = rng.randint(1, ny - h - 1)
        x0 = rng.randint(1, nx - w - 1)
        blocks[y0 : y0 + h, x0 : x0 + w] = False

    for _ in range(max(1, min(ny, nx) // 8)):
        if rng.random() < 0.5:
            y = rng.randint(1, ny - 2)
            blocks[y : min(ny - 1, y + 1), 1 : nx - 1] = True
        else:
            x = rng.randint(1, nx - 2)
            blocks[1 : ny - 1, x : min(nx - 1, x + 1)] = True
    return blocks


def sample_geometry_mask_hw(
    height: int,
    width: int,
    base: int,
    family: str,
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    if height % base != 0 or width % base != 0:
        raise ValueError("geometry dimensions must be divisible by the block base")
    ny = height // base
    nx = width // base
    if ny < 4 or nx < 4:
        raise ValueError("geometry needs at least 4x4 blocks")

    if family == "rect":
        blocks = _rect_blocks_hw(rng, ny, nx)
    elif family == "notch":
        blocks = _notch_blocks_hw(rng, ny, nx)
    elif family == "obstacle":
        blocks = _obstacle_blocks_hw(rng, ny, nx, count=1)
    elif family == "multi_obstacle":
        blocks = _obstacle_blocks_hw(rng, ny, nx, count=3)
    elif family == "lshape":
        blocks = _lshape_blocks_hw(rng, ny, nx)
    elif family == "ushape":
        blocks = _ushape_blocks_hw(rng, ny, nx)
    elif family == "rooms":
        blocks = _rooms_blocks_hw(rng, ny, nx)
    elif family == "labyrinth":
        blocks = _labyrinth_blocks_hw(rng, ny, nx)
    elif family == "patch_rooms":
        blocks = _patch_rooms_blocks_hw(rng, ny, nx)
    elif family == "random_blocks":
        blocks = _random_blocks_hw(rng, ny, nx)
    else:
        raise ValueError(f"unknown geometry family: {family}")
    return _blocks_to_mask(blocks, base, device)
