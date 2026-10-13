from __future__ import annotations

import random

Patch = tuple[int, int, int, int]


def uniform_patches(h: int, w: int, size: int) -> list[Patch]:
    if h % size != 0 or w % size != 0:
        raise ValueError(f"domain {(h, w)} is not tileable by patch size {size}")
    return [(y, y + size, x, x + size) for y in range(0, h, size) for x in range(0, w, size)]


def jittered_grid_patches(
    h: int,
    w: int,
    size: int,
    rng: random.Random,
    jitter_frac: float = 0.25,
    min_size: int = 2,
) -> list[Patch]:
    if h % size != 0 or w % size != 0:
        raise ValueError(f"domain {(h, w)} is not tileable by patch size {size}")
    if min_size < 1:
        raise ValueError("min_size must be at least 1")
    jitter = max(0, int(round(size * jitter_frac)))

    def breaks(length: int) -> list[int]:
        n = length // size
        out = [0]
        prev = 0
        for idx in range(1, n):
            nominal = idx * size
            remaining = n - idx
            lo = max(prev + min_size, nominal - jitter)
            hi = min(length - remaining * min_size, nominal + jitter)
            if lo > hi:
                point = max(prev + min_size, min(length - remaining * min_size, nominal))
            else:
                point = rng.randint(lo, hi)
            out.append(point)
            prev = point
        out.append(length)
        return out

    ys = breaks(h)
    xs = breaks(w)
    return [(y0, y1, x0, x1) for y0, y1 in zip(ys[:-1], ys[1:]) for x0, x1 in zip(xs[:-1], xs[1:])]


def patch_size_histogram(patches: list[Patch]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for y0, y1, x0, x1 in patches:
        key = f"{y1 - y0}x{x1 - x0}"
        hist[key] = hist.get(key, 0) + 1
    return hist
