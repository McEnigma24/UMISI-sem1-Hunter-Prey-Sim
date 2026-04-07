"""Torus topology: coordinates wrap on a rectangular grid."""

from __future__ import annotations

import random


def wrap_x(x: int, w: int) -> int:
    return x % w


def wrap_y(y: int, h: int) -> int:
    return y % h


def wrap_xy(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    return x % w, y % h


def torus_manhattan(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> int:
    dx = abs(x1 - x2)
    dx = min(dx, w - dx)
    dy = abs(y1 - y2)
    dy = min(dy, h - dy)
    return dx + dy


def torus_offset_1d(a: int, b: int, dim: int) -> int:
    """Signed minimal offset from a to b along one torus dimension."""
    d = (b - a) % dim
    if d > dim // 2:
        d -= dim
    return d


def step_toward_torus(
    x: int,
    y: int,
    tx: int,
    ty: int,
    w: int,
    h: int,
    rng: random.Random,
) -> tuple[int, int]:
    """One 4-neighbour step that reduces toroidal Manhattan distance to (tx, ty)."""
    ox = torus_offset_1d(x, tx, w)
    oy = torus_offset_1d(y, ty, h)
    if ox == 0 and oy == 0:
        return 0, 0
    if ox != 0 and oy != 0:
        if rng.random() < 0.5:
            return (1 if ox > 0 else -1), 0
        return 0, (1 if oy > 0 else -1)
    if ox != 0:
        return (1 if ox > 0 else -1), 0
    return 0, (1 if oy > 0 else -1)
