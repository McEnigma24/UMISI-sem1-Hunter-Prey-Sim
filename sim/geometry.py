"""Continuous torus geometry: shortest displacement, distance, wrapped angles."""

from __future__ import annotations

import math


def wrap_coord(v: float, size: float) -> float:
    """Wrap scalar coordinate into [0, size)."""
    w = v % size
    if w < 0.0:
        w += size
    return w


def wrap_xy(x: float, y: float, width: float, height: float) -> tuple[float, float]:
    return wrap_coord(x, width), wrap_coord(y, height)


def torus_delta(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    width: float,
    height: float,
) -> tuple[float, float]:
    """Vector from A to B along shortest path on a torus (B - A)."""
    dx = bx - ax
    dy = by - ay
    if dx > width * 0.5:
        dx -= width
    elif dx < -width * 0.5:
        dx += width
    if dy > height * 0.5:
        dy -= height
    elif dy < -height * 0.5:
        dy += height
    return dx, dy


def torus_distance(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    width: float,
    height: float,
) -> float:
    dx, dy = torus_delta(ax, ay, bx, by, width, height)
    return math.hypot(dx, dy)


def wrap_angle(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    while a <= -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi
    return a


def angle_to(dx: float, dy: float) -> float:
    return math.atan2(dy, dx)
