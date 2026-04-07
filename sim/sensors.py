"""Vision cone on a grid: range + focus tradeoff (narrow/long vs wide/short)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from sim.config import SimConfig
from sim.genome import VISION_FOCUS, VISION_RANGE
from sim.torus import torus_manhattan

DIRS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _norm(level: float, cfg: SimConfig, trait_idx: int) -> float:
    lo, hi = cfg.trait_bounds[trait_idx]
    if hi <= lo:
        return 0.5
    return (level - lo) / float(hi - lo)


def vision_arc_deg(cfg: SimConfig, vf_level: float) -> float:
    vf_n = _norm(vf_level, cfg, VISION_FOCUS)
    return cfg.vision_arc_max_deg - (cfg.vision_arc_max_deg - cfg.vision_arc_min_deg) * vf_n


def vision_reach_cells(cfg: SimConfig, vr_level: float, vf_level: float) -> int:
    vr_n = _norm(vr_level, cfg, VISION_RANGE)
    arc = vision_arc_deg(cfg, vf_level)
    base = cfg.vision_min_reach_cells + vr_n * cfg.vision_reach_cells_coeff * cfg.vision_reach_per_level
    if arc >= 300.0:
        base *= 0.78
    elif arc <= 110.0:
        base *= 1.12
    return max(1, int(round(base)))


def facing_index(dx: int, dy: int) -> int:
    for i, (fx, fy) in enumerate(DIRS_4):
        if fx == dx and fy == dy:
            return i
    return 0


def iter_visible_cells(
    ax: int,
    ay: int,
    facing_idx: int,
    w: int,
    h: int,
    cfg: SimConfig,
    vr_level: float,
    vf_level: float,
) -> Iterable[tuple[int, int]]:
    r = vision_reach_cells(cfg, vr_level, vf_level)
    arc = vision_arc_deg(cfg, vf_level)
    fdx, fdy = DIRS_4[facing_idx % 4]
    fwd = math.atan2(fdy, fdx)
    full = arc >= 359.5
    half = math.radians(arc / 2.0)
    seen: set[tuple[int, int]] = set()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if abs(dx) + abs(dy) > r:
                continue
            if dx == 0 and dy == 0:
                continue
            tx = (ax + dx) % w
            ty = (ay + dy) % h
            if (tx, ty) in seen:
                continue
            seen.add((tx, ty))
            if full:
                yield tx, ty
                continue
            ang = math.atan2(dy, dx)
            da = ang - fwd
            while da > math.pi:
                da -= 2 * math.pi
            while da < -math.pi:
                da += 2 * math.pi
            if abs(da) <= half + 1e-9:
                yield tx, ty


def nearest_visible_agent(
    ax: int,
    ay: int,
    facing_idx: int,
    agent_kind_grid: list[list[int]],
    w: int,
    h: int,
    cfg: SimConfig,
    traits: list[list[list[float]]],
    target_kind: int,
) -> tuple[int, int] | None:
    vr = traits[VISION_RANGE][ay][ax]
    vf = traits[VISION_FOCUS][ay][ax]
    best: tuple[int, int] | None = None
    best_key: tuple[int, int, int] | None = None
    for tx, ty in iter_visible_cells(ax, ay, facing_idx, w, h, cfg, vr, vf):
        if agent_kind_grid[ty][tx] != target_kind:
            continue
        d = torus_manhattan(ax, ay, tx, ty, w, h)
        cand = (d, tx, ty)
        if best_key is None or cand < best_key:
            best_key = cand
            best = (tx, ty)
    return best
