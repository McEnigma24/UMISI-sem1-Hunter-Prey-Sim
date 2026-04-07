"""Local r×r observations with periodic (torus) padding — Park et al. Sec. 2.1."""

from __future__ import annotations

import numpy as np

from park_env.constants import CELL_EMPTY, CELL_PREDATOR, CELL_PREY, ParkEnvConfig


def local_window_one_hot(
    occupancy: np.ndarray,
    cx: int,
    cy: int,
    cfg: ParkEnvConfig,
) -> np.ndarray:
    """
    occupancy: (H, W) int8/ints with CELL_* values.
    Returns array shape (r, r, 3) float32 one-hot per cell (torus wrap).
    """
    h, w = occupancy.shape
    r = cfg.obs_side
    assert r == cfg.obs_radius
    half = r // 2
    out = np.zeros((r, r, 3), dtype=np.float32)
    for yi in range(r):
        for xi in range(r):
            gx = (cx + xi - half) % w
            gy = (cy + yi - half) % h
            c = int(occupancy[gy, gx])
            if c == CELL_EMPTY:
                out[yi, xi, 0] = 1.0
            elif c == CELL_PREDATOR:
                out[yi, xi, 1] = 1.0
            elif c == CELL_PREY:
                out[yi, xi, 2] = 1.0
            else:
                raise ValueError(f"bad cell {c}")
    return out


def global_state_one_hot(occupancy: np.ndarray) -> np.ndarray:
    """H×W×3 one-hot tensor (float32) for logging/plots."""
    h, w = occupancy.shape
    t = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            c = int(occupancy[y, x])
            t[y, x, c] = 1.0
    return t
