"""Park et al. 2021 (PMC8069842) defaults: Table 2 + action layout (Fig. 2)."""

from __future__ import annotations

from dataclasses import dataclass

# Cell / channel encoding
CELL_EMPTY = 0
CELL_PREDATOR = 1
CELL_PREY = 2

# Actions: 0 up-left, 1 up, 2 up-right, 3 left, 4 remain, 5 right, 6 down-left, 7 down, 8 down-right
# Grid: x increases right, y increases down (screen coords).
ACTION_DELTAS: list[tuple[int, int]] = [
    (-1, -1),  # 0 up-left
    (0, -1),   # 1 up
    (1, -1),   # 2 up-right
    (-1, 0),   # 3 left
    (0, 0),    # 4 remain
    (1, 0),    # 5 right
    (-1, 1),   # 6 down-left
    (0, 1),    # 7 down
    (1, 1),    # 8 down-right
]

NUM_ACTIONS = 9


@dataclass
class ParkEnvConfig:
    """Simulation hyperparameters from the paper (Sec. 2.1, 3; Table 2)."""

    grid_size: int = 50
    n_predators_init: int = 100
    n_prey_init: int = 500
    b_predator: float = 0.2  # b_X reproduction prob after kill move
    b_prey: float = 0.6  # b_Y reproduction prob after move to empty
    max_starvation_predator: int = 15  # T_X
    max_age_prey: int = 30  # T_Y
    obs_radius: int = 5  # r×r neighborhood; paper uses r=5 (5×5 window)
    max_episode_steps: int = 10_000
    rng_seed: int | None = None

    @property
    def obs_side(self) -> int:
        return self.obs_radius

    @property
    def obs_dim_flat(self) -> int:
        s = self.obs_side
        return s * s * 3  # one-hot channels: empty, predator, prey


def wrap_coord(v: int, size: int) -> int:
    return v % size
