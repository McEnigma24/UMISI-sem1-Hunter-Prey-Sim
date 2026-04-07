"""Grid-world predator–prey environment — Park et al. 2021 Sec. 2.1."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from park_env.constants import (
    ACTION_DELTAS,
    CELL_EMPTY,
    CELL_PREDATOR,
    CELL_PREY,
    NUM_ACTIONS,
    ParkEnvConfig,
    wrap_coord,
)
from park_env.observation import global_state_one_hot, local_window_one_hot


@dataclass
class StepResult:
    rewards: dict[int, float]
    terminated: bool
    truncated: bool
    n_predators: int
    n_prey: int


@dataclass
class ParkGridEnv:
    """
    One agent per cell; periodic boundaries; sequential action resolution
    (sorted by (y, x, agent_id) each step for reproducibility).
    """

    cfg: ParkEnvConfig = field(default_factory=ParkEnvConfig)
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.cfg.rng_seed)
        self._occupancy: np.ndarray | None = None
        self._agent_id: np.ndarray | None = None
        self._next_id: int = 1
        self._pred_starvation: dict[int, int] = {}
        self._prey_age: dict[int, int] = {}
        self._species: dict[int, int] = {}  # agent_id -> CELL_PREDATOR or CELL_PREY
        self._step_count: int = 0

    @property
    def occupancy(self) -> np.ndarray:
        assert self._occupancy is not None
        return self._occupancy

    @property
    def agent_id_grid(self) -> np.ndarray:
        assert self._agent_id is not None
        return self._agent_id

    def reset(self) -> None:
        c = self.cfg
        n = c.grid_size
        self._occupancy = np.zeros((n, n), dtype=np.int8)
        self._agent_id = np.zeros((n, n), dtype=np.int32)
        self._next_id = 1
        self._pred_starvation.clear()
        self._prey_age.clear()
        self._species.clear()
        self._step_count = 0

        cells = [(x, y) for y in range(n) for x in range(n)]
        self.rng.shuffle(cells)
        idx = 0
        for _ in range(c.n_predators_init):
            while idx < len(cells):
                x, y = cells[idx]
                idx += 1
                if int(self._occupancy[y, x]) == CELL_EMPTY:
                    self._spawn_at(x, y, CELL_PREDATOR)
                    break
        for _ in range(c.n_prey_init):
            while idx < len(cells):
                x, y = cells[idx]
                idx += 1
                if int(self._occupancy[y, x]) == CELL_EMPTY:
                    self._spawn_at(x, y, CELL_PREY)
                    break

    def _alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _spawn_at(self, x: int, y: int, species: int) -> int:
        aid = self._alloc_id()
        self._occupancy[y, x] = species  # type: ignore[index]
        self._agent_id[y, x] = aid  # type: ignore[index]
        self._species[aid] = species
        if species == CELL_PREDATOR:
            self._pred_starvation[aid] = 0
        else:
            self._prey_age[aid] = 0
        return aid

    def _remove_agent_at(self, x: int, y: int) -> None:
        aid = int(self._agent_id[y, x])
        if aid == 0:
            return
        sp = int(self._occupancy[y, x])
        self._occupancy[y, x] = CELL_EMPTY
        self._agent_id[y, x] = 0
        self._species.pop(aid, None)
        if sp == CELL_PREDATOR:
            self._pred_starvation.pop(aid, None)
        else:
            self._prey_age.pop(aid, None)

    def _list_agent_positions(self) -> list[tuple[int, int, int]]:
        """(y, x, agent_id) for sorting."""
        n = self.cfg.grid_size
        out: list[tuple[int, int, int]] = []
        occ = self._occupancy
        ids = self._agent_id
        assert occ is not None and ids is not None
        for y in range(n):
            for x in range(n):
                aid = int(ids[y, x])
                if aid != 0:
                    out.append((y, x, aid))
        out.sort(key=lambda t: (t[0], t[1], t[2]))
        return out

    def get_global_state_tensor(self) -> np.ndarray:
        return global_state_one_hot(self.occupancy)

    def observation_for_agent_at(self, x: int, y: int) -> np.ndarray:
        return local_window_one_hot(self.occupancy, x, y, self.cfg).reshape(-1)

    def build_observations_dict(self) -> dict[int, np.ndarray]:
        obs: dict[int, np.ndarray] = {}
        for y, x, aid in self._list_agent_positions():
            obs[aid] = self.observation_for_agent_at(x, y)
        return obs

    def agent_positions(self) -> dict[int, tuple[int, int]]:
        pos: dict[int, tuple[int, int]] = {}
        for y, x, aid in self._list_agent_positions():
            pos[aid] = (x, y)
        return pos

    def n_predators(self) -> int:
        return int(np.sum(self.occupancy == CELL_PREDATOR))

    def n_prey(self) -> int:
        return int(np.sum(self.occupancy == CELL_PREY))

    def step(self, actions: dict[int, int]) -> StepResult:
        """
        Apply all agents' actions in deterministic row-major order.
        actions: agent_id -> action in [0, 8]. Missing -> remain (4).
        """
        c = self.cfg
        n = c.grid_size
        rewards: dict[int, float] = {}
        occ = self._occupancy
        ids = self._agent_id
        assert occ is not None and ids is not None

        ids_at_step_start = {t[2] for t in self._list_agent_positions()}
        eaten_predator_ids: set[int] = set()

        for y, x, aid in self._list_agent_positions():
            if aid not in self._species:
                continue
            if int(ids[y, x]) != aid:
                continue

            sp = self._species[aid]
            a = int(actions.get(aid, 4))
            if a < 0 or a >= NUM_ACTIONS:
                a = 4
            dx, dy = ACTION_DELTAS[a]
            tx = wrap_coord(x + dx, n)
            ty = wrap_coord(y + dy, n)

            if sp == CELL_PREDATOR:
                self._step_predator(x, y, tx, ty, aid, rewards, eaten_predator_ids)
            else:
                self._step_prey(x, y, tx, ty, aid, rewards)

        self._end_of_step_metabolism(eaten_predator_ids, ids_at_step_start)

        self._step_count += 1
        npred = self.n_predators()
        nprey = self.n_prey()
        terminated = npred == 0 or nprey == 0
        truncated = self._step_count >= c.max_episode_steps
        return StepResult(
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            n_predators=npred,
            n_prey=nprey,
        )

    def _step_predator(
        self,
        x: int,
        y: int,
        tx: int,
        ty: int,
        aid: int,
        rewards: dict[int, float],
        eaten_predator_ids: set[int],
    ) -> None:
        occ = self._occupancy
        ids = self._agent_id
        assert occ is not None and ids is not None

        if tx == x and ty == y:
            return

        target = int(occ[ty, tx])
        if target == CELL_PREDATOR:
            return

        if target == CELL_EMPTY:
            occ[y, x] = CELL_EMPTY
            ids[y, x] = 0
            occ[ty, tx] = CELL_PREDATOR
            ids[ty, tx] = aid
            return

        # eat prey at (tx, ty)
        prey_id = int(ids[ty, tx])
        rewards[aid] = rewards.get(aid, 0.0) + 1.0
        rewards[prey_id] = rewards.get(prey_id, 0.0) - 1.0
        eaten_predator_ids.add(aid)

        self._remove_agent_at(tx, ty)

        occ[y, x] = CELL_EMPTY
        ids[y, x] = 0
        occ[ty, tx] = CELL_PREDATOR
        ids[ty, tx] = aid

        if self.rng.random() < self.cfg.b_predator:
            self._spawn_at(x, y, CELL_PREDATOR)

    def _step_prey(
        self,
        x: int,
        y: int,
        tx: int,
        ty: int,
        aid: int,
        rewards: dict[int, float],
    ) -> None:
        occ = self._occupancy
        ids = self._agent_id
        assert occ is not None and ids is not None

        if tx == x and ty == y:
            return

        target = int(occ[ty, tx])
        if target == CELL_PREY:
            return

        if target == CELL_PREDATOR:
            rewards[aid] = rewards.get(aid, 0.0) - 1.0
            self._remove_agent_at(x, y)
            return

        # move to empty
        occ[y, x] = CELL_EMPTY
        ids[y, x] = 0
        occ[ty, tx] = CELL_PREY
        ids[ty, tx] = aid

        if self.rng.random() < self.cfg.b_prey:
            self._spawn_at(x, y, CELL_PREY)

    def _end_of_step_metabolism(
        self, eaten_predator_ids: set[int], ids_at_step_start: set[int]
    ) -> None:
        c = self.cfg
        n = c.grid_size
        occ = self._occupancy
        ids = self._agent_id
        assert occ is not None and ids is not None

        to_remove: list[tuple[int, int]] = []

        for y in range(n):
            for x in range(n):
                aid = int(ids[y, x])
                if aid == 0:
                    continue
                if aid not in ids_at_step_start:
                    continue
                sp = int(occ[y, x])
                if sp == CELL_PREDATOR:
                    if aid in eaten_predator_ids:
                        self._pred_starvation[aid] = 0
                    else:
                        self._pred_starvation[aid] = self._pred_starvation.get(aid, 0) + 1
                        if self._pred_starvation[aid] >= c.max_starvation_predator:
                            to_remove.append((x, y))
                else:
                    self._prey_age[aid] = self._prey_age.get(aid, 0) + 1
                    if self._prey_age[aid] >= c.max_age_prey:
                        to_remove.append((x, y))

        for x, y in to_remove:
            self._remove_agent_at(x, y)
