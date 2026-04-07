"""Unit tests for Park et al. 2021 grid environment."""

from __future__ import annotations

import unittest

import numpy as np

from park_env.constants import CELL_EMPTY, CELL_PREDATOR, CELL_PREY, ParkEnvConfig
from park_env.grid_env import ParkGridEnv


class TestParkGridEnv(unittest.TestCase):
    def test_predator_eat_rewards_and_removes_prey(self) -> None:
        cfg = ParkEnvConfig(
            grid_size=5,
            n_predators_init=0,
            n_prey_init=0,
            b_predator=0.0,
            b_prey=0.0,
            max_starvation_predator=99,
            max_age_prey=99,
            obs_radius=5,
            rng_seed=1,
        )
        env = ParkGridEnv(cfg)
        env.reset()
        env._occupancy[:] = CELL_EMPTY  # type: ignore[index]
        env._agent_id[:] = 0  # type: ignore[index]
        env._species.clear()
        env._pred_starvation.clear()
        env._prey_age.clear()
        env._next_id = 1
        env._spawn_at(0, 2, CELL_PREDATOR)
        env._spawn_at(1, 2, CELL_PREY)
        pid = int(env._agent_id[2, 0])
        yid = int(env._agent_id[2, 1])
        # Predator moves right (action 5) into prey
        r = env.step({pid: 5, yid: 4})
        self.assertEqual(r.rewards.get(pid, 0.0), 1.0)
        self.assertEqual(r.rewards.get(yid, 0.0), -1.0)
        self.assertEqual(int(env.occupancy[2, 1]), CELL_PREDATOR)
        self.assertEqual(env.n_prey(), 0)

    def test_prey_walks_into_predator_only_prey_penalty(self) -> None:
        cfg = ParkEnvConfig(
            grid_size=5,
            n_predators_init=0,
            n_prey_init=0,
            b_predator=0.0,
            b_prey=0.0,
            max_starvation_predator=99,
            max_age_prey=99,
            obs_radius=5,
            rng_seed=2,
        )
        env = ParkGridEnv(cfg)
        env.reset()
        env._occupancy[:] = CELL_EMPTY  # type: ignore[index]
        env._agent_id[:] = 0  # type: ignore[index]
        env._species.clear()
        env._pred_starvation.clear()
        env._prey_age.clear()
        env._next_id = 1
        env._spawn_at(1, 2, CELL_PREDATOR)
        env._spawn_at(0, 2, CELL_PREY)
        pid = int(env._agent_id[2, 1])
        yid = int(env._agent_id[2, 0])
        # Prey moves right into predator
        r = env.step({pid: 4, yid: 5})
        self.assertEqual(r.rewards.get(pid, 0.0), 0.0)
        self.assertEqual(r.rewards.get(yid, 0.0), -1.0)
        self.assertEqual(env.n_predators(), 1)
        self.assertEqual(env.n_prey(), 0)

    def test_torus_wrap_move(self) -> None:
        cfg = ParkEnvConfig(
            grid_size=5,
            n_predators_init=0,
            n_prey_init=0,
            b_predator=0.0,
            b_prey=0.0,
            max_starvation_predator=99,
            max_age_prey=99,
            obs_radius=5,
            rng_seed=3,
        )
        env = ParkGridEnv(cfg)
        env.reset()
        env._occupancy[:] = CELL_EMPTY  # type: ignore[index]
        env._agent_id[:] = 0  # type: ignore[index]
        env._species.clear()
        env._pred_starvation.clear()
        env._prey_age.clear()
        env._next_id = 1
        env._spawn_at(0, 0, CELL_PREY)
        yid = int(env._agent_id[0, 0])
        # up-left from (0,0) -> (4,4) on 5x5 torus
        env.step({yid: 0})
        self.assertEqual(int(env.occupancy[4, 4]), CELL_PREY)
        self.assertEqual(env.n_prey(), 1)

    def test_observation_shape_and_one_hot(self) -> None:
        cfg = ParkEnvConfig(grid_size=7, n_predators_init=1, n_prey_init=1, obs_radius=5, rng_seed=0)
        env = ParkGridEnv(cfg)
        env.reset()
        obs = env.build_observations_dict()
        self.assertEqual(len(obs), 2)
        for v in obs.values():
            self.assertEqual(v.shape, (cfg.obs_radius * cfg.obs_radius * 3,))
            self.assertAlmostEqual(float(np.sum(v)), cfg.obs_radius * cfg.obs_radius, places=5)

    def test_global_state_tensor_shape(self) -> None:
        cfg = ParkEnvConfig(grid_size=4, n_predators_init=1, n_prey_init=0, rng_seed=0)
        env = ParkGridEnv(cfg)
        env.reset()
        t = env.get_global_state_tensor()
        self.assertEqual(t.shape, (4, 4, 3))


if __name__ == "__main__":
    unittest.main()
