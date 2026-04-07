"""Population trace rollout (per env step)."""

from __future__ import annotations

import unittest

from park_env.constants import ParkEnvConfig
from park_train.co_evolution_loop import CoevolutionTrainer


class TestPopulationTrace(unittest.TestCase):
    def test_trace_length_includes_t0(self) -> None:
        cfg = ParkEnvConfig(
            grid_size=6,
            n_predators_init=2,
            n_prey_init=4,
            rng_seed=0,
            max_episode_steps=200,
        )
        tr = CoevolutionTrainer(cfg, batch_size=32)
        n = 15
        p, y = tr.population_trace(n, random_actions=True)
        self.assertEqual(len(p), n + 1)
        self.assertEqual(len(y), n + 1)


if __name__ == "__main__":
    unittest.main()
