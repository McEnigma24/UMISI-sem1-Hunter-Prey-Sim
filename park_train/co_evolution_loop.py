"""
Co-evolution training (Park et al. Table 1, Sec. 2.3):
roll out with shared policies, then AlgaeDICE updates on species-specific buffers.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from park_env.constants import CELL_PREDATOR, CELL_PREY, NUM_ACTIONS, ParkEnvConfig
from park_env.grid_env import ParkGridEnv, StepResult
from park_train.algaedice import DiscreteAlgaeDICE
from park_train.buffer import ReplayBuffer


def _to_torch(
    batch: tuple[np.ndarray, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Returns (states, actions, rewards, next_states, masks)."""
    s, a, r, s2, m = batch
    return (
        torch.as_tensor(s, device=device, dtype=torch.float32),
        torch.as_tensor(a, device=device, dtype=torch.int64),
        torch.as_tensor(r, device=device, dtype=torch.float32),
        torch.as_tensor(s2, device=device, dtype=torch.float32),
        torch.as_tensor(m, device=device, dtype=torch.float32),
    )


def _batch_for_algaedice(
    batch: tuple[np.ndarray, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Order expected by `DiscreteAlgaeDICE.train_step`: s, a, s2, r, m."""
    s, a, r, s2, m = _to_torch(batch, device)
    return (s, a, s2, r, m)


class CoevolutionTrainer:
    def __init__(
        self,
        cfg: ParkEnvConfig,
        *,
        device: str | torch.device = "cpu",
        buffer_capacity: int = 500_000,
        batch_size: int = 256,
        gamma: float = 0.99,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.env = ParkGridEnv(cfg)
        od = cfg.obs_dim_flat
        self.pred_algae = DiscreteAlgaeDICE(od, gamma=gamma).to(self.device)
        self.prey_algae = DiscreteAlgaeDICE(od, gamma=gamma).to(self.device)
        self.pred_buf = ReplayBuffer(od, buffer_capacity)
        self.prey_buf = ReplayBuffer(od, buffer_capacity)
        self._np_rng = np.random.default_rng(
            cfg.rng_seed if cfg.rng_seed is not None else 42
        )

    def collect_steps(
        self,
        max_steps: int,
        *,
        on_env_step: Callable[[ParkGridEnv, StepResult, int], bool] | None = None,
    ) -> dict[str, float]:
        """
        Up to `max_steps` env transitions (resets on terminate/truncate).

        If `on_env_step(env, step, step_index)` returns False, rollout stops early (e.g. user closed the window).
        """
        if self.env._occupancy is None:
            self.env.reset()
        elif self.env.n_predators() == 0 or self.env.n_prey() == 0:
            self.env.reset()

        total_r_p = 0.0
        total_r_y = 0.0
        n_p = 0
        n_y = 0
        last_np, last_ny = self.env.n_predators(), self.env.n_prey()

        for step_idx in range(max_steps):
            obs0 = self.env.build_observations_dict()
            species0 = dict(self.env._species)
            actions: dict[int, int] = {}
            for aid, ob in obs0.items():
                t = torch.as_tensor(ob, device=self.device, dtype=torch.float32).unsqueeze(0)
                sp = species0[aid]
                if sp == CELL_PREDATOR:
                    d = self.pred_algae.actor.dist(t)
                else:
                    d = self.prey_algae.actor.dist(t)
                actions[aid] = int(d.sample().item())

            pre_obs = {k: v.copy() for k, v in obs0.items()}
            step = self.env.step(actions)
            post = self.env.build_observations_dict()
            od = self.cfg.obs_dim_flat

            for aid, s in pre_obs.items():
                sp = species0[aid]
                a = actions[aid]
                r = float(step.rewards.get(aid, 0.0))
                if sp == CELL_PREDATOR:
                    total_r_p += r
                    n_p += 1
                else:
                    total_r_y += r
                    n_y += 1

                if aid in self.env._species:
                    s2 = post[aid]
                    mask = 0.0 if (step.terminated or step.truncated) else 1.0
                else:
                    s2 = np.zeros(od, dtype=np.float32)
                    mask = 0.0

                if sp == CELL_PREDATOR:
                    self.pred_buf.add(s, a, r, s2, mask)
                else:
                    self.prey_buf.add(s, a, r, s2, mask)

            if step.terminated or step.truncated:
                self.env.reset()

            last_np, last_ny = step.n_predators, step.n_prey

            if on_env_step is not None and not on_env_step(self.env, step, step_idx):
                break

        out: dict[str, float] = {
            "n_pred_trans": float(n_p),
            "n_prey_trans": float(n_y),
            "mean_r_pred": float(total_r_p / max(1, n_p)),
            "mean_r_prey": float(total_r_y / max(1, n_y)),
            "n_predators": float(last_np),
            "n_prey": float(last_ny),
        }
        return out

    def train_species(
        self,
        algae: DiscreteAlgaeDICE,
        buf: ReplayBuffer,
        n_updates: int,
    ) -> dict[str, float]:
        if buf.size < self.batch_size:
            return {}
        agg: dict[str, list[float]] = {}
        for _ in range(n_updates):
            batch = buf.sample(self.batch_size, self._np_rng)
            tensors = _batch_for_algaedice(batch, self.device)
            stats = algae.train_step(*tensors)
            for k, v in stats.items():
                agg.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in agg.items()}

    def population_trace(
        self,
        num_steps: int,
        *,
        random_actions: bool = False,
        random_prefix_steps: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Record predator/prey counts along one continuous rollout (paper Fig. 3–5: axis `step`).

        Includes t=0 after `reset()`, then one sample after each env step (length num_steps + 1).
        On episode end, the env resets like `collect_steps` and logging continues until
        `num_steps` transitions are executed.

        - `random_actions=True`: uniform random discrete actions (Fig. 4 style).
        - `random_prefix_steps=k`: first k *transitions* use random actions, then learned policy
          (Fig. 5: use k=500, num_steps=2000).
        """
        self.env.reset()
        pred_hist: list[int] = [self.env.n_predators()]
        prey_hist: list[int] = [self.env.n_prey()]

        for t in range(num_steps):
            obs0 = self.env.build_observations_dict()
            species0 = dict(self.env._species)
            actions: dict[int, int] = {}
            use_random = random_actions or (t < random_prefix_steps)
            for aid, ob in obs0.items():
                if use_random:
                    actions[aid] = int(self.env.rng.randint(0, NUM_ACTIONS - 1))
                else:
                    tt = torch.as_tensor(ob, device=self.device, dtype=torch.float32).unsqueeze(0)
                    sp = species0[aid]
                    if sp == CELL_PREDATOR:
                        d = self.pred_algae.actor.dist(tt)
                    else:
                        d = self.prey_algae.actor.dist(tt)
                    actions[aid] = int(d.sample().item())

            step = self.env.step(actions)
            pred_hist.append(step.n_predators)
            prey_hist.append(step.n_prey)

            if step.terminated or step.truncated:
                self.env.reset()

        return np.asarray(pred_hist, dtype=np.int32), np.asarray(prey_hist, dtype=np.int32)

    def iteration(
        self,
        sampling_horizon: int = 70,
        br_prey_updates: int = 4,
        br_pred_updates: int = 4,
        joint_prey_updates: int = 2,
        joint_pred_updates: int = 2,
        *,
        on_env_step: Callable[[ParkGridEnv, StepResult, int], bool] | None = None,
    ) -> dict[str, float]:
        """
        One outer iteration: data collection, then best-response-style inner passes
        (prey / predator), then short joint refinement.
        """
        roll = self.collect_steps(sampling_horizon, on_env_step=on_env_step)

        py = self.train_species(self.prey_algae, self.prey_buf, br_prey_updates)
        pr = self.train_species(self.pred_algae, self.pred_buf, br_pred_updates)
        py2 = self.train_species(self.prey_algae, self.prey_buf, joint_prey_updates)
        pr2 = self.train_species(self.pred_algae, self.pred_buf, joint_pred_updates)

        out = {**roll}
        for k, v in py.items():
            out[f"prey_br_{k}"] = v
        for k, v in pr.items():
            out[f"pred_br_{k}"] = v
        for k, v in py2.items():
            out[f"prey_joint_{k}"] = v
        for k, v in pr2.items():
            out[f"pred_joint_{k}"] = v
        return out
