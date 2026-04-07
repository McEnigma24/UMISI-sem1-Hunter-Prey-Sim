#!/usr/bin/env python3
"""Realtime Park grid with uniform random actions (no RL). Example: `python watch_park.py --grid 20`"""

from __future__ import annotations

import argparse
import random
import sys

from park_env.constants import NUM_ACTIONS, ParkEnvConfig
from park_env.grid_env import ParkGridEnv
from park_visual.park_pygame import ParkRealtimeViewer


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="Watch Park et al. 2021 grid (random moves).")
    p.add_argument("--grid", type=int, default=25, help="Grid size N")
    p.add_argument("--pred", type=int, default=30, help="Initial predators")
    p.add_argument("--prey", type=int, default=120, help="Initial prey")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delay", type=float, default=0.02)
    p.add_argument("--every", type=int, default=1, help="Draw every N steps")
    args = p.parse_args(argv)

    cfg = ParkEnvConfig(
        grid_size=args.grid,
        n_predators_init=args.pred,
        n_prey_init=args.prey,
        rng_seed=args.seed,
    )
    env = ParkGridEnv(cfg)
    env.reset()
    rng = random.Random(args.seed)
    view = ParkRealtimeViewer(cfg.grid_size, delay_sec=max(0.0, args.delay), frame_every=max(1, args.every))

    step_idx = 0
    try:
        while not view.quit_requested:
            obs0 = env.build_observations_dict()
            actions = {aid: rng.randrange(NUM_ACTIONS) for aid in obs0}
            step = env.step(actions)
            step_idx += 1
            if not view.tick(env, step, subtitle=f"random policy  step {step_idx}"):
                break
            if step.terminated or step.truncated:
                env.reset()
    finally:
        view.close()


if __name__ == "__main__":
    main()
