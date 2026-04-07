#!/usr/bin/env python3
"""
Reproduce Park et al. 2021-style Fig. 3–5 (population vs env step).

Fig. 3: trained policies, three RNG seeds (separate co-evolution runs).
Fig. 4: uniform random actions (same seeds for env init).
Fig. 5: first `switch_step` transitions random, then trained policy (paper: 500 / 2000).

Example:
  python reproduce_paper_figures.py --smoke
  python reproduce_paper_figures.py --out-dir picks --eval-steps 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from park_env.constants import ParkEnvConfig
from park_train.co_evolution_loop import CoevolutionTrainer
from park_visual.paper_figures import (
    plot_figure3_co_evolution,
    plot_figure4_random,
    plot_figure5_random_then_trained,
    save_step_trace_csv,
)


def _train(
    cfg: ParkEnvConfig,
    *,
    outer_iters: int,
    horizon: int,
    br: int,
    joint: int,
    device: str,
    batch_size: int,
) -> CoevolutionTrainer:
    t = CoevolutionTrainer(cfg, device=device, batch_size=batch_size)
    for _ in range(outer_iters):
        t.iteration(
            sampling_horizon=horizon,
            br_prey_updates=br,
            br_pred_updates=br,
            joint_prey_updates=joint,
            joint_pred_updates=joint,
        )
    return t


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="Park et al. 2021 — Fig. 3–5 reproduction (step-wise population).")
    p.add_argument("--smoke", action="store_true", help="Tiny grid / few iters / short eval (sanity).")
    p.add_argument("--out-dir", type=str, default="picks", help="Directory for PNG + CSV traces.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Training / env seeds (paper: three).")
    p.add_argument("--train-iters", type=int, default=30, help="Outer co-evolution iterations per seed.")
    p.add_argument("--horizon", type=int, default=70, help="Sampling horizon h (paper: 70).")
    p.add_argument("--eval-steps", type=int, default=2000, help="Env steps logged per curve (paper: 2000).")
    p.add_argument("--switch-step", type=int, default=500, help="Fig. 5 random prefix length (paper: 500).")
    p.add_argument("--grid", type=int, default=50, help="Grid size N (paper: 50).")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    switch_step = args.switch_step
    if args.smoke:
        cfg_kw: dict = dict(
            grid_size=10,
            n_predators_init=5,
            n_prey_init=25,
            max_episode_steps=500,
        )
        outer_iters = 3
        horizon = 15
        br, joint = 2, 1
        eval_steps = 80
        batch = min(64, args.batch)
        xmax, ymax = None, None
        switch_step = min(25, max(1, eval_steps // 3))
    else:
        cfg_kw = dict(
            grid_size=args.grid,
            n_predators_init=100,
            n_prey_init=500,
            max_episode_steps=10_000,
        )
        outer_iters = args.train_iters
        horizon = args.horizon
        br, joint = 4, 2
        eval_steps = args.eval_steps
        batch = args.batch
        xmax = float(eval_steps)
        ymax = 2500.0

    row_labels = [f"seed {s}" for s in args.seeds]

    trained: list[CoevolutionTrainer] = []
    traces_fig3: list[tuple] = []
    traces_fig5: list[tuple] = []
    traces_fig4: list[tuple] = []

    print("Training + Fig. 3 / Fig. 5 traces (per seed)...")
    for s in args.seeds:
        cfg = ParkEnvConfig(rng_seed=s, **cfg_kw)
        tr = _train(cfg, outer_iters=outer_iters, horizon=horizon, br=br, joint=joint, device=args.device, batch_size=batch)
        trained.append(tr)
        p3, y3 = tr.population_trace(eval_steps, random_actions=False)
        traces_fig3.append((p3, y3))
        save_step_trace_csv(p3, y3, out / f"trace_fig3_seed{s}.csv")

        p5, y5 = tr.population_trace(eval_steps, random_prefix_steps=switch_step)
        traces_fig5.append((p5, y5))
        save_step_trace_csv(p5, y5, out / f"trace_fig5_seed{s}.csv")
        print(f"  seed {s}: trained, eval len = {len(p3)}")

    print("Fig. 4 traces (random actions, no training)...")
    for s in args.seeds:
        cfg = ParkEnvConfig(rng_seed=s, **cfg_kw)
        tr = CoevolutionTrainer(cfg, device=args.device, batch_size=batch)
        p4, y4 = tr.population_trace(eval_steps, random_actions=True)
        traces_fig4.append((p4, y4))
        save_step_trace_csv(p4, y4, out / f"trace_fig4_seed{s}.csv")
        print(f"  seed {s}: random trace len = {len(p4)}")

    f3 = out / "figure3_co_evolution.png"
    f4 = out / "figure4_random.png"
    f5 = out / "figure5_random_then_trained.png"

    plot_figure3_co_evolution(
        traces_fig3,
        f3,
        suptitle="Figure 3 — co-evolution (trained policies)",
        xmax=xmax,
        ymax=ymax,
        row_labels=row_labels,
    )
    plot_figure4_random(
        traces_fig4,
        f4,
        suptitle="Figure 4 — random policies",
        xmax=xmax,
        ymax=ymax,
        row_labels=row_labels,
    )
    plot_figure5_random_then_trained(
        traces_fig5,
        f5,
        switch_step=switch_step,
        suptitle="Figure 5 — random → trained at t",
        xmax=xmax,
        ymax=ymax,
        row_labels=row_labels,
    )

    print(f"Wrote:\n  {f3.resolve()}\n  {f4.resolve()}\n  {f5.resolve()}")


if __name__ == "__main__":
    main()
