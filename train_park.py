#!/usr/bin/env python3
"""CLI: Park et al. 2021 co-evolution (PMC8069842). Example: `python train_park.py --smoke`"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from park_env.constants import ParkEnvConfig
from park_train.co_evolution_loop import CoevolutionTrainer


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="Park et al. 2021 predator–prey MARL (AlgaeDICE).")
    p.add_argument("--smoke", action="store_true", help="Tiny grid, few iterations (CI / sanity).")
    p.add_argument("--iters", type=int, default=30, help="Outer co-evolution iterations (paper: tens).")
    p.add_argument("--horizon", type=int, default=70, help="Sampling horizon h between updates (paper: 70).")
    p.add_argument("--grid", type=int, default=50, help="Grid size N (paper: 50).")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    p.add_argument("--device", type=str, default="cpu", help="torch device, e.g. cpu or cuda.")
    p.add_argument("--batch", type=int, default=256, help="Minibatch size for AlgaeDICE.")
    p.add_argument(
        "--plot",
        type=str,
        default="",
        help="If set, overrides --plot-out (same as saving to that path).",
    )
    p.add_argument(
        "--plot-out",
        type=str,
        default="park_population.png",
        help="Population figure path (PNG). Ignored if --no-plot.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not save population PNG/CSV at the end.",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write CSV next to the population PNG (same basename, .csv).",
    )
    p.add_argument(
        "--plot-live",
        action="store_true",
        help="Open a matplotlib window and refresh the population plot after each outer iteration.",
    )
    p.add_argument("--viz", action="store_true", help="Realtime Pygame grid during rollout (red/green).")
    p.add_argument(
        "--viz-delay",
        type=float,
        default=0.02,
        help="Seconds between drawn frames (0 = fastest; use [ ] keys in window to tune).",
    )
    p.add_argument(
        "--viz-every",
        type=int,
        default=1,
        help="Draw every N-th env step (larger = faster on big grids).",
    )
    args = p.parse_args(argv)

    if args.smoke:
        cfg = ParkEnvConfig(
            grid_size=8,
            n_predators_init=3,
            n_prey_init=8,
            max_episode_steps=200,
            rng_seed=args.seed,
        )
        iters = 2
        horizon = 20
        br = 2
        joint = 1
    else:
        cfg = ParkEnvConfig(
            grid_size=args.grid,
            n_predators_init=100,
            n_prey_init=500,
            max_episode_steps=10_000,
            rng_seed=args.seed,
        )
        iters = args.iters
        horizon = args.horizon
        br = 4
        joint = 2

    if args.viz:
        try:
            from park_visual.park_pygame import ParkRealtimeViewer
        except ImportError:  # pragma: no cover
            print("Pygame not available; install pygame and retry.", file=sys.stderr)
            sys.exit(1)

    want_save = not args.no_plot
    need_matplotlib = want_save or args.plot_live
    live_plot = None
    save_png = None
    save_csv = None
    if need_matplotlib:
        import matplotlib

        if args.plot_live:
            matplotlib.use("TkAgg")
        else:
            matplotlib.use("Agg")

        from park_visual.population_charts import (
            LivePopulationPlot,
            save_park_population_csv,
            save_park_population_png,
        )

        save_png = save_park_population_png
        save_csv = save_park_population_csv

        if args.plot_live:
            live_plot = LivePopulationPlot()

    bs = min(64, args.batch) if args.smoke else args.batch
    trainer = CoevolutionTrainer(cfg, device=args.device, batch_size=bs)

    viewer = None
    if args.viz:
        ve = max(1, args.viz_every)
        if not args.smoke and cfg.grid_size >= 40 and args.viz_every == 1:
            ve = max(ve, 3)
        viewer = ParkRealtimeViewer(
            cfg.grid_size,
            delay_sec=max(0.0, args.viz_delay),
            frame_every=ve,
        )

    preds: list[int] = []
    preys: list[int] = []

    try:
        for it in range(iters):
            on_env_step = None
            if viewer is not None:
                cap_it = it

                def on_env_step(env, step, step_idx, _cap=cap_it):
                    return viewer.tick(
                        env,
                        step,
                        subtitle=f"iter {_cap + 1}/{iters}  rollout {step_idx + 1}/{horizon}",
                    )

            stats = trainer.iteration(
                sampling_horizon=horizon,
                br_prey_updates=br,
                br_pred_updates=br,
                joint_prey_updates=joint,
                joint_pred_updates=joint,
                on_env_step=on_env_step,
            )
            preds.append(int(stats["n_predators"]))
            preys.append(int(stats["n_prey"]))

            if live_plot is not None:
                live_plot.update(preds, preys)

            if args.smoke or (it + 1) % max(1, iters // 5) == 0:
                print(
                    f"iter {it + 1}/{iters}  predators={stats['n_predators']:.0f}  "
                    f"prey={stats['n_prey']:.0f}  "
                    f"buf_p={trainer.pred_buf.size} buf_y={trainer.prey_buf.size}"
                )

            if viewer is not None and viewer.quit_requested:
                print("Window closed; stopping early.")
                break
    finally:
        if viewer is not None:
            viewer.close()
        if live_plot is not None:
            live_plot.close()

    if want_save and preds and save_png is not None:
        out_png = Path(args.plot.strip() or args.plot_out)
        save_png(preds, preys, out_png)
        print(f"Saved population plot: {out_png.resolve()}")
        if not args.no_csv and save_csv is not None:
            csv_path = out_png.with_suffix(".csv")
            save_csv(preds, preys, csv_path)
            print(f"Saved population CSV: {csv_path.resolve()}")
    elif want_save and not preds:
        print("No data to plot (stopped before first iteration).", file=sys.stderr)


if __name__ == "__main__":
    main()
