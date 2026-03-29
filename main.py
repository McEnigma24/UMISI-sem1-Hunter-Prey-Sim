"""Hunter–prey ecosystem: run `python main.py` from this directory."""

from __future__ import annotations

import argparse
import sys

from sim.config import SimConfig
from visual.run_pygame import run_pygame


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="2D hunter–prey ecosystem simulation (Pygame view).")
    p.add_argument("--width", type=int, default=40, help="Grid width")
    p.add_argument("--height", type=int, default=30, help="Grid height")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (default: from config)")
    p.add_argument("--delay-preset", type=int, default=2, help="Initial delay preset index 0..6 (see runner)")
    p.add_argument("--spf", type=int, default=1, help="Initial steps per frame (turbo: use 64–4096)")
    p.add_argument(
        "--window",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=None,
        help="Fixed window size in pixels (default: sized to fit grid on current display)",
    )
    p.add_argument(
        "--parallel-env-threads",
        type=int,
        default=None,
        metavar="N",
        help="Parallel env step: 1=off, 0=auto (CPU), N=worker count (see SimConfig.parallel_env_threads)",
    )
    args = p.parse_args(argv)

    cfg = SimConfig(width=args.width, height=args.height)
    if args.seed is not None:
        cfg.rng_seed = args.seed
    if args.parallel_env_threads is not None:
        cfg.parallel_env_threads = args.parallel_env_threads

    win = (args.window[0], args.window[1]) if args.window is not None else None
    run_pygame(
        cfg,
        window_size=win,
        initial_delay_index=args.delay_preset,
        initial_steps_per_frame=max(1, args.spf),
    )


if __name__ == "__main__":
    main()
