"""Hunter–prey continuous torus: policy evolution. Run: python main.py from this directory."""

from __future__ import annotations

import argparse
import sys

from sim.config import SimConfig
from visual.run_pygame import run_pygame


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(
        description="Continuous 2D hunter–prey (torus), evolving movement policies."
    )
    p.add_argument("--width", type=float, default=100.0, help="World width (torus)")
    p.add_argument("--height", type=float, default=100.0, help="World height (torus)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (default: from config)")
    p.add_argument("--delay-preset", type=int, default=0, help="Initial delay preset index 0..6 (0 = no delay)")
    p.add_argument("--spf", type=int, default=1, help="Initial steps per frame (turbo: 64–4096)")
    p.add_argument(
        "--window",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=None,
        help="Fixed window size in pixels",
    )
    p.add_argument(
        "--evolve",
        choices=("both", "prey", "predator", "none"),
        default="both",
        help="Which species may mutate policy genomes in offspring (default: both).",
    )
    args = p.parse_args(argv)

    cfg = SimConfig(width=args.width, height=args.height)
    if args.seed is not None:
        cfg.rng_seed = args.seed
    if args.evolve == "both":
        cfg.evolve_prey = True
        cfg.evolve_predator = True
    elif args.evolve == "prey":
        cfg.evolve_prey = True
        cfg.evolve_predator = False
    elif args.evolve == "predator":
        cfg.evolve_prey = False
        cfg.evolve_predator = True
    else:
        cfg.evolve_prey = False
        cfg.evolve_predator = False

    win = (args.window[0], args.window[1]) if args.window is not None else None
    run_pygame(
        cfg,
        window_size=win,
        initial_delay_index=args.delay_preset,
        initial_steps_per_frame=max(1, args.spf),
    )


if __name__ == "__main__":
    main()
