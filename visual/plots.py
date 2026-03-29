from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from sim.world import World


def save_history_csv(world: World, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "plants", "prey", "hunters"])
        for t, pl, pr, hu in zip(
            world.history_t,
            world.history_plants,
            world.history_prey,
            world.history_hunters,
        ):
            w.writerow([t, pl, pr, hu])


def show_population_plot(world: World, title: str = "Population over time") -> None:
    if not world.history_t:
        return
    plt.figure(figsize=(9, 5))
    plt.plot(world.history_t, world.history_plants, label="plants (cells)", color="green", alpha=0.85)
    plt.plot(world.history_t, world.history_prey, label="prey", color="blue", alpha=0.85)
    plt.plot(world.history_t, world.history_hunters, label="hunters", color="red", alpha=0.85)
    plt.xlabel("Step")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
