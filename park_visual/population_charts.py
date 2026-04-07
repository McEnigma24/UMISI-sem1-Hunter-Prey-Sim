"""Population vs outer iteration — save PNG / optional live matplotlib (Park MARL)."""

from __future__ import annotations

import csv
from pathlib import Path


def save_park_population_png(
    predators: list[int],
    prey: list[int],
    path: str | Path,
    *,
    title: str = "Park et al. 2021 — population (end of each sampling horizon)",
    xlabel: str = "Outer co-evolution iteration",
    ylabel: str = "Population (individuals)",
) -> None:
    """Non-interactive save (Agg backend safe if set before pyplot import)."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = list(range(1, len(predators) + 1))
    ax.plot(x, predators, color="tab:red", lw=2.0, marker="o", markersize=3, label="Predators")
    ax.plot(x, prey, color="tab:green", lw=2.0, marker="o", markersize=3, label="Prey")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_park_population_csv(predators: list[int], prey: list[int], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["outer_iteration", "n_predators", "n_prey"])
        for i, (npred, nprey) in enumerate(zip(predators, prey, strict=True), start=1):
            w.writerow([i, npred, nprey])


class LivePopulationPlot:
    """
    Interactive window; call `update` after each outer iteration.
    Create only after `matplotlib.use(\"TkAgg\")` (or similar) if needed.
    """

    def __init__(
        self,
        *,
        title: str = "Park et al. 2021 — population (live)",
        xlabel: str = "Outer co-evolution iteration",
        ylabel: str = "Population (individuals)",
    ) -> None:
        import matplotlib.pyplot as plt

        self._plt = plt
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 4.5))
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, alpha=0.35)
        (self._line_p,) = self.ax.plot([], [], color="tab:red", lw=2.0, marker="o", markersize=4, label="Predators")
        (self._line_y,) = self.ax.plot([], [], color="tab:green", lw=2.0, marker="o", markersize=4, label="Prey")
        self.ax.legend(loc="upper right")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.show()

    def update(self, predators: list[int], prey: list[int]) -> None:
        x = list(range(1, len(predators) + 1))
        self._line_p.set_data(x, predators)
        self._line_y.set_data(x, prey)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._plt.pause(0.05)

    def close(self) -> None:
        self._plt.close(self.fig)
        self._plt.ioff()
