from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sim.policy import PREDATOR_MODES, PREY_MODES
from sim.world import World

PREY_MODE_LABELS = ("flee", "random", "zigzag")
PRED_MODE_LABELS = ("chase", "lead", "patrol")


def _layout_tight(fig: plt.Figure) -> None:
    fig.tight_layout(pad=2.8, h_pad=3.0, rect=(0.03, 0.02, 0.97, 0.95))


def _mutation_phase_span_color(mh: int, mp: int) -> str | None:
    if mh:
        return "tab:red"
    if mp:
        return "tab:blue"
    return None


def _draw_mutation_phase_spans(ax, world: World, *, alpha: float = 0.38) -> None:
    """Color [t[i], t[i+1]) by mutation gates for the step() that *starts* at step_index == int(t[i])."""
    for p in list(ax.patches):
        p.remove()
    t = world.history_t
    n = len(t)
    if n < 2:
        return
    for i in range(n - 1):
        k = int(t[i])
        mh, mp = world.mutation_gates_at_step_index(k)
        c = _mutation_phase_span_color(mh, mp)
        if c is None:
            continue
        x0 = float(t[i])
        x1 = float(t[i + 1])
        ax.axvspan(x0, x1, facecolor=c, alpha=alpha, linewidth=0, zorder=0)


def save_history_csv(world: World, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["step", "prey", "predators", "mutation_prey", "mutation_predator"]
    for name in PREY_MODE_LABELS:
        header.append(f"prey_mode_mean_{name}")
    for name in PRED_MODE_LABELS:
        header.append(f"pred_mode_mean_{name}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, t in enumerate(world.history_t):
            row: list[object] = [
                t,
                world.history_prey[i],
                world.history_predators[i],
                world.history_mutation_prey[i],
                world.history_mutation_predator[i],
            ]
            for j in range(PREY_MODES):
                row.append(f"{world.history_prey_mode_probs[i][j]:.6f}")
            for j in range(PREDATOR_MODES):
                row.append(f"{world.history_pred_mode_probs[i][j]:.6f}")
            w.writerow(row)


def show_population_plot(world: World, title: str = "Hunter–Prey — continuous 2D policy") -> None:
    if not world.history_t:
        return
    fig, (ax_pop, ax_pm, ax_dm, ax_mut) = plt.subplots(4, 1, figsize=(11, 15.5), num=title)
    fig.suptitle(title, fontsize=12, y=0.98)
    _plot_body_count(ax_pop, world)
    _plot_mode_probs(ax_pm, world, species="prey")
    _plot_mode_probs(ax_dm, world, species="predator")
    _plot_mutation_gates(ax_mut, world)
    _layout_tight(fig)
    plt.show()


def _plot_body_count(ax, world: World) -> None:
    t = world.history_t
    ax.plot(t, world.history_prey, color="tab:blue", lw=2.0, label="Prey")
    ax.plot(t, world.history_predators, color="tab:red", lw=2.0, label="Predators")
    ax.set_ylabel("Population")
    ax.set_xlabel("Step")
    ax.set_title("Population")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right")


def _plot_mode_probs(ax, world: World, *, species: str) -> None:
    t = world.history_t
    if species == "prey":
        rows = world.history_prey_mode_probs
        labels = PREY_MODE_LABELS
        title = "Prey — mean softmax mode weights"
        colors = ("#1f77b4", "#9467bd", "#2ca02c")
    else:
        rows = world.history_pred_mode_probs
        labels = PRED_MODE_LABELS
        title = "Predators — mean softmax mode weights"
        colors = ("#d62728", "#ff7f0e", "#8c564b")
    ax.set_title(title)
    n = PREY_MODES if species == "prey" else PREDATOR_MODES
    for j in range(n):
        y = [row[j] for row in rows]
        ax.plot(t, y, color=colors[j], lw=1.5, label=labels[j])
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=3)


def _plot_mutation_gates(ax, world: World) -> None:
    _draw_mutation_phase_spans(ax, world)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Step")
    ax.set_title(
        "Mutation phases (aligned to sim step start) — red = predators, blue = prey"
    )
    ax.grid(True, alpha=0.35, zorder=1)
    ax.legend(
        handles=[
            Patch(facecolor="tab:red", alpha=0.45, label="Predator mutation active"),
            Patch(facecolor="tab:blue", alpha=0.45, label="Prey mutation active"),
        ],
        loc="upper right",
        fontsize=8,
    )


class LiveCharts:
    """Non-blocking: population + policy mode means + mutation bands."""

    def __init__(self, title: str = "Hunter–Prey — live stats") -> None:
        plt.ion()
        self._title = title
        self.fig, (self.ax_pop, self.ax_pm, self.ax_dm, self.ax_mut) = plt.subplots(
            4, 1, figsize=(11, 15.5), num=title
        )
        self.fig.suptitle(title, fontsize=12, y=0.98)

        (self._ln_prey_pop,) = self.ax_pop.plot([], [], color="tab:blue", lw=2.0, label="Prey")
        (self._ln_pred_pop,) = self.ax_pop.plot([], [], color="tab:red", lw=2.0, label="Predators")
        self.ax_pop.set_ylabel("Population")
        self.ax_pop.set_xlabel("Step")
        self.ax_pop.set_title("Population")
        self.ax_pop.grid(True, alpha=0.35)
        self.ax_pop.legend(loc="upper right")

        self._ln_prey_m: list = []
        colors_p = ("#1f77b4", "#9467bd", "#2ca02c")
        for j in range(PREY_MODES):
            (ln,) = self.ax_pm.plot([], [], color=colors_p[j], lw=1.45, label=PREY_MODE_LABELS[j])
            self._ln_prey_m.append(ln)
        self.ax_pm.set_ylabel("Probability")
        self.ax_pm.set_ylim(0.0, 1.0)
        self.ax_pm.set_xlabel("Step")
        self.ax_pm.set_title("Prey — mean mode weights")
        self.ax_pm.grid(True, alpha=0.3)
        self.ax_pm.legend(loc="upper right", fontsize=8, ncol=3)

        self._ln_pred_m: list = []
        colors_d = ("#d62728", "#ff7f0e", "#8c564b")
        for j in range(PREDATOR_MODES):
            (ln,) = self.ax_dm.plot([], [], color=colors_d[j], lw=1.45, label=PRED_MODE_LABELS[j])
            self._ln_pred_m.append(ln)
        self.ax_dm.set_ylabel("Probability")
        self.ax_dm.set_ylim(0.0, 1.0)
        self.ax_dm.set_xlabel("Step")
        self.ax_dm.set_title("Predators — mean mode weights")
        self.ax_dm.grid(True, alpha=0.3)
        self.ax_dm.legend(loc="upper right", fontsize=8, ncol=3)

        self.ax_mut.set_ylim(0.0, 1.0)
        self.ax_mut.set_yticks([])
        self.ax_mut.set_xlabel("Step")
        self.ax_mut.set_title("Mutation phases (step start index on x-interval)")
        self.ax_mut.grid(True, alpha=0.35, zorder=1)
        self.ax_mut.legend(
            handles=[
                Patch(facecolor="tab:red", alpha=0.45, label="Predator mutation"),
                Patch(facecolor="tab:blue", alpha=0.45, label="Prey mutation"),
            ],
            loc="upper right",
            fontsize=8,
        )

        _layout_tight(self.fig)
        plt.show(block=False)
        self.fig.canvas.flush_events()

    def update(self, world: World) -> None:
        if not world.history_t:
            return
        if not plt.fignum_exists(self.fig.number):
            return
        t = world.history_t
        self._ln_prey_pop.set_data(t, world.history_prey)
        self._ln_pred_pop.set_data(t, world.history_predators)
        for j in range(PREY_MODES):
            yj = [row[j] for row in world.history_prey_mode_probs]
            self._ln_prey_m[j].set_data(t, yj)
        for j in range(PREDATOR_MODES):
            yj = [row[j] for row in world.history_pred_mode_probs]
            self._ln_pred_m[j].set_data(t, yj)

        _draw_mutation_phase_spans(self.ax_mut, world)

        self.ax_pop.relim()
        self.ax_pop.autoscale_view()
        self.ax_pm.relim()
        self.ax_pm.autoscale_view()
        self.ax_dm.relim()
        self.ax_dm.autoscale_view()
        self.ax_pm.set_ylim(0.0, 1.0)
        self.ax_dm.set_ylim(0.0, 1.0)
        self.ax_mut.relim()
        self.ax_mut.autoscale_view()
        self.ax_mut.set_ylim(0.0, 1.0)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        if hasattr(self, "fig") and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
