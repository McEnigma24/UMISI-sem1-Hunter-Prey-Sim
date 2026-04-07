from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sim.genome import TRAIT_COUNT, TRAIT_LABELS
from sim.world import World

# Same order as TRAIT_LABELS: paired hues (blue / red / combat / green / health).
SKILL_COLORS: tuple[str, ...] = (
    "#7EB8EE",  # vision_range — jaśniejszy niebieski
    "#2E5FA3",  # vision_focus — ciemniejszy niebieski
    "#A61E2D",  # speed — ciemniejszy czerwony
    "#F5B4B8",  # agility — jaśniejszy czerwony
    "#141414",  # attack — czarny
    "#A8B0C4",  # armor — stalowy / srebrny
    "#1B6E3A",  # stamina_max — ciemniejszy zielony
    "#8FE8A8",  # stamina_regen — jaśniejszy zielony
    "#E8A838",  # health — bursztyn
)


def _layout_tight(fig: plt.Figure) -> None:
    """Extra vertical gap between stacked plots."""
    fig.tight_layout(pad=2.8, h_pad=3.2, rect=(0.03, 0.02, 0.97, 0.95))


def save_history_csv(world: World, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["step", "plants", "prey", "hunters", "mutation_prey", "mutation_hunter"]
    for name in TRAIT_LABELS:
        header.append(f"prey_{name}")
    for name in TRAIT_LABELS:
        header.append(f"hunter_{name}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, t in enumerate(world.history_t):
            row: list[object] = [
                t,
                world.history_plants[i],
                world.history_prey[i],
                world.history_hunters[i],
                world.history_mutation_prey[i],
                world.history_mutation_hunter[i],
            ]
            for j in range(TRAIT_COUNT):
                row.append(f"{world.history_prey_mean_traits[i][j]:.4f}")
            for j in range(TRAIT_COUNT):
                row.append(f"{world.history_hunter_mean_traits[i][j]:.4f}")
            w.writerow(row)


def _skill_colors() -> tuple[str, ...]:
    """One color per skill — same in prey and hunter panels."""
    return SKILL_COLORS


def show_population_plot(world: World, title: str = "Hunter–Prey — history") -> None:
    if not world.history_t:
        return
    fig, (ax_pop, ax_prey, ax_hunt, ax_mut) = plt.subplots(4, 1, figsize=(11, 15.5), num=title)
    fig.suptitle(title, fontsize=12, y=0.98)
    _plot_body_count(ax_pop, world)
    _plot_traits_one_species(ax_prey, world, species="prey")
    _plot_traits_one_species(ax_hunt, world, species="hunter")
    _plot_mutation_gates(ax_mut, world)
    _layout_tight(fig)
    plt.show()


def _plot_body_count(ax, world: World) -> None:
    t = world.history_t
    ax.plot(t, world.history_prey, color="tab:blue", lw=2.0, label="Prey (count)")
    ax.plot(t, world.history_hunters, color="tab:red", lw=2.0, label="Hunters (count)")
    ax.set_ylabel("Population (individuals)")
    ax.set_xlabel("Step")
    ax.set_title("Body count")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right")


def _plot_traits_one_species(ax, world: World, *, species: str) -> None:
    t = world.history_t
    colors = _skill_colors()
    rows = (
        world.history_prey_mean_traits
        if species == "prey"
        else world.history_hunter_mean_traits
    )
    title = "Prey — mean traits" if species == "prey" else "Hunters — mean traits"
    ax.set_title(title)
    for i in range(TRAIT_COUNT):
        y = [row[i] for row in rows]
        ax.plot(t, y, color=colors[i], lw=1.5, label=TRAIT_LABELS[i])
    ax.set_ylabel("Mean raw stat (domain 0–1, autoscale y)")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.94)
    ax.relim()
    ax.autoscale_view()


def _mutation_phase_span_color(mh: int, mp: int) -> str | None:
    if mh:
        return "tab:red"
    if mp:
        return "tab:blue"
    return None


def _draw_mutation_phase_spans(ax, world: World, *, alpha: float = 0.38) -> None:
    """Alternating colored bands: red = hunter mutation window, blue = prey (mutually exclusive)."""
    for p in list(ax.patches):
        p.remove()
    t = world.history_t
    mp = world.history_mutation_prey
    mh = world.history_mutation_hunter
    n = len(t)
    if n == 0:
        return
    dx = (t[-1] - t[-2]) if n > 1 else 1.0
    for i in range(n):
        c = _mutation_phase_span_color(mh[i], mp[i])
        if c is None:
            continue
        x0 = float(t[i])
        x1 = float(t[i + 1]) if i + 1 < n else x0 + dx
        ax.axvspan(x0, x1, facecolor=c, alpha=alpha, linewidth=0, zorder=0)


def _plot_mutation_gates(ax, world: World) -> None:
    _draw_mutation_phase_spans(ax, world)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Step")
    ax.set_title("Mutation phases — shaded: red = hunters mutate, blue = prey mutate")
    ax.grid(True, alpha=0.35, zorder=1)
    ax.legend(
        handles=[
            Patch(facecolor="tab:red", alpha=0.45, label="Hunter mutation active"),
            Patch(facecolor="tab:blue", alpha=0.45, label="Prey mutation active"),
        ],
        loc="upper right",
        fontsize=8,
    )


class LiveCharts:
    """Non-blocking: body count + prey traits + hunter traits + mutation gates."""

    def __init__(self, title: str = "Hunter–Prey — live stats") -> None:
        plt.ion()
        self._title = title
        self.fig, (self.ax_pop, self.ax_prey, self.ax_hunt, self.ax_mut) = plt.subplots(
            4, 1, figsize=(11, 15.5), num=title
        )
        self.fig.suptitle(title, fontsize=12, y=0.98)

        (self._ln_prey_pop,) = self.ax_pop.plot(
            [], [], color="tab:blue", lw=2.0, label="Prey (count)"
        )
        (self._ln_hunt_pop,) = self.ax_pop.plot(
            [], [], color="tab:red", lw=2.0, label="Hunters (count)"
        )
        self.ax_pop.set_ylabel("Population (individuals)")
        self.ax_pop.set_xlabel("Step")
        self.ax_pop.set_title("Body count")
        self.ax_pop.grid(True, alpha=0.35)
        self.ax_pop.legend(loc="upper right")

        colors = _skill_colors()
        self._ln_prey_t: list = []
        self._ln_hunt_t: list = []
        for i in range(TRAIT_COUNT):
            (lp,) = self.ax_prey.plot([], [], color=colors[i], lw=1.45, label=TRAIT_LABELS[i])
            (lh,) = self.ax_hunt.plot([], [], color=colors[i], lw=1.45, label=TRAIT_LABELS[i])
            self._ln_prey_t.append(lp)
            self._ln_hunt_t.append(lh)

        self.ax_prey.set_ylabel("Mean raw stat (domain 0–1, autoscale y)")
        self.ax_prey.set_xlabel("Step")
        self.ax_prey.set_title("Prey — mean traits")
        self.ax_prey.grid(True, alpha=0.3)
        self.ax_prey.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.94)

        self.ax_hunt.set_ylabel("Mean raw stat (domain 0–1, autoscale y)")
        self.ax_hunt.set_xlabel("Step")
        self.ax_hunt.set_title("Hunters — mean traits")
        self.ax_hunt.grid(True, alpha=0.3)
        self.ax_hunt.legend(loc="upper left", fontsize=7, ncol=2, framealpha=0.94)

        self.ax_mut.set_ylim(0.0, 1.0)
        self.ax_mut.set_yticks([])
        self.ax_mut.set_xlabel("Step")
        self.ax_mut.set_title("Mutation phases — shaded: red = hunters, blue = prey")
        self.ax_mut.grid(True, alpha=0.35, zorder=1)
        self.ax_mut.legend(
            handles=[
                Patch(facecolor="tab:red", alpha=0.45, label="Hunter mutation active"),
                Patch(facecolor="tab:blue", alpha=0.45, label="Prey mutation active"),
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
        self._ln_hunt_pop.set_data(t, world.history_hunters)
        hp = world.history_prey_mean_traits
        hh = world.history_hunter_mean_traits
        for i in range(TRAIT_COUNT):
            yi = [row[i] for row in hp]
            self._ln_prey_t[i].set_data(t, yi)
            yh = [row[i] for row in hh]
            self._ln_hunt_t[i].set_data(t, yh)

        _draw_mutation_phase_spans(self.ax_mut, world)

        self.ax_pop.relim()
        self.ax_pop.autoscale_view()
        self.ax_prey.relim()
        self.ax_prey.autoscale_view()
        self.ax_hunt.relim()
        self.ax_hunt.autoscale_view()
        self.ax_mut.relim()
        self.ax_mut.autoscale_view()
        self.ax_mut.set_ylim(0.0, 1.0)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        if hasattr(self, "fig") and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
