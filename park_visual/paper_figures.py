"""
Figure 3–5 style plots (Park et al. 2021, PMC8069842): population vs env step.

Paper styling: red = predator, blue = prey (Fig. 3); x-axis `step`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Match typical MDPI / paper line colors (prey blue, predator red)
COLOR_PREDATOR = "#d62728"
COLOR_PREY = "#1f77b4"


def save_step_trace_csv(
    pred: np.ndarray,
    prey: np.ndarray,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "n_predators", "n_prey"])
        for i in range(len(pred)):
            w.writerow([i, int(pred[i]), int(prey[i])])


def _style_axis(
    ax,
    *,
    xmax: float | None,
    ymax: float | None,
    show_legend: bool,
    ylabel: str,
) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    if show_legend:
        ax.legend(loc="upper right")


def plot_figure3_co_evolution(
    traces: list[tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    *,
    suptitle: str = "Figure 3 — co-evolution (trained policies)",
    xmax: float | None = 2000,
    ymax: float | None = 2500,
    row_labels: list[str] | None = None,
    figsize_width: float = 7.5,
) -> None:
    """Three (or fewer) stacked panels like the paper's Fig. 3."""
    import matplotlib.pyplot as plt

    n = len(traces)
    if n == 0:
        raise ValueError("no traces")
    fig_h = max(3.8, 3.6 * n)
    fig, axes = plt.subplots(n, 1, figsize=(figsize_width, fig_h), sharex=True)
    ax_list = [axes] if n == 1 else list(np.ravel(axes))
    for i, (pred, prey) in enumerate(traces):
        ax = ax_list[i]
        steps = np.arange(len(pred))
        ax.plot(steps, pred, color=COLOR_PREDATOR, lw=1.35, label="predator")
        ax.plot(steps, prey, color=COLOR_PREY, lw=1.35, label="prey")
        title = row_labels[i] if row_labels and i < len(row_labels) else f"seed panel {i + 1}"
        ax.set_title(title, fontsize=10)
        _style_axis(
            ax,
            xmax=xmax,
            ymax=ymax,
            show_legend=True,
            ylabel="population",
        )
    ax_list[-1].set_xlabel("step")
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_figure4_random(
    traces: list[tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    *,
    suptitle: str = "Figure 4 — random policies",
    xmax: float | None = 2000,
    ymax: float | None = 2500,
    row_labels: list[str] | None = None,
) -> None:
    """Same layout as Fig. 3 but for uniform-random actions."""
    plot_figure3_co_evolution(
        traces,
        out_path,
        suptitle=suptitle,
        xmax=xmax,
        ymax=ymax,
        row_labels=row_labels,
    )


def plot_figure5_random_then_trained(
    traces: list[tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    *,
    switch_step: int = 500,
    suptitle: str = "Figure 5 — random → trained at t",
    xmax: float | None = 2000,
    ymax: float | None = 2500,
    row_labels: list[str] | None = None,
) -> None:
    """Vertical line at `switch_step` (paper: first 500 steps random)."""
    import matplotlib.pyplot as plt

    n = len(traces)
    if n == 0:
        raise ValueError("no traces")
    fig_h = max(3.8, 3.6 * n)
    fig, axes = plt.subplots(n, 1, figsize=(7.5, fig_h), sharex=True)
    ax_list = [axes] if n == 1 else list(np.ravel(axes))
    for i, (pred, prey) in enumerate(traces):
        ax = ax_list[i]
        steps = np.arange(len(pred))
        ax.plot(steps, pred, color=COLOR_PREDATOR, lw=1.35, label="predator")
        ax.plot(steps, prey, color=COLOR_PREY, lw=1.35, label="prey")
        ax.axvline(switch_step, color="0.35", ls="--", lw=1.0, alpha=0.85)
        title = row_labels[i] if row_labels and i < len(row_labels) else f"seed panel {i + 1}"
        ax.set_title(title, fontsize=10)
        _style_axis(
            ax,
            xmax=xmax,
            ymax=ymax,
            show_legend=True,
            ylabel="population",
        )
    ax_list[-1].set_xlabel("step")
    full_title = f"{suptitle} = {switch_step}"
    fig.suptitle(full_title, fontsize=11, y=1.01)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
