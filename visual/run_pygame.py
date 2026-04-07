from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import pygame

from sim.config import SimConfig
from sim.entities import AGENT_HUNTER, AGENT_PREY
from sim.genome import VISION_RANGE
from sim.world import World
from visual.plots import LiveCharts, save_history_csv, show_population_plot

DELAY_PRESETS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.6)

# Layout: HUD text at top; grid uses the rest
_HUD_LINES_PX = 72
_PAD = 14
_MIN_CELL_PX = 2
_MAX_CELL_PX = 28


def _compute_window_for_grid(
    gw: int,
    gh: int,
    *,
    max_w: int,
    max_h: int,
) -> tuple[int, int]:
    """Window size so the grid fits with a reasonable cell size (capped by monitor)."""
    inner_w = max(1, max_w - 2 * _PAD)
    inner_h = max(1, max_h - _HUD_LINES_PX - _PAD)
    cell = min(
        _MAX_CELL_PX,
        inner_w // max(1, gw),
        inner_h // max(1, gh),
    )
    cell = max(_MIN_CELL_PX, cell)
    win_w = min(max_w, gw * cell + 2 * _PAD)
    win_h = min(max_h, gh * cell + _HUD_LINES_PX + _PAD)
    return max(320, win_w), max(240, win_h)


def run_pygame(
    cfg: SimConfig | None = None,
    window_size: tuple[int, int] | None = None,
    initial_delay_index: int = 0,
    initial_steps_per_frame: int = 1,
) -> None:
    cfg = cfg or SimConfig()
    world = World(cfg)

    live_charts: LiveCharts | None = None
    try:
        live_charts = LiveCharts()
    except Exception:
        live_charts = None

    last_live_update = 0.0

    pygame.init()
    info = pygame.display.Info()
    max_w = max(800, int(info.current_w * 0.96))
    max_h = max(600, int(info.current_h * 0.92))
    if window_size is None:
        win_w, win_h = _compute_window_for_grid(cfg.width, cfg.height, max_w=max_w, max_h=max_h)
    else:
        win_w, win_h = window_size
    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
    pygame.display.set_caption("Hunter–Prey ecosystem")

    clock = pygame.time.Clock()
    paused = False
    delay_idx = max(0, min(initial_delay_index, len(DELAY_PRESETS) - 1))
    step_delay_sec = DELAY_PRESETS[delay_idx]
    steps_per_frame = max(1, initial_steps_per_frame)
    running = True
    font = pygame.font.SysFont("consolas", 18)

    def update_caption() -> None:
        pygame.display.set_caption(
            f"Hunter–Prey {cfg.width}x{cfg.height} | step={world.step_index} delay={step_delay_sec}s spf={steps_per_frame}"
            + (" PAUSED" if paused else "")
        )

    def cell_pixel_size() -> tuple[int, int, int]:
        gw, gh = cfg.width, cfg.height
        sw, sh = screen.get_size()
        avail_w = sw - 2 * _PAD
        avail_h = sh - _HUD_LINES_PX - _PAD
        if avail_w < gw or avail_h < gh:
            cell = max(_MIN_CELL_PX, min(avail_w // max(1, gw), avail_h // max(1, gh)))
        else:
            cell = max(
                _MIN_CELL_PX,
                min(_MAX_CELL_PX, avail_w // gw, avail_h // gh),
            )
        grid_pw = cell * gw
        grid_ph = cell * gh
        ox = _PAD + (avail_w - grid_pw) // 2
        oy = _HUD_LINES_PX + (avail_h - grid_ph) // 2
        return cell, ox, oy

    def draw_world() -> None:
        nonlocal last_live_update
        screen.fill((24, 24, 28))
        cell, ox, oy = cell_pixel_size()
        gw, gh = cfg.width, cfg.height
        lo, hi = cfg.trait_bounds[VISION_RANGE]

        for y in range(gh):
            for x in range(gw):
                rect = pygame.Rect(ox + x * cell, oy + y * cell, cell, cell)
                pygame.draw.rect(screen, (32, 32, 38), rect)

        for y in range(gh):
            for x in range(gw):
                k = world.agent_kind[y][x]
                if k == 0:
                    continue
                rect = pygame.Rect(ox + x * cell, oy + y * cell, cell, cell)
                vr = world.traits[VISION_RANGE][y][x]
                t = (vr - lo) / max(1e-6, float(hi - lo))
                if k == AGENT_PREY:
                    col = (int(80 + 100 * t), int(140 + 80 * (1 - t)), 255)
                else:
                    col = (255, int(60 + 120 * (1 - t)), int(60 + 80 * t))
                inset = max(1, cell // 6)
                pygame.draw.rect(
                    screen,
                    col,
                    rect.inflate(-inset * 2, -inset * 2),
                    border_radius=max(1, cell // 8),
                )

        for y in range(gh):
            for x in range(gw):
                rect = pygame.Rect(ox + x * cell, oy + y * cell, cell, cell)
                pygame.draw.rect(screen, (50, 50, 55), rect, 1)

        pmv = world.mean_trait_prey(VISION_RANGE) if world.count_prey() else 0.0
        hmv = world.mean_trait_hunter(VISION_RANGE) if world.count_hunters() else 0.0
        hud = (
            f"SPACE pause | +/- delay | [/] spf | P snapshot plot | S save CSV | R reset | Q quit\n"
            f"prey={world.count_prey()} hunters={world.count_hunters()} | live charts in Matplotlib window | vision μ: {pmv:.2f} / {hmv:.2f}"
        )
        for i, line in enumerate(hud.split("\n")):
            surf = font.render(line, True, (220, 220, 220))
            screen.blit(surf, (_PAD, 8 + i * 20))
        pygame.display.flip()

        if live_charts is not None:
            now = time.monotonic()
            if now - last_live_update >= 0.12:
                live_charts.update(world)
                last_live_update = now

    update_caption()
    draw_world()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                nw = max(320, event.w)
                nh = max(240, event.h)
                screen = pygame.display.set_mode((nw, nh), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    update_caption()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    delay_idx = min(len(DELAY_PRESETS) - 1, delay_idx + 1)
                    step_delay_sec = DELAY_PRESETS[delay_idx]
                    update_caption()
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    delay_idx = max(0, delay_idx - 1)
                    step_delay_sec = DELAY_PRESETS[delay_idx]
                    update_caption()
                elif event.key == pygame.K_LEFTBRACKET:
                    steps_per_frame = max(1, steps_per_frame // 2)
                    update_caption()
                elif event.key == pygame.K_RIGHTBRACKET:
                    steps_per_frame = min(10_000, steps_per_frame * 2)
                    update_caption()
                elif event.key == pygame.K_r:
                    world.reset()
                    update_caption()
                elif event.key == pygame.K_p:
                    show_population_plot(world)
                elif event.key == pygame.K_s:
                    save_history_csv(world, Path(__file__).resolve().parent.parent / "plots" / "population_history.csv")

        if not paused:
            batch = steps_per_frame
            done = 0
            while done < batch:
                world.step()
                done += 1
                if done % 64 == 0:
                    pygame.event.pump()

        draw_world()

        if not paused and step_delay_sec > 0.0 and steps_per_frame == 1:
            time.sleep(step_delay_sec)

        clock.tick(60)

    if live_charts is not None:
        live_charts.close()
    pygame.quit()
