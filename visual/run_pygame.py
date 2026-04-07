from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import pygame

from sim.config import SimConfig
from sim.world import SPECIES_PREDATOR, SPECIES_PREY, World
from visual.plots import LiveCharts, save_history_csv, show_population_plot

DELAY_PRESETS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.6)

_HUD_LINES_PX = 72
_PAD = 14


def _compute_window(
    aspect: float,
    *,
    max_w: int,
    max_h: int,
) -> tuple[int, int]:
    inner_w = max(1, max_w - 2 * _PAD)
    inner_h = max(1, max_h - _HUD_LINES_PX - _PAD)
    if inner_w / inner_h > aspect:
        h = inner_h
        w = int(h * aspect)
    else:
        w = inner_w
        h = int(w / aspect)
    w = max(400, min(w, inner_w))
    h = max(300, min(h, inner_h))
    return max(400, w + 2 * _PAD), max(320, h + _HUD_LINES_PX + _PAD)


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
    aspect = cfg.width / max(1e-6, cfg.height)
    if window_size is None:
        win_w, win_h = _compute_window(aspect, max_w=max_w, max_h=max_h)
    else:
        win_w, win_h = window_size
    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
    pygame.display.set_caption("Hunter–Prey continuous 2D")

    clock = pygame.time.Clock()
    paused = False
    delay_idx = max(0, min(initial_delay_index, len(DELAY_PRESETS) - 1))
    step_delay_sec = DELAY_PRESETS[delay_idx]
    steps_per_frame = max(1, initial_steps_per_frame)
    running = True
    font = pygame.font.SysFont("consolas", 18)

    def update_caption() -> None:
        pygame.display.set_caption(
            f"Hunter–Prey continuous | step={world.step_index} delay={step_delay_sec}s spf={steps_per_frame}"
            + (" PAUSED" if paused else "")
        )

    def draw_world() -> None:
        nonlocal last_live_update
        screen.fill((20, 22, 28))
        sw, sh = screen.get_size()
        avail_w = sw - 2 * _PAD
        avail_h = sh - _HUD_LINES_PX - _PAD
        ox = _PAD
        oy = _HUD_LINES_PX
        W, H = cfg.width, cfg.height
        scale_x = avail_w / W
        scale_y = avail_h / H

        field_rect = pygame.Rect(ox, oy, avail_w, avail_h)
        pygame.draw.rect(screen, (32, 36, 44), field_rect)
        pygame.draw.rect(screen, (60, 64, 72), field_rect, 2)

        r_pre = max(2, int(min(scale_x, scale_y) * 0.35))
        r_pred = max(3, int(min(scale_x, scale_y) * 0.42))

        for a in world.agents:
            sx = ox + a.x * scale_x
            sy = oy + a.y * scale_y
            if a.species == SPECIES_PREY:
                col = (100, 160, 255)
                r = r_pre
            else:
                col = (255, 90, 90)
                r = r_pred
            pygame.draw.circle(screen, col, (int(sx), int(sy)), r)
            if abs(a.vx) + abs(a.vy) > 1e-3:
                vm = max(1e-6, (a.vx * a.vx + a.vy * a.vy) ** 0.5)
                vx, vy = a.vx / vm * 6.0, a.vy / vm * 6.0
                pygame.draw.line(
                    screen,
                    (200, 200, 200),
                    (int(sx), int(sy)),
                    (int(sx + vx * scale_x / 2), int(sy + vy * scale_y / 2)),
                    1,
                )

        ep = "on" if cfg.evolve_prey else "off"
        ed = "on" if cfg.evolve_predator else "off"
        hud = (
            "SPACE pause | +/- delay | [/] spf | P snapshot plot | S save CSV | R reset | Q quit\n"
            f"prey={world.count_prey()} predators={world.count_predators()} | "
            f"world {W:.0f}x{H:.0f} (torus) | evolve: prey={ep} pred={ed}"
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
                    save_history_csv(
                        world,
                        Path(__file__).resolve().parent.parent / "plots" / "population_history.csv",
                    )

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
