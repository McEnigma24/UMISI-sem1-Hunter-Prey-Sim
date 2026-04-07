"""Realtime Pygame view for Park et al. 2021 grid (predators=red, prey=green)."""

from __future__ import annotations

import time

import pygame

from park_env.constants import CELL_EMPTY, CELL_PREDATOR, CELL_PREY
from park_env.grid_env import ParkGridEnv, StepResult

_HUD_PX = 80
_PAD = 10
_MIN_CELL = 2
_MAX_CELL = 14


def _window_size(n: int, max_w: int, max_h: int) -> tuple[int, int, int]:
    inner_w = max(1, max_w - 2 * _PAD)
    inner_h = max(1, max_h - _HUD_PX - _PAD)
    cell = min(_MAX_CELL, inner_w // max(1, n), inner_h // max(1, n))
    cell = max(_MIN_CELL, cell)
    win_w = min(max_w, n * cell + 2 * _PAD)
    win_h = min(max_h, n * cell + _HUD_PX + _PAD)
    return max(320, win_w), max(260, win_h), cell


class ParkRealtimeViewer:
    """
    Draw `ParkGridEnv` after each env step. Call `tick` from the training rollout loop.

    Keys: **Esc** / close window → `quit_requested`; training should stop when `tick` returns False.
    **[** / **]** — wolniej / szybciej opóźnienie; **+** / **-** — co ile kroków rysować.
    """

    def __init__(
        self,
        grid_size: int,
        *,
        title: str = "Park et al. 2021 — predator–prey",
        delay_sec: float = 0.02,
        frame_every: int = 1,
    ) -> None:
        self.grid_size = grid_size
        self.delay_sec = max(0.0, float(delay_sec))
        self.frame_every = max(1, int(frame_every))
        self.quit_requested = False
        self._roll_step = 0

        pygame.init()
        pygame.display.set_caption(title)
        info = pygame.display.Info()
        max_w = max(640, int(info.current_w * 0.92))
        max_h = max(480, int(info.current_h * 0.88))
        win_w, win_h, cell = _window_size(grid_size, max_w=max_w, max_h=max_h)
        self._cell = cell
        self._screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
        self._font = pygame.font.Font(None, 22)
        self._font_small = pygame.font.Font(None, 18)

        self._bg = (24, 24, 28)
        self._empty = (45, 48, 55)
        self._pred = (210, 55, 55)
        self._prey = (65, 195, 85)

    def close(self) -> None:
        pygame.quit()

    def _pump_events(self) -> None:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.quit_requested = True
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.quit_requested = True
                elif e.key in (pygame.K_LEFTBRACKET, pygame.K_COMMA):
                    self.delay_sec = max(0.0, self.delay_sec - 0.01)
                elif e.key in (pygame.K_RIGHTBRACKET, pygame.K_PERIOD):
                    self.delay_sec = min(0.5, self.delay_sec + 0.01)
                elif e.key == pygame.K_MINUS or e.key == pygame.K_KP_MINUS:
                    self.frame_every = min(32, self.frame_every + 1)
                elif e.key == pygame.K_EQUALS or e.key == pygame.K_PLUS or e.key == pygame.K_KP_PLUS:
                    self.frame_every = max(1, self.frame_every - 1)
            elif e.type == pygame.VIDEORESIZE:
                w, h = max(320, e.w), max(260, e.h)
                self._screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)

    def tick(self, env: ParkGridEnv, step: StepResult, *, subtitle: str = "") -> bool:
        """
        Process events, optionally draw. Returns False if user closed window / Esc (stop training).
        """
        self._pump_events()
        if self.quit_requested:
            return False

        self._roll_step += 1
        if self._roll_step % self.frame_every != 0:
            return True

        self._draw(env, step, subtitle)
        pygame.display.flip()
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        return not self.quit_requested

    def _draw(self, env: ParkGridEnv, step: StepResult, subtitle: str) -> None:
        screen = self._screen
        n = self.grid_size
        sw, sh = screen.get_size()
        grid_px = min(sw - 2 * _PAD, sh - _HUD_PX - _PAD)
        cell = max(_MIN_CELL, grid_px // max(1, n))
        ox = (sw - n * cell) // 2
        oy = _PAD + (sh - _HUD_PX - _PAD - n * cell) // 2

        screen.fill(self._bg)
        occ = env.occupancy

        for gy in range(n):
            for gx in range(n):
                c = int(occ[gy, gx])
                if c == CELL_PREDATOR:
                    col = self._pred
                elif c == CELL_PREY:
                    col = self._prey
                else:
                    col = self._empty
                r = pygame.Rect(ox + gx * cell, oy + gy * cell, cell, cell)
                pygame.draw.rect(screen, col, r)
                if cell >= 6:
                    pygame.draw.rect(screen, (18, 18, 22), r, 1)

        lines = [
            f"Predators: {step.n_predators}   Prey: {step.n_prey}",
            f"delay={self.delay_sec:.2f}s  draw every {self.frame_every} step(s)   "
            f"[ ] delay  +/- skip   Esc quit",
        ]
        if subtitle:
            lines.insert(0, subtitle)
        y = 8
        for line in lines:
            surf = self._font.render(line, True, (230, 230, 235))
            screen.blit(surf, (_PAD, y))
            y += 24
        hint = self._font_small.render(
            "Green=prey, red=predator (Park et al. 2021 Fig. 1)", True, (160, 160, 170)
        )
        screen.blit(hint, (_PAD, sh - 22))
