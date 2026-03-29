from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from sim.config import SimConfig
from sim.entities import AGENT_HUNTER, AGENT_NONE, AGENT_PREY

DIRS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _copy_f(g: list[list[float]]) -> list[list[float]]:
    return [row[:] for row in g]


def _copy_i(g: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in g]


def _row_chunks(height: int, n_parts: int) -> list[tuple[int, int]]:
    if n_parts <= 1 or height <= 0:
        return [(0, height)]
    n_parts = min(n_parts, height)
    base, rem = divmod(height, n_parts)
    chunks: list[tuple[int, int]] = []
    y = 0
    for i in range(n_parts):
        sz = base + (1 if i < rem else 0)
        chunks.append((y, y + sz))
        y += sz
    return chunks


def _env_chunk_parallel(
    y0: int,
    y1: int,
    plants_in: list[list[float]],
    plants_out: list[list[float]],
    ce_in: list[list[float]],
    ca_in: list[list[int]],
    ce_out: list[list[float]],
    ca_out: list[list[int]],
    cfg: SimConfig,
    step_index: int,
    rng_seed: int | None,
) -> None:
    w = cfg.width
    sb = rng_seed if rng_seed is not None else 0

    def spawn_u(x: int, y: int) -> float:
        s = (sb * 0x9E3779B1 ^ step_index * 0x85EBCA77 ^ x * 0xC2B2AE3D ^ y * 0x165667B1) & 0xFFFFFFFF
        return random.Random(s).random()

    for y in range(y0, y1):
        for x in range(w):
            pin = plants_in[y][x]
            if pin > 0:
                plants_out[y][x] = pin
            else:
                raw_n = 0
                for dx, dy in DIRS_4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < cfg.height and plants_in[ny][nx] > 0:
                        raw_n += 1
                n = min(raw_n, cfg.max_neighbors_for_bonus)
                p = min(1.0, cfg.plant_spawn_base_prob + cfg.plant_neighbor_bonus * n)
                plants_out[y][x] = cfg.plant_energy_value if spawn_u(x, y) < p else 0.0

            e = ce_in[y][x]
            if e <= 0:
                ce_out[y][x] = 0.0
                ca_out[y][x] = 0
                continue
            age = ca_in[y][x] + 1
            ne = e
            if age > cfg.carrion_fresh_steps:
                ne *= cfg.carrion_decay_factor
            if ne < cfg.carrion_min_energy:
                ce_out[y][x] = 0.0
                ca_out[y][x] = 0
            else:
                ce_out[y][x] = ne
                ca_out[y][x] = age


class World:
    """
    Stan: na komórce max 1 roślina (float), max 1 padlina (float + wiek), max 1 agent (prey XOR hunter).
    Fazy prey / hunter: snapshot (read-only) → zapis do bufora next → podmiana referencji.
    """

    def __init__(self, cfg: SimConfig | None = None) -> None:
        self._shutdown_executor()
        self.cfg = cfg or SimConfig()
        self.rng = random.Random(self.cfg.rng_seed)
        w, h = self.cfg.width, self.cfg.height
        self.plants: list[list[float]] = [[0.0 for _ in range(w)] for _ in range(h)]
        self.carrion_energy: list[list[float]] = [[0.0 for _ in range(w)] for _ in range(h)]
        self.carrion_age: list[list[int]] = [[0 for _ in range(w)] for _ in range(h)]
        self.agent_kind: list[list[int]] = [[AGENT_NONE for _ in range(w)] for _ in range(h)]
        self.agent_energy: list[list[float]] = [[0.0 for _ in range(w)] for _ in range(h)]
        self.agent_id: list[list[int]] = [[0 for _ in range(w)] for _ in range(h)]
        self.agent_move_acc: list[list[float]] = [[0.0 for _ in range(w)] for _ in range(h)]
        self._next_uid = 1

        self._plants_next: list[list[float]] = [[0.0 for _ in range(w)] for _ in range(h)]
        self._carrion_energy_next: list[list[float]] = [[0.0 for _ in range(w)] for _ in range(h)]
        self._carrion_age_next: list[list[int]] = [[0 for _ in range(w)] for _ in range(h)]

        self._executor: ThreadPoolExecutor | None = None
        self._executor_workers: int | None = None

        self.step_index = 0
        self.history_t: list[int] = []
        self.history_plants: list[int] = []
        self.history_prey: list[int] = []
        self.history_hunters: list[int] = []

        self._bootstrap_population()

    def _bootstrap_population(self) -> None:
        c = self.cfg
        w, h = c.width, c.height
        area = w * h
        n_prey = max(8, area // 80)
        n_hunter = max(2, area // 400)
        for _ in range(n_prey):
            self._spawn_at_random(AGENT_PREY, c.prey_start_energy)
        for _ in range(n_hunter):
            self._spawn_at_random(AGENT_HUNTER, c.hunter_start_energy)
        for _ in range(area // 25):
            x, y = self.rng.randrange(w), self.rng.randrange(h)
            if self.plants[y][x] == 0 and self.agent_kind[y][x] == AGENT_NONE:
                self.plants[y][x] = c.plant_energy_value
        self._record_history_snapshot()

    def _alloc_uid(self) -> int:
        u = self._next_uid
        self._next_uid += 1
        return u

    def _spawn_at_random(self, kind: int, energy: float) -> bool:
        w, h = self.cfg.width, self.cfg.height
        for _ in range(200):
            x, y = self.rng.randrange(w), self.rng.randrange(h)
            if self.agent_kind[y][x] == AGENT_NONE:
                self.agent_kind[y][x] = kind
                self.agent_energy[y][x] = energy
                self.agent_id[y][x] = self._alloc_uid()
                self.agent_move_acc[y][x] = 0.0
                return True
        return False

    def reset(self) -> None:
        self.__init__(self.cfg)

    def _shutdown_executor(self) -> None:
        ex = getattr(self, "_executor", None)
        if ex is not None:
            ex.shutdown(wait=True)
        self._executor = None
        self._executor_workers = None

    def _effective_env_workers(self) -> int:
        c = self.cfg.parallel_env_threads
        if c == 1:
            return 1
        if c <= 0:
            c = min(8, os.cpu_count() or 4)
        return max(1, c)

    def _swap_env_buffers(self) -> None:
        self.plants, self._plants_next = self._plants_next, self.plants
        self.carrion_energy, self._carrion_energy_next = self._carrion_energy_next, self.carrion_energy
        self.carrion_age, self._carrion_age_next = self._carrion_age_next, self.carrion_age

    def _parallel_env_step(self) -> None:
        workers = self._effective_env_workers()
        h = self.cfg.height
        chunks = _row_chunks(h, workers)
        if self._executor is None or self._executor_workers != workers:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            self._executor = ThreadPoolExecutor(max_workers=workers)
            self._executor_workers = workers
        ex = self._executor
        futures = []
        for y0, y1 in chunks:
            futures.append(
                ex.submit(
                    _env_chunk_parallel,
                    y0,
                    y1,
                    self.plants,
                    self._plants_next,
                    self.carrion_energy,
                    self.carrion_age,
                    self._carrion_energy_next,
                    self._carrion_age_next,
                    self.cfg,
                    self.step_index,
                    self.cfg.rng_seed,
                )
            )
        for f in futures:
            f.result()
        self._swap_env_buffers()

    def _manhattan(self, x1: int, y1: int, x2: int, y2: int) -> int:
        return abs(x1 - x2) + abs(y1 - y2)

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.cfg.width and 0 <= y < self.cfg.height

    def _iter_disk(self, cx: int, cy: int, r: int) -> Iterable[tuple[int, int]]:
        w, h = self.cfg.width, self.cfg.height
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) + abs(dy) > r:
                    continue
                x, y = cx + dx, cy + dy
                if 0 <= x < w and 0 <= y < h:
                    yield x, y

    def _nearest_agent_cell(
        self, sk: list[list[int]], x: int, y: int, radius: int, kind: int
    ) -> tuple[int, int] | None:
        best = None
        best_d = radius + 1
        for px, py in self._iter_disk(x, y, radius):
            if sk[py][px] == kind:
                d = self._manhattan(x, y, px, py)
                if d < best_d:
                    best_d = d
                    best = (px, py)
        return best

    def _nearest_plant_cell(
        self, sp: list[list[float]], x: int, y: int, radius: int
    ) -> tuple[int, int] | None:
        best = None
        best_d = radius + 1
        for px, py in self._iter_disk(x, y, radius):
            if sp[py][px] > 0:
                d = self._manhattan(x, y, px, py)
                if d < best_d:
                    best_d = d
                    best = (px, py)
        return best

    def _nearest_carrion_cell(
        self, ce: list[list[float]], x: int, y: int, radius: int
    ) -> tuple[int, int] | None:
        best = None
        best_d = radius + 1
        for px, py in self._iter_disk(x, y, radius):
            if ce[py][px] > 0:
                d = self._manhattan(x, y, px, py)
                if d < best_d:
                    best_d = d
                    best = (px, py)
        return best

    def _count_plant_neighbors(self, p: list[list[float]], x: int, y: int) -> int:
        n = 0
        w, h = self.cfg.width, self.cfg.height
        for dx, dy in DIRS_4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and p[ny][nx] > 0:
                n += 1
        return n

    def _grow_plants(self) -> None:
        c = self.cfg
        w, h = c.width, c.height
        for y in range(h):
            for x in range(w):
                if self.plants[y][x] > 0:
                    continue
                raw_n = self._count_plant_neighbors(self.plants, x, y)
                n = min(raw_n, c.max_neighbors_for_bonus)
                p = min(1.0, c.plant_spawn_base_prob + c.plant_neighbor_bonus * n)
                if self.rng.random() < p:
                    self.plants[y][x] = c.plant_energy_value

    def _decay_carrion(self) -> None:
        c = self.cfg
        h, w = len(self.carrion_energy), len(self.carrion_energy[0])
        for y in range(h):
            for x in range(w):
                e = self.carrion_energy[y][x]
                if e <= 0:
                    self.carrion_age[y][x] = 0
                    continue
                self.carrion_age[y][x] += 1
                if self.carrion_age[y][x] > c.carrion_fresh_steps:
                    e *= c.carrion_decay_factor
                    self.carrion_energy[y][x] = e
                if self.carrion_energy[y][x] < c.carrion_min_energy:
                    self.carrion_energy[y][x] = 0.0
                    self.carrion_age[y][x] = 0

    def _deposit_carrion(
        self, ce: list[list[float]], ca: list[list[int]], x: int, y: int, amount: float
    ) -> None:
        if amount <= 0:
            return
        ce[y][x] += amount
        ca[y][x] = 0

    def _carrion_drop(self, kind: int, energy: float) -> float:
        c = self.cfg
        base = c.prey_start_energy if kind == AGENT_PREY else c.hunter_start_energy
        return c.meat_fraction_of_body * max(25.0, base * 0.4 + max(0.0, energy))

    def _step_toward(self, x: int, y: int, tx: int, ty: int) -> tuple[int, int]:
        dx = (1 if tx > x else -1) if tx != x else 0
        dy = (1 if ty > y else -1) if ty != y else 0
        if dx != 0 and dy != 0:
            if self.rng.random() < 0.5:
                dy = 0
            else:
                dx = 0
        return dx, dy

    def _cell_blocked_prey_dynamic(
        self, nk: list[list[int]], nid: list[list[int]], nx: int, ny: int, uid: int
    ) -> bool:
        """True = nie można wejść (kolizja z innym agentem w nk)."""
        if not self._in_bounds(nx, ny):
            return True
        k = nk[ny][nx]
        if k == AGENT_NONE:
            return False
        if k == AGENT_HUNTER:
            return True
        if k == AGENT_PREY:
            return nid[ny][nx] != uid
        return True

    def _best_flee_dir_dynamic(
        self,
        nk: list[list[int]],
        nid: list[list[int]],
        ax: int,
        ay: int,
        hx: int,
        hy: int,
        uid: int,
    ) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        best_gain = -999
        d0 = self._manhattan(ax, ay, hx, hy)
        for dx, dy in DIRS_4:
            nx, ny = ax + dx, ay + dy
            if self._cell_blocked_prey_dynamic(nk, nid, nx, ny, uid):
                continue
            d1 = self._manhattan(nx, ny, hx, hy)
            gain = d1 - d0
            if gain > best_gain:
                best_gain = gain
                best = (dx, dy)
        return best

    def _random_valid_dir_dynamic(
        self, nk: list[list[int]], nid: list[list[int]], ax: int, ay: int, uid: int
    ) -> tuple[int, int] | None:
        opts: list[tuple[int, int]] = []
        for dx, dy in DIRS_4:
            nx, ny = ax + dx, ay + dy
            if not self._cell_blocked_prey_dynamic(nk, nid, nx, ny, uid):
                opts.append((dx, dy))
        return self.rng.choice(opts) if opts else None

    def _random_valid_hunter_dynamic(
        self, nk: list[list[int]], nid: list[list[int]], ax: int, ay: int, uid: int
    ) -> tuple[int, int] | None:
        """Losowy krok tylko na puste pole (bez ataku — polowanie idzie przez AI)."""
        opts = []
        for dx, dy in DIRS_4:
            nx, ny = ax + dx, ay + dy
            if not self._in_bounds(nx, ny):
                continue
            if nk[ny][nx] != AGENT_NONE:
                continue
            opts.append((dx, dy))
        return self.rng.choice(opts) if opts else None

    def _prey_substep_walk(
        self,
        nk: list[list[int]],
        ne: list[list[float]],
        nid: list[list[int]],
        nacc: list[list[float]],
        cx: int,
        cy: int,
        dx: int,
        dy: int,
        uid: int,
    ) -> tuple[bool, int, int]:
        nx, ny = cx + dx, cy + dy
        if self._cell_blocked_prey_dynamic(nk, nid, nx, ny, uid):
            return False, cx, cy
        e = ne[cy][cx] - self.cfg.move_cost
        val = nacc[cy][cx]
        rem = val - 1.0
        nk[cy][cx] = AGENT_NONE
        ne[cy][cx] = 0.0
        nid[cy][cx] = 0
        nacc[cy][cx] = 0.0
        nk[ny][nx] = AGENT_PREY
        ne[ny][nx] = e
        nid[ny][nx] = uid
        nacc[ny][nx] = rem
        return True, nx, ny

    def _hunter_substep(
        self,
        nk: list[list[int]],
        ne: list[list[float]],
        nid: list[list[int]],
        nacc: list[list[float]],
        cx: int,
        cy: int,
        dx: int,
        dy: int,
        uid: int,
    ) -> tuple[bool, int, int]:
        nx, ny = cx + dx, cy + dy
        if not self._in_bounds(nx, ny):
            return False, cx, cy
        e = ne[cy][cx]
        val = nacc[cy][cx]
        rem = val - 1.0
        if nk[ny][nx] == AGENT_PREY:
            meal = max(15.0, ne[ny][nx] * 0.9)
            nk[ny][nx] = AGENT_NONE
            ne[ny][nx] = 0.0
            nid[ny][nx] = 0
            nacc[ny][nx] = 0.0
            nk[cy][cx] = AGENT_NONE
            ne[cy][cx] = 0.0
            nid[cy][cx] = 0
            nacc[cy][cx] = 0.0
            nk[ny][nx] = AGENT_HUNTER
            ne[ny][nx] = e + meal - self.cfg.move_cost
            nid[ny][nx] = uid
            nacc[ny][nx] = rem
            return True, nx, ny
        if nk[ny][nx] == AGENT_HUNTER and nid[ny][nx] != uid:
            return False, cx, cy
        if nk[ny][nx] != AGENT_NONE:
            return False, cx, cy
        new_e = e - self.cfg.move_cost
        nk[cy][cx] = AGENT_NONE
        ne[cy][cx] = 0.0
        nid[cy][cx] = 0
        nacc[cy][cx] = 0.0
        nk[ny][nx] = AGENT_HUNTER
        ne[ny][nx] = new_e
        nid[ny][nx] = uid
        nacc[ny][nx] = rem
        return True, nx, ny

    def _prey_phase(self) -> None:
        c = self.cfg
        w, h = c.width, c.height
        sp = _copy_f(self.plants)
        sce, sca = _copy_f(self.carrion_energy), _copy_i(self.carrion_age)
        sk = _copy_i(self.agent_kind)
        ssid = _copy_i(self.agent_id)

        np_ = _copy_f(sp)
        nce, nca = _copy_f(sce), _copy_i(sca)
        nk = _copy_i(sk)
        ne = _copy_f(self.agent_energy)
        nid = _copy_i(self.agent_id)
        nacc = _copy_f(self.agent_move_acc)

        positions = [(x, y) for y in range(h) for x in range(w) if sk[y][x] == AGENT_PREY]
        self.rng.shuffle(positions)

        for x, y in positions:
            sid = ssid[y][x]
            if nk[y][x] != AGENT_PREY or nid[y][x] != sid:
                continue
            e = ne[y][x] - c.idle_cost_prey
            if e <= 0:
                nk[y][x] = AGENT_NONE
                ne[y][x] = 0.0
                nid[y][x] = 0
                nacc[y][x] = 0.0
                self._deposit_carrion(nce, nca, x, y, self._carrion_drop(AGENT_PREY, e))
                continue
            ne[y][x] = e

            cx, cy = x, y
            uid = sid
            nacc[cy][cx] += c.prey_move_stride
            sub = 0
            while nacc[cy][cx] >= 1.0 - 1e-12 and sub < c.max_submoves_per_tick:
                sub += 1
                if nk[cy][cx] != AGENT_PREY or nid[cy][cx] != uid:
                    break
                vr = c.prey_vision_radius
                threat = self._nearest_agent_cell(sk, cx, cy, vr, AGENT_HUNTER)
                dx, dy = 0, 0
                if threat is not None:
                    hx, hy = threat
                    flee = self._best_flee_dir_dynamic(nk, nid, cx, cy, hx, hy, uid)
                    if flee is not None:
                        dx, dy = flee
                if dx == 0 and dy == 0:
                    tgt = self._nearest_plant_cell(sp, cx, cy, vr)
                    if tgt is not None:
                        tx, ty = tgt
                        dx, dy = self._step_toward(cx, cy, tx, ty)
                if dx == 0 and dy == 0:
                    rd = self._random_valid_dir_dynamic(nk, nid, cx, cy, uid)
                    if rd is not None:
                        dx, dy = rd
                if dx == 0 and dy == 0:
                    break
                ok, cx, cy = self._prey_substep_walk(nk, ne, nid, nacc, cx, cy, dx, dy, uid)
                if not ok:
                    break

            if nk[cy][cx] == AGENT_PREY and np_[cy][cx] > 0:
                ne[cy][cx] += np_[cy][cx]
                np_[cy][cx] = 0.0

            if nk[cy][cx] != AGENT_PREY:
                continue
            e = ne[cy][cx]
            if e <= 0:
                nk[cy][cx] = AGENT_NONE
                ne[cy][cx] = 0.0
                nid[cy][cx] = 0
                nacc[cy][cx] = 0.0
                self._deposit_carrion(nce, nca, cx, cy, self._carrion_drop(AGENT_PREY, e))
                continue

            if e >= c.prey_breed_threshold:
                opts = []
                for ddx, ddy in DIRS_4:
                    tx, ty = cx + ddx, cy + ddy
                    if self._in_bounds(tx, ty) and nk[ty][tx] == AGENT_NONE:
                        opts.append((tx, ty))
                if opts:
                    tx, ty = self.rng.choice(opts)
                    nk[ty][tx] = AGENT_PREY
                    ne[ty][tx] = c.prey_start_energy
                    nid[ty][tx] = self._alloc_uid()
                    nacc[ty][tx] = 0.0
                    ne[cy][cx] = e - c.prey_breed_cost

            e = ne[cy][cx]
            if nk[cy][cx] == AGENT_PREY and e <= 0:
                nk[cy][cx] = AGENT_NONE
                ne[cy][cx] = 0.0
                nid[cy][cx] = 0
                nacc[cy][cx] = 0.0
                self._deposit_carrion(nce, nca, cx, cy, self._carrion_drop(AGENT_PREY, e))

        self.plants, self.carrion_energy, self.carrion_age = np_, nce, nca
        self.agent_kind, self.agent_energy, self.agent_id = nk, ne, nid
        self.agent_move_acc = nacc

    def _hunter_phase(self) -> None:
        c = self.cfg
        w, h = c.width, c.height
        sce, sca = _copy_f(self.carrion_energy), _copy_i(self.carrion_age)
        sk, se = _copy_i(self.agent_kind), _copy_f(self.agent_energy)
        ssid = _copy_i(self.agent_id)

        np_, nce, nca = _copy_f(self.plants), _copy_f(sce), _copy_i(sca)
        nk, ne = _copy_i(sk), _copy_f(se)
        nid = _copy_i(self.agent_id)
        nacc = _copy_f(self.agent_move_acc)

        positions = [(x, y) for y in range(h) for x in range(w) if sk[y][x] == AGENT_HUNTER]
        self.rng.shuffle(positions)

        for x, y in positions:
            sid = ssid[y][x]
            if nk[y][x] != AGENT_HUNTER or nid[y][x] != sid:
                continue
            e = ne[y][x] - c.idle_cost_hunter
            if e <= 0:
                nk[y][x] = AGENT_NONE
                ne[y][x] = 0.0
                nid[y][x] = 0
                nacc[y][x] = 0.0
                self._deposit_carrion(nce, nca, x, y, self._carrion_drop(AGENT_HUNTER, e))
                continue

            ne[y][x] = e
            hx, hy = x, y
            uid = sid
            nacc[hy][hx] += c.hunter_move_stride
            sub = 0
            while nacc[hy][hx] >= 1.0 - 1e-12 and sub < c.max_submoves_per_tick:
                sub += 1
                if nk[hy][hx] != AGENT_HUNTER or nid[hy][hx] != uid:
                    break
                vr = c.hunter_vision_radius
                prey_cell = self._nearest_agent_cell(nk, hx, hy, vr, AGENT_PREY)
                dx, dy = 0, 0
                if prey_cell is not None:
                    px, py = prey_cell
                    dx, dy = self._step_toward(hx, hy, px, py)
                if dx == 0 and dy == 0:
                    cc = self._nearest_carrion_cell(nce, hx, hy, vr)
                    if cc is not None:
                        cx, cy = cc
                        dx, dy = self._step_toward(hx, hy, cx, cy)
                if dx == 0 and dy == 0:
                    rd = self._random_valid_hunter_dynamic(nk, nid, hx, hy, uid)
                    if rd is not None:
                        dx, dy = rd
                if dx == 0 and dy == 0:
                    break
                ok, hx, hy = self._hunter_substep(nk, ne, nid, nacc, hx, hy, dx, dy, uid)
                if not ok:
                    break

            if nk[hy][hx] == AGENT_HUNTER and nce[hy][hx] > 0:
                ne[hy][hx] += nce[hy][hx] * 0.95
                nce[hy][hx] = 0.0
                nca[hy][hx] = 0

            if nk[hy][hx] != AGENT_HUNTER:
                continue
            e = ne[hy][hx]
            if e <= 0:
                nk[hy][hx] = AGENT_NONE
                ne[hy][hx] = 0.0
                nid[hy][hx] = 0
                nacc[hy][hx] = 0.0
                self._deposit_carrion(nce, nca, hx, hy, self._carrion_drop(AGENT_HUNTER, e))
                continue

            if e >= c.hunter_breed_threshold:
                opts = []
                for ddx, ddy in DIRS_4:
                    tx, ty = hx + ddx, hy + ddy
                    if self._in_bounds(tx, ty) and nk[ty][tx] == AGENT_NONE:
                        opts.append((tx, ty))
                if opts:
                    tx, ty = self.rng.choice(opts)
                    nk[ty][tx] = AGENT_HUNTER
                    ne[ty][tx] = c.hunter_start_energy
                    nid[ty][tx] = self._alloc_uid()
                    nacc[ty][tx] = 0.0
                    ne[hy][hx] = e - c.hunter_breed_cost

            e = ne[hy][hx]
            if nk[hy][hx] == AGENT_HUNTER and e <= 0:
                nk[hy][hx] = AGENT_NONE
                ne[hy][hx] = 0.0
                nid[hy][hx] = 0
                nacc[hy][hx] = 0.0
                self._deposit_carrion(nce, nca, hx, hy, self._carrion_drop(AGENT_HUNTER, e))

        self.plants, self.carrion_energy, self.carrion_age = np_, nce, nca
        self.agent_kind, self.agent_energy, self.agent_id = nk, ne, nid
        self.agent_move_acc = nacc

    def count_plants(self) -> int:
        return sum(1 for row in self.plants for v in row if v > 0)

    def count_prey(self) -> int:
        return sum(1 for row in self.agent_kind for v in row if v == AGENT_PREY)

    def count_hunters(self) -> int:
        return sum(1 for row in self.agent_kind for v in row if v == AGENT_HUNTER)

    def _record_history_snapshot(self) -> None:
        self.history_t.append(self.step_index)
        self.history_plants.append(self.count_plants())
        self.history_prey.append(self.count_prey())
        self.history_hunters.append(self.count_hunters())

    def step(self) -> None:
        if self._effective_env_workers() <= 1:
            self._grow_plants()
            self._decay_carrion()
        else:
            self._parallel_env_step()

        self._prey_phase()
        self._hunter_phase()

        self.step_index += 1
        self._record_history_snapshot()
