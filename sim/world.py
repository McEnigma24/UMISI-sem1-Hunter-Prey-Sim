from __future__ import annotations

import random

from sim.builds import mutate_trait_random_step
from sim.config import SimConfig
from sim.entities import AGENT_HUNTER, AGENT_NONE, AGENT_PREY
from sim.genome import (
    AGILITY,
    ARMOR,
    ATTACK,
    Genome,
    SPEED,
    STAMINA_MAX,
    STAMINA_REGEN,
    TRAIT_COUNT,
    copy_traits_to_grid,
    read_genome_from_grids,
)
from sim.sensors import DIRS_4, facing_index, nearest_visible_agent
from sim.torus import step_toward_torus, torus_manhattan, wrap_xy

DIRS = DIRS_4


def _copy_i(g: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in g]


def _copy_f(g: list[list[float]]) -> list[list[float]]:
    return [row[:] for row in g]


class World:
    """
    Grid: max one agent per cell. No plants/carrion.
    Genomes: float trait grids per cell (only meaningful where agent_kind != NONE).
    Combat: symmetric attack vs armor threshold (no HP pool).
    """

    def __init__(self, cfg: SimConfig | None = None) -> None:
        self.cfg = cfg or SimConfig()
        self.rng = random.Random(self.cfg.rng_seed)
        w, h = self.cfg.width, self.cfg.height
        self.agent_kind: list[list[int]] = [[AGENT_NONE for _ in range(w)] for _ in range(h)]
        self.agent_id: list[list[int]] = [[0 for _ in range(w)] for _ in range(h)]
        self.agent_stamina: list[list[float]] = [[0.0 for _ in range(w)] for _ in range(h)]
        self.agent_age: list[list[int]] = [[0 for _ in range(w)] for _ in range(h)]
        self.agent_facing: list[list[int]] = [[0 for _ in range(w)] for _ in range(h)]
        self.agent_last_dx: list[list[int]] = [[0 for _ in range(w)] for _ in range(h)]
        self.agent_last_dy: list[list[int]] = [[0 for _ in range(w)] for _ in range(h)]
        self.agent_move_acc: list[list[float]] = [[0.0 for _ in range(w)] for _ in range(h)]
        self.traits: list[list[list[float]]] = [
            [[0.0 for _ in range(w)] for _ in range(h)] for _ in range(TRAIT_COUNT)
        ]

        self._next_uid = 1
        self.step_index = 0
        self.history_t: list[int] = []
        self.history_plants: list[int] = []
        self.history_prey: list[int] = []
        self.history_hunters: list[int] = []
        self.history_prey_mean_traits: list[list[float]] = []
        self.history_hunter_mean_traits: list[list[float]] = []
        self.history_mutation_prey: list[int] = []
        self.history_mutation_hunter: list[int] = []

        self._bootstrap_population()

    def _alloc_uid(self) -> int:
        u = self._next_uid
        self._next_uid += 1
        return u

    def _center_genome(self) -> Genome:
        return Genome.from_center(self.cfg)

    def _mutation_phase_allows_at_step(self, species: int, step: int) -> bool:
        """Whether `species` may mutate on simulation step index `step` (hunters first block, then prey)."""
        c = self.cfg
        if not c.mutation_phase_alternate or c.mutation_phase_steps <= 0:
            return True
        cycle = 2 * c.mutation_phase_steps
        pos = step % cycle
        hunter_window = pos < c.mutation_phase_steps
        if species == AGENT_HUNTER:
            return hunter_window
        return not hunter_window

    def _mutation_phase_allows(self, species: int) -> bool:
        """When True, offspring/spawn may apply random trait mutation; when False, exact copy (spawn still uses neighbor mean)."""
        return self._mutation_phase_allows_at_step(species, self.step_index)

    def _offspring_genome(self, parent: Genome, species: int) -> Genome:
        """Breeding / spawn after local mean: clone if mutation phase off, else random trait step."""
        g = parent.clone().clamped(self.cfg)
        if not self._mutation_phase_allows(species):
            return g
        return mutate_trait_random_step(g, self.rng, self.cfg).clamped(self.cfg)

    def _trait_norm_at(self, trait_idx: int, x: int, y: int) -> float:
        c = self.cfg
        lo, hi = c.trait_bounds[trait_idx]
        v = float(self.traits[trait_idx][y][x])
        return (v - lo) / max(1e-9, float(hi - lo))

    def _stamina_cap_at(self, x: int, y: int) -> float:
        c = self.cfg
        lvl = float(self.traits[STAMINA_MAX][y][x])
        lo, hi = c.trait_bounds[STAMINA_MAX]
        t = (lvl - lo) / max(1e-9, float(hi - lo))
        return c.stamina_base_max + t * float(hi - lo) * c.stamina_per_level

    def _stamina_regen_at(self, x: int, y: int) -> float:
        c = self.cfg
        lvl = float(self.traits[STAMINA_REGEN][y][x])
        lo, hi = c.trait_bounds[STAMINA_REGEN]
        t = (lvl - lo) / max(1e-9, float(hi - lo))
        return c.stamina_regen_base + t * float(hi - lo) * c.stamina_regen_per_level * 0.15

    def _stride_at(self, x: int, y: int) -> float:
        c = self.cfg
        lvl = float(self.traits[SPEED][y][x])
        lo, hi = c.trait_bounds[SPEED]
        t = (lvl - lo) / max(1e-9, float(hi - lo))
        base = c.speed_stride_min + t * (c.speed_stride_max - c.speed_stride_min)
        arm = self._trait_norm_at(ARMOR, x, y)
        base *= max(0.25, 1.0 - c.armor_speed_burden * arm)
        return base

    def _agility_effective_norm_at(self, x: int, y: int) -> float:
        c = self.cfg
        ag = self._trait_norm_at(AGILITY, x, y)
        arm = self._trait_norm_at(ARMOR, x, y)
        return max(0.0, ag * (1.0 - c.armor_agility_burden * arm))

    def _turn_stride_factor(self, ldx: int, ldy: int, dx: int, dy: int, x: int, y: int) -> float:
        c = self.cfg
        if ldx == 0 and ldy == 0:
            return 1.0
        if (dx, dy) == (ldx, ldy):
            return 1.0
        mit = c.agility_turn_mitigation * 10.0 * self._agility_effective_norm_at(x, y)
        if (dx, dy) == (-ldx, -ldy):
            pen = c.turn_penalty_180 * max(0.0, 1.0 - mit)
            return max(0.35, 1.0 - min(0.6, pen))
        pen = c.turn_penalty_90 * max(0.0, 1.0 - mit)
        return max(0.45, 1.0 - min(0.45, pen))

    def _bootstrap_population(self) -> None:
        c = self.cfg
        w, h = c.width, c.height
        area = w * h
        n = max(8, area // 80)
        g0 = self._center_genome()
        for _ in range(n):
            self._spawn_at_random(AGENT_PREY, g0.clone(), age=0)
        for _ in range(n):
            self._spawn_at_random(AGENT_HUNTER, g0.clone(), age=0)
        self._record_history_snapshot()

    def _spawn_at_random(self, kind: int, genome: Genome, age: int = 0) -> bool:
        w, h = self.cfg.width, self.cfg.height
        for _ in range(400):
            x, y = self.rng.randrange(w), self.rng.randrange(h)
            if self.agent_kind[y][x] == AGENT_NONE:
                uid = self._alloc_uid()
                self.agent_kind[y][x] = kind
                self.agent_id[y][x] = uid
                self.agent_move_acc[y][x] = 0.0
                copy_traits_to_grid(self.traits, x, y, genome)
                self.agent_stamina[y][x] = self._stamina_cap_at(x, y)
                self.agent_age[y][x] = age
                self.agent_facing[y][x] = self.rng.randrange(4)
                self.agent_last_dx[y][x] = 0
                self.agent_last_dy[y][x] = 0
                return True
        return False

    def _genome_for_new_hunter_at(self, x: int, y: int) -> Genome:
        """Mean trait vector of K nearest hunters (torus), else center; then optional mutation if phase allows."""
        c = self.cfg
        k = max(1, c.hunter_spawn_neighbor_count)
        local = self._mean_genome_k_nearest(x, y, k, AGENT_HUNTER)
        base = local if local is not None else self._center_genome()
        return self._offspring_genome(base, AGENT_HUNTER)

    def reset(self) -> None:
        self.__init__(self.cfg)

    def _step_toward(self, x: int, y: int, tx: int, ty: int) -> tuple[int, int]:
        w, h = self.cfg.width, self.cfg.height
        return step_toward_torus(x, y, tx, ty, w, h, self.rng)

    def _cell_blocked_prey(
        self, nk: list[list[int]], nid: list[list[int]], nx: int, ny: int, uid: int
    ) -> bool:
        w, h = self.cfg.width, self.cfg.height
        nx, ny = wrap_xy(nx, ny, w, h)
        k = nk[ny][nx]
        if k == AGENT_NONE:
            return False
        if k == AGENT_HUNTER:
            return True
        if k == AGENT_PREY:
            return nid[ny][nx] != uid
        return True

    def _best_flee_dir(
        self,
        nk: list[list[int]],
        nid: list[list[int]],
        ax: int,
        ay: int,
        hx: int,
        hy: int,
        uid: int,
    ) -> tuple[int, int] | None:
        w, h = self.cfg.width, self.cfg.height
        best: tuple[int, int] | None = None
        best_gain = -999
        d0 = torus_manhattan(ax, ay, hx, hy, w, h)
        for dx, dy in DIRS:
            nx, ny = wrap_xy(ax + dx, ay + dy, w, h)
            if self._cell_blocked_prey(nk, nid, nx, ny, uid):
                continue
            d1 = torus_manhattan(nx, ny, hx, hy, w, h)
            gain = d1 - d0
            if gain > best_gain:
                best_gain = gain
                best = (dx, dy)
        return best

    def _random_valid_dir_prey(
        self, nk: list[list[int]], nid: list[list[int]], ax: int, ay: int, uid: int
    ) -> tuple[int, int] | None:
        w, h = self.cfg.width, self.cfg.height
        opts: list[tuple[int, int]] = []
        for dx, dy in DIRS:
            tx, ty = wrap_xy(ax + dx, ay + dy, w, h)
            if not self._cell_blocked_prey(nk, nid, tx, ty, uid):
                opts.append((dx, dy))
        return self.rng.choice(opts) if opts else None

    def _random_valid_dir_hunter_empty(
        self, nk: list[list[int]], nx: int, ny: int
    ) -> tuple[int, int] | None:
        w, h = self.cfg.width, self.cfg.height
        opts: list[tuple[int, int]] = []
        for dx, dy in DIRS:
            tx, ty = wrap_xy(nx + dx, ny + dy, w, h)
            if nk[ty][tx] == AGENT_NONE:
                opts.append((dx, dy))
        return self.rng.choice(opts) if opts else None

    def _try_prey_breed(
        self,
        nk: list[list[int]],
        ne: list[list[float]],
        nid: list[list[int]],
        nacc: list[list[float]],
        cx: int,
        cy: int,
    ) -> None:
        c = self.cfg
        if self.rng.random() >= c.p_prey_breed:
            return
        w, h = c.width, c.height
        opts: list[tuple[int, int]] = []
        for ddx, ddy in DIRS:
            tx, ty = wrap_xy(cx + ddx, cy + ddy, w, h)
            if nk[ty][tx] == AGENT_NONE:
                opts.append((tx, ty))
        if not opts:
            return
        tx, ty = self.rng.choice(opts)
        g = read_genome_from_grids(self.traits, cx, cy)
        child = self._offspring_genome(g, AGENT_PREY)
        uid = self._alloc_uid()
        nk[ty][tx] = AGENT_PREY
        nid[ty][tx] = uid
        nacc[ty][tx] = 0.0
        copy_traits_to_grid(self.traits, tx, ty, child)
        ne[ty][tx] = self._stamina_cap_at(tx, ty)
        self.agent_age[ty][tx] = 0
        self.agent_facing[ty][tx] = self.rng.randrange(4)
        self.agent_last_dx[ty][tx] = 0
        self.agent_last_dy[ty][tx] = 0

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
        w, h = self.cfg.width, self.cfg.height
        nx, ny = wrap_xy(cx + dx, cy + dy, w, h)
        if self._cell_blocked_prey(nk, nid, nx, ny, uid):
            return False, cx, cy
        ldx, ldy = self.agent_last_dx[cy][cx], self.agent_last_dy[cy][cx]
        fac = self._turn_stride_factor(ldx, ldy, dx, dy, cx, cy)
        rem = nacc[cy][cx] - 1.0
        cost = self.cfg.move_cost_base * max(0.25, fac)
        e = ne[cy][cx] - cost
        if e <= 0:
            return False, cx, cy
        saved = tuple(self.traits[k][cy][cx] for k in range(TRAIT_COUNT))
        age = self.agent_age[cy][cx]
        nk[cy][cx] = AGENT_NONE
        ne[cy][cx] = 0.0
        nid[cy][cx] = 0
        nacc[cy][cx] = 0.0
        for k in range(TRAIT_COUNT):
            self.traits[k][cy][cx] = 0.0
        self.agent_age[cy][cx] = 0

        nk[ny][nx] = AGENT_PREY
        ne[ny][nx] = e
        nid[ny][nx] = uid
        nacc[ny][nx] = rem
        self.agent_age[ny][nx] = age
        for k in range(TRAIT_COUNT):
            self.traits[k][ny][nx] = saved[k]
        self.agent_facing[ny][nx] = facing_index(dx, dy)
        self.agent_last_dx[ny][nx], self.agent_last_dy[ny][nx] = dx, dy
        cap = self._stamina_cap_at(nx, ny)
        if ne[ny][nx] > cap:
            ne[ny][nx] = cap
        return True, nx, ny

    def _clear_cell(
        self,
        nk: list[list[int]],
        ne: list[list[float]],
        nid: list[list[int]],
        nacc: list[list[float]],
        x: int,
        y: int,
    ) -> None:
        nk[y][x] = AGENT_NONE
        ne[y][x] = 0.0
        nid[y][x] = 0
        nacc[y][x] = 0.0
        for k in range(TRAIT_COUNT):
            self.traits[k][y][x] = 0.0
        self.agent_age[y][x] = 0

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
    ) -> tuple[str, int, int]:
        """Returns (result, x, y)."""
        w, h = self.cfg.width, self.cfg.height
        nx, ny = wrap_xy(cx + dx, cy + dy, w, h)
        ldx, ldy = self.agent_last_dx[cy][cx], self.agent_last_dy[cy][cx]
        fac = self._turn_stride_factor(ldx, ldy, dx, dy, cx, cy)
        rem = nacc[cy][cx] - 1.0
        cost = self.cfg.move_cost_base * max(0.25, fac)
        e = ne[cy][cx]

        if nk[ny][nx] == AGENT_PREY:
            atk_h_n = self._trait_norm_at(ATTACK, cx, cy)
            atk_p_n = self._trait_norm_at(ATTACK, nx, ny)
            arm_h_n = self._trait_norm_at(ARMOR, cx, cy)
            arm_p_n = self._trait_norm_at(ARMOR, nx, ny)
            hunter_kills = atk_h_n > arm_p_n
            prey_kills = atk_p_n > arm_h_n

            if hunter_kills and prey_kills:
                self._clear_cell(nk, ne, nid, nacc, cx, cy)
                self._clear_cell(nk, ne, nid, nacc, nx, ny)
                return "both_dead", cx, cy

            if prey_kills and not hunter_kills:
                self._clear_cell(nk, ne, nid, nacc, cx, cy)
                return "hunter_died", cx, cy

            if hunter_kills and not prey_kills:
                meal = 28.0 + 0.4 * ne[ny][nx]
                hunter_traits = tuple(self.traits[k][cy][cx] for k in range(TRAIT_COUNT))
                self._clear_cell(nk, ne, nid, nacc, ny, nx)
                self._clear_cell(nk, ne, nid, nacc, cx, cy)

                nk[ny][nx] = AGENT_HUNTER
                new_e = e - cost + meal
                cap = self._stamina_cap_at(nx, ny)
                ne[ny][nx] = min(cap, new_e)
                nid[ny][nx] = uid
                nacc[ny][nx] = rem
                for k in range(TRAIT_COUNT):
                    self.traits[k][ny][nx] = hunter_traits[k]

                self.agent_facing[ny][nx] = facing_index(dx, dy)
                self.agent_last_dx[ny][nx], self.agent_last_dy[ny][nx] = dx, dy

                c = self.cfg
                if self.rng.random() < c.p_hunter_breed_on_kill:
                    opts: list[tuple[int, int]] = []
                    for ddx, ddy in DIRS:
                        tx, ty = wrap_xy(nx + ddx, ny + ddy, w, h)
                        if nk[ty][tx] == AGENT_NONE:
                            opts.append((tx, ty))
                    if opts:
                        bx, by = self.rng.choice(opts)
                        parent = read_genome_from_grids(self.traits, nx, ny)
                        child_g = self._offspring_genome(parent, AGENT_HUNTER)
                        nk[by][bx] = AGENT_HUNTER
                        nid[by][bx] = self._alloc_uid()
                        nacc[by][bx] = 0.0
                        copy_traits_to_grid(self.traits, bx, by, child_g)
                        ne[by][bx] = self._stamina_cap_at(bx, by)
                        self.agent_age[by][bx] = 0
                        self.agent_facing[by][bx] = self.rng.randrange(4)
                        self.agent_last_dx[by][bx] = 0
                        self.agent_last_dy[by][bx] = 0

                return "ate", nx, ny

            ne[cy][cx] = max(0.0, e - cost - self.cfg.failed_hunt_extra_cost)
            nacc[cy][cx] = rem
            if ne[cy][cx] <= 0:
                self._clear_cell(nk, ne, nid, nacc, cx, cy)
                return "failed_hunt", cx, cy
            return "failed_hunt", cx, cy

        if nk[ny][nx] == AGENT_HUNTER:
            return "blocked", cx, cy
        if nk[ny][nx] != AGENT_NONE:
            return "blocked", cx, cy

        new_e = e - cost
        saved = tuple(self.traits[k][cy][cx] for k in range(TRAIT_COUNT))
        nk[cy][cx] = AGENT_NONE
        ne[cy][cx] = 0.0
        nid[cy][cx] = 0
        nacc[cy][cx] = 0.0
        for k in range(TRAIT_COUNT):
            self.traits[k][cy][cx] = 0.0
        self.agent_age[cy][cx] = 0

        nk[ny][nx] = AGENT_HUNTER
        ne[ny][nx] = new_e
        nid[ny][nx] = uid
        nacc[ny][nx] = rem
        for k in range(TRAIT_COUNT):
            self.traits[k][ny][nx] = saved[k]

        self.agent_facing[ny][nx] = facing_index(dx, dy)
        self.agent_last_dx[ny][nx], self.agent_last_dy[ny][nx] = dx, dy
        cap_s = self._stamina_cap_at(nx, ny)
        if ne[ny][nx] > cap_s:
            ne[ny][nx] = cap_s
        return "moved", nx, ny

    def _prey_phase(self) -> None:
        c = self.cfg
        w, h = c.width, c.height
        sk = _copy_i(self.agent_kind)
        ssid = _copy_i(self.agent_id)

        nk = _copy_i(sk)
        ne = _copy_f(self.agent_stamina)
        nid = _copy_i(self.agent_id)
        nacc = _copy_f(self.agent_move_acc)

        positions = [(x, y) for y in range(h) for x in range(w) if sk[y][x] == AGENT_PREY]
        self.rng.shuffle(positions)

        for x, y in positions:
            sid = ssid[y][x]
            if nk[y][x] != AGENT_PREY or nid[y][x] != sid:
                continue

            self.agent_age[y][x] += 1
            if self.agent_age[y][x] > c.prey_max_age:
                self._clear_cell(nk, ne, nid, nacc, x, y)
                continue

            cap = self._stamina_cap_at(x, y)
            e = min(cap, ne[y][x] + self._stamina_regen_at(x, y))
            e -= c.idle_cost_prey
            if e < 1.0:
                e = 1.0
            ne[y][x] = e

            cx, cy = x, y
            uid = sid
            nacc[cy][cx] += self._stride_at(cx, cy)
            sub = 0
            while nacc[cy][cx] >= 1.0 - 1e-12 and sub < c.max_submoves_per_tick:
                sub += 1
                if nk[cy][cx] != AGENT_PREY or nid[cy][cx] != uid:
                    break
                fac_idx = self.agent_facing[cy][cx]
                threat = nearest_visible_agent(
                    cx, cy, fac_idx, sk, w, h, c, self.traits, AGENT_HUNTER
                )
                dx, dy = 0, 0
                if threat is not None:
                    hx, hy = threat
                    flee = self._best_flee_dir(nk, nid, cx, cy, hx, hy, uid)
                    if flee is not None:
                        dx, dy = flee
                if dx == 0 and dy == 0:
                    rd = self._random_valid_dir_prey(nk, nid, cx, cy, uid)
                    if rd is not None:
                        dx, dy = rd
                if dx == 0 and dy == 0:
                    break
                ok, cx, cy = self._prey_substep_walk(nk, ne, nid, nacc, cx, cy, dx, dy, uid)
                if not ok:
                    break

            if nk[cy][cx] == AGENT_PREY and nid[cy][cx] == uid:
                self._try_prey_breed(nk, ne, nid, nacc, cx, cy)

            if nk[cy][cx] == AGENT_PREY:
                cap = self._stamina_cap_at(cx, cy)
                if ne[cy][cx] > cap:
                    ne[cy][cx] = cap

        self.agent_kind = nk
        self.agent_stamina = ne
        self.agent_id = nid
        self.agent_move_acc = nacc

    def _hunter_phase(self) -> None:
        c = self.cfg
        w, h = c.width, c.height
        sk = _copy_i(self.agent_kind)
        ssid = _copy_i(self.agent_id)

        nk = _copy_i(sk)
        ne = _copy_f(self.agent_stamina)
        nid = _copy_i(self.agent_id)
        nacc = _copy_f(self.agent_move_acc)

        positions = [(x, y) for y in range(h) for x in range(w) if sk[y][x] == AGENT_HUNTER]
        self.rng.shuffle(positions)

        for x, y in positions:
            sid = ssid[y][x]
            if nk[y][x] != AGENT_HUNTER or nid[y][x] != sid:
                continue

            e = ne[y][x] - c.idle_cost_hunter
            ne[y][x] = e
            if ne[y][x] <= 0:
                self._clear_cell(nk, ne, nid, nacc, x, y)
                continue

            cap = self._stamina_cap_at(x, y)
            ne[y][x] = min(cap, ne[y][x] + self._stamina_regen_at(x, y))

            hx, hy = x, y
            uid = sid
            nacc[hy][hx] += self._stride_at(hx, hy)
            sub = 0
            while nacc[hy][hx] >= 1.0 - 1e-12 and sub < c.max_submoves_per_tick:
                sub += 1
                if nk[hy][hx] != AGENT_HUNTER or nid[hy][hx] != uid:
                    break
                fac_idx = self.agent_facing[hy][hx]
                prey_cell = nearest_visible_agent(
                    hx, hy, fac_idx, sk, w, h, c, self.traits, AGENT_PREY
                )
                dx, dy = 0, 0
                if prey_cell is not None:
                    px, py = prey_cell
                    dx, dy = self._step_toward(hx, hy, px, py)
                if dx == 0 and dy == 0:
                    rd = self._random_valid_dir_hunter_empty(nk, hx, hy)
                    if rd is not None:
                        dx, dy = rd
                if dx == 0 and dy == 0:
                    break
                res, hx, hy = self._hunter_substep(nk, ne, nid, nacc, hx, hy, dx, dy, uid)
                if res in ("blocked", "failed_hunt", "hunter_died", "both_dead"):
                    break

            if nk[hy][hx] == AGENT_HUNTER:
                cap = self._stamina_cap_at(hx, hy)
                if ne[hy][hx] > cap:
                    ne[hy][hx] = cap
                if ne[hy][hx] <= 0:
                    self._clear_cell(nk, ne, nid, nacc, hx, hy)

        self.agent_kind = nk
        self.agent_stamina = ne
        self.agent_id = nid
        self.agent_move_acc = nacc

    def _mean_genome_k_nearest(
        self, x: int, y: int, k: int, species: int
    ) -> Genome | None:
        """Arithmetic mean of traits over `k` nearest agents of `species` (torus Manhattan, d>0)."""
        c = self.cfg
        w, h = c.width, c.height
        cand: list[tuple[int, int, int]] = []
        for ty in range(h):
            for tx in range(w):
                if self.agent_kind[ty][tx] != species:
                    continue
                d = torus_manhattan(x, y, tx, ty, w, h)
                if d == 0:
                    continue
                cand.append((d, tx, ty))
        if not cand:
            return None
        cand.sort(key=lambda t: t[0])
        take = cand[: min(k, len(cand))]
        acc = [0.0] * TRAIT_COUNT
        for _, tx, ty in take:
            for i in range(TRAIT_COUNT):
                acc[i] += float(self.traits[i][ty][tx])
        n = len(take)
        bounds = c.trait_bounds
        out: list[float] = []
        for i in range(TRAIT_COUNT):
            v = acc[i] / n
            lo, hi = bounds[i]
            out.append(max(lo, min(hi, v)))
        return Genome(traits=tuple(out))

    def _spontaneous_prey_spawn(self) -> None:
        c = self.cfg
        w, h = c.width, c.height
        g0 = self._center_genome()
        k = max(1, c.prey_spawn_neighbor_count)
        for y in range(h):
            for x in range(w):
                if self.agent_kind[y][x] != AGENT_NONE:
                    continue
                if self.rng.random() >= c.p_prey_spawn_empty:
                    continue
                uid = self._alloc_uid()
                local = self._mean_genome_k_nearest(x, y, k, AGENT_PREY)
                base = local if local is not None else g0
                child = self._offspring_genome(base, AGENT_PREY)
                self.agent_kind[y][x] = AGENT_PREY
                self.agent_id[y][x] = uid
                self.agent_move_acc[y][x] = 0.0
                copy_traits_to_grid(self.traits, x, y, child)
                self.agent_stamina[y][x] = self._stamina_cap_at(x, y)
                self.agent_age[y][x] = 0
                self.agent_facing[y][x] = self.rng.randrange(4)
                self.agent_last_dx[y][x] = 0
                self.agent_last_dy[y][x] = 0

    def _spontaneous_hunter_spawn(self) -> None:
        c = self.cfg
        w, h = c.width, c.height
        for y in range(h):
            for x in range(w):
                if self.agent_kind[y][x] != AGENT_NONE:
                    continue
                if self.rng.random() >= c.p_hunter_spawn_empty:
                    continue
                uid = self._alloc_uid()
                child = self._genome_for_new_hunter_at(x, y)
                self.agent_kind[y][x] = AGENT_HUNTER
                self.agent_id[y][x] = uid
                self.agent_move_acc[y][x] = 0.0
                copy_traits_to_grid(self.traits, x, y, child)
                self.agent_stamina[y][x] = self._stamina_cap_at(x, y)
                self.agent_age[y][x] = 0
                self.agent_facing[y][x] = self.rng.randrange(4)
                self.agent_last_dx[y][x] = 0
                self.agent_last_dy[y][x] = 0

    def count_plants(self) -> int:
        return 0

    def count_prey(self) -> int:
        return sum(1 for row in self.agent_kind for v in row if v == AGENT_PREY)

    def count_hunters(self) -> int:
        return sum(1 for row in self.agent_kind for v in row if v == AGENT_HUNTER)

    def mean_trait_prey(self, trait_idx: int) -> float:
        s = 0.0
        n = 0
        for y in range(self.cfg.height):
            for x in range(self.cfg.width):
                if self.agent_kind[y][x] == AGENT_PREY:
                    s += float(self.traits[trait_idx][y][x])
                    n += 1
        return s / n if n else 0.0

    def mean_trait_hunter(self, trait_idx: int) -> float:
        s = 0.0
        n = 0
        for y in range(self.cfg.height):
            for x in range(self.cfg.width):
                if self.agent_kind[y][x] == AGENT_HUNTER:
                    s += float(self.traits[trait_idx][y][x])
                    n += 1
        return s / n if n else 0.0

    def _record_history_snapshot(self) -> None:
        self.history_t.append(self.step_index)
        self.history_plants.append(0)
        self.history_prey.append(self.count_prey())
        self.history_hunters.append(self.count_hunters())
        self.history_prey_mean_traits.append([self.mean_trait_prey(i) for i in range(TRAIT_COUNT)])
        self.history_hunter_mean_traits.append([self.mean_trait_hunter(i) for i in range(TRAIT_COUNT)])
        phase_step = 0 if self.step_index == 0 else self.step_index - 1
        self.history_mutation_prey.append(
            1 if self._mutation_phase_allows_at_step(AGENT_PREY, phase_step) else 0
        )
        self.history_mutation_hunter.append(
            1 if self._mutation_phase_allows_at_step(AGENT_HUNTER, phase_step) else 0
        )

    def step(self) -> None:
        self._prey_phase()
        self._hunter_phase()
        self._spontaneous_prey_spawn()
        self._spontaneous_hunter_spawn()
        self.step_index += 1
        self._record_history_snapshot()
