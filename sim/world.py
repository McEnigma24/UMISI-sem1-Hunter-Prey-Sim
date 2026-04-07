"""Continuous torus hunter–prey: nearest-neighbor sensing, mixture policies, evolution."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from sim.config import SimConfig
from sim.geometry import torus_distance, wrap_xy
from sim.policy import (
    PolicyGenome,
    compute_desired_velocity_predator,
    compute_desired_velocity_prey,
    mean_policy_probs,
)

SPECIES_PREY = 0
SPECIES_PREDATOR = 1


@dataclass
class Agent:
    id: int
    species: int
    x: float
    y: float
    genome: PolicyGenome
    vx: float = 0.0
    vy: float = 0.0

    def heading(self) -> float:
        if abs(self.vx) < 1e-9 and abs(self.vy) < 1e-9:
            return 0.0
        return math.atan2(self.vy, self.vx)


class World:
    def __init__(self, cfg: SimConfig | None = None) -> None:
        self.cfg = cfg or SimConfig()
        self.rng = random.Random(self.cfg.rng_seed)
        self.agents: list[Agent] = []
        self._next_id = 1
        self.step_index = 0

        self.history_t: list[int] = []
        self.history_prey: list[int] = []
        self.history_predators: list[int] = []
        self.history_prey_mode_probs: list[list[float]] = []
        self.history_pred_mode_probs: list[list[float]] = []
        self.history_mutation_prey: list[int] = []
        self.history_mutation_predator: list[int] = []

        self._bootstrap()

    def reset(self) -> None:
        self.__init__(self.cfg)

    def _alloc_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _random_pos(self) -> tuple[float, float]:
        c = self.cfg
        return self.rng.uniform(0, c.width), self.rng.uniform(0, c.height)

    def _bootstrap(self) -> None:
        c = self.cfg
        for _ in range(c.initial_prey_count):
            x, y = self._random_pos()
            g = PolicyGenome.random(self.rng, c)
            self.agents.append(
                Agent(self._alloc_id(), SPECIES_PREY, x, y, genome=g)
            )
        for _ in range(c.initial_predator_count):
            x, y = self._random_pos()
            g = PolicyGenome.random(self.rng, c)
            self.agents.append(
                Agent(self._alloc_id(), SPECIES_PREDATOR, x, y, genome=g)
            )
        self._record_history_snapshot()

    def _prey_list(self) -> list[Agent]:
        return [a for a in self.agents if a.species == SPECIES_PREY]

    def _pred_list(self) -> list[Agent]:
        return [a for a in self.agents if a.species == SPECIES_PREDATOR]

    def _nearest_in(
        self, ax: float, ay: float, others: list[Agent]
    ) -> Agent | None:
        if not others:
            return None
        best: Agent | None = None
        best_d = float("inf")
        c = self.cfg
        for o in others:
            d = torus_distance(ax, ay, o.x, o.y, c.width, c.height)
            if d < best_d:
                best_d = d
                best = o
        return best

    def _mutation_phase_allows_at_step(self, species: int, step: int) -> bool:
        c = self.cfg
        if not c.mutation_phase_alternate or c.mutation_phase_steps <= 0:
            return True
        cycle = 2 * c.mutation_phase_steps
        pos = step % cycle
        predator_window = pos < c.mutation_phase_steps
        if species == SPECIES_PREDATOR:
            return predator_window
        return not predator_window

    def _mutation_phase_allows(self, species: int) -> bool:
        return self._mutation_phase_allows_at_step(species, self.step_index)

    def _species_evolution_enabled(self, species: int) -> bool:
        c = self.cfg
        if species == SPECIES_PREY:
            return c.evolve_prey
        return c.evolve_predator

    def _offspring_genome(self, parent: PolicyGenome, species: int) -> PolicyGenome:
        g = parent.clone()
        if not self._species_evolution_enabled(species):
            return g
        if not self._mutation_phase_allows(species):
            return g
        return g.mutate(self.rng, self.cfg)

    def _spawn_near_parent(self, parent: Agent) -> tuple[float, float]:
        c = self.cfg
        ang = self.rng.random() * 2.0 * math.pi
        r = self.rng.uniform(0.4 * c.offspring_spawn_radius, c.offspring_spawn_radius)
        return wrap_xy(
            parent.x + math.cos(ang) * r,
            parent.y + math.sin(ang) * r,
            c.width,
            c.height,
        )

    def _effective_mutation_gate_at_step(self, species: int, step: int) -> bool:
        """Whether offspring of this species can mutate on this step (evolution on + phase window)."""
        if not self._species_evolution_enabled(species):
            return False
        return self._mutation_phase_allows_at_step(species, step)

    def mutation_gates_at_step_index(self, step_index_at_step_start: int) -> tuple[int, int]:
        """
        (predator_gate, prey_gate) each 0/1 for the step() call that *begins* with
        world.step_index == step_index_at_step_start (must match _offspring_genome / phases).
        """
        mh = (
            1
            if self._effective_mutation_gate_at_step(
                SPECIES_PREDATOR, step_index_at_step_start
            )
            else 0
        )
        mp = (
            1
            if self._effective_mutation_gate_at_step(SPECIES_PREY, step_index_at_step_start)
            else 0
        )
        return mh, mp

    def step(self) -> None:
        c = self.cfg
        dt = c.dt
        phase = float(self.step_index)

        preys = self._prey_list()
        preds = self._pred_list()

        for a in self.agents:
            if a.species == SPECIES_PREY:
                target = self._nearest_in(a.x, a.y, preds)
                if target is None:
                    ang = self.rng.random() * 2.0 * math.pi
                    sp = c.prey_max_speed
                    a.vx, a.vy = math.cos(ang) * sp, math.sin(ang) * sp
                else:
                    a.vx, a.vy = compute_desired_velocity_prey(
                        a.genome,
                        a.x,
                        a.y,
                        target.x,
                        target.y,
                        c.width,
                        c.height,
                        a.heading(),
                        phase + 0.01 * float(a.id),
                        self.rng,
                        c,
                    )
            else:
                target = self._nearest_in(a.x, a.y, preys)
                if target is None:
                    ang = self.rng.random() * 2.0 * math.pi
                    sp = c.predator_max_speed
                    a.vx, a.vy = math.cos(ang) * sp, math.sin(ang) * sp
                else:
                    a.vx, a.vy = compute_desired_velocity_predator(
                        a.genome,
                        a.x,
                        a.y,
                        target.x,
                        target.y,
                        c.width,
                        c.height,
                        a.heading(),
                        phase + 0.01 * float(a.id),
                        self.rng,
                        c,
                    )

        for a in self.agents:
            a.x += a.vx * dt
            a.y += a.vy * dt
            a.x, a.y = wrap_xy(a.x, a.y, c.width, c.height)

        preys = self._prey_list()
        preds = self._pred_list()
        to_remove: set[int] = set()
        killers: list[Agent] = []

        for p in preys:
            best_h: Agent | None = None
            best_d = float("inf")
            for h in preds:
                d = torus_distance(p.x, p.y, h.x, h.y, c.width, c.height)
                if d < c.capture_radius and d < best_d:
                    best_d = d
                    best_h = h
            if best_h is not None:
                to_remove.add(p.id)
                killers.append(best_h)

        self.agents = [a for a in self.agents if a.id not in to_remove]

        for h in killers:
            if self.rng.random() < c.p_predator_birth_on_kill and len(self._pred_list()) < c.max_predators:
                child_g = self._offspring_genome(h.genome, SPECIES_PREDATOR)
                nx, ny = self._spawn_near_parent(h)
                self.agents.append(
                    Agent(self._alloc_id(), SPECIES_PREDATOR, nx, ny, genome=child_g)
                )

        preys = self._prey_list()
        preds = self._pred_list()

        if (
            self.rng.random() < c.p_prey_reproduce_per_step
            and len(preys) < c.max_prey
            and len(preys) > 0
        ):
            parent = self.rng.choice(preys)
            child_g = self._offspring_genome(parent.genome, SPECIES_PREY)
            x, y = self._spawn_near_parent(parent)
            self.agents.append(Agent(self._alloc_id(), SPECIES_PREY, x, y, genome=child_g))

        preys = self._prey_list()
        if (
            self.rng.random() < c.p_prey_spawn_random
            and len(preys) < c.max_prey
        ):
            if preys:
                parent = self.rng.choice(preys)
                child_g = self._offspring_genome(parent.genome, SPECIES_PREY)
            else:
                child_g = PolicyGenome.random(self.rng, c)
            rx, ry = self._random_pos()
            self.agents.append(Agent(self._alloc_id(), SPECIES_PREY, rx, ry, genome=child_g))

        preds = self._pred_list()
        if (
            self.rng.random() < c.p_predator_reproduce_per_step
            and len(preds) < c.max_predators
            and len(preds) > 0
        ):
            parent = self.rng.choice(preds)
            child_g = self._offspring_genome(parent.genome, SPECIES_PREDATOR)
            x, y = self._spawn_near_parent(parent)
            self.agents.append(
                Agent(self._alloc_id(), SPECIES_PREDATOR, x, y, genome=child_g)
            )

        self.step_index += 1
        self._record_history_snapshot()

    def count_prey(self) -> int:
        return sum(1 for a in self.agents if a.species == SPECIES_PREY)

    def count_predators(self) -> int:
        return sum(1 for a in self.agents if a.species == SPECIES_PREDATOR)

    def _record_history_snapshot(self) -> None:
        preys = self._prey_list()
        preds = self._pred_list()
        pl = [a.genome.prey_logits for a in preys]
        dl = [a.genome.pred_logits for a in preds]
        mp, md = mean_policy_probs(pl, dl)

        self.history_t.append(self.step_index)
        self.history_prey.append(len(preys))
        self.history_predators.append(len(preds))
        self.history_prey_mode_probs.append(mp)
        self.history_pred_mode_probs.append(md)

        # Interval [t[i], t[i+1]) on the plot = one step() that *starts* with step_index == t[i].
        mh, mp = self.mutation_gates_at_step_index(self.step_index)
        self.history_mutation_predator.append(mh)
        self.history_mutation_prey.append(mp)
