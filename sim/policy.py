"""Evolvable mixture policies: softmax weights over discrete movement modes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from sim.config import SimConfig
from sim.geometry import angle_to, torus_delta, wrap_angle


PREY_MODES = 3
PREDATOR_MODES = 3


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    ex = [math.exp(x - m) for x in logits]
    s = sum(ex)
    return [e / s for e in ex]


def _norm(dx: float, dy: float) -> tuple[float, float]:
    h = math.hypot(dx, dy)
    if h < 1e-12:
        return 1.0, 0.0
    return dx / h, dy / h


@dataclass
class PolicyGenome:
    prey_logits: tuple[float, float, float]
    pred_logits: tuple[float, float, float]

    def clone(self) -> PolicyGenome:
        return PolicyGenome(
            prey_logits=self.prey_logits,
            pred_logits=self.pred_logits,
        )

    @classmethod
    def random(cls, rng: random.Random, cfg: SimConfig) -> PolicyGenome:
        s = cfg.genome_logit_span
        return cls(
            prey_logits=tuple(rng.uniform(-s, s) for _ in range(PREY_MODES)),
            pred_logits=tuple(rng.uniform(-s, s) for _ in range(PREDATOR_MODES)),
        )

    def mutate(self, rng: random.Random, cfg: SimConfig) -> PolicyGenome:
        m = cfg.mutation_sigma
        pl = [x + rng.gauss(0.0, m) for x in self.prey_logits]
        dl = [x + rng.gauss(0.0, m) for x in self.pred_logits]
        return PolicyGenome(
            prey_logits=tuple(pl),
            pred_logits=tuple(dl),
        )


def prey_mode_vectors(
    dx: float,
    dy: float,
    dist: float,
    phase: float,
    rng: random.Random,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Three unit-ish directions: flee, random, zigzag blend."""
    if dist < 1e-9:
        dx, dy = 1.0, 0.0
        dist = 1.0
    flee_x, flee_y = _norm(-dx, -dy)
    ang = rng.random() * 2.0 * math.pi
    rx, ry = math.cos(ang), math.sin(ang)
    # Zigzag: rotate flee toward perpendicular by phase
    px, py = -dy / dist, dx / dist
    c, s = math.cos(phase), math.sin(phase)
    zx, zy = _norm(flee_x * c + px * s, flee_y * c + py * s)
    return (flee_x, flee_y), (rx, ry), (zx, zy)


def predator_mode_vectors(
    dx: float,
    dy: float,
    dist: float,
    lead_angle: float,
    rng: random.Random,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Chase toward prey, lead (rotated chase), patrol random."""
    if dist < 1e-9:
        dx, dy = 1.0, 0.0
    cx, cy = _norm(dx, dy)
    ca, sa = math.cos(lead_angle), math.sin(lead_angle)
    lx, ly = _norm(cx * ca - cy * sa, cx * sa + cy * ca)
    ang = rng.random() * 2.0 * math.pi
    px, py = math.cos(ang), math.sin(ang)
    return (cx, cy), (lx, ly), (px, py)


def compute_desired_velocity_prey(
    genome: PolicyGenome,
    ax: float,
    ay: float,
    tx: float,
    ty: float,
    width: float,
    height: float,
    heading: float,
    step_phase: float,
    rng: random.Random,
    cfg: SimConfig,
) -> tuple[float, float]:
    dx, dy = torus_delta(ax, ay, tx, ty, width, height)
    dist = math.hypot(dx, dy)
    d_obs = min(cfg.perception_max_dist, max(0.0, dist + rng.gauss(0.0, cfg.obs_distance_noise_sigma)))
    phi = wrap_angle(angle_to(dx, dy) - heading + rng.gauss(0.0, cfg.obs_angle_noise_sigma))
    _ = d_obs, phi  # available for future richer policies
    phase = step_phase * cfg.prey_zigzag_frequency
    v0, v1, v2 = prey_mode_vectors(dx, dy, dist, phase, rng)
    pi = _softmax(list(genome.prey_logits))
    mx = pi[0] * v0[0] + pi[1] * v1[0] + pi[2] * v2[0]
    my = pi[0] * v0[1] + pi[1] * v1[1] + pi[2] * v2[1]
    mx, my = _norm(mx, my)
    sp = cfg.prey_max_speed
    return mx * sp, my * sp


def compute_desired_velocity_predator(
    genome: PolicyGenome,
    ax: float,
    ay: float,
    tx: float,
    ty: float,
    width: float,
    height: float,
    heading: float,
    step_phase: float,
    rng: random.Random,
    cfg: SimConfig,
) -> tuple[float, float]:
    dx, dy = torus_delta(ax, ay, tx, ty, width, height)
    dist = math.hypot(dx, dy)
    _ = min(cfg.perception_max_dist, max(0.0, dist + rng.gauss(0.0, cfg.obs_distance_noise_sigma)))
    _ = wrap_angle(angle_to(dx, dy) - heading + rng.gauss(0.0, cfg.obs_angle_noise_sigma))
    lead_angle = cfg.predator_lead_angle + 0.08 * math.sin(step_phase * 0.5)
    v0, v1, v2 = predator_mode_vectors(dx, dy, dist, lead_angle, rng)
    pi = _softmax(list(genome.pred_logits))
    mx = pi[0] * v0[0] + pi[1] * v1[0] + pi[2] * v2[0]
    my = pi[0] * v0[1] + pi[1] * v1[1] + pi[2] * v2[1]
    mx, my = _norm(mx, my)
    sp = cfg.predator_max_speed
    return mx * sp, my * sp


def mean_policy_probs(prey_logits_batch: list[tuple[float, float, float]], pred_logits_batch: list[tuple[float, float, float]]) -> tuple[list[float], list[float]]:
    """Mean softmax probabilities across populations (for plotting)."""
    if not prey_logits_batch:
        mp = [1.0 / PREY_MODES] * PREY_MODES
    else:
        acc = [0.0] * PREY_MODES
        for lg in prey_logits_batch:
            p = _softmax(list(lg))
            for i in range(PREY_MODES):
                acc[i] += p[i]
        mp = [a / len(prey_logits_batch) for a in acc]
    if not pred_logits_batch:
        md = [1.0 / PREDATOR_MODES] * PREDATOR_MODES
    else:
        acc = [0.0] * PREDATOR_MODES
        for lg in pred_logits_batch:
            p = _softmax(list(lg))
            for i in range(PREDATOR_MODES):
                acc[i] += p[i]
        md = [a / len(pred_logits_batch) for a in acc]
    return mp, md
