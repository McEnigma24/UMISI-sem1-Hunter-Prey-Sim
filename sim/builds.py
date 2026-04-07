"""Mutation: pick one trait at random and nudge it by ±trait_mutation_step (then clamp to bounds)."""

from __future__ import annotations

import random

from sim.config import SimConfig
from sim.genome import Genome, TRAIT_COUNT


def mutate_trait_random_step(genome: Genome, rng: random.Random, cfg: SimConfig) -> Genome:
    if rng.random() >= cfg.mutation_prob:
        return genome
    i = rng.randrange(TRAIT_COUNT)
    delta = cfg.trait_mutation_step if rng.random() < 0.5 else -cfg.trait_mutation_step
    t = list(genome.traits)
    t[i] += delta
    return Genome(traits=tuple(t)).clamped(cfg)
