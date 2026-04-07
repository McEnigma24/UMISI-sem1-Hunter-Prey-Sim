"""Continuous trait genome (floats in trait_bounds); evolution via random single-trait steps in sim/builds."""

from __future__ import annotations

from dataclasses import dataclass

from sim.config import SimConfig

TRAIT_COUNT = 9

VISION_RANGE = 0
VISION_FOCUS = 1
SPEED = 2
AGILITY = 3
ATTACK = 4
ARMOR = 5
STAMINA_MAX = 6
STAMINA_REGEN = 7
HEALTH = 8

TRAIT_LABELS: tuple[str, ...] = (
    "vision_range",
    "vision_focus",
    "speed",
    "agility",
    "attack",
    "armor",
    "stamina_max",
    "stamina_regen",
    "health",
)


@dataclass
class Genome:
    traits: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.traits) != TRAIT_COUNT:
            raise ValueError(f"Genome must have {TRAIT_COUNT} traits, got {len(self.traits)}")

    @classmethod
    def from_center(cls, cfg: SimConfig) -> Genome:
        t = tuple((float(a) + float(b)) * 0.5 for a, b in cfg.trait_bounds)
        return cls(traits=t)

    def clone(self) -> Genome:
        return Genome(traits=self.traits)

    def clamped(self, cfg: SimConfig) -> Genome:
        t = []
        for i, v in enumerate(self.traits):
            lo, hi = cfg.trait_bounds[i]
            t.append(max(lo, min(hi, float(v))))
        return Genome(traits=tuple(t))


def copy_traits_to_grid(
    traits_out: list[list[list[float]]],
    x: int,
    y: int,
    g: Genome,
) -> None:
    for k in range(TRAIT_COUNT):
        traits_out[k][y][x] = g.traits[k]


def read_genome_from_grids(traits: list[list[list[float]]], x: int, y: int) -> Genome:
    return Genome(traits=tuple(float(traits[k][y][x]) for k in range(TRAIT_COUNT)))
