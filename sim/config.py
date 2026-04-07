from dataclasses import dataclass, field


@dataclass
class SimConfig:
    width: int = 40
    height: int = 30

    # Per-trait inclusive bounds (raw stats). Attack & armor use a narrow band so they saturate early and selection pressures other traits.
    # Order: vision_range, vision_focus, speed, agility, attack, armor, stamina_max, stamina_regen, health
    trait_bounds: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.40, 0.60),
            (0.40, 0.60),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
        ]
    )

    mutation_prob: float = 0.35
    # One random trait ± this amount per mutation event (then clamped to trait_bounds).
    trait_mutation_step: float = 0.032

    prey_max_age: int = 180
    p_prey_breed: float = 0.004
    # Spontaneous spawn on empty cells (separate rolls). Prey rate higher so random fill favors prey over time.
    p_prey_spawn_empty: float = 0.00042
    p_hunter_spawn_empty: float = 0.00009
    # Spontaneous spawn: arithmetic mean of this many nearest same-species agents (torus Manhattan), then optional mutation.
    prey_spawn_neighbor_count: int = 5
    hunter_spawn_neighbor_count: int = 5
    # Alternating mutation windows: first `mutation_phase_steps` steps = hunter mutation on, prey off; next block = prey on, hunter off (repeat).
    mutation_phase_alternate: bool = True
    mutation_phase_steps: int = 100
    # Hunters get fewer mutation events than prey (breed-on-kill + rare spawn only); keep this a bit high.
    p_hunter_breed_on_kill: float = 0.58

    rng_seed: int | None = 42

    max_submoves_per_tick: int = 16

    # Higher idle/move costs → stamina & speed matter more before contact happens.
    idle_cost_prey: float = 1.05
    idle_cost_hunter: float = 1.12
    move_cost_base: float = 1.22
    failed_hunt_extra_cost: float = 2.75

    turn_penalty_90: float = 0.45
    turn_penalty_180: float = 0.9

    speed_stride_min: float = 0.55
    speed_stride_max: float = 1.35

    agility_turn_mitigation: float = 0.06
    # Heavy armor reduces effective agility and stride (both species, symmetric).
    armor_agility_burden: float = 0.78
    armor_speed_burden: float = 0.42

    # Vision: larger spread of reach (vr) and arc (vf) — scouts see threats earlier / narrower builds go farther.
    vision_reach_per_level: float = 1.12
    vision_reach_cells_coeff: float = 14.0
    vision_min_reach_cells: int = 1
    vision_arc_min_deg: float = 72.0
    vision_arc_max_deg: float = 360.0

    stamina_base_max: float = 52.0
    stamina_per_level: float = 10.0
    stamina_regen_base: float = 0.4
    stamina_regen_per_level: float = 0.52
    # Combat: symmetric threshold — kill if attacker_norm > target_armor_norm (no HP pool; same tick for both sides).

