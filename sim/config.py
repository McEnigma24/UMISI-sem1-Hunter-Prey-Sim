from dataclasses import dataclass


@dataclass
class SimConfig:
    """Continuous torus world; policy evolution via logits over movement modes."""

    # World size (torus units)
    width: float = 100.0
    height: float = 100.0
    dt: float = 0.35

    rng_seed: int | None = 42

    initial_prey_count: int = 60
    initial_predator_count: int = 60

    prey_max_speed: float = 3.1
    predator_max_speed: float = 3.0

    capture_radius: float = 1.85

    # Perception (nearest opponent only); Gaussian noise on observations
    perception_max_dist: float = 200.0
    obs_distance_noise_sigma: float = 1.2
    obs_angle_noise_sigma: float = 0.12

    prey_zigzag_frequency: float = 0.25
    predator_lead_angle: float = 0.18

    # Policy genome: random logits in [-span, span]; mutation Gaussian sigma
    genome_logit_span: float = 1.0
    mutation_sigma: float = 0.15

    # If False, offspring keep parent's genome (no mutation) for that species.
    evolve_prey: bool = True
    evolve_predator: bool = True

    # Reproduction: mostly near parent; extra random-torus prey spawn to avoid predator wipeout.
    p_prey_reproduce_per_step: float = 0.35
    p_prey_spawn_random: float = 0.12
    p_predator_reproduce_per_step: float = 0.02
    offspring_spawn_radius: float = 4.0
    p_predator_birth_on_kill: float = 0.22
    max_prey: int = 400
    max_predators: int = 250

    # Alternating mutation windows (only matters when evolve_* is True for that species)
    mutation_phase_alternate: bool = True
    mutation_phase_steps: int = 100
