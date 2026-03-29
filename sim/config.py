from dataclasses import dataclass


@dataclass
class SimConfig:
    width: int = 40
    height: int = 30

    # Energy (kcal)
    prey_start_energy: float = 80.0
    hunter_start_energy: float = 120.0
    plant_energy_value: float = 35.0
    move_cost: float = 2.0
    idle_cost_prey: float = 1.5
    idle_cost_hunter: float = 5.0

    prey_breed_threshold: float = 140.0
    prey_breed_cost: float = 70.0

    hunter_breed_threshold: float = 200.0
    hunter_breed_cost: float = 100.0

    # Vision (Manhattan distance)
    prey_vision_radius: int = 4
    hunter_vision_radius: int = 8

    # Plants
    plant_spawn_base_prob: float = 0.012
    plant_neighbor_bonus: float = 0.08
    max_neighbors_for_bonus: int = 4

    # Carrion
    carrion_fresh_steps: int = 3
    carrion_decay_factor: float = 0.92
    carrion_min_energy: float = 0.5
    meat_fraction_of_body: float = 0.85

    rng_seed: int | None = 42

    # Równoległy krok środowiska (rośliny + padlina): read → next → swap. Agenty: fazy prey/hunter na snapshotach.
    parallel_env_threads: int = 0
