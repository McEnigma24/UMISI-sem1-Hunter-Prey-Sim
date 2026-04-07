"""Replay buffers for predator / prey transitions."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim: int, capacity: int = 500_000) -> None:
        self.obs_dim = obs_dim
        self.capacity = capacity
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.m = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s2: np.ndarray,
        mask: float,
    ) -> None:
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.m[i] = mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        if self.size == 0:
            raise ValueError("empty buffer")
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.s[idx],
            self.a[idx],
            self.r[idx],
            self.s2[idx],
            self.m[idx],
        )
