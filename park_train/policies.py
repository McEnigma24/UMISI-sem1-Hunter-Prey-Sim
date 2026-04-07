"""Shared MLP policies and Q-networks for discrete actions (Park et al. Sec. 2.2)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from park_env.constants import NUM_ACTIONS


def mlp(sizes: list[int], activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class DiscreteActor(nn.Module):
    """Categorical policy: two hidden layers (Sec. 2.2)."""

    def __init__(self, obs_dim: int, hidden1: int = 256, hidden2: int = 256) -> None:
        super().__init__()
        self.trunk = mlp([obs_dim, hidden1, hidden2, NUM_ACTIONS])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.forward(obs)
        return torch.distributions.Categorical(logits=logits)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a)

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        d = self.dist(obs)
        return d.log_prob(actions)


class DoubleQCritic(nn.Module):
    """State–action Q; actions as long indices (embedded via one-hot)."""

    def __init__(self, obs_dim: int, hidden1: int = 256, hidden2: int = 256) -> None:
        super().__init__()
        in_dim = obs_dim + NUM_ACTIONS
        self.q1 = mlp([in_dim, hidden1, hidden2, 1])
        self.q2 = mlp([in_dim, hidden1, hidden2, 1])

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a_oh = F.one_hot(actions.long(), num_classes=NUM_ACTIONS).float()
        x = torch.cat([obs, a_oh], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def soft_update(net: nn.Module, target: nn.Module, tau: float) -> None:
    for p, tp in zip(net.parameters(), target.parameters(), strict=True):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


def hard_update(net: nn.Module, target: nn.Module) -> None:
    soft_update(net, target, tau=1.0)


def discrete_entropy_target(num_actions: int = NUM_ACTIONS) -> float:
    """Target max-entropy ~ uniform over actions."""
    return 0.5 * math.log(float(num_actions))
