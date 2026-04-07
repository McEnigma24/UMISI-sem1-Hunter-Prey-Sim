"""
Discrete-action AlgaeDICE in PyTorch, adapted from
`google-research/algae_dice/algae.py` (continuous → categorical policy).
Matches Park et al. f(x)=|x|^2/2 (exponent 2) and entropy regularization (Sec. 2.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from park_train.policies import (
    DiscreteActor,
    DoubleQCritic,
    discrete_entropy_target,
    hard_update,
    soft_update,
)


def _f(residual: torch.Tensor) -> torch.Tensor:
    return 0.5 * residual.pow(2)


def _fgrad(residual: torch.Tensor) -> torch.Tensor:
    return torch.clamp(residual, -50.0, 50.0)


class DiscreteAlgaeDICE(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        *,
        gamma: float = 0.99,
        tau: float = 0.005,
        algae_alpha: float = 1.0,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        learn_alpha: bool = True,
        target_entropy: float | None = None,
        actor_update_every: int = 2,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.algae_alpha = algae_alpha
        self.learn_alpha = learn_alpha
        self.target_entropy = (
            target_entropy if target_entropy is not None else discrete_entropy_target()
        )
        self.actor_update_every = actor_update_every

        self.actor = DiscreteActor(obs_dim)
        self.critic = DoubleQCritic(obs_dim)
        self.critic_target = DoubleQCritic(obs_dim)
        hard_update(self.critic, self.critic_target)

        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.actor_opt = optim.AdamW(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.AdamW(self.critic.parameters(), lr=critic_lr)
        self.alpha_opt = optim.AdamW([self.log_alpha], lr=alpha_lr)

        self._critic_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def critic_mix(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            tq1, tq2 = self.critic_target(obs, actions)
        q1, q2 = self.critic(obs, actions)
        mix1 = 0.05 * q1 + 0.95 * tq1
        mix2 = 0.05 * q2 + 0.95 * tq2
        return mix1, mix2

    def fit_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        init_states: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            d_next = self.actor.dist(next_states)
            next_actions = d_next.sample()
            next_log_probs = d_next.log_prob(next_actions)
            d0 = self.actor.dist(init_states)
            init_actions = d0.sample()

            target_q1, target_q2 = self.critic_mix(next_states, next_actions)
            target_q1 = rewards + self.gamma * masks * (
                target_q1 - self.alpha * next_log_probs
            )
            target_q2 = rewards + self.gamma * masks * (
                target_q2 - self.alpha * next_log_probs
            )
            target_q1 = target_q1.detach()
            target_q2 = target_q2.detach()

        q1, q2 = self.critic(states, actions)
        init_q1, init_q2 = self.critic(init_states, init_actions.detach())

        term = 1.0 - self.gamma
        loss1 = (_f(target_q1 - q1) + term * self.algae_alpha * init_q1).mean()
        loss2 = (_f(target_q2 - q2) + term * self.algae_alpha * init_q2).mean()
        loss = 0.5 * (loss1 + loss2)

        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        return loss.detach()

    def fit_actor(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        init_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_next = self.actor.dist(next_states)
        next_actions = d_next.sample()
        next_log_probs = d_next.log_prob(next_actions)

        d0 = self.actor.dist(init_states)
        init_actions = d0.sample()

        target_q1, target_q2 = self.critic_mix(next_states, next_actions)
        target_q1 = rewards + self.gamma * masks * (
            target_q1 - self.alpha.detach() * next_log_probs
        )
        target_q2 = rewards + self.gamma * masks * (
            target_q2 - self.alpha.detach() * next_log_probs
        )

        q1, q2 = self.critic(states, actions)
        init_q1, init_q2 = self.critic(init_states, init_actions)

        term = 1.0 - self.gamma
        r1 = target_q1 - q1
        r2 = target_q2 - q2
        actor_loss1 = -(_fgrad(r1).detach() * r1 + term * self.algae_alpha * init_q1).mean()
        actor_loss2 = -(_fgrad(r2).detach() * r2 + term * self.algae_alpha * init_q2).mean()
        actor_loss = 0.5 * (actor_loss1 + actor_loss2)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = torch.tensor(0.0, device=states.device)
        if self.learn_alpha:
            d_a = self.actor.dist(next_states)
            samp = d_a.sample()
            nlp = d_a.log_prob(samp)
            alpha_loss = (self.alpha * (-nlp - self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        ent = (-next_log_probs.detach()).mean()
        return actor_loss.detach(), alpha_loss.detach(), ent.detach()

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ) -> dict[str, float]:
        """Args: s, a, s', r, mask (mask=1 if not terminal)."""
        init_states = states
        c_loss = self.fit_critic(states, actions, next_states, rewards, masks, init_states)
        self._critic_steps += 1
        out: dict[str, float] = {"critic_loss": float(c_loss.item())}
        if self._critic_steps % self.actor_update_every == 0:
            a_loss, al_loss, ent = self.fit_actor(
                states, actions, next_states, rewards, masks, init_states
            )
            soft_update(self.critic, self.critic_target, self.tau)
            out["actor_loss"] = float(a_loss.item())
            out["alpha_loss"] = float(al_loss.item())
            out["entropy"] = float(ent.item())
        return out
