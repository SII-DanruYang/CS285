from typing import Callable, Optional, Sequence, Tuple
import copy

import torch
from torch import nn
import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
        # Actor-critic configuration
        actor_gradient_type: str = "reinforce",  # One of "reinforce" or "reparametrize"
        num_actor_samples: int = 1,
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        target_critic_backup_type: str = "mean",  # One of "doubleq", "min", "redq", or "mean"
        # Soft actor-critic
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
    ):
        super().__init__()

        assert target_critic_backup_type in [
            "doubleq",
            "min",
            "mean",
            "redq",
        ], f"{target_critic_backup_type} is not a valid target critic backup type"

        assert actor_gradient_type in [
            "reinforce",
            "reparametrize",
        ], f"{actor_gradient_type} is not a valid type of actor gradient update"

        assert (
            target_update_period is not None or soft_target_update_rate is not None
        ), "Must specify either target_update_period or soft_target_update_rate"

        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        self.critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.critic_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critic_lr_scheduler = make_critic_schedule(self.critic_optimizer)
        self.target_critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )
        self.update_target_critic()

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        self.target_update_period = target_update_period
        self.target_critic_backup_type = target_critic_backup_type
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.actor_gradient_type = actor_gradient_type
        self.num_actor_samples = num_actor_samples
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        with torch.no_grad():
            observation = ptu.from_numpy(observation)[None]

            action_distribution: torch.distributions.Distribution = self.actor(observation)
            action: torch.Tensor = action_distribution.sample()

            assert action.shape == (1, self.action_dim), action.shape
            return ptu.to_numpy(action).squeeze(0)

    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        return torch.stack([critic(obs, action) for critic in self.critics], dim=0)

    def target_critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        return torch.stack(
            [critic(obs, action) for critic in self.target_critics], dim=0
        )

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """Combine target Q-values from one or more target critics."""
        assert next_qs.ndim == 2, f"next_qs should have shape (num_critics, batch_size), got {next_qs.shape}"
        num_critic_networks, batch_size = next_qs.shape
        assert num_critic_networks == self.num_critic_networks

        if self.target_critic_backup_type == "doubleq":
            if self.num_critic_networks == 1:
                pass
            elif self.num_critic_networks == 2:
                next_qs = torch.stack([next_qs[1], next_qs[0]], dim=0)
            else:
                next_qs = torch.roll(next_qs, shifts=1, dims=0)

        elif self.target_critic_backup_type == "min":
            next_qs = torch.min(next_qs, dim=0).values

        elif self.target_critic_backup_type == "mean":
            next_qs = torch.mean(next_qs, dim=0)

        elif self.target_critic_backup_type == "redq":
            idx = torch.randint(low=0, high=self.num_critic_networks, size=(2, batch_size), device=next_qs.device)
            sampled_qs = torch.gather(next_qs, dim=0, index=idx)
            next_qs = torch.min(sampled_qs, dim=0).values

        if next_qs.shape == (batch_size,):
            next_qs = next_qs[None].expand((self.num_critic_networks, batch_size)).contiguous()

        assert next_qs.shape == (self.num_critic_networks, batch_size), next_qs.shape
        return next_qs

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """Update critic networks by minimizing Bellman error."""
        (batch_size,) = reward.shape

        with torch.no_grad():
            next_action_distribution: torch.distributions.Distribution = self.actor(next_obs)
            next_action = next_action_distribution.sample()
            assert next_action.shape == (batch_size, self.action_dim), next_action.shape

            next_qs = self.target_critic(next_obs, next_action)
            if next_qs.ndim == 3 and next_qs.shape[-1] == 1:
                next_qs = next_qs.squeeze(-1)
            assert next_qs.shape == (self.num_critic_networks, batch_size), next_qs.shape

            next_qs = self.q_backup_strategy(next_qs)
            assert next_qs.shape == (self.num_critic_networks, batch_size), next_qs.shape

            if self.use_entropy_bonus and self.backup_entropy:
                next_action_entropy = self.entropy(next_action_distribution)
                assert next_action_entropy.shape == (batch_size,), next_action_entropy.shape
                next_qs = next_qs + self.temperature * next_action_entropy[None, :]

            target_values = reward[None, :] + self.discount * (1 - done.float())[None, :] * next_qs
            assert target_values.shape == (self.num_critic_networks, batch_size), target_values.shape

        q_values = self.critic(obs, action)
        if q_values.ndim == 3 and q_values.shape[-1] == 1:
            q_values = q_values.squeeze(-1)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        loss = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        """Approximate entropy H(pi) = E[-log pi(a|s)]."""
        action = action_distribution.rsample()
        log_prob = action_distribution.log_prob(action)

        if log_prob.ndim > 1 and log_prob.shape[-1] == self.action_dim:
            log_prob = log_prob.sum(dim=-1)

        return -log_prob

    def actor_loss_reinforce(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        action_distribution: torch.distributions.Distribution = self.actor(obs)

        with torch.no_grad():
            action = action_distribution.sample((self.num_actor_samples,))
            assert action.shape == (self.num_actor_samples, batch_size, self.action_dim), action.shape

            obs_expanded = obs[None].expand(self.num_actor_samples, batch_size, *obs.shape[1:])
            obs_expanded = obs_expanded.reshape(self.num_actor_samples * batch_size, *obs.shape[1:])
            action_flat = action.reshape(self.num_actor_samples * batch_size, self.action_dim)

            q_values = self.critic(obs_expanded, action_flat)
            if q_values.ndim == 3 and q_values.shape[-1] == 1:
                q_values = q_values.squeeze(-1)
            q_values = q_values.reshape(self.num_critic_networks, self.num_actor_samples, batch_size)
            assert q_values.shape == (self.num_critic_networks, self.num_actor_samples, batch_size), q_values.shape

            q_values = torch.mean(q_values, dim=0)
            advantage = q_values

        log_probs = action_distribution.log_prob(action)
        if log_probs.ndim > 2 and log_probs.shape[-1] == self.action_dim:
            log_probs = log_probs.sum(dim=-1)
        assert log_probs.shape == (self.num_actor_samples, batch_size), log_probs.shape

        loss = -(log_probs * advantage).mean()

        return loss, torch.mean(self.entropy(action_distribution))

    def actor_loss_reparametrize(self, obs: torch.Tensor):
        batch_size = obs.shape[0]

        action_distribution: torch.distributions.Distribution = self.actor(obs)

        action = action_distribution.rsample()
        assert action.shape == (batch_size, self.action_dim), action.shape

        q_values = self.critic(obs, action)
        if q_values.ndim == 3 and q_values.shape[-1] == 1:
            q_values = q_values.squeeze(-1)
        assert q_values.shape == (self.num_critic_networks, batch_size), q_values.shape

        q_values = torch.mean(q_values, dim=0)
        loss = -q_values.mean()

        return loss, torch.mean(self.entropy(action_distribution))

    def update_actor(self, obs: torch.Tensor):
        """
        Update the actor by one gradient step using either REPARAMETRIZE or REINFORCE.
        """

        if self.actor_gradient_type == "reparametrize":
            loss, entropy = self.actor_loss_reparametrize(obs)
        elif self.actor_gradient_type == "reinforce":
            loss, entropy = self.actor_loss_reinforce(obs)

        # Add entropy if necessary
        if self.use_entropy_bonus:
            loss -= self.temperature * entropy

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": loss.item(), "entropy": entropy.item()}

    def update_target_critic(self):
        self.soft_update_target_critic(1.0)

    def soft_update_target_critic(self, tau):
        for target_critic, critic in zip(self.target_critics, self.critics):
            for target_param, param in zip(
                target_critic.parameters(), critic.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        """Update actor, critics, and target critics."""

        critic_infos = []

        for _ in range(self.num_critic_updates):
            critic_infos.append(
                self.update_critic(
                    observations,
                    actions,
                    rewards,
                    next_observations,
                    dones,
                )
            )

        actor_info = self.update_actor(observations)

        if self.target_update_period is not None:
            if step % self.target_update_period == 0:
                self.update_target_critic()

        if self.soft_target_update_rate is not None:
            self.soft_update_target_critic(self.soft_target_update_rate)

        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        return {
            **actor_info,
            **critic_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }

