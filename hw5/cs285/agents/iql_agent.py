from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn

from cs285.agents.awac_agent import AWACAgent


class IQLAgent(AWACAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_value_critic: Callable[[Tuple[int, ...]], nn.Module],
        make_value_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        expectile: float,
        **kwargs
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )

        self.value_critic = make_value_critic(observation_shape)
        self.target_value_critic = make_value_critic(observation_shape)
        self.target_value_critic.load_state_dict(self.value_critic.state_dict())

        self.value_critic_optimizer = make_value_critic_optimizer(
            self.value_critic.parameters()
        )
        self.expectile = expectile

    @staticmethod
    def _squeeze_value(values: torch.Tensor) -> torch.Tensor:
        """
        Convert value-network output to shape (batch_size,).
        Some networks return (B,), while others return (B, 1).
        """
        if values.ndim == 2 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        return values

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        """
        IQL advantage used for the actor update:
            A(s, a_data) = Q(s, a_data) - V(s)

        The actor update itself is inherited from AWAC:
            - log pi(a_data | s) * exp(A / temperature)
        """
        batch_size = observations.shape[0]
        actions = actions.long()

        with torch.no_grad():
            qa_values = self.critic(observations)
            assert qa_values.shape == (batch_size, self.num_actions), qa_values.shape

            q_values = torch.gather(
                qa_values,
                dim=1,
                index=actions.unsqueeze(1),
            ).squeeze(1)
            assert q_values.shape == (batch_size,), q_values.shape

            vs = self._squeeze_value(self.value_critic(observations))
            assert vs.shape == (batch_size,), vs.shape

            advantages = q_values - vs
            assert advantages.shape == (batch_size,), advantages.shape

        return advantages

    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a) with the IQL critic target:
            target = r + gamma * (1 - done) * V(s')
        """
        batch_size = observations.shape[0]
        actions = actions.long()

        with torch.no_grad():
            next_vs = self._squeeze_value(self.target_value_critic(next_observations))
            assert next_vs.shape == (batch_size,), next_vs.shape

            target_values = rewards + self.discount * (1.0 - dones.float()) * next_vs
            assert target_values.shape == (batch_size,), target_values.shape

        qa_values = self.critic(observations)
        assert qa_values.shape == (batch_size, self.num_actions), qa_values.shape

        q_values = torch.gather(
            qa_values,
            dim=1,
            index=actions.unsqueeze(1),
        ).squeeze(1)
        assert q_values.shape == target_values.shape, (q_values.shape, target_values.shape)

        loss = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        metrics = {
            "q_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "q_target_values": target_values.mean().item(),
            "target_values": target_values.mean().item(),
            "q_grad_norm": grad_norm.item(),
        }

        return metrics

    @staticmethod
    def iql_expectile_loss(
        expectile: float, vs: torch.Tensor, target_qs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the IQL expectile loss.

        Let u = Q(s, a) - V(s).
        The expectile loss is:
            |tau - 1{u <= 0}| * u^2

        If Q > V, the weight is tau.
        If Q <= V, the weight is 1 - tau.
        """
        diff = target_qs - vs
        weights = torch.where(
            diff > 0,
            torch.full_like(diff, expectile),
            torch.full_like(diff, 1.0 - expectile),
        )
        loss = torch.mean(weights * diff.pow(2))
        return loss

    def update_v(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update V(s) using expectile regression toward Q(s, a_data).
        """
        batch_size = observations.shape[0]
        actions = actions.long()

        with torch.no_grad():
            qa_values = self.target_critic(observations)
            assert qa_values.shape == (batch_size, self.num_actions), qa_values.shape

            target_values = torch.gather(
                qa_values,
                dim=1,
                index=actions.unsqueeze(1),
            ).squeeze(1)
            assert target_values.shape == (batch_size,), target_values.shape

        vs = self._squeeze_value(self.value_critic(observations))
        assert vs.shape == target_values.shape, (vs.shape, target_values.shape)

        loss = self.iql_expectile_loss(
            expectile=self.expectile,
            vs=vs,
            target_qs=target_values,
        )

        self.value_critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.value_critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.value_critic_optimizer.step()

        return {
            "v_loss": loss.item(),
            "vs_adv": (vs - target_values).mean().item(),
            "vs": vs.mean().item(),
            "v_target_values": target_values.mean().item(),
            "v_grad_norm": grad_norm.item(),
        }

    def update_critic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update both Q(s, a) and V(s).
        """
        metrics_q = self.update_q(
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        )
        metrics_v = self.update_v(observations, actions)

        return {**metrics_q, **metrics_v}

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics = self.update_critic(
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        )

        # Actor update is inherited from AWAC, but the advantage is IQL-style:
        # A(s, a) = Q(s, a) - V(s).
        metrics["actor_loss"] = self.update_actor(observations, actions)

        if step % self.target_update_period == 0:
            self.update_target_critic()
            self.update_target_value_critic()

        return metrics

    def update_target_value_critic(self):
        self.target_value_critic.load_state_dict(self.value_critic.state_dict())
