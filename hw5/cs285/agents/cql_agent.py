from typing import Sequence, Tuple

import torch

from cs285.agents.dqn_agent import DQNAgent


class CQLAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        cql_alpha: float,
        cql_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            observation_shape=observation_shape, num_actions=num_actions, **kwargs
        )
        self.cql_alpha = cql_alpha
        self.cql_temperature = cql_temperature

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict, dict]:
        # First compute the standard DQN Bellman error.
        loss, metrics, variables = super().compute_critic_loss(
            obs,
            action,
            reward,
            next_obs,
            done,
        )

        qa_values = variables["qa_values"]  # Q(s, all actions), shape: (B, num_actions)
        q_values = variables["q_values"]    # Q(s, a_data), shape: (B,)

        # CQL regularizer:
        # temperature * logsumexp(Q(s, a) / temperature) - Q(s, a_data)
        logsumexp_q = self.cql_temperature * torch.logsumexp(
            qa_values / self.cql_temperature,
            dim=1,
        )

        cql_loss = torch.mean(logsumexp_q - q_values)

        # Total critic loss = DQN TD loss + alpha * CQL regularizer
        loss = loss + self.cql_alpha * cql_loss

        metrics["cql_loss"] = cql_loss.item()
        metrics["cql_alpha"] = self.cql_alpha
        metrics["cql_temperature"] = self.cql_temperature
        metrics["total_critic_loss"] = loss.item()
        metrics["cql_logsumexp_q"] = logsumexp_q.mean().item()

        return loss, metrics, variables
