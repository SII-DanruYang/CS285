from typing import Callable, Optional, Sequence, Tuple

import torch
import numpy as np
from torch import nn

from cs285.agents.dqn_agent import DQNAgent
import cs285.infrastructure.pytorch_util as ptu


class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)

        self.actor = make_actor(observation_shape, num_actions)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature

    def get_action(self, observation, epsilon: float = 0.0) -> int:
        """
        AWAC should act with the learned actor, not with the DQN critic.
        During evaluation epsilon=0, so we take the actor's most likely action.
        During online fine-tuning epsilon can still inject random exploration.
        """
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.num_actions))

        observation = ptu.from_numpy(np.asarray(observation))[None]

        with torch.no_grad():
            action_dist = self._get_action_dist(observation)
            action = torch.argmax(action_dist.probs, dim=1)

        return ptu.to_numpy(action).squeeze(0).item()

    def _get_action_dist(self, observations: torch.Tensor) -> torch.distributions.Categorical:
        """
        The actor may either directly return a Categorical distribution,
        or return logits. This helper makes the code robust to both cases.
        """
        out = self.actor(observations)

        if isinstance(out, torch.distributions.Categorical):
            return out

        # If actor returns raw logits, convert them to a categorical distribution.
        return torch.distributions.Categorical(logits=out)

    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        batch_size = observations.shape[0]
        actions = actions.long()

        with torch.no_grad():
            # Actor distribution at next states: pi(a' | s')
            next_action_dist = self._get_action_dist(next_observations)
            next_action_probs = next_action_dist.probs
            assert next_action_probs.shape == (batch_size, self.num_actions), next_action_probs.shape

            # Target critic Q(s', all actions)
            next_qa_values = self.target_critic(next_observations)
            assert next_qa_values.shape == (batch_size, self.num_actions), next_qa_values.shape

            # AWAC critic backup uses E_{a' ~ pi}[Q_target(s', a')]
            next_qs = torch.sum(next_action_probs * next_qa_values, dim=1)
            assert next_qs.shape == (batch_size,), next_qs.shape

            # TD target: r + gamma * (1 - done) * E[Q(s', a')]
            target_values = rewards + self.discount * (1.0 - dones.float()) * next_qs
            assert target_values.shape == (batch_size,), target_values.shape

        # Current critic Q(s, all actions)
        qa_values = self.critic(observations)
        assert qa_values.shape == (batch_size, self.num_actions), qa_values.shape

        # Q(s, a_data)
        q_values = torch.gather(qa_values, dim=1, index=actions.unsqueeze(1)).squeeze(1)
        assert q_values.shape == target_values.shape, (q_values.shape, target_values.shape)

        loss = self.critic_loss(q_values, target_values)

        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        """
        AWAC advantage:
            A(s, a_data) = Q(s, a_data) - E_{a ~ pi(.|s)}[Q(s, a)]

        This tells us whether the dataset action is better or worse than
        the actor's average action under the current critic.
        """
        batch_size = observations.shape[0]
        actions = actions.long()

        if action_dist is None:
            action_dist = self._get_action_dist(observations)

        with torch.no_grad():
            qa_values = self.critic(observations)
            assert qa_values.shape == (batch_size, self.num_actions), qa_values.shape

            q_values = torch.gather(qa_values, dim=1, index=actions.unsqueeze(1)).squeeze(1)
            assert q_values.shape == (batch_size,), q_values.shape

            action_probs = action_dist.probs
            assert action_probs.shape == (batch_size, self.num_actions), action_probs.shape

            values = torch.sum(action_probs * qa_values, dim=1)
            assert values.shape == (batch_size,), values.shape

            advantages = q_values - values
            assert advantages.shape == (batch_size,), advantages.shape

        return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        AWAC actor update:
            maximize log pi(a_data | s) * exp(A(s, a_data) / temperature)

        In code, we minimize the negative of this objective.
        """
        actions = actions.long()

        action_dist = self._get_action_dist(observations)
        log_probs = action_dist.log_prob(actions)

        advantages = self.compute_advantage(
            observations=observations,
            actions=actions,
            action_dist=action_dist,
        )

        # Exponential advantage weights.
        # Clamp to avoid numerical explosion from exp().
        weights = torch.exp(advantages / self.temperature)
        weights = torch.clamp(weights, max=100.0).detach()

        loss = -(log_probs * weights).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics = super().update(
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            step,
        )

        # Update the actor.
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
