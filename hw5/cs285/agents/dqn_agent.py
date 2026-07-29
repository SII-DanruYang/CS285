from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int], num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Used for evaluation and data collection.

        With probability epsilon, choose a random action.
        Otherwise, choose the action with the largest Q-value.
        """
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.num_actions))

        observation = ptu.from_numpy(np.asarray(observation))[None]

        with torch.no_grad():
            qa_values = self.critic(observation)
            action = torch.argmax(qa_values, dim=1)

        return ptu.to_numpy(action).squeeze(0).item()

    def compute_critic_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict, dict]:
        """
        Compute the loss for the DQN critic.

        Returns:
         - loss: torch.Tensor, the MSE loss for the critic
         - metrics: dict, a dictionary of metrics to log
         - variables: dict, a dictionary of variables that can be used in subsequent calculations
        """
        batch_size = obs.shape[0]

        # Q(s, all actions)
        qa_values = self.critic(obs)
        assert qa_values.shape == (batch_size, self.num_actions), qa_values.shape

        # Q(s, a), where a is the action actually taken in the batch
        action = action.long()
        q_values = torch.gather(qa_values, dim=1, index=action.unsqueeze(1)).squeeze(1)
        assert q_values.shape == (batch_size,), q_values.shape

        with torch.no_grad():
            # Target-network Q-values for next states
            next_qa_values = self.target_critic(next_obs)
            assert next_qa_values.shape == (batch_size, self.num_actions), next_qa_values.shape

            if self.use_double_q:
                # Double DQN:
                # use the online critic to choose the next action,
                # but use the target critic to evaluate that action.
                next_actions_online = torch.argmax(self.critic(next_obs), dim=1)
                next_q_values = torch.gather(
                    next_qa_values,
                    dim=1,
                    index=next_actions_online.unsqueeze(1),
                ).squeeze(1)
            else:
                # Standard DQN:
                # directly take max over target critic Q-values.
                next_q_values = torch.max(next_qa_values, dim=1).values

            assert next_q_values.shape == (batch_size,), next_q_values.shape

            # Bellman target:
            # y = r + gamma * (1 - done) * max_a' Q_target(s', a')
            done_float = done.float()
            target_values = reward + self.discount * (1.0 - done_float) * next_q_values
            assert target_values.shape == (batch_size,), target_values.shape

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
                "target_values": target_values,
            },
        )

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats for logging."""
        loss, metrics, _ = self.compute_critic_loss(obs, action, reward, next_obs, done)

        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        metrics["grad_norm"] = grad_norm.item()
        self.critic_optimizer.step()

        self.lr_scheduler.step()

        return metrics

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)

        if step % self.target_update_period == 0:
            self.update_target_critic()

        return critic_stats
