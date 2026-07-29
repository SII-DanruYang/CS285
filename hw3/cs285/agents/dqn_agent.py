from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
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

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        if np.random.random() < epsilon:
            action = torch.randint(self.num_actions, (1,), device=observation.device)
        else:
            with torch.no_grad():
                qa_values = self.critic(observation)
                action = torch.argmax(qa_values, dim=1)

        return ptu.to_numpy(action).squeeze(0).item()

    def update_critic(self, obs, action, reward, next_obs, done):
        # 不对 target 求梯度；target 只作为监督信号
        with torch.no_grad():
            # target critic 负责评估下一个状态 s' 的所有动作 Q_target(s', a')
            next_qa_values = self.target_critic(next_obs)

            # Double DQN：用 online critic 选动作，用 target critic 估值
            if self.use_double_q:
                # online critic 只用于选择 argmax 动作，减少 max 带来的过估计偏差
                next_qa_values_online = self.critic(next_obs)
                # 对每个 next state 选出 online critic 认为最好的动作
                next_action = torch.argmax(next_qa_values_online, dim=1)
            else:
                # 普通 DQN：直接用 target critic 选择最大 Q 对应的动作
                next_action = torch.argmax(next_qa_values, dim=1)

            # 从 Q_target(s', 所有动作) 中取出 Q_target(s', next_action)
            next_q_values = torch.gather(
                next_qa_values,
                dim=1,
                index=next_action.unsqueeze(1),
            ).squeeze(1)

            # Bellman target：y = r + gamma * (1 - done) * Q_target(s', a')
            target_values = reward + self.discount * (1 - done.float()) * next_q_values

        # online critic 计算当前状态 s 下所有动作的 Q(s, a)
        qa_values = self.critic(obs)

        # 只取 replay buffer 中真实执行过的 action 对应的 Q(s, a)
        q_values = torch.gather(
            qa_values,
            dim=1,
            index=action.long().unsqueeze(1),
        ).squeeze(1)

        # critic loss：让当前 Q(s,a) 拟合 Bellman target
        loss = self.critic_loss(q_values, target_values)

        # 标准 PyTorch 更新三连：清梯度、反向传播、优化器更新
        self.critic_optimizer.zero_grad()
        loss.backward()

        # 如果配置文件设置了 clip_grad_norm，就进行梯度裁剪
        if self.clip_grad_norm is not None:
            # 梯度裁剪，防止 Q-learning 训练不稳定时梯度爆炸
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.clip_grad_norm,
            )
        else:
            # 如果没有设置梯度裁剪，就不裁剪；日志里记为 0.0
            grad_norm = 0.0

        # 更新 online critic 参数
        self.critic_optimizer.step()

        # 更新学习率 schedule
        self.lr_scheduler.step()

        # 返回日志，后面 logger 会写入 TensorBoard
        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
        }
    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, next_obs: np.ndarray, done: np.ndarray, step: int):
        # 用当前 batch 更新 critic，也就是训练 Q 网络
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)

        # 每隔 target_update_period 步，将 critic 参数复制到 target critic
        if step % self.target_update_period == 0:
            self.update_target_critic()

        # 返回日志信息，例如 critic loss、Q 值均值、target 均值等
        return critic_stats
