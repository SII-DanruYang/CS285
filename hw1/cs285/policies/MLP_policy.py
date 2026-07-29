"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    2. forward
    3. update
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers)
    return mlp


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to continuous
    actions.

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter(

            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.FloatTensor) -> Any:
        """
        根据 observation 构造连续动作的高斯分布。

        mean_net 输出动作均值，logstd 是可学习的对数标准差。
        """
        mean = self.mean_net(observation)
        std = torch.exp(self.logstd)

        return distributions.Normal(mean, std)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        根据一个 observation 或一批 observations 采样动作。
        """
        obs = np.asarray(obs, dtype=np.float32)

        # 单个状态 [ob_dim] 转成批量形式 [1, ob_dim]
        if obs.ndim == 1:
            obs = obs[None, :]

        observation = ptu.from_numpy(obs)

        # 与环境交互时不需要计算梯度
        with torch.no_grad():
            action_distribution = self.forward(observation)
            action = action_distribution.sample()

        return ptu.to_numpy(action)

    def update(self, observations, actions):
        """
        使用专家的 observation-action 数据更新行为克隆策略。
        """
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)

        if observations.shape[0] != actions.shape[0]:
            raise ValueError(
                "observations 和 actions 的样本数量必须一致："
                f"{observations.shape[0]} != {actions.shape[0]}"
            )

        observations_tensor = ptu.from_numpy(observations)
        actions_tensor = ptu.from_numpy(actions)

        # 得到 πθ(a|s)
        action_distribution = self.forward(observations_tensor)

        # 最大化专家动作的似然，等价于最小化负对数似然
        log_prob = action_distribution.log_prob(actions_tensor)
        loss = -log_prob.sum(dim=-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss),
        }
