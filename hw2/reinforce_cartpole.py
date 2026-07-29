import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm


# ========== 1. 固定随机种子，方便复现 ==========
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ========== 2. 平滑曲线函数 ==========
def moving_average(data, window_size=9):
    data = np.array(data)
    result = []

    for i in range(len(data)):
        left = max(0, i - window_size // 2)
        right = min(len(data), i + window_size // 2 + 1)
        result.append(np.mean(data[left:right]))

    return np.array(result)


# ========== 3. 策略网络 ==========
class PolicyNet(nn.Module):
    """
    输入：当前状态 state
    输出：每个动作的概率

    CartPole 有两个动作：
    0：向左推车
    1：向右推车
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs


# ========== 4. REINFORCE 智能体 ==========
class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        """
        根据当前状态选择动作。
        注意：不是永远选概率最大的动作，而是按照概率随机采样。
        这样智能体才有探索能力。
        """

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        probs = self.policy_net(state)

        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()

        return action.item()

    def update(self, transition_dict):
        """
        用一整条 episode 的数据更新策略网络。

        transition_dict 里面保存：
        states  : 每一步的状态
        actions : 每一步采取的动作
        rewards : 每一步得到的奖励
        """

        states = transition_dict["states"]
        actions = transition_dict["actions"]
        rewards = transition_dict["rewards"]

        G = 0
        loss_list = []

        # 从最后一步往前算回报 G
        for i in reversed(range(len(rewards))):
            reward = rewards[i]
            state = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0).to(self.device)
            action = torch.tensor(actions[i], dtype=torch.long).to(self.device)

            probs = self.policy_net(state)
            action_dist = torch.distributions.Categorical(probs)
            log_prob = action_dist.log_prob(action)

            # G_t = r_t + gamma * G_{t+1}
            G = reward + self.gamma * G

            # 策略梯度是最大化 log_prob * G
            # PyTorch 默认是梯度下降，所以这里加负号
            loss = -log_prob * G
            loss_list.append(loss)

        total_loss = torch.stack(loss_list).sum()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


# ========== 5. 主训练函数 ==========
def train():
    set_seed(0)

    # 超参数
    learning_rate = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    env_name = "CartPole-v1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("状态维度 state_dim =", state_dim)
    print("动作数量 action_dim =", action_dim)

    agent = REINFORCE(
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        device=device
    )

    return_list = []

    for i in range(10):
        with tqdm(total=num_episodes // 10, desc=f"Iteration {i}") as pbar:
            for i_episode in range(num_episodes // 10):
                episode_return = 0

                transition_dict = {
                    "states": [],
                    "actions": [],
                    "rewards": [],
                }

                episode_id = i * (num_episodes // 10) + i_episode

                state, info = env.reset(seed=episode_id)
                done = False

                while not done:
                    action = agent.take_action(state)

                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["rewards"].append(reward)

                    state = next_state
                    episode_return += reward

                return_list.append(episode_return)
                agent.update(transition_dict)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        "episode": episode_id + 1,
                        "return": f"{np.mean(return_list[-10:]):.3f}"
                    })

                pbar.update(1)

    env.close()

    # ========== 6. 画原始回报曲线 ==========
    episodes = list(range(len(return_list)))

    plt.figure()
    plt.plot(episodes, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title(f"REINFORCE on {env_name}")
    plt.savefig("reinforce_return_raw.png", dpi=300)
    plt.show()

    # ========== 7. 画平滑后的回报曲线 ==========
    mv_return = moving_average(return_list, window_size=9)

    plt.figure()
    plt.plot(episodes, mv_return)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title(f"REINFORCE on {env_name} - Moving Average")
    plt.savefig("reinforce_return_smooth.png", dpi=300)
    plt.show()

    print("训练完成！")
    print("最后 10 个 episode 平均回报：", np.mean(return_list[-10:]))


if __name__ == "__main__":
    train()