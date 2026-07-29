"""A
Some miscellaneous utility functions

Functions to edit:
    1. sample_trajectory
"""

from collections import OrderedDict
import cv2
import numpy as np
import time

from cs285.infrastructure import pytorch_util as ptu


def sample_trajectory(env, policy, max_path_length, render=False):
    """使用 policy 在环境中采集一条完整轨迹。"""

    # Gym 旧版 reset 返回 observation；
    # Gym 新版 reset 返回 (observation, info)。
    reset_output = env.reset()
    if isinstance(reset_output, tuple):
        ob = reset_output[0]
    else:
        ob = reset_output

    obs = []
    acs = []
    rewards = []
    next_obs = []
    terminals = []
    image_obs = []

    steps = 0

    while True:
        if render:
            if hasattr(env, "sim"):
                img = env.sim.render(
                    camera_name="track",
                    height=500,
                    width=500,
                )[::-1]
            else:
                try:
                    img = env.render()
                except TypeError:
                    img = env.render(mode="single_rgb_array")

            if img is not None:
                image_obs.append(
                    cv2.resize(
                        img,
                        dsize=(250, 250),
                        interpolation=cv2.INTER_CUBIC,
                    )
                )

        # 当前策略根据当前状态产生动作
        ac = policy.get_action(
            np.asarray(ob, dtype=np.float32)
        )[0]

        # 避免高斯策略采样出动作空间以外的值
        ac = np.clip(
            ac,
            env.action_space.low,
            env.action_space.high,
        )

        step_output = env.step(ac)

        # 兼容 Gym 新旧 API
        if len(step_output) == 5:
            next_ob, rew, terminated, truncated, _ = step_output
            done = bool(terminated or truncated)
        else:
            next_ob, rew, done, _ = step_output
            done = bool(done)

        steps += 1

        # 环境终止或达到最大轨迹长度时结束
        rollout_done = bool(
            done or steps >= max_path_length
        )

        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob

        if rollout_done:
            break

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
    """Collect rollouts until we have collected min_timesteps_per_batch steps."""

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        #collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)

        #count steps
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False):
    """Collect ntraj rollouts."""

    paths = []
    for i in range(ntraj):
        # collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
    return paths


########################################
########################################


def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals


########################################
########################################
            

def compute_metrics(paths, eval_paths):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [path["reward"].sum() for path in paths]
    eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

    # episode lengths, for logging
    train_ep_lens = [len(path["reward"]) for path in paths]
    eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


############################################
############################################


def get_pathlength(path):
    return len(path["reward"])
