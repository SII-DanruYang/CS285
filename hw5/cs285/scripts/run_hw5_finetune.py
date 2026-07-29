import argparse
import pickle
import os
import shutil
from datetime import datetime

from cs285.agents import agents as agent_types
from cs285.envs import Pointmass

import gym
import numpy as np
import torch
import cs285.infrastructure.pytorch_util as ptu
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.replay_buffer import ReplayBuffer

from scripting_utils import make_logger, make_config
from run_hw5_explore import visualize

MAX_NVIDEO = 2


def copy_replay_buffer_to_larger_buffer(dataset: ReplayBuffer, capacity: int) -> ReplayBuffer:
    """
    Create a new replay buffer and copy the valid transitions from an existing
    replay buffer into it. This lets us keep the offline dataset and still have
    enough capacity to insert new online transitions during fine-tuning.
    """
    n_initial = min(dataset.size, dataset.max_size)
    replay_buffer = ReplayBuffer(capacity=capacity)

    for i in range(n_initial):
        replay_buffer.insert(
            observation=dataset.observations[i],
            action=dataset.actions[i],
            reward=dataset.rewards[i],
            next_observation=dataset.next_observations[i],
            done=dataset.dones[i],
        )

    return replay_buffer


def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # Set random seeds.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # Give every run a unique timestamp so that files are never overwritten.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = config.get(
        "log_name",
        config.get("exp_name", "pointmass_finetune"),
    )
    safe_run_name = str(run_name).replace("/", "_").replace(" ", "_")

    artifact_dir = os.path.join(
        "finetune_artifacts",
        f"{safe_run_name}_{run_timestamp}",
    )
    image_dir = os.path.join(artifact_dir, "images")
    config_dir = os.path.join(artifact_dir, "config")

    os.makedirs(image_dir, exist_ok=False)
    os.makedirs(config_dir, exist_ok=False)

    source_config_name = os.path.basename(args.config_file)
    config_snapshot_path = os.path.join(
        config_dir,
        f"{os.path.splitext(source_config_name)[0]}_{run_timestamp}.yaml",
    )
    shutil.copy2(args.config_file, config_snapshot_path)

    print("Run timestamp:", run_timestamp)
    print("Artifacts directory:", artifact_dir)
    print("Config snapshot:", config_snapshot_path)

    # Make the gym environment.
    env = config["make_env"]()

    # Use an independent environment for evaluation. Evaluation resets and
    # steps the environment, so sharing it with online collection would make
    # the stored training observation inconsistent with the true env state.
    eval_env = config["make_env"]()

    exploration_schedule = config.get("exploration_schedule", None)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert discrete, "DQN only supports discrete action spaces"

    # Build agent.
    agent_cls = agent_types[config["agent"]]
    agent = agent_cls(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )

    ep_len = env.spec.max_episode_steps or env.max_episode_steps
    env_pointmass: Pointmass = env.unwrapped
    eval_env_pointmass: Pointmass = eval_env.unwrapped

    # Load offline dataset.
    offline_dataset_file = os.path.join(args.dataset_dir, f"{config['dataset_name']}.pkl")
    with open(offline_dataset_file, "rb") as f:
        offline_dataset = pickle.load(f)

    num_offline_steps = config["offline_steps"]
    num_online_steps = config["total_steps"] - num_offline_steps

    # Keep all offline data and leave room for online data.
    n_initial = min(offline_dataset.size, offline_dataset.max_size)
    # By default, keep enough capacity for all offline + online transitions.
    # For fine-tuning experiments, we can optionally use a smaller replay_capacity
    # to focus learning on more recent online data.
    if "replay_capacity" in config:
        replay_capacity = max(int(config["replay_capacity"]), n_initial + 1)
    else:
        replay_capacity = max(config["total_steps"], n_initial + num_online_steps + 1)

    replay_buffer = copy_replay_buffer_to_larger_buffer(
        dataset=offline_dataset,
        capacity=replay_capacity,
    )

    # Keep online transitions in a separate buffer. This makes it possible to
    # explicitly control the offline/online sampling ratio during fine-tuning.
    online_buffer = ReplayBuffer(
        capacity=max(
            num_online_steps + 1,
            int(config["batch_size"]) + 1,
        )
    )

    print(f"Loaded offline dataset from {offline_dataset_file}")
    print(f"Offline transitions copied: {n_initial}")
    print(f"Replay buffer capacity: {replay_buffer.max_size}")
    print(f"Offline training steps: {num_offline_steps}")
    print(f"Online fine-tuning steps: {num_online_steps}")

    # Initialize environment state for later online interaction.
    # Optional behavior cloning warm-up for actor-based methods such as AWAC.
    # This is useful when the offline dataset has high-quality trajectories and
    # we want the actor to imitate them before online fine-tuning.
    bc_pretrain_steps = int(config.get("bc_pretrain_steps", 0))
    if bc_pretrain_steps > 0:
        if not hasattr(agent, "actor") or not hasattr(agent, "_get_action_dist"):
            print("bc_pretrain_steps is set, but this agent has no actor. Skipping BC warm-up.")
        else:
            print(f"Running BC warm-up for {bc_pretrain_steps} steps...")
            for bc_step in range(bc_pretrain_steps):
                batch = replay_buffer.sample(config["batch_size"])

                obs_bc = ptu.from_numpy(batch["observations"])
                act_bc = ptu.from_numpy(batch["actions"]).long()
                if act_bc.ndim > 1:
                    act_bc = act_bc.squeeze(-1)

                action_dist = agent._get_action_dist(obs_bc)
                bc_loss = -action_dist.log_prob(act_bc).mean()

                agent.actor_optimizer.zero_grad()
                bc_loss.backward()
                agent.actor_optimizer.step()

                if bc_step % 1000 == 0:
                    print(f"BC step {bc_step}: loss={bc_loss.item():.4f}")

            print("Finished BC warm-up.")

    observation = env.reset()
    recent_observations = []

    # Default online exploration epsilon.
    # If the config provides an exploration schedule, we use that instead.
    online_epsilon = config.get("online_epsilon", 0.1)

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        is_online = step >= num_offline_steps

        # During online fine-tuning, optionally reduce CQL conservatism.
        # Offline stage still uses the original cql_alpha from the config.
        if is_online and "online_cql_alpha" in config and hasattr(agent, "cql_alpha"):
            agent.cql_alpha = config["online_cql_alpha"]
        epsilon = None

        # Online interaction starts only after offline training is finished.
        if is_online:
            # Prefer fine-tuning-specific online epsilon settings.
            # Some base configs provide an exploration_schedule by default,
            # but for fine-tuning experiments we want the yaml-defined
            # online_epsilon_start/end schedule to take precedence.
            if "online_epsilon_start" in config and "online_epsilon_end" in config:
                online_step = step - num_offline_steps
                decay_steps = config.get("online_epsilon_decay_steps", num_online_steps)
                frac = min(1.0, online_step / max(1, decay_steps))
                epsilon = (
                    config["online_epsilon_start"]
                    + frac * (config["online_epsilon_end"] - config["online_epsilon_start"])
                )
            elif exploration_schedule is not None:
                epsilon = exploration_schedule.value(step - num_offline_steps)
            else:
                epsilon = online_epsilon

            action = agent.get_action(observation, epsilon)
            next_observation, reward, done, info = env.step(action)
            next_observation = np.asarray(next_observation)

            truncated = info.get("TimeLimit.truncated", False)

            transition_done = done and not truncated

            # The combined buffer is saved at the end of training.
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=transition_done,
            )

            # The online-only buffer is used for controlled mixed sampling.
            online_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=transition_done,
            )

            recent_observations.append(observation)

            if done:
                observation = env.reset()

                if "episode" in info:
                    logger.log_scalar(info["episode"]["r"], "train_return", step)
                    logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
            else:
                observation = next_observation

        # Main training loop.
        #
        # Offline stage:
        #     sample entirely from the original offline dataset.
        #
        # Online stage:
        #     sample a controlled mixture from the frozen offline dataset and
        #     the newly collected online transitions.
        batch_size = int(config["batch_size"])
        effective_online_ratio = 0.0

        if is_online:
            requested_online_ratio = float(
                config.get("online_ratio", 0.5)
            )
            requested_online_ratio = float(
                np.clip(requested_online_ratio, 0.05, 0.95)
            )

            online_batch_size = int(
                round(batch_size * requested_online_ratio)
            )
            online_batch_size = max(
                1,
                min(batch_size - 1, online_batch_size),
            )
            offline_batch_size = batch_size - online_batch_size

            # Wait until the online buffer contains enough distinct samples.
            # Before that, use the combined replay buffer normally.
            if len(online_buffer) >= online_batch_size:
                offline_batch = offline_dataset.sample(
                    offline_batch_size
                )
                online_batch = online_buffer.sample(
                    online_batch_size
                )

                batch = {
                    key: np.concatenate(
                        [offline_batch[key], online_batch[key]],
                        axis=0,
                    )
                    for key in offline_batch
                }

                # Shuffle the mixed batch so that offline and online samples
                # are not placed in fixed contiguous regions.
                permutation = np.random.permutation(batch_size)
                batch = {
                    key: value[permutation]
                    for key, value in batch.items()
                }

                effective_online_ratio = (
                    online_batch_size / batch_size
                )
            else:
                batch = replay_buffer.sample(batch_size)
        else:
            batch = offline_dataset.sample(batch_size)

        batch = {
            k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in batch.items()
        }

        # For fine-tuning, use the real environment reward by default.
        use_reward = args.use_reward or config.get("use_reward", True)
        rewards = batch["rewards"] if use_reward else batch["rewards"] * 0.0

        update_info = agent.update(
            batch["observations"],
            batch["actions"],
            rewards,
            batch["next_observations"],
            batch["dones"],
            step,
        )

        update_info["is_online"] = float(is_online)
        update_info["buffer_size"] = float(len(replay_buffer))
        update_info["online_buffer_size"] = float(len(online_buffer))
        update_info["effective_online_ratio"] = float(
            effective_online_ratio
        )

        if epsilon is not None:
            update_info["epsilon"] = epsilon

        if step % args.log_interval == 0:
            for k, v in update_info.items():
                logger.log_scalar(v, k, step)
            logger.flush()

        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            trajectory_figures = [
                eval_env_pointmass.plot_trajectory(
                    trajectory["next_observation"]
                )
                for trajectory in trajectories
            ]

            for trajectory_index, trajectory_figure in enumerate(
                trajectory_figures
            ):
                trajectory_path = os.path.join(
                    image_dir,
                    (
                        f"eval_trajectory_{run_timestamp}"
                        f"_step_{step:07d}"
                        f"_trajectory_{trajectory_index:02d}.png"
                    ),
                )
                trajectory_figure.savefig(
                    trajectory_path,
                    dpi=200,
                    bbox_inches="tight",
                )

            logger.log_figures(
                trajectory_figures,
                "trajectories",
                step,
                "eval",
            )

        if step % args.visualize_interval == 0 and len(recent_observations) > 0:
            observations_for_plot = np.stack(recent_observations)
            recent_observations = []

            exploration_figure = visualize(
                env_pointmass,
                agent,
                observations_for_plot,
            )
            exploration_path = os.path.join(
                image_dir,
                (
                    f"online_exploration_{run_timestamp}"
                    f"_step_{step:07d}.png"
                ),
            )
            exploration_figure.savefig(
                exploration_path,
                dpi=200,
                bbox_inches="tight",
            )

            logger.log_figure(
                exploration_figure,
                "online_exploration_trajectories",
                step,
                "eval",
            )

    # Save the final replay buffer with a unique timestamp.
    final_dataset_base = config.get(
        "finetuned_dataset_name",
        f"{config['dataset_name']}_finetuned",
    )
    dataset_file = os.path.join(
        args.dataset_dir,
        f"{final_dataset_base}_{run_timestamp}.pkl",
    )
    with open(dataset_file, "wb") as f:
        pickle.dump(replay_buffer, f)
    print("Saved final fine-tuning replay buffer to", dataset_file)

    # Render and save the final heatmap in this run's unique image directory.
    n_plot = min(len(replay_buffer), replay_buffer.max_size)
    fig = visualize(
        env_pointmass,
        agent,
        replay_buffer.observations[:n_plot],
    )
    fig.suptitle("Fine-tuning State Coverage")

    final_heatmap_path = os.path.join(
        image_dir,
        f"final_heatmap_{run_timestamp}.png",
    )
    fig.savefig(
        final_heatmap_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("Saved final heatmap to", final_heatmap_path)
    print("All run artifacts are stored in", artifact_dir)


banner = """
======================================================================
Online Fine-Tuning

Fine-tuning on the {env} environment using algorithm {alg}.
The initial offline dataset is loaded from {dataset_dir}.
======================================================================
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--visualize_interval", "-vi", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1)

    parser.add_argument("--use_reward", action="store_true")
    parser.add_argument("--dataset_dir", type=str, required=True)

    args = parser.parse_args()

    logdir_prefix = "hw5_finetune_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    os.makedirs(args.dataset_dir, exist_ok=True)
    print(
        banner.format(
            env=config["env_name"], alg=config["agent"], dataset_dir=args.dataset_dir
        )
    )

    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
