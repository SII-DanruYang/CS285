import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUTDIR = Path("figures_hw3_fixed")
OUTDIR.mkdir(exist_ok=True)


def get_run_dir_from_log(log_file):
    text = Path(log_file).read_text(errors="ignore")
    matches = re.findall(r"logging outputs to\s+(.+)", text)
    if not matches:
        raise RuntimeError(f"No data dir found in {log_file}")
    run_dir = os.path.normpath(matches[-1].strip())
    if not os.path.isdir(run_dir):
        raise RuntimeError(f"Run dir does not exist for {log_file}: {run_dir}")
    return run_dir


def load_scalar(run_dir, tag):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        raise RuntimeError(f"Tag {tag} not found in {run_dir}. Available: {tags}")
    vals = ea.Scalars(tag)
    xs = [v.step for v in vals]
    ys = [v.value for v in vals]
    return xs, ys


def plot_from_logs(title, specs, tag, filename, ylabel=None):
    plt.figure(figsize=(8, 5))
    print(f"\n=== {title} | tag={tag} ===")

    for label, log_file in specs:
        run_dir = get_run_dir_from_log(log_file)
        xs, ys = load_scalar(run_dir, tag)
        plt.plot(xs, ys, label=label)
        print(f"{label:<24} log={log_file:<38} last_step={xs[-1]:<8} last={ys[-1]:.3f} max={max(ys):.3f}")
        print(f"  run={run_dir}")

    plt.title(title)
    plt.xlabel("Environment steps")
    plt.ylabel(ylabel or tag)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = OUTDIR / filename
    plt.savefig(out, dpi=200)
    plt.close()
    print("[SAVED]", out)


# 1. CartPole learning-rate sensitivity
plot_from_logs(
    title="CartPole-v1: Learning Rate Sensitivity",
    tag="eval_return",
    filename="q2_cartpole_lr_eval_return.png",
    ylabel="Evaluation return",
    specs=[
        ("lr00001", "cartpole_lr00001.log"),
        ("lr0001",  "cartpole_lr0001.log"),
        ("lr001",   "cartpole_lr001.log"),
        ("lr005",   "cartpole_lr005.log"),
    ],
)

# 2. LunarLander vanilla DQN vs Double DQN
plot_from_logs(
    title="LunarLander-v2: DQN vs Double DQN",
    tag="eval_return",
    filename="q2_lunar_dqn_vs_doubleq_eval_return.png",
    ylabel="Evaluation return",
    specs=[
        ("DQN seed1",     "lunar_dqn_seed1.log"),
        ("DQN seed2",     "lunar_dqn_seed2.log"),
        ("DQN seed3",     "lunar_dqn_seed3.log"),
        ("DoubleQ seed1", "lunar_doubleq_seed1.log"),
        ("DoubleQ seed2", "lunar_doubleq_seed2.log"),
        ("DoubleQ seed3", "lunar_doubleq_seed3.log"),
    ],
)

# 3. MsPacman eval_return and train_return
plot_from_logs(
    title="MsPacman-v0: Evaluation Return",
    tag="eval_return",
    filename="q2_mspacman_eval_return.png",
    ylabel="Evaluation return",
    specs=[
        ("MsPacman eval", "mspacman.log"),
    ],
)

plot_from_logs(
    title="MsPacman-v0: Training Return",
    tag="train_return",
    filename="q2_mspacman_train_return.png",
    ylabel="Training return",
    specs=[
        ("MsPacman train", "mspacman.log"),
    ],
)

# 4. HalfCheetah actor gradient estimators
plot_from_logs(
    title="HalfCheetah-v4: Actor Gradient Estimators",
    tag="eval_return",
    filename="q3_halfcheetah_actor_estimators_eval_return.png",
    ylabel="Evaluation return",
    specs=[
        ("REINFORCE-1",   "sac_halfcheetah_reinforce1_remote.log"),
        ("REINFORCE-10",  "sac_halfcheetah_reinforce10.log"),
        ("REPARAMETRIZE", "sac_halfcheetah_reparametrize.log"),
    ],
)

# 5. Hopper backup comparison: eval_return
plot_from_logs(
    title="Hopper-v4: Target Backup Strategy",
    tag="eval_return",
    filename="q3_hopper_backup_eval_return.png",
    ylabel="Evaluation return",
    specs=[
        ("single-Q",  "sac_hopper_singleq.log"),
        ("double-Q",  "sac_hopper_doubleq.log"),
        ("clipped-Q", "sac_hopper_clipq.log"),
    ],
)

# 6. Hopper backup comparison: q_values
plot_from_logs(
    title="Hopper-v4: Q-value Estimates",
    tag="q_values",
    filename="q3_hopper_backup_q_values.png",
    ylabel="Logged Q values",
    specs=[
        ("single-Q",  "sac_hopper_singleq.log"),
        ("double-Q",  "sac_hopper_doubleq.log"),
        ("clipped-Q", "sac_hopper_clipq.log"),
    ],
)

# 7. Humanoid 500K
plot_from_logs(
    title="Humanoid-v4: SAC clipped-Q 500K",
    tag="eval_return",
    filename="q3_humanoid_clipq_500k_eval_return.png",
    ylabel="Evaluation return",
    specs=[
        ("Humanoid clipped-Q", "sac_humanoid_clipq_500k.log"),
    ],
)

print("\n[DONE] Fixed figures saved to:", OUTDIR.resolve())
