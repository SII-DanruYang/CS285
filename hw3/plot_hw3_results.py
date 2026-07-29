import os
import glob
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


OUTDIR = Path("figures_hw3")
OUTDIR.mkdir(exist_ok=True)


def load_scalar(run_dir, tag):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        return None, None
    vals = ea.Scalars(tag)
    if not vals:
        return None, None
    xs = [v.step for v in vals]
    ys = [v.value for v in vals]
    return xs, ys


def choose_best_run(patterns, tag="eval_return"):
    """从多个 glob pattern 中选 scalar 记录最完整的 run。"""
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    best = None
    best_last_step = -1
    best_n = -1

    for run in candidates:
        xs, ys = load_scalar(run, tag)
        if xs is None:
            continue
        last_step = xs[-1]
        n = len(xs)
        if (last_step, n) > (best_last_step, best_n):
            best = run
            best_last_step = last_step
            best_n = n

    return best


def plot_runs(title, run_specs, tag, filename, ylabel=None):
    plt.figure(figsize=(8, 5))

    printed = []
    for label, patterns in run_specs:
        run = choose_best_run(patterns, tag=tag)
        if run is None:
            print(f"[MISS] {label}: no run for tag {tag}")
            continue

        xs, ys = load_scalar(run, tag)
        plt.plot(xs, ys, label=label)
        printed.append((label, run, xs[-1], ys[-1], max(ys), min(ys)))

    plt.title(title)
    plt.xlabel("Environment steps")
    plt.ylabel(ylabel or tag)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = OUTDIR / filename
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"\n[SAVED] {out}")
    for label, run, last_step, last, mx, mn in printed:
        print(f"  {label:<20} last_step={last_step:<8} last={last:.3f} max={mx:.3f} min={mn:.3f}")
        print(f"    run={run}")


# 1. CartPole learning-rate sensitivity
plot_runs(
    title="CartPole-v1: Learning Rate Sensitivity",
    tag="eval_return",
    filename="q2_cartpole_lr_eval_return.png",
    ylabel="Evaluation return",
    run_specs=[
        ("lr=0.0001", ["data/*CartPole*lr0.0001*", "data/*CartPole*"]),
        ("lr=0.001",  ["data/*CartPole*lr0.001*", "data/*CartPole*"]),
        ("lr=0.01",   ["data/*CartPole*lr0.01*", "data/*CartPole*"]),
        ("lr=0.05",   ["data/*CartPole*lr0.05*", "data/*CartPole*"]),
    ],
)

# 上面如果因为 log_string 不含 lr 分不清，下面用 mtime 顺序兜底打印所有 CartPole
print("\n[INFO] All CartPole candidate runs:")
for r in sorted(glob.glob("data/*CartPole*"), key=os.path.getmtime):
    xs, ys = load_scalar(r, "eval_return")
    if xs:
        print(f"  {r} | last_step={xs[-1]} last={ys[-1]:.3f} max={max(ys):.3f}")


# 2. LunarLander DQN vs Double-DQN
plot_runs(
    title="LunarLander-v2: DQN vs Double DQN",
    tag="eval_return",
    filename="q2_lunar_dqn_vs_doubleq_eval_return.png",
    ylabel="Evaluation return",
    run_specs=[
        ("DQN seed1",        ["data/*LunarLander*dqn*seed1*", "data/*LunarLander*dqn*"]),
        ("DQN seed2",        ["data/*LunarLander*dqn*seed2*", "data/*LunarLander*dqn*"]),
        ("DQN seed3",        ["data/*LunarLander*dqn*seed3*", "data/*LunarLander*dqn*"]),
        ("DoubleQ seed1",    ["data/*LunarLander*doubleq*seed1*", "data/*LunarLander*doubleq*"]),
        ("DoubleQ seed2",    ["data/*LunarLander*doubleq*seed2*", "data/*LunarLander*doubleq*"]),
        ("DoubleQ seed3",    ["data/*LunarLander*doubleq*seed3*", "data/*LunarLander*doubleq*"]),
    ],
)

# 3. MsPacman train_return and eval_return
plot_runs(
    title="MsPacman-v0: Train Return and Eval Return",
    tag="eval_return",
    filename="q2_mspacman_eval_return.png",
    ylabel="Evaluation return",
    run_specs=[
        ("MsPacman eval", ["data/*MsPacman*", "data/*mspacman*"]),
    ],
)

plot_runs(
    title="MsPacman-v0: Training Return",
    tag="train_return",
    filename="q2_mspacman_train_return.png",
    ylabel="Training return",
    run_specs=[
        ("MsPacman train", ["data/*MsPacman*", "data/*mspacman*"]),
    ],
)

# 4. HalfCheetah actor gradient estimators
plot_runs(
    title="HalfCheetah-v4: Actor Gradient Estimators",
    tag="eval_return",
    filename="q3_halfcheetah_actor_estimators_eval_return.png",
    ylabel="Evaluation return",
    run_specs=[
        ("REINFORCE-1",   ["data/*HalfCheetah*reinforce1*", "data/*halfcheetah*reinforce1*"]),
        ("REINFORCE-10",  ["data/*HalfCheetah*reinforce10*", "data/*halfcheetah*reinforce10*"]),
        ("REPARAMetrize", ["data/*HalfCheetah*reparametrize*", "data/*halfcheetah*reparametrize*"]),
    ],
)

# 5. Hopper eval_return
plot_runs(
    title="Hopper-v4: Target Backup Strategy",
    tag="eval_return",
    filename="q3_hopper_backup_eval_return.png",
    ylabel="Evaluation return",
    run_specs=[
        ("single-Q",   ["data/*hopper_singlecritic*", "data/*Hopper*singlecritic*"]),
        ("double-Q",   ["data/*hopper_doubleq*", "data/*Hopper*doubleq*"]),
        ("clipped-Q",  ["data/*hopper_clipq*", "data/*Hopper*clipq*", "data/*Hopper*min*"]),
    ],
)

# 6. Hopper q_values
plot_runs(
    title="Hopper-v4: Q-value Estimates",
    tag="q_values",
    filename="q3_hopper_backup_q_values.png",
    ylabel="Logged Q values",
    run_specs=[
        ("single-Q",   ["data/*hopper_singlecritic*", "data/*Hopper*singlecritic*"]),
        ("double-Q",   ["data/*hopper_doubleq*", "data/*Hopper*doubleq*"]),
        ("clipped-Q",  ["data/*hopper_clipq*", "data/*Hopper*clipq*", "data/*Hopper*min*"]),
    ],
)

# 7. Humanoid 500K
plot_runs(
    title="Humanoid-v4: SAC clipped-Q 500K",
    tag="eval_return",
    filename="q3_humanoid_clipq_500k_eval_return.png",
    ylabel="Evaluation return",
    run_specs=[
        ("Humanoid clipped-Q", ["data/*humanoid*min*", "data/*Humanoid*min*"]),
    ],
)

print("\n[DONE] Figures saved to:", OUTDIR.resolve())
