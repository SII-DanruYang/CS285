import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def read_curve(run_dir):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    tags = ea.Tags()["scalars"]

    tag = None
    for candidate in ["eval_return", "Eval_AverageReturn", "eval/return", "evaluation_return"]:
        if candidate in tags:
            tag = candidate
            break

    if tag is None:
        candidates = [t for t in tags if "eval" in t.lower() and "return" in t.lower()]
        if candidates:
            tag = candidates[0]
        else:
            raise RuntimeError(f"找不到 eval return: {run_dir}, tags={tags}")

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

vanilla_dirs = sorted(glob.glob("data/hw3_dqn_dqn_LunarLander-v2_seed*_clean*"))
doubleq_dirs = sorted(glob.glob("data/hw3_dqn_doubleq_LunarLander-v2_seed*"))

print("Vanilla DQN dirs:")
for d in vanilla_dirs:
    print(" ", d)

print("Double DQN dirs:")
for d in doubleq_dirs:
    print(" ", d)

if len(vanilla_dirs) < 3:
    raise RuntimeError("clean vanilla DQN 少于 3 个目录，请先确认三个 seed 是否跑完。")

if len(doubleq_dirs) < 3:
    raise RuntimeError("Double DQN 少于 3 个目录。")

plt.figure()

for i, run_dir in enumerate(vanilla_dirs[:3], 1):
    steps, values = read_curve(run_dir)
    plt.plot(steps, values, color="blue", alpha=0.55, label="Vanilla DQN" if i == 1 else None)
    print(f"vanilla seed {i}: final={values[-1]:.2f}, max={max(values):.2f}")

for i, run_dir in enumerate(doubleq_dirs[:3], 1):
    steps, values = read_curve(run_dir)
    plt.plot(steps, values, color="red", alpha=0.65, label="Double DQN" if i == 1 else None)
    print(f"doubleq seed {i}: final={values[-1]:.2f}, max={max(values):.2f}")

plt.xlabel("Environment steps")
plt.ylabel("Evaluation return")
plt.title("Vanilla DQN vs Double DQN on LunarLander-v2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("lunarlander_dqn_vs_doubleq_clean.png", dpi=200)

print("Saved: lunarlander_dqn_vs_doubleq_clean.png")
