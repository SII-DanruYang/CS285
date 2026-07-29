import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

patterns = {
    "lr=1e-4": "data/*CartPole-v1_lr0001*",
    "lr=1e-3": "data/hw3_dqn_dqn_CartPole-v1_s64_l2_d0.99_*",
    "lr=1e-2": "data/*CartPole-v1_lr001*",
    "lr=5e-2": "data/*CartPole-v1_lr005*",
}

def get_latest_run(pattern):
    runs = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not runs:
        return None
    return runs[-1]

def read_eval_return(run_dir):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    tags = ea.Tags()["scalars"]

    tag = None
    for candidate in ["eval_return", "Eval_AverageReturn", "eval/return"]:
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
    return steps, values, tag

plt.figure()

for label, pattern in patterns.items():
    run = get_latest_run(pattern)
    if run is None:
        print("Skip", label, "no run found")
        continue

    steps, values, tag = read_eval_return(run)
    plt.plot(steps, values, label=label)
    print(label, "run =", run)
    print(label, "final =", values[-1], "max =", max(values), "tag =", tag)

plt.xlabel("Environment steps")
plt.ylabel("Evaluation return")
plt.title("CartPole-v1 DQN Learning Rate Sweep")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("cartpole_lr_sweep_eval_return.png", dpi=200)

print("Saved: cartpole_lr_sweep_eval_return.png")
