import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_eval_curve(run_dir):
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
            raise RuntimeError(f"找不到 eval return，已有 tags: {tags}")

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values, tag

runs = sorted(glob.glob("data/hw3_dqn_doubleq_LunarLander-v2_seed*"))

print("Double DQN runs:")
for r in runs:
    print("  ", r)

plt.figure()

for run in runs:
    steps, values, tag = get_eval_curve(run)
    name = os.path.basename(run).split("_17-")[0]
    plt.plot(steps, values, label=name)
    print(name, "final =", values[-1], "max =", max(values), "tag =", tag)

plt.xlabel("Environment steps")
plt.ylabel("Evaluation return")
plt.title("Double DQN on LunarLander-v2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("lunarlander_doubleq_eval_return.png", dpi=200)

print("Saved: lunarlander_doubleq_eval_return.png")
