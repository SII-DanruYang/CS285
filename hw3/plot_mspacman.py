import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = sorted(glob.glob("data/*MsPacman*"), key=os.path.getmtime)

if not runs:
    raise RuntimeError("没有找到 MsPacman 日志目录")

run = runs[-1]
print("Using run:", run)

ea = EventAccumulator(run)
ea.Reload()
tags = ea.Tags()["scalars"]
print("Scalar tags:", tags)

def find_tag(keywords):
    for t in tags:
        low = t.lower()
        if all(k.lower() in low for k in keywords):
            return t
    return None

eval_tag = find_tag(["eval", "return"])
train_tag = find_tag(["train", "return"])

if eval_tag is None:
    raise RuntimeError("找不到 eval_return，请看 Scalar tags")
if train_tag is None:
    raise RuntimeError("找不到 train_return，请看 Scalar tags")

plt.figure()

for tag, label in [(train_tag, "train_return"), (eval_tag, "eval_return")]:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    plt.plot(steps, values, label=label)
    print(label, "tag =", tag, "final =", values[-1], "max =", max(values), "min =", min(values))

plt.xlabel("Environment steps")
plt.ylabel("Return")
plt.title("DQN on MsPacman-v0")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mspacman_train_eval_return.png", dpi=200)

print("Saved: mspacman_train_eval_return.png")
