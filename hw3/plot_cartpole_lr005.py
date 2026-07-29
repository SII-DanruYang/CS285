import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = sorted(glob.glob("data/*CartPole-v1_lr005*"), key=os.path.getmtime)

if not runs:
    raise RuntimeError("没有找到 CartPole lr=0.05 的日志目录")

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

tag_eval = find_tag(["eval", "return"])
tag_q = find_tag(["q"])
tag_loss = find_tag(["critic", "loss"])

# 如果 q 或 loss 没用关键词找到，就做一些兜底
if tag_q is None:
    for t in tags:
        if "q" in t.lower():
            tag_q = t
            break

if tag_loss is None:
    for t in tags:
        if "loss" in t.lower() or "error" in t.lower():
            tag_loss = t
            break

for name, tag in [
    ("eval_return", tag_eval),
    ("q_values", tag_q),
    ("critic_loss", tag_loss),
]:
    if tag is None:
        print(f"Skip {name}: no matching tag")
        continue

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    print(f"{name}: tag={tag}, final={values[-1]:.4f}, max={max(values):.4f}, min={min(values):.4f}")

    plt.figure()
    plt.plot(steps, values)
    plt.xlabel("Environment steps")
    plt.ylabel(name)
    plt.title(f"CartPole-v1 lr=0.05: {name}")
    plt.grid(True)
    plt.tight_layout()
    out = f"cartpole_lr005_{name}.png"
    plt.savefig(out, dpi=200)
    print("Saved:", out)
