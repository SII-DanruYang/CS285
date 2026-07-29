import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = sorted(glob.glob("data/hw3_dqn_dqn_CartPole-v1*"), key=os.path.getmtime)

if not runs:
    raise RuntimeError("没有找到 CartPole 日志")

run = runs[-1]
print("Using run:", run)

ea = EventAccumulator(run)
ea.Reload()

print("Available scalar tags:")
print(ea.Tags()["scalars"])

tag = "eval_return"
if tag not in ea.Tags()["scalars"]:
    candidates = [t for t in ea.Tags()["scalars"] if "eval" in t.lower() and "return" in t.lower()]
    if candidates:
        tag = candidates[0]
    else:
        raise RuntimeError("没找到 eval_return，请看上面打印的 scalar tags")

events = ea.Scalars(tag)
steps = [e.step for e in events]
values = [e.value for e in events]

plt.figure()
plt.plot(steps, values)
plt.xlabel("Environment steps")
plt.ylabel("Evaluation return")
plt.title("DQN on CartPole-v1")
plt.grid(True)
plt.tight_layout()
plt.savefig("cartpole_eval_return.png", dpi=200)

print("Saved: cartpole_eval_return.png")
