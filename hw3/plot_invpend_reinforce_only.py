import glob, os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = sorted(glob.glob("data/*sanity_invpendulum_reinforce*"), key=os.path.getmtime)

best = None
for run in runs:
    ea = EventAccumulator(run, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if "eval_return" not in tags:
        continue
    vals = ea.Scalars("eval_return")
    if not vals:
        continue
    last_step = vals[-1].step
    max_return = max(v.value for v in vals)
    candidate = (last_step, max_return, run, ea, tags)
    if best is None or candidate[:2] > best[:2]:
        best = candidate

if best is None:
    raise RuntimeError("No valid InvertedPendulum reinforce run found.")

last_step, max_return, run, ea, tags = best
print("Using:", run)
print("last_step:", last_step)
print("max_eval_return:", max_return)

def get(tag):
    vals = ea.Scalars(tag)
    return [v.step for v in vals], [v.value for v in vals]

# eval_return
steps, values = get("eval_return")
plt.figure()
plt.plot(steps, values, label="eval_return")
plt.xlabel("Environment steps")
plt.ylabel("Evaluation return")
plt.title("SAC REINFORCE Sanity Check on InvertedPendulum-v4")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("sac_sanity_invpendulum_reinforce_eval_return.png", dpi=200)
print("Saved: sac_sanity_invpendulum_reinforce_eval_return.png")
print("final eval_return:", values[-1])
print("max eval_return:", max(values))

# train vs eval
plt.figure()
steps, values = get("eval_return")
plt.plot(steps, values, label="eval_return")

if "train_return" in tags:
    train_steps, train_values = get("train_return")
    plt.plot(train_steps, train_values, label="train_return")

plt.xlabel("Environment steps")
plt.ylabel("Return")
plt.title("InvertedPendulum-v4 SAC REINFORCE: Train vs Eval Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("sac_sanity_invpendulum_reinforce_train_eval_return.png", dpi=200)
print("Saved: sac_sanity_invpendulum_reinforce_train_eval_return.png")
