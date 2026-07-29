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

    steps = [v.step for v in vals]
    returns = [v.value for v in vals]

    last_step = steps[-1]
    max_return = max(returns)

    candidate = (last_step, max_return, run, steps, returns)
    if best is None or candidate[:2] > best[:2]:
        best = candidate

if best is None:
    raise RuntimeError("No valid InvertedPendulum reinforce run found.")

last_step, max_return, run, steps, returns = best

print("Using:", run)
print("last_step:", last_step)
print("final eval_return:", returns[-1])
print("max eval_return:", max_return)

plt.figure(figsize=(8, 5))
plt.plot(steps, returns, marker="o", label="eval_return")
plt.axhline(1000, linestyle="--", linewidth=1, label="max return = 1000")

plt.xlabel("Environment steps")
plt.ylabel("Evaluation return")
plt.title("SAC REINFORCE on InvertedPendulum-v4")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("sac_invpend_reinforce_eval_return_clean.png", dpi=200)
print("Saved: sac_invpend_reinforce_eval_return_clean.png")
