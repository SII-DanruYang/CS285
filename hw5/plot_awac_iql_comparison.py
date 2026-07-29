from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA_DIR = Path("data")
OUT_DIR = Path("figures_hw5")
OUT_DIR.mkdir(exist_ok=True)

runs = [
    (
        "AWAC, temp=10.0",
        sorted(DATA_DIR.glob("hw5_offline_PointmassHard-v0_awac10.0_*"))[-1],
    ),
    (
        "IQL, expectile=0.99, temp=10.0",
        sorted(DATA_DIR.glob("hw5_offline_PointmassHard-v0_iql0.99_temp10.0_*"))[-1],
    ),
]

def load_scalar(run_dir, tag):
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        print(f"[Missing] {tag} in {run_dir.name}")
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    vals = np.array([e.value for e in events])
    return steps, vals

print("Using runs:")
for label, run_dir in runs:
    print(f"{label:35s} -> {run_dir}")

# 1. Main required figure: evaluation return
plt.figure(figsize=(8, 5))
for label, run_dir in runs:
    steps, vals = load_scalar(run_dir, "eval_return")
    if steps is not None:
        plt.plot(steps, vals, label=label)

plt.xlabel("Training step")
plt.ylabel("Evaluation return")
plt.title("PointmassHard Offline RL: AWAC vs IQL")
plt.legend()
plt.tight_layout()
out = OUT_DIR / "q42_awac_iql_eval_return.png"
plt.savefig(out, dpi=200)
print("Saved:", out)

# 2. Diagnostic figure: Q-values
plt.figure(figsize=(8, 5))
for label, run_dir in runs:
    steps, vals = load_scalar(run_dir, "q_values")
    if steps is not None:
        plt.plot(steps, vals, label=label)

plt.xlabel("Training step")
plt.ylabel("Mean Q-value")
plt.title("PointmassHard Offline RL: AWAC vs IQL Q-values")
plt.legend()
plt.tight_layout()
out = OUT_DIR / "q42_awac_iql_q_values.png"
plt.savefig(out, dpi=200)
print("Saved:", out)

# 3. Diagnostic figure: actor loss
plt.figure(figsize=(8, 5))
for label, run_dir in runs:
    steps, vals = load_scalar(run_dir, "actor_loss")
    if steps is not None:
        plt.plot(steps, vals, label=label)

plt.xlabel("Training step")
plt.ylabel("Actor loss")
plt.title("PointmassHard Offline RL: AWAC vs IQL Actor Loss")
plt.legend()
plt.tight_layout()
out = OUT_DIR / "q42_awac_iql_actor_loss.png"
plt.savefig(out, dpi=200)
print("Saved:", out)
