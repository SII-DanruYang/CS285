from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA_DIR = Path("data")
OUT_DIR = Path("figures_hw5")
OUT_DIR.mkdir(exist_ok=True)

# These are the four newest CQL alpha=0.1 runs from the data-ablation loop.
# Assuming the loop order was 1000, 5000, 10000, 20000.
run_dirs = [
    ("RND dataset 1000 steps",  DATA_DIR / "hw5_offline_PointmassMedium-v0_cql0.1_07-07-2026_12-42-11"),
    ("RND dataset 5000 steps",  DATA_DIR / "hw5_offline_PointmassMedium-v0_cql0.1_07-07-2026_12-52-02"),
    ("RND dataset 10000 steps", DATA_DIR / "hw5_offline_PointmassMedium-v0_cql0.1_07-07-2026_13-01-51"),
    ("RND dataset 20000 steps", DATA_DIR / "hw5_offline_PointmassMedium-v0_cql0.1_07-07-2026_13-11-38"),
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

def smooth(vals, weight=0.6):
    if len(vals) == 0:
        return vals
    out = []
    last = vals[0]
    for v in vals:
        last = last * weight + (1 - weight) * v
        out.append(last)
    return np.array(out)

print("Using runs:")
for label, d in run_dirs:
    print(label, "->", d, "exists:", d.exists())

# 1. Required main figure: evaluation return
plt.figure(figsize=(8, 5))
for label, d in run_dirs:
    steps, vals = load_scalar(d, "eval_return")
    if steps is not None:
        plt.plot(steps, vals, label=label)

plt.xlabel("Training step")
plt.ylabel("Evaluation return")
plt.title("PointmassMedium CQL Data Ablation: Evaluation Return")
plt.legend()
plt.tight_layout()
out = OUT_DIR / "q43_data_ablation_eval_return.png"
plt.savefig(out, dpi=200)
print("Saved:", out)

# 2. Diagnostic figure: Q-values
plt.figure(figsize=(8, 5))
for label, d in run_dirs:
    steps, vals = load_scalar(d, "q_values")
    if steps is not None:
        plt.plot(steps, smooth(vals), label=label)

plt.xlabel("Training step")
plt.ylabel("Mean Q-value")
plt.title("PointmassMedium CQL Data Ablation: Mean Q-values")
plt.legend()
plt.tight_layout()
out = OUT_DIR / "q43_data_ablation_q_values.png"
plt.savefig(out, dpi=200)
print("Saved:", out)

# 3. Diagnostic figure: target values
plt.figure(figsize=(8, 5))
for label, d in run_dirs:
    steps, vals = load_scalar(d, "target_values")
    if steps is not None:
        plt.plot(steps, smooth(vals), label=label)

plt.xlabel("Training step")
plt.ylabel("Mean target value")
plt.title("PointmassMedium CQL Data Ablation: Target Values")
plt.legend()
plt.tight_layout()
out = OUT_DIR / "q43_data_ablation_target_values.png"
plt.savefig(out, dpi=200)
print("Saved:", out)
