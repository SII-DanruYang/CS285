from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA_DIR = Path("data")
OUT_DIR = Path("figures_hw5")
OUT_DIR.mkdir(exist_ok=True)

offline_steps = 100000

runs = []

alpha1 = sorted(DATA_DIR.glob("hw5_finetune_PointmassHard-v0_cql1.0_*"))
if alpha1:
    runs.append(("CQL fine-tune alpha=1.0", alpha1[-1]))

alpha01 = sorted(DATA_DIR.glob("hw5_finetune_PointmassHard-v0_cql0.1_*"))
if alpha01:
    runs.append(("CQL fine-tune alpha=0.1", alpha01[-1]))

print("Using runs:")
for label, d in runs:
    print(label, "->", d)

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

plt.figure(figsize=(8, 5))

for label, d in runs:
    steps, vals = load_scalar(d, "eval_return")
    if steps is None:
        continue
    print()
    print(label)
    print("Eval steps:", steps)
    print("Eval returns:", vals)
    print("Best eval return:", vals.max())
    print("Final eval return:", vals[-1])
    plt.plot(steps, vals, marker="o", label=label)

plt.axvline(offline_steps, linestyle="--", label="online fine-tuning begins")
plt.axhline(-20, linestyle=":", label="target return = -20")
plt.xlabel("Training step")
plt.ylabel("Evaluation return")
plt.title("PointmassHard CQL Online Fine-Tuning Comparison")
plt.legend()
plt.tight_layout()

out = OUT_DIR / "q5_finetune_eval_return_compare.png"
plt.savefig(out, dpi=200)
print("Saved:", out)

plt.figure(figsize=(8, 5))

for label, d in runs:
    steps, vals = load_scalar(d, "q_values")
    if steps is None:
        continue
    plt.plot(steps, vals, label=label)

plt.axvline(offline_steps, linestyle="--", label="online fine-tuning begins")
plt.xlabel("Training step")
plt.ylabel("Mean Q-value")
plt.title("PointmassHard CQL Fine-Tuning Q-values Comparison")
plt.legend()
plt.tight_layout()

out = OUT_DIR / "q5_finetune_q_values_compare.png"
plt.savefig(out, dpi=200)
print("Saved:", out)
