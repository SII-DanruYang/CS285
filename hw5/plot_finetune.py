from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA_DIR = Path("data")
OUT_DIR = Path("figures_hw5")
OUT_DIR.mkdir(exist_ok=True)

run_dir = sorted(DATA_DIR.glob("hw5_finetune_PointmassHard-v0_cql1.0_*"))[-1]
offline_steps = 100000

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

print("Using run:", run_dir)

# Evaluation return, required by the assignment.
steps, vals = load_scalar(run_dir, "eval_return")
print("Eval steps:", steps)
print("Eval returns:", vals)
print("Best eval return:", vals.max())
print("Final eval return:", vals[-1])

plt.figure(figsize=(8, 5))
plt.plot(steps, vals, marker="o", label="CQL fine-tuning")
plt.axvline(offline_steps, linestyle="--", label="online fine-tuning begins")
plt.axhline(-20, linestyle=":", label="target return = -20")
plt.xlabel("Training step")
plt.ylabel("Evaluation return")
plt.title("PointmassHard CQL Online Fine-Tuning")
plt.legend()
plt.tight_layout()
out = OUT_DIR / "q5_finetune_eval_return.png"
plt.savefig(out, dpi=200)
print("Saved:", out)

# Q-values diagnostic.
steps_q, vals_q = load_scalar(run_dir, "q_values")
if steps_q is not None:
    plt.figure(figsize=(8, 5))
    plt.plot(steps_q, vals_q, label="Mean Q-value")
    plt.axvline(offline_steps, linestyle="--", label="online fine-tuning begins")
    plt.xlabel("Training step")
    plt.ylabel("Mean Q-value")
    plt.title("PointmassHard CQL Fine-Tuning: Q-values")
    plt.legend()
    plt.tight_layout()
    out = OUT_DIR / "q5_finetune_q_values.png"
    plt.savefig(out, dpi=200)
    print("Saved:", out)

# Buffer size diagnostic.
steps_b, vals_b = load_scalar(run_dir, "buffer_size")
if steps_b is not None:
    plt.figure(figsize=(8, 5))
    plt.plot(steps_b, vals_b, label="Replay buffer size")
    plt.axvline(offline_steps, linestyle="--", label="online fine-tuning begins")
    plt.xlabel("Training step")
    plt.ylabel("Buffer size")
    plt.title("Replay Buffer Growth During Online Fine-Tuning")
    plt.legend()
    plt.tight_layout()
    out = OUT_DIR / "q5_finetune_buffer_size.png"
    plt.savefig(out, dpi=200)
    print("Saved:", out)
