from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA_DIR = Path("data")
OUT_DIR = Path("figures_hw5")
OUT_DIR.mkdir(exist_ok=True)

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

runs = []

# CQL alpha runs
for d in DATA_DIR.glob("hw5_offline_PointmassMedium-v0_cql*"):
    m = re.search(r"cql([0-9.]+)_", d.name)
    if m:
        alpha = float(m.group(1))
        runs.append((alpha, f"CQL alpha={alpha:g}", d))

# DQN baseline
for d in DATA_DIR.glob("hw5_offline_pointmass_medium_dqn*"):
    runs.append((-1, "DQN baseline", d))

runs = sorted(runs, key=lambda x: x[0])

print("Using runs:")
for _, label, d in runs:
    print(f"{label:18s} -> {d}")

def plot_scalar(tag, title, ylabel, filename, do_smooth=True):
    plt.figure(figsize=(8, 5))

    for _, label, d in runs:
        steps, vals = load_scalar(d, tag)
        if steps is None:
            continue

        plot_vals = smooth(vals) if do_smooth else vals
        plt.plot(steps, plot_vals, label=label)

    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / filename
    plt.savefig(out_path, dpi=200)
    print("Saved:", out_path)

plot_scalar(
    tag="eval_return",
    title="PointmassMedium Offline RL: Evaluation Return",
    ylabel="Evaluation return",
    filename="q4_cql_alpha_eval_return.png",
    do_smooth=False,
)

plot_scalar(
    tag="q_values",
    title="PointmassMedium CQL Alpha Sweep: Mean Q-values",
    ylabel="Mean Q-value",
    filename="q4_cql_alpha_q_values.png",
    do_smooth=True,
)

plot_scalar(
    tag="target_values",
    title="PointmassMedium CQL Alpha Sweep: Target Values",
    ylabel="Mean target value",
    filename="q4_cql_alpha_target_values.png",
    do_smooth=True,
)
