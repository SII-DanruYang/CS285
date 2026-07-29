import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUTDIR = Path("figures_hw3_fixed")
OUTDIR.mkdir(exist_ok=True)

def get_run_dir_from_log(log_file):
    text = Path(log_file).read_text(errors="ignore")
    matches = re.findall(r"logging outputs to\s+(.+)", text)
    if not matches:
        raise RuntimeError(f"No data dir found in {log_file}")
    run_dir = os.path.normpath(matches[-1].strip())
    if not os.path.isdir(run_dir):
        raise RuntimeError(f"Run dir does not exist: {run_dir}")
    return run_dir

def load_scalar(run_dir, tag):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    vals = ea.Scalars(tag)
    xs = [v.step for v in vals]
    ys = [v.value for v in vals]
    return xs, ys

specs = [
    ("DQN seed1",     "lunar_dqn_seed1_clean.log"),
    ("DQN seed2",     "lunar_dqn_seed2_clean.log"),
    ("DQN seed3",     "lunar_dqn_seed3_clean.log"),
    ("DoubleQ seed1", "lunar_doubleq_seed1.log"),
    ("DoubleQ seed2", "lunar_doubleq_seed2.log"),
    ("DoubleQ seed3", "lunar_doubleq_seed3.log"),
]

plt.figure(figsize=(8, 5))

print("=== LunarLander-v2 fixed plot ===")
for label, log_file in specs:
    run_dir = get_run_dir_from_log(log_file)
    xs, ys = load_scalar(run_dir, "eval_return")
    plt.plot(xs, ys, label=label)
    print(f"{label:<16} last_step={xs[-1]:<8} last={ys[-1]:.3f} max={max(ys):.3f} min={min(ys):.3f}")
    print("  run =", run_dir)

plt.title("LunarLander-v2: DQN vs Double DQN")
plt.xlabel("Environment steps")
plt.ylabel("Evaluation return")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

out = OUTDIR / "q2_lunar_dqn_vs_doubleq_eval_return.png"
plt.savefig(out, dpi=200)
plt.close()

print("[SAVED]", out)
