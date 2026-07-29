import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_run(pattern, require_last_step=None):
    runs = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not runs:
        raise RuntimeError(f"No runs found for pattern: {pattern}")

    candidates = []
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
        candidates.append((last_step, max_return, run, ea, tags))

    if not candidates:
        raise RuntimeError(f"No valid TensorBoard scalar runs for pattern: {pattern}")

    # 优先选跑得最完整的 run，避免误读 partial run
    candidates.sort(key=lambda x: (x[0], x[1]))
    last_step, max_return, run, ea, tags = candidates[-1]

    print("=" * 80)
    print("Selected run:", run)
    print("last eval step:", last_step)
    print("max eval_return:", max_return)
    print("available tags:", tags)

    if require_last_step is not None and last_step < require_last_step:
        print(f"WARNING: last_step={last_step}, expected at least {require_last_step}")

    return run, ea, tags


def get_scalar(ea, tag):
    vals = ea.Scalars(tag)
    steps = [v.step for v in vals]
    values = [v.value for v in vals]
    return steps, values


# ============================================================
# 1. Pendulum sanity: eval_return
# ============================================================
pend_run, pend_ea, pend_tags = load_run(
    "data/*sanity_pendulum*",
    require_last_step=295000,
)

plt.figure()
steps, values = get_scalar(pend_ea, "eval_return")
plt.plot(steps, values, label="eval_return")
plt.xlabel("Environment steps")
plt.ylabel("Evaluation return")
plt.title("SAC Sanity Check on Pendulum-v1")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("sac_sanity_pendulum_eval_return.png", dpi=200)
print("Saved: sac_sanity_pendulum_eval_return.png")
print("Pendulum final eval_return:", values[-1])
print("Pendulum max eval_return:", max(values))


# ============================================================
# 2. Pendulum sanity: q_values vs target_values
# 这个图用来看 critic 的 Q 估计和 target 是否在同一量级
# ============================================================
plt.figure()
for tag in ["q_values", "target_values"]:
    if tag in pend_tags:
        steps, values = get_scalar(pend_ea, tag)
        plt.plot(steps, values, label=tag)

plt.xlabel("Environment steps")
plt.ylabel("Value")
plt.title("Pendulum-v1 Critic Sanity: Q Values vs Target Values")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("sac_sanity_pendulum_q_vs_target.png", dpi=200)
print("Saved: sac_sanity_pendulum_q_vs_target.png")


# ============================================================
# 3. Pendulum sanity: critic_loss
# ============================================================
if "critic_loss" in pend_tags:
    plt.figure()
    steps, values = get_scalar(pend_ea, "critic_loss")
    plt.plot(steps, values, label="critic_loss")
    plt.xlabel("Environment steps")
    plt.ylabel("Critic loss")
    plt.title("Pendulum-v1 Critic Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sac_sanity_pendulum_critic_loss.png", dpi=200)
    print("Saved: sac_sanity_pendulum_critic_loss.png")


# ============================================================
# 4. InvertedPendulum REINFORCE sanity: eval_return
# 注意：自动选择 last_step 最大的完整 run，避免选到 25000 step 的 partial run
# ============================================================
inv_run, inv_ea, inv_tags = load_run(
    "data/*sanity_invpendulum_reinforce*",
    require_last_step=295000,
)

plt.figure()
steps, values = get_scalar(inv_ea, "eval_return")
plt.plot(steps, values, label="eval_return")
plt.xlabel("Environment steps")
plt.ylabel("Evaluation return")
plt.title("SAC REINFORCE Sanity Check on InvertedPendulum-v4")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("sac_sanity_invpendulum_reinforce_eval_return.png", dpi=200)
print("Saved: sac_sanity_invpendulum_reinforce_eval_return.png")
print("InvertedPendulum final eval_return:", values[-1])
print("InvertedPendulum max eval_return:", max(values))


# ============================================================
# 5. InvertedPendulum REINFORCE: train_return 和 eval_return 同图
# ============================================================
plt.figure()

steps, values = get_scalar(inv_ea, "eval_return")
plt.plot(steps, values, label="eval_return")

if "train_return" in inv_tags:
    steps, values = get_scalar(inv_ea, "train_return")
    plt.plot(steps, values, label="train_return")

plt.xlabel("Environment steps")
plt.ylabel("Return")
plt.title("InvertedPendulum-v4 SAC REINFORCE: Train vs Eval Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("sac_sanity_invpendulum_reinforce_train_eval_return.png", dpi=200)
print("Saved: sac_sanity_invpendulum_reinforce_train_eval_return.png")
