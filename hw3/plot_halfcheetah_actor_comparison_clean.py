import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = {
    "REINFORCE-1": "data/hw3_sac_reinforce1_HalfCheetah-v4_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.2_stu0.005_19-06-2026_12-34-56",
    "REINFORCE-10": "data/hw3_sac_reinforce10_HalfCheetah-v4_reinforce_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.2_stu0.005_19-06-2026_12-45-51",
    "Reparameterization": "data/hw3_sac_reparametrize_HalfCheetah-v4_reparametrize_s128_l3_alr0.0003_clr0.0003_b128_d0.99_t0.1_stu0.005_19-06-2026_12-46-28",
}

plt.figure(figsize=(8, 5))

for label, run in runs.items():
    ea = EventAccumulator(run, size_guidance={"scalars": 0})
    ea.Reload()

    vals = ea.Scalars("eval_return")
    steps = [v.step for v in vals]
    returns = [v.value for v in vals]

    print("=" * 80)
    print(label)
    print("run:", run)
    print("last_step:", steps[-1])
    print("final_eval_return:", returns[-1])
    print("max_eval_return:", max(returns))

    plt.plot(steps, returns, label=label)

plt.xlabel("Environment steps")
plt.ylabel("Evaluation return")
plt.title("HalfCheetah-v4 SAC Actor Gradient Estimator Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("halfcheetah_actor_estimator_comparison_clean.png", dpi=200)

print("Saved: halfcheetah_actor_estimator_comparison_clean.png")
