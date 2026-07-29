import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

patterns = {
    "REINFORCE-1": "data/hw3_sac_reinforce1_HalfCheetah-v4_*",
    "REINFORCE-10": "data/hw3_sac_reinforce10_HalfCheetah-v4_*",
    "REPARAM": "data/hw3_sac_reparametrize_HalfCheetah-v4_*",
}

for name, pattern in patterns.items():
    print("\n" + "=" * 80)
    print(name)
    runs = sorted(glob.glob(pattern), key=os.path.getmtime)

    if not runs:
        print("No run found.")
        continue

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
        final_return = returns[-1]
        max_return = max(returns)

        print("run:", run)
        print("  last_step:", last_step)
        print("  final_eval_return:", final_return)
        print("  max_eval_return:", max_return)

        candidate = (last_step, max_return, final_return, run)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    if best is not None:
        last_step, max_return, final_return, run = best
        print("\nSelected best complete-looking run:")
        print(" ", run)
        print(" last_step:", last_step)
        print(" final_eval_return:", final_return)
        print(" max_eval_return:", max_return)
