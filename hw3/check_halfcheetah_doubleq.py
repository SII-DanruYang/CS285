import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = sorted(glob.glob("data/*doubleq*HalfCheetah*"), key=os.path.getmtime)

print("Found runs:")
for r in runs:
    print(" ", r)

if not runs:
    print("\nNo doubleq run found.")
    raise SystemExit

for run in runs:
    print("\n" + "=" * 80)
    print("Run:", run)

    try:
        ea = EventAccumulator(run, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        print("tags:", tags)

        if "eval_return" not in tags:
            print("No eval_return found.")
            continue

        vals = ea.Scalars("eval_return")
        steps = [v.step for v in vals]
        returns = [v.value for v in vals]

        print("num eval points:", len(vals))
        print("last_step:", steps[-1])
        print("final_eval_return:", returns[-1])
        print("max_eval_return:", max(returns))

        if steps[-1] >= 990000:
            print("STATUS: COMPLETE")
        else:
            print("STATUS: PARTIAL")
    except Exception as e:
        print("ERROR reading run:", e)
