import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = sorted(glob.glob("data/*sanity_invpendulum_reinforce*"), key=os.path.getmtime)

print("runs:")
for r in runs:
    print(" ", r)

for run in runs:
    ea = EventAccumulator(run, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]

    if "eval_return" not in tags:
        print("\nSkip:", run, "no eval_return")
        continue

    vals = ea.Scalars("eval_return")
    steps = [v.step for v in vals]
    returns = [v.value for v in vals]

    print("\nUsing:", run)
    print("num eval points:", len(vals))
    print("first step:", steps[0], "first return:", returns[0])
    print("last step:", steps[-1], "last return:", returns[-1])
    print("max return:", max(returns))
