import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = sorted(glob.glob("data/*sanity_invpendulum_reinforce*"), key=os.path.getmtime)
print("runs:")
for r in runs[-3:]:
    print(" ", r)

run = runs[-1]
ea = EventAccumulator(run)
ea.Reload()
tags = ea.Tags()["scalars"]
print("Using:", run)
print("tags:", tags)

candidates = [t for t in tags if "eval" in t.lower() and "return" in t.lower()]
print("eval return candidates:", candidates)

tag = candidates[0]
vals = ea.Scalars(tag)
steps = [v.step for v in vals]
returns = [v.value for v in vals]

print("tag:", tag)
print("final step:", steps[-1])
print("final eval_return:", returns[-1])
print("max eval_return:", max(returns))
