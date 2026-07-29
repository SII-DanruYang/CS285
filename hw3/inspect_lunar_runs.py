import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = sorted(glob.glob("data/*LunarLander*"), key=os.path.getmtime)

for r in runs:
    try:
        ea = EventAccumulator(r, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if "eval_return" not in tags:
            print("[NO eval_return]", r)
            continue
        vals = ea.Scalars("eval_return")
        xs = [v.step for v in vals]
        ys = [v.value for v in vals]
        print(f"{os.path.basename(r)}")
        print(f"  n={len(xs)} last_step={xs[-1]} last={ys[-1]:.3f} max={max(ys):.3f} min={min(ys):.3f}")
    except Exception as e:
        print("[ERR]", r, e)
