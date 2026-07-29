import glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

patterns = {
    "single-Q": "data/*hopper_singlecritic*",
    "double-Q": "data/*hopper_doubleq*",
    "clipped-Q": "data/*hopper_clipq*",
}

for name, pattern in patterns.items():
    runs = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not runs:
        print(name, "NO RUN")
        continue

    run = runs[-1]
    ea = EventAccumulator(run, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]

    print("\n===", name, "===")
    print("run:", run)

    for tag in ["eval_return", "q_values"]:
        if tag not in tags:
            print(tag, "not found")
            continue
        vals = ea.Scalars(tag)
        steps = [v.step for v in vals]
        ys = [v.value for v in vals]
        print(f"{tag}: last_step={steps[-1]}, last={ys[-1]:.2f}, max={max(ys):.2f}, min={min(ys):.2f}")
