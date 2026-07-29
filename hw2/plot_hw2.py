import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CANDIDATE_DATA_DIRS = ["data", "cs285/data"]
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


def existing_data_dirs():
    dirs = []
    for d in CANDIDATE_DATA_DIRS:
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


def find_runs(keyword):
    matched = []
    for data_dir in existing_data_dirs():
        pattern = os.path.join(data_dir, f"*{keyword}*")
        matched.extend(glob.glob(pattern))
    matched = sorted(set(matched))
    return matched


def read_scalars(logdir):
    event_files = glob.glob(os.path.join(logdir, "events.out.tfevents*"))
    if len(event_files) == 0:
        raise RuntimeError(f"No TensorBoard event file found in {logdir}")

    ea = EventAccumulator(logdir)
    ea.Reload()

    tags = ea.Tags().get("scalars", [])

    def get_tag(tag):
        if tag not in tags:
            return {}
        return {e.step: e.value for e in ea.Scalars(tag)}

    envsteps = get_tag("Train_EnvstepsSoFar")
    eval_return = get_tag("Eval_AverageReturn")
    baseline_loss = get_tag("Baseline_Loss")

    if not envsteps:
        raise RuntimeError(f"No Train_EnvstepsSoFar in {logdir}. Available tags: {tags}")

    if not eval_return:
        raise RuntimeError(f"No Eval_AverageReturn in {logdir}. Available tags: {tags}")

    common_steps = sorted(set(envsteps.keys()) & set(eval_return.keys()))

    data = {
        "envsteps": np.array([envsteps[s] for s in common_steps]),
        "eval_return": np.array([eval_return[s] for s in common_steps]),
    }

    if baseline_loss:
        common_bl = sorted(set(envsteps.keys()) & set(baseline_loss.keys()))
        data["baseline_envsteps"] = np.array([envsteps[s] for s in common_bl])
        data["baseline_loss"] = np.array([baseline_loss[s] for s in common_bl])

    return data


def plot_runs(keywords, title, filename, metric="eval_return"):
    plt.figure(figsize=(8, 5))
    plotted = 0

    for keyword in keywords:
        runs = find_runs(keyword)

        if len(runs) == 0:
            print(f"[Warning] no runs found for {keyword}")
            continue

        run = runs[-1]
        print(f"[Use] {keyword} -> {run}")

        try:
            data = read_scalars(run)
        except Exception as e:
            print(f"[Warning] failed to read {run}: {e}")
            continue

        if metric == "eval_return":
            x = data["envsteps"]
            y = data["eval_return"]
            ylabel = "Eval Average Return"
        elif metric == "baseline_loss":
            if "baseline_loss" not in data:
                print(f"[Warning] no baseline loss for {keyword}")
                continue
            x = data["baseline_envsteps"]
            y = data["baseline_loss"]
            ylabel = "Baseline_Loss"
        else:
            raise ValueError(metric)

        plt.plot(x, y, label=keyword)
        plotted += 1

    plt.xlabel("Environment steps")
    plt.ylabel(ylabel if plotted > 0 else metric)
    plt.title(title)

    if plotted > 0:
        plt.legend()
    else:
        plt.text(
            0.5,
            0.5,
            "No matching experiment logs found",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, dpi=200)
    print("saved:", out)


def plot_average_over_seeds(groups, title, filename):
    plt.figure(figsize=(8, 5))
    plotted = 0

    for label, keyword in groups:
        runs = find_runs(keyword)

        if len(runs) == 0:
            print(f"[Warning] no runs found for {keyword}")
            continue

        all_x = []
        all_y = []

        print(f"[Group] {label}: {len(runs)} runs")

        for run in runs:
            try:
                data = read_scalars(run)
            except Exception as e:
                print(f"[Warning] failed to read {run}: {e}")
                continue

            all_x.append(data["envsteps"])
            all_y.append(data["eval_return"])

        if len(all_y) == 0:
            continue

        min_len = min(len(y) for y in all_y)
        x = all_x[0][:min_len]
        ys = np.stack([y[:min_len] for y in all_y], axis=0)

        mean_y = ys.mean(axis=0)
        std_y = ys.std(axis=0)

        plt.plot(x, mean_y, label=label)
        plt.fill_between(x, mean_y - std_y, mean_y + std_y, alpha=0.2)
        plotted += 1

    plt.xlabel("Environment steps")
    plt.ylabel("Eval Average Return")
    plt.title(title)

    if plotted > 0:
        plt.legend()
    else:
        plt.text(
            0.5,
            0.5,
            "No matching experiment logs found",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    plt.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, dpi=200)
    print("saved:", out)


if __name__ == "__main__":
    print("Searching logs in:", existing_data_dirs())

    print("\nExisting experiment folders:")
    for d in existing_data_dirs():
        for x in sorted(glob.glob(os.path.join(d, "*")))[:50]:
            print(" ", x)

    # Experiment 1 small batch
    plot_runs(
        [
            "cartpole_CartPole-v0",
            "cartpole_rtg_CartPole-v0",
            "cartpole_na_CartPole-v0",
            "cartpole_rtg_na_CartPole-v0",
        ],
        "CartPole small batch",
        "exp1_cartpole_small.png",
    )

    # Experiment 1 large batch
    plot_runs(
        [
            "cartpole_lb_CartPole-v0",
            "cartpole_lb_rtg_CartPole-v0",
            "cartpole_lb_na_CartPole-v0",
            "cartpole_lb_rtg_na_CartPole-v0",
        ],
        "CartPole large batch",
        "exp1_cartpole_large.png",
    )

    # Experiment 2 eval return
    plot_runs(
        [
            "cheetah_HalfCheetah-v4",
            "cheetah_baseline_HalfCheetah-v4",
            "cheetah_baseline_bgs1_HalfCheetah-v4",
        ],
        "HalfCheetah eval return",
        "exp2_cheetah_eval_return.png",
    )

    # Experiment 2 baseline loss
    plot_runs(
        [
            "cheetah_baseline_HalfCheetah-v4",
            "cheetah_baseline_bgs1_HalfCheetah-v4",
        ],
        "HalfCheetah baseline loss",
        "exp2_cheetah_baseline_loss.png",
        metric="baseline_loss",
    )

    # Experiment 3
    plot_runs(
        [
            "lunar_lander_lambda0_LunarLander-v2",
            "lunar_lander_lambda0.95_LunarLander-v2",
            "lunar_lander_lambda0.98_LunarLander-v2",
            "lunar_lander_lambda0.99_LunarLander-v2",
            "lunar_lander_lambda1_LunarLander-v2",
        ],
        "LunarLander GAE lambda",
        "exp3_lunar_lander_lambda.png",
    )

    # Experiment 4
    plot_average_over_seeds(
        [
            ("default", "pendulum_default"),
            ("tuned", "pendulum_tuned_b1000_lr5e3_gae095"),
        ],
        "InvertedPendulum default vs tuned",
        "exp4_pendulum_default_vs_tuned.png",
    )
