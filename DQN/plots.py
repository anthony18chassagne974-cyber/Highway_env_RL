import numpy as np
import matplotlib.pyplot as plt


def moving_average(values, window=20):
    values = np.array(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_rewards(logs_dict, window=20):
    plt.figure(figsize=(12, 5))
    for name, logs in logs_dict.items():
        rewards = logs["episode_rewards"]
        smoothed = moving_average(rewards, window=window)
        x = np.arange(len(smoothed)) + window - 1
        plt.plot(x, smoothed, label=name)

    plt.title(f"Average reward / episode (moving average, window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_training_td_loss(logs_dict):
    plt.figure(figsize=(12, 5))
    for name, logs in logs_dict.items():
        if "episode_avg_losses" in logs:
            values = logs["episode_avg_losses"]
            x = np.arange(len(values))
            plt.plot(x, values, label=name)
        else:
            print(f"[WARN] {name}: pas de 'episode_avg_losses' dans les logs")

    plt.title("Average TD loss / episode")
    plt.xlabel("Episode")
    plt.ylabel("TD loss")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_eval_metric_boxplot(df, metric, ylabel=None):
    agents = list(df["agent"].unique())
    data = [df[df["agent"] == agent][metric].dropna().values for agent in agents]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data, tick_labels=agents)
    plt.title(metric)
    plt.ylabel(ylabel if ylabel is not None else metric)
    plt.grid(True)
    plt.show()


def plot_eval_metric_bar(df, metric, ylabel=None):
    grouped = df.groupby("agent")[metric].agg(["mean", "std"]).reset_index()

    plt.figure(figsize=(10, 5))
    plt.bar(grouped["agent"], grouped["mean"], yerr=grouped["std"], capsize=5)
    plt.title(metric)
    plt.ylabel(ylabel if ylabel is not None else metric)
    plt.grid(True, axis="y")
    plt.show()


def plot_benchmark_summary(df):
    metrics = [
        "reward",
        "avg_speed",
        "episode_length",
        "lane_changes",
        "mean_abs_acceleration",
        "mean_abs_jerk",
        "progress_x",
        "collision",
    ]

    for metric in metrics:
        plot_eval_metric_bar(df, metric, ylabel=metric)