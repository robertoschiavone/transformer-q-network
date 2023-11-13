import numpy as np
from matplotlib import pyplot as plt

from tqn.utils.preprocess import rolling_average


def plot_score_vs_episode(env_id: str, scores: dict[str, np.ndarray],
                          config: dict[str, int]) -> None:
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4), tight_layout=True)

    for i, (key, values) in enumerate(scores.items()):
        y = np.mean(values, axis=0)
        x = range(len(y))
        axes[i].plot(x, y, color="b", label="Scores")

        ra = rolling_average(y, len(y) // 10)
        x = range(len(y) - len(ra), len(y))
        axes[i].plot(x, ra, color="r", label="Rolling Average")

        axes[i].axhline(config["target_score"], linestyle="--", color="k",
                        label="Target Score")

        axes[i].set_ylim([config["min_score"] - config["tick_pad"],
                          config["max_score"] - config["tick_pad"]])
        axes[i].set_yticks(np.arange(config["min_score"],
                                     config["max_score"],
                                     config["tick_interval"]))

        axes[i].tick_params(direction="inout")
        axes[i].grid(visible=True, which="both", axis="both")

        axes[i].set_xlabel("Episode Number")
        axes[i].set_ylabel("Score")

        axes[i].set_title(key)
        axes[i].legend()

    fig.suptitle(env_id, fontweight="bold")
    filename = f"score-vs-episode_{'-'.join(scores.keys())}_{env_id}"
    fig.savefig(f"./plots/{filename}.pdf")
    fig.savefig(f"./plots/{filename}.png")
