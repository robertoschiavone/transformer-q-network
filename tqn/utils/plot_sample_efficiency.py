import copy

import numpy as np
from matplotlib import pyplot as plt
from rliable import library as rly
from rliable import metrics

from tqn.utils.colormap import COLORBLIND_COLORMAP
from tqn.utils.preprocess import downsample, normalize_0_tau


def plot_sample_efficiency(scores: dict[str, np.ndarray], ax: plt.Axes,
                           config: dict[str, int | str]) -> None:
    scores = copy.deepcopy(scores)
    for key, values in scores.items():
        for i, value in enumerate(values):
            scores[key][i] = downsample(value)
        scores[key] = np.transpose(normalize_0_tau(np.array(scores[key]),
                                                   config["target_score"]))

    sample_efficiency = {key: [] for key in scores.keys()}

    for key, values in scores.items():
        for i, value in enumerate(values):
            value = np.array(value).reshape(-1, 1)
            aggregate_scores, confidence_intervals = rly.get_interval_estimates(
                {key: value}, metrics.aggregate_iqm)
            sample_efficiency[key] += [{
                "min": confidence_intervals[key][0][0],
                "value": aggregate_scores[key],
                "max": confidence_intervals[key][1][0],
            }]

    for i, key in enumerate(scores.keys()):
        y_value = [elem["value"] for elem in sample_efficiency[key]]
        y_min = [elem["min"] for elem in sample_efficiency[key]]
        y_max = [elem["max"] for elem in sample_efficiency[key]]
        x = range(11)
        ax.plot(x, y_value, linewidth=2, marker="o", label=key,
                color=COLORBLIND_COLORMAP[i])
        ax.fill_between(x, y_min, y_max, alpha=0.2,
                        color=COLORBLIND_COLORMAP[i])

    ax.grid(visible=True, which="both", axis="both")
    ax.set_title(f"Sample Efficiency Curve")
    ax.set_xlabel("Number of Steps (in Thousands)")
    ax.set_ylabel(r"IQM Normalized Score ($\tau$)")
    ax.set_xticks([0, 2, 4, 6, 8, 10], ["0", "20", "40", "60", "80", "100"])
    ax.legend()


def plot_probability_improvement(scores: dict[str, np.ndarray], ax: plt.Axes,
                                 config: dict[str, int | str]) -> None:
    scores = copy.deepcopy(scores)
    for key, values in scores.items():
        scores[key] = np.transpose(normalize_0_tau(np.array(values),
                                                   config["target_score"]))

    ax.set_yticks([1], ["DQN"])
    ax2 = ax.twinx()
    ax2.set_ylim([0.5, 2.5])
    ax2.set_yticks([1], ["TQN"])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params(width=0)
    ax2.spines[["left", "right", "top"]].set_visible(False)
    ax2.tick_params(width=0)

    probability_improvement, confidence_intervals = rly.get_interval_estimates(
        {"x,y": scores.values()}, metrics.probability_of_improvement)

    ax.barh(y=1, left=confidence_intervals["x,y"][0][0],
            height=0.5,
            width=confidence_intervals["x,y"][1][0]
                  - confidence_intervals["x,y"][0][0],
            color=COLORBLIND_COLORMAP[0])
    ax.vlines(x=probability_improvement["x,y"],
              ymin=0.75, ymax=1.25, color="k")

    ax.set_title(f"Probability of Improvement")
    ax.set_xlabel(r"$P(X > Y)$")

    ax.set_xlim([0, 1])
    ax.set_ylim([0.5, 2.5])


def plot_sample_efficiency_probability_improvement(
        env_id: str, scores: dict[str, np.ndarray],
        config: dict[str, int | str]) -> None:
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4), tight_layout=True)
    plot_sample_efficiency(scores, axes[0], config)
    plot_probability_improvement(scores, axes[1], config)
    fig.suptitle(env_id, fontweight="bold")
    filename = (f"sample-efficiency-probability-improvement_"
                f"{'-'.join(scores.keys())}_{env_id}")
    fig.savefig(f"./plots/{filename}.pdf")
    fig.savefig(f"./plots/{filename}.png")
