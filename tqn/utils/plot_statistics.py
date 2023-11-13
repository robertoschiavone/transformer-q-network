import numpy as np
from matplotlib import pyplot as plt
from rliable import library as rly
from rliable import metrics

from tqn.utils.colormap import COLORBLIND_COLORMAP
from tqn.utils.preprocess import normalize_0_tau


def plot_statistics(env_id: str, scores: dict[str, np.ndarray],
                    config: dict[str, int]) -> None:
    fig, axes = plt.subplots(ncols=4, figsize=(12, 3), sharey=True,
                             tight_layout=True)
    stats = ["Median", "IQM", "Mean", "Optimality Gap"]
    models = sorted(scores.keys())

    for stat, ax in zip(stats, axes):
        ax.set_title(stat)

    axes[0].set_yticks(range(1, len(models) + 1), models)
    axes[0].set_ylim([0.5, len(models) + 0.5])
    fig.supxlabel("Score")

    for ax in axes:
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.tick_params(width=0)

    for key, value in scores.items():
        scores[key] = np.transpose(normalize_0_tau(value,
                                                   config["target_score"]))

    aggregate_scores, confidence_intervals = rly.get_interval_estimates(
        scores, lambda x: np.array([
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x)
        ]))
    for i in range(len(stats)):
        for j, model in enumerate(models):
            axes[i].barh(y=(j + 1), left=confidence_intervals[model][0][i],
                         height=0.5,
                         width=confidence_intervals[model][1][i]
                               - confidence_intervals[model][0][i],
                         color=COLORBLIND_COLORMAP[j])
            axes[i].vlines(x=aggregate_scores[model][i],
                           ymin=j + 0.75, ymax=j + 1.25, color="k")

    fig.supxlabel(r"Normalized Score ($\tau$)")
    fig.suptitle(env_id, fontweight="bold")
    filename = f"statistics_{'-'.join(scores.keys())}_{env_id}"
    plt.savefig(f"./plots/{filename}.pdf")
    plt.savefig(f"./plots/{filename}.png")
