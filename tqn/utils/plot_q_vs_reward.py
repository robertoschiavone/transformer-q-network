import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm.rich import tqdm

from tqn.utils.evaluate_policy import evaluate_policy
from tqn.utils.preprocess import normalize_minus1_1
from tqn.utils.seed_everything import seed_everything


def plot_q_vs_reward(env_id: str) -> None:
    nbins = 100
    models_folder = f"./models/{env_id}/"
    models = os.listdir(models_folder)

    fig, axes = plt.subplots(ncols=2, figsize=(4 * len(models), 4), sharex=True,
                             tight_layout=True)
    fig.supxlabel("Reward")
    axes[0].set_ylabel("Estimated Q")

    for ax in axes:
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(direction="inout")

    for m, model in enumerate(sorted(models)):
        total_q_values = []
        total_rewards = []
        episode_rewards = []
        model_path = os.path.join(models_folder, model)
        seeds = os.listdir(model_path)
        seeds = [int(seed.replace(".zip", "")) for seed in seeds]
        pbar = tqdm(total=len(seeds) ** 2)
        for seed in seeds:
            model_file = os.path.join(model_path, f"{seed}.zip")
            for i in seeds:
                seed_everything(int(seed))
                (q_values, rewards), (episode_reward, _), _ = evaluate_policy(
                    env_id, model_file, seed=int(i))
                total_q_values += [*normalize_minus1_1(np.array(q_values))]
                total_rewards += [*normalize_minus1_1(np.array(rewards))]
                episode_rewards += [episode_reward]
                pbar.update(1)
        pbar.close()

        x = np.array(total_rewards)
        y = np.array(total_q_values)

        k = scipy.stats.gaussian_kde((x, y))

        xi, yi = np.mgrid[x.min():x.max():nbins * 1j,
                 y.min():y.max():nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        axes[m].set_title(model)
        axes[m].pcolormesh(xi, yi, zi.reshape(xi.shape), shading="gouraud",
                           cmap=plt.cm.Greys)
        axes[m].plot([-1, 1], [-1, 1], linestyle="dashed", c="k")

    fig.suptitle(env_id, fontweight="bold")
    filename = f"q-vs-reward_{env_id}"
    fig.savefig(f"./plots/{filename}.pdf")
    fig.savefig(f"./plots/{filename}.png")
