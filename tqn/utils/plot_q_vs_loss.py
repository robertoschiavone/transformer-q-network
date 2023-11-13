import numpy as np
from matplotlib import pyplot as plt

from tqn.utils.colormap import COLORBLIND_COLORMAP
from tqn.utils.preprocess import rolling_average


def plot_q_vs_loss(env_id: str, q_values: dict[str, np.ndarray],
                   losses: dict[str, np.ndarray]) -> None:
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4), sharex=True,
                                   tight_layout=True)
    acrobot_lunarlander_xticks = [0, 5_000, 10_000, 15_000, 20_000, 25_000]
    cartpole_xticks = [0, 40, 80, 120, 160, 200]
    acrobot_lunarlander_xlabels = ["0", "20", "40", "60", "80", "100"]
    cartpole_xlabels = ["0", "10", "20", "30", "40", "50"]
    xticks = acrobot_lunarlander_xticks if env_id in [
        "Acrobot-v1",
        "LunarLander-v2"] else cartpole_xticks
    xlabels = acrobot_lunarlander_xlabels if env_id in [
        "Acrobot-v1",
        "LunarLander-v2"] else cartpole_xlabels
    ax1.set_xticks(xticks, xlabels)

    ax1.set_title("Q-values")
    ax2.set_title("Loss")

    for i, model in enumerate(sorted(q_values.keys())):
        y = np.mean(q_values[model], axis=0)
        y = rolling_average(y, len(y) // 100)
        ax1.plot(y, label=f"{model}", color=COLORBLIND_COLORMAP[i])

    for i, model in enumerate(sorted(q_values.keys())):
        y = np.mean(losses[model], axis=0)
        y = rolling_average(y, len(y) // 100)
        ax2.plot(y, label=f"{model}", color=COLORBLIND_COLORMAP[i])

    ax1.grid(True)
    ax2.grid(True)
    ax1.legend()
    ax2.legend()
    fig.supxlabel("Number of Steps (in Thousands)")
    fig.suptitle(env_id, fontweight="bold")
    filename = f"q-vs-loss_{'-'.join(q_values.keys())}_{env_id}"
    plt.savefig(f"./plots/{filename}.pdf")
    plt.savefig(f"./plots/{filename}.png")
