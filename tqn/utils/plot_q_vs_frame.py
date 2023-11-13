import os

import matplotlib.pyplot as plt

from tqn.utils.evaluate_policy import evaluate_policy
from tqn.utils.seed_everything import seed_everything


def plot_q_vs_frame(env_id: str):
    best_seeds = {
        "Acrobot-v1": {
            "DQN": 1694822400,
            "TQN": 1693699200
        },
        "CartPole-v1": {
            "DQN": 1694908800,
            "TQN": 1695081600
        },
        "LunarLander-v2": {
            "DQN": 1694131200,
            "TQN": 1694390400
        },
    }

    models_folder = f"./models/{env_id}/"
    models = os.listdir(models_folder)

    for model in sorted(models):
        seed = best_seeds[env_id][model]
        seed_everything(seed)
        model_path = os.path.join(models_folder, model, f"{seed}.zip")
        (q_value, _), _, frames = evaluate_policy(env_id, model_path,
                                                  seed, "rgb_array_list")

        frames = frames[25], frames[30], frames[35]
        y_value = q_value[25], q_value[30], q_value[35]

        fig, axes = plt.subplots(ncols=4, figsize=(24, 4), tight_layout=True)

        for i, frame in enumerate(frames):
            axes[i + 1].imshow(frame)
            axes[i + 1].set_xticks([])
            axes[i + 1].set_yticks([])

        for x, y, text in zip([5, 10, 15], y_value, ["A", "B", "C"]):
            axes[0].text(x, y, text, fontsize=20, color="b")

        axes[0].plot(q_value[20:41], marker="o", markevery=[5, 10, 15],
                     color="midnightblue", linestyle="dashed")
        axes[0].set_xlabel(r"Frame \#", fontsize=20)
        axes[0].set_ylabel("Q", fontsize=20)
        axes[0].set_xticks(range(0, len(q_value[20:41]), 5),
                           range(0, len(q_value[20:41]), 5), fontsize=16)
        axes[0].tick_params(axis="y", labelsize=16)
        filename = f"q-vs-frame_{model}_{env_id}"
        fig.savefig(f"./plots/{filename}.pdf")
        fig.savefig(f"./plots/{filename}.png")
