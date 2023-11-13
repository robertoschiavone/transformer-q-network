import warnings

import gymnasium as gym
import matplotlib.pyplot as plt


def plot_environments() -> None:
    warnings.simplefilter("ignore", category=UserWarning)
    environments = ["Acrobot-v1", "CartPole-v1", "LunarLander-v2"]

    fig, axes = plt.subplots(ncols=3, figsize=(12, 4), sharex=False,
                             sharey=False, tight_layout=True)

    for ax, env_id in zip(axes, environments):
        env = gym.make(env_id, render_mode="rgb_array")
        env.reset(seed=1693526400)
        for _ in range(20):
            env.step(env.action_space.sample())

        frame = env.render()
        height, width, _ = frame.shape

        ax.set_title(env_id)
        ax.set_xlim([0, width])
        ax.tick_params(direction="inout")
        ax.set_xticks(list(range(0, width + 1, 100)))

        ax.set_ylim([height, 0])
        ax.set_yticks(list(range(0, height + 1, 100)),
                      list(range(height, -1, -100)))

        ax.imshow(frame)
    filename = "environments"
    fig.savefig(f"./plots/{filename}.pdf")
    fig.savefig(f"./plots/{filename}.png")
