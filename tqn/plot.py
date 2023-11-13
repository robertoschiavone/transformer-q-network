import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from tqdm.rich import tqdm

from tqn.utils.plot_environments import plot_environments
from tqn.utils.plot_q_vs_frame import plot_q_vs_frame
from tqn.utils.plot_q_vs_loss import plot_q_vs_loss
from tqn.utils.plot_q_vs_reward import plot_q_vs_reward
from tqn.utils.plot_sample_efficiency import \
    plot_sample_efficiency_probability_improvement
from tqn.utils.plot_score_vs_episode import plot_score_vs_episode
from tqn.utils.plot_statistics import plot_statistics
from tqn.utils.preprocess import upsample


def get_config(env_id: str) -> dict:
    if env_id == "Acrobot-v1":
        return {
            "min_score": -500,
            "max_score": 0,
            "target_score": -100,
            "tick_interval": 100,
            "tick_pad": 50,
        }
    elif env_id == "CartPole-v1":
        return {
            "min_score": 0,
            "max_score": 600,
            "target_score": 500,
            "tick_interval": 100,
            "tick_pad": 50,
        }
    elif env_id == "LunarLander-v2":
        return {
            "min_score": -300,
            "max_score": 300,
            "target_score": 200,
            "tick_interval": 100,
            "tick_pad": 50,
        }
    else:
        raise ValueError(f"Environment {env_id} is not supported.")


def load_values(env_id: str, models: list[str]) \
        -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
        dict[str, np.ndarray]]:
    scores = {key: [] for key in models}
    q_values = {key: [] for key in models}
    losses = {key: [] for key in models}

    for model in sorted(models):
        log_folder = f"logs/{env_id}/{model}"
        seeds = os.listdir(log_folder)
        max_len = 0
        scores[model] = [[] for _ in range(len(seeds))]
        print(f"Loading {model} data...")
        for n, seed in enumerate(tqdm(seeds)):
            log = os.path.join(log_folder, seed)
            ea = event_accumulator.EventAccumulator(log, size_guidance={
                event_accumulator.SCALARS: 0
            })
            ea.Reload()

            q_values[model] += [[q_values.value for q_values in
                                 ea.Scalars("charts/q_values")]]
            losses[model] += [[loss.value for loss in
                               ea.Scalars("charts/loss")]]

            for i, score in enumerate(ea.Scalars("charts/score")):
                if i == len(scores[model][n]):
                    scores[model][n] += [score.value]
                else:
                    scores[model][n][i] += score.value
                if i >= max_len:
                    max_len = i + 1

        for i in range(len(scores[model])):
            scores[model][i] = upsample(scores[model][i], max_len)

    return scores, q_values, losses


def plot(env_id: str, plot_type: str) -> None:
    print(f"Environment: {env_id}")

    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"

    if plot_type == "environments":
        plot_environments()
    elif plot_type == "q-vs-frame":
        plot_q_vs_frame(env_id)
    elif plot_type == "q-vs-reward":
        plot_q_vs_reward(env_id)
    else:
        logs_folder = f"./logs/{env_id}/"
        models = sorted(os.listdir(logs_folder))
        config = get_config(env_id)
        scores, q_values, losses = load_values(env_id, models)
        match plot_type:
            case "q-vs-loss":
                plot_q_vs_loss(env_id, q_values, losses)
            case "sample-efficiency-probability-improvement":
                plot_sample_efficiency_probability_improvement(env_id, scores,
                                                               config)
            case "score-vs-episode":
                plot_score_vs_episode(env_id, scores, config)
            case "statistics":
                plot_statistics(env_id, scores, config)
