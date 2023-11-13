import gymnasium as gym
import torch as T
from gymnasium.wrappers import RecordEpisodeStatistics

from tqn.utils.play_episode import play_episode


def play(env_id: str, model: str, seed: int | None = None):
    return evaluate_policy(env_id, model, seed, render_mode="human")


def evaluate_policy(env_id: str, model: str, seed: int | None = None,
                    render_mode: str | None = None):
    env = gym.make(env_id, max_episode_steps=500, render_mode=render_mode)
    env = RecordEpisodeStatistics(env)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    frames = None

    net = T.load(model)
    net.eval()

    (
        (q_value, reward),
        (episode_reward, episode_length),
    ) = play_episode(net, env)

    if render_mode:
        frames = env.render()

    return (q_value, reward), (episode_reward, episode_length), frames
