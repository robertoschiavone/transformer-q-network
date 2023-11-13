import gymnasium as gym
import torch as T
from torch import nn


@T.no_grad()
def play_episode(net: nn.Module, env: gym.Env, seed: int | None = None) -> \
        tuple[tuple[list[float], list[float]], tuple[float, int]]:
    device = next(net.parameters()).device
    done = False

    state, info = env.reset(seed=seed)

    q_values = []
    rewards = []

    while not done:
        state = T.as_tensor(state)
        state = state.view(1, -1)
        state = state.to(device)
        q_value = net(state).max().item()
        action = net(state).argmax().item()
        state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        q_values += [q_value]
        rewards += [reward]

    return (q_values, rewards), (info["episode"]["r"], info["episode"]["l"])
