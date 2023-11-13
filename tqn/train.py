import json
import os

import torch as T

from tqn.utils.agent import Agent
from tqn.utils.count_parameters import count_parameters
from tqn.utils.seed_everything import seed_everything


def train(env_id: str, seed: int, network: str):
    seed_everything(seed)

    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    config_dir = "./config"
    config_name = f"{env_id}.json"
    config_file = os.path.join(config_dir, config_name)

    if os.path.exists(config_file):
        with open(config_file) as config_file:
            config = json.load(config_file)
    else:
        config = {}

    max_steps = config.pop("max_steps", 1_000_000)
    gradient_clip_val = config.pop("gradient_clip_val", 10.)
    accumulate_grad_batches = config.pop("accumulate_grad_batches", 1)
    batch_size = config.pop("batch_size", 32)

    agent = Agent(env_id, seed, device, **config, network=network)

    print(f"Environment: {env_id}")
    print(f"Observation shape: {agent.env.observation_space.shape[0]}")
    print(f"Action shape: {agent.env.action_space.n}")

    print(f"\nDevice: {device}\n")

    count_parameters(agent.policy)

    agent.fit(max_steps, gradient_clip_val, accumulate_grad_batches, batch_size)
