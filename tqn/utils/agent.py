import os
from functools import cached_property, cache

import gymnasium as gym
import numpy as np
import torch as T
from gymnasium.wrappers import RecordEpisodeStatistics, AutoResetWrapper
from torch import nn, optim as O
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.rich import trange

from tqn.data.batch import Batch
from tqn.data.replay_buffer import ReplayBuffer
from tqn.net.dqn import DQN
from tqn.net.tqn import TQN


class Agent:
    def __init__(self,
                 env_id: str,
                 seed: int,
                 device: T.device,
                 learning_rate: float = 1e-3,
                 buffer_size: int = 1_000_000,
                 learning_starts: int = 50_000,
                 tau: float = 1,
                 gamma: float = 0.99,
                 train_frequency: int = 4,
                 target_update_interval: int = 10_000,
                 exploration_fraction: float = 0.01,
                 epsilon_max: float = 1.0,
                 epsilon_min: float = 0.05,
                 network: str = "DQN"):
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.train_frequency = train_frequency
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min

        self.env = gym.make(env_id, max_episode_steps=500)
        self.env = AutoResetWrapper(self.env)
        self.env = RecordEpisodeStatistics(self.env)
        self.env.observation_space.seed(seed)
        self.env.action_space.seed(seed)

        self.state, _ = self.env.reset(seed=seed)

        observation_shape = self.env.observation_space.shape[0]
        action_shape = self.env.action_space.n

        Network = DQN if network == "DQN" else TQN

        self.policy = Network(observation_shape, action_shape)
        self.policy.to(device)

        self.target = Network(observation_shape, action_shape)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.target.to(device)

        self.optimizer = O.Adam(self.policy.parameters(), learning_rate)

        self.memory = ReplayBuffer(buffer_size, observation_shape, 1, seed)

        self.log_dir = f"./logs/{env_id}/{self.policy.__class__.__name__}"
        self.model_dir = f"./models/{env_id}/{self.policy.__class__.__name__}"

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.run_name = f"{seed}"
        log_file = os.path.join(self.log_dir, self.run_name)
        self.writer = SummaryWriter(log_file)

        self.rng = np.random.default_rng(seed)

        self.best_score = float("-inf")
        self.episode = 0

    @cached_property
    def device(self) -> T.device:
        return next(self.policy.parameters()).device

    @cache
    def epsilon_decay(self, max_steps: int) -> float:
        total_iterations = max_steps * self.exploration_fraction
        epsilon_decay = (self.epsilon_max - self.epsilon_min) / total_iterations
        return epsilon_decay

    def get_epsilon(self, global_step: int, max_steps: int) -> float:
        epsilon_decay = self.epsilon_decay(max_steps)
        epsilon_min = self.epsilon_min
        epsilon_max = self.epsilon_max
        epsilon = max(-epsilon_decay * global_step + epsilon_max, epsilon_min)
        return epsilon

    def choose_action(self, net: nn.Module, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = T.as_tensor(self.state)
            state = state.to(self.device)
            state = state.view((1, -1))
            q_values = net(state)
            action = T.argmax(q_values).item()

        return action

    def step(self, global_step: int, max_steps: int) -> None:
        epsilon = self.get_epsilon(global_step, max_steps)
        self.writer.add_scalar("charts/epsilon", epsilon, global_step)

        action = self.choose_action(self.policy, epsilon)

        next_state, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        if done:
            score = info["episode"]["r"]
            length = info["episode"]["l"]
            self.writer.add_scalar("charts/score", score, self.episode)
            self.writer.add_scalar("charts/length", length, self.episode)
            self.episode += 1
            if score > self.best_score:
                self.best_score = score
                model_name = f"{self.run_name}.zip"
                model_file = os.path.join(self.model_dir, model_name)
                T.save(self.policy, model_file)

        transition = (self.state, action, next_state, reward, float(done))
        self.memory.add(*transition)

        self.state = next_state

    def update_target_network(self) -> None:
        if self.tau == 1.:
            self.target.load_state_dict(self.policy.state_dict())
        else:
            parameters = zip(self.policy.parameters(), self.target.parameters())
            for from_param, to_param in parameters:
                updated_value = (self.tau * from_param.data +
                                 (1 - self.tau) * to_param.data)
                to_param.data.copy_(updated_value)

    def get_predictions(self, batch: Batch) -> T.Tensor:
        state, action, *_ = batch
        state = state.to(self.device)
        action = action.to(self.device)
        predictions = self.policy(state).gather(1, action).squeeze()
        return predictions

    @T.no_grad()
    def get_targets(self, batch: Batch) -> T.Tensor:
        _, _, next_state, reward, done = batch
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)
        next_q = self.target(next_state).max(dim=1).values
        next_q[done.squeeze().bool()] = 0.
        targets = reward.squeeze() + self.gamma * next_q
        return targets

    def fit(self, max_steps: int, gradient_clip_val: float,
            accumulate_grad_batches: int, batch_size: int) -> None:
        for global_step in trange(max_steps):
            self.step(global_step, max_steps)

            if global_step % self.target_update_interval == 0:
                self.update_target_network()

            if global_step >= self.learning_starts and \
                    (global_step % self.train_frequency) == 0:
                batch = self.memory.sample(batch_size)

                self.optimizer.zero_grad(set_to_none=True)

                predictions = self.get_predictions(batch)
                self.writer.add_scalar("charts/q_values",
                                       predictions.mean().item(), global_step)
                targets = self.get_targets(batch)

                loss = F.mse_loss(predictions, targets)
                self.writer.add_scalar("charts/loss", loss, global_step)
                loss.backward()

                T.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                           gradient_clip_val)

                if global_step % accumulate_grad_batches == 0:
                    self.optimizer.step()

    def __del__(self):
        self.writer.close()
        self.env.close()
