import numpy as np
import torch as T

from tqn.data.batch import Batch
from tqn.data.transition import Transition


class ReplayBuffer:
    def __init__(self, buffer_size: int, observation_shape: int,
                 action_shape: int, seed: int):
        self.capacity = buffer_size

        self.state = T.empty((buffer_size, observation_shape),
                             dtype=T.float32, pin_memory=True)
        self.action = T.empty((buffer_size, action_shape),
                              dtype=T.int64, pin_memory=True)
        self.next_state = T.empty((buffer_size, observation_shape),
                                  dtype=T.float32, pin_memory=True)
        self.reward = T.empty((buffer_size, 1), pin_memory=True)
        self.done = T.empty((buffer_size, 1), pin_memory=True)

        self.rng = np.random.default_rng(seed)
        self.ptr = 0
        self.length = 0

    def add(self, *args: Transition) -> None:
        state, action, next_state, reward, done = args
        self.state[self.ptr] = T.as_tensor(state)
        self.action[self.ptr] = T.as_tensor(action)
        self.next_state[self.ptr] = T.as_tensor(next_state)
        self.reward[self.ptr] = T.as_tensor(reward)
        self.done[self.ptr] = T.as_tensor(done)

        self.length += 1
        full = self.ptr == (self.capacity - 1)
        self.ptr = 0 if full else self.ptr + 1

    def sample(self, batch_size: int) -> Batch:
        stop = min(self.length, self.capacity)
        indices = self.rng.integers(low=0, high=stop, size=batch_size)

        state = self.state[indices]
        action = self.action[indices]
        next_state = self.next_state[indices]
        reward = self.reward[indices]
        done = self.done[indices]

        return Batch((state, action, next_state, reward, done))

    def __len__(self):
        return self.length
