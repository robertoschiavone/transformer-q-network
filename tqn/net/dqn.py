import torch as T
from torch import nn
from torch.nn import functional as F


class DQN(nn.Module):

    def __init__(self, n_input: int, out_features: int):
        super().__init__()

        self.fc1 = nn.Linear(n_input, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, out_features)

    def forward(self, x: T.Tensor):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
