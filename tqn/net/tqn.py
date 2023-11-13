import torch as T
from torch import nn
from torch.nn import functional as F


class TQN(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(in_features,
                                                   dim_feedforward=256,
                                                   nhead=in_features,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        decoder_layer = nn.TransformerDecoderLayer(in_features,
                                                   dim_feedforward=256,
                                                   nhead=in_features,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, out_features)

    def forward(self, x: T.Tensor):
        x = x.view(1, *x.shape)
        memory = self.encoder(x)
        x = self.decoder(x, memory)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = x.squeeze()

        return x
