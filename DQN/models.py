import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values