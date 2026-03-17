import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models import DQNNetwork, DuelingDQNNetwork


class BaseDQNAgent:
    def __init__(
        self,
        obs_shape,
        n_actions: int,
        device: torch.device,
        lr: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 256,
        target_update_freq: int = 1000,
    ):
        self.device = device
        self.gamma = gamma
        self.n_actions = n_actions
        self.target_update_freq = target_update_freq
        self.update_count = 0

        self.input_dim = int(np.prod(obs_shape))

        self.q_net = self.build_network(hidden_dim).to(device)
        self.target_net = self.build_network(hidden_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def build_network(self, hidden_dim: int):
        raise NotImplementedError

    def act(self, state, epsilon: float):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)

        state_tensor = torch.tensor(
            np.array(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def compute_targets(self, rewards, next_states, dones):
        raise NotImplementedError

    def update(self, replay_buffer, batch_size: int):
        if len(replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(
            batch_size, self.device
        )

        q_values = self.q_net(states).gather(1, actions)
        targets = self.compute_targets(rewards, next_states, dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())


class DQNAgent(BaseDQNAgent):
    def build_network(self, hidden_dim: int):
        return DQNNetwork(self.input_dim, self.n_actions, hidden_dim)

    def compute_targets(self, rewards, next_states, dones):
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values
        return targets


class DoubleDQNAgent(DQNAgent):
    def compute_targets(self, rewards, next_states, dones):
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values
        return targets


class DuelingDQNAgent(BaseDQNAgent):
    def build_network(self, hidden_dim: int):
        return DuelingDQNNetwork(self.input_dim, self.n_actions, hidden_dim)

    def compute_targets(self, rewards, next_states, dones):
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values
        return targets


class DuelingDoubleDQNAgent(DuelingDQNAgent):
    def compute_targets(self, rewards, next_states, dones):
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            targets = rewards + self.gamma * (1.0 - dones) * next_q_values
        return targets