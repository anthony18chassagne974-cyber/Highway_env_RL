from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from common_io import append_row_csv, save_json

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

@dataclass
class ReinforceConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    hidden_sizes: tuple = (128, 128)
    max_steps_per_episode: int = 250
    batch_episodes: int = 8
    device: str = "cpu"

class ReinforceAgent:
    def __init__(self, obs_shape, n_actions, config: ReinforceConfig):
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.config = config
        self.device = torch.device(config.device)
        input_dim = int(np.prod(obs_shape))
        self.policy = PolicyNetwork(input_dim, n_actions, config.hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
    def _obs_tensor(self, obs):
        return torch.tensor(np.array(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
    def act(self, obs, deterministic=False):
        logits = self.policy(self._obs_tensor(obs))
        dist = Categorical(logits=logits)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        return int(dist.sample().item())
    def predict(self, obs, deterministic=True):
        return self.act(obs, deterministic=deterministic), None
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.policy.state_dict(),
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
            "config": self.config.__dict__,
        }, path)
    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ReinforceConfig(**{**ckpt["config"], "device": device})
        agent = cls(ckpt["obs_shape"], ckpt["n_actions"], cfg)
        agent.policy.load_state_dict(ckpt["state_dict"])
        return agent

def discounted_returns(rewards, gamma):
    out = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    out.reverse()
    arr = np.array(out, dtype=np.float32)
    if len(arr) > 1 and arr.std() > 1e-8:
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return arr

def train_reinforce(env_factory, episodes, run_dir, config: ReinforceConfig, seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = env_factory(render_mode=None)
    agent = ReinforceAgent(env.observation_space.shape, env.action_space.n, config)
    train_csv = Path(run_dir) / "train_metrics.csv"
    batch_log_probs, batch_returns = [], []
    episode_rewards = []
    best_reward = -np.inf
    for episode in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        rewards, log_probs = [], []
        total_reward = 0.0
        steps = 0
        while not done and steps < config.max_steps_per_episode:
            obs_t = agent._obs_tensor(obs)
            logits = agent.policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
            done = terminated or truncated
            log_probs.append(dist.log_prob(action))
            rewards.append(float(reward))
            total_reward += float(reward)
            obs = next_obs
            steps += 1
        returns = discounted_returns(rewards, config.gamma)
        batch_log_probs.extend(log_probs)
        batch_returns.extend(returns.tolist())
        episode_rewards.append(total_reward)
        row = {
            "episode": episode,
            "episode_reward": total_reward,
            "episode_length": steps,
            "rolling_mean_20": float(np.mean(episode_rewards[-20:])),
        }
        append_row_csv(train_csv, row)
        print(f"[REINFORCE] Episode {episode}/{episodes} | reward={total_reward:.2f} | len={steps} | mean20={row['rolling_mean_20']:.2f}")
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(Path(run_dir) / "best_model.pt")
        if (episode % config.batch_episodes == 0) or (episode == episodes):
            returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=agent.device)
            loss = 0.0
            for log_prob, ret in zip(batch_log_probs, returns_t):
                loss = loss - log_prob * ret
            loss = loss / max(1, len(batch_log_probs))
            agent.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=10.0)
            agent.optimizer.step()
            batch_log_probs, batch_returns = [], []
    agent.save(Path(run_dir) / "final_model.pt")
    save_json(Path(run_dir) / "train_summary.json", {
        "episodes": episodes,
        "best_reward": float(best_reward),
        "final_mean_20": float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0,
    })
    env.close()
    return agent
