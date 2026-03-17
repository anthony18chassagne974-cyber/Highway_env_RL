import os
import numpy as np

from replay_buffer import ReplayBuffer
from agents import (
    DQNAgent,
    DoubleDQNAgent,
    DuelingDQNAgent,
    DuelingDoubleDQNAgent,
)


def linear_epsilon_schedule(
    step: int,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
):
    if step >= epsilon_decay_steps:
        return epsilon_end

    fraction = step / epsilon_decay_steps
    return epsilon_start + fraction * (epsilon_end - epsilon_start)


def train_dqn(
    env,
    device,
    agent_type: str = "dqn",
    num_episodes: int = 300,
    batch_size: int = 64,
    buffer_capacity: int = 50000,
    min_buffer_size: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    hidden_dim: int = 256,
    target_update_freq: int = 1000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 30000,
    max_steps_per_episode: int = 200,
    model_save_path: str = "results/model.pt",
):
    os.makedirs("results", exist_ok=True)

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent_type = agent_type.lower()

    if agent_type == "dqn":
        agent = DQNAgent(
            obs_shape=obs_shape,
            n_actions=n_actions,
            device=device,
            lr=lr,
            gamma=gamma,
            hidden_dim=hidden_dim,
            target_update_freq=target_update_freq,
        )
    elif agent_type == "double_dqn":
        agent = DoubleDQNAgent(
            obs_shape=obs_shape,
            n_actions=n_actions,
            device=device,
            lr=lr,
            gamma=gamma,
            hidden_dim=hidden_dim,
            target_update_freq=target_update_freq,
        )
    elif agent_type == "dueling_dqn":
        agent = DuelingDQNAgent(
            obs_shape=obs_shape,
            n_actions=n_actions,
            device=device,
            lr=lr,
            gamma=gamma,
            hidden_dim=hidden_dim,
            target_update_freq=target_update_freq,
        )
    elif agent_type == "dueling_double_dqn":
        agent = DuelingDoubleDQNAgent(
            obs_shape=obs_shape,
            n_actions=n_actions,
            device=device,
            lr=lr,
            gamma=gamma,
            hidden_dim=hidden_dim,
            target_update_freq=target_update_freq,
        )
    else:
        raise ValueError(f"Agent type inconnu : {agent_type}")

    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    episode_rewards = []
    episode_lengths = []
    losses = []
    epsilons = []

    global_step = 0
    best_reward = -np.inf

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        episode_loss_values = []

        for step_in_episode in range(max_steps_per_episode):
            epsilon = linear_epsilon_schedule(
                global_step,
                epsilon_start,
                epsilon_end,
                epsilon_decay_steps,
            )

            action = agent.act(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)

            terminal = done or truncated
            replay_buffer.push(state, action, reward, next_state, terminal)

            state = next_state
            episode_reward += reward
            global_step += 1
            epsilons.append(epsilon)

            if len(replay_buffer) >= min_buffer_size:
                loss = agent.update(replay_buffer, batch_size)
                if loss is not None:
                    losses.append(loss)
                    episode_loss_values.append(loss)

            if terminal:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_in_episode + 1)

        mean_loss = np.mean(episode_loss_values) if episode_loss_values else None
        reward_100 = np.mean(episode_rewards[-100:])

        print(
            f"[{agent_type.upper()}] Episode {episode + 1}/{num_episodes} | "
            f"Reward: {episode_reward:.2f} | "
            f"Avg100: {reward_100:.2f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Len: {step_in_episode + 1} | "
            f"Loss: {mean_loss if mean_loss is not None else 'N/A'}"
        )

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(model_save_path)

    logs = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "losses": losses,
        "epsilons": epsilons,
        "agent_type": agent_type,
    }

    return agent, logs