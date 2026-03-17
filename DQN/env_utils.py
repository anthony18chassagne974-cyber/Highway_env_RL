import numpy as np

from config import ENV_CONFIG, SEED
from custom_highway_env import CustomHighwayEnv


def make_env(render_mode=None):
    env = CustomHighwayEnv(config=ENV_CONFIG, render_mode=render_mode)
    env.reset(seed=SEED)
    np.random.seed(SEED)
    return env


def print_env_info(env):
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    obs, info = env.reset()

    print("Observation shape:", obs.shape)
    print("Sample observation:")
    print(obs)
    print("duration:", env.unwrapped.config.get("duration"))
    print("policy_frequency:", env.unwrapped.config.get("policy_frequency"))