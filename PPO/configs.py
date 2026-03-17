from copy import deepcopy

BASE_ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 20,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "absolute": False,
        "order": "sorted",
        "normalize": True,
    },
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 3,
    "vehicles_count": 18,
    "duration": 250,
    "initial_spacing": 2,
    "policy_frequency": 2,
}

BASE_REWARD_CONFIG = {
    "w_speed": 1.2,
    "w_progress": 0.08,
    "w_collision": 20.0,
    "w_too_close": 2.5,
    "w_lane_change": 0.03,
    "w_overtake": 2.0,
    "w_being_overtaken": 1.0,
    "w_idle_behind": 0.8,
    "v_min": 20.0,
    "v_target": 28.0,
    "d_safe": 18.0,
    "side_window": 10.0,
    "pass_margin": 4.0,
    "rear_window": 25.0,
    "front_window": 35.0,
}

BASE_PPO_ALGO = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.02,
    "vf_coef": 0.5,
    "net_arch": [128, 128],
    "n_envs": 8,
}

EXPERIMENTS = {
    "ppo_balanced": {
        "env": deepcopy(BASE_ENV_CONFIG),
        "reward": deepcopy(BASE_REWARD_CONFIG),
        "algo": {
            **deepcopy(BASE_PPO_ALGO),
            "ent_coef": 0.01,
        },
    },
    "ppo_overtake": {
        "env": deepcopy(BASE_ENV_CONFIG),
        "reward": {
            **deepcopy(BASE_REWARD_CONFIG),
            "w_overtake": 3.5,
            "w_being_overtaken": 1.5,
            "w_idle_behind": 1.2,
            "w_lane_change": 0.02,
        },
        "algo": deepcopy(BASE_PPO_ALGO),
    },
    "ppo_explore": {
        "env": deepcopy(BASE_ENV_CONFIG),
        "reward": {
            **deepcopy(BASE_REWARD_CONFIG),
            "w_overtake": 3.5,
            "w_being_overtaken": 1.5,
            "w_idle_behind": 1.2,
            "w_lane_change": 0.01,
        },
        "algo": {
            **deepcopy(BASE_PPO_ALGO),
            "learning_rate": 1e-4,
            "n_steps": 2048,
            "batch_size": 128,
            "gamma": 0.995,
            "gae_lambda": 0.98,
            "clip_range": 0.15,
            "ent_coef": 0.05,
            "net_arch": [256, 256],
        },
    },
    "a2c_baseline": {
        "env": deepcopy(BASE_ENV_CONFIG),
        "reward": deepcopy(BASE_REWARD_CONFIG),
        "algo": {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 128,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "net_arch": [128, 128],
            "n_envs": 8,
        },
    },
    "reinforce_baseline": {
        "env": deepcopy(BASE_ENV_CONFIG),
        "reward": deepcopy(BASE_REWARD_CONFIG),
        "algo": {
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "hidden_sizes": [128, 128],
            "episodes": 500,
            "max_steps_per_episode": 250,
            "batch_episodes": 8,
        },
    },
    "ppo_reward_very_simple": {
        "env": deepcopy(BASE_ENV_CONFIG),
        "reward": {
            **deepcopy(BASE_REWARD_CONFIG),
            "w_too_close": 0.0,
            "w_lane_change": 0.0,
            "w_overtake": 0.0,
            "w_being_overtaken": 0.0,
            "w_idle_behind": 0.0,
        },
        "algo": deepcopy(BASE_PPO_ALGO),
    },
    "ppo_reward_simple": {
        "env": deepcopy(BASE_ENV_CONFIG),
        "reward": {
            **deepcopy(BASE_REWARD_CONFIG),
            "w_too_close": 0.0,
            "w_lane_change": 0.0,
            "w_idle_behind": 0.0,
            "w_overtake": 3.5,
            "w_being_overtaken": 1.5,
        },
        "algo": deepcopy(BASE_PPO_ALGO),
    },
    "ppo_overtake_aggressive": {
        "env": deepcopy(BASE_ENV_CONFIG),
        "reward": {
            **deepcopy(BASE_REWARD_CONFIG),
            "w_overtake": 5.0,
            "w_being_overtaken": 2.5,
            "w_idle_behind": 1.8,
            "w_lane_change": 0.015,
            "w_too_close": 2.0,
        },
        "algo": deepcopy(BASE_PPO_ALGO),
    },
    "ppo_overtake_arch_128x3": {
        "env": deepcopy(BASE_ENV_CONFIG),
        "reward": {
            **deepcopy(BASE_REWARD_CONFIG),
            "w_overtake": 3.5,
            "w_being_overtaken": 1.5,
            "w_idle_behind": 1.2,
            "w_lane_change": 0.02,
        },
        "algo": {
            **deepcopy(BASE_PPO_ALGO),
            "net_arch": [128, 128, 128],
        },
    },
    "ppo_overtake_arch_256x2": {
        "env": deepcopy(BASE_ENV_CONFIG),
        "reward": {
            **deepcopy(BASE_REWARD_CONFIG),
            "w_overtake": 3.5,
            "w_being_overtaken": 1.5,
            "w_idle_behind": 1.2,
            "w_lane_change": 0.02,
        },
        "algo": {
            **deepcopy(BASE_PPO_ALGO),
            "net_arch": [256, 256],
        },
    },
    "ppo_overtake_arch_256x3": {
        "env": deepcopy(BASE_ENV_CONFIG),
        "reward": {
            **deepcopy(BASE_REWARD_CONFIG),
            "w_overtake": 3.5,
            "w_being_overtaken": 1.5,
            "w_idle_behind": 1.2,
            "w_lane_change": 0.02,
        },
        "algo": {
            **deepcopy(BASE_PPO_ALGO),
            "net_arch": [256, 256, 256],
        },
    },
}
