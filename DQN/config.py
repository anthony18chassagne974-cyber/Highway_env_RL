ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "sorted",
        "normalize": True
    },

    "duration": 500,
    "simulation_frequency": 15,
    "policy_frequency": 1,

    "collision_reward": -10.0,

    "relative_speed_bonus": 2.0,
    "desired_speed_margin": 2.5,
    "speed_tolerance": 2.0,

    "lane_change_penalty": 0.18,
    "left_lane_penalty": 0.04,
    "overtake_bonus": 0.25,

    "acceleration_penalty_weight": 0.10,
    "jerk_penalty_weight": 0.24,
    "action_acceleration_penalty": 0.03,
    "stable_speed_bonus": 0.10,
}

SEED = 42