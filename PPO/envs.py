import gymnasium as gym
import highway_env

from reward_v2 import compute_reward_and_metrics


class HighwayRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_config):
        super().__init__(env)
        self.reward_config = reward_config
        self.prev_x = None
        self.prev_speed = None
        self.prev_lane = None
        self.prev_action = None
        self.prev_side_snapshot = None
        self.prev_vehicle_positions = None
        self.episode_step = 0
        self.episode_return = 0.0
        self.metric_sums = {}

    def _reset_episode_stats(self):
        self.episode_step = 0
        self.episode_return = 0.0
        self.metric_sums = {
            "speed": 0.0,
            "abs_acceleration": 0.0,
            "overtakes": 0,
            "overtaken_by_others": 0,
            "lane_change": 0,
            "collision": 0,
            "progress": 0.0,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ego = self.unwrapped.vehicle
        self.prev_x = float(ego.position[0])
        self.prev_speed = float(ego.speed)
        self.prev_lane = ego.lane_index[2] if ego.lane_index is not None else None
        self.prev_action = None
        self.prev_side_snapshot = None
        self.prev_vehicle_positions = None
        self._reset_episode_stats()
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward, metrics, next_state = compute_reward_and_metrics(
            self,
            self.reward_config,
            prev_x=self.prev_x,
            prev_action=self.prev_action,
            prev_side_snapshot=self.prev_side_snapshot,
            prev_vehicle_positions=self.prev_vehicle_positions,
            prev_speed=self.prev_speed,
            prev_lane=self.prev_lane,
        )
        self.prev_x = next_state["prev_x"]
        self.prev_speed = next_state["prev_speed"]
        self.prev_lane = next_state["prev_lane"]
        self.prev_action = action
        self.prev_side_snapshot = next_state["prev_side_snapshot"]
        self.prev_vehicle_positions = next_state["prev_vehicle_positions"]

        self.episode_step += 1
        self.episode_return += float(reward)
        self.metric_sums["speed"] += float(metrics.get("speed", 0.0))
        self.metric_sums["abs_acceleration"] += float(metrics.get("abs_acceleration", 0.0))
        self.metric_sums["overtakes"] += int(metrics.get("overtakes", 0))
        self.metric_sums["overtaken_by_others"] += int(metrics.get("overtaken_by_others", 0))
        self.metric_sums["lane_change"] += int(metrics.get("lane_change", 0))
        self.metric_sums["collision"] = max(self.metric_sums["collision"], int(metrics.get("collision", 0)))
        self.metric_sums["progress"] += float(metrics.get("progress", 0.0))

        info.update(metrics)
        if terminated or truncated:
            denom = max(1, self.episode_step)
            info["custom_episode"] = {
                "episode_return": float(self.episode_return),
                "episode_length": int(self.episode_step),
                "mean_speed": float(self.metric_sums["speed"] / denom),
                "mean_abs_acceleration": float(self.metric_sums["abs_acceleration"] / denom),
                "overtakes": int(self.metric_sums["overtakes"]),
                "overtaken_by_others": int(self.metric_sums["overtaken_by_others"]),
                "lane_changes": int(self.metric_sums["lane_change"]),
                "collision": int(self.metric_sums["collision"]),
                "total_progress": float(self.metric_sums["progress"]),
            }
        return obs, reward, terminated, truncated, info


def make_env(env_config, reward_config, render_mode=None):
    env = gym.make("highway-fast-v0", config=env_config, render_mode=render_mode)
    env = HighwayRewardWrapper(env, reward_config)
    return env
