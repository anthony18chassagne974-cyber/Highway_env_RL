from pathlib import Path
import numpy as np
from video_utils import save_video


def _predict_fn(model, obs, deterministic=True):
    if hasattr(model, "predict"):
        out = model.predict(obs, deterministic=deterministic)
        if isinstance(out, tuple):
            return int(out[0])
        return int(out)
    return int(model.act(obs, deterministic=deterministic))


def evaluate_policy_generic(env_factory, model, episodes=20, deterministic=True, video_every=None, video_dir=None, max_steps=None):
    rows = []
    saved_videos = []
    for ep in range(episodes):
        env = env_factory(render_mode="rgb_array" if (video_every and (ep % video_every == 0)) else None)
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        speeds, abs_accels = [], []
        overtakes = overtaken = lane_changes = collision = 0
        frames = []
        steps = 0
        while not done:
            if video_every and (ep % video_every == 0):
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            action = _predict_fn(model, obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            speeds.append(float(info.get("speed", 0.0)))
            abs_accels.append(float(info.get("abs_acceleration", 0.0)))
            overtakes += int(info.get("overtakes", 0))
            overtaken += int(info.get("overtaken_by_others", 0))
            lane_changes += int(info.get("lane_change", 0))
            collision = max(collision, int(info.get("collision", 0)))
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        rows.append({
            "episode": ep + 1,
            "episode_return": total_reward,
            "episode_length": steps,
            "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            "mean_abs_acceleration": float(np.mean(abs_accels)) if abs_accels else 0.0,
            "overtakes": overtakes,
            "overtaken_by_others": overtaken,
            "lane_changes": lane_changes,
            "collision": collision,
        })
        if frames and video_dir is not None:
            video_path = Path(video_dir) / f"eval_episode_{ep+1}.mp4"
            save_video(frames, video_path, fps=env.unwrapped.config.get("policy_frequency", 2))
            saved_videos.append(str(video_path))
        env.close()
    summary = {
        "episodes": episodes,
        "mean_return": float(np.mean([r["episode_return"] for r in rows])) if rows else 0.0,
        "mean_episode_length": float(np.mean([r["episode_length"] for r in rows])) if rows else 0.0,
        "mean_speed": float(np.mean([r["mean_speed"] for r in rows])) if rows else 0.0,
        "mean_abs_acceleration": float(np.mean([r["mean_abs_acceleration"] for r in rows])) if rows else 0.0,
        "mean_overtakes": float(np.mean([r["overtakes"] for r in rows])) if rows else 0.0,
        "mean_overtaken_by_others": float(np.mean([r["overtaken_by_others"] for r in rows])) if rows else 0.0,
        "mean_lane_changes": float(np.mean([r["lane_changes"] for r in rows])) if rows else 0.0,
        "collision_rate": float(np.mean([r["collision"] for r in rows])) if rows else 0.0,
        "video_paths": saved_videos,
    }
    return summary, rows
