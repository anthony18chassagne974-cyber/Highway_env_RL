import numpy as np
import pandas as pd


def _get_front_gap(env):
    ego = env.unwrapped.vehicle
    gaps = []

    for veh in env.unwrapped.road.vehicles:
        if veh is ego:
            continue

        # Même voie
        if veh.lane_index == ego.lane_index:
            dx = veh.position[0] - ego.position[0]
            if dx > 0:
                gaps.append(dx)

    if len(gaps) == 0:
        return np.nan

    return float(min(gaps))


def evaluate_agent(
    make_env_fn,
    agent,
    agent_name: str,
    n_episodes: int = 20,
    max_steps: int = 500,
    seed_offset: int = 1000,
):
    rows = []

    for ep in range(n_episodes):
        env = make_env_fn(render_mode=None)
        state, info = env.reset(seed=seed_offset + ep)

        done = False
        truncated = False

        dt = 1.0 / env.unwrapped.config["policy_frequency"]

        episode_reward = 0.0
        speeds = []
        abs_accelerations = []
        abs_jerks = []
        front_gaps = []

        lane_changes = 0
        collision = 0
        termination_reason = "time_limit"

        prev_speed = float(env.unwrapped.vehicle.speed)
        prev_acc = 0.0
        prev_lane = env.unwrapped.vehicle.lane_index[2]

        start_x = float(env.unwrapped.vehicle.position[0])

        step_count = 0

        while not (done or truncated) and step_count < max_steps:
            action = agent.act(state, epsilon=0.0)

            next_state, reward, done, truncated, info = env.step(action)

            veh = env.unwrapped.vehicle
            speed = float(veh.speed)
            lane = veh.lane_index[2]
            x = float(veh.position[0])

            acc = (speed - prev_speed) / dt
            jerk = (acc - prev_acc) / dt

            speeds.append(speed)
            abs_accelerations.append(abs(acc))
            abs_jerks.append(abs(jerk))

            gap = _get_front_gap(env)
            if not np.isnan(gap):
                front_gaps.append(gap)

            if lane != prev_lane:
                lane_changes += 1

            if veh.crashed:
                collision = 1
                termination_reason = "collision"

            episode_reward += reward
            state = next_state

            prev_speed = speed
            prev_acc = acc
            prev_lane = lane

            step_count += 1
            print(
            f"Episode {ep} | steps={step_count} | "
            f"done={done} | truncated={truncated} | "
            f"crashed={env.unwrapped.vehicle.crashed} | collision={collision}"
            )

            
        end_x = float(env.unwrapped.vehicle.position[0])
        if collision == 0:
            if done:
                termination_reason = "done_non_collision"
            elif truncated:
                termination_reason = "time_limit"

        rows.append(
            {
                "agent": agent_name,
                "episode": ep,
                "reward": episode_reward,
                "avg_speed": float(np.mean(speeds)) if len(speeds) > 0 else 0.0,
                "max_speed": float(np.max(speeds)) if len(speeds) > 0 else 0.0,
                "episode_length": step_count,
                "collision": collision,
                "lane_changes": lane_changes,
                "mean_abs_acceleration": float(np.mean(abs_accelerations)) if len(abs_accelerations) > 0 else 0.0,
                "mean_abs_jerk": float(np.mean(abs_jerks)) if len(abs_jerks) > 0 else 0.0,
                "progress_x": end_x - start_x,
                "min_front_gap": float(np.min(front_gaps)) if len(front_gaps) > 0 else np.nan,
                "collision": collision,
                "termination_reason": termination_reason,
            }
        )

        env.close()

    return pd.DataFrame(rows)