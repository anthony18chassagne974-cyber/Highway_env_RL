import numpy as np


def _lane_id(lane_index):
    try:
        return lane_index[2]
    except Exception:
        return None


def front_vehicle_distance(env):
    ego = env.unwrapped.vehicle
    ego_x = float(ego.position[0])
    ego_lane = _lane_id(ego.lane_index)
    best = None
    for veh in env.unwrapped.road.vehicles:
        if veh is ego:
            continue
        if _lane_id(veh.lane_index) != ego_lane:
            continue
        dx = float(veh.position[0]) - ego_x
        if dx > 0 and (best is None or dx < best):
            best = dx
    return best


def side_snapshot(env, side_window=10.0):
    ego = env.unwrapped.vehicle
    ego_x = float(ego.position[0])
    ego_lane = _lane_id(ego.lane_index)
    out = {}
    for veh in env.unwrapped.road.vehicles:
        if veh is ego:
            continue
        veh_lane = _lane_id(veh.lane_index)
        if veh_lane is None or ego_lane is None:
            continue
        if abs(veh_lane - ego_lane) != 1:
            continue
        veh_x = float(veh.position[0])
        if abs(veh_x - ego_x) <= side_window:
            out[id(veh)] = {"x": veh_x, "speed": float(veh.speed)}
    return out


def vehicle_positions(env):
    return {id(v): float(v.position[0]) for v in env.unwrapped.road.vehicles}


def rear_vehicle_count(env, rear_window=25.0):
    ego = env.unwrapped.vehicle
    ego_x = float(ego.position[0])
    c = 0
    for veh in env.unwrapped.road.vehicles:
        if veh is ego:
            continue
        dx = ego_x - float(veh.position[0])
        if 0 < dx < rear_window:
            c += 1
    return c


def overtake_count_from_side(env, prev_side_snapshot, pass_margin=4.0):
    if prev_side_snapshot is None:
        return 0
    ego = env.unwrapped.vehicle
    ego_x = float(ego.position[0])
    current = {id(v): v for v in env.unwrapped.road.vehicles if v is not ego}
    count = 0
    for vid, prev in prev_side_snapshot.items():
        veh = current.get(vid)
        if veh is None:
            continue
        veh_x = float(veh.position[0])
        if veh_x < ego_x - pass_margin and float(ego.speed) > float(veh.speed):
            count += 1
    return count


def being_overtaken_count(env, prev_vehicle_positions, pass_margin=4.0):
    if prev_vehicle_positions is None:
        return 0
    ego = env.unwrapped.vehicle
    ego_x = float(ego.position[0])
    count = 0
    for veh in env.unwrapped.road.vehicles:
        if veh is ego:
            continue
        vid = id(veh)
        if vid not in prev_vehicle_positions:
            continue
        prev_veh_x = prev_vehicle_positions[vid]
        veh_x = float(veh.position[0])
        was_behind = prev_veh_x < ego_x
        now_ahead = veh_x > ego_x + pass_margin
        if was_behind and now_ahead:
            count += 1
    return count


def compute_reward_and_metrics(env, cfg, prev_x=None, prev_action=None,
                               prev_side_snapshot=None, prev_vehicle_positions=None,
                               prev_speed=None, prev_lane=None):
    ego = env.unwrapped.vehicle
    v = float(ego.speed)
    x = float(ego.position[0])
    lane = _lane_id(ego.lane_index)

    speed_reward = cfg["w_speed"] * np.clip(v / cfg["v_target"], 0.0, 1.2)
    progress = 0.0 if prev_x is None else max(0.0, x - prev_x)
    progress_reward = cfg["w_progress"] * progress

    collision_penalty = cfg["w_collision"] if ego.crashed else 0.0

    front_d = front_vehicle_distance(env)
    too_close_penalty = 0.0
    blocked = 0
    if front_d is not None and front_d < cfg["d_safe"]:
        blocked = 1
        too_close_penalty = cfg["w_too_close"] * (1.0 - front_d / cfg["d_safe"])

    lane_change = 0
    lane_change_penalty = 0.0
    if prev_lane is not None and lane is not None and prev_lane != lane:
        lane_change = 1
        lane_change_penalty = cfg["w_lane_change"]

    overtakes = overtake_count_from_side(env, prev_side_snapshot, cfg["pass_margin"])
    overtake_bonus = cfg["w_overtake"] * overtakes

    overtaken_by_others = being_overtaken_count(env, prev_vehicle_positions, cfg["pass_margin"])
    overtaken_penalty = cfg["w_being_overtaken"] * overtaken_by_others

    rear_count = rear_vehicle_count(env, cfg["rear_window"])
    idle_behind_penalty = cfg["w_idle_behind"] if rear_count == 0 and blocked else 0.0

    accel = 0.0 if prev_speed is None else v - prev_speed
    abs_accel = abs(accel)

    reward = (
        speed_reward
        + progress_reward
        + overtake_bonus
        - collision_penalty
        - too_close_penalty
        - lane_change_penalty
        - overtaken_penalty
        - idle_behind_penalty
    )

    metrics = {
        "speed": v,
        "progress": progress,
        "collision": int(ego.crashed),
        "front_distance": -1.0 if front_d is None else float(front_d),
        "too_close_penalty": float(too_close_penalty),
        "lane_change": lane_change,
        "overtakes": overtakes,
        "overtaken_by_others": overtaken_by_others,
        "rear_vehicle_count": rear_count,
        "blocked": blocked,
        "acceleration": float(accel),
        "abs_acceleration": float(abs_accel),
        "reward_speed": float(speed_reward),
        "reward_progress": float(progress_reward),
        "reward_overtake": float(overtake_bonus),
        "penalty_collision": float(collision_penalty),
        "penalty_idle_behind": float(idle_behind_penalty),
    }

    next_state = {
        "prev_x": x,
        "prev_speed": v,
        "prev_lane": lane,
        "prev_action": None if prev_action is None else prev_action,
        "prev_side_snapshot": side_snapshot(env, cfg["side_window"]),
        "prev_vehicle_positions": vehicle_positions(env),
    }

    return float(reward), metrics, next_state
