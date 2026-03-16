"""
CMA-ES Neuroevolution for highway-fast-v0
==========================================
Evolves the weights of a small MLP policy using CMA-ES.
Parallelises fitness evaluation across CPU cores via multiprocessing.

Environment config (fixed)
--------------------------
- Kinematics observation: 15 vehicles × 5 features = 75 inputs
- duration: 250 steps, policy_frequency: 2 Hz
- Custom reward: collision, speed, lane-change, overtake, smoothness terms

Usage
-----
# Basic run (all defaults)
python cmaes_highway.py

# Named experiment with custom settings
python cmaes_highway.py --exp-name dense_traffic --generations 200 --rollouts 10 --hidden 32 --sigma0 0.4

# Resume from a checkpoint
python cmaes_highway.py --exp-name dense_traffic --resume

# Evaluate a saved policy
python cmaes_highway.py --exp-name dense_traffic --evaluate

Experiment output
-----------------
results/
  <exp_name>/
    config.json          — full config used for this run
    best_policy.npz      — weights of the best policy found
    checkpoint.pkl       — full CMA-ES state (resume from here)
    log.csv              — generation, mean_fitness, best_fitness, sigma
    final_policy.npz     — weights at end of last generation
"""

import argparse
import csv
import json
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import cma
import gymnasium as gym
import highway_env  # noqa: F401 — registers the envs
import numpy as np

# ---------------------------------------------------------------------------
# Environment configuration (matches your project setup)
# ---------------------------------------------------------------------------

ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 15,
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
    "duration": 250,
    "simulation_frequency": 15,
    "policy_frequency": 2,
}

# Observation dimensions derived from ENV_CONFIG
OBS_VEHICLES = 15
OBS_FEATURES = 5   # presence, x, y, vx, vy
OBS_DIM      = OBS_VEHICLES * OBS_FEATURES  # 75
N_ACTIONS    = 5

# Custom reward weights (from your reward config)
REWARD_WEIGHTS = {
    "collision_reward":            -10.0,
    "relative_speed_bonus":          2.0,
    "desired_speed_margin":          2.5,   # m/s tolerance around target speed
    "speed_tolerance":               2.0,
    "lane_change_penalty":           0.18,
    "left_lane_penalty":             0.04,
    "overtake_bonus":                0.25,
    "acceleration_penalty_weight":   0.10,
    "jerk_penalty_weight":           0.24,
    "action_acceleration_penalty":   0.03,
    "stable_speed_bonus":            0.10,
}

# ---------------------------------------------------------------------------
# Custom reward function
# ---------------------------------------------------------------------------

def compute_reward(obs, prev_obs, action, info, prev_action, prev_speed, rw=REWARD_WEIGHTS):
    """
    Compute the custom reward from raw environment state.

    Terms
    -----
    collision_term            : large penalty on crash
    speed_term                : bonus for driving near desired speed, relative to traffic
    lane_change_term          : penalty for changing lanes unnecessarily
    left_lane_term            : soft penalty for occupying left lanes
    overtake_term             : bonus for overtaking slower vehicles
    acceleration_term         : penalty for large physical acceleration (jerk proxy)
    jerk_term                 : penalty for change in acceleration (smoothness)
    action_acceleration_term  : penalty for choosing aggressive meta-actions repeatedly
    stable_speed_term         : bonus for maintaining consistent speed
    """
    crashed = info.get("crashed", False)

    # --- collision ---
    collision_term = rw["collision_reward"] if crashed else 0.0

    # Ego vehicle row is always index 0 in the kinematic obs
    # obs shape: (vehicles_count, n_features) — features: presence,x,y,vx,vy
    ego_vx    = float(obs[0, 3])   # normalised forward velocity
    prev_vx   = float(prev_obs[0, 3]) if prev_obs is not None else ego_vx

    # --- speed term ---
    # Reward driving near desired speed; bonus for going faster than surrounding traffic
    desired_vx = 1.0   # normalised (~30 m/s at max)
    speed_error = abs(ego_vx - desired_vx)
    speed_term = 0.0
    if speed_error < rw["speed_tolerance"] / 20.0:   # scale tolerance to normalised space
        speed_term += rw["stable_speed_bonus"]
    # Relative speed bonus: ego faster than observable vehicles
    other_vx = obs[1:, 3][obs[1:, 0] > 0]           # only present vehicles
    if len(other_vx) > 0:
        rel_speed = ego_vx - float(np.mean(other_vx))
        speed_term += rw["relative_speed_bonus"] * np.clip(rel_speed, 0, 1)

    # --- lane change term ---
    # Actions: 0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER
    lane_changed = (action in (0, 2)) and (prev_action not in (0, 2))
    lane_change_term = -rw["lane_change_penalty"] if lane_changed else 0.0

    # --- left lane term ---
    # obs[0,2] is normalised lateral position; negative = left of centre
    ego_y = float(obs[0, 2])
    left_lane_term = -rw["left_lane_penalty"] * max(0.0, -ego_y)

    # --- overtake term ---
    # Bonus when ego is faster than a nearby vehicle (present, close in x, same lane-ish)
    overtake_term = 0.0
    for i in range(1, OBS_VEHICLES):
        if obs[i, 0] < 0.5:
            continue
        dx = float(obs[i, 1])   # relative x (normalised): negative = vehicle is behind
        dy = abs(float(obs[i, 2]))
        rel_v = ego_vx - float(obs[i, 3])
        if -0.3 < dx < 0.0 and dy < 0.1 and rel_v > 0:
            overtake_term += rw["overtake_bonus"]
            break   # one overtake bonus per step

    # --- acceleration / jerk terms ---
    delta_vx = ego_vx - prev_vx
    acceleration_term      = -rw["acceleration_penalty_weight"] * abs(delta_vx)
    jerk_term              = -rw["jerk_penalty_weight"] * abs(delta_vx - (prev_vx - float(prev_obs[0, 3]) if prev_obs is not None else 0))
    action_accel_term      = -rw["action_acceleration_penalty"] * (1 if action in (3, 4) else 0)

    # --- stable speed bonus ---
    stable_speed_term = rw["stable_speed_bonus"] if abs(delta_vx) < 0.02 else 0.0

    reward = (
        collision_term
        + speed_term
        + lane_change_term
        + left_lane_term
        + overtake_term
        + acceleration_term
        + jerk_term
        + action_accel_term
        + stable_speed_term
    )
    return reward


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class MLP:
    """
    Minimal MLP: obs -> hidden (tanh) -> logits -> argmax action.
    Weights are stored as a flat numpy array for CMA-ES.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int):
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        # Layer shapes
        self.shapes = [
            (obs_dim, hidden_dim),   # W1
            (hidden_dim,),           # b1
            (hidden_dim, n_actions), # W2
            (n_actions,),            # b2
        ]
        self.n_params = sum(np.prod(s) for s in self.shapes)

    def unpack(self, weights: np.ndarray):
        """Split flat weight vector into layer tensors."""
        params = []
        idx = 0
        for shape in self.shapes:
            size = int(np.prod(shape))
            params.append(weights[idx: idx + size].reshape(shape))
            idx += size
        return params

    def forward(self, obs: np.ndarray, weights: np.ndarray) -> int:
        W1, b1, W2, b2 = self.unpack(weights)
        x = obs.flatten()
        x = np.tanh(x @ W1 + b1)
        logits = x @ W2 + b2
        return int(np.argmax(logits))


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def make_env(vehicles_density: float = 1.0):
    """Create a highway-fast-v0 environment with the project ENV_CONFIG."""
    cfg = {**ENV_CONFIG, "vehicles_density": vehicles_density}
    return gym.make("highway-fast-v0", config=cfg)


def rollout(args) -> float:
    """
    Run one episode with the given weights and return total reward.
    This function is called in a subprocess, so imports are re-evaluated.
    Uses the custom reward function instead of the environment's default.
    """
    weights, hidden_dim, vehicles_density, seed = args

    import gymnasium as gym
    import highway_env  # noqa
    import numpy as np

    cfg = {**ENV_CONFIG, "vehicles_density": vehicles_density}
    env = gym.make("highway-fast-v0", config=cfg)

    policy = MLP(OBS_DIM, hidden_dim, N_ACTIONS)

    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    done = truncated = False
    prev_obs    = None
    prev_action = 1    # IDLE
    prev_speed  = float(obs[0, 3])

    while not (done or truncated):
        action = policy.forward(obs, weights)
        next_obs, _env_reward, done, truncated, info = env.step(action)

        reward = compute_reward(obs, prev_obs, action, info, prev_action, prev_speed)
        total_reward += reward

        prev_obs    = obs
        prev_action = action
        prev_speed  = float(obs[0, 3])
        obs         = next_obs

    env.close()
    return total_reward


def evaluate_weights(
    weights: np.ndarray,
    hidden_dim: int,
    n_rollouts: int,
    vehicles_density: float,
    pool: Pool,
    base_seed: int = 0,
) -> float:
    """
    Average reward over n_rollouts episodes using a process pool.
    Each rollout uses a different seed for stable fitness estimates.
    """
    args = [
        (weights, hidden_dim, vehicles_density, base_seed + i)
        for i in range(n_rollouts)
    ]
    rewards = pool.map(rollout, args)
    return float(np.mean(rewards))


# ---------------------------------------------------------------------------
# Experiment I/O
# ---------------------------------------------------------------------------

def make_exp_dir(exp_name: str) -> Path:
    path = Path("results") / exp_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_config(exp_dir: Path, config: dict):
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def save_policy(exp_dir: Path, weights: np.ndarray, filename: str = "best_policy.npz"):
    np.savez(exp_dir / filename, weights=weights)
    print(f"  Saved {filename} -> {exp_dir / filename}")


def load_policy(exp_dir: Path, filename: str = "best_policy.npz") -> np.ndarray:
    data = np.load(exp_dir / filename)
    return data["weights"]


def save_checkpoint(exp_dir: Path, es: cma.CMAEvolutionStrategy, best_weights, best_fitness):
    with open(exp_dir / "checkpoint.pkl", "wb") as f:
        pickle.dump({"es": es, "best_weights": best_weights, "best_fitness": best_fitness}, f)


def load_checkpoint(exp_dir: Path):
    with open(exp_dir / "checkpoint.pkl", "rb") as f:
        data = pickle.load(f)
    return data["es"], data["best_weights"], data["best_fitness"]


def init_log(exp_dir: Path):
    log_path = exp_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "mean_fitness", "best_fitness", "sigma", "elapsed_s"])
    return log_path


def append_log(log_path: Path, generation: int, mean_fitness: float, best_fitness: float, sigma: float, elapsed: float):
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([generation, f"{mean_fitness:.4f}", f"{best_fitness:.4f}", f"{sigma:.6f}", f"{elapsed:.1f}"])


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict):
    exp_dir = make_exp_dir(config["exp_name"])
    save_config(exp_dir, config)

    # Network dimensions — derived from ENV_CONFIG
    obs_dim    = OBS_DIM    # 15 vehicles × 5 features = 75
    hidden_dim = config["hidden"]
    n_actions  = N_ACTIONS
    policy     = MLP(obs_dim, hidden_dim, n_actions)
    n_params   = policy.n_params
    print(f"\nPolicy: {obs_dim} -> {hidden_dim} -> {n_actions}  ({n_params} parameters)")

    # CMA-ES setup or resume
    if config["resume"] and (exp_dir / "checkpoint.pkl").exists():
        print("Resuming from checkpoint...")
        es, best_weights, best_fitness = load_checkpoint(exp_dir)
        start_gen = es.result.iterations
    else:
        x0 = np.zeros(n_params)
        es = cma.CMAEvolutionStrategy(
            x0,
            config["sigma0"],
            {
                "popsize":   config["popsize"] or (4 + int(3 * np.log(n_params))),
                "maxiter":   config["generations"],
                "tolx":      1e-6,
                "tolfun":    1e-6,
                "verbose":   -9,   # suppress CMA's own output
            },
        )
        best_weights = x0.copy()
        best_fitness = -np.inf
        start_gen = 0

    log_path = init_log(exp_dir) if not config["resume"] else exp_dir / "log.csv"
    n_workers = config["workers"] or max(1, os.cpu_count() - 1)
    print(f"Workers: {n_workers}  |  Rollouts per individual: {config['rollouts']}")
    print(f"Population size: {es.popsize}  |  Generations: {config['generations']}\n")

    t_start = time.time()

    with Pool(processes=n_workers) as pool:
        generation = start_gen
        while not es.stop() and generation < config["generations"]:
            generation += 1
            t_gen = time.time()

            # Sample population
            solutions = es.ask()

            # Evaluate in parallel — each individual averaged over K rollouts
            fitnesses = [
                evaluate_weights(
                    w,
                    hidden_dim,
                    config["rollouts"],
                    config["vehicles_density"],
                    pool,
                    base_seed=generation * 1000,
                )
                for w in solutions
            ]

            # CMA-ES minimises, so negate reward
            es.tell(solutions, [-f for f in fitnesses])

            # Track best
            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_fitness = fitnesses[gen_best_idx]
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_weights = solutions[gen_best_idx].copy()
                save_policy(exp_dir, best_weights, "best_policy.npz")

            mean_fitness = float(np.mean(fitnesses))
            elapsed = time.time() - t_start

            print(
                f"Gen {generation:4d}/{config['generations']} | "
                f"mean: {mean_fitness:6.3f}  best: {gen_best_fitness:6.3f}  "
                f"all-time best: {best_fitness:6.3f}  σ: {es.sigma:.4f}  "
                f"({time.time()-t_gen:.1f}s)"
            )
            append_log(log_path, generation, mean_fitness, gen_best_fitness, es.sigma, elapsed)

            # Checkpoint every 10 generations
            if generation % 10 == 0:
                save_checkpoint(exp_dir, es, best_weights, best_fitness)
                save_policy(exp_dir, np.array(es.result.xbest if es.result.xbest is not None else best_weights), "final_policy.npz")

    # Final save
    save_policy(exp_dir, best_weights, "best_policy.npz")
    save_checkpoint(exp_dir, es, best_weights, best_fitness)
    print(f"\nDone. Best fitness: {best_fitness:.4f}")
    print(f"Results saved to: {exp_dir}/")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(config: dict, n_episodes: int = 20):
    exp_dir = Path("results") / config["exp_name"]
    weights = load_policy(exp_dir, "best_policy.npz")

    hidden_dim = config["hidden"]
    policy     = MLP(OBS_DIM, hidden_dim, N_ACTIONS)
    env        = make_env(config["vehicles_density"])
    rewards    = []

    print(f"\nEvaluating best policy over {n_episodes} episodes...")
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        total       = 0.0
        done        = truncated = False
        prev_obs    = None
        prev_action = 1
        prev_speed  = float(obs[0, 3])

        while not (done or truncated):
            action = policy.forward(obs, weights)
            next_obs, _, done, truncated, info = env.step(action)
            reward      = compute_reward(obs, prev_obs, action, info, prev_action, prev_speed)
            total      += reward
            prev_obs    = obs
            prev_action = action
            prev_speed  = float(obs[0, 3])
            obs         = next_obs

        rewards.append(total)
        print(f"  Episode {ep+1:2d}: {total:.3f}")

    env.close()
    print(f"\nMean: {np.mean(rewards):.3f}  Std: {np.std(rewards):.3f}  "
          f"Min: {np.min(rewards):.3f}  Max: {np.max(rewards):.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CMA-ES for highway-fast-v0")

    # Experiment
    p.add_argument("--exp-name",   type=str,   default="exp_01",  help="Name of the experiment (used for output folder)")
    p.add_argument("--resume",     action="store_true",            help="Resume from checkpoint")
    p.add_argument("--evaluate",   action="store_true",            help="Evaluate saved policy instead of training")

    # Architecture
    p.add_argument("--hidden",     type=int,   default=16,         help="Hidden layer size (default: 16)")

    # CMA-ES
    p.add_argument("--generations",type=int,   default=150,        help="Max generations (default: 150)")
    p.add_argument("--sigma0",     type=float, default=0.5,        help="Initial step size (default: 0.5)")
    p.add_argument("--popsize",    type=int,   default=None,       help="Population size (default: auto = 4+3*ln(n))")

    # Fitness evaluation
    p.add_argument("--rollouts",   type=int,   default=5,          help="Rollouts per individual for stable fitness (default: 5)")

    # Environment
    p.add_argument("--vehicles-density", type=float, default=1.0,  help="Traffic density (default: 1.0)")

    # Compute
    p.add_argument("--workers",    type=int,   default=None,       help="CPU workers (default: cpu_count - 1)")

    return p.parse_args()


if __name__ == "__main__":
    # On macOS, multiprocessing requires spawn start method
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    args = parse_args()
    config = vars(args)

    if args.evaluate:
        evaluate(config)
    else:
        train(config)