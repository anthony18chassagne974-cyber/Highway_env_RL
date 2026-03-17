import argparse
from pathlib import Path
from stable_baselines3 import PPO, A2C
from configs import EXPERIMENTS
from envs import make_env
from generic_eval import evaluate_policy_generic
from common_io import save_json, save_rows_csv
from reinforce import ReinforceAgent


def load_model(model_path, algo):
    if algo == "ppo":
        return PPO.load(model_path)
    if algo == "a2c":
        return A2C.load(model_path)
    if algo == "reinforce":
        return ReinforceAgent.load(model_path)
    raise ValueError(algo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "a2c", "reinforce"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    exp_cfg = EXPERIMENTS[args.config]
    run_dir = Path(args.model).resolve().parent
    model = load_model(args.model, args.algo)
    env_factory = lambda render_mode=None: make_env(exp_cfg["env"], exp_cfg["reward"], render_mode=render_mode)
    summary, rows = evaluate_policy_generic(env_factory, model, episodes=args.episodes, deterministic=args.deterministic, video_every=args.episodes, video_dir=run_dir / "videos")
    save_json(run_dir / "evaluation_summary.json", summary)
    save_rows_csv(run_dir / "evaluation_episodes.csv", rows)
    print(summary)


if __name__ == "__main__":
    main()
