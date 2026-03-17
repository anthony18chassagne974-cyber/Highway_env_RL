import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from configs import EXPERIMENTS
from envs import make_env
from common_io import save_json, append_row_csv

EPISODE_FIELDS = [
    "episode", "timesteps", "episode_return", "episode_length", "mean_speed",
    "mean_abs_acceleration", "overtakes", "overtaken_by_others", "lane_changes",
    "collision", "total_progress"
]

BATCH_FIELDS = [
    "update", "timesteps", "episodes_in_batch", "episode_return", "episode_length", "mean_speed",
    "mean_abs_acceleration", "overtakes", "overtaken_by_others", "lane_changes",
    "collision", "total_progress"
]

class PersistMetricsCallback(BaseCallback):
    def __init__(self, episode_csv_path, batch_csv_path, verbose=0):
        super().__init__(verbose)
        self.episode_csv_path = Path(episode_csv_path)
        self.batch_csv_path = Path(batch_csv_path)
        self.episode_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_count = 0
        self.update_count = 0
        self.pending_batch_rows = []

    def _extract_episode_row(self, info):
        custom = info.get("custom_episode", {})
        ep_info = info.get("episode", {})
        return {
            "episode": self.episode_count,
            "timesteps": int(self.num_timesteps),
            "episode_return": float(custom.get("episode_return", ep_info.get("r", 0.0))),
            "episode_length": int(custom.get("episode_length", ep_info.get("l", 0))),
            "mean_speed": float(custom.get("mean_speed", 0.0)),
            "mean_abs_acceleration": float(custom.get("mean_abs_acceleration", 0.0)),
            "overtakes": int(custom.get("overtakes", 0)),
            "overtaken_by_others": int(custom.get("overtaken_by_others", 0)),
            "lane_changes": int(custom.get("lane_changes", 0)),
            "collision": int(custom.get("collision", 0)),
            "total_progress": float(custom.get("total_progress", 0.0)),
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done and ("episode" in info or "custom_episode" in info):
                self.episode_count += 1
                row = self._extract_episode_row(info)
                append_row_csv(self.episode_csv_path, row)
                self.pending_batch_rows.append(row)
                print(
                    f"Episode {self.episode_count} | reward={row['episode_return']:.2f} | length={row['episode_length']} | "
                    f"ovt={row['overtakes']} | ovt_by={row['overtaken_by_others']} | coll={row['collision']}",
                    flush=True,
                )
        return True

    def _on_rollout_end(self) -> None:
        self.update_count += 1
        rows = self.pending_batch_rows
        if rows:
            agg = {
                "update": self.update_count,
                "timesteps": int(self.num_timesteps),
                "episodes_in_batch": len(rows),
            }
            for field in ["episode_return", "episode_length", "mean_speed", "mean_abs_acceleration", "overtakes", "overtaken_by_others", "lane_changes", "collision", "total_progress"]:
                agg[field] = float(np.mean([r[field] for r in rows]))
        else:
            agg = {
                "update": self.update_count,
                "timesteps": int(self.num_timesteps),
                "episodes_in_batch": 0,
                "episode_return": np.nan,
                "episode_length": np.nan,
                "mean_speed": np.nan,
                "mean_abs_acceleration": np.nan,
                "overtakes": np.nan,
                "overtaken_by_others": np.nan,
                "lane_changes": np.nan,
                "collision": np.nan,
                "total_progress": np.nan,
            }
        append_row_csv(self.batch_csv_path, agg)
        self.pending_batch_rows = []


def build_algo(algo_name, hp, env, tensorboard_dir):
    policy_kwargs = dict(net_arch=hp["net_arch"])
    if algo_name == "ppo":
        return PPO(hp["policy"], env, learning_rate=hp["learning_rate"], n_steps=hp["n_steps"], batch_size=hp["batch_size"], gamma=hp["gamma"], gae_lambda=hp["gae_lambda"], clip_range=hp["clip_range"], ent_coef=hp["ent_coef"], vf_coef=hp["vf_coef"], policy_kwargs=policy_kwargs, tensorboard_log=str(tensorboard_dir), verbose=1)
    if algo_name == "a2c":
        return A2C(hp["policy"], env, learning_rate=hp["learning_rate"], n_steps=hp["n_steps"], gamma=hp["gamma"], gae_lambda=hp["gae_lambda"], ent_coef=hp["ent_coef"], vf_coef=hp["vf_coef"], policy_kwargs=policy_kwargs, tensorboard_log=str(tensorboard_dir), verbose=1)
    raise ValueError(algo_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "a2c"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--timesteps", type=int, default=200_000)
    args = parser.parse_args()
    exp_cfg = EXPERIMENTS[args.config]
    run_dir = Path("runs") / f"{args.algo}_{args.config}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tb"

    def make_one():
        return Monitor(make_env(exp_cfg["env"], exp_cfg["reward"]))

    env = make_vec_env(make_one, n_envs=exp_cfg["algo"].get("n_envs", 8))
    model = build_algo(args.algo, exp_cfg["algo"], env, tb_dir)
    callback = PersistMetricsCallback(run_dir / "train_episode_metrics.csv", run_dir / "train_batch_metrics.csv")
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(run_dir / "final_model")
    save_json(run_dir / "run_config.json", {"algo": args.algo, "config_name": args.config, "timesteps": args.timesteps, "env": exp_cfg["env"], "reward": exp_cfg["reward"], "algo_hparams": exp_cfg["algo"]})
    print(f"Saved model to {run_dir / 'final_model.zip'}")

if __name__ == "__main__":
    main()
