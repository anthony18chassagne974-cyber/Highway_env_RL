# Highway RL - Policy-Gradient Algorithms

This README provides an overview of the implementation of policy-gradient algorithms for autonomous driving in the Highway environment. It explains the purpose of each file and the project structure.

## File Descriptions

- `algo_comparison.ipynb`: Jupyter notebook for executing and comparing PPO, A2C, and REINFORCE algorithms. Trains models, saves them along with performance metrics and videos, and supports reloading results across kernel sessions.
- `reward_architecture_study.ipynb`: Jupyter notebook for investigating various reward functions and neural network architectures using PPO. Retrains model variants, saves metrics, and visualizes results.
- `common_io.py`: Utility functions for CSV and JSON file operations.
- `configs.py`: Configuration file containing all experiment setups.
- `envs.py`: Environment creation and reward wrapper application.
- `evaluate_and_record.py`: Model loading, evaluation, video recording, and result saving.
- `generic_eval.py`: Shared evaluation logic across the project.
- `reinforce.py`: PyTorch implementation of the REINFORCE algorithm.
- `reward_v2.py`: Reward function definitions and associated metrics.
- `train_sb3.py`: Training script for PPO and A2C using Stable Baselines3, including model and data saving.
- `video_utils.py`: Video saving and display utilities.

## Results Directory Structure

Results are organized in the `runs/` directory, with each experiment in its own subdirectory (e.g., `ppo_ppo_balanced/`, `reinforce_reinforce_baseline/`).

Each experiment directory includes:
- Trained model files (e.g., `final_model.zip` or `final_model.pt`)
- Training metrics in CSV format (`train_metrics.csv`, `train_batch_metrics.csv`, `train_episode_metrics.csv`)
- Evaluation data (CSV and JSON files: `evaluation_episodes.csv`, `evaluation_summary.json`)
- Videos in MP4 format within the `videos/` subdirectory
