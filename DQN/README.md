# Reinforcement Learning - Highway Driving

This project implements and compares several Reinforcement Learning approaches (DQN variants) on a highway driving task using the `highway-env` simulator.

## Environment

This project runs in the **same Python environment as the course labs**.  
No additional dependencies are required beyond those already used in the labs.

---

## Project Structure

### Core Files

- **train.py**  
  Main training loop for DQN-based agents.

- **agents.py**  
  Implementation of the different agents (DQN, Double DQN, Dueling, etc.).

- **models.py**  
  Neural network architectures used by the agents.

- **replay_buffer.py**  
  Experience replay buffer for storing and sampling transitions.

---

### Environment & Configuration

- **custom_highway_env.py**  
  Custom version of the highway environment with modified reward function.

- **config.py**  
  Environment and reward configuration (all hyperparameters and reward weights).

- **env_utils.py**  
  Helper functions to create and initialize environments.

---

### Evaluation & Visualization

- **evaluate.py**  
  Script to evaluate trained agents and compute benchmark metrics.

- **plots.py**  
  Utilities to generate training and evaluation plots.

- **video_utils.py**  
  Functions to record and display agent behavior as videos.

---

### Notebook

- **Notebook.ipynb**  
  Main notebook used for training, evaluation, and visualization.

---

### Results

- **results/**  
  Directory containing trained models and saved outputs.

- **random_agent.gif**  
  Example of environment behavior with a random policy.

---

### Misc

- **README.md**  
  Project documentation.

---

## Notes

- All experiments are based on a **custom reward function** designed to balance:
  - safety (collision avoidance),
  - efficiency (speed and overtaking),
  - comfort (smooth acceleration and low jerk).

- The same reward and environment configuration are used across all models for fair comparison.

---