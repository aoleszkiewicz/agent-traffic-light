# Traffic Light RL Optimization

Multi-agent reinforcement learning system that optimizes traffic light control at a two-road intersection. Two RL agents (one per traffic light) learn cooperative signal timing to minimize vehicle queue lengths using PPO.

Built as a semester project for the **Regression Analysis and Time Series** course.

## How It Works

### The Problem

A single intersection with two roads (A and B). Vehicles arrive randomly following a Poisson distribution. Only one road can have a green light at a time. The goal: minimize total waiting vehicles across both roads.

### The Environment

The simulation (`environment/`) models the intersection as a [PettingZoo](https://pettingzoo.farama.org/) multi-agent parallel environment with two agents (`light_A`, `light_B`):

- **Observation** (per agent): `[own_queue, other_queue, own_phase, time_in_phase]`
- **Action** (per agent): `0` = keep current phase, `1` = request phase change
- **Reward**: `-(own_queue + 0.3 * other_queue)` — penalizes long queues, with a cooperation term encouraging agents to care about the other road
- **Constraints**: mutual exclusion (only one green at a time), minimum green duration (4 steps), yellow transition period (2 steps)

Vehicle arrivals are Poisson-distributed (`lambda_a=0.6`, `lambda_b=0.4` vehicles/step by default). On green, up to 3 vehicles depart per step.

### The Agent

A shared [PPO](https://stable-baselines3.readthedocs.io/) policy (MlpPolicy) controls both traffic lights. Since the environment is symmetric (each agent sees `[own_queue, other_queue, ...]`), a single neural network learns the optimal switching strategy for both lights.

[SuperSuit](https://github.com/Farama-Foundation/SuperSuit) converts the PettingZoo environment into an SB3-compatible vectorized environment for training.

### Baseline Comparison

A fixed-cycle policy (`agent/baseline.py`) alternates green every N steps regardless of traffic conditions. This serves as the benchmark to demonstrate that the RL agent learns adaptive behavior.

### Hyperparameter Optimization

[Optuna](https://optuna.org/) searches over PPO hyperparameters (learning rate, batch size, clip range, entropy coefficient, cooperation alpha, etc.) using shorter 100k-step training runs per trial. Results are persisted in SQLite.

### Live Monitoring

Training metrics stream to [Redis](https://redis.io/) in real time. A [FastAPI](https://fastapi.tiangolo.com/) REST API exposes endpoints for monitoring state, launching training, reading metrics, and running evaluations.

## Project Structure

```
agent-traffic-signal/
├── config.py                          # Central config (env, training, Redis)
├── docker-compose.yml                 # Redis service
├── environment/
│   ├── traffic_logic.py               # Pure simulation: Poisson arrivals, queue drain, phase logic
│   ├── traffic_env.py                 # PettingZoo ParallelEnv (2 agents)
│   └── wrappers.py                    # SuperSuit → SB3 VecEnv conversion
├── agent/
│   ├── ppo_agent.py                   # SB3 PPO training loop + Redis metrics callback
│   ├── hyperopt.py                    # Optuna hyperparameter search
│   └── baseline.py                    # Fixed-cycle baseline policy
├── state/
│   └── redis_manager.py              # Redis read/write helpers
├── api/
│   └── server.py                      # FastAPI REST endpoints
├── visualization/
│   ├── renderer.py                    # Matplotlib intersection view
│   └── plots.py                       # Training curves, comparison charts
├── scripts/
│   ├── train.py                       # CLI: train the agent
│   ├── evaluate.py                    # CLI: evaluate & compare with baseline
│   └── run_api.py                     # CLI: start FastAPI server
├── tests/
│   ├── test_env.py                    # Environment logic + PettingZoo API compliance
│   ├── test_agent.py                  # Baseline policy tests
│   └── test_api.py                    # API endpoint tests
└── notebooks/
    └── analysis.ipynb                 # Final analysis & plots for report
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Start Redis (optional, for live metrics)
docker compose up -d
```

## Usage

### Train the agent

```bash
# Default: 300k timesteps
uv run python scripts/train.py

# Custom timesteps
uv run python scripts/train.py --timesteps 50000

# Without Redis (if Redis isn't running)
uv run python scripts/train.py --timesteps 50000 --no-redis
```

The trained model is saved to `models/ppo_traffic_final.zip`, with the best checkpoint in `models/best/`.

### Evaluate and compare with baseline

```bash
# Headless evaluation with baseline comparison
uv run python scripts/evaluate.py --compare-baseline

# Visual evaluation (renders the intersection)
uv run python scripts/evaluate.py --render --episodes 1
```

### Run hyperparameter optimization

```bash
uv run python scripts/train.py --hyperopt --hyperopt-trials 50
```

Results are saved to `optuna_study.db` and can be viewed via the API.

### Start the REST API

```bash
uv run python scripts/run_api.py
```

Open http://localhost:8000/docs for the interactive API documentation.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/state` | Current intersection state |
| `GET` | `/metrics` | Training metrics history |
| `POST` | `/train/start` | Launch training |
| `GET` | `/train/status` | Training status (idle/running/finished) |
| `GET` | `/config` | Current configuration |
| `PUT` | `/config` | Update configuration |
| `POST` | `/evaluate` | Run evaluation (optionally compare with baseline) |
| `GET` | `/hyperopt/results` | Optuna optimization results |

### Run tests

```bash
uv run python -m pytest tests/ -v
```

## Configuration

All defaults are in `config.py` and can be adjusted via the API or by editing the file directly.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_a` | 0.6 | Poisson arrival rate, road A (vehicles/step) |
| `lambda_b` | 0.4 | Poisson arrival rate, road B (vehicles/step) |
| `green_drain_rate` | 3 | Vehicles departing per green step |
| `min_green_steps` | 4 | Minimum green phase duration |
| `yellow_steps` | 2 | Yellow transition duration |
| `max_steps` | 500 | Steps per episode |
| `alpha` | 0.3 | Cooperation coefficient in reward |
| `max_queue` | 50 | Queue capacity per road |
| `total_timesteps` | 300,000 | Training duration |
| `learning_rate` | 3e-4 | PPO learning rate |

## Tech Stack

- **RL Framework**: Stable-Baselines3 (PPO) + PettingZoo (multi-agent) + SuperSuit (wrappers)
- **Simulation**: NumPy (Poisson arrivals, queue dynamics)
- **API**: FastAPI + Uvicorn
- **State**: Redis
- **Hyperparameters**: Optuna
- **Visualization**: Matplotlib
- **Package Management**: uv
