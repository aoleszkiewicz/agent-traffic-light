# Deep Dive: Understanding the Traffic Light RL System

A technical study guide for all team members. Read this to understand every component, why it's designed this way, and how it all fits together.

---

## Table of Contents

1. [Core Concepts](#1-core-concepts)
2. [The Simulation Layer](#2-the-simulation-layer)
3. [The PettingZoo Environment](#3-the-pettingzoo-environment)
4. [SB3 Integration & Shared Policy](#4-sb3-integration--shared-policy)
5. [PPO Training](#5-ppo-training)
6. [Reward Engineering](#6-reward-engineering)
7. [The Baseline & Fair Comparison](#7-the-baseline--fair-comparison)
8. [Hyperparameter Optimization](#8-hyperparameter-optimization)
9. [Redis & API Layer](#9-redis--api-layer)
10. [Bugs We Hit & How We Fixed Them](#10-bugs-we-hit--how-we-fixed-them)
11. [How to Read the Code](#11-how-to-read-the-code)
12. [Key Equations](#12-key-equations)
13. [Glossary](#13-glossary)

---

## 1. Core Concepts

### What is Reinforcement Learning?

An agent interacts with an environment in a loop:

```
state → agent picks action → environment returns (next_state, reward) → repeat
```

The agent's goal: maximize cumulative reward over time. It learns a **policy** — a function mapping observations to actions.

### What is PPO?

Proximal Policy Optimization is an RL algorithm that:
1. Collects a batch of experience (2048 steps in our case)
2. Computes how much better/worse each action was vs. expected (advantage)
3. Updates the neural network, but clips the update to prevent too-large changes
4. Repeats

The "proximal" part means: don't change the policy too much in one step. This makes training stable.

### What is Multi-Agent RL?

Multiple agents act simultaneously in the same environment. In our case:
- `light_A` controls road A's traffic light
- `light_B` controls road B's traffic light
- They share the same neural network (parameter sharing)
- Each sees the world from its own perspective: `[own_queue, other_queue, ...]`

### What is PettingZoo?

A library for multi-agent environments (like Gymnasium, but for multiple agents). We use the `ParallelEnv` API — both agents act at the same time each step.

---

## 2. The Simulation Layer

**File**: `environment/traffic_logic.py`

This file contains pure functions with no framework dependencies. Easy to test, easy to understand.

### Vehicle Arrivals

```python
def sample_arrivals(lambda_a, lambda_b, rng):
    return int(rng.poisson(lambda_a)), int(rng.poisson(lambda_b))
```

Vehicles arrive following a **Poisson distribution**:
- P(k arrivals) = (λ^k × e^(-λ)) / k!
- λ_A = 0.6 means on average 0.6 vehicles arrive per step on road A
- λ_B = 0.4 means road B has lighter traffic

**Why Poisson?** It models independent random events occurring at a constant rate — exactly how traffic works at a macro level. This connects directly to the course material on stochastic processes.

### Vehicle Departures

```python
def process_departures(queue, is_green, rate):
    if not is_green:
        return queue
    return max(0, queue - rate)
```

- If green: up to `rate=3` vehicles leave per step
- If red or yellow: no vehicles leave
- Queue can't go below 0

**Why rate=3?** With λ_A=0.6 arrivals/step, a drain rate of 3 means the queue can clear quickly when green. This ensures an achievable equilibrium — if the drain rate were only 1, queues would grow faster than they drain.

### Phase Change Logic

```python
def resolve_phase_change(action_a, action_b, current_green, time_in_phase, min_green):
```

This is the most important function. The rules:

1. **Minimum green**: If `time_in_phase < 4`, no switching allowed. Prevents flickering.
2. **Cancel-out**: If both agents request change → no switch. Prevents oscillation.
3. **Either agent can trigger**: If exactly one agent requests change → switch to other road.
4. **Default**: If neither requests change → stay on current road.

**Why can both agents trigger a switch?** Initially only the green agent could switch. This created a "greedy hold" problem — once green, the agent never voluntarily gave up green because its own queue was low. Allowing the red agent to demand green gives the starving road agency.

**Why cancel-out?** If both request simultaneously, it's ambiguous — the green agent wants to give up, but the red agent also wants to give up its request... or does it want green? The cancel-out rule is a simple coordination mechanism that prevents rapid oscillation.

---

## 3. The PettingZoo Environment

**File**: `environment/traffic_env.py`

### The State

```python
_queue_a: int          # vehicles waiting on road A
_queue_b: int          # vehicles waiting on road B
_current_green: str    # "A" or "B"
_time_in_phase: int    # steps since last phase change
_yellow_remaining: int # countdown for yellow transition
_step_count: int       # total steps in episode
```

### Observation Design

Each agent sees a 4-element vector, normalized to [0, 1]:

```python
[own_queue / 50,              # how congested is my road? (0-1)
 other_queue / 50,            # how congested is the other road? (0-1)
 own_phase / 2.0,             # am I red (0), green (0.5), or yellow (1)?
 time_in_phase / 500]         # how long has the current phase lasted? (0-1)
```

**Why normalize?** Neural networks learn best when inputs are in similar ranges. Raw queues (0-50) and raw time (0-500) would dominate the smaller phase signal (0-2).

**Why symmetric?** Both agents see `[own, other, ...]` rather than `[queue_A, queue_B, ...]`. This means light_A and light_B see the same structure, just from their own perspective. This enables a **shared policy** — one neural network for both agents.

### Step Execution Order

Each `step()` call:

```
1. Increment step counter
2. Sample Poisson arrivals → add to queues (capped at 50)
3. If yellow phase active:
   → Decrement yellow counter, skip phase change logic
4. Else:
   → Check agent actions via resolve_phase_change()
   → If switching: start yellow countdown, update current_green
   → If not switching: increment time_in_phase
   → Process departures (only if not yellow)
5. Compute rewards (normalized)
6. Check truncation (step >= 500)
7. Return (obs, rewards, terminations, truncations, infos)
```

**Why no departures during yellow?** In reality, traffic stops during the amber phase. This also creates a cost for switching — you lose 2 steps of throughput.

### Episode Lifecycle

- Starts with road A green, all queues at 0
- Runs for 500 steps (configurable)
- Ends via truncation (not termination) — the environment doesn't have a "win" condition
- When truncated: `self.agents = []` signals PettingZoo that the episode is over

---

## 4. SB3 Integration & Shared Policy

**File**: `environment/wrappers.py`

### The Problem

SB3 (Stable-Baselines3) expects a standard Gymnasium `VecEnv`. PettingZoo provides a multi-agent `ParallelEnv`. They don't speak the same language.

### The Solution: SuperSuit

```python
env = TrafficLightEnv()
env.reset(seed=seed)

# Step 1: Flatten 2-agent env into a VecEnv with 2 "sub-environments"
env = ss.pettingzoo_env_to_vec_env_v1(env)

# Step 2: Stack into SB3-compatible format
env = ss.concat_vec_envs_v1(env, n_envs, num_cpus=1, base_class="stable_baselines3")
```

After this conversion:
- SB3 sees a VecEnv with `num_envs = 2` (one per agent)
- Each "environment" gets its own observation and produces its own action
- But they all share the same underlying TrafficLightEnv instance
- During training, SB3 feeds experience from both agents into the same neural network

**Key insight**: This is **parameter sharing** — the cheapest form of multi-agent RL. Both agents learn from each other's experience because they share the same policy network.

### The Seed Bug

SB3 calls `env.seed()` during setup, but SuperSuit's `ConcatVecEnv` doesn't implement it. We monkey-patch:

```python
inner = env.venv if hasattr(env, "venv") else env
if not hasattr(inner, "seed"):
    inner.seed = lambda seed=None: [seed] * env.num_envs
```

We patch `env.venv` (the inner ConcatVecEnv), not `env` itself (the SB3 wrapper), because SB3's wrapper delegates `seed()` to `self.venv`.

---

## 5. PPO Training

**File**: `agent/ppo_agent.py`

### The PPO Hyperparameters Explained

```python
PPO(
    "MlpPolicy",          # 2-layer MLP (64×64 by default)
    env,
    learning_rate=3e-4,    # How fast to update weights. Too high = unstable, too low = slow
    n_steps=2048,          # Collect this many steps before updating. Bigger = more stable, slower
    batch_size=64,         # Split the 2048 steps into minibatches of 64 for gradient descent
    n_epochs=10,           # Run 10 gradient descent passes over each batch of 2048 steps
    gamma=0.99,            # Discount factor. 0.99 = care about rewards ~100 steps into the future
    gae_lambda=0.95,       # GAE smoothing. Balances bias vs variance in advantage estimation
    clip_range=0.2,        # PPO clipping. Prevents policy from changing more than 20% per update
    ent_coef=0.01,         # Entropy bonus. Encourages exploration (trying random actions)
    vf_coef=0.5,           # Value function loss weight. How much to prioritize value prediction
)
```

### What Happens During `model.learn()`

```
Repeat until total_timesteps reached:
    1. Collect 2048 steps using current policy (rollout)
       - Both agents contribute steps (so ~1024 per agent per rollout)
    2. Compute advantages using GAE (Generalized Advantage Estimation)
    3. For 10 epochs:
       - Shuffle data into minibatches of 64
       - For each minibatch:
         - Compute policy loss (clipped surrogate objective)
         - Compute value loss (MSE between predicted and actual returns)
         - Compute entropy bonus
         - Total loss = policy_loss + 0.5 × value_loss - 0.01 × entropy
         - Backpropagate and update weights
    4. Every 10,000 steps: evaluate on separate eval environment
```

### Callbacks

**EvalCallback**: Every 10k steps, runs 10 episodes on a separate environment. Saves the best model checkpoint. This prevents overfitting — the "best" model might not be the final one.

**RedisMetricsCallback**: Every 1000 steps, pushes average reward and episode length to Redis. Enables live monitoring via the API.

---

## 6. Reward Engineering

This was the hardest part of the project. Three iterations:

### Attempt 1: Raw rewards (FAILED)

```python
reward = -(queue_a + 0.3 * queue_b)  # alpha=0.3
```

Problem: Rewards ranged from 0 to -65 per step. Over 500 steps, cumulative returns reached -14,000. The value function (which predicts cumulative future reward) couldn't learn to predict values that large. `explained_variance` stuck near 0.

### Attempt 2: Normalized rewards + higher alpha (SUCCESS)

```python
scale = max_queue * (1.0 + alpha)  # = 50 * 1.5 = 75
reward = -(queue_a + 0.5 * queue_b) / scale  # normalized to [-1, 0]
```

Changes:
- **Normalization**: Dividing by `scale` keeps rewards in [-1, 0]. The value function can now predict returns in a reasonable range (roughly [-500, 0] over an episode).
- **Alpha 0.3 → 0.5**: With only 30% weight on the other road, agents didn't care enough to switch. At 50%, the penalty for letting the other road's queue grow is strong enough to motivate cooperation.

### Why Reward Normalization Matters

PPO's value function predicts V(s) = expected cumulative reward from state s. If rewards are huge (-14,000), the value function needs to predict huge numbers accurately — a much harder regression problem than predicting values in [-500, 0].

The `explained_variance` metric tells you how well the value function predicts:
- ~0: predictions are useless (before normalization)
- ~0.1: starting to learn (after normalization)
- ~1.0: near-perfect predictions

---

## 7. The Baseline & Fair Comparison

**File**: `agent/baseline.py`

### Fixed-Cycle Policy

```python
def predict(self, obs):
    own_phase = obs[2]                        # 0=red, 0.5=green, 1=yellow
    is_green = abs(own_phase - 0.5) < 0.01   # am I green?
    self._step += 1
    if is_green and self._step % 15 == 0:     # every 15 steps, if green
        return 1                               # request change
    return 0                                   # otherwise keep
```

**Critical design choice**: Only the green agent requests changes. If both agents used the same step counter and both requested at the same time, the cancel-out rule would prevent any switching.

### Why This is a Fair Baseline

- Fixed-cycle is the simplest reasonable traffic control strategy
- Cycle length of 15 gives each road a fair share of green time
- It doesn't adapt to traffic conditions — proves that adaptation helps
- Both PPO and baseline operate under the same environment rules

### Results

| Metric | PPO Agent | Baseline |
|--------|-----------|----------|
| Mean Reward | -24.69 | -66.43 |
| Std Deviation | ±2.48 | ±6.52 |

PPO is 63% better and much more consistent (lower variance).

---

## 8. Hyperparameter Optimization

**File**: `agent/hyperopt.py`

### How Optuna Works

```
for each trial:
    1. Optuna suggests hyperparameter values (smart sampling)
    2. Train PPO for 100k steps with those values
    3. Evaluate: compute mean reward
    4. Report back to Optuna
    5. Optuna updates its model of what works
```

Optuna uses **Tree-structured Parzen Estimators (TPE)** — it builds a probabilistic model of which hyperparameters lead to good results and samples more from promising regions.

### What We Search Over

```python
learning_rate:  1e-5 to 1e-3   (log scale — because learning rate spans orders of magnitude)
n_steps:        [256, 512, 1024, 2048]  (categorical — must be powers of 2 for efficiency)
batch_size:     [32, 64, 128, 256]
n_epochs:       3 to 15
gamma:          0.95 to 0.999
gae_lambda:     0.8 to 0.99
clip_range:     0.1 to 0.4
ent_coef:       1e-4 to 0.1   (log scale)
alpha:          0.0 to 1.0    (the cooperation coefficient!)
```

**Why include alpha?** The cooperation coefficient directly affects agent behavior. Optuna can discover the optimal balance between selfishness and cooperation.

### Why 100k Steps Per Trial?

Full training takes 300k steps (~65 seconds). With 50 trials at 300k each, that's ~54 minutes. At 100k per trial, it's ~18 minutes. The ranking of hyperparameters is usually stable even with shorter runs.

---

## 9. Redis & API Layer

### Redis State Manager (`state/redis_manager.py`)

Redis stores four types of data:

| Key | Type | Content |
|-----|------|---------|
| `traffic:state` | Hash | Current env state (queues, phase, step) |
| `traffic:metrics` | List | Time-series of training metrics |
| `traffic:config` | String | JSON-encoded current config |
| `traffic:status` | String | "idle", "running", "finished" |

**Why Redis?** It's fast, simple, and decouples the training process from the monitoring API. Training writes metrics; the API reads them. They don't need to share memory.

### FastAPI Server (`api/server.py`)

Key design decisions:
- **Background thread for training**: `POST /train/start` spawns a daemon thread. The API remains responsive during training.
- **Lazy imports**: Expensive modules (torch, SB3) are imported inside endpoint functions, not at module level. This makes server startup instant.
- **Graceful degradation**: If Redis is down, most endpoints return sensible defaults instead of crashing.

### API Endpoints Summary

```
GET  /state              → current queues, phase
GET  /metrics            → training reward history
POST /train/start        → launch training
GET  /train/status       → idle/running/finished
GET  /config             → current settings
PUT  /config             → update settings
POST /evaluate           → run PPO + baseline evaluation
GET  /hyperopt/results   → Optuna best params
```

---

## 10. Bugs We Hit & How We Fixed Them

### Bug 1: SuperSuit seed() missing

**Symptom**: `AttributeError: 'ConcatVecEnv' object has no attribute 'seed'`

**Root cause**: SB3 calls `env.seed()` during initialization. SuperSuit's `ConcatVecEnv` doesn't implement it. SB3's wrapper delegates `seed()` to `self.venv`, so patching the outer wrapper isn't enough.

**Fix**: Monkey-patch the inner `env.venv`:
```python
inner = env.venv if hasattr(env, "venv") else env
inner.seed = lambda seed=None: [seed] * env.num_envs
```

### Bug 2: Agent permanently holds green (greedy hold)

**Symptom**: PPO reward didn't improve. Agent kept road A green forever; road B queue hit 50.

**Root cause**: Only the currently-green agent could trigger a phase change. Once green, the agent had no incentive to switch (its own queue was low). Alpha=0.3 wasn't enough cooperation penalty.

**Fix**: Allow either agent to request a change. Increased alpha from 0.3 to 0.5.

### Bug 3: Value function can't learn (reward scale too large)

**Symptom**: `explained_variance ≈ 0` throughout training. Reward stuck at -14,000/episode.

**Root cause**: Raw rewards of -30 per step created cumulative returns of -14,000. The value function needed to regress huge values — a much harder problem.

**Fix**: Normalize rewards to [-1, 0]:
```python
scale = max_queue * (1.0 + alpha)
reward = -(own_queue + alpha * other_queue) / scale
```

### Bug 4: Same action applied to both agents in evaluation

**Symptom**: During manual evaluation, PPO appeared to perform same as a stuck agent. Queue dynamics showed no switching.

**Root cause**: Evaluation code called `model.predict(obs_a)` once and used the same action for both agents. When both output 1, the cancel-out rule prevented switching.

**Fix**: Predict independently for each agent:
```python
action_a, _ = model.predict(obs["light_A"], deterministic=True)
action_b, _ = model.predict(obs["light_B"], deterministic=True)
```

### Bug 5: Baseline broken by cancel-out

**Symptom**: Baseline never switched phases. Both agents used identical cycle counters and always requested change simultaneously.

**Root cause**: Two `FixedCyclePolicy` with same cycle length fire on the same step → both request change → cancel-out.

**Fix**: Only the green agent requests changes. The red agent always returns 0:
```python
if is_green and self._step % self.cycle_length == 0:
    return 1
return 0
```

---

## 11. How to Read the Code

### Recommended reading order

1. **`config.py`** — understand all the parameters first
2. **`environment/traffic_logic.py`** — pure simulation, no framework code
3. **`environment/traffic_env.py`** — how the PettingZoo env wraps the logic
4. **`environment/wrappers.py`** — short file, understand the SuperSuit conversion
5. **`agent/baseline.py`** — simple policy, good contrast with RL
6. **`agent/ppo_agent.py`** — the training loop
7. **`agent/hyperopt.py`** — Optuna integration
8. **`state/redis_manager.py`** — Redis operations
9. **`api/server.py`** — FastAPI endpoints

### Data flow during training

```
scripts/train.py
  → config.py (load defaults or CLI overrides)
  → agent/ppo_agent.py: train_ppo()
    → environment/wrappers.py: make_sb3_env()
      → environment/traffic_env.py: TrafficLightEnv()
      → SuperSuit conversion
    → SB3 PPO.learn()
      → Each step: traffic_env.step()
        → traffic_logic.sample_arrivals()
        → traffic_logic.resolve_phase_change()
        → traffic_logic.process_departures()
      → Every 1000 steps: RedisMetricsCallback → redis_manager
      → Every 10k steps: EvalCallback → evaluate on separate env
    → Save model to models/
```

### Data flow during evaluation

```
scripts/evaluate.py
  → agent/ppo_agent.py: load_model()
  → environment/wrappers.py: make_sb3_env() (eval env)
  → SB3 evaluate_policy()
  → agent/baseline.py: evaluate_baseline() (comparison)
    → environment/traffic_env.py: TrafficLightEnv() (direct, no SB3)
    → FixedCyclePolicy.predict() per agent per step
```

---

## 12. Key Equations

### Poisson Distribution (vehicle arrivals)

```
P(X = k) = (λ^k × e^(-λ)) / k!

E[X] = λ           (expected arrivals per step)
Var(X) = λ          (variance equals mean)

Road A: λ_A = 0.6 vehicles/step
Road B: λ_B = 0.4 vehicles/step
```

### Reward Function

```
scale = max_queue × (1 + α) = 50 × 1.5 = 75

reward_A = -(queue_A + α × queue_B) / scale
reward_B = -(queue_B + α × queue_A) / scale

where α = 0.5 (cooperation coefficient)
```

Range: [-1, 0] where 0 is optimal (empty queues) and -1 is worst (both queues at capacity).

### PPO Clipped Objective

```
L(θ) = E[ min(r(θ) × A, clip(r(θ), 1-ε, 1+ε) × A) ]

where:
  r(θ) = π_new(a|s) / π_old(a|s)    (probability ratio)
  A = advantage estimate              (how much better than expected)
  ε = 0.2                             (clip range)
```

This prevents the policy from changing too much in one update. If `r(θ)` would go above 1.2 or below 0.8, the gradient is clipped.

### Generalized Advantage Estimation (GAE)

```
A_t = Σ_{l=0}^{∞} (γλ)^l × δ_{t+l}

where δ_t = r_t + γ × V(s_{t+1}) - V(s_t)    (TD error)

γ = 0.99    (discount factor)
λ = 0.95    (GAE smoothing)
```

GAE smooths advantage estimates over multiple timesteps. λ=0 would give high-bias/low-variance estimates; λ=1 gives low-bias/high-variance. 0.95 is a good balance.

### Queue Dynamics

```
queue_A(t+1) = min(queue_A(t) + arrivals_A(t) - departures_A(t), max_queue)

departures_A(t) = min(queue_A(t), drain_rate)  if green_A
                  0                               otherwise
```

The system reaches equilibrium when expected departures match expected arrivals. With drain_rate=3 and λ_A=0.6, road A can clear its queue quickly when green.

---

## 13. Glossary

| Term | Definition |
|------|-----------|
| **PPO** | Proximal Policy Optimization — RL algorithm that clips policy updates for stability |
| **Policy** | A function π(a\|s) mapping states to action probabilities |
| **Value function** | V(s) = expected cumulative future reward from state s |
| **Advantage** | A(s,a) = Q(s,a) - V(s) — how much better action a is vs average |
| **GAE** | Generalized Advantage Estimation — smoothed advantage computation |
| **Rollout** | A batch of collected experience (2048 steps in our case) |
| **Episode** | One complete simulation run (500 steps) |
| **Truncation** | Episode ended by time limit (not by reaching a terminal state) |
| **VecEnv** | Vectorized environment — multiple envs running in parallel for SB3 |
| **Parameter sharing** | Multiple agents using the same neural network |
| **PettingZoo** | Multi-agent RL environment library (parallel + turn-based) |
| **SuperSuit** | Wrapper library that converts PettingZoo envs to SB3-compatible VecEnvs |
| **Optuna** | Hyperparameter optimization library using Bayesian optimization (TPE) |
| **Poisson distribution** | Models count of independent events in a fixed interval |
| **Entropy** | Measure of randomness in the policy. Higher = more exploration |
| **Clip range** | PPO parameter (ε=0.2) limiting how much the policy can change per update |
| **Alpha (α)** | Cooperation coefficient — weight on other agent's queue in reward |
| **Yellow phase** | 2-step transition when switching lights — no departures during this time |
| **Min green** | 4-step minimum — prevents rapid switching (flickering) |
| **Cancel-out** | When both agents request change simultaneously, no switch occurs |
| **Explained variance** | How well the value function predicts returns (0=useless, 1=perfect) |
| **MlpPolicy** | SB3 policy using a Multi-Layer Perceptron (2 hidden layers of 64 units) |
