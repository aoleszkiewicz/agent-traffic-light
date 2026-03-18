# Presentation Plan: Traffic Light RL Optimization

Target audience: professor + students of "Regression Analysis and Time Series" course.
Recommended duration: 15–20 minutes + 5 min Q&A.

---

## Slide 1 — Title

**Intelligent Traffic Light Control with Multi-Agent Reinforcement Learning**

- Team members (3 names)
- Course: Regression Analysis and Time Series
- Date

---

## Slide 2 — The Problem

"How should traffic lights decide when to switch?"

- Single intersection, two roads (A and B)
- Vehicles arrive randomly — we don't know when the next car comes
- Only one road can be green at a time
- Goal: minimize total waiting time across both roads

**Visual**: Use `figures/intersection_snapshots.png` (step 1 — the clean starting state)

**Talking point**: Fixed-cycle lights waste time — green for an empty road while cars pile up elsewhere. Can we do better?

---

## Slide 3 — Our Approach

**Reinforcement Learning**: agents learn by trial and error

```
Agent observes state → chooses action → receives reward → learns
```

- Two agents (one per traffic light) share the same brain (neural network)
- They learn cooperatively — each agent cares about the other road's queue too
- Framework: PPO (Proximal Policy Optimization) — state-of-the-art RL algorithm

**Visual**: Simple diagram:
```
[Observation] → [Neural Network] → [Action: keep / switch]
                                          ↓
                              [Environment steps forward]
                                          ↓
                              [Reward: -queue penalty]
```

---

## Slide 4 — The Simulation Environment

Explain what the agents see and do:

| Component | Details |
|-----------|---------|
| **Observation** | Own queue, other queue, light phase, time in phase (all normalized to [0,1]) |
| **Action** | 0 = keep current phase, 1 = request change |
| **Reward** | `-(own_queue + 0.5 × other_queue) / scale` |
| **Constraints** | Mutual exclusion, minimum green (4 steps), yellow transition (2 steps) |
| **Traffic model** | Poisson arrivals: λ_A=0.6, λ_B=0.4 vehicles/step |
| **Episode** | 500 steps |

**Talking point**: The Poisson distribution connects to the time-series content of the course — vehicle arrivals are a stochastic process.

---

## Slide 5 — The Cooperation Mechanism

Why `alpha = 0.5` matters:

```
reward_A = -(queue_A + 0.5 × queue_B) / scale
```

- If alpha = 0: selfish — agent only cares about its own road → never switches
- If alpha = 1: fully altruistic — values other road equally
- alpha = 0.5: balanced cooperation — agent switches when the other road needs it

**Talking point**: This is the key insight. Without the cooperation term, agents learn to hoard green and starve the other road.

---

## Slide 6 — Phase Change Logic

How switching actually works:

1. Either agent can request a change (action = 1)
2. If **both** request simultaneously → requests cancel out (no change)
3. Min green enforced — can't switch too fast
4. Yellow transition — 2-step pause between switches

**Talking point**: The cancel-out rule is important — it prevents oscillation and forces agents to coordinate implicitly.

---

## Slide 7 — Technical Architecture

```
┌─────────────────────────────────────────────┐
│  PettingZoo Environment (2 agents)          │
│  ┌───────────────┐  ┌───────────────┐       │
│  │   light_A     │  │   light_B     │       │
│  └───────┬───────┘  └───────┬───────┘       │
│          │    SuperSuit         │            │
│          └──────┬───────────────┘            │
│                 ▼                            │
│        SB3 VecEnv (shared policy)           │
└─────────────────┬───────────────────────────┘
                  ▼
        PPO Training (MlpPolicy)
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
  Redis       Models       FastAPI
 (metrics)  (checkpoints)   (API)
```

Technologies: PettingZoo, Stable-Baselines3, SuperSuit, FastAPI, Redis, Optuna

---

## Slide 8 — Training Process

- PPO algorithm: learns by collecting experience, then updating the policy to increase reward
- 300,000 timesteps (~65 seconds on CPU)
- Shared policy: both agents use the same neural network (symmetric observations enable this)
- Evaluation every 10k steps — best model checkpointed automatically

**Show**: Training log snippet showing reward improving from -221 → -25 over training

---

## Slide 9 — Results: Queue Dynamics

**Visual**: `figures/queue_dynamics_comparison.png`

**Left (PPO)**: Both queues stay bounded (0–10 vehicles). Active, adaptive switching.

**Right (Baseline)**: Fixed-cycle can't adapt — queues spike to 15-18 regularly. Wastes green time on low-traffic road.

**Bottom row**: Phase timeline shows PPO switches dynamically based on demand. Baseline follows rigid schedule.

---

## Slide 10 — Results: Reward Comparison

**Visual**: `figures/reward_comparison.png`

| Metric | PPO Agent | Fixed-Cycle Baseline |
|--------|-----------|---------------------|
| Mean Reward | -24.69 | -66.43 |
| Std Deviation | ±2.48 | ±6.52 |

**PPO is 63% better** and more consistent (lower variance).

---

## Slide 11 — Results: Cumulative Reward

**Visual**: `figures/cumulative_reward.png`

- PPO curve stays above baseline throughout the entire episode
- Gap widens over time — PPO prevents queue buildup, baseline lets queues grow
- Final: PPO ≈ -50, Baseline ≈ -70

---

## Slide 12 — The Baseline Problem & What We Learned

Initially the baseline was broken — both agents requested changes at the same step, triggering the cancel-out rule. The fix: only the green agent requests phase changes.

Key lessons:
1. **Reward normalization** was critical — raw rewards (-14,000/episode) prevented the value function from learning
2. **Cooperation coefficient** (alpha) must be high enough to incentivize switching
3. **Both agents need agency** — originally only the green agent could switch, which created a "greedy hold" problem

---

## Slide 13 — Hyperparameter Optimization

Optuna searches over 9 parameters (100k steps per trial):

| Parameter | Search Range |
|-----------|-------------|
| learning_rate | 1e-5 to 1e-3 (log) |
| n_steps | {256, 512, 1024, 2048} |
| batch_size | {32, 64, 128, 256} |
| n_epochs | 3 to 15 |
| gamma | 0.95 to 0.999 |
| clip_range | 0.1 to 0.4 |
| ent_coef | 1e-4 to 0.1 (log) |
| alpha | 0.0 to 1.0 |

**Talking point**: Alpha is included as a tunable hyperparameter — interesting research question: what is the optimal cooperation level?

---

## Slide 14 — Live Demo (Optional)

If time allows, show:

1. `uv run python scripts/train.py --timesteps 50000 --no-redis` — live training
2. `uv run python scripts/evaluate.py --compare-baseline` — comparison
3. `uv run python scripts/run_api.py` → open http://localhost:8000/docs — API

---

## Slide 15 — Course Connection: Time Series & Stochastic Processes

How this project connects to course material:

- **Poisson process**: vehicle arrivals modeled as Poisson-distributed random variables
- **Stochastic optimization**: RL optimizes a policy under uncertainty (random arrivals)
- **Time series of rewards**: training curves show non-stationary reward over time
- **Statistical evaluation**: mean ± std over multiple episodes; significance of improvement

---

## Slide 16 — Conclusions

1. RL agents learn adaptive traffic control that outperforms fixed-cycle by 63%
2. Cooperation between agents is essential — the alpha parameter drives this
3. Reward design and normalization are as important as the algorithm choice
4. Full system: simulation → training → evaluation → API → visualization

**Future work**: More complex intersections, real traffic data integration, multi-intersection coordination

---

## Slide 17 — Q&A

Questions?

**Backup answers to prepare**:
- "Why PPO?" — stable, sample-efficient, works well with discrete actions and shared policies
- "Why not DQN?" — PPO handles multi-agent via parameter sharing more naturally
- "How does this scale?" — more intersections = more agents, same shared policy approach
- "Is this realistic?" — simplified model, but Poisson arrivals and signal constraints mirror real systems
- "What about pedestrians?" — future work: additional agents or constraints

---

## Presentation Tips

- **Lead with the visual** — show the queue dynamics comparison early, it's the most convincing result
- **Live demo** makes a strong impression if time permits
- **Emphasize the failures** — the broken baseline, the greedy-hold problem, reward normalization. These show real engineering work
- **Connect to the course** — Poisson processes, stochastic optimization, statistical evaluation
- **Keep equations minimal** — the reward formula and Poisson λ are enough; refer to code for details
