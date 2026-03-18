"""Central configuration for the traffic light RL system."""

from pydantic import BaseModel, Field


class EnvConfig(BaseModel):
    lambda_a: float = Field(default=0.6, description="Poisson arrival rate for road A")
    lambda_b: float = Field(default=0.4, description="Poisson arrival rate for road B")
    green_drain_rate: int = Field(default=3, description="Vehicles per step on green")
    min_green_steps: int = Field(default=4, description="Minimum green phase duration")
    yellow_steps: int = Field(default=2, description="Yellow transition duration")
    max_steps: int = Field(default=500, description="Episode length")
    alpha: float = Field(default=0.3, description="Cooperation coefficient for reward")
    max_queue: int = Field(default=50, description="Maximum queue length per road")


class TrainConfig(BaseModel):
    total_timesteps: int = 300_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    seed: int = 42
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    model_dir: str = "models"


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    state_key: str = "traffic:state"
    metrics_key: str = "traffic:metrics"
    config_key: str = "traffic:config"
    status_key: str = "traffic:status"


class AppConfig(BaseModel):
    env: EnvConfig = EnvConfig()
    train: TrainConfig = TrainConfig()
    redis: RedisConfig = RedisConfig()


DEFAULT_CONFIG = AppConfig()
